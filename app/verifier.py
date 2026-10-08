"""Evidence verification: LLM judges, a local NLI model, and an ensemble of several verifiers."""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass
from typing import Protocol

NLI_LABELS = {"entailment", "contradiction", "neutral"}
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])")


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    """Split a passage into sentences for sentence-level NLI; fragments shorter than min_words are dropped."""
    sentences = [s.strip() for s in _SENTENCE_END.split(text) if len(s.split()) >= min_words]
    return sentences or [text.strip()]


@dataclass(frozen=True)
class NLIScores:
    entailment: float
    contradiction: float
    neutral: float


class Verifier(Protocol):
    def score(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        """Score (premise = evidence, hypothesis = claim) pairs."""


def make_hypothesis(question: str, answer: str) -> str:
    """The claim the NLI model checks. A full-sentence answer is already a claim; a short one ("Neil Armstrong")
    needs the question for context."""
    answer = answer.strip()
    if len(answer.split()) >= 4:
        return answer
    return f'The answer to the question "{question.strip()}" is: {answer}'


JUDGE_PROMPT = """You are a strict fact-checker. Judge each numbered evidence sentence against the claim, using only what that sentence says.

Claim: "{claim}"

Labels:
- supports: the sentence states that the claim is true
- partial: the sentence supports part of the claim but not all of it
- contradicts: the sentence states something that makes the claim false
- neutral: the sentence is unrelated to the claim or does not settle it

Evidence sentences:
{sentences}

Reply with exactly one line per sentence, in the form "<number>: <label>", and nothing else."""

# Label -> (entailment, contradiction, neutral), on the same 0-1 scale as the NLI model.
_JUDGE_SCORES = {
    "supports": NLIScores(0.95, 0.0, 0.05),
    "partial": NLIScores(0.5, 0.0, 0.5),
    "contradicts": NLIScores(0.0, 0.95, 0.05),
    "neutral": NLIScores(0.0, 0.0, 1.0),
}
_JUDGE_LINE = re.compile(r"^\W*(\d+)\W+(supports|partial|contradicts|neutral)\b", re.IGNORECASE | re.MULTILINE)


class Completer(Protocol):
    def complete(self, prompt: str, max_output_tokens: int | None = None) -> str: ...


class LLMJudgeVerifier:
    """Verifier that asks the SLM itself to label evidence sentences, one API call per claim.
    Unparseable or missing lines count as neutral, so a bad reply can never create support."""

    def __init__(self, llm: Completer, max_output_tokens: int = 600):
        self.llm = llm
        self.max_output_tokens = max_output_tokens

    def score(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        results: list[NLIScores] = [_JUDGE_SCORES["neutral"]] * len(pairs)
        by_claim: dict[str, list[int]] = {}
        for index, (_, claim) in enumerate(pairs):
            by_claim.setdefault(claim, []).append(index)
        for claim, indices in by_claim.items():
            numbered = "\n".join(f"[{n}] {pairs[i][0]}" for n, i in enumerate(indices, 1))
            reply = self.llm.complete(JUDGE_PROMPT.format(claim=claim, sentences=numbered), self.max_output_tokens)
            for number, label in _JUDGE_LINE.findall(reply):
                n = int(number)
                if 1 <= n <= len(indices):
                    results[indices[n - 1]] = _JUDGE_SCORES[label.lower()]
        return results


@dataclass
class EnsembleResult:
    combined: list[NLIScores]                 # mean over the verifiers that succeeded
    per_verifier: dict[str, list[NLIScores]]  # each verifier's own scores
    errors: dict[str, str]                    # verifiers that failed on this call (skipped)


class EnsembleVerifier:
    """Multiple verifiers scoring the same pairs in parallel; scores are averaged. A failing verifier (quota, outage)
    is skipped and reported, so one bad provider doesn't fail the request. Raises only if every verifier fails."""

    def __init__(self, members: list[tuple[str, "Verifier"]], timeout: float | None = None):
        if not members:
            raise ValueError("an ensemble needs at least one verifier")
        self.members = members
        self.timeout = timeout  # a verifier slower than this is skipped for this call (the others decide)
        self._pool = ThreadPoolExecutor(max_workers=max(4, 3 * len(members)))

    @property
    def names(self) -> list[str]:
        return [name for name, _ in self.members]

    def score(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        return self.score_detailed(pairs).combined

    def score_detailed(self, pairs: list[tuple[str, str]]) -> EnsembleResult:
        if not pairs:
            return EnsembleResult([], {name: [] for name in self.names}, {})
        futures = {name: self._pool.submit(verifier.score, pairs) for name, verifier in self.members}
        wait(futures.values(), timeout=self.timeout)
        per_verifier, errors, last_error = {}, {}, None
        for name, future in futures.items():
            if not future.done():  # too slow (e.g. an overloaded API): decide without it
                errors[name] = f"timed out after {self.timeout:g}s"
                last_error = TimeoutError(f"{name} timed out")
                continue
            try:
                per_verifier[name] = future.result()
            except Exception as exc:  # a provider failing must not take the whole check down
                errors[name] = str(exc)
                last_error = exc
        if not per_verifier:
            raise last_error
        n = len(per_verifier)
        combined = [NLIScores(entailment=sum(s[i].entailment for s in per_verifier.values()) / n,
                              contradiction=sum(s[i].contradiction for s in per_verifier.values()) / n,
                              neutral=sum(s[i].neutral for s in per_verifier.values()) / n)
                    for i in range(len(pairs))]
        return EnsembleResult(combined, per_verifier, errors)


class NLIVerifier:
    """Cross-encoder NLI model, run locally. Loads on first use."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._tokenizer = None
        self._model = None
        self._labels: dict[int, str] = {}
        self._lock = threading.Lock()  # local model: one call at a time

    def _load(self) -> None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
        self._model.eval()
        self._labels = {int(i): label.lower() for i, label in self._model.config.id2label.items()}
        if set(self._labels.values()) != NLI_LABELS:
            raise ValueError(f"{self.model_name} is not a 3-way NLI model (labels: {self._labels})")

    def score(self, pairs: list[tuple[str, str]]) -> list[NLIScores]:
        if not pairs:
            return []
        import torch

        with self._lock:
            if self._model is None:
                self._load()
            encoded = self._tokenizer([p for p, _ in pairs], [h for _, h in pairs], padding=True, truncation=True,
                                      max_length=512, return_tensors="pt")
            with torch.no_grad():
                probs = self._model(**encoded).logits.softmax(dim=-1).tolist()
        return [NLIScores(**{self._labels[i]: p for i, p in enumerate(row)}) for row in probs]
