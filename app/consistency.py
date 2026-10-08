"""Paraphrase consistency: ask the same question in other words and check the SLM's answers agree."""
from __future__ import annotations

import logging
import re
from concurrent.futures import Executor
from dataclasses import dataclass, field

from .generator import Generator, SLMError
from .verifier import Verifier, make_hypothesis

logger = logging.getLogger("knowsure")

PARAPHRASE_PROMPT = """Rewrite the question below in {n} different ways that keep exactly the same meaning.
Reply with one rewritten question per line, with no numbering and nothing else.

Question: {question}"""

_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")


@dataclass
class ParaphraseCheck:
    question: str
    answer: str
    agreement: float  # 1 = agrees with the original answer, 0.5 = partial/unclear, 0 = contradicts it
    label: str


@dataclass
class ConsistencyResult:
    score: float | None  # mean agreement; None when the check couldn't run
    checks: list[ParaphraseCheck] = field(default_factory=list)
    error: str | None = None


def _agreement(entailment: float, contradiction: float) -> tuple[float, str]:
    if contradiction >= 0.6:
        return 0.0, "contradicts"
    if entailment >= 0.7:
        return 1.0, "agrees"
    if entailment >= 0.35:
        return 0.5, "partial"
    return 0.5, "unclear"


def check_consistency(generator: Generator, verifier: Verifier, pool: Executor, question: str, candidate: str,
                      n: int) -> ConsistencyResult:
    try:
        reply = generator.complete(PARAPHRASE_PROMPT.format(n=n, question=question), 300)
        paraphrases = [_LIST_MARKER.sub("", line).strip() for line in reply.splitlines()]
        paraphrases = [p for p in dict.fromkeys(paraphrases) if p and p.lower() != question.lower()][:n]
        if not paraphrases:
            return ConsistencyResult(None, error="the model returned no paraphrases")
        answers = list(pool.map(generator.generate, paraphrases))
        claim = make_hypothesis(question, candidate)
        scores = verifier.score([(answer, claim) for answer in answers])
    except SLMError as exc:  # consistency is an extra signal: report it, don't fail the request
        logger.warning("Paraphrase consistency check failed: %s", exc)
        return ConsistencyResult(None, error=str(exc))
    checks = [ParaphraseCheck(p, a, *_agreement(s.entailment, s.contradiction))
              for p, a, s in zip(paraphrases, answers, scores)]
    return ConsistencyResult(round(sum(c.agreement for c in checks) / len(checks), 4), checks)
