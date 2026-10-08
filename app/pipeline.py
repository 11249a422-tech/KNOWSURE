"""KnowSure pipeline: Generate -> Retrieve -> Verify -> Decide, plus the SLM-only and SLM + RAG baselines.

The knowsure mode adds:
- claim-level detection: the answer is split into atomic claims and each claim is verified on its own;
- adaptive retrieval: a claim not settled by article leads triggers a second round over full articles;
- multiple verifiers: an ensemble scores every evidence sentence, and their (dis)agreement is a signal;
- paraphrase consistency: the question is re-asked in other words and the answers must agree;
- calibrated confidence: the signals are mapped to P(answer is correct).
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .calibration import Calibrator
from .claims import split_claims
from .config import Settings
from .consistency import ConsistencyResult, check_consistency
from .decision import Decision, Signals, Verdict, decide
from .generator import Generator, build_gemini, build_generator
from .retrieval import (Chunk, Embedder, FastEmbedEmbedder, LexicalEmbedder, SentenceTransformerEmbedder, chunk_text,
                        top_k)
from .schemas import (AskResponse, ClaimResult, ConsistencyOut, EvidenceItem, Mode, ParaphraseOut, SignalsOut)
from .verifier import (EnsembleVerifier, LLMJudgeVerifier, NLIVerifier, Verifier, make_hypothesis,
                       split_sentences)
from .wikipedia import EvidenceSource, EvidenceUnavailable, Page, WikipediaClient

logger = logging.getLogger("knowsure")
ABSTAIN_MESSAGE = "I don't know. There isn't enough trustworthy evidence to answer this."
_DONT_KNOW = re.compile(r"\b(i\s+don'?t\s+know|i\s+do\s+not\s+know)\b", re.IGNORECASE)


def says_dont_know(text: str) -> bool:
    return bool(_DONT_KNOW.search(text))


@dataclass
class PassageCheck:
    entailment: float = 0.0
    contradiction: float = 0.0
    support_sentence: str | None = None
    conflict_sentence: str | None = None

    def key_sentence(self, reason: str) -> str | None:
        """The sentence that best explains the decision, for highlighting in the UI."""
        if reason == "conflict_detected":
            return self.conflict_sentence or self.support_sentence
        return self.support_sentence


@dataclass
class ClaimCheck:
    text: str
    verdict: Verdict
    signals: Signals
    hits: list[tuple[Chunk, float]]
    checks: list[PassageCheck]
    agreement: float | None
    rounds: int = 1
    errors: dict[str, str] = field(default_factory=dict)

    def key(self) -> tuple[Chunk, PassageCheck] | None:
        """The passage that best explains this claim's verdict."""
        pairs = list(zip([c for c, _ in self.hits], self.checks))
        if not pairs:
            return None
        if self.verdict.reason == "conflict_detected":
            return max(pairs, key=lambda p: p[1].contradiction)
        return max(pairs, key=lambda p: p[1].entailment)


class KnowSure:
    def __init__(self, settings: Settings, generator: Generator, embedder: Embedder, verifier: Verifier,
                 evidence_source: EvidenceSource, calibrator: Calibrator | None = None):
        self.settings = settings
        self.generator = generator
        self.embedder = embedder
        self.verifier = verifier if isinstance(verifier, EnsembleVerifier) else EnsembleVerifier([("verifier", verifier)])
        self.evidence_source = evidence_source
        self.calibrator = calibrator or Calibrator.load(Path(settings.calibration_file))
        self._page_cache: dict[str, tuple[list[Chunk], np.ndarray]] = {}  # in memory, per process
        self._full_chunks: set[str] = set()   # chunk ids that came from full articles (adaptive retrieval)
        self._lock = threading.Lock()         # one request at a time is fine for a demo
        self._embed_lock = threading.Lock()   # the local embedding model is used from several threads
        # Separate pools so tasks never wait on a pool they are running in.
        self._io = ThreadPoolExecutor(max_workers=8, thread_name_prefix="knowsure-io")      # Wikipedia
        self._tasks = ThreadPoolExecutor(max_workers=6, thread_name_prefix="knowsure-task")  # claims, consistency
        self._llm = ThreadPoolExecutor(max_workers=4, thread_name_prefix="knowsure-llm")     # paraphrase answers

    # ---------------------------------------------------------------- evidence
    def _embed(self, texts: list[str]) -> np.ndarray:
        with self._embed_lock:
            return self.embedder.encode(texts)

    def _passages(self, page: Page) -> tuple[list[Chunk], np.ndarray]:
        key = getattr(page, "key", page.url)
        if key not in self._page_cache:
            chunks = chunk_text(page.text, page.title, page.url, self.settings.chunk_words, self.settings.chunk_overlap)
            vectors = self._embed([c.text for c in chunks]) if chunks else np.zeros((0, 0), np.float32)
            if getattr(page, "section", "lead") == "full":
                self._full_chunks.update(c.id for c in chunks)
            if len(self._page_cache) >= 128:
                self._page_cache.pop(next(iter(self._page_cache)))
            self._page_cache[key] = (chunks, vectors)
        return self._page_cache[key]

    @staticmethod
    def _gather(futures: list[Future]) -> dict[str, Page]:
        pages: dict[str, Page] = {}
        for future in futures:
            for page in future.result():  # re-raises EvidenceUnavailable from the worker thread
                pages.setdefault(getattr(page, "key", page.url), page)
        return pages

    def _rank(self, pages: dict[str, Page], rank_query: str) -> list[tuple[Chunk, float]]:
        chunks: list[Chunk] = []
        blocks = []
        for page in pages.values():
            page_chunks, vectors = self._passages(page)
            if page_chunks:
                chunks += page_chunks
                blocks.append(vectors)
        if not chunks:
            return []
        hits = top_k(self._embed([rank_query])[0], np.vstack(blocks), self.settings.top_k)
        return [(chunks[i], score) for i, score in hits]

    def retrieve(self, search_queries: list[str], rank_query: str,
                 pending: list[Future] | None = None) -> list[tuple[Chunk, float]]:
        """Fetch pages for every search query (in parallel, plus searches already started), then return the
        top-k passages most similar to rank_query."""
        futures = list(pending or []) + [self._io.submit(self.evidence_source.search, q) for q in search_queries]
        return self._rank(self._gather(futures), rank_query)

    def warmup(self) -> None:
        """Load the local models now so the first real question isn't slow (call before a live demo)."""
        with self._lock:
            self._embed(["warmup"])
            for _, member in self.verifier.members:
                if isinstance(member, NLIVerifier):  # LLM judges have nothing to load locally
                    member.score([("The sky is blue.", "The sky is blue.")])

    # ---------------------------------------------------------------- questions
    def ask(self, question: str, mode: Mode = "knowsure") -> AskResponse:
        start = time.perf_counter()
        with self._lock:
            if mode == "slm":
                response = self._ask_slm(question)
            elif mode == "rag":
                response = self._ask_rag(question)
            else:
                response = self._ask_knowsure(question)
        response.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        self._log(response)
        return response

    def _ask_slm(self, question: str) -> AskResponse:
        answer = self.generator.generate(question)
        return AskResponse(question=question, mode="slm", answer=answer, candidate_answer=answer,
                           abstained=says_dont_know(answer), latency_ms=0)

    def _ask_rag(self, question: str) -> AskResponse:
        try:
            hits = self.retrieve([question], question)
        except EvidenceUnavailable as exc:
            logger.warning("Evidence unavailable for the RAG baseline: %s", exc)
            hits = []
        answer = self.generator.generate(question, [c.text for c, _ in hits] or None)
        return AskResponse(question=question, mode="rag", answer=answer, candidate_answer=answer,
                           abstained=says_dont_know(answer), evidence=[self._evidence(c, s) for c, s in hits],
                           citations=_citations([c for c, _ in hits]), latency_ms=0)

    def _ask_knowsure(self, question: str) -> AskResponse:
        s = self.settings
        # The question search doesn't depend on the answer, so start it while the SLM is generating.
        question_search = self._io.submit(self.evidence_source.search, question)
        # 1. Generate.
        try:
            candidate = self.generator.generate(question)
        except Exception:
            question_search.cancel()
            raise
        # Paraphrase consistency runs alongside everything else.
        consistency_job = (self._tasks.submit(check_consistency, self.generator, self.verifier, self._llm, question,
                                              candidate, s.paraphrases) if s.paraphrases > 0 else None)
        # Claim-level detection: split the answer into atomic claims (one claim if it's a single fact).
        claims = (split_claims(self.generator, question, candidate, s.max_claims) if s.claim_level
                  else [make_hypothesis(question, candidate)])
        # 2. Retrieve: evidence for the question and for question + answer, shared by all claims.
        claim_query = f"{question} {' '.join(candidate.split()[:20])}"
        try:
            base_pages = self._gather([question_search, self._io.submit(self.evidence_source.search, claim_query)])
        except EvidenceUnavailable as exc:
            logger.warning("Evidence unavailable, abstaining: %s", exc)
            return self._abstain(question, candidate, Verdict(Decision.UNKNOWN, "evidence_unavailable"),
                                 consistency=self._consistency(consistency_job))
        # 3. Verify each claim (in parallel), with adaptive retrieval for unsettled claims.
        claim_checks = list(self._tasks.map(lambda c: self._check_claim(c, base_pages, len(claims) > 1), claims))
        consistency = self._consistency(consistency_job)
        # 4. Decide.
        return self._respond(question, candidate, claim_checks, consistency)

    # ---------------------------------------------------------------- per claim
    def _check_claim(self, claim: str, base_pages: dict[str, Page], own_search: bool) -> ClaimCheck:
        s = self.settings
        pages = dict(base_pages)
        if own_search:  # a claim may be about something the question didn't mention
            try:
                pages.update(self._gather([self._io.submit(self.evidence_source.search, claim)]))
            except EvidenceUnavailable:
                pass
        hits = self._rank(pages, claim)
        checks, agreement, errors = self._verify(claim, [c for c, _ in hits])
        result = self._claim_verdict(claim, hits, checks, agreement, errors)
        # Adaptive retrieval: leads didn't settle it, so read the full articles for this claim.
        if s.adaptive_retrieval and result.verdict.decision != Decision.KNOWN \
                and result.verdict.reason != "conflict_detected":
            try:
                full_pages = self._gather([self._io.submit(self.evidence_source.search, claim,
                                                           s.adaptive_full_articles)])
            except EvidenceUnavailable:
                full_pages = {}
            if full_pages:
                more_hits = self._rank(full_pages, claim)
                more_checks, more_agreement, more_errors = self._verify(claim, [c for c, _ in more_hits])
                agreements = [a for a in (agreement, more_agreement) if a is not None]
                result = self._claim_verdict(claim, hits + more_hits, checks + more_checks,
                                             min(agreements) if agreements else None, {**errors, **more_errors})
                result.rounds = 2
        return result

    def _claim_verdict(self, claim: str, hits: list[tuple[Chunk, float]], checks: list[PassageCheck],
                       agreement: float | None, errors: dict[str, str]) -> ClaimCheck:
        relevant = [(sim, chk) for (_, sim), chk in zip(hits, checks) if sim >= self.settings.min_retrieval_score]
        signals = Signals(retrieval_score=max((sim for _, sim in hits), default=0.0),
                          support=max((chk.entailment for _, chk in relevant), default=0.0),
                          contradiction=max((chk.contradiction for _, chk in relevant), default=0.0))
        return ClaimCheck(claim, decide(signals, self.settings), signals, hits, checks, agreement, errors=errors)

    def _verify(self, hypothesis: str, chunks: list[Chunk]) -> tuple[list[PassageCheck], float | None, dict[str, str]]:
        """Sentence-level verification by the verifier ensemble. A passage's support is its best-supporting sentence.
        A contradiction only counts when the sentence is about the claim (embedding similarity >= topic_similarity),
        so off-topic text can't flag a correct answer. Also returns how closely the verifiers agree (1 = fully)."""
        checks = [PassageCheck() for _ in chunks]
        sentences = [(i, sent) for i, chunk in enumerate(chunks) for sent in split_sentences(chunk.text)]
        if not sentences:
            return checks, None, {}
        claim_vector = self._embed([hypothesis])[0]
        topical = self._embed([sent for _, sent in sentences]) @ claim_vector
        # Only the most on-topic sentences can support or contradict the claim; the rest count as neutral.
        keep = sorted(np.argsort(-topical)[: self.settings.max_verify_sentences])
        sentences = [sentences[k] for k in keep]
        topical = topical[keep]
        on_topic = topical >= self.settings.topic_similarity
        result = self.verifier.score_detailed([(sent, hypothesis) for _, sent in sentences])
        for (i, sent), is_on_topic, nli in zip(sentences, on_topic, result.combined):
            check = checks[i]
            if nli.entailment > check.entailment:
                check.entailment, check.support_sentence = nli.entailment, sent
            if is_on_topic and nli.contradiction > check.contradiction:
                check.contradiction, check.conflict_sentence = nli.contradiction, sent
        agreement = None
        if len(result.per_verifier) >= 2:
            support = [max((s.entailment for s in scores), default=0.0) for scores in result.per_verifier.values()]
            conflict = [max((s.contradiction for s, t in zip(scores, on_topic) if t), default=0.0)
                        for scores in result.per_verifier.values()]
            agreement = round(1.0 - max(max(support) - min(support), max(conflict) - min(conflict)), 4)
        return checks, agreement, result.errors

    # ---------------------------------------------------------------- combine
    def _consistency(self, job: Future | None) -> ConsistencyResult | None:
        if job is None:
            return None
        try:
            return job.result()
        except Exception as exc:  # never fail the request because of an extra signal
            logger.warning("Consistency check failed: %s", exc)
            return ConsistencyResult(None, error=str(exc))

    def _overall(self, claim_checks: list[ClaimCheck]) -> Verdict:
        if len(claim_checks) == 1:
            return claim_checks[0].verdict
        verdicts = [c.verdict for c in claim_checks]
        if any(v.reason == "conflict_detected" for v in verdicts):
            return Verdict(Decision(self.settings.conflict_decision), "conflict_detected")
        if all(v.decision == Decision.KNOWN for v in verdicts):
            return Verdict(Decision.KNOWN, "supported_by_evidence")
        if any(v.decision == Decision.KNOWN for v in verdicts):
            return Verdict(Decision.UNCERTAIN, "partial_support")
        if any(v.decision == Decision.UNCERTAIN for v in verdicts):
            return Verdict(Decision.UNCERTAIN, "weak_support")
        if all(v.reason == "no_relevant_evidence" for v in verdicts):
            return Verdict(Decision.UNKNOWN, "no_relevant_evidence")
        return Verdict(Decision.UNKNOWN, "insufficient_support")

    def _respond(self, question: str, candidate: str, claim_checks: list[ClaimCheck],
                 consistency: ConsistencyResult | None) -> AskResponse:
        s = self.settings
        verdict = self._overall(claim_checks)
        agreements = [c.agreement for c in claim_checks if c.agreement is not None]
        agreement = round(sum(agreements) / len(agreements), 4) if agreements else None
        consistency_score = consistency.score if consistency else None
        # Extra checks can only lower confidence in a KNOWN answer, never raise it.
        if verdict.decision == Decision.KNOWN and agreement is not None and agreement < s.verifier_agreement_threshold:
            verdict = Verdict(Decision.UNCERTAIN, "verifier_disagreement")
        if verdict.decision == Decision.KNOWN and consistency_score is not None \
                and consistency_score < s.consistency_threshold:
            verdict = Verdict(Decision.UNCERTAIN, "inconsistent_answers")

        signals = Signals(retrieval_score=max(c.signals.retrieval_score for c in claim_checks),
                          support=min(c.signals.support for c in claim_checks),  # as strong as the weakest claim
                          contradiction=max(c.signals.contradiction for c in claim_checks))
        claims_supported = round(sum(c.verdict.decision == Decision.KNOWN for c in claim_checks) / len(claim_checks), 4)
        features = {"support": signals.support, "contradiction": signals.contradiction,
                    "retrieval": signals.retrieval_score, "consistency": consistency_score,
                    "verifier_agreement": agreement, "claims_supported": claims_supported}
        signals_out = SignalsOut(retrieval_score=round(signals.retrieval_score, 4), support=round(signals.support, 4),
                                 contradiction=round(signals.contradiction, 4), consistency=consistency_score,
                                 verifier_agreement=agreement, claims_supported=claims_supported)

        evidence = self._merged_evidence(claim_checks, verdict.reason)
        claims_out, cited = [], []
        for c in claim_checks:
            key = c.key()
            chunk, check = key if key else (None, None)
            claims_out.append(ClaimResult(
                text=c.text, decision=c.verdict.decision.value, reason=c.verdict.reason,
                support=round(c.signals.support, 4), contradiction=round(c.signals.contradiction, 4),
                verifier_agreement=c.agreement, key_sentence=check.key_sentence(c.verdict.reason) if check else None,
                source=chunk.source if chunk else None, url=chunk.url if chunk else None, retrieval_rounds=c.rounds))
            if chunk and c.verdict.decision != Decision.UNKNOWN:
                cited.append(chunk)
        errors = {k: v for c in claim_checks for k, v in c.errors.items()}
        abstained = verdict.decision == Decision.UNKNOWN
        return AskResponse(
            question=question, mode="knowsure", decision=verdict.decision.value, reason=verdict.reason,
            answer=ABSTAIN_MESSAGE if abstained else candidate, candidate_answer=candidate, abstained=abstained,
            signals=signals_out, evidence=evidence, citations=[] if abstained else _citations(cited),
            claims=claims_out, consistency=_consistency_out(consistency),
            confidence=round(self.calibrator.predict(features), 4), confidence_calibrated=self.calibrator.calibrated,
            verifiers=self.verifier.names, verifier_errors=errors, latency_ms=0)

    def _merged_evidence(self, claim_checks: list[ClaimCheck], reason: str) -> list[EvidenceItem]:
        best: dict[str, tuple[Chunk, float, PassageCheck]] = {}
        for c in claim_checks:
            for (chunk, sim), check in zip(c.hits, c.checks):
                seen = best.get(chunk.id)
                if seen is None or max(check.entailment, check.contradiction) > max(seen[2].entailment,
                                                                                     seen[2].contradiction):
                    best[chunk.id] = (chunk, sim, check)
        ranked = sorted(best.values(), key=lambda t: -max(t[2].entailment, t[2].contradiction, t[1] / 10))
        return [self._evidence(chunk, sim, chk.entailment, chk.contradiction, chk.key_sentence(reason))
                for chunk, sim, chk in ranked[:8]]

    def _abstain(self, question: str, candidate: str, verdict: Verdict,
                 consistency: ConsistencyResult | None = None) -> AskResponse:
        return AskResponse(question=question, mode="knowsure", decision=verdict.decision.value, reason=verdict.reason,
                           answer=ABSTAIN_MESSAGE, candidate_answer=candidate, abstained=True,
                           signals=SignalsOut(retrieval_score=0.0, support=0.0, contradiction=0.0),
                           consistency=_consistency_out(consistency), verifiers=self.verifier.names,
                           confidence_calibrated=self.calibrator.calibrated, latency_ms=0)

    def _evidence(self, chunk: Chunk, score: float, entailment: float | None = None,
                  contradiction: float | None = None, key_sentence: str | None = None) -> EvidenceItem:
        return EvidenceItem(chunk_id=chunk.id, source=chunk.source, url=chunk.url, text=chunk.text,
                            retrieval_score=round(score, 4), key_sentence=key_sentence,
                            full_article=chunk.id in self._full_chunks,
                            entailment=None if entailment is None else round(entailment, 4),
                            contradiction=None if contradiction is None else round(contradiction, 4))

    def _log(self, response: AskResponse) -> None:
        if not self.settings.log_runs:
            return
        path = self.settings.runs_log
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"time": datetime.now(timezone.utc).isoformat(), **response.model_dump()}
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _consistency_out(result: ConsistencyResult | None) -> ConsistencyOut | None:
    if result is None:
        return None
    return ConsistencyOut(score=result.score, error=result.error,
                          checks=[ParaphraseOut(question=c.question, answer=c.answer, agreement=c.agreement,
                                                label=c.label) for c in result.checks])


def _citations(chunks: list[Chunk]) -> list[str]:
    return list(dict.fromkeys(c.url for c in chunks))


def build_verifiers(settings: Settings, generator: Generator) -> EnsembleVerifier:
    members: list[tuple[str, Verifier]] = []
    for spec in settings.verifier_specs:
        if spec == "llm":
            members.append((f"llm-judge:{settings.active_slm}", LLMJudgeVerifier(generator)))
        elif spec.startswith("llm:"):
            model = spec.split(":", 1)[1]
            # An extra judge must not hold up the answer: short HTTP timeout, one retry.
            judge = build_gemini(settings, model, timeout=settings.verifier_timeout, retries=1)
            members.append((f"llm-judge:{model}", LLMJudgeVerifier(judge)))
        elif spec == "nli":
            members.append((f"nli:{settings.nli_model}", NLIVerifier(settings.nli_model)))
    # With a single verifier there is nothing to fall back on, so it gets no ensemble timeout.
    return EnsembleVerifier(members, timeout=settings.verifier_timeout if len(members) > 1 else None)


def build_pipeline(settings: Settings) -> KnowSure:
    generator = build_generator(settings)
    if settings.embed_backend == "lexical":
        embedder: Embedder = LexicalEmbedder()
        # Lexical similarity runs on a lower scale than neural similarity: use the lexical thresholds.
        settings = replace(settings, min_retrieval_score=settings.lexical_min_retrieval_score,
                           topic_similarity=settings.lexical_topic_similarity, top_k=settings.lexical_top_k)
    elif settings.embed_backend == "fastembed":
        embedder = FastEmbedEmbedder(settings.embed_model)
    else:
        embedder = SentenceTransformerEmbedder(settings.embed_model)
    return KnowSure(settings, generator, embedder,
                    build_verifiers(settings, generator),
                    WikipediaClient(settings.wikipedia_lang, settings.wiki_search_results, settings.wiki_max_chars,
                                    settings.request_timeout, contact=settings.wiki_contact,
                                    full_max_chars=settings.wiki_full_max_chars))
