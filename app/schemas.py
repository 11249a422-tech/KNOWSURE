"""Request and response models for the HTTP API."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# knowsure = full reliability layer; rag and slm are the baselines from the evaluation plan.
Mode = Literal["knowsure", "rag", "slm"]
DecisionLabel = Literal["KNOWN", "UNCERTAIN", "UNKNOWN"]
Reason = Literal["supported_by_evidence", "weak_support", "partial_support", "conflict_detected",
                 "no_relevant_evidence", "insufficient_support", "evidence_unavailable", "inconsistent_answers",
                 "verifier_disagreement"]


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Mode = "knowsure"


class EvidenceItem(BaseModel):
    chunk_id: str
    source: str               # Wikipedia article title
    url: str                  # link to the article
    page: int | None = None   # always None for Wikipedia (kept for frontend compatibility)
    text: str
    retrieval_score: float
    entailment: float | None = None      # best sentence-level support in this passage
    contradiction: float | None = None   # best on-topic sentence-level contradiction in this passage
    key_sentence: str | None = None      # the sentence behind the score, for highlighting
    full_article: bool = False           # found by adaptive retrieval in the full article


class SignalsOut(BaseModel):
    retrieval_score: float
    support: float
    contradiction: float
    consistency: float | None = None         # paraphrase agreement, 0-1 (None if not measured)
    verifier_agreement: float | None = None  # how closely the verifiers agree, 0-1 (None with one verifier)
    claims_supported: float | None = None    # fraction of claims that are KNOWN


class ClaimResult(BaseModel):
    text: str
    decision: DecisionLabel
    reason: Reason
    support: float
    contradiction: float
    verifier_agreement: float | None = None
    key_sentence: str | None = None
    source: str | None = None
    url: str | None = None
    retrieval_rounds: int = 1                # 2 = adaptive retrieval read full articles for this claim


class ParaphraseOut(BaseModel):
    question: str
    answer: str
    agreement: float
    label: str


class ConsistencyOut(BaseModel):
    score: float | None
    checks: list[ParaphraseOut] = []
    error: str | None = None


class AskResponse(BaseModel):
    question: str
    mode: Mode
    decision: DecisionLabel | None = None  # None for the baseline modes, which don't decide
    reason: Reason | None = None
    answer: str                            # what to show the user (the abstain message when abstaining)
    candidate_answer: str                  # the raw SLM output, kept for transparency and evaluation
    abstained: bool
    signals: SignalsOut | None = None
    evidence: list[EvidenceItem] = []
    citations: list[str] = []              # article URLs
    claims: list[ClaimResult] = []         # claim-level verdicts
    consistency: ConsistencyOut | None = None
    confidence: float | None = None        # P(answer is correct) from the calibration model
    confidence_calibrated: bool = False    # False until scripts/calibrate.py has been run
    verifiers: list[str] = []
    verifier_errors: dict[str, str] = {}   # verifiers that failed on this request and were skipped
    latency_ms: float


class HealthResponse(BaseModel):
    status: str
    slm_provider: str
    slm_model: str
    api_key_configured: bool
    evidence_source: str
    wikipedia_contact_configured: bool
    verifier: str
    verifiers: list[str]
    features: dict[str, bool | int]
    confidence_calibrated: bool
    limits: dict[str, int] = {}  # usage limits on a public deployment (0 = off)
    embed_model: str
    nli_model: str
