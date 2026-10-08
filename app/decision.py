"""The decision engine: turn reliability signals into KNOWN / UNCERTAIN / UNKNOWN."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import Settings


class Decision(str, Enum):
    KNOWN = "KNOWN"          # the answer is sufficiently supported by available evidence
    UNCERTAIN = "UNCERTAIN"  # support is weak/incomplete, or evidence conflicts with the answer: ask to verify
    UNKNOWN = "UNKNOWN"      # no trustworthy evidence: abstain


@dataclass(frozen=True)
class Signals:
    retrieval_score: float  # best cosine similarity between the query and any retrieved passage
    support: float          # best entailment probability over relevant passages
    contradiction: float    # best contradiction probability over relevant passages


@dataclass(frozen=True)
class Verdict:
    decision: Decision
    reason: str


def decide(signals: Signals, settings: Settings) -> Verdict:
    if signals.retrieval_score < settings.min_retrieval_score:
        return Verdict(Decision.UNKNOWN, "no_relevant_evidence")
    if signals.contradiction >= settings.conflict_contradiction and signals.contradiction > signals.support:
        return Verdict(Decision(settings.conflict_decision), "conflict_detected")
    if signals.support >= settings.known_support:
        return Verdict(Decision.KNOWN, "supported_by_evidence")
    if signals.support >= settings.uncertain_support:
        return Verdict(Decision.UNCERTAIN, "weak_support")
    return Verdict(Decision.UNKNOWN, "insufficient_support")
