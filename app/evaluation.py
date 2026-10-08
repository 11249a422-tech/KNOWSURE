"""Benchmark loading and the core metrics from the evaluation plan."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .calibration import expected_calibration_error
from .schemas import AskResponse

DECISIONS = {"KNOWN", "UNCERTAIN", "UNKNOWN"}


@dataclass(frozen=True)
class BenchmarkItem:
    question: str
    expected_decision: str             # what a reliable system should do: KNOWN, UNCERTAIN or UNKNOWN
    answers: list[str] = field(default_factory=list)  # accepted answer strings (case-insensitive substring match)
    category: str = ""


def load_benchmark(path: Path) -> list[BenchmarkItem]:
    items = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        item = BenchmarkItem(**json.loads(line))
        if item.expected_decision not in DECISIONS:
            raise ValueError(f"{path}:{n}: expected_decision must be one of {sorted(DECISIONS)}")
        items.append(item)
    return items


def gives_confident_answer(response: AskResponse) -> bool:
    """KnowSure only commits to an answer when KNOWN; the baselines commit whenever they don't say 'I don't know'."""
    if response.mode == "knowsure":
        return response.decision == "KNOWN"
    return not response.abstained


def is_correct(item: BenchmarkItem, response: AskResponse) -> bool:
    text = response.candidate_answer.lower()
    return any(a.lower() in text for a in item.answers)


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 4) if denominator else None


def compute_metrics(items: list[BenchmarkItem], responses: list[AskResponse]) -> dict[str, float | None]:
    pairs = list(zip(items, responses))
    answered = [(i, r) for i, r in pairs if gives_confident_answer(r)]
    answered_with_key = [(i, r) for i, r in answered if i.answers]
    answerable = [i for i in items if i.expected_decision != "UNKNOWN" and i.answers]
    unsupported = [(i, r) for i, r in answered
                   if i.expected_decision == "UNKNOWN" or (i.answers and not is_correct(i, r))]
    metrics = {
        "answer_accuracy": _ratio(sum(is_correct(i, r) for i, r in answered_with_key), len(answered_with_key)),
        "unsupported_answer_rate": _ratio(len(unsupported), len(answered)),
        "abstention_accuracy": _ratio(sum(r.abstained == (i.expected_decision == "UNKNOWN") for i, r in pairs),
                                      len(pairs)),
        "useful_answer_coverage": _ratio(sum(is_correct(i, r) for i, r in answered_with_key), len(answerable)),
        "decision_accuracy": None,
    }
    if pairs and all(r.decision for _, r in pairs):
        metrics["decision_accuracy"] = _ratio(sum(r.decision == i.expected_decision for i, r in pairs), len(pairs))
    # Calibration: does "confidence 0.8" mean right 80% of the time? Only gradable items (with an answer key) count.
    graded = [(r.confidence, is_correct(i, r)) for i, r in pairs if i.answers and r.confidence is not None]
    metrics["ece"] = (expected_calibration_error([c for c, _ in graded], [y for _, y in graded])
                      if graded else None)
    return metrics
