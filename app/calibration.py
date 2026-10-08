"""Calibration: turn the reliability signals into a confidence that the answer is correct.

The confidence is a logistic model over the signals. Out of the box it uses hand-set starting weights and is
reported as uncalibrated. scripts/calibrate.py fits the weights on labelled benchmark results (Platt-style logistic
regression) and records the Expected Calibration Error (ECE) before and after.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

FEATURES = ["support", "contradiction", "retrieval", "consistency", "verifier_agreement", "claims_supported"]
# Hand-set starting weights (not fitted): fully supported ~0.99, half-supported ~0.45, unsupported or
# contradicted ~0.0. Replace them by running scripts/calibrate.py on your benchmark.
DEFAULT_BIAS = -9.0
DEFAULT_WEIGHTS = {"support": 8.0, "contradiction": -5.0, "retrieval": 1.0, "consistency": 2.0,
                   "verifier_agreement": 1.0, "claims_supported": 3.0}


def feature_vector(features: dict[str, float | None]) -> np.ndarray:
    # A signal that wasn't measured (e.g. consistency switched off) counts as neutral-good, i.e. 1.0.
    return np.array([1.0 if features.get(name) is None else float(features[name]) for name in FEATURES])


@dataclass
class Calibrator:
    bias: float = DEFAULT_BIAS
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    calibrated: bool = False   # True only after fitting on labelled data
    ece: float | None = None   # Expected Calibration Error on the fitting data, after fitting
    samples: int = 0

    def predict(self, features: dict[str, float | None]) -> float:
        z = self.bias + float(np.dot([self.weights.get(n, 0.0) for n in FEATURES], feature_vector(features)))
        return 1.0 / (1.0 + math.exp(-z))

    @classmethod
    def load(cls, path: Path) -> "Calibrator":
        if not path.is_file():
            return cls()
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(data["bias"], data["weights"], True, data.get("ece_after"), data.get("samples", 0))

    def save(self, path: Path, ece_before: float | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"bias": self.bias, "weights": self.weights, "ece_before": ece_before,
                                    "ece_after": self.ece, "samples": self.samples}, indent=2), encoding="utf-8")


def fit(X: np.ndarray, y: np.ndarray, l2: float = 0.01, steps: int = 3000, lr: float = 0.5) -> tuple[float, np.ndarray]:
    """L2-regularised logistic regression by gradient descent. X: (n, len(FEATURES)) in [0, 1]; y: 0/1."""
    w = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        grad = p - y
        w -= lr * (X.T @ grad / len(y) + l2 * w)
        b -= lr * grad.mean()
    return float(b), w


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, bins: int = 10) -> float:
    """Average |accuracy - confidence| over equal-width confidence bins, weighted by bin size."""
    probs, labels = np.asarray(probs, float), np.asarray(labels, float)
    if len(probs) == 0:
        return 0.0
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if mask.any():
            ece += mask.mean() * abs(labels[mask].mean() - probs[mask].mean())
    return round(float(ece), 4)
