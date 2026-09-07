from __future__ import annotations

import numpy as np


def risk_coverage(scores: np.ndarray, correct: np.ndarray, bins: int = 10):
    scores, correct = np.asarray(scores, float), np.asarray(correct, bool)
    order = np.argsort(-scores, kind="stable")
    rows = []
    for fraction in np.linspace(0.1, 1.0, bins):
        n = max(1, int(round(len(order) * fraction)))
        rows.append({"coverage": n / len(order) if len(order) else 0.0,
                     "risk": float(1 - correct[order[:n]].mean()) if len(order) else None})
    return rows

