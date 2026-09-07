from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import numpy as np

from mat.core.types import Assignment, IdentityDescriptor


@dataclass(frozen=True)
class UpdateDecision:
    action: str
    reasons: tuple[str, ...]


class UpdatePolicy:
    """Conservative update gate; thresholds must be calibrated on source/dev."""

    def __init__(self, min_score: float = 0.5, min_margin: float = 0.05,
                 min_quality: float = 0.5, min_evidence_windows: int = 2):
        self.min_score = min_score
        self.min_margin = min_margin
        self.min_quality = min_quality
        self.min_evidence_windows = min_evidence_windows

    def evaluate(self, assignment: Assignment, evidence, *, confirmed: bool = False) -> UpdateDecision:
        scores = assignment.candidate_scores
        reasons: list[str] = []
        if assignment.persistent_uid is None:
            return UpdateDecision("reject", ("unknown_assignment",))
        if not scores or scores[0] < self.min_score:
            reasons.append("score_below_gate")
        if len(scores) > 1 and scores[0] - scores[1] < self.min_margin:
            reasons.append("candidate_margin_below_gate")
        if isinstance(evidence, dict):
            quality = float(evidence.get("quality", 0.0))
            windows = int(evidence.get("independent_windows", 0))
        else:
            quality = float(getattr(evidence, "quality", 0.0))
            windows = int(getattr(evidence, "independent_windows", 0))
        if quality < self.min_quality:
            reasons.append("quality_below_gate")
        if windows < self.min_evidence_windows:
            reasons.append("insufficient_independent_time_windows")
        if reasons:
            return UpdateDecision("wait", tuple(reasons))
        return UpdateDecision("promote" if confirmed else "accept_to_quarantine", ("all_gates_passed",))
