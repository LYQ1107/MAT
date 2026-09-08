"""Deterministic gates for safe longitudinal memory updates."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MemoryCommitDecision:
    action: str
    reasons: tuple[str, ...]


class MemoryCommitGate:
    """Allow pending-to-confirmed only with strong, non-conflicting evidence."""

    def __init__(self, high_threshold: float = 0.70, margin_threshold: float = 0.08,
                 quality_threshold: float = 0.60, min_independent_windows: int = 2):
        self.high_threshold = float(high_threshold)
        self.margin_threshold = float(margin_threshold)
        self.quality_threshold = float(quality_threshold)
        self.min_independent_windows = int(min_independent_windows)

    def evaluate(self, *, top1_score: float, top2_score: float = float("-inf"),
                 margin: float | None = None, tracklet_quality: float,
                 common_part_fraction: float = 0.0, descriptor_consistency: float = 1.0,
                 independent_window_count: int, has_conflict: bool,
                 current_role: str = "pending") -> MemoryCommitDecision:
        reasons: list[str] = []
        margin_value = float(top1_score - top2_score) if margin is None and top2_score != float("-inf") else (float(margin) if margin is not None else float("inf"))
        if not top1_score >= self.high_threshold:
            reasons.append("top1_below_high_threshold")
        if margin_value < self.margin_threshold:
            reasons.append("margin_below_threshold")
        if tracklet_quality < self.quality_threshold:
            reasons.append("quality_below_threshold")
        if independent_window_count < self.min_independent_windows:
            reasons.append("insufficient_independent_time_windows")
        if has_conflict:
            reasons.append("cannot_link_conflict")
        if reasons:
            return MemoryCommitDecision("reject" if has_conflict else "wait", tuple(reasons))
        if current_role == "pending":
            return MemoryCommitDecision("promote", ("all_gates_passed",))
        if current_role in {"anchor", "confirmed"}:
            return MemoryCommitDecision("accept_to_quarantine", ("all_gates_passed",))
        return MemoryCommitDecision("reject", ("unknown_memory_role",))


__all__ = ["MemoryCommitGate", "MemoryCommitDecision"]
