from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping
import math

from mat.core.errors import ValidationError


@dataclass(frozen=True)
class IdentityMatchingConfig:
    """The subset of YAML matching settings that controls model assignment."""

    solver: str = "deterministic_greedy"
    accept_threshold: float | None = None
    top_k: int = 4
    threshold_calibration: str = "development"
    calibration_objective: str = "macro_f1"
    mode: str = "global_only"

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any] | None) -> "IdentityMatchingConfig":
        if mapping is None:
            mapping = {}
        if not isinstance(mapping, Mapping):
            raise ValidationError("matching config must be a mapping")
        solver = str(mapping.get("solver", cls.solver))
        if solver != "deterministic_greedy":
            raise ValidationError(f"unsupported identity matching solver: {solver}")
        threshold = mapping.get("accept_threshold", cls.accept_threshold)
        if threshold is not None:
            try:
                threshold = float(threshold)
            except (TypeError, ValueError) as exc:
                raise ValidationError("accept_threshold must be numeric or null") from exc
            if not math.isfinite(threshold):
                raise ValidationError("accept_threshold must be finite or null")
        try:
            top_k = int(mapping.get("top_k", cls.top_k))
        except (TypeError, ValueError) as exc:
            raise ValidationError("top_k must be a positive integer") from exc
        if top_k <= 0:
            raise ValidationError("top_k must be a positive integer")
        calibration = str(mapping.get("threshold_calibration", cls.threshold_calibration))
        if calibration not in {"development", "fixed", "none"}:
            raise ValidationError(f"unsupported threshold_calibration: {calibration}")
        objective = str(mapping.get("calibration_objective", cls.calibration_objective))
        if objective != "macro_f1":
            raise ValidationError(f"unsupported calibration_objective: {objective}")
        mode = str(mapping.get("mode", cls.mode))
        if mode not in {"global_only", "part_aware"}:
            raise ValidationError(f"unsupported identity matching mode: {mode}")
        return cls(solver, threshold, top_k, calibration, objective, mode)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


__all__ = ["IdentityMatchingConfig"]
