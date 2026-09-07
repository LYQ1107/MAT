from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ValidationError


REQUIRED_PROTOCOL = {
    "enrollment", "session_order", "prediction_mode", "update_timing",
    "target_identity_labels_after_reference", "target_pose_labels_for_training",
    "future_sessions_visible",
}


def require_keys(mapping: Mapping[str, Any], keys: set[str], where: str = "config") -> None:
    missing = sorted(keys - mapping.keys())
    if missing:
        raise ValidationError(f"{where} missing required keys: {', '.join(missing)}")


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    require_keys(protocol, REQUIRED_PROTOCOL, "protocol")
    if protocol["enrollment"] not in {"human", "oracle_reference", "auto"}:
        raise ValidationError("protocol.enrollment must be human/oracle_reference/auto")
    if protocol["session_order"] != "chronological":
        raise ValidationError("MAT main protocol requires chronological session order")
    if protocol["update_timing"] != "end_of_session":
        raise ValidationError("MAT main protocol commits updates at end_of_session")
    if protocol["future_sessions_visible"]:
        raise ValidationError("future sessions must not be visible in causal protocol")
    if protocol["target_identity_labels_after_reference"]:
        raise ValidationError("target identity labels after reference are forbidden")

