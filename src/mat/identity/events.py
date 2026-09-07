from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RegistryEvent:
    event_id: str
    event_type: str
    cohort_uid: str
    from_version: str
    to_version: str
    payload: dict[str, Any]
    created_at: str

