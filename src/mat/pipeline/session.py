from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class SessionState:
    session_uid: str
    gallery_version_at_start: str
    status: str
    prediction_path: str | None = None
    update_version_at_end: str | None = None


def chronological_sessions(sessions):
    return sorted(sessions, key=lambda s: (s.recorded_at is None, s.recorded_at or "", s.session_uid))

