from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mat.core.types import IdentityDescriptor, PersistentIdentity, LocalTracklet

EnrollmentProtocol = Literal["human", "oracle_reference", "auto"]


@dataclass(frozen=True)
class ReferenceVerification:
    """S0-only human/oracle grouping; no target-session labels are accepted."""

    groups: dict[str, tuple[str, ...]]
    uncertain_tracklets: tuple[str, ...] = ()
    operation_count: int = 0
    actual_human_seconds: float | None = None
    provenance: str = "human"


@dataclass(frozen=True)
class EnrollmentResult:
    cohort_uid: str
    reference_session_uid: str
    identities: tuple[PersistentIdentity, ...]
    descriptors: dict[str, IdentityDescriptor]
    unresolved_tracklets: tuple[str, ...]
    expected_coverage: float | None
    provenance: str
    status: str = "SUCCEEDED"


@dataclass(frozen=True)
class EvidenceBundle:
    descriptor: IdentityDescriptor
    quality: float
    independent_windows: int
    source_observation_uids: tuple[str, ...] = ()
    provenance: dict[str, Any] = field(default_factory=dict)
