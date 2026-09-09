"""Frozen, label-aware longitudinal protocol for the prepared gerbil bundle.

The generic :mod:`mat.data.splits` helper intentionally remains hash-based for
other datasets.  This module is specific to the SLEAP gerbil manifests and is
the only place that reads the private provider identity truth while constructing
the reference/development/sealed session roles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from mat.core.errors import ValidationError


DEFAULT_IDENTITY_LABELS = ("female", "male", "pup shaved", "pup unshaved")


def _sha256(path: Path) -> str:
    if not path.is_file():
        raise ValidationError(f"missing protocol input manifest: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_datetime(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


@dataclass(frozen=True)
class GerbilLongitudinalProtocol:
    protocol_id: str
    seed: int
    ordering_basis: str
    reference_sessions: tuple[str, ...]
    source_sessions: tuple[str, ...]
    development_sessions: tuple[str, ...]
    sealed_test_sessions: tuple[str, ...]
    identity_labels: tuple[str, ...]
    session_manifest_sha256: str
    truth_manifest_sha256: str
    provider_identity_evidence: str
    all_provider_sessions: tuple[str, ...] = ()
    labeled_sessions: tuple[str, ...] = ()
    status: str = "FROZEN"

    def __post_init__(self) -> None:
        roles = (
            *self.reference_sessions,
            *self.source_sessions,
            *self.development_sessions,
            *self.sealed_test_sessions,
        )
        if not self.identity_labels or len(set(self.identity_labels)) != len(self.identity_labels):
            raise ValidationError("identity_labels must be non-empty and unique")
        if len(set(roles)) != len(roles):
            raise ValidationError("a session cannot occur in more than one protocol role")
        if self.reference_sessions and not set(self.reference_sessions).issubset(set(self.labeled_sessions or roles)):
            raise ValidationError("reference sessions must be labeled sessions")
        if self.status != "FROZEN":
            raise ValidationError("gerbil longitudinal protocol must be written as FROZEN")

    @property
    def all_role_sessions(self) -> tuple[str, ...]:
        return (
            *self.reference_sessions,
            *self.source_sessions,
            *self.development_sessions,
            *self.sealed_test_sessions,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in (
            "reference_sessions", "source_sessions", "development_sessions",
            "sealed_test_sessions", "identity_labels", "all_provider_sessions",
            "labeled_sessions",
        ):
            payload[key] = list(payload[key])
        payload["role_counts"] = {
            "all_provider_sessions": len(self.all_provider_sessions),
            "labeled_sessions": len(self.labeled_sessions),
            "reference_sessions": len(self.reference_sessions),
            "source_sessions": len(self.source_sessions),
            "development_sessions": len(self.development_sessions),
            "sealed_test_sessions": len(self.sealed_test_sessions),
        }
        return payload

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8")

    @classmethod
    def read(cls, path: Path) -> "GerbilLongitudinalProtocol":
        raw = json.loads(path.read_text(encoding="utf-8"))
        tuple_keys = (
            "reference_sessions", "source_sessions", "development_sessions",
            "sealed_test_sessions", "identity_labels", "all_provider_sessions",
            "labeled_sessions",
        )
        for key in tuple_keys:
            raw[key] = tuple(str(value) for value in raw.get(key, ()))
        raw.pop("role_counts", None)
        return cls(**raw)


def _ordered_sessions(rows: Sequence[Mapping[str, Any]], labeled: set[str]) -> tuple[tuple[str, ...], str]:
    participating = [row for row in rows if str(row.get("session_uid", "")) in labeled]
    dates = {str(row.get("session_uid")): _valid_datetime(row.get("recording_datetime_if_parseable"))
             for row in participating}
    if participating and all(value is not None for value in dates.values()):
        ordered = sorted(participating, key=lambda row: (dates[str(row["session_uid"])], str(row["session_uid"])))
        return tuple(str(row["session_uid"]) for row in ordered), "provider_recording_datetime"
    return tuple(sorted(labeled)), "deterministic_session_uid_fallback"


def _role_split(remaining: Sequence[str]) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    values = tuple(remaining)
    n = len(values)
    if n == 0:
        return (), (), ()
    if n == 1:
        # It is impossible to satisfy both minimum holdout roles with one
        # session; keep the only remaining recording sealed and report the
        # empty development role in the protocol itself.
        return (), (), values
    if n == 2:
        return (), (values[0],), (values[1],)
    source_count = max(1, int(n * 0.60))
    development_count = max(1, int(n * 0.20))
    if source_count + development_count >= n:
        source_count = max(1, n - development_count - 1)
    source = values[:source_count]
    development = values[source_count:source_count + development_count]
    sealed = values[source_count + development_count:]
    if not sealed:
        sealed = (development[-1],)
        development = development[:-1]
    return tuple(source), tuple(development), tuple(sealed)


def build_gerbil_longitudinal_protocol(
    work_root: Path,
    *,
    seed: int = 17,
    identity_labels: Sequence[str] = DEFAULT_IDENTITY_LABELS,
    output: Path | None = None,
) -> GerbilLongitudinalProtocol:
    """Build and freeze the protocol from session inventory and private truth."""
    root = Path(work_root).expanduser().resolve() / "prepared" / "sleap_gerbils" / "manifests"
    session_path = root / "session_inventory.json"
    truth_path = root / "private_pose_identity_truth.jsonl"
    raw_sessions = json.loads(session_path.read_text(encoding="utf-8"))
    session_rows = list(raw_sessions.get("sessions", ()))
    if not session_rows:
        raise ValidationError("session inventory contains no provider sessions")
    truth_rows = [json.loads(line) for line in truth_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    truth_sessions: dict[str, set[str]] = {}
    for row in truth_rows:
        session = row.get("session_uid")
        if session is None:
            # Prepared truth historically omits session_uid; join by its
            # observation UID through the neutral manifest without exposing the
            # join to any model runner.
            continue
        truth_sessions.setdefault(str(session), set()).add(str(row.get("gt_identity")))
    observation_path = root / "observations.jsonl"
    if any(not row.get("session_uid") for row in truth_rows):
        obs_to_session = {}
        for line in observation_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                obs_to_session[str(row["observation_uid"])] = str(row["session_uid"])
        for row in truth_rows:
            session = obs_to_session.get(str(row.get("observation_uid")))
            if session:
                truth_sessions.setdefault(session, set()).add(str(row.get("gt_identity")))
    all_provider = tuple(sorted({str(row.get("session_uid")) for row in session_rows if row.get("session_uid")}))
    labeled = {
        str(row.get("session_uid")) for row in session_rows
        if int(row.get("labeled_frame_count", 0) or 0) > 0
    }
    labeled.update(session for session, labels in truth_sessions.items() if labels)
    labeled &= set(all_provider) | set(truth_sessions)
    ordered, ordering_basis = _ordered_sessions(session_rows, labeled)
    expected = tuple(str(label) for label in identity_labels)
    covered: set[str] = set()
    reference: list[str] = []
    for session in ordered:
        reference.append(session)
        covered.update(truth_sessions.get(session, set()))
        if set(expected).issubset(covered):
            break
    if not set(expected).issubset(covered):
        raise ValidationError(f"reference sessions do not cover all identities: missing={sorted(set(expected) - covered)}")
    reference_set = set(reference)
    remaining = tuple(session for session in ordered if session not in reference_set)
    source, development, sealed = _role_split(remaining)
    role_payload = {
        "seed": int(seed),
        "reference_sessions": list(reference),
        "source_sessions": list(source),
        "development_sessions": list(development),
        "sealed_test_sessions": list(sealed),
        "session_manifest_sha256": _sha256(session_path),
        "truth_manifest_sha256": _sha256(truth_path),
    }
    protocol_id = hashlib.sha256(json.dumps(role_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:16]
    protocol = GerbilLongitudinalProtocol(
        protocol_id=protocol_id, seed=int(seed), ordering_basis=ordering_basis,
        reference_sessions=tuple(reference), source_sessions=source,
        development_sessions=development, sealed_test_sessions=sealed,
        identity_labels=expected, session_manifest_sha256=role_payload["session_manifest_sha256"],
        truth_manifest_sha256=role_payload["truth_manifest_sha256"],
        provider_identity_evidence=(
            "SLEAP provider source track names in private truth; no independent "
            "cross-day biological-ID mapping; NOT_STRICT_CHRONOLOGICAL_DATE_ORDER "
            "because fallback UID order is not a recorded date order"
            if ordering_basis != "provider_recording_datetime" else
            "SLEAP provider source track names in private truth; no independent cross-day biological-ID mapping"
        ),
        all_provider_sessions=all_provider, labeled_sessions=tuple(ordered),
    )
    if output is None:
        output = Path(work_root).expanduser().resolve() / "assets" / "manifests" / "splits" / "gerbils_longitudinal_v1.json"
    protocol.write(output)
    return protocol


__all__ = ["DEFAULT_IDENTITY_LABELS", "GerbilLongitudinalProtocol", "build_gerbil_longitudinal_protocol"]
