from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

from mat.core.errors import ProtocolError, ValidationError
from mat.core.types import PersistentIdentity
from .base import EnrollmentResult, ReferenceVerification


class ManualRegistrar:
    """Import explicit S0 grouping; the implementation never opens target truth."""

    def build(self, reference, verification: ReferenceVerification | dict[str, Any] | Path | None = None,
              *, cohort_uid: str | None = None) -> EnrollmentResult:
        tracklets = list(reference.tracklets if hasattr(reference, "tracklets") else reference["tracklets"])
        descriptors = reference.descriptors if hasattr(reference, "descriptors") else reference["descriptors"]
        session_uid = reference.session_uid if hasattr(reference, "session_uid") else reference["session_uid"]
        cohort_uid = cohort_uid or (reference.cohort_uid if hasattr(reference, "cohort_uid") else reference.get("cohort_uid", "cohort:unresolved"))
        if verification is None:
            return EnrollmentResult(cohort_uid, session_uid, (), {}, tuple(t.tracklet_uid for t in tracklets),
                                    0.0, "H_human", "BLOCKED_NEEDS_REFERENCE_REVIEW")
        if isinstance(verification, (str, Path)):
            verification = json.loads(Path(verification).read_text(encoding="utf-8"))
        if isinstance(verification, dict):
            verification = ReferenceVerification(
                {str(k): tuple(v) for k, v in verification.get("groups", {}).items()},
                tuple(verification.get("uncertain_tracklets", ())), int(verification.get("operation_count", 0)),
                verification.get("actual_human_seconds"), verification.get("provenance", "human"))
        if verification.provenance not in {"human", "oracle_reference"}:
            raise ProtocolError("verification provenance must be human or oracle_reference")
        if verification.operation_count < 0 or (verification.actual_human_seconds is not None and verification.actual_human_seconds < 0):
            raise ValidationError("verification operation count/time must be non-negative")
        if verification.provenance == "oracle_reference" and verification.actual_human_seconds is not None:
            raise ProtocolError("oracle_reference cannot claim human time")
        allowed = {t.tracklet_uid for t in tracklets}
        identities: list[PersistentIdentity] = []
        out_desc = {}
        consumed = set()
        for name, refs in sorted(verification.groups.items()):
            if any(r not in allowed for r in refs):
                raise ValidationError("verification references a non-S0 tracklet")
            if consumed.intersection(refs):
                raise ValidationError("a tracklet appears in multiple verification groups")
            consumed.update(refs)
            uid = "id:" + hashlib.sha256(f"{cohort_uid}:{session_uid}:{name}".encode()).hexdigest()[:20]
            identities.append(PersistentIdentity(uid, cohort_uid, tuple(refs), (), (), session_uid,
                                                 {"protocol": "H_human" if verification.provenance == "human" else "H_oracle_reference",
                                                  "group_label": str(name), "operation_count": verification.operation_count,
                                                  "actual_human_seconds": verification.actual_human_seconds}))
            out_desc[uid] = descriptors[refs[0]]
        unresolved = tuple(sorted(allowed - consumed - set(verification.uncertain_tracklets)))
        unresolved += tuple(sorted(set(verification.uncertain_tracklets) & allowed))
        coverage = len(consumed) / len(allowed) if allowed else 0.0
        provenance = "H_human" if verification.provenance == "human" else "H_oracle_reference"
        return EnrollmentResult(cohort_uid, session_uid, tuple(identities), out_desc, unresolved,
                                coverage, provenance, "SUCCEEDED")
