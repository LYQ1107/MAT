from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

from .manifests import iter_manifest_records


FORBIDDEN_MODEL_FIELDS = {
    "gt_id", "eid", "identity", "gt_identity", "gt_keypoints", "gt_visibility",
    "treatment", "behavior", "disease",
}


def validate_neutral_manifest(path: Path) -> dict[str, Any]:
    count = 0
    forbidden: list[str] = []
    for row in iter_manifest_records(path):
        count += 1
        forbidden.extend(sorted(FORBIDDEN_MODEL_FIELDS.intersection(row)))
        if "observation_uid" not in row or "session_uid" not in row:
            raise ValueError(f"neutral manifest row lacks observation/session UID: {path}")
    return {"rows": count, "forbidden_fields": sorted(set(forbidden)), "ok": not forbidden}


def manifest_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def leakage_check(observations: Path, truth: Path) -> dict[str, Any]:
    neutral = validate_neutral_manifest(observations)
    truth_ids = {row.get("observation_uid") for row in iter_manifest_records(truth)}
    obs_ids = {row.get("observation_uid") for row in iter_manifest_records(observations)}
    return {**neutral, "truth_rows": len(truth_ids), "observation_truth_overlap": len(obs_ids & truth_ids)}
