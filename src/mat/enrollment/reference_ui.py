from __future__ import annotations

from pathlib import Path
import json

from mat.core.errors import ValidationError


def export_contact_sheet(tracklets, output_dir: Path, max_frames_per_tracklet: int = 8) -> Path:
    """Export a neutral review manifest; rendering is intentionally delegated to a local UI."""
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [{"tracklet_uid": t.tracklet_uid, "representative_budget": max_frames_per_tracklet,
             "session_uid": t.session_uid, "status": "awaiting_reference_review"} for t in tracklets]
    path = output_dir / "reference_contact_sheet.json"
    path.write_text(json.dumps({"schema_version": "mat.reference_review.v1", "rows": rows}, indent=2) + "\n", encoding="utf-8")
    return path


def validate_reference_review(path: Path, allowed_tracklet_uids: set[str]) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    groups = payload.get("groups", {})
    refs = {ref for values in groups.values() for ref in values}
    if not refs <= allowed_tracklet_uids:
        raise ValidationError("reference review contains a non-S0 tracklet")
    return {"groups": len(groups), "tracklets": len(refs), "provenance": payload.get("provenance", "unknown")}

