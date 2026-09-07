from __future__ import annotations

from typing import Iterable


def validate_annotation_rows(rows: Iterable[dict]) -> dict[str, int | bool]:
    rows = list(rows)
    missing_uid = sum("observation_uid" not in row for row in rows)
    missing_points = sum("keypoints" not in row for row in rows)
    return {"rows": len(rows), "missing_uid": missing_uid, "missing_keypoints": missing_points,
            "ok": not (missing_uid or missing_points)}

