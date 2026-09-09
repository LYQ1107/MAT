"""Small diagnostic for predicted-vs-labeled instance counts."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Mapping, Any
import json


def summarize_instance_counts(
    gt_by_frame: Mapping[Any, int] | Iterable[Mapping[str, Any]],
    pred_by_frame: Mapping[Any, int] | Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Return per-frame count differences without changing pose thresholds.

    Inputs may be frame→count maps or rows containing a frame key and (for
    predictions) one row per instance.  The result is intentionally descriptive
    rather than a detector metric: it helps distinguish expected over-detection
    from an SLP parser accounting error.
    """
    def _counts(value, *, prediction: bool) -> Counter:
        if isinstance(value, Mapping):
            return Counter({str(key): int(count) for key, count in value.items()})
        counter: Counter = Counter()
        for row in value:
            key = row.get("frame_uid", row.get("frame_index", row.get("frame_idx")))
            if key is None:
                raise ValueError("pose count rows require frame_uid/frame_index/frame_idx")
            if "count" in row:
                counter[str(key)] += int(row["count"])
            else:
                counter[str(key)] += 1
        return counter

    gt = _counts(gt_by_frame, prediction=False)
    pred = _counts(pred_by_frame, prediction=True)
    frames = sorted(set(gt) | set(pred))
    per_frame = {
        frame: {"gt_instance_count": int(gt.get(frame, 0)), "pred_instance_count": int(pred.get(frame, 0)),
                "difference": int(pred.get(frame, 0) - gt.get(frame, 0))}
        for frame in frames
    }
    equal = sum(int(gt.get(frame, 0) == pred.get(frame, 0)) for frame in frames)
    over = sum(int(pred.get(frame, 0) > gt.get(frame, 0)) for frame in frames)
    under = sum(int(pred.get(frame, 0) < gt.get(frame, 0)) for frame in frames)
    return {
        "per_frame_gt_instance_count": {frame: int(gt.get(frame, 0)) for frame in frames},
        "per_frame_pred_instance_count": {frame: int(pred.get(frame, 0)) for frame in frames},
        "mean_gt_count": (sum(gt.get(frame, 0) for frame in frames) / len(frames)) if frames else None,
        "mean_pred_count": (sum(pred.get(frame, 0) for frame in frames) / len(frames)) if frames else None,
        "frames_pred_gt_equal": equal,
        "frames_over_detect": over,
        "frames_under_detect": under,
        "total_gt": sum(gt.values()),
        "total_pred": sum(pred.values()),
        "frames": len(frames),
        "per_frame": per_frame,
    }


def write_instance_count_diagnostic(path, gt_by_frame, pred_by_frame) -> dict[str, Any]:
    result = summarize_instance_counts(gt_by_frame, pred_by_frame)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


__all__ = ["summarize_instance_counts", "write_instance_count_diagnostic"]
