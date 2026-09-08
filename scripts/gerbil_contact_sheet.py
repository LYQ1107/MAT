#!/usr/bin/env python3
"""Create a private, deterministic SLEAP gerbil GT contact sheet.

The output belongs under MAT_WORK_ROOT and is intentionally not committed.
Provider track names are shown only for this audit; they are not model inputs.
"""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import os

from PIL import Image, ImageDraw


def main() -> int:
    work = Path(os.environ.get("MAT_WORK_ROOT", "/data2/usr_for_deadline/MAT_workspace"))
    root = work / "prepared" / "sleap_gerbils"
    manifest = root / "manifests"
    observations = [json.loads(line) for line in (manifest / "observations.jsonl").read_text().splitlines() if line]
    truth = {row["observation_uid"]: row for row in
             (json.loads(line) for line in (manifest / "private_pose_identity_truth.jsonl").read_text().splitlines() if line)}
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in observations:
        gt = truth.get(row["observation_uid"], {})
        if gt.get("source_split") == "train":
            by_frame[row["frame_uid"]].append({"observation": row, "truth": gt})
    frame_ids = sorted(by_frame)
    rng = random.Random(17)
    rng.shuffle(frame_ids)
    selected = sorted(frame_ids[:8])
    if len(selected) < 8:
        raise RuntimeError(f"expected at least 8 train frames, got {len(selected)}")
    tiles = []
    audit_rows = []
    for frame_uid in selected:
        entries = by_frame[frame_uid]
        first = entries[0]["observation"]
        image_path = root / first["image_ref"]
        image = Image.open(image_path).convert("RGB")
        image.thumbnail((480, 360))
        tile = Image.new("RGB", (520, 430), "white")
        tile.paste(image, ((520 - image.width) // 2, 12))
        draw = ImageDraw.Draw(tile)
        labels = []
        scale_x = image.width / Image.open(image_path).width
        scale_y = image.height / Image.open(image_path).height
        xoff = (520 - image.width) // 2
        for index, entry in enumerate(entries):
            gt = entry["truth"]
            colour = ((220, 40, 40), (30, 100, 210), (30, 150, 70), (160, 70, 180))[index % 4]
            points = gt.get("gt_keypoints", [])
            for point in points:
                if point[0] is None or point[1] is None:
                    continue
                x = xoff + float(point[0]) * scale_x
                y = 12 + float(point[1]) * scale_y
                draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=colour)
            identity = gt.get("gt_identity") or "untracked"
            labels.append(f"{identity} ({len([p for p in points if p[0] is not None])} kp)")
        draw.text((8, 382), f"{first['session_uid']} frame={first['frame_index']}", fill="black")
        draw.text((8, 400), "; ".join(labels), fill="black")
        tiles.append(tile)
        audit_rows.append({"frame_uid": frame_uid, "session_uid": first["session_uid"],
                           "source_video_name": first.get("source_video_name", first["session_uid"]),
                           "frame_index": first["frame_index"],
                           "provider_identities": sorted({e["truth"].get("gt_identity") for e in entries if e["truth"].get("gt_identity")})})
    sheet = Image.new("RGB", (520 * 2, 430 * 4), "#d0d0d0")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % 2) * 520, (i // 2) * 430))
    output = work / "runs" / "data_sanity" / "gerbils_gt_contact_sheet.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, format="PNG")
    (output.with_suffix(".json")).write_text(json.dumps({"seed": 17, "frames": audit_rows}, indent=2) + "\n")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
