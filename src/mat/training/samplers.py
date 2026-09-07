from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import random


class CrossSessionPairSampler:
    """Pairs only where provider/source truth explicitly establishes identity."""

    def __init__(self, rows, seed: int = 17):
        self.rows = list(rows)
        self.seed = seed

    def pairs(self):
        grouped = defaultdict(list)
        for row in self.rows:
            if row.get("gt_id") is not None and row.get("session_uid") is not None:
                grouped[str(row["gt_id"])].append(row)
        out = []
        rng = random.Random(self.seed)
        for identity, rows in sorted(grouped.items()):
            rows = sorted(rows, key=lambda r: (str(r["session_uid"]), str(r.get("observation_uid", ""))))
            for i, left in enumerate(rows):
                for right in rows[i + 1:]:
                    if left["session_uid"] != right["session_uid"]:
                        out.append((left["observation_uid"], right["observation_uid"], identity))
        rng.shuffle(out)
        return out

