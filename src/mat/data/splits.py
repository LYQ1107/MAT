from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import hashlib
import json

from .manifests import iter_manifest_records, write_jsonl


@dataclass(frozen=True)
class FrozenSplit:
    split_id: str
    seed: int
    unit: str
    source: tuple[str, ...]
    development: tuple[str, ...]
    sealed_test: tuple[str, ...]
    manifest_hash: str
    status: str = "FROZEN"

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


def freeze_by_field(manifest: Path, field: str = "cohort_uid", seed: int = 17,
                    source_fraction: float = 0.6, dev_fraction: float = 0.2) -> FrozenSplit:
    values = sorted({str(row.get(field, "unknown")) for row in iter_manifest_records(manifest)})
    # Hash ranking is deterministic and does not depend on Python's randomized hash().
    ranked = sorted(values, key=lambda v: hashlib.sha256(f"{seed}:{v}".encode()).hexdigest())
    n = len(ranked)
    ns = int(n * source_fraction)
    nd = int(n * dev_fraction)
    if n and ns == 0:
        ns = 1
    if n > 1 and nd == 0:
        nd = 1
    if ns + nd >= n and n > 1:
        nd = max(0, n - ns - 1)
    source = tuple(ranked[:ns])
    development = tuple(ranked[ns:ns + nd])
    sealed = tuple(ranked[ns + nd:])
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    split_id = hashlib.sha256(json.dumps([field, seed, source, development, sealed]).encode()).hexdigest()[:16]
    return FrozenSplit(split_id, seed, field, source, development, sealed, digest)

