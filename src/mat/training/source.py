from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

from mat.core.errors import DependencyUnavailableError


@dataclass
class TrainingReceipt:
    status: str
    seed: int
    optimizer_steps: int
    checkpoint: str | None
    parameter_hash_before: str | None
    parameter_hash_after: str | None
    reason: str | None = None

    def write(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")


class SourceIdentityTrainer:
    def __init__(self, module=None, *, seed: int = 17):
        self.module = module
        self.seed = seed

    def fit(self, batches, output_dir: Path, max_epochs: int = 30) -> TrainingReceipt:
        if self.module is None:
            receipt = TrainingReceipt("BLOCKED_MISSING_TRAINABLE_MODULE", self.seed, 0, None, None, None,
                                      "No verified torch projection module supplied")
            receipt.write(output_dir / "training_receipt.json")
            return receipt
        try:
            import torch
        except ImportError as exc:
            raise DependencyUnavailableError("PyTorch required for source training") from exc
        # Training implementation intentionally requires caller-owned loss/batches so no
        # random labels can masquerade as source identity supervision.
        raise NotImplementedError("provide verified source batches and loss; no fabricated training run")

