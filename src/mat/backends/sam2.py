from __future__ import annotations

from pathlib import Path

from mat.core.errors import DependencyUnavailableError, MissingAssetError


class Sam2MaskBackend:
    """Optional mask front-end; masks never become persistent identities."""

    def __init__(self, checkpoint: Path | None = None, config: Path | None = None):
        self.checkpoint, self.config = checkpoint, config

    def verify_assets(self):
        if self.checkpoint is None or self.config is None or not self.checkpoint.is_file() or not self.config.is_file():
            raise MissingAssetError("SAM2 is optional but requires explicit local checkpoint and config")

    def propagate(self, *args, **kwargs):
        self.verify_assets()
        raise DependencyUnavailableError("SAM2 runtime is not installed; no mask/ID compatibility is fabricated")

