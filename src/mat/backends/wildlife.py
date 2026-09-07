from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import hashlib
import numpy as np

from mat.core.errors import DependencyUnavailableError, MissingAssetError, ValidationError
from mat.core.types import DescriptorBatch


class GlobalIdentityBackend:
    """Offline wrapper for a verified local WildlifeTools/MegaDescriptor model."""

    def __init__(self, model: Any | None = None, *, model_hash: str | None = None,
                 preprocess_fingerprint: str = "unresolved", device: str = "cpu"):
        self.model = model
        self.model_hash = model_hash
        self.preprocess_fingerprint = preprocess_fingerprint
        self.device = device
        if model is None:
            raise MissingAssetError("MegaDescriptor model must be supplied from a verified local checkpoint")
        self.fingerprint = hashlib.sha256(
            f"{model_hash}:{preprocess_fingerprint}:{type(model).__module__}.{type(model).__qualname__}".encode()
        ).hexdigest()[:20]

    def encode(self, images: Any) -> DescriptorBatch:
        """Encode neutral images. This method does not accept identity labels."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise DependencyUnavailableError("PyTorch is required for MegaDescriptor inference") from exc
        if hasattr(self.model, "eval"):
            self.model.eval()
        with torch.no_grad():
            tensor = images
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.as_tensor(np.asarray(images))
            output = self.model(tensor.to(self.device))
            if isinstance(output, (tuple, list)):
                output = output[0]
            output = output.detach().cpu().numpy().astype(np.float32)
        if output.ndim != 2:
            raise ValidationError("identity model must return [B,D]")
        norms = np.linalg.norm(output, axis=1, keepdims=True)
        output = output / np.where(norms > 0, norms, 1.0)
        parts = np.zeros((len(output), 0, output.shape[1]), dtype=np.float32)
        return DescriptorBatch(output, parts, np.zeros((len(output), 0), bool),
                               np.zeros((len(output), 0), np.float32), self.fingerprint)

    @classmethod
    def from_local(cls, checkpoint: Path, *, architecture: str = "swin_tiny_patch4_window7_224",
                   config_path: Path | None = None, expected_sha256: str | None = None,
                   device: str = "cpu") -> "GlobalIdentityBackend":
        """Construct a model without ``hf-hub``/``pretrained=True`` network access."""
        if not checkpoint.is_file():
            raise MissingAssetError(f"missing identity checkpoint: {checkpoint}")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise MissingAssetError("identity checkpoint SHA-256 mismatch")
        try:
            import torch
            import timm
        except ImportError as exc:  # pragma: no cover
            raise DependencyUnavailableError("offline identity loading requires torch and timm") from exc
        model = timm.create_model(architecture, pretrained=False, num_classes=0)
        raw = torch.load(str(checkpoint), map_location="cpu")
        state = raw.get("state_dict", raw.get("model", raw)) if isinstance(raw, dict) else raw
        if not isinstance(state, dict):
            raise ValidationError("unsupported checkpoint structure")
        cleaned = {}
        for key, value in state.items():
            key = key.removeprefix("module.")
            if any(key.startswith(prefix) for prefix in ("head.", "fc.", "classifier.")):
                continue
            cleaned[key] = value
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        non_head_missing = [k for k in missing if not any(k.startswith(p) for p in ("head.", "fc.", "classifier."))]
        non_head_unexpected = [k for k in unexpected if not any(k.startswith(p) for p in ("head.", "fc.", "classifier."))]
        if non_head_missing or non_head_unexpected:
            raise MissingAssetError(f"checkpoint backbone mismatch: missing={non_head_missing}, unexpected={non_head_unexpected}")
        model.to(device).eval()
        preprocess_fingerprint = hashlib.sha256(config_path.read_bytes()).hexdigest() if config_path else "config-unresolved"
        return cls(model, model_hash=digest, preprocess_fingerprint=preprocess_fingerprint, device=device)


class NumpyFixtureEncoder:
    """Small deterministic encoder used only by TEST_FIXTURE/unit tests."""

    fingerprint = "TEST_FIXTURE:numpy-color-hist-v1"

    def encode(self, images: np.ndarray) -> DescriptorBatch:
        arr = np.asarray(images, dtype=np.float32)
        if arr.ndim != 4 or arr.shape[-1] not in {1, 3}:
            raise ValidationError("fixture images must be B,H,W,C")
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        means = arr.mean(axis=(1, 2)) / 255.0
        std = arr.std(axis=(1, 2)) / 255.0
        hist = np.concatenate([means, std], axis=1)
        hist = hist / np.where(np.linalg.norm(hist, axis=1, keepdims=True) > 0,
                               np.linalg.norm(hist, axis=1, keepdims=True), 1.0)
        return DescriptorBatch(hist.astype(np.float32), np.zeros((len(hist), 0, hist.shape[1]), np.float32),
                               np.zeros((len(hist), 0), bool), np.zeros((len(hist), 0), np.float32), self.fingerprint)
