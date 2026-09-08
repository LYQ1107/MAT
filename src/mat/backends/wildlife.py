from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable
import hashlib
import json
import numpy as np

from mat.core.errors import DependencyUnavailableError, MissingAssetError, ValidationError
from mat.core.types import DescriptorBatch


try:  # The lightweight audit interpreter may not ship PyTorch.
    import torch
    from torch import nn
except Exception:  # pragma: no cover
    torch = None
    nn = object


@dataclass(frozen=True)
class IdentityPreprocessSpec:
    """Audited MegaDescriptor-T-224 input contract.

    The official WildlifeTools example uses a fixed 224x224 resize, RGB
    ``ToTensor`` scaling and ImageNet normalization.  The model repository's
    ``config.json`` is authoritative for interpolation; the pinned T-224
    revision reports bicubic.
    """

    size: tuple[int, int] = (224, 224)
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    interpolation: str = "bicubic"
    antialias: bool = True
    source: str = "BVRA/MegaDescriptor-T-224:pretrained_cfg"

    def __post_init__(self) -> None:
        if len(self.size) != 2 or any(int(x) <= 0 for x in self.size):
            raise ValidationError("identity preprocess size must be two positive integers")
        if len(self.mean) != 3 or len(self.std) != 3 or any(float(x) <= 0 for x in self.std):
            raise ValidationError("identity preprocess mean/std must have three channels")
        if self.interpolation not in {"nearest", "bilinear", "bicubic", "area"}:
            raise ValidationError(f"unsupported identity interpolation: {self.interpolation}")

    @classmethod
    def official_t224(cls, config: dict[str, Any] | None = None) -> "IdentityPreprocessSpec":
        cfg = (config or {}).get("pretrained_cfg", {}) if isinstance(config, dict) else {}
        input_size = cfg.get("input_size", [3, 224, 224])
        size = tuple(int(x) for x in input_size[-2:]) if len(input_size) >= 2 else (224, 224)
        mean = tuple(float(x) for x in cfg.get("mean", cls.mean))
        std = tuple(float(x) for x in cfg.get("std", cls.std))
        interpolation = str(cfg.get("interpolation", cls.interpolation)).lower()
        return cls(size=size, mean=mean, std=std, interpolation=interpolation,
                   antialias=True, source="BVRA/MegaDescriptor-T-224:pretrained_cfg")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["fingerprint"] = self.fingerprint
        return value

    def write(self, path: Path, *, equivalence: dict[str, Any] | None = None) -> None:
        payload = {"schema_version": "mat.identity_preprocess.v1", "spec": self.to_dict()}
        if equivalence is not None:
            payload["equivalence"] = equivalence
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def apply(self, images: Any):
        """Convert numpy BHWC uint8 or torch BCHW input to normalized BCHW."""
        if torch is None:  # pragma: no cover - minimal audit interpreter
            raise DependencyUnavailableError("PyTorch is required for MegaDescriptor preprocessing")
        if isinstance(images, torch.Tensor):
            tensor = images
            if tensor.ndim == 3:
                tensor = tensor.unsqueeze(0)
            if tensor.ndim != 4 or tensor.shape[1] not in (1, 3, 4):
                raise ValidationError("torch identity images must be BCHW with 1, 3 or 4 channels")
        else:
            array = np.asarray(images)
            if array.ndim == 3:
                array = array[None, ...]
            if array.ndim != 4 or array.shape[-1] not in (1, 3, 4):
                raise ValidationError("numpy identity images must be BHWC with 1, 3 or 4 channels")
            tensor = torch.as_tensor(array).permute(0, 3, 1, 2)
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        elif tensor.shape[1] == 4:
            tensor = tensor[:, :3]
        # ``ToTensor`` is applied *after* torchvision Resize in the official
        # pipeline.  Keep uint8/0..255 values through interpolation; scaling
        # before bicubic resize changes the result because of rounding and is
        # measurably non-equivalent.
        scale_255 = bool(tensor.numel() and float(tensor.detach().amax().cpu()) > 1.0)
        # Prefer torchvision's public functional resize: for uint8 inputs it
        # preserves the exact rounding semantics of the official
        # ``Resize(...); ToTensor()`` pipeline.  Fall back to torch functional
        # interpolation only in a minimal runtime without torchvision.
        try:
            from torchvision.transforms import InterpolationMode
            from torchvision.transforms.functional import resize as tv_resize
            interpolation_mode = getattr(InterpolationMode, self.interpolation.upper())
            tensor = tv_resize(tensor, list(self.size), interpolation=interpolation_mode,
                               antialias=self.antialias)
        except (ImportError, AttributeError, RuntimeError, TypeError):
            try:
                import torch.nn.functional as functional
                tensor = tensor.to(dtype=torch.float32)
                tensor = functional.interpolate(
                    tensor, size=self.size, mode=self.interpolation,
                    align_corners=False if self.interpolation in {"bilinear", "bicubic"} else None,
                    antialias=self.antialias,
                )
            except TypeError:  # torch versions predating interpolate(antialias=...)
                import torch.nn.functional as functional
                tensor = tensor.to(dtype=torch.float32)
                kwargs = {"align_corners": False} if self.interpolation in {"bilinear", "bicubic"} else {}
                tensor = functional.interpolate(tensor, size=self.size, mode=self.interpolation, **kwargs)
        tensor = tensor.to(dtype=torch.float32)
        if scale_255:
            tensor = tensor / 255.0
        mean = torch.tensor(self.mean, dtype=torch.float32, device=tensor.device).view(1, 3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32, device=tensor.device).view(1, 3, 1, 1)
        return (tensor - mean) / std


class MegaDescriptorRuntime(nn.Module if torch is not None else object):
    """Frozen backbone plus the verified preprocessing contract."""

    def __init__(self, backbone: Any, preprocess_spec: IdentityPreprocessSpec | None = None):
        if torch is None:  # pragma: no cover
            raise DependencyUnavailableError("PyTorch is required for MegaDescriptorRuntime")
        super().__init__()
        self.backbone = backbone
        self.preprocess_spec = preprocess_spec or IdentityPreprocessSpec.official_t224()
        self.output_dim = int(getattr(backbone, "num_features", 0) or 0)

    def preprocess(self, images: Any):
        return self.preprocess_spec.apply(images)

    def forward(self, images: Any):
        tensor = self.preprocess(images)
        output = self.backbone(tensor)
        if isinstance(output, dict):
            output = output.get("embedding", output.get("features", output.get("x", output)))
        if isinstance(output, (tuple, list)):
            output = output[0]
        if not isinstance(output, torch.Tensor):
            output = torch.as_tensor(output, device=tensor.device)
        if output.ndim == 4:
            output = output.mean(dim=(2, 3))
        elif output.ndim == 3:
            output = output.mean(dim=1)
        elif output.ndim != 2:
            output = output.flatten(1)
        if output.ndim != 2:
            raise ValidationError("MegaDescriptor backbone must return [B,D] features")
        if self.output_dim <= 0:
            self.output_dim = int(output.shape[1])
        return output

    def write_preprocess_audit(self, path: Path, *, images: Any | None = None,
                               reference_transform: Callable[[Any], Any] | None = None) -> dict[str, Any]:
        """Save spec and, when supplied, numerical local-vs-official equivalence."""
        audit: dict[str, Any] = {"status": "NOT_RUN_NO_REFERENCE"}
        if images is not None and reference_transform is not None:
            local = self.preprocess(images).detach().cpu().numpy()
            reference = reference_transform(images)
            if hasattr(reference, "detach"):
                reference = reference.detach().cpu().numpy()
            reference = np.asarray(reference, dtype=np.float32)
            delta = np.abs(local.astype(np.float32) - reference)
            max_error = float(delta.max()) if delta.size else 0.0
            audit = {"status": "EQUIVALENT" if np.allclose(local, reference, atol=1e-5, rtol=1e-5) else "MISMATCH",
                     "max_abs_error": max_error,
                     "mean_abs_error": float(delta.mean())}
        self.preprocess_spec.write(path, equivalence=audit)
        return audit


class GlobalIdentityBackend:
    """Offline wrapper for a verified local WildlifeTools/MegaDescriptor model."""

    def __init__(self, model: Any | None = None, *, runtime: MegaDescriptorRuntime | None = None,
                 preprocess_spec: IdentityPreprocessSpec | None = None,
                 model_hash: str | None = None, preprocess_fingerprint: str = "unresolved",
                 device: str = "cpu"):
        if runtime is None and model is None:
            raise MissingAssetError("MegaDescriptor model must be supplied from a verified local checkpoint")
        if runtime is None:
            runtime = MegaDescriptorRuntime(model, preprocess_spec=preprocess_spec)
        self.runtime = runtime
        self.model = runtime.backbone
        self.preprocess_spec = runtime.preprocess_spec
        self.model_hash = model_hash
        self.preprocess_fingerprint = preprocess_fingerprint
        self.device = device
        self.fingerprint = hashlib.sha256(
            f"{model_hash}:{preprocess_fingerprint}:{self.preprocess_spec.fingerprint}:"
            f"{type(self.model).__module__}.{type(self.model).__qualname__}".encode()
        ).hexdigest()[:20]

    def encode(self, images: Any) -> DescriptorBatch:
        """Encode neutral images. This method does not accept identity labels."""
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise DependencyUnavailableError("PyTorch is required for MegaDescriptor inference") from exc
        if hasattr(self.runtime, "eval"):
            self.runtime.eval()
        with torch.no_grad():
            output = self.runtime(images)
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
        """Construct a model without ``hf-hub``/``pretrained=True`` network access.

        A MegaDescriptor checkpoint is not interchangeable with an ImageNet
        Swin checkpoint.  Therefore a local model config is mandatory (or must
        sit next to the checkpoint as ``config.json``) and its architecture and
        feature dimension are checked before any state dict is loaded.
        """
        if not checkpoint.is_file():
            raise MissingAssetError(f"missing identity checkpoint: {checkpoint}")
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise MissingAssetError("identity checkpoint SHA-256 mismatch")
        config_path = config_path or checkpoint.with_name("config.json")
        if not config_path.is_file():
            raise MissingAssetError("MegaDescriptor config.json is required to verify architecture")
        try:
            import json
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ValidationError(f"invalid identity model config: {config_path}") from exc
        configured_architecture = config.get("architecture")
        if configured_architecture != architecture:
            raise ValidationError(
                f"identity architecture mismatch: config={configured_architecture!r}, requested={architecture!r}"
            )
        if config.get("num_classes", 0) not in (0, None):
            raise ValidationError("identity checkpoint config must have num_classes=0")
        configured_features = config.get("num_features")
        if configured_features is not None and int(configured_features) <= 0:
            raise ValidationError("identity checkpoint config must declare a positive num_features")
        try:
            import torch
            import timm
        except ImportError as exc:  # pragma: no cover
            raise DependencyUnavailableError("offline identity loading requires torch and timm") from exc
        model = timm.create_model(architecture, pretrained=False, num_classes=0)
        if configured_features is not None and int(getattr(model, "num_features", 0)) != int(configured_features):
            raise ValidationError(
                f"identity feature dimension mismatch: config={configured_features}, "
                f"runtime={getattr(model, 'num_features', None)}"
            )
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
        # MegaDescriptor-T-224 was published with an older timm Swin layout
        # where PatchMerging is named on the preceding stage.  Current timm
        # attaches that module to the following stage.  Apply this explicit,
        # shape-checked rename only for this known architecture; any other
        # mismatch remains a hard validation error below.
        if architecture == "swin_tiny_patch4_window7_224":
            remapped = {}
            for key, value in cleaned.items():
                parts = key.split(".")
                if len(parts) >= 4 and parts[0] == "layers" and parts[2] == "downsample":
                    try:
                        stage = int(parts[1])
                    except ValueError:
                        stage = -1
                    if 0 <= stage <= 2:
                        parts[1] = str(stage + 1)
                        key = ".".join(parts)
                remapped[key] = value
            cleaned = remapped
        # Relative-position indices and attention masks are deterministic
        # buffers whose persistence changed between timm releases.  They are
        # regenerated by the current model and are safe to omit; arbitrary
        # unexpected tensors are still rejected below.
        regenerated = {key for key in cleaned
                       if key.endswith("relative_position_index") or key.endswith("attn_mask")}
        cleaned = {key: value for key, value in cleaned.items() if key not in regenerated}
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        non_head_missing = [k for k in missing if not any(k.startswith(p) for p in ("head.", "fc.", "classifier."))]
        non_head_unexpected = [k for k in unexpected if not any(k.startswith(p) for p in ("head.", "fc.", "classifier."))]
        if non_head_missing or non_head_unexpected:
            raise MissingAssetError(f"checkpoint backbone mismatch: missing={non_head_missing}, unexpected={non_head_unexpected}")
        resolved_device = device
        if device == "auto":
            resolved_device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(resolved_device).eval()
        preprocess_spec = IdentityPreprocessSpec.official_t224(config)
        preprocess_fingerprint = preprocess_spec.fingerprint
        runtime = MegaDescriptorRuntime(model, preprocess_spec=preprocess_spec)
        # Persist the exact local contract next to the verified model.  Do not
        # erase a previously completed numerical equivalence audit merely
        # because a later offline process reloads the checkpoint.
        audit_path = checkpoint.with_name("megadescriptor_preprocess_spec.json")
        equivalence: dict[str, Any] = {"status": "NOT_RUN_NO_REFERENCE"}
        if audit_path.is_file():
            try:
                previous = json.loads(audit_path.read_text(encoding="utf-8"))
                if (previous.get("spec", {}).get("fingerprint") == preprocess_spec.fingerprint
                        and isinstance(previous.get("equivalence"), dict)):
                    equivalence = dict(previous["equivalence"])
            except (OSError, ValueError, TypeError):
                pass
        preprocess_spec.write(audit_path, equivalence=equivalence)
        return cls(runtime=runtime, model_hash=digest, preprocess_fingerprint=preprocess_fingerprint, device=resolved_device)


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
