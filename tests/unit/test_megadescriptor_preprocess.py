import numpy as np
import pytest

torch = pytest.importorskip("torch")
torchvision = pytest.importorskip("torchvision")
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from mat.backends.wildlife import IdentityPreprocessSpec, MegaDescriptorRuntime, GlobalIdentityBackend


class _Backbone(torch.nn.Module):
    num_features = 4

    def forward(self, x):
        # Keep the test independent of a checkpoint while proving that the
        # runtime receives normalized BCHW tensors.
        return x.mean(dim=(2, 3))[:, :1].repeat(1, 4)


def test_numpy_bhwc_and_torch_bchw_are_identical_and_official_bicubic():
    spec = IdentityPreprocessSpec.official_t224({"pretrained_cfg": {
        "input_size": [3, 224, 224], "interpolation": "bicubic",
        "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}})
    images = np.zeros((2, 11, 17, 3), dtype=np.uint8)
    images[0, :, :, 0] = np.arange(17, dtype=np.uint8)[None, :]
    images[1, 2:8, 3:12] = [10, 80, 240]
    numpy_out = spec.apply(images)
    torch_out = spec.apply(torch.from_numpy(images).permute(0, 3, 1, 2))
    assert numpy_out.shape == (2, 3, 224, 224)
    assert numpy_out.dtype == torch.float32
    assert torch.allclose(numpy_out, torch_out, atol=0, rtol=0)
    # Independent public torchvision reference: Resize acts on uint8 before
    # ToTensor scaling, which catches accidental pre-interpolation /255.
    reference = torch.stack([
        TF.resize(item, [224, 224], interpolation=InterpolationMode.BICUBIC, antialias=True)
        for item in torch.from_numpy(images).permute(0, 3, 1, 2)
    ]).float() / 255.0
    reference = (reference - torch.tensor(spec.mean).view(1, 3, 1, 1)) / torch.tensor(spec.std).view(1, 3, 1, 1)
    assert torch.allclose(numpy_out, reference, atol=0, rtol=0)

    runtime = MegaDescriptorRuntime(_Backbone(), spec)
    backend = GlobalIdentityBackend(runtime=runtime, model_hash="fixture", preprocess_fingerprint=spec.fingerprint)
    encoded = backend.encode(images)
    assert encoded.global_features.shape == (2, 4)
    assert np.allclose(np.linalg.norm(encoded.global_features, axis=1), 1.0)


def test_preprocess_audit_records_equivalence(tmp_path):
    spec = IdentityPreprocessSpec()
    runtime = MegaDescriptorRuntime(_Backbone(), spec)
    images = np.full((1, 4, 5, 3), 128, dtype=np.uint8)
    reference = lambda x: spec.apply(x)
    result = runtime.write_preprocess_audit(tmp_path / "preprocess.json", images=images,
                                            reference_transform=reference)
    assert result["status"] == "EQUIVALENT"
    assert '"status": "EQUIVALENT"' in (tmp_path / "preprocess.json").read_text()
