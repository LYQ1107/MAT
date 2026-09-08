"""Optional integration check for the verified local MegaDescriptor asset."""

from pathlib import Path
import os
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mat.backends.wildlife import GlobalIdentityBackend
from mat.core.types import SpeciesSpec
from mat.models.part_encoder import PartAwareIdentityEncoder, PosePartCropper


def test_real_megadescriptor_pose_rois_have_part_evidence():
    work = Path(os.environ.get("MAT_WORK_ROOT", "/data2/usr_for_deadline/MAT_workspace"))
    checkpoint = work / "assets" / "identity" / "megadescriptor_t_224" / "pytorch_model.bin"
    config = checkpoint.with_name("config.json")
    if not checkpoint.is_file() or not config.is_file():
        pytest.skip("verified MegaDescriptor-T-224 asset is not locally imported")
    try:
        backend = GlobalIdentityBackend.from_local(checkpoint, config_path=config)
    except Exception as exc:
        pytest.skip(f"local timm runtime unavailable: {type(exc).__name__}")
    names = ("nose", "left_eye", "right_eye", "left_ear", "right_ear", "spine1",
             "spine2", "spine3", "spine4", "spine5", "tail1", "tail2", "tail3", "tail4")
    species = SpeciesSpec("gerbil", "gerbil-14-v1", names, ((5, 0),),
                          {"head": (0, 1, 2, 3, 4, 5), "rear_trunk": (7, 8, 9)},
                          None, ("top",), "sleap_gerbils")
    cropper = PosePartCropper(species, output_size=64)
    encoder = PartAwareIdentityEncoder(backend, species, cropper=cropper)
    image = torch.zeros((1, 3, 128, 128), dtype=torch.uint8)
    image[:, :, 8:52, 8:60] = [220, 100, 50]
    image[:, :, 58:116, 64:120] = [20, 180, 210]
    boxes = torch.tensor([[4.0, 4.0, 124.0, 124.0]])
    keypoints = torch.tensor([[[15, 20], [25, 16], [35, 16], [20, 28], [34, 28], [28, 38],
                               [54, 60], [70, 72], [84, 82], [98, 92], [108, 100], [114, 106],
                               [118, 112], [120, 116]]], dtype=torch.float32)
    scores = torch.ones((1, 14))
    valid = torch.ones((1, 14), dtype=torch.bool)
    original = encoder(image, boxes, keypoints, scores, valid)
    occluded = image.clone(); occluded[:, :, 8:52, 8:60] = 0
    altered = encoder(occluded, boxes, keypoints, scores, valid)
    assert not torch.allclose(original[1][:, 0], altered[1][:, 0])
    assert not torch.allclose(original[1][:, 0], original[1][:, 1])
