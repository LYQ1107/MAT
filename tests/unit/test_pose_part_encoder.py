import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mat.core.types import SpeciesSpec
from mat.models.part_encoder import PartAwareIdentityEncoder, PosePartCropper


def _species():
    return SpeciesSpec(
        "fixture", "v1", ("a", "b", "c", "d"), ((0, 1), (1, 2), (2, 3)),
        {"head": (0, 1), "tail": (2, 3)}, None, ("top",), "fixture",
    )


def test_pose_crops_are_distinct_and_coordinates_are_detached():
    species = _species()
    images = torch.zeros((1, 3, 32, 32), dtype=torch.float32)
    images[:, :, :16, :16] = 1.0
    images[:, 0, 16:, 16:] = 0.5
    boxes = torch.tensor([[0.0, 0.0, 32.0, 32.0]], requires_grad=True)
    keypoints = torch.tensor([[[4.0, 4.0], [12.0, 12.0], [24.0, 24.0], [28.0, 28.0]]], requires_grad=True)
    scores = torch.ones((1, 4))
    valid = torch.ones((1, 4), dtype=torch.bool)
    cropper = PosePartCropper(species, output_size=8, keypoint_threshold=0.2)
    batch = cropper(images, boxes, keypoints, scores, valid)
    assert batch.crops.shape == (3, 3, 8, 8)
    assert not torch.allclose(batch.crops[1], batch.crops[2])
    assert batch.part_valid.tolist() == [[True, True]]
    assert torch.all((batch.part_quality >= 0) & (batch.part_quality <= 1))

    backbone = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 8 * 8, 8))
    encoder = PartAwareIdentityEncoder(backbone, 8, 6, species, cropper=cropper)
    global_feature, parts, part_valid, _, _ = encoder(images, boxes, keypoints, scores, valid)
    loss = global_feature.sum() + parts.sum()
    loss.backward()
    assert global_feature.shape == (1, 6)
    assert parts.shape == (1, 2, 6)
    assert part_valid.tolist() == [[True, True]]
    assert boxes.grad is None and keypoints.grad is None


def test_invalid_part_is_masked_not_zero_feature():
    species = _species()
    images = torch.ones((1, 3, 20, 20), dtype=torch.float32)
    cropper = PosePartCropper(species, output_size=8, keypoint_threshold=0.9)
    boxes = torch.tensor([[0.0, 0.0, 20.0, 20.0]])
    keypoints = torch.full((1, 4, 2), float("nan"))
    scores = torch.zeros((1, 4))
    valid = torch.zeros((1, 4), dtype=torch.bool)
    batch = cropper(images, boxes, keypoints, scores, valid)
    assert batch.part_valid.tolist() == [[False, False]]
    assert torch.all(batch.part_quality == 0)
    assert torch.any(batch.crops[1:] != 0)
