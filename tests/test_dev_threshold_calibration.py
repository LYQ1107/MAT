import numpy as np
import pytest

from mat.identity.calibration import calibrate_accept_threshold


def test_development_threshold_uses_standard_metrics_and_deterministic_ties():
    result = calibrate_accept_threshold(
        np.asarray([0.95, 0.90, 0.60, 0.55, 0.40, 0.10]),
        ["A", "A", "B", "A", "C", "D"],
        ["A", "A", "B", "B", "C", "D"],
        identity_labels=("A", "B", "C", "D"),
    )
    assert result.split_role == "development"
    assert result.threshold == pytest.approx(0.1 - 1e-12)
    assert result.macro_f1 is not None
    assert result.candidate_count == 8


def test_threshold_calibration_rejects_nonfinite_or_mismatched_inputs():
    with pytest.raises(ValueError):
        calibrate_accept_threshold(np.asarray([0.5, np.nan]), ["A", "A"], ["A", "A"], identity_labels=("A",))
    with pytest.raises(ValueError):
        calibrate_accept_threshold(np.asarray([0.5]), ["A", "A"], ["A"], identity_labels=("A",))
