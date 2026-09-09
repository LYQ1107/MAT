import pytest

from mat.config.identity import IdentityMatchingConfig


def test_matching_config_requires_the_implemented_solver_and_parses_values():
    config = IdentityMatchingConfig.from_mapping({
        "mode": "global_only", "solver": "deterministic_greedy", "top_k": 4,
        "accept_threshold": None, "threshold_calibration": "development",
        "calibration_objective": "macro_f1",
    })
    assert config.top_k == 4
    assert config.accept_threshold is None
    with pytest.raises(ValueError):
        IdentityMatchingConfig.from_mapping({"solver": "hungarian"})
