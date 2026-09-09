from pathlib import Path
import yaml


def test_b1_configs_share_frozen_roles_and_backbone_with_b0():
    root = Path(__file__).parents[1] / "configs" / "experiments" / "gerbils"
    b0 = yaml.safe_load((root / "B0_oracle_strict.yaml").read_text())
    b1 = yaml.safe_load((root / "B1_oracle_part_strict.yaml").read_text())
    for key in ("protocol_file", "reference_role", "development_role", "sealed_test_role"):
        assert b1[key] == b0[key]
    assert b1["identity"]["backend"] == b0["identity"]["backend"]
    assert b1["identity"]["descriptor"] == b0["identity"]["descriptor"]
    assert b1["matching"]["solver"] == b0["matching"]["solver"]
    assert b1["diagnostic_only"] is True
