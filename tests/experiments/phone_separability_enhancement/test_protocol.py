from __future__ import annotations

from pathlib import Path

import yaml

from scripts.experiments.phone_separability_enhancement.config import protocol_snapshot, validate_run_id


def test_config_has_fixed_protocol_and_parent() -> None:
    path = Path("scripts/configs/phone_separability_enhancement_v2.yaml")
    config = yaml.safe_load(path.read_text())
    assert config["protocol_id"] == "phone_separability_enhancement_v2"
    assert config["measurement_version"] == "signed_margin_fixed_support_v2"
    assert config["parent_run"]


def test_run_id_rejects_path_traversal() -> None:
    validate_run_id("ok_run-1")
    try:
        validate_run_id("../bad")
    except ValueError:
        pass
    else:
        raise AssertionError("path traversal must be rejected")


def test_protocol_snapshot_contains_input_and_code_hashes() -> None:
    path = Path("scripts/configs/phone_separability_enhancement_v2.yaml")
    config = yaml.safe_load(path.read_text())
    snapshot = protocol_snapshot(config, path, repo_root=Path.cwd())
    assert snapshot["input_hashes"]["manifest"]
    assert snapshot["code_hashes"]
