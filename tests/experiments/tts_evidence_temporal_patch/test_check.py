from __future__ import annotations

from pathlib import Path

import pytest

from scripts.experiments.tts_evidence_temporal_patch.check import validate_run
from scripts.experiments.tts_evidence_temporal_patch.protocol import PROTOCOL_ID, RunPaths, finalize_artifact, write_json


def _minimal_inventory() -> dict:
    cells = []
    for index in range(108):
        cells.append({"cell_key": f"g{index % 36}::s{index}::natural", "source_group": f"g{index % 36}", "sample_id": f"s{index}", "arm": "natural"})
    return finalize_artifact({"schema_version": 1, "protocol_id": PROTOCOL_ID, "video_cells": 108, "source_group_count": 36, "cells": cells})


def test_check_accepts_engineering_run_but_keeps_human_pending(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    paths.ensure_dirs()
    inventory = _minimal_inventory()
    protocol = finalize_artifact({"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "FROZEN", "design": {"lambda": 0.5}})
    write_json(paths.protocol / "parent_inventory.json", inventory)
    write_json(paths.protocol / "protocol.json", protocol)
    result = validate_run(tmp_path)
    assert result["status"] == "INCOMPLETE"
    assert result["scientific_status"] == "PENDING_HUMAN"


def test_check_rejects_tampered_patch_lambda(tmp_path: Path) -> None:
    paths = RunPaths(tmp_path)
    paths.ensure_dirs()
    write_json(paths.protocol / "parent_inventory.json", _minimal_inventory())
    write_json(paths.protocol / "protocol.json", finalize_artifact({"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "FROZEN", "design": {"lambda": 0.5}}))
    receipt = finalize_artifact({"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "complete", "latent_layer": "audio_encoder.output", "lambda": 0.4, "hook_removed": True, "outputs": {}})
    write_json(paths.patch_video / "x" / "receipt.json", receipt)
    with pytest.raises(Exception):
        validate_run(tmp_path)
