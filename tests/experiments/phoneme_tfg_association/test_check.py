from __future__ import annotations

from scripts.experiments.phoneme_tfg_association.check import validate_run
from scripts.experiments.phoneme_tfg_association.protocol import file_sha256, write_json


def test_checker_reports_missing_media_without_self_accepting(tmp_path):
    root = tmp_path / "run"
    protocol = {
        "protocol_id": "phoneme_tfg_association_v1",
        "smoke": True,
        "blocks": [{"source_group": "g0", "pair": ["qwen_cloud", "qwen_local"]}],
        "cells": [{"cell_key": f"g0::s0::{arm}", "sample_id": "s0", "source_group": "g0", "tfg": "wav2lip", "arm": arm} for arm in ("natural", "qwen_cloud", "qwen_local")],
        "video_budget": {"planned_unique_wav2lip_videos": 3},
        "config": {"pair_replicates": 6},
    }
    write_json(root / "00_protocol/protocol.json", protocol)
    result = validate_run(root)
    assert result["status"] == "PASS"
    assert result["counts"]["missing_media_count"] == 3
    assert result["warnings"]


def test_checker_rejects_forbidden_visual_binding_and_accepts_external_hash(tmp_path):
    root = tmp_path / "run"
    visual = tmp_path / "external.mp4"
    visual.write_bytes(b"external visual")
    reference = {
        "visual_source": str(visual),
        "visual_source_sha256": file_sha256(visual),
        "visual_source_frame_policy": "first_frame_only",
        "visual_source_used_for_generation": "one_frame_repeated_for_each_mel_chunk",
    }
    protocol = {
        "protocol_id": "phoneme_tfg_association_v1",
        "smoke": True,
        "blocks": [{"source_group": "g0", "sample_id": "s0", "pair": ["qwen_cloud", "qwen_local"], "reference": reference}],
        "cells": [],
        "video_budget": {"planned_unique_wav2lip_videos": 0},
        "config": {"pair_replicates": 6},
    }
    write_json(root / "00_protocol/protocol.json", protocol)
    result = validate_run(root)
    assert result["status"] == "PASS"
    forbidden = tmp_path / "lrs3" / "source.mp4"
    forbidden.parent.mkdir()
    forbidden.write_bytes(b"forbidden visual")
    protocol["blocks"][0]["reference"]["visual_source"] = str(forbidden)
    protocol["blocks"][0]["reference"]["visual_source_sha256"] = file_sha256(forbidden)
    write_json(root / "00_protocol/protocol.json", protocol)
    result = validate_run(root)
    assert result["status"] == "FAIL"
    assert any("forbidden or missing external visual source" in error for error in result["errors"])
    visual.unlink()
