from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.experiments.lrs3_phone_rules import audit_inputs, run_stage_a
from scripts.experiments.lrs3_phone_rules_worker import sha256_file, write_pcm16

TEXTGRID = '''File type = "ooTextFile"
Object class = "TextGrid"
xmin = 0
xmax = 0.4
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 0.4
        intervals: size = 3
        intervals [1]: xmin = 0 xmax = 0.2 text = "a"
        intervals [2]: xmin = 0.2 xmax = 0.25 text = "sil"
        intervals [3]: xmin = 0.25 xmax = 0.4 text = "b"
'''


def _make_fixture(tmp_path: Path) -> tuple[dict, dict, Path]:
    textgrid = tmp_path / "sample.TextGrid"
    textgrid.write_text(TEXTGRID, encoding="utf-8")
    natural_paths = []
    tts_paths = []
    for name in ("natural_train", "tts_train", "natural_eval", "tts_eval"):
        path = tmp_path / f"{name}.wav"
        write_pcm16(path, np.zeros(6400, dtype=np.int16))
        (natural_paths if name.startswith("natural") else tts_paths).append(path)
    tokens = [
        {"token": "a", "label": "a", "start_s": 0.0, "end_s": 0.2, "duration_s": 0.2, "silence": False},
        {"token": "sil", "label": "sil", "start_s": 0.2, "end_s": 0.25, "duration_s": 0.05, "silence": True},
        {"token": "b", "label": "b", "start_s": 0.25, "end_s": 0.4, "duration_s": 0.15, "silence": False},
    ]
    records = []
    token_records = {}
    for index, (split, group, natural, tts) in enumerate([
        ("train", "train_group", natural_paths[0], tts_paths[0]),
        ("evaluation", "eval_group", natural_paths[1], tts_paths[1]),
    ]):
        sample_id = f"sample_{index}"
        record = {
            "sample_id": sample_id,
            "source_group": group,
            "protocol_split": split,
            "transcript": "A B",
            "mfa_transcript": "A B",
            "transcript_sha256": "fixture",
            "natural_audio_path": str(natural),
            "natural_audio_sha256": sha256_file(natural),
            "tts_audio_path": str(tts),
            "tts_audio_sha256": sha256_file(tts),
            "tts_audio_origin": "fixture",
        }
        records.append(record)
        token_records[sample_id] = {
            "natural": {"textgrid": str(textgrid), "textgrid_sha256": sha256_file(textgrid), "tokens": tokens},
            "tts": {"textgrid": str(textgrid), "textgrid_sha256": sha256_file(textgrid), "tokens": tokens},
        }
    manifest = {"records": records}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    tokens_payload = {"source_manifest_sha256": sha256_file(manifest_path), "records": token_records}
    tokens_path = tmp_path / "tokens.json"
    tokens_path.write_text(json.dumps(tokens_payload), encoding="utf-8")
    config = {
        "source": {
            "manifest": str(manifest_path),
            "manifest_sha256": sha256_file(manifest_path),
            "tokens": str(tokens_path),
            "tokens_sha256": sha256_file(tokens_path),
            "sample_rate": 16000,
            "expected_records": 2,
            "expected_train_records": 1,
            "expected_eval_records": 1,
            "expected_train_source_groups": 1,
            "expected_eval_source_groups": 1,
        }
    }
    return config, {"records": records}, textgrid


def test_audit_binds_full_sample_ids_and_audio_textgrids(tmp_path) -> None:
    config, _, _ = _make_fixture(tmp_path)
    run_dir = tmp_path / "run"
    cohort = audit_inputs(config, run_dir)
    assert cohort["status"] == "COMPLETE"
    assert cohort["record_count"] == 2
    assert cohort["evaluation_source_groups"] == ["eval_group"]
    assert cohort["records"][0]["natural"]["textgrid_token_signature"]


def test_stage_a_negative_result_does_not_create_candidate_wavs(tmp_path, monkeypatch) -> None:
    config, _, _ = _make_fixture(tmp_path)
    config.update({
        "models": {
            "primary": {"key": "hubert", "layer": 6},
            "cross_encoder": {"key": "xlsr", "layer": 10},
            "diagnostic_layers": {},
        },
        "probe": {
            "min_tokens_per_label": 1,
            "min_groups_per_label": 1,
            "min_labels_per_pair": 2,
            "min_tokens_per_pair_side": 2,
            "min_speech_coverage": 1.0,
            "min_eval_source_groups": 1,
            "stage_a_min_effect": 0.02,
            "bootstrap_draws": 50,
            "bootstrap_seed": 20260920,
        },
    })
    run_dir = tmp_path / "run"
    cohort = audit_inputs(config, run_dir)

    def fake_extract(config, run_dir, rows, model_cfg, *, smoke):
        result = {}
        for row in rows:
            sample = {}
            for condition in ("natural", "tts"):
                sample[condition] = [
                    {"token_id": f"{row['sample_id']}:0", "label": "a", "speech": True, "valid": True, "embedding": [1.0, 0.0], "condition": condition, "source_group": row["source_group"]},
                    {"token_id": f"{row['sample_id']}:1", "label": "b", "speech": True, "valid": True, "embedding": [0.0, 1.0], "condition": condition, "source_group": row["source_group"]},
                ]
            result[row["sample_id"]] = {"sample_id": row["sample_id"], "source_group": row["source_group"], "protocol_split": row["protocol_split"], **sample}
        return result, {"model_key": model_cfg["key"]}

    monkeypatch.setattr("scripts.experiments.lrs3_phone_rules._extract_model_records", fake_extract)
    result = run_stage_a(config, run_dir, cohort, smoke=False)
    assert result["decision"]["science_decision"] == "INSUFFICIENT_SUPPORT"
    assert not list((run_dir / "04_rules").glob("**/*.wav")) if (run_dir / "04_rules").exists() else True
    assert json.loads((run_dir / "05_stage_b" / "decision.json").read_text())["status"] == "SKIPPED_GATE_NOT_PASSED"
