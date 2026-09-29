from __future__ import annotations

import json

from scripts.experiments.check_lrs3_phone_rules import check_run


def _write(path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        path.write_text(value, encoding="utf-8")
    else:
        path.write_text(json.dumps(value), encoding="utf-8")


def test_checker_accepts_a_complete_scientific_negative_result(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write(run_dir / "00_audit" / "cohort.json", {"status": "COMPLETE", "failures": [], "record_count": 0})
    _write(run_dir / "protocol.json", {"schema_version": 1})
    _write(run_dir / "status.json", {"status": "COMPLETE"})
    _write(run_dir / "03_stage_a" / "decision.json", {"science_decision": "NO_CLEAR_ADVANTAGE"})
    _write(run_dir / "03_stage_a" / "statistics.json", {"models": {}})
    _write(run_dir / "03_stage_a" / "per_pair.jsonl", "")
    _write(run_dir / "05_stage_b" / "decision.json", {"status": "SKIPPED_GATE_NOT_PASSED", "science_decision": "SKIPPED_GATE_NOT_PASSED"})
    result = check_run(run_dir)
    assert result["engineering_pass"] is True
    assert result["science_decision"] == "SKIPPED_GATE_NOT_PASSED"
    assert result["waveform"]["status"] == "NOT_APPLICABLE"


def test_checker_accepts_empty_smoke_stage_a_statistics(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write(run_dir / "00_audit" / "cohort.json", {"status": "COMPLETE", "failures": [], "record_count": 2})
    _write(run_dir / "protocol.json", {"schema_version": 1})
    _write(run_dir / "status.json", {"status": "COMPLETE"})
    _write(run_dir / "03_stage_a" / "decision.json", {"science_decision": "SMOKE_NO_SCIENTIFIC_DECISION"})
    _write(
        run_dir / "03_stage_a" / "statistics.json",
        {
            "models": {
                "hubert": {
                    "n_pairs": 0,
                    "n_source_groups": 0,
                    "mean_natural": None,
                    "mean_tts": None,
                    "effect_tts_minus_natural": {
                        "estimate": None,
                        "ci_low": None,
                        "ci_high": None,
                        "n_groups": 0,
                        "seed": 20260920,
                        "draws": 10000,
                    },
                }
            }
        },
    )
    _write(run_dir / "03_stage_a" / "per_pair.jsonl", "")
    _write(run_dir / "05_stage_b" / "decision.json", {"status": "SKIPPED_GATE_NOT_PASSED"})

    result = check_run(run_dir)

    assert result["engineering_pass"] is True
    assert result["recomputed"]["models"]["hubert"]["mean_natural"] is None


def test_checker_rejects_orphan_rule_wavs_after_negative_stage_a(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write(run_dir / "00_audit" / "cohort.json", {"status": "COMPLETE", "failures": [], "record_count": 0})
    _write(run_dir / "protocol.json", {"schema_version": 1})
    _write(run_dir / "status.json", {"status": "COMPLETE"})
    _write(run_dir / "03_stage_a" / "decision.json", {"science_decision": "NO_CLEAR_ADVANTAGE"})
    _write(run_dir / "03_stage_a" / "statistics.json", {"models": {}})
    _write(run_dir / "03_stage_a" / "per_pair.jsonl", "")
    _write(run_dir / "05_stage_b" / "decision.json", {"status": "SKIPPED_GATE_NOT_PASSED"})
    orphan = run_dir / "04_rules" / "wav" / "spectral_drc" / "orphan.wav"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_bytes(b"not a valid scientific artifact")
    try:
        check_run(run_dir)
    except RuntimeError as exc:
        assert "B WAVs exist" in str(exc)
    else:
        raise AssertionError("orphan B audio was not detected")
