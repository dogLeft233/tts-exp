from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/experiments/world_phone_transfer_20261001_overlay.py"
SPEC = importlib.util.spec_from_file_location("world_phone_transfer_20261001_overlay_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
world = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = world
SPEC.loader.exec_module(world)


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _fingerprint(payload: dict) -> str:
    return world.fingerprint(payload)


def test_overlay_validation_preserves_original_gate_and_builds_physical_view(tmp_path):
    mfa_out = tmp_path / "mfa"
    mfa_out.mkdir()
    sample_id = "S0064_BAC009S0064W0121"
    source_identity = "source-id"
    transcript = "侵蚀"
    transcript_sha = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
    audio_sha = {"natural": "n-audio", "tts": "t-audio"}
    old_grids = {}
    new_grids = {}
    labs = {}
    probe_conditions = {}
    overlay_conditions = {}
    qualification_conditions = {}
    producer_conditions = {}

    for condition in ("natural", "tts"):
        old_grid = tmp_path / f"old_{condition}.TextGrid"
        old_grid.write_text(f"old-{condition}\n", encoding="utf-8")
        new_grid = tmp_path / f"new_{condition}.TextGrid"
        new_grid.write_text(f"new-{condition}\n", encoding="utf-8")
        lab = tmp_path / f"new_{condition}.lab"
        lab.write_text("侵蚀\n", encoding="utf-8")
        old_grids[condition] = old_grid
        new_grids[condition] = new_grid
        labs[condition] = lab
        old_sha = hashlib.sha256(old_grid.read_bytes()).hexdigest()
        new_sha = hashlib.sha256(new_grid.read_bytes()).hexdigest()
        lab_sha = hashlib.sha256(lab.read_bytes()).hexdigest()
        producer_conditions[condition] = {
            "source_audio_sha256": audio_sha[condition],
            "expected_textgrid_path": str(old_grid.resolve()),
            "normalized_lab_text": "侵 蚀",
        }
        qualification_conditions[condition] = {"qualified": False, "textgrid_sha256": old_sha, "errors": ["unknown_or_oov_phone"]}
        probe_conditions[condition] = {
            "source_audio_sha256": audio_sha[condition],
            "lab_sha256": lab_sha,
            "probe_lab_text": "侵蚀",
            "textgrid_sha256": new_sha,
            "qualified_by_frozen_structure_rules": True,
            "unknown_oov_phone_count": 0,
        }
        overlay_conditions[condition] = {
            "source_audio_sha256": audio_sha[condition],
            "original_textgrid_path": str(old_grid.resolve()),
            "original_textgrid_sha256": old_sha,
            "physical_lab_path": str(lab.resolve()),
            "physical_lab_sha256": lab_sha,
            "physical_lab_text": "侵蚀",
            "physical_textgrid_path": str(new_grid.resolve()),
            "physical_textgrid_sha256": new_sha,
        }

    eligibility_path = mfa_out / "evaluation_endpoint_eligibility.json"
    eligibility_path.write_text("{}\n", encoding="utf-8")
    annex_path = tmp_path / "alignment-overlay.md"
    annex_path.write_text("frozen overlay\n", encoding="utf-8")
    bindings_path = tmp_path / "execution-bindings.md"
    bindings_path.write_text("frozen bindings\n", encoding="utf-8")

    producer = {
        "manifest_sha256": "producer-seal",
        "records": {
            sample_id: {
                "id": sample_id,
                "speaker": "S0064",
                "split": "evaluation",
                "source_identity_sha256": source_identity,
                "natural_transcript_sha256": transcript_sha,
                "tts_transcript_sha256": transcript_sha,
                "conditions": producer_conditions,
            }
        },
    }
    protocol = {"manifest_sha256": "protocol-seal"}
    original_qualification = {
        sample_id: {"id": sample_id, "pair_qualified": False, **qualification_conditions}
    }

    probe = {
        "publisher_transcript": transcript,
        "conditions": probe_conditions,
    }
    probe["manifest_sha256"] = _fingerprint(probe)
    probe_path = tmp_path / "probe_manifest.json"
    _write_json(probe_path, probe)

    row = {
        "id": sample_id,
        "speaker": "S0064",
        "split": "evaluation",
        "source_identity_sha256": source_identity,
        "publisher_transcript": transcript,
        "physical_lab_text": "侵蚀",
        "conditions": overlay_conditions,
    }
    row["record_sha256"] = _fingerprint(row)
    overlay_manifest = {
        "status": "FROZEN_PHYSICAL_ONLY",
        "evaluation_effects_read": False,
        "alignment_overlay_annex_sha256": hashlib.sha256(annex_path.read_bytes()).hexdigest(),
        "execution_bindings_sha256": hashlib.sha256(bindings_path.read_bytes()).hexdigest(),
        "mfa_producer_manifest_sha256": producer["manifest_sha256"],
        "mfa_protocol_manifest_sha256": protocol["manifest_sha256"],
        "original_evaluation_eligibility_raw_sha256": hashlib.sha256(eligibility_path.read_bytes()).hexdigest(),
        "probe_manifest_path": str(probe_path.resolve()),
        "probe_manifest_raw_sha256": hashlib.sha256(probe_path.read_bytes()).hexdigest(),
        "probe_manifest_fingerprint": probe["manifest_sha256"],
        "records": [row],
    }
    overlay_manifest["manifest_sha256"] = _fingerprint(overlay_manifest)
    overlay_path = tmp_path / "overlay.json"
    _write_json(overlay_path, overlay_manifest)

    overlay = world._load_alignment_overlay(
        overlay_path,
        annex_path,
        bindings_path,
        mfa_out,
        producer,
        protocol,
        original_qualification,
    )
    run_records, effective_qualification, original_rows = world._physical_run_views(
        producer, original_qualification, overlay
    )

    assert run_records[sample_id]["conditions"]["natural"]["normalized_lab_text"] == "侵蚀"
    assert run_records[sample_id]["conditions"]["natural"]["expected_textgrid_path"] == str(new_grids["natural"].resolve())
    assert producer["records"][sample_id]["conditions"]["natural"]["expected_textgrid_path"] == str(old_grids["natural"].resolve())
    assert original_rows[sample_id]["pair_qualified"] is False
    assert effective_qualification[sample_id]["pair_qualified"] is True
    assert effective_qualification[sample_id]["original_pair_qualified"] is False
    assert effective_qualification[sample_id]["natural"]["textgrid_sha256"] == hashlib.sha256(new_grids["natural"].read_bytes()).hexdigest()

