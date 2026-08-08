from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location(
    "prepare_mdc_english_pairs", SCRIPTS / "prepare_mdc_english_pairs.py"
)
assert spec is not None and spec.loader is not None
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _record(sample_id: str, path: str, *, author: str = "a") -> dict:
    return {
        "sample_id": sample_id,
        "language_code": "en",
        "file_path": path,
        "text": f"Text for {sample_id}",
        "source_author_id": author,
        "source_row_index": int(sample_id.split("_")[1]),
    }


def test_load_english_manifest_exposes_language_metadata(tmp_path: Path):
    path = tmp_path / "mdc.json"
    path.write_text(
        json.dumps({
            "languages": {"en": {"dataset_id": "dataset-1", "archive_filename": "archive.tar.gz"}},
            "samples": [_record("en_001", "source.wav")],
        }),
        encoding="utf-8",
    )
    records, meta = module.load_english_manifest(path)
    assert records[0]["sample_id"] == "en_001"
    assert meta["dataset_id"] == "dataset-1"
    assert meta["archive_filename"] == "archive.tar.gz"


def test_select_records_preserves_stable_string_ids():
    records = [_record("en_002", "2.wav"), _record("en_001", "1.wav")]
    selected = module.select_records(records)
    assert [row["sample_id"] for row in selected] == ["en_001", "en_002"]


def test_select_records_rejects_duplicate_or_missing_ids():
    records = [_record("en_001", "1.wav")]
    try:
        module.select_records(records, ["en_001", "en_001"])
    except ValueError as exc:
        assert "duplicate" in str(exc)
    else:
        raise AssertionError("duplicate IDs should fail")
    try:
        module.select_records(records, ["en_002"])
    except ValueError as exc:
        assert "missing" in str(exc)
    else:
        raise AssertionError("missing IDs should fail")


def test_full_run_requires_exact_50_ids():
    with_ids = [f"en_{index:03d}" for index in range(1, 50)]
    try:
        module.require_complete_sample_set(with_ids)
    except ValueError as exc:
        assert "en_001..en_050" in str(exc)
    else:
        raise AssertionError("incomplete full run should fail")


def test_full_run_accepts_exact_50_ids():
    module.require_complete_sample_set(module.EXPECTED_SAMPLE_IDS)


def test_prepare_natural_pairs_rebuilds_stale_canonical_audio(tmp_path: Path):
    source = tmp_path / "source.wav"
    sf.write(source, np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    manifest = tmp_path / "mdc.json"
    manifest.write_text(json.dumps({"samples": [_record("en_001", "source.wav")]}), encoding="utf-8")
    run_id = "stale_run"
    destination = tmp_path / "runs" / run_id / "00_pairs" / "natural" / "en_001.wav"
    destination.parent.mkdir(parents=True)
    sf.write(destination, np.full(16000, 0.2, dtype=np.float32), 16000, subtype="PCM_16")
    result = module.prepare_natural_pairs(
        tmp_path, manifest, run_id, sample_ids=["en_001"], ffmpeg_bin="ffmpeg"
    )
    assert result["records"][0]["canonical_qc"]["rms"] < 0.15


def test_prepare_natural_pairs_writes_manifest_and_preserves_null_unavailable_fields(tmp_path: Path):
    source = tmp_path / "source.wav"
    sf.write(source, np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    manifest = tmp_path / "mdc.json"
    manifest.write_text(
        json.dumps({"samples": [_record("en_001", "source.wav")]}),
        encoding="utf-8",
    )
    result = module.prepare_natural_pairs(
        repo=tmp_path,
        manifest_path=manifest,
        run_id="test_run",
        sample_ids=["en_001"],
        ffmpeg_bin="ffmpeg",
    )
    assert result["sample_keys"] == ["en_001"]
    assert result["complete_natural"] is False
    assert result["complete_pairs"] is False
    record = result["records"][0]
    assert record["paired_key"] == "en_001"
    assert record["utterance_id"] == "mdc_tts/en_001/natural"
    assert record["condition"] == "natural"
    assert record["tts_status"] == "pending"
    assert record["canonical_qc"]["sample_rate_hz"] == 16000
    assert record["canonical_qc"]["subtype"] == "PCM_16"
    assert record["canonical_qc"]["clipped_fraction"] == 0.0
    assert record["source_dataset_id"] is None
    assert record["source_archive"] is None
    assert record["source_audio_id"] is None
    written = json.loads(
        (tmp_path / "runs/test_run/00_pairs/pair_manifest.json").read_text()
    )
    assert written["source_manifest_sha256"] == module.sha256_file(manifest)
    assert written["records"][0]["sample_key"] == "en_001"


def test_prepare_natural_pairs_falls_back_to_language_provenance(tmp_path: Path):
    source = tmp_path / "source.wav"
    sf.write(source, np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    manifest = tmp_path / "mdc.json"
    manifest.write_text(
        json.dumps({
            "languages": {"en": {"dataset_id": "dataset-1", "archive_filename": "archive.tar.gz"}},
            "samples": [_record("en_001", "source.wav")],
        }),
        encoding="utf-8",
    )
    result = module.prepare_natural_pairs(
        tmp_path, manifest, "provenance_run", sample_ids=["en_001"], ffmpeg_bin="ffmpeg"
    )
    record = result["records"][0]
    assert record["source_dataset_id"] == "dataset-1"
    assert record["source_archive"] == "archive.tar.gz"
    assert record["source_archive_member"] is None
    assert record["source_audio_id"] is None
