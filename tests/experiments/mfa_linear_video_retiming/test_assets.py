from __future__ import annotations

import hashlib
import json
import wave
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.experiments.mfa_linear_video_retiming import assets
from scripts.experiments.mfa_linear_video_retiming.common import ProtocolError


def _write_wav(path: Path, values: np.ndarray, rate: int = 16000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(np.asarray(values, dtype="<i2").tobytes())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def frozen_assets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    ids = ("1", "101", "201")
    rows = []
    results: dict[str, dict[str, object]] = {}
    natural_source: dict[str, str] = {}
    transcripts: dict[str, str] = {}
    legacy_records = []
    audio_root = tmp_path / "natural"
    mfa_root = tmp_path / "mfa"
    for index, sample_id in enumerate(ids):
        n_path = audio_root / f"{sample_id}.wav"
        m_path = mfa_root / f"{sample_id}.wav"
        n_values = np.arange(16000, dtype=np.int16) + index
        m_values = np.arange(16000, dtype=np.int16) - index
        _write_wav(n_path, n_values)
        _write_wav(m_path, m_values)
        speaker = f"spk{sample_id}"
        key = f"aishell1__BAC{sample_id}"
        transcript = f"一句话{sample_id}"
        natural_hash = _sha(n_path)
        natural_source[sample_id] = str(n_path)
        transcripts[sample_id] = transcript
        rows.append({
            "sample_id": int(sample_id),
            "paired_key": key,
            "speaker_id": speaker,
            "split": "valid",
            "transcript": transcript,
            "audio_path": str(n_path),
            "natural_source_sha256": natural_hash,
        })
        results[sample_id] = {
            "sample_id": sample_id,
            "paired_key": key,
            "speaker_id": speaker,
            "audio_path": str(m_path),
            "audio_sha256": _sha(m_path),
            "natural_samples": 16000,
            "output_samples": 16000,
            "exact_natural_length": True,
            "mfa_linear_meta": {"policy": "mfa_linear"},
        }
        legacy_records.append({
            "sample_id": sample_id,
            "paired_key": None,
            "speaker_id": speaker,
            "arm": "natural_raw",
            "audio": str(n_path),
            "audio_sha256": natural_hash,
        })

    mini_path = tmp_path / "mini_manifest.json"
    mini_path.write_text(json.dumps({"schema_version": 1, "records": rows}), encoding="utf-8")
    clean_path = tmp_path / "manifest.json"
    clean_path.write_text(json.dumps({"natural_source": natural_source, "transcripts": transcripts}), encoding="utf-8")
    tokens_path = tmp_path / "tokens.json"
    tokens_path.write_text("{}\n", encoding="utf-8")
    summary_path = mfa_root / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({
        "arm": "mfa_linear",
        "samples_total": 3,
        "samples_ok": 3,
        "failures": [],
        "cohort_manifest": str(mini_path),
        "cohort_manifest_sha256": _sha(mini_path),
        "tokens_path": str(tokens_path),
        "tokens_sha256": _sha(tokens_path),
        "results": results,
    }), encoding="utf-8")
    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    (legacy_dir / "records.json").write_text(json.dumps({"records": legacy_records}), encoding="utf-8")
    (legacy_dir / "summary.json").write_text(json.dumps({"status": "complete"}), encoding="utf-8")

    portrait_rows = {}
    for portrait_id, generation_box, score_box in (
        ("3", [138, 90, 357, 387], [33, 18, 462, 447]),
        ("6", [185, 54, 384, 333], [83, -14, 486, 389]),
        ("9", [136, 97, 341, 381], [33, 28, 443, 438]),
    ):
        image_path = tmp_path / "images" / f"{portrait_id}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8), mode="RGB")
        image.save(image_path)
        rgb_sha = hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
        portrait_rows[portrait_id] = {
            "portrait_id": portrait_id,
            "path": str(image_path),
            "container_sha256": _sha(image_path),
            "rgb_pixel_sha256": rgb_sha,
            "width": 512,
            "height": 512,
            "generation_box_xyxy": generation_box,
            "score_box": {"order": "x1,y1,x2,y2; unbounded square with zero padding", "box": score_box},
            "source_frame_indices": [0],
            "detector": {"selected": {"score": 1.0}},
            "geometry_contract": "sfd_selected_face_v1",
        }
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"portraits": portrait_rows}), encoding="utf-8")

    wav2lip_model = tmp_path / "wav2lip.pth"
    syncnet_model = tmp_path / "syncnet.model"
    wav2lip_model.write_bytes(b"fake wav2lip weights")
    syncnet_model.write_bytes(b"fake syncnet weights")
    monkeypatch.setattr(assets, "EXPECTED_WAV2LIP_SHA256", _sha(wav2lip_model))
    monkeypatch.setattr(assets, "EXPECTED_SYNCNET_SHA256", _sha(syncnet_model))
    monkeypatch.setattr(assets, "EXPECTED_PORTRAIT_3_RGB_SHA256", hashlib.sha256(np.zeros((512, 512, 3), dtype=np.uint8).tobytes()).hexdigest())

    paths = {
        "mfa_summary": str(summary_path),
        "clean_manifest": str(clean_path),
        "mini_manifest": str(mini_path),
        "legacy_eval_dir": str(legacy_dir),
        "portrait_registry": str(registry_path),
        "wav2lip_checkpoint": str(wav2lip_model),
        "syncnet_model": str(syncnet_model),
    }
    return {"config": {"protocol": "mfa_linear_video_retiming_v1", "sample_ids": list(ids), "paths": paths}, "paths": paths}


def test_freeze_inputs_binds_n_m_roles_and_pair_identity(frozen_assets: dict[str, object]) -> None:
    config = frozen_assets["config"]
    assert isinstance(config, dict)

    result = assets.freeze_inputs(config)

    assert result["sample_ids"] == ["1", "101", "201"]
    row = result["records"][1]
    assert row["sample_id"] == "101"
    assert row["paired_key"] == "aishell1__BAC101"
    assert row["audio"]["natural"]["path"].endswith("/natural/101.wav")
    assert row["audio"]["mfa_linear"]["path"].endswith("/mfa/101.wav")
    assert row["audio"]["sample_count"] == 16000
    assert row["legacy_natural_binding"]["paired_key_present"] is False


def test_float32_mfa_wav_is_hash_compared_through_pcm16_without_changing_source(tmp_path: Path) -> None:
    natural_path = tmp_path / "natural.wav"
    mfa_path = tmp_path / "mfa_float.wav"
    pcm = np.asarray([-32768, -1, 0, 1, 32767], dtype=np.int16)
    _write_wav(natural_path, pcm)
    float_samples = (pcm.astype(np.float32) / np.float32(32768.0)).astype(np.float32)
    assets.wavfile.write(mfa_path, 16000, float_samples)

    pair = assets.validate_audio_pair(natural_path, mfa_path)

    assert pair["sample_count"] == len(pcm)
    assert pair["pcm_identity"] is True
    assert pair["pcm_hash_encoding"].startswith("normalized_signed_pcm16_le")
    assert pair["mfa_linear"]["sha256"] == _sha(mfa_path)
    assert pair["mfa_linear"]["pcm_sha256"] == hashlib.sha256(pcm.astype("<i2").tobytes()).hexdigest()


def test_float32_mfa_wav_rejects_nonfinite_and_out_of_range_samples(tmp_path: Path) -> None:
    mfa_path = tmp_path / "invalid_float.wav"
    assets.wavfile.write(mfa_path, 16000, np.asarray([1.1], dtype=np.float32))

    with pytest.raises(ProtocolError, match=r"normalized to \[-1, 1\]"):
        assets.read_pcm16_mono(mfa_path)


def test_freeze_inputs_rejects_natural_sha_mismatch(frozen_assets: dict[str, object]) -> None:
    config = frozen_assets["config"]
    assert isinstance(config, dict)
    mini_path = Path(config["paths"]["mini_manifest"])
    mini = json.loads(mini_path.read_text())
    mini["records"][0]["natural_source_sha256"] = "0" * 64
    mini_path.write_text(json.dumps(mini), encoding="utf-8")

    with pytest.raises(ProtocolError, match="cohort manifest hash mismatch"):
        assets.freeze_inputs(config)


def test_freeze_inputs_rejects_mfa_role_or_summary_version_drift(frozen_assets: dict[str, object]) -> None:
    config = frozen_assets["config"]
    assert isinstance(config, dict)
    summary_path = Path(config["paths"]["mfa_summary"])
    summary = json.loads(summary_path.read_text())
    summary["arm"] = "natural_raw"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ProtocolError, match="MFA-linear summary"):
        assets.freeze_inputs(config)


def test_audio_pair_rejects_wrong_rate_and_missing_files(tmp_path: Path) -> None:
    natural = tmp_path / "n.wav"
    mfa = tmp_path / "m.wav"
    _write_wav(natural, np.zeros(8000, dtype=np.int16), rate=8000)
    _write_wav(mfa, np.zeros(8000, dtype=np.int16), rate=8000)
    with pytest.raises(ProtocolError, match="16 kHz"):
        assets.validate_audio_pair(natural, mfa)
    with pytest.raises(ProtocolError, match="missing"):
        assets.validate_audio_pair(natural, tmp_path / "absent.wav")


def test_portrait_audit_rejects_full_frame_and_bad_container_hash(frozen_assets: dict[str, object]) -> None:
    config = frozen_assets["config"]
    assert isinstance(config, dict)
    registry_path = Path(config["paths"]["portrait_registry"])
    registry = json.loads(registry_path.read_text())
    registry["portraits"]["6"]["generation_box_xyxy"] = [0, 0, 512, 512]
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    with pytest.raises(ProtocolError, match="full-frame"):
        assets.audit_portraits(config)
