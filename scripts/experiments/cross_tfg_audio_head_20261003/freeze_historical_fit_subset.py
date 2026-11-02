"""Freeze a small, explicitly historical fit-only subset for the audio head.

This builder reads only the selected historical AISHELL self-clone reference
WAVs.  It never opens development or confirmation audio.  Each selected WAV
contributes one deterministic, center-near 25 fps-aligned 1.4 s training
window so differentiable Wav2Lip/SyncNet fitting stays within a small GPU
memory budget.  The complete source WAV and exact source interval remain
bound in each row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import wave
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.audio_enhancement.audio_io import read_canonical_wav, write_float32_wav


EXPECTED_QWEN_MANIFEST_RAW = "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b"
EXPECTED_QWEN_MANIFEST_PAYLOAD = "b18f83ad07a660e6c0bdef3b8780c59b7b12497370a79916eb95c7dfe07303bf"
EXPECTED_DATA_PROTOCOL_RAW = "cf615529edc8ae0b6ffff0a34246db71b3ec8ce5ac17d9f0daf2618ac624cf87"
FIT_SPEAKERS = ("S0052", "S0053", "S0054", "S0055")
SAMPLE_IDS = (
    "S0052_BAC009S0052W0121",
    "S0052_BAC009S0052W0122",
    "S0053_BAC009S0053W0121",
    "S0053_BAC009S0053W0122",
    "S0054_BAC009S0054W0121",
    "S0054_BAC009S0054W0122",
    "S0055_BAC009S0055W0121",
    "S0055_BAC009S0055W0122",
)
WINDOW_SAMPLES = 22_400  # 1.4 s; official Wav2Lip chunking yields 32 frames.
SAMPLES_PER_VIDEO_FRAME = 640  # 16 kHz / 25 fps.
MODEL_ID = "wav2lip_syncnet_training_proxy"
METHOD_ID = "shared_spectral_residual_head"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_payload_sha(value: dict[str, Any], self_key: str) -> str:
    body = {key: item for key, item in value.items() if key != self_key}
    encoded = json.dumps(
        body, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return sha256_bytes(encoded)


def _read_pcm16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if (
            handle.getnchannels() != 1
            or handle.getframerate() != 16_000
            or handle.getsampwidth() != 2
            or handle.getcomptype() != "NONE"
        ):
            raise ValueError(f"expected mono 16 kHz PCM16 source: {path}")
        frames = handle.getnframes()
        raw = handle.readframes(frames)
    values = np.frombuffer(raw, dtype="<i2").astype(np.int16, copy=True)
    if values.size != frames or values.size == 0:
        raise ValueError(f"invalid PCM16 frame payload: {path}")
    return values


def _canonical_sha(path: Path) -> str:
    return canonical_payload_sha(json.loads(path.read_text(encoding="utf-8")), "manifest_sha256")


def freeze(output_dir: Path, remote_audio_dir: str) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse historical fit subset output: {output_dir}")

    data_protocol_path = ROOT / "openspec/changes/develop-cross-tfg-audio-head-20261003/data-protocol.json"
    qwen_manifest_path = ROOT / "runs/recovered_20260930/local/qwen_source_axis/hf_selected/input_manifest.json"
    protocol_raw = data_protocol_path.read_bytes()
    protocol_sha = sha256_bytes(protocol_raw)
    if protocol_sha != EXPECTED_DATA_PROTOCOL_RAW:
        raise ValueError(f"data protocol changed: {protocol_sha}")
    protocol = json.loads(protocol_raw.decode("utf-8"))
    qwen_raw = qwen_manifest_path.read_bytes()
    qwen_raw_sha = sha256_bytes(qwen_raw)
    qwen_manifest = json.loads(qwen_raw.decode("utf-8"))
    qwen_payload_sha = canonical_payload_sha(qwen_manifest, "manifest_sha256")
    if qwen_raw_sha != EXPECTED_QWEN_MANIFEST_RAW or qwen_payload_sha != EXPECTED_QWEN_MANIFEST_PAYLOAD:
        raise ValueError("historical Qwen input manifest differs from the reviewed source binding")
    if qwen_manifest.get("status") != "complete" or qwen_manifest.get("dataset", {}).get("revision") != protocol.get("dataset_revision"):
        raise ValueError("historical Qwen source manifest is not complete or has a different AISHELL revision")

    fit_speakers = list(protocol["fit_speaker_ids"])
    dev_speakers = list(protocol["dev_speaker_ids"])
    confirm_speakers = list(protocol["confirm_speaker_ids"])
    fit_set, dev_set, confirm_set = map(set, (fit_speakers, dev_speakers, confirm_speakers))
    if fit_set & dev_set or fit_set & confirm_set or dev_set & confirm_set:
        raise ValueError("frozen speaker rosters overlap")
    if not set(FIT_SPEAKERS).issubset(fit_set):
        raise ValueError("historical subset contains speakers outside the frozen fit roster")
    if set(FIT_SPEAKERS) & (dev_set | confirm_set):
        raise ValueError("historical fit subset overlaps development/confirmation speakers")

    records = {str(row.get("id")): row for row in qwen_manifest.get("records", [])}
    if len(records) != len(qwen_manifest.get("records", [])):
        raise ValueError("historical source manifest has duplicate sample IDs")
    if not set(SAMPLE_IDS).issubset(records):
        raise ValueError("one or more explicitly selected historical samples are absent")

    audio_dir = output_dir / "fit_audio_windows"
    audio_dir.mkdir(parents=True, exist_ok=False)
    remote_audio_root = remote_audio_dir.rstrip("/")
    if not remote_audio_root.startswith("/"):
        raise ValueError("remote audio staging directory must be an absolute POSIX path")
    rows: list[dict[str, Any]] = []
    for sample_id in SAMPLE_IDS:
        source = records[sample_id]
        speaker_id = str(source["speaker"])
        if speaker_id not in fit_set or source.get("split") != "calibration":
            raise ValueError(f"selected source is not a historical fit-roster calibration row: {sample_id}")
        source_path = (ROOT / Path(source["ref_path"])).resolve(strict=True)
        source_file_sha = sha256_file(source_path)
        if source_file_sha != str(source["ref_sha256"]).lower():
            raise ValueError(f"historical fit WAV file SHA mismatch: {sample_id}")
        source_audio, source_info = read_canonical_wav(source_path)
        source_pcm16 = _read_pcm16(source_path)
        if (
            source_info.sample_rate != 16_000
            or source_info.source_sample_rate != 16_000
            or source_info.source_channels != 1
            or source_info.frames != source_pcm16.size
            or not np.array_equal(
                np.rint(np.clip(source_audio.astype(np.float64) * 32768.0, -32768.0, 32767.0)).astype(np.int16),
                source_pcm16,
            )
        ):
            raise ValueError(f"historical fit source is not exactly canonical mono PCM16: {sample_id}")
        if source_pcm16.size < WINDOW_SAMPLES:
            raise ValueError(f"historical fit source is shorter than the fixed training window: {sample_id}")

        # Nearest center start, rounded down to the original 25 fps sample clock.
        center_start = (source_pcm16.size - WINDOW_SAMPLES) // 2
        start_sample = (center_start // SAMPLES_PER_VIDEO_FRAME) * SAMPLES_PER_VIDEO_FRAME
        stop_sample = start_sample + WINDOW_SAMPLES
        window_audio = np.ascontiguousarray(source_audio[start_sample:stop_sample], dtype=np.float32)
        window_pcm16 = np.ascontiguousarray(source_pcm16[start_sample:stop_sample], dtype=np.int16)
        window_id = f"{sample_id}__center_aligned_32f"
        window_path = audio_dir / f"{window_id}.wav"
        write_info = write_float32_wav(window_path, window_audio, sample_rate=16_000, overwrite=False)
        window_file_sha = sha256_file(window_path)
        reread, reread_info = read_canonical_wav(window_path)
        if not np.array_equal(reread, window_audio) or reread_info.pcm_float32_sha256 != write_info.pcm_float32_sha256:
            raise ValueError(f"training window float32 WAV failed exact readback: {sample_id}")

        archive = source["source_archive"]
        archive_path = (ROOT / Path(archive["path"])).resolve(strict=True)
        archive_sha = sha256_file(archive_path)
        if archive_sha != str(archive["sha256"]).lower():
            raise ValueError(f"historical source archive SHA mismatch: {sample_id}")

        rows.append({
            "row_id": window_id,
            "split": "fit",
            "sample_id": sample_id,
            "speaker_id": speaker_id,
            "model_id": MODEL_ID,
            "method_id": METHOD_ID,
            "historical_exposed": True,
            "historical_exposure_source": "AISHELL S0052-S0055 Qwen self-clone calibration input manifest; explicitly allowed for fit, not fresh confirmation",
            "source_pcm": {
                "path": str(source_path),
                "file_sha256": source_file_sha,
                "pcm_sha256": source_info.pcm_float32_sha256,
                "pcm16_sha256": sha256_bytes(source_pcm16.astype("<i2", copy=False).tobytes()),
                "sample_rate_hz": 16_000,
                "channels": 1,
                "sample_count": int(source_pcm16.size),
                "source_archive_path": str(archive_path),
                "source_archive_sha256": archive_sha,
                "source_archive_member": source["source_archive_member"],
                "source_manifest_row_id": source["id"],
            },
            "training_window": {
                "rule": "single 22400-sample center-near window; start=floor(floor((N-22400)/2)/640)*640; 16 kHz mono; 640 samples/video frame",
                "start_sample": int(start_sample),
                "stop_sample_exclusive": int(stop_sample),
                "sample_count": int(WINDOW_SAMPLES),
                "duration_seconds": WINDOW_SAMPLES / 16_000,
                "source_offset_seconds": start_sample / 16_000,
                "source_frame_aligned": True,
            },
            "natural_pcm": {
                "path": f"{remote_audio_root}/{window_path.name}",
                "transfer_local_path": str(window_path.resolve()),
                "file_sha256": window_file_sha,
                "pcm_sha256": reread_info.pcm_float32_sha256,
                "sample_rate_hz": 16_000,
                "channels": 1,
                "sample_count": int(reread.size),
                "source_encoding": "IEEE_FLOAT32_WAV derived from the bound original PCM16 interval",
                "direct_source_pcm16_sha256": sha256_bytes(window_pcm16.astype("<i2", copy=False).tobytes()),
            },
        })

    runtime = {
        "adapter_factory": "scripts.audio_enhancement.wav2lip_fit_runtime:create_runtime",
        "device": "cuda",
        "wav2lip": {
            "root": "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip",
            "git_head": "bac9a81e63ecc153202353372e5724b83d9e6322",
            "checkpoint": "/root/autodl-tmp/wav2lip-syncnet/Wav2Lip/checkpoints/wav2lip_gan.pth",
            "checkpoint_sha256": "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8",
            "source_sha256": {
                "audio.py": "71d5e927cac3d0b7103e7671a43b82b41b6a56398a092bbfa2e96d3041249c4c",
                "hparams.py": "4a4c31dc51ebd8140da18ac24841e582b2472389f7c65e7f0a6b4447fc0c71cf",
                "inference.py": "99662bfce2fdaf17c1cff8e71977ba02cb80c684058cd114c4e0648b12059fc6",
                "models/wav2lip.py": "a02b54288f8200332ee4f0ea0c8013bca542b94d8149e514ac25a742a7a70e66",
            },
        },
        "syncnet": {
            "root": "/root/autodl-tmp/wav2lip-syncnet/syncnet_python",
            "git_head": "907c0b579c2e2d83f0eae1b2ac9e720cde4e5623",
            "checkpoint": "/root/autodl-tmp/wav2lip-syncnet/syncnet_python/data/syncnet_v2.model",
            "checkpoint_sha256": "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442",
            "source_sha256": {
                "SyncNetInstance.py": "bd722bf844700d6a923dec2f10d7cacd65c0c6b22d09aa5d44d2509660e900d6",
                "SyncNetModel.py": "41b85aed0006546cb2d6dc9180c5bc3dac737ab80eff643022e397a349c30c22",
                "run_pipeline.py": "3654a3cf22c74cb3071fca6fb97ca9fb7dd1a5943fe864169ff8c0b57eb80acf",
                "run_syncnet.py": "c78fe9f75665aa9e7a2bb523e4e7e3734a857137ff70b29c88a7b4893c422ded",
            },
        },
        "context": {
            "npz_path": "/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/context_export_attempt01/context.npz",
            "npz_sha256": "6616e649412cdf5c5f78ce51fa7d8403364c56528f93777a2e35eb82fc941ec1",
            "metadata_path": "/dev/shm/cross_tfg_audio_head_20261003/oldpilot_qc_attempt04/context_export_attempt01/context.json",
            "metadata_sha256": "46bab1ff647cbaabc3eea247dc315431d6ad976ca142a9c5e331ce4e81094450",
            "training_context": "fixed portrait face crop and first SyncNet scoring crop repeated over each short fit audio window; differentiable training proxy only",
        },
        "source_context_is_training_proxy_only": True,
        "fit_audio_pcm16_policy": "direct PCM16 interval recovered from bound historical source PCM16 WAV; no Wav2Lip intermediate MP3 or oldpilot SyncNet track crop",
    }

    fit_config = {
        "optimizer": "Adam",
        "learning_rate": 1e-4,
        "steps_requested": 24,
        "epochs": 3,
        "selected_rows": list(SAMPLE_IDS),
        "window_samples": WINDOW_SAMPLES,
        "source_sample_count_total": int(sum(row["source_pcm"]["sample_count"] for row in rows)),
        "proxy_loss": {
            "kind": "median_lag_distance_minus_normalized_softmin",
            "temperature": 0.1,
            "lag_count": 31,
            "support": "all source-clock SyncNet windows available in each 32-frame proxy",
        },
        "transform": {
            "method": "identity-initialized bounded phase-preserving STFT residual head",
            "n_fft": 512,
            "hop_length": 128,
            "hidden_channels": 24,
            "max_residual_db": 3.0,
            "global_rms": True,
            "target_rms": 0.0376838172675746,
            "rms_stage_peak_ceiling": 0.99,
            "local_envelope": False,
            "post_rms_match": "target",
            "peak_ceiling": 0.99,
        },
        "status_scope": "exploratory historical fit subset; no candidate selection or dev/confirmation scoring",
    }
    config_encoded = json.dumps(fit_config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    plan: dict[str, Any] = {
        "schema_version": 1,
        "schema": "cross_tfg_audio_head_historical_fit_subset_v1",
        "run_id": "cross_tfg_audio_head_fit_subset_20261003_attempt01",
        "status": "FIT_ONLY_FROZEN",
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.PIPE
        ).strip().lower(),
        "split_manifest_path": str(data_protocol_path.resolve()),
        "split_manifest_sha256": protocol_sha,
        "split_manifest_status": protocol.get("status"),
        "source_manifest_path": str(qwen_manifest_path.resolve()),
        "source_manifest_raw_sha256": qwen_raw_sha,
        "source_manifest_payload_sha256": qwen_payload_sha,
        "dataset": qwen_manifest["dataset"],
        "historical_fit_subset": True,
        "historical_exposed_speaker_ids": list(FIT_SPEAKERS),
        "no_development_or_confirmation_rows_in_this_fit_plan": True,
        "fit_speaker_ids": fit_speakers,
        "dev_speaker_ids": dev_speakers,
        "confirm_speaker_ids": confirm_speakers,
        "models": [{"model_id": MODEL_ID, "role": "frozen_Wav2Lip_training_proxy", "scoring_model_id": "pinned_SyncNet_V2"}],
        "methods": [{"method_id": METHOD_ID, "kind": "learned_shared_waveform_head", "config_sha256": sha256_bytes(config_encoded)}],
        "primary_transfer_model_id": MODEL_ID,
        "primary_method_id": METHOD_ID,
        "protocol": {
            "sample_rate_hz": 16_000,
            "channels": 1,
            "waveform_output_sample_count_must_match": True,
            "training_window_rule": "center-near 22400-sample interval with start rounded down to 640-sample/25fps boundary",
            "audio_score_source": "direct PCM16 recovered from the bound original natural WAV interval; does not use intermediate MP3",
            "face_conditioning": "fixed portrait/crop geometry repeated through the window; differentiable proxy only; not native generated mouth/video",
            "head_loss": fit_config["proxy_loss"],
            "optimizer": {"class": "torch.optim.Adam", "learning_rate": 1e-4},
        },
        "runtime": runtime,
        "fit_config": fit_config,
        "config_sha256": sha256_bytes(config_encoded),
        "rows": rows,
    }
    plan["plan_payload_sha256"] = canonical_payload_sha(plan, "plan_payload_sha256")
    output_dir.mkdir(parents=True, exist_ok=False)
    plan_path = output_dir / "historical_fit_subset_plan.json"
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    plan_raw_sha = sha256_file(plan_path)
    receipt = {
        "schema": "cross_tfg_audio_head_historical_fit_subset_build_receipt_v1",
        "status": "FIT_ONLY_PLAN_READY",
        "plan_path": str(plan_path.resolve()),
        "plan_bytes": plan_path.stat().st_size,
        "plan_raw_sha256": plan_raw_sha,
        "plan_payload_sha256": plan["plan_payload_sha256"],
        "source_manifest_raw_sha256": qwen_raw_sha,
        "source_manifest_payload_sha256": qwen_payload_sha,
        "split_manifest_sha256": protocol_sha,
        "fit_rows": len(rows),
        "fit_speakers_used": sorted({row["speaker_id"] for row in rows}),
        "development_audio_opened": False,
        "confirmation_audio_opened": False,
        "candidate_selected_on_development": False,
        "rows": [
            {
                "row_id": row["row_id"],
                "speaker_id": row["speaker_id"],
                "sample_id": row["sample_id"],
                "source_file_sha256": row["source_pcm"]["file_sha256"],
                "source_archive_sha256": row["source_pcm"]["source_archive_sha256"],
                "source_window_start_sample": row["training_window"]["start_sample"],
                "natural_pcm": row["natural_pcm"],
            }
            for row in rows
        ],
    }
    receipt_path = output_dir / "historical_fit_subset_build_receipt.json"
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--remote-audio-dir", required=True)
    args = parser.parse_args()
    receipt = freeze(args.output_dir, args.remote_audio_dir)
    print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
