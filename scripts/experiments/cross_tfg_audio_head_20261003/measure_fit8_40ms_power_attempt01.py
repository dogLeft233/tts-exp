"""Fit-only 40 ms natural/RMS waveform-power accounting.

This source-only diagnostic checks the eight historical fit WAVs named by the
sealed native-input probe plan.  It reads no labels, models, generators, scores,
development clips, or confirmation clips.  It never writes waveform copies.
The separate fit40 manifest is verified as the planned expansion roster; the
eight-row diagnostic cohort is not presented as the fit40 roster.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


EXPECTED_PROBE_PLAN_SHA256 = "0b6f7ba4935ccb7b411148f67b8024cf056a58bafdbb55dce7f008761cf88f5a"
EXPECTED_RESTORED_ARCHIVE_MANIFEST_SHA256 = "0d73826ef19a33698c41b4d50408adc7adc76a69e007c2ea8a9f4081aa0f642b"
EXPECTED_FIT40_MANIFEST_SHA256 = "0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020"
EXPECTED_FIT_PLAN_SHA256 = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
EXPECTED_AUDIO_IO_SHA256 = "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004"
EXPECTED_BASELINES_SHA256 = "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
EXPECTED_CANDIDATE_SOURCE_SHA256 = "45a287c288cc4a8d66d47359f60c1b7f0789a18b917500d86f6845e3a0eebdc3"
SAMPLE_RATE_HZ = 16_000
FRAME_MS = 40
FRAME_SAMPLES = SAMPLE_RATE_HZ * FRAME_MS // 1000
TARGET_RMS = 0.0376838172675746


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            size += len(block)
            digest.update(block)
    return size, digest.hexdigest()


def _canonical_pcm_sha256(samples: np.ndarray) -> str:
    canonical = np.asarray(samples, dtype=np.dtype("<f4"))
    if canonical.ndim != 1:
        raise ValueError("canonical PCM digest requires a mono vector")
    return hashlib.sha256(np.ascontiguousarray(canonical).tobytes()).hexdigest()


def _rms64(samples: np.ndarray) -> float:
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("RMS requires nonempty finite samples")
    return float(np.sqrt(np.mean(x * x, dtype=np.float64)))


def _power_summary(values: np.ndarray) -> dict[str, Any]:
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("power summary requires nonempty finite values")
    return {
        "count": int(x.size),
        "mean": float(np.mean(x, dtype=np.float64)),
        "median": float(np.median(x)),
        "p05": float(np.quantile(x, 0.05)),
        "p95": float(np.quantile(x, 0.95)),
        "minimum": float(np.min(x)),
        "maximum": float(np.max(x)),
    }


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _checked_file(repo_root: Path, relative_path: str, expected_sha256: str) -> Path:
    path = (repo_root / relative_path).resolve(strict=True)
    try:
        path.relative_to(repo_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError(f"pinned input escapes repository: {relative_path}") from exc
    _, observed = _hash_file(path)
    if observed != expected_sha256:
        raise ValueError(f"raw SHA mismatch for {relative_path}: {observed}")
    return path


def _validate_fit40_manifest(manifest: dict[str, Any]) -> None:
    if manifest.get("schema") != "cross_tfg_v5_fit40_source_transfer_manifest_adr013_v1":
        raise ValueError("fit40 roster manifest schema changed")
    rows = manifest.get("rows")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ValueError("the frozen expansion manifest must retain all 40 planned rows")
    sample_ids = [row.get("sample_id") for row in rows]
    speakers = [row.get("speaker_id") for row in rows]
    if len(set(sample_ids)) != 40 or len(set(speakers)) != 20:
        raise ValueError("fit40 roster must contain 40 unique clips from 20 speakers")
    if any(row.get("sample_rate_hz") != SAMPLE_RATE_HZ or row.get("channels") != 1 for row in rows):
        raise ValueError("fit40 source manifest has an unexpected audio clock/shape")
    scope = manifest.get("scope")
    if not isinstance(scope, dict):
        raise ValueError("fit40 scope record is missing")
    forbidden = (
        "candidate_transform_executed",
        "confirm_audio_read",
        "dev_audio_read",
        "model_forward",
        "render",
        "score",
    )
    if any(scope.get(key) is not False for key in forbidden):
        raise ValueError("fit40 source manifest no longer describes a source-only roster")
    selection_scan = manifest.get("selection_scan")
    if not isinstance(selection_scan, dict) or selection_scan.get("fit_plan_raw_sha256") != EXPECTED_FIT_PLAN_SHA256:
        raise ValueError("fit40 manifest is not bound to the expected frozen historical-fit plan")


def run(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve(strict=True)
    repo_root = Path(__file__).resolve().parents[3]
    config = _load_json(config_path)
    if config.get("schema") != "cross_tfg_fit8_40ms_power_measurement_config_v1":
        raise ValueError("unexpected measurement configuration schema")
    if (
        config.get("scope") != "historical_fit8_waveform_power_only"
        or config.get("frame_samples") != FRAME_SAMPLES
        or config.get("frame_ms") != FRAME_MS
        or config.get("sample_rate_hz") != SAMPLE_RATE_HZ
        or config.get("target_rms") != TARGET_RMS
        or config.get("peak_ceiling") is not None
    ):
        raise ValueError("configuration changed the declared cohort or 40 ms clock")
    measurement_source = config.get("measurement_source")
    candidate_source = config.get("candidate_source")
    if not isinstance(measurement_source, dict) or not isinstance(candidate_source, dict):
        raise ValueError("configuration must bind both source files")
    expected_measurement_path = "scripts/experiments/cross_tfg_audio_head_20261003/measure_fit8_40ms_power_attempt01.py"
    if measurement_source.get("path") != expected_measurement_path:
        raise ValueError("measurement source path changed")
    if candidate_source.get("path") != "scripts/audio_enhancement/shared_recipient_local_power_spectral_head_v1.py":
        raise ValueError("candidate source path changed")
    _checked_file(repo_root, measurement_source["path"], measurement_source["sha256"])
    _checked_file(repo_root, candidate_source["path"], candidate_source["sha256"])

    plan_path = _checked_file(
        repo_root, config["probe_plan_path"], EXPECTED_PROBE_PLAN_SHA256
    )
    archive_manifest_path = _checked_file(
        repo_root,
        config["restored_archive_manifest_path"],
        EXPECTED_RESTORED_ARCHIVE_MANIFEST_SHA256,
    )
    fit40_manifest_path = _checked_file(
        repo_root, config["fit40_manifest_path"], EXPECTED_FIT40_MANIFEST_SHA256
    )
    fit_plan_path = _checked_file(
        repo_root, config["fit_plan_path"], EXPECTED_FIT_PLAN_SHA256
    )
    _checked_file(repo_root, "scripts/audio_enhancement/audio_io.py", EXPECTED_AUDIO_IO_SHA256)
    _checked_file(repo_root, "scripts/audio_enhancement/baselines.py", EXPECTED_BASELINES_SHA256)
    _checked_file(
        repo_root,
        "scripts/audio_enhancement/shared_recipient_local_power_spectral_head_v1.py",
        EXPECTED_CANDIDATE_SOURCE_SHA256,
    )

    plan = _load_json(plan_path)
    if plan.get("schema") != "cross_tfg_native_audio_encoder_probe_plan_v1":
        raise ValueError("native fit8 probe plan schema changed")
    rows = plan.get("input_rows")
    if not isinstance(rows, list) or len(rows) != 8:
        raise ValueError("the sealed historical diagnostic denominator must remain eight clips")
    if any(row.get("split") != "fit" or row.get("seed") != 20261003 for row in rows):
        raise ValueError("probe-plan rows are not the frozen fit split and seed")
    scope = plan.get("scope")
    if (
        not isinstance(scope, dict)
        or scope.get("development_audio_opened") is not False
        or scope.get("confirmation_audio_opened") is not False
    ):
        raise ValueError("probe plan does not prove development/confirmation closure")
    sample_ids = [row.get("sample_id") for row in rows]
    if len(set(sample_ids)) != 8:
        raise ValueError("fit8 probe plan has duplicate sample IDs")
    if plan.get("source_plan", {}).get("sha256") != EXPECTED_FIT_PLAN_SHA256:
        raise ValueError("fit8 probe plan is not bound to the expected frozen fit plan")

    archive_manifest = _load_json(archive_manifest_path)
    archived_inputs = archive_manifest.get("inputs")
    if not isinstance(archived_inputs, list) or len(archived_inputs) != 8:
        raise ValueError("strictly restored audio archive does not bind all eight fit inputs")
    archive_by_id = {item.get("sample_id"): item for item in archived_inputs}
    if set(archive_by_id) != set(sample_ids):
        raise ValueError("restored archive input IDs differ from the frozen eight-row probe plan")
    for planned in rows:
        archive_item = archive_by_id[planned["sample_id"]]
        pcm_record = planned["natural_pcm"]
        if (
            archive_item.get("speaker_id") != planned["speaker_id"]
            or archive_item.get("encoding") != "PCM_16"
            or archive_item.get("sample_rate_hz") != SAMPLE_RATE_HZ
            or archive_item.get("sha256") != pcm_record["file_sha256"]
            or archive_item.get("pcm_float32_sha256") != pcm_record["pcm_sha256"]
            or archive_item.get("sample_count") != pcm_record["sample_count"]
        ):
            raise ValueError(f"restored archive provenance differs from probe plan: {planned['sample_id']}")

    fit40_manifest = _load_json(fit40_manifest_path)
    _validate_fit40_manifest(fit40_manifest)
    if _hash_file(fit_plan_path)[1] != fit40_manifest["selection_scan"]["fit_plan_raw_sha256"]:
        raise ValueError("frozen fit plan and fit40 source manifest disagree")

    # Import only the qualified PCM decoder and RMS helper after checking their
    # exact file pins. These modules do not initialize Torch or a frontend.
    repo_text = str(repo_root)
    if repo_text not in sys.path:
        sys.path.insert(0, repo_text)
    from scripts.audio_enhancement.audio_io import read_canonical_wav
    from scripts.audio_enhancement.baselines import apply_global_rms, FIXED_POSITIVE_CONTROL_RMS

    if importlib.util.find_spec("torch") is not None and "torch" in sys.modules:
        raise RuntimeError("unexpected Torch initialization in the source-only CPU measurement")
    if float(FIXED_POSITIVE_CONTROL_RMS) != TARGET_RMS:
        raise ValueError("qualified RMS helper target changed")

    input_root = (repo_root / config["fit8_restored_input_dir"]).resolve(strict=True)
    try:
        input_root.relative_to(repo_root.resolve(strict=True))
    except ValueError as exc:
        raise ValueError("restored input directory escapes the repository") from exc
    frame_records: list[dict[str, Any]] = []
    sample_records: list[dict[str, Any]] = []
    for planned in rows:
        pcm_record = planned["natural_pcm"]
        source_name = Path(pcm_record["path"]).name
        wav_path = (input_root / source_name).resolve(strict=True)
        if wav_path.parent != input_root:
            raise ValueError(f"WAV path escaped the readback inputs folder: {source_name}")
        file_bytes, file_sha = _hash_file(wav_path)
        if file_sha != pcm_record["file_sha256"]:
            raise ValueError(f"source WAV raw SHA mismatch: {planned['sample_id']}")
        natural, read_info = read_canonical_wav(wav_path, target_rate=SAMPLE_RATE_HZ)
        if (
            read_info.source_sample_rate != SAMPLE_RATE_HZ
            or read_info.source_channels != 1
            or read_info.source_encoding != "PCM_16"
            or read_info.resampler != "none"
            or read_info.frames != pcm_record["sample_count"]
            or read_info.pcm_float32_sha256 != pcm_record["pcm_sha256"]
        ):
            raise ValueError(f"canonical source PCM differs from frozen plan: {planned['sample_id']}")

        rms_audio, rms_info = apply_global_rms(
            natural,
            target_rms=TARGET_RMS,
            peak_ceiling=None,
        )
        if rms_audio.shape != natural.shape or rms_info.peak_limited:
            raise ValueError(f"RMS helper changed the clock or activated a peak limit: {planned['sample_id']}")
        if _canonical_pcm_sha256(natural) != pcm_record["pcm_sha256"]:
            raise ValueError(f"source canonical PCM digest mismatch: {planned['sample_id']}")

        frame_count = natural.size // FRAME_SAMPLES
        trailing = natural.size - frame_count * FRAME_SAMPLES
        nat_frames = natural[: frame_count * FRAME_SAMPLES].reshape(frame_count, FRAME_SAMPLES).astype(
            np.float64
        )
        rms_frames = rms_audio[: frame_count * FRAME_SAMPLES].reshape(frame_count, FRAME_SAMPLES).astype(
            np.float64
        )
        nat_power = np.mean(nat_frames * nat_frames, axis=1, dtype=np.float64)
        rms_power = np.mean(rms_frames * rms_frames, axis=1, dtype=np.float64)
        nat_frame_rms = np.sqrt(nat_power)
        rms_frame_rms = np.sqrt(rms_power)
        valid_ratio = nat_power > 0.0
        ratio = np.full(frame_count, np.nan, dtype=np.float64)
        ratio_db = np.full(frame_count, np.nan, dtype=np.float64)
        ratio[valid_ratio] = rms_power[valid_ratio] / nat_power[valid_ratio]
        ratio_db[valid_ratio] = 10.0 * np.log10(ratio[valid_ratio])

        for index in range(frame_count):
            frame_start = index * FRAME_SAMPLES
            frame_records.append(
                {
                    "sample_id": planned["sample_id"],
                    "speaker_id": planned["speaker_id"],
                    "frame_index": index,
                    "source_start_sample": frame_start,
                    "source_stop_sample_exclusive": frame_start + FRAME_SAMPLES,
                    "source_center_sample": frame_start + FRAME_SAMPLES // 2,
                    "natural_frame_rms": float(nat_frame_rms[index]),
                    "rms_frame_rms": float(rms_frame_rms[index]),
                    "natural_frame_power": float(nat_power[index]),
                    "rms_frame_power": float(rms_power[index]),
                    "rms_to_natural_power_ratio": (
                        float(ratio[index]) if valid_ratio[index] else ""
                    ),
                    "rms_to_natural_power_change_db": (
                        float(ratio_db[index]) if valid_ratio[index] else ""
                    ),
                    "ratio_supported_nonzero_natural_power": bool(valid_ratio[index]),
                }
            )

        expected_gain = rms_info.applied_gain
        supported_db = ratio_db[valid_ratio]
        sample_records.append(
            {
                "sample_id": planned["sample_id"],
                "speaker_id": planned["speaker_id"],
                "split": "fit",
                "seed": int(planned["seed"]),
                "source_path": str(wav_path.relative_to(repo_root)).replace("\\", "/"),
                "source_wav_bytes": file_bytes,
                "source_wav_sha256": file_sha,
                "source_pcm_float32_sha256": read_info.pcm_float32_sha256,
                "sample_rate_hz": SAMPLE_RATE_HZ,
                "source_sample_count": int(natural.size),
                "complete_40ms_frame_count": int(frame_count),
                "trailing_unframed_sample_count": int(trailing),
                "canonical_target_rms": TARGET_RMS,
                "natural_global_rms": _rms64(natural),
                "rms_output_global_rms": _rms64(rms_audio),
                "rms_applied_gain": expected_gain,
                "rms_applied_gain_db": 20.0 * math.log10(expected_gain),
                "natural_peak": float(np.max(np.abs(natural.astype(np.float64)))),
                "rms_output_peak": float(np.max(np.abs(rms_audio.astype(np.float64)))),
                "peak_limited": False,
                "nonzero_natural_power_frame_count": int(valid_ratio.sum()),
                "zero_natural_power_frame_count": int((~valid_ratio).sum()),
                "frame_power_ratio_summary": _power_summary(ratio[valid_ratio]),
                "frame_power_change_db_summary": _power_summary(supported_db),
                "max_abs_frame_power_ratio_error_from_global_gain_squared": float(
                    np.max(np.abs(ratio[valid_ratio] - expected_gain * expected_gain))
                ) if np.any(valid_ratio) else None,
            }
        )

    if len(sample_records) != 8:
        raise RuntimeError("the complete fit8 diagnostic denominator was not emitted")
    by_speaker: dict[str, list[dict[str, Any]]] = {}
    for row in sample_records:
        by_speaker.setdefault(row["speaker_id"], []).append(row)
    if len(by_speaker) != 4 or any(len(rows_for_speaker) != 2 for rows_for_speaker in by_speaker.values()):
        raise ValueError("fit8 diagnostic cohort must preserve four speakers and both planned clips")
    out_path = Path(config["fresh_output_dir"])
    if not out_path.is_absolute():
        out_path = repo_root / out_path
    out_path = out_path.resolve()
    if out_path.exists():
        raise FileExistsError(f"refusing to overwrite diagnostic output: {out_path}")
    out_path.mkdir(parents=True)

    frame_csv = out_path / "fit8_40ms_frame_power.csv"
    fieldnames = list(frame_records[0])
    with frame_csv.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_records)
        handle.flush()
        os.fsync(handle.fileno())

    summary = {
        "schema": "cross_tfg_fit8_40ms_power_measurement_v1",
        "scope": {
            "fit8_waveform_power_only": True,
            "fit40_roster_only_verified_not_measured": True,
            "development_audio_read": False,
            "confirmation_audio_read": False,
            "model_forward": False,
            "render": False,
            "score": False,
            "optimizer": False,
            "waveform_copy_written": False,
        },
        "source_bindings": {
            "measurement_source_raw_sha256": measurement_source["sha256"],
            "candidate_head_source_sha256": candidate_source["sha256"],
            "probe_plan_raw_sha256": EXPECTED_PROBE_PLAN_SHA256,
            "restored_archive_manifest_raw_sha256": EXPECTED_RESTORED_ARCHIVE_MANIFEST_SHA256,
            "fit40_source_manifest_raw_sha256": EXPECTED_FIT40_MANIFEST_SHA256,
            "historical_fit_plan_raw_sha256": EXPECTED_FIT_PLAN_SHA256,
            "audio_io_raw_sha256": EXPECTED_AUDIO_IO_SHA256,
            "baselines_raw_sha256": EXPECTED_BASELINES_SHA256,
        },
        "measurement": {
            "cohort": "eight historical fit utterances from the native encoder input probe",
            "frame_clock": "non-overlapping source intervals [640*j,640*(j+1)) at 16 kHz",
            "frame_samples": FRAME_SAMPLES,
            "trailing_samples_policy": "excluded from complete 40 ms frames and counted per clip",
            "ratio_support": "natural frame power strictly greater than zero; unsupported intervals remain counted",
            "rms_target": TARGET_RMS,
            "peak_ceiling": None,
            "sample_rows": sample_records,
            "equal_speaker_mean_rms_to_natural_power_ratio": float(
                np.mean(
                    [
                        np.mean(
                            [row["frame_power_ratio_summary"]["mean"] for row in rows_for_speaker]
                        )
                        for rows_for_speaker in by_speaker.values()
                    ]
                )
            ),
            "equal_speaker_mean_rms_to_natural_power_change_db": float(
                np.mean(
                    [
                        np.mean(
                            [row["frame_power_change_db_summary"]["mean"] for row in rows_for_speaker]
                        )
                        for rows_for_speaker in by_speaker.values()
                    ]
                )
            ),
            "complete_frame_row_count": len(frame_records),
        },
        "output_files": {
            frame_csv.name: {"bytes": frame_csv.stat().st_size, "sha256": _hash_file(frame_csv)[1]}
        },
    }
    summary_path = out_path / "fit8_40ms_power_summary.json"
    summary_bytes = (json.dumps(summary, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    with summary_path.open("xb") as handle:
        handle.write(summary_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    receipt = {
        "schema": "cross_tfg_fit8_40ms_power_measurement_receipt_v1",
        "status": "COMPLETE_SOURCE_ONLY_WAVEFORM_POWER",
        "summary": {"path": summary_path.name, "bytes": len(summary_bytes), "sha256": hashlib.sha256(summary_bytes).hexdigest()},
        "frame_csv": {"path": frame_csv.name, "bytes": frame_csv.stat().st_size, "sha256": _hash_file(frame_csv)[1]},
        "sample_count": len(sample_records),
        "frame_row_count": len(frame_records),
        "scope": summary["scope"],
    }
    receipt_path = out_path / "measurement_receipt.json"
    receipt_bytes = (json.dumps(receipt, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode(
        "utf-8"
    )
    with receipt_path.open("xb") as handle:
        handle.write(receipt_bytes)
        handle.flush()
        os.fsync(handle.fileno())
    return {
        "status": "COMPLETE_SOURCE_ONLY_WAVEFORM_POWER",
        "output_dir": str(out_path),
        "receipt_bytes": len(receipt_bytes),
        "receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "sample_count": len(sample_records),
        "frame_row_count": len(frame_records),
        "scope": summary["scope"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.config)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
