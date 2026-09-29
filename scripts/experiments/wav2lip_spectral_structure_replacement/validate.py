from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from . import analysis, config, protocol
from .common import (
    ExperimentError,
    bytes_sha256,
    file_sha256,
    read_pcm16_wav,
    verify_self_hashed_json,
)


def _read(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def _rebuild_transformed(natural_pcm: bytes, mfa_pcm: bytes, arm: str) -> bytes:
    natural = np.frombuffer(natural_pcm, dtype="<i2").astype(np.float64) / 32768.0
    mfa = np.frombuffer(mfa_pcm, dtype="<i2").astype(np.float64) / 32768.0
    window = torch.hann_window(config.STFT_WIN_LENGTH, periodic=True, dtype=torch.float64)
    sn = torch.stft(torch.from_numpy(natural), n_fft=config.STFT_NFFT, hop_length=config.STFT_HOP_LENGTH, win_length=config.STFT_WIN_LENGTH, window=window, center=True, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    sm = torch.stft(torch.from_numpy(mfa), n_fft=config.STFT_NFFT, hop_length=config.STFT_HOP_LENGTH, win_length=config.STFT_WIN_LENGTH, window=window, center=True, pad_mode="reflect", normalized=False, onesided=True, return_complex=True)
    floor = config.MAGNITUDE_FLOOR
    mn = sn.abs().clamp_min(floor)
    mm = sm.abs().clamp_min(floor)
    ln = mn.log()
    lm = mm.log()
    if arm == config.ARM_RT:
        target = ln
    elif arm == config.ARM_MAG:
        target = ln + config.ALPHA * (lm - ln)
    elif arm == config.ARM_ENV:
        target = ln + config.ALPHA * (lm - ln).mean(dim=-1, keepdim=True)
    else:
        raise ExperimentError(f"invalid transformed arm: {arm}")
    rebuilt = torch.istft(target.exp() * (sn / mn), n_fft=config.STFT_NFFT, hop_length=config.STFT_HOP_LENGTH, win_length=config.STFT_WIN_LENGTH, window=window, center=True, normalized=False, onesided=True, length=natural.size).numpy()
    natural_rms = float(np.sqrt(np.mean(natural * natural)))
    rebuilt_rms = float(np.sqrt(np.mean(rebuilt * rebuilt)))
    rebuilt = rebuilt * (natural_rms / rebuilt_rms)
    peak = float(np.max(np.abs(rebuilt)))
    if peak >= 0.999:
        rebuilt = rebuilt * (0.999 / peak)
    return np.clip(np.rint(rebuilt * 32768.0), -32768, 32767).astype("<i2", copy=False).tobytes()


def _validate_audio(root: Path, inputs: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    manifest = _read(root / "audio/manifest.json")
    rows = {str(row["sample_id"]): row for row in manifest.get("rows", []) if isinstance(row, Mapping)}
    if set(rows) != {str(row["sample_id"]) for row in inputs["rows"]}:
        return ["audio manifest IDs differ from frozen inputs"]
    for frozen in inputs["rows"]:
        sid = str(frozen["sample_id"])
        row = rows[sid]
        arms = row.get("arms", {})
        natural_pcm = frozen["_natural_pcm"]
        mfa_pcm = frozen["_mfa_pcm"]
        expected: dict[str, bytes] = {config.ARM_N: natural_pcm, config.ARM_N_REPEAT: natural_pcm}
        for arm in (config.ARM_RT, config.ARM_MAG, config.ARM_ENV):
            expected[arm] = _rebuild_transformed(natural_pcm, mfa_pcm, arm)
        expected[config.ARM_P] = frozen["_p_pcm"]
        for arm, pcm in expected.items():
            item = arms.get(arm)
            if not isinstance(item, Mapping):
                errors.append(f"{sid}/{arm}: missing manifest arm")
                continue
            path = Path(str(item.get("path", ""))).resolve()
            try:
                observed, values, _params = read_pcm16_wav(path)
                if observed != pcm:
                    errors.append(f"{sid}/{arm}: PCM differs from independent reconstruction")
                if str(item.get("container_sha256")) != file_sha256(path) or str(item.get("decoded_pcm_sha256")) != bytes_sha256(observed):
                    errors.append(f"{sid}/{arm}: audio hash binding differs")
                if values.size != int(frozen["sample_count"]):
                    errors.append(f"{sid}/{arm}: sample count differs")
            except (OSError, ValueError, ExperimentError) as exc:
                errors.append(f"{sid}/{arm}: {type(exc).__name__}: {exc}")
    return errors


def _validate_score_rows(root: Path, inputs: Mapping[str, Any], expected_count: int) -> tuple[list[str], list[dict[str, Any]]]:
    errors: list[str] = []
    payload = _read(root / "scores/manifest.json")
    rows = [dict(row) for row in payload.get("scores", []) if isinstance(row, Mapping)]
    if len(rows) != expected_count:
        errors.append(f"score cell count {len(rows)} != {expected_count}")
    keys = {(str(row.get("sample_id")), str(row.get("video_arm")), str(row.get("audio_arm"))) for row in rows}
    expected = {(str(row["sample_id"]), arm, config.ARM_N) for row in inputs["rows"] for arm in config.STAGE_A_ARMS}
    expected |= {(str(row["sample_id"]), config.ARM_N, config.ARM_P) for row in inputs["rows"]}
    if expected_count == config.EXPECTED_RECORD_COUNT * 6:
        expected |= {(str(row["sample_id"]), arm, config.ARM_N) for row in inputs["rows"] for arm in config.STAGE_B_ARMS}
    if keys != expected:
        errors.append(f"score cell key set differs: {len(keys)}/{len(expected)}")
    for row in rows:
        try:
            media_path = Path(str(row["media"])).resolve()
            matrix_path = Path(str(row["matrix"])).resolve()
            worker_path = Path(str(row["worker_result"])).resolve()
            if file_sha256(media_path) != str(row["media_sha256"]):
                errors.append(f"media hash differs: {row.get('sample_id')}/{row.get('video_arm')}/{row.get('audio_arm')}")
            if file_sha256(matrix_path) != str(row["matrix_sha256"]):
                errors.append(f"matrix hash differs: {row.get('sample_id')}/{row.get('video_arm')}/{row.get('audio_arm')}")
            matrix = np.asarray(np.load(matrix_path, allow_pickle=False))
            if list(matrix.shape) != list(row.get("matrix_shape", [])) or matrix.ndim != 2 or matrix.shape[1] != config.MATRIX_COLUMNS or not np.isfinite(matrix).all():
                errors.append(f"matrix shape/value invalid: {row.get('sample_id')}/{row.get('video_arm')}/{row.get('audio_arm')}")
            worker = verify_self_hashed_json(worker_path)
            if worker.get("matrix_sha256") != row.get("matrix_sha256"):
                errors.append(f"worker/matrix binding differs: {worker_path}")
        except (OSError, KeyError, ValueError, ExperimentError) as exc:
            errors.append(f"malformed score row: {type(exc).__name__}: {exc}")
    return errors, rows


def _validate_videos(root: Path, expected_count: int) -> list[str]:
    errors: list[str] = []
    payload = _read(root / "videos/manifest.json")
    rows = [row for row in payload.get("rows", []) if isinstance(row, Mapping)]
    videos = [video for row in rows for video in row.get("videos", {}).values() if isinstance(video, Mapping)]
    if len(videos) != expected_count:
        errors.append(f"generated video count {len(videos)} != {expected_count}")
    for item in videos:
        path = Path(str(item.get("output", ""))).resolve()
        try:
            if file_sha256(path) != str(item.get("output_sha256")):
                errors.append(f"video hash differs: {path}")
            if item.get("device") != "cuda":
                errors.append(f"video was not generated on CUDA: {path}")
        except OSError as exc:
            errors.append(f"video missing: {path}: {exc}")
    return errors


def validate_stage_a(run_root: Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    try:
        inputs = protocol.load_frozen_inputs()
        errors = []
        errors.extend(_validate_audio(root, inputs))
        errors.extend(_validate_videos(root, config.EXPECTED_RECORD_COUNT * len(config.STAGE_A_ARMS)))
        score_errors, scores = _validate_score_rows(root, inputs, config.EXPECTED_RECORD_COUNT * 4)
        errors.extend(score_errors)
        payload = _read(root / "protocol.json")
        if payload.get("expected_stage_a_cell_count") != config.EXPECTED_RECORD_COUNT * 4:
            errors.append("protocol stage-A budget differs")
        analysis_path = root / "control_analysis.json"
        control = _read(analysis_path)
        recomputed = analysis.analyze_stage_a(payload, scores)
        if bool(recomputed.get("passes")) != bool(control.get("passes")):
            errors.append("control analysis does not match independent score recomputation")
        return {"schema_version": 1, "stage_id": "control_validation", "valid": not errors, "status": "valid" if not errors else "invalid", "error_count": len(errors), "errors": errors, "record_count": config.EXPECTED_RECORD_COUNT, "score_cell_count": len(scores), "checks": {"input_reload": True, "audio_reconstruction": True, "video_identity": not bool(_validate_videos(root, config.EXPECTED_RECORD_COUNT * len(config.STAGE_A_ARMS))), "matrix_shape_and_hash": not bool(score_errors), "analysis_recomputed": True}}
    except (OSError, KeyError, ValueError, ExperimentError, json.JSONDecodeError) as exc:
        return {"schema_version": 1, "stage_id": "control_validation", "valid": False, "status": "invalid", "error_count": 1, "errors": [f"{type(exc).__name__}: {exc}"]}


def validate_complete(run_root: Path) -> dict[str, Any]:
    root = Path(run_root).resolve()
    try:
        inputs = protocol.load_frozen_inputs()
        errors = []
        errors.extend(_validate_audio(root, inputs))
        errors.extend(_validate_videos(root, config.EXPECTED_RECORD_COUNT * (len(config.STAGE_A_ARMS) + len(config.STAGE_B_ARMS))))
        score_errors, scores = _validate_score_rows(root, inputs, config.EXPECTED_RECORD_COUNT * 6)
        errors.extend(score_errors)
        payload = _read(root / "protocol.json")
        control = _read(root / "control_analysis.json")
        analysis_payload = _read(root / "analysis.json")
        stage_a_scores = [row for row in scores if str(row["video_arm"]) in config.STAGE_A_ARMS or (str(row["video_arm"]) == config.ARM_N and str(row["audio_arm"]) == config.ARM_P)]
        recomputed_a = analysis.analyze_stage_a(payload, stage_a_scores)
        recomputed_b = analysis.analyze_stage_b(payload, scores, recomputed_a)
        if recomputed_b.get("decision") != analysis_payload.get("decision"):
            errors.append("final analysis decision differs from independent recomputation")
        if not bool(control.get("passes")):
            errors.append("complete run has a failed Stage-A control")
        return {
            "schema_version": 1,
            "stage_id": "validation",
            "valid": not errors,
            "status": "valid" if not errors else "invalid",
            "error_count": len(errors),
            "errors": errors,
            "record_count": config.EXPECTED_RECORD_COUNT,
            "score_cell_count": len(scores),
            "decision": recomputed_b["decision"],
            "checks": {
                "input_reload": True,
                "audio_reconstruction": not bool(_validate_audio(root, inputs)),
                "video_identity": not bool(_validate_videos(root, config.EXPECTED_RECORD_COUNT * (len(config.STAGE_A_ARMS) + len(config.STAGE_B_ARMS)))),
                "matrix_shape_and_hash": not bool(score_errors),
                "analysis_recomputed": True,
            },
        }
    except (OSError, KeyError, ValueError, ExperimentError, json.JSONDecodeError) as exc:
        return {"schema_version": 1, "stage_id": "validation", "valid": False, "status": "invalid", "error_count": 1, "errors": [f"{type(exc).__name__}: {exc}"]}


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Validate the Wav2Lip spectral structure replacement run")
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate_complete(args.run_root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("valid") else 2


if __name__ == "__main__":
    raise SystemExit(main())
