#!/usr/bin/env python3
"""Run the frozen local-clock waveform pilot against report44 assets."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

REPO = Path(__file__).resolve().parents[3]
BASE = REPO / "runs/repeated_take_crossover_20261015_attempt01"
OUT = Path("/dev/shm/local_clock_crossover_20261016_main_attempt01")
SPEC_PATH = REPO / ".codex/scratch/local_clock_spec_20261016.md"
SPEAKERS = ("S0003", "S0006", "S0007", "S0008")
ROLES = ("N1", "N2", "T1", "T2")
REFERENCES = ("N1", "T1")
SAMPLE_RATE = 16_000
HOP = 80
N_FFT = 512
TARGET_RMS = 0.0376838172675746
POSITIVE_AMPLITUDE_S = 0.240
WAV2LIP_PY = Path("/home/wjj/.venvs/wav2lip/bin/python")
SYNCNET_PY = Path("/home/wjj/.venvs/syncnet/bin/python")
CPU_BUDGET_S = 30 * 60
GPU_BUDGET_S = 10 * 60
SPEC_SHA256 = {
    "basic_memory_spec_file": "349a679a0eda2296e52cf8d5b0704b7fe0a9838200d5a88dd71a406146d953f3",
    "frozen_scratch_spec_v1_1_file": "fa8f004f7b159410c961080b5ba053509480d7910f62e7c9982c36e35f30780d",
}
PRIOR_PREFLIGHT_FAILURE = {
    "status": "FAILED_BEFORE_ATTEMPT_CREATION",
    "source_sha256": "5a495348ebdc154ad99759625ad746696ab3a90968e111c348b793b51c5ad542",
    "error": "PilotError: report44 fixed support exceeds real visual rows: S0003/N1",
    "attempt_directory_created": False,
    "audio_mfa_or_syncnet_started": False,
}
PRIOR_FAILED_CONTROL_ATTEMPT = {
    "status": "CONTROL_GATE_FAIL_BEFORE_AUDIO_FORWARD",
    "run_path": str(REPO / "runs/local_clock_crossover_20261016_attempt01"),
    "runner_sha256": "f494ddc45fb2e2ace797ab0611b6203347c9906490c83ffa3e6be3430f84a6b8",
    "error": "librosa 1.0 phase_vocoder received both rate=1.0 and t_out; exactly one is required",
    "render_errors": 20,
    "audio_wavs_created": 0,
    "mfa_receipts_created": 0,
    "scored_audio_conditions": 0,
}


class PilotError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def hash_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def process_start_ticks(pid: int) -> int:
    stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    closing_paren = stat.rfind(")")
    if closing_paren < 0:
        raise PilotError(f"cannot parse /proc/{pid}/stat")
    fields = stat[closing_paren + 2 :].split()
    if len(fields) <= 19:
        raise PilotError(f"/proc/{pid}/stat lacks starttime field for pid {pid}")
    return int(fields[19])


def load_bindings() -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    paths = {
        "audio_manifest": BASE / "audio_manifest.json",
        "canonical32_manifest": BASE / "timing_diagnostics_canonical32/input_manifest.json",
        "matrix_scores": BASE / "matrix_scores.json",
        "video_manifest": BASE / "video_manifest.json",
        "frozen_protocol_spec": SPEC_PATH,
    }
    for label, path in paths.items():
        if not path.is_file():
            raise PilotError(f"missing report44 {label}: {path}")
    if sha256_file(SPEC_PATH) != SPEC_SHA256["frozen_scratch_spec_v1_1_file"]:
        raise PilotError("uploaded frozen protocol spec hash mismatch")
    audio_doc = json.loads(paths["audio_manifest"].read_text(encoding="utf-8"))
    grid_doc = json.loads(paths["canonical32_manifest"].read_text(encoding="utf-8"))
    score_doc = json.loads(paths["matrix_scores"].read_text(encoding="utf-8"))
    video_doc = json.loads(paths["video_manifest"].read_text(encoding="utf-8"))
    audio_by = {(row["speaker"], row["role"]): row for row in audio_doc["records"]}
    grid_by = {(row["speaker"], row["role"]): row for row in grid_doc["items"]}
    matrix_by = {row["speaker"]: row for row in score_doc["speaker_results"]}
    video_by = {(row["speaker"], row["video_role"]): row for row in video_doc["records"]}
    bindings: dict[str, dict[str, Any]] = {}
    for speaker in SPEAKERS:
        if speaker not in matrix_by:
            raise PilotError(f"speaker missing from report44 score matrix: {speaker}")
        role_rows: dict[str, Any] = {}
        for role in ROLES:
            key = (speaker, role)
            if key not in audio_by or key not in grid_by:
                raise PilotError(f"report44 lacks {speaker}/{role} audio or canonical32 transcript")
            audio = audio_by[key]
            grid = grid_by[key]
            audio_path = Path(audio["canonical_path"])
            if not audio_path.is_file() or sha256_file(audio_path) != audio["canonical_sha256"]:
                raise PilotError(f"report44 canonical audio hash mismatch: {speaker}/{role}")
            grid_path = BASE / "timing_diagnostics_canonical32/textgrids/canonical32" / speaker / f"{grid['input_stem']}.TextGrid"
            if not grid_path.is_file():
                raise PilotError(f"canonical32 TextGrid missing: {speaker}/{role}")
            feature_path = BASE / "features" / speaker / f"visual_{role}.npy"
            if not feature_path.is_file():
                raise PilotError(f"cached visual embedding missing: {speaker}/{role}")
            cell = matrix_by[speaker]["cells"].get(f"V_{role}_A_{role}")
            if not isinstance(cell, dict):
                raise PilotError(f"report44 self cell missing: {speaker}/{role}")
            distance_path = Path(cell["distance_path"])
            if not distance_path.is_file() or sha256_file(distance_path) != cell["distance_sha256"]:
                raise PilotError(f"report44 self distance hash mismatch: {speaker}/{role}")
            q0 = int(matrix_by[speaker]["support"]["query_start_window_inclusive"])
            q1 = int(matrix_by[speaker]["support"]["query_end_window_exclusive"])
            if q0 < 0 or q1 <= q0:
                raise PilotError(f"invalid report44 fixed support interval: {speaker}/{role}")
            support = list(range(q0, q1))
            primary_index = int(cell["argmin_index"])
            if primary_index < 0 or primary_index >= 31 or len(cell["lag_frames"]) != 31:
                raise PilotError(f"invalid frozen report44 primary lag: {speaker}/{role}")
            visual = np.load(feature_path, allow_pickle=False)
            original_distance = np.load(distance_path, allow_pickle=False)
            if visual.ndim != 2 or visual.shape[1] != 1024 or original_distance.shape != tuple(cell["distance_shape"]):
                raise PilotError(f"report44 feature geometry mismatch: {speaker}/{role}")
            if original_distance.shape != (q1 - q0, 31):
                raise PilotError(f"report44 distance artifact does not match its fixed support rows: {speaker}/{role}")
            if q1 > visual.shape[0]:
                raise PilotError(f"report44 fixed support exceeds real visual rows: {speaker}/{role}")
            baseline_distance = float(np.mean(original_distance[:, primary_index], dtype=np.float64))
            baseline_curve = np.mean(original_distance, axis=0, dtype=np.float64)
            baseline_c = float(np.median(baseline_curve) - np.min(baseline_curve))
            role_rows[role] = {
                "audio_path": audio_path,
                "audio_sha256": audio["canonical_sha256"],
                "audio_samples": int(audio["samples"]),
                "audio_rms": float(audio["rms_after_pcm16"]),
                "textgrid_path": grid_path,
                "textgrid_sha256": sha256_file(grid_path),
                "input_stem": str(grid["input_stem"]),
                "transcript": str(grid["transcript"]),
                "visual_path": feature_path,
                "visual_sha256": sha256_file(feature_path),
                "video": video_by.get(key),
                "support": support,
                "primary_index": primary_index,
                "primary_lag_frames": int(cell["lag_frames"][primary_index]),
                "baseline_primary_distance": baseline_distance,
                "baseline_C": baseline_c,
                "reported_baseline_C": float(cell["C"]),
                "distance_path": distance_path,
                "distance_sha256": cell["distance_sha256"],
                "distance_shape": list(original_distance.shape),
                "self_cell": f"V_{role}_A_{role}",
            }
        for ref in REFERENCES:
            video = video_by.get((speaker, ref))
            if not isinstance(video, dict):
                raise PilotError(f"report44 reference video missing: {speaker}/{ref}")
            video_path = Path(video["video_path"])
            if not video_path.is_file() or sha256_file(video_path) != video["video_sha256"]:
                raise PilotError(f"report44 reference video hash mismatch: {speaker}/{ref}")
        bindings[speaker] = role_rows
    receipt = {
        "source": "report44 repeated_take_crossover_20261015_attempt01",
        "files": {name: hash_record(path) for name, path in paths.items()},
        "frozen_protocol_sha256": SPEC_SHA256,
        "speaker_roles": {
            speaker: {
                role: {
                    "audio_path": str(row["audio_path"]), "audio_sha256": row["audio_sha256"],
                    "textgrid_path": str(row["textgrid_path"]), "textgrid_sha256": row["textgrid_sha256"],
                    "visual_path": str(row["visual_path"]), "visual_sha256": row["visual_sha256"],
                    "distance_path": str(row["distance_path"]), "distance_sha256": row["distance_sha256"],
                    "support_rows": row["support"], "primary_index": row["primary_index"],
                    "primary_lag_frames": row["primary_lag_frames"],
                    "baseline_primary_distance": row["baseline_primary_distance"],
                    "baseline_C": row["baseline_C"],
                    "video_path": (row["video"].get("video_path") if row["video"] else None),
                    "video_sha256": (row["video"].get("video_sha256") if row["video"] else None),
                }
                for role, row in roles.items()
            }
            for speaker, roles in bindings.items()
        },
    }
    return receipt, bindings


def read_pcm16(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with wave.open(str(path), "rb") as handle:
        ch, width, rate, frames = handle.getnchannels(), handle.getsampwidth(), handle.getframerate(), handle.getnframes()
        raw = handle.readframes(frames)
    if (ch, width, rate) != (1, 2, SAMPLE_RATE) or frames < 1:
        raise PilotError(f"expected mono/16k/PCM16 input: {path}")
    pcm = np.frombuffer(raw, dtype="<i2").copy()
    if pcm.size != frames:
        raise PilotError(f"PCM frame count mismatch: {path}")
    values = pcm.astype(np.float64) / 32768.0
    return values, {"samples": int(frames), "duration_s": frames / SAMPLE_RATE, "rms": float(np.sqrt(np.mean(values * values))), "peak": float(np.max(np.abs(values))), "pcm_sha256": hashlib.sha256(raw).hexdigest()}


def write_pcm16(path: Path, values: np.ndarray) -> dict[str, Any]:
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1 or y.size < 1 or not np.isfinite(y).all():
        raise PilotError("renderer returned invalid audio")
    rms = float(np.sqrt(np.mean(y * y)))
    if not math.isfinite(rms) or rms <= 0:
        raise PilotError("renderer returned zero or nonfinite RMS")
    y = y * (TARGET_RMS / rms)
    peak_before_pcm = float(np.max(np.abs(y)))
    if peak_before_pcm >= 1.0:
        raise PilotError(f"RMS normalization would clip: peak={peak_before_pcm:.9g}")
    pcm64 = np.rint(y * 32768.0)
    if np.any(pcm64 < -32768) or np.any(pcm64 > 32767):
        raise PilotError("PCM16 conversion would clip")
    pcm = pcm64.astype("<i2")
    raw = pcm.tobytes()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PilotError(f"refusing to overwrite WAV: {path}")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(SAMPLE_RATE)
        handle.writeframes(raw)
    actual_rms = float(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)) / 32768.0)
    if abs(actual_rms - TARGET_RMS) > 2e-6:
        raise PilotError(f"quantized RMS outside frozen tolerance: {actual_rms:.12g}")
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "pcm_sha256": hashlib.sha256(raw).hexdigest(), "samples": int(pcm.size), "sample_rate_hz": SAMPLE_RATE, "channels": 1, "sample_width_bytes": 2, "rms": actual_rms, "rms_error": abs(actual_rms - TARGET_RMS), "peak": float(np.max(np.abs(pcm.astype(np.float64) / 32768.0))), "peak_before_pcm": peak_before_pcm, "clip_count": 0, "bytes": path.stat().st_size}


def render_pv(source: Path, destination: Path, target_samples: int, map_spec: Mapping[str, Any]) -> dict[str, Any]:
    import librosa

    x, input_meta = read_pcm16(source)
    stft = librosa.stft(x, n_fft=N_FFT, hop_length=HOP, win_length=N_FFT, window="hann", center=True, pad_mode="constant", dtype=np.complex128)
    frame_count = target_samples // HOP + 1
    output_times = np.arange(frame_count, dtype=np.float64) * (HOP / SAMPLE_RATE)
    kind = str(map_spec["kind"])
    if kind == "identity":
        raw_t = np.arange(frame_count, dtype=np.float64)
    elif kind == "positive_sine":
        duration = float(map_spec["source_duration_s"])
        raw_t = (output_times + POSITIVE_AMPLITUDE_S * np.sin(2.0 * np.pi * output_times / duration)) * SAMPLE_RATE / HOP
    else:
        raw_t = np.interp(output_times, np.asarray(map_spec["output_anchor_s"], dtype=np.float64), np.asarray(map_spec["input_anchor_s"], dtype=np.float64)) * SAMPLE_RATE / HOP
    clipped_t = np.clip(raw_t, 0.0, float(stft.shape[1] - 1))
    stretched = librosa.phase_vocoder(stft, hop_length=HOP, n_fft=N_FFT, t_out=clipped_t)
    y = librosa.istft(stretched, hop_length=HOP, win_length=N_FFT, window="hann", center=True, length=int(target_samples))
    if y.size != target_samples:
        raise PilotError("ISTFT did not return the prescribed sample count")
    meta = write_pcm16(destination, y)
    t_path = destination.with_suffix(".t_out.npy")
    if t_path.exists():
        raise PilotError(f"refusing to overwrite phase map: {t_path}")
    np.save(t_path, clipped_t, allow_pickle=False)
    meta.update({
        "source_path": str(source.resolve()), "source_sha256": sha256_file(source), "source_pcm": input_meta,
        "renderer": "librosa.phase_vocoder", "librosa_version": importlib.metadata.version("librosa"),
        "n_fft": N_FFT, "win_length": N_FFT, "hop_length": HOP, "window": "hann", "center": True,
        "pad_mode": "constant", "output_frame_count": frame_count,
        "unclipped_t_out_min": float(np.min(raw_t)), "unclipped_t_out_max": float(np.max(raw_t)),
        "stft_frame_count": int(stft.shape[1]), "edge_clipped_frame_count": int(np.count_nonzero(raw_t != clipped_t)),
        "t_out_path": str(t_path.resolve()), "t_out_sha256": sha256_file(t_path), "map": dict(map_spec),
    })
    return meta


def load_timing_helper() -> Any:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    from scripts.experiments.repeated_take_crossover_20261015 import timing_diagnostics
    return timing_diagnostics


def recognized_grid(helper: Any, path: Path) -> tuple[list[dict[str, Any]], list[str], int]:
    tiers = helper.parse_textgrid(path)
    recognized, _, _, labels, unknown_count, _ = helper._phone_intervals(tiers)
    return recognized, labels, unknown_count


def nonsilence_count(labels: Sequence[str], helper: Any) -> int:
    return sum(str(label).casefold() not in helper.ACTUAL_SILENCE_LABELS for label in labels)


def fit_phone_map(helper: Any, reference_grid: Path, source_grid: Path, reference_duration: float, source_duration: float) -> dict[str, Any]:
    ref, ref_labels, _ = recognized_grid(helper, reference_grid)
    src, src_labels, _ = recognized_grid(helper, source_grid)
    pairs = helper._lcs_pairs(ref, src)
    nr, ns = nonsilence_count(ref_labels, helper), nonsilence_count(src_labels, helper)
    cov_ref = len(pairs) / nr if nr else 0.0
    cov_src = len(pairs) / ns if ns else 0.0
    result: dict[str, Any] = {"status": "UNSUPPORTED", "matched_count": len(pairs), "reference_non_silence_count": nr, "source_non_silence_count": ns, "reference_coverage": cov_ref, "source_coverage": cov_src, "matched_phones": []}
    if cov_ref < 0.85 or cov_src < 0.85:
        result["reason"] = "phone_coverage_below_0.85"
        return result
    out_anchors = [0.0]
    in_anchors = [0.0]
    for i, j in pairs:
        r, s = ref[i], src[j]
        out_center = (float(r["start_s"]) + float(r["end_s"])) / 2.0
        in_center = (float(s["start_s"]) + float(s["end_s"])) / 2.0
        out_anchors.append(out_center)
        in_anchors.append(in_center)
        result["matched_phones"].append({"label": str(r["label"]), "reference_center_s": out_center, "source_center_s": in_center, "reference_index": int(r["index"]), "source_index": int(s["index"])})
    out_anchors.append(float(reference_duration))
    in_anchors.append(float(source_duration))
    out_a, in_a = np.asarray(out_anchors), np.asarray(in_anchors)
    if np.any(np.diff(out_a) <= 0) or np.any(np.diff(in_a) <= 0):
        result["reason"] = "duplicate_or_nonmonotone_anchors"
        return result
    slopes = np.diff(in_a) / np.diff(out_a)
    result.update({"output_anchor_s": out_anchors, "input_anchor_s": in_anchors, "segment_slopes": [float(x) for x in slopes], "slope_min": float(np.min(slopes)), "slope_max": float(np.max(slopes))})
    if np.any(slopes < 0.25) or np.any(slopes > 4.0):
        result["reason"] = "slope_outside_0.25_4"
        return result
    result.update({"status": "SUPPORTED", "kind": "piecewise_linear_phone_centers", "unmatched_span_rule": "linear interpolation between nearest surrounding matched anchors"})
    return result


def grid_clock_error(helper: Any, reference_grid: Path, actual_grid: Path) -> dict[str, Any]:
    ref, ref_labels, _ = recognized_grid(helper, reference_grid)
    act, act_labels, _ = recognized_grid(helper, actual_grid)
    pairs = helper._lcs_pairs(ref, act)
    nr, na = nonsilence_count(ref_labels, helper), nonsilence_count(act_labels, helper)
    errors = [abs(((float(act[j]["start_s"]) + float(act[j]["end_s"])) / 2.0) - ((float(ref[i]["start_s"]) + float(ref[i]["end_s"])) / 2.0)) for i, j in pairs]
    return {"matched_count": len(pairs), "reference_coverage": len(pairs) / nr if nr else 0.0, "actual_coverage": len(pairs) / na if na else 0.0, "mae_s": float(np.mean(errors)) if errors else None, "median_abs_error_s": float(np.median(errors)) if errors else None, "errors_s": errors}


def inverse_positive(t: float, duration: float) -> float:
    lo, hi = 0.0, duration
    target = float(t)
    for _ in range(64):
        mid = (lo + hi) / 2.0
        mapped = mid + POSITIVE_AMPLITUDE_S * math.sin(2.0 * math.pi * mid / duration)
        if mapped < target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def positive_clock_check(helper: Any, original_grid: Path, distorted_grid: Path, sham_grid: Path, repair_grid: Path, duration: float) -> dict[str, Any]:
    original, original_labels, _ = recognized_grid(helper, original_grid)
    distorted, distorted_labels, _ = recognized_grid(helper, distorted_grid)
    pairs = helper._lcs_pairs(original, distorted)
    predicted: list[float] = []
    actual: list[float] = []
    original_centers: list[float] = []
    for i, j in pairs:
        c0 = (float(original[i]["start_s"]) + float(original[i]["end_s"])) / 2.0
        ca = (float(distorted[j]["start_s"]) + float(distorted[j]["end_s"])) / 2.0
        original_centers.append(c0)
        predicted.append(inverse_positive(c0, duration))
        actual.append(ca)
    err = np.abs(np.asarray(actual) - np.asarray(predicted)) if predicted else np.asarray([], dtype=np.float64)
    shift = np.abs(np.asarray(predicted) - np.asarray(original_centers)) if predicted else np.asarray([], dtype=np.float64)
    d_clock = grid_clock_error(helper, original_grid, distorted_grid)
    sham_clock = grid_clock_error(helper, original_grid, sham_grid)
    repair_clock = grid_clock_error(helper, original_grid, repair_grid)
    return {
        "matched_count": len(pairs), "expected_vs_actual_median_error_s": float(np.median(err)) if err.size else None,
        "expected_vs_actual_mae_s": float(np.mean(err)) if err.size else None,
        "expected_center_displacement_median_s": float(np.median(shift)) if shift.size else None,
        "actual_distorted_clock_mae_s": d_clock["mae_s"], "distorted_sham_clock_mae_s": sham_clock["mae_s"],
        "repair_clock_mae_s": repair_clock["mae_s"], "repair_clock_median_error_s": repair_clock["median_abs_error_s"],
        "repair_mae_reduction_vs_distorted_sham": (1.0 - repair_clock["mae_s"] / sham_clock["mae_s"]) if sham_clock["mae_s"] and repair_clock["mae_s"] is not None else None,
        "sham_retained_distorted_clock_error": (sham_clock["mae_s"] / d_clock["mae_s"]) if d_clock["mae_s"] else None,
        "distorted_alignment": d_clock, "sham_alignment": sham_clock, "repair_alignment": repair_clock,
    }


def mfa_run(helper: Any, stage_dir: Path, speaker: str, tag: str, entries: Sequence[Mapping[str, Any]], deadline: float) -> dict[str, Any]:
    record: dict[str, Any] = {"speaker": speaker, "tag": tag, "status": "FAILED", "entries": []}
    try:
        pins = helper._check_mfa_pins()
        input_dir = stage_dir / "mfa_input" / speaker / tag
        output_dir = stage_dir / "textgrids" / speaker / tag
        temp_dir = stage_dir / "mfa_temp" / speaker / tag
        for path in (input_dir, output_dir, temp_dir):
            path.mkdir(parents=True, exist_ok=False)
        for item in entries:
            stem = str(item["stem"])
            source_wav = Path(item["wav_path"])
            wav_out = input_dir / f"{stem}.wav"
            shutil.copyfile(source_wav, wav_out)
            (input_dir / f"{stem}.txt").write_text(str(item["transcript"]).strip() + "\n", encoding="utf-8")
            record["entries"].append({"stem": stem, "wav_sha256": sha256_file(wav_out), "transcript_sha256": sha256_file(input_dir / f"{stem}.txt")})
        command = helper._mfa_command(input_dir, output_dir, temp_dir)
        environment = dict(os.environ)
        environment["PATH"] = str(helper.MFA.parent) + os.pathsep + environment.get("PATH", "")
        log_path = stage_dir / "logs" / "mfa" / speaker / f"{tag}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        if log_path.exists():
            raise PilotError(f"refusing to overwrite MFA log: {log_path}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PilotError("CPU stage exceeded 30 minute budget before MFA")
        started = time.monotonic()
        with log_path.open("xb") as log:
            process = subprocess.Popen(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT)
            record.update({"pid": process.pid, "process_start_ticks": process_start_ticks(process.pid), "command": command, "cwd": str(REPO), "input_dir": str(input_dir), "output_dir": str(output_dir), "temporary_directory": str(temp_dir), "log": str(log_path), "mfa_pins": pins, "status": "RUNNING"})
            json_write(stage_dir / "mfa_receipts" / f"{speaker}__{tag}.json", record)
            try:
                returncode = process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                returncode = process.returncode
                record["status"] = "TIMEOUT"
            record.update({"returncode": returncode, "elapsed_seconds": time.monotonic() - started, "log_sha256": sha256_file(log_path)})
        grids = {}
        for item in entries:
            grid = output_dir / f"{item['stem']}.TextGrid"
            if grid.is_file():
                grids[str(item["stem"])] = hash_record(grid)
        record["textgrids"] = grids
        if record["status"] != "TIMEOUT" and record["returncode"] == 0 and len(grids) == len(entries):
            record["status"] = "PASS"
        elif record["status"] != "TIMEOUT":
            record["status"] = "FAILED"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
    json_write(stage_dir / "mfa_receipts" / f"{speaker}__{tag}.json", record)
    return record


def run_score_worker(stage_dir: Path, stage_name: str, jobs: list[dict[str, Any]], deadline: float) -> dict[str, Any]:
    job_path = stage_dir / "scoring" / f"{stage_name}_jobs.json"
    json_write(job_path, {"jobs": jobs})
    log_path = stage_dir / "logs" / f"{stage_name}_score_worker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    worker = REPO / "scripts/experiments/local_clock_crossover_20261016/score_worker.py"
    remaining = deadline - time.monotonic()
    result: dict[str, Any] = {"status": "FAILED", "jobs_file": hash_record(job_path), "worker": hash_record(worker), "python": str(SYNCNET_PY)}
    if remaining <= 0:
        result["error"] = "GPU stage budget exhausted before worker start"
        return result
    with log_path.open("xb") as log:
        process = subprocess.Popen([str(SYNCNET_PY), "-B", str(worker), "--jobs", str(job_path)], cwd=REPO, stdout=log, stderr=subprocess.STDOUT)
        result.update({"pid": process.pid, "process_start_ticks": process_start_ticks(process.pid), "log": str(log_path), "status": "RUNNING"})
        json_write(stage_dir / "scoring" / f"{stage_name}_worker_receipt.json", result)
        try:
            rc = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            rc = process.returncode
            result["status"] = "TIMEOUT"
        result.update({"returncode": rc, "elapsed_seconds": time.monotonic() - (deadline - remaining), "log_sha256": sha256_file(log_path)})
    if result["status"] != "TIMEOUT":
        result["status"] = "PASS" if rc == 0 else "FAILED"
    json_write(stage_dir / "scoring" / f"{stage_name}_worker_receipt.json", result)
    return result


def score_job(output_id: str, speaker: str, role: str, wav: Path, visual: Path, support: Sequence[int], primary_index: int, stage_dir: Path) -> dict[str, Any]:
    out = stage_dir / "scores" / output_id
    return {"id": output_id, "speaker": speaker, "reference_role": role, "wav_path": str(wav), "wav_sha256": sha256_file(wav), "visual_path": str(visual), "visual_sha256": sha256_file(visual), "support": list(support), "primary_index": primary_index, "audio_feature_path": str((out / "A.npy").resolve()), "distance_path": str((out / "D.npy").resolve()), "curve_path": str((out / "curve.npy").resolve()), "receipt_path": str((out / "score.json").resolve())}


def fresh_stage(stage: str, root_admitted: bool) -> Path:
    if stage == "controls":
        if OUT.exists():
            raise PilotError(f"refusing to reuse existing run directory: {OUT}")
        OUT.mkdir(parents=True, exist_ok=False)
        target = OUT / "controls"
        target.mkdir()
        return target
    if not root_admitted:
        raise PilotError("main stage requires root's explicit --root-admitted launch decision")
    control_file = OUT / "controls/control_summary.json"
    if not control_file.is_file():
        raise PilotError("main stage requires completed controls summary")
    summary = json.loads(control_file.read_text(encoding="utf-8"))
    if summary.get("status") != "CONTROL_GATE_PASS":
        raise PilotError("main stage blocked because controls did not pass")
    target = OUT / "main"
    target.mkdir(parents=True, exist_ok=False)
    return target


def prepare_run_receipt(stage_dir: Path, inputs: dict[str, Any], stage: str, admitted: bool) -> None:
    helper = load_timing_helper()
    receipt = {
        "schema_version": 1, "status": "RUNNING", "stage": stage, "speaker_ids": list(SPEAKERS),
        "started_unix": time.time(), "pid": os.getpid(), "process_start_ticks": process_start_ticks(os.getpid()), "cwd": str(REPO), "root_admitted": admitted,
        "prior_preflight_failures": [PRIOR_PREFLIGHT_FAILURE],
        "prior_failed_control_attempts": [PRIOR_FAILED_CONTROL_ATTEMPT],
        "source_hashes": {Path(__file__).name: hash_record(Path(__file__)), "score_worker.py": hash_record(REPO / "scripts/experiments/local_clock_crossover_20261016/score_worker.py"), "report44_timing_diagnostics.py": hash_record(Path(helper.__file__)), "report44_syncnet.py": hash_record(REPO / "scripts/experiments/tts_native_gain_attribution/syncnet.py"), "report44_syncnet_config.py": hash_record(REPO / "scripts/experiments/tts_native_gain_attribution/config.py")},
        "input_bindings": inputs,
        "runtime": {name: importlib.metadata.version(name) for name in ("numpy", "scipy", "librosa", "soundfile")},
        "contract": {"target_rms": TARGET_RMS, "sample_rate_hz": SAMPLE_RATE, "n_fft": N_FFT, "win_length": N_FFT, "hop_length": HOP, "window": "hann", "center": True, "pad_mode": "constant", "phase_vocoder_pitch_policy": "phase increment accumulated once per output hop; not multiplied by local map slope", "q_support": "report44 common support rows only; all 31 lag columns must be real", "primary_lag_source": "report44 self-cell argmin_index", "max_cpu_seconds": CPU_BUDGET_S, "max_gpu_seconds": GPU_BUDGET_S},
    }
    json_write(stage_dir / "run_receipt.json", receipt)


def write_run_status(stage_dir: Path, status: str, extras: Mapping[str, Any]) -> None:
    path = stage_dir / "run_receipt.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    value.update(extras)
    value["status"] = status
    value["finished_unix"] = time.time()
    json_write(path, value)


def run_controls(stage_dir: Path, inputs: dict[str, Any], bindings: dict[str, dict[str, Any]]) -> None:
    started = time.monotonic()
    deadline = started + CPU_BUDGET_S
    helper = load_timing_helper()
    prepare_run_receipt(stage_dir, inputs, "controls", False)
    outputs: dict[str, dict[str, Any]] = {s: {"identities": {}, "positive": {}, "mfa": [], "errors": []} for s in SPEAKERS}
    for speaker in SPEAKERS:
        role_items = bindings[speaker]
        for role in ROLES:
            row = role_items[role]
            name = f"{speaker}__identity__{role}"
            try:
                wav = stage_dir / "audio" / speaker / f"{name}.wav"
                meta = render_pv(row["audio_path"], wav, row["audio_samples"], {"kind": "identity"})
                outputs[speaker]["identities"][role] = {"wav": meta, "textgrid_reference": str(row["textgrid_path"])}
            except Exception as exc:
                outputs[speaker]["errors"].append(f"identity/{role}: {type(exc).__name__}: {exc}")
        row = role_items["N1"]
        try:
            x, meta = read_pcm16(row["audio_path"])
            duration = meta["duration_s"]
            deriv = 1.0 + (2.0 * math.pi * POSITIVE_AMPLITUDE_S / duration) * np.cos(2.0 * np.pi * np.linspace(0.0, duration, 10001) / duration)
            if float(np.min(deriv)) <= 0:
                raise PilotError("known positive mapping is not monotone")
            d_name, sham_name, repair_name = (f"{speaker}__positive_{suffix}" for suffix in ("D", "D_sham", "R"))
            d_path = stage_dir / "audio" / speaker / f"{d_name}.wav"
            d_meta = render_pv(row["audio_path"], d_path, row["audio_samples"], {"kind": "positive_sine", "formula": "t+0.240*sin(2*pi*t/D)", "source_duration_s": duration, "amplitude_s": POSITIVE_AMPLITUDE_S, "derivative_min_sampled": float(np.min(deriv)), "derivative_max_sampled": float(np.max(deriv))})
            sham_path = stage_dir / "audio" / speaker / f"{sham_name}.wav"
            sham_meta = render_pv(d_path, sham_path, row["audio_samples"], {"kind": "identity", "source_condition": d_name})
            outputs[speaker]["positive"].update({"D": d_meta, "D_sham": sham_meta, "duration_s": duration, "names": {"D": d_name, "D_sham": sham_name, "R": repair_name}})
            aligned = mfa_run(helper, stage_dir, speaker, "positive_D_and_sham", [{"stem": d_name, "wav_path": d_path, "transcript": row["transcript"]}, {"stem": sham_name, "wav_path": sham_path, "transcript": row["transcript"]}], deadline)
            outputs[speaker]["mfa"].append(aligned)
            if aligned.get("status") != "PASS":
                outputs[speaker]["errors"].append("positive D/D_sham MFA did not pass")
                continue
            d_grid = Path(aligned["textgrids"][d_name]["path"])
            original_grid = row["textgrid_path"]
            repair_map = fit_phone_map(helper, original_grid, d_grid, duration, duration)
            outputs[speaker]["positive"]["repair_map"] = repair_map
            if repair_map.get("status") != "SUPPORTED":
                outputs[speaker]["errors"].append("positive repair map unsupported")
                continue
            repair_spec = {"kind": "piecewise_linear_phone_centers", "output_anchor_s": repair_map["output_anchor_s"], "input_anchor_s": repair_map["input_anchor_s"], "matched_phones": repair_map["matched_phones"], "segment_slopes": repair_map["segment_slopes"], "unmatched_span_rule": repair_map["unmatched_span_rule"], "source_condition": d_name, "fit_source": "fresh MFA D phone centers to original N1 centers"}
            repair_path = stage_dir / "audio" / speaker / f"{repair_name}.wav"
            repair_meta = render_pv(d_path, repair_path, row["audio_samples"], repair_spec)
            outputs[speaker]["positive"]["R"] = repair_meta
            repaired = mfa_run(helper, stage_dir, speaker, "positive_R", [{"stem": repair_name, "wav_path": repair_path, "transcript": row["transcript"]}], deadline)
            outputs[speaker]["mfa"].append(repaired)
            if repaired.get("status") != "PASS":
                outputs[speaker]["errors"].append("positive repair MFA did not pass")
                continue
            repair_grid = Path(repaired["textgrids"][repair_name]["path"])
            sham_grid = Path(aligned["textgrids"][sham_name]["path"])
            outputs[speaker]["positive"]["clock_check"] = positive_clock_check(helper, original_grid, d_grid, sham_grid, repair_grid, duration)
        except Exception as exc:
            outputs[speaker]["errors"].append(f"positive: {type(exc).__name__}: {exc}")
        json_write(stage_dir / "controls_receipt.json", {"speakers": outputs})

    jobs = []
    for speaker in SPEAKERS:
        for role in ROLES:
            item = outputs[speaker]["identities"].get(role)
            if item:
                row = bindings[speaker][role]
                output_id = f"{speaker}__identity__{role}"
                jobs.append(score_job(output_id, speaker, role, Path(item["wav"]["path"]), row["visual_path"], row["support"], row["primary_index"], stage_dir))
        positive = outputs[speaker]["positive"]
        row = bindings[speaker]["N1"]
        for condition in ("D", "D_sham", "R"):
            item = positive.get(condition)
            if item:
                output_id = f"{speaker}__positive_{condition}"
                jobs.append(score_job(output_id, speaker, "N1", Path(item["path"]), row["visual_path"], row["support"], row["primary_index"], stage_dir))
    worker = run_score_worker(stage_dir, "controls", jobs, time.monotonic() + GPU_BUDGET_S)
    gate = control_gates(stage_dir, bindings, outputs)
    json_write(stage_dir / "controls_receipt.json", {"speakers": outputs, "scoring_worker": worker, "score_job_count": len(jobs), "expected_score_job_count": 28})
    json_write(stage_dir / "control_summary.json", gate)
    write_run_status(stage_dir, "CONTROL_GATE_PASS" if gate["passing_speakers"] >= 3 else "CONTROL_GATE_FAIL", {"outputs": "controls_receipt.json", "summary": "control_summary.json", "elapsed_seconds": time.monotonic() - started})


def control_gates(stage_dir: Path, bindings: dict[str, dict[str, Any]], outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    speakers: dict[str, Any] = {}
    for speaker in SPEAKERS:
        identity_rows = []
        for role in ROLES:
            out_path = stage_dir / "scores" / f"{speaker}__identity__{role}" / "score.json"
            base = bindings[speaker][role]
            if not out_path.is_file():
                identity_rows.append({"role": role, "status": "MISSING_SCORE"})
                continue
            scored = json.loads(out_path.read_text(encoding="utf-8"))
            if scored.get("status") != "PASS" or not isinstance(scored.get("metrics"), dict):
                identity_rows.append({"role": role, "status": scored.get("status", "INVALID_SCORE"), "error": scored.get("error")})
                continue
            fixed_distance = float(scored["metrics"]["primary_fixedlag_distance"])
            c_change = abs(float(scored["metrics"]["C"]) - base["baseline_C"])
            score_support_ok = int(scored["metrics"].get("support_count", -1)) == len(base["support"]) and scored["metrics"].get("all_lag_columns_real_features") is True
            expected_samples = base["audio_samples"]
            audio_qc = outputs[speaker]["identities"].get(role, {}).get("wav", {})
            audio_qc_ok = audio_qc.get("samples") == expected_samples and audio_qc.get("rms_error", 9.0) <= 2e-6 and audio_qc.get("clip_count") == 0
            identity_rows.append({"role": role, "status": "PASS", "primary_fixedlag_distance": fixed_distance, "distance_abs_change": abs(fixed_distance - base["baseline_primary_distance"]), "C_abs_change": c_change, "distance_pass": abs(fixed_distance - base["baseline_primary_distance"]) <= 0.10, "C_pass": c_change <= 0.25, "score_support_pass": score_support_ok, "audio_qc_pass": audio_qc_ok, "audio_qc": audio_qc})
        pos = outputs[speaker]["positive"]
        clock = pos.get("clock_check", {})
        score_paths = {c: stage_dir / "scores" / f"{speaker}__positive_{c}" / "score.json" for c in ("D", "D_sham", "R")}
        score_values = {}
        for condition, path in score_paths.items():
            scored = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
            score_values[condition] = scored if scored and scored.get("status") == "PASS" and isinstance(scored.get("metrics"), dict) else None
        identity_n1 = next((row for row in identity_rows if row.get("role") == "N1" and row.get("status") == "PASS"), {})
        base = identity_n1.get("primary_fixedlag_distance")
        d_distance = score_values["D"]["metrics"]["primary_fixedlag_distance"] if score_values["D"] else None
        sham_distance = score_values["D_sham"]["metrics"]["primary_fixedlag_distance"] if score_values["D_sham"] else None
        r_distance = score_values["R"]["metrics"]["primary_fixedlag_distance"] if score_values["R"] else None
        degradation = (sham_distance - base) if sham_distance is not None and base is not None else None
        recovery = ((sham_distance - r_distance) / degradation) if degradation and r_distance is not None else None
        dist_mae = clock.get("actual_distorted_clock_mae_s")
        sham_mae = clock.get("distorted_sham_clock_mae_s")
        repair_mae = clock.get("repair_clock_mae_s")
        positive_audio_qc = {}
        for condition in ("D", "D_sham", "R"):
            item = pos.get(condition)
            qc = item if isinstance(item, dict) else {}
            positive_audio_qc[condition] = bool(qc and qc.get("samples") == bindings[speaker]["N1"]["audio_samples"] and qc.get("rms_error", 9.0) <= 2e-6 and qc.get("clip_count") == 0)
        positive_support_pass = all(score_values[c] is not None and int(score_values[c]["metrics"].get("support_count", -1)) == len(bindings[speaker]["N1"]["support"]) and score_values[c]["metrics"].get("all_lag_columns_real_features") is True for c in ("D", "D_sham", "R"))
        positive_alignment = {
            "D": clock.get("distorted_alignment", {}),
            "D_sham": clock.get("sham_alignment", {}),
            "R": clock.get("repair_alignment", {}),
        }
        positive_mfa_coverage = {
            condition: {
                "reference_coverage": alignment.get("reference_coverage", 0.0),
                "actual_coverage": alignment.get("actual_coverage", 0.0),
                "pass": alignment.get("reference_coverage", 0.0) >= 0.85 and alignment.get("actual_coverage", 0.0) >= 0.85,
            }
            for condition, alignment in positive_alignment.items()
        }
        positive_mfa_coverage_pass = all(row["pass"] for row in positive_mfa_coverage.values())
        identity_pass = len(identity_rows) == 4 and all(row.get("distance_pass") and row.get("C_pass") and row.get("score_support_pass") and row.get("audio_qc_pass") for row in identity_rows)
        gates = {
            "identity_all_roles_pass": identity_pass,
            "positive_output_pcm_rms_length_pass": all(positive_audio_qc.values()),
            "positive_common_support_and_31_real_lags_pass": positive_support_pass,
            "positive_fresh_mfa_coverage_ge_0_85": positive_mfa_coverage_pass,
            "expected_center_displacement_ge_80ms": clock.get("expected_center_displacement_median_s") is not None and clock["expected_center_displacement_median_s"] >= 0.080,
            "distorted_actual_vs_analytic_median_le_40ms": clock.get("expected_vs_actual_median_error_s") is not None and clock["expected_vs_actual_median_error_s"] <= 0.040,
            "repair_median_le_40ms": clock.get("repair_clock_median_error_s") is not None and clock["repair_clock_median_error_s"] <= 0.040,
            "repair_mae_reduction_ge_60pct": clock.get("repair_mae_reduction_vs_distorted_sham") is not None and clock["repair_mae_reduction_vs_distorted_sham"] >= 0.60,
            "sham_retains_ge_75pct_distortion": clock.get("sham_retained_distorted_clock_error") is not None and clock["sham_retained_distorted_clock_error"] >= 0.75,
            "sham_fixed_lag_degradation_ge_0_10": degradation is not None and degradation >= 0.10,
            "repair_recovers_ge_50pct_fixed_lag_degradation": recovery is not None and recovery >= 0.50,
        }
        speakers[speaker] = {"status": "PASS" if all(gates.values()) else "FAIL", "gates": gates, "identity_rows": identity_rows, "positive_audio_qc": positive_audio_qc, "positive_mfa_coverage": positive_mfa_coverage, "positive_clock": clock, "positive_fixedlag_distances": {"baseline_identity": base, "D": d_distance, "D_sham": sham_distance, "R": r_distance, "D_sham_degradation": degradation, "repair_recovery_fraction": recovery, "repair_residual_excess_diagnostic_only": (r_distance - base) if r_distance is not None and base is not None else None}, "errors": outputs[speaker]["errors"]}
    passing = sum(row["status"] == "PASS" for row in speakers.values())
    return {"status": "CONTROL_GATE_PASS" if passing >= 3 else "CONTROL_GATE_FAIL", "passing_speakers": passing, "required_speakers": 3, "speaker_results": speakers, "expected_identity_audio_forwards": 16, "expected_positive_audio_forwards": 12, "expected_total_audio_forwards": 28, "observed_scored_audio_conditions": sum(1 for p in (stage_dir / "scores").glob("*/score.json")), "launch_policy": "main stage remains gated by explicit root admission"}


def run_main(stage_dir: Path, inputs: dict[str, Any], bindings: dict[str, dict[str, Any]], admitted: bool) -> None:
    started = time.monotonic()
    deadline = started + CPU_BUDGET_S
    helper = load_timing_helper()
    prepare_run_receipt(stage_dir, inputs, "main", admitted)
    outputs: dict[str, list[dict[str, Any]]] = {s: [] for s in SPEAKERS}
    jobs: list[dict[str, Any]] = []
    for speaker in SPEAKERS:
        rows = bindings[speaker]
        for ref in REFERENCES:
            ref_row = rows[ref]
            _, ref_pcm = read_pcm16(ref_row["audio_path"])
            for source_role in ROLES:
                if source_role == ref:
                    continue
                src_row = rows[source_role]
                _, src_pcm = read_pcm16(src_row["audio_path"])
                local = fit_phone_map(helper, ref_row["textgrid_path"], src_row["textgrid_path"], ref_pcm["duration_s"], src_pcm["duration_s"])
                pair = {"speaker": speaker, "reference_role": ref, "source_role": source_role, "local_map": local, "outputs": {}}
                if local.get("status") != "SUPPORTED":
                    pair["status"] = "UNSUPPORTED_LOCAL_MAP"
                    outputs[speaker].append(pair)
                    continue
                maps = {
                    "endpoint_affine": {"kind": "endpoint_affine", "output_anchor_s": [0.0, ref_pcm["duration_s"]], "input_anchor_s": [0.0, src_pcm["duration_s"]], "slope": src_pcm["duration_s"] / ref_pcm["duration_s"]},
                    "local": {"kind": local["kind"], "output_anchor_s": local["output_anchor_s"], "input_anchor_s": local["input_anchor_s"], "matched_phones": local["matched_phones"], "segment_slopes": local["segment_slopes"], "unmatched_span_rule": local["unmatched_span_rule"]},
                }
                for method, map_spec in maps.items():
                    if method == "endpoint_affine" and not 0.25 <= map_spec["slope"] <= 4.0:
                        pair["outputs"][method] = {"status": "UNSUPPORTED_AFFINE_SLOPE", "slope": map_spec["slope"]}
                        continue
                    output_id = f"{speaker}__{ref}__{source_role}__{method}"
                    wav = stage_dir / "audio" / speaker / f"{output_id}.wav"
                    try:
                        rendered = render_pv(src_row["audio_path"], wav, ref_row["audio_samples"], map_spec)
                        pair["outputs"][method] = {"status": "RENDERED", "wav": rendered, "map": map_spec}
                        jobs.append(score_job(output_id, speaker, ref, wav, ref_row["visual_path"], ref_row["support"], ref_row["primary_index"], stage_dir))
                    except Exception as exc:
                        pair["outputs"][method] = {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}
                pair["status"] = "RENDERED" if all(pair["outputs"].get(k, {}).get("status") == "RENDERED" for k in ("endpoint_affine", "local")) else "PARTIAL"
                outputs[speaker].append(pair)
    json_write(stage_dir / "main_receipt.json", {"outputs": outputs, "audio_job_count": len(jobs), "expected_max_audio_job_count": 48})
    for speaker in SPEAKERS:
        entries = []
        for pair in outputs[speaker]:
            if pair["status"] not in {"RENDERED", "PARTIAL"}:
                continue
            transcript = bindings[speaker][pair["source_role"]]["transcript"]
            for method, item in pair["outputs"].items():
                if item.get("status") == "RENDERED":
                    wav = Path(item["wav"]["path"])
                    entries.append({"stem": wav.stem, "wav_path": wav, "transcript": transcript})
        if entries:
            mfa = mfa_run(helper, stage_dir, speaker, "main_outputs", entries, deadline)
            for pair in outputs[speaker]:
                for method, item in pair["outputs"].items():
                    if item.get("status") != "RENDERED":
                        continue
                    stem = Path(item["wav"]["path"]).stem
                    grid_record = mfa.get("textgrids", {}).get(stem)
                    if grid_record:
                        item["textgrid"] = grid_record
                        item["clock_check"] = grid_clock_error(helper, bindings[speaker][pair["reference_role"]]["textgrid_path"], Path(grid_record["path"]))
                    else:
                        item["textgrid_status"] = mfa.get("status", "FAILED")
    worker = run_score_worker(stage_dir, "main", jobs, time.monotonic() + GPU_BUDGET_S)
    for speaker in SPEAKERS:
        for pair in outputs[speaker]:
            for method, item in pair["outputs"].items():
                if item.get("status") != "RENDERED":
                    continue
                score_path = stage_dir / "scores" / Path(item["wav"]["path"]).stem / "score.json"
                if score_path.is_file():
                    item["score"] = json.loads(score_path.read_text(encoding="utf-8"))
            affine = pair["outputs"].get("endpoint_affine", {})
            local = pair["outputs"].get("local", {})
            ac, lc = affine.get("clock_check", {}), local.get("clock_check", {})
            affine_mae, local_mae = ac.get("mae_s"), lc.get("mae_s")
            local_median = lc.get("median_abs_error_s")
            coverage_ok = all(
                result.get(key, 0.0) >= 0.85
                for result in (ac, lc)
                for key in ("reference_coverage", "actual_coverage")
            )
            score_ok = all(
                isinstance(item.get("score"), dict)
                and item["score"].get("status") == "PASS"
                and item["score"].get("metrics", {}).get("support_count") == len(bindings[speaker][pair["reference_role"]]["support"])
                and item["score"].get("metrics", {}).get("all_lag_columns_real_features") is True
                for item in (affine, local)
            )
            audio_ok = all(
                item.get("status") == "RENDERED"
                and item.get("wav", {}).get("samples") == bindings[speaker][pair["reference_role"]]["audio_samples"]
                and item.get("wav", {}).get("rms_error", 9.0) <= 2e-6
                and item.get("wav", {}).get("clip_count") == 0
                for item in (affine, local)
            )
            if affine_mae is None or local_mae is None or local_median is None:
                clock_status = "UNQUALIFIED_MISSING_MFA"
            elif not coverage_ok or not audio_ok or not score_ok:
                clock_status = "UNQUALIFIED_TECHNICAL_GATE"
            elif local_median > 0.040:
                clock_status = "FAIL_LOCAL_MEDIAN_OVER_40MS"
            elif float(affine_mae) < 0.020:
                clock_status = "ALREADY_ALIGNED"
            elif float(affine_mae) > 0 and float(local_mae) <= 0.40 * float(affine_mae):
                clock_status = "PASS"
            else:
                clock_status = "FAIL_LOCAL_MAE_REDUCTION_UNDER_60PCT"
            pair["clock_comparison"] = {
                "status": clock_status, "qualified": clock_status == "PASS",
                "affine_mae_s": affine_mae, "local_mae_s": local_mae, "local_median_abs_error_s": local_median,
                "local_mae_reduction_fraction": (1.0 - float(local_mae) / float(affine_mae)) if affine_mae and local_mae is not None else None,
                "phone_coverage_pass": coverage_ok, "audio_qc_pass": audio_ok, "fixed_support_pass": score_ok,
            }
            pair["status"] = "QUALIFIED" if pair["clock_comparison"]["qualified"] else pair.get("status", "UNQUALIFIED")
    extension_speakers = {}
    role_class = {"N1": "N", "N2": "N", "T1": "T", "T2": "T"}
    for speaker in SPEAKERS:
        by_reference = {}
        for ref in REFERENCES:
            qualified = [p for p in outputs[speaker] if p["reference_role"] == ref and p.get("clock_comparison", {}).get("qualified")]
            within = [p for p in qualified if role_class[p["source_role"]] == role_class[ref]]
            cross = [p for p in qualified if role_class[p["source_role"]] != role_class[ref]]
            by_reference[ref] = {"qualified_within_count": len(within), "qualified_cross_count": len(cross), "within_and_cross_pass": bool(within) and len(cross) >= 1}
        extension_speakers[speaker] = {"references": by_reference, "eligible_for_extension": all(x["within_and_cross_pass"] for x in by_reference.values())}
    extension_count = sum(row["eligible_for_extension"] for row in extension_speakers.values())
    json_write(stage_dir / "main_receipt.json", {"outputs": outputs, "audio_job_count": len(jobs), "expected_max_audio_job_count": 48, "scoring_worker": worker, "elapsed_seconds": time.monotonic() - started})
    json_write(stage_dir / "main_technical_summary.json", {"speaker_results": extension_speakers, "extension_eligible_speakers": extension_count, "extension_gate_pass": extension_count >= 3, "extension_policy": "technical and actual-clock gates only; no score-sign selection"})
    write_run_status(stage_dir, "MAIN_COMPLETE", {"outputs": "main_receipt.json", "elapsed_seconds": time.monotonic() - started})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("controls", "main"))
    parser.add_argument("--root-admitted", action="store_true", help="record root's explicit main-stage admission")
    args = parser.parse_args()
    inputs, bindings = load_bindings()
    stage_dir = fresh_stage(args.stage, args.root_admitted)
    json_write(stage_dir / "input_bindings.json", inputs)
    if args.stage == "controls":
        run_controls(stage_dir, inputs, bindings)
    else:
        run_main(stage_dir, inputs, bindings, args.root_admitted)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"LOCAL_CLOCK_PILOT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
