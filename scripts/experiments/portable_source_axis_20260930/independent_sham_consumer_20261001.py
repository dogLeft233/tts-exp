#!/usr/bin/env python3
"""Independently rebuild and summarize the sealed 80-sample resampling sham.

This CPU-only consumer intentionally imports neither the producer nor its
``ops``/``scoring`` modules. It validates the saved row/artifact grid, rebuilds
raw and unit embedding cross-distance, zero-distance, lag, anchor, and support
metrics, then computes equal-speaker bootstrap contrasts from cached arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import wave
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np


EXPECTED_PROTOCOL_SHA256 = "2b54680fb73c158faf3bd7b7c5ec34894c60ac576a2ecbde308c4c6eb79e1eb4"
EXPECTED_ROW_SHA256 = "65fa7638eb2c3f73086db5c742c849161aa7eaf87aae10af3b4b9da9d429f934"
EXPECTED_INPUT_SHA256 = "6b0ce3252cf776a86cf84b19ab53afd936526216e4ddb30dc587a72a66409b4b"
EXPECTED_RESAMPLING_SHA256 = "160b2a0a26ae9f489734751e98128ec4461f9bbfd1d9e8fa6b9c9ebdc28a07ae"
EXPECTED_CODE_SHA256 = {
    "resampling_sham.py": "736b779c45b6ba0431f083bff1d73d0efc0b2b4d48c7182aa92ffc078e465aea",
    "evaluate.py": "e50a31eaa2493b83e87cb4426b742e8f28e73836cbb9374b2319e7b60c940880",
    "ops.py": "64eecefea72a494cc07573c8bbdd872351e23b9f7998e52b280483af3b9c0a3b",
    "scoring.py": "9411436d9fc199e91116a44523eeeccb66be0e59f7f20f491b0b9c8cc5a3ad87",
    "wav2lip.py": "f0a6cd180679aa1bb218df845632211731651e122b5baa706a38a3db32ccfe9d",
}
CONDITIONS = {
    "q00": ("V_RAW_N", "A_RAW_N"),
    "q10": ("V_RAW_S", "A_RAW_N"),
    "q01": ("V_RAW_N", "A_RAW_S"),
    "q11": ("V_RAW_S", "A_RAW_S"),
}
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C", "C_anchor", "B", "D", "search_gain")
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_930
VSHIFT = 15
EPSILON = np.float32(1e-6)
DISTANCE_ATOL = 5e-4
METRIC_ATOL = 5e-4


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(payload: dict[str, Any], field: str) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in payload.items() if k != field})).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(result, dict):
        raise AuditError(f"expected JSON object: {path}")
    return result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise AuditError(f"JSONL item is not an object: {path}:{line_no}")
            rows.append(value)
    return rows


def safe_path(root: Path, rel: str) -> Path:
    parts = PurePosixPath(str(rel)).parts
    if not parts or PurePosixPath(str(rel)).is_absolute() or any(x in ("", "..") for x in parts):
        raise AuditError(f"artifact path escapes the sealed run: {rel}")
    path = root.joinpath(*parts).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise AuditError(f"artifact path escapes the sealed run: {rel}") from exc
    return path


def check_bytes(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise AuditError(f"missing {label}: {path}")
    actual = sha256_file(path)
    if actual != expected:
        raise AuditError(f"{label} hash mismatch: expected {expected}, got {actual}")


def read_mono_pcm16(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as wav:
        rate, channels, width = wav.getframerate(), wav.getnchannels(), wav.getsampwidth()
        if rate != 16000 or channels != 1 or width != 2:
            raise AuditError(f"expected mono 16-kHz PCM16 WAV: {path}")
        payload = bytearray()
        while True:
            block = wav.readframes(1 << 20)
            if not block:
                break
            payload.extend(block)
    if len(payload) % 2:
        raise AuditError(f"PCM16 WAV payload has an odd byte count: {path}")
    return int(rate), np.frombuffer(payload, dtype="<i2").copy()


def expected_supports(n: int) -> dict[str, np.ndarray]:
    all_i = np.arange(n, dtype=np.int32)
    return {
        "full": all_i,
        "valid": all_i.copy(),
        "guard20": np.arange(min(20, n), max(min(20, n), n - 20), dtype=np.int32),
        "guard15": np.arange(min(15, n), max(min(15, n), n - 15), dtype=np.int32),
    }


def unit_rows(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value, dtype=np.float32)
    norm = np.sqrt(np.sum(data * data, axis=1, keepdims=True, dtype=np.float32), dtype=np.float32)
    return (data / np.maximum(norm, np.float32(1e-12))).astype(np.float32, copy=False)


def cross_and_zero(v: np.ndarray, a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    visual = np.asarray(v, dtype=np.float32)
    audio = np.asarray(a, dtype=np.float32)
    if visual.ndim != 2 or audio.ndim != 2 or visual.shape[1] != audio.shape[1] or not len(visual) or not len(audio):
        raise AuditError("visual/audio embedding arrays are not aligned nonempty matrices")
    result = np.empty((len(visual), len(audio)), dtype=np.float32)
    # Bounded batches keep the 75x75x1024 temporary small on CPU.
    for start in range(0, len(visual), 4):
        stop = min(start + 4, len(visual))
        delta = visual[start:stop, None, :] - audio[None, :, :]
        np.add(delta, EPSILON, out=delta)
        squared = np.multiply(delta, delta, dtype=np.float32)
        squared_sum = np.sum(squared, axis=2, dtype=np.float32)
        np.sqrt(squared_sum, out=result[start:stop])
    zero_delta = visual.copy()
    np.add(zero_delta, EPSILON, out=zero_delta)
    zero_sum = np.sum(np.multiply(zero_delta, zero_delta, dtype=np.float32), axis=1, dtype=np.float32)
    zero = np.sqrt(zero_sum, dtype=np.float32).astype(np.float32, copy=False)
    return result, zero


def lag_matrix(cross: np.ndarray, zero: np.ndarray, vshift: int) -> np.ndarray:
    value = np.asarray(cross, dtype=np.float32)
    boundary = np.asarray(zero, dtype=np.float32)
    if value.ndim != 2 or boundary.shape != (value.shape[0],):
        raise AuditError("cross/zero geometry is invalid")
    out = np.empty((value.shape[0], 2 * vshift + 1), dtype=np.float32)
    for i in range(value.shape[0]):
        for col in range(2 * vshift + 1):
            j = i + col - vshift
            out[i, col] = value[i, j] if 0 <= j < value.shape[1] else boundary[i]
    return out


def metric_payload(matrix: np.ndarray, supports: dict[str, np.ndarray], anchor_col: int, vshift: int) -> dict[str, Any]:
    out = {}
    for name, indices in supports.items():
        if len(indices) == 0:
            out[name] = {"status": "too_short", "support_n": 0}
            continue
        curve = np.asarray(matrix[indices], dtype=np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
        best = int(np.argmin(curve))
        median = float(np.median(curve))
        dmin = float(curve[best])
        d0 = float(curve[vshift])
        danchor = float(curve[anchor_col])
        out[name] = {
            "status": "ok", "support_n": int(len(indices)), "window_indices": indices.tolist(),
            "curve": curve.tolist(), "lag_offsets_frames": [int(vshift - i) for i in range(len(curve))],
            "optimal_column": best, "optimal_offset_frames": int(vshift - best),
            "D": dmin, "D_min": dmin, "B": median, "B_median": median,
            "C": median - dmin, "C_best": median - dmin,
            "D_anchor": danchor, "D_fixed_anchor": danchor,
            "C_anchor": median - danchor, "C_fixed_anchor": median - danchor,
            "search_gain": d0 - dmin, "D_zero": d0,
            "fixed_anchor_column": int(anchor_col), "fixed_anchor_offset_frames": int(vshift - anchor_col),
        }
    return out


def compare_matrix(actual: np.ndarray, expected: np.ndarray, label: str) -> float:
    value = np.asarray(actual)
    if value.dtype != np.float32 or value.shape != expected.shape or not np.isfinite(value).all():
        raise AuditError(f"saved {label} has wrong dtype/shape or nonfinite values")
    max_abs = float(np.max(np.abs(value - expected)))
    if max_abs > DISTANCE_ATOL:
        raise AuditError(f"saved {label} differs from independent rebuild by {max_abs:.8g} (limit {DISTANCE_ATOL})")
    return max_abs


def validate_metric_match(saved: dict[str, Any], rebuilt: dict[str, Any], label: str) -> float:
    largest = 0.0
    for support in SUPPORTS:
        s, r = saved.get(support), rebuilt.get(support)
        if not isinstance(s, dict) or not isinstance(r, dict):
            raise AuditError(f"{label} missing support {support}")
        if s.get("status") != r.get("status") or int(s.get("support_n", -1)) != int(r.get("support_n", -2)):
            raise AuditError(f"{label} support status/count differs for {support}")
        if r["status"] == "too_short":
            continue
        if s.get("window_indices") != r.get("window_indices"):
            raise AuditError(f"{label} support indices differ for {support}")
        for key in ("D_min", "B_median", "C_best", "D_fixed_anchor", "C_fixed_anchor"):
            diff = abs(float(s[key]) - float(r[key]))
            largest = max(largest, diff)
            if diff > METRIC_ATOL:
                raise AuditError(f"{label} {support}.{key} differs by {diff:.8g}")
        curve_diff = float(np.max(np.abs(np.asarray(s["curve"], dtype=np.float64) - np.asarray(r["curve"], dtype=np.float64))))
        largest = max(largest, curve_diff)
        if curve_diff > METRIC_ATOL:
            raise AuditError(f"{label} {support}.curve differs by {curve_diff:.8g}")
    return largest


def speaker_bootstrap(rows_by_speaker: dict[str, list[float]]) -> dict[str, Any]:
    speakers = sorted(rows_by_speaker)
    if len(speakers) != 40 or any(len(rows_by_speaker[s]) != 2 for s in speakers):
        raise AuditError("bootstrap requires exactly two clips per each of 40 speakers")
    values = np.asarray([np.mean(rows_by_speaker[s], dtype=np.float64) for s in speakers], dtype=np.float64)
    generator = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = generator.integers(0, len(speakers), size=(BOOTSTRAP_DRAWS, len(speakers)), dtype=np.int64)
    boot = values[draws].mean(axis=1, dtype=np.float64)
    low, high = np.quantile(boot, [0.005, 0.995], method="linear")
    return {
        "estimate_equal_speaker_mean": float(values.mean(dtype=np.float64)),
        "speaker_mean_sd": float(values.std(ddof=1, dtype=np.float64)),
        "speaker_count": int(len(speakers)), "clips_per_speaker": 2,
        "bootstrap": {"method": "equal-speaker cluster bootstrap; sample 40 speakers with replacement; each has two clip means", "rng": "numpy.PCG64", "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "confidence_level": 0.99, "ci_percentiles": [0.5, 99.5], "ci_low": float(low), "ci_high": float(high)},
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root = args.run_root.resolve(strict=True)
    out = args.outdir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite independent attempt: {out}")
    out.mkdir(parents=True, exist_ok=True)
    protocol_path = run_root / "evaluation_protocol.json"
    row_path = run_root / "evaluation_rows.jsonl"
    result_path = run_root / "evaluation_results.jsonl"
    status_path = run_root / "run_status.json"
    seal_path = run_root / "code_seal.json"
    all_hashes_path = run_root / "all_files_sha256.json"
    protocol = read_json(protocol_path)
    status = read_json(status_path)
    code_seal = read_json(seal_path)
    all_hashes = read_json(all_hashes_path)
    if protocol.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256 or fingerprint(protocol, "protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise AuditError("sealed sham protocol fingerprint differs")
    if sha256_file(row_path) != EXPECTED_ROW_SHA256 or protocol.get("row_manifest_sha256") != EXPECTED_ROW_SHA256:
        raise AuditError("sealed sham row plan bytes/hash differ")
    if status.get("status") != "complete" or any(int(status.get(k, -1)) != v for k, v in (("planned_samples", 80), ("completed_samples", 80), ("failed_samples", 0), ("planned_rows", 320), ("completed_rows", 320), ("failed_rows", 0))):
        raise AuditError("sealed sham producer run status is incomplete")
    for name in ("protocol_sha256", "row_manifest_sha256", "code_sha256"):
        if status.get(name) != protocol.get({"protocol_sha256": "protocol_sha256", "row_manifest_sha256": "row_manifest_sha256", "code_sha256": "code_sha256"}[name]):
            raise AuditError(f"run status and protocol differ for {name}")
    if code_seal.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise AuditError("code seal is not bound to the frozen sham protocol")
    if protocol.get("evaluation_count") != 80 or protocol.get("speaker_count") != 40 or protocol.get("planned_rows") != 320:
        raise AuditError("protocol cohort/grid counts differ")
    if protocol.get("conditions") != {k: list(v) for k, v in CONDITIONS.items()}:
        raise AuditError("frozen condition-to-visual/audio cells differ")
    if (protocol.get("frozen_anchor_k"), protocol.get("frozen_anchor_column"), protocol.get("frozen_anchor_official_syncnet_offset"), protocol.get("vshift")) != (3, 18, -3, 15):
        raise AuditError("frozen +3 sample anchor / -3 official offset / 31 lag geometry differs")
    required_flags = {"cudnn_benchmark": False, "cudnn_deterministic": True, "deterministic_algorithms": True}
    if protocol.get("post_detection_flags") != required_flags:
        raise AuditError("post-detection forward flags are not the frozen deterministic contract")
    code_hashes = protocol.get("code_sha256", {})
    if code_hashes != EXPECTED_CODE_SHA256 or code_seal.get("code_sha256") != EXPECTED_CODE_SHA256:
        raise AuditError("producer source seal differs from the frozen code hashes")
    input_meta = protocol.get("input_manifest", {})
    resampling_meta = protocol.get("resampling_manifest", {})
    input_path, resampling_path = Path(input_meta.get("path", "")), Path(resampling_meta.get("path", ""))
    check_bytes(input_path, EXPECTED_INPUT_SHA256, "frozen source input manifest")
    check_bytes(resampling_path, EXPECTED_RESAMPLING_SHA256, "frozen resampling manifest")
    if input_meta.get("sha256") != EXPECTED_INPUT_SHA256 or resampling_meta.get("sha256") != EXPECTED_RESAMPLING_SHA256:
        raise AuditError("protocol external input hash bindings differ")
    input_manifest = read_json(input_path)
    resampling_manifest = read_json(resampling_path)
    source_rows = {str(r["id"]): r for r in input_manifest.get("records", []) if r.get("split") == "evaluation"}
    resampling_rows = {str(r["id"]): r for r in resampling_manifest.get("records", []) if r.get("split") == "evaluation"}
    if len(source_rows) != 80 or set(source_rows) != set(resampling_rows):
        raise AuditError("frozen input and resampling manifests do not match the 80-sample evaluation set")

    rows, results = read_jsonl(row_path), read_jsonl(result_path)
    if len(rows) != 320 or len(results) != 320:
        raise AuditError("planned/results row count differs from 320")
    row_by_id = {str(r["row_id"]): r for r in rows}
    result_by_id = {str(r["row_id"]): r for r in results}
    if len(row_by_id) != 320 or len(result_by_id) != 320 or set(row_by_id) != set(result_by_id):
        raise AuditError("planned/results row identities do not form the same unique 320 rows")
    sample_order = []
    sample_speakers: dict[str, str] = {}
    rows_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample = str(row.get("sample_id")); speaker = str(row.get("speaker_id"))
        if row.get("family") != "resampling_sham" or row.get("split") != "evaluation" or row.get("face_id") != "primary":
            raise AuditError(f"unexpected sham row family/split/face: {row.get('row_id')}")
        if row.get("condition") not in CONDITIONS:
            raise AuditError(f"unexpected sham cross cell: {row.get('row_id')}")
        expected_v, expected_a = CONDITIONS[str(row["condition"])]
        if row.get("visual_id") != expected_v or row.get("audio_id") != expected_a:
            raise AuditError(f"row condition maps to unexpected V/A cell: {row.get('row_id')}")
        if row.get("status") != "planned":
            raise AuditError(f"plan row was unexpectedly changed: {row.get('row_id')}")
        rows_by_sample[sample].append(row)
        sample_speakers[sample] = speaker
        if sample not in sample_order:
            sample_order.append(sample)
    if len(sample_order) != 80 or set(sample_speakers.values()) != {str(r["speaker"]) for r in source_rows.values()}:
        raise AuditError("sample order or speaker identity differs from the frozen input manifest")
    if len(set(sample_speakers.values())) != 40 or any(Counter(sample_speakers.values())[s] != 2 for s in set(sample_speakers.values())):
        raise AuditError("cohort is not 40 speakers with two clips each")
    if any(len(rows_by_sample[s]) != 4 or {r["condition"] for r in rows_by_sample[s]} != set(CONDITIONS) for s in sample_order):
        raise AuditError("each sample must have exactly the four planned q cross cells")

    if not isinstance(all_hashes, dict):
        raise AuditError("all_files_sha256.json must be a path-to-digest object")
    for rel, expected in (("evaluation_protocol.json", sha256_file(protocol_path)), ("evaluation_rows.jsonl", EXPECTED_ROW_SHA256), ("evaluation_results.jsonl", sha256_file(result_path)), ("run_status.json", sha256_file(status_path)), ("code_seal.json", sha256_file(seal_path))):
        if all_hashes.get(rel) != expected:
            raise AuditError(f"all-files seal differs for {rel}")
    for code_name, expected in EXPECTED_CODE_SHA256.items():
        code_path = args.code_root.resolve() / code_name
        check_bytes(code_path, expected, f"sealed producer source {code_name}")

    if len(status.get("forward_events", [])) == 0:
        raise AuditError("producer recorded no forward events")
    bad_events = [e for e in status["forward_events"] if e.get("flags_at_call") != required_flags]
    if bad_events:
        raise AuditError(f"{len(bad_events)} producer forward events violate deterministic flags")
    if status.get("face_flags", {}).get("after_override") != required_flags:
        raise AuditError("face detector forward flags were not restored to the frozen deterministic state")

    detailed_rows: list[dict[str, Any]] = []
    metrics_by_sample: dict[str, dict[str, dict[str, dict[str, dict[str, Any]]]]] = defaultdict(lambda: defaultdict(dict))
    maximum_differences: dict[str, float] = defaultdict(float)
    verified_contexts = verified_visuals = verified_scores = 0
    pcm_frame_counts: list[int] = []
    recompute_dir = out / "recomputed" / "scores"
    recompute_dir.mkdir(parents=True, exist_ok=True)
    required_support_arrays = {"full": "common_support_full_int32", "guard20": "common_support_guard20_int32", "guard15": "common_support_guard15_int32"}
    for sample_index, sample_id in enumerate(sample_order):
        sample_rows = rows_by_sample[sample_id]
        source = source_rows.get(sample_id)
        sham_source = resampling_rows.get(sample_id)
        if source is None or sham_source is None or sample_speakers[sample_id] != str(source["speaker"]):
            raise AuditError(f"sample has no matching frozen source/resampling identity: {sample_id}")
        expected_natural_sha = str(source.get("ref_sha256"))
        if sham_source.get("source_natural16_file_sha256") != expected_natural_sha:
            raise AuditError(f"resampling source WAV identity mismatch: {sample_id}")
        preserved = sham_source.get("length_preserved_audio", {})
        if not preserved.get("file_sha256"):
            raise AuditError(f"length-preserved resampling entry missing: {sample_id}")
        sample_results = [result_by_id[str(r["row_id"])] for r in sample_rows]
        if any(r.get("status") != "complete" for r in sample_results):
            raise AuditError(f"one or more sample rows did not complete: {sample_id}")
        canonical_result = sorted(sample_results, key=lambda r: str(r["condition"]))[0]
        context_path = safe_path(run_root, str(canonical_result["context_npz"]))
        context_rel = context_path.relative_to(run_root).as_posix()
        expected_context_sha = str(canonical_result["context_sha256"])
        check_bytes(context_path, expected_context_sha, f"{sample_id} context artifact")
        if all_hashes.get(context_rel) != expected_context_sha:
            raise AuditError(f"all-files seal differs for context {sample_id}")
        with np.load(context_path, allow_pickle=False) as archive:
            context = {k: archive[k].copy() for k in archive.files}
        verified_contexts += 1
        n_pcm = np.asarray(context["natural_pcm16"])
        s_pcm = np.asarray(context["sham_pcm16"])
        if n_pcm.dtype != np.int16 or s_pcm.dtype != np.int16 or n_pcm.ndim != 1 or s_pcm.ndim != 1 or len(n_pcm) != len(s_pcm):
            raise AuditError(f"natural/sham PCM must be mono PCM16 with the same full sample length: {sample_id}")
        n_frames = int(len(n_pcm))
        if n_frames != int(sham_source.get("reference_natural_frames", -1)) or n_frames != int(preserved.get("decoded_frames", -2)):
            raise AuditError(f"natural/sham PCM length differs from frozen resampling manifest: {sample_id}")
        pcm_frame_counts.append(n_frames)
        utterance = sample_id.split("_", 1)[1]
        input_base = run_root.parent.parent
        natural_wav = input_base / "inputs" / "hf_selected" / "audio" / str(source["speaker"]) / f"{utterance}.wav"
        sham_wav = Path(str(protocol["resampling_audio_root"])) / "audio" / f"{sample_id}_length_preserved.wav"
        check_bytes(natural_wav, expected_natural_sha, f"{sample_id} raw natural source WAV")
        check_bytes(sham_wav, str(preserved["file_sha256"]), f"{sample_id} length-preserved resampling WAV")
        n_rate, n_pcm_from_wav = read_mono_pcm16(natural_wav)
        s_rate, s_pcm_from_wav = read_mono_pcm16(sham_wav)
        if n_rate != 16000 or s_rate != 16000 or not np.array_equal(n_pcm_from_wav, n_pcm) or not np.array_equal(s_pcm_from_wav, s_pcm):
            raise AuditError(f"context PCM differs from frozen natural/sham WAV payload: {sample_id}")
        expected_raw_n = (n_pcm.astype(np.float64) / np.float64(32768.0)).astype(np.float32)
        expected_raw_s = (s_pcm.astype(np.float64) / np.float64(32768.0)).astype(np.float32)
        if not np.array_equal(context["natural_raw_float32"], expected_raw_n) or not np.array_equal(context["sham_raw_float32"], expected_raw_s):
            raise AuditError(f"context float waveform is not the exact conversion from saved PCM16: {sample_id}")
        if not all(np.isfinite(context[k]).all() for k in ("natural_audio_embeddings_raw_float32", "sham_audio_embeddings_raw_float32", "natural_lip_embeddings_raw_float32", "sham_lip_embeddings_raw_float32")):
            raise AuditError(f"nonfinite audio/video embeddings in context: {sample_id}")
        v_n = np.asarray(context["natural_lip_embeddings_raw_float32"], dtype=np.float32)
        v_s = np.asarray(context["sham_lip_embeddings_raw_float32"], dtype=np.float32)
        a_n = np.asarray(context["natural_audio_embeddings_raw_float32"], dtype=np.float32)
        a_s = np.asarray(context["sham_audio_embeddings_raw_float32"], dtype=np.float32)
        n_windows = len(v_n)
        if any(x.shape != (n_windows, 1024) for x in (v_n, v_s, a_n, a_s)) or n_windows != int(canonical_result["common_window_count"]):
            raise AuditError(f"cached N/S embeddings have inconsistent windows/feature dimensions: {sample_id}")
        supports = expected_supports(n_windows)
        for key, name in required_support_arrays.items():
            saved_support = np.asarray(context[name])
            if saved_support.dtype != np.int32 or not np.array_equal(saved_support, supports[key]):
                raise AuditError(f"cached common {key} support differs from the frozen guard formula: {sample_id}")

        for result in sample_results:
            row = row_by_id[str(result["row_id"])]
            if any(result.get(k) != row.get(k) for k in ("ordered_eval_index", "sample_id", "speaker_id", "split", "family", "condition", "face_id", "visual_id", "audio_id")):
                raise AuditError(f"result metadata differs from the row plan: {result['row_id']}")
            if int(result.get("fixed_anchor_k", -1)) != 3 or int(result.get("fixed_anchor_column", -1)) != 18 or int(result.get("fixed_anchor_offset_frames", 999)) != -3:
                raise AuditError(f"result anchor differs from frozen +3/-3 convention: {result['row_id']}")
            if result.get("flags_at_forward") != required_flags:
                raise AuditError(f"row forward flags differ from frozen deterministic values: {result['row_id']}")
            visual = v_n if row["visual_id"] == "V_RAW_N" else v_s
            audio = a_n if row["audio_id"] == "A_RAW_N" else a_s
            cross_path = safe_path(run_root, str(result["score_npz"]))
            score_rel = cross_path.relative_to(run_root).as_posix()
            visual_path = safe_path(run_root, str(result["visual_npz"]))
            visual_rel = visual_path.relative_to(run_root).as_posix()
            check_bytes(cross_path, str(result["score_sha256"]), f"{result['row_id']} score artifact")
            check_bytes(visual_path, str(result["visual_sha256"]), f"{result['row_id']} visual artifact")
            if all_hashes.get(score_rel) != result["score_sha256"] or all_hashes.get(visual_rel) != result["visual_sha256"]:
                raise AuditError(f"all-files seal differs for score/visual artifact: {result['row_id']}")
            verified_scores += 1
            with np.load(visual_path, allow_pickle=False) as visual_archive:
                visual_keys = set(visual_archive.files)
                if "lip_embeddings_float32" in visual_keys:
                    cached_visual = visual_archive["lip_embeddings_float32"]
                    if cached_visual.shape != visual.shape or not np.array_equal(cached_visual, visual):
                        raise AuditError(f"visual cache raw embeddings differ from sample context: {result['row_id']}")
                else:
                    raise AuditError(f"visual artifact lacks lip embeddings: {result['row_id']}")
                expected_z = context["natural_z_raw_float32"] if row["visual_id"] == "V_RAW_N" else context["sham_z_raw_float32"]
                if "z_condition_float32" not in visual_keys or not np.array_equal(visual_archive["z_condition_float32"], expected_z):
                    raise AuditError(f"visual cache latent differs from sample context: {result['row_id']}")
            verified_visuals += 1
            with np.load(cross_path, allow_pickle=False) as score_archive:
                score = {k: score_archive[k].copy() for k in score_archive.files}
            for support_name, support_key in (("full", "common_support_full_int32"), ("guard20", "common_support_guard20_int32"), ("guard15", "common_support_guard15_int32")):
                if support_key not in score or score[support_key].dtype != np.int32 or not np.array_equal(score[support_key], supports[support_name]):
                    raise AuditError(f"saved score {support_name} support differs from frozen common support: {result['row_id']}")
            if not np.array_equal(score["visual_embeddings_raw_float32"], visual) or not np.array_equal(score["audio_embeddings_raw_float32"], audio):
                raise AuditError(f"saved score embeddings do not match row's context V/A cell: {result['row_id']}")
            visual_unit = unit_rows(visual)
            audio_unit = unit_rows(audio)
            if not np.array_equal(score["visual_embeddings_unit_float32"], visual_unit) or not np.array_equal(score["audio_embeddings_unit_float32"], audio_unit):
                raise AuditError(f"saved row-unit embeddings differ from independent normalization: {result['row_id']}")
            raw_cross, raw_zero = cross_and_zero(visual, audio)
            unit_cross, unit_zero = cross_and_zero(visual_unit, audio_unit)
            raw_lags = lag_matrix(raw_cross, raw_zero, VSHIFT)
            unit_lags = lag_matrix(unit_cross, unit_zero, VSHIFT)
            diffs = {
                "raw_cross": compare_matrix(score["cross_distance_raw_float32"], raw_cross, f"{result['row_id']} raw cross"),
                "raw_zero": compare_matrix(score["cross_distance_to_zero_raw_float32"], raw_zero, f"{result['row_id']} raw zero"),
                "raw_lag": compare_matrix(score["lag_distance_raw_float32"], raw_lags, f"{result['row_id']} raw lag"),
                "unit_cross": compare_matrix(score["cross_distance_unit_float32"], unit_cross, f"{result['row_id']} unit cross"),
                "unit_zero": compare_matrix(score["cross_distance_to_zero_unit_float32"], unit_zero, f"{result['row_id']} unit zero"),
                "unit_lag": compare_matrix(score["lag_distance_unit_float32"], unit_lags, f"{result['row_id']} unit lag"),
            }
            rebuilt = {
                "raw": metric_payload(raw_lags, supports, 18, VSHIFT),
                "unit": metric_payload(unit_lags, supports, 18, VSHIFT),
            }
            metric_diff_raw = validate_metric_match(result.get("raw_support_metrics", {}), rebuilt["raw"], f"{result['row_id']} raw")
            metric_diff_unit = validate_metric_match(result.get("unit_support_metrics", {}), rebuilt["unit"], f"{result['row_id']} unit")
            maximum_differences["raw_metrics"] = max(maximum_differences["raw_metrics"], metric_diff_raw)
            maximum_differences["unit_metrics"] = max(maximum_differences["unit_metrics"], metric_diff_unit)
            for name, value in diffs.items():
                maximum_differences[name] = max(maximum_differences[name], value)
            cond = str(result["condition"])
            for geom in GEOMETRIES:
                metrics_by_sample[sample_id][geom][cond] = rebuilt[geom]
            independent_npz = recompute_dir / f"{sample_id}__{cond}.npz"
            if independent_npz.exists():
                raise FileExistsError(independent_npz)
            np.savez_compressed(
                independent_npz,
                cross_distance_raw_float32=raw_cross,
                cross_distance_to_zero_raw_float32=raw_zero,
                lag_distance_raw_float32=raw_lags,
                cross_distance_unit_float32=unit_cross,
                cross_distance_to_zero_unit_float32=unit_zero,
                lag_distance_unit_float32=unit_lags,
                common_support_full_int32=supports["full"],
                common_support_guard20_int32=supports["guard20"],
                common_support_guard15_int32=supports["guard15"],
            )
            detail = {
                "row_id": result["row_id"], "sample_id": sample_id, "speaker_id": sample_speakers[sample_id], "condition": cond,
                "visual_id": row["visual_id"], "audio_id": row["audio_id"], "source_result_sha256": result["score_sha256"],
                "independent_score_path": independent_npz.relative_to(out).as_posix(), "independent_score_sha256": sha256_file(independent_npz),
                "max_abs_saved_matrix_differences": diffs, "max_abs_saved_metric_difference": {"raw": metric_diff_raw, "unit": metric_diff_unit},
                "metrics": rebuilt,
            }
            detailed_rows.append(detail)

    if len(detailed_rows) != 320 or verified_contexts != 80 or verified_scores != 320 or verified_visuals != 320:
        raise AuditError("independent cache/row coverage is incomplete")
    if any(set(metrics_by_sample[s][g]) != set(CONDITIONS) for s in sample_order for g in GEOMETRIES):
        raise AuditError("some sample/geometry is missing one or more cross-cell curves")

    effects: dict[str, Any] = {}
    effect_records: list[dict[str, Any]] = []
    for geom in GEOMETRIES:
        effects[geom] = {}
        for support in SUPPORTS:
            effects[geom][support] = {}
            for metric in METRICS:
                speaker_values = {"q11_minus_q00": defaultdict(list), "G_q10_minus_q00": defaultdict(list), "E_q01_minus_q00": defaultdict(list), "I_interaction": defaultdict(list)}
                clip_values = {k: [] for k in speaker_values}
                for sample_id in sample_order:
                    cell = metrics_by_sample[sample_id][geom]
                    q00 = float(cell["q00"][support][metric])
                    q10 = float(cell["q10"][support][metric])
                    q01 = float(cell["q01"][support][metric])
                    q11 = float(cell["q11"][support][metric])
                    current = {"q11_minus_q00": q11 - q00, "G_q10_minus_q00": q10 - q00, "E_q01_minus_q00": q01 - q00, "I_interaction": q11 - q10 - q01 + q00}
                    speaker = sample_speakers[sample_id]
                    for key, value in current.items():
                        speaker_values[key][speaker].append(value)
                        clip_values[key].append(value)
                    effect_records.append({"sample_id": sample_id, "speaker_id": speaker, "geometry": geom, "support": support, "metric": metric, **current})
                for contrast, grouped in speaker_values.items():
                    effects[geom][support][f"{metric}:{contrast}"] = {
                        "clip_count": len(clip_values[contrast]),
                        "clip_mean_descriptive": float(np.mean(clip_values[contrast], dtype=np.float64)),
                        **speaker_bootstrap({speaker: values for speaker, values in grouped.items()}),
                    }

    with (out / "independent_cell_metrics.jsonl").open("x", encoding="utf-8") as stream:
        for record in detailed_rows:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    with (out / "independent_effect_rows.jsonl").open("x", encoding="utf-8") as stream:
        for record in effect_records:
            stream.write(json.dumps(record, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")

    consumer_path = Path(__file__).resolve()
    report = {
        "schema_version": 1,
        "manifest_type": "independent_resampling_sham_cpu_consumer",
        "status": "COMPLETE",
        "classification": "independent CPU recomputation of the frozen sham; all 80 samples retained",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "consumer_source": {"path": str(consumer_path), "sha256": sha256_file(consumer_path), "numpy_version": np.__version__, "imports_producer_ops_or_scoring": False, "device": "CPU"},
        "input_seals": {
            "producer_run_root": str(run_root), "protocol_raw_sha256": sha256_file(protocol_path), "protocol_fingerprint": protocol["protocol_sha256"],
            "row_manifest_sha256": sha256_file(row_path), "results_jsonl_sha256": sha256_file(result_path), "run_status_sha256": sha256_file(status_path),
            "all_files_manifest_sha256": sha256_file(all_hashes_path), "code_seal_sha256": sha256_file(seal_path),
            "producer_code_sha256": code_hashes, "external_input_manifest_sha256": EXPECTED_INPUT_SHA256,
            "resampling_manifest_sha256": EXPECTED_RESAMPLING_SHA256,
        },
        "verification": {
            "producer_status_complete": True, "planned_samples": 80, "completed_samples": 80, "failed_samples": 0,
            "planned_rows": 320, "completed_rows": 320, "failed_rows": 0,
            "unique_speakers": 40, "clips_per_speaker": {"min": 2, "max": 2},
            "contexts_verified": verified_contexts, "score_artifacts_recomputed": verified_scores,
            "visual_artifacts_verified": verified_visuals, "all_artifact_hashes_checked": True,
            "all_embedding_distances_finite": True, "natural_sham_pcm16_equal_length_all_samples": True,
            "same_N_S_pcm_frames_per_sample": {"min": min(pcm_frame_counts), "max": max(pcm_frame_counts), "unique_lengths": sorted(set(pcm_frame_counts))},
            "deterministic_forward_flags_all_rows_and_events": required_flags,
            "condition_cells_per_sample": {"q00": "V_RAW_N/A_RAW_N", "q10": "V_RAW_S/A_RAW_N", "q01": "V_RAW_N/A_RAW_S", "q11": "V_RAW_S/A_RAW_S"},
            "fixed_anchor": {"k": 3, "column": 18, "official_offset_frames": -3},
            "max_abs_saved_vs_independent": dict(maximum_differences),
            "comparison_tolerances": {"saved_matrices_atol": DISTANCE_ATOL, "saved_metrics_atol": METRIC_ATOL},
        },
        "estimands": {
            "cells": ["q00", "q10", "q01", "q11"], "primary_contrast": "q11-q00",
            "generator_component_G": "q10-q00", "resampling_component_E": "q01-q00",
            "interaction_I": "q11-q10-q01+q00", "identity": "q11-q00 = G+E+I",
            "metrics": {"C": "B-D_min", "C_anchor": "B-D_fixed_anchor", "B": "median lag distance", "D": "minimum lag distance", "search_gain": "D_zero-D_min"},
            "anchor_k": 3, "anchor_column": 18, "lag_domain": "31 columns; k=-15..+15; out-of-audio samples use distance-to-zero",
            "supports": {"full": "all common frames", "valid": "all common frames", "guard20": "indices [20,N-20)", "guard15": "indices [15,N-15)"},
        },
        "bootstrap": {"rng": "numpy.PCG64", "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "ci": "99%, percentile [0.5,99.5]", "unit": "equal speaker; resample 40 speakers, retain each speaker's two clip mean values"},
        "effects": effects,
    }
    report["manifest_sha256"] = fingerprint(report, "manifest_sha256")
    report_path = out / "independent_sham_summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    output_index = {}
    for path in sorted(p for p in out.rglob("*") if p.is_file()):
        if path.name == "independent_output_sha256.json":
            continue
        output_index[path.relative_to(out).as_posix()] = sha256_file(path)
    output_payload = {"schema_version": 1, "manifest_type": "independent_sham_consumer_output_files", "consumer_sha256": sha256_file(consumer_path), "summary_manifest_sha256": report["manifest_sha256"], "files": output_index}
    output_payload["manifest_sha256"] = fingerprint(output_payload, "manifest_sha256")
    (out / "independent_output_sha256.json").write_text(json.dumps(output_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = run(args)
        print(json.dumps({"status": report["status"], "summary": str(args.outdir.resolve() / "independent_sham_summary.json"), "sha256": report["manifest_sha256"], "verification": report["verification"]}, separators=(",", ":"), allow_nan=False))
        return 0
    except Exception as exc:
        args.outdir.mkdir(parents=True, exist_ok=True)
        failure = {"status": "FAILED", "error_type": type(exc).__name__, "error": str(exc), "created_utc": datetime.now(timezone.utc).isoformat(), "consumer_source_sha256": sha256_file(Path(__file__).resolve())}
        (args.outdir / "independent_sham_failure.json").write_text(json.dumps(failure, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure, ensure_ascii=False, separators=(",", ":")), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
