#!/usr/bin/env python3
"""Independent CPU consumer for the frozen mel partition four-cell run.

This module reconstructs all partition identities, distance matrices, fixed
supports, temporal permutation nulls, and equal-speaker intervals directly
from sealed arrays and embeddings. It does not import a producer or model
scoring implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np


SPEC_SHA256 = "abeced0f6f50a94fe49fc085af40358071dacc5091c7c796c26c5edafbe63ee8"
INPUT_IDENTITY_SHA256 = "deb85e134904879d662ffce58990c04992420a25f0796184e907729f99af1f27"
TARGET_RMS = 0.0376838172675746
CONDITIONS = ("q00", "q10", "q01", "q11")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
MEL_RATE_HZ = 80.0
VIDEO_FPS = 25
MEL_STEP_SIZE = 16
MEL_CHUNK_RULE = "for i=0.., start=int(i*80/25), append mel[:,start:start+16], when start+16>mel_frames append mel[:,-16:] and stop"
METRICS = ("C", "C_anchor", "B", "D", "D0", "search_gain")
CONTRASTS = {
    "interior_q10_minus_q00": {"q10": 1.0, "q00": -1.0},
    "boundary_q01_minus_q00": {"q01": 1.0, "q00": -1.0},
    "joint_q11_minus_q00": {"q11": 1.0, "q00": -1.0},
    "interaction_q11_minus_q10_minus_q01_plus_q00": {"q11": 1.0, "q10": -1.0, "q01": -1.0, "q00": 1.0},
    "q10_minus_q01": {"q10": 1.0, "q01": -1.0},
}
EPS = np.float32(1e-6)
VSHIFT = 15
ZERO_COLUMN = 15
ANCHOR_COLUMN = 18
EMBEDDING_DIM = 1024
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20260930
PERMUTATION_COUNT = 100
PERMUTATION_SEED = 20260930
TORCH_FLAGS = {
    "deterministic_algorithms": True,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False,
    "cudnn_allow_tf32": False,
    "matmul_allow_tf32": False,
}


class ConsumerError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def _sha_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _payload_sha(value: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical({key: item for key, item in value.items() if key != "payload_sha256"})).hexdigest()


def _official_mel_chunk_starts(mel_frames: int) -> list[int]:
    """Independently reconstruct Wav2Lip's frame-to-mel chunk loop."""
    if mel_frames < MEL_STEP_SIZE:
        raise ConsumerError(f"mel has fewer than {MEL_STEP_SIZE} frames")
    multiplier = MEL_RATE_HZ / float(VIDEO_FPS)
    starts: list[int] = []
    index = 0
    while True:
        start = int(index * multiplier)
        if start + MEL_STEP_SIZE > mel_frames:
            starts.append(mel_frames - MEL_STEP_SIZE)
            break
        starts.append(start)
        index += 1
    return starts


def _read_json(path: str | Path) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConsumerError(f"cannot read JSON object {path}: {type(exc).__name__}") from exc
    if not isinstance(value, dict):
        raise ConsumerError(f"expected JSON object: {path}")
    return value


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConsumerError(f"malformed JSONL at {path}:{number}") from exc
        if not isinstance(value, dict):
            raise ConsumerError(f"JSONL row is not an object at {path}:{number}")
        result.append(value)
    return result


def _require_sha(path: str | Path, expected: str, label: str) -> Path:
    target = Path(path).resolve(strict=True)
    actual = _sha_file(target)
    if actual != expected:
        raise ConsumerError(f"{label} file SHA mismatch: {target} ({actual} != {expected})")
    return target


def _verify_ledger(root: Path, ledger_path: Path, expected_raw_sha: str | None, label: str) -> dict[str, str]:
    if expected_raw_sha:
        _require_sha(ledger_path, expected_raw_sha, f"{label} ledger")
    ledger = _read_json(ledger_path)
    if not ledger or any(not isinstance(key, str) or not isinstance(value, str) or len(value) != 64 for key, value in ledger.items()):
        raise ConsumerError(f"malformed {label} file ledger: {ledger_path}")
    root = root.resolve(strict=True)
    for relative, digest in ledger.items():
        candidate = (root / relative).resolve()
        if root not in candidate.parents:
            raise ConsumerError(f"{label} ledger path escapes its root: {relative}")
        _require_sha(candidate, digest, f"{label} ledger entry {relative}")
    found = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.resolve() != ledger_path.resolve()
    }
    if found != set(ledger):
        raise ConsumerError(f"{label} ledger membership differs: missing={sorted(set(ledger)-found)[:3]}, extra={sorted(found-set(ledger))[:3]}")
    return {str(key): str(value) for key, value in ledger.items()}


def _deep_match(saved: Any, actual: Any, label: str, *, float_atol: float = 1e-11) -> None:
    if isinstance(saved, Mapping) and isinstance(actual, Mapping):
        if set(saved) != set(actual):
            raise ConsumerError(f"{label} field inventory differs")
        for key in saved:
            _deep_match(saved[key], actual[key], f"{label}/{key}", float_atol=float_atol)
        return
    if isinstance(saved, list) and isinstance(actual, list):
        if len(saved) != len(actual):
            raise ConsumerError(f"{label} list length differs")
        for index, (left, right) in enumerate(zip(saved, actual)):
            _deep_match(left, right, f"{label}/{index}", float_atol=float_atol)
        return
    if isinstance(saved, bool) or isinstance(actual, bool):
        if saved is not actual:
            raise ConsumerError(f"{label} boolean differs: {saved!r} != {actual!r}")
        return
    if isinstance(saved, (int, float)) and isinstance(actual, (int, float)):
        if not math.isclose(float(saved), float(actual), rel_tol=1e-12, abs_tol=float_atol):
            raise ConsumerError(f"{label} numeric value differs: {saved!r} != {actual!r}")
        return
    if saved != actual:
        raise ConsumerError(f"{label} differs: {saved!r} != {actual!r}")


def _cpu_run(root: Path, split: str, *, verify_ledger: bool) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    root = root.resolve(strict=True)
    plan_path, manifest_path = root / "plan.json", root / "run_manifest.json"
    rows_path, ledger_path = root / "rows.jsonl", root / "all_files_sha256.json"
    plan, manifest = _read_json(plan_path), _read_json(manifest_path)
    if plan.get("payload_sha256") != _payload_sha(plan):
        raise ConsumerError(f"{split} CPU plan payload fingerprint mismatch")
    if manifest.get("payload_sha256") != _payload_sha(manifest):
        raise ConsumerError(f"{split} CPU manifest payload fingerprint mismatch")
    if plan.get("spec", {}).get("sha256") != SPEC_SHA256 or manifest.get("input_identity_sha256") != INPUT_IDENTITY_SHA256:
        raise ConsumerError(f"{split} CPU source is not bound to the frozen mel spec and input identity")
    if plan.get("input_identity_sha256") != INPUT_IDENTITY_SHA256 or plan.get("split") != split or manifest.get("split") != split:
        raise ConsumerError(f"{split} CPU split/identity differs from the frozen protocol")
    if plan.get("model_forward_performed") is not False or plan.get("effects_read") is not False:
        raise ConsumerError(f"{split} CPU plan is not a no-forward/no-effects source")
    if plan.get("status") != "frozen_pre_effects" or float(plan.get("target_rms", -1)) != TARGET_RMS:
        raise ConsumerError(f"{split} CPU plan status or fixed target RMS differs")
    frontend = plan.get("frontend")
    if not isinstance(frontend, Mapping) or frontend.get("chunk_rule") != MEL_CHUNK_RULE:
        raise ConsumerError(f"{split} CPU plan does not bind the frozen official mel chunk rule")
    source_manifest = plan.get("input_manifest")
    pair_manifest = plan.get("tts_pair_manifest")
    for name, source in (("input", source_manifest), ("TTS pair", pair_manifest)):
        if not isinstance(source, Mapping) or not _is_sha256(source.get("raw_sha256")):
            raise ConsumerError(f"{split} CPU plan lacks the raw SHA for its {name} manifest")
        if source.get("canonical_payload_matches_embedded_fingerprint") is not True:
            raise ConsumerError(f"{split} CPU {name} manifest fingerprint was not verified")
    if manifest.get("status") != "complete_cpu_preflight" or manifest.get("model_forward_performed") is not False or manifest.get("effects_read") is not False:
        raise ConsumerError(f"{split} CPU run is incomplete or not effects-free")
    expected_audio_rows = 40 if split == "calibration" else 80
    if manifest.get("audio_rows") != expected_audio_rows or manifest.get("generation_rows") != expected_audio_rows * 4:
        raise ConsumerError(f"{split} CPU source counts differ from the frozen design")
    if manifest.get("plan_raw_sha256") != _sha_file(plan_path) or manifest.get("rows_raw_sha256") != _sha_file(rows_path):
        raise ConsumerError(f"{split} CPU run does not bind its plan/rows bytes")
    if verify_ledger:
        ledger = _verify_ledger(root, ledger_path, None, f"{split} CPU")
    else:
        ledger = _read_json(ledger_path)
        if not isinstance(ledger, dict):
            raise ConsumerError(f"{split} CPU file ledger is malformed")
    if set(ledger) != {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file() and path.resolve() != ledger_path.resolve()
    }:
        raise ConsumerError(f"{split} CPU file ledger membership differs")
    rows = _read_jsonl(rows_path)
    plan_rows = plan.get("rows")
    artifacts = manifest.get("artifacts")
    if not isinstance(plan_rows, list) or len(plan_rows) != expected_audio_rows or len(rows) != expected_audio_rows:
        raise ConsumerError(f"{split} CPU planned row count differs")
    if [(str(a.get("id")), str(a.get("speaker")), str(a.get("split")), str(a.get("source_condition")), str(a.get("expected_wav_sha256")), a.get("ordered_index")) for a in plan_rows] != [
        (str(a.get("id")), str(a.get("speaker")), str(a.get("split")), str(a.get("source_condition")), str(a.get("expected_wav_sha256")), a.get("ordered_index")) for a in rows
    ]:
        raise ConsumerError(f"{split} CPU JSONL rows differ from the embedded plan rows")
    if not isinstance(artifacts, list) or len(artifacts) != expected_audio_rows:
        raise ConsumerError(f"{split} CPU artifact count differs")
    if split == "calibration":
        if plan.get("scope") != "calibration20_both_sources" or manifest.get("audio_rows") != 40:
            raise ConsumerError("CAL CPU run is not the frozen 20-speaker, both-source cohort")
    else:
        if plan.get("scope") != "evaluation80_natural_only":
            raise ConsumerError("EVAL CPU run is not the frozen natural-only cohort")
    row_ids: dict[str, tuple[str, set[str]]] = {}
    for row in plan_rows:
        sample_id, speaker, condition = str(row.get("id")), str(row.get("speaker")), str(row.get("source_condition"))
        if sample_id in row_ids:
            previous_speaker, conditions = row_ids[sample_id]
            if previous_speaker != speaker or condition in conditions:
                raise ConsumerError(f"{split} CPU plan has a duplicate or inconsistent source identity: {sample_id}")
            conditions.add(condition)
        else:
            row_ids[sample_id] = (speaker, {condition})
    by_speaker: dict[str, set[str]] = {}
    for sample_id, (speaker, conditions) in row_ids.items():
        if split == "calibration" and conditions != {"natural", "tts"}:
            raise ConsumerError(f"CAL CPU pair lacks exactly natural/TTS rows: {sample_id}")
        if split == "evaluation" and conditions != {"natural"}:
            raise ConsumerError(f"EVAL CPU row is not natural-only: {sample_id}")
        by_speaker.setdefault(speaker, set()).add(sample_id)
    expected_speakers, clips_per_speaker = (10, 2) if split == "calibration" else (40, 2)
    if len(row_ids) != (20 if split == "calibration" else 80) or len(by_speaker) != expected_speakers or any(len(ids) != clips_per_speaker for ids in by_speaker.values()):
        raise ConsumerError(f"{split} CPU cohort does not meet the frozen speaker/clip counts")
    artifact_map = {(str(row.get("id")), str(row.get("source_condition"))): row for row in artifacts}
    if len(artifact_map) != expected_audio_rows:
        raise ConsumerError(f"{split} CPU artifacts contain duplicate identities")
    for index, row in enumerate(rows):
        # CAL rows are paired natural/TTS sources: both rows share the clip-pair
        # index. EVAL has one natural row per clip and therefore uses row index.
        expected_ordered_index = index if split == "evaluation" else index // 2
        if row.get("ordered_index") != expected_ordered_index or row.get("split") != split or row.get("status") != "preflight_pass":
            raise ConsumerError(f"{split} CPU row order/status differs at {index}")
        if split == "calibration":
            expected_condition = "natural" if index % 2 == 0 else "tts"
            if row.get("source_condition") != expected_condition:
                raise ConsumerError(f"CAL CPU source-condition pair order differs at row {index}")
            if index % 2 == 1:
                prior = rows[index - 1]
                if (row.get("id"), row.get("speaker")) != (prior.get("id"), prior.get("speaker")):
                    raise ConsumerError(f"CAL CPU natural/TTS rows do not share clip identity at pair {index // 2}")
        identity = (str(row.get("id")), str(row.get("source_condition")))
        artifact = artifact_map.get(identity)
        if artifact is None or artifact.get("status") != "prepared" or artifact.get("speaker") != row.get("speaker"):
            raise ConsumerError(f"{split} CPU row/artifact identity differs for {identity}")
        if artifact.get("source_wav_sha256") != row.get("expected_wav_sha256"):
            raise ConsumerError(f"{split} CPU source WAV identity differs for {identity}")
    binding = {
        "root": str(root),
        "plan_sha256": _sha_file(plan_path),
        "run_manifest_sha256": _sha_file(manifest_path),
        "rows_sha256": _sha_file(rows_path),
        "all_files_ledger_sha256": _sha_file(ledger_path),
        "ledger_entries_verified": len(ledger),
        "input_identity_sha256": manifest.get("input_identity_sha256"),
        "input_manifest_raw_sha256": source_manifest["raw_sha256"],
        "tts_pair_manifest_raw_sha256": pair_manifest["raw_sha256"],
        "producer_code_sha256": plan.get("producer_code", {}).get("sha256"),
        "frontend": plan.get("frontend"),
        "runtime": plan.get("runtime"),
    }
    if split == "calibration":
        binding["calibration_qa_sha256"] = _sha_file(root / "calibration_qa.json")
    return plan, manifest, rows, binding


def _array_record(values: Mapping[str, np.ndarray], metadata: Mapping[str, Any], label: str) -> None:
    if set(values) != set(metadata):
        raise ConsumerError(f"{label} array inventory differs")
    for name, value in values.items():
        item = metadata[name]
        if not isinstance(item, Mapping) or list(value.shape) != item.get("shape") or str(value.dtype) != item.get("dtype"):
            raise ConsumerError(f"{label}/{name} array shape/dtype differs")
        if _sha_array(value) != item.get("array_sha256"):
            raise ConsumerError(f"{label}/{name} array SHA differs")


def _read_partition_arrays(artifact: Mapping[str, Any], label: str) -> tuple[dict[str, np.ndarray], Path]:
    path = _require_sha(artifact.get("mel_path", ""), str(artifact.get("mel_file_sha256", "")), f"{label} mel NPZ")
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except Exception as exc:
        raise ConsumerError(f"cannot decode {label} mel NPZ: {type(exc).__name__}") from exc
    metadata = artifact.get("arrays")
    if not isinstance(metadata, Mapping):
        raise ConsumerError(f"{label} has no per-array metadata")
    _array_record(arrays, metadata, label)
    required = {"M0", "M1", "U", "S", "mel_starts", *CONDITIONS, *(f"chunks_{condition}" for condition in CONDITIONS)}
    if set(arrays) != required:
        raise ConsumerError(f"{label} NPZ array key set differs: {sorted(set(arrays) ^ required)}")
    return arrays, path


def _partition_summary(m0: np.ndarray, m1: np.ndarray, u: np.ndarray, cells: Mapping[str, np.ndarray], gain: float) -> dict[str, Any]:
    lo, hi = np.float32(-4.0), np.float32(4.0)
    inward_lo = np.nextafter(lo, np.float32(0.0), dtype=np.float32)
    inward_hi = np.nextafter(hi, np.float32(0.0), dtype=np.float32)
    delta = m1.astype(np.float64) - m0.astype(np.float64)
    q10_delta = cells["q10"].astype(np.float64) - m0.astype(np.float64)
    q01_delta = cells["q01"].astype(np.float64) - m0.astype(np.float64)
    closure = delta - (q10_delta + q01_delta)
    prediction = float(1.6 * np.log10(np.float64(gain)))
    interior_delta = delta[u]
    prediction_error = interior_delta - prediction

    def count(mask: np.ndarray) -> int:
        return int(np.count_nonzero(mask))

    return {
        "shape": list(m0.shape),
        "dtype": "float32",
        "total_bins": int(m0.size),
        "U_interior_bins": int(np.count_nonzero(u)),
        "S_complement_bins": int(np.count_nonzero(~u)),
        "raw_lower_lock": count((m0 == lo) & (m1 == lo)),
        "raw_lower_unlock": count((m0 == lo) & (m1 > lo)),
        "fixed_lower_entry": count((m0 > lo) & (m1 == lo)),
        "raw_upper_lock": count((m0 == hi) & (m1 == hi)),
        "raw_upper_unlock": count((m0 == hi) & (m1 < hi)),
        "fixed_upper_entry": count((m0 < hi) & (m1 == hi)),
        "upper_cross_raw_interior_to_fixed_upper": count((m0 < hi) & (m1 == hi)),
        "upper_cross_raw_upper_to_fixed_interior": count((m0 == hi) & (m1 < hi)),
        "inward_one_ulp": {
            "raw_lower": count(m0 == inward_lo), "raw_upper": count(m0 == inward_hi),
            "fixed_lower": count(m1 == inward_lo), "fixed_upper": count(m1 == inward_hi),
        },
        "delta": {
            "l1_sum": float(np.abs(delta).sum(dtype=np.float64)),
            "l2": float(np.linalg.norm(delta.ravel())),
            "linf": float(np.max(np.abs(delta), initial=0.0)),
            "interior_l2": float(np.linalg.norm(interior_delta)),
            "boundary_l2": float(np.linalg.norm(delta[~u])),
        },
        "interior_waveform_gain_prediction": {
            "gain": float(gain), "predicted_mel_offset": prediction, "count": int(interior_delta.size),
            "mean_error": float(prediction_error.mean()) if prediction_error.size else None,
            "rmse_error": float(np.sqrt(np.mean(np.square(prediction_error)))) if prediction_error.size else None,
            "max_abs_error": float(np.max(np.abs(prediction_error), initial=0.0)),
        },
        "four_cell": {
            "q00_equals_M0": bool(np.array_equal(cells["q00"], m0)),
            "q11_equals_M1": bool(np.array_equal(cells["q11"], m1)),
            "q10_equals_where_U_M1_else_M0": bool(np.array_equal(cells["q10"], np.where(u, m1, m0))),
            "q01_equals_where_U_M0_else_M1": bool(np.array_equal(cells["q01"], np.where(u, m0, m1))),
            "float64_closure_max_abs": float(np.max(np.abs(closure), initial=0.0)),
            "float64_closure_exact": bool(np.array_equal(closure, np.zeros_like(closure))),
        },
    }


def _verify_partition(
    artifact: Mapping[str, Any],
    *,
    split: str,
    saved_cal_summary: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = f"{split}/{artifact.get('id')}/{artifact.get('source_condition')}"
    arrays, npz_path = _read_partition_arrays(artifact, label)
    m0, m1 = arrays["M0"], arrays["M1"]
    u, complement = arrays["U"], arrays["S"]
    if m0.dtype != np.float32 or m1.dtype != np.float32 or m0.ndim != 2 or m0.shape != m1.shape or m0.shape[0] != 80:
        raise ConsumerError(f"{label} endpoints violate float32 [80,frames] contract")
    if not np.isfinite(m0).all() or not np.isfinite(m1).all() or np.any(m0 < -4) or np.any(m0 > 4) or np.any(m1 < -4) or np.any(m1 > 4):
        raise ConsumerError(f"{label} mel endpoint has nonfinite/out-of-range bins")
    expected_u = (m0 > np.float32(-4)) & (m0 < np.float32(4)) & (m1 > np.float32(-4)) & (m1 < np.float32(4))
    if u.dtype != np.bool_ or complement.dtype != np.bool_ or not np.array_equal(u, expected_u) or not np.array_equal(complement, ~expected_u):
        raise ConsumerError(f"{label} U/S masks do not follow the exact strict-interior rule")
    cells = {condition: arrays[condition] for condition in CONDITIONS}
    expected_cells = {
        "q00": m0,
        "q10": np.where(u, m1, m0).astype(np.float32, copy=False),
        "q01": np.where(u, m0, m1).astype(np.float32, copy=False),
        "q11": m1,
    }
    for condition in CONDITIONS:
        if cells[condition].dtype != np.float32 or not np.array_equal(cells[condition], expected_cells[condition]):
            raise ConsumerError(f"{label} four-cell copy identity failed: {condition}")
    starts = arrays["mel_starts"]
    if starts.dtype != np.int32 or starts.ndim != 1 or not starts.size:
        raise ConsumerError(f"{label} chunk-start array is malformed")
    expected_starts = np.asarray(_official_mel_chunk_starts(m0.shape[1]), dtype=np.int32)
    if not np.array_equal(starts, expected_starts):
        raise ConsumerError(f"{label} chunk starts do not match the independently reconstructed Wav2Lip sequence")
    chunks_hashes = {}
    for condition in CONDITIONS:
        expected_chunks = np.stack([cells[condition][:, int(start):int(start)+MEL_STEP_SIZE] for start in starts], axis=0).astype(np.float32, copy=False)
        chunks = arrays[f"chunks_{condition}"]
        if chunks.dtype != np.float32 or not np.array_equal(chunks, expected_chunks):
            raise ConsumerError(f"{label} chunk array is not an exact slice of {condition}")
        chunks_hashes[condition] = _sha_array(chunks)
    if int(artifact.get("chunk_count", -1)) != len(starts):
        raise ConsumerError(f"{label} recorded chunk count differs")
    gain_meta = artifact.get("gain")
    if not isinstance(gain_meta, Mapping) or float(gain_meta.get("target_rms", -1)) != TARGET_RMS:
        raise ConsumerError(f"{label} source RMS target differs")
    gain = float(gain_meta.get("gain", 0.0))
    if not math.isfinite(gain) or gain <= 0:
        raise ConsumerError(f"{label} has invalid waveform gain")
    summary = _partition_summary(m0, m1, u, cells, gain)
    if not summary["four_cell"]["float64_closure_exact"]:
        raise ConsumerError(f"{label} float64 partition closure is not exact")
    if saved_cal_summary is not None:
        _deep_match(saved_cal_summary, summary, f"{label} saved CAL QA")
    record = {
        "id": str(artifact["id"]), "speaker": str(artifact["speaker"]), "source_condition": str(artifact["source_condition"]),
        "mel_npz_path": str(npz_path), "mel_npz_sha256": _sha_file(npz_path),
        "M0_array_sha256": _sha_array(m0), "M1_array_sha256": _sha_array(m1),
        "U_array_sha256": _sha_array(u), "S_array_sha256": _sha_array(complement),
        "U_bins": int(np.count_nonzero(u)), "S_bins": int(np.count_nonzero(complement)),
        "mel_shape": list(m0.shape), "chunk_count": int(len(starts)), "chunk_array_sha256": chunks_hashes,
        "chunk_start_verification": {
            "status": "PASS", "fps": VIDEO_FPS, "mel_rate_hz": MEL_RATE_HZ,
            "step_size": MEL_STEP_SIZE, "algorithm": MEL_CHUNK_RULE,
            "starts_array_sha256": _sha_array(starts), "reconstructed_starts_array_sha256": _sha_array(expected_starts),
            "exact_sequence_match": True,
        },
        "fixed_audio_gain": dict(gain_meta), "partition_dose": summary,
    }
    return record, arrays


def _verify_pcm_gain(artifact: Mapping[str, Any], label: str) -> dict[str, Any]:
    raw_path = _require_sha(artifact.get("source_wav_path", ""), str(artifact.get("source_wav_sha256", "")), f"{label} raw WAV")
    fixed_path = _require_sha(artifact.get("fixed_pcm_path", ""), str(artifact.get("fixed_pcm_file_sha256", "")), f"{label} fixed waveform")
    try:
        from scipy.io import wavfile

        rate, pcm = wavfile.read(str(raw_path))
    except Exception as exc:
        raise ConsumerError(f"cannot decode {label} source WAV: {type(exc).__name__}") from exc
    if rate != 16_000 or pcm.ndim != 1 or pcm.dtype != np.int16 or len(pcm) != int(artifact.get("sample_count", -1)):
        raise ConsumerError(f"{label} source PCM clock/format differs")
    waveform = (pcm.astype(np.float64) / np.float64(32768.0)).astype(np.float32)
    waveform_sha = _sha_array(waveform)
    pcm_sha = _sha_array(np.asarray(pcm, dtype="<i2"))
    if waveform_sha != artifact.get("source_waveform_array_sha256"):
        raise ConsumerError(f"{label} decoded float32 waveform SHA differs from the sealed artifact")
    # The frozen producer artifact and its plan seal the WAV bytes and the
    # canonical float32 waveform, but do not include a separate int16-array
    # digest. Keep checking that digest if a producer explicitly supplies it;
    # otherwise report our recomputation as an unbound diagnostic below.
    sealed_pcm_sha = artifact.get("source_pcm_array_sha256")
    if sealed_pcm_sha is not None and pcm_sha != sealed_pcm_sha:
        raise ConsumerError(f"{label} decoded int16 PCM SHA differs from the sealed artifact")
    try:
        fixed = np.load(fixed_path, allow_pickle=False)
    except Exception as exc:
        raise ConsumerError(f"cannot read {label} fixed waveform: {type(exc).__name__}") from exc
    if fixed.dtype != np.float32 or fixed.ndim != 1 or fixed.shape != waveform.shape or not np.isfinite(fixed).all():
        raise ConsumerError(f"{label} fixed waveform type/clock is invalid")
    if _sha_array(fixed) != artifact.get("fixed_pcm_array_sha256"):
        raise ConsumerError(f"{label} fixed waveform array SHA differs")
    gain_meta = artifact.get("gain")
    if not isinstance(gain_meta, Mapping):
        raise ConsumerError(f"{label} fixed waveform gain record is missing")
    raw_rms = float(np.sqrt(np.mean(np.square(waveform.astype(np.float64)), dtype=np.float64)))
    gain = float(TARGET_RMS / np.float64(raw_rms))
    expected_fixed = np.multiply(waveform.astype(np.float64), np.float64(gain), dtype=np.float64).astype(np.float32)
    fixed_rms = float(np.sqrt(np.mean(np.square(fixed.astype(np.float64)), dtype=np.float64)))
    if not np.array_equal(fixed, expected_fixed) or abs(fixed_rms - TARGET_RMS) > 2e-7:
        raise ConsumerError(f"{label} fixed waveform is not the exact no-clipping RMS transform")
    for key, value in (("input_rms", raw_rms), ("gain", gain), ("realized_rms_float32", fixed_rms)):
        if not math.isclose(float(gain_meta.get(key, float("nan"))), value, rel_tol=1e-12, abs_tol=1e-12):
            raise ConsumerError(f"{label} fixed gain metadata differs at {key}")
    return {
        "raw_wav_path": str(raw_path), "raw_wav_sha256": _sha_file(raw_path), "pcm16_array_sha256": pcm_sha,
        "raw_float32_array_sha256": waveform_sha,
        "int16_pcm_digest_binding": "verified_against_artifact" if sealed_pcm_sha is not None else "recomputed_unbound_not_present_in_producer_schema",
        "fixed_waveform_path": str(fixed_path),
        "fixed_waveform_file_sha256": _sha_file(fixed_path), "fixed_waveform_array_sha256": _sha_array(fixed),
        "raw_rms": raw_rms, "gain": gain, "gain_db": float(20.0 * np.log10(gain)),
        "fixed_rms": fixed_rms, "fixed_target_rms": TARGET_RMS, "exact_float32_transform": True,
    }


def _verify_cpu_calibration_qa(root: Path, manifest: Mapping[str, Any]) -> tuple[dict[str, Any], dict[tuple[str, str], Mapping[str, Any]]]:
    path = root / "calibration_qa.json"
    ledger = _read_json(root / "all_files_sha256.json")
    if not isinstance(ledger, Mapping) or "calibration_qa.json" not in ledger:
        raise ConsumerError("CAL all-files ledger does not include calibration_qa.json")
    _require_sha(path, str(ledger["calibration_qa.json"]), "CAL QA")
    qa = _read_json(path)
    manifest_path = root / "run_manifest.json"
    if qa.get("status") != "PASS" or qa.get("split") != "calibration" or qa.get("effects_read") is not False:
        raise ConsumerError("CAL QA is not a frozen PASS no-effect report")
    if qa.get("rule") != "CAL20 CPU frontend/partition QA only; no model forward or effect inspection":
        raise ConsumerError("CAL QA rule differs from the frozen CPU-only rule")
    if qa.get("input_rows") != 40 or qa.get("prepared_input_rows") != 40 or qa.get("source_rows") != 20:
        raise ConsumerError("CAL QA does not contain all 20 natural/TTS pairs")
    if qa.get("required_generation_rows") != 160 or qa.get("planned_generation_rows") != 160 or qa.get("prepared_generation_rows") != 160:
        raise ConsumerError("CAL QA four-cell generation counts differ")
    if qa.get("all_finite_in_range") is not True or qa.get("exact_endpoint_and_partition_copies") is not True or qa.get("float64_closure_exact") is not True:
        raise ConsumerError("CAL QA finite/endpoint/closure gates are incomplete")
    if qa.get("run_manifest_sha256") != _sha_file(manifest_path) or qa.get("spec_sha256") != SPEC_SHA256:
        raise ConsumerError("CAL QA does not bind the sealed CPU manifest/spec")
    if not qa.get("run_manifest_sha256") or qa.get("run_manifest_sha256") != _sha_file(root / "run_manifest.json"):
        raise ConsumerError("CAL QA does not bind the exact run manifest bytes")
    rows = qa.get("rows")
    if not isinstance(rows, list) or len(rows) != 40:
        raise ConsumerError("CAL QA row manifest is incomplete")
    mapping: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("id")), str(row.get("source_condition")))
        if key in mapping or row.get("status") != "prepared":
            raise ConsumerError(f"duplicate or failed CAL QA row: {key}")
        mapping[key] = row
    return {"path": str(path.resolve()), "raw_sha256": _sha_file(path), "status": qa["status"], "rows": len(rows)}, mapping


def _sha_and_root(record: Mapping[str, Any], root: Path, label: str) -> Path:
    value = Path(str(record.get("path", "")))
    if not value.is_absolute():
        value = root / value
    return _require_sha(value, str(record.get("file_sha256", "")), label)


def _load_array_record(record: Mapping[str, Any], root: Path, label: str, *, expected_dim: int | None = None) -> tuple[np.ndarray, Path]:
    path = _sha_and_root(record, root, label)
    try:
        value = np.load(path, allow_pickle=False)
    except Exception as exc:
        raise ConsumerError(f"cannot decode {label}: {type(exc).__name__}") from exc
    if not isinstance(value, np.ndarray) or value.dtype != np.float32 or value.ndim != 2 or (expected_dim is not None and value.shape[1] != expected_dim):
        raise ConsumerError(f"{label} has invalid dtype/shape: {getattr(value, 'shape', None)}/{getattr(value, 'dtype', None)}")
    if not np.isfinite(value).all():
        raise ConsumerError(f"{label} contains nonfinite values")
    if list(value.shape) != record.get("shape") or str(value.dtype) != record.get("dtype") or _sha_array(value) != record.get("array_sha256"):
        raise ConsumerError(f"{label} array metadata/hash mismatch")
    return value, path


def _load_distance_artifact(record: Mapping[str, Any], root: Path, label: str) -> tuple[dict[str, np.ndarray], Path]:
    path = _sha_and_root(record, root, label)
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {key: np.asarray(archive[key]) for key in archive.files}
    except Exception as exc:
        raise ConsumerError(f"cannot read {label} distance archive: {type(exc).__name__}") from exc
    metadata = record.get("arrays")
    if not isinstance(metadata, Mapping):
        raise ConsumerError(f"{label} distance archive lacks array metadata")
    _array_record(arrays, metadata, label)
    required = {"cross_v_to_a_float32", "zero_boundary_float32", "lag_distance_float32", "lag_columns_int32", "lag_k_int32", *(f"support_{name}_int32" for name in SUPPORTS)}
    if set(arrays) != required:
        raise ConsumerError(f"{label} distance array inventory differs")
    return arrays, path


def _row_unit(value: np.ndarray) -> np.ndarray:
    x = np.asarray(value, dtype=np.float32)
    norm = np.sqrt(np.sum(x * x, axis=1, keepdims=True, dtype=np.float32)).astype(np.float32, copy=False)
    return (x / np.maximum(norm, np.float32(1e-12))).astype(np.float32, copy=False)


def _cross_distance(video: np.ndarray, audio: np.ndarray) -> np.ndarray:
    if video.ndim != 2 or audio.ndim != 2 or video.shape[1] != audio.shape[1] or not len(video) or not len(audio):
        raise ConsumerError(f"invalid V×A geometry: {video.shape}/{audio.shape}")
    result = np.empty((len(video), len(audio)), dtype=np.float32)
    for start in range(0, len(video), 8):
        stop = min(start + 8, len(video))
        delta = video[start:stop, None, :] - audio[None, :, :]
        delta += EPS
        result[start:stop] = np.sqrt(np.sum(delta * delta, axis=-1, dtype=np.float32)).astype(np.float32, copy=False)
    if not np.isfinite(result).all():
        raise ConsumerError("independently reconstructed V×A matrix is nonfinite")
    return result


def _zero_distance(video: np.ndarray) -> np.ndarray:
    shifted = np.asarray(video, dtype=np.float32) + EPS
    value = np.sqrt(np.sum(shifted * shifted, axis=1, dtype=np.float32)).astype(np.float32, copy=False)
    if not np.isfinite(value).all():
        raise ConsumerError("independently reconstructed zero-distance vector is nonfinite")
    return value


def _lag_distance(cross: np.ndarray, zero: np.ndarray, query_count: int, audio_count: int) -> np.ndarray:
    result = np.empty((query_count, 31), dtype=np.float32)
    for query in range(query_count):
        for column, lag in enumerate(range(-VSHIFT, VSHIFT + 1)):
            audio_index = query + lag
            result[query, column] = cross[query, audio_index] if 0 <= audio_index < audio_count else zero[query]
    return result


def _supports(count: int) -> dict[str, np.ndarray]:
    all_rows = np.arange(count, dtype=np.int32)
    return {
        "full": all_rows,
        "valid": all_rows.copy(),
        "guard20": np.arange(min(20, count), max(min(20, count), count - 20), dtype=np.int32),
        "guard15": np.arange(min(15, count), max(min(15, count), count - 15), dtype=np.int32),
    }


def _metrics(matrix: np.ndarray, supports: Mapping[str, np.ndarray]) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for name in SUPPORTS:
        indices = supports[name]
        if not len(indices):
            output[name] = {"status": "too_short", "support_n": 0, "indices": []}
            continue
        curve = np.asarray(matrix[indices], dtype=np.float32).mean(axis=0, dtype=np.float32).astype(np.float64)
        minimum_idx = int(np.argmin(curve))
        d_min = float(curve[minimum_idx])
        d_zero = float(curve[ZERO_COLUMN])
        median = float(np.median(curve))
        output[name] = {
            "status": "ok", "support_n": int(len(indices)), "indices": indices.tolist(), "curve": curve.tolist(),
            "C": median - d_min, "C_anchor": median - float(curve[ANCHOR_COLUMN]), "B": median,
            "D": d_min, "D0": d_zero, "search_gain": d_zero - d_min,
            "D_anchor": float(curve[ANCHOR_COLUMN]), "best_lag": int(minimum_idx - VSHIFT),
        }
    return output


def _metric_match(saved: Mapping[str, Any], rebuilt: Mapping[str, Any], label: str) -> None:
    for support in SUPPORTS:
        left, right = saved.get(support), rebuilt.get(support)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or left.get("status") != right.get("status"):
            raise ConsumerError(f"{label} support status differs: {support}")
        if left.get("status") != "ok":
            continue
        if left.get("support_n") != right.get("support_n") or left.get("indices") != right.get("indices") or left.get("best_lag") != right.get("best_lag"):
            raise ConsumerError(f"{label} support coordinates differ: {support}")
        for metric in (*METRICS, "D_anchor"):
            if not math.isclose(float(left[metric]), float(right[metric]), rel_tol=1e-5, abs_tol=2e-5):
                raise ConsumerError(f"{label} {support}/{metric} differs: {left[metric]} vs {right[metric]}")
        if not np.allclose(np.asarray(left["curve"], dtype=np.float64), np.asarray(right["curve"], dtype=np.float64), rtol=1e-5, atol=2e-5):
            raise ConsumerError(f"{label} {support} lag curve differs")


def _array_diff(actual: np.ndarray, expected: np.ndarray, label: str, *, rtol: float = 2e-5, atol: float = 2e-5) -> dict[str, Any]:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise ConsumerError(f"{label} shape/dtype differs: {actual.shape}/{actual.dtype} != {expected.shape}/{expected.dtype}")
    delta = actual.astype(np.float64) - expected.astype(np.float64)
    max_abs = float(np.max(np.abs(delta), initial=0.0))
    mean_abs = float(np.mean(np.abs(delta), dtype=np.float64))
    exact = bool(np.array_equal(actual, expected))
    if not exact and not np.allclose(actual, expected, rtol=rtol, atol=atol):
        raise ConsumerError(f"{label} independent reconstruction differs (max_abs={max_abs})")
    return {
        "exact": exact, "max_abs_difference": max_abs, "mean_abs_difference": mean_abs,
        "actual_array_sha256": _sha_array(actual), "saved_array_sha256": _sha_array(expected),
        "tolerance": {"rtol": rtol, "atol": atol},
    }


def _metric_contrast(values: Mapping[str, Mapping[str, Mapping[str, float]]], contrast: Mapping[str, float], geometry: str, support: str, metric: str) -> float:
    return float(sum(weight * float(values[condition][geometry][support][metric]) for condition, weight in contrast.items()))


def _temporal_permutations(audio_count: int, ordered_index: int) -> tuple[list[np.ndarray], list[str]]:
    rng = np.random.Generator(np.random.PCG64(PERMUTATION_SEED + int(ordered_index)))
    expected = np.arange(audio_count, dtype=np.int64)
    permutations, hashes = [], []
    for _ in range(PERMUTATION_COUNT):
        pi = rng.permutation(audio_count).astype(np.int64, copy=False)
        if not np.array_equal(np.sort(pi), expected):
            raise ConsumerError("PCG64 did not return a complete RAW-A row permutation")
        permutations.append(pi)
        hashes.append(hashlib.sha256(np.asarray(pi, dtype="<i8").tobytes()).hexdigest())
    return permutations, hashes


def _permuted_lag(cross: np.ndarray, zero: np.ndarray, query_count: int, permutation: np.ndarray) -> np.ndarray:
    audio_count = cross.shape[1]
    queries = np.arange(query_count, dtype=np.int64)[:, None]
    lag_offsets = np.arange(-VSHIFT, VSHIFT + 1, dtype=np.int64)[None, :]
    source = queries + lag_offsets
    valid = (source >= 0) & (source < audio_count)
    query_grid = np.broadcast_to(queries, source.shape)
    result = np.empty(source.shape, dtype=np.float32)
    result[valid] = cross[query_grid[valid], permutation[source[valid]]]
    result[~valid] = zero[query_grid[~valid]]
    return result


def _verify_events(
    control: Path,
    plan: Mapping[str, Any],
    run: Mapping[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    runtime_path = _require_sha(run.get("execution_runtime_path", ""), str(run.get("execution_runtime_raw_sha256", "")), "mel execution runtime")
    runtime = _read_json(runtime_path)
    if runtime.get("torch_flags_after_deterministic_override_before_forward") != TORCH_FLAGS:
        raise ConsumerError("mel runtime forward flags differ from frozen deterministic policy")
    if runtime.get("deterministic_override") != {"applied": True, "seed": 20260930, "cublas_workspace_config": ":4096:8"}:
        raise ConsumerError("mel runtime deterministic override metadata differs")
    if runtime.get("model_forward_performed_at_creation") is not False or runtime.get("effects_read") is not False:
        raise ConsumerError("mel runtime record has invalid creation/effects state")
    event_path = _require_sha(run.get("execution_events_path", ""), str(run.get("execution_events_raw_sha256", "")), "mel execution event log")
    events = _read_jsonl(event_path)
    if len(events) != run.get("execution_event_count") or not events:
        raise ConsumerError("mel execution event count differs from run manifest")
    event_map: dict[Any, dict[str, Any]] = {}
    forward_counts: dict[str, int] = {}
    for event in events:
        key = event.get("event_id")
        if key is None or key in event_map or event.get("torch_flags_before") != TORCH_FLAGS:
            raise ConsumerError("mel event IDs are duplicated/missing or a forward used incorrect flags")
        forward = event.get("forward")
        if not isinstance(forward, str) or not forward:
            raise ConsumerError("mel event lacks forward name")
        event_map[key] = event
        forward_counts[forward] = forward_counts.get(forward, 0) + 1
    expected_forward_kinds = {"wav2lip_face_encoder_cache", "wav2lip_face_encoder_block", "wav2lip_audio_encoder", "wav2lip_decoder", "syncnet_forward_aud", "syncnet_forward_lip"}
    if not expected_forward_kinds.issubset(forward_counts):
        raise ConsumerError(f"mel event log lacks a required forward kind: {sorted(expected_forward_kinds-set(forward_counts))}")
    generation_batch_size = int(plan.get("generation", {}).get("batch_size", -1))
    syncnet_batch_size = int(plan.get("syncnet", {}).get("batch_size", -1))
    if generation_batch_size != 64 or syncnet_batch_size != 20:
        raise ConsumerError("mel plan generation/SyncNet batch sizes differ from the frozen execution")
    referenced_event_ids: set[Any] = set()

    def ref(event: Mapping[str, Any], *, forward: str, sample_id: str, condition: str | None, label: str) -> None:
        found = event_map.get(event.get("event_id"))
        if found is None or dict(found) != dict(event):
            raise ConsumerError(f"{label} forward event is not present byte-equivalent in the sealed event log")
        if event.get("forward") != forward or event.get("sample_id") != sample_id or event.get("torch_flags_before") != TORCH_FLAGS:
            raise ConsumerError(f"{label} forward event identity/flags differ")
        if condition is not None and event.get("condition") != condition:
            raise ConsumerError(f"{label} forward condition differs")
        referenced_event_ids.add(event["event_id"])

    def check_batched_events(
        batch_events: list[Mapping[str, Any]],
        *, start_key: str,
        expected_count: int,
        batch_size: int,
        label: str,
    ) -> None:
        ordered_events = sorted(batch_events, key=lambda event: int(event.get(start_key, -1)))
        cursor = 0
        for event in ordered_events:
            start = int(event.get(start_key, -1))
            stop = int(event.get("batch_stop", -1)) if start_key == "batch_start" else start + int((event.get("input_shape") or [0])[0])
            shape = event.get("input_shape")
            if not isinstance(shape, list) or not shape or start != cursor or stop <= start or stop - start > batch_size or int(shape[0]) != stop - start:
                raise ConsumerError(f"{label} event batches are not a contiguous frozen-size partition")
            cursor = stop
        if cursor != expected_count:
            raise ConsumerError(f"{label} event batches cover {cursor} rows, expected {expected_count}")

    for row in rows:
        sample_id = str(row.get("id"))
        shared = row.get("shared_A")
        if not isinstance(shared, Mapping):
            raise ConsumerError(f"{sample_id} has no shared A forward record")
        aud_events = shared.get("forward_events")
        if not isinstance(aud_events, list) or not aud_events:
            raise ConsumerError(f"{sample_id} lacks shared SyncNet-A forward events")
        for event in aud_events:
            ref(event, forward="syncnet_forward_aud", sample_id=sample_id, condition=None, label=f"{sample_id}/A")
        audio_count = int(shared.get("feature_meta", {}).get("feature_count", -1))
        check_batched_events(aud_events, start_key="batch_start", expected_count=audio_count, batch_size=syncnet_batch_size, label=f"{sample_id}/A")
        for condition in CONDITIONS:
            cell = row.get("cells", {}).get(condition)
            if not isinstance(cell, Mapping):
                raise ConsumerError(f"{sample_id}/{condition} cell is absent")
            latent = cell.get("latents", {})
            encoder_events = latent.get("encoder_forward_events") if isinstance(latent, Mapping) else None
            if not isinstance(encoder_events, list) or not encoder_events:
                raise ConsumerError(f"{sample_id}/{condition} lacks encoder events")
            for event in encoder_events:
                ref(event, forward="wav2lip_audio_encoder", sample_id=sample_id, condition=condition, label=f"{sample_id}/{condition}/encoder")
            latent_count = int(latent.get("shape", [-1])[0])
            check_batched_events(encoder_events, start_key="batch_start", expected_count=latent_count, batch_size=generation_batch_size, label=f"{sample_id}/{condition}/encoder")
            video = cell.get("video", {})
            decoder_event = video.get("decoder_event_last") if isinstance(video, Mapping) else None
            if not isinstance(decoder_event, Mapping):
                raise ConsumerError(f"{sample_id}/{condition} lacks its decoder event")
            ref(decoder_event, forward="wav2lip_decoder", sample_id=sample_id, condition=condition, label=f"{sample_id}/{condition}/decoder")
            decoder_events = [event for event in events if event.get("forward") == "wav2lip_decoder" and event.get("sample_id") == sample_id and event.get("condition") == condition]
            check_batched_events(decoder_events, start_key="batch_start", expected_count=int(video.get("frame_count", -1)), batch_size=generation_batch_size, label=f"{sample_id}/{condition}/decoder")
            if not decoder_events or max(decoder_events, key=lambda event: int(event.get("batch_start", -1))) != dict(decoder_event):
                raise ConsumerError(f"{sample_id}/{condition} last decoder event does not match its event log sequence")
            for event in decoder_events:
                referenced_event_ids.add(event["event_id"])
            v_events = cell.get("V", {}).get("forward_events")
            if not isinstance(v_events, list) or not v_events:
                raise ConsumerError(f"{sample_id}/{condition} lacks SyncNet-V forward events")
            for event in v_events:
                ref(event, forward="syncnet_forward_lip", sample_id=sample_id, condition=condition, label=f"{sample_id}/{condition}/V")
            visual_count = int(cell.get("V", {}).get("feature_meta", {}).get("feature_count", -1))
            check_batched_events(v_events, start_key="frame_start", expected_count=visual_count, batch_size=syncnet_batch_size, label=f"{sample_id}/{condition}/V")
    face_cache_events = [event for event in events if event.get("forward") == "wav2lip_face_encoder_cache"]
    face_block_events = [event for event in events if event.get("forward") == "wav2lip_face_encoder_block"]
    if len(face_cache_events) != 1 or face_cache_events[0].get("cache_shape") != [64, 6, 96, 96]:
        raise ConsumerError("shared Wav2Lip face-cache event differs from the sealed execution")
    if not face_block_events or any(event.get("sample_id") != "__shared_primary_face__" or event.get("cache_batch_size") != 64 for event in face_block_events):
        raise ConsumerError("shared Wav2Lip face-encoder block event set is malformed")
    if sorted(int(event.get("block_index", -1)) for event in face_block_events) != list(range(len(face_block_events))):
        raise ConsumerError("shared Wav2Lip face-encoder block event indices are not contiguous")
    referenced_event_ids.update(event["event_id"] for event in face_cache_events + face_block_events)
    if referenced_event_ids != set(event_map):
        raise ConsumerError(f"forward event log has missing or unreferenced records: missing={len(set(event_map)-referenced_event_ids)}, extra={len(referenced_event_ids-set(event_map))}")
    return {
        "runtime_path": str(runtime_path), "runtime_raw_sha256": _sha_file(runtime_path),
        "runtime_forward_flags": runtime["torch_flags_after_deterministic_override_before_forward"],
        "event_path": str(event_path), "event_raw_sha256": _sha_file(event_path),
        "event_count": len(events), "forward_counts": forward_counts,
        "all_event_records_accounted_for": True,
        "referenced_audio_event_count": sum(len(row["shared_A"]["forward_events"]) for row in rows),
        "referenced_wav2lip_audio_encoder_events": sum(len(row["cells"][condition]["latents"]["encoder_forward_events"]) for row in rows for condition in CONDITIONS),
        "referenced_wav2lip_decoder_last_events": len(rows) * len(CONDITIONS),
        "referenced_syncnet_visual_events": sum(len(row["cells"][condition]["V"]["forward_events"]) for row in rows for condition in CONDITIONS),
        "all_referenced_events_match_log": True,
    }


def _verify_output_ledger(run: Mapping[str, Any]) -> tuple[Path, dict[str, str]]:
    root = Path(str(run.get("artifact_root", ""))).resolve(strict=True)
    ledger = run.get("artifact_ledger")
    if not isinstance(ledger, Mapping) or len(ledger) != int(run.get("artifact_count", -1)):
        raise ConsumerError("mel output artifact ledger/count differs")
    checked = {}
    for relative, expected in ledger.items():
        candidate = (root / str(relative)).resolve()
        if root not in candidate.parents:
            raise ConsumerError(f"mel output ledger path escapes artifact root: {relative}")
        _require_sha(candidate, str(expected), f"mel output artifact {relative}")
        checked[str(relative)] = str(expected)
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual != set(checked):
        raise ConsumerError(f"mel output artifact ledger membership differs: missing={sorted(set(checked)-actual)[:3]}, extra={sorted(actual-set(checked))[:3]}")
    return root, checked


def _verify_recorded_output(record: Mapping[str, Any], artifact_root: Path, ledger: Mapping[str, str], label: str, *, sha_field: str) -> Path:
    raw_path = Path(str(record.get("path", "")))
    if not raw_path.is_absolute():
        raw_path = artifact_root / raw_path
    path = raw_path.resolve(strict=True)
    try:
        relative = path.relative_to(artifact_root.resolve(strict=True)).as_posix()
    except ValueError as exc:
        raise ConsumerError(f"{label} path escapes sealed output root") from exc
    file_sha = str(record.get(sha_field, ""))
    if ledger.get(relative) != file_sha:
        raise ConsumerError(f"{label} path/file SHA differs from sealed output ledger")
    _require_sha(path, file_sha, label)
    return path


def _score_one_row(
    planned: Mapping[str, Any],
    row: Mapping[str, Any],
    artifact_root: Path,
    ledger: Mapping[str, str],
    event_map: Mapping[Any, Mapping[str, Any]],
    *,
    ordered_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    sample_id, speaker = str(planned.get("id")), str(planned.get("speaker"))
    if row.get("id") != sample_id or row.get("speaker") != speaker or row.get("split") != "evaluation" or row.get("status") != "complete":
        raise ConsumerError(f"EVAL row identity/status differs for {sample_id}")
    if row.get("model_forward_performed") is not True or row.get("effects_read") is not False:
        raise ConsumerError(f"EVAL row forward/effect state differs for {sample_id}")
    shared_a_record = row.get("shared_A")
    if not isinstance(shared_a_record, Mapping):
        raise ConsumerError(f"{sample_id} has no shared RAW-N A record")
    audio, audio_path = _load_array_record(shared_a_record, artifact_root, f"{sample_id} shared RAW-N A", expected_dim=EMBEDDING_DIM)
    audio_meta = shared_a_record.get("feature_meta")
    if not isinstance(audio_meta, Mapping) or audio_meta.get("feature_count") != len(audio) or audio_meta.get("sample_rate_hz") != 16_000:
        raise ConsumerError(f"{sample_id} shared A feature metadata differs")
    if audio_meta.get("source_pcm_array_sha256") != planned.get("source_pcm_array_sha256") or audio_meta.get("sample_count") != planned.get("sample_count"):
        raise ConsumerError(f"{sample_id} shared RAW-N A is not bound to the planned RAW PCM")
    _verify_recorded_output(shared_a_record, artifact_root, ledger, f"{sample_id} A feature file", sha_field="file_sha256")
    cell_data: dict[str, dict[str, Any]] = {}
    cell_metric_values: dict[str, dict[str, dict[str, float]]] = {}
    v_counts = {}
    for condition in CONDITIONS:
        cell = row.get("cells", {}).get(condition)
        if not isinstance(cell, Mapping) or cell.get("status") != "generated_scored" or cell.get("condition") != condition:
            raise ConsumerError(f"{sample_id}/{condition} is not a generated/scored cell")
        z_record, v_record = cell.get("latents"), cell.get("V")
        if not isinstance(z_record, Mapping) or not isinstance(v_record, Mapping):
            raise ConsumerError(f"{sample_id}/{condition} lacks latent/V array record")
        latents, _ = _load_array_record(z_record, artifact_root, f"{sample_id}/{condition} audio-encoder latents", expected_dim=512)
        _verify_recorded_output(z_record, artifact_root, ledger, f"{sample_id}/{condition} latent file", sha_field="file_sha256")
        if latents.shape[0] != int(planned.get("chunk_count", -1)):
            raise ConsumerError(f"{sample_id}/{condition} latent count differs from sealed mel chunks")
        if z_record.get("input_chunk_file_sha256") != planned.get("mel_file_sha256") or z_record.get("input_chunk_array_sha256") != planned.get("mel_arrays", {}).get(f"chunks_{condition}", {}).get("array_sha256"):
            raise ConsumerError(f"{sample_id}/{condition} latent is not bound to its exact mel chunk source")
        video_record, crop_record = cell.get("video"), cell.get("scored_video")
        if not isinstance(video_record, Mapping) or not isinstance(crop_record, Mapping):
            raise ConsumerError(f"{sample_id}/{condition} lacks decoded video metadata")
        video_path = _verify_recorded_output(video_record, artifact_root, ledger, f"{sample_id}/{condition} video", sha_field="sha256")
        crop_path = _verify_recorded_output(crop_record, artifact_root, ledger, f"{sample_id}/{condition} scored crop", sha_field="sha256")
        frame_count = int(video_record.get("frame_count", -1))
        crop_frames = int(crop_record.get("frame_count", -1))
        v, _ = _load_array_record(v_record, artifact_root, f"{sample_id}/{condition} V", expected_dim=EMBEDDING_DIM)
        _verify_recorded_output(v_record, artifact_root, ledger, f"{sample_id}/{condition} V file", sha_field="file_sha256")
        v_meta = v_record.get("feature_meta")
        if not isinstance(v_meta, Mapping) or v_meta.get("frame_count") != frame_count or v_meta.get("feature_count") != len(v):
            raise ConsumerError(f"{sample_id}/{condition} SyncNet V clock metadata differs")
        if frame_count != crop_frames or frame_count != len(latents) or len(v) != frame_count - 5:
            raise ConsumerError(f"{sample_id}/{condition} latent/video/crop/V clocks do not satisfy N-5")
        v_counts[condition] = len(v)
        cell_data[condition] = {
            "latents": latents, "V": v, "video_path": str(video_path), "video_sha256": _sha_file(video_path),
            "scored_crop_path": str(crop_path), "scored_crop_sha256": _sha_file(crop_path),
            "latent_array_sha256": _sha_array(latents), "V_array_sha256": _sha_array(v),
            "frames": frame_count, "A_frames": len(audio),
        }
    common_n = min(len(audio), *(v_counts[name] for name in CONDITIONS))
    score_record = row.get("scores")
    if not isinstance(score_record, Mapping) or score_record.get("common_support_n") != common_n or score_record.get("A_count") != len(audio):
        raise ConsumerError(f"{sample_id} saved common support/A count differs")
    if score_record.get("V_counts") != v_counts:
        raise ConsumerError(f"{sample_id} saved V counts differ")
    support_map = _supports(common_n)
    for geometry in GEOMETRIES:
        audio_features = audio if geometry == "raw" else _row_unit(audio)
        for condition in CONDITIONS:
            video = cell_data[condition]["V"]
            video_features = video if geometry == "raw" else _row_unit(video)
            cross = _cross_distance(video_features, audio_features)
            zero = _zero_distance(video_features)
            lag = _lag_distance(cross, zero, common_n, len(audio_features))
            geometry_row = score_record.get("geometries", {}).get(geometry, {}).get(condition)
            if not isinstance(geometry_row, Mapping):
                raise ConsumerError(f"{sample_id}/{geometry}/{condition} saved score row is missing")
            saved_meta = geometry_row.get("distance_artifact")
            saved_metrics = geometry_row.get("metrics")
            if not isinstance(saved_meta, Mapping) or not isinstance(saved_metrics, Mapping):
                raise ConsumerError(f"{sample_id}/{geometry}/{condition} saved score matrices/metrics missing")
            distance_arrays, distance_path = _load_distance_artifact(saved_meta, artifact_root, f"{sample_id}/{geometry}/{condition} distance")
            _verify_recorded_output(saved_meta, artifact_root, ledger, f"{sample_id}/{geometry}/{condition} distance file", sha_field="file_sha256")
            comparisons = {
                "cross_v_to_a_float32": _array_diff(cross, distance_arrays["cross_v_to_a_float32"], f"{sample_id}/{geometry}/{condition} cross"),
                "zero_boundary_float32": _array_diff(zero, distance_arrays["zero_boundary_float32"], f"{sample_id}/{geometry}/{condition} zero"),
                "lag_distance_float32": _array_diff(lag, distance_arrays["lag_distance_float32"], f"{sample_id}/{geometry}/{condition} lag"),
            }
            if not np.array_equal(distance_arrays["lag_columns_int32"], np.arange(31, dtype=np.int32)) or not np.array_equal(distance_arrays["lag_k_int32"], np.arange(-15, 16, dtype=np.int32)):
                raise ConsumerError(f"{sample_id}/{geometry}/{condition} frozen lag coordinate arrays differ")
            for name, expected in support_map.items():
                if not np.array_equal(distance_arrays[f"support_{name}_int32"], expected):
                    raise ConsumerError(f"{sample_id}/{geometry}/{condition} {name} support indices differ")
            rebuilt_metrics = _metrics(lag, support_map)
            _metric_match(saved_metrics, rebuilt_metrics, f"{sample_id}/{geometry}/{condition}")
            cell_metric_values.setdefault(condition, {})[geometry] = {
                support: {metric: float(rebuilt_metrics[support][metric]) for metric in METRICS}
                for support in SUPPORTS if rebuilt_metrics[support].get("status") == "ok"
            }
            cell_data[condition].setdefault("matrix_qa", {})[geometry] = {
                "reconstructed_cross_shape": list(cross.shape), "reconstructed_cross_array_sha256": _sha_array(cross),
                "saved_cross_array_sha256": _sha_array(distance_arrays["cross_v_to_a_float32"]),
                "reconstructed_zero_shape": list(zero.shape), "reconstructed_lag_shape": list(lag.shape),
                "distance_npz_path": str(distance_path), "distance_npz_file_sha256": _sha_file(distance_path),
                "array_comparisons": comparisons, "support_counts": {name: len(value) for name, value in support_map.items()},
                "independent_metric_reconstruction_matches": True,
            }
    permutations, permutation_hashes = _temporal_permutations(len(audio), ordered_index)
    null_values: dict[str, dict[str, dict[str, dict[str, np.ndarray]]]] = {}
    for geometry in GEOMETRIES:
        a = audio if geometry == "raw" else _row_unit(audio)
        for condition in CONDITIONS:
            v = cell_data[condition]["V"]
            v_geometry = v if geometry == "raw" else _row_unit(v)
            cross = _cross_distance(v_geometry, a)
            zero = _zero_distance(v_geometry)
            distributions = {support: {metric: np.empty(PERMUTATION_COUNT, dtype=np.float64) for metric in METRICS} for support in SUPPORTS}
            for draw_index, pi in enumerate(permutations):
                null_lag = _permuted_lag(cross, zero, common_n, pi)
                null_metrics = _metrics(null_lag, support_map)
                for support in SUPPORTS:
                    if null_metrics[support].get("status") == "ok":
                        for metric in METRICS:
                            distributions[support][metric][draw_index] = float(null_metrics[support][metric])
            null_values.setdefault(geometry, {})[condition] = distributions
    contrast_values: dict[str, dict[str, dict[str, dict[str, float]]]] = {}
    null_contrast: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for contrast_name, weights in CONTRASTS.items():
        contrast_values[contrast_name] = {}
        null_contrast[contrast_name] = {}
        for geometry in GEOMETRIES:
            contrast_values[contrast_name][geometry] = {}
            null_contrast[contrast_name][geometry] = {}
            for support in SUPPORTS:
                if any(support not in cell_metric_values[condition][geometry] for condition in weights):
                    continue
                contrast_values[contrast_name][geometry][support] = {}
                null_contrast[contrast_name][geometry][support] = {}
                for metric in METRICS:
                    observed = _metric_contrast(cell_metric_values, weights, geometry, support, metric)
                    null_draws = np.zeros(PERMUTATION_COUNT, dtype=np.float64)
                    for condition, weight in weights.items():
                        null_draws += weight * null_values[geometry][condition][support][metric]
                    mean_null = float(np.mean(null_draws, dtype=np.float64))
                    contrast_values[contrast_name][geometry][support][metric] = observed
                    null_contrast[contrast_name][geometry][support][metric] = {
                        "observed": observed, "null_mean": mean_null, "observed_minus_mean_null": observed - mean_null,
                        "null_p05": float(np.quantile(null_draws, 0.05)), "null_p50": float(np.quantile(null_draws, 0.50)),
                        "null_p95": float(np.quantile(null_draws, 0.95)),
                    }
    row_output = {
        "id": sample_id, "speaker": speaker, "ordered_index": ordered_index,
        "A": {
            "path": str(audio_path), "file_sha256": _sha_file(audio_path), "array_sha256": _sha_array(audio),
            "shape": list(audio.shape), "dtype": str(audio.dtype), "source_pcm_array_sha256": audio_meta["source_pcm_array_sha256"],
            "frame_count_by_condition": v_counts, "common_support_n": common_n,
        },
        "cells": cell_data,
        "scores": {
            condition: {
                geometry: {
                    support: {metric: float(cell_metric_values[condition][geometry][support][metric]) for metric in METRICS}
                    for support in SUPPORTS if support in cell_metric_values[condition][geometry]
                }
                for geometry in GEOMETRIES
            }
            for condition in CONDITIONS
        },
        "contrasts": contrast_values,
        "temporal_null": {
            "permutation_count": PERMUTATION_COUNT, "generator": "numpy.PCG64",
            "seed": PERMUTATION_SEED + ordered_index, "full_audio_domain_n": len(audio),
            "permutation_sha256": permutation_hashes, "same_permutation_across_all_cells_and_geometries": True,
            "contrast_observed_minus_mean_null": null_contrast,
        },
    }
    return row_output, {"cell_metrics": cell_metric_values, "contrast_metrics": contrast_values, "null_contrasts": null_contrast}


def _speaker_bootstrap(speaker_values: Mapping[str, list[float]], draws: np.ndarray) -> dict[str, Any]:
    if len(speaker_values) != 40 or any(len(values) != 2 for values in speaker_values.values()):
        return {"status": "incomplete_speaker_support", "speaker_count": len(speaker_values), "clip_counts": {speaker: len(values) for speaker, values in speaker_values.items()}}
    speakers = sorted(speaker_values)
    speaker_means = np.asarray([np.mean(speaker_values[speaker], dtype=np.float64) for speaker in speakers], dtype=np.float64)
    boot = speaker_means[draws].mean(axis=1, dtype=np.float64)
    lower, upper = np.quantile(boot, [0.005, 0.995])
    return {
        "status": "complete",
        "speaker_count": 40,
        "clips_per_speaker": 2,
        "equal_speaker_mean": float(speaker_means.mean(dtype=np.float64)),
        "speaker_mean_median": float(np.median(speaker_means)),
        "speaker_means": {speaker: float(value) for speaker, value in zip(speakers, speaker_means)},
        "bootstrap": {
            "rng": "numpy.PCG64", "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS,
            "resampling_unit": "40 speakers with replacement; each speaker contributes its mean across two clips",
            "ci": "99 percent percentile [0.5,99.5]", "ci99": [float(lower), float(upper)],
        },
    }


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(results, key=lambda row: int(row["ordered_index"]))
    speakers = sorted({str(row["speaker"]) for row in ordered})
    if len(ordered) != 80 or len(speakers) != 40 or any(sum(row["speaker"] == speaker for row in ordered) != 2 for speaker in speakers):
        raise ConsumerError("aggregate cohort is not complete 80 clips/40 speakers/two clips each")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draws = rng.integers(0, len(speakers), size=(BOOTSTRAP_DRAWS, len(speakers)), dtype=np.int64)
    contrasts = {}
    for contrast_name in CONTRASTS:
        contrasts[contrast_name] = {}
        for geometry in GEOMETRIES:
            contrasts[contrast_name][geometry] = {}
            for support in SUPPORTS:
                contrasts[contrast_name][geometry][support] = {}
                for metric in METRICS:
                    values_by_speaker: dict[str, list[float]] = {}
                    for row in ordered:
                        value = row["contrasts"].get(contrast_name, {}).get(geometry, {}).get(support, {}).get(metric)
                        if value is not None:
                            values_by_speaker.setdefault(str(row["speaker"]), []).append(float(value))
                    contrasts[contrast_name][geometry][support][metric] = _speaker_bootstrap(values_by_speaker, draws)
    nulls = {}
    for contrast_name in CONTRASTS:
        nulls[contrast_name] = {}
        for geometry in GEOMETRIES:
            nulls[contrast_name][geometry] = {}
            for support in SUPPORTS:
                nulls[contrast_name][geometry][support] = {}
                for metric in METRICS:
                    by_speaker: dict[str, list[float]] = {}
                    null_mean_by_speaker: dict[str, list[float]] = {}
                    for row in ordered:
                        item = row["temporal_null"]["contrast_observed_minus_mean_null"].get(contrast_name, {}).get(geometry, {}).get(support, {}).get(metric)
                        if item is not None:
                            by_speaker.setdefault(str(row["speaker"]), []).append(float(item["observed_minus_mean_null"]))
                            null_mean_by_speaker.setdefault(str(row["speaker"]), []).append(float(item["null_mean"]))
                    nulls[contrast_name][geometry][support][metric] = {
                        "observed_minus_mean_null": _speaker_bootstrap(by_speaker, draws),
                        "mean_null": _speaker_bootstrap(null_mean_by_speaker, draws),
                    }
    return {
        "clip_count": 80, "speaker_count": 40, "clips_per_speaker": 2,
        "contrasts": contrasts, "temporal_null_contrasts": nulls,
        "bootstrap_draw_array_sha256": _sha_array(draws),
    }


def _consumer(args: argparse.Namespace) -> None:
    cal_root, eval_root, gpu_control = args.calibration_root, args.evaluation_root, args.gpu_control
    _require_sha(args.spec, SPEC_SHA256, "frozen mel partition specification")
    cal_plan, cal_manifest, cal_rows, cal_binding = _cpu_run(cal_root, "calibration", verify_ledger=not args.skip_full_file_hashes)
    eval_plan, eval_manifest, eval_rows, eval_binding = _cpu_run(eval_root, "evaluation", verify_ledger=not args.skip_full_file_hashes)
    cal_qa_record, cal_qa_rows = _verify_cpu_calibration_qa(cal_root.resolve(), cal_manifest)
    for field in ("input_manifest_raw_sha256", "tts_pair_manifest_raw_sha256"):
        if cal_binding[field] != eval_binding[field]:
            raise ConsumerError(f"CAL and EVAL CPU plans bind different {field}")
    for binding_name, expected in (
        ("calibration", cal_binding), ("evaluation", eval_binding),
    ):
        if expected["input_identity_sha256"] != INPUT_IDENTITY_SHA256:
            raise ConsumerError(f"{binding_name} input identity differs")
    cal_artifacts = cal_manifest["artifacts"]
    eval_artifacts = eval_manifest["artifacts"]
    cal_artifact_map = {(str(row["id"]), str(row["source_condition"])): row for row in cal_artifacts}
    cal_partition_rows = []
    for artifact in cal_artifacts:
        key = (str(artifact["id"]), str(artifact["source_condition"]))
        qa_row = cal_qa_rows.get(key)
        if qa_row is None:
            raise ConsumerError(f"CAL QA lacks artifact identity: {key}")
        partition, arrays = _verify_partition(artifact, split="calibration", saved_cal_summary={k: v for k, v in qa_row.items() if k not in ("id", "speaker", "source_condition", "status")})
        waveform = _verify_pcm_gain(artifact, f"CAL/{key[0]}/{key[1]}")
        partition["waveform_provenance"] = waveform
        cal_partition_rows.append(partition)
    eval_partition_rows = []
    eval_artifact_map = {str(row["id"]): row for row in eval_artifacts}
    eval_partition_arrays: dict[str, dict[str, np.ndarray]] = {}
    for artifact in eval_artifacts:
        key = str(artifact["id"])
        partition, arrays = _verify_partition(artifact, split="evaluation", saved_cal_summary=None)
        waveform = _verify_pcm_gain(artifact, f"EVAL/{key}/natural")
        partition["waveform_provenance"] = waveform
        eval_partition_rows.append(partition)
        eval_partition_arrays[key] = arrays
    if len(cal_partition_rows) != 40 or len(eval_partition_rows) != 80:
        raise ConsumerError("CAL/EVAL partition QA row counts differ")
    cal_qa = cal_qa_record

    gpu_control = gpu_control.resolve(strict=True)
    plan_path = gpu_control / "execution_plan.json"
    rows_path = gpu_control / "execution_results.jsonl"
    run_path = gpu_control / "run_manifest.json"
    plan, run = _read_json(plan_path), _read_json(run_path)
    if plan.get("payload_sha256") != _payload_sha(plan) or run.get("payload_sha256") != _payload_sha(run):
        raise ConsumerError("GPU execution plan/run manifest payload fingerprint mismatch")
    if plan.get("spec", {}).get("sha256") != SPEC_SHA256 or plan.get("input_identity_sha256") != INPUT_IDENTITY_SHA256:
        raise ConsumerError("GPU plan differs from the frozen mel spec/cohort identity")
    if plan.get("generation", {}).get("fps") != VIDEO_FPS:
        raise ConsumerError("GPU plan FPS differs from the independently verified mel chunk clock")
    if plan.get("eval_cpu_plan_input_manifest_sha256") != eval_binding["input_manifest_raw_sha256"] or plan.get("eval_cpu_tts_pair_manifest_sha256") != eval_binding["tts_pair_manifest_raw_sha256"]:
        raise ConsumerError("GPU plan does not bind the exact EVAL source and TTS pair manifest bytes")
    if run.get("status") != "complete" or run.get("model_forward_performed") is not True or run.get("effects_read") is not False or run.get("mel_frontend_recomputed") is not False:
        raise ConsumerError("mel GPU run is incomplete, effects-contaminated, or recomputed the frontend")
    if run.get("planned_rows") != 80 or run.get("completed_rows") != 80 or run.get("planned_cells") != 320 or run.get("failed_cells_or_rows") != 0:
        raise ConsumerError("mel GPU run counts do not satisfy 80/320 zero-failure plan")
    if Path(str(run.get("execution_plan_path", ""))).resolve() != plan_path.resolve() or run.get("execution_plan_raw_sha256") != _sha_file(plan_path) or run.get("execution_plan_payload_sha256") != plan.get("payload_sha256"):
        raise ConsumerError("GPU run manifest does not bind the provided execution plan")
    if Path(str(run.get("rows_path", ""))).resolve() != rows_path.resolve() or run.get("rows_raw_sha256") != _sha_file(rows_path):
        raise ConsumerError("GPU run manifest does not bind the provided result rows")
    if run.get("input_identity_sha256") != INPUT_IDENTITY_SHA256:
        raise ConsumerError("GPU run input identity differs")
    expected_cpu_cal = plan.get("cpu_calibration", {})
    expected_cpu_eval = plan.get("cpu_evaluation", {})
    if expected_cpu_cal != {key: cal_binding[key] for key in expected_cpu_cal}:
        raise ConsumerError("GPU plan is not bound to these exact CAL CPU seals")
    if expected_cpu_eval != {key: eval_binding[key] for key in expected_cpu_eval}:
        raise ConsumerError("GPU plan is not bound to these exact EVAL CPU seals")
    if cal_plan.get("frontend") != eval_plan.get("frontend") or cal_plan.get("producer_code", {}).get("sha256") != eval_plan.get("producer_code", {}).get("sha256"):
        raise ConsumerError("CAL and EVAL CPU plans do not share the exact frontend and producer source")
    if run.get("cpu_calibration") != expected_cpu_cal or run.get("cpu_evaluation") != expected_cpu_eval:
        raise ConsumerError("GPU run manifest CPU bindings differ from its frozen plan")
    results = _read_jsonl(rows_path)
    planned = plan.get("planned_samples")
    if not isinstance(planned, list) or len(planned) != 80 or len(results) != 80:
        raise ConsumerError("GPU planned/result row list does not contain all 80 clips")
    output_root, output_ledger = _verify_output_ledger(run)
    events_summary = _verify_events(gpu_control, plan, run, results)
    result_map = {str(row.get("id")): row for row in results}
    if len(result_map) != 80:
        raise ConsumerError("GPU result rows contain duplicate sample IDs")
    score_rows, internal_score_rows = [], []
    partition_by_id = eval_artifact_map
    for index, planned_row in enumerate(planned):
        if int(planned_row.get("ordered_index", -1)) != index:
            raise ConsumerError(f"GPU planned row order differs at index {index}")
        sample_id = str(planned_row.get("id"))
        result = result_map.get(sample_id)
        if result is None:
            raise ConsumerError(f"GPU result row missing for planned clip {sample_id}")
        cpu_artifact = partition_by_id.get(sample_id)
        if cpu_artifact is None or planned_row.get("mel_file_sha256") != cpu_artifact.get("mel_file_sha256"):
            raise ConsumerError(f"{sample_id} GPU plan does not bind the sealed EVAL mel artifact")
        if planned_row.get("speaker") != cpu_artifact.get("speaker") or planned_row.get("source_wav_file_sha256") != cpu_artifact.get("source_wav_sha256"):
            raise ConsumerError(f"{sample_id} GPU plan/source artifact identity differs")
        if planned_row.get("input_identity_sha256") not in (None, INPUT_IDENTITY_SHA256):
            raise ConsumerError(f"{sample_id} planned input identity differs")
        if set(planned_row.get("mel_arrays", {})) != set(cpu_artifact.get("arrays", {})) or planned_row.get("mel_arrays") != {
            name: {"shape": cpu_artifact["arrays"][name]["shape"], "dtype": cpu_artifact["arrays"][name]["dtype"], "array_sha256": cpu_artifact["arrays"][name]["array_sha256"]}
            for name in planned_row.get("mel_arrays", {})
        }:
            raise ConsumerError(f"{sample_id} GPU planned mel array references differ from CPU manifest")
        partition, score_detail = _score_one_row(planned_row, result, output_root, output_ledger, {}, ordered_index=index)
        # Confirm the consumer independently reloaded the same NPZ and mask/counts.
        if partition["U_bins"] != int(cpu_artifact.get("U_count", -1)) or partition["S_bins"] != int(cpu_artifact.get("S_count", -1)):
            raise ConsumerError(f"{sample_id} partition mask counts differ from the CPU manifest")
        score_detail["partition_qa"] = partition
        score_rows.append(score_detail)
        internal_score_rows.append({
            "id": partition["id"], "speaker": partition["speaker"], "ordered_index": index,
            "contrasts": score_detail["contrast_metrics"], "temporal_null": {
                "contrast_observed_minus_mean_null": score_detail["null_contrasts"],
            },
        })
    if set(result_map) != {str(row["id"]) for row in planned}:
        raise ConsumerError("GPU results contain rows outside the sealed plan")
    aggregate = _aggregate(internal_score_rows)

    report = {
        "schema_version": 1,
        "experiment_id": "confirm-mel-level-partition-20261001",
        "consumer_id": "independent-mel-level-partition-consumer-20261001-attempt06",
        "status": "PASS",
        "created_at_utc": _utc_now(),
        "consumer_source": {"path": str(Path(__file__).resolve()), "sha256": _sha_file(Path(__file__))},
        "spec": {"path": str(args.spec.resolve(strict=True)), "sha256": _sha_file(args.spec)},
        "constants": {
            "input_identity_sha256": INPUT_IDENTITY_SHA256, "target_rms": TARGET_RMS,
            "conditions": list(CONDITIONS), "geometries": list(GEOMETRIES), "supports": list(SUPPORTS),
            "mel_chunking": {"fps": VIDEO_FPS, "mel_rate_hz": MEL_RATE_HZ, "step_size": MEL_STEP_SIZE,
                             "rule": MEL_CHUNK_RULE, "start_sequence_independently_reconstructed": True},
            "metrics": list(METRICS), "contrasts": CONTRASTS, "anchor_column": ANCHOR_COLUMN,
            "anchor_k": 3, "official_offset": -3, "eps": float(EPS), "temporal_permutations": PERMUTATION_COUNT,
            "temporal_seed_rule": "20260930+ordered_eval_index", "bootstrap_draws": BOOTSTRAP_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_ci": "99% percentile [0.5,99.5]",
        },
        "input_seals": {
            "calibration_cpu": cal_binding, "evaluation_cpu": eval_binding, "calibration_qa": cal_qa,
            "mel_gpu_plan": {"path": str(plan_path), "raw_sha256": _sha_file(plan_path), "payload_sha256": plan["payload_sha256"]},
            "mel_gpu_run_manifest": {"path": str(run_path), "raw_sha256": _sha_file(run_path), "payload_sha256": run["payload_sha256"]},
            "mel_gpu_result_rows": {"path": str(rows_path), "raw_sha256": _sha_file(rows_path), "row_count": len(results)},
            "mel_gpu_event_log": events_summary,
            "mel_gpu_output_artifact_root": str(output_root), "mel_gpu_output_artifact_count": len(output_ledger),
            "mel_gpu_output_artifact_ledger": dict(output_ledger),
            "mel_gpu_runner_source_sha256": plan.get("code", {}).get("sha256"),
            "execution_assets_raw_sha256": plan.get("execution_assets_raw_sha256"),
            "runtime": plan.get("runtime"),
            "actual_wav2lip_imports": plan.get("actual_wav2lip_imports"),
            "frontend_provenance": eval_plan.get("frontend"),
            "model_forward_performed": True,
            "effects_read_in_producer": False,
            "effects_reconstructed_by_consumer": True,
            "mel_frontend_recomputed_by_consumer": False,
        },
        "calibration_partition_qa": {
            "input_rows": len(cal_partition_rows),
            "unique_speakers": len({row["speaker"] for row in cal_partition_rows}),
            "rows": cal_partition_rows,
            "no_endpoint_based_selection": True,
        },
        "evaluation_partition_qa": {
            "input_rows": len(eval_partition_rows),
            "unique_speakers": len({row["speaker"] for row in eval_partition_rows}),
            "rows": eval_partition_rows,
            "no_endpoint_based_selection": True,
        },
        "evaluation_score_reconstruction": {
            "status": "PASS", "planned_rows": 80, "complete_rows": len(score_rows),
            "clip_geometry_score_views": 160, "cell_geometry_score_records": 640, "support_metric_sets": 2560,
            "cells": 320, "raw_unit_full_distance_matrices_reconstructed": 640,
            "fixed_support_domains": list(SUPPORTS), "all_scores_reconstructed_without_producer_imports": True,
            "rows": score_rows,
        },
        "aggregate_statistics": aggregate,
        "main_rms_joint_bridge": {
            "status": "computed_as_mel_q11_minus_q00",
            "note": "The joint contrast is reported for direct comparison to the RMS main G. No main-run summary was supplied to this consumer, so this report does not assert cross-run numeric equality.",
        },
        "model_forward_performed_by_consumer": False,
        "effects_read_by_consumer": True,
        "payload_sha256": "",
    }
    report["payload_sha256"] = _payload_sha(report)
    output_path = args.output.resolve()
    if output_path.exists():
        raise ConsumerError(f"consumer output already exists; choose a fresh attempt file: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({
        "status": "PASS", "rows": len(score_rows), "cells": 320,
        "report": str(output_path), "report_raw_sha256": _sha_file(output_path),
        "payload_sha256": report["payload_sha256"], "effects_read_by_consumer": True,
        "calibration_partition_rows": len(cal_partition_rows), "evaluation_partition_rows": len(eval_partition_rows),
    }, ensure_ascii=False, sort_keys=True), flush=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration-root", required=True, type=Path, help="sealed mel CPU CAL run directory")
    parser.add_argument("--evaluation-root", required=True, type=Path, help="sealed mel CPU EVAL run directory")
    parser.add_argument("--gpu-control", required=True, type=Path, help="mel05 GPU control directory containing plan/results/events/runtime/run_manifest")
    parser.add_argument("--spec", required=True, type=Path, help="exact frozen mel partition specification")
    parser.add_argument("--output", required=True, type=Path, help="new, absent JSON report path")
    parser.add_argument("--skip-full-file-hashes", action="store_true", help="verify manifest membership and referenced files only, rather than rehashing every CPU-ledger file")
    parser.set_defaults(func=_consumer)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:2000]}, ensure_ascii=False, sort_keys=True), file=__import__("sys").stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
