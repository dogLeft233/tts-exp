#!/usr/bin/env python3
"""Execute the frozen MFA-linear trajectory ablation protocol.

The script intentionally keeps the experiment in small, restartable stages:
``prepare``, ``audio``, ``render``, ``score`` and ``analyze``.  The optional
``reuse`` stage creates a new support30 run from validated parent metadata. The
protocol is defined in the Basic Memory spec named ``MFA-linear 连续轨迹机制消融实验 Spec``.
No model is trained and no historical score is copied into the new matrix.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import shutil
import subprocess
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO / "scripts"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from knn_vc_retrieval import frame_owners, matched_span_map  # noqa: I001


RUN_ID = "mfa_linear_trajectory_ablation_20260916"
DEFAULT_OUTPUT = REPO / "runs" / RUN_ID
DEFAULT_A_SUMMARY = REPO / "runs/knn_vc_poc_valid15_mfa_linear_bridge_20260813/summary.json"
DEFAULT_B_ALIGNMENT = REPO / "runs/two_stage_hubert_aishell1_20260810/data_boundary/aishell1_400_raw_mfa_faster_qwen3_heldout.json"
DEFAULT_C_MANIFEST = REPO / "runs/rhythm_timing/20260813_syncnet_valid15_mfa_bridge/manifest.json"
DEFAULT_KNN_SOURCE = REPO / "third_party/knn-vc"
DEFAULT_WAV2LIP = REPO / "third_party/Wav2Lip"
DEFAULT_WAV2LIP_PYTHON = Path.home() / ".venvs/wav2lip/bin/python"
DEFAULT_SYNCNET = REPO / "third_party/syncnet_python"
DEFAULT_SYNCNET_PYTHON = Path.home() / ".venvs/syncnet/bin/python"
DEFAULT_WAV2LIP_CHECKPOINT = DEFAULT_WAV2LIP / "checkpoints/wav2lip_gan.pth"
DEFAULT_SYNCNET_MODEL = DEFAULT_SYNCNET / "data/syncnet_v2.model"

SAMPLE_RATE = 16_000
FRAME_STRIDE_SAMPLES = 320
FEATURE_DIM = 1024
FPS = 25.0
FACE_DET_THRESHOLD = 0.9
VSHIFT = 15
MIN_COMMON_WINDOWS = 50
SENSITIVITY_THRESHOLDS = (30, 35, 40, 45, 50)
V2_PROTOCOL = "mfa_linear_trajectory_ablation_v2_support30"
BOOTSTRAP_SEED = 20260916
BOOTSTRAP_DRAWS = 10_000

ARMS = ("N_RAW", "T_RAW", "N_100", "N_050", "N_000", "T_100", "T_050", "T_000")
RESYNTH_ARMS = ("N_100", "N_050", "N_000", "T_100", "T_050", "T_000")
KEEP_BY_ARM = {"N_100": 1.0, "N_050": 0.5, "N_000": 0.0, "T_100": 1.0, "T_050": 0.5, "T_000": 0.0}
SOURCE_BY_ARM = {"N_100": "N", "N_050": "N", "N_000": "N", "T_100": "T", "T_050": "T", "T_000": "T"}


class ProtocolError(RuntimeError):
    """An input or output violates the frozen protocol."""


def _validate_min_common_windows(value: Any) -> int:
    """Validate the number of shared SyncNet windows used by one cell."""
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"min-common-windows must be a positive integer: {value!r}") from exc
    if integer <= 0:
        raise ProtocolError(f"min-common-windows must be positive: {integer}")
    return integer


def _effective_min_common_windows(args: argparse.Namespace | None = None, inputs: Mapping[str, Any] | None = None) -> int:
    """Resolve one threshold and reject CLI/metadata disagreements."""
    explicit = getattr(args, "min_common_windows", None) if args is not None else None
    metadata = inputs.get("min_common_windows") if inputs is not None else None
    if explicit is not None:
        value = _validate_min_common_windows(explicit)
        if metadata is not None and value != _validate_min_common_windows(metadata):
            raise ProtocolError(f"CLI min-common-windows {value} disagrees with inputs.json {metadata}")
        return value
    if metadata is not None:
        return _validate_min_common_windows(metadata)
    return MIN_COMMON_WINDOWS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return bytes_sha256(encoded)


def write_json(path: Path, value: Mapping[str, Any] | Sequence[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"cannot read JSON: {path}: {exc}") from exc


def resolve_path(value: str | Path, *, expected_sha256: str | None = None) -> Path:
    """Resolve old absolute paths without silently accepting a hash mismatch."""
    raw = Path(str(value)).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        candidates.append(raw)
        parts = raw.parts
        if "tts-exp" in parts:
            index = len(parts) - 1 - parts[::-1].index("tts-exp")
            candidates.append(REPO.joinpath(*parts[index + 1 :]))
        candidates.append(REPO / raw.name)
    else:
        candidates.extend((REPO / raw, raw))
    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_file():
            continue
        seen.add(candidate)
        if expected_sha256 is None or file_sha256(candidate) == expected_sha256:
            return candidate
    expected = f" with sha256={expected_sha256}" if expected_sha256 else ""
    raise ProtocolError(f"asset not found{expected}: {value}")


def _require_file(path: Path, label: str) -> Path:
    if not path.is_file():
        raise ProtocolError(f"missing {label}: {path}")
    return path


def _audio(path: Path) -> tuple[np.ndarray, int]:
    try:
        values, rate = sf.read(str(path), dtype="float32", always_2d=False)
    except Exception as exc:  # pragma: no cover - backend-specific error text
        raise ProtocolError(f"cannot decode audio {path}: {exc}") from exc
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 1 or int(rate) != SAMPLE_RATE:
        raise ProtocolError(f"audio must be mono {SAMPLE_RATE} Hz: {path} shape={values.shape} rate={rate}")
    return values, int(rate)


def audio_qc(values: np.ndarray, *, natural_count: int, raw_decoder_count: int | None = None, action: str = "none") -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    finite = bool(np.isfinite(values).all())
    absolute = np.abs(values) if values.size else np.zeros(0, dtype=np.float32)
    return {
        "sample_count": int(values.size),
        "natural_sample_count": int(natural_count),
        "duration_s": float(values.size / SAMPLE_RATE),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "finite": finite,
        "nonfinite_sample_count": int((~np.isfinite(values)).sum()),
        "out_of_range_sample_count": int((absolute > 1.0).sum()),
        "peak": float(absolute.max()) if values.size else 0.0,
        "rms": float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0,
        "exact_natural_sample_count": bool(values.size == natural_count),
        "raw_decoder_sample_count": int(raw_decoder_count if raw_decoder_count is not None else values.size),
        "length_adjustment_action": action,
        "length_adjustment_sample_count": int(abs((raw_decoder_count if raw_decoder_count is not None else values.size) - natural_count)),
    }


def exact_natural_length(values: np.ndarray, count: int) -> tuple[np.ndarray, dict[str, Any]]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    raw_count = int(values.size)
    if raw_count > count:
        output = values[:count].copy()
        action = "right_crop"
    elif raw_count < count:
        output = np.pad(values, (0, count - raw_count)).astype(np.float32)
        action = "right_zero_pad"
    else:
        output = values.copy()
        action = "none"
    return output, {
        "raw_decoder_sample_count": raw_count,
        "target_natural_sample_count": int(count),
        "action": action,
        "adjustment_sample_count": abs(raw_count - count),
    }


def ablate_phone_dynamics(features: Any, occurrences: Sequence[Mapping[str, Any]], keep: float) -> Any:
    """Apply the exact per-occurrence mean-preserving residual scaling rule."""
    import torch

    values = torch.as_tensor(features, dtype=torch.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("features must be a non-empty [T,D] matrix")
    if not math.isfinite(float(keep)) or not 0.0 <= float(keep) <= 1.0:
        raise ValueError("keep must be in [0, 1]")
    if keep == 1.0:
        return values.clone()
    output = values.clone()
    for occurrence in occurrences:
        if occurrence.get("eligible") is not True:
            continue
        indices = [int(index) for index in occurrence.get("frame_indices", [])]
        if len(indices) < 2:
            continue
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=values.device)
        source = values.index_select(0, index_tensor)
        mean = source.mean(dim=0, keepdim=True)
        output[index_tensor] = mean + float(keep) * (source - mean)
    return output


def _occurrence_mask(natural_tokens: Sequence[Mapping[str, Any]], tts_tokens: Sequence[Mapping[str, Any]], frame_count: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    owners = frame_owners(frame_count, natural_tokens, frame_stride_samples=FRAME_STRIDE_SAMPLES, sample_rate=SAMPLE_RATE)
    mapping, match_stats = matched_span_map(natural_tokens, tts_tokens)
    by_span: dict[int, list[int]] = defaultdict(list)
    for frame_index, owner in enumerate(owners):
        by_span[owner.span_index].append(frame_index)
    occurrences: list[dict[str, Any]] = []
    for span_index, token in enumerate(natural_tokens):
        label = str(token.get("token", token.get("label", "")))
        frame_indices = by_span.get(span_index, [])
        tts_span = mapping.get(span_index)
        silence = bool(token.get("is_silence") or token.get("is_non_speech") or label.casefold() in {"", "sil", "sp", "spn", "<sil>"})
        eligible = bool(not silence and tts_span is not None and len(frame_indices) >= 2)
        if silence:
            reason = "silence"
        elif tts_span is None:
            reason = "unmatched"
        elif len(frame_indices) < 2:
            reason = "fewer_than_two_frames"
        else:
            reason = "eligible"
        occurrences.append({
            "occurrence_index": span_index,
            "label": label,
            "tts_span_index": tts_span,
            "frame_indices": frame_indices,
            "frame_count": len(frame_indices),
            "is_silence": silence,
            "eligible": eligible,
            "reason": reason,
        })
    eligible_frames = sorted({index for occurrence in occurrences if occurrence["eligible"] for index in occurrence["frame_indices"]})
    metadata = {
        "natural_frame_count": int(frame_count),
        "occurrence_count": len(occurrences),
        "eligible_occurrence_count": sum(bool(row["eligible"]) for row in occurrences),
        "eligible_frame_count": len(eligible_frames),
        "eligible_frame_fraction": float(len(eligible_frames) / frame_count),
        "matched_coverage": float(sum(owner.span_index in mapping for owner in owners) / frame_count),
        "fallback_frame_count": int(sum(owner.span_index not in mapping for owner in owners)),
        "match_stats": match_stats,
    }
    return occurrences, metadata


def trajectory_diagnostics(features: Any, occurrences: Sequence[Mapping[str, Any]]) -> dict[str, float | int]:
    import torch

    values = torch.as_tensor(features, dtype=torch.float32)
    internal: list[float] = []
    boundary: list[float] = []
    for occurrence in occurrences:
        indices = [int(value) for value in occurrence.get("frame_indices", [])]
        for left, right in pairwise(indices):
            internal.append(float(torch.sum((values[right] - values[left]) ** 2).item()))
        if indices and indices[-1] + 1 < values.shape[0]:
            right = indices[-1] + 1
            boundary.append(float(torch.sum((values[right] - values[indices[-1]]) ** 2).item()))
    return {
        "internal_pair_count": len(internal),
        "internal_difference_energy": float(np.mean(internal)) if internal else 0.0,
        "boundary_pair_count": len(boundary),
        "boundary_difference_energy": float(np.mean(boundary)) if boundary else 0.0,
    }


def _score_box(selected: Mapping[str, Any]) -> dict[str, Any]:
    width = float(selected["x2"]) - float(selected["x1"])
    height = float(selected["y2"]) - float(selected["y1"])
    side = max(1, round(1.5 * max(width, height)))
    center_x = (float(selected["x1"]) + float(selected["x2"])) / 2.0
    center_y = (float(selected["y1"]) + float(selected["y2"])) / 2.0
    left = math.floor(center_x - side / 2.0)
    top = math.floor(center_y - side / 2.0)
    return {
        "order": "x1,y1,x2,y2; unbounded square with zero padding",
        "center": [center_x, center_y],
        "side": side,
        "box": [left, top, left + side, top + side],
        "scale_from_detection": 1.5,
    }


def _crop_zero_padded(frame: np.ndarray, box: Mapping[str, Any], output_size: int = 224) -> np.ndarray:
    import cv2

    x1, y1, x2, y2 = [int(value) for value in box["box"]]
    side = int(box["side"])
    if x2 - x1 != side or y2 - y1 != side or side <= 0:
        raise ProtocolError("invalid square score box")
    canvas = np.zeros((side, side, 3), dtype=np.uint8)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(frame.shape[1], x2), min(frame.shape[0], y2)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - y1 : sy2 - y1, sx1 - x1 : sx2 - x1] = frame[sy1:sy2, sx1:sx2]
    return cv2.resize(canvas, (output_size, output_size), interpolation=cv2.INTER_LINEAR)


def _video_info(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open video: {path}")
    count = 0
    first = middle = last = None
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if first is None:
                first = frame.copy()
            frames.append(frame)
            count += 1
    finally:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        capture.release()
    if count < 25 or first is None:
        raise ProtocolError(f"video has insufficient frames: {path} ({count})")
    middle = frames[count // 2]
    last = frames[-1]
    if abs(fps - FPS) > 0.01:
        raise ProtocolError(f"video is not {FPS:g} fps: {path} ({fps})")
    return {"frame_count": count, "fps": fps, "width": int(first.shape[1]), "height": int(first.shape[0]), "first": first, "middle": middle, "last": last}


def _validate_crop_coverage(video: Path, score_box: Mapping[str, Any]) -> dict[str, Any]:
    info = _video_info(video)
    ratios = []
    for frame in (info["first"], info["middle"], info["last"]):
        crop = _crop_zero_padded(frame, score_box)
        ratios.append(float(np.mean(np.any(crop != 0, axis=2))))
    # This is only a geometry sanity check.  It is deliberately independent of
    # SyncNet scores and cannot be used to select a better crop.
    if min(ratios) <= 0.05:
        raise ProtocolError(f"fixed crop is mostly outside the face video: {video}, ratios={ratios}")
    return {"frame_count": info["frame_count"], "fps": info["fps"], "nonzero_crop_fraction_first_middle_last": ratios}


def _find_executable(explicit: Path | None, candidates: Sequence[Path], label: str) -> Path:
    if explicit is not None:
        return _require_file(explicit.expanduser(), label)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ProtocolError(f"cannot find {label}; tried {[str(path) for path in candidates]}")


def _selected_records(a: Mapping[str, Any], b: Mapping[str, Any], c: Mapping[str, Any], *, a_path: Path, b_path: Path, c_path: Path, wav2lip_checkpoint: Path, syncnet_model: Path, wav2lip_python: Path) -> list[dict[str, Any]]:
    selection = a.get("selection")
    ordered = list(selection.get("ordered_paired_keys", [])) if isinstance(selection, Mapping) else []
    if selection.get("speaker_group") != "S0765" or selection.get("selected_count") != 15 or len(ordered) != 15:
        raise ProtocolError("A summary must contain the frozen S0765 ordered 15-sample cohort")
    a_by_key = {str(item.get("paired_key")): item for item in a.get("items", [])}
    b_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in b.get("records", []):
        key = (str(row.get("paired_key")), str(row.get("condition")))
        if key[1] in {"natural", "tts"}:
            b_by_key[key] = row
    c_by_key: dict[tuple[str, str], Mapping[str, Any]] = {}
    for row in c.get("records", []):
        key = (str(row.get("paired_key")), str(row.get("arm")))
        c_by_key[key] = row
    if c.get("wav2lip_checkpoint_sha256") != file_sha256(wav2lip_checkpoint) or c.get("syncnet_model_sha256") != file_sha256(syncnet_model):
        raise ProtocolError("Wav2Lip/SyncNet checkpoint hash differs from frozen C manifest")
    if a.get("manifest", {}).get("alignment_sha256") != file_sha256(b_path):
        raise ProtocolError("B alignment manifest hash differs from A.manifest.alignment_sha256")
    rows: list[dict[str, Any]] = []
    for order, paired_key in enumerate(ordered, 1):
        item = a_by_key.get(str(paired_key))
        if item is None or int(item.get("sample_id")) != order or str(item.get("speaker_group", "S0765")) not in {"S0765", ""}:
            raise ProtocolError(f"A item/order mismatch at {order}: {paired_key}")
        natural_b = b_by_key.get((str(paired_key), "natural"))
        tts_b = b_by_key.get((str(paired_key), "tts"))
        natural_c = c_by_key.get((str(paired_key), "natural_raw"))
        tts_c = c_by_key.get((str(paired_key), "raw_tts"))
        if natural_b is None or tts_b is None or natural_c is None or tts_c is None:
            raise ProtocolError(f"missing natural/TTS source row for {paired_key}")
        for row in (natural_b, tts_b):
            if row.get("speaker_id") != "S0765" or row.get("split") != "valid" or row.get("variant") != "raw":
                raise ProtocolError(f"split/speaker/variant mismatch for {paired_key}")
        feature_rows = [row for row in item.get("conditions", []) if row.get("condition") == "paired_tts_mfa_linear"]
        if len(feature_rows) != 1:
            raise ProtocolError(f"A has no unique paired_tts_mfa_linear row: {paired_key}")
        feature_row = feature_rows[0]
        feature_path = resolve_path(feature_row["feature_path"], expected_sha256=str(feature_row.get("conditioning_sha256")))
        import torch

        payload = torch.load(feature_path, map_location="cpu", weights_only=False)
        if not isinstance(payload, Mapping) or "conditioning" not in payload or "output" not in payload:
            raise ProtocolError(f"feature file must contain conditioning and output: {feature_path}")
        conditioning = torch.as_tensor(payload["conditioning"], dtype=torch.float32)
        if list(conditioning.shape) != list(feature_row.get("conditioning_shape", [])):
            raise ProtocolError(f"conditioning shape metadata mismatch: {feature_path}")
        natural_audio = resolve_path(natural_c["audio"], expected_sha256=str(natural_c.get("audio_sha256")))
        tts_audio = resolve_path(tts_c["audio"], expected_sha256=str(tts_c.get("audio_sha256")))
        a_natural = resolve_path(item["natural_source"], expected_sha256=str(item.get("natural_source_sha256")))
        a_tts = resolve_path(item["tts_source"], expected_sha256=str(item.get("tts_source_sha256")))
        # The historical A/B TTS source is the 24 kHz waveform used for MFA
        # alignment, while C's ``raw_tts`` is its already-resampled 16 kHz
        # scoring input.  They are intentionally different files; retain both
        # identities instead of silently replacing one with the other.
        if file_sha256(natural_audio) != file_sha256(a_natural):
            raise ProtocolError(f"A/C natural source identity mismatch: {paired_key}")
        if str(natural_b.get("source_sha256")) != str(item.get("natural_source_sha256")) or str(tts_b.get("source_sha256")) != str(item.get("tts_source_sha256")):
            raise ProtocolError(f"B source hash mismatch: {paired_key}")
        if canonical_sha256(natural_b.get("tokens")) != str(item.get("natural_tokens_sha256")) or canonical_sha256(tts_b.get("tokens")) != str(item.get("tts_tokens_sha256")):
            raise ProtocolError(f"B token hash mismatch: {paired_key}")
        face = resolve_path(natural_c["face"], expected_sha256=str(natural_c.get("face_sha256")))
        tts_face = resolve_path(tts_c["face"], expected_sha256=str(tts_c.get("face_sha256")))
        if file_sha256(face) != file_sha256(tts_face):
            raise ProtocolError(f"natural/TTS face differs: {paired_key}")
        rows.append({
            "sample_id": int(item["sample_id"]),
            "paired_key": str(paired_key),
            "transcript": str(item.get("transcript", "")),
            "speaker_id": "S0765",
            "split": "valid",
            "natural_audio": str(natural_audio),
            "natural_audio_sha256": file_sha256(natural_audio),
            "tts_audio": str(tts_audio),
            "tts_audio_sha256": file_sha256(tts_audio),
            "face_video": str(face),
            "face_video_sha256": file_sha256(face),
            "natural_tokens": list(natural_b["tokens"]),
            "tts_tokens": list(tts_b["tokens"]),
            "natural_tokens_sha256": canonical_sha256(natural_b["tokens"]),
            "tts_tokens_sha256": canonical_sha256(tts_b["tokens"]),
            "historical_feature_path": str(feature_path),
            "historical_feature_sha256": file_sha256(feature_path),
            "historical_conditioning_shape": list(conditioning.shape),
            "historical_conditioning_sha256": str(feature_row.get("conditioning_sha256")),
            "a_natural_source": str(a_natural),
            "a_tts_source": str(a_tts),
            "a_tts_source_sha256": file_sha256(a_tts),
            "c_tts_audio_sha256": file_sha256(tts_audio),
        })
    if len(rows) != 15 or [row["paired_key"] for row in rows] != ordered:
        raise ProtocolError("frozen sample order is not preserved")
    return rows


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    min_common_windows = _effective_min_common_windows(args)
    a_path = args.a_summary.resolve()
    b_path = args.b_alignment.resolve()
    c_path = args.c_manifest.resolve()
    wav2lip = args.wav2lip.resolve()
    syncnet = args.syncnet.resolve()
    wav2lip_checkpoint = _require_file(args.wav2lip_checkpoint.resolve(), "Wav2Lip checkpoint")
    syncnet_model = _require_file(args.syncnet_model.resolve(), "SyncNet checkpoint")
    # Do not call Path.resolve() on these environment launchers: the venv
    # entry point is a symlink whose target lacks the venv site-packages.
    wav2lip_python = _require_file(args.wav2lip_python.expanduser(), "Wav2Lip Python")
    a, b, c = read_json(a_path), read_json(b_path), read_json(c_path)
    records = _selected_records(a, b, c, a_path=a_path, b_path=b_path, c_path=c_path, wav2lip_checkpoint=wav2lip_checkpoint, syncnet_model=syncnet_model, wav2lip_python=wav2lip_python)

    import cv2

    frame_dir = output / "inputs" / "face_frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    requests = []
    for row in records:
        face = Path(row["face_video"])
        info = _video_info(face)
        frame_path = frame_dir / f"{row['sample_id']:04d}.png"
        if not frame_path.is_file() and not cv2.imwrite(str(frame_path), info["first"]):
            raise ProtocolError(f"cannot save face frame 0: {frame_path}")
        requests.append({"sample_id": str(row["sample_id"]), "image": str(frame_path), "image_sha256": bytes_sha256(cv2.cvtColor(info["first"], cv2.COLOR_BGR2RGB).tobytes())})
    request_path = output / "inputs" / "face_detection_requests.json"
    detection_path = output / "inputs" / "face_detections.json"
    write_json(request_path, requests)
    detect_worker = REPO / "scripts/experiments/static_image_bridge/detect_worker.py"
    if not detection_path.is_file():
        log = output / "logs" / "prepare_face_detection.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("w", encoding="utf-8") as handle:
            result = subprocess.run([str(wav2lip_python), str(detect_worker), "--requests", str(request_path), "--output", str(detection_path), "--threshold", str(FACE_DET_THRESHOLD)], cwd=str(REPO), stdout=handle, stderr=subprocess.STDOUT, check=False)
        if result.returncode != 0:
            raise ProtocolError(f"face detection failed; see {log}")
    detections = read_json(detection_path)
    detection_by_id = {str(row["sample_id"]): row for row in detections.get("records", [])}
    if len(detection_by_id) != 15:
        raise ProtocolError("face detection output does not contain 15 records")
    prepared: list[dict[str, Any]] = []
    for row in records:
        det = detection_by_id[str(row["sample_id"])]
        selected = det.get("selected")
        if not isinstance(selected, Mapping):
            eligible = [value for value in det.get("detections", []) if float(value.get("score", 0.0)) >= FACE_DET_THRESHOLD]
            eligible.sort(key=lambda value: (-float(value["score"]), float(value["x1"]), float(value["y1"])))
            if not eligible:
                raise ProtocolError(f"no eligible face: {row['sample_id']}")
            selected = eligible[0]
        score_box = _score_box(selected)
        coverage = _validate_crop_coverage(Path(row["face_video"]), score_box)
        copy = dict(row)
        copy["score_box"] = score_box
        copy["score_box_sha256"] = canonical_sha256(score_box)
        copy["face_geometry_qc"] = coverage
        score_box_path = output / "inputs" / f"score_box_{int(row['sample_id']):04d}.json"
        write_json(score_box_path, {"sample_id": int(row["sample_id"]), "paired_key": row["paired_key"], **score_box})
        copy["score_box_path"] = str(score_box_path)
        prepared.append(copy)
    model = a.get("model", {})
    inputs = {
        "schema_version": 1,
        "status": "complete",
        "protocol": "mfa_linear_trajectory_ablation_v1",
        "run_id": RUN_ID,
        "sample_count": 15,
        "active_sample_count": 15,
        "arms": list(ARMS),
        "resynthesis_arms": list(RESYNTH_ARMS),
        "keep_by_arm": KEEP_BY_ARM,
        "sample_rate": SAMPLE_RATE,
        "frame_stride_samples": FRAME_STRIDE_SAMPLES,
        "feature_dim": FEATURE_DIM,
        "frame_center_rule": "(index+0.5)*320/16000",
        "face_det_threshold": FACE_DET_THRESHOLD,
        "score_lag_range": list(range(-VSHIFT, VSHIFT + 1)),
        "min_common_windows": min_common_windows,
        "sources": {
            "a_summary": str(a_path), "a_summary_sha256": file_sha256(a_path),
            "b_alignment": str(b_path), "b_alignment_sha256": file_sha256(b_path),
            "c_manifest": str(c_path), "c_manifest_sha256": file_sha256(c_path),
        },
        "models": {
            "knn_vc_repository": model.get("repository"), "knn_vc_revision": model.get("revision"),
            "wavlm_checkpoint_sha256": model.get("wavlm_checkpoint_sha256"), "vocoder_checkpoint_sha256": model.get("vocoder_checkpoint_sha256"),
            "wav2lip_checkpoint": str(wav2lip_checkpoint), "wav2lip_checkpoint_sha256": file_sha256(wav2lip_checkpoint),
            "syncnet_model": str(syncnet_model), "syncnet_model_sha256": file_sha256(syncnet_model),
            "wav2lip_python": str(wav2lip_python), "syncnet_python": str(args.syncnet_python),
        },
        "paths": {"repo": str(REPO), "wav2lip": str(wav2lip), "syncnet": str(syncnet), "knn_vc_source": str(args.knn_source.resolve())},
        "runtime": {"python": platform.python_version(), "git_commit": _git_commit(), "argv": sys.argv},
        "records": prepared,
    }
    write_json(output / "inputs.json", inputs)
    return inputs


def _validate_reuse_source(source: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, str]]:
    """Validate every read-only artifact that a support30 run will reuse."""
    def source_file(value: str | Path, label: str) -> Path:
        raw = Path(str(value)).expanduser()
        if not raw.is_absolute():
            raw = source / raw
        return _require_file(raw, label)

    inputs_path = _require_file(source / "inputs.json", "source inputs.json")
    audio_path = _require_file(source / "audio_manifest.json", "source audio_manifest.json")
    videos_path = _require_file(source / "videos_manifest.json", "source videos_manifest.json")
    source_inputs = read_json(inputs_path)
    audio_manifest = read_json(audio_path)
    videos_manifest = read_json(videos_path)
    if source_inputs.get("status") != "complete" or source_inputs.get("protocol") != "mfa_linear_trajectory_ablation_v1":
        raise ProtocolError("reuse source must be a complete MFA-linear v1 run")
    if _validate_min_common_windows(source_inputs.get("min_common_windows")) != MIN_COMMON_WINDOWS:
        raise ProtocolError("reuse source must use the original 50-window threshold")
    if list(source_inputs.get("arms", [])) != list(ARMS) or len(source_inputs.get("records", [])) != 15:
        raise ProtocolError("reuse source does not contain the frozen 15 x 8 cohort")
    if audio_manifest.get("status") != "complete" or videos_manifest.get("status") != "complete":
        raise ProtocolError("reuse source audio/videos manifests must be complete")
    if int(audio_manifest.get("audio_rows", -1)) != 120 or int(videos_manifest.get("expected_videos", -1)) != 120:
        raise ProtocolError("reuse source must contain exactly 120 audio and 120 video artifacts")

    for record in source_inputs["records"]:
        for field, label in (("natural_audio", "natural audio"), ("tts_audio", "TTS audio"), ("face_video", "face video")):
            path = source_file(record[field], f"source {label}")
            if file_sha256(path) != str(record[f"{field}_sha256"]):
                raise ProtocolError(f"source {label} hash changed: {path}")
        score_box = record.get("score_box")
        if not isinstance(score_box, Mapping) or canonical_sha256(score_box) != str(record.get("score_box_sha256")):
            raise ProtocolError(f"source score box hash mismatch: sample {record.get('sample_id')}")
        score_box_path = source_file(record["score_box_path"], "source score box")
        stored_box = read_json(score_box_path)
        if any(stored_box.get(key) != score_box.get(key) for key in ("order", "center", "side", "box", "scale_from_detection")):
            raise ProtocolError(f"source score box contents changed: {score_box_path}")
        feature_path = source_file(record["historical_feature_path"], "source historical feature")
        if file_sha256(feature_path) != str(record["historical_feature_sha256"]):
            raise ProtocolError(f"source feature hash changed: {feature_path}")

    audio_rows = [row for record in audio_manifest["records"] for row in record.get("audio", [])]
    if len(audio_rows) != 120:
        raise ProtocolError(f"source audio manifest has {len(audio_rows)} rows, expected 120")
    expected_artifact_keys = {(str(record["paired_key"]), arm) for record in source_inputs["records"] for arm in ARMS}
    actual_audio_keys = {(str(row["paired_key"]), str(row["arm"])) for row in audio_rows}
    if actual_audio_keys != expected_artifact_keys:
        raise ProtocolError("source audio manifest keys do not match the frozen 15 x 8 cohort")
    for row in audio_rows:
        path = source_file(row["audio_pcm16"], "source PCM audio")
        if file_sha256(path) != str(row["audio_sha256"]):
            raise ProtocolError(f"source audio hash changed: {path}")
        condition_path = row.get("conditioning_path")
        condition_hash = row.get("conditioning_sha256")
        if condition_path and condition_hash:
            condition = source_file(condition_path, "source conditioning")
            if file_sha256(condition) != str(condition_hash):
                raise ProtocolError(f"source conditioning hash changed: {condition}")

    video_rows = list(videos_manifest["videos"])
    if len(video_rows) != 120:
        raise ProtocolError(f"source video manifest has {len(video_rows)} rows, expected 120")
    actual_video_keys = {(str(row["paired_key"]), str(row["arm"])) for row in video_rows}
    if actual_video_keys != expected_artifact_keys:
        raise ProtocolError("source video manifest keys do not match the frozen 15 x 8 cohort")
    for row in video_rows:
        video = source_file(row["video"], "source generated video")
        if file_sha256(video) != str(row["video_sha256"]):
            raise ProtocolError(f"source video hash changed: {video}")
        face = source_file(row["face"], "source face video")
        if file_sha256(face) != str(row["face_sha256"]):
            raise ProtocolError(f"source face video hash changed: {face}")

    for model_field, hash_field, label in (("syncnet_model", "syncnet_model_sha256", "SyncNet model"), ("wav2lip_checkpoint", "wav2lip_checkpoint_sha256", "Wav2Lip checkpoint")):
        model_path = source_file(source_inputs["models"][model_field], f"source {label}")
        if file_sha256(model_path) != str(source_inputs["models"][hash_field]):
            raise ProtocolError(f"source {label} hash changed: {model_path}")

    hashes = {
        "inputs": file_sha256(inputs_path),
        "audio_manifest": file_sha256(audio_path),
        "videos_manifest": file_sha256(videos_path),
    }
    return source_inputs, audio_manifest, videos_manifest, hashes


def prepare_reuse(args: argparse.Namespace) -> dict[str, Any]:
    """Create a new support30 run that references a validated v1 run."""
    output = args.output_dir.resolve()
    source = args.source_run.resolve()
    if output == source:
        raise ProtocolError("reuse output-dir must differ from source-run")
    min_common_windows = _effective_min_common_windows(args)
    if min_common_windows != 30:
        raise ProtocolError("support30 reuse requires --min-common-windows 30")
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ProtocolError(f"existing output requires --resume: {output}")
    output.mkdir(parents=True, exist_ok=True)
    source_inputs, audio_manifest, videos_manifest, source_hashes = _validate_reuse_source(source)

    # The manifests are copied as metadata only; every media/feature path in
    # them remains an absolute path into the validated parent run.
    shutil.copy2(source / "audio_manifest.json", output / "audio_manifest.json")
    shutil.copy2(source / "videos_manifest.json", output / "videos_manifest.json")
    inputs = json.loads(json.dumps(source_inputs, ensure_ascii=False))
    inputs["protocol"] = V2_PROTOCOL
    inputs["run_id"] = output.name
    inputs["min_common_windows"] = min_common_windows
    inputs["parent_run"] = str(source)
    inputs["parent_protocol"] = str(source_inputs.get("protocol"))
    inputs["parent"] = {
        "run": str(source),
        "protocol": str(source_inputs.get("protocol")),
        "inputs_sha256": source_hashes["inputs"],
        "audio_manifest_sha256": source_hashes["audio_manifest"],
        "videos_manifest_sha256": source_hashes["videos_manifest"],
    }
    inputs["reuse"] = {
        "mode": "read_only_parent_artifacts",
        "parent_run": str(source),
        "media_and_features_are_reused": True,
        "score_boxes_are_reused": True,
        "new_score_and_analysis_outputs": True,
    }
    inputs["runtime"] = dict(inputs.get("runtime", {}))
    inputs["runtime"].update({"python": platform.python_version(), "git_commit": _git_commit(), "argv": sys.argv})
    write_json(output / "inputs.json", inputs)
    write_json(output / "reuse_manifest.json", {
        "schema_version": 1,
        "status": "complete",
        "protocol": V2_PROTOCOL,
        "min_common_windows": min_common_windows,
        "parent_run": str(source),
        "parent_protocol": str(source_inputs.get("protocol")),
        "parent_inputs_sha256": source_hashes["inputs"],
        "parent_audio_manifest_sha256": source_hashes["audio_manifest"],
        "parent_videos_manifest_sha256": source_hashes["videos_manifest"],
        "sample_count": 15,
        "audio_rows": int(audio_manifest["audio_rows"]),
        "video_rows": len(videos_manifest["videos"]),
        "code_version": _git_commit(),
    })
    return inputs


def _git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def _records_for_stage(inputs: Mapping[str, Any], smoke: bool) -> list[Mapping[str, Any]]:
    rows = list(inputs.get("records", []))
    if len(rows) != 15:
        raise ProtocolError("inputs.json must contain all 15 frozen records")
    return rows[:1] if smoke else rows


def _save_pcm_audio(float_path: Path, pcm_path: Path, values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    qc = audio_qc(values, natural_count=values.size)
    if not qc["finite"] or qc["out_of_range_sample_count"]:
        raise ProtocolError(f"generated waveform is non-finite or outside [-1,1]: {float_path}")
    float_path.parent.mkdir(parents=True, exist_ok=True)
    pcm_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(float_path), values, SAMPLE_RATE, subtype="FLOAT")
    sf.write(str(pcm_path), values, SAMPLE_RATE, subtype="PCM_16")
    decoded, rate = _audio(pcm_path)
    if rate != SAMPLE_RATE:
        raise ProtocolError(f"PCM16 output rate changed: {pcm_path}")
    qc["pcm16_sample_count"] = int(decoded.size)
    qc["pcm16_peak"] = float(np.abs(decoded).max()) if decoded.size else 0.0
    qc["pcm16_rms"] = float(np.sqrt(np.mean(np.square(decoded)))) if decoded.size else 0.0
    return qc


def _load_historical_conditioning(row: Mapping[str, Any]) -> Any:
    import torch

    path = Path(str(row["historical_feature_path"]))
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or "conditioning" not in payload:
        raise ProtocolError(f"historical feature file has no conditioning: {path}")
    conditioning = torch.as_tensor(payload["conditioning"], dtype=torch.float32)
    if list(conditioning.shape) != list(row["historical_conditioning_shape"]):
        raise ProtocolError(f"historical conditioning shape changed: {path}")
    if file_sha256(path) != row["historical_feature_sha256"]:
        raise ProtocolError(f"historical feature hash changed: {path}")
    return conditioning


def audio_stage(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    output = args.output_dir.resolve()
    inputs = read_json(output / "inputs.json")
    rows = _records_for_stage(inputs, args.smoke)
    from wavlm_knn_vc_adapter import KNN_VC_REVISION, WavLMKNNVCAdapter

    if inputs["models"].get("knn_vc_revision") != KNN_VC_REVISION:
        raise ProtocolError("inputs.json kNN-VC revision is not the pinned revision")
    adapter = WavLMKNNVCAdapter.load_pretrained(device=args.device, source=args.knn_source.resolve(), revision=KNN_VC_REVISION)
    metadata = adapter.metadata()
    for key in ("wavlm_checkpoint_sha256", "vocoder_checkpoint_sha256"):
        expected = inputs["models"].get(key)
        if expected and metadata.get(key) != expected:
            raise ProtocolError(f"{key} differs from A model hash: {metadata.get(key)} != {expected}")

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        sid = int(row["sample_id"])
        key = str(row["paired_key"])
        try:
            natural, _ = _audio(Path(row["natural_audio"]))
            tts, _ = _audio(Path(row["tts_audio"]))
            if file_sha256(Path(row["natural_audio"])) != row["natural_audio_sha256"] or file_sha256(Path(row["tts_audio"])) != row["tts_audio_sha256"]:
                raise ProtocolError("raw source hash changed")
            z_n = adapter.extract(torch.from_numpy(natural).unsqueeze(0)).cpu()
            z_t = _load_historical_conditioning(row)
            if z_n.ndim != 2 or z_n.shape[1] != FEATURE_DIM or z_t.shape != z_n.shape:
                raise ProtocolError(f"Z_N/Z_T shape mismatch: {tuple(z_n.shape)} vs {tuple(z_t.shape)}")
            expected_shape = list(row.get("historical_conditioning_shape", []))
            if expected_shape and expected_shape != list(z_n.shape):
                raise ProtocolError(f"Z_N shape differs from historical natural grid: {tuple(z_n.shape)} vs {expected_shape}")
            occurrences, mask_meta = _occurrence_mask(row["natural_tokens"], row["tts_tokens"], int(z_n.shape[0]))
            feature_dir = output / "features"
            feature_dir.mkdir(parents=True, exist_ok=True)
            z_n_path = feature_dir / f"{key}__Z_N.pt"
            z_t_path = feature_dir / f"{key}__Z_T.pt"
            torch.save({"features": z_n, "source": "N_RAW", "sample_id": sid, "paired_key": key}, z_n_path)
            torch.save({"features": z_t, "source": "T_100 historical conditioning", "sample_id": sid, "paired_key": key}, z_t_path)
            mask_path = feature_dir / f"{key}__occurrences.json"
            write_json(mask_path, {"sample_id": sid, "paired_key": key, "occurrences": occurrences, "metadata": mask_meta})
            conditions: dict[str, Any] = {}
            audio_rows: list[dict[str, Any]] = []
            raw_specs = (("N_RAW", natural, Path(row["natural_audio"]), None), ("T_RAW", tts, Path(row["tts_audio"]), None))
            for arm, values, source_path_raw, _ in raw_specs:
                pcm_path = output / "audio" / f"{key}__{arm}.wav"
                pcm_path.parent.mkdir(parents=True, exist_ok=True)
                if not pcm_path.is_file():
                    shutil.copy2(source_path_raw, pcm_path)
                elif file_sha256(pcm_path) != file_sha256(source_path_raw):
                    raise ProtocolError(f"existing raw audio copy has the wrong identity: {pcm_path}")
                decoded, rate = _audio(pcm_path)
                if rate != SAMPLE_RATE:
                    raise ProtocolError(f"raw audio rate changed: {pcm_path}")
                qc = audio_qc(decoded, natural_count=int(natural.size), raw_decoder_count=int(decoded.size), action="source_copy")
                audio_rows.append({"sample_id": sid, "paired_key": key, "arm": arm, "audio_float": str(pcm_path), "audio_pcm16": str(pcm_path), "audio_sha256": file_sha256(pcm_path), "source_sha256": file_sha256(source_path_raw), "conditioning_path": None, "occurrence_count": mask_meta["occurrence_count"], "eligible_occurrence_count": mask_meta["eligible_occurrence_count"], "eligible_frame_fraction": mask_meta["eligible_frame_fraction"], "matched_coverage": mask_meta["matched_coverage"], "fallback_frame_count": mask_meta["fallback_frame_count"], "internal_boundary_before": {"N": trajectory_diagnostics(z_n, occurrences), "T": trajectory_diagnostics(z_t, occurrences)}, "internal_boundary_after": {"source": "raw", "diagnostics": None}, "length_adjustment": {"action": "source_copy", "raw_decoder_sample_count": int(decoded.size), "target_natural_sample_count": int(natural.size), "adjustment_sample_count": abs(int(decoded.size) - int(natural.size))}, "qc": qc})
            for arm in RESYNTH_ARMS:
                source = z_n if SOURCE_BY_ARM[arm] == "N" else z_t
                conditioning = ablate_phone_dynamics(source, occurrences, KEEP_BY_ARM[arm])
                if KEEP_BY_ARM[arm] == 1.0:
                    # Keep this arm byte-for-byte tied to its source tensor.
                    conditioning = source.clone()
                conditioning_path = feature_dir / f"{key}__{arm}__conditioning.pt"
                torch.save({"conditioning": conditioning.cpu(), "source": SOURCE_BY_ARM[arm], "keep": KEEP_BY_ARM[arm], "occurrences_path": str(mask_path)}, conditioning_path)
                raw_decoded = adapter.vocode(conditioning).cpu().numpy()
                output_values, adjustment = exact_natural_length(raw_decoded, int(natural.size))
                float_path = output / "audio" / f"{key}__{arm}__float.wav"
                pcm_path = output / "audio" / f"{key}__{arm}.wav"
                qc = _save_pcm_audio(float_path, pcm_path, output_values)
                qc.update({"raw_decoder_sample_count": adjustment["raw_decoder_sample_count"], "length_adjustment_action": adjustment["action"], "length_adjustment_sample_count": adjustment["adjustment_sample_count"], "natural_sample_count": int(natural.size), "exact_natural_sample_count": True})
                audio_rows.append({"sample_id": sid, "paired_key": key, "arm": arm, "audio_float": str(float_path), "audio_pcm16": str(pcm_path), "audio_sha256": file_sha256(pcm_path), "source_sha256": None, "conditioning_path": str(conditioning_path), "conditioning_sha256": file_sha256(conditioning_path), "source": SOURCE_BY_ARM[arm], "keep": KEEP_BY_ARM[arm], "occurrence_count": mask_meta["occurrence_count"], "eligible_occurrence_count": mask_meta["eligible_occurrence_count"], "eligible_frame_fraction": mask_meta["eligible_frame_fraction"], "matched_coverage": mask_meta["matched_coverage"], "fallback_frame_count": mask_meta["fallback_frame_count"], "internal_boundary_before": {"N": trajectory_diagnostics(z_n, occurrences), "T": trajectory_diagnostics(z_t, occurrences)}, "internal_boundary_after": {"source": SOURCE_BY_ARM[arm], "diagnostics": trajectory_diagnostics(conditioning, occurrences)}, "length_adjustment": adjustment, "qc": qc})
                conditions[arm] = {"conditioning_path": str(conditioning_path), "conditioning_sha256": file_sha256(conditioning_path), "source": SOURCE_BY_ARM[arm], "keep": KEEP_BY_ARM[arm], "shape": list(conditioning.shape)}
            records.append({"sample_id": sid, "paired_key": key, "z_n_path": str(z_n_path), "z_t_path": str(z_t_path), "occurrences_path": str(mask_path), "z_n_shape": list(z_n.shape), "z_t_shape": list(z_t.shape), "mask": mask_meta, "conditions": conditions, "audio": audio_rows})
            print(f"audio {index}/{len(rows)} complete sample={sid}", flush=True)
        except Exception as exc:  # noqa: BLE001 - retain per-utterance failure details
            failures.append({"sample_id": sid, "paired_key": key, "stage": "audio", "error": str(exc)})
            print(f"audio {index}/{len(rows)} FAILED sample={sid}: {exc}", flush=True)
    all_audio = [audio for record in records for audio in record["audio"]]
    status = "complete" if not failures and len(records) == len(rows) and len(all_audio) == len(rows) * len(ARMS) else "incomplete"
    manifest = {"schema_version": 1, "status": status, "sample_count": len(records), "expected_sample_count": len(rows), "expected_audio_rows": len(rows) * len(ARMS), "audio_rows": len(all_audio), "arms": list(ARMS), "records": records, "failures": failures, "model": metadata, "heldout_excluded": True}
    write_json(output / "audio_manifest.json", manifest)
    _write_audio_qc_csv(output / "audio_qc.csv", all_audio)
    if status != "complete":
        raise ProtocolError(f"audio stage incomplete: {failures}")
    return manifest


def _write_audio_qc_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fieldnames = ["sample_id", "paired_key", "arm", "audio_float", "audio_pcm16", "audio_sha256", "source_sha256", "conditioning_path", "conditioning_sha256", "source", "keep", "occurrence_count", "eligible_occurrence_count", "eligible_frame_fraction", "matched_coverage", "fallback_frame_count", "sample_count", "natural_sample_count", "duration_s", "sample_rate", "channels", "finite", "nonfinite_sample_count", "out_of_range_sample_count", "peak", "rms", "exact_natural_sample_count", "raw_decoder_sample_count", "length_adjustment_action", "length_adjustment_sample_count"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            qc = dict(row.get("qc", {}))
            writer.writerow({key: row.get(key, qc.get(key, "")) for key in fieldnames})


def render_stage(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    inputs = read_json(output / "inputs.json")
    audio_manifest = read_json(output / "audio_manifest.json")
    if audio_manifest.get("status") != "complete":
        raise ProtocolError("render requires a complete audio_manifest.json")
    rows = _records_for_stage(inputs, args.smoke)
    audio_by_key_arm = {(str(row["paired_key"]), str(row["arm"])): row for record in audio_manifest["records"] for row in record["audio"]}
    wav2lip_python = _require_file(args.wav2lip_python.expanduser(), "Wav2Lip Python")
    inference = _require_file(args.wav2lip / "inference.py", "Wav2Lip inference.py")
    checkpoint = _require_file(args.wav2lip_checkpoint.resolve(), "Wav2Lip checkpoint")
    existing_manifest_path = output / "videos_manifest.json"
    existing_manifest = read_json(existing_manifest_path) if existing_manifest_path.is_file() else {}
    existing_by_key_arm = {(str(value.get("paired_key")), str(value.get("arm"))): value for value in existing_manifest.get("videos", [])}
    videos: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for index, row in enumerate(rows, 1):
        key = str(row["paired_key"])
        face = Path(row["face_video"])
        for arm in ARMS:
            try:
                audio = audio_by_key_arm[(key, arm)]
                video = output / "videos" / arm / f"{int(row['sample_id']):04d}.mp4"
                work = output / "render_work" / arm / str(row["sample_id"])
                # Wav2Lip writes the intermediate ``temp/result.avi`` using a
                # path relative to cwd; each cell therefore gets its own temp
                # directory to prevent cross-arm contamination.
                (work / "temp").mkdir(parents=True, exist_ok=True)
                video.parent.mkdir(parents=True, exist_ok=True)
                log = output / "logs" / "render" / arm / f"{int(row['sample_id']):04d}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                existing = existing_by_key_arm.get((key, arm), {})
                reusable = False
                if video.is_file():
                    try:
                        video_sha256 = file_sha256(video)
                        existing_identity = (
                            existing.get("video") == str(video)
                            and existing.get("audio_sha256") == audio["audio_sha256"]
                            and existing.get("face_sha256") == row["face_video_sha256"]
                            and existing.get("wav2lip_checkpoint_sha256") == file_sha256(checkpoint)
                            and existing.get("video_sha256") == video_sha256
                        )
                        # A previous render can be interrupted before its manifest is
                        # written.  The output path is deterministic; validate such a
                        # file directly so a resumed run does not discard completed
                        # cells.  The current audio/face/checkpoint hashes are still
                        # recorded in the new manifest below.
                        interrupted_run_output = not existing
                        if existing_identity or interrupted_run_output:
                            info = _video_info(video)
                            reusable = info["frame_count"] >= 25 and abs(info["fps"] - FPS) <= 0.01
                    except Exception:  # noqa: BLE001 - invalid partial output is rerendered
                        reusable = False
                if not reusable:
                    command = [str(wav2lip_python), str(inference), "--checkpoint_path", str(checkpoint), "--face", str(face), "--audio", str(audio["audio_pcm16"]), "--outfile", str(video), "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--nosmooth"]
                    with log.open("w", encoding="utf-8") as handle:
                        result = subprocess.run(command, cwd=str(work), stdout=handle, stderr=subprocess.STDOUT, check=False)
                    if result.returncode != 0:
                        raise ProtocolError(f"Wav2Lip returncode={result.returncode}; see {log}")
                info = _video_info(video)
                videos.append({"sample_id": int(row["sample_id"]), "paired_key": key, "arm": arm, "video": str(video), "video_sha256": file_sha256(video), "face": str(face), "face_sha256": row["face_video_sha256"], "audio": audio["audio_pcm16"], "audio_sha256": audio["audio_sha256"], "fps": info["fps"], "frame_count": info["frame_count"], "wav2lip_checkpoint_sha256": file_sha256(checkpoint), "parameters": {"face_det_batch_size": 4, "wav2lip_batch_size": 4, "nosmooth": True, "cwd": str(work)}})
            except Exception as exc:  # noqa: BLE001 - retain per-cell failure details
                failures.append({"sample_id": int(row["sample_id"]), "paired_key": key, "arm": arm, "stage": "render", "error": str(exc)})
                print(f"render FAILED sample={row['sample_id']} arm={arm}: {exc}", flush=True)
        print(f"render {index}/{len(rows)} sample={row['sample_id']}", flush=True)
    expected = len(rows) * len(ARMS)
    status = "complete" if not failures and len(videos) == expected else "incomplete"
    manifest = {"schema_version": 1, "status": status, "sample_count": len(rows), "expected_videos": expected, "videos": videos, "failures": failures, "wav2lip_checkpoint_sha256": file_sha256(checkpoint), "heldout_excluded": True}
    write_json(output / "videos_manifest.json", manifest)
    if status != "complete":
        raise ProtocolError(f"render stage incomplete: {failures}")
    return manifest


def _crop_video(path: Path, score_box: Mapping[str, Any]) -> np.ndarray:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open generated video: {path}")
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(_crop_zero_padded(frame, score_box))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if len(frames) < 25 or abs(fps - FPS) > 0.01:
        raise ProtocolError(f"video support invalid for scoring: {path}, frames={len(frames)}, fps={fps}")
    return np.stack(frames, axis=0)


def _visual_embedding(cropped: np.ndarray, scorer: Any, device: str, torch: Any, batch_size: int = 20) -> tuple[np.ndarray, int]:
    values = []
    count = int(cropped.shape[0] - 4)
    for start in range(0, count, batch_size):
        stop = min(count, start + batch_size)
        sequences = np.stack([cropped[index : index + 5] for index in range(start, stop)], axis=0)
        tensor = torch.from_numpy(np.transpose(sequences, (0, 4, 1, 2, 3))).float().to(device)
        with torch.no_grad():
            values.append(scorer.__S__.forward_lip(tensor).detach().cpu())
    if not values:
        raise ProtocolError("video has no SyncNet visual windows")
    return torch.cat(values, dim=0).numpy().astype(np.float32), count


def _audio_embedding(path: Path, scorer: Any, device: str, torch: Any, batch_size: int = 20) -> tuple[np.ndarray, int, int]:
    import python_speech_features
    from scipy.io import wavfile

    rate, values = wavfile.read(str(path))
    values = np.asarray(values)
    if int(rate) != SAMPLE_RATE or values.ndim != 1:
        raise ProtocolError(f"scoring audio must be mono 16k PCM: {path}")
    mfcc = np.asarray(list(zip(*python_speech_features.mfcc(values, rate))), dtype=np.float32)
    if mfcc.ndim != 2 or mfcc.shape[0] != 13:
        raise ProtocolError(f"unexpected MFCC shape: {path} {mfcc.shape}")
    count = (int(mfcc.shape[1]) - 20) // 4 + 1
    if count <= 0:
        raise ProtocolError(f"audio has insufficient SyncNet support: {path}")
    batches = []
    for start in range(0, count, batch_size):
        indices = range(start, min(count, start + batch_size))
        values_batch = [mfcc[:, index * 4 : index * 4 + 20] for index in indices]
        tensor = torch.from_numpy(np.asarray(values_batch, dtype=np.float32)[:, None, :, :]).to(device)
        with torch.no_grad():
            batches.append(scorer.__S__.forward_aud(tensor).detach().cpu())
    return torch.cat(batches, dim=0).numpy().astype(np.float32), count, int(values.size)


def _curve(visual: np.ndarray, audio: np.ndarray, t_indices: Sequence[int], *, vshift: int = VSHIFT, min_common_windows: int = MIN_COMMON_WINDOWS) -> tuple[list[int], list[float], int]:
    import torch

    threshold = _validate_min_common_windows(min_common_windows)
    if len(t_indices) < threshold:
        raise ProtocolError(f"common SyncNet support has only {len(t_indices)} windows (< {threshold})")
    lags = list(range(-vshift, vshift + 1))
    values: list[float] = []
    for lag in lags:
        left = torch.from_numpy(np.asarray([visual[t] for t in t_indices], dtype=np.float32))
        right = torch.from_numpy(np.asarray([audio[t + lag] for t in t_indices], dtype=np.float32))
        distances = torch.nn.functional.pairwise_distance(left, right).numpy()
        values.append(float(np.mean(distances)))
    return lags, values, len(t_indices)


def _curve_metrics(lags: Sequence[int], values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != (2 * VSHIFT + 1,) or not np.isfinite(arr).all():
        raise ProtocolError("SyncNet curve must contain 31 finite values")
    minimum = float(np.min(arr))
    index = int(np.flatnonzero(arr == minimum)[0])
    return {"C": float(np.median(arr) - minimum), "D": minimum, "curve_median": float(np.median(arr)), "k_star": int(lags[index]), "d_zero": float(arr[VSHIFT])}


def score_worker(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if not torch.cuda.is_available():
        raise ProtocolError("SyncNet scoring requires CUDA")
    syncnet_root = args.syncnet.resolve()
    if str(syncnet_root) not in sys.path:
        sys.path.insert(0, str(syncnet_root))
    from SyncNetInstance import SyncNetInstance

    output = args.output_dir.resolve()
    inputs = read_json(output / "inputs.json")
    min_common_windows = _effective_min_common_windows(args, inputs)
    audio_manifest = read_json(output / "audio_manifest.json")
    videos_manifest = read_json(output / "videos_manifest.json")
    if audio_manifest.get("status") != "complete" or videos_manifest.get("status") != "complete":
        raise ProtocolError("score requires complete audio and video manifests")
    rows = _records_for_stage(inputs, args.smoke)
    by_audio = {(str(r["paired_key"]), str(r["arm"])): r for record in audio_manifest["records"] for r in record["audio"]}
    by_video = {(str(r["paired_key"]), str(r["arm"])): r for r in videos_manifest["videos"]}
    checkpoint = _require_file(Path(inputs["models"]["syncnet_model"]), "SyncNet checkpoint")
    if file_sha256(checkpoint) != inputs["models"]["syncnet_model_sha256"]:
        raise ProtocolError("SyncNet model hash changed")
    scorer = SyncNetInstance(device="cuda")
    scorer.loadParameters(str(checkpoint))
    scorer.eval()
    scores: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in rows:
        key = str(row["paired_key"])
        sid = int(row["sample_id"])
        try:
            visual: dict[str, np.ndarray] = {}
            v_counts: dict[str, int] = {}
            audio_features: dict[str, np.ndarray] = {}
            a_counts: dict[str, int] = {}
            for arm in ARMS:
                video_row = by_video[(key, arm)]
                audio_row = by_audio[(key, arm)]
                visual[arm], v_counts[arm] = _visual_embedding(_crop_video(Path(video_row["video"]), row["score_box"]), scorer, "cuda", torch)
                audio_features[arm], a_counts[arm], _ = _audio_embedding(Path(audio_row["audio_pcm16"]), scorer, "cuda", torch)
            common_f = min(min(v_counts.values()), min(a_counts[arm] for arm in ARMS if arm != "T_RAW"))
            common_t = list(range(VSHIFT, common_f - VSHIFT))
            if len(common_t) < min_common_windows:
                raise ProtocolError(
                    f"common support {len(common_t)} < {min_common_windows} "
                    f"(F={common_f}, visual_min={min(v_counts.values())}, "
                    f"audio_min_excluding_T_RAW={min(a_counts[arm] for arm in ARMS if arm != 'T_RAW')})"
                )
            t_raw_f = min(v_counts["T_RAW"], a_counts["T_RAW"])
            t_raw_t = list(range(VSHIFT, t_raw_f - VSHIFT))
            if len(t_raw_t) < min_common_windows:
                raise ProtocolError(f"T_RAW support {len(t_raw_t)} < {min_common_windows}")
            candidate_cells = [(arm, arm) for arm in ARMS] + [(arm, "N_RAW") for arm in RESYNTH_ARMS] + [("N_RAW", "T_100")]
            pending: list[dict[str, Any]] = []
            for video_arm, audio_arm in candidate_cells:
                t_indices = t_raw_t if audio_arm == "T_RAW" else common_t
                lags, values, support = _curve(visual[video_arm], audio_features[audio_arm], t_indices, min_common_windows=min_common_windows)
                metrics = _curve_metrics(lags, values)
                fixed = None
                if audio_arm == "N_RAW":
                    pending.append({"video_arm": video_arm, "audio_arm": audio_arm, "lags": lags, "values": values, "support": support, "metrics": metrics, "fixed": fixed})
                else:
                    pending.append({"video_arm": video_arm, "audio_arm": audio_arm, "lags": lags, "values": values, "support": support, "metrics": metrics, "fixed": fixed})
            natural_native = next(item for item in pending if item["video_arm"] == "N_RAW" and item["audio_arm"] == "N_RAW")
            k_n = int(natural_native["metrics"]["k_star"])
            k_index = k_n + VSHIFT
            for item in pending:
                metrics = item["metrics"]
                fixed_value = float(item["values"][k_index]) if item["audio_arm"] == "N_RAW" else None
                video_row = by_video[(key, item["video_arm"])]
                audio_row = by_audio[(key, item["audio_arm"])]
                score_row = {"schema_version": 1, "status": "complete", "sample_id": sid, "paired_key": key, "video_arm": item["video_arm"], "audio_arm": item["audio_arm"], "C": metrics["C"], "D": metrics["D"], "curve_median": metrics["curve_median"], "k_star": metrics["k_star"], "d_zero": metrics["d_zero"], "common_support_count": item["support"], "fixed_k_n": k_n if item["audio_arm"] == "N_RAW" else None, "d_fixed_natural_lag": fixed_value, "video": video_row["video"], "video_sha256": video_row["video_sha256"], "audio": audio_row["audio_pcm16"], "audio_sha256": audio_row["audio_sha256"], "score_box_sha256": row["score_box_sha256"], "syncnet_model_sha256": inputs["models"]["syncnet_model_sha256"]}
                scores.append(score_row)
                curves.append({"sample_id": sid, "paired_key": key, "video_arm": item["video_arm"], "audio_arm": item["audio_arm"], "lags": item["lags"], "values": item["values"], "common_support_count": item["support"], "k_n": k_n if item["audio_arm"] == "N_RAW" else None})
            print(f"score sample={sid} cells=15 support={len(common_t)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - retain per-utterance failure details
            failures.append({"sample_id": sid, "paired_key": key, "stage": "score", "cell": "all_15_cells", "error": str(exc)})
            print(f"score FAILED sample={sid}: {exc}", flush=True)
    expected = len(rows) * 15
    status = "complete" if not failures and len(scores) == expected and len(curves) == expected else "incomplete"
    scores_path = output / "scores.csv"
    _write_scores_csv(scores_path, scores)
    write_json(output / "curves.json", {"schema_version": 1, "lags": list(range(-VSHIFT, VSHIFT + 1)), "curves": curves})
    manifest = {"schema_version": 1, "status": status, "sample_count": len(rows), "expected_cells": expected, "score_rows": len(scores), "curve_rows": len(curves), "failures": failures, "syncnet_model_sha256": inputs["models"]["syncnet_model_sha256"], "scoring_protocol": {"vshift": VSHIFT, "min_common_windows": min_common_windows, "common_t_rule": "range(15,F-15)", "fixed_crop": True}}
    write_json(output / "scores_manifest.json", manifest)
    if status != "complete":
        raise ProtocolError(f"score stage incomplete: {failures}")
    return manifest


SCORE_FIELDS = ["schema_version", "status", "sample_id", "paired_key", "video_arm", "audio_arm", "C", "D", "curve_median", "k_star", "d_zero", "common_support_count", "fixed_k_n", "d_fixed_natural_lag", "video", "video_sha256", "audio", "audio_sha256", "score_box_sha256", "syncnet_model_sha256"]


def _write_scores_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCORE_FIELDS)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in SCORE_FIELDS} for row in rows)


def _read_scores(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ProtocolError(f"scores.csv is missing: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        copy = dict(row)
        for key in ("sample_id", "k_star", "common_support_count", "fixed_k_n"):
            if copy.get(key) not in (None, ""):
                copy[key] = int(copy[key])
        for key in ("C", "D", "curve_median", "d_zero", "d_fixed_natural_lag"):
            if copy.get(key) not in (None, ""):
                copy[key] = float(copy[key])
        parsed.append(copy)
    return parsed


def _bootstrap_indices(n: int, *, seed: int, draws: int = BOOTSTRAP_DRAWS) -> np.ndarray:
    if n <= 0:
        raise ProtocolError("bootstrap requires at least one utterance")
    return np.random.default_rng(seed).integers(0, n, size=(draws, n), endpoint=False)


def _ci(values: np.ndarray, indices: np.ndarray, lower: float, upper: float) -> list[float]:
    estimates = np.mean(values[indices], axis=1)
    return [float(np.quantile(estimates, lower, method="linear")), float(np.quantile(estimates, upper, method="linear"))]


def _comparison(values: Sequence[float], indices: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ProtocolError("comparison contains non-finite values")
    return {"n": int(arr.size), "mean": float(np.mean(arr)), "median": float(np.median(arr)), "positive_count": int(np.sum(arr > 0)), "negative_count": int(np.sum(arr < 0)), "ci95": _ci(arr, indices, 0.025, 0.975), "ci_bonferroni_98_333": _ci(arr, indices, 0.008333333333333333, 0.9916666666666667), "values": arr.tolist()}


def _cell_map(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    cells: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (int(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"]))
        if key in cells:
            raise ProtocolError(f"duplicate score cell: {key}")
        for metric in ("C", "D"):
            if not math.isfinite(float(row[metric])):
                raise ProtocolError(f"non-finite score cell: {key}")
        cells[key] = row
    return cells


def _q(cells: Mapping[tuple[int, str, str], Mapping[str, Any]], sample_ids: Sequence[int], video_arm: str, audio_arm: str, metric: str) -> np.ndarray:
    values = []
    for sid in sample_ids:
        row = cells[(sid, video_arm, audio_arm)]
        value = float(row[metric])
        values.append(-value if metric == "D" else value)
    return np.asarray(values, dtype=np.float64)


def _paired_delta(cells: Mapping[tuple[int, str, str], Mapping[str, Any]], sample_ids: Sequence[int], candidate: tuple[str, str], baseline: tuple[str, str], metric: str) -> np.ndarray:
    return _q(cells, sample_ids, *candidate, metric) - _q(cells, sample_ids, *baseline, metric)


def _sample_common_support(cells: Mapping[tuple[int, str, str], Mapping[str, Any]], sample_id: int) -> int:
    rows = [row for (sid, _video, audio), row in cells.items() if sid == int(sample_id) and audio != "T_RAW"]
    if len(rows) != 14:
        raise ProtocolError(f"sample {sample_id} must have 14 non-T_RAW mechanism cells, found {len(rows)}")
    supports = [int(row["common_support_count"]) for row in rows]
    if any(value <= 0 for value in supports):
        raise ProtocolError(f"sample {sample_id} has non-positive common support")
    if len(set(supports)) != 1:
        raise ProtocolError(f"sample {sample_id} mechanism cells do not share one common support: {sorted(set(supports))}")
    return min(supports)


def _direction(value: float | None) -> str:
    if value is None:
        return "no_eligible_samples"
    if value > 0:
        return "positive"
    if value < 0:
        return "negative"
    return "zero"


def _sensitivity_table(cells: Mapping[tuple[int, str, str], Mapping[str, Any]], sample_ids: Sequence[int], thresholds: Sequence[int] = SENSITIVITY_THRESHOLDS) -> list[dict[str, Any]]:
    support_by_sample = {int(sample_id): _sample_common_support(cells, int(sample_id)) for sample_id in sample_ids}
    table: list[dict[str, Any]] = []
    for raw_threshold in thresholds:
        threshold = _validate_min_common_windows(raw_threshold)
        eligible = [int(sample_id) for sample_id in sample_ids if support_by_sample[int(sample_id)] >= threshold]
        effects: dict[str, Any] = {}
        if eligible:
            gown = {metric: float(np.mean(_paired_delta(cells, eligible, ("T_100", "T_100"), ("N_RAW", "N_RAW"), metric))) for metric in ("C", "D")}
            fixed_natural: dict[str, dict[str, float]] = {}
            for name, candidate, baseline in (("R", ("N_100", "N_RAW"), ("N_RAW", "N_RAW")), ("E", ("T_100", "N_RAW"), ("N_100", "N_RAW"))):
                fixed_natural[name] = {metric: float(np.mean(_paired_delta(cells, eligible, candidate, baseline, metric))) for metric in ("C", "D")}
            fixed_natural["I"] = {
                metric: float(np.mean(_paired_delta(cells, eligible, ("T_100", "N_RAW"), ("T_000", "N_RAW"), metric) - _paired_delta(cells, eligible, ("N_100", "N_RAW"), ("N_000", "N_RAW"), metric)))
                for metric in ("C", "D")
            }
            effects = {
                "G_own": {"C": gown["C"], "D_improvement": gown["D"]},
                "fixed_natural_track": {name: {"C": values["C"], "D_improvement": values["D"]} for name, values in fixed_natural.items()},
                "effect_direction": {
                    "G_own_C": _direction(gown["C"]),
                    "G_own_D_improvement": _direction(gown["D"]),
                    **{f"{name}_C": _direction(values["C"]) for name, values in fixed_natural.items()},
                    **{f"{name}_D_improvement": _direction(values["D"]) for name, values in fixed_natural.items()},
                },
            }
        else:
            effects = {"G_own": {"C": None, "D_improvement": None}, "fixed_natural_track": {}, "effect_direction": {}}
        table.append({"threshold": threshold, "n": len(eligible), "sample_ids": eligible, "support_by_sample": {str(sid): support_by_sample[sid] for sid in eligible}, **effects})
    return table


def _render_report(analysis: Mapping[str, Any], output: Path) -> None:
    positive = analysis["positive_reference"]
    main = analysis["main_comparisons"]["q_N"]
    sensitivity = analysis.get("sensitivity", [])
    lines = [
        "# MFA-linear 连续轨迹机制消融实验",
        "",
        f"这是探索性 support30 补充运行，固定 S0765 的 {analysis['sample_count']} 条历史样本，完成状态为 **{analysis['status']}**，最小共同支持为 {analysis.get('min_common_windows', MIN_COMMON_WINDOWS)} 个窗口。阳性参照 G_own = q_own(T_100) - q_own(N_RAW) 的 Sync-C 均值为 {positive['C']['mean']:.3f}，普通 95% CI 为 [{positive['C']['ci95'][0]:.3f}, {positive['C']['ci95'][1]:.3f}]；{positive['statement']}。D 方向使用基线 D 减候选 D。",
        "",
        f"固定自然音轨视角的主对比为 R={main['R']['C']['mean']:.3f}、E={main['E']['C']['mean']:.3f}、I={main['I']['C']['mean']:.3f}（Sync-C；主 CI 为 Bonferroni 98.333% 区间）。R+E=G 是同一视角下的代数分解，不是两个独立因果效应。",
        "",
        "## 运行边界",
        "",
        "这是单说话人、历史已见样本的探索性诊断。WavLM-Large L6、prematched HiFi-GAN、Wav2Lip 和 SyncNet 均冻结；音素内动态在自然时间网格中按逐 occurrence 均值保持残差缩放。结果仍可能混有声码器适配、音质和可懂度变化，不能宣称真实嘴型运动改善。",
        "",
        "## 主要结果",
        "",
        _markdown_comparison_table(main),
        "",
        "## 支持阈值敏感性（描述性）",
        "",
        "下表只从当前 225 个 cell 按每条样本的 14 个非 T_RAW cell 的共同支持数筛选；不重评分、不重切片、不做显著性检验。D_improvement 已按低 D 为正方向报告。",
        "",
        _markdown_sensitivity_table(sensitivity),
        "",
        "剂量曲线见 `keep_dose.png`。报告同时保存了 225 个 cell、31 点 lag 曲线、固定 crop、共同支持数和每条 utterance 的 bootstrap 输入，主数据可从 `scores.csv` 与 `curves.json` 重算。",
        "",
        "听检：未完成（本轮没有把主观听感作为删样本或结论依据）。",
        "",
        "下一步只在需要时另立协议扩展说话人或加入真实视频真值；本轮不据此调整强度、crop 或模型。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _render_incomplete_report(analysis: Mapping[str, Any], output: Path) -> None:
    failures = analysis.get("failures", [])
    threshold = analysis.get("min_common_windows", MIN_COMMON_WINDOWS)
    lines = [
        "# MFA-linear 连续轨迹机制消融实验",
        "",
        "本轮状态为 **incomplete**。prepare、audio 和 render 已完成 15 条样本、120 个视频；SyncNet 按 spec 的固定共同支持规则只完成了部分评分，不能生成 225 cell 的机制统计。",
        "",
        f"固定规则为 `range(15,F-15)` 且每条至少需要 {threshold} 个 t。本轮成功评分 {analysis['score_cell_count']} / {analysis['expected_score_cell_count']} 个 cell；{len(failures)} 条样本因共同支持不足失败。未降低阈值、未换样本、未删除失败样本。",
        "",
        "## 失败清单",
        "",
        "| sample_id | cell | paired_key | 原因 |",
        "|---:|---|---|---|",
    ]
    for failure in failures:
        lines.append(f"| {failure.get('sample_id')} | {failure.get('cell', 'all')} | {failure.get('paired_key')} | {failure.get('error')} |")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "已有的 120 个 cell 和曲线仅作为技术诊断保留，不能代替预先规定的 15 条完整配对统计；本轮不报告阳性参照、R/E/I 或剂量结论。要得到完整实验，需另行批准降低最小支持窗口或改选更长的固定样本，并作为新协议运行。",
            "",
            f"详细状态见 `scores_manifest.json`（{analysis['score_cell_count']} 个成功 cell、{len(failures)} 个失败样本）；原始输入、音频质量和视频 manifest 仍可复核。",
        ]
    )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stale_plot = output / "keep_dose.png"
    if stale_plot.exists():
        stale_plot.unlink()


def _markdown_comparison_table(main: Mapping[str, Any]) -> str:
    lines = ["| 对比 | Sync-C 均值 | CI95 | Bonferroni 98.333% CI | 正向条数/n |", "|---|---:|---:|---:|---:|"]
    for label in ("R", "E", "I"):
        row = main[label]["C"]
        lines.append(f"| {label} | {row['mean']:.3f} | [{row['ci95'][0]:.3f}, {row['ci95'][1]:.3f}] | [{row['ci_bonferroni_98_333'][0]:.3f}, {row['ci_bonferroni_98_333'][1]:.3f}] | {row['positive_count']}/{row['n']} |")
    return "\n".join(lines)


def _markdown_sensitivity_table(rows: Sequence[Mapping[str, Any]]) -> str:
    lines = ["| 最小支持 | n | sample_ids | G_own C | G_own方向 | R C | R方向 | E C | E方向 | I C | I方向 |", "|---:|---:|---|---:|---|---:|---|---:|---|---:|---|"]
    for row in rows:
        effects = row.get("fixed_natural_track", {})
        gown = row.get("G_own", {}).get("C")
        values = [gown, effects.get("R", {}).get("C"), effects.get("E", {}).get("C"), effects.get("I", {}).get("C")]
        formatted = ["—" if value is None else f"{float(value):.3f}" for value in values]
        directions = row.get("effect_direction", {})
        labels = [directions.get("G_own_C", "no_eligible_samples"), directions.get("R_C", "no_eligible_samples"), directions.get("E_C", "no_eligible_samples"), directions.get("I_C", "no_eligible_samples")]
        cells = []
        for value, label in zip(formatted, labels, strict=True):
            cells.extend((value, label))
        lines.append(f"| {row.get('threshold')} | {row.get('n')} | {','.join(str(value) for value in row.get('sample_ids', [])) or '—'} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _dose_plot(analysis: Mapping[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    dose = analysis["dose"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    for axis, metric in zip(axes, ("C", "D_improvement"), strict=True):
        for source, color in (("N", "tab:blue"), ("T", "tab:orange")):
            points = dose[metric][source]
            axis.plot(points["keep"], points["q_own"], marker="o", color=color, label=f"{source} q_own")
            axis.plot(points["keep"], points["q_N"], marker="x", linestyle="--", color=color, label=f"{source} q_N")
        axis.set_xlabel("keep")
        axis.set_ylabel(metric)
        axis.set_xticks([0.0, 0.5, 1.0])
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output / "keep_dose.png", dpi=160)
    plt.close(fig)


def analyze_stage(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output_dir.resolve()
    inputs = read_json(output / "inputs.json")
    min_common_windows = _effective_min_common_windows(args, inputs)
    score_manifest = read_json(output / "scores_manifest.json")
    manifest_threshold = score_manifest.get("scoring_protocol", {}).get("min_common_windows")
    if manifest_threshold is not None and _validate_min_common_windows(manifest_threshold) != min_common_windows:
        raise ProtocolError(f"scores_manifest threshold {manifest_threshold} disagrees with inputs/CLI {min_common_windows}")
    if score_manifest.get("status") != "complete":
        failures = list(score_manifest.get("failures", []))
        rows = _read_scores(output / "scores.csv")
        analysis = {
            "schema_version": 1,
            "status": "incomplete",
            "analysis": "mfa_linear_trajectory_ablation",
            "sample_ids": [int(row["sample_id"]) for row in inputs.get("records", [])],
            "sample_count": len(inputs.get("records", [])),
            "score_cell_count": len(rows),
            "expected_score_cell_count": int(score_manifest.get("expected_cells", 225)),
            "min_common_windows": min_common_windows,
            "failures": failures,
            "reason": f"fixed common support range(15,F-15) requires at least {min_common_windows} windows; score manifest is incomplete",
            "source_files": {"inputs": str(output / "inputs.json"), "scores": str(output / "scores.csv"), "curves": str(output / "curves.json")},
        }
        write_json(output / "analysis.json", analysis)
        _render_incomplete_report(analysis, output)
        return analysis
    rows = _read_scores(output / "scores.csv")
    expected_ids = [int(row["sample_id"]) for row in inputs["records"]]
    active_ids = expected_ids[:1] if args.smoke else expected_ids
    if not args.smoke and len(active_ids) != 15:
        raise ProtocolError("full analysis requires 15 input records")
    expected_cells = len(active_ids) * 15
    if len(rows) != expected_cells:
        raise ProtocolError(f"scores.csv has {len(rows)} rows; expected {expected_cells}")
    cells = _cell_map(rows)
    expected_cell_set = {(sid, arm, arm) for sid in active_ids for arm in ARMS} | {(sid, arm, "N_RAW") for sid in active_ids for arm in RESYNTH_ARMS} | {(sid, "N_RAW", "T_100") for sid in active_ids}
    if set(cells) != expected_cell_set:
        missing, extra = sorted(expected_cell_set - set(cells)), sorted(set(cells) - expected_cell_set)
        raise ProtocolError(f"score cell set mismatch; missing={missing[:3]} extra={extra[:3]}")
    if not args.smoke and len(rows) != 225:
        raise ProtocolError("full analysis requires exactly 225 score cells")
    for row in rows:
        support = int(row["common_support_count"])
        if support < min_common_windows:
            raise ProtocolError(f"score cell support {support} is below threshold {min_common_windows}: {row['sample_id']} {row['video_arm']} {row['audio_arm']}")
    curves_payload = read_json(output / "curves.json")
    curve_rows = list(curves_payload.get("curves", []))
    if len(curve_rows) != len(rows):
        raise ProtocolError(f"curves.json has {len(curve_rows)} rows; expected {len(rows)}")
    for curve in curve_rows:
        if len(curve.get("lags", [])) != 2 * VSHIFT + 1 or len(curve.get("values", [])) != 2 * VSHIFT + 1:
            raise ProtocolError("every SyncNet curve must contain 31 lag/value points")
        if int(curve.get("common_support_count", 0)) < min_common_windows:
            raise ProtocolError(f"curve support is below threshold {min_common_windows}")
    indices = _bootstrap_indices(len(active_ids), seed=BOOTSTRAP_SEED)
    qown: dict[str, dict[str, np.ndarray]] = {metric: {} for metric in ("C", "D")}
    qn: dict[str, dict[str, np.ndarray]] = {metric: {} for metric in ("C", "D")}
    for metric in ("C", "D"):
        for arm in ARMS:
            qown[metric][arm] = _q(cells, active_ids, arm, arm, metric)
        for arm in ("N_RAW",) + RESYNTH_ARMS:
            qn[metric][arm] = _q(cells, active_ids, arm, "N_RAW", metric)
    def view_comparisons(view: Mapping[str, Mapping[str, np.ndarray]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name, candidate, baseline in (("R", "N_100", "N_RAW"), ("E", "T_100", "N_100")):
            result[name] = {metric: _comparison(view[metric][candidate] - view[metric][baseline], indices) for metric in ("C", "D")}
        result["I"] = {metric: _comparison((view[metric]["T_100"] - view[metric]["T_000"]) - (view[metric]["N_100"] - view[metric]["N_000"]), indices) for metric in ("C", "D")}
        return result
    comparisons = {"q_own": view_comparisons(qown), "q_N": view_comparisons(qn)}
    positive_c = _comparison(qown["C"]["T_100"] - qown["C"]["N_RAW"], indices)
    positive_d = _comparison(qown["D"]["T_100"] - qown["D"]["N_RAW"], indices)
    positive_c["direction"] = "candidate C - baseline C"
    positive_d["direction"] = "baseline D - candidate D"
    positive = {"C": positive_c, "D": positive_d}
    if positive_c["mean"] <= 0:
        statement = "本轮协议没有复现原生正向效应"
    elif positive_c["ci95"][0] <= 0:
        statement = "均值为正但阳性参照的普通 95% CI 跨 0"
    else:
        statement = "阳性参照在本历史样本/当前协议中成立"
    positive["statement"] = statement
    dose: dict[str, dict[str, dict[str, list[float]]]] = {"C": {}, "D_improvement": {}}
    for metric in ("C", "D"):
        out_metric = "D_improvement" if metric == "D" else metric
        for source, arms in (("N", ("N_000", "N_050", "N_100")), ("T", ("T_000", "T_050", "T_100"))):
            keeps = [0.0, 0.5, 1.0]
            dose[out_metric][source] = {"keep": keeps, "q_own": [float(np.mean(qown[metric][arm])) for arm in arms], "q_N": [float(np.mean(qn[metric][arm])) if arm in qn[metric] else float("nan") for arm in arms]}
            if metric == "D":
                dose[out_metric][source]["q_own"] = [-value for value in dose[out_metric][source]["q_own"]]
                dose[out_metric][source]["q_N"] = [-value for value in dose[out_metric][source]["q_N"]]
    two_by_two = {
        "fixed_A_N_video_T100_minus_NRAW": _comparison(_q(cells, active_ids, "T_100", "N_RAW", "C") - _q(cells, active_ids, "N_RAW", "N_RAW", "C"), indices),
        "fixed_A_T_video_T100_minus_NRAW": _comparison(_q(cells, active_ids, "T_100", "T_100", "C") - _q(cells, active_ids, "N_RAW", "T_100", "C"), indices),
        "fixed_V_N_audio_T100_minus_NRAW": _comparison(_q(cells, active_ids, "N_RAW", "T_100", "C") - _q(cells, active_ids, "N_RAW", "N_RAW", "C"), indices),
        "fixed_V_T_audio_T100_minus_NRAW": _comparison(_q(cells, active_ids, "T_100", "T_100", "C") - _q(cells, active_ids, "T_100", "N_RAW", "C"), indices),
    }
    analysis = {"schema_version": 1, "status": "complete", "analysis": "mfa_linear_trajectory_ablation", "protocol": inputs.get("protocol"), "parent_run": inputs.get("parent_run"), "min_common_windows": min_common_windows, "sample_ids": active_ids, "sample_count": len(active_ids), "score_cell_count": len(rows), "directions": {"C": "higher_is_better", "D": "lower_is_better; reported q=-D"}, "bootstrap": {"seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "unit": "utterance", "main_ci": "Bonferroni 98.333% for q_N R/E/I"}, "positive_reference": positive, "main_comparisons": comparisons, "dose": dose, "two_by_two": two_by_two, "sensitivity": _sensitivity_table(cells, active_ids), "limitations": ["single speaker S0765", "historical seen cohort", "feature mean ablation can change vocoder-domain quality", "no human mouth-motion ground truth"], "listening_check": "not completed", "source_files": {"inputs": str(output / "inputs.json"), "scores": str(output / "scores.csv"), "curves": str(output / "curves.json")}}
    write_json(output / "analysis.json", analysis)
    _render_report(analysis, output)
    _dose_plot(analysis, output)
    return analysis


def launch_score(args: argparse.Namespace) -> dict[str, Any]:
    syncnet_python = _require_file(args.syncnet_python.expanduser(), "SyncNet Python")
    output = args.output_dir.resolve()
    inputs = read_json(output / "inputs.json")
    min_common_windows = _effective_min_common_windows(args, inputs)
    command = [str(syncnet_python), str(Path(__file__).resolve()), "--stage", "score-worker", "--output-dir", str(output), "--wav2lip", str(args.wav2lip.resolve()), "--syncnet", str(args.syncnet.resolve()), "--syncnet-model", str(args.syncnet_model.resolve()), "--min-common-windows", str(min_common_windows), "--smoke" if args.smoke else "--no-smoke"]
    log = output / "logs" / "score_worker.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(args.syncnet.resolve()), stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise ProtocolError(f"score worker failed; see {log}")
    return read_json(output / "scores_manifest.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("prepare", "reuse", "audio", "render", "score", "score-worker", "analyze"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--source-run", type=Path, default=DEFAULT_OUTPUT, help="completed v1 run to reuse for --stage reuse")
    parser.add_argument("--min-common-windows", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="allow rebuilding metadata in an existing output directory")
    parser.add_argument("--a-summary", type=Path, default=DEFAULT_A_SUMMARY)
    parser.add_argument("--b-alignment", type=Path, default=DEFAULT_B_ALIGNMENT)
    parser.add_argument("--c-manifest", type=Path, default=DEFAULT_C_MANIFEST)
    parser.add_argument("--knn-source", type=Path, default=DEFAULT_KNN_SOURCE)
    parser.add_argument("--wav2lip", type=Path, default=DEFAULT_WAV2LIP)
    parser.add_argument("--syncnet", type=Path, default=DEFAULT_SYNCNET)
    parser.add_argument("--wav2lip-python", type=Path, default=DEFAULT_WAV2LIP_PYTHON)
    parser.add_argument("--syncnet-python", type=Path, default=DEFAULT_SYNCNET_PYTHON)
    parser.add_argument("--wav2lip-checkpoint", type=Path, default=DEFAULT_WAV2LIP_CHECKPOINT)
    parser.add_argument("--syncnet-model", type=Path, default=DEFAULT_SYNCNET_MODEL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", dest="smoke", action="store_true")
    parser.add_argument("--no-smoke", dest="smoke", action="store_false")
    parser.set_defaults(smoke=False)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.stage == "prepare":
            result = prepare(args)
        elif args.stage == "reuse":
            result = prepare_reuse(args)
        elif args.stage == "audio":
            result = audio_stage(args)
        elif args.stage == "render":
            result = render_stage(args)
        elif args.stage == "score":
            result = launch_score(args)
        elif args.stage == "score-worker":
            result = score_worker(args)
        else:
            result = analyze_stage(args)
    except Exception as exc:  # noqa: BLE001 - CLI must report a concise protocol failure
        print(f"{args.stage} FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"stage": args.stage, "status": result.get("status"), "output_dir": str(args.output_dir.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
