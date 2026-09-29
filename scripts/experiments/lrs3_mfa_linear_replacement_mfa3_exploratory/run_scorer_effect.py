"""Run the symmetric MFA-linear scorer-effect experiment.

This is a new score-only protocol revision over the frozen external-fit
face-ready cohort.  It binds the already completed Wav2Lip renders and scores
all four ``G``/``E`` cells:

``G_N_E_N``
    natural audio generated video, natural audio scored;
``G_M_E_N``
    MFA-linear audio generated video, natural audio scored;
``G_N_E_M``
    natural audio generated video, MFA-linear audio scored;
``G_M_E_M``
    MFA-linear audio generated video, MFA-linear audio scored.

The six candidate WAVs whose historical paths disappeared are regenerated
only when their SHA-256 matches the audio hash recorded by the completed
Wav2Lip render.  The old strict 24-record run is intentionally not relaxed:
``spn`` remains an unknown-speech alignment failure rather than being silently
treated as silence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from scripts.experiments.lrs3_mfa_linear_replacement.candidate_audio import (
    canonical_pcm_s16le,
    candidate_from_features,
    validate_wavlm_interface,
)
from scripts.experiments.lrs3_mfa_linear_replacement.mfa_alignment import (
    build_frame_mapping,
    trace_rows,
)
from scripts.experiments.lrs3_mfa_linear_replacement.strict_mux import (
    file_sha256,
    mux_and_verify,
    verify_mux_integrity,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.protocol import (
    KNN_VC_REVISION,
    VOCODER_CHECKPOINT,
    WAVLM_CHECKPOINT,
    load_json,
    write_json,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_candidate_audio import (
    _assert_exploratory_gpu_ready,
    _read_pcm16,
    _write_pcm16,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.run_strict_replacement import (
    MIN_TRACK,
    _parse_syncnet,
    _run_logged,
)
from scripts.experiments.lrs3_mfa_linear_replacement_mfa3_exploratory.scorer_effect import (
    CELL_NAMES,
    analyze_scorer_effect,
)
from scripts.wavlm_knn_vc_adapter import WavLMKNNVCAdapter

REPO = Path(__file__).resolve().parents[3]
SOURCE_ROOT = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826"
STAGE00_PATH = SOURCE_ROOT / "00_external_fit_supplement_protocol_retry4/manifest.json"
ALIGNMENT_PATH = SOURCE_ROOT / "01_external_fit_mfa3_screen_retry4/alignment_manifest.json"
CANDIDATE_PATH = SOURCE_ROOT / "02_external_fit_candidate_audio_retry4/candidate_manifest.json"
FACE_PREFLIGHT_PATH = SOURCE_ROOT / "03_external_fit_face_preflight_retry4/face_preflight_manifest.json"
RENDER_PATH = SOURCE_ROOT / "04_external_fit_wav2lip_render_retry6/render_manifest.json"

PROTOCOL_ID = "lrs3_mfa_linear_scorer_effect_20260923"
DEFAULT_RUN_ROOT = REPO / "runs/lrs3_mfa_linear_scorer_effect_20260923"
SAMPLE_RATE = 16_000
FFMPEG = Path(shutil.which("ffmpeg") or "ffmpeg")
FFPROBE = Path(shutil.which("ffprobe") or "ffprobe")
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_PY = Path.home() / ".venvs/syncnet/bin/python"
SYNCNET_PIPELINE = SYNCNET_ROOT / "run_pipeline.py"
SYNCNET_SCORE = SYNCNET_ROOT / "run_syncnet.py"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"

# The historical external-fit candidate hashes were produced by this exact
# runtime.  PyTorch 2.5 produces numerically close, but byte-different,
# WavLM features on the same inputs, so candidate repair must fail before
# writing an audio artifact when the runtime is not reproducible.
CANDIDATE_PYTHON = Path("/home/wjj/miniconda3/bin/python")
CANDIDATE_TORCH_VERSION = "2.13.0+cu126"
CANDIDATE_TORCH_CUDA = "12.6"
CANDIDATE_CUDNN_VERSION = 91002

CELL_SPECS: dict[str, dict[str, str]] = {
    "G_N_E_N": {"video_key": "natural_video", "audio_key": "natural_audio"},
    "G_M_E_N": {"video_key": "candidate_video", "audio_key": "natural_audio"},
    "G_N_E_M": {"video_key": "natural_video", "audio_key": "candidate_audio"},
    "G_M_E_M": {"video_key": "candidate_video", "audio_key": "candidate_audio"},
}


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _asset(path: Path) -> dict[str, str]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {"path": str(resolved), "sha256": file_sha256(resolved)}


def _executable_asset(path: Path) -> dict[str, str]:
    """Hash an executable target while preserving venv launch semantics."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _write_hashed_json(path: Path, value: Mapping[str, Any]) -> str:
    payload = dict(value)
    payload.pop("artifact_sha256", None)
    payload["artifact_sha256"] = _sha256_json(payload)
    write_json(path, payload)
    return str(payload["artifact_sha256"])


def _load_hashed_json(path: Path) -> dict[str, Any]:
    payload = load_json(path)
    stored = payload.pop("artifact_sha256", None)
    if not isinstance(stored, str) or stored != _sha256_json(payload):
        raise ValueError(f"artifact hash mismatch: {path}")
    payload["artifact_sha256"] = stored
    return payload


def _path_binding(path: Path, expected_sha256: str, *, allow_missing: bool = False) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        if allow_missing:
            return {"path": str(resolved), "sha256": expected_sha256, "exists": False}
        raise FileNotFoundError(resolved)
    actual = file_sha256(resolved)
    if actual != expected_sha256:
        raise ValueError(f"SHA-256 mismatch: {resolved}")
    return {"path": str(resolved), "sha256": actual, "exists": True}


def _candidate_runtime() -> dict[str, Any]:
    return {
        "python": str(Path(sys.executable).resolve()),
        "torch_version": str(torch.__version__),
        "torch_cuda": str(torch.version.cuda),
        "cudnn_version": int(torch.backends.cudnn.version() or 0),
    }


def _assert_candidate_runtime() -> dict[str, Any]:
    actual = _candidate_runtime()
    expected_python = CANDIDATE_PYTHON.resolve()
    expected = {
        "python": str(expected_python),
        "torch_version": CANDIDATE_TORCH_VERSION,
        "torch_cuda": CANDIDATE_TORCH_CUDA,
        "cudnn_version": CANDIDATE_CUDNN_VERSION,
    }
    if actual != expected:
        raise RuntimeError(
            "candidate repair requires the historical runtime "
            f"({expected}); got {actual}. Run with {CANDIDATE_PYTHON}."
        )
    return actual


def _selected_ids(render: Mapping[str, Any], *, max_records: int | None, sample_ids: Sequence[str]) -> list[str]:
    all_ids = [str(value) for value in render.get("face_ready_sample_ids", [])]
    if not all_ids or len(all_ids) != len(set(all_ids)):
        raise ValueError("render manifest face-ready IDs are missing or duplicated")
    if sample_ids:
        requested = {str(value) for value in sample_ids}
        missing = requested - set(all_ids)
        if missing:
            raise ValueError(f"requested sample IDs are outside the frozen face-ready cohort: {sorted(missing)}")
        selected = [sample_id for sample_id in all_ids if sample_id in requested]
    else:
        selected = list(all_ids)
    if max_records is not None:
        if max_records <= 0:
            raise ValueError("max-records must be positive")
        selected = selected[:max_records]
    if not selected:
        raise ValueError("selected scorer-effect cohort is empty")
    return selected


def _video_frame_count(path: Path) -> int:
    result = subprocess.run(
        [
            str(FFPROBE), "-v", "error", "-count_frames", "-select_streams", "v:0",
            "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe frame count failed for {path}: {result.stderr.strip()}")
    try:
        count = int(result.stdout.strip())
    except ValueError as exc:
        raise ValueError(f"invalid video frame count for {path}: {result.stdout!r}") from exc
    if count <= 0:
        raise ValueError(f"video contains no frames: {path}")
    return count


def build_protocol(
    run_root: Path,
    *,
    max_records: int | None = None,
    sample_ids: Sequence[str] = (),
) -> dict[str, Any]:
    stage00 = load_json(STAGE00_PATH)
    alignment = load_json(ALIGNMENT_PATH)
    candidate = load_json(CANDIDATE_PATH)
    face = load_json(FACE_PREFLIGHT_PATH)
    render = load_json(RENDER_PATH)
    if stage00.get("protocol_id") != "lrs3_mfa_linear_replacement_mfa3_external_fit_supplement_20260826" or stage00.get("status") != "complete":
        raise ValueError("external-fit Stage00 is not complete")
    if alignment.get("clean_record_count") != len(alignment.get("records", [])) or alignment.get("status") not in {"partial", "complete"}:
        raise ValueError("external-fit alignment manifest is malformed")
    if candidate.get("status") != "complete" or candidate.get("record_count") != len(candidate.get("results", [])):
        raise ValueError("external-fit candidate manifest is not complete")
    if face.get("status") != "complete" or face.get("engineering_decision") != "GO":
        raise ValueError("external-fit face preflight is not complete GO")
    if render.get("status") != "complete" or render.get("render_count") != len(render.get("renders", [])):
        raise ValueError("external-fit render manifest is not complete")

    selected = _selected_ids(render, max_records=max_records, sample_ids=sample_ids)
    stage_by_id = {str(row["sample_id"]): row for row in stage00["cohort"]["records"]}
    candidate_by_id = {str(row["sample_id"]): row for row in candidate["results"]}
    render_by_key = {(str(row["sample_id"]), str(row["arm"])): row for row in render["renders"]}
    alignment_ids = {str(row["sample_id"]) for row in alignment["records"]}
    repair_audio_dir = run_root / "01_repair" / "audio"
    records: list[dict[str, Any]] = []
    excluded_short_videos: list[dict[str, Any]] = []
    for sample_id in selected:
        if sample_id not in stage_by_id or sample_id not in candidate_by_id or sample_id not in alignment_ids:
            raise ValueError(f"selected sample lacks a complete external-fit join: {sample_id}")
        source = stage_by_id[sample_id]
        candidate_row = candidate_by_id[sample_id]
        natural_render = render_by_key.get((sample_id, "natural"))
        candidate_render = render_by_key.get((sample_id, "candidate"))
        if natural_render is None or candidate_render is None:
            raise ValueError(f"selected sample lacks natural/candidate render: {sample_id}")
        natural_audio = Path(str(source["natural_audio"])).resolve()
        candidate_history_path = Path(str(candidate_row["candidate_audio"])).resolve()
        expected_candidate_sha256 = str(candidate_row["candidate_audio_sha256"])
        if str(candidate_render["audio_sha256"]) != expected_candidate_sha256:
            raise ValueError(f"candidate render audio hash disagrees with candidate manifest: {sample_id}")
        if natural_render["audio_sha256"] != str(source["natural_audio_sha256"]):
            raise ValueError(f"natural render audio hash disagrees with Stage00: {sample_id}")
        candidate_audio = candidate_history_path if candidate_history_path.is_file() else repair_audio_dir / f"{sample_id}.wav"
        _path_binding(natural_audio, str(source["natural_audio_sha256"]))
        _path_binding(candidate_audio, expected_candidate_sha256, allow_missing=True)
        _path_binding(Path(str(natural_render["video"])), str(natural_render["video_sha256"]))
        _path_binding(Path(str(candidate_render["video"])), str(candidate_render["video_sha256"]))
        natural_video_frames = _video_frame_count(Path(str(natural_render["video"])))
        candidate_video_frames = _video_frame_count(Path(str(candidate_render["video"])))
        if min(natural_video_frames, candidate_video_frames) <= MIN_TRACK:
            excluded_short_videos.append({
                "sample_id": sample_id,
                "source_group": str(source["source_group"]),
                "natural_video_frames": natural_video_frames,
                "candidate_video_frames": candidate_video_frames,
                "reason": f"source video has at most {MIN_TRACK} frames; SyncNet requires a longer face track",
            })
            continue
        records.append({
            "sample_id": sample_id,
            "source_group": str(source["source_group"]),
            "natural_audio": str(natural_audio),
            "natural_audio_sha256": str(source["natural_audio_sha256"]),
            "candidate_audio": str(candidate_audio),
            "candidate_audio_sha256": expected_candidate_sha256,
            "candidate_history_path": str(candidate_history_path),
            "face_video": str(natural_render["face_video"]),
            "face_video_sha256": str(natural_render["face_video_sha256"]),
            "natural_video": str(natural_render["video"]),
            "natural_video_sha256": str(natural_render["video_sha256"]),
            "candidate_video": str(candidate_render["video"]),
            "candidate_video_sha256": str(candidate_render["video_sha256"]),
            "natural_video_frames": natural_video_frames,
            "candidate_video_frames": candidate_video_frames,
            "candidate_generation_audio_sha256": expected_candidate_sha256,
            "natural_generation_audio_sha256": str(natural_render["audio_sha256"]),
            "natural_samples": int(source["natural_audio_samples"]),
            "candidate_samples": int(candidate_row["candidate_samples"]),
        })
    groups = [str(row["source_group"]) for row in records]
    if len(set(groups)) > 24:
        raise ValueError("scorer-effect protocol requires at most 24 source groups for exact sign-flip inference")
    protocol = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "frozen",
        "parents": {
            "stage00": _asset(STAGE00_PATH),
            "alignment": _asset(ALIGNMENT_PATH),
            "candidate": _asset(CANDIDATE_PATH),
            "face_preflight": _asset(FACE_PREFLIGHT_PATH),
            "render": _asset(RENDER_PATH),
        },
        "selection": {
            "cohort": "external-fit face-ready render cohort with scoreable source-video length",
            "source_face_ready_record_count": len(selected),
            "minimum_source_video_frames_exclusive": MIN_TRACK,
            "excluded_short_videos": excluded_short_videos,
            "record_count": len(records),
            "source_group_count": len(set(groups)),
            "source_order_preserved": True,
            "score_based_selection": False,
            "visual_metric_used_for_selection": False,
            "syncnet_used_for_selection": False,
            "smoke_prefix": max_records is not None or bool(sample_ids),
        },
        "cells": CELL_SPECS,
        "primary_question": "does the SyncNet audio domain change the measured visual effect?",
        "audio_policy": {
            "sample_rate_hz": SAMPLE_RATE,
            "channels": 1,
            "candidate_exact_length": True,
            "spn_policy": "unknown speech remains excluded by the strict MFA screen; never relabeled as silence",
        },
        "assets": {
            "ffmpeg": _asset(FFMPEG),
            "ffprobe": _asset(FFPROBE),
            "syncnet_python": _executable_asset(SYNCNET_PY),
            "syncnet_pipeline": _asset(SYNCNET_PIPELINE),
            "syncnet_score": _asset(SYNCNET_SCORE),
            "syncnet_model": _asset(SYNCNET_MODEL),
            "wavlm": _asset(WAVLM_CHECKPOINT),
            "vocoder": _asset(VOCODER_CHECKPOINT),
            "candidate_generation_python": _executable_asset(CANDIDATE_PYTHON),
        },
        "candidate_generation_runtime": {
            "python": str(CANDIDATE_PYTHON.resolve()),
            "torch_version": CANDIDATE_TORCH_VERSION,
            "torch_cuda": CANDIDATE_TORCH_CUDA,
            "cudnn_version": CANDIDATE_CUDNN_VERSION,
        },
        "records": records,
    }
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(run_root / "protocol.json", protocol)
    (run_root / "protocol.sha256").write_text(file_sha256(run_root / "protocol.json") + "\n", encoding="utf-8")
    return protocol


def _candidate_repair_result(
    *,
    row: Mapping[str, Any],
    source: Mapping[str, Any],
    alignment: Mapping[str, Any],
    adapter: WavLMKNNVCAdapter,
    output: Path,
) -> dict[str, Any]:
    natural_values = _read_pcm16(Path(str(source["natural_audio"])))
    tts_values = _read_pcm16(Path(str(source["tts_audio"])))
    natural_features = adapter.extract(torch.from_numpy(natural_values).unsqueeze(0)).cpu().numpy()
    tts_features = adapter.extract(torch.from_numpy(tts_values).unsqueeze(0)).cpu().numpy()
    mapping, mapping_stats = build_frame_mapping(
        natural_features.shape[0],
        tts_features.shape[0],
        alignment["natural_tokens"],
        alignment["tts_tokens"],
    )
    candidate, metadata = candidate_from_features(
        natural_features=natural_features,
        tts_features=tts_features,
        mapping=mapping,
        natural_audio_samples=natural_values.size,
        vocode=lambda conditioning: adapter.vocode(torch.from_numpy(conditioning)).cpu().numpy(),
    )
    pcm, pcm_qc = canonical_pcm_s16le(candidate, natural_values.size)
    _write_pcm16(output, pcm)
    return {
        "sample_id": str(row["sample_id"]),
        "candidate_audio": str(output.resolve()),
        "candidate_audio_sha256": file_sha256(output),
        "candidate_samples": int(candidate.size),
        "natural_samples": int(natural_values.size),
        "mapping": mapping_stats,
        "candidate": metadata,
        "pcm_qc": pcm_qc,
        "trace_rows_sha256": _sha256_json(trace_rows(mapping)),
        "model_interface": adapter.metadata(),
        "generation_runtime": _candidate_runtime(),
    }


def repair_candidate_audio(protocol: Mapping[str, Any], run_root: Path, *, local_knn_vc: Path) -> dict[str, Any]:
    records = list(protocol["records"])
    missing = [row for row in records if not Path(str(row["candidate_audio"])).is_file()]
    if not missing:
        existing_manifest = run_root / "01_repair/manifest.json"
        if existing_manifest.is_file():
            prior = _load_hashed_json(existing_manifest)
            current_protocol_sha256 = file_sha256(run_root / "protocol.json")
            parents = dict(prior.get("parents", {}))
            if parents.get("protocol_sha256") != current_protocol_sha256:
                parents["protocol_sha256"] = current_protocol_sha256
                prior["parents"] = parents
                _write_hashed_json(existing_manifest, prior)
            return prior
        result = {"status": "complete", "repaired_count": 0, "records": []}
        _write_hashed_json(run_root / "01_repair/manifest.json", result)
        return result

    stage00 = load_json(STAGE00_PATH)
    alignment = load_json(ALIGNMENT_PATH)
    stage_by_id = {str(row["sample_id"]): row for row in stage00["cohort"]["records"]}
    alignment_by_id = {str(row["sample_id"]): row for row in alignment["records"]}
    _assert_candidate_runtime()
    _assert_exploratory_gpu_ready()
    adapter = WavLMKNNVCAdapter.load_pretrained(device="cuda", source=local_knn_vc, revision=KNN_VC_REVISION)
    validate_wavlm_interface(
        adapter.metadata(),
        revision=KNN_VC_REVISION,
        wavlm_checkpoint_sha256=file_sha256(WAVLM_CHECKPOINT),
        vocoder_checkpoint_sha256=file_sha256(VOCODER_CHECKPOINT),
    )
    repaired: list[dict[str, Any]] = []
    for row in missing:
        sample_id = str(row["sample_id"])
        output = Path(str(row["candidate_audio"])).resolve()
        result = _candidate_repair_result(
            row=row,
            source=stage_by_id[sample_id],
            alignment=alignment_by_id[sample_id],
            adapter=adapter,
            output=output,
        )
        if result["candidate_audio_sha256"] != row["candidate_audio_sha256"]:
            raise ValueError(f"repaired candidate hash does not match render-time hash: {sample_id}")
        if result["candidate_samples"] != int(row["natural_samples"]):
            raise ValueError(f"repaired candidate length mismatch: {sample_id}")
        repaired.append(result)
    result = {
        "status": "complete",
        "repaired_count": len(repaired),
        "records": repaired,
        "parents": {"protocol_sha256": file_sha256(run_root / "protocol.json")},
        "generation_runtime": _candidate_runtime(),
    }
    _write_hashed_json(run_root / "01_repair/manifest.json", result)
    return result


def _parse_score(log_path: Path) -> dict[str, float | int]:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    confidence = re.search(r"Confidence:\s+([-+0-9.eE]+)", text)
    distance = re.search(r"Min dist:\s+([-+0-9.eE]+)", text)
    offset = re.search(r"AV offset:\s+(-?\d+)", text)
    if confidence is None or distance is None:
        raise ValueError(f"SyncNet score fields missing: {log_path}")
    values: dict[str, float | int] = {
        "sync_c": float(confidence.group(1)),
        "sync_d": float(distance.group(1)),
        "av_offset": int(offset.group(1)) if offset else 0,
    }
    if not all(math.isfinite(float(values[key])) for key in ("sync_c", "sync_d", "av_offset")):
        raise ValueError(f"non-finite SyncNet score: {log_path}")
    return values


def _wait_for_gpu_ready() -> None:
    """Allow nvidia-smi's utilization sample to settle after our subprocess exits."""
    for attempt in range(31):
        try:
            _assert_exploratory_gpu_ready()
            return
        except RuntimeError as exc:
            if "is occupied: utilization=" not in str(exc) or attempt == 30:
                raise
            time.sleep(1)


def _score_cell(
    row: Mapping[str, Any],
    cell: str,
    output: Path,
    assets: Mapping[str, Any],
) -> dict[str, Any]:
    spec = CELL_SPECS[cell]
    video = Path(str(row[spec["video_key"]])).resolve()
    audio = Path(str(row[spec["audio_key"]])).resolve()
    expected_video_hash = str(row[f"{spec['video_key']}_sha256"])
    expected_audio_hash = str(row[f"{spec['audio_key']}_sha256"])
    if file_sha256(video) != expected_video_hash or file_sha256(audio) != expected_audio_hash:
        raise ValueError(f"input hash changed for {row['sample_id']} {cell}")
    mux_path = output / "mux" / cell / f"{row['sample_id']}.mkv"
    score_path = output / "scores" / cell / f"{row['sample_id']}.json"
    if score_path.is_file() and mux_path.is_file():
        prior = _load_hashed_json(score_path)
        if prior.get("cell") != cell or prior.get("muxed_file_sha256") != file_sha256(mux_path):
            raise ValueError(f"existing score identity changed: {row['sample_id']} {cell}")
        return prior
    if score_path.exists() and not score_path.is_file():
        raise ValueError(f"score artifact is not a regular file: {score_path}")
    if score_path.exists() and not mux_path.is_file():
        raise ValueError(f"score exists without its muxed media: {row['sample_id']} {cell}")
    if mux_path.exists() and not mux_path.is_file():
        raise ValueError(f"mux artifact is not a regular file: {mux_path}")

    mux_path.parent.mkdir(parents=True, exist_ok=True)
    score_path.parent.mkdir(parents=True, exist_ok=True)
    if mux_path.is_file():
        mux = verify_mux_integrity(
            source_video=video,
            muxed_file=mux_path,
            expected_audio=audio,
            ffmpeg=str(assets["ffmpeg"]["path"]),
            ffprobe=str(assets["ffprobe"]["path"]),
        )
    else:
        mux = mux_and_verify(
            source_video=video,
            expected_audio=audio,
            output_path=mux_path,
            ffmpeg=str(assets["ffmpeg"]["path"]),
            ffprobe=str(assets["ffprobe"]["path"]),
        )
    reference = f"lrs3_mfa_scorer_effect_{row['sample_id']}_{cell}"
    sync_dir = output / "syncnet" / cell / str(row["sample_id"])
    pipeline_log = output / "logs" / "syncnet" / cell / f"{row['sample_id']}.pipeline.log"
    _wait_for_gpu_ready()
    pipeline_rc = _run_logged(
        [
            str(assets["syncnet_python"]["path"]),
            str(assets["syncnet_pipeline"]["path"]),
            "--videofile", str(mux_path),
            "--reference", reference,
            "--data_dir", str(sync_dir),
            "--min_track", str(MIN_TRACK),
            "--overwrite",
        ],
        cwd=SYNCNET_ROOT,
        log_path=pipeline_log,
    )
    if pipeline_rc != 0:
        raise RuntimeError(f"SyncNet pipeline failed: {row['sample_id']} {cell}")
    score_log = output / "logs" / "syncnet" / cell / f"{row['sample_id']}.score.log"
    _wait_for_gpu_ready()
    score_rc = _run_logged(
        [
            str(assets["syncnet_python"]["path"]),
            str(assets["syncnet_score"]["path"]),
            "--videofile", str(mux_path),
            "--reference", reference,
            "--data_dir", str(sync_dir),
            "--initial_model", str(assets["syncnet_model"]["path"]),
        ],
        cwd=SYNCNET_ROOT,
        log_path=score_log,
    )
    if score_rc != 0:
        raise RuntimeError(f"SyncNet scorer failed: {row['sample_id']} {cell}")
    parsed = _parse_score(score_log)
    score = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "sample_id": str(row["sample_id"]),
        "source_group": str(row["source_group"]),
        "cell": cell,
        "video": str(video),
        "video_sha256": file_sha256(video),
        "audio": str(audio),
        "audio_sha256": file_sha256(audio),
        "muxed_file": str(mux_path),
        "muxed_file_sha256": file_sha256(mux_path),
        "mux_verification": mux,
        "syncnet_model_sha256": assets["syncnet_model"]["sha256"],
        "syncnet_python_sha256": assets["syncnet_python"]["sha256"],
        "pipeline_log": str(pipeline_log),
        "score_log": str(score_log),
        **parsed,
    }
    _write_hashed_json(score_path, score)
    return score


def score_stage(protocol: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    output = run_root / "02_scores"
    assets = protocol["assets"]
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(protocol["records"], 1):
        cells: dict[str, Any] = {}
        for cell in CELL_NAMES:
            cells[cell] = _score_cell(row, cell, output, assets)
            print(f"SCORE {index}/{len(protocol['records'])} {cell} {row['sample_id']}", flush=True)
        rows.append({"sample_id": row["sample_id"], "source_group": row["source_group"], **cells})
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "record_count": len(rows),
        "cell_count": len(rows) * len(CELL_NAMES),
        "protocol_sha256": file_sha256(run_root / "protocol.json"),
        "scores": rows,
    }
    _write_hashed_json(output / "manifest.json", manifest)
    return manifest


def analysis_stage(protocol: Mapping[str, Any], score_manifest: Mapping[str, Any], run_root: Path) -> dict[str, Any]:
    if score_manifest.get("record_count") != len(protocol["records"]):
        raise ValueError("score/protocol record count mismatch")
    expected = [(str(row["sample_id"]), str(row["source_group"])) for row in protocol["records"]]
    actual = [(str(row["sample_id"]), str(row["source_group"])) for row in score_manifest["scores"]]
    if actual != expected:
        raise ValueError("score order is not the frozen protocol order")
    matrix = [dict(row) for row in score_manifest["scores"]]
    result = analyze_scorer_effect(matrix)
    result.update({
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": file_sha256(run_root / "protocol.json"),
        "score_manifest_sha256": file_sha256(run_root / "02_scores/manifest.json"),
        "selection": dict(protocol["selection"]),
    })
    _write_hashed_json(run_root / "03_analysis.json", result)
    _write_report(result, run_root)
    return result


def _write_report(result: Mapping[str, Any], run_root: Path) -> None:
    selection = result["selection"]
    excluded = selection.get("excluded_short_videos", [])
    lines = [
        "# LRS3 MFA-linear scorer effect",
        "",
        "同一批 external-fit face-ready LRS3 样本上，对 natural/MFA-linear 生成音频与最终评分音轨做完整 2×2 交叉。Sync-C 按高为好、Sync-D 按低为好统一转成正向改善；推断单位为 source group 均值。",
        "",
        f"样本数：{result['record_count']}；source group 数：{result['source_group_count']}。",
        f"原始 face-ready 样本数：{selection.get('source_face_ready_record_count', result['record_count'])}；"
        f"短视频结构性排除：{len(excluded)}。",
        "",
        "| 生成视频所用音频 | 评分音轨 | Sync-C 均值 | Sync-D 均值 |",
        "|---|---|---:|---:|",
    ]
    for cell, generation, scoring in (
        ("G_N_E_N", "natural", "natural"),
        ("G_M_E_N", "MFA-linear", "natural"),
        ("G_N_E_M", "natural", "MFA-linear"),
        ("G_M_E_M", "MFA-linear", "MFA-linear"),
    ):
        means = result["cell_means_record_weighted"][cell]
        lines.append(f"|{generation}|{scoring}|{means['sync_c']:.3f}|{means['sync_d']:.3f}|")
    lines.extend([
        "",
        "| 对比 | Sync-C group mean [95% CI] | p(正向) | Sync-D improvement group mean [95% CI] | p(正向) |",
        "|---|---:|---:|---:|---:|",
    ])
    for label, contrast in result["contrasts"].items():
        c = contrast["sync_c"]
        d = contrast["sync_d"]
        lines.append(
            f"|{label}|{c['group_mean']:+.3f} [{c['bootstrap_group_mean_ci95'][0]:+.3f},{c['bootstrap_group_mean_ci95'][1]:+.3f}]|"
            f"{c['sign_flip_p_value_one_sided_positive']:.3g}|{d['group_mean']:+.3f} "
            f"[{d['bootstrap_group_mean_ci95'][0]:+.3f},{d['bootstrap_group_mean_ci95'][1]:+.3f}]|"
            f"{d['sign_flip_p_value_one_sided_positive']:.3g}|"
        )
    natural_swap = result["contrasts"]["mfa_audio_effect_on_natural_video"]
    mfa_swap = result["contrasts"]["mfa_audio_effect_on_mfa_video"]
    audio_main = result["contrasts"]["mfa_scoring_audio_main_effect"]
    lines.extend([
        "",
        f"固定 natural 视频，换 MFA-linear 音轨：组均 ΔSync-C={natural_swap['sync_c']['group_mean']:+.3f}，"
        f"Sync-D 改善={natural_swap['sync_d']['group_mean']:+.3f}。固定 MFA-linear 视频，同样换轨："
        f"组均 ΔSync-C={mfa_swap['sync_c']['group_mean']:+.3f}，"
        f"Sync-D 改善={mfa_swap['sync_d']['group_mean']:+.3f}。"
        f"两种固定视频等权平均的 MFA-linear 评分音轨效应："
        f"ΔSync-C={audio_main['sync_c']['group_mean']:+.3f}，"
        f"Sync-D 改善={audio_main['sync_d']['group_mean']:+.3f}。",
    ])
    if all(
        audio_main[metric]["bootstrap_group_mean_ci95"][1] < 0
        and natural_swap[metric]["bootstrap_group_mean_ci95"][1] < 0
        and mfa_swap[metric]["bootstrap_group_mean_ci95"][0] > 0
        for metric in ("sync_c", "sync_d")
    ):
        lines.extend([
            "",
            "结论：本批 LRS3 / Wav2Lip / MFA-linear 数据不支持 SyncNet 对 MFA-linear 音轨普遍加分。"
            "音轨与生成视频的匹配关系主导分数变化；这不能单凭 SyncNet 分数推出人眼感知的唇形质量，也不能推广到所有 TTS。",
        ])
    lines.extend([
        "",
        "解释重点：`visual_under_natural_audio` 是在自然评分音轨下的视觉生成差异；`scorer_visual_interaction` 衡量这个视觉差异是否随最终评分音轨改成 MFA-linear 而改变。`mfa_scoring_audio_main_effect` 是固定视频后换成 MFA-linear 评分音轨的平均效应。`mfa_audio_effect_on_natural_video` 是自然视频换音轨的单独效应；换音轨不会改变视频帧。",
        "",
    ])
    if excluded:
        lines.extend(["排除的短视频：" + ", ".join(row["sample_id"] for row in excluded) + "。", ""])
    (run_root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run(
    *,
    stage: str,
    run_root: Path,
    max_records: int | None = None,
    sample_ids: Sequence[str] = (),
    local_knn_vc: Path,
) -> dict[str, Any]:
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    protocol_path = run_root / "protocol.json"
    if stage in {"protocol", "all"}:
        protocol = build_protocol(run_root, max_records=max_records, sample_ids=sample_ids)
    else:
        protocol = load_json(protocol_path)
    if stage == "protocol":
        return protocol
    if stage in {"repair", "score", "analysis", "all"}:
        if stage in {"repair", "all"}:
            repair_candidate_audio(protocol, run_root, local_knn_vc=local_knn_vc)
        elif any(not Path(str(row["candidate_audio"])).is_file() for row in protocol["records"]):
            raise FileNotFoundError("candidate audio repair is incomplete")
    if stage in {"score", "analysis", "all"}:
        scores = score_stage(protocol, run_root)
    else:
        scores = None
    if stage in {"analysis", "all"}:
        if scores is None:
            scores = _load_hashed_json(run_root / "02_scores/manifest.json")
        return analysis_stage(protocol, scores, run_root)
    return {"status": "complete", "stage": stage, "record_count": len(protocol["records"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("protocol", "repair", "score", "analysis", "all"), default="all")
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--local-knn-vc", type=Path, default=Path.home() / ".cache/torch/hub/bshall_knn-vc_c616845c4e309e24d5927f15adbdf277a3d65358")
    args = parser.parse_args()
    try:
        result = run(
            stage=args.stage,
            run_root=args.run_root,
            max_records=args.max_records,
            sample_ids=args.sample_id,
            local_knn_vc=args.local_knn_vc.resolve(),
        )
    except Exception as exc:
        failure = {"protocol_id": PROTOCOL_ID, "status": "blocked", "stage": args.stage, "error": str(exc)}
        args.run_root.mkdir(parents=True, exist_ok=True)
        write_json(args.run_root / "failure.json", failure)
        print(json.dumps(failure, ensure_ascii=False, indent=2))
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
