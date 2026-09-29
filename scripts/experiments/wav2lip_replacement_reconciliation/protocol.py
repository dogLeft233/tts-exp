from __future__ import annotations

import hashlib
import pickle
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ReconciliationError,
    bytes_sha256,
    file_sha256,
    load_matrix,
    probe_media,
    read_pcm16_wav,
    require_hash,
    verify_self_hashed_json,
)


def _fixed_json(label: str) -> dict[str, Any]:
    path, expected = config.ROOT_INPUTS[label]
    require_hash(path, expected, label)
    return verify_self_hashed_json(path)


def _ordered_id_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def _index(rows: Sequence[Mapping[str, Any]], key: str = "sample_id") -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row.get(key, ""))
        if not value or value in result:
            raise ReconciliationError(f"duplicate or empty {key}: {value}")
        result[value] = row
    return result


def _cell_index(rows: Sequence[Mapping[str, Any]], key_fields: tuple[str, ...]) -> dict[tuple[str, ...], Mapping[str, Any]]:
    result: dict[tuple[str, ...], Mapping[str, Any]] = {}
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if any(not value for value in key) or key in result:
            raise ReconciliationError(f"duplicate or incomplete cell: {key}")
        result[key] = row
    return result


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config.REPO / path).resolve()


def _asset_registry() -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    assets: list[dict[str, Any]] = []
    seen: dict[str, dict[str, Any]] = {}

    def add(path: Path, expected: str | None, label: str, provenance: str) -> dict[str, Any]:
        resolved = path.resolve()
        key = str(resolved)
        if key in seen:
            existing = seen[key]
            if expected and existing["sha256"] != expected:
                raise ReconciliationError(f"conflicting asset hash: {resolved}")
            return existing
        if not resolved.is_file():
            raise ReconciliationError(f"missing bound asset: {label}: {resolved}")
        actual = file_sha256(resolved)
        if expected and actual != expected:
            raise ReconciliationError(f"asset hash mismatch: {label}: {resolved}: {actual} != {expected}")
        if any(token in key.lower() for token in config.NO_SEALED_MEDIA_TOKENS):
            raise ReconciliationError(f"sealed asset path crossed: {resolved}")
        item = {"path": key, "sha256": actual, "label": label, "provenance": provenance}
        seen[key] = item
        assets.append(item)
        return item

    return assets, {"add": add}  # type: ignore[return-value]


def _parse_score_log(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    confidence = [float(value) for value in re.findall(r"Confidence:\s+([0-9.]+)", text)]
    distance = [float(value) for value in re.findall(r"Min dist:\s+([0-9.]+)", text)]
    offsets = [int(value) for value in re.findall(r"AV offset:\s+(-?\d+)", text)]
    if len(confidence) != 1 or len(distance) != 1 or len(offsets) != 1:
        raise ReconciliationError(f"score log does not contain exactly one C/D/offset set: {path}")
    return {"sync_c": confidence[0], "sync_d": distance[0], "av_offset": offsets[0], "match_count": 1}


def _track_and_matrix(track_path: Path, matrix_path: Path) -> tuple[int, tuple[int, int]]:
    try:
        tracks = pickle.load(track_path.open("rb"))
        matrices = pickle.load(matrix_path.open("rb"))
    except (OSError, EOFError, pickle.PickleError, ValueError) as exc:
        raise ReconciliationError(f"cannot load legacy SyncNet cache: {matrix_path}: {exc}") from exc
    if not isinstance(tracks, list) or not isinstance(matrices, list) or len(tracks) != 1 or len(matrices) != 1:
        raise ReconciliationError(f"legacy cache must contain exactly one track and matrix: {matrix_path}")
    track = tracks[0]
    frames = np.asarray(track.get("track", {}).get("frame", []), dtype=np.int64)
    matrix = np.asarray(matrices[0])
    expected_frames = np.arange(frames.size, dtype=np.int64)
    if frames.size < 1 or not np.array_equal(frames, expected_frames):
        raise ReconciliationError(f"legacy track clock is not zero-start continuous: {track_path}")
    if matrix.ndim != 2 or matrix.shape[1] != config.MATRIX_COLUMNS or matrix.shape[0] < 1 or matrix.shape[0] > frames.size or not np.isfinite(matrix).all():
        raise ReconciliationError(f"legacy matrix/track shape mismatch: {matrix_path}: {matrix.shape}/{frames.size}")
    return int(frames.size), (int(matrix.shape[0]), int(matrix.shape[1]))


def rebuild_distance(visual: np.ndarray, audio: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    visual_array = np.asarray(visual, dtype=np.float32)
    audio_array = np.asarray(audio, dtype=np.float32)
    if visual_array.ndim != 2 or audio_array.ndim != 2 or visual_array.shape != audio_array.shape:
        raise ReconciliationError(f"embedding shapes differ: {visual_array.shape}/{audio_array.shape}")
    padded = np.pad(audio_array, ((config.VSHIFT, config.VSHIFT), (0, 0)))
    rows = []
    for row in range(visual_array.shape[0]):
        candidate = padded[row : row + 2 * config.VSHIFT + 1]
        diff = visual_array[row : row + 1] - candidate
        rows.append(np.sqrt(np.sum((diff + np.float32(epsilon)) ** 2, axis=1, dtype=np.float32)))
    return np.asarray(rows, dtype=np.float32)


def _audio_item(path: Path, expected: str, label: str, add_asset: Any, construction: Any = None) -> tuple[dict[str, Any], bytes, np.ndarray]:
    add_asset(path, expected, label, "manifest_bound")
    raw, values, params = read_pcm16_wav(path)
    metadata = {
        "path": str(path.resolve()),
        "container_sha256": file_sha256(path),
        "decoded_pcm_sha256": bytes_sha256(raw),
        "sample_count": int(values.size),
        "sample_rate": int(params["sample_rate"]),
        "channels": int(params["channels"]),
        "sample_width": int(params["sample_width"]),
    }
    if construction is not None:
        metadata["construction"] = construction
    return metadata, raw, values


def _diff_pcm(left: bytes, right: bytes) -> dict[str, Any]:
    left_values = np.frombuffer(left, dtype="<i2").astype(np.int32)
    right_values = np.frombuffer(right, dtype="<i2").astype(np.int32)
    count = int(min(left_values.size, right_values.size))
    if count:
        difference = left_values[:count] - right_values[:count]
        max_abs = int(np.max(np.abs(difference)))
        rms = float(np.sqrt(np.mean(np.asarray(difference, dtype=np.float64) ** 2)))
    else:
        max_abs = 0
        rms = 0.0
    return {
        "left_sample_count": int(left_values.size),
        "right_sample_count": int(right_values.size),
        "pcm_equal": bool(left == right),
        "max_abs_diff_lsb": max_abs,
        "rms_diff_lsb": rms,
        "compared_sample_count": count,
    }


def _probe_summary(probe: Mapping[str, Any]) -> dict[str, Any]:
    video = probe.get("video", {})
    return {
        "codec_name": video.get("codec_name"),
        "width": video.get("width"),
        "height": video.get("height"),
        "r_frame_rate": video.get("r_frame_rate"),
        "time_base": video.get("time_base"),
        "nb_frames": video.get("nb_frames"),
    }


def _build_audio_audit(
    records: Sequence[Mapping[str, Any]],
    h_audio: Mapping[str, Any],
    s_audio: Mapping[str, Any],
    add_asset: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    h_rows = _index(h_audio.get("rows", []))
    s_rows = _index(s_audio.get("rows", []))
    rows: list[dict[str, Any]] = []
    for record in records:
        sid = str(record["sample_id"])
        h_row = h_rows[sid]
        s_row = s_rows[sid]
        h_arms = {str(item["arm"]): item for item in h_row.get("arms", [])}
        s_arms = s_row.get("arms", {})
        if not isinstance(s_arms, Mapping):
            raise ReconciliationError(f"S audio arm mapping is not a dict: {sid}")
        source_meta: dict[str, Any] = {}
        source_raw: dict[str, bytes] = {}
        generated_meta: dict[str, Any] = {}
        generated_raw: dict[str, bytes] = {}
        natural_path = _resolve(str(record["natural_audio"]["path"]))
        mfa_path = _resolve(str(record["mfa_linear_audio"]["path"]))
        source_meta["N_source"], source_raw["N_source"], _ = _audio_item(natural_path, str(record["natural_audio"]["sha256"]), f"H source natural {sid}", add_asset)
        source_meta["M_source"], source_raw["M_source"], _ = _audio_item(mfa_path, str(record["mfa_linear_audio"]["sha256"]), f"H source MFA {sid}", add_asset)
        for arm in ("N", "N_REPEAT", "BRIDGE_075"):
            item = h_arms.get(arm)
            if not isinstance(item, Mapping):
                raise ReconciliationError(f"missing H audio arm: {sid}/{arm}")
            path = _resolve(str(item["output"]))
            generated_meta[f"H_{arm}"], generated_raw[f"H_{arm}"], _ = _audio_item(path, str(item["output_sha256"]), f"H audio {sid}/{arm}", add_asset, item.get("construction"))
        for arm in ("N", "N_REPEAT", "RT", "MAG", "ENV"):
            item = s_arms.get(arm)
            if not isinstance(item, Mapping):
                raise ReconciliationError(f"missing S audio arm: {sid}/{arm}")
            path = _resolve(str(item["path"]))
            generated_meta[f"S_{arm}"], generated_raw[f"S_{arm}"], _ = _audio_item(path, str(item["container_sha256"]), f"S audio {sid}/{arm}", add_asset, item.get("construction"))
        for name, source_name in (("H_N", "N_source"), ("S_N", "N_source"), ("H_N_REPEAT", "N_source"), ("S_N_REPEAT", "N_source")):
            if generated_raw[name] != source_raw[source_name]:
                raise ReconciliationError(f"natural identity mismatch: {sid}/{name}")
        if generated_raw["H_N"] != generated_raw["S_N"] or generated_raw["H_N_REPEAT"] != generated_raw["S_N_REPEAT"]:
            raise ReconciliationError(f"H/S N identity mismatch: {sid}")
        if source_meta["M_source"]["sample_count"] != generated_meta["H_N"]["sample_count"]:
            raise ReconciliationError(f"N/M length mismatch: {sid}")
        row = {
            "sample_id": sid,
            "source_group": str(record["source_group"]),
            "source": source_meta,
            "H": {key: generated_meta[key] for key in generated_meta if key.startswith("H_")},
            "S": {key: generated_meta[key] for key in generated_meta if key.startswith("S_")},
            "comparisons": {
                "H_BRIDGE_075_vs_S_MAG": _diff_pcm(generated_raw["H_BRIDGE_075"], generated_raw["S_MAG"]),
                "S_RT_vs_S_N": _diff_pcm(generated_raw["S_RT"], generated_raw["S_N"]),
                "H_N_vs_S_N": _diff_pcm(generated_raw["H_N"], generated_raw["S_N"]),
                "H_M_vs_S_M_source": _diff_pcm(source_raw["M_source"], source_raw["M_source"]),
            },
        }
        rows.append(row)
    return rows, {
        "h_pcm_field_note": "H format.pcm_sha256 is treated as a container/file hash; decoded PCM is recomputed independently.",
        "candidate_pair": "H BRIDGE_075 versus S MAG; unequal PCM is retained as a confound, not repaired.",
        "rows": rows,
    }


def _chain_table(h_video: Mapping[str, Any], h_mux: Mapping[str, Any], h_score: Mapping[str, Any], s_video: Mapping[str, Any], s_worker: Mapping[str, Any], s_probe: Mapping[str, Any]) -> list[dict[str, Any]]:
    h_geometry = h_video.get("geometry", {})
    h_command = h_video.get("command", [])
    return [
        {"field": "generation_box", "H": h_geometry.get("mode", "unknown"), "S": "per-frame ROI from frozen boxes", "evidence": "H videos_manifest.geometry; S videos_manifest.boxes/command"},
        {"field": "generation_batch", "H": {"face_det": _command_value(h_command, "--face_det_batch_size"), "wav2lip": _command_value(h_command, "--wav2lip_batch_size")}, "S": s_video.get("batch_size", "unknown"), "evidence": "frozen generation commands"},
        {"field": "generation_checkpoint", "H": h_video.get("checkpoint_sha256"), "S": s_video.get("checkpoint_sha256"), "evidence": "frozen manifests"},
        {"field": "video_codec_and_clock", "H": h_mux.get("video_stream", {}), "S": _probe_summary(s_probe), "evidence": "H mux manifest; S score media probe"},
        {"field": "audio_mux", "H": h_mux.get("audio_stream", {}), "S": {"worker_audio_format": s_worker.get("extraction", {}).get("audio_format", "unknown")}, "evidence": "frozen mux/worker metadata"},
        {"field": "syncnet_crop_and_frontend", "H": "legacy run_pipeline.py pycrop/00000.avi then run_syncnet.py", "S": "fresh worker frame JPEG + audio extraction + forward_lip/forward_aud", "evidence": "H score_log/cache paths; S worker extraction"},
        {"field": "score_support", "H": "FULL legacy matrix includes boundary zero-embedding rows", "S": "matrix rows include padded boundaries; I/U are diagnostic re-summaries", "evidence": "activesd and worker contract"},
        {"field": "embedding_or_distance_precision", "H": "unknown embeddings; cached matrix float32", "S": s_worker.get("dtype", "unknown"), "evidence": "legacy cache boundary versus worker.json"},
        {"field": "unresolved_confound", "H": "generation/crop/codec/legacy SyncNet chain", "S": "ROI/fresh extraction/codec/worker chain", "evidence": "multiple frozen settings change together"},
    ]


def _command_value(command: Sequence[Any], flag: str) -> Any:
    try:
        index = list(command).index(flag)
        return command[index + 1]
    except (ValueError, IndexError):
        return "unknown"


def build_audit(run_id: str) -> dict[str, Any]:
    roots = {label: _fixed_json(label) for label in config.ROOT_INPUTS}
    h_cohort = roots["H/cohort"]
    s_protocol = roots["S/protocol"]
    h_records = list(h_cohort.get("records", []))
    s_records = list(s_protocol.get("records", []))
    if len(h_records) != config.EXPECTED_RECORD_COUNT or len(s_records) != config.EXPECTED_RECORD_COUNT:
        raise ReconciliationError("H/S record count is not exactly 22")
    h_ids = [str(row.get("sample_id")) for row in h_records]
    s_index = _index(s_records)
    if len(set(h_ids)) != len(h_ids) or _ordered_id_hash(h_ids) != config.ORDERED_SAMPLE_ID_SHA256:
        raise ReconciliationError("ordered H sample IDs are not the frozen 22-record cohort")
    records: list[dict[str, Any]] = []
    for h_row in h_records:
        sid = str(h_row["sample_id"])
        s_row = s_index.get(sid)
        if not isinstance(s_row, Mapping) or str(s_row.get("source_group")) != str(h_row.get("source_group")):
            raise ReconciliationError(f"keyed H/S join failed: {sid}")
        for key in ("natural_audio", "mfa_linear_audio", "face_video"):
            if not isinstance(h_row.get(key), Mapping):
                raise ReconciliationError(f"H cohort asset missing: {sid}/{key}")
        records.append(
            {
                "sample_id": sid,
                "source_group": str(h_row["source_group"]),
                "natural_audio": dict(h_row["natural_audio"]),
                "mfa_linear_audio": dict(h_row["mfa_linear_audio"]),
                "face_video": dict(h_row["face_video"]),
                "frame_count": int(s_row["frame_count"]),
                "sample_count": int(s_row["sample_count"]),
                "plus_rows": [int(value) for value in s_row["plus_rows"]],
                "minus_rows": [int(value) for value in s_row["minus_rows"]],
                "u_rows": [int(value) for value in [*s_row["plus_rows"], *s_row["minus_rows"]]],
                "break_frame": s_row.get("break_frame"),
                "break_sample": s_row.get("break_sample"),
            }
        )
    if len({row["source_group"] for row in records}) != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ReconciliationError("source groups are not unique")
    if any(len(row["u_rows"]) != len(set(row["u_rows"])) for row in records):
        raise ReconciliationError("U contains duplicate rows")
    assets, registry = _asset_registry()
    add_asset = registry["add"]
    h_audio = roots["H/audio_manifest"]
    s_audio = roots["S/audio_manifest"]
    audio_audit_rows, audio_audit = _build_audio_audit(records, h_audio, s_audio, add_asset)

    h_scores = roots["H/scores_manifest"]
    h_videos = roots["H/videos_manifest"]
    s_scores = roots["S/scores_manifest"]
    s_videos = roots["S/videos_manifest"]
    h_score_index = _cell_index(h_scores.get("scores", []), ("sample_id", "cell"))
    h_mux_index = _cell_index(h_scores.get("muxes", []), ("sample_id", "cell"))
    s_score_index = _cell_index(s_scores.get("scores", []), ("sample_id", "video_arm", "audio_arm"))
    h_video_index = _index(h_videos.get("rows", []))
    # The S producer writes one manifest row for stage-A arms and one for
    # stage-B arms.  Reconcile them by sample_id before selecting cells.
    s_video_index: dict[str, dict[str, Any]] = {}
    for video_row in s_videos.get("rows", []):
        sid = str(video_row.get("sample_id", ""))
        if not sid:
            raise ReconciliationError("S video row has no sample_id")
        merged = s_video_index.setdefault(sid, {"sample_id": sid, "source_group": str(video_row.get("source_group")), "videos": {}, "cells": {}})
        if merged["source_group"] != str(video_row.get("source_group")):
            raise ReconciliationError(f"S video source_group differs: {sid}")
        for arm, item in video_row.get("videos", {}).items():
            if arm in merged["videos"]:
                raise ReconciliationError(f"duplicate S video arm: {sid}/{arm}")
            merged["videos"][arm] = item
        for cell, item in video_row.get("cells", {}).items():
            if cell in merged["cells"]:
                raise ReconciliationError(f"duplicate S video cell: {sid}/{cell}")
            merged["cells"][cell] = item
    if len(s_video_index) != config.EXPECTED_RECORD_COUNT:
        raise ReconciliationError(f"S video sample count differs: {len(s_video_index)}")
    s_audio_index = _index(s_audio.get("rows", []))
    matrix_bindings: list[dict[str, Any]] = []
    processing_chain: list[dict[str, Any]] | None = None
    probe_cache: dict[str, dict[str, Any]] = {}
    for record in records:
        sid = record["sample_id"]
        h_video_row = h_video_index[sid]
        s_video_row = s_video_index[sid]
        s_audio_row = s_audio_index[sid]
        h_face_path = _resolve(str(h_video_row["face"]))
        add_asset(h_face_path, str(h_video_row["face_sha256"]), f"H face {sid}", "manifest_bound")
        s_n_audio = s_audio_row["arms"]["N"]
        for arm in config.H_VIDEO_ARMS:
            cell = f"V_{arm}/A_N"
            score = h_score_index.get((sid, cell))
            if not isinstance(score, Mapping):
                raise ReconciliationError(f"missing H score cell: {sid}/{cell}")
            reference = str(score["reference"])
            cache_root = config.H_ROOT / "03_scores/syncnet" / reference
            pywork = cache_root / "pywork" / reference
            pycrop = cache_root / "pycrop" / reference
            track_path = pywork / "tracks.pckl"
            matrix_path = pywork / "activesd.pckl"
            add_asset(track_path, None, f"H track {sid}/{arm}", "legacy_cache_locked_at_audit")
            add_asset(matrix_path, None, f"H distance matrix {sid}/{arm}", "legacy_cache_locked_at_audit")
            for sidecar in sorted(pywork.iterdir()):
                if sidecar.is_file():
                    add_asset(sidecar, None, f"H cache sidecar {sid}/{arm}/{sidecar.name}", "legacy_cache_locked_at_audit")
            crop_files = sorted(pycopy for pycopy in pycrop.glob("*.avi") if pycopy.is_file())
            if len(crop_files) != 1:
                raise ReconciliationError(f"H pycrop must contain one AVI: {sid}/{arm}")
            add_asset(crop_files[0], None, f"H pycrop {sid}/{arm}", "legacy_cache_locked_at_audit")
            row_count, shape = _track_and_matrix(track_path, matrix_path)
            matrix_sha = file_sha256(matrix_path)
            media_path = _resolve(str(score["media"]))
            add_asset(media_path, str(score["media_sha256"]), f"H score media {sid}/{arm}", "manifest_bound")
            if str(score.get("syncnet_model_sha256")) != config.SYNCNET_MODEL_SHA256:
                raise ReconciliationError(f"H SyncNet model binding changed: {sid}/{arm}")
            if str(score.get("cell")) != cell:
                raise ReconciliationError(f"H score key differs: {sid}/{arm}")
            score_log = _resolve(str(score["score_log"]))
            add_asset(score_log, None, f"H score log {sid}/{arm}", "manifest_bound")
            log_score = _parse_score_log(score_log)
            if abs(float(score["sync_c"]) - log_score["sync_c"]) > 0.000501 or abs(float(score["sync_d"]) - log_score["sync_d"]) > 0.000501 or int(score["av_offset"]) != int(log_score["av_offset"]):
                raise ReconciliationError(f"H score log/manifest mismatch: {sid}/{arm}")
            h_video_arm = h_video_row["arms"].get(arm)
            if not isinstance(h_video_arm, Mapping):
                raise ReconciliationError(f"H video arm missing: {sid}/{arm}")
            h_video_path = _resolve(str(h_video_arm["output"]))
            add_asset(h_video_path, str(h_video_arm["output_sha256"]), f"H generated video {sid}/{arm}", "manifest_bound")
            # Every selected H matrix is candidate video arm versus natural audio N.
            h_audio_arm = {str(item["arm"]): item for item in h_audio_row(h_audio, sid)}.get("N")
            if not isinstance(h_audio_arm, Mapping):
                raise ReconciliationError(f"H generated audio arm missing: {sid}/{arm}")
            audio_path = _resolve(str(h_audio_arm["output"]))
            add_asset(audio_path, str(h_audio_arm["output_sha256"]), f"H generated audio binding {sid}/{arm}", "manifest_bound")
            mux = h_mux_index.get((sid, cell))
            if not isinstance(mux, Mapping):
                raise ReconciliationError(f"H mux binding missing: {sid}/{cell}")
            if str(mux.get("video_source_sha256")) != str(h_video_arm.get("output_sha256")):
                raise ReconciliationError(f"H mux/video binding differs: {sid}/{arm}")
            if str(mux.get("audio_source_sha256")) != str(h_audio_arm.get("output_sha256")):
                raise ReconciliationError(f"H mux/audio binding differs: {sid}/{arm}")
            if str(h_video_arm.get("face_sha256")) != str(h_video_row.get("face_sha256")):
                raise ReconciliationError(f"H face binding differs: {sid}/{arm}")
            probe = probe_cache.setdefault(str(media_path), probe_media(media_path))
            if str(probe["video"].get("r_frame_rate")) != "25/1":
                raise ReconciliationError(f"H media is not 25 fps: {media_path}")
            binding = {
                "origin": "H",
                "sample_id": sid,
                "source_group": record["source_group"],
                "video_arm": arm,
                "audio_arm": "N",
                "cell": cell,
                "reference": reference,
                "matrix": str(matrix_path.resolve()),
                "matrix_sha256": matrix_sha,
                "matrix_shape": list(shape),
                "track": str(track_path.resolve()),
                "track_sha256": file_sha256(track_path),
                "track_frame_count": row_count,
                "matrix_row_count": int(shape[0]),
                "pycrop": str(crop_files[0].resolve()),
                "pycrop_sha256": file_sha256(crop_files[0]),
                "media": str(media_path.resolve()),
                "media_sha256": file_sha256(media_path),
                "score_log": str(score_log.resolve()),
                "score_log_sha256": file_sha256(score_log),
                "historical_score": {"sync_c": float(score["sync_c"]), "sync_d": float(score["sync_d"]), "av_offset": int(score["av_offset"])},
                "log_score": log_score,
                "probe": _probe_summary(probe),
                "legacy_cache": True,
                "legacy_embeddings_available": False,
            }
            matrix_bindings.append(binding)
            if processing_chain is None:
                s_representative = s_video_row["videos"].get("N")
                s_score = s_score_index.get((sid, "N", "N"))
                if not isinstance(s_representative, Mapping) or not isinstance(s_score, Mapping):
                    raise ReconciliationError(f"processing-chain representative is missing: {sid}")
                s_media = _resolve(str(s_score["media"]))
                s_probe = probe_cache.setdefault(str(s_media), probe_media(s_media))
                processing_chain = _chain_table(h_video_row, mux, score, s_representative, {}, s_probe)
        for arm in config.S_VIDEO_ARMS:
            score = s_score_index.get((sid, arm, "N"))
            if not isinstance(score, Mapping):
                raise ReconciliationError(f"missing S score cell: {sid}/{arm}/N")
            media_path = _resolve(str(score["media"]))
            add_asset(media_path, str(score["media_sha256"]), f"S score media {sid}/{arm}", "manifest_bound")
            matrix_path = _resolve(str(score["matrix"]))
            visual_path = _resolve(str(score["visual"]))
            audio_embedding_path = _resolve(str(score["audio_embedding"]))
            worker_path = _resolve(str(score["worker_result"]))
            add_asset(matrix_path, str(score["matrix_sha256"]), f"S matrix {sid}/{arm}", "manifest_bound")
            add_asset(visual_path, str(score["visual_sha256"]), f"S visual embedding {sid}/{arm}", "manifest_bound")
            add_asset(audio_embedding_path, str(score["audio_embedding_sha256"]), f"S audio embedding {sid}/{arm}", "manifest_bound")
            add_asset(worker_path, str(score["worker_result_sha256"]), f"S worker result {sid}/{arm}", "manifest_bound")
            worker = verify_self_hashed_json(worker_path)
            if worker.get("new_forward") is not True or worker.get("model_sha256") != config.SYNCNET_MODEL_SHA256:
                raise ReconciliationError(f"S worker is not the frozen fresh-forward result: {sid}/{arm}")
            visual = np.load(visual_path, allow_pickle=False)
            audio_embedding = np.load(audio_embedding_path, allow_pickle=False)
            matrix = load_matrix(matrix_path, str(score["matrix_sha256"]), f"S matrix {sid}/{arm}")
            rebuilt = rebuild_distance(visual, audio_embedding)
            if rebuilt.shape != matrix.shape or not np.allclose(rebuilt, matrix, atol=1e-4, rtol=0.0):
                raise ReconciliationError(f"S matrix does not rebuild from embeddings: {sid}/{arm}")
            if list(matrix.shape) != list(score.get("matrix_shape", [])) or worker.get("matrix_sha256") != score.get("matrix_sha256"):
                raise ReconciliationError(f"S matrix binding differs: {sid}/{arm}")
            s_video = s_video_row["videos"].get(arm)
            if not isinstance(s_video, Mapping):
                raise ReconciliationError(f"S video arm missing: {sid}/{arm}")
            video_path = _resolve(str(s_video["output"]))
            add_asset(video_path, str(s_video["output_sha256"]), f"S generated video {sid}/{arm}", "manifest_bound")
            for field in ("boxes", "generation_result", "log"):
                if s_video.get(field):
                    expected = s_video.get(f"{field}_sha256")
                    add_asset(_resolve(str(s_video[field])), str(expected) if expected else None, f"S video {field} {sid}/{arm}", "manifest_bound")
            if str(s_video.get("face_sha256")) != str(h_video_row.get("face_sha256")):
                raise ReconciliationError(f"S face binding differs: {sid}/{arm}")
            if str(score.get("audio_container_sha256")) != str(s_n_audio.get("container_sha256")):
                raise ReconciliationError(f"S score/audio binding differs: {sid}/{arm}")
            if _resolve(str(score.get("audio"))) != _resolve(str(s_n_audio.get("path"))):
                raise ReconciliationError(f"S score/audio path differs: {sid}/{arm}")
            probe = probe_cache.setdefault(str(media_path), probe_media(media_path))
            if str(probe["video"].get("r_frame_rate")) != "25/1":
                raise ReconciliationError(f"S media is not 25 fps: {media_path}")
            if int(s_video.get("frame_count", -1)) != int(record["frame_count"]):
                raise ReconciliationError(f"S generated frame count differs: {sid}/{arm}")
            binding = {
                "origin": "S",
                "sample_id": sid,
                "source_group": record["source_group"],
                "video_arm": arm,
                "audio_arm": "N",
                "cell": f"V_{arm}/A_N",
                "matrix": str(matrix_path.resolve()),
                "matrix_sha256": str(score["matrix_sha256"]),
                "matrix_shape": list(matrix.shape),
                "visual": str(visual_path.resolve()),
                "visual_sha256": str(score["visual_sha256"]),
                "audio_embedding": str(audio_embedding_path.resolve()),
                "audio_embedding_sha256": str(score["audio_embedding_sha256"]),
                "worker_result": str(worker_path.resolve()),
                "worker_result_sha256": str(score["worker_result_sha256"]),
                "media": str(media_path.resolve()),
                "media_sha256": str(score["media_sha256"]),
                "probe": _probe_summary(probe),
                "worker": {key: worker.get(key) for key in ("model_sha256", "device", "new_forward", "dtype", "distance_epsilon", "visual_shape", "audio_embedding_shape", "matrix_shape", "extraction")},
                "legacy_cache": False,
                "legacy_embeddings_available": True,
            }
            matrix_bindings.append(binding)
            if processing_chain is not None and arm == "N":
                s_worker = worker
                processing_chain = _chain_table(
                    h_video_index[sid]["arms"]["N"],
                    next((item for item in h_scores.get("muxes", []) if item.get("sample_id") == sid and item.get("cell") == "V_N/A_N"), {}),
                    h_score_index[(sid, "V_N/A_N")],
                    s_video_row["videos"]["N"],
                    s_worker,
                    probe,
                )
    if len(matrix_bindings) != config.EXPECTED_MATRIX_COUNT:
        raise ReconciliationError(f"matrix binding count differs: {len(matrix_bindings)}")
    if processing_chain is None:
        raise ReconciliationError("processing chain evidence is empty")
    per_sample_lengths: dict[str, list[int]] = {}
    for binding in matrix_bindings:
        per_sample_lengths.setdefault(str(binding["sample_id"]), []).append(int(binding["matrix_shape"][0]))
    for record in records:
        sid = record["sample_id"]
        minimum = min(per_sample_lengths[sid])
        interior = list(range(config.VSHIFT, minimum - config.VSHIFT))
        if not interior or any(row < config.VSHIFT or row >= minimum - config.VSHIFT for row in record["u_rows"]):
            raise ReconciliationError(f"fixed I/U support failed: {sid}")
        record["common_interior_rows"] = interior
        record["common_interior_count"] = len(interior)
        record["u_count"] = len(record["u_rows"])
    for binding in matrix_bindings:
        if binding["origin"] == "H":
            if int(binding["matrix_shape"][0]) > int(binding["track_frame_count"]):
                raise ReconciliationError(f"H matrix exceeds track clock: {binding['cell']}")
        else:
            if int(binding["matrix_shape"][0]) != int(next(row["frame_count"] for row in records if row["sample_id"] == binding["sample_id"])) - 5:
                raise ReconciliationError(f"S matrix frame support differs from frozen frame count: {binding['cell']}")
    audio_by_sample = {str(row["sample_id"]): row for row in audio_audit_rows}
    input_audit = {
        "schema_version": 1,
        "stage_id": "input_audit",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "classification": "retrospective_cache_only_diagnostic",
        "record_count": len(records),
        "source_group_count": len({row["source_group"] for row in records}),
        "join": {"key": "sample_id", "order": "H cohort order", "record_count": len(records), "source_group_count": len({row["source_group"] for row in records}), "discovery_23_record_cohort_included": False},
        "audio": audio_audit,
        "records": [{**row, "audio_comparison": audio_by_sample[row["sample_id"]]["comparisons"]} for row in records],
        "processing_chain": processing_chain,
        "matrix_count": {"H": config.EXPECTED_H_MATRIX_COUNT, "S": config.EXPECTED_S_MATRIX_COUNT, "total": config.EXPECTED_MATRIX_COUNT},
        "endpoint_support_preflight": {"minimum_u_count": min(row["u_count"] for row in records), "minimum_common_interior_count": min(row["common_interior_count"] for row in records), "all_u_in_common_interior": True},
        "new_media_count": 0,
        "model_forward_count": 0,
        "sealed_media_accessed": False,
    }
    root_bindings = {
        label: {"path": str(path.resolve()), "sha256": expected}
        for label, (path, expected) in config.ROOT_INPUTS.items()
    }
    code_bindings = {str(path.relative_to(config.REPO)): file_sha256(path) for path in config.CODE_FILES}
    spec_bindings = {str(path.relative_to(config.REPO)): file_sha256(path) for path in config.SPEC_FILES}
    protocol = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "run_id": run_id,
        "status": "frozen",
        "classification": "retrospective_cache_only_diagnostic",
        "budget": {"gpu": 0, "network": 0, "new_media": 0, "model_forward": 0, "training": False},
        "root_inputs": root_bindings,
        "code_bindings": code_bindings,
        "spec_bindings": spec_bindings,
        "frozen_config": config.frozen_config(),
        "input_audit_sha256": "__filled_by_runner__",
        "records": records,
        "matrix_bindings": matrix_bindings,
        "asset_bindings": assets,
        "selected_cells": {
            "H": [f"{row['sample_id']}::{arm}::N" for row in records for arm in config.H_VIDEO_ARMS],
            "S": [f"{row['sample_id']}::{arm}::N" for row in records for arm in config.S_VIDEO_ARMS],
        },
        "processing_chain": processing_chain,
        "legacy_evidence_boundary": "H activesd caches contain distance matrices and tracks, not original embeddings; H is cache-consistency evidence only.",
        "discovery_cohort": {"record_count": 23, "included": False},
        "training_authorized": False,
        "generalization_established": False,
        "historical_gate_repaired": False,
    }
    return {"protocol": protocol, "input_audit": input_audit}


def h_audio_row(manifest: Mapping[str, Any], sample_id: str) -> list[Mapping[str, Any]]:
    for row in manifest.get("rows", []):
        if str(row.get("sample_id")) == sample_id:
            return [item for item in row.get("arms", []) if isinstance(item, Mapping)]
    raise ReconciliationError(f"missing H audio row: {sample_id}")


def load_bound_matrices(protocol: Mapping[str, Any]) -> dict[tuple[str, str, str, str], np.ndarray]:
    result: dict[tuple[str, str, str, str], np.ndarray] = {}
    for binding in protocol.get("matrix_bindings", []):
        key = (str(binding["origin"]), str(binding["sample_id"]), str(binding["video_arm"]), str(binding["audio_arm"]))
        if key in result:
            raise ReconciliationError(f"duplicate bound matrix: {key}")
        result[key] = load_matrix(Path(str(binding["matrix"])), str(binding["matrix_sha256"]), f"bound matrix {key}")
    if len(result) != config.EXPECTED_MATRIX_COUNT:
        raise ReconciliationError(f"bound matrix count differs: {len(result)}")
    return result
