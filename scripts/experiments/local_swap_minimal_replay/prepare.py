from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .audio import quarter_swap, read_pcm16, slice_pcm, write_pcm16
from .common import DiagnosticError, canonical_json_sha256, file_sha256, verify_self_hashed_json, write_self_hashed_json
from .media import decode_video_frames, encode_ffv1, lock_source_crop, mux_pcm, strict_mux, verify_media


def _history_rows(history: Mapping[str, Any]) -> tuple[dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]], dict[str, Mapping[str, Any]]]:
    audio_rows = {str(row["sample_id"]): row for row in history["audio"].get("rows", [])}
    video_rows = {str(row["sample_id"]): row for row in history["videos"].get("rows", [])}
    cohort_rows = {str(row["sample_id"]): row for row in history["cohort"].get("records", [])}
    return cohort_rows, audio_rows, video_rows


def _audio_arm(audio_row: Mapping[str, Any], arm: str) -> Path:
    for row in audio_row.get("arms", []):
        if str(row.get("arm")) == arm:
            return Path(str(row["output"])).resolve()
    raise DiagnosticError(f"history audio arm missing: {audio_row.get('sample_id')}/{arm}")


def _cell_row(
    *,
    sample_id: str,
    source_group: str,
    family: str,
    cell: str,
    video_kind: str,
    audio_kind: str,
    media: Path,
    media_meta: Mapping[str, Any],
    video_source: Path,
    audio_source: Path,
    frame_count: int,
    audio_sample_count: int,
    frame_map: list[int],
    audio_sample_range: list[int],
    score_mode: str,
    notes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "ready",
        "sample_id": sample_id,
        "source_group": source_group,
        "family": family,
        "cell": cell,
        "video_kind": video_kind,
        "audio_kind": audio_kind,
        "media": str(media.resolve()),
        "media_sha256": file_sha256(media),
        "video_source": str(video_source.resolve()),
        "video_source_sha256": file_sha256(video_source),
        "audio_source": str(audio_source.resolve()),
        "audio_source_sha256": file_sha256(audio_source),
        "frame_count": int(frame_count),
        "audio_sample_count": int(audio_sample_count),
        "frame_map": [int(value) for value in frame_map],
        "frame_map_sha256": canonical_json_sha256([int(value) for value in frame_map]),
        "audio_sample_range": [int(value) for value in audio_sample_range],
        "score_mode": score_mode,
        "media_meta": dict(media_meta),
        "notes": dict(notes or {}),
    }


def _blocked_cell(sample_id: str, source_group: str, family: str, cell: str, error: Exception) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "blocked",
        "sample_id": sample_id,
        "source_group": source_group,
        "family": family,
        "cell": cell,
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _deferred_cell(sample_id: str, source_group: str, family: str, cell: str, reason: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "deferred",
        "sample_id": sample_id,
        "source_group": source_group,
        "family": family,
        "cell": cell,
        "reason": reason,
    }


def _write_audio(path: Path, values: np.ndarray) -> dict[str, Any]:
    return write_pcm16(path, np.asarray(values, dtype="<i2"))


def _prepare_a(paths: config.RunPaths, sample_id: str, source_group: str, video_path: Path, n_path: Path, s_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    g_n = paths.media / sample_id / "A_G_N.mkv"
    g_s = paths.media / sample_id / "A_G_S.mkv"
    for cell, audio_path, output in (("G_N", n_path, g_n), ("G_S", s_path, g_s)):
        meta = strict_mux(video_path, audio_path, output, paths.media / sample_id / f"{cell}.mux.log")
        frame_count = int(meta["video"]["frame_count"])
        audio_count = int(meta["audio_sample_count"])
        rows.append(_cell_row(sample_id=sample_id, source_group=source_group, family="A", cell=cell, video_kind="G", audio_kind="N" if cell == "G_N" else "S", media=output, media_meta=meta, video_source=video_path, audio_source=audio_path, frame_count=frame_count, audio_sample_count=audio_count, frame_map=list(range(frame_count)), audio_sample_range=[0, audio_count], score_mode="official_pipeline", notes={"old_generated_video": True, "full_history_audio": True}))
    return rows


def _prepare_b(paths: config.RunPaths, sample_id: str, source_group: str, face_video: Path, n_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from scripts.experiments.lrs3_real_video_local_timing import media as real_media

    rows: list[dict[str, Any]] = []
    natural, natural_meta = read_pcm16(n_path)
    track, crops, crop_meta = lock_source_crop(face_video)
    frame_count = len(crops)
    audio_frames = natural.size // config.SAMPLES_PER_FRAME
    k = min(frame_count, audio_frames) // 4
    if k < config.MIN_K:
        raise DiagnosticError(f"known-pairing control has insufficient support: k={k}")
    original_frames = crops[: 4 * k]
    swapped_frames, frame_mapping = quarter_swap(np.asarray(original_frames), k)
    original_audio = natural[: 4 * k * config.SAMPLES_PER_FRAME]
    swapped_audio, audio_mapping = quarter_swap(original_audio, k * config.SAMPLES_PER_FRAME)
    assets = paths.assets / sample_id / "B"
    assets.mkdir(parents=True, exist_ok=True)
    audio_paths = {
        "N0": assets / "N0.wav",
        "S0": assets / "S0.wav",
    }
    audio_meta = {"N0": _write_audio(audio_paths["N0"], original_audio), "S0": _write_audio(audio_paths["S0"], swapped_audio)}
    video_paths = {"R": assets / "R.mkv", "Rs": assets / "Rs.mkv"}
    video_meta = {"R": encode_ffv1(original_frames, video_paths["R"]), "Rs": encode_ffv1(swapped_frames, video_paths["Rs"])}
    frame_maps = {"R": list(range(4 * k)), "Rs": [int(value) for value in frame_mapping["output_to_input"]]}
    audio_maps = {"N0": list(range(4 * k * config.SAMPLES_PER_FRAME)), "S0": [int(value) for value in audio_mapping["output_to_input"]]}
    support = {
        "sample_id": sample_id,
        "source_group": source_group,
        "face_video": str(face_video.resolve()),
        "face_video_sha256": file_sha256(face_video),
        "source_track": track,
        "crop": crop_meta,
        "F": int(frame_count),
        "L": int(natural.size),
        "audio_frames": int(audio_frames),
        "k": int(k),
        "frame_count_used": int(4 * k),
        "audio_sample_count_used": int(4 * k * config.SAMPLES_PER_FRAME),
        "chunk_frame_ranges": [[int(i * k), int((i + 1) * k)] for i in range(4)],
        "chunk_audio_ranges": [[int(i * k * config.SAMPLES_PER_FRAME), int((i + 1) * k * config.SAMPLES_PER_FRAME)] for i in range(4)],
        "joint_swap_order": [0, 2, 1, 3],
        "frame_maps": frame_maps,
        "audio_maps": {"N0": audio_maps["N0"], "S0": audio_maps["S0"]},
        "audio_meta": audio_meta,
        "video_meta": video_meta,
        "common_support": [0, int(4 * k * config.SAMPLES_PER_FRAME)],
        "source_pcm_sha256": natural_meta["pcm_sha256"],
    }
    for video_kind in ("R", "Rs"):
        for audio_kind in ("N0", "S0"):
            cell = f"{video_kind}_{audio_kind}"
            output = paths.media / sample_id / f"B_{cell}.mkv"
            mux = real_media.mux_pcm(video_paths[video_kind], audio_paths[audio_kind], output)
            expected_frames = original_frames if video_kind == "R" else swapped_frames
            expected_pcm = original_audio.tobytes() if audio_kind == "N0" else swapped_audio.tobytes()
            verification = verify_media(output, expected_frames, expected_pcm)
            rows.append(_cell_row(sample_id=sample_id, source_group=source_group, family="B", cell=cell, video_kind=video_kind, audio_kind=audio_kind, media=output, media_meta={"mux": mux, "verification": verification, "crop_video": video_meta[video_kind]}, video_source=video_paths[video_kind], audio_source=audio_paths[audio_kind], frame_count=4 * k, audio_sample_count=4 * k * config.SAMPLES_PER_FRAME, frame_map=frame_maps[video_kind], audio_sample_range=[0, 4 * k * config.SAMPLES_PER_FRAME], score_mode="fixed_forward", notes={"N0_is_not_history_N": True, "S0_is_not_history_S": True, "joint_swap_order": [0, 2, 1, 3]}))
    return rows, support


def _prepare_c(paths: config.RunPaths, sample_id: str, source_group: str, generated_crop_video: Path, n_path: Path, s_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    natural, _ = read_pcm16(n_path)
    swapped, _ = read_pcm16(s_path)
    generated_frames = decode_video_frames(generated_crop_video)
    frame_count = min(len(generated_frames), natural.size // config.SAMPLES_PER_FRAME, swapped.size // config.SAMPLES_PER_FRAME)
    if frame_count < config.WINDOW_FRAMES + 2:
        raise DiagnosticError(f"fixed generated video has insufficient support: {frame_count}")
    frames = generated_frames[:frame_count]
    audio_count = frame_count * config.SAMPLES_PER_FRAME
    assets = paths.assets / sample_id / "C"
    assets.mkdir(parents=True, exist_ok=True)
    video_path = assets / "Gc.mkv"
    video_meta = encode_ffv1(frames, video_path)
    audio_paths = {"Nc": assets / "Nc.wav", "Sc": assets / "Sc.wav"}
    audio_meta = {"Nc": _write_audio(audio_paths["Nc"], natural[:audio_count]), "Sc": _write_audio(audio_paths["Sc"], swapped[:audio_count])}
    support = {
        "sample_id": sample_id,
        "source_group": source_group,
        "generated_crop_video": str(generated_crop_video.resolve()),
        "generated_crop_video_sha256": file_sha256(generated_crop_video),
        "frame_count_original": len(generated_frames),
        "frame_count_used": frame_count,
        "audio_sample_count_used": audio_count,
        "absolute_time_support": [0, audio_count],
        "video_frame_hashes": video_meta["frame_hashes"],
        "audio_meta": audio_meta,
        "video_meta": video_meta,
        "fixed_same_video_frames": True,
        "independent_detector_per_audio": False,
    }
    for cell, audio_kind in (("Gc_Nc", "Nc"), ("Gc_Sc", "Sc")):
        output = paths.media / sample_id / f"C_{cell}.mkv"
        mux = mux_pcm(video_path, audio_paths[audio_kind], output)
        verification = verify_media(output, frames, (natural if audio_kind == "Nc" else swapped)[:audio_count].tobytes())
        rows.append(_cell_row(sample_id=sample_id, source_group=source_group, family="C", cell=cell, video_kind="Gc", audio_kind=audio_kind, media=output, media_meta={"mux": mux, "verification": verification, "video_meta": video_meta}, video_source=video_path, audio_source=audio_paths[audio_kind], frame_count=frame_count, audio_sample_count=audio_count, frame_map=list(range(frame_count)), audio_sample_range=[0, audio_count], score_mode="fixed_forward", notes={"same_generated_video_frames": True, "history_N_or_S_exact_slice": True}))
    return rows, support


def materialize_c_from_a_pipeline(
    paths: config.RunPaths,
    history: Mapping[str, Any],
    media_manifest: Mapping[str, Any],
    score_rows: Mapping[tuple[str, str, str], Mapping[str, Any]],
) -> dict[str, Any]:
    """Replace deferred C cells with the crop selected by A/G_N's official run."""
    cohort_rows, audio_rows, video_rows = _history_rows(history)
    support = verify_self_hashed_json(paths.support)
    replacements: dict[tuple[str, str, str], dict[str, Any]] = {}
    for sample_id in config.SAMPLE_IDS:
        source = cohort_rows[sample_id]
        audio_row = audio_rows[sample_id]
        video_row = video_rows[sample_id]
        source_group = str(source["source_group"])
        try:
            a_score = score_rows.get((sample_id, "A", "G_N"))
            if not isinstance(a_score, Mapping) or a_score.get("status") != "complete":
                raise DiagnosticError("A/G_N official score is unavailable; C cannot lock its crop")
            crop_video = Path(str(a_score["scored_media"])).resolve()
            expected_hash = str(a_score.get("scored_media_sha256", ""))
            if not crop_video.is_file() or file_sha256(crop_video) != expected_hash:
                raise DiagnosticError("A/G_N selected crop changed before C materialization")
            score_json = Path(str(a_score.get("official_worker", ""))).resolve().parent / "score.json"
            if not score_json.is_file():
                raise DiagnosticError("A/G_N score identity is missing")
            n_path = _audio_arm(audio_row, "N")
            s_path = _audio_arm(audio_row, "LOCAL_SWAP")
            c_rows, c_support = _prepare_c(paths, sample_id, source_group, crop_video, n_path, s_path)
            generated_source = Path(str(video_row["arms"]["LOCAL_SWAP"]["output"])).resolve()
            c_support.update(
                {
                    "source_generated_video": str(generated_source),
                    "source_generated_video_sha256": file_sha256(generated_source),
                    "a_pipeline_source": {
                        "sample_id": sample_id,
                        "family": "A",
                        "cell": "G_N",
                        "score_json": str(score_json),
                        "score_json_sha256": file_sha256(score_json),
                        "scored_media": str(crop_video),
                        "scored_media_sha256": expected_hash,
                        "pipeline_track_count": a_score.get("pipeline_track_count"),
                        "selection_rule": "first official G/N pipeline crop, matching A score cell",
                    },
                    "fixed_same_pipeline_crop_frames": True,
                }
            )
            support["samples"][sample_id]["C"] = {"status": "ready", **c_support}
            replacements.update({(sample_id, "C", str(row["cell"])): row for row in c_rows})
        except Exception as exc:
            support["samples"][sample_id]["C"] = {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}
            support.setdefault("failures", []).append({"sample_id": sample_id, "family": "C", "error": str(exc)})
            for cell in ("Gc_Nc", "Gc_Sc"):
                replacements[(sample_id, "C", cell)] = _blocked_cell(sample_id, source_group, "C", cell, exc)
    replaced: list[dict[str, Any]] = []
    for row in media_manifest.get("cells", []):
        key = (str(row.get("sample_id")), str(row.get("family")), str(row.get("cell")))
        replaced.append(replacements.get(key, dict(row)))
    support["status"] = "complete" if not support.get("failures") else "blocked"
    support["cell_count"] = len(replaced)
    write_self_hashed_json(paths.support, support)
    manifest = dict(media_manifest)
    manifest["status"] = "complete" if len(replaced) == config.EXPECTED_NEW_CELL_COUNT and support["status"] == "complete" else "blocked"
    manifest["cells"] = replaced
    manifest["cell_count"] = len(replaced)
    manifest["support_sha256"] = file_sha256(paths.support)
    write_self_hashed_json(paths.media / "manifest.json", manifest)
    return manifest


def prepare_media(paths: config.RunPaths, history: Mapping[str, Any], selected_ids: tuple[str, ...], *, defer_c: bool = True) -> dict[str, Any]:
    cohort_rows, audio_rows, video_rows = _history_rows(history)
    all_cells: list[dict[str, Any]] = []
    support: dict[str, Any] = {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "samples": {}, "failures": []}
    for sample_id in selected_ids:
        source = cohort_rows[sample_id]
        audio_row = audio_rows[sample_id]
        video_row = video_rows[sample_id]
        source_group = str(source["source_group"])
        n_path = _audio_arm(audio_row, "N")
        s_path = _audio_arm(audio_row, "LOCAL_SWAP")
        generated = Path(str(video_row["arms"]["LOCAL_SWAP"]["output"])).resolve()
        face_video = Path(str(source["face_video"]["path"])).resolve()
        sample_support: dict[str, Any] = {"sample_id": sample_id, "source_group": source_group, "A": {}, "B": {}, "C": {}}
        try:
            rows = _prepare_a(paths, sample_id, source_group, generated, n_path, s_path)
            all_cells.extend(rows)
            sample_support["A"] = {"status": "ready", "generated_video": str(generated), "N": str(n_path), "S": str(s_path)}
        except Exception as exc:
            sample_support["A"] = {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}
            for cell in ("G_N", "G_S"):
                all_cells.append(_blocked_cell(sample_id, source_group, "A", cell, exc))
            support["failures"].append({"sample_id": sample_id, "family": "A", "error": str(exc)})
        try:
            rows, b_support = _prepare_b(paths, sample_id, source_group, face_video, n_path)
            all_cells.extend(rows)
            sample_support["B"] = {"status": "ready", **b_support}
        except Exception as exc:
            sample_support["B"] = {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}
            for cell in ("R_N0", "R_S0", "Rs_N0", "Rs_S0"):
                all_cells.append(_blocked_cell(sample_id, source_group, "B", cell, exc))
            support["failures"].append({"sample_id": sample_id, "family": "B", "error": str(exc)})
        if defer_c:
            sample_support["C"] = {"status": "deferred", "reason": "awaiting_A_G_N_pipeline_crop"}
            for cell in ("Gc_Nc", "Gc_Sc"):
                all_cells.append(_deferred_cell(sample_id, source_group, "C", cell, "awaiting_A_G_N_pipeline_crop"))
        else:
            try:
                rows, c_support = _prepare_c(paths, sample_id, source_group, generated, n_path, s_path)
                all_cells.extend(rows)
                sample_support["C"] = {"status": "ready", **c_support}
            except Exception as exc:
                sample_support["C"] = {"status": "blocked", "error_type": type(exc).__name__, "error": str(exc)}
                for cell in ("Gc_Nc", "Gc_Sc"):
                    all_cells.append(_blocked_cell(sample_id, source_group, "C", cell, exc))
                support["failures"].append({"sample_id": sample_id, "family": "C", "error": str(exc)})
        support["samples"][sample_id] = sample_support
    support["status"] = "deferred" if defer_c and not support["failures"] else ("complete" if not support["failures"] else "blocked")
    support["expected_cell_count"] = config.EXPECTED_NEW_CELL_COUNT
    support["cell_count"] = len(all_cells)
    write_self_hashed_json(paths.support, support)
    manifest = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "deferred" if defer_c and len(all_cells) == config.EXPECTED_NEW_CELL_COUNT and not support["failures"] else ("complete" if len(all_cells) == config.EXPECTED_NEW_CELL_COUNT and not support["failures"] else "blocked"),
        "expected_cell_count": config.EXPECTED_NEW_CELL_COUNT,
        "cell_count": len(all_cells),
        "cells": all_cells,
        "support_sha256": file_sha256(paths.support),
        "zero_wav2lip_inference": True,
        "zero_tts_generation": True,
        "zero_bridge_rescoring": True,
    }
    write_self_hashed_json(paths.media / "manifest.json", manifest)
    return manifest
