"""Build and analyse the blinded human-assessment package.

Sync items use two silent videos plus one separate lossless A0 WAV.  This
prevents an embedded, possibly different, video audio track from changing a
human comparison.  Operator-only joins live in ``private_mapping.json``;
public rows contain no sample, source, seed, condition, or private path.
"""

from __future__ import annotations

import csv
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import (
    ProtocolError,
    copy_atomic,
    csv_read,
    executable_path,
    ffprobe_json,
    file_sha256,
    read_self_hashed_json,
    run_command,
    write_self_hashed_json,
)

SYNC_TEMPLATE_FIELDS = ("rater_id", "anonymous_id", "sync_preference", "artifact_preference", "notes")
QUALITY_TEMPLATE_FIELDS = (
    "rater_id",
    "anonymous_id",
    "clarity_left_1_5",
    "clarity_right_1_5",
    "naturalness_left_1_5",
    "naturalness_right_1_5",
    "notes",
)


def _anon_id(kind: str, index: int) -> str:
    return f"{kind.lower()}_{index:04d}"


def _resolve(value: str | Path) -> Path:
    target = Path(value)
    if not target.is_absolute():
        target = config.REPO / target
    return target.resolve()


def _generation_map(generation: Mapping[str, Any]) -> dict[tuple[int, str, str, int], Mapping[str, Any]]:
    return {
        (int(row["id"]), str(row["source"]), str(row["driver_condition"]), int(row["seed"])): row
        for row in generation.get("videos", [])
        if isinstance(row, Mapping) and row.get("stage") == "B_GENERATION"
    }


def _audio_map(audio: Mapping[str, Any]) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    return {
        (int(row["sample_id"]), str(row["source"]), str(row["condition"])): row
        for row in audio.get("records", [])
        if isinstance(row, Mapping)
    }


def _shuffle_sides(rng: np.random.Generator, left: Mapping[str, Any], right: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    return (dict(left), dict(right)) if int(rng.integers(0, 2)) == 0 else (dict(right), dict(left))


def _video_summary(path: Path) -> dict[str, Any]:
    probe = ffprobe_json(path)
    streams = [stream for stream in probe.get("streams", []) if isinstance(stream, Mapping)]
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1 or any(stream.get("codec_type") == "audio" for stream in streams):
        raise ProtocolError(f"blind video must contain one video stream and no audio: {path}")
    stream = videos[0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not isinstance(rate, str) or "/" not in rate:
        raise ProtocolError(f"blind video has no auditable frame rate: {path}")
    numerator, denominator = rate.split("/", 1)
    try:
        fps = float(numerator) / float(denominator)
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        raise ProtocolError(f"blind video frame rate is malformed: {path}") from exc
    if abs(fps - config.FPS) > 0.01:
        raise ProtocolError(f"blind video is not {config.FPS} fps: {path}")
    duration = stream.get("duration") or probe.get("format", {}).get("duration")
    try:
        if duration is not None and float(duration) <= 0:
            raise ProtocolError(f"blind video has no positive duration: {path}")
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"blind video duration is malformed: {path}") from exc
    return {"file_sha256": file_sha256(path), "bytes": int(path.stat().st_size), "fps": fps, "stream": dict(stream)}


def _make_silent_video(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise ProtocolError(f"source B video is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp.mp4")
    temporary.unlink(missing_ok=True)
    command = (
        str(executable_path(config.FFMPEG, "ffmpeg")),
        "-y",
        "-v",
        "error",
        "-i",
        str(source),
        "-map",
        "0:v:0",
        "-an",
        "-c:v",
        "copy",
        "-copyts",
        "-avoid_negative_ts",
        "disabled",
        str(temporary),
    )
    run_command(command)
    temporary.replace(destination)
    summary = _video_summary(destination)
    summary.update({"source_path": str(source), "source_sha256": file_sha256(source)})
    return summary


def _copy_lossless_audio(source: Path, destination: Path) -> dict[str, Any]:
    if not source.is_file():
        raise ProtocolError(f"audio is missing: {source}")
    copy_atomic(source, destination)
    return {"path": str(destination), "file_sha256": file_sha256(destination), "source_path": str(source), "source_sha256": file_sha256(source)}


def _materialize_sync_media(root: Path, anonymous_id: str, left_source: Path | None, right_source: Path | None, audio_source: Path | None) -> tuple[dict[str, Any], bool]:
    media_root = root / "public" / "sync" / anonymous_id
    result: dict[str, Any] = {"left_public": None, "right_public": None, "audio_public": None, "left_sha256": None, "right_sha256": None, "audio_sha256": None}
    if left_source is None or right_source is None or audio_source is None:
        return result, False
    try:
        left = _make_silent_video(left_source, media_root / "left.mp4")
        right = _make_silent_video(right_source, media_root / "right.mp4")
        audio = _copy_lossless_audio(audio_source, media_root / "audio.wav")
        result.update({"left_public": str(media_root / "left.mp4"), "right_public": str(media_root / "right.mp4"), "audio_public": str(media_root / "audio.wav"), "left_sha256": left["file_sha256"], "right_sha256": right["file_sha256"], "audio_sha256": audio["file_sha256"]})
        return result, True
    except (OSError, ProtocolError, ValueError):
        for path in (media_root / "left.mp4", media_root / "right.mp4", media_root / "audio.wav"):
            path.unlink(missing_ok=True)
        return result, False


def _materialize_quality_media(root: Path, anonymous_id: str, left_source: Path | None, right_source: Path | None) -> tuple[dict[str, Any], bool]:
    result: dict[str, Any] = {"left_public": None, "right_public": None, "left_sha256": None, "right_sha256": None}
    if left_source is None or right_source is None:
        return result, False
    media_root = root / "public" / "quality" / anonymous_id
    try:
        left = _copy_lossless_audio(left_source, media_root / "left.wav")
        right = _copy_lossless_audio(right_source, media_root / "right.wav")
        result.update({"left_public": str(media_root / "left.wav"), "right_public": str(media_root / "right.wav"), "left_sha256": left["file_sha256"], "right_sha256": right["file_sha256"]})
        return result, True
    except (OSError, ProtocolError, ValueError):
        (media_root / "left.wav").unlink(missing_ok=True)
        (media_root / "right.wav").unlink(missing_ok=True)
        return result, False


def _input_bindings(paths: config.RunPaths) -> dict[str, Any]:
    audio_path = paths.audio / "manifest.json"
    generation_path = paths.generation / "manifest.json"
    return {
        "audio_manifest_sha256": file_sha256(audio_path) if audio_path.is_file() else None,
        "generation_manifest_sha256": file_sha256(generation_path) if generation_path.is_file() else None,
        "protocol_id": config.PROTOCOL_ID,
        "counts": {"sync_formal": config.EXPECTED_SYNC_PAIRS, "quality_formal": config.EXPECTED_QUALITY_PAIRS},
    }


def _package_cache_valid(path: Path, paths: config.RunPaths) -> bool:
    try:
        package = read_self_hashed_json(path)
        if package.get("protocol_id") != config.PROTOCOL_ID or package.get("blind_schema_version") != 3:
            return False
        if package.get("input_bindings") != _input_bindings(paths) or package.get("status") != "COMPLETE":
            return False
        if len(package.get("sync_pairs", [])) != config.EXPECTED_SYNC_PAIRS + 10 or len(package.get("quality_pairs", [])) != config.EXPECTED_QUALITY_PAIRS + 5:
            return False
        for row in (*package["sync_pairs"], *package["quality_pairs"]):
            for key in ("left_public", "right_public"):
                if not isinstance(row.get(key), str) or not Path(row[key]).is_file():
                    return False
            if row.get("audio_public") is not None and not Path(str(row["audio_public"])).is_file():
                return False
        return True
    except (OSError, ProtocolError, TypeError, ValueError, KeyError):
        return False


def _ensure_template(path: Path, fields: Sequence[str]) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerow(fields)


def build_perception_package(paths: config.RunPaths, assets: Mapping[str, Any], audio_manifest: Mapping[str, Any], generation_manifest: Mapping[str, Any] | None) -> dict[str, Any]:
    root = paths.perception
    root.mkdir(parents=True, exist_ok=True)
    package_path = root / "package.json"
    if _package_cache_valid(package_path, paths):
        return read_self_hashed_json(package_path)
    rng = np.random.Generator(np.random.PCG64(config.BOOTSTRAP_SEED))
    groups = {int(row["sample_id"]): str(row["source_group"]) for row in assets.get("records", []) if isinstance(row, Mapping)}
    audios = _audio_map(audio_manifest)
    generated = _generation_map(generation_manifest or {})
    private: list[dict[str, Any]] = []
    sync_public: list[dict[str, Any]] = []
    quality_public: list[dict[str, Any]] = []
    media_complete = True

    def add_sync(sample_id: int, source: str, seed: int, driver: str, *, hidden: bool, source_pair_id: str | None, anonymous_id: str) -> None:
        nonlocal media_complete
        base = generated.get((sample_id, source, "A0", seed), {})
        altered = generated.get((sample_id, source, driver, seed), {})
        left, right = _shuffle_sides(rng, {"role": "V0", "path": base.get("path")}, {"role": f"V{driver}", "path": altered.get("path")})
        left_path = _resolve(str(left["path"])) if left.get("path") else None
        right_path = _resolve(str(right["path"])) if right.get("path") else None
        audio_record = audios.get((sample_id, source, "A0"), {})
        audio_path = _resolve(str(audio_record["path"])) if audio_record.get("path") else None
        media, available = _materialize_sync_media(root, anonymous_id, left_path, right_path, audio_path)
        media_complete = media_complete and available
        private.append({"anonymous_id": anonymous_id, "kind": "sync", "sample_id": sample_id, "source_group": groups.get(sample_id), "source": source, "seed": seed, "driver": driver, "left_role": left["role"], "right_role": right["role"], "left_source_path": str(left_path) if left_path else None, "right_source_path": str(right_path) if right_path else None, "audio_source_path": str(audio_path) if audio_path else None, "audio_pcm_sha256": audio_record.get("pcm_sha256"), "hidden_repeat": hidden, "source_pair_id": source_pair_id, "media": media})
        sync_public.append({"anonymous_id": anonymous_id, **media, "audio_start_seconds": 0.0, "prompt": "使用同一条声音，判断哪一个视频的嘴部动作与声音更同步。允许平局。", "available": available})

    sync_index = 0
    for sample_id in config.SAMPLE_IDS:
        for source in ("N", "T"):
            for seed in config.SEEDS:
                for driver in ("NOISE", "DENOISE"):
                    sync_index += 1
                    add_sync(sample_id, source, seed, driver, hidden=False, source_pair_id=None, anonymous_id=_anon_id("sync", sync_index))
    for source_row in list(sync_public)[:10]:
        sync_index += 1
        original = next(item for item in private if item["anonymous_id"] == source_row["anonymous_id"])
        add_sync(int(original["sample_id"]), str(original["source"]), int(original["seed"]), str(original["driver"]), hidden=True, source_pair_id=source_row["anonymous_id"], anonymous_id=_anon_id("sync", sync_index))

    quality_index = 0

    def add_quality(sample_id: int, source: str, driver: str, *, hidden: bool, source_pair_id: str | None, anonymous_id: str) -> None:
        nonlocal media_complete
        left, right = _shuffle_sides(rng, {"role": "A0", "path": audios.get((sample_id, source, "A0"), {}).get("path")}, {"role": f"A{driver}", "path": audios.get((sample_id, source, driver), {}).get("path")})
        left_path = _resolve(str(left["path"])) if left.get("path") else None
        right_path = _resolve(str(right["path"])) if right.get("path") else None
        media, available = _materialize_quality_media(root, anonymous_id, left_path, right_path)
        media_complete = media_complete and available
        private.append({"anonymous_id": anonymous_id, "kind": "quality", "sample_id": sample_id, "source_group": groups.get(sample_id), "source": source, "driver": driver, "left_role": left["role"], "right_role": right["role"], "left_source_path": str(left_path) if left_path else None, "right_source_path": str(right_path) if right_path else None, "hidden_repeat": hidden, "source_pair_id": source_pair_id, "media": media})
        quality_public.append({"anonymous_id": anonymous_id, **media, "prompt": "只听两段声音，分别评价清晰度和自然度。不要猜测生成方式。", "available": available})

    for sample_id in config.SAMPLE_IDS:
        for source in ("N", "T"):
            for driver in ("NOISE", "DENOISE"):
                quality_index += 1
                add_quality(sample_id, source, driver, hidden=False, source_pair_id=None, anonymous_id=_anon_id("quality", quality_index))
    for source_row in list(quality_public)[:5]:
        quality_index += 1
        original = next(item for item in private if item["anonymous_id"] == source_row["anonymous_id"])
        add_quality(int(original["sample_id"]), str(original["source"]), str(original["driver"]), hidden=True, source_pair_id=source_row["anonymous_id"], anonymous_id=_anon_id("quality", quality_index))

    sync_public = [sync_public[int(index)] for index in rng.permutation(len(sync_public))]
    quality_public = [quality_public[int(index)] for index in rng.permutation(len(quality_public))]
    sync_positions = {row["anonymous_id"]: index for index, row in enumerate(sync_public)}
    quality_positions = {row["anonymous_id"]: index for index, row in enumerate(quality_public)}
    for row in private:
        row["public_order"] = (sync_positions if row["kind"] == "sync" else quality_positions)[row["anonymous_id"]]

    mapping_path = root / "private_mapping.json"
    write_self_hashed_json(mapping_path, {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "seed": config.BOOTSTRAP_SEED, "mapping": private, "operator_only": True})
    _ensure_template(root / "sync_ratings_template.csv", SYNC_TEMPLATE_FIELDS)
    _ensure_template(root / "quality_ratings_template.csv", QUALITY_TEMPLATE_FIELDS)
    result = {
        "schema_version": 1,
        "blind_schema_version": 3,
        "protocol_id": config.PROTOCOL_ID,
        "stage": "06_perception",
        "status": "COMPLETE" if media_complete else "PARTIAL_MEDIA",
        "media_status": "COMPLETE" if media_complete else "PARTIAL_MEDIA",
        "sync_pairs": sync_public,
        "quality_pairs": quality_public,
        "sync_pair_count": config.EXPECTED_SYNC_PAIRS,
        "quality_pair_count": config.EXPECTED_QUALITY_PAIRS,
        "sync_total_with_hidden": len(sync_public),
        "quality_total_with_hidden": len(quality_public),
        "hidden_sync_count": 10,
        "hidden_quality_count": 5,
        "order_seed": config.BOOTSTRAP_SEED,
        "input_bindings": _input_bindings(paths),
        "private_mapping_sha256": file_sha256(mapping_path),
        "templates": {"sync": str(root / "sync_ratings_template.csv"), "quality": str(root / "quality_ratings_template.csv")},
        "no_condition_provider_labels_in_prompts": True,
        "same_a0_pcm_for_sync_sides": True,
        "embedded_video_audio_disabled": True,
        "public_identity_fields_removed": True,
    }
    return write_self_hashed_json(package_path, result)


def _rating_optional(raw: Any, label: str) -> float | None:
    value = str(raw).strip().lower()
    if value in {"", "na", "n/a", "unjudgeable", "unjudgable", "cannot_judge"}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ProtocolError(f"invalid rating {label}: {raw}") from exc
    if not np.isfinite(number) or not 1.0 <= number <= 5.0:
        raise ProtocolError(f"rating outside 1..5: {label}")
    return number


def _preference(raw: Any, label: str) -> float | None:
    value = str(raw).strip().lower()
    if value in {"left", "a", "1"}:
        return 1.0
    if value in {"right", "b", "2"}:
        return 0.0
    if value in {"tie", "equal", "0.5"}:
        return 0.5
    if value in {"", "na", "n/a", "unjudgeable", "unjudgable", "cannot_judge"}:
        return None
    raise ProtocolError(f"invalid preference {label}: {raw}")


def _target_value(value: float | None, left_role: str, intervention_role: str) -> float | None:
    if value is None:
        return None
    return value if left_role == intervention_role else 1.0 - value


def _bootstrap_group_values(values: Mapping[str, float], *, metric: str) -> dict[str, Any]:
    labels = sorted(values)
    if len(labels) != len(config.SAMPLE_IDS):
        return {"status": "INCOMPLETE", "metric": metric, "group_count": len(labels), "expected_group_count": len(config.SAMPLE_IDS)}
    array = np.asarray([float(values[label]) for label in labels], dtype=np.float64)
    indices = np.random.Generator(np.random.PCG64(config.BOOTSTRAP_SEED)).integers(0, len(labels), size=(config.BOOTSTRAP_DRAWS, len(labels)), dtype=np.int64)
    draws = array[indices].mean(axis=1, dtype=np.float64)
    quantile = lambda probability: float(np.quantile(draws, probability, method="linear"))
    tail = 0.05 / (2 * 6)
    mean = float(array.mean())
    return {"status": "COMPLETE", "metric": metric, "mean": mean, "winner_rate": mean if metric.endswith("win_probability") else None, "excess_over_0.5": mean - 0.5 if metric.endswith("win_probability") else None, "ci95": [quantile(0.025), quantile(0.975)], "ci99_166667_bonferroni": [quantile(tail), quantile(1 - tail)], "group_labels": labels, "group_means": {label: float(value) for label, value in zip(labels, array, strict=True)}, "draws": config.BOOTSTRAP_DRAWS, "seed": config.BOOTSTRAP_SEED, "quantile_method": "linear", "source_group_equal_weight": True}


def _validate_rating_rows(rows: Sequence[Mapping[str, Any]], mapping: Mapping[str, Any], *, kind: str) -> tuple[list[dict[str, str]], dict[str, Mapping[str, Any]], set[str], set[str]]:
    mapping_by_id = {str(row["anonymous_id"]): row for row in mapping.get("mapping", []) if isinstance(row, Mapping) and row.get("kind") == kind}
    formal = {key for key, row in mapping_by_id.items() if not row.get("hidden_repeat")}
    hidden = {key for key, row in mapping_by_id.items() if row.get("hidden_repeat")}
    seen: set[tuple[str, str]] = set()
    normalized: list[dict[str, str]] = []
    for index, raw in enumerate(rows):
        row = {str(key): str(value) for key, value in raw.items()}
        rater = row.get("rater_id", "").strip()
        anonymous = row.get("anonymous_id", "").strip()
        if not rater or not anonymous:
            raise ProtocolError(f"{kind} rating row {index} has no rater_id/anonymous_id")
        if anonymous not in mapping_by_id:
            raise ProtocolError(f"{kind} rating references unknown anonymous_id: {anonymous}")
        if (rater, anonymous) in seen:
            raise ProtocolError(f"duplicate {kind} rating row: {(rater, anonymous)}")
        seen.add((rater, anonymous))
        if kind == "sync":
            _preference(row.get("sync_preference", ""), f"{anonymous}/sync")
            _preference(row.get("artifact_preference", ""), f"{anonymous}/artifact")
        else:
            for field in ("clarity_left_1_5", "clarity_right_1_5", "naturalness_left_1_5", "naturalness_right_1_5"):
                _rating_optional(row.get(field, ""), f"{anonymous}/{field}")
        normalized.append(row)
    return normalized, mapping_by_id, formal, hidden


def _hidden_consistency(rows: Sequence[Mapping[str, str]], mapping: Mapping[str, Mapping[str, Any]], *, kind: str) -> dict[str, Any]:
    matches = 0
    comparisons = 0
    rows_by_key = {(str(row["rater_id"]), str(row["anonymous_id"])): row for row in rows}
    for hidden_id, hidden in mapping.items():
        if not hidden.get("hidden_repeat"):
            continue
        source_id = str(hidden.get("source_pair_id"))
        source = mapping.get(source_id)
        if source is None:
            continue
        hidden_row = rows_by_key.get((str(next((row["rater_id"] for row in rows if str(row["anonymous_id"]) == hidden_id), "")), hidden_id))
        if hidden_row is None:
            continue
        rater = str(hidden_row["rater_id"])
        source_row = rows_by_key.get((rater, source_id))
        if source_row is None:
            continue
        if kind == "sync":
            current = _preference(hidden_row.get("sync_preference", ""), f"{hidden_id}/sync")
            original = _preference(source_row.get("sync_preference", ""), f"{source_id}/sync")
            current = _target_value(current, str(hidden["left_role"]), f"V{hidden['driver']}")
            original = _target_value(original, str(source["left_role"]), f"V{source['driver']}")
            if current is None or original is None:
                continue
            same = abs(current - original) <= 1e-12
        else:
            def delta(row: Mapping[str, str], item: Mapping[str, Any]) -> float | None:
                left = _rating_optional(row.get("clarity_left_1_5", ""), "hidden clarity left")
                right = _rating_optional(row.get("clarity_right_1_5", ""), "hidden clarity right")
                if left is None or right is None:
                    return None
                return left - right if item.get("left_role") == f"A{item['driver']}" else right - left
            current = delta(hidden_row, hidden)
            original = delta(source_row, source)
            if current is None or original is None:
                continue
            same = current == 0.0 or original == 0.0 or current * original > 0.0
        comparisons += 1
        matches += int(same)
    return {"status": "ASSESSED" if comparisons else "NOT_ASSESSED", "comparison_count": comparisons, "consistent_count": matches, "consistency_rate": None if not comparisons else float(matches / comparisons), "hidden_repeats_excluded_from_formal": True}


def _analyze_ratings(path: Path, mapping: Mapping[str, Any], *, kind: str) -> dict[str, Any]:
    rows = csv_read(path) if path.is_file() else []
    empty_status = "PERCEPTION_NOT_ASSESSED" if kind == "sync" else "QUALITY_NOT_ASSESSED"
    if not rows:
        return {"status": empty_status, "rating_row_count": 0, "complete_rater_count": 0, "reason": f"{path.name} is empty; no human result is inferred", "hidden_repeats_excluded": True}
    normalized, mapping_by_id, formal_ids, hidden_ids = _validate_rating_rows(rows, mapping, kind=kind)
    by_rater: dict[str, dict[str, Mapping[str, str]]] = {}
    for row in normalized:
        by_rater.setdefault(row["rater_id"], {})[row["anonymous_id"]] = row
    complete = sorted(rater for rater, items in by_rater.items() if formal_ids.issubset(items))
    hidden_consistency = _hidden_consistency(normalized, mapping_by_id, kind=kind)
    if len(complete) < 3:
        return {"status": "PERCEPTION_INSUFFICIENT" if kind == "sync" else "QUALITY_INSUFFICIENT", "rating_row_count": len(rows), "rater_count": len(by_rater), "complete_rater_count": len(complete), "reason": "at least three common raters must complete every formal pair", "hidden_repeats_excluded": True, "hidden_consistency": hidden_consistency}
    metric_names = ("sync_win_probability", "artifact_win_probability") if kind == "sync" else ("clarity_delta", "naturalness_delta")
    pair_values: list[dict[str, Any]] = []
    for anonymous in sorted(formal_ids):
        item = mapping_by_id[anonymous]
        values: dict[str, list[float]] = {metric: [] for metric in metric_names}
        for rater in complete:
            row = by_rater[rater][anonymous]
            if kind == "sync":
                preference = _target_value(_preference(row.get("sync_preference", ""), f"{anonymous}/sync"), str(item["left_role"]), f"V{item['driver']}")
                artifact = _target_value(_preference(row.get("artifact_preference", ""), f"{anonymous}/artifact"), str(item["left_role"]), f"V{item['driver']}")
                if preference is not None:
                    values["sync_win_probability"].append(preference)
                if artifact is not None:
                    values["artifact_win_probability"].append(artifact)
            else:
                lc = _rating_optional(row.get("clarity_left_1_5", ""), f"{anonymous}/clarity_left")
                rc = _rating_optional(row.get("clarity_right_1_5", ""), f"{anonymous}/clarity_right")
                ln = _rating_optional(row.get("naturalness_left_1_5", ""), f"{anonymous}/naturalness_left")
                rn = _rating_optional(row.get("naturalness_right_1_5", ""), f"{anonymous}/naturalness_right")
                if lc is not None and rc is not None:
                    values["clarity_delta"].append(lc - rc if item["left_role"] == f"A{item['driver']}" else rc - lc)
                if ln is not None and rn is not None:
                    values["naturalness_delta"].append(ln - rn if item["left_role"] == f"A{item['driver']}" else rn - ln)
        pair = {"anonymous_id": anonymous, "source_group": item.get("source_group"), "source": item.get("source"), "driver": item.get("driver"), "rater_count": len(complete)}
        for metric in metric_names:
            pair[metric] = None if not values[metric] else float(np.mean(values[metric], dtype=np.float64))
            pair[f"{metric}_valid_count"] = len(values[metric])
        pair_values.append(pair)
    source_level: list[dict[str, Any]] = []
    by_group: dict[tuple[str, str, str], dict[str, list[float]]] = {}
    for pair in pair_values:
        key = (str(pair["source_group"]), str(pair["source"]), str(pair["driver"]))
        target = by_group.setdefault(key, {metric: [] for metric in metric_names})
        for metric in metric_names:
            if pair[metric] is not None:
                target[metric].append(float(pair[metric]))
    for key, values in sorted(by_group.items()):
        row = {"source_group": key[0], "source": key[1], "driver": key[2], "pair_count": max((len(values[metric]) for metric in metric_names), default=0)}
        row.update({metric: None if not values[metric] else float(np.mean(values[metric], dtype=np.float64)) for metric in metric_names})
        source_level.append(row)
    bootstrap: list[dict[str, Any]] = []
    for source in ("N", "T"):
        for driver in ("NOISE", "DENOISE"):
            for metric in metric_names:
                values = {row["source_group"]: float(row[metric]) for row in source_level if row["source"] == source and row["driver"] == driver and row[metric] is not None}
                bootstrap.append({"source": source, "driver": driver, **_bootstrap_group_values(values, metric=metric)})
    return {"status": "COMPLETE", "rating_row_count": len(rows), "rater_count": len(by_rater), "complete_rater_count": len(complete), "formal_pair_count": len(formal_ids), "hidden_pair_count": len(hidden_ids), "pair_level": pair_values, "source_level": source_level, "bootstrap": bootstrap, "hidden_consistency": hidden_consistency, "hidden_repeats_excluded": True, "estimand": "average qualifying raters within pair, then source_group equal-weight bootstrap", "human_results_are_not_inferred_from_syncnet": True}


def perception_analyze_stage(paths: config.RunPaths) -> dict[str, Any]:
    package = read_self_hashed_json(paths.perception / "package.json")
    mapping = read_self_hashed_json(paths.perception / "private_mapping.json")
    sync = _analyze_ratings(paths.perception / "sync_ratings_template.csv", mapping, kind="sync")
    quality = _analyze_ratings(paths.perception / "quality_ratings_template.csv", mapping, kind="quality")
    status = "COMPLETE" if package.get("status") == "COMPLETE" else "PARTIAL_MEDIA"
    return write_self_hashed_json(paths.perception / "analysis.json", {"schema_version": 1, "protocol_id": config.PROTOCOL_ID, "stage": "06_perception_analysis", "status": status, "sync": sync, "quality": quality, "package_sha256": file_sha256(paths.perception / "package.json"), "human_results_are_not_inferred_from_syncnet": True})
