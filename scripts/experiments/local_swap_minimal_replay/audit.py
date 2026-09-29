from __future__ import annotations

import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .audio import local_swap, read_pcm16
from .common import DiagnosticError, bytes_sha256, file_sha256, write_self_hashed_json
from .media import decode_pcm16, ffprobe_streams, video_bitstream


def parse_syncnet_log(path: Path) -> list[dict[str, float | int]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    confidence = re.findall(r"Confidence:\s+([-+]?\d+(?:\.\d+)?)", text)
    distance = re.findall(r"Min dist:\s+([-+]?\d+(?:\.\d+)?)", text)
    offset = re.findall(r"AV offset:\s+(-?\d+)", text)
    if not (len(confidence) == len(distance) == len(offset)):
        raise DiagnosticError(f"ambiguous SyncNet log: {path}")
    return [{"sync_c": float(c), "sync_d": float(d), "av_offset": int(o)} for c, d, o in zip(confidence, distance, offset, strict=True)]


def _arm_map(row: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    arms = {str(item.get("arm")): item for item in row.get("arms", [])}
    return arms


def _codec(path: Path) -> str:
    streams = ffprobe_streams(path)
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    if len(videos) != 1:
        raise DiagnosticError(f"expected exactly one video stream: {path}")
    return str(videos[0].get("codec_name", ""))


def audit_history(history: Mapping[str, Any], output_path: Path | None = None) -> dict[str, Any]:
    cohort = history["cohort"]
    audio_manifest = history["audio"]
    video_manifest = history["videos"]
    scores_manifest = history["scores"]
    audio_rows = {str(row["sample_id"]): row for row in audio_manifest.get("rows", [])}
    video_rows = {str(row["sample_id"]): row for row in video_manifest.get("rows", [])}
    score_rows = {(str(row["sample_id"]), str(row["cell"])): row for row in scores_manifest.get("scores", [])}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    c_n: list[float] = []
    c_s: list[float] = []
    d_n: list[float] = []
    d_s: list[float] = []
    for source in cohort.get("records", []):
        sample_id = str(source["sample_id"])
        errors: list[str] = []
        audio_row = audio_rows.get(sample_id)
        video_row = video_rows.get(sample_id)
        checks: dict[str, Any] = {}
        if not isinstance(audio_row, Mapping) or not isinstance(video_row, Mapping):
            errors.append("manifest_row_missing")
        else:
            arms = _arm_map(audio_row)
            video_arm = video_row.get("arms", {}).get("LOCAL_SWAP")
            if "N" not in arms or "LOCAL_SWAP" not in arms or not isinstance(video_arm, Mapping):
                errors.append("LOCAL_SWAP_binding_missing")
            else:
                try:
                    natural, natural_meta = read_pcm16(Path(str(arms["N"]["output"])))
                    swapped, swapped_meta = read_pcm16(Path(str(arms["LOCAL_SWAP"]["output"])))
                    expected_swap, boundaries = local_swap(natural)
                    checks["natural_sha256_matches_manifest"] = natural_meta["sha256"] == str(arms["N"].get("output_sha256"))
                    checks["local_swap_sha256_matches_manifest"] = swapped_meta["sha256"] == str(arms["LOCAL_SWAP"].get("output_sha256"))
                    checks["local_swap_pcm_recomputed"] = bool(np.array_equal(swapped, expected_swap))
                    checks["sample_count_unchanged"] = int(swapped.size) == int(natural.size) == int(source.get("natural_sample_count", -1))
                    checks["video_audio_binding"] = str(video_arm.get("audio_sha256")) == str(arms["LOCAL_SWAP"].get("output_sha256"))
                    checks["audio_hashes"] = {
                        "N": natural_meta["sha256"],
                        "LOCAL_SWAP": swapped_meta["sha256"],
                        "swap_pcm_sha256": bytes_sha256(np.asarray(expected_swap, dtype="<i2").tobytes()),
                        "bound_boundaries": boundaries,
                    }
                    if not all(value is True for key, value in checks.items() if key not in {"audio_hashes"}):
                        errors.append("pcm_or_audio_binding_mismatch")
                    generated_video = Path(str(video_arm["output"]))
                    if not generated_video.is_file() or file_sha256(generated_video) != str(video_arm.get("output_sha256")):
                        errors.append("generated_video_hash_mismatch")
                    if str(video_arm.get("face_sha256")) != str(source["face_video"].get("sha256")):
                        errors.append("generated_video_face_binding_mismatch")
                    checks["generated_video_hash"] = file_sha256(generated_video) if generated_video.is_file() else None
                except (OSError, ValueError, DiagnosticError, KeyError) as exc:
                    errors.append(f"media_read_error:{type(exc).__name__}:{exc}")
                    natural = swapped = None
            for cell, tag in (("V_LOCAL_SWAP/A_N", "N"), ("V_LOCAL_SWAP/A_LOCAL_SWAP", "S")):
                score_row = score_rows.get((sample_id, cell))
                score_check: dict[str, Any] = {"manifest_row": score_row is not None}
                if not isinstance(score_row, Mapping):
                    errors.append(f"score_row_missing:{cell}")
                    continue
                media = Path(str(score_row.get("media", "")))
                log_path = Path(str(score_row.get("score_log", "")))
                try:
                    parsed = parse_syncnet_log(log_path)
                    score_check["log_values"] = parsed
                    score_check["single_log_record"] = len(parsed) == 1
                    score_check["manifest_log_match"] = len(parsed) == 1 and all(
                        abs(float(parsed[0][key]) - float(score_row[key])) <= (0.001 if key != "av_offset" else 0)
                        for key in ("sync_c", "sync_d", "av_offset")
                    )
                    score_check["media_hash_match"] = media.is_file() and file_sha256(media) == str(score_row.get("media_sha256"))
                    expected_audio_path = Path(str(_arm_map(audio_row)["N" if tag == "N" else "LOCAL_SWAP"]["output"]))
                    score_check["decoded_audio_match"] = media.is_file() and decode_pcm16(media) == decode_pcm16(expected_audio_path)
                    generated = Path(str(video_row["arms"]["LOCAL_SWAP"]["output"]))
                    if media.is_file() and generated.is_file():
                        codec = _codec(generated)
                        score_check["video_elementary_stream_match"] = video_bitstream(media, codec) == video_bitstream(generated, codec)
                    else:
                        score_check["video_elementary_stream_match"] = False
                    if not all(value is True for key, value in score_check.items() if key not in {"log_values"}):
                        errors.append(f"score_media_or_log_mismatch:{cell}")
                    if len(parsed) == 1:
                        if tag == "N":
                            c_n.append(float(parsed[0]["sync_c"]))
                            d_n.append(float(parsed[0]["sync_d"]))
                        else:
                            c_s.append(float(parsed[0]["sync_c"]))
                            d_s.append(float(parsed[0]["sync_d"]))
                except (OSError, ValueError, DiagnosticError, KeyError) as exc:
                    errors.append(f"score_audit_error:{cell}:{type(exc).__name__}:{exc}")
                checks[cell] = score_check
        row = {
            "sample_id": sample_id,
            "source_group": str(source.get("source_group", "")),
            "passed": not errors,
            "errors": errors,
            "checks": checks,
        }
        records.append(row)
        failures.extend({"sample_id": sample_id, "error": error} for error in errors)
    pair_count = min(len(c_n), len(c_s), len(d_n), len(d_s))
    c_gap = [c_n[index] - c_s[index] for index in range(pair_count)]
    d_gap = [d_s[index] - d_n[index] for index in range(pair_count)]
    payload = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "audit": "history_local_swap_read_only_recompute",
        "status": "complete" if not failures and len(records) == config.EXPECTED_HISTORY_COUNT else "blocked",
        "record_count": len(records),
        "expected_record_count": config.EXPECTED_HISTORY_COUNT,
        "passed_count": sum(bool(row["passed"]) for row in records),
        "failure_count": len(failures),
        "records": records,
        "failures": failures,
        "aggregate": {
            "pair_count": pair_count,
            "c_local_swap_minus_natural_mean": float(np.mean(c_gap)) if c_gap else None,
            "d_natural_minus_local_swap_mean": float(np.mean(d_gap)) if d_gap else None,
            "natural_better_both_count": sum(c > 0 and d > 0 for c, d in zip(c_gap, d_gap, strict=True)),
            "c_gap": c_gap,
            "d_gap": d_gap,
        },
        "claims": {
            "history_record_verified": not failures and len(records) == config.EXPECTED_HISTORY_COUNT,
            "new_forward_replay_not_yet_done": True,
        },
    }
    if output_path is not None:
        write_self_hashed_json(output_path, payload)
        payload["artifact_sha256"] = json_hash(output_path)
    return payload


def json_hash(path: Path) -> str:
    import json

    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["artifact_sha256"])
