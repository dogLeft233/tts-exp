from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any

from . import config
from .common import (
    CalibrationError,
    assert_not_sealed,
    canonical_json_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .control import local_swap_pcm16, read_pcm16
from .protocol import load_history

CONTROL_CELLS = (
    "V_N/A_N",
    "V_N_REPEAT/A_N",
    "V_LOCAL_SWAP/A_N",
    "V_LOCAL_SWAP/A_LOCAL_SWAP",
)
CONTROL_VIDEO_ARMS = (config.NATURAL_ARM, config.REPEAT_ARM, config.LOCAL_SWAP_ARM)
EXPECTED_WAV2LIP_BINDINGS = {
    "wav2lip_python": config.WAV2LIP_PYTHON_SHA256,
    "wav2lip_checkpoint": config.WAV2LIP_CHECKPOINT_SHA256,
    "wav2lip_inference": config.WAV2LIP_INFERENCE_SHA256,
}


def _check(
    check: str,
    status: str,
    evidence: str,
    *,
    sample_id: str | None = None,
    cell: str | None = None,
    expected: Any = None,
    actual: Any = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {"check": check, "status": status, "evidence": evidence}
    if sample_id is not None:
        row["sample_id"] = sample_id
    if cell is not None:
        row["cell"] = cell
    if expected is not None:
        row["expected"] = expected
    if actual is not None:
        row["actual"] = actual
    return row


def _required_file(path: Path, description: str) -> None:
    assert_not_sealed(path)
    if not path.is_file():
        raise CalibrationError(f"{description} is missing: {path}")


def _stream_info(path: Path) -> list[dict[str, Any]]:
    result = subprocess.run(
        [str(config.FFPROBE), "-v", "error", "-show_streams", "-of", "json", str(path)],
        capture_output=True,
        check=True,
    )
    payload = json.loads(result.stdout.decode("utf-8"))
    streams = payload.get("streams")
    if not isinstance(streams, list):
        raise CalibrationError(f"ffprobe returned no stream list: {path}")
    return streams


def _decode_pcm16(path: Path) -> bytes:
    result = subprocess.run(
        [
            str(config.FFMPEG),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(config.SAMPLE_RATE),
            "-ac",
            "1",
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return bytes(result.stdout)


def _video_elementary(path: Path, codec_name: str) -> bytes:
    output_format = {"h264": "h264", "hevc": "hevc", "mpeg4": "m4v", "vp8": "ivf", "vp9": "ivf"}.get(codec_name)
    if output_format is None:
        raise CalibrationError(f"unsupported video codec for elementary-stream audit: {codec_name}")
    result = subprocess.run(
        [
            str(config.FFMPEG),
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-c:v",
            "copy",
            "-f",
            output_format,
            "pipe:1",
        ],
        capture_output=True,
        check=True,
    )
    return bytes(result.stdout)


def _seconds(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _stream_duration(stream: Mapping[str, Any]) -> float | None:
    value = _seconds(stream.get("duration"))
    if value is not None:
        return value
    tags = stream.get("tags")
    text = tags.get("DURATION") if isinstance(tags, Mapping) else None
    if not isinstance(text, str):
        return None
    parts = text.split(":")
    if len(parts) != 3:
        return None
    try:
        hours, minutes, seconds = (float(part) for part in parts)
    except ValueError:
        return None
    value = hours * 3600.0 + minutes * 60.0 + seconds
    return value if math.isfinite(value) else None


def _validate_timeline(video: Mapping[str, Any], audio: Mapping[str, Any], path: Path) -> dict[str, Any]:
    video_start = _seconds(video.get("start_time"))
    audio_start = _seconds(audio.get("start_time"))
    video_duration = _stream_duration(video)
    audio_duration = _stream_duration(audio)
    if video_start is None or audio_start is None or video_duration is None or audio_duration is None:
        raise CalibrationError(f"mux timeline metadata is incomplete: {path}")
    if abs(video_start) > 1e-3 or abs(audio_start) > 1e-3:
        raise CalibrationError(f"mux streams do not start at zero: {path}")
    if abs(video_start - audio_start) > 1.0 / 25.0:
        raise CalibrationError(f"mux stream start times differ: {path}")
    frame_rate = str(video.get("r_frame_rate", ""))
    try:
        rate = float(Fraction(frame_rate))
    except (ValueError, ZeroDivisionError):
        raise CalibrationError(f"mux video frame rate is invalid: {path}") from None
    if abs(rate - 25.0) > 1e-6:
        raise CalibrationError(f"mux video frame rate is not 25 fps: {path}")
    return {
        "video_start_time": video_start,
        "audio_start_time": audio_start,
        "video_duration": video_duration,
        "audio_duration": audio_duration,
        "duration_difference": video_duration - audio_duration,
        "frame_rate": rate,
        "video_nb_frames": video.get("nb_frames"),
        "time_base": video.get("time_base"),
    }


def audit_mux(video: Path, audio: Path, mux: Path) -> dict[str, Any]:
    """Verify the actual muxed bytes and timeline, rather than trusting sidecars."""

    _required_file(video, "source video")
    _required_file(audio, "source audio")
    _required_file(mux, "muxed media")
    streams = _stream_info(mux)
    videos = [stream for stream in streams if stream.get("codec_type") == "video"]
    audios = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if len(videos) != 1 or len(audios) != 1:
        raise CalibrationError(f"mux must contain one video and one audio stream: {mux}")
    audio_stream = audios[0]
    if (
        audio_stream.get("codec_name") != "pcm_s16le"
        or int(audio_stream.get("sample_rate", 0)) != config.SAMPLE_RATE
        or int(audio_stream.get("channels", 0)) != config.PCM_CHANNELS
    ):
        raise CalibrationError(f"mux audio format is not 16-kHz mono PCM16: {mux}")
    expected_pcm = _decode_pcm16(audio)
    actual_pcm = _decode_pcm16(mux)
    if actual_pcm != expected_pcm:
        raise CalibrationError(f"mux decoded PCM differs from specified audio: {mux}")
    codec_name = str(videos[0].get("codec_name", ""))
    expected_video = _video_elementary(video, codec_name)
    actual_video = _video_elementary(mux, codec_name)
    if actual_video != expected_video:
        raise CalibrationError(f"mux video elementary stream differs from source: {mux}")
    source_streams = _stream_info(video)
    source_videos = [stream for stream in source_streams if stream.get("codec_type") == "video"]
    if len(source_videos) != 1:
        raise CalibrationError(f"source video stream count is invalid: {video}")
    timeline = _validate_timeline(videos[0], audio_stream, mux)
    return {
        "mux_sha256": file_sha256(mux),
        "source_video_sha256": file_sha256(video),
        "source_audio_sha256": file_sha256(audio),
        "decoded_audio_pcm_sha256": hashlib.sha256(actual_pcm).hexdigest(),
        "source_video_elementary_sha256": hashlib.sha256(expected_video).hexdigest(),
        "mux_video_elementary_sha256": hashlib.sha256(actual_video).hexdigest(),
        "audio_pcm_exact": True,
        "video_elementary_exact": True,
        "stream": {
            "video_codec": codec_name,
            "audio_codec": audio_stream.get("codec_name"),
            "audio_sample_rate": audio_stream.get("sample_rate"),
            "audio_channels": audio_stream.get("channels"),
        },
        "timeline": timeline,
    }


def _parse_syncnet_log(path: Path) -> dict[str, float | int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = {
        "sync_c": re.findall(r"Confidence:\s+([-+]?\d+(?:\.\d+)?)", text),
        "sync_d": re.findall(r"Min dist:\s+([-+]?\d+(?:\.\d+)?)", text),
        "av_offset": re.findall(r"AV offset:\s+(-?\d+)", text),
    }
    if any(len(values) != 1 for values in matches.values()):
        raise CalibrationError(f"SyncNet log has missing or ambiguous metrics: {path}")
    result: dict[str, float | int] = {
        "sync_c": float(matches["sync_c"][0]),
        "sync_d": float(matches["sync_d"][0]),
        "av_offset": int(matches["av_offset"][0]),
    }
    if not math.isfinite(result["sync_c"]) or not math.isfinite(result["sync_d"]):
        raise CalibrationError(f"SyncNet log has non-finite metrics: {path}")
    return result


def _find_row(rows: list[Mapping[str, Any]], sample_id: str, name: str) -> Mapping[str, Any]:
    found = [row for row in rows if str(row.get("sample_id")) == sample_id]
    if len(found) != 1:
        raise CalibrationError(f"historical {name} row is missing or duplicated: {sample_id}")
    return found[0]


def _audit_audio(history: Mapping[str, Any], items: list[dict[str, Any]]) -> None:
    cohort = history["cohort"]
    manifest = history["audio"]
    for record in cohort["records"]:
        sample_id = str(record["sample_id"])
        row = _find_row(list(manifest["rows"]), sample_id, "audio")
        arms = {str(item.get("arm")): item for item in row.get("arms", [])}
        for arm in (config.NATURAL_ARM, config.REPEAT_ARM, config.LOCAL_SWAP_ARM):
            item_name = f"audio_sidecar/{arm}"
            arm_row = arms.get(arm)
            try:
                if arm_row is None:
                    raise CalibrationError(f"audio arm is missing: {sample_id}/{arm}")
                output = Path(str(arm_row.get("output")))
                sidecar = output.with_suffix(".json")
                sidecar_payload = verify_self_hashed_json(sidecar)
                natural = Path(str(record["natural_audio"]["path"]))
                natural_values, _natural_meta = read_pcm16(natural)
                actual_values, actual_meta = read_pcm16(output)
                if sidecar_payload.get("sample_id") != sample_id or sidecar_payload.get("arm") != arm:
                    raise CalibrationError(f"audio sidecar identity mismatch: {sidecar}")
                if file_sha256(output) != arm_row.get("output_sha256") or actual_meta["pcm_sha256"] != arm_row.get("output_sha256"):
                    raise CalibrationError(f"audio output hash mismatch: {output}")
                if arm in (config.NATURAL_ARM, config.REPEAT_ARM):
                    if actual_values.tobytes() != natural_values.tobytes():
                        raise CalibrationError(f"{arm} is not byte-identical to natural audio: {sample_id}")
                    expected = "byte-identical natural PCM"
                    actual = actual_meta["pcm_sha256"]
                else:
                    expected_values, boundaries = local_swap_pcm16(natural_values)
                    if actual_values.tobytes() != expected_values.tobytes():
                        raise CalibrationError(f"LOCAL_SWAP PCM differs from deterministic reconstruction: {sample_id}")
                    expected = boundaries
                    actual = arm_row.get("construction", {})
                items.append(_check(item_name, "PASS", str(sidecar), sample_id=sample_id, expected=expected, actual=actual))
            except (OSError, ValueError, TypeError, CalibrationError) as exc:
                items.append(_check(item_name, "FAIL", str(exc), sample_id=sample_id))


def _audit_videos(history: Mapping[str, Any], items: list[dict[str, Any]]) -> None:
    cohort = history["cohort"]
    audio_manifest = history["audio"]
    video_manifest = history["videos"]
    for record in cohort["records"]:
        sample_id = str(record["sample_id"])
        audio_row = _find_row(list(audio_manifest["rows"]), sample_id, "audio")
        audio_by_arm = {str(item.get("arm")): item for item in audio_row.get("arms", [])}
        video_row = _find_row(list(video_manifest["rows"]), sample_id, "video")
        video_by_arm = dict(video_row.get("arms", {}))
        for arm in CONTROL_VIDEO_ARMS:
            try:
                arm_row = video_by_arm.get(arm)
                if not isinstance(arm_row, Mapping):
                    raise CalibrationError(f"video arm is missing: {sample_id}/{arm}")
                output = Path(str(arm_row.get("output")))
                sidecar = output.with_suffix(".json")
                payload = verify_self_hashed_json(sidecar)
                face = Path(str(record["face_video"]["path"]))
                audio = Path(str(audio_by_arm[arm]["output"]))
                command = [str(value) for value in payload.get("command", [])]
                if not output.is_file() or file_sha256(output) != arm_row.get("output_sha256"):
                    raise CalibrationError(f"video output hash mismatch: {output}")
                if payload.get("sample_id") != sample_id or payload.get("arm") != arm:
                    raise CalibrationError(f"video sidecar identity mismatch: {sidecar}")
                if payload.get("face_sha256") != file_sha256(face) or payload.get("audio_sha256") != file_sha256(audio):
                    raise CalibrationError(f"video source hash mismatch: {sample_id}/{arm}")
                if payload.get("checkpoint_sha256") != config.WAV2LIP_CHECKPOINT_SHA256 or payload.get("runtime_bindings") != EXPECTED_WAV2LIP_BINDINGS:
                    raise CalibrationError(f"Wav2Lip runtime binding mismatch: {sample_id}/{arm}")
                if "--face" not in command or Path(command[command.index("--face") + 1]).resolve() != face.resolve():
                    raise CalibrationError(f"Wav2Lip face binding mismatch: {sample_id}/{arm}")
                if "--audio" not in command or Path(command[command.index("--audio") + 1]).resolve() != audio.resolve():
                    raise CalibrationError(f"Wav2Lip audio binding mismatch: {sample_id}/{arm}")
                if "--checkpoint_path" not in command or Path(command[command.index("--checkpoint_path") + 1]).resolve() != config.WAV2LIP_CHECKPOINT.resolve():
                    raise CalibrationError(f"Wav2Lip checkpoint command binding mismatch: {sample_id}/{arm}")
                work_dir = config.HISTORY_ROOT / "02_videos" / "work" / arm / sample_id
                if not (work_dir / "temp").is_dir():
                    raise CalibrationError(f"Wav2Lip isolated work directory is missing: {work_dir}")
                if payload.get("geometry_sha256") != canonical_json_sha256(video_row.get("geometry", {})):
                    raise CalibrationError(f"video geometry binding mismatch: {sample_id}/{arm}")
                items.append(_check("video_sidecar", "PASS", str(sidecar), sample_id=sample_id, cell=f"V_{arm}"))
            except (OSError, ValueError, TypeError, KeyError, CalibrationError) as exc:
                items.append(_check("video_sidecar", "FAIL", str(exc), sample_id=sample_id, cell=f"V_{arm}"))


def _audit_scores(history: Mapping[str, Any], items: list[dict[str, Any]]) -> None:
    cohort = history["cohort"]
    audio_manifest = history["audio"]
    video_manifest = history["videos"]
    score_manifest = history["scores"]
    scores = {(str(row.get("sample_id")), str(row.get("cell"))): row for row in score_manifest.get("scores", [])}
    muxes = {(str(row.get("sample_id")), str(row.get("cell"))): row for row in score_manifest.get("muxes", [])}
    score_root = config.HISTORY_ROOT / "03_scores"
    for record in cohort["records"]:
        sample_id = str(record["sample_id"])
        audio_row = _find_row(list(audio_manifest["rows"]), sample_id, "audio")
        audio_by_arm = {str(item.get("arm")): item for item in audio_row.get("arms", [])}
        video_row = _find_row(list(video_manifest["rows"]), sample_id, "video")
        for cell in CONTROL_CELLS:
            try:
                score_row = scores.get((sample_id, cell))
                mux_row = muxes.get((sample_id, cell))
                if score_row is None or mux_row is None:
                    raise CalibrationError(f"score or mux row is missing: {sample_id}/{cell}")
                video_arm = cell.split("/")[0].removeprefix("V_")
                audio_arm = cell.split("/")[1].removeprefix("A_")
                video_row_arm = video_row["arms"][video_arm]
                video = Path(str(video_row_arm["output"]))
                audio = Path(str(audio_by_arm[audio_arm]["output"]))
                mux = Path(str(score_row.get("media")))
                score_sidecar = score_root / "scores" / video_arm / audio_arm / f"{sample_id}.json"
                mux_sidecar = mux.with_suffix(".json")
                verify_self_hashed_json(score_sidecar)
                mux_payload = verify_self_hashed_json(mux_sidecar)
                if score_row.get("media_sha256") != file_sha256(mux) or mux_payload.get("sha256") != file_sha256(mux):
                    raise CalibrationError(f"score/mux media hash mismatch: {sample_id}/{cell}")
                if score_row.get("syncnet_model_sha256") != config.SYNCNET_MODEL_SHA256 or score_row.get("syncnet_python_sha256") != config.SYNCNET_PYTHON_SHA256 or int(score_row.get("min_track", -1)) != config.MIN_TRACK:
                    raise CalibrationError(f"SyncNet runtime binding mismatch: {sample_id}/{cell}")
                if mux_payload.get("video_source_sha256") != file_sha256(video) or mux_payload.get("audio_source_sha256") != file_sha256(audio):
                    raise CalibrationError(f"mux source binding mismatch: {sample_id}/{cell}")
                if mux_payload.get("audio_pcm_verified") is not True or mux_payload.get("video_stream_copy_verified") is not True:
                    raise CalibrationError(f"mux sidecar did not assert both byte checks: {sample_id}/{cell}")
                mux_evidence = audit_mux(video, audio, mux)
                parsed = _parse_syncnet_log(Path(str(score_row.get("score_log"))))
                for name in ("sync_c", "sync_d", "av_offset"):
                    if parsed[name] != score_row.get(name):
                        raise CalibrationError(f"score row differs from raw log: {sample_id}/{cell}/{name}")
                reference = str(score_row.get("reference", ""))
                expected_reference = f"{sample_id}__{video_arm}_{audio_arm}"
                if reference != expected_reference:
                    raise CalibrationError(f"score reference identity mismatch: {sample_id}/{cell}")
                data_dir = score_root / "syncnet" / reference
                if not data_dir.is_dir():
                    raise CalibrationError(f"SyncNet data directory is missing: {data_dir}")
                items.append(_check("score_cell", "PASS", str(score_sidecar), sample_id=sample_id, cell=cell, expected={"reference": expected_reference}, actual={"mux": mux_evidence, "score": parsed}))
            except (OSError, ValueError, TypeError, KeyError, subprocess.SubprocessError, CalibrationError) as exc:
                items.append(_check("score_cell", "FAIL", str(exc), sample_id=sample_id, cell=cell))


def audit_history() -> dict[str, Any]:
    """Audit the prior run and return a branch decision without new scoring."""

    try:
        history = load_history()
    except (OSError, ValueError, TypeError, CalibrationError) as exc:
        return {
            "schema_version": 1,
            "stage_id": "00_audit",
            "protocol_id": config.PROTOCOL_ID,
            "status": "complete",
            "audit_decision": "INCONCLUSIVE",
            "repair_verified": False,
            "history_final_sha256": config.HISTORY_FINAL_SHA256,
            "history_cohort_sha256": config.HISTORY_COHORT_SHA256,
            "history_audio_sha256": config.HISTORY_AUDIO_SHA256,
            "history_videos_sha256": config.HISTORY_VIDEOS_SHA256,
            "history_scores_sha256": config.HISTORY_SCORES_SHA256,
            "record_count": config.EXPECTED_RECORD_COUNT,
            "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
            "new_score_read_count": 0,
            "items": [_check("history_provenance", "INCONCLUSIVE", str(exc))],
            "summary": {"pass": 0, "fail": 0, "inconclusive": 1},
            "reason": str(exc),
        }
    items: list[dict[str, Any]] = []
    _audit_audio(history, items)
    _audit_videos(history, items)
    _audit_scores(history, items)
    pass_count = sum(item["status"] == "PASS" for item in items)
    fail_count = sum(item["status"] == "FAIL" for item in items)
    inconclusive_count = sum(item["status"] == "INCONCLUSIVE" for item in items)
    if inconclusive_count:
        decision = "INCONCLUSIVE"
    elif fail_count:
        decision = "DEFECT_FOUND"
    else:
        decision = "NO_DEFECT_FOUND"
    return {
        "schema_version": 1,
        "stage_id": "00_audit",
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "audit_decision": decision,
        "repair_verified": False,
        "history_final_sha256": config.HISTORY_FINAL_SHA256,
        "history_cohort_sha256": config.HISTORY_COHORT_SHA256,
        "history_audio_sha256": config.HISTORY_AUDIO_SHA256,
        "history_videos_sha256": config.HISTORY_VIDEOS_SHA256,
        "history_scores_sha256": config.HISTORY_SCORES_SHA256,
        "record_count": config.EXPECTED_RECORD_COUNT,
        "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        "classification": "fit_only_control_calibration",
        "new_score_read_count": 0,
        "items": items,
        "summary": {"pass": pass_count, "fail": fail_count, "inconclusive": inconclusive_count},
        "checks": {
            "audio_arms": [config.NATURAL_ARM, config.REPEAT_ARM, config.LOCAL_SWAP_ARM],
            "score_cells": list(CONTROL_CELLS),
            "actual_mux_and_raw_log_verified": True,
        },
    }


def run_audit(paths: config.RunPaths) -> dict[str, Any]:
    if paths.audit.exists() and (paths.audit / "audit.json").is_file():
        existing = verify_self_hashed_json(paths.audit / "audit.json")
        if existing.get("protocol_id") != config.PROTOCOL_ID:
            raise CalibrationError("existing audit belongs to another protocol")
        return existing
    paths.audit.mkdir(parents=True, exist_ok=True)
    audit = audit_history()
    write_self_hashed_json(paths.audit / "audit.json", audit)
    return verify_self_hashed_json(paths.audit / "audit.json")
