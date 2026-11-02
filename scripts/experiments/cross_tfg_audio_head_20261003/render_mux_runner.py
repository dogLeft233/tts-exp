"""Render frozen audio/method rows, then losslessly attach canonical natural PCM.

This producer does not score SyncNet or select candidates.  It consumes a
pre-frozen plan and writes one render/mux result for every planned row,
including failures.  Model-specific renderers are argv templates in the plan;
no shell is involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.audio_enhancement.audio_io import read_canonical_wav
from scripts.audio_enhancement.overdub import mux_original_pcm


SCHEMA = "cross_tfg_audio_head_render_mux_v1"
IDENTITY_FIELDS = (
    "row_id", "split", "sample_id", "speaker_id", "model_id", "audio_frontend",
    "model_role", "face_id", "seed", "method_id", "method_config_sha256",
    "checkpoint_sha256",
)
SAFE_COMPONENT = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,160}$")


class RowFailure(RuntimeError):
    """A row-local failure that should be recorded without dropping the row."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                      allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected one JSON object")
            rows.append(value)
    return rows


def _write_jsonl_row(handle, row: Mapping[str, Any]) -> None:
    raw = (json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    handle.write(raw)
    handle.flush()
    os.fsync(handle.fileno())


def _safe_component(value: Any, field: str) -> str:
    text = str(value)
    if not SAFE_COMPONENT.fullmatch(text) or text in {".", ".."}:
        raise ValueError(f"unsafe {field} path component: {text!r}")
    return text


def _check_ref(ref: Mapping[str, Any], *, label: str) -> Path:
    raw_path = ref.get("path")
    expected = ref.get("file_sha256")
    if not isinstance(raw_path, str) or not raw_path:
        raise RowFailure(f"{label}: missing path")
    path = Path(raw_path)
    if not path.is_file():
        raise RowFailure(f"{label}: file is missing: {path}")
    actual = sha256_file(path)
    if not isinstance(expected, str) or actual != expected:
        raise RowFailure(f"{label}: file SHA-256 mismatch at {path}")
    return path


def _check_canonical_wav(ref: Mapping[str, Any], *, label: str) -> tuple[Path, dict[str, Any]]:
    path = _check_ref(ref, label=label)
    try:
        samples, info = read_canonical_wav(path)
    except Exception as exc:
        raise RowFailure(f"{label}: canonical WAV read failed: {type(exc).__name__}: {exc}") from exc
    if info.source_sample_rate != 16000 or info.source_channels != 1 or info.source_frames != samples.size:
        raise RowFailure(f"{label}: planned WAV is not already canonical 16 kHz mono")
    expected = {
        "sample_rate_hz": info.sample_rate,
        "channels": 1,
        "sample_count": info.frames,
        "pcm_sha256": info.pcm_float32_sha256,
    }
    for key, actual in expected.items():
        if ref.get(key) != actual:
            raise RowFailure(f"{label}: {key} binding mismatch")
    return path, {
        "file_sha256": info.file_sha256,
        "pcm_sha256": info.pcm_float32_sha256,
        "sample_rate_hz": info.sample_rate,
        "channels": 1,
        "sample_count": info.frames,
        "source_encoding": info.source_encoding,
    }


def _hash_declared_sources(model: Mapping[str, Any]) -> None:
    for item in model.get("source_files", []):
        if not isinstance(item, Mapping):
            raise RowFailure("model source_files entry is not an object")
        path = Path(str(item.get("path", "")))
        expected = item.get("file_sha256")
        if not path.is_file() or not isinstance(expected, str) or sha256_file(path) != expected:
            raise RowFailure(f"model code source seal mismatch: {path}")
    checkpoint = model.get("checkpoint")
    expected_checkpoint = model.get("checkpoint_sha256")
    if checkpoint is not None:
        path = Path(str(checkpoint))
        if not path.is_file() or not isinstance(expected_checkpoint, str):
            raise RowFailure(f"model checkpoint is missing or unsealed: {path}")
        if sha256_file(path) != expected_checkpoint:
            raise RowFailure(f"model checkpoint SHA-256 mismatch: {path}")


def _frame_rate(stream: Mapping[str, Any]) -> tuple[int, int]:
    for key in ("avg_frame_rate", "r_frame_rate"):
        value = stream.get(key)
        if isinstance(value, str) and value not in {"0/0", "N/A", ""}:
            parsed = Fraction(value)
            if parsed.numerator > 0 and parsed.denominator > 0:
                return parsed.numerator, parsed.denominator
    raise RowFailure("ffprobe returned no positive rational frame rate")


def _probe_frames(path: Path, *, ffprobe: str, timeout: float) -> dict[str, Any]:
    command = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams",
        "-show_frames", "-show_entries",
        "stream=codec_name,width,height,time_base,avg_frame_rate,r_frame_rate,nb_frames,start_time,duration:frame=best_effort_timestamp,pkt_duration",
        "-of", "json", str(path),
    ]
    completed = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, timeout=timeout, check=False)
    if completed.returncode != 0:
        message = completed.stderr.decode("utf-8", errors="replace")[-3000:]
        raise RowFailure(f"ffprobe frame enumeration failed ({completed.returncode}): {message}")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except Exception as exc:
        raise RowFailure("ffprobe frame enumeration returned invalid JSON") from exc
    streams = payload.get("streams", [])
    frames = payload.get("frames", [])
    if len(streams) != 1 or not frames:
        raise RowFailure("expected one video stream and at least one decoded frame")
    stream = streams[0]
    time_base = Fraction(str(stream.get("time_base", "0/1")))
    if time_base.numerator <= 0 or time_base.denominator <= 0:
        raise RowFailure("video stream has invalid time_base")
    timestamps = []
    previous = None
    for frame_index, frame in enumerate(frames):
        raw_pts = frame.get("best_effort_timestamp")
        if raw_pts is None:
            raise RowFailure(f"video frame {frame_index} has no best_effort_timestamp")
        pts = int(raw_pts)
        if previous is not None and pts < previous:
            raise RowFailure("video frame PTS is not monotonic")
        previous = pts
        timestamps.append({
            "frame_index": frame_index,
            "pts": pts,
            "time_num": pts * time_base.numerator,
            "time_den": time_base.denominator,
            "duration_pts": int(frame["pkt_duration"]) if frame.get("pkt_duration") is not None else None,
        })
    fps_num, fps_den = _frame_rate(stream)
    return {
        "codec_name": stream.get("codec_name"),
        "width": int(stream["width"]) if stream.get("width") is not None else None,
        "height": int(stream["height"]) if stream.get("height") is not None else None,
        "time_base_num": time_base.numerator,
        "time_base_den": time_base.denominator,
        "fps_num": fps_num,
        "fps_den": fps_den,
        "frame_count": len(timestamps),
        "stream_start_time": stream.get("start_time"),
        "stream_duration": stream.get("duration"),
        "frame_timestamps": timestamps,
        "ffprobe_command": command,
    }


def _run_logged(argv: Sequence[str], *, cwd: Path, stdout_path: Path, stderr_path: Path,
                timeout: float, env: Mapping[str, str] | None = None) -> dict[str, Any]:
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    try:
        with stdout_path.open("xb") as out, stderr_path.open("xb") as err:
            completed = subprocess.run(
                list(argv), cwd=str(cwd), env=dict(env) if env is not None else None,
                stdin=subprocess.DEVNULL, stdout=out, stderr=err,
                timeout=timeout, check=False,
            )
        return {
            "returncode": completed.returncode,
            "started_unix": started,
            "ended_unix": time.time(),
            "stdout_path": str(stdout_path),
            "stdout_sha256": sha256_file(stdout_path),
            "stdout_bytes": stdout_path.stat().st_size,
            "stderr_path": str(stderr_path),
            "stderr_sha256": sha256_file(stderr_path),
            "stderr_bytes": stderr_path.stat().st_size,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": None,
            "timed_out": True,
            "started_unix": started,
            "ended_unix": time.time(),
            "stdout_path": str(stdout_path),
            "stdout_sha256": sha256_file(stdout_path) if stdout_path.exists() else None,
            "stderr_path": str(stderr_path),
            "stderr_sha256": sha256_file(stderr_path) if stderr_path.exists() else None,
            "error": f"TimeoutExpired: {exc}",
        }


def _command_for(model: Mapping[str, Any], row: Mapping[str, Any], *, output_video: Path,
                 natural_audio: Path, enhanced_audio: Path) -> tuple[list[str], Path, dict[str, str]]:
    generator = model.get("generator")
    if not isinstance(generator, Mapping):
        raise RowFailure("model is missing a generator command contract")
    template = generator.get("argv")
    if not isinstance(template, list) or not template or not all(isinstance(x, str) for x in template):
        raise RowFailure("model generator.argv must be a non-empty string array")
    face = row.get("source_face", {})
    video = row.get("source_video", {})
    values = {
        "python": str(generator.get("python", "python")),
        "repo": str(generator.get("repo", "")),
        "checkpoint": str(model.get("checkpoint", "")),
        "source_face": str(face.get("path", "")) if isinstance(face, Mapping) else "",
        "source_video": str(video.get("path", "")) if isinstance(video, Mapping) else "",
        "natural_audio": str(natural_audio),
        "enhanced_audio": str(enhanced_audio),
        "output_video": str(output_video),
        "model_id": str(row.get("model_id", "")),
        "method_id": str(row.get("method_id", "")),
        "sample_id": str(row.get("sample_id", "")),
        "speaker_id": str(row.get("speaker_id", "")),
        "face_id": str(row.get("face_id", "")),
        "seed": str(row.get("seed", "")),
    }
    custom = generator.get("template_values", {})
    if not isinstance(custom, Mapping):
        raise RowFailure("generator.template_values must be an object")
    for key, value in custom.items():
        if not isinstance(key, str) or key in values:
            raise RowFailure(f"invalid or reserved command template value: {key!r}")
        values[key] = str(value)
    try:
        argv = [part.format_map(values) for part in template]
    except KeyError as exc:
        raise RowFailure(f"unbound generator argv template field: {exc}") from exc
    cwd = Path(str(generator.get("cwd", "."))).resolve()
    if not cwd.is_dir():
        raise RowFailure(f"generator working directory is missing: {cwd}")
    env = os.environ.copy()
    override = generator.get("env", {})
    if not isinstance(override, Mapping) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in override.items()):
        raise RowFailure("generator.env must be a string-to-string object")
    env.update(override)
    return argv, cwd, env


def _identity(row: Mapping[str, Any]) -> dict[str, Any]:
    return {key: row.get(key) for key in IDENTITY_FIELDS}


def process_row(row: Mapping[str, Any], *, model: Mapping[str, Any], out_root: Path,
                ffmpeg: str, ffprobe: str, timeout_seconds: float) -> dict[str, Any]:
    result: dict[str, Any] = {
        **_identity(row),
        "schema_version": SCHEMA,
        "status": "FAILED",
        "failure_reason": None,
        "render": {"status": "NOT_STARTED"},
        "mux_audio_readback": {"status": "NOT_STARTED"},
        "score": {"status": "NOT_RUN"},
    }
    stage_root = out_root / "rows" / _safe_component(row.get("row_id"), "row_id")
    logs = stage_root / "logs"
    try:
        if model.get("model_id") != row.get("model_id"):
            raise RowFailure("plan row model_id does not match model configuration")
        if model.get("checkpoint_sha256") != row.get("checkpoint_sha256"):
            raise RowFailure("plan row checkpoint SHA does not match model configuration")
        _hash_declared_sources(model)
        natural_path, natural_info = _check_canonical_wav(row.get("natural_pcm", {}), label="natural_pcm")
        enhanced_path, enhanced_info = _check_canonical_wav(row.get("enhanced_pcm", {}), label="enhanced_pcm")
        if natural_info["sample_count"] != enhanced_info["sample_count"]:
            raise RowFailure("natural and enhanced canonical PCM sample counts differ")
        if row.get("score_audio_sha256") != natural_info["pcm_sha256"]:
            raise RowFailure("score_audio_sha256 is not the canonical natural PCM hash")
        if row.get("score_audio_path") != str(natural_path):
            raise RowFailure("score_audio_path does not resolve to planned natural PCM")
        for key in ("source_face", "source_video"):
            ref = row.get(key)
            if not isinstance(ref, Mapping):
                raise RowFailure(f"missing {key} binding")
            _check_ref(ref, label=key)

        model_id = _safe_component(row.get("model_id"), "model_id")
        method_id = _safe_component(row.get("method_id"), "method_id")
        sample_id = _safe_component(row.get("sample_id"), "sample_id")
        render_dir = stage_root / "render"
        render_dir.mkdir(parents=True, exist_ok=False)
        generated_video = render_dir / "generated_video.mp4"
        silent_video = render_dir / "silent_video.mkv"
        muxed_video = render_dir / "natural_pcm_mux.mkv"
        timestamps_path = render_dir / "video_timestamps.json"
        argv, cwd, env = _command_for(model, row, output_video=generated_video,
                                      natural_audio=natural_path, enhanced_audio=enhanced_path)
        _atomic_json(render_dir / "generator_invocation.json", {
            "argv": argv,
            "cwd": str(cwd),
            "environment_overrides": model.get("generator", {}).get("env", {}),
            "natural_audio_path": str(natural_path),
            "natural_audio_file_sha256": natural_info["file_sha256"],
            "natural_pcm_float32_sha256": natural_info["pcm_sha256"],
            "enhanced_audio_path": str(enhanced_path),
            "enhanced_audio_file_sha256": enhanced_info["file_sha256"],
            "enhanced_pcm_float32_sha256": enhanced_info["pcm_sha256"],
        })
        generation = _run_logged(argv, cwd=cwd,
            stdout_path=logs / "generator.stdout.log", stderr_path=logs / "generator.stderr.log",
            timeout=float(model.get("generator", {}).get("timeout_seconds", timeout_seconds)), env=env)
        if generation.get("returncode") != 0:
            raise RowFailure(f"generator exited unsuccessfully: {generation.get('returncode')}")
        if not generated_video.is_file() or generated_video.stat().st_size == 0:
            raise RowFailure("generator returned success without a non-empty video")
        generated_sha = sha256_file(generated_video)
        generated_clock = _probe_frames(generated_video, ffprobe=ffprobe, timeout=timeout_seconds)

        strip_command = [ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(generated_video),
                         "-map", "0:v:0", "-c:v", "copy", "-an", "-map_metadata", "-1",
                         "-f", "matroska", str(silent_video)]
        strip_result = _run_logged(strip_command, cwd=render_dir,
            stdout_path=logs / "strip_audio.stdout.log", stderr_path=logs / "strip_audio.stderr.log",
            timeout=timeout_seconds)
        if strip_result.get("returncode") != 0 or not silent_video.is_file():
            raise RowFailure(f"video-only extraction failed: {strip_result.get('returncode')}")
        silent_clock = _probe_frames(silent_video, ffprobe=ffprobe, timeout=timeout_seconds)
        if generated_clock["frame_timestamps"] != silent_clock["frame_timestamps"]:
            raise RowFailure("video timestamp sequence changed while removing generator audio")
        timestamp_payload = {
            "schema_version": "cross_tfg_video_clock_v1",
            "model_id": model_id,
            "method_id": method_id,
            "sample_id": sample_id,
            "stream_time_base": [generated_clock["time_base_num"], generated_clock["time_base_den"]],
            "frame_count": generated_clock["frame_count"],
            "frames": generated_clock["frame_timestamps"],
        }
        _atomic_json(timestamps_path, timestamp_payload)
        mux = mux_original_pcm(silent_video, natural_path, muxed_video,
                               ffmpeg=ffmpeg, ffprobe=ffprobe, timeout_seconds=timeout_seconds)
        mux_clock = _probe_frames(muxed_video, ffprobe=ffprobe, timeout=timeout_seconds)
        if mux_clock["frame_timestamps"] != generated_clock["frame_timestamps"]:
            raise RowFailure("video timestamp sequence changed while muxing natural PCM")
        mux_probe_command = [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(muxed_video)]
        mux_probe_run = subprocess.run(mux_probe_command, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                       timeout=timeout_seconds, check=False)
        if mux_probe_run.returncode != 0:
            raise RowFailure("ffprobe could not inspect final PCM mux")
        mux_probe = json.loads(mux_probe_run.stdout.decode("utf-8"))
        audio_streams = [s for s in mux_probe.get("streams", []) if s.get("codec_type") == "audio"]
        if len(audio_streams) != 1:
            raise RowFailure("final mux does not contain exactly one audio stream")
        audio_stream = audio_streams[0]
        audio_start = audio_stream.get("start_time", "0")
        render = {
            "status": "PASS",
            "generated_video_path": str(generated_video.resolve()),
            "generated_video_sha256": generated_sha,
            "silent_video_path": str(silent_video.resolve()),
            "silent_video_sha256": sha256_file(silent_video),
            "muxed_video_path": str(muxed_video.resolve()),
            "muxed_video_sha256": mux.output_file_sha256,
            "video_stream_sha256": mux.video_stream_sha256,
            "fps_num": generated_clock["fps_num"],
            "fps_den": generated_clock["fps_den"],
            "frame_count": generated_clock["frame_count"],
            "timestamps_path": str(timestamps_path.resolve()),
            "timestamps_sha256": sha256_file(timestamps_path),
            "video_time_base_num": generated_clock["time_base_num"],
            "video_time_base_den": generated_clock["time_base_den"],
            "video_codec": generated_clock["codec_name"],
            "generation": generation,
            "strip_audio": strip_result,
            "mux_command": [
                ffmpeg, "-v", "error", "-nostdin", "-y", "-i", str(silent_video),
                "-i", str(natural_path), "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "copy", "-c:a", "copy", "-map_metadata", "0", "-f", "matroska", str(muxed_video),
            ],
        }
        mux_readback = {
            "status": "PASS" if mux.audio_matches_original_decoded_pcm else "FAIL",
            "codec": mux.audio_codec,
            "sample_rate_hz": mux.audio_sample_rate,
            "channels": mux.audio_channels,
            "sample_count": mux.decoded_audio_sample_count,
            "pcm_sha256": mux.decoded_audio_float32_sha256,
            "start_time_s": str(audio_start),
            "matches_natural_canonical_pcm": mux.audio_matches_original_decoded_pcm,
            "video_stream_identity_verified": True,
        }
        if mux_readback["pcm_sha256"] != natural_info["pcm_sha256"]:
            raise RowFailure("mux readback PCM hash differs from canonical natural PCM")
        result.update({
            "status": "RENDER_MUX_COMPLETE",
            "failure_reason": None,
            "render": render,
            "mux_audio_readback": mux_readback,
            "score": {"status": "NOT_RUN"},
        })
    except Exception as exc:
        result["status"] = "FAILED"
        result["failure_reason"] = f"{type(exc).__name__}: {exc}"
        stage_root.mkdir(parents=True, exist_ok=True)
        try:
            _atomic_json(stage_root / "row_failure.json", {
                **_identity(row), "error_type": type(exc).__name__, "error": str(exc),
                "observed_unix": time.time(),
            })
        except Exception:
            pass
        render_status = result.get("render", {}).get("status")
        result["render"] = {"status": render_status or "FAILED"}
        result["mux_audio_readback"] = {"status": "FAILED" if render_status == "PASS" else "NOT_REACHED"}
    return result


def run_plan(plan_path: Path, planned_rows_path: Path, output_root: Path, *,
             expected_plan_sha256: str, ffmpeg: str, ffprobe: str,
             timeout_seconds: float) -> dict[str, Any]:
    plan_raw = plan_path.read_bytes()
    actual_plan_sha = hashlib.sha256(plan_raw).hexdigest()
    if actual_plan_sha != expected_plan_sha256:
        raise ValueError("plan raw SHA-256 differs from caller pin")
    plan = json.loads(plan_raw.decode("utf-8"))
    rows = _read_jsonl(planned_rows_path)
    models = plan.get("models")
    if not isinstance(models, list):
        raise ValueError("plan.models must be an array")
    by_model = {str(item.get("model_id")): item for item in models if isinstance(item, Mapping)}
    if len(by_model) != len(models):
        raise ValueError("plan has duplicate or malformed model definitions")
    row_ids = [str(row.get("row_id", "")) for row in rows]
    if not rows or any(not row_id for row_id in row_ids) or len(row_ids) != len(set(row_ids)):
        raise ValueError("planned rows are empty or have missing/duplicate row_id values")
    output_root = output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"refusing to write into non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    results_path = output_root / "render_mux_results.jsonl"
    if results_path.exists():
        raise FileExistsError(f"refusing to overwrite results: {results_path}")
    complete = failed = 0
    with results_path.open("xb") as handle:
        for row in rows:
            model = by_model.get(str(row.get("model_id")))
            if model is None:
                item = {**_identity(row), "schema_version": SCHEMA, "status": "FAILED",
                        "failure_reason": "plan row references an unknown model_id",
                        "render": {"status": "NOT_STARTED"},
                        "mux_audio_readback": {"status": "NOT_STARTED"},
                        "score": {"status": "NOT_RUN"}}
            else:
                item = process_row(row, model=model, out_root=output_root,
                                   ffmpeg=ffmpeg, ffprobe=ffprobe,
                                   timeout_seconds=timeout_seconds)
            _write_jsonl_row(handle, item)
            if item["status"] == "RENDER_MUX_COMPLETE":
                complete += 1
            else:
                failed += 1
    manifest = {
        "schema_version": SCHEMA,
        "status": "RENDER_MUX_COMPLETE" if failed == 0 else "RENDER_MUX_COMPLETE_WITH_FAILURES",
        "plan_path": str(plan_path.resolve()),
        "plan_sha256": actual_plan_sha,
        "planned_rows_path": str(planned_rows_path.resolve()),
        "planned_rows_sha256": sha256_file(planned_rows_path),
        "planned_rows": len(rows),
        "render_mux_complete_rows": complete,
        "failed_rows": failed,
        "result_path": str(results_path.resolve()),
        "result_sha256": sha256_file(results_path),
        "source_retired": False,
        "scores_read_or_generated": False,
    }
    _atomic_json(output_root / "render_mux_manifest.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--planned-rows", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--expected-plan-sha256", required=True)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument("--ffprobe", default="ffprobe")
    parser.add_argument("--timeout-seconds", type=float, default=1800.0)
    args = parser.parse_args(argv)
    try:
        result = run_plan(args.plan, args.planned_rows, args.output_root,
                          expected_plan_sha256=args.expected_plan_sha256,
                          ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
                          timeout_seconds=args.timeout_seconds)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True))
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0 if result["failed_rows"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
