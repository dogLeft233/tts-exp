#!/usr/bin/env python3
"""One-pass CPU RGB comparison of the six accepted MuseTalk H.264 outputs."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np


PROJECT = Path("/home/wjj/tts-audio/tts-exp")
OUTPUT_ROOT = Path(
    "/dev/shm/cross_tfg_native_attribute_response_probe_20261015_"
    "output_owned_musetalk_render_output_attempt01"
)
MAP_PATH = PROJECT / (
    ".codex/scratch/native_attribute_response_probe_20261015_output_owned_musetalk_render_stage_attempt01/"
    "provenance/output_owned_actual_frame_map_attempt01.json"
)
REMOTE_RUN_ROOT = PROJECT / ".codex/scratch/"
REMOTE_RUN_ROOT /= "native_attribute_response_probe_20261015_h264_rgb_pixel_diagnostic_attempt01"
RESULT_PATH = REMOTE_RUN_ROOT / "h264_rgb_pixel_diagnostic_attempt01.json"
EXECUTION_PATH = REMOTE_RUN_ROOT / "h264_rgb_pixel_execution_receipt_attempt01.json"
FFMPEG = "/usr/bin/ffmpeg"
EXPECTED_MAP_SHA256 = "127f5b8398521766592a73807f0ae5c4a5f81aaefd579f3ce4e83e937ab3b11f"
EXPECTED_CONTROLLER_SHA256 = "adf7f5caaa06cacb0eabaf5f67e4ceeae382f453eb565ed0704d91692d23b05a"
SAMPLES = ("BAC009S0022W0123", "BAC009S0024W0122")
ARMS = ("zero", "plus", "minus")
CHANNELS = 3


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def proc_start_ticks(pid: int) -> str | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
        return stat[stat.rfind(")") + 2 :].split()[19]
    except (OSError, IndexError):
        return None


def write_json_create_only(path: Path, payload: dict) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
        stream.write("\n")


def read_exact(stream, size: int) -> bytes:
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def decode_rgb24_once(video_path: Path, frame_count: int, height: int, width: int):
    command = [
        FFMPEG,
        "-nostdin",
        "-hide_banner",
        "-loglevel",
        "error",
        "-hwaccel",
        "none",
        "-threads",
        "1",
        "-i",
        str(video_path),
        "-map",
        "0:v:0",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    frame_bytes = height * width * CHANNELS
    process = subprocess.Popen(
        command,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    )
    identity = {
        "pid": process.pid,
        "start_ticks": proc_start_ticks(process.pid),
        "argv": command,
        "video_path": str(video_path),
    }
    frames: list[bytes] = []
    try:
        assert process.stdout is not None
        for frame_index in range(frame_count):
            raw = read_exact(process.stdout, frame_bytes)
            if len(raw) != frame_bytes:
                raise RuntimeError(
                    f"decoded frame {frame_index} has {len(raw)} bytes; "
                    f"expected {frame_bytes}"
                )
            frames.append(raw)
        tail = process.stdout.read(frame_bytes + 1)
        if tail:
            while process.stdout.read(1024 * 1024):
                pass
            raise RuntimeError(
                f"decoder emitted bytes after the {frame_count} receipt-bound frames"
            )
        returncode = process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        identity.update(
            {
                "returncode": returncode,
                "decoded_frame_count": len(frames),
                "rgb24_bytes": len(frames) * frame_bytes,
                "stderr": stderr,
            }
        )
        if returncode != 0:
            raise RuntimeError(f"ffmpeg returned {returncode}: {stderr}")
        return frames, identity
    except Exception:
        if process.poll() is None:
            process.kill()
        process.wait()
        raise
    finally:
        if process.stdout:
            process.stdout.close()
        if process.stderr:
            process.stderr.close()


def frame_hashes(frames: list[bytes]) -> list[dict]:
    return [
        {"frame_index": index, "rgb_sha256": hashlib.sha256(raw).hexdigest()}
        for index, raw in enumerate(frames)
    ]


def compare_arm(
    sample_id: str,
    arm: str,
    zero_frames: list[bytes],
    candidate_frames: list[bytes],
    owned_indices: set[int],
    height: int,
    width: int,
) -> dict:
    frame_changed_owned = 0
    frame_changed_nonowned = 0
    nonowned_sum = 0
    nonowned_values = 0
    nonowned_max = 0
    for index, (zero_raw, candidate_raw) in enumerate(zip(zero_frames, candidate_frames)):
        zero = np.frombuffer(zero_raw, dtype=np.uint8).reshape(height, width, CHANNELS)
        candidate = np.frombuffer(candidate_raw, dtype=np.uint8).reshape(height, width, CHANNELS)
        delta = np.abs(candidate.astype(np.int16) - zero.astype(np.int16))
        changed = bool(np.any(delta))
        if index in owned_indices:
            frame_changed_owned += int(changed)
        else:
            frame_changed_nonowned += int(changed)
            nonowned_sum += int(delta.sum(dtype=np.uint64))
            nonowned_values += int(delta.size)
            nonowned_max = max(nonowned_max, int(delta.max()))
    return {
        "sample_id": sample_id,
        "candidate_arm": arm,
        "frame_count": len(zero_frames),
        "owned_frame_indices": sorted(owned_indices),
        "owned_frame_count": len(owned_indices),
        "nonowned_frame_count": len(zero_frames) - len(owned_indices),
        "changed_frame_count_owned": frame_changed_owned,
        "changed_frame_count_nonowned": frame_changed_nonowned,
        "nonowned_mean_absolute_rgb_byte_difference": (
            nonowned_sum / nonowned_values if nonowned_values else None
        ),
        "nonowned_max_absolute_rgb_byte_difference": nonowned_max,
        "mean_definition": "mean absolute difference over all RGB channel bytes in all nonowned frames",
    }


def run(source_sha256: str) -> tuple[dict, dict]:
    actual_source_sha256 = sha256_file(Path(__file__))
    if actual_source_sha256 != source_sha256:
        raise RuntimeError("source file SHA256 does not match the execution pin")
    if sha256_file(MAP_PATH) != EXPECTED_MAP_SHA256:
        raise RuntimeError("actual changed-output frame map SHA256 mismatch")
    controller_path = OUTPUT_ROOT / "controller_receipt.json"
    if sha256_file(controller_path) != EXPECTED_CONTROLLER_SHA256:
        raise RuntimeError("controller receipt SHA256 mismatch")
    if not Path(FFMPEG).is_file():
        raise RuntimeError(f"pinned ffmpeg executable is missing: {FFMPEG}")

    frame_map = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    controller = json.loads(controller_path.read_text(encoding="utf-8"))
    if frame_map.get("status") != "PASS_ACTUAL_CHANGED_ROWS":
        raise RuntimeError("frame map is not the accepted actual-changed-rows map")
    if controller.get("status") != "PASS_RENDER_ONLY_NO_SCORE" or controller.get("row_count") != 6:
        raise RuntimeError("controller receipt is not the accepted six-row render-only receipt")
    rows_by_key = {(row["sample_id"], row["arm"]): row for row in controller["rows"]}
    if len(rows_by_key) != 6 or set(rows_by_key) != {
        (sample, arm) for sample in SAMPLES for arm in ARMS
    }:
        raise RuntimeError("controller rows do not exactly match the frozen two-sample/three-arm design")
    outdir = REMOTE_RUN_ROOT
    outdir.mkdir(parents=True, exist_ok=False)

    decode_runs = []
    hashes = {}
    comparisons = []
    sample_summaries = []
    for sample_id in SAMPLES:
        owned = set(frame_map["owned_output_frame_indices_by_sample"][sample_id])
        arm_rows = {arm: rows_by_key[(sample_id, arm)] for arm in ARMS}
        frame_counts = {int(row["frame_count"]) for row in arm_rows.values()}
        fps = {(int(row["fps_num"]), int(row["fps_den"])) for row in arm_rows.values()}
        if len(frame_counts) != 1 or len(fps) != 1 or fps != {(25, 1)}:
            raise RuntimeError(f"row framecount/fps receipt mismatch for {sample_id}")
        frame_count = frame_counts.pop()
        if len(owned) != 7 or min(owned) < 0 or max(owned) >= frame_count:
            raise RuntimeError(f"frozen owned-frame mask is invalid for {sample_id}")
        shape_values = [row["precodec_frames"][0]["shape"] for row in arm_rows.values()]
        if any(shape != [512, 512, CHANNELS] for shape in shape_values):
            raise RuntimeError(f"accepted precodec RGB shape is not 512x512x3 for {sample_id}")
        height, width, _ = shape_values[0]
        for arm, row in arm_rows.items():
            if row["run_status"]["status"] != "PASS_RENDER_ONLY" or row["run_status"]["returncode"] != 0:
                raise RuntimeError(f"renderer row is not PASS for {sample_id}/{arm}")
            if len(row["precodec_frames"]) != frame_count:
                raise RuntimeError(f"PNG manifest count differs from frozen clock for {sample_id}/{arm}")
            video_path = Path(row["silent_video"]["path"])
            actual_video_sha256 = sha256_file(video_path)
            if actual_video_sha256 != row["silent_video"]["sha256"]:
                raise RuntimeError(f"silent video hash mismatch for {sample_id}/{arm}")
            frames, decode_identity = decode_rgb24_once(video_path, frame_count, height, width)
            decode_identity.update(
                {
                    "sample_id": sample_id,
                    "arm": arm,
                    "source_video_sha256": actual_video_sha256,
                }
            )
            if len(frames) != frame_count:
                raise RuntimeError(f"decoded frame count differs from accepted clock for {sample_id}/{arm}")
            decode_runs.append(decode_identity)
            hashes.setdefault(sample_id, {})[arm] = frame_hashes(frames)
            arm_rows[arm]["_decoded_rgb_frames"] = frames

        zero_frames = arm_rows["zero"].pop("_decoded_rgb_frames")
        sample_summaries.append(
            {
                "sample_id": sample_id,
                "frame_count_from_controller_receipt": frame_count,
                "fps_from_controller_receipt": {"num": 25, "den": 1},
                "owned_output_frame_indices_from_actual_cpu_map": sorted(owned),
                "nonowned_frame_count": frame_count - len(owned),
            }
        )
        for arm in ("plus", "minus"):
            candidate_frames = arm_rows[arm].pop("_decoded_rgb_frames")
            comparisons.append(
                compare_arm(sample_id, arm, zero_frames, candidate_frames, owned, height, width)
            )
            del candidate_frames
        del zero_frames

    completed = utc_now()
    data = {
        "schema": "musetalk_h264_decoded_rgb24_nonowned_pixel_diagnostic_v1",
        "status": "PASS_CPU_DECODED_PIXELS_ONLY",
        "scientific_scope": "exact RGB24 bytes decoded from the six existing silent H.264 MP4s; compare plus/minus against zero; no scoring, geometry extraction, PCM reprocessing, or media generation",
        "source_sha256": actual_source_sha256,
        "controller_receipt": {"path": str(controller_path), "sha256": EXPECTED_CONTROLLER_SHA256},
        "actual_output_frame_map": {"path": str(MAP_PATH), "sha256": EXPECTED_MAP_SHA256},
        "decode": {
            "ffmpeg": FFMPEG,
            "software_only": True,
            "output_pixel_format": "rgb24",
            "frame_shape_from_accepted_png_manifest": [512, 512, 3],
            "frame_count_and_clock_source": "accepted controller receipt only; no ffprobe or PTS recomputation in this diagnostic",
            "per_video_decode_count": 1,
        },
        "samples": sample_summaries,
        "comparisons_vs_zero": comparisons,
        "per_frame_rgb_sha256": hashes,
        "ffmpeg_decode_processes": decode_runs,
        "completed_utc": completed,
    }
    write_json_create_only(RESULT_PATH, data)
    result_sha256 = sha256_file(RESULT_PATH)
    execution = {
        "schema": "musetalk_h264_decoded_rgb24_execution_receipt_v1",
        "status": "PASS",
        "returncode": 0,
        "started_utc": STARTED_UTC,
        "finished_utc": completed,
        "python_pid": os.getpid(),
        "python_start_ticks": PYTHON_START_TICKS,
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "argv": sys.argv,
        "source_path": str(Path(__file__)),
        "source_sha256": actual_source_sha256,
        "controller_receipt_sha256": EXPECTED_CONTROLLER_SHA256,
        "actual_output_frame_map_sha256": EXPECTED_MAP_SHA256,
        "ffmpeg_decode_process_count": len(decode_runs),
        "ffmpeg_decode_returncodes": sorted({item["returncode"] for item in decode_runs}),
        "result_path": str(RESULT_PATH),
        "result_sha256": result_sha256,
    }
    write_json_create_only(EXECUTION_PATH, execution)
    return data, execution


STARTED_UTC = utc_now()
PYTHON_START_TICKS = proc_start_ticks(os.getpid())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--source-sha256", required=True)
    args = parser.parse_args()
    try:
        data, execution = run(args.source_sha256)
    except Exception:
        traceback_text = traceback.format_exc()
        failure = {
            "schema": "musetalk_h264_decoded_rgb24_execution_receipt_v1",
            "status": "FAIL",
            "returncode": 1,
            "started_utc": STARTED_UTC,
            "finished_utc": utc_now(),
            "python_pid": os.getpid(),
            "python_start_ticks": PYTHON_START_TICKS,
            "cwd": os.getcwd(),
            "python_executable": sys.executable,
            "argv": sys.argv,
            "source_path": str(Path(__file__)),
            "source_sha256": sha256_file(Path(__file__)),
            "traceback": traceback_text,
        }
        try:
            REMOTE_RUN_ROOT.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass
        try:
            write_json_create_only(EXECUTION_PATH, failure)
        except FileExistsError:
            pass
        print(traceback_text, file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "status": data["status"],
                "result_path": execution["result_path"],
                "result_sha256": execution["result_sha256"],
                "execution_receipt_path": str(EXECUTION_PATH),
                "comparisons_vs_zero": data["comparisons_vs_zero"],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
