#!/usr/bin/env python3
"""Receipt-bound Stage-A adapter for the qualified Ditto runtime.

The Stage-A core supplies canonical generator audio and a source portrait via
the shared five-argument CLI.  This shim calls the already-qualified native
renderer with immutable source/model/runtime pins, then records the exact
silent-video, input-audio, source-clock, and native-render receipt.  Stage-A
remuxes and scores the original natural PCM independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from fractions import Fraction
from pathlib import Path
from typing import Any

REMOTE_PROJECT = Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003")
DETERMINISM_BOOTSTRAP = REMOTE_PROJECT / "scripts/experiments/cross_tfg_audio_head_20261003/runtime_determinism_attempt02"
MODEL_DEPLOY_ROOT = Path("/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt03/models/ditto_pytorch")
SOURCE_ROOT = REMOTE_PROJECT / "model_sources_attempt02/ditto-talkinghead"
CONFIG_PATH = MODEL_DEPLOY_ROOT / "ditto_cfg/v0.4_hubert_cfg_pytorch.pkl"
RENDERER_PATH = REMOTE_PROJECT / "model_deploy_attempt18/code/render_silent_video_attempt07.py"
RENDERER_SHA256 = "1415ee28206ba6afe70e32fb728b29fb3b49bb638bdc21d3c2a7e89c66131df0"
EXPECTED_FFMPEG_SHA256 = "36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45"
FFMPEG = "/usr/bin/ffmpeg"
FFPROBE = "/usr/bin/ffprobe"
DITTO_SOURCE_COMMIT = "c3e47eee2e626500017a0556b470d6d4182f85e8"
PROFILE = "ditto_hubert_onnx_cpu_torch_cuda_fixed_portrait"
RECEIPT_SCHEMA = "cross_tfg_ditto_stagea_adapter_receipt_v1"
GPU_MIN_FREE_MIB = 18000
GPU_WAIT_TIMEOUT_SECONDS = 1800
GPU_POLL_SECONDS = 15
GPU_PROGRESS_SECONDS = 30

IMPORT_REPORT = REMOTE_PROJECT / "job_control/model_deploy_attempt13/ditto_streamsdk_import_probe_attempt12/ditto_streamsdk_import_probe_attempt12.json"
IMPORT_REPORT_SHA256 = "2ba259045dd79c797f7c0e34403f600ab1d0accc40fa03ca2aa820c8c80f7806"
RUNTIME_EXTENSION = REMOTE_PROJECT / "model_deploy_attempt13/code/import_probe_native_cache_attempt12/pyximport/lib.linux-x86_64-3.10/core/utils/blend/blend.cpython-310-x86_64-linux-gnu.so"
RUNTIME_EXTENSION_SHA256 = "ce1524957cf0ef6d6f8065630c3caebb654485576867a2a91450779277451509"
RUNTIME_EXTENSION_BYTES = 281144
HEADER_REPORT_SHA256 = "b997b9e98ed61be480788442237bf1008f4a5cef24e74dd55bdb71ea55b1f0cd"
HEADER_TREE_SHA256 = "e9dcde923150324ed29f3e8104d18f9ca61f61b7b32b07a6a1874e47c15a2a6e"
SETUPTOOLS_STAGE_SHA256 = "2ecaf424eb3a2ccdacb903fab9ac80134d21417ed8d27e6d322d1be51e1d7746"

OVERLAYS = (
    REMOTE_PROJECT / "model_deploy_attempt13/code/overlays/setuptools_overlay_attempt12",
    REMOTE_PROJECT / "model_deploy_attempt13/code/overlays/cython_overlay_attempt12",
    REMOTE_PROJECT / "model_deploy_attempt13/code/overlays/filetype_overlay",
    REMOTE_PROJECT / "model_deploy_attempt10/code/future_overlay_attempt02",
    REMOTE_PROJECT / "model_deploy_attempt09/code/overlay_python310_attempt02",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_gate_progress(control_dir: Path, event: dict[str, Any]) -> None:
    control_dir.mkdir(parents=True, exist_ok=True)
    progress_path = control_dir / "ditto_gpu_gate_current.json"
    temporary = progress_path.with_name(progress_path.name + ".tmp")
    temporary.write_text(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    os.replace(temporary, progress_path)
    with (control_dir / "ditto_gpu_gate_events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def wait_for_gpu_headroom(row_id: str) -> dict[str, Any]:
    raw_control = os.environ.get("CROSS_TFG_CONTROL_DIR")
    if not raw_control:
        raise RuntimeError("CROSS_TFG_CONTROL_DIR is required for the per-row GPU wait gate")
    control_dir = Path(raw_control)
    started = time.monotonic()
    last_progress = -GPU_PROGRESS_SECONDS
    while True:
        elapsed = time.monotonic() - started
        try:
            probe = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                check=False, timeout=10,
            )
            if probe.returncode != 0:
                raise RuntimeError(probe.stderr.decode("utf-8", "replace")[-1000:])
            free_mib = int(probe.stdout.decode("ascii", "strict").splitlines()[0].strip())
            probe_error = None
        except Exception as exc:
            free_mib = None
            probe_error = repr(exc)
        now = time.time()
        if free_mib is not None and free_mib >= GPU_MIN_FREE_MIB:
            event = {
                "schema": "cross_tfg_ditto_gpu_wait_v1", "status": "PASS_HEADROOM",
                "row_id": row_id, "gpu_free_mib": free_mib,
                "minimum_free_mib": GPU_MIN_FREE_MIB,
                "wait_seconds": round(elapsed, 3), "timestamp_unix": now,
            }
            _write_gate_progress(control_dir, event)
            print(json.dumps({"event": "DITTO_GPU_GATE_PASS", **event}, sort_keys=True), flush=True)
            return event
        if elapsed >= GPU_WAIT_TIMEOUT_SECONDS:
            event = {
                "schema": "cross_tfg_ditto_gpu_wait_v1", "status": "BOUNDED_WAIT_EXPIRED",
                "row_id": row_id, "gpu_free_mib": free_mib,
                "minimum_free_mib": GPU_MIN_FREE_MIB,
                "wait_seconds": round(elapsed, 3), "timestamp_unix": now,
                "probe_error": probe_error,
            }
            _write_gate_progress(control_dir, event)
            raise RuntimeError("DITTO_GPU_GATE_BOUNDED_WAIT_EXPIRED: " + json.dumps(event, sort_keys=True))
        if elapsed - last_progress >= GPU_PROGRESS_SECONDS:
            event = {
                "schema": "cross_tfg_ditto_gpu_wait_v1", "status": "WAITING_FOR_HEADROOM",
                "row_id": row_id, "gpu_free_mib": free_mib,
                "minimum_free_mib": GPU_MIN_FREE_MIB,
                "wait_seconds": round(elapsed, 3), "timestamp_unix": now,
                "probe_error": probe_error,
            }
            _write_gate_progress(control_dir, event)
            print(json.dumps({"event": "DITTO_GPU_GATE_WAIT", **event}, sort_keys=True), flush=True)
            last_progress = elapsed
        time.sleep(GPU_POLL_SECONDS)


def verify_runtime() -> dict[str, Any]:
    if not RENDERER_PATH.is_file() or sha256_file(RENDERER_PATH) != RENDERER_SHA256:
        raise RuntimeError("qualified Ditto renderer is absent or has changed")
    if not IMPORT_REPORT.is_file() or sha256_file(IMPORT_REPORT) != IMPORT_REPORT_SHA256:
        raise RuntimeError("qualified full StreamSDK import report is absent or has changed")
    report = json.loads(IMPORT_REPORT.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_FULL_STREAMSDK_IMPORT_ONLY" or report.get("forward_performed") is not False:
        raise RuntimeError("full StreamSDK import receipt has wrong status/scope")
    if not RUNTIME_EXTENSION.is_file() or RUNTIME_EXTENSION.stat().st_size != RUNTIME_EXTENSION_BYTES:
        raise RuntimeError("qualified runtime blend extension is absent or has changed size")
    runtime_extension_sha = sha256_file(RUNTIME_EXTENSION)
    if runtime_extension_sha != RUNTIME_EXTENSION_SHA256:
        raise RuntimeError("qualified runtime blend extension SHA mismatch")
    if not CONFIG_PATH.is_file() or not SOURCE_ROOT.is_dir():
        raise RuntimeError("pinned Ditto source/config root is absent")
    ffmpeg = Path(FFMPEG).resolve(strict=True)
    if sha256_file(ffmpeg) != EXPECTED_FFMPEG_SHA256:
        raise RuntimeError("Stage-A/renderer FFmpeg pin mismatch")
    ffprobe = Path(FFPROBE).resolve(strict=True)
    missing = [str(path) for path in OVERLAYS if not path.is_dir()]
    if missing:
        raise RuntimeError(f"qualified Ditto runtime overlay is absent: {missing}")
    return {
        "renderer_path": str(RENDERER_PATH), "renderer_sha256": RENDERER_SHA256,
        "ditto_source_commit": DITTO_SOURCE_COMMIT,
        "source_root": str(SOURCE_ROOT), "weights_root": str(MODEL_DEPLOY_ROOT),
        "config_path": str(CONFIG_PATH), "config_sha256": sha256_file(CONFIG_PATH),
        "import_report_path": str(IMPORT_REPORT), "import_report_sha256": IMPORT_REPORT_SHA256,
        "header_report_sha256": HEADER_REPORT_SHA256, "header_tree_sha256": HEADER_TREE_SHA256,
        "setuptools_stage_sha256": SETUPTOOLS_STAGE_SHA256,
        "runtime_extension_path": str(RUNTIME_EXTENSION),
        "runtime_extension_sha256": runtime_extension_sha, "runtime_extension_bytes": RUNTIME_EXTENSION_BYTES,
        "overlay_paths": [str(path) for path in OVERLAYS],
        "ffmpeg_path": str(ffmpeg), "ffmpeg_sha256": EXPECTED_FFMPEG_SHA256,
        "ffprobe_path": str(ffprobe),
    }


def frame_clock(path: Path, ffprobe: str) -> dict[str, Any]:
    argv = [ffprobe, "-v", "error", "-select_streams", "v:0", "-show_frames",
            "-show_entries", "stream=codec_type,width,height,r_frame_rate,avg_frame_rate:frame=best_effort_timestamp_time,pkt_pts_time",
            "-of", "json", str(path)]
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, timeout=120)
    if proc.returncode != 0:
        raise RuntimeError(f"Ditto output clock probe failed: {proc.stderr.decode('utf-8', 'replace')[-2000:]}")
    probe = json.loads(proc.stdout.decode("utf-8"))
    streams = probe.get("streams", [])
    frames = probe.get("frames", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "video" or not frames:
        raise RuntimeError("Ditto adapter output must contain exactly one video stream and frames")
    rate_text = streams[0].get("avg_frame_rate") or streams[0].get("r_frame_rate")
    if Fraction(str(rate_text)) != Fraction(25, 1):
        raise RuntimeError(f"Ditto output FPS differs from frozen 25/1 source clock: {rate_text}")
    pts: list[dict[str, int]] = []
    for index, frame in enumerate(frames):
        raw = frame.get("best_effort_timestamp_time", frame.get("pkt_pts_time"))
        if raw is None:
            raise RuntimeError(f"Ditto frame {index} has no presentation timestamp")
        timestamp = Fraction(str(raw))
        expected = Fraction(index, 25)
        if timestamp != expected:
            raise RuntimeError(f"Ditto PTS discontinuity at frame {index}: {timestamp} != {expected}")
        pts.append({"frame_index": index, "timestamp_num": timestamp.numerator,
                    "timestamp_den": timestamp.denominator})
    return {"fps_num": 25, "fps_den": 1, "frame_count": len(frames),
            "width": int(streams[0].get("width", 0)), "height": int(streams[0].get("height", 0)),
            "pts_policy": "decoded frames start at zero and each PTS equals frame_index/25; no retiming",
            "frames": pts, "ffprobe_argv": argv}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", type=Path, required=True)
    parser.add_argument("--face", type=Path, required=True)
    parser.add_argument("--outfile", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fps", type=int, choices=(25,), default=25)
    args = parser.parse_args()

    row_id = os.environ.get("CROSS_TFG_ROW_ID", "")
    env_model_id = os.environ.get("CROSS_TFG_MODEL_ID", "")
    env_frontend = os.environ.get("CROSS_TFG_AUDIO_FRONTEND", "")
    if not row_id or env_model_id != "ditto_hubert_fixed_portrait_attempt18" or not env_frontend:
        raise RuntimeError("Stage-A model/row/frontend bindings are missing or differ from the frozen Ditto plan")
    if int(os.environ.get("CROSS_TFG_RENDER_SEED", "-1")) != args.seed:
        raise RuntimeError("Stage-A wrapper seed differs from adapter seed")
    gpu_gate = wait_for_gpu_headroom(row_id)
    audio = args.audio.resolve(strict=True)
    face = args.face.resolve(strict=True)
    output = args.outfile.resolve()
    if output.exists():
        raise FileExistsError(output)
    report_path = output.with_suffix(".ditto_native_report.json")
    receipt_path = output.with_suffix(".adapter_receipt.json")
    native_mux = output.with_name(output.stem + "_native_mux.mp4")
    for path in (report_path, receipt_path, native_mux, Path(str(native_mux) + ".tmp.mp4")):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite Ditto render artifact: {path}")

    runtime = verify_runtime()
    ffprobe = runtime["ffprobe_path"]
    renderer_env = os.environ.copy()
    renderer_env.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "20261003",
        "TFG_DETERMINISM_SEED": str(args.seed),
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONNOUSERSITE": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONPATH": os.pathsep.join((str(DETERMINISM_BOOTSTRAP), *[str(path) for path in OVERLAYS], str(SOURCE_ROOT))),
        "CFLAGS": "-I/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt13/code/python310_headers_attempt14/overlay/usr/include "
                  "-I/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt13/code/python310_headers_attempt14/overlay/usr/include/python3.10 "
                  "-I/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt13/code/python310_headers_attempt14/overlay/usr/include/x86_64-linux-gnu/python3.10",
    })
    argv = [sys.executable, "-B", str(RENDERER_PATH), "--model", "ditto",
            "--audio", str(audio), "--face", str(face), "--outfile", str(output),
            "--seed", str(args.seed), "--fps", "25", "--source-root", str(SOURCE_ROOT),
            "--weights-root", str(MODEL_DEPLOY_ROOT), "--config", str(CONFIG_PATH),
            "--ffmpeg", runtime["ffmpeg_path"], "--ffprobe", ffprobe, "--report", str(report_path)]
    stdout_path = output.with_suffix(".ditto_renderer.stdout.log")
    stderr_path = output.with_suffix(".ditto_renderer.stderr.log")
    done = subprocess.run(argv, cwd=str(SOURCE_ROOT), env=renderer_env,
                          stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, timeout=1800)
    stdout_path.write_bytes(done.stdout)
    stderr_path.write_bytes(done.stderr)
    if done.returncode != 0:
        raise RuntimeError(f"pinned Ditto renderer exited {done.returncode}; stderr={stderr_path}")
    if not report_path.is_file() or not output.is_file() or not native_mux.is_file():
        raise RuntimeError("Ditto renderer did not produce its pinned report, silent video, and native mux")
    report_raw = report_path.read_bytes()
    report = json.loads(report_raw.decode("utf-8"))
    if report.get("status") != "PASS_ENGINEERING_ONLY" or report.get("profile") != "ditto_hubert_onnx_cpu_renderer_torch_cuda":
        raise RuntimeError("native Ditto report status/profile differs from qualified attempt18")
    if report.get("audio", {}).get("file_sha256") != sha256_file(audio):
        raise RuntimeError("native Ditto report does not bind the exact generator input WAV")
    if report.get("face", {}).get("file_sha256") != sha256_file(face):
        raise RuntimeError("native Ditto report does not bind the exact planned face image")
    if report.get("silent_video", {}).get("file_sha256") != sha256_file(output):
        raise RuntimeError("native Ditto report does not bind the silent output")
    if report.get("runtime", {}).get("cython_cache", {}).get("blend_extension_sha256_after") != RUNTIME_EXTENSION_SHA256:
        raise RuntimeError("actual Ditto forward did not use the qualified runtime Cython extension")
    sessions = report.get("runtime", {}).get("ort_sessions", [])
    if not sessions or any(session.get("providers") != ["CPUExecutionProvider"] for session in sessions):
        raise RuntimeError("all Ditto ONNX auxiliary sessions must be pinned to CPUExecutionProvider")
    clock = frame_clock(output, ffprobe)
    if clock["frame_count"] != int(report.get("silent_video", {}).get("probe", {}).get("frame_count", -1)):
        raise RuntimeError("independent PTS frame count differs from the native renderer report")
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "status": "PASS_ENGINEERING_ONLY",
        "row_id": row_id,
        "seed": args.seed,
        "model_id": env_model_id,
        "audio_frontend": env_frontend,
        "adapter_path": str(Path(__file__).resolve()),
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "model_profile": PROFILE,
        "renderer_sha256": RENDERER_SHA256,
        "source_commit": DITTO_SOURCE_COMMIT,
        "native_report_path": str(report_path),
        "native_report_sha256": sha256_file(report_path),
        "native_mux_path": str(native_mux),
        "native_mux_sha256": sha256_file(native_mux),
        "runtime_bindings": runtime,
        "gpu_gate": gpu_gate,
        "gpu_gate_progress_path": str(Path(os.environ["CROSS_TFG_CONTROL_DIR"]) / "ditto_gpu_gate_current.json"),
        "input_audio": {
            "path": str(audio), "file_sha256": sha256_file(audio),
            "pcm_float32_sha256": report["audio"].get("pcm_float32_sha256"),
            "sample_rate_hz": report["audio"].get("sample_rate_hz"),
            "channels": report["audio"].get("channels"),
            "sample_count": report["audio"].get("sample_count"),
        },
        "source_face": {"path": str(face), "file_sha256": sha256_file(face)},
        "silent_video": {
            "path": str(output), "file_sha256": sha256_file(output),
            "bytes": output.stat().st_size, **clock,
        },
        "original_pcm_policy": "adapter does not use its native AAC mux for Stage-A scoring; the pinned Stage-A core remuxes the original natural PCM_F32LE and derives the official PCM16 score window without retiming",
    }
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True,
                                       separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps({"status": receipt["status"], "row_id": row_id, "seed": args.seed,
                      "silent_video": receipt["silent_video"],
                      "adapter_receipt": str(receipt_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
