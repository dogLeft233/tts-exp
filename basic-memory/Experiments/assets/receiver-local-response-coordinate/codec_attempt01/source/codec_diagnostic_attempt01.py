#!/usr/bin/env python3
"""Source-only runner for a six-row FFV1 codec diagnostic.

This file is intentionally not executed during source preparation. It consumes the
already-rendered 504 pre-codec PNGs, creates a separate FFV1 512->224 chain, muxes
the pinned natural PCM, and calls the existing Stage-A SyncNet evaluator serially.
Its outputs are diagnostic-only and never replace the formal H.264 readout.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


INPUT_SHA256 = "673ee09cdba130ee70698553f6bd52ca4b4633cf85c94e2092de66bbb36b4889"
BASELINE_READOUT_SHA256 = "1b4f454463a4387b470c744b0f9269b991e2c4e4fd0164132c40a1c87a655c4f"
HELPER_SHA256 = "0febe6f2b43ed4fb089bdf36e2e208cc1cd61681d8e0ebfd7f59cd56dd00b4b6"
EVALUATOR_SHA256 = "b15762e1837fd28eaba8fe9e261ffb35b164246da555cb3a7e3bbc7c645f818b"
SUMMARY_SOURCE_SHA256 = "79bb2df7c3f0ccb8642ae778dce84856e887fce86035df6d0057766980e68f44"
CHECKPOINT_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SYNCNET_INSTANCE_SHA256 = "bd722bf844700d6a923dec2f10d7cacd65c0c6b22d09aa5d44d2509660e900d6"
FPS = Fraction(25, 1)
VSHIFT = 15
EXPECTED = {
    "BAC009S0022W0123": {"speaker_id": "S0022", "frames": 80},
    "BAC009S0024W0122": {"speaker_id": "S0024", "frames": 88},
}
ARM_TO_BASELINE_DIR = {
    "zero": "native_height_zero",
    "plus": "native_height_plus",
    "minus": "native_height_minus",
}
EXPECTED_ROW_IDS = {
    f"{sample_id}/musetalk/{arm}"
    for sample_id in EXPECTED
    for arm in ARM_TO_BASELINE_DIR
}
ROW_ORDER = [
    f"{sample_id}/musetalk/{arm}"
    for sample_id in EXPECTED
    for arm in ("zero", "plus", "minus")
]


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def require_pin(path: Path, expected_sha256: str, label: str) -> Path:
    path = path.resolve(strict=True)
    if path.is_symlink() or not path.is_file() or sha_file(path) != expected_sha256:
        raise RuntimeError(f"{label} source/input pin mismatch: {path}")
    return path


def run_logged(argv: list[str], *, cwd: Path, label: str, log_dir: Path) -> dict[str, Any]:
    result = subprocess.run(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8", errors="replace", check=False)
    record = {
        "label": label,
        "argv": argv,
        "returncode": int(result.returncode),
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
    write_new(log_dir / f"{label}.json", record)
    if result.returncode != 0:
        raise RuntimeError(f"{label} failed with rc={result.returncode}: {result.stderr[-2000:]}")
    return record


def probe_video(ffprobe: str, path: Path, *, size: int, frame_count: int) -> dict[str, Any]:
    argv = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-show_streams", "-show_frames",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,start_time:"
        "frame=best_effort_timestamp_time",
        "-of", "json", str(path),
    ]
    result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE, text=True, encoding="utf-8",
                           errors="replace", check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {result.stderr[-2000:]}")
    document = json.loads(result.stdout)
    streams = document.get("streams", [])
    frames = document.get("frames", [])
    if len(streams) != 1 or len(frames) != frame_count:
        raise RuntimeError(f"unexpected stream/frame count for {path}: {len(streams)}/{len(frames)}")
    stream = streams[0]
    if (stream.get("codec_name") != "ffv1" or stream.get("pix_fmt") != "yuv420p"
            or (int(stream.get("width", -1)), int(stream.get("height", -1))) != (size, size)
            or Fraction(stream.get("avg_frame_rate", "0/1")) != FPS
            or Fraction(stream.get("r_frame_rate", "0/1")) != FPS
            or Fraction(stream.get("start_time", "0")) != 0):
        raise RuntimeError(f"FFV1 video contract failed for {path}: {stream}")
    pts = [Fraction(frame["best_effort_timestamp_time"]) for frame in frames]
    if pts != [Fraction(index, 25) for index in range(frame_count)]:
        raise RuntimeError(f"video PTS differs from canonical j/25 clock: {path}")
    return {"path": str(path), "sha256": sha_file(path), "stream": stream,
            "frame_count": len(frames), "pts_j_over_25": True}


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned helper: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--baseline-readout-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overdub-source", type=Path, required=True)
    parser.add_argument("--stagea-evaluator", type=Path, required=True)
    parser.add_argument("--summary-source", type=Path, required=True)
    parser.add_argument("--syncnet-python", type=Path, required=True)
    parser.add_argument("--syncnet-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ffmpeg", required=True)
    parser.add_argument("--ffprobe", required=True)
    args = parser.parse_args()

    input_manifest_path = require_pin(args.input_manifest, INPUT_SHA256, "six-row PNG input manifest")
    baseline_manifest_path = require_pin(args.baseline_readout_manifest, BASELINE_READOUT_SHA256,
                                         "formal H.264 readout manifest")
    helper_path = require_pin(args.overdub_source, HELPER_SHA256, "strict original-PCM mux helper")
    evaluator_path = require_pin(args.stagea_evaluator, EVALUATOR_SHA256, "Stage-A SyncNet evaluator")
    summary_path = require_pin(args.summary_source, SUMMARY_SOURCE_SHA256, "full/guard20 metric helper")
    checkpoint = require_pin(args.checkpoint, CHECKPOINT_SHA256, "SyncNet checkpoint")
    require_pin(args.syncnet_root / "SyncNetInstance.py", SYNCNET_INSTANCE_SHA256,
                "official SyncNetInstance source")
    if args.output_root.exists() or args.output_root.is_symlink():
        raise FileExistsError(f"codec-diagnostic output root must be absent: {args.output_root}")

    input_manifest = load_json(input_manifest_path)
    baseline_manifest = load_json(baseline_manifest_path)
    if input_manifest.get("schema") != "output_owned_musetalk_readout_input_v1" or input_manifest.get("status") != "PASS":
        raise ValueError("unexpected frozen six-row PNG input manifest")
    if baseline_manifest.get("status") != "PASS" or baseline_manifest.get("row_count") != 6:
        raise ValueError("formal H.264 baseline manifest is not the six-row PASS readout")
    input_rows = {row["row_id"]: row for row in input_manifest.get("rows", [])}
    baseline_rows = {row["row_id"]: row for row in baseline_manifest.get("rows", [])}
    if set(input_rows) != EXPECTED_ROW_IDS or set(baseline_rows) != EXPECTED_ROW_IDS:
        raise ValueError("codec diagnostic must bind exactly the existing two-sample, three-arm rows")

    # Validate every retained PNG and its order before creating any media.
    for row_id in ROW_ORDER:
        row = input_rows[row_id]
        sample_id = row["sample_id"]
        if (row.get("model") != "musetalk" or row.get("seed") != 20261003
                or row.get("fps_num") != 25 or row.get("fps_den") != 1
                or row.get("expected_frames") != EXPECTED[sample_id]["frames"]):
            raise ValueError(f"frozen frame/seed/model identity differs: {row_id}")
        pngs = row.get("precodec_frames", [])
        if len(pngs) != EXPECTED[sample_id]["frames"]:
            raise ValueError(f"pre-codec PNG frame count differs: {row_id}")
        for index, ref in enumerate(pngs):
            if ref.get("frame_index") != index:
                raise ValueError(f"pre-codec PNG frame order differs: {row_id}/{index}")
            png_path = require_pin(Path(ref["path"]), ref["sha256"], f"pre-codec PNG {row_id}/{index}")
            if png_path.name != f"{index:08d}.png":
                raise ValueError(f"pre-codec PNG name does not match canonical sequence: {png_path}")

    baseline_evaluators: dict[str, dict[str, Any]] = {}
    for row_id in ROW_ORDER:
        baseline_row_ref = baseline_rows[row_id]
        row_receipt_path = Path(baseline_row_ref["row_receipt_path"]).resolve(strict=True)
        if sha_file(row_receipt_path) != baseline_row_ref["row_receipt_sha256"]:
            raise RuntimeError(f"formal H.264 row receipt hash mismatch: {row_id}")
        row_receipt = load_json(row_receipt_path)
        evaluator_receipt_path = Path(row_receipt["syncnet"]["evaluator_receipt"]).resolve(strict=True)
        if sha_file(evaluator_receipt_path) != row_receipt["syncnet"]["evaluator_receipt_sha256"]:
            raise RuntimeError(f"formal H.264 evaluator receipt hash mismatch: {row_id}")
        evaluator_receipt = load_json(evaluator_receipt_path)
        if evaluator_receipt.get("status") != "PASS":
            raise RuntimeError(f"formal H.264 evaluator row is not PASS: {row_id}")
        baseline_evaluators[row_id] = evaluator_receipt

    output_root = args.output_root.resolve(strict=False)
    output_root.mkdir(parents=True, exist_ok=False)
    started = {
        "schema": "native_attribute_response_ffv1_codec_diagnostic_v1",
        "status": "RUNNING",
        "scope": "diagnostic-only; existing formal H.264 full/guard20 artifacts are immutable",
        "input_manifest_sha256": sha_file(input_manifest_path),
        "baseline_readout_manifest_sha256": sha_file(baseline_manifest_path),
        "source_pins": {
            "strict_mux_helper_sha256": sha_file(helper_path),
            "stagea_syncnet_evaluator_sha256": sha_file(evaluator_path),
            "full_guard20_summary_source_sha256": sha_file(summary_path),
            "syncnet_checkpoint_sha256": sha_file(checkpoint),
        },
        "codec_chain": [
            {"stage": "512px", "codec": "ffv1", "pix_fmt": "yuv420p"},
            {"stage": "224px", "scale": "scale=224:224:flags=lanczos",
             "codec": "ffv1", "pix_fmt": "yuv420p"},
        ],
        "clock": {"fps_num": 25, "fps_den": 1, "frame_pts": "j/25", "time_warp": False},
        "score_contract": "existing Stage-A evaluator; natural original PCM; full and guard20 from the pinned readout summary helper",
        "row_count_expected": 6,
        "score_invocations_expected": 6,
        "rows": [],
    }
    write_new(output_root / "attempt_started.json", started)
    muxer = load_module("codec_diagnostic_overdub", helper_path)
    summary = load_module("codec_diagnostic_score_summary", summary_path)

    for row_id in sorted(EXPECTED_ROW_IDS):
        row = input_rows[row_id]
        sample_id = row["sample_id"]
        arm = row["arm"]
        frame_count = EXPECTED[sample_id]["frames"]
        natural = input_manifest["audio_contracts"][sample_id]
        natural_path = require_pin(Path(natural["natural_path"]), natural["natural_wav_sha256"],
                                   f"natural PCM WAV {sample_id}")
        row_root = output_root / sample_id / "musetalk" / f"native_height_{arm}"
        row_root.mkdir(parents=True, exist_ok=False)
        log_dir = row_root / "logs"
        log_dir.mkdir()
        png_dir = Path(row["precodec_frames"][0]["path"]).parent
        png_pattern = png_dir / "%08d.png"
        video512 = row_root / "codec_diagnostic_512_ffv1_yuv420p.mkv"
        video224 = row_root / "codec_diagnostic_224_lanczos_ffv1_yuv420p.mkv"
        muxed = row_root / "codec_diagnostic_natural_pcm.mkv"

        encode512 = [args.ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n",
                     "-framerate", "25", "-start_number", "0", "-i", str(png_pattern),
                     "-frames:v", str(frame_count), "-an", "-fps_mode", "passthrough",
                     "-c:v", "ffv1", "-pix_fmt", "yuv420p", str(video512)]
        command512 = run_logged(encode512, cwd=output_root, label="encode_512_ffv1", log_dir=log_dir)
        probe512 = probe_video(args.ffprobe, video512, size=512, frame_count=frame_count)

        encode224 = [args.ffmpeg, "-hide_banner", "-v", "error", "-nostdin", "-n",
                     "-i", str(video512), "-vf", "scale=224:224:flags=lanczos", "-an",
                     "-fps_mode", "passthrough", "-c:v", "ffv1", "-pix_fmt", "yuv420p",
                     str(video224)]
        command224 = run_logged(encode224, cwd=output_root, label="encode_224_ffv1", log_dir=log_dir)
        probe224 = probe_video(args.ffprobe, video224, size=224, frame_count=frame_count)

        mux_info = muxer.mux_original_pcm(video224, natural_path, muxed,
                                          ffmpeg=args.ffmpeg, ffprobe=args.ffprobe,
                                          overwrite=False)
        baseline_eval = baseline_evaluators[row_id]
        if (baseline_eval.get("audio_pcm16_sha256") is None
                or baseline_eval.get("audio_sample_count") != natural["sample_count"]):
            raise RuntimeError(f"formal natural PCM16 contract unavailable or inconsistent: {row_id}")

        matrix_path = row_root / "codec_diagnostic_distance_by_query_lag.npz"
        eval_path = row_root / "codec_diagnostic_syncnet_evaluator_receipt.json"
        reference = f"codec_diag_{sample_id}_{arm}"
        score_argv = [
            str(args.syncnet_python), "-B", str(evaluator_path),
            "--syncnet-root", str(args.syncnet_root), "--checkpoint", str(checkpoint),
            "--input-mkv", str(muxed), "--tmp-dir", str(row_root / "syncnet_tmp"),
            "--output-npz", str(matrix_path), "--output-json", str(eval_path),
            "--reference", reference,
            "--expected-pcm16-sha256", baseline_eval["audio_pcm16_sha256"],
            "--expected-sample-count", str(natural["sample_count"]),
            "--expected-video-frame-count", str(frame_count),
            "--batch-size", "20", "--vshift", str(VSHIFT),
        ]
        score_command = run_logged(score_argv, cwd=output_root, label="stagea_syncnet_score", log_dir=log_dir)
        evaluator_receipt = load_json(eval_path)
        if (evaluator_receipt.get("status") != "PASS"
                or evaluator_receipt.get("audio_pcm16_sha256") != baseline_eval["audio_pcm16_sha256"]
                or evaluator_receipt.get("audio_sample_count") != natural["sample_count"]
                or evaluator_receipt.get("video_frame_count") != frame_count):
            raise RuntimeError(f"codec diagnostic evaluator contract differs: {row_id}")

        score_summary = summary.save_full_guard20(matrix_path, row_root)
        row_receipt_path = row_root / "codec_diagnostic_row_receipt.json"
        write_new(row_receipt_path, {
            "schema": "native_attribute_response_ffv1_codec_diagnostic_row_v1",
            "status": "PASS",
            "classification": "CODEC_DIAGNOSTIC_ONLY_NOT_FORMAL_HEAD_ENDPOINT",
            "row_id": row_id,
            "sample_id": sample_id,
            "speaker_id": row["speaker_id"],
            "arm": arm,
            "seed": row["seed"],
            "frame_count": frame_count,
            "fps": "25/1",
            "precodec_frame_dir": str(png_dir),
            "natural_wav_path": str(natural_path),
            "natural_wav_sha256": natural["natural_wav_sha256"],
            "natural_pcm_float32_sha256": natural["natural_pcm_float32_sha256"],
            "original_h264_row_receipt_sha256": baseline_rows[row_id]["row_receipt_sha256"],
            "original_h264_pcm16_sha256": baseline_eval["audio_pcm16_sha256"],
            "ffmpeg_512_command": command512,
            "ffv1_512_probe": probe512,
            "ffmpeg_224_command": command224,
            "ffv1_224_probe": probe224,
            "mux_original_pcm": mux_info.__dict__,
            "stagea_score_command": score_command,
            "stagea_evaluator_receipt": str(eval_path),
            "stagea_evaluator_receipt_sha256": sha_file(eval_path),
            "distance_matrix_npz": str(matrix_path),
            "distance_matrix_npz_sha256": sha_file(matrix_path),
            "full_and_guard20_diagnostic_summary": score_summary,
            "formal_h264_artifacts_modified": False,
        })
        started["rows"].append({
            "row_id": row_id,
            "status": "PASS",
            "row_receipt_path": str(row_receipt_path),
            "row_receipt_sha256": sha_file(row_receipt_path),
        })

    final = {
        **started,
        "status": "PASS",
        "rows": started["rows"],
        "row_count_completed": 6,
        "score_invocations_completed": 6,
        "formal_h264_artifacts_modified": False,
    }
    write_new(output_root / "codec_diagnostic_manifest.json", final)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
