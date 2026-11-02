#!/usr/bin/env python3
"""Read-only visual/timing check plus exact natural-F32 delivery mux for Ditto18.

This is an engineering artifact check, not a SyncNet effect evaluation. It copies
the already-rendered video stream and the unchanged natural PCM_F32LE track.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any

EXPECTED_VIDEO_SHA256 = "2e0c042d44efb42f6db6254ff52bb65277aaad70efffdff7f0ec73941d87791c"
EXPECTED_AUDIO_WAV_SHA256 = "d12ced7d44e7d70713a7e201abe47ff60ec190af20e9cd447cba5d325c11caa4"
EXPECTED_AUDIO_PCM_F32_SHA256 = "b9a24dd67fd557e012e58339ebe5b1d22655240cf13dd3e31ae7173f3bd0d15a"
EXPECTED_VIDEO_BYTES = 222207
EXPECTED_AUDIO_SAMPLES = 92225
EXPECTED_FFMPEG_SHA256 = "2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def run(argv: list[str], timeout: int = 300) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, check=False, timeout=timeout)


def ffmpeg_bytes(ffmpeg: Path, argv: list[str], *, timeout: int = 300) -> bytes:
    proc = run([str(ffmpeg), *argv], timeout=timeout)
    if proc.returncode:
        raise RuntimeError(f"FFmpeg command failed ({proc.returncode}): {proc.stderr.decode('utf-8', 'replace')[-4000:]}")
    return proc.stdout


def frame_pts(ffmpeg: Path, video: Path) -> list[Fraction]:
    proc = run([str(ffmpeg), "-hide_banner", "-nostdin", "-i", str(video), "-vf", "showinfo",
                "-an", "-f", "null", "NUL"], timeout=300)
    if proc.returncode:
        # FFmpeg's null muxer can return non-zero on some Windows builds when
        # NUL is treated as an input-like filename; showinfo records are still
        # accepted only if all frames were decoded and exactly timestamped.
        text = proc.stderr.decode("utf-8", "replace")
        if "frame=" not in text:
            raise RuntimeError(f"FFmpeg frame-clock decode failed ({proc.returncode}): {text[-3000:]}")
    text = proc.stderr.decode("utf-8", "replace")
    found: list[tuple[int, Fraction]] = []
    pattern = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:([0-9]+(?:\.[0-9]+)?)")
    for match in pattern.finditer(text):
        found.append((int(match.group(1)), Fraction(match.group(3))))
    if not found:
        raise RuntimeError("FFmpeg showinfo emitted no frame PTS records")
    if [i for i, _ in found] != list(range(len(found))):
        raise RuntimeError("decoded frame indices are not contiguous from zero")
    pts = [stamp for _, stamp in found]
    expected = [Fraction(i, 25) for i in range(len(pts))]
    if pts != expected:
        raise RuntimeError("decoded Ditto frame PTS do not equal index/25 exactly")
    return pts


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--silent-video", type=Path, required=True)
    p.add_argument("--natural-wav", type=Path, required=True)
    p.add_argument("--ffmpeg", type=Path, required=True)
    p.add_argument("--output-mkv", type=Path, required=True)
    p.add_argument("--contact-sheet", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    args = p.parse_args()

    video = args.silent_video.resolve(strict=True)
    audio = args.natural_wav.resolve(strict=True)
    ffmpeg = args.ffmpeg.resolve(strict=True)
    if video.stat().st_size != EXPECTED_VIDEO_BYTES or sha_file(video) != EXPECTED_VIDEO_SHA256:
        raise ValueError("Ditto18 silent video does not match the actual remote readback seal")
    if sha_file(audio) != EXPECTED_AUDIO_WAV_SHA256:
        raise ValueError("natural float32 WAV does not match the qualified Ditto input seal")
    if sha_file(ffmpeg) != EXPECTED_FFMPEG_SHA256:
        raise ValueError("local FFmpeg binary differs from the pinned contact-sheet tool")
    for path in (args.output_mkv, args.contact_sheet, args.report):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)

    input_pcm = ffmpeg_bytes(ffmpeg, ["-v", "error", "-i", str(audio), "-map", "0:a:0", "-vn",
                                       "-f", "f32le", "pipe:1"])
    if len(input_pcm) != EXPECTED_AUDIO_SAMPLES * 4 or sha_bytes(input_pcm) != EXPECTED_AUDIO_PCM_F32_SHA256:
        raise ValueError("natural WAV decoded float32 samples differ from the frozen PCM binding")

    mux_argv = [str(ffmpeg), "-v", "error", "-nostdin", "-y", "-i", str(video), "-i", str(audio),
                "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "copy",
                "-map_metadata", "0", "-f", "matroska", str(args.output_mkv)]
    mux = run(mux_argv, timeout=300)
    if mux.returncode or not args.output_mkv.is_file():
        raise RuntimeError(f"natural PCM Matroska mux failed: {mux.stderr.decode('utf-8', 'replace')[-4000:]}")

    mux_pcm = ffmpeg_bytes(ffmpeg, ["-v", "error", "-i", str(args.output_mkv), "-map", "0:a:0", "-vn",
                                     "-f", "f32le", "pipe:1"])
    if mux_pcm != input_pcm:
        raise RuntimeError("decoded Matroska audio is not bit-identical to the natural float32 PCM")
    video_es_before = ffmpeg_bytes(ffmpeg, ["-v", "error", "-i", str(video), "-map", "0:v:0",
                                             "-c:v", "copy", "-f", "h264", "pipe:1"])
    video_es_after = ffmpeg_bytes(ffmpeg, ["-v", "error", "-i", str(args.output_mkv), "-map", "0:v:0",
                                            "-c:v", "copy", "-f", "h264", "pipe:1"])
    if video_es_before != video_es_after:
        raise RuntimeError("the lossless audio mux changed the video elementary stream")

    pts = frame_pts(ffmpeg, video)
    contact = run([str(ffmpeg), "-v", "error", "-nostdin", "-i", str(video), "-vf",
                   "select='not(mod(n,9))',scale=384:-1,tile=4x4", "-frames:v", "1", "-vsync", "0",
                   "-y", str(args.contact_sheet)], timeout=300)
    if contact.returncode or not args.contact_sheet.is_file():
        raise RuntimeError(f"contact sheet generation failed: {contact.stderr.decode('utf-8', 'replace')[-3000:]}")

    report: dict[str, Any] = {
        "schema": "DITTO18_ORIGINAL_PCM_F32_MUX_QC_V1",
        "status": "PASS_ENGINEERING_ONLY",
        "scope": "existing qualified Ditto portrait; unchanged original natural PCM_F32LE mux and decoded frame-clock check; no SyncNet effects",
        "inputs": {
            "silent_video": {"path": str(video), "bytes": video.stat().st_size, "sha256": sha_file(video)},
            "natural_wav": {"path": str(audio), "bytes": audio.stat().st_size, "sha256": sha_file(audio),
                            "pcm_float32_sha256": sha_bytes(input_pcm), "sample_rate_hz": 16000,
                            "channels": 1, "sample_count": len(input_pcm) // 4},
            "ffmpeg": {"path": str(ffmpeg), "sha256": sha_file(ffmpeg)},
        },
        "output": {"path": str(args.output_mkv.resolve()), "bytes": args.output_mkv.stat().st_size,
                   "sha256": sha_file(args.output_mkv), "audio_codec": "pcm_f32le",
                   "decoded_audio_exact": True, "video_es_sha256": sha_bytes(video_es_after),
                   "video_es_unchanged": True},
        "video_clock": {"fps_num": 25, "fps_den": 1, "frame_count": len(pts),
                        "all_pts_equal_frame_index_over_25": True,
                        "frame_pts_num_den": [{"num": t.numerator, "den": t.denominator} for t in pts]},
        "contact_sheet": {"path": str(args.contact_sheet.resolve()), "bytes": args.contact_sheet.stat().st_size,
                          "sha256": sha_file(args.contact_sheet), "sampled_frame_indices": list(range(0, len(pts), 9))[:16]},
        "argv": mux_argv,
    }
    args.report.write_text(json.dumps(report, sort_keys=True, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(args.report.resolve()),
                      "mux_sha256": report["output"]["sha256"], "frames": len(pts),
                      "contact_sheet": str(args.contact_sheet.resolve())}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
