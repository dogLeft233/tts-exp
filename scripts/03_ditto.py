#!/usr/bin/env python3
"""03_ditto.py - Ditto talking-head inference for 4-condition experiment (issue #4).

Each sample i (1..10) × 4 conditions = 40 videos:
  natural_raw    → natural wav fed directly to Ditto
  natural_resamp → natural wav resampled to 16k via ffmpeg before Ditto
  tts_raw        → TTS wav fed directly to Ditto
  tts_resamp     → TTS wav resampled to 16k via ffmpeg before Ditto

Image: data/data/image/{i}.png (same for all 4 conditions).
Uses ditto-talkinghead/inference.py via subprocess.
TRT online mode first; falls back to PyTorch if TRT engine load fails (plan Q33).

Output: runs/<run_id>/03_ditto/{condition}/{i}.mp4 + ditto_meta.json
"""

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_config(repo: Path) -> dict:
    cfg_path = repo / "scripts" / "config.yaml"
    return yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}


def resample_16k(src: Path, dst: Path, ditto_bin: str = "") -> bool:
    """ffmpeg resample to 16kHz mono 16-bit. Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    env = None
    ffmpeg = "ffmpeg"
    if ditto_bin:
        ffmpeg = str(Path(ditto_bin) / "ffmpeg")
        env = {"PATH": f"{ditto_bin}:{os.environ.get('PATH', '')}"}
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30, env=env)
        return True
    except Exception:
        return False


def run_ditto(
    repo: Path,
    run_id: str,
    cfg: dict,
    audio_path: Path,
    source_path: Path,
    output_path: Path,
    mode: str,
    retries: int = 1,
) -> bool:
    """Call ditto inference.py. Returns True on success."""
    if mode == "trt_online":
        data_root = cfg["ditto"]["trt_online"]["data_root"]
        cfg_pkl = cfg["ditto"]["trt_online"]["cfg_pkl"]
    else:
        data_root = cfg["ditto"]["pytorch_fallback"]["data_root"]
        cfg_pkl = cfg["ditto"]["pytorch_fallback"]["cfg_pkl"]

    # Ensure ffmpeg from the ditto conda env is in PATH for audio muxing
    env = os.environ.copy()
    ditto_bin = str(Path(cfg["paths"]["envs_dir"]) / "ditto" / "bin")
    env["PATH"] = ditto_bin + ":" + env.get("PATH", "")
    if "LD_LIBRARY_PATH" not in env:
        libdir = str(Path(cfg["paths"]["envs_dir"]) / "ditto" / "opt" / "cudnn8" / "lib")
        if Path(libdir).is_dir():
            env["LD_LIBRARY_PATH"] = libdir

    ditto_dir = repo / cfg["paths"]["third_party"] / "ditto-talkinghead"
    for script_name in ("inference.py", "inference_online.py"):
        script = ditto_dir / script_name
        if not script.exists():
            continue
        cmd = [
            "python", str(script),
            "--data_root", data_root,
            "--cfg_pkl", cfg_pkl,
            "--audio_path", str(audio_path),
            "--source_path", str(source_path),
            "--output_path", str(output_path),
        ]
        for attempt in range(retries + 1):
            try:
                if attempt > 0:
                    time.sleep(1)
                subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300, env=env)
                return True
            except subprocess.CalledProcessError as e:
                if "libcudnn" in (e.stderr or "") or "Could not load" in (e.stderr or ""):
                    return False
        return False
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Ditto 4-condition inference")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    run_dir = repo / "runs" / args.run_id
    out_base = run_dir / "03_ditto"
    out_base.mkdir(parents=True, exist_ok=True)

    cfg = load_config(repo)
    ditto_bin = str(Path(cfg["paths"]["envs_dir"]) / "ditto" / "bin")
    sample_ids = [1] if args.smoke else list(range(1, 11))
    conditions = cfg.get("ditto", {}).get("conditions", ["natural_raw", "natural_resamp", "tts_raw", "tts_resamp"])

    audio_dir = repo / "data" / "data" / "audio"
    tts_dir = run_dir / "02_tts"
    img_dir = repo / "data" / "data" / "image"

    # Check if tts wavs exist
    _ = all((tts_dir / f"{i}.wav").exists() for i in sample_ids)  # noqa

    mode = "trt_online"  # start with TRT, fall back to pytorch if needed
    results: dict = {}
    failed: list[dict] = []
    t0 = time.monotonic()

    for cond in conditions:
        cond_dir = out_base / cond
        cond_dir.mkdir(exist_ok=True)

        for i in sample_ids:
            img_path = img_dir / f"{i}.png"

            # Determine audio source
            if cond.startswith("natural"):
                audio_src = audio_dir / f"{i}.wav"
            else:  # tts
                audio_src = tts_dir / f"{i}.wav"
                if not audio_src.exists():
                    failed.append({"condition": cond, "sample_id": i, "error": "tts audio missing"})
                    continue

            # Resample if needed
            if cond.endswith("_resamp"):
                tmp_audio = cond_dir / f"{i}_16k.wav"
                if not resample_16k(audio_src, tmp_audio, ditto_bin):
                    failed.append({"condition": cond, "sample_id": i, "error": "resample failed"})
                    continue
                feed_audio = tmp_audio
            else:
                feed_audio = audio_src

            out_video = cond_dir / f"{i}.mp4"
            if out_video.exists():
                results[f"{cond}:{i}"] = "ok-cached"
                continue
            print(f"[ditto] {cond}:{i} ...")

            ok = run_ditto(repo, args.run_id, cfg, feed_audio, img_path, out_video, mode)
            if ok:
                results[f"{cond}:{i}"] = "ok"
                print(f"  -> {out_video}")
            elif mode == "trt_online":
                # TRT failed — retry with PyTorch
                print(f"  TRT failed, switching to PyTorch ...")
                mode = "pytorch"
                ok = run_ditto(repo, args.run_id, cfg, feed_audio, img_path, out_video, mode)
                if ok:
                    results[f"{cond}:{i}"] = "ok-pytorch"
                else:
                    failed.append({"condition": cond, "sample_id": i, "error": "ditto inference failed"})
            else:
                failed.append({"condition": cond, "sample_id": i, "error": "ditto inference failed"})

    elapsed = time.monotonic() - t0
    meta = {
        "mode_used": mode,
        "conditions": conditions,
        "samples": sample_ids,
        "time_s": round(elapsed, 1),
        "results": results,
        "failed": failed,
    }
    (out_base / "ditto_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    ok_count = sum(1 for v in results.values() if "ok" in v)
    print(f"[ditto] {ok_count} ok, {len(failed)} failed ({elapsed:.0f}s) mode={mode}")


if __name__ == "__main__":
    main()
