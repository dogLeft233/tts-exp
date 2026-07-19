#!/usr/bin/env python3
"""03_ditto.py - Ditto talking-head inference for 4-condition experiment (issue #4).

Each sample i (1..10) × 4 conditions = 40 videos:
  All arms: EBU R128 two-pass loudness normalisation to -24 LUFS before Ditto.
  natural_raw    → loudnorm natural wav → Ditto
  natural_resamp → loudnorm natural wav → resample 16k → Ditto
  tts_raw        → loudnorm TTS wav → Ditto
  tts_resamp     → loudnorm TTS wav → resample 16k → Ditto

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

from utils import detect_sample_ids, load_config, resolve_repo_path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ffmpeg_bin(ditto_bin: str) -> str:
    return str(Path(ditto_bin) / "ffmpeg") if ditto_bin else "ffmpeg"


def resample_16k(src: Path, dst: Path, ditto_bin: str = "") -> bool:
    """ffmpeg resample to 16kHz mono 16-bit. Returns True on success."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = _ffmpeg_bin(ditto_bin)
    cmd = [ffmpeg, "-y", "-v", "error", "-i", str(src), "-ar", "16000", "-ac", "1", "-sample_fmt", "s16", str(dst)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        return True
    except Exception:
        return False


def loudnorm_audio(
    src: Path,
    dst: Path,
    ditto_bin: str = "",
    target_lufs: float = -24.0,
) -> dict | None:
    """EBU R128 two-pass loudness normalisation via ffmpeg loudnorm.

    Pass 1 measures integrated loudness / LRA / true-peak.
    Pass 2 applies linear normalisation with measured values, limiting
    write-out to ``dst`` as 24kHz mono s16 (matching Ditto native).
    Returns measurement dict on success, None on failure.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    ff = _ffmpeg_bin(ditto_bin)
    env = os.environ.copy()
    if ditto_bin:
        env["PATH"] = f"{ditto_bin}:{env.get('PATH', '')}"

    def _measure(inpath):
        cmd = [
            ff, "-y", "-v", "info", "-nostats",
            "-i", str(inpath),
            "-af", f"loudnorm=I={target_lufs}:LRA=11:tp=-1.5:print_format=json",
            "-f", "null", "-",
        ]
        try:
            out = subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=30, env=env)
            idx = out.rfind("{")
            if idx >= 0:
                return json.loads(out[idx:].rstrip("\x00"))
        except subprocess.CalledProcessError:
            pass
        return None

    # pass 1 — measure
    m = _measure(src)
    if not m:
        return None

    # pass 2 — apply linear gain with measured refs
    cmd2 = [
        ff, "-y", "-v", "error",
        "-i", str(src),
        "-af",
        f"loudnorm=I={target_lufs}:LRA=11:tp=-1.5"
        f":measured_I={m['input_i']}:measured_LRA={m['input_lra']}"
        f":measured_tp={m['input_tp']}:measured_thresh={m['input_thresh']}"
        f":linear=true:print_format=summary",
        "-ac", "1", "-sample_fmt", "s16", str(dst),
    ]
    try:
        subprocess.run(cmd2, check=True, capture_output=True, text=True, timeout=60, env=env)
        return m
    except subprocess.CalledProcessError:
        return None


def run_ditto(
    repo: Path,
    run_id: str,
    cfg: dict,
    audio_path: Path,
    source_path: Path,
    output_path: Path,
    mode: str,
    retries: int = 1,
    seed: int | None = None,
) -> bool:
    """Call Ditto through the seeded adapter."""
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

    # ditto-talkinghead has a single `inference.py` entrypoint; the cfg_pkl
# (v0.4_hubert_cfg_trt_online.pkl vs v0.4_hubert_cfg_trt.pkl) selects online
# vs offline streaming internally. ditto.md names `inference_online.py` but the
# file does not exist in this checked-out ditto version, so we always use
# inference.py and pass the cfg_pkl through (verified manually).
    ditto_dir = repo / cfg["paths"]["third_party"] / "ditto-talkinghead"
    script = ditto_dir / "inference.py"
    if not script.exists():
        return False
    effective_seed = int(seed if seed is not None else cfg.get("ditto", {}).get("seed", 42))
    adapter = repo / "scripts" / "ditto_seeded_inference.py"
    cmd = [
        "python", str(adapter),
        "--ditto-dir", str(ditto_dir),
        "--data-root", data_root,
        "--cfg-pkl", cfg_pkl,
        "--audio-path", str(audio_path),
        "--source-path", str(source_path),
        "--output-path", str(output_path),
        "--seed", str(effective_seed),
    ]
    last_error = "Ditto inference failed"
    for attempt in range(retries + 1):
        try:
            if attempt > 0:
                time.sleep(1)
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=300, env=env)
            return True
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr or ""
            last_error = stderr[-1000:] or str(exc)
            if "libcudnn" in stderr or "Could not load" in stderr:
                return False
        except (subprocess.TimeoutExpired, OSError) as exc:
            last_error = str(exc)
    if last_error:
        print(f"[ditto] inference error: {last_error}")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(description="Ditto 4-condition inference")
    ap.add_argument("--run_id", "--run-id", dest="run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default="")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--no-cache", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    run_dir = repo / "runs" / args.run_id
    out_base = run_dir / "03_ditto"
    out_base.mkdir(parents=True, exist_ok=True)

    cfg = load_config(repo, args.config or None)
    ditto_bin = str(Path(cfg["paths"]["envs_dir"]) / "ditto" / "bin")
    sample_ids = detect_sample_ids(repo, args.smoke, cfg=cfg)
    conditions = cfg.get("ditto", {}).get("conditions", ["natural_raw", "natural_resamp", "tts_raw", "tts_resamp"])
    do_loudnorm = cfg.get("ditto", {}).get("loudnorm", True)
    seed = int(args.seed if args.seed is not None else cfg.get("ditto", {}).get("seed", 42))

    input_stage = cfg.get("ditto", {}).get("input_stage")
    if input_stage:
        pair_base = run_dir / input_stage
        audio_dir = pair_base / "natural"
        tts_dir = pair_base / "tts"
    else:
        audio_dir = resolve_repo_path(repo, cfg.get("paths", {}).get("audio_dir", "data/data/audio"))
        configured_tts = cfg.get("paths", {}).get("tts_audio_dir")
        tts_dir = resolve_repo_path(repo, configured_tts) if configured_tts else run_dir / "02_tts"
    img_dir = resolve_repo_path(repo, cfg.get("paths", {}).get("image_dir", "data/data/image"))

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
            if cond.startswith("natural") or cond == "natural":
                audio_src = audio_dir / f"{i}.wav"
            else:  # tts
                audio_src = tts_dir / f"{i}.wav"
                if not audio_src.exists():
                    failed.append({"condition": cond, "sample_id": i, "error": "tts audio missing"})
                    continue

            # Step A: EBU R128 loudness normalise (both arms) → -24 LUFS
            if do_loudnorm:
                feed_audio = cond_dir / f"{i}_loud.wav"
                if not feed_audio.exists():
                    lm = loudnorm_audio(audio_src, feed_audio, ditto_bin)
                    if lm is None:
                        failed.append({"condition": cond, "sample_id": i, "error": "loudnorm failed"})
                        continue
            else:
                feed_audio = audio_src

            # Step B: resample 16k for resamp arm
            if cond.endswith("_resamp"):
                tmp_audio = cond_dir / f"{i}_loud_16k.wav"
                if not tmp_audio.exists():
                    if not resample_16k(feed_audio, tmp_audio, ditto_bin):
                        failed.append({"condition": cond, "sample_id": i, "error": "resample failed"})
                        continue
                feed_audio = tmp_audio

            out_video = cond_dir / f"{i}.mp4"
            if out_video.exists() and not args.no_cache:
                results[f"{cond}:{i}"] = "ok-cached"
                continue
            if out_video.exists():
                out_video.unlink()
            print(f"[ditto] {cond}:{i} seed={seed} ...")

            ok = run_ditto(
                repo, args.run_id, cfg, feed_audio, img_path, out_video, mode,
                retries=int(cfg.get("ditto", {}).get("retry", 1)), seed=seed,
            )
            if ok:
                results[f"{cond}:{i}"] = "ok"
                print(f"  -> {out_video}")
            elif mode == "trt_online":
                print("  TRT failed, switching to PyTorch ...")
                mode = "pytorch"
                ok = run_ditto(
                    repo, args.run_id, cfg, feed_audio, img_path, out_video, mode,
                    retries=int(cfg.get("ditto", {}).get("retry", 1)), seed=seed,
                )
                if ok:
                    results[f"{cond}:{i}"] = "ok-pytorch"
                else:
                    failed.append({"condition": cond, "sample_id": i, "error": "ditto inference failed"})
            else:
                failed.append({"condition": cond, "sample_id": i, "error": "ditto inference failed"})

    elapsed = time.monotonic() - t0
    meta = {
        "mode_used": mode,
        "seed": seed,
        "config_override": args.config or None,
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
