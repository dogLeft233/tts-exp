#!/usr/bin/env python3
"""04_eval.py - SyncNet evaluation (issue #5).

For each successfully-generated ditto video, run syncnet_python/demo_syncnet.py
and parse stdout for Confidence (→ Sync-C), Min dist (→ Sync-D), AV offset.

Uses the syncnet conda env for execution (avoids dependency conflicts with ditto env).

Output: runs/<run_id>/04_eval/{condition}/{i}/syncnet.json + eval_meta.json
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

import yaml

# Regex patterns for SyncNet stdout
RE_CONFIDENCE = re.compile(r"Confidence:\s+([\d.]+)")
RE_MIN_DIST = re.compile(r"Min dist:\s+([\d.]+)")
RE_AV_OFFSET = re.compile(r"AV offset:\s+(\d+)")

# ---------------------------------------------------------------------------


def load_config(repo: Path) -> dict:
    cfg_path = repo / "scripts" / "config.yaml"
    return yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}


def parse_syncnet_output(stdout: str) -> dict | None:
    """Parse Confidence, Min dist, AV offset from SyncNet stdout."""
    m_c = RE_CONFIDENCE.search(stdout)
    m_d = RE_MIN_DIST.search(stdout)
    m_o = RE_AV_OFFSET.search(stdout)
    if m_c and m_d:
        return {
            "sync_c": float(m_c.group(1)),
            "sync_d": float(m_d.group(1)),
            "av_offset": int(m_o.group(1)) if m_o else None,
        }
    return None


def find_videos(ditto_dir: Path, conditions: list[str]) -> dict[str, dict[int, Path]]:
    """Scan 03_ditto for existing .mp4 files."""
    videos: dict[str, dict[int, Path]] = {}
    for cond in conditions:
        cond_dir = ditto_dir / cond
        if not cond_dir.is_dir():
            continue
        videos[cond] = {}
        for mp4 in sorted(cond_dir.glob("*.mp4")):
            try:
                sid = int(mp4.stem)
                videos[cond][sid] = mp4
            except ValueError:
                pass
    return videos


def run_syncnet_pipeline(
    syncnet_dir: Path,
    syncnet_python: str,
    syncnet_bin: str,
    video_path: Path,
    data_dir: Path,
    reference: str,
    model_path: Path,
) -> tuple[str | None, str]:
    """Run full SyncNet pipeline: detect+crop faces, then evaluate.

    Step 1: run_pipeline.py — detects and crops face tracks
    Step 2: run_syncnet.py — computes sync scores on cropped video

    Returns (stdout from run_syncnet, stderr), or (None, error_msg) on failure.
    """
    env = os.environ.copy()
    env["PATH"] = f"{syncnet_bin}:{env.get('PATH', '')}"

    # Step 1: face detection + cropping
    cmd1 = [
        syncnet_python,
        str(syncnet_dir / "run_pipeline.py"),
        "--videofile", str(video_path),
        "--reference", reference,
        "--data_dir", str(data_dir),
    ]
    try:
        subprocess.run(cmd1, check=True, capture_output=True, text=True, timeout=180, env=env, cwd=str(syncnet_dir))
    except subprocess.CalledProcessError as e:
        return None, f"run_pipeline failed: {e.stderr[-300:]}"

    # Step 2: sync evaluation
    cmd2 = [
        syncnet_python,
        str(syncnet_dir / "run_syncnet.py"),
        "--videofile", str(video_path),
        "--reference", reference,
        "--data_dir", str(data_dir),
        "--initial_model", str(model_path),
    ]
    try:
        result = subprocess.run(cmd2, check=True, capture_output=True, text=True, timeout=120, env=env, cwd=str(syncnet_dir))
        return result.stdout + result.stderr, ""
    except subprocess.CalledProcessError as e:
        return None, f"run_syncnet failed: {e.stderr[-300:]}"


def main() -> None:
    ap = argparse.ArgumentParser(description="SyncNet evaluation")
    ap.add_argument("--run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    run_dir = repo / "runs" / args.run_id
    ditto_dir = run_dir / "03_ditto"
    out_base = run_dir / "04_eval"
    out_base.mkdir(parents=True, exist_ok=True)

    cfg = load_config(repo)
    conditions = cfg.get("ditto", {}).get("conditions", ["natural_raw", "natural_resamp", "tts_raw", "tts_resamp"])
    syncnet_dir = repo / cfg["paths"]["syncnet_repo"]
    syn_env = Path(cfg["paths"]["envs_dir"]) / "syncnet"
    syncnet_python = str(syn_env / "bin" / "python")
    syncnet_bin = str(syn_env / "bin")
    syncnet_model = syncnet_dir / "data" / "syncnet_v2.model"

    videos = find_videos(ditto_dir, conditions)
    print(f"[eval] found videos in {len(videos)} conditions: {list(videos.keys())}")

    results: dict[str, dict] = {}
    failed: list[dict] = []

    for cond, sample_videos in videos.items():
        cond_out = out_base / cond
        cond_out.mkdir(exist_ok=True)
        for sid, vpath in sorted(sample_videos.items()):
            sample_dir = cond_out / str(sid)
            sample_dir.mkdir(exist_ok=True)
            data_dir = sample_dir / "syncnet_data"
            data_dir.mkdir(exist_ok=True)
            reference = f"{cond}_{sid}"

            print(f"[eval] {cond}:{sid} ...")
            for attempt in range(2):
                try:
                    stdout, stderr = run_syncnet_pipeline(
                        syncnet_dir, syncnet_python, syncnet_bin,
                        vpath, data_dir, reference, syncnet_model,
                    )
                    if stdout is None:
                        if attempt == 1:
                            failed.append({"condition": cond, "sample_id": sid, "error": stderr})
                        else:
                            time.sleep(1)
                        continue
                    parsed = parse_syncnet_output(stdout)
                    if parsed:
                        parsed["sample_id"] = sid
                        parsed["condition"] = cond
                        sample_out = {
                            "sample_id": sid,
                            "condition": cond,
                            **parsed,
                        }
                        (sample_dir / "syncnet.json").write_text(
                            json.dumps(sample_out, indent=2)
                        )
                        results[f"{cond}:{sid}"] = sample_out
                        c, d = parsed["sync_c"], parsed["sync_d"]
                        print(f"  -> Sync-C={c:.3f} Sync-D={d:.3f}")
                        break
                    else:
                        failed.append({"condition": cond, "sample_id": sid, "error": "parse failed", "stdout": stdout[:500]})
                        # Don't retry on parse failure
                        break
                except Exception as e:
                    if attempt == 1:
                        failed.append({"condition": cond, "sample_id": sid, "error": str(e)})
                    time.sleep(1)
            else:
                print(f"  -> FAILED")

    meta = {
        "samples_total": sum(len(v) for v in videos.values()),
        "samples_ok": len(results),
        "samples_failed": len(failed),
        "results": results,
        "failed": failed,
    }
    (out_base / "eval_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"[eval] {len(results)} ok, {len(failed)} failed")


if __name__ == "__main__":
    main()
