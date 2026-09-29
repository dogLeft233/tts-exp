#!/usr/bin/env python3
"""04_eval.py - SyncNet evaluation (issue #5).

For each successfully-generated ditto video, run the official SyncNet
run_pipeline.py and run_syncnet.py stages in order, then parse their scores.
Different videos can be evaluated concurrently without sharing work directories.

Uses the syncnet conda env for execution (avoids dependency conflicts with ditto env).

Output: runs/<run_id>/04_eval/{condition}/{i}/syncnet.json + eval_meta.json
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import json
import os
import re
import subprocess
import time
from pathlib import Path

from utils import detect_sample_ids, load_config

# Regex patterns for SyncNet stdout
RE_CONFIDENCE = re.compile(r"Confidence:\s+([\d.]+)")
RE_MIN_DIST = re.compile(r"Min dist:\s+([\d.]+)")
RE_AV_OFFSET = re.compile(r"AV offset:\s+(\d+)")

# ---------------------------------------------------------------------------


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
    min_track: int = 100,
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
        "--overwrite",
        "--min_track", str(min_track),
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


def evaluate_video(
    cond: str,
    sid: int,
    vpath: Path,
    *,
    out_base: Path,
    syncnet_dir: Path,
    syncnet_python: str,
    syncnet_bin: str,
    syncnet_model: Path,
    min_track: int,
    no_cache: bool,
) -> tuple[dict | None, dict | None, bool]:
    """Evaluate one isolated video; return (score, failure, cache_hit)."""
    sample_dir = out_base / cond / str(sid)
    sample_dir.mkdir(parents=True, exist_ok=True)
    sync_path = sample_dir / "syncnet.json"
    if sync_path.exists() and not no_cache:
        try:
            cached = json.loads(sync_path.read_text())
            if cached.get("sync_c") is not None and cached.get("sync_d") is not None:
                return cached, None, True
        except (json.JSONDecodeError, OSError):
            pass

    data_dir = sample_dir / "syncnet_data"
    data_dir.mkdir(exist_ok=True)
    reference = f"{cond}_{sid}"
    last_error = ""
    for attempt in range(2):
        try:
            stdout, stderr = run_syncnet_pipeline(
                syncnet_dir, syncnet_python, syncnet_bin,
                vpath, data_dir, reference, syncnet_model,
                min_track=min_track,
            )
            if stdout is None:
                last_error = stderr
            else:
                parsed = parse_syncnet_output(stdout)
                if parsed is None:
                    return None, {
                        "condition": cond, "sample_id": sid,
                        "error": "parse failed", "stdout": stdout[:500],
                    }, False
                sample_out = {"sample_id": sid, "condition": cond, **parsed}
                sync_path.write_text(json.dumps(sample_out, indent=2))
                return sample_out, None, False
        except Exception as exc:
            last_error = str(exc)
        if attempt == 0:
            time.sleep(1)
    return None, {"condition": cond, "sample_id": sid, "error": last_error}, False


def main() -> None:
    ap = argparse.ArgumentParser(description="SyncNet evaluation")
    ap.add_argument("--run_id", "--run-id", dest="run_id", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--config", default="")
    ap.add_argument("--no-cache", action="store_true")
    ap.add_argument("--min-track", type=int, default=100)
    ap.add_argument("--workers", type=int, default=None, help="concurrent videos (default: eval.workers, or 2)")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    run_dir = repo / "runs" / args.run_id
    ditto_dir = run_dir / "03_ditto"
    out_base = run_dir / "04_eval"
    out_base.mkdir(parents=True, exist_ok=True)

    cfg = load_config(repo, args.config or None)
    workers = args.workers if args.workers is not None else int(cfg.get("eval", {}).get("workers", 2))
    if workers < 1:
        ap.error("--workers must be at least 1")
    conditions = cfg.get("ditto", {}).get("conditions", ["natural_raw", "natural_resamp", "tts_raw", "tts_resamp"])
    expected_ids = detect_sample_ids(repo, args.smoke, cfg=cfg)
    syncnet_dir = repo / cfg["paths"]["syncnet_repo"]
    syn_env = Path(cfg["paths"]["envs_dir"]) / "syncnet"
    syncnet_python = str(syn_env / "bin" / "python")
    syncnet_bin = str(syn_env / "bin")
    syncnet_model = syncnet_dir / "data" / "syncnet_v2.model"

    videos = find_videos(ditto_dir, conditions)
    print(f"[eval] found videos in {len(videos)} conditions: {list(videos.keys())}")

    results: dict[str, dict] = {}
    failed: list[dict] = []

    jobs = []
    for cond, sample_videos in videos.items():
        cond_out = out_base / cond
        cond_out.mkdir(exist_ok=True)
        for sid, vpath in sorted(sample_videos.items()):
            jobs.append((cond, sid, vpath))

    worker = partial(
        evaluate_video,
        out_base=out_base,
        syncnet_dir=syncnet_dir,
        syncnet_python=syncnet_python,
        syncnet_bin=syncnet_bin,
        syncnet_model=syncnet_model,
        min_track=args.min_track,
        no_cache=args.no_cache,
    )
    print(f"[eval] scoring {len(jobs)} videos with {workers} worker(s)")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for (cond, sid, _), (score, error, cached) in zip(
            jobs, pool.map(lambda job: worker(*job), jobs)
        ):
            if score is not None:
                results[f"{cond}:{sid}"] = score
                if cached:
                    print(f"[eval] {cond}:{sid} cached")
                else:
                    print(f"[eval] {cond}:{sid} -> Sync-C={score['sync_c']:.3f} Sync-D={score['sync_d']:.3f}")
            elif error is not None:
                failed.append(error)
                print(f"[eval] {cond}:{sid} -> FAILED: {error['error']}")

    successful_by_condition = {
        condition: sorted(
            sid for sid in expected_ids if f"{condition}:{sid}" in results
        )
        for condition in conditions
    }
    complete_ids = sorted(
        set(expected_ids).intersection(
            *(set(ids) for ids in successful_by_condition.values())
        )
    ) if conditions else []
    require_complete = bool(
        cfg.get("eval", {}).get("require_complete_pairs", False)
    )
    incomplete_ids = sorted(set(expected_ids) - set(complete_ids))
    meta = {
        "config_override": args.config or None,
        "workers": workers,
        "conditions": conditions,
        "expected_sample_ids": expected_ids,
        "successful_by_condition": successful_by_condition,
        "complete_case_ids": complete_ids,
        "incomplete_ids": incomplete_ids,
        "complete": not incomplete_ids,
        "samples_total": sum(len(v) for v in videos.values()),
        "samples_ok": len(results),
        "samples_failed": len(failed),
        "results": results,
        "failed": failed,
    }
    (out_base / "eval_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(
        f"[eval] {len(results)} ok, {len(failed)} failed; "
        f"complete cases={len(complete_ids)}/{len(expected_ids)}"
    )
    if require_complete and incomplete_ids:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
