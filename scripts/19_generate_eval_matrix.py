#!/usr/bin/env python3
"""19_generate_eval_matrix.py — GxE (Generator x Evaluator) audio matrix.

Separates TFG generation effects from SyncNet scorer biases by evaluating
each video with both the original generation audio and its counterpart.

For each sample, the 2x2 matrix cells are:

    G (generate with)    E (evaluate with)
    natural              natural   → existing natural_raw video + existing SyncNet
    natural              tts       → natural_raw video, replace audio with TTS, re-eval
    tts                  natural   → tts_raw video, replace audio with natural, re-eval
    tts                  tts       → existing tts_raw video + existing SyncNet

Diagonal cells (G=E) reuse existing SyncNet results from 04_eval/.
Off-diagonal cells (G≠E) replace the audio track with ffmpeg then run SyncNet.

SyncNet execution uses the syncnet conda env — this needs to happen on the server.

Output: runs/<run_id>/04_eval/gxe_matrix.json
"""

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import yaml

from tfg_feature_common import STUDY_SAMPLES

RE_CONFIDENCE = re.compile(r"Confidence:\s+([\d.]+)")
RE_MIN_DIST = re.compile(r"Min dist:\s+([\d.]+)")
RE_AV_OFFSET = re.compile(r"AV offset:\s+(\d+)")


def load_config(repo: Path) -> dict:
    cfg_path = repo / "scripts" / "config.yaml"
    return yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}


def parse_syncnet_output(stdout: str) -> dict | None:
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


def find_existing_videos(ditto_dir: Path, conditions: list[str]) -> dict[str, dict[int, Path]]:
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


def find_existing_results(eval_base: Path, conditions: list[str]) -> dict[str, dict[int, dict]]:
    results: dict[str, dict[int, dict]] = {}
    for cond in conditions:
        cond_dir = eval_base / cond
        if not cond_dir.is_dir():
            continue
        results[cond] = {}
        for sid_dir in cond_dir.iterdir():
            if not sid_dir.is_dir():
                continue
            try:
                sid = int(sid_dir.name)
            except ValueError:
                continue
            sj = sid_dir / "syncnet.json"
            if sj.exists():
                try:
                    results[cond][sid] = json.loads(sj.read_text())
                except (json.JSONDecodeError, KeyError):
                    pass
    return results


def run_syncnet_pipeline(
    syncnet_dir: Path,
    syncnet_python: str,
    syncnet_bin: str,
    video_path: Path,
    data_dir: Path,
    reference: str,
    model_path: Path,
) -> tuple[str | None, str]:
    env = os.environ.copy()
    env["PATH"] = f"{syncnet_bin}:{env.get('PATH', '')}"

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


def build_ffmpeg_command(video_path: Path, audio_path: Path, output_path: Path) -> list[str]:
    return [
        "ffmpeg", "-y",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        str(output_path),
    ]


def gxe_key(generator: str, evaluator: str) -> str:
    return f"G_{generator}_E_{evaluator}"


def condition_for_axis(axis_label: str, conditions: list[str]) -> str:
    mapping = {
        "natural": "natural_raw",
        "tts": "tts_raw",
    }
    if axis_label in mapping:
        mapped = mapping[axis_label]
        if mapped in conditions:
            return mapped
    if axis_label in conditions:
        return axis_label
    return axis_label


def decompose(total: dict, gen_effect: dict, scorer_effect: dict) -> dict:
    interaction = {}
    for key in total:
        interaction[key] = round(total[key] - gen_effect[key] - scorer_effect[key], 4)
    return interaction


def classify_sample(gen_effect: dict, scorer_effect: dict) -> str:
    gen_magnitude = sum(abs(v) for v in gen_effect.values())
    score_magnitude = sum(abs(v) for v in scorer_effect.values())
    if score_magnitude == 0 and gen_magnitude == 0:
        return "mixed"
    ratio = score_magnitude / gen_magnitude if gen_magnitude > 0 else float("inf")
    if ratio > 2.0:
        return "scorer_driven"
    elif gen_magnitude / score_magnitude > 2.0 if score_magnitude > 0 else True:
        return "generator_driven"
    return "mixed"


def print_decomposition(samples_results: dict[str, dict]) -> None:
    gc_keys = ["sync_c", "sync_d"]
    nat_nat = {k: [] for k in gc_keys}
    tts_tts = {k: [] for k in gc_keys}
    tts_nat = {k: [] for k in gc_keys}
    nat_tts = {k: [] for k in gc_keys}

    for sid, cells in sorted(samples_results.items()):
        for cell_key, vals in cells.items():
            if cell_key == "G_natural_E_natural":
                for k in gc_keys:
                    nat_nat[k].append(vals.get(k, 0))
            elif cell_key == "G_tts_E_tts":
                for k in gc_keys:
                    tts_tts[k].append(vals.get(k, 0))
            elif cell_key == "G_tts_E_natural":
                for k in gc_keys:
                    tts_nat[k].append(vals.get(k, 0))
            elif cell_key == "G_natural_E_tts":
                for k in gc_keys:
                    nat_tts[k].append(vals.get(k, 0))

    print("\n=== GxE Decomposition ===")
    for k in gc_keys:
        _nat_nat = sum(nat_nat[k]) / len(nat_nat[k]) if nat_nat[k] else 0
        _tts_tts = sum(tts_tts[k]) / len(tts_tts[k]) if tts_tts[k] else 0
        _tts_nat = sum(tts_nat[k]) / len(tts_nat[k]) if tts_nat[k] else 0
        _nat_tts = sum(nat_tts[k]) / len(nat_tts[k]) if nat_tts[k] else 0

        total = round(_tts_tts - _nat_nat, 4)
        gen_effect = round(_tts_nat - _nat_nat, 4)
        scorer_effect = round(_nat_tts - _nat_nat, 4)
        interaction = round(total - gen_effect - scorer_effect, 4)

        print(f"\n  {k}:")
        print(f"    Total TTS effect (G_tts_E_tts - G_nat_E_nat): {total:+.4f}")
        print(f"    Generator effect (G_tts_E_nat - G_nat_E_nat): {gen_effect:+.4f}")
        print(f"    Scorer effect (G_nat_E_tts - G_nat_E_nat):  {scorer_effect:+.4f}")
        print(f"    Interaction (unexplained):                    {interaction:+.4f}")

    print("\nInterpretation:")
    print("  If Generator effect > 0: TTS audio genuinely improves TFG output (better features)")
    print("  If Scorer effect > 0: SyncNet itself prefers TTS audio characteristics (e.g., loudness)")

    print("\n=== Per-Sample Classification ===")
    for sid, cells in sorted(samples_results.items()):
        nat_nat_vals = cells.get("G_natural_E_natural", {})
        tts_nat_vals = cells.get("G_tts_E_natural", {})
        nat_tts_vals = cells.get("G_natural_E_tts", {})

        gen_vals = {}
        scorer_vals = {}
        for k in gc_keys:
            gen_vals[k] = tts_nat_vals.get(k, 0) - nat_nat_vals.get(k, 0)
            scorer_vals[k] = nat_tts_vals.get(k, 0) - nat_nat_vals.get(k, 0)

        classification = classify_sample(gen_vals, scorer_vals)
        print(f"  sample {sid}: {classification}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="GxE audio matrix — separate TFG generation effects from SyncNet scorer biases"
    )
    ap.add_argument("--run-id", required=True, help="Run identifier (e.g., aishell1_strict_20260707T081223Z)")
    ap.add_argument("--samples", default=",".join(str(s) for s in STUDY_SAMPLES),
                    help="Comma-separated sample IDs (default: STUDY_SAMPLES)")
    ap.add_argument("--matrix", default="natural,tts", help="Comma-separated axis labels (default: natural,tts)")
    ap.add_argument("--output-dir", default=None, help="Override output directory for results JSON")
    ap.add_argument("--dry-run", action="store_true", help="Show what would be done without executing")
    ap.add_argument("--no-cache", action="store_true", help="Re-evaluate even if syncnet.json exists")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    run_dir = repo / "runs" / args.run_id
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}")
        return

    cfg = load_config(repo)
    conditions = cfg.get("ditto", {}).get("conditions", ["natural_raw", "tts_raw"])
    syncnet_dir = repo / cfg["paths"]["syncnet_repo"]
    syn_env = Path(cfg["paths"]["envs_dir"]) / "syncnet"
    syncnet_python = str(syn_env / "bin" / "python")
    syncnet_bin = str(syn_env / "bin")
    syncnet_model = syncnet_dir / "data" / "syncnet_v2.model"

    matrix_axes = [a.strip() for a in args.matrix.split(",")]
    if len(matrix_axes) < 2:
        print("ERROR: --matrix must have at least 2 labels (e.g., natural,tts)")
        return

    sample_ids = [int(s.strip()) for s in args.samples.split(",") if s.strip()]

    ditto_dir = run_dir / "03_ditto"
    eval_base = run_dir / "04_eval"
    output_dir = Path(args.output_dir) if args.output_dir else eval_base

    axis_conditions = [condition_for_axis(a, conditions) for a in matrix_axes]
    axis_names = {ax: cond for ax, cond in zip(matrix_axes, axis_conditions)}

    videos = find_existing_videos(ditto_dir, axis_conditions)
    existing_results = find_existing_results(eval_base, axis_conditions)

    audio_base = repo / "data" / "data" / "audio"
    tts_audio_dir = run_dir / "02_tts"

    audio_sources = {}
    for ax, cond in axis_names.items():
        if ax == "natural":
            audio_sources[ax] = audio_base
        elif ax == "tts":
            audio_sources[ax] = tts_audio_dir
        else:
            audio_sources[ax] = audio_base

    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        print("ERROR: ffmpeg not found in PATH")
        return

    gxe_generated_dir = ditto_dir / "gxe_matrix"
    gxe_eval_dir = output_dir / "gxe_matrix"
    gxe_generated_dir.mkdir(parents=True, exist_ok=True)
    gxe_eval_dir.mkdir(parents=True, exist_ok=True)

    results_by_sample: dict[str, dict] = {}
    failed: list[dict] = []

    generators = [a for a in matrix_axes]
    evaluators = [a for a in matrix_axes]

    print(f"[gxe] run_id={args.run_id}")
    print(f"[gxe] matrix axes: {matrix_axes} → conditions: {axis_conditions}")
    print(f"[gxe] samples: {sample_ids}")
    print(f"[gxe] videos found: {[(c, list(v.keys())) for c, v in videos.items()]}")
    if args.dry_run:
        print("[gxe] DRY-RUN mode — no ffmpeg or SyncNet will be executed\n")

    for sid in sample_ids:
        sample_results: dict = {}

        for gen_axis in generators:
            gen_cond = axis_names[gen_axis]
            for eval_axis in evaluators:
                cell_key = gxe_key(gen_axis, eval_axis)
                is_diagonal = (gen_axis == eval_axis)
                ref_video_cond = gen_cond
                eval_cond_name = gen_cond if is_diagonal else f"G_{gen_axis}_E_{eval_axis}"

                if is_diagonal:
                    if gen_cond in existing_results and sid in existing_results[gen_cond]:
                        sample_results[cell_key] = {
                            k: existing_results[gen_cond][sid][k]
                            for k in ["sync_c", "sync_d", "av_offset"]
                            if k in existing_results[gen_cond][sid]
                        }
                        print(f"[gxe] sample {sid} {cell_key}: cached (diagonal)")
                        continue
                    else:
                        failed.append({"sample_id": sid, "cell": cell_key, "error": "no existing SyncNet result for diagonal"})
                        print(f"[gxe] sample {sid} {cell_key}: MISSING (no existing result)")
                        continue

                video_cond_videos = videos.get(ref_video_cond, {})
                if sid not in video_cond_videos:
                    failed.append({"sample_id": sid, "cell": cell_key, "error": f"video missing for {ref_video_cond}"})
                    print(f"[gxe] sample {sid} {cell_key}: SKIP (no video)")
                    continue

                video_path = video_cond_videos[sid]
                eval_audio_dir = audio_sources.get(eval_axis, audio_base)
                eval_audio_path = eval_audio_dir / f"{sid}.wav"

                if not eval_audio_path.exists():
                    failed.append({"sample_id": sid, "cell": cell_key, "error": f"eval audio missing: {eval_audio_path}"})
                    print(f"[gxe] sample {sid} {cell_key}: SKIP (no eval audio {eval_audio_path})")
                    continue

                output_dir_gxe = gxe_generated_dir / f"G_{gen_axis}_E_{eval_axis}"
                output_dir_gxe.mkdir(exist_ok=True)
                gxe_video = output_dir_gxe / f"{sid}.mp4"

                eval_out_dir = gxe_eval_dir / f"G_{gen_axis}_E_{eval_axis}" / str(sid)
                eval_out_dir.mkdir(parents=True, exist_ok=True)

                cached_json = eval_out_dir / "syncnet.json"
                if not args.no_cache and cached_json.exists():
                    try:
                        cached = json.loads(cached_json.read_text())
                        sample_results[cell_key] = {
                            k: cached[k] for k in ["sync_c", "sync_d", "av_offset"] if k in cached
                        }
                        print(f"[gxe] sample {sid} {cell_key}: cached (syncnet.json)")
                        continue
                    except (json.JSONDecodeError, KeyError):
                        pass

                ffmpeg_cmd = build_ffmpeg_command(video_path, eval_audio_path, gxe_video)

                if args.dry_run:
                    print(f"[gxe] sample {sid} {cell_key}: WOULD run {shlex.join(ffmpeg_cmd)}")
                else:
                    print(f"[gxe] sample {sid} {cell_key}: re-encoding audio...")
                    try:
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True, timeout=60)
                    except subprocess.CalledProcessError as e:
                        failed.append({"sample_id": sid, "cell": cell_key, "error": f"ffmpeg failed: {e.stderr[-200:]}"})
                        print(f"[gxe] sample {sid} {cell_key}: ffmpeg FAILED")
                        continue

                data_dir = eval_out_dir / "syncnet_data"
                data_dir.mkdir(exist_ok=True)
                reference = f"{gen_cond}_{eval_axis}_{sid}"

                if args.dry_run:
                    print(f"[gxe] sample {sid} {cell_key}: WOULD run SyncNet on {gxe_video}")
                    sample_results[cell_key] = {"sync_c": 0.0, "sync_d": 0.0, "av_offset": 0}
                    continue

                print(f"[gxe] sample {sid} {cell_key}: running SyncNet...")
                for attempt in range(2):
                    try:
                        stdout, stderr = run_syncnet_pipeline(
                            syncnet_dir, syncnet_python, syncnet_bin,
                            gxe_video, data_dir, reference, syncnet_model,
                        )
                        if stdout is None:
                            if attempt == 1:
                                failed.append({"sample_id": sid, "cell": cell_key, "error": stderr})
                            else:
                                time.sleep(1)
                            continue
                        parsed = parse_syncnet_output(stdout)
                        if parsed:
                            parsed["sample_id"] = sid
                            parsed["cell"] = cell_key
                            cached_json.write_text(json.dumps(parsed, indent=2))
                            sample_results[cell_key] = {
                                k: parsed[k] for k in ["sync_c", "sync_d", "av_offset"] if k in parsed
                            }
                            c, d = parsed["sync_c"], parsed["sync_d"]
                            print(f"  -> Sync-C={c:.3f} Sync-D={d:.3f}")
                            break
                        else:
                            failed.append({"sample_id": sid, "cell": cell_key, "error": "parse failed", "stdout": stdout[:500]})
                            break
                    except Exception as e:
                        if attempt == 1:
                            failed.append({"sample_id": sid, "cell": cell_key, "error": str(e)})
                        time.sleep(1)

        if sample_results:
            results_by_sample[str(sid)] = sample_results

    output_json = {
        "run_id": args.run_id,
        "samples": sample_ids,
        "matrix_axes": matrix_axes,
        "results": results_by_sample,
        "failed": failed,
    }

    out_path = output_dir / "gxe_matrix.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
        out_path.write_text(json.dumps(output_json, indent=2, ensure_ascii=False))
    print(f"\n[gxe] results: {len(results_by_sample)} samples with results, {len(failed)} failures")

    if results_by_sample:
        print_decomposition(results_by_sample)


if __name__ == "__main__":
    main()
