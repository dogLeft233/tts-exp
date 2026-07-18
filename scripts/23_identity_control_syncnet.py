#!/usr/bin/env python3
"""23_identity_control_syncnet.py — Identity control for the dose-response pipeline.

This script exercises the exact same audio → Ditto → SyncNet pipeline used by
script 22 (`22_dose_response_syncnet.py`), but with an **identity transform**:
the TTS source audio is written back to disk as 16-bit PCM (the format Ditto
ingests) without any acoustic modification, then re-run through Ditto and
SyncNet. ΔSync-C is computed against the original `tts_raw` diagonal baseline.

Interpretation
--------------

- |ΔSync-C| ≲ 0.2  → the write/Ditto/SyncNet pipeline itself is clean.
  The −1.0 to −1.2 Sync-C drops observed in the dose-response can be
  attributed to feature-specific perturbations, not pipeline artifacts.
- ΔSync-C ≈ −1.0  → the pipeline alone accounts for most of the
  "common-mode" TTS-source drop. The dose-response findings would then need
  to be reinterpreted as Ditto + SyncNet being sensitive to *any* re-write
  of TTS audio, regardless of feature content.

Usage:
    python scripts/23_identity_control_syncnet.py --run-id aishell1_strict_20260707T081223Z --dry-run
    python scripts/23_identity_control_syncnet.py --run-id aishell1_strict_20260707T081223Z --samples 3
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO / "scripts"))

from tfg_feature_common import STUDY_SAMPLES, TARGET_SR, load_audio_mono, ensure_output_dirs


def _load_script_module(filename: str):
    """Load a sibling script as a module by file path (mirrors script 22)."""
    path = _REPO / "scripts" / filename
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_s22 = _load_script_module("22_dose_response_syncnet.py")

Intervention = _s22.Intervention
run_intervention_pipeline = _s22.run_intervention_pipeline
load_baseline_results = _s22.load_baseline_results
aggregate_deltas = _s22.aggregate_deltas


# ---------------------------------------------------------------------------
# Identity intervention
# ---------------------------------------------------------------------------


def _identity_tts(y_tts, y_nat, sr, sid):
    """No-op transform: return TTS audio unchanged.

    The point is to exercise the downstream write/Ditto/SyncNet pipeline
    with bit-identical audio — any ΔSync-C must therefore come from the
    pipeline itself, not from a feature modification.
    """
    return y_tts


def _build_identity_interventions() -> list[Intervention]:
    return [
        Intervention(
            name="identity_tts",
            source="tts",
            baseline_cond="tts_raw",
            transform_description="No-op: TTS audio re-written as PCM_16 unchanged",
            expected_sync_direction="no change",
            transform=_identity_tts,
        ),
    ]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Identity control for the dose-response pipeline (script 22)"
    )
    ap.add_argument("--run-id", required=True)
    ap.add_argument(
        "--samples",
        default=",".join(str(s) for s in STUDY_SAMPLES),
        help="Comma-separated sample IDs (default: STUDY_SAMPLES)",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--no-cache-audio", action="store_true")
    ap.add_argument("--no-cache-video", action="store_true")
    ap.add_argument("--no-cache-syncnet", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parent.parent
    run_dir = repo / "runs" / args.run_id
    if not run_dir.is_dir():
        print(f"ERROR: run directory not found: {run_dir}")
        return

    cfg_path = repo / "scripts" / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}

    sample_ids = [int(s.strip()) for s in args.samples.split(",") if s.strip()]
    interventions = _build_identity_interventions()

    eval_base = run_dir / "04_eval"
    baseline_conds = sorted({iv.baseline_cond for iv in interventions})
    baselines = load_baseline_results(eval_base, baseline_conds)

    nat_audio_dir = repo / "data" / "data" / "audio"
    tts_audio_dir = run_dir / "02_tts"
    if not tts_audio_dir.is_dir():
        print(f"ERROR: TTS audio dir not found: {tts_audio_dir}")
        return
    img_dir = repo / "data" / "data" / "image"

    natural_audio: dict[int, np.ndarray] = {}
    tts_audio: dict[int, np.ndarray] = {}
    loaded_ids: list[int] = []
    for sid in sample_ids:
        nat_p = nat_audio_dir / f"{sid}.wav"
        tts_p = tts_audio_dir / f"{sid}.wav"
        if nat_p.exists() and tts_p.exists():
            natural_audio[sid], _ = load_audio_mono(nat_p, target_sr=TARGET_SR)
            tts_audio[sid], _ = load_audio_mono(tts_p, target_sr=TARGET_SR)
            loaded_ids.append(sid)
        else:
            print(f"WARN: sample {sid} missing audio (nat={nat_p.exists()}, tts={tts_p.exists()})")

    sr = TARGET_SR

    ditto_intv_base = run_dir / "03_ditto" / "interventions"
    eval_intv_base = run_dir / "04_eval" / "interventions"
    out_dir = args.output_dir if args.output_dir else ensure_output_dirs()["metrics"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[identity] run_id={args.run_id}")
    print(f"[identity] samples loaded: {loaded_ids}")
    print(f"[identity] interventions: {[iv.name for iv in interventions]}")
    print(f"[identity] baselines: {[(c, len(v)) for c, v in baselines.items()]}")
    if args.dry_run:
        print("[identity] DRY-RUN — no Ditto/SyncNet will be executed\n")

    results: dict[str, dict[int, dict]] = {iv.name: {} for iv in interventions}
    failures: list[dict] = []
    t0 = time.monotonic()

    for iv in interventions:
        print(f"\n=== {iv.name} ({iv.transform_description}) ===")
        for sid in loaded_ids:
            img_path = img_dir / f"{sid}.png"
            if not img_path.exists():
                failures.append({"intervention": iv.name, "sample_id": sid, "error": "image missing"})
                continue

            audio_out = ditto_intv_base / iv.name
            video_out = ditto_intv_base / iv.name
            eval_out = eval_intv_base / iv.name

            res, status = run_intervention_pipeline(
                intervention=iv,
                sid=sid,
                y_tts=tts_audio[sid],
                y_nat=natural_audio[sid],
                sr=sr,
                img_path=img_path,
                audio_out_dir=audio_out,
                video_out_dir=video_out,
                eval_out_dir=eval_out,
                repo=repo,
                cfg=cfg,
                run_id=args.run_id,
                skip_audio=not args.no_cache_audio,
                skip_video=not args.no_cache_video,
                skip_syncnet=not args.no_cache_syncnet,
                dry_run=args.dry_run,
            )

            if res is None:
                failures.append({"intervention": iv.name, "sample_id": sid, "error": status})
                print(f"  sample {sid}: FAIL — {status}")
            else:
                results[iv.name][sid] = res
                c = res.get("sync_c", 0.0)
                base = baselines.get(iv.baseline_cond, {}).get(sid, {})
                bc = base.get("sync_c")
                if bc is not None:
                    print(f"  sample {sid}: Sync-C={c:.3f} (base={bc:.3f}, Δ={c - bc:+.3f}) — {status}")
                else:
                    print(f"  sample {sid}: Sync-C={c:.3f} (no baseline) — {status}")

    elapsed = time.monotonic() - t0
    print(f"\n[identity] completed in {elapsed:.0f}s; {sum(len(v) for v in results.values())} ok, {len(failures)} failures")

    if args.dry_run:
        print("[identity] DRY-RUN — no JSON written")
        return

    summary = aggregate_deltas(results, baselines, interventions)

    # Interpretation
    iv_name = interventions[0].name
    s = summary.get(iv_name, {})
    if s.get("n_samples"):
        mean_dc = s["mean_delta_c"]
        if mean_dc is not None:
            if abs(mean_dc) < 0.2:
                verdict = "PIPELINE_CLEAN"
                verdict_msg = (
                    f"Identity ΔSync-C = {mean_dc:+.3f} (|Δ| < 0.2). The write/Ditto/SyncNet "
                    "pipeline itself does not account for the dose-response drops. "
                    "The −1.0 to −1.2 Sync-C drops in script 22 can be attributed to "
                    "feature-specific perturbations."
                )
            elif mean_dc <= -0.5:
                verdict = "PIPELINE_CONFOUNDS_DOSE_RESPONSE"
                verdict_msg = (
                    f"Identity ΔSync-C = {mean_dc:+.3f} (≤ −0.5). The pipeline alone accounts "
                    "for a substantial fraction of the dose-response drop. The TTS-source "
                    "intervention findings in script 22 should be reinterpreted: the common-mode "
                    "degradation is partly a write/Ditto/SyncNet re-run artifact, not purely a "
                    "feature-specific effect."
                )
            else:
                verdict = "AMBIGUOUS"
                verdict_msg = (
                    f"Identity ΔSync-C = {mean_dc:+.3f} (0.2 ≤ |Δ| < 0.5). Pipeline contributes "
                    "a small but non-negligible offset; interpret dose-response magnitudes with "
                    "this baseline in mind."
                )
        else:
            verdict = "NO_DATA"
            verdict_msg = "No samples produced results."
    else:
        verdict = "NO_DATA"
        verdict_msg = "No samples produced results."

    output = {
        "run_id": args.run_id,
        "samples": loaded_ids,
        "baselines": {
            cond: {str(sid): vals for sid, vals in baselines.get(cond, {}).items()}
            for cond in baseline_conds
        },
        "interventions": summary,
        "failed": failures,
        "elapsed_s": round(elapsed, 1),
        "verdict": verdict,
        "verdict_message": verdict_msg,
    }

    out_path = out_dir / "identity_control.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False, default=str))
    print(f"[identity] results -> {out_path}")

    print("\n=== Identity Control Summary ===")
    if s.get("n_samples"):
        print(
            f"{iv_name:<24} n={s['n_samples']:>2}  "
            f"ΔSync-C={s['mean_delta_c']:+.4f}  ΔSync-D={s['mean_delta_d']:+.4f}"
        )
    print(f"\nVERDICT: {verdict}")
    print(verdict_msg)


if __name__ == "__main__":
    main()