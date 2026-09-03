"""Run the minimal confirmation matrix through frozen Wav2Lip and SyncNet."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.masked_tts_tfg_probe.cohort import freeze_confirmation_cohort
from scripts.experiments.masked_tts_tfg_probe.mel_drivers import SEEDS, build_drivers
from scripts.experiments.masked_tts_tfg_probe.phase_a import file_sha256, read_json, write_json
from scripts.experiments.masked_tts_tfg_probe.run import render_all, run_parity, score_all

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PARENT = REPO / "runs/lrs3_masked_tts_retention_exploratory_20260901"
DEFAULT_FIRST_PROBE = REPO / "runs/lrs3_masked_tts_tfg_probe_20260902/01_cohort/manifest.json"
DEFAULT_RUN = REPO / "runs/lrs3_masked_tts_tfg_confirmation_20260902"
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260902
EXPECTED_RECORDS = 16
EXPECTED_GROUPS = 8
DRIVERS_PER_RECORD = 10


def _bootstrap(values: np.ndarray) -> dict[str, Any]:
    if values.shape != (EXPECTED_GROUPS,) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires eight finite group values")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    medians = np.median(values[indices], axis=1)
    return {
        "median": float(np.median(values)),
        "ci95": [float(np.quantile(medians, 0.025, method="linear")), float(np.quantile(medians, 0.975, method="linear"))],
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "unit": "whole_source_group",
        "method": "numpy_quantile_linear",
    }


def _incomplete(run_dir: Path, reason: str) -> tuple[dict[str, Any], dict[str, Any]]:
    analysis = {"schema_version": 1, "status": "incomplete", "reason": reason, "sealed_splits_accessed": False}
    decision = {
        "schema_version": 1,
        "engineering": "NO_GO",
        "science": "NOT_EVALUATED",
        "confirmatory": True,
        "claim": "confirmation matrix was not complete",
        "claim_boundary": ["no scientific conclusion from an incomplete matrix"],
        "sealed_splits_accessed": False,
    }
    output = run_dir / "05_analysis"
    write_json(output / "analysis.json", analysis)
    write_json(output / "decision.json", decision)
    write_json(run_dir / "decision.json", decision)
    return analysis, decision


def analyze_scores(run_dir: Path, cohort_path: Path, score_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cohort = read_json(cohort_path)
    scores = read_json(score_path)
    if scores.get("status") != "complete" or int(scores.get("score_count", -1)) != EXPECTED_RECORDS * DRIVERS_PER_RECORD:
        return _incomplete(run_dir, "confirmation score matrix is incomplete")
    records = cohort.get("records", [])
    groups = [str(group) for group in cohort.get("groups", [])]
    if len(records) != EXPECTED_RECORDS or len(groups) != EXPECTED_GROUPS:
        return _incomplete(run_dir, "confirmation cohort dimensions are invalid")
    by_key: dict[tuple[str, int, str], Mapping[str, Any]] = {}
    for row in scores.get("scores", []):
        condition = str(row.get("condition"))
        if condition == "NATURAL_MEL":
            continue
        key = (str(row["sample_id"]), int(row["seed"]), condition)
        if key in by_key:
            raise ValueError(f"duplicate confirmation score cell: {key}")
        by_key[key] = row
    record_rows: list[dict[str, Any]] = []
    seed_rows: list[dict[str, Any]] = []
    for record in records:
        sample_id = str(record["sample_id"])
        group = str(record["source_group"])
        for seed in SEEDS:
            try:
                paired = by_key[(sample_id, seed, "PAIRED_TTS")]
                centroid = by_key[(sample_id, seed, "PHONE_CENTROID")]
                nat = by_key[(sample_id, seed, "NAT_ONLY")]
            except KeyError as error:
                raise ValueError(f"missing confirmation score cell: {error}")
            seed_rows.append({
                "sample_id": sample_id,
                "source_group": group,
                "seed": int(seed),
                "token_C_gain": float(paired["sync_c"]) - float(centroid["sync_c"]),
                "token_D_gain": float(centroid["sync_d"]) - float(paired["sync_d"]),
                "modality_C_gain": float(paired["sync_c"]) - float(nat["sync_c"]),
                "modality_D_gain": float(nat["sync_d"]) - float(paired["sync_d"]),
            })
        local_seed_rows = [row for row in seed_rows if row["sample_id"] == sample_id]
        record_rows.append({
            "sample_id": sample_id,
            "source_group": group,
            "seed_count": len(local_seed_rows),
            **{name: float(np.median([row[name] for row in local_seed_rows])) for name in ("token_C_gain", "token_D_gain", "modality_C_gain", "modality_D_gain")},
        })
    if len(record_rows) != EXPECTED_RECORDS or any(row["seed_count"] != len(SEEDS) for row in record_rows):
        raise ValueError("confirmation record aggregation is incomplete")
    group_rows: list[dict[str, Any]] = []
    for group in groups:
        local = [row for row in record_rows if row["source_group"] == group]
        if len(local) != 2:
            raise ValueError(f"confirmation group does not contain two records: {group}")
        group_rows.append({
            "source_group": group,
            "record_count": len(local),
            **{name: float(np.median([row[name] for row in local])) for name in ("token_C_gain", "token_D_gain", "modality_C_gain", "modality_D_gain")},
        })
    metric_names = ("token_C_gain", "token_D_gain", "modality_C_gain", "modality_D_gain")
    bootstrap = {name: _bootstrap(np.asarray([row[name] for row in group_rows], dtype=np.float64)) for name in metric_names}
    wins = {name: int(sum(float(row[name]) > 0 for row in group_rows)) for name in metric_names}
    modality_pass = all(bootstrap[name]["ci95"][0] > 0 and wins[name] >= 7 for name in ("modality_C_gain", "modality_D_gain"))
    token_pass = modality_pass and all(bootstrap[name]["ci95"][0] > 0 and wins[name] >= 7 for name in ("token_C_gain", "token_D_gain"))
    if token_pass:
        status = "CONFIRMED_TOKEN_SIGNAL"
    elif modality_pass:
        status = "CONFIRMED_MODALITY_ONLY"
    else:
        status = "NO_CONFIRMATORY_TFG_GAIN"
    claim = {
        "CONFIRMED_TOKEN_SIGNAL": "paired TTS-conditioned mel has a confirmatory direct-mel frozen-Wav2Lip signal beyond phone-centroid conditioning on the 16 held-back parent records",
        "CONFIRMED_MODALITY_ONLY": "TTS-side conditioning has a confirmatory direct-mel frozen-Wav2Lip value relative to NAT_ONLY on the 16 held-back parent records, without a token-specific signal",
        "NO_CONFIRMATORY_TFG_GAIN": "the exploratory modality gain did not reproduce on the 16 held-back parent records at this frozen Wav2Lip/SyncNet endpoint",
    }[status]
    claim_boundary = [
        "direct-mel frozen-Wav2Lip validation only",
        "primary audio is untouched natural audio after strict replacement",
        "not audible TTS-feature retention",
        "not waveform reachability",
        "not audio quality",
        "not natural prosody preservation",
        "not population generalization",
        "not a deployable replacement system",
    ]
    analysis = {
        "schema_version": 1,
        "status": "complete",
        "confirmatory": True,
        "record_count": EXPECTED_RECORDS,
        "group_count": EXPECTED_GROUPS,
        "score_count": int(scores["score_count"]),
        "cohort_manifest_sha256": file_sha256(cohort_path),
        "score_manifest_sha256": file_sha256(score_path),
        "record_rows": record_rows,
        "seed_rows": seed_rows,
        "group_rows": group_rows,
        "bootstrap": bootstrap,
        "wins": wins,
        "rules": {
            "modality": "both lower bounds > 0 and at least 7/8 positive source groups on Sync-C and Sync-D",
            "token": "both lower bounds > 0 and at least 7/8 positive source groups on Sync-C and Sync-D, with modality rule passing",
        },
        "claim_boundary": claim_boundary,
        "sealed_splits_accessed": False,
    }
    decision = {
        "schema_version": 1,
        "engineering": "GO",
        "science": status,
        "confirmatory": True,
        "claim": claim,
        "claim_boundary": claim_boundary,
        "next_stage_gate": {
            "waveform_decoder_feasibility": status == "CONFIRMED_TOKEN_SIGNAL",
            "reason": "token-specific rule passed" if status == "CONFIRMED_TOKEN_SIGNAL" else "do not advance to waveform decoding as a fine-grained TTS-retention experiment",
        },
        "sealed_splits_accessed": False,
    }
    output = run_dir / "05_analysis"
    write_json(output / "analysis.json", analysis)
    write_json(output / "decision.json", decision)
    lines = [
        "# Direct-mel frozen-Wav2Lip confirmation",
        "",
        f"Status: **{status}**.",
        "",
        "The confirmation matrix uses all 16 parent evaluation records not used by the first eight-record probe. Each rendered video was muxed with its record's untouched natural audio before official SyncNet V2 scoring.",
        "",
        "## Group-level gains",
        "",
        "| Group | Token C | Token D | Modality C | Modality D |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in group_rows:
        lines.append(f"| `{row['source_group']}` | {row['token_C_gain']:.4f} | {row['token_D_gain']:.4f} | {row['modality_C_gain']:.4f} | {row['modality_D_gain']:.4f} |")
    lines.extend(["", "## Record-level gains", "", "| Record | Group | Token C | Token D | Modality C | Modality D |", "|---|---|---:|---:|---:|---:|"])
    for row in record_rows:
        lines.append(f"| `{row['sample_id']}` | `{row['source_group']}` | {row['token_C_gain']:.4f} | {row['token_D_gain']:.4f} | {row['modality_C_gain']:.4f} | {row['modality_D_gain']:.4f} |")
    lines.extend(["", "## Bootstrap and wins", ""])
    for name, row in bootstrap.items():
        lines.append(f"- `{name}` median `{row['median']:.5f}`, 95% CI `[{row['ci95'][0]:.5f}, {row['ci95'][1]:.5f}]`; positive groups `{wins[name]}/8`.")
    lines.extend(["", "## Claim boundary", "", claim, "", "This is not evidence of audible TTS-feature retention, waveform reachability, audio quality, natural-prosody preservation, population generalization, or a deployable replacement system.", ""])
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    write_json(run_dir / "decision.json", decision)
    write_json(run_dir / "summary.json", {
        "schema_version": 1,
        "run": "lrs3_masked_tts_tfg_confirmation_20260902",
        "status": status,
        "cohort_records": EXPECTED_RECORDS,
        "cohort_groups": EXPECTED_GROUPS,
        "driver_count": EXPECTED_RECORDS * DRIVERS_PER_RECORD,
        "render_count": EXPECTED_RECORDS * DRIVERS_PER_RECORD,
        "score_count": EXPECTED_RECORDS * DRIVERS_PER_RECORD,
        "decision": decision,
        "sealed_splits_accessed": False,
    })
    return analysis, decision


def execute(args: argparse.Namespace) -> int:
    parent = args.parent.resolve()
    run_dir = args.run.resolve()
    cohort_path = run_dir / "00_cohort/manifest.json"
    drivers_dir = run_dir / "01_mels"
    drivers_path = drivers_dir / "drivers.json"
    parity_path = run_dir / "02_renders/parity.json"
    render_path = run_dir / "02_renders/render_manifest.json"
    score_path = run_dir / "05_syncnet/summary.json"
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume and args.stage == "all":
        raise ValueError(f"refusing non-empty output directory without --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "score":
        if not cohort_path.is_file() or not render_path.is_file():
            raise ValueError("confirmation cohort and render manifest are required before scoring")
        score_all(run_dir, cohort_path, render_path)
        return 0
    if args.stage == "analysis":
        if not score_path.is_file():
            raise ValueError("confirmation SyncNet summary is required before analysis")
        _, decision = analyze_scores(run_dir, cohort_path, score_path)
        print(json.dumps({"status": decision["science"], "output": str((run_dir / "05_analysis").resolve())}, ensure_ascii=False))
        return 0
    if args.stage in {"cohort", "mels", "parity", "render", "all"}:
        freeze_confirmation_cohort(parent, args.first_probe, cohort_path)
    if args.stage in {"mels", "parity", "render", "score", "analysis", "all"}:
        build_drivers(parent, cohort_path, drivers_dir)
    if args.stage in {"parity", "render", "score", "analysis", "all"}:
        run_parity(run_dir, cohort_path, drivers_path, run_dir / "02_renders")
    if args.stage in {"render", "score", "analysis", "all"}:
        render_all(run_dir, cohort_path, drivers_path, parity_path, run_dir / "02_renders")
    if args.stage in {"score", "analysis", "all"}:
        score_all(run_dir, cohort_path, render_path)
    if args.stage in {"analysis", "all"}:
        _, decision = analyze_scores(run_dir, cohort_path, score_path)
        print(json.dumps({"status": decision["science"], "output": str((run_dir / "05_analysis").resolve())}, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--first-probe", type=Path, default=DEFAULT_FIRST_PROBE)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=("cohort", "mels", "parity", "render", "score", "analysis", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
