"""Execute the frozen direct-mel Wav2Lip and SyncNet probe."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cohort import freeze_cohort
from .mel_drivers import build_drivers
from .phase_a import SEEDS, diagnose, file_sha256, read_json, write_json

REPO = Path(__file__).resolve().parents[3]
DEFAULT_PARENT = REPO / "runs/lrs3_masked_tts_retention_exploratory_20260901"
DEFAULT_RUN = REPO / "runs/lrs3_masked_tts_tfg_probe_20260902"
WAV2LIP_PY = Path.home() / ".venvs/wav2lip/bin/python"
SYNCNET_PY = Path.home() / ".venvs/syncnet/bin/python"
WAV2LIP = REPO / "third_party/Wav2Lip"
SYNCNET = REPO / "third_party/syncnet_python"
WAV2LIP_CHECKPOINT = WAV2LIP / "checkpoints/wav2lip_gan.pth"
SYNCNET_MODEL = SYNCNET / "data/syncnet_v2.model"
EXPECTED_WAV2LIP_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
EXPECTED_SYNCNET_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
MIN_TRACK = 50
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260902


def _run_logged(command: Sequence[str], cwd: Path, log: Path) -> str:
    log.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(list(command), cwd=str(cwd), capture_output=True, text=True, check=False)
    log.write_text(result.stdout + result.stderr, encoding="utf-8")
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")
    return result.stdout + result.stderr


def _last_json(text: str) -> dict[str, Any]:
    for line in reversed(text.splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("command did not emit a JSON result")


def run_parity(run_dir: Path, cohort_path: Path, drivers_path: Path, render_root: Path | None = None) -> dict[str, Any]:
    cohort = read_json(cohort_path)
    drivers = read_json(drivers_path)
    record = cohort["records"][0]
    natural_driver = next(row for row in drivers["drivers"] if row["sample_id"] == record["sample_id"] and row["condition"] == "NATURAL_MEL")
    parity_dir = (render_root or (run_dir / "03_renders")).resolve()
    log = parity_dir / "parity.log"
    output = _run_logged([
        str(WAV2LIP_PY), str(REPO / "scripts/experiments/masked_tts_tfg_probe/direct_mel.py"), "parity",
        "--audio", str(record["natural_audio"]), "--mel", str(natural_driver["path"]), "--fps", "25",
    ], REPO, log)
    result = _last_json(output)
    result.update({
        "record": record["sample_id"],
        "source_group": record["source_group"],
        "driver_sha256": natural_driver["sha256"],
        "log": str(log.resolve()),
        "wav2lip_audio_sha256": file_sha256(WAV2LIP / "audio.py"),
        "wav2lip_hparams_sha256": file_sha256(WAV2LIP / "hparams.py"),
    })
    write_json(parity_dir / "parity.json", result)
    return result


def render_all(run_dir: Path, cohort_path: Path, drivers_path: Path, parity_path: Path, render_root: Path | None = None) -> dict[str, Any]:
    parity = read_json(parity_path)
    if parity.get("status") != "PASS":
        raise ValueError("natural-mel parity has not passed")
    if file_sha256(WAV2LIP_CHECKPOINT) != EXPECTED_WAV2LIP_SHA256:
        raise ValueError("Wav2Lip checkpoint hash changed")
    cohort = read_json(cohort_path)
    drivers = read_json(drivers_path)
    records_by_id = {str(row["sample_id"]): row for row in cohort["records"]}
    render_dir = (render_root or (run_dir / "03_renders")).resolve()
    expected_count = int(drivers.get("driver_count", -1))
    if expected_count <= 0 or len(drivers.get("drivers", [])) != expected_count:
        raise ValueError("driver manifest is incomplete")
    render_rows: list[dict[str, Any]] = []
    for driver in drivers["drivers"]:
        sample_id = str(driver["sample_id"])
        record = records_by_id[sample_id]
        output = render_dir / "videos" / f"{driver['driver_id']}.mp4"
        boxes = render_dir / "boxes" / f"{sample_id}.json"
        log = render_dir / "logs" / f"{driver['driver_id']}.log"
        command = [
            str(WAV2LIP_PY), str(REPO / "scripts/experiments/masked_tts_tfg_probe/direct_mel.py"), "render",
            "--checkpoint", str(WAV2LIP_CHECKPOINT), "--face", str(record["face"]),
            "--mel", str(driver["path"]), "--outfile", str(output), "--face-det-batch-size", "4",
            "--wav2lip-batch-size", "4", "--nosmooth",
        ]
        if boxes.is_file():
            command.extend(["--boxes-input", str(boxes)])
        else:
            command.extend(["--boxes-output", str(boxes)])
        if not output.is_file() or file_sha256(output) != str(driver.get("rendered_video_sha256", "")):
            output.parent.mkdir(parents=True, exist_ok=True)
            text = _run_logged(command, REPO, log)
            direct_result = _last_json(text)
        else:
            direct_result = {"outfile": str(output), "resumed": True}
        if not output.is_file() or output.stat().st_size == 0:
            raise RuntimeError(f"missing direct-mel render: {output}")
        row = {
            "driver_id": driver["driver_id"], "sample_id": sample_id, "source_group": record["source_group"],
            "condition": driver["condition"], "seed": driver["seed"], "driver_sha256": driver["sha256"],
            "face": record["face"], "face_sha256": record["face_sha256"], "video": str(output.resolve()),
            "video_sha256": file_sha256(output), "boxes": str(boxes.resolve()), "log": str(log.resolve()),
            "command": command, "direct_result": direct_result,
        }
        render_rows.append(row)
        print(f"rendered {len(render_rows)}/{expected_count} {driver['driver_id']}", flush=True)
    if len(render_rows) != expected_count:
        raise ValueError(f"render matrix incomplete: {len(render_rows)}/{expected_count}")
    result = {
        "schema_version": 1, "manifest_type": "lrs3_masked_tts_direct_mel_wav2lip_renders", "status": "complete",
        "cohort_manifest_sha256": file_sha256(cohort_path), "drivers_manifest_sha256": file_sha256(drivers_path),
        "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256, "render_count": len(render_rows), "renders": render_rows,
        "sealed_splits_accessed": False,
    }
    write_json(render_dir / "render_manifest.json", result)
    return result


def _parse_syncnet(log: Path) -> dict[str, float | int]:
    text = log.read_text(encoding="utf-8", errors="replace")
    confidence = re.search(r"Confidence:\s+([0-9.]+)", text)
    distance = re.search(r"Min dist:\s+([0-9.]+)", text)
    offset = re.search(r"AV offset:\s+(-?\d+)", text)
    if confidence is None or distance is None:
        raise ValueError(f"SyncNet score missing from {log}")
    return {"sync_c": float(confidence.group(1)), "sync_d": float(distance.group(1)), "av_offset": int(offset.group(1)) if offset else 0}


def score_all(run_dir: Path, cohort_path: Path, render_path: Path) -> dict[str, Any]:
    from scripts.experiments.lrs3_mfa_linear_replacement.strict_mux import mux_and_verify

    if file_sha256(SYNCNET_MODEL) != EXPECTED_SYNCNET_SHA256:
        raise ValueError("SyncNet model hash changed")
    cohort = read_json(cohort_path)
    renders = read_json(render_path)
    records_by_id = {str(row["sample_id"]): row for row in cohort["records"]}
    expected_count = int(renders.get("render_count", -1))
    if expected_count <= 0 or len(renders.get("renders", [])) != expected_count:
        raise ValueError("render manifest is incomplete")
    replacement_dir = run_dir / "04_replacement"
    sync_dir = run_dir / "05_syncnet"
    replacement_dir.mkdir(parents=True, exist_ok=True)
    sync_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for index, render in enumerate(renders["renders"], 1):
        sample_id = str(render["sample_id"])
        record = records_by_id[sample_id]
        muxed = replacement_dir / f"{render['driver_id']}.mkv"
        mux_log = replacement_dir / "logs" / f"{render['driver_id']}.log"
        if muxed.is_file():
            mux = {"path": str(muxed.resolve()), "sha256": file_sha256(muxed), "resumed": True}
        else:
            mux = mux_and_verify(
                source_video=Path(str(render["video"])), expected_audio=Path(str(record["natural_audio"])),
                output_path=muxed,
            )
        cell_sync_dir = sync_dir / str(render["driver_id"])
        cell_sync_dir.mkdir(parents=True, exist_ok=True)
        reference = f"masked_tts_tfg_{sample_id}"
        pipeline_log = sync_dir / "logs" / f"{render['driver_id']}.pipeline.log"
        score_log = sync_dir / "logs" / f"{render['driver_id']}.score.log"
        if not (cell_sync_dir / "syncnet_v2.model").exists():
            _run_logged([
                str(SYNCNET_PY), "run_pipeline.py", "--videofile", str(muxed), "--reference", reference,
                "--data_dir", str(cell_sync_dir), "--min_track", str(MIN_TRACK), "--overwrite",
            ], SYNCNET, pipeline_log)
        _run_logged([
            str(SYNCNET_PY), "run_syncnet.py", "--videofile", str(muxed), "--reference", reference,
            "--data_dir", str(cell_sync_dir), "--initial_model", str(SYNCNET_MODEL),
        ], SYNCNET, score_log)
        scores = _parse_syncnet(score_log)
        row = {
            "driver_id": render["driver_id"], "sample_id": sample_id, "source_group": record["source_group"],
            "condition": render["condition"], "seed": render["seed"], "driver_sha256": render["driver_sha256"],
            "render_video_sha256": render["video_sha256"], "replacement": mux, "replacement_path": str(muxed.resolve()),
            "replacement_sha256": file_sha256(muxed), "natural_audio": record["natural_audio"],
            "natural_audio_sha256": record["natural_audio_sha256"], "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256,
            "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256, "min_track": MIN_TRACK, "reference": reference,
            "pipeline_log": str(pipeline_log.resolve()), "score_log": str(score_log.resolve()), **scores,
        }
        rows.append(row)
        print(f"scored {index}/{expected_count} {render['driver_id']} C={scores['sync_c']:.3f} D={scores['sync_d']:.3f}", flush=True)
    if len(rows) != expected_count:
        raise ValueError(f"SyncNet matrix incomplete: {len(rows)}/{expected_count}")
    result = {
        "schema_version": 1, "manifest_type": "lrs3_masked_tts_direct_mel_replacement_syncnet", "status": "complete",
        "cohort_manifest_sha256": file_sha256(cohort_path), "render_manifest_sha256": file_sha256(render_path),
        "wav2lip_checkpoint_sha256": EXPECTED_WAV2LIP_SHA256, "syncnet_model_sha256": EXPECTED_SYNCNET_SHA256,
        "min_track": MIN_TRACK, "score_count": len(rows), "scores": rows, "primary_audio": "untouched_natural_audio",
        "sealed_splits_accessed": False,
    }
    write_json(sync_dir / "summary.json", result)
    return result


def _bootstrap(values: np.ndarray) -> dict[str, Any]:
    if values.shape != (8,) or not np.isfinite(values).all():
        raise ValueError("bootstrap requires eight finite group values")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    medians = np.median(values[indices], axis=1)
    return {
        "median": float(np.median(values)),
        "ci95": [float(np.quantile(medians, 0.025, method="linear")), float(np.quantile(medians, 0.975, method="linear"))],
        "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "whole_source_group", "method": "numpy_quantile_linear",
    }


def analyze_scores(run_dir: Path, cohort_path: Path, score_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    cohort = read_json(cohort_path)
    scores = read_json(score_path)
    if scores.get("status") != "complete" or int(scores.get("score_count", -1)) != 80:
        return ({"schema_version": 1, "status": "incomplete"}, {"schema_version": 1, "science": "NOT_EVALUATED"})
    by_key = {(str(row["source_group"]), int(row["seed"]), str(row["condition"])): row for row in scores["scores"] if row["condition"] != "NATURAL_MEL"}
    groups = [str(row["source_group"]) for row in cohort["records"]]
    seed_rows: list[dict[str, Any]] = []
    for group in groups:
        for seed in SEEDS:
            try:
                paired = by_key[(group, seed, "PAIRED_TTS")]
                centroid = by_key[(group, seed, "PHONE_CENTROID")]
                nat = by_key[(group, seed, "NAT_ONLY")]
            except KeyError as error:
                raise ValueError(f"missing score cell: {error}")
            seed_rows.append({
                "source_group": group, "seed": seed,
                "token_C_gain": float(paired["sync_c"]) - float(centroid["sync_c"]),
                "token_D_gain": float(centroid["sync_d"]) - float(paired["sync_d"]),
                "modality_C_gain": float(paired["sync_c"]) - float(nat["sync_c"]),
                "modality_D_gain": float(nat["sync_d"]) - float(paired["sync_d"]),
                "paired_sync_c": float(paired["sync_c"]), "paired_sync_d": float(paired["sync_d"]),
                "phone_centroid_sync_c": float(centroid["sync_c"]), "phone_centroid_sync_d": float(centroid["sync_d"]),
                "nat_only_sync_c": float(nat["sync_c"]), "nat_only_sync_d": float(nat["sync_d"]),
            })
    metric_names = ("token_C_gain", "token_D_gain", "modality_C_gain", "modality_D_gain")
    group_rows: list[dict[str, Any]] = []
    for group in groups:
        local = [row for row in seed_rows if row["source_group"] == group]
        group_rows.append({"source_group": group, "seed_count": len(local), **{name: float(np.median([row[name] for row in local])) for name in metric_names}})
    bootstrap = {name: _bootstrap(np.asarray([row[name] for row in group_rows], dtype=np.float64)) for name in metric_names}
    wins = {name: int(sum(float(row[name]) > 0 for row in group_rows)) for name in metric_names}
    token_pass = bootstrap["token_C_gain"]["ci95"][0] > 0 and bootstrap["token_D_gain"]["ci95"][0] > 0 and wins["token_C_gain"] >= 7 and wins["token_D_gain"] >= 7
    modality_pass = bootstrap["modality_C_gain"]["ci95"][0] > 0 and bootstrap["modality_D_gain"]["ci95"][0] > 0 and wins["modality_C_gain"] >= 7 and wins["modality_D_gain"] >= 7
    if token_pass:
        status = "EXPLORATORY_TOKEN_SIGNAL"
    elif modality_pass:
        status = "EXPLORATORY_MODALITY_ONLY"
    else:
        status = "NO_EXPLORATORY_TFG_GAIN"
    natural_reference = [row for row in scores["scores"] if row["condition"] == "NATURAL_MEL"]
    analysis = {
        "schema_version": 1, "status": "complete", "phase": "B", "score_count": len(scores["scores"]),
        "cohort_manifest_sha256": file_sha256(cohort_path), "score_manifest_sha256": file_sha256(score_path),
        "seed_rows": seed_rows, "group_rows": group_rows, "bootstrap": bootstrap, "wins": wins,
        "rules": {"token": "both lower bounds > 0 and at least 7/8 positive groups on both metrics", "modality": "both lower bounds > 0 and at least 7/8 positive groups on both metrics"},
        "natural_mel_reference": [{"source_group": row["source_group"], "sync_c": row["sync_c"], "sync_d": row["sync_d"]} for row in natural_reference],
        "sealed_splits_accessed": False,
        "claim_boundary": [
            "direct-mel frozen-Wav2Lip sensitivity probe only",
            "primary audio is untouched natural audio after strict replacement",
            "not audible TTS-feature retention", "not waveform reachability", "not audio quality",
            "not natural prosody preservation", "not population generalization", "not a deployable replacement system",
        ],
    }
    decision = {
        "schema_version": 1, "engineering": "GO", "science": status, "exploratory": True,
        "claim": (
            "paired TTS-conditioned mel drives frozen Wav2Lip motion that is more compatible with untouched natural audio than phone-centroid drives in this eight-group direct-mel probe"
            if status == "EXPLORATORY_TOKEN_SIGNAL" else
            "TTS-side conditioning has downstream value relative to NAT_ONLY in this eight-group direct-mel probe"
            if status == "EXPLORATORY_MODALITY_ONLY" else
            "the feature-space masked-TTS contrast did not produce a robust gain at this frozen Wav2Lip/SyncNet endpoint"
        ),
        "claim_boundary": analysis["claim_boundary"], "sealed_splits_accessed": False,
    }
    output = run_dir / "06_analysis"
    write_json(output / "analysis.json", analysis)
    write_json(output / "decision.json", decision)
    report = render_analysis_report(analysis, decision)
    (output / "report.md").write_text(report, encoding="utf-8")
    write_json(run_dir / "decision.json", decision)
    write_json(run_dir / "summary.json", {
        "schema_version": 1, "run": "lrs3_masked_tts_tfg_probe_20260902", "status": status,
        "cohort_records": 8, "driver_count": 80, "render_count": 80, "score_count": 80,
        "decision": decision, "sealed_splits_accessed": False,
    })
    return analysis, decision


def render_analysis_report(analysis: Mapping[str, Any], decision: Mapping[str, Any]) -> str:
    lines = [
        "# Direct-mel frozen-Wav2Lip probe",
        "",
        f"Status: **{decision['science']}**.",
        "",
        "The matrix uses one score-independent record from each of the eight parent evaluation groups. Every rendered video was muxed with that record's untouched natural audio before official SyncNet V2 scoring.",
        "",
        "## Group-level gains",
        "",
        "| Group | Token C | Token D | Modality C | Modality D |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in analysis["group_rows"]:
        lines.append(f"| `{row['source_group']}` | {row['token_C_gain']:.4f} | {row['token_D_gain']:.4f} | {row['modality_C_gain']:.4f} | {row['modality_D_gain']:.4f} |")
    lines.extend(["", "## Bootstrap and wins", ""])
    for name, row in analysis["bootstrap"].items():
        lines.append(f"- `{name}` median `{row['median']:.5f}`, 95% CI `[{row['ci95'][0]:.5f}, {row['ci95'][1]:.5f}]`; positive groups `{analysis['wins'][name]}/8`.")
    lines.extend(["", "## Claim boundary", "", decision["claim"], "", "This is not evidence of audible TTS-feature retention, waveform reachability, audio quality, natural-prosody preservation, population generalization, or a deployable replacement system.", ""])
    return "\n".join(lines)


def execute(args: argparse.Namespace) -> int:
    parent = args.parent.resolve()
    run_dir = args.run.resolve()
    diagnosis_path = run_dir / "00_diagnosis/analysis.json"
    cohort_path = run_dir / "01_cohort/manifest.json"
    drivers_dir = run_dir / "02_mels"
    drivers_path = drivers_dir / "drivers.json"
    parity_path = run_dir / "03_renders/parity.json"
    render_path = run_dir / "03_renders/render_manifest.json"
    score_path = run_dir / "05_syncnet/summary.json"
    if args.stage == "score":
        if not render_path.is_file():
            raise ValueError("render manifest is required before scoring")
        score_all(run_dir, cohort_path, render_path)
        return 0
    if args.stage == "analysis":
        if not score_path.is_file():
            raise ValueError("SyncNet summary is required before analysis")
        _, decision = analyze_scores(run_dir, cohort_path, score_path)
        print(json.dumps({"status": decision["science"], "output": str((run_dir / "06_analysis").resolve())}, ensure_ascii=False))
        return 0
    if args.stage in {"diagnosis", "all"}:
        diagnose(parent, run_dir / "00_diagnosis")
    if args.stage == "diagnosis":
        return 0
    if not diagnosis_path.is_file():
        raise ValueError("Phase A diagnosis is required before Phase B")
    if args.stage in {"cohort", "mels", "parity", "render", "score", "analysis", "all"}:
        freeze_cohort(parent, diagnosis_path, cohort_path)
    if args.stage in {"mels", "parity", "render", "score", "analysis", "all"}:
        build_drivers(parent, cohort_path, drivers_dir)
    if args.stage in {"parity", "render", "score", "analysis", "all"}:
        run_parity(run_dir, cohort_path, drivers_path)
    if args.stage in {"render", "score", "analysis", "all"}:
        render_all(run_dir, cohort_path, drivers_path, parity_path)
    if args.stage in {"score", "analysis", "all"}:
        score_all(run_dir, cohort_path, render_path)
    if args.stage in {"analysis", "all"}:
        _, decision = analyze_scores(run_dir, cohort_path, score_path)
        print(json.dumps({"status": decision["science"], "output": str((run_dir / "06_analysis").resolve())}, ensure_ascii=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--run", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--stage", choices=("diagnosis", "cohort", "mels", "parity", "render", "score", "analysis", "all"), default="all")
    args = parser.parse_args(argv)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
