from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.experiments.lrs3_mfa_linear_replacement.strict_mux import mux_and_verify

from . import config
from .diagonal import _parse_syncnet, _run_logged
from .protocol import (
    ProtocolError,
    canonical_sha256,
    load_json,
    sha256_file,
    write_json_once,
)
from .statistics import cluster_bootstrap

PROTOCOL_ID = "lrs3_mfa_dtw_exploratory_replacement_20260904"
SOURCE_ROOT = config.REPO / "runs/lrs3_mfa_dtw_replacement_short_phone_20260904"
OUTPUT_ROOT = config.REPO / "runs/lrs3_mfa_dtw_exploratory_replacement_20260904"


def _validate_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage00 = load_json(SOURCE_ROOT / "00_protocol" / "manifest.json")
    diagonal = load_json(SOURCE_ROOT / "02_diagonal" / "diagonal_manifest.json")
    analysis = load_json(SOURCE_ROOT / "03_diagonal_analysis" / "analysis.json")
    if (
        stage00.get("protocol_id") != "lrs3_mfa_dtw_short_phone_support_20260904"
        or stage00.get("status") != "complete"
        or stage00.get("cohort", {}).get("record_count") != config.EXPECTED_RECORD_COUNT
    ):
        raise ProtocolError("short-phone Stage 00 is not complete")
    if (
        diagonal.get("protocol_id") != stage00.get("protocol_id")
        or diagonal.get("status") != "complete"
        or len(diagonal.get("scores", [])) != config.EXPECTED_RECORD_COUNT
    ):
        raise ProtocolError("short-phone diagonal matrix is not complete")
    if (
        analysis.get("protocol_id") != stage00.get("protocol_id")
        or analysis.get("decision") != "NO_DTW_TFG_ADVANTAGE"
        or analysis.get("replacement_authorized") is not False
    ):
        raise ProtocolError("registered diagonal decision is not the expected sealed no-advantage result")
    stage_rows = stage00["cohort"]["records"]
    diagonal_rows = diagonal["scores"]
    if [str(row["sample_id"]) for row in diagonal_rows] != [str(row["sample_id"]) for row in stage_rows]:
        raise ProtocolError("diagonal order differs from frozen cohort")
    return stage00, diagonal, analysis


def _score_cell(
    stage_row: dict[str, Any],
    diagonal_row: dict[str, Any],
    output: Path,
) -> dict[str, Any]:
    sample_id = str(stage_row["sample_id"])
    video = Path(diagonal_row["video"])
    natural_audio = Path(stage_row["natural_audio"]["path"])
    if not video.is_file() or sha256_file(video) != diagonal_row["video_sha256"]:
        raise ProtocolError(f"DTW video hash changed: {sample_id}")
    if not natural_audio.is_file() or sha256_file(natural_audio) != stage_row["natural_audio"]["sha256"]:
        raise ProtocolError(f"natural audio hash changed: {sample_id}")
    muxed = output / "mux" / f"{sample_id}.mkv"
    muxed.parent.mkdir(parents=True, exist_ok=True)
    verification = mux_and_verify(
        source_video=video,
        expected_audio=natural_audio,
        output_path=muxed,
        ffmpeg=str(config.FFMPEG),
        ffprobe=str(config.FFPROBE),
    )
    sync_dir = output / "syncnet" / sample_id
    pipeline_log = output / "logs" / f"{sample_id}.pipeline.log"
    score_log = output / "logs" / f"{sample_id}.score.log"
    reference = f"lrs3_mfa_dtw_exploratory_replacement_{sample_id}"
    pipeline_command = [
        str(config.SYNCNET_PYTHON), "run_pipeline.py", "--videofile", str(muxed),
        "--reference", reference, "--data_dir", str(sync_dir), "--min_track", str(config.MIN_TRACK), "--overwrite",
    ]
    if _run_logged(pipeline_command, cwd=config.SYNCNET_ROOT, log_path=pipeline_log) != 0:
        raise RuntimeError(f"SyncNet pipeline failed for replacement {sample_id}")
    score_command = [
        str(config.SYNCNET_PYTHON), "run_syncnet.py", "--videofile", str(muxed),
        "--reference", reference, "--data_dir", str(sync_dir), "--initial_model", str(config.SYNCNET_MODEL),
    ]
    if _run_logged(score_command, cwd=config.SYNCNET_ROOT, log_path=score_log) != 0:
        raise RuntimeError(f"SyncNet score failed for replacement {sample_id}")
    parsed = _parse_syncnet(score_log)
    row = {
        "schema_version": 1,
        "sample_id": sample_id,
        "source_group": stage_row["source_group"],
        "video": str(video),
        "video_sha256": diagonal_row["video_sha256"],
        "audio": str(natural_audio),
        "audio_sha256": stage_row["natural_audio"]["sha256"],
        "muxed_file": str(muxed),
        "muxed_file_sha256": sha256_file(muxed),
        "mux_verification": verification,
        "syncnet_model_sha256": sha256_file(config.SYNCNET_MODEL),
        "min_track": config.MIN_TRACK,
        "pipeline_log": str(pipeline_log),
        "score_log": str(score_log),
        "sync_c": float(parsed["sync_c"]),
        "sync_d": float(parsed["sync_d"]),
        "av_offset": int(parsed["av_offset"]),
    }
    write_json_once(output / "scores" / f"{sample_id}.json", row)
    return row


def _contrast_rows(stage_rows: list[dict[str, Any]], dtw_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    contrasts = {"dtw_vs_natural": [], "linear_vs_natural": [], "dtw_vs_linear_replacement": []}
    for dtw, source in zip(dtw_rows, stage_rows, strict=True):
        if str(dtw["sample_id"]) != str(source["sample_id"]):
            raise ProtocolError("replacement order differs from frozen cohort")
        natural = source["historical_scores"]["G_N_E_N"]
        linear = source["historical_scores"]["G_M_E_N"]
        common = {"sample_id": str(dtw["sample_id"]), "source_group": source["source_group"]}
        contrasts["dtw_vs_natural"].append({
            **common,
            "delta_C": float(dtw["sync_c"] - natural["sync_c"]),
            "delta_D": float(natural["sync_d"] - dtw["sync_d"]),
        })
        contrasts["linear_vs_natural"].append({
            **common,
            "delta_C": float(linear["sync_c"] - natural["sync_c"]),
            "delta_D": float(natural["sync_d"] - linear["sync_d"]),
        })
        contrasts["dtw_vs_linear_replacement"].append({
            **common,
            "delta_C": float(dtw["sync_c"] - linear["sync_c"]),
            "delta_D": float(linear["sync_d"] - dtw["sync_d"]),
        })
    return contrasts


def _summarize_contrast(rows: list[dict[str, Any]]) -> dict[str, Any]:
    bootstrap = {
        "delta_C": cluster_bootstrap(rows, "delta_C"),
        "delta_D": cluster_bootstrap(rows, "delta_D"),
    }
    return {
        "record_count": len(rows),
        "source_group_count": len({str(row["source_group"]) for row in rows}),
        "bootstrap": bootstrap,
        "wins": {
            "C_positive_count": int(sum(row["delta_C"] > 0 for row in rows)),
            "D_positive_count": int(sum(row["delta_D"] > 0 for row in rows)),
            "joint_positive_count": int(sum(row["delta_C"] > 0 and row["delta_D"] > 0 for row in rows)),
        },
    }


def run(output_dir: str | Path = OUTPUT_ROOT) -> dict[str, Any]:
    output = Path(output_dir).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"exploratory replacement output is already populated: {output}")
    stage00, diagonal, registered_analysis = _validate_inputs()
    output.mkdir(parents=True, exist_ok=True)
    stage_rows = list(stage00["cohort"]["records"])
    dtw_rows: list[dict[str, Any]] = []
    for index, (stage_row, diagonal_row) in enumerate(zip(stage_rows, diagonal["scores"], strict=True), 1):
        row = _score_cell(stage_row, diagonal_row, output)
        dtw_rows.append(row)
        print(f"OK {index}/{len(stage_rows)} {row['sample_id']} C={row['sync_c']:.3f} D={row['sync_d']:.3f}", flush=True)
    if len(dtw_rows) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("exploratory replacement matrix is incomplete")
    contrasts = _contrast_rows(stage_rows, dtw_rows)
    statistics = {name: _summarize_contrast(rows) for name, rows in contrasts.items()}
    parents = {
        "stage00_manifest": {
            "path": str((SOURCE_ROOT / "00_protocol" / "manifest.json").resolve()),
            "sha256": sha256_file(SOURCE_ROOT / "00_protocol" / "manifest.json"),
        },
        "diagonal_manifest": {
            "path": str((SOURCE_ROOT / "02_diagonal" / "diagonal_manifest.json").resolve()),
            "sha256": sha256_file(SOURCE_ROOT / "02_diagonal" / "diagonal_manifest.json"),
        },
        "registered_analysis": {
            "path": str((SOURCE_ROOT / "03_diagonal_analysis" / "analysis.json").resolve()),
            "sha256": sha256_file(SOURCE_ROOT / "03_diagonal_analysis" / "analysis.json"),
        },
    }
    manifest = {
        "schema_version": 1,
        "manifest_type": "lrs3_mfa_dtw_exploratory_replacement",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "descriptive_only",
        "registered_replacement_authorized": False,
        "registered_diagonal_decision": registered_analysis["decision"],
        "interpretation": "descriptive replacement diagnostic; does not alter or promote the registered experiment",
        "cohort_sha256": config.EXPECTED_COHORT_HASH,
        "record_count": len(dtw_rows),
        "parents": parents,
        "replacement_contract": {
            "video_source": "registered DTW diagonal video",
            "audio_source": "untouched natural PCM",
            "video_stream_copy": True,
            "audio_codec": "pcm_s16le",
            "decoded_pcm_exact_match_required": True,
            "official_syncnet_v2": True,
        },
        "scores": dtw_rows,
        "contrasts": contrasts,
        "statistics": statistics,
        "cohort_order_sha256": canonical_sha256([str(row["sample_id"]) for row in stage_rows]),
    }
    write_json_once(output / "replacement_manifest.json", manifest)
    summary = {
        "schema_version": 1,
        "stage_id": "exploratory_replacement",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "engineering_decision": "GO",
        "scientific_decision": "descriptive_only",
        "registered_replacement_authorized": False,
        "record_count": len(dtw_rows),
        "replacement_pcm_verified": all(row["mux_verification"]["audio_pcm_verified"] for row in dtw_rows),
        "video_stream_copy_verified": all(row["mux_verification"]["video_stream_copy_verified"] for row in dtw_rows),
        "statistics": statistics,
    }
    write_json_once(output / "summary.json", summary)
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args(argv)
    result = run(args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
