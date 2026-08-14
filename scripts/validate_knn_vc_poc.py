#!/usr/bin/env python3
"""Validate provenance, QC, and WavLM geometry for a kNN-VC PoC run."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from knn_vc_poc_schema import (
    EXPECTED_CONDITIONS,
    file_sha256,
    load_json,
    validate_summary,
    write_json,
)


def _mean(rows: Sequence[Mapping[str, Any]], path: Sequence[str]) -> float:
    values: list[float] = []
    for row in rows:
        value: Any = row
        for key in path:
            value = value[key]
        values.append(float(value))
    result = float(np.mean(values))
    if not np.isfinite(result):
        raise FloatingPointError("aggregate metric is non-finite")
    return result


def validate_run(summary_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    summary_path = Path(summary_path).resolve()
    summary = load_json(summary_path)
    validate_summary(summary)
    failures: list[str] = []
    by_condition: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for item in summary["items"]:
        for row in item["conditions"]:
            condition = str(row["condition"])
            by_condition[condition].append(row)
            wav_path = Path(row["output_path"])
            feature_path = Path(row["feature_path"])
            if not wav_path.is_file() or file_sha256(wav_path) != row["output_sha256"]:
                failures.append(f"{item['paired_key']}:{condition}: output hash mismatch")
                continue
            if not feature_path.is_file() or file_sha256(feature_path) != row["conditioning_sha256"]:
                failures.append(f"{item['paired_key']}:{condition}: feature hash mismatch")
            audio, sample_rate = sf.read(wav_path, dtype="float32", always_2d=False)
            if audio.ndim != 1 or sample_rate != 16_000:
                failures.append(f"{item['paired_key']}:{condition}: output is not mono 16 kHz")
            qc = row["audio"]
            if len(audio) != qc["natural_sample_count"] or qc["exact_natural_sample_count"] is not True:
                failures.append(f"{item['paired_key']}:{condition}: output is not exact natural N")
            if not np.isfinite(audio).all() or not qc["finite"] or qc["rms"] <= 0:
                failures.append(f"{item['paired_key']}:{condition}: non-finite or silent output")
            if int(qc["clipped_sample_count"]) != 0:
                failures.append(f"{item['paired_key']}:{condition}: clipped output")
            retrieval = row["retrieval"]
            if not 0.0 <= float(retrieval["coverage"]) <= 1.0 or int(retrieval["fallback_frames"]) < 0:
                failures.append(f"{item['paired_key']}:{condition}: invalid retrieval coverage")
            if row["conditioning_shape"][1] != 1024:
                failures.append(f"{item['paired_key']}:{condition}: conditioning is not WavLM 1024-D")
    if tuple(by_condition) != EXPECTED_CONDITIONS:
        failures.append("aggregate condition order/set mismatch")

    aggregates: dict[str, Any] = {}
    for condition in EXPECTED_CONDITIONS:
        rows = by_condition[condition]
        aggregates[condition] = {
            "item_count": len(rows),
            "mean_output_to_aligned_tts_cosine": _mean(
                rows, ("wavlm_synthesis_1024", "output_to_aligned_tts", "cosine")
            ),
            "mean_natural_to_aligned_tts_cosine": _mean(
                rows, ("wavlm_synthesis_1024", "natural_query_to_aligned_tts", "cosine")
            ),
            "mean_output_to_conditioning_cosine": _mean(
                rows, ("wavlm_synthesis_1024", "output_to_conditioning", "cosine")
            ),
            "closer_to_aligned_tts_fraction": float(np.mean([
                row["wavlm_synthesis_1024"]["output_to_aligned_tts"]["cosine"]
                < row["wavlm_synthesis_1024"]["natural_query_to_aligned_tts"]["cosine"]
                for row in rows
            ])),
            "mean_length_adjustment_samples": _mean(rows, ("length_adjustment", "adjustment_sample_count")),
        }
    same = aggregates["paired_tts_same_phone"]["mean_output_to_aligned_tts_cosine"]
    positioned = aggregates["paired_tts_same_phone_position"]["mean_output_to_aligned_tts_cosine"]
    wrong = aggregates["paired_tts_wrong_phone"]["mean_output_to_aligned_tts_cosine"]
    negative_control_pass = bool(min(same, positioned) < wrong)
    if not negative_control_pass:
        failures.append("same-phone conditions do not beat wrong-phone negative in WavLM space")

    report = {
        "schema_version": 1,
        "validation_type": "wavlm_knn_vc_poc_gate",
        "summary_path": str(summary_path),
        "summary_sha256": file_sha256(summary_path),
        "item_count": len(summary["items"]),
        "heldout_excluded": True,
        "metric_spaces": {
            "primary": "wavlm_synthesis_1024",
            "project_hubert_768": "not_run_by_this_validator_use_41_hubert_proximity_audit",
            "output_phone_boundaries": "not_measured_mfa_cli_unavailable",
        },
        "aggregates": aggregates,
        "negative_control_pass": negative_control_pass,
        "failures": failures,
        "gate_pass": not failures,
    }
    write_json(output_path, report)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args(argv)
    report = validate_run(args.summary, args.output)
    print(json.dumps({"gate_pass": report["gate_pass"], "failures": report["failures"]}, indent=2))
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
