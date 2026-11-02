#!/usr/bin/env python3
"""Summarize prespecified source-axis contrasts from the sealed independent row report."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


SOURCE_SUMMARY_SHA256 = "55988932b41c0410f10acdb87db43ac30d21315d9ef1a5e06f423462c5403f1c"
EXPECTED_ROWS = 2560
EXPECTED_CLIPS = 80
EXPECTED_SPEAKERS = 40
FACES = ("primary", "secondary")
CONDITIONS = ("baseline", "orth_plus", "orth_minus")
ALL_AXIS_CONDITIONS = ("baseline", "full_plus", "full_minus", "orth_plus", "orth_minus", "random1", "random2", "random3")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C", "C_anchor", "B", "D", "search_gain")
CONTRASTS = {
    "orth_plus_minus_baseline": ("orth_plus", "baseline"),
    "orth_plus_minus_orth_minus": ("orth_plus", "orth_minus"),
}
BOOTSTRAP_SEED = 20260930
BOOTSTRAP_DRAWS = 20_000


class AnalysisError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _speaker_from_sample(sample_id: str) -> str:
    match = re.match(r"^(S\d+)_", sample_id)
    if not match:
        raise AnalysisError(f"sample id does not begin with a frozen speaker id: {sample_id}")
    return match.group(1)


def _bootstrap(values_by_sample: dict[str, float], sample_ids: list[str]) -> dict[str, Any]:
    by_speaker: dict[str, list[float]] = defaultdict(list)
    for sample_id in sample_ids:
        by_speaker[_speaker_from_sample(sample_id)].append(float(values_by_sample[sample_id]))
    if len(by_speaker) != EXPECTED_SPEAKERS or any(len(values) != 2 for values in by_speaker.values()):
        raise AnalysisError("the frozen sample IDs do not form 40 speakers with two clips each")
    speakers = sorted(by_speaker)
    speaker_values = {
        speaker: float(np.mean(np.asarray(by_speaker[speaker], dtype=np.float64), dtype=np.float64))
        for speaker in speakers
    }
    vector = np.asarray([speaker_values[speaker] for speaker in speakers], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(speakers), size=(BOOTSTRAP_DRAWS, len(speakers)), endpoint=False)
    means = vector[indices].mean(axis=1, dtype=np.float64)
    low, high = np.quantile(means, [0.005, 0.995], method="linear")
    return {
        "equal_speaker_mean": float(vector.mean(dtype=np.float64)),
        "speaker_bootstrap_99_ci": [float(low), float(high)],
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "complete_speaker_count": len(speakers),
        "complete_clip_count": len(sample_ids),
        "speaker_values": speaker_values,
    }


def analyze(source_summary: Path, output_path: Path) -> dict[str, Any]:
    source_summary = source_summary.resolve(strict=True)
    actual_sha = sha256_file(source_summary)
    if actual_sha != SOURCE_SUMMARY_SHA256:
        raise AnalysisError(f"sealed source summary SHA mismatch: {actual_sha}")
    source = json.loads(source_summary.read_text(encoding="utf-8"))
    if source.get("status") != "complete" or source.get("protocol_sha256") != "8ba27fcb395dccbb493ad4831e8b68ac18c485b61e39d6e3b7b3018fd668cf39":
        raise AnalysisError("source consumer report is not the expected sealed retry3 result")
    if source.get("row_manifest_sha256") != "19f805f6935c85475ad28319e17d2694a333a85d055d3e19a0eb93bda7390dab":
        raise AnalysisError("source consumer report binds another row plan")
    rows = source.get("score_support_summaries_by_row")
    if not isinstance(rows, dict) or len(rows) != EXPECTED_ROWS:
        raise AnalysisError("source consumer report does not contain all 2,560 validated score rows")

    axis: dict[tuple[str, str, str], dict[str, Any]] = {}
    sample_speakers: dict[str, str] = {}
    for row_id, value in rows.items():
        parts = str(row_id).split("__")
        if len(parts) != 5:
            raise AnalysisError(f"malformed source row identity: {row_id}")
        sample_id, face, family, source_name, condition = parts
        if family != "source_axis" or source_name != "natural":
            continue
        if face not in FACES or condition not in ALL_AXIS_CONDITIONS:
            raise AnalysisError(f"unexpected source-axis row identity: {row_id}")
        if condition not in CONDITIONS:
            continue
        key = (sample_id, face, condition)
        if key in axis:
            raise AnalysisError(f"duplicate source-axis row: {row_id}")
        axis[key] = value
        speaker = _speaker_from_sample(sample_id)
        if sample_id in sample_speakers and sample_speakers[sample_id] != speaker:
            raise AnalysisError(f"sample speaker mapping changed: {sample_id}")
        sample_speakers[sample_id] = speaker

    samples = sorted(sample_speakers)
    if len(samples) != EXPECTED_CLIPS or len(set(sample_speakers.values())) != EXPECTED_SPEAKERS:
        raise AnalysisError("source-axis subset is not the complete frozen 80/40 cohort")
    expected = {(sample, face, condition) for sample in samples for face in FACES for condition in CONDITIONS}
    if set(axis) != expected:
        raise AnalysisError(f"source-axis rows are incomplete: found {len(axis)}, expected {len(expected)}")

    contrasts: dict[str, Any] = {}
    for contrast_name, (left, right) in CONTRASTS.items():
        contrasts[contrast_name] = {}
        for face in FACES:
            contrasts[contrast_name][face] = {}
            for geometry in GEOMETRIES:
                contrasts[contrast_name][face][geometry] = {}
                for support in SUPPORTS:
                    contrasts[contrast_name][face][geometry][support] = {}
                    for metric in METRICS:
                        clip_values: dict[str, float] = {}
                        for sample in samples:
                            left_value = axis[(sample, face, left)][geometry][support].get(metric)
                            right_value = axis[(sample, face, right)][geometry][support].get(metric)
                            if left_value is None or right_value is None:
                                raise AnalysisError(f"missing {metric}/{support} value for {sample}/{face}/{left}/{right}")
                            delta = float(left_value) - float(right_value)
                            if not np.isfinite(delta):
                                raise AnalysisError(f"non-finite contrast for {sample}/{face}/{metric}/{support}")
                            clip_values[sample] = delta
                        contrasts[contrast_name][face][geometry][support][metric] = {
                            **_bootstrap(clip_values, samples),
                            "clip_values": clip_values,
                        }

    formal = source.get("primary_gates")
    if not isinstance(formal, dict) or formal.get("eligible") is not True:
        raise AnalysisError("the sealed original formal primary guard20 gates are not eligible")
    output = {
        "schema_version": 1,
        "analysis_type": "independent_source_axis_prespecified_sensitivity_addendum",
        "status": "COMPLETE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "consumer_source": {"path": str(Path(__file__).resolve()), "sha256": sha256_file(Path(__file__).resolve())},
        "input": {"path": str(source_summary), "sha256": actual_sha, "protocol_sha256": source["protocol_sha256"], "row_manifest_sha256": source["row_manifest_sha256"]},
        "classification": "Additive contrasts computed from the sealed independent all-row metric table; no producer scoring functions, model forwards, or row selection.",
        "verification": {
            "all_required_source_rows_present": True,
            "selected_source_axis_rows": len(axis),
            "expected_source_axis_rows": len(expected),
            "clips": len(samples), "speakers": len(set(sample_speakers.values())),
            "faces": list(FACES), "conditions": list(CONDITIONS), "supports": list(SUPPORTS),
            "metrics": list(METRICS),
            "speaker_weighting": "two clips averaged within speaker, then 40 speakers equally",
            "ci": "20,000 PCG64 speaker bootstrap draws; seed 20260930; percentile 0.005–0.995",
        },
        "contrasts": contrasts,
        "original_formal_primary_guard20_gates": formal,
    }
    output_path = output_path.resolve()
    if output_path.exists():
        raise AnalysisError(f"refusing to replace an existing addendum: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.source_summary, args.output)
    print(json.dumps({"status": result["status"], "selected_source_axis_rows": result["verification"]["selected_source_axis_rows"], "output": str(args.output.resolve())}, sort_keys=True))


if __name__ == "__main__":
    main()
