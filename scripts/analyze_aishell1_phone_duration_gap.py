#!/usr/bin/env python3
"""Compare paired natural/TTS phone durations with MFA-linear outcomes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from collections import defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Mapping, Sequence

EXPECTED_SPEAKERS = ("S0765", "S0901", "S0906", "S0912", "S0913")
DEFAULT_GOOD = ("S0765", "S0913")
DEFAULT_POOR = ("S0901", "S0912")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label(token: Mapping[str, Any]) -> str:
    return str(token.get("token", token.get("label", ""))).strip().casefold()


def _duration(token: Mapping[str, Any]) -> float:
    start = float(token["start_s"])
    end = float(token["end_s"])
    if end <= start:
        raise ValueError(f"non-positive token duration: {token}")
    return end - start


def _is_silence(token: Mapping[str, Any]) -> bool:
    return bool(token.get("is_silence") or token.get("is_non_speech"))


def _mean(values: Sequence[float]) -> float | None:
    return float(statistics.mean(values)) if values else None


def _median(values: Sequence[float]) -> float | None:
    return float(statistics.median(values)) if values else None


def _percentile(values: Sequence[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)


def _summary(values: Sequence[float]) -> dict[str, Any]:
    return {
        "n": len(values),
        "mean": _mean(values),
        "median": _median(values),
        "p90": _percentile(values, 0.9),
    }


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for position in range(index, end):
            ranks[ordered[position][0]] = rank
        index = end
    return ranks


def _load_audio_artifacts(path: Path | None) -> tuple[dict[str, Mapping[str, Any]], str | None]:
    if path is None:
        return {}, None
    rows: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("arm") == "mfa_linear_resample_poly":
                sample_id = str(row["sample_id"])
                if sample_id in rows:
                    raise ValueError(f"duplicate MFA audio-audit row: {sample_id}")
                rows[sample_id] = row
    if len(rows) != 25:
        raise ValueError(f"audio audit must contain 25 MFA rows, found {len(rows)}")
    return rows, sha256_file(path)


def _correlation(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    denominator = math.sqrt(
        sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys)
    )
    return float(numerator / denominator) if denominator else None


def _matched_pairs(
    natural_tokens: Sequence[Mapping[str, Any]],
    tts_tokens: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    natural_labels = [_label(token) for token in natural_tokens]
    tts_labels = [_label(token) for token in tts_tokens]
    matcher = SequenceMatcher(a=natural_labels, b=tts_labels, autojunk=False)
    pairs: list[dict[str, Any]] = []
    equal_count = 0
    speech_pairs = 0
    silence_pairs = 0
    for tag, natural_start, natural_end, tts_start, tts_end in matcher.get_opcodes():
        if tag != "equal":
            continue
        equal_count += natural_end - natural_start
        for natural_index, tts_index in zip(
            range(natural_start, natural_end), range(tts_start, tts_end)
        ):
            natural = natural_tokens[natural_index]
            tts = tts_tokens[tts_index]
            natural_silence = _is_silence(natural)
            tts_silence = _is_silence(tts)
            if natural_silence and tts_silence:
                silence_pairs += 1
            elif not natural_silence and not tts_silence:
                speech_pairs += 1
                natural_duration = _duration(natural)
                tts_duration = _duration(tts)
                delta = tts_duration - natural_duration
                pairs.append(
                    {
                        "natural_index": natural_index,
                        "tts_index": tts_index,
                        "phone": _label(natural),
                        "natural_duration_s": natural_duration,
                        "tts_duration_s": tts_duration,
                        "delta_s": delta,
                        "abs_delta_s": abs(delta),
                        "relative_abs_delta": abs(delta) / max(natural_duration, 1e-6),
                    }
                )
    natural_speech_count = sum(not _is_silence(token) for token in natural_tokens)
    tts_speech_count = sum(not _is_silence(token) for token in tts_tokens)
    unknown_labels = sorted(
        {
            _label(token)
            for token in [*natural_tokens, *tts_tokens]
            if _label(token) in {"spn", "<unk>", "unk"}
        }
    )
    return pairs, {
        "natural_token_count": len(natural_tokens),
        "tts_token_count": len(tts_tokens),
        "natural_speech_phone_count": natural_speech_count,
        "tts_speech_phone_count": tts_speech_count,
        "matched_token_count": equal_count,
        "matched_speech_phone_count": speech_pairs,
        "matched_silence_count": silence_pairs,
        "unmatched_natural_token_count": len(natural_tokens) - equal_count,
        "unmatched_tts_token_count": len(tts_tokens) - equal_count,
        "speech_match_rate": speech_pairs / max(natural_speech_count, tts_speech_count, 1),
        "unknown_speech_labels": unknown_labels,
    }


def _sample_metrics(
    sample_id: str,
    record: Mapping[str, Any],
    score_by_arm: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    natural_tokens = list(record["natural"]["tokens"])
    tts_tokens = list(record["tts"]["tokens"])
    pairs, match_stats = _matched_pairs(natural_tokens, tts_tokens)
    abs_deltas = [float(pair["abs_delta_s"]) for pair in pairs]
    relative_deltas = [float(pair["relative_abs_delta"]) for pair in pairs]
    signed_deltas = [float(pair["delta_s"]) for pair in pairs]
    natural_speech_duration = sum(_duration(token) for token in natural_tokens if not _is_silence(token))
    tts_speech_duration = sum(_duration(token) for token in tts_tokens if not _is_silence(token))
    natural_silence_duration = sum(_duration(token) for token in natural_tokens if _is_silence(token))
    tts_silence_duration = sum(_duration(token) for token in tts_tokens if _is_silence(token))
    natural_score = score_by_arm["natural_raw"]
    mfa_score = score_by_arm["mfa_linear_resample_poly"]
    raw_score = score_by_arm["raw_tts"]
    return {
        "sample_id": sample_id,
        "paired_key": record["paired_key"],
        "speaker_id": record["speaker_id"],
        "transcript": record["transcript"],
        **match_stats,
        "duration_mae_s": _mean(abs_deltas),
        "duration_median_abs_s": _median(abs_deltas),
        "duration_p90_abs_s": _percentile(abs_deltas, 0.9),
        "duration_mean_signed_delta_s": _mean(signed_deltas),
        "duration_mean_relative_abs_delta": _mean(relative_deltas),
        "duration_median_relative_abs_delta": _median(relative_deltas),
        "natural_speech_duration_s": natural_speech_duration,
        "tts_speech_duration_s": tts_speech_duration,
        "speech_duration_delta_s": tts_speech_duration - natural_speech_duration,
        "natural_silence_duration_s": natural_silence_duration,
        "tts_silence_duration_s": tts_silence_duration,
        "silence_duration_delta_s": tts_silence_duration - natural_silence_duration,
        "mfa_sync_c": float(mfa_score["sync_c"]),
        "mfa_sync_d": float(mfa_score["sync_d"]),
        "natural_sync_c": float(natural_score["sync_c"]),
        "natural_sync_d": float(natural_score["sync_d"]),
        "raw_tts_sync_c": float(raw_score["sync_c"]),
        "raw_tts_sync_d": float(raw_score["sync_d"]),
        "mfa_minus_natural_sync_c": float(mfa_score["sync_c"] - natural_score["sync_c"]),
        "mfa_minus_natural_sync_d": float(mfa_score["sync_d"] - natural_score["sync_d"]),
        "mfa_minus_raw_sync_c": float(mfa_score["sync_c"] - raw_score["sync_c"]),
        "mfa_minus_raw_sync_d": float(mfa_score["sync_d"] - raw_score["sync_d"]),
        "phone_pairs": pairs,
    }


def _speaker_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "duration_mae_s",
        "duration_median_abs_s",
        "duration_p90_abs_s",
        "duration_mean_signed_delta_s",
        "duration_mean_relative_abs_delta",
        "duration_median_relative_abs_delta",
        "speech_duration_delta_s",
        "silence_duration_delta_s",
        "speech_match_rate",
        "mfa_minus_natural_sync_c",
        "mfa_minus_natural_sync_d",
        "mfa_minus_raw_sync_c",
        "mfa_minus_raw_sync_d",
    )
    result: dict[str, Any] = {"speaker_id": rows[0]["speaker_id"], "sample_count": len(rows)}
    for metric in metric_names:
        result[metric] = _summary([float(row[metric]) for row in rows])
    result["matched_speech_phone_count"] = _summary([float(row["matched_speech_phone_count"]) for row in rows])
    result["unknown_speech_labels"] = sorted({label for row in rows for label in row["unknown_speech_labels"]})
    return result


def _group_metrics(
    speaker_rows: Mapping[str, Mapping[str, Any]],
    speaker_group: Mapping[str, str],
) -> dict[str, Any]:
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for speaker, row in speaker_rows.items():
        groups[speaker_group[speaker]].append(row)
    metrics = (
        "duration_mae_s",
        "duration_mean_relative_abs_delta",
        "duration_mean_signed_delta_s",
        "speech_duration_delta_s",
        "mfa_minus_natural_sync_c",
        "mfa_minus_natural_sync_d",
    )
    result: dict[str, Any] = {}
    for group, rows in sorted(groups.items()):
        result[group] = {
            "speakers": sorted(row["speaker_id"] for row in rows),
            "speaker_count": len(rows),
            "metrics": {
                metric: _summary([float(row[metric]["mean"]) for row in rows]) for metric in metrics
            },
        }
    if "good" in result and "poor" in result:
        result["poor_minus_good"] = {
            metric: result["poor"]["metrics"][metric]["mean"] - result["good"]["metrics"][metric]["mean"]
            for metric in metrics
        }
    return result


def _validate_inputs(tokens: Mapping[str, Any], scores: Mapping[str, Any]) -> None:
    records = tokens.get("records", {})
    if tokens.get("complete") is not True or len(records) != 25:
        raise ValueError("tokens input must be complete n25")
    if scores.get("status") != "complete" or len(scores.get("scores", [])) != 75:
        raise ValueError("score input must be complete 75-cell n25 matrix")
    speakers = {str(record.get("speaker_id")) for record in records.values()}
    if speakers != set(EXPECTED_SPEAKERS):
        raise ValueError(f"unexpected speaker set: {sorted(speakers)}")
    if "S0770" in speakers:
        raise ValueError("heldout S0770 is forbidden")


def analyze(
    tokens_path: Path,
    scores_path: Path,
    outdir: Path,
    *,
    good_speakers: Sequence[str] = DEFAULT_GOOD,
    poor_speakers: Sequence[str] = DEFAULT_POOR,
    audio_audit_path: Path | None = None,
) -> dict[str, Any]:
    if outdir.exists() and any(outdir.iterdir()):
        raise ValueError(f"refusing non-empty output directory: {outdir}")
    tokens = json.loads(tokens_path.read_text(encoding="utf-8"))
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    _validate_inputs(tokens, scores)
    audio_artifacts, audio_artifacts_sha256 = _load_audio_artifacts(audio_audit_path)
    good = set(good_speakers)
    poor = set(poor_speakers)
    if good & poor or not good or not poor:
        raise ValueError("good and poor speaker groups must be non-empty and disjoint")
    if not (good | poor) <= set(EXPECTED_SPEAKERS):
        raise ValueError("speaker grouping contains unknown speaker")
    speaker_group = {speaker: "good" if speaker in good else "poor" if speaker in poor else "intermediate" for speaker in EXPECTED_SPEAKERS}

    score_by_sample: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for score in scores["scores"]:
        arm = str(score["arm"])
        if arm not in {"natural_raw", "raw_tts", "mfa_linear_resample_poly"}:
            raise ValueError(f"unexpected score arm: {arm}")
        score_by_sample[str(score["sample_id"])][arm] = score
    if any(set(arms) != {"natural_raw", "raw_tts", "mfa_linear_resample_poly"} for arms in score_by_sample.values()):
        raise ValueError("each sample must have all three score arms")

    rows: list[dict[str, Any]] = []
    for sample_id, record in sorted(tokens["records"].items(), key=lambda item: int(item[0])):
        if str(record["sample_id"]) != str(sample_id):
            raise ValueError(f"sample id mismatch for {sample_id}")
        score_arms = score_by_sample[str(sample_id)]
        for arm, score in score_arms.items():
            if str(score["paired_key"]) != str(record["paired_key"]) or str(score["speaker_id"]) != str(record["speaker_id"]):
                raise ValueError(f"score identity mismatch for {sample_id} {arm}")
        row = _sample_metrics(str(sample_id), record, score_arms)
        if audio_artifacts:
            artifact = audio_artifacts.get(str(sample_id))
            if artifact is None:
                raise ValueError(f"audio audit missing sample {sample_id}")
            if str(artifact.get("paired_key")) != str(record["paired_key"]) or str(artifact.get("speaker_id")) != str(record["speaker_id"]):
                raise ValueError(f"audio audit identity mismatch for {sample_id}")
            for source_key, output_key in (
                ("artifact_event_count", "audio_artifact_event_count"),
                ("repetition_count", "audio_repetition_count"),
                ("boundary_event_count", "audio_boundary_event_count"),
                ("artifact_event_duration_s", "audio_artifact_event_duration_s"),
                ("mean_high_band_ratio", "audio_mean_high_band_ratio"),
                ("peak", "audio_peak"),
                ("crest_factor", "audio_crest_factor"),
            ):
                row[output_key] = float(artifact.get(source_key, 0.0) or 0.0)
        row["group"] = speaker_group[str(record["speaker_id"])]
        rows.append(row)

    by_speaker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_speaker[str(row["speaker_id"])].append(row)
    speaker_rows = {speaker: _speaker_metrics(values) for speaker, values in sorted(by_speaker.items())}
    speaker_correlations: dict[str, Any] = {}
    metric_keys = ("duration_mae_s", "duration_mean_relative_abs_delta", "speech_duration_delta_s")
    outcome_keys = ("mfa_minus_natural_sync_c", "mfa_minus_natural_sync_d", "mfa_minus_raw_sync_c", "mfa_minus_raw_sync_d")
    for metric in metric_keys:
        xs = [float(speaker_rows[speaker][metric]["mean"]) for speaker in EXPECTED_SPEAKERS]
        for outcome in outcome_keys:
            ys = [float(speaker_rows[speaker][outcome]["mean"]) for speaker in EXPECTED_SPEAKERS]
            speaker_correlations[f"{metric}_vs_{outcome}"] = {
                "n_speakers": len(EXPECTED_SPEAKERS),
                "pearson_r": _correlation(xs, ys),
                "spearman_r": _correlation(_rank(xs), _rank(ys)),
                "interpretation": "exploratory speaker-level association; not causal",
            }

    audio_correlations: dict[str, Any] = {}
    if audio_artifacts:
        audio_metrics = (
            "audio_artifact_event_count",
            "audio_repetition_count",
            "audio_boundary_event_count",
            "audio_artifact_event_duration_s",
        )
        audio_by_speaker: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in rows:
            audio_by_speaker[str(row["speaker_id"])].append(row)
        for duration_metric in metric_keys:
            duration_values = [float(speaker_rows[speaker][duration_metric]["mean"]) for speaker in EXPECTED_SPEAKERS]
            for audio_metric in audio_metrics:
                audio_values = [
                    statistics.mean(float(row[audio_metric]) for row in audio_by_speaker[speaker])
                    for speaker in EXPECTED_SPEAKERS
                ]
                audio_correlations[f"{duration_metric}_vs_{audio_metric}"] = {
                    "n_speakers": len(EXPECTED_SPEAKERS),
                    "pearson_r": _correlation(duration_values, audio_values),
                    "spearman_r": _correlation(_rank(duration_values), _rank(audio_values)),
                    "interpretation": "exploratory speaker-level association; detector counts are review candidates, not diagnoses",
                }

    result = {
        "schema_version": 1,
        "analysis": "aishell1_n25_phone_duration_gap",
        "scope": "historical strict-token exploratory analysis; not clean MFA-3 n25",
        "inputs": {
            "tokens": str(tokens_path.resolve()),
            "tokens_sha256": sha256_file(tokens_path),
            "scores": str(scores_path.resolve()),
            "scores_sha256": sha256_file(scores_path),
            "audio_audit": str(audio_audit_path.resolve()) if audio_audit_path else None,
            "audio_audit_sha256": audio_artifacts_sha256,
        },
        "cohort": {
            "speakers": list(EXPECTED_SPEAKERS),
            "samples": len(rows),
            "samples_per_speaker": 5,
            "heldout_excluded": True,
            "good_speakers": sorted(good),
            "poor_speakers": sorted(poor),
            "intermediate_speakers": sorted(set(EXPECTED_SPEAKERS) - good - poor),
            "grouping_basis": "predefined from prior n25 speaker-level MFA-linear outcome; no duration values used for grouping",
        },
        "interpretation": {
            "phone_duration_metric": "matched speech-phone TTS minus natural duration; SequenceMatcher aligns ordered labels and excludes silence from duration error",
            "relative_metric_denominator": "natural phone duration",
            "syncnet_role": "exploratory outcome association under single fixed-face protocol",
            "audio_artifact_role": "exploratory association with detector review candidates from the separate audio audit",
            "mfa_status": "full clean MFA-3 n25 tokens were not yet available; historical strict tokens are retained and analyzed without overwrite",
        },
        "speaker_summary": speaker_rows,
        "group_summary": _group_metrics(speaker_rows, speaker_group),
        "speaker_correlations": speaker_correlations,
        "audio_correlations": audio_correlations,
        "rows": [{key: value for key, value in row.items() if key != "phone_pairs"} for row in rows],
    }
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "analysis.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    with (outdir / "per_sample.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    (outdir / "speaker_summary.json").write_text(json.dumps(speaker_rows, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (outdir / "group_summary.json").write_text(json.dumps(result["group_summary"], indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    (outdir / "report.md").write_text(render_report(result), encoding="utf-8")
    return result


def render_report(result: Mapping[str, Any]) -> str:
    cohort = result["cohort"]
    lines = [
        "# AISHELL-1 n25 phone-duration discrepancy audit",
        "",
        "## Scope",
        "",
        "This is a speaker-balanced exploratory analysis of paired natural/TTS MFA phone durations. Speaker groups were fixed from the prior n25 MFA-linear outcome before duration metrics were computed.",
        "",
        f"- Good speakers: `{', '.join(cohort['good_speakers'])}`",
        f"- Poor speakers: `{', '.join(cohort['poor_speakers'])}`",
        f"- Intermediate speaker: `{', '.join(cohort['intermediate_speakers'])}`",
        "- MFA limitation: full clean MFA-3 n25 tokens were not yet available; these are historical strict tokens.",
        "",
        "## Speaker summary",
        "",
        "| Speaker | Group | phone MAE (s) | relative abs gap | signed gap (s) | speech duration gap (s) | MFA C delta | MFA D delta |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for speaker in EXPECTED_SPEAKERS:
        row = result["speaker_summary"][speaker]
        lines.append(
            f"| {speaker} | {('good' if speaker in cohort['good_speakers'] else 'poor' if speaker in cohort['poor_speakers'] else 'intermediate')} | "
            f"{row['duration_mae_s']['mean']:.4f} | {row['duration_mean_relative_abs_delta']['mean']:.4f} | "
            f"{row['duration_mean_signed_delta_s']['mean']:.4f} | {row['speech_duration_delta_s']['mean']:.4f} | "
            f"{row['mfa_minus_natural_sync_c']['mean']:.4f} | {row['mfa_minus_natural_sync_d']['mean']:.4f} |"
        )
    lines.extend(["", "## Group comparison", ""])
    group_summary = result["group_summary"]
    for group in ("good", "intermediate", "poor"):
        if group not in group_summary:
            continue
        metrics = group_summary[group]["metrics"]
        lines.append(
            f"- `{group}` ({', '.join(group_summary[group]['speakers'])}): "
            f"phone MAE `{metrics['duration_mae_s']['mean']:.4f}s`, "
            f"relative gap `{metrics['duration_mean_relative_abs_delta']['mean']:.4f}`, "
            f"speech-duration gap `{metrics['speech_duration_delta_s']['mean']:.4f}s`."
        )
    if "poor_minus_good" in group_summary:
        diff = group_summary["poor_minus_good"]
        lines.extend([
            "",
            "Poor minus good speaker-group difference:",
            f"- phone MAE: `{diff['duration_mae_s']:.4f}s`",
            f"- relative absolute gap: `{diff['duration_mean_relative_abs_delta']:.4f}`",
            f"- speech-duration gap: `{diff['speech_duration_delta_s']:.4f}s`",
        ])
    lines.extend([
        "",
        "## Interpretation",
        "",
        "A larger phone-duration gap would be supported if the poor group consistently exceeded the good group in both absolute and relative per-phone error. The result must remain exploratory because there are only five speakers, each with five utterances, and the alignment is not yet the clean MFA-3 n25 rerun.",
        "",
        "Speaker-level correlations with SyncNet are descriptive only. The fixed-face SyncNet protocol cannot separate speaker acoustic mismatch from face mismatch and does not establish causality.",
        "",
    ])
    if result.get("audio_correlations"):
        lines.extend(["## Duration versus audio-artifact candidates", "", "Correlations use speaker means and are detector associations, not diagnoses:", ""])
        for key, value in result["audio_correlations"].items():
            lines.append(f"- `{key}`: Pearson r={value['pearson_r']}, Spearman r={value['spearman_r']}")
        lines.append("")
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--audio-audit", type=Path)
    parser.add_argument("--good-speakers", nargs="+", default=list(DEFAULT_GOOD))
    parser.add_argument("--poor-speakers", nargs="+", default=list(DEFAULT_POOR))
    args = parser.parse_args(argv)
    result = analyze(
        args.tokens.resolve(),
        args.scores.resolve(),
        args.outdir.resolve(),
        good_speakers=args.good_speakers,
        poor_speakers=args.poor_speakers,
        audio_audit_path=args.audio_audit.resolve() if args.audio_audit else None,
    )
    print(json.dumps({"analysis": result["analysis"], "samples": result["cohort"]["samples"], "outdir": str(args.outdir.resolve())}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
