#!/usr/bin/env python3
"""Extract paired TTS rhythm factors and associate them with valid SyncNet gains.

This is a CPU-only diagnostic. It does not generate audio, train a model, or
read heldout metrics for any choice. MFA tokens are kept on the full timeline,
including silence intervals.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import math
import statistics
import wave
from pathlib import Path
from typing import Any, Iterable, Mapping


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ALIGNMENT_MANIFEST = (
    REPO_ROOT
    / "runs/two_stage_hubert_aishell1_20260810/data_boundary/"
    "aishell1_400_raw_mfa_faster_qwen3_heldout.json"
)
DEFAULT_SYNCNET_SUMMARY = Path("/tmp/wav2lip_valid_full/summary.json")
DEFAULT_OUTPUT = REPO_ROOT / "runs/rhythm_timing/20260812_factor_audit"
LEGACY_ROOT = "/mnt/e/Documents/tts-audio/tts-exp"
SILENCE_LABELS = {"", "sil", "sp", "spn", "<eps>", "h#", "pau", "pause"}
INCLUDED_SPLITS = {"train", "valid"}


class AuditError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(raw_path: str | Path) -> Path:
    raw = str(raw_path)
    candidates = [Path(raw)]
    if raw.startswith(LEGACY_ROOT + "/"):
        candidates.append(REPO_ROOT / raw[len(LEGACY_ROOT) + 1 :])
    elif not Path(raw).is_absolute():
        candidates.append(REPO_ROOT / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"cannot resolve source path: {raw}")


def audio_duration(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            sample_rate = handle.getframerate()
        if sample_rate <= 0:
            raise AuditError(f"invalid sample rate in {path}")
        return float(frames / sample_rate)
    except (wave.Error, EOFError) as exc:
        raise AuditError(f"cannot read WAV metadata: {path}: {exc}") from exc


def finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if finite(value)]
    return statistics.fmean(values) if values else None


def median(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if finite(value)]
    return statistics.median(values) if values else None


def safe_log_ratio(numerator: float, denominator: float) -> float | None:
    if numerator <= 0.0 or denominator <= 0.0:
        return None
    return math.log(numerator / denominator)


def rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        average = (index + end - 1) / 2.0 + 1.0
        for position in order[index:end]:
            result[position] = average
        index = end
    return result


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 3:
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    x_centered = [value - x_mean for value in xs]
    y_centered = [value - y_mean for value in ys]
    denominator = math.sqrt(sum(value * value for value in x_centered) * sum(value * value for value in y_centered))
    if denominator <= 1e-12:
        return None
    return float(sum(x * y for x, y in zip(x_centered, y_centered)) / denominator)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys))


def normalized_label(token: Mapping[str, Any]) -> str:
    return str(token.get("token", token.get("raw_token", ""))).strip().lower()


def is_silence(token: Mapping[str, Any]) -> bool:
    if bool(token.get("is_silence")) or bool(token.get("is_non_speech")):
        return True
    return normalized_label(token) in SILENCE_LABELS


def parse_tokens(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for raw in record.get("tokens", []):
        try:
            start = float(raw["start_s"])
            end = float(raw["end_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditError(f"invalid token in {record.get('utterance_id')}: {raw}") from exc
        if not math.isfinite(start) or not math.isfinite(end) or end <= start:
            continue
        result.append(
            {
                "label": normalized_label(raw),
                "start_s": start,
                "end_s": end,
                "duration_s": end - start,
                "is_silence": is_silence(raw),
            }
        )
    return sorted(result, key=lambda item: (item["start_s"], item["end_s"]))


def token_matches(natural: list[dict[str, Any]], tts: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    natural_phone = [token for token in natural if not token["is_silence"]]
    tts_phone = [token for token in tts if not token["is_silence"]]
    natural_labels = [token["label"] for token in natural_phone]
    tts_labels = [token["label"] for token in tts_phone]
    matcher = difflib.SequenceMatcher(a=natural_labels, b=tts_labels, autojunk=False)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            pairs.append((natural_phone[block.a + offset], tts_phone[block.b + offset]))
    return pairs


def js_divergence(xs: list[float], ys: list[float]) -> float | None:
    if not xs or not ys or len(xs) != len(ys):
        return None
    x_total = sum(xs)
    y_total = sum(ys)
    if x_total <= 0.0 or y_total <= 0.0:
        return None
    px = [value / x_total for value in xs]
    py = [value / y_total for value in ys]
    midpoint = [(x + y) / 2.0 for x, y in zip(px, py)]

    def kl(left: list[float], right: list[float]) -> float:
        return sum(value * math.log(value / other) for value, other in zip(left, right) if value > 0.0 and other > 0.0)

    return float((kl(px, midpoint) + kl(py, midpoint)) / 2.0)


def pause_stats(tokens: list[dict[str, Any]], total_duration: float) -> dict[str, Any]:
    pauses = [token for token in tokens if token["is_silence"]]
    epsilon = 0.01
    leading = [token for token in pauses if token["start_s"] <= epsilon]
    trailing = [token for token in pauses if token["end_s"] >= total_duration - epsilon]
    internal = [token for token in pauses if token not in leading and token not in trailing]
    return {
        "pause_count": len(pauses),
        "internal_pause_count": len(internal),
        "pause_total_s": sum(token["duration_s"] for token in pauses),
        "internal_pause_total_s": sum(token["duration_s"] for token in internal),
        "pause_fraction": sum(token["duration_s"] for token in pauses) / max(total_duration, 1e-9),
        "internal_pause_fraction": sum(token["duration_s"] for token in internal) / max(total_duration, 1e-9),
        "pause_positions": [token["start_s"] / max(total_duration, 1e-9) for token in internal],
        "pause_durations": [token["duration_s"] for token in internal],
    }


def match_pause_events(
    natural_pause: Mapping[str, Any],
    tts_pause: Mapping[str, Any],
) -> list[tuple[float, float, float, float]]:
    natural_positions = list(natural_pause["pause_positions"])
    tts_positions = list(tts_pause["pause_positions"])
    natural_durations = list(natural_pause["pause_durations"])
    tts_durations = list(tts_pause["pause_durations"])
    if not natural_positions or not tts_positions:
        return []
    candidates = sorted(
        (
            abs(natural_position - tts_position),
            natural_index,
            tts_index,
        )
        for natural_index, natural_position in enumerate(natural_positions)
        for tts_index, tts_position in enumerate(tts_positions)
    )
    used_natural: set[int] = set()
    used_tts: set[int] = set()
    matches: list[tuple[float, float, float, float]] = []
    for _, natural_index, tts_index in candidates:
        if natural_index in used_natural or tts_index in used_tts:
            continue
        used_natural.add(natural_index)
        used_tts.add(tts_index)
        matches.append(
            (
                natural_positions[natural_index],
                tts_positions[tts_index],
                natural_durations[natural_index],
                tts_durations[tts_index],
            )
        )
    return matches


def paired_factors(natural_record: Mapping[str, Any], tts_record: Mapping[str, Any]) -> dict[str, Any]:
    natural_tokens = parse_tokens(natural_record)
    tts_tokens = parse_tokens(tts_record)
    if not natural_tokens or not tts_tokens:
        raise AuditError(f"empty token sequence for {natural_record.get('paired_key')}")

    natural_path = resolve_path(str(natural_record["audio_path"]))
    tts_path = resolve_path(str(tts_record["audio_path"]))
    natural_total = audio_duration(natural_path)
    tts_total = audio_duration(tts_path)
    natural_pause = pause_stats(natural_tokens, natural_total)
    tts_pause = pause_stats(tts_tokens, tts_total)
    matches = token_matches(natural_tokens, tts_tokens)
    natural_phones = [token for token in natural_tokens if not token["is_silence"]]
    tts_phones = [token for token in tts_tokens if not token["is_silence"]]
    natural_durations = [pair[0]["duration_s"] for pair in matches]
    tts_durations = [pair[1]["duration_s"] for pair in matches]
    duration_logs = [math.log(max(tts, 1e-6) / max(natural, 1e-6)) for natural, tts in zip(natural_durations, tts_durations)]
    boundary_displacements = []
    boundary_displacements_s = []
    for natural, tts in matches:
        for field in ("start_s", "end_s"):
            natural_norm = natural[field] / max(natural_total, 1e-9)
            tts_norm = tts[field] / max(tts_total, 1e-9)
            boundary_displacements.append(abs(tts_norm - natural_norm))
            boundary_displacements_s.append(abs((tts[field] / max(tts_total, 1e-9)) * natural_total - natural[field]))

    matched_coverage = len(matches) / max(len(natural_phones), len(tts_phones), 1)
    allocation_jsd = js_divergence(natural_durations, tts_durations)
    matched_natural_speech = sum(natural_durations)
    matched_tts_speech = sum(tts_durations)
    pause_pairs = match_pause_events(natural_pause, tts_pause)
    pause_position_deltas = [abs(tts - natural) for natural, tts, _, _ in pause_pairs]
    pause_duration_deltas = [abs(tts - natural) for _, _, natural, tts in pause_pairs]

    return {
        "paired_key": str(natural_record["paired_key"]),
        "sample_id": int(natural_record.get("sample_id", -1)),
        "speaker_id": str(natural_record["speaker_id"]),
        "split": str(natural_record["split"]),
        "natural_audio": str(natural_path),
        "tts_audio": str(tts_path),
        "natural_audio_sha256": str(natural_record.get("source_sha256", "")),
        "tts_audio_sha256": str(tts_record.get("source_sha256", "")),
        "natural_duration_s": natural_total,
        "tts_duration_s": tts_total,
        "log_total_duration_ratio": safe_log_ratio(tts_total, natural_total),
        "natural_speech_duration_s": natural_total - natural_pause["pause_total_s"],
        "tts_speech_duration_s": tts_total - tts_pause["pause_total_s"],
        "natural_non_silence_duration_s": natural_total - natural_pause["pause_total_s"],
        "tts_non_silence_duration_s": tts_total - tts_pause["pause_total_s"],
        "log_non_silence_duration_ratio": safe_log_ratio(
            tts_total - tts_pause["pause_total_s"], natural_total - natural_pause["pause_total_s"]
        ),
        "natural_phone_count": len(natural_phones),
        "tts_phone_count": len(tts_phones),
        "matched_phone_count": len(matches),
        "matched_phone_coverage": matched_coverage,
        "natural_phone_rate": len(natural_phones) / max(natural_total - natural_pause["pause_total_s"], 1e-9),
        "tts_phone_rate": len(tts_phones) / max(tts_total - tts_pause["pause_total_s"], 1e-9),
        "log_phone_rate_ratio": safe_log_ratio(
            len(tts_phones) / max(tts_total - tts_pause["pause_total_s"], 1e-9),
            len(natural_phones) / max(natural_total - natural_pause["pause_total_s"], 1e-9),
        ),
        "natural_pause_count": natural_pause["pause_count"],
        "tts_pause_count": tts_pause["pause_count"],
        "pause_count_delta": tts_pause["pause_count"] - natural_pause["pause_count"],
        "natural_internal_pause_count": natural_pause["internal_pause_count"],
        "tts_internal_pause_count": tts_pause["internal_pause_count"],
        "internal_pause_count_delta": tts_pause["internal_pause_count"] - natural_pause["internal_pause_count"],
        "natural_pause_fraction": natural_pause["pause_fraction"],
        "tts_pause_fraction": tts_pause["pause_fraction"],
        "pause_fraction_delta": tts_pause["pause_fraction"] - natural_pause["pause_fraction"],
        "natural_internal_pause_fraction": natural_pause["internal_pause_fraction"],
        "tts_internal_pause_fraction": tts_pause["internal_pause_fraction"],
        "internal_pause_fraction_delta": tts_pause["internal_pause_fraction"] - natural_pause["internal_pause_fraction"],
        "pause_position_mae_norm": mean(pause_position_deltas),
        "pause_duration_mae_s": mean(pause_duration_deltas),
        "phone_duration_log_ratio_mean": mean(duration_logs),
        "phone_duration_log_ratio_abs_mean": mean(abs(value) for value in duration_logs),
        "phone_duration_mae_s": mean(abs(tts - natural) for natural, tts in zip(natural_durations, tts_durations)),
        "phone_duration_spearman": spearman(natural_durations, tts_durations),
        "phone_duration_allocation_jsd": allocation_jsd,
        "boundary_displacement_mean_norm": mean(boundary_displacements),
        "boundary_displacement_mean_s": mean(boundary_displacements_s),
        "boundary_displacement_max_norm": max(boundary_displacements, default=None),
        "matched_natural_speech_s": matched_natural_speech,
        "matched_tts_speech_s": matched_tts_speech,
    }


def load_pairs(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise AuditError("alignment manifest records must be a list")
    pairs: dict[str, dict[str, Any]] = {}
    for raw in records:
        record = dict(raw)
        split = str(record.get("split", ""))
        if split not in INCLUDED_SPLITS:
            continue
        key = str(record.get("paired_key", ""))
        condition = str(record.get("condition", ""))
        if not key or condition not in {"natural", "tts"}:
            continue
        group = pairs.setdefault(key, {})
        if condition in group:
            raise AuditError(f"duplicate {condition} record for {key}")
        group[condition] = record
    return pairs, payload


def load_sync_deltas(path: Path) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    scores = payload.get("scores", [])
    by_key: dict[str, dict[str, dict[str, float]]] = {}
    for score in scores:
        if score.get("condition") not in {"natural_raw", "tts_raw"}:
            continue
        key = str(score.get("paired_key", ""))
        if not key:
            continue
        by_key.setdefault(key, {})[str(score["condition"])] = {
            "sync_c": float(score["sync_c"]),
            "sync_d": float(score["sync_d"]),
            "av_offset": float(score.get("av_offset", 0)),
        }
    result: dict[str, dict[str, float]] = {}
    for key, conditions in by_key.items():
        natural = conditions.get("natural_raw")
        tts = conditions.get("tts_raw")
        if natural is None or tts is None:
            continue
        result[key] = {
            "natural_sync_c": natural["sync_c"],
            "tts_sync_c": tts["sync_c"],
            "sync_c_delta_tts_minus_natural": tts["sync_c"] - natural["sync_c"],
            "natural_sync_d": natural["sync_d"],
            "tts_sync_d": tts["sync_d"],
            "sync_d_delta_tts_minus_natural": tts["sync_d"] - natural["sync_d"],
            "sync_d_improvement_tts_minus_natural": natural["sync_d"] - tts["sync_d"],
            "natural_av_offset": natural["av_offset"],
            "tts_av_offset": tts["av_offset"],
            "av_offset_delta_tts_minus_natural": tts["av_offset"] - natural["av_offset"],
        }
    return result


def association(records: list[dict[str, Any]], x_name: str, y_name: str) -> dict[str, Any] | None:
    pairs = [(float(record[x_name]), float(record[y_name])) for record in records if finite(record.get(x_name)) and finite(record.get(y_name))]
    if len(pairs) < 3:
        return None
    xs, ys = zip(*pairs)
    return {
        "feature": x_name,
        "target": y_name,
        "n": len(pairs),
        "pearson": pearson(list(xs), list(ys)),
        "spearman": spearman(list(xs), list(ys)),
    }


def write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    summary = result["summary"]
    lines = [
        "# TTS Rhythm Factor Audit",
        "",
        "This CPU-only audit uses natural/TTS MFA token timelines and the existing valid-only SyncNet summary. Heldout S0770 records and metrics are excluded before factor extraction and are not used for any selection.",
        "",
        "## Scope",
        "",
        f"- Pairs analyzed: {summary['pair_count']} ({', '.join(summary['split_counts'])})",
        f"- Valid SyncNet pairs: {summary['valid_sync_pair_count']}",
        f"- Heldout excluded: {result['provenance']['heldout_excluded']}",
        f"- Alignment manifest SHA256: `{result['provenance']['alignment_manifest_sha256']}`",
        "",
        "## Valid associations",
        "",
        "Features are associated with `tts - natural` paired deltas; this is correlational and does not establish causality.",
        "",
        "| Feature | Target | N | Pearson | Spearman |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["valid_associations"]:
        lines.append(f"| `{item['feature']}` | `{item['target']}` | {item['n']} | {item['pearson']:.4f} | {item['spearman']:.4f} |")
    lines.extend(
        [
            "",
            "## Interpretation guardrails",
            "",
            "- These associations only prioritize later counterfactual tests; they do not justify changing the enhancer or choosing a heldout result.",
            "- Full TTS duration transfer remains a negative control because the earlier exact-length duration audit worsened the HuBERT L6 gap as alpha increased.",
            "- The next causal step should use exact-N, no-op, and timing-shuffle controls before adding a timing side-channel.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(alignment_manifest: Path, sync_summary: Path, output_dir: Path) -> dict[str, Any]:
    pairs, manifest_payload = load_pairs(alignment_manifest)
    sync_deltas = load_sync_deltas(sync_summary)
    records: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for key in sorted(pairs):
        group = pairs[key]
        if "natural" not in group or "tts" not in group:
            rejected.append({"paired_key": key, "reason": "missing_natural_or_tts"})
            continue
        try:
            record = paired_factors(group["natural"], group["tts"])
        except (AuditError, FileNotFoundError) as exc:
            rejected.append({"paired_key": key, "reason": str(exc)})
            continue
        if key in sync_deltas:
            record.update(sync_deltas[key])
        records.append(record)

    valid_records = [record for record in records if record["split"] == "valid" and "sync_c_delta_tts_minus_natural" in record]
    feature_names = [
        key
        for key, value in records[0].items()
        if key not in {"paired_key", "sample_id", "speaker_id", "split", "natural_audio", "tts_audio", "natural_audio_sha256", "tts_audio_sha256"}
        and finite(value)
    ] if records else []
    associations: list[dict[str, Any]] = []
    for feature in feature_names:
        if feature.startswith("sync_") or feature.endswith("_sync_c") or feature.endswith("_sync_d") or "av_offset" in feature:
            continue
        item = association(valid_records, feature, "sync_c_delta_tts_minus_natural")
        if item is not None:
            associations.append(item)
    associations.sort(key=lambda item: abs(item["spearman"] or 0.0), reverse=True)

    split_counts: dict[str, int] = {}
    for record in records:
        split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1
    result = {
        "schema_version": 1,
        "provenance": {
            "alignment_manifest": str(alignment_manifest.resolve()),
            "alignment_manifest_sha256": sha256_file(alignment_manifest),
            "syncnet_summary": str(sync_summary.resolve()) if sync_summary.exists() else None,
            "syncnet_summary_sha256": sha256_file(sync_summary) if sync_summary.exists() else None,
            "included_splits": sorted(INCLUDED_SPLITS),
            "heldout_excluded": True,
            "heldout_speaker": "S0770",
            "valid_speaker": "S0765",
            "sample_rate_hz": 16000,
            "frame_hop_samples": 320,
            "factor_type": "MFA_full_timeline_duration_pause_boundary",
        },
        "summary": {
            "pair_count": len(records),
            "split_counts": [f"{split}:{split_counts[split]}" for split in sorted(split_counts)],
            "rejected_count": len(rejected),
            "valid_sync_pair_count": len(valid_records),
            "valid_associations": associations,
        },
        "records": records,
        "rejected": rejected,
        "source_manifest_metadata": {
            "dataset": manifest_payload.get("dataset"),
            "dataset_version": manifest_payload.get("dataset_version"),
            "split_speakers": manifest_payload.get("split_speakers"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rhythm_factors.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    write_markdown(output_dir / "rhythm_factors.md", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment-manifest", type=Path, default=DEFAULT_ALIGNMENT_MANIFEST)
    parser.add_argument("--syncnet-summary", type=Path, default=DEFAULT_SYNCNET_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.alignment_manifest, args.syncnet_summary, args.output_dir)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"output={args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
