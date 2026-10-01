"""Paired q00-versus-q11 native-gap addendum for the sealed native report.

This CPU-only transform leaves the immutable parent report untouched. It
reconstructs the paired speaker contrast (T-N at q00) minus (T-N at q11)
from the parent's per-speaker and shared-permutation summaries, and retains
the original intended denominator when either endpoint is unavailable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


EXPECTED_REPORT_SHA256 = "7b4c8e96155e103bb93a149c55bc97d2bdeb7fa99986bd8d5a36467a38ddd8f9"
EXPECTED_CLIPS = 80
EXPECTED_SPEAKERS = 40
EXPECTED_PERMUTATIONS = 100
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_930
FACES = ("primary", "secondary")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C", "C_anchor", "B", "D", "search_gain")


class AddendumError(ValueError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _paired_bootstrap(values: Mapping[str, float]) -> dict[str, Any] | None:
    speakers = sorted(values)
    if len(speakers) < 2:
        return None
    vector = np.asarray([values[speaker] for speaker in speakers], dtype=np.float64)
    if not np.isfinite(vector).all():
        raise AddendumError("nonfinite paired speaker contrast")
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    draw_indices = rng.integers(0, len(speakers), size=(BOOTSTRAP_DRAWS, len(speakers)))
    draw_means = vector[draw_indices].mean(axis=1, dtype=np.float64)
    return {
        "speaker_count": len(speakers),
        "speakers": speakers,
        "equal_speaker_mean": float(vector.mean(dtype=np.float64)),
        "speaker_bootstrap_99_ci": [
            float(np.quantile(draw_means, 0.005, method="linear")),
            float(np.quantile(draw_means, 0.995, method="linear")),
        ],
        "draws": BOOTSTRAP_DRAWS,
        "seed": BOOTSTRAP_SEED,
        "unit": "speaker",
    }


def _speaker_map(payload: Mapping[str, Any], label: str) -> dict[str, float]:
    values = payload.get("speaker_values")
    if not isinstance(values, dict):
        raise AddendumError(f"{label} does not contain speaker-level values")
    return {str(key): float(value) for key, value in values.items()}


def _contrast_for_pair(
    q00: Mapping[str, Any],
    q11: Mapping[str, Any],
    *,
    expected_speakers: set[str],
    label: str,
) -> dict[str, Any]:
    q00_obs = _speaker_map(q00["observed"], f"{label}/q00/observed")
    q11_obs = _speaker_map(q11["observed"], f"{label}/q11/observed")
    q00_null_mean = _speaker_map(q00["mean_null"], f"{label}/q00/mean_null")
    q11_null_mean = _speaker_map(q11["mean_null"], f"{label}/q11/mean_null")
    q00_excess = _speaker_map(q00["observed_minus_mean_null"], f"{label}/q00/excess")
    q11_excess = _speaker_map(q11["observed_minus_mean_null"], f"{label}/q11/excess")
    speaker_sets = [set(value) for value in (q00_obs, q11_obs, q00_null_mean, q11_null_mean, q00_excess, q11_excess)]
    common = set.intersection(*speaker_sets) if speaker_sets else set()
    if not common:
        raise AddendumError(f"{label} has no paired speakers across q00 and q11")
    if len(expected_speakers) != EXPECTED_SPEAKERS:
        raise AddendumError(f"{label} expected speaker roster does not contain all 40 speakers")

    observed_delta = {speaker: q00_obs[speaker] - q11_obs[speaker] for speaker in sorted(common)}
    null_mean_delta = {speaker: q00_null_mean[speaker] - q11_null_mean[speaker] for speaker in sorted(common)}
    excess_delta = {speaker: q00_excess[speaker] - q11_excess[speaker] for speaker in sorted(common)}
    derived_excess_delta = {speaker: observed_delta[speaker] - null_mean_delta[speaker] for speaker in sorted(common)}
    max_excess_identity_error = max(abs(excess_delta[speaker] - derived_excess_delta[speaker]) for speaker in common)
    if max_excess_identity_error > 2e-8:
        raise AddendumError(f"{label} parent excess values do not preserve paired speaker arithmetic")

    q00_permutation = np.asarray(q00.get("null_equal_speaker_mean_by_permutation", []), dtype=np.float64)
    q11_permutation = np.asarray(q11.get("null_equal_speaker_mean_by_permutation", []), dtype=np.float64)
    if q00_permutation.shape != (EXPECTED_PERMUTATIONS,) or q11_permutation.shape != (EXPECTED_PERMUTATIONS,):
        raise AddendumError(f"{label} does not contain the complete frozen 100-permutation arrays")
    if not np.isfinite(q00_permutation).all() or not np.isfinite(q11_permutation).all():
        raise AddendumError(f"{label} has nonfinite permutation values")
    null_delta_by_permutation = q00_permutation - q11_permutation
    observed_equal_speaker_delta = float(np.mean(list(observed_delta.values()), dtype=np.float64))
    null_equal_speaker_delta = float(np.mean(list(null_mean_delta.values()), dtype=np.float64))
    mean_paired_null = float(np.mean(null_delta_by_permutation, dtype=np.float64))
    if abs(null_equal_speaker_delta - mean_paired_null) > 2e-7:
        raise AddendumError(f"{label} q00/q11 null summaries are not aligned on the same permutation draws")

    observed_bootstrap = _paired_bootstrap(observed_delta)
    null_mean_bootstrap = _paired_bootstrap(null_mean_delta)
    excess_bootstrap = _paired_bootstrap(excess_delta)
    q00_full_support = bool(q00.get("support_eligible_for_full_cohort", False))
    q11_full_support = bool(q11.get("support_eligible_for_full_cohort", False))
    support_eligible = bool(
        q00_full_support and q11_full_support
        and len(common) == EXPECTED_SPEAKERS
        and int(q00.get("null_complete_speaker_count", -1)) == EXPECTED_SPEAKERS
        and int(q11.get("null_complete_speaker_count", -1)) == EXPECTED_SPEAKERS
    )
    return {
        "estimand": "(T-N at q00) - (T-N at q11)",
        "q00_minus_q11_observed_gap_change": observed_bootstrap,
        "q00_minus_q11_mean_null_gap_change": null_mean_bootstrap,
        "q00_minus_q11_observed_minus_mean_null_excess": excess_bootstrap,
        "paired_speaker_values": {
            "observed_gap_change": observed_delta,
            "mean_null_gap_change": null_mean_delta,
            "time_excess": excess_delta,
        },
        "null_gap_change_by_permutation": null_delta_by_permutation.tolist(),
        "null_gap_change_percentile_99": [
            float(np.quantile(null_delta_by_permutation, 0.005, method="linear")),
            float(np.quantile(null_delta_by_permutation, 0.995, method="linear")),
        ],
        "fraction_null_gap_change_ge_observed": float(
            (1 + np.count_nonzero(null_delta_by_permutation >= observed_equal_speaker_delta))
            / (EXPECTED_PERMUTATIONS + 1)
        ),
        "permutation_count": EXPECTED_PERMUTATIONS,
        "same_shared_permutation_order": True,
        "paired_speaker_count": len(common),
        "expected_speaker_count": EXPECTED_SPEAKERS,
        "missing_from_paired_support": sorted(expected_speakers - common),
        "unexpected_paired_speakers": sorted(common - expected_speakers),
        "q00_support_eligible_for_full_cohort": q00_full_support,
        "q11_support_eligible_for_full_cohort": q11_full_support,
        "eligible_for_full_cohort": support_eligible,
        "max_speaker_arithmetic_identity_error": max_excess_identity_error,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
    }


def build_addendum(report_path: str | Path) -> dict[str, Any]:
    source_path = Path(report_path).resolve(strict=True)
    source_sha = _sha256(source_path)
    if source_sha != EXPECTED_REPORT_SHA256:
        raise AddendumError(f"native report SHA differs from frozen parent report: {source_sha}")
    report = json.loads(source_path.read_text(encoding="utf-8"))
    if report.get("status") != "COMPLETE":
        raise AddendumError("native parent report is not complete")
    verification = report.get("verification", {})
    if (verification.get("planned_rows") != 2560
            or verification.get("result_rows") != 2560
            or verification.get("native_fourcell_rows") != 1280
            or verification.get("verified_contexts") != EXPECTED_CLIPS
            or verification.get("verified_face_source_groups") != 320
            or verification.get("fixed_anchor") != {"column": 18, "k": 3, "official_offset": -3}):
        raise AddendumError("native parent report verification counts or anchor differ from the sealed run")
    permutation = report.get("permutation_protocol", {})
    if (permutation.get("draws") != EXPECTED_PERMUTATIONS
            or permutation.get("seed_rule") != "20260930 + ordered_eval_index"
            or not isinstance(permutation.get("domains_and_hashes"), Mapping)):
        raise AddendumError("native parent report does not bind the shared 100-draw permutation design")
    permutation_domains = permutation["domains_and_hashes"]
    if len(permutation_domains) != EXPECTED_CLIPS * 2:
        raise AddendumError("native parent report does not bind both source domains for all 80 clips")
    source_domains: dict[tuple[str, str], Mapping[str, Any]] = {}
    for key, record in permutation_domains.items():
        if not isinstance(record, Mapping):
            raise AddendumError(f"malformed permutation-domain record: {key}")
        sample_id, separator, source = str(key).partition(":")
        if not separator or source not in ("natural", "tts") or record.get("sample_id") != sample_id or record.get("source") != source:
            raise AddendumError(f"permutation-domain identity differs: {key}")
        ordered_index = record.get("ordered_eval_index")
        hashes = record.get("permutation_sha256")
        if (not isinstance(ordered_index, int) or not 0 <= ordered_index < EXPECTED_CLIPS
                or record.get("seed") != BOOTSTRAP_SEED + ordered_index
                or record.get("permutation_count") != EXPECTED_PERMUTATIONS
                or not isinstance(hashes, list) or len(hashes) != EXPECTED_PERMUTATIONS
                or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
                or record.get("same_permutation_for_raw_fixed_audio") is not True
                or record.get("same_permutation_across_faces_and_cells") is not True):
            raise AddendumError(f"permutation-domain seed/count/hash contract differs: {key}")
        domain_key = (sample_id, source)
        if domain_key in source_domains:
            raise AddendumError(f"duplicate permutation-domain record: {key}")
        source_domains[domain_key] = record
    if len({sample for sample, source in source_domains if source == "natural"}) != EXPECTED_CLIPS or len(source_domains) != EXPECTED_CLIPS * 2:
        raise AddendumError("permutation-domain sample coverage differs from the frozen cohort")
    ordered_indices = [int(record["ordered_eval_index"]) for (sample, source), record in source_domains.items() if source == "natural"]
    if sorted(ordered_indices) != list(range(EXPECTED_CLIPS)):
        raise AddendumError("natural permutation domains do not cover each frozen ordered index exactly once")
    for sample_id in {sample for sample, _source in source_domains}:
        natural = source_domains.get((sample_id, "natural"))
        tts = source_domains.get((sample_id, "tts"))
        if natural is None or tts is None or natural.get("ordered_eval_index") != tts.get("ordered_eval_index") or natural.get("seed") != tts.get("seed"):
            raise AddendumError(f"natural/TTS permutation seed pairing differs for {sample_id}")

    domains = {
        "primary_legacy_common_audio_cutoff": report["summaries"],
        "supplement_full_audio_cutoff": report["padding_cutoff_sensitivity"]["supplement_full_audio_cutoff"],
    }
    contrasts: dict[str, Any] = {}
    for domain_name, domain in domains.items():
        face_summary = domain.get("by_face", {})
        if set(face_summary) != set(FACES):
            raise AddendumError(f"{domain_name} has an unexpected face set")
        contrasts[domain_name] = {}
        for face in FACES:
            contrasts[domain_name][face] = {}
            for geometry in GEOMETRIES:
                supports = face_summary[face]["by_geometry"][geometry]["by_support"]
                if set(supports) != set(SUPPORTS):
                    raise AddendumError(f"{domain_name}/{face}/{geometry} has an unexpected support set")
                full_gap = supports["full"].get("native_T_minus_N", {})
                full_q00 = full_gap.get("q00", {}).get("C", {}).get("observed", {})
                full_q11 = full_gap.get("q11", {}).get("C", {}).get("observed", {})
                expected_speakers = set(_speaker_map(full_q00, f"{domain_name}/{face}/{geometry}/full/q00 cohort"))
                full_q11_speakers = set(_speaker_map(full_q11, f"{domain_name}/{face}/{geometry}/full/q11 cohort"))
                if expected_speakers != full_q11_speakers or len(expected_speakers) != EXPECTED_SPEAKERS:
                    raise AddendumError(f"{domain_name}/{face}/{geometry} full-support q00/q11 speaker roster differs")
                contrasts[domain_name][face][geometry] = {}
                for support in SUPPORTS:
                    native_gap = supports[support].get("native_T_minus_N", {})
                    if set(native_gap) != {"q00", "q11"}:
                        raise AddendumError(f"{domain_name}/{face}/{geometry}/{support} lacks q00/q11 native gaps")
                    q00, q11 = native_gap["q00"], native_gap["q11"]
                    if set(q00) != set(METRICS) or set(q11) != set(METRICS):
                        raise AddendumError(f"{domain_name}/{face}/{geometry}/{support} metric grid differs")
                    contrasts[domain_name][face][geometry][support] = {
                        metric: _contrast_for_pair(
                            q00[metric], q11[metric], expected_speakers=expected_speakers,
                            label=f"{domain_name}/{face}/{geometry}/{support}/{metric}",
                        )
                        for metric in METRICS
                    }
    return {
        "schema_version": 1,
        "analysis_type": "independent paired native q00-q11 gap-change and shared-null addendum",
        "status": "COMPLETE",
        "classification": "additive summary transform; parent report and score artifacts unchanged",
        "input_report": {"path": str(source_path), "sha256": source_sha},
        "input_protocol_sha256": report.get("input_seals", {}).get("protocol_sha256"),
        "input_row_manifest_sha256": report.get("input_seals", {}).get("row_manifest_sha256"),
        "estimand": "(T q00 - N q00) - (T q11 - N q11), paired by speaker; null contrast uses the same 100 PCG64 draws",
        "bootstrap": {"generator": "NumPy PCG64", "seed": BOOTSTRAP_SEED, "draws": BOOTSTRAP_DRAWS, "resampling_unit": "speaker"},
        "permutation": {"generator": "NumPy PCG64", "draws": EXPECTED_PERMUTATIONS, "seed_rule": "20260930 + ordered_eval_index", "shared_q00_q11_draws_required": True},
        "intended_denominator": {"clips": EXPECTED_CLIPS, "speakers": EXPECTED_SPEAKERS, "clips_per_speaker": 2},
        "domains": contrasts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = build_addendum(args.report)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": value["status"], "output": str(output), "input_report_sha256": value["input_report"]["sha256"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
