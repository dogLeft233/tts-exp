"""Tiny deterministic tables for implementation checks; never real cohort data."""

from __future__ import annotations

import copy
import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .modeling import CATEGORICAL_BASE_FEATURES, DEFAULT_RELATIVE_FIELDS, _all_feature_columns


def make_tables(
    schema: Mapping[str, Any], addendum: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    assets: list[dict[str, Any]] = []
    features: list[dict[str, Any]] = []
    scores: list[dict[str, Any]] = []
    feature_columns = _all_feature_columns(schema)
    relative_map = addendum.get("feature_variant_mapping", {}).get("relative_silence", DEFAULT_RELATIVE_FIELDS)
    numeric_index = {name: index for index, name in enumerate(feature_columns)}
    pair_specs: list[dict[str, Any]] = []
    for dataset_index, dataset in enumerate(("fixture_A", "fixture_B")):
        for group_index in range(6):
            group = f"{dataset}_source_{group_index:02d}"
            for pair_index in range(2):
                pair_uid = f"{dataset}_{group_index:02d}_{pair_index:02d}"
                pair_specs.append({
                    "dataset": dataset,
                    "dataset_index": dataset_index,
                    "group": group,
                    "group_index": group_index,
                    "pair_index": pair_index,
                    "pair_uid": pair_uid,
                })
    for pair in pair_specs:
        n_value = 0.15 * pair["group_index"] + 0.04 * pair["pair_index"]
        t_value = n_value + 0.12 + 0.015 * pair["dataset_index"]
        arm_values: dict[str, dict[str, Any]] = {}
        for condition, arm_shift in (("natural", 0.0), ("tts", 1.0)):
            uid = f"{pair['pair_uid']}__{condition}"
            digest = hashlib.sha256(uid.encode("utf-8")).hexdigest()
            asset = {
                "audio_uid": uid,
                "pair_uid": pair["pair_uid"],
                "source_group": pair["group"],
                "source_group_basis": "synthetic speaker identity",
                "dataset": pair["dataset"],
                "language": "zh",
                "condition": condition,
                "source_layer": "generator",
                "source_path": f"synthetic://{uid}",
                "location": "synthetic_fixture_only",
                "file_sha256": digest,
                "pcm_sha256": hashlib.sha256((uid + "_pcm").encode("utf-8")).hexdigest(),
                "lineage": json.dumps({"fixture": True, "parent": pair["pair_uid"]}),
                "missing_reason": None,
                "tts_provider": "fixture_provider",
                "tts_version": "fixture_1",
                "tfg": "fixture_tfg",
                "tfg_version": "fixture_tfg_1",
                "run": "synthetic_only",
                "portrait_reference": "fixture_portrait",
                "score_contract": "fixture_contract",
            }
            assets.append(asset)
            row: dict[str, Any] = {
                "audio_uid": uid,
                "pair_uid": pair["pair_uid"],
                "source_group": pair["group"],
                "dataset": pair["dataset"],
                "language": "zh",
                "condition": condition,
                "source_layer": "generator",
                "file_sha256": asset["file_sha256"],
                "pcm_sha256": asset["pcm_sha256"],
                "source_path": asset["source_path"],
                "missing_reasons_json": "{}",
                "tts_provider": "fixture_provider",
                "tfg": "fixture_tfg",
            }
            for column in feature_columns:
                if column in CATEGORICAL_BASE_FEATURES:
                    categories = {
                        "encoding_codec": ("pcm_s16le", "pcm_f32le"),
                        "encoding_container": ("wav", "wav"),
                        "encoding_intermediate_codec": ("none", "fixture_codec"),
                    }
                    row[column] = categories[column][int(arm_shift)]
                else:
                    value = (
                        0.1 * numeric_index[column]
                        + 0.03 * pair["group_index"]
                        + 0.01 * pair["pair_index"]
                        + 0.09 * arm_shift
                        + 0.02 * pair["dataset_index"]
                    )
                    if column.endswith("_hz"):
                        value = 16000.0 + 1000.0 * arm_shift
                    elif column.endswith("_s") and "sample_rate" not in column:
                        value = 0.6 + 0.02 * pair["group_index"] + 0.03 * arm_shift
                    elif column == "silence_fraction":
                        value = 0.08 + 0.01 * pair["group_index"] + 0.02 * arm_shift
                    elif column == "duration_s":
                        value = 1.0 + 0.01 * pair["pair_index"] + 0.05 * arm_shift
                    elif column == "timeline_fps":
                        value = 25.0
                    elif column in {"timeline_audio_sample_count", "timeline_video_frame_count"}:
                        value = 16000.0 if column.endswith("sample_count") else 25.0
                    if column == "band_ratio_5500_7500" and (pair["group_index"] + pair["pair_index"]) % 7 == 0:
                        value = np.nan
                    row[column] = value
            for base, alternate in relative_map.items():
                base_value = row[base]
                if pd.isna(base_value):
                    row[alternate] = np.nan
                else:
                    row[alternate] = float(base_value) + 0.005 * (1 + arm_shift)
            features.append(row)
            arm_values[condition] = row

        n = arm_values["natural"]
        t = arm_values["tts"]
        delta = 0.03 * (float(t["silence_leading_s"]) - float(n["silence_leading_s"]))
        delta += 0.02 * (float(t["band_ratio_250_1000"]) - float(n["band_ratio_250_1000"]))
        delta += 0.002 * (pair["group_index"] - 2.5)
        scores.append({
            "score_uid": f"score_{pair['pair_uid']}",
            "pair_uid": pair["pair_uid"],
            "source_group": pair["group"],
            "dataset": pair["dataset"],
            "language": "zh",
            "tts_provider": "fixture_provider",
            "tts_version": "fixture_1",
            "tfg": "fixture_tfg",
            "tfg_version": "fixture_tfg_1",
            "run": "synthetic_only",
            "score_contract": "fixture_contract",
            "contract_fingerprint": "fixture_contract_fp",
            "feature_layer": "generator",
            "natural_audio_uid": f"{pair['pair_uid']}__natural",
            "tts_audio_uid": f"{pair['pair_uid']}__tts",
            "C_n": 0.42,
            "C_t": 0.42 + delta,
            "delta_c": delta,
            "score_source": "synthetic_fixture_only",
            "score_source_sha256": hashlib.sha256(b"synthetic-score").hexdigest(),
            "missing_reason": None,
            "native_or_unified_pcm_or_replacement": "unified_pcm",
            "scorer_source_version": "fixture",
            "scorer_weight_identity": "fixture",
            "roi_policy": "fixture",
            "window_support": "fixture",
            "min_track": "fixture",
            "fps": "25/1",
            "audio_chain": "fixture",
            "B_n": 0.2,
            "B_t": 0.21,
            "Dmin_n": 3.0,
            "Dmin_t": 2.9,
            "D0_n": 4.0,
            "D0_t": 4.1,
        })
    return pd.DataFrame(assets), pd.DataFrame(features), pd.DataFrame(scores)


def make_test_configuration(
    schema: Mapping[str, Any], addendum: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    test_schema = copy.deepcopy(schema)
    test_addendum = copy.deepcopy(addendum)
    test_schema["analysis"]["permutation_L1"] = "11 synthetic group swaps"
    test_addendum["stable_incremental_within_stratum"]["bootstrap_draws"] = 24
    protocol = {
        "schema_version": "processing_chain_audit_synthetic_test_v1",
        "status": "SYNTHETIC_FIXTURE_ONLY",
        "cohort_binding": {"synthetic_fixture": True},
        "analysis": {"uncertainty": {"bootstrap_draws": 24, "permutation_draws": 11}},
        "feature_variant_mapping": copy.deepcopy(
            test_addendum["feature_variant_mapping"]
        ),
    }
    return test_schema, test_addendum, protocol


def materialize_fixture(output_dir: str | Path, schema: Mapping[str, Any], addendum: Mapping[str, Any]) -> dict[str, Any]:
    """Write only fake rows to a fresh directory for deterministic implementation checks."""
    output = Path(output_dir).expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"synthetic fixture directory must be fresh: {output}")
    output.mkdir(parents=True)
    assets, features, scores = make_tables(schema, addendum)
    assets_path = output / "assets.synthetic.json"
    features_path = output / "features.synthetic.csv"
    scores_path = output / "paired_scores.synthetic.csv"
    assets_path.write_text(
        json.dumps({"fixture_only": True, "assets": assets.to_dict(orient="records")},
                   ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    features.to_csv(features_path, index=False, encoding="utf-8")
    scores.to_csv(scores_path, index=False, encoding="utf-8")
    manifest = {
        "status": "SYNTHETIC_FIXTURE_ONLY_NOT_COHORT_DATA",
        "datasets": 2,
        "source_groups": int(assets["source_group"].nunique()),
        "complete_pairs": int(scores["pair_uid"].nunique()),
        "assets_rows": int(len(assets)),
        "features_rows": int(len(features)),
        "scores_rows": int(len(scores)),
        "files": {},
    }
    for path in (assets_path, features_path, scores_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": digest}
    manifest_path = output / "manifest.synthetic.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Create synthetic-only input tables for implementation checks.")
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--addendum", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    addendum = json.loads(args.addendum.read_text(encoding="utf-8"))
    print(json.dumps(materialize_fixture(args.output_dir, schema, addendum), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
