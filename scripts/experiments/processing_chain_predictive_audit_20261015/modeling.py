from __future__ import annotations

import itertools
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class AuditInputError(ValueError):
    """Raised when a frozen input contract is violated."""


SCHEMA_VERSION = "processing_chain_audit_v1"
DEFAULT_RELATIVE_FIELDS = {
    "silence_leading_s": "silence_leading_s_relative",
    "silence_trailing_s": "silence_trailing_s_relative",
    "silence_fraction": "silence_fraction_relative",
    "energy_active_rms_dbfs": "energy_active_rms_dbfs_relative",
    "duration_active_s": "duration_active_s_relative",
}
CATEGORICAL_BASE_FEATURES = {
    "encoding_codec",
    "encoding_container",
    "encoding_intermediate_codec",
}
L1_VIEW_FAMILIES = {
    "metadata_only": ("sample_rate", "encoding", "timeline"),
    "acoustic_only": ("energy", "silence", "band", "duration"),
    "joint": (
        "energy",
        "silence",
        "sample_rate",
        "encoding",
        "band",
        "duration",
        "timeline",
    ),
}
L2_VIEW_FAMILIES = {
    "metadata_only_plus_context": ("sample_rate", "encoding", "timeline"),
    "acoustic_only_plus_context": ("energy", "silence", "band", "duration"),
    "all_seven_plus_context": (
        "energy",
        "silence",
        "sample_rate",
        "encoding",
        "band",
        "duration",
        "timeline",
    ),
}
L2_CONTEXT = ("dataset", "tfg", "tts_provider")
L1_PAIR_KEYS = ("dataset", "pair_uid", "source_layer")
MISSING_CATEGORY = "__AUDIT_MISSING__"


@dataclass(frozen=True)
class AuditConfig:
    seed: int
    min_groups: int
    min_pairs: int
    max_folds: int
    bootstrap_draws: int
    permutation_draws: int
    ordinary_coverage: float
    adjusted_coverage: float
    relative_fields: Mapping[str, str]
    cpu_threads: int = 2


class FoldTransformer:
    """Small train-only imputer/standardizer/one-hot encoder for one fold."""

    def __init__(self, numeric_columns: Sequence[str], categorical_columns: Sequence[str]):
        self.numeric_columns = tuple(dict.fromkeys(numeric_columns))
        self.categorical_columns = tuple(dict.fromkeys(categorical_columns))
        self.numeric_fit: dict[str, tuple[float, float]] = {}
        self.category_levels: dict[str, tuple[str, ...]] = {}
        self.numeric_indicators: list[str] = []
        self.output_names: list[str] = []
        self.diagnostics: dict[str, Any] = {
            "all_missing_training_columns": [],
            "constant_training_columns": [],
            "unseen_test_categories": {},
        }

    @staticmethod
    def _numeric(series: pd.Series) -> np.ndarray:
        values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
        values[~np.isfinite(values)] = np.nan
        return values

    @staticmethod
    def _category(value: Any) -> str:
        if pd.isna(value):
            return MISSING_CATEGORY
        return str(value)

    def fit(self, frame: pd.DataFrame) -> "FoldTransformer":
        names: list[str] = []
        for column in self.numeric_columns:
            values = self._numeric(frame[column])
            finite = values[np.isfinite(values)]
            if finite.size == 0:
                self.diagnostics["all_missing_training_columns"].append(column)
                continue
            median = float(np.median(finite))
            imputed = np.where(np.isfinite(values), values, median)
            mean = float(np.mean(imputed))
            scale = float(np.std(imputed, ddof=0))
            self.numeric_fit[column] = (median, scale if scale > 1e-12 else 1.0)
            if scale <= 1e-12:
                self.diagnostics["constant_training_columns"].append(column)
            else:
                names.append(column)
            indicator = f"{column}__missing"
            self.numeric_indicators.append(column)
            names.append(indicator)
            if not np.any(~np.isfinite(values)):
                self.diagnostics["constant_training_columns"].append(indicator)

        for column in self.categorical_columns:
            values = [self._category(value) for value in frame[column].tolist()]
            levels = tuple(sorted(set(values)))
            if not levels or levels == (MISSING_CATEGORY,):
                self.diagnostics["all_missing_training_columns"].append(column)
                continue
            self.category_levels[column] = levels
            if len(levels) <= 1:
                self.diagnostics["constant_training_columns"].append(column)
                continue
            for level in levels[1:]:
                names.append(f"{column}=={level}")
        self.output_names = names
        return self

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        blocks: list[np.ndarray] = []
        for column in self.numeric_columns:
            if column not in self.numeric_fit:
                continue
            median, scale = self.numeric_fit[column]
            values = self._numeric(frame[column])
            missing = ~np.isfinite(values)
            imputed = np.where(missing, median, values)
            # The train mean is recovered from a stored attribute below; no test statistics enter.
            mean = self._means[column]
            train_std = self._stds[column]
            if train_std > 1e-12:
                blocks.append(((imputed - mean) / train_std).reshape(-1, 1))
            blocks.append(missing.astype(float).reshape(-1, 1))

        for column, levels in self.category_levels.items():
            values = [self._category(value) for value in frame[column].tolist()]
            allowed = set(levels)
            unseen = sorted(set(values) - allowed)
            if unseen:
                self.diagnostics["unseen_test_categories"][column] = unseen
            baseline = levels[0]
            for level in levels[1:]:
                blocks.append(np.asarray([value == level for value in values], dtype=float).reshape(-1, 1))
        if not blocks:
            return np.zeros((len(frame), 0), dtype=float)
        result = np.concatenate(blocks, axis=1)
        if result.shape[1] != len(self.output_names):
            raise AuditInputError(
                f"preprocessor output-name mismatch: {result.shape[1]} values vs {len(self.output_names)} names"
            )
        return result

    def fit_transform_pair(
        self, train: pd.DataFrame, test: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray]:
        # Keep fold statistics explicitly so transform never derives them from heldout rows.
        self._means: dict[str, float] = {}
        self._stds: dict[str, float] = {}
        for column in self.numeric_columns:
            values = self._numeric(train[column])
            finite = values[np.isfinite(values)]
            if finite.size:
                median = float(np.median(finite))
                imputed = np.where(np.isfinite(values), values, median)
                self._means[column] = float(np.mean(imputed))
                std = float(np.std(imputed, ddof=0))
                self._stds[column] = std
        self.fit(train)
        return self.transform(train), self.transform(test)


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as stream:
        return json.load(stream)


def load_input_tables(
    assets_json: str, features_csv: str, paired_scores_csv: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    payload = load_json(assets_json)
    if not isinstance(payload, dict) or not isinstance(payload.get("assets"), list):
        raise AuditInputError("assets JSON must contain an 'assets' list")
    return (
        pd.DataFrame(payload["assets"]),
        pd.read_csv(features_csv, dtype={"audio_uid": str}),
        pd.read_csv(paired_scores_csv, dtype={"score_uid": str}),
    )


def _family_columns(schema: Mapping[str, Any]) -> dict[str, list[str]]:
    families = schema["families"]
    result: dict[str, list[str]] = {}
    for name, spec in families.items():
        result[name] = list(spec.get("acoustic", ())) + list(spec.get("metadata", ()))
    return result


def _all_feature_columns(schema: Mapping[str, Any]) -> list[str]:
    return list(dict.fromkeys(column for columns in _family_columns(schema).values() for column in columns))


def _required_table_columns(schema: Mapping[str, Any]) -> tuple[set[str], set[str], set[str]]:
    tables = schema["tables"]
    return (
        set(tables["assets_json"]["required"]),
        set(tables["features_csv"]["required_metadata"]) | set(_all_feature_columns(schema)),
        set(tables["paired_scores_csv"]["required"]),
    )


def validate_input_tables(
    schema: Mapping[str, Any],
    assets: pd.DataFrame,
    features: pd.DataFrame,
    scores: pd.DataFrame,
    relative_fields: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if schema.get("schema_version") != SCHEMA_VERSION:
        raise AuditInputError(f"unsupported schema_version: {schema.get('schema_version')!r}")
    required_assets, required_features, required_scores = _required_table_columns(schema)
    # Relative-silence measurements are a frozen sensitivity requirement for every
    # decoded waveform. A producer omission is a schema error, even if the base
    # columns happen to be present.
    required_features |= set((relative_fields or DEFAULT_RELATIVE_FIELDS).values())
    errors: list[str] = []
    for label, frame, required in (
        ("assets", assets, required_assets),
        ("features", features, required_features),
        ("paired_scores", scores, required_scores),
    ):
        missing = sorted(required - set(frame.columns))
        if missing:
            errors.append(f"{label} missing columns: {missing}")
    if errors:
        raise AuditInputError("; ".join(errors))
    if assets["audio_uid"].isna().any() or assets["audio_uid"].duplicated().any():
        raise AuditInputError("assets audio_uid must be non-null and unique")
    if features["audio_uid"].isna().any() or features["audio_uid"].duplicated().any():
        raise AuditInputError("features audio_uid must be non-null and unique")
    if scores["score_uid"].isna().any() or scores["score_uid"].duplicated().any():
        raise AuditInputError("paired score_uid must be non-null and unique")
    asset_uids = set(assets["audio_uid"].astype(str))
    feature_uids = set(features["audio_uid"].astype(str))
    if asset_uids != feature_uids:
        missing = sorted(asset_uids - feature_uids)[:10]
        extra = sorted(feature_uids - asset_uids)[:10]
        raise AuditInputError(
            f"asset/feature rows must align one-to-one; missing feature rows={missing}, extra feature rows={extra}"
        )
    for col in ("pair_uid", "source_group", "dataset", "language", "condition", "source_layer"):
        if assets[col].isna().any():
            errors.append(f"assets.{col} contains null values")
    if not set(assets["condition"].dropna().astype(str)).issubset({"natural", "tts"}):
        errors.append("assets condition contains labels outside natural/tts")
    if not set(assets["source_layer"].dropna().astype(str)).issubset(
        {"original", "generator", "mux", "scorer"}
    ):
        errors.append("assets source_layer contains an unknown layer")
    for _, row in assets.iterrows():
        if pd.isna(row["file_sha256"]) or pd.isna(row["pcm_sha256"]):
            reason = "" if pd.isna(row["missing_reason"]) else str(row["missing_reason"]).strip()
            if not reason:
                errors.append(f"asset {row['audio_uid']} has a null hash without missing_reason")
    if errors:
        raise AuditInputError("; ".join(errors))

    score_math_checked = 0
    score_math_mismatch = []
    for _, row in scores.iterrows():
        values = [row.get("C_n"), row.get("C_t"), row.get("delta_c")]
        if all(pd.notna(value) for value in values):
            expected = float(row["C_t"]) - float(row["C_n"])
            score_math_checked += 1
            if not math.isclose(float(row["delta_c"]), expected, rel_tol=0.0, abs_tol=1e-9):
                score_math_mismatch.append(str(row["score_uid"]))
    if score_math_mismatch:
        raise AuditInputError(
            f"delta_c != C_t-C_n for {len(score_math_mismatch)} rows; first={score_math_mismatch[:5]}"
        )
    return {
        "status": "PASS",
        "assets_rows": int(len(assets)),
        "feature_rows": int(len(features)),
        "paired_score_rows": int(len(scores)),
        "assets_with_pcm_hash": int(assets["pcm_sha256"].notna().sum()),
        "assets_missing_pcm_hash": int(assets["pcm_sha256"].isna().sum()),
        "score_arithmetic_rows_checked": int(score_math_checked),
        "score_arithmetic_mismatches": 0,
    }


def _pcm_label_collisions(assets: pd.DataFrame) -> list[dict[str, Any]]:
    valid = assets.dropna(subset=["pcm_sha256"])
    collisions: list[dict[str, Any]] = []
    for pcm, group in valid.groupby("pcm_sha256", sort=True):
        labels = sorted(set(group["condition"].astype(str)))
        if len(labels) > 1:
            collisions.append(
                {
                    "pcm_sha256": str(pcm),
                    "conditions": labels,
                    "audio_uids": sorted(group["audio_uid"].astype(str).unique().tolist()),
                    "source_groups": sorted(group["source_group"].astype(str).unique().tolist()),
                }
            )
    return collisions


def _pcm_source_group_collisions(assets: pd.DataFrame) -> list[dict[str, Any]]:
    valid = assets.dropna(subset=["pcm_sha256", "source_group"])
    collisions: list[dict[str, Any]] = []
    for pcm, group in valid.groupby("pcm_sha256", sort=True):
        source_groups = sorted(set(group["source_group"].astype(str)))
        if len(source_groups) > 1:
            collisions.append(
                {
                    "pcm_sha256": str(pcm),
                    "source_groups": source_groups,
                    "conditions": sorted(set(group["condition"].astype(str))),
                    "audio_uids": sorted(group["audio_uid"].astype(str).unique().tolist()),
                }
            )
    return collisions


def join_assets_features(
    schema: Mapping[str, Any], assets: pd.DataFrame, features: pd.DataFrame
) -> pd.DataFrame:
    key_columns = (
        "audio_uid",
        "pair_uid",
        "source_group",
        "dataset",
        "language",
        "condition",
        "source_layer",
        "file_sha256",
        "pcm_sha256",
        "source_path",
    )
    missing = [column for column in key_columns if column not in features.columns]
    if missing:
        raise AuditInputError(f"feature table missing required join metadata: {missing}")
    merged = assets.merge(
        features,
        on="audio_uid",
        how="left",
        suffixes=("_asset", ""),
        validate="one_to_one",
        indicator=True,
    )
    joined_rows: list[dict[str, Any]] = []
    mismatches: list[str] = []
    feature_columns = _all_feature_columns(schema)
    extra_columns = [
        column
        for column in features.columns
        if column not in set(key_columns) | set(feature_columns)
    ]
    for _, row in merged.iterrows():
        if row["_merge"] != "both":
            continue
        record: dict[str, Any] = {}
        for column in key_columns:
            asset_value = row.get(f"{column}_asset", row.get(column))
            feature_value = row.get(column)
            if pd.notna(asset_value) and pd.notna(feature_value) and str(asset_value) != str(feature_value):
                mismatches.append(f"{row['audio_uid']}:{column}")
            record[column] = asset_value if pd.notna(asset_value) else feature_value
        for column in feature_columns + extra_columns:
            record[column] = row.get(column)
        # Retain context only for audit output; it is never selected as an acoustic predictor.
        for column in ("tts_provider", "tts_version", "tfg", "tfg_version", "run", "portrait_reference", "score_contract"):
            if column in assets.columns:
                record[column] = row.get(column)
        joined_rows.append(record)
    if mismatches:
        raise AuditInputError(f"asset/feature identity mismatch: {mismatches[:10]}")
    return pd.DataFrame(joined_rows)


def make_group_fold_map(group_ids: Iterable[Any], max_folds: int = 5) -> dict[str, int]:
    groups = sorted({str(group) for group in group_ids if pd.notna(group)})
    if len(groups) < 2:
        return {group: 0 for group in groups}
    n_splits = min(max_folds, len(groups))
    splitter = GroupKFold(n_splits=n_splits)
    x = np.zeros((len(groups), 1), dtype=float)
    encoded = np.asarray(groups, dtype=object)
    result: dict[str, int] = {}
    for fold, (_, test_indices) in enumerate(splitter.split(x, groups=encoded)):
        for index in test_indices:
            result[groups[int(index)]] = fold
    if len(result) != len(groups):
        raise AuditInputError("failed to assign every source group to a deterministic fold")
    return result


def _complete_l1_pairs(joined: pd.DataFrame, layer: str) -> tuple[pd.DataFrame, int]:
    frame = joined.loc[
        (joined["source_layer"].astype(str) == str(layer))
        & joined["condition"].isin(["natural", "tts"])
    ].copy()
    frame = frame.dropna(subset=["file_sha256", "pcm_sha256", "source_group", "pair_uid"])
    keys = list(L1_PAIR_KEYS)
    valid_pairs: list[tuple[str, str, str]] = []
    for values, group in frame.groupby(keys, dropna=False, sort=True):
        if set(group["condition"].astype(str)) == {"natural", "tts"}:
            if group["source_group"].astype(str).nunique() != 1:
                raise AuditInputError(f"pair crosses source groups: {values}")
            valid_pairs.append(tuple(map(str, values)))
    if not valid_pairs:
        return frame.iloc[0:0].copy(), 0
    valid_keys = set(valid_pairs)
    key_tuples = list(zip(*(frame[column].astype(str) for column in keys)))
    complete = frame.loc[[key in valid_keys for key in key_tuples]].copy()
    pair_count = len(valid_keys)
    # PCM identity wins over container bytes; keep one deterministic file hash as provenance.
    complete = complete.sort_values(["source_group", "condition", "pcm_sha256", "file_sha256", "audio_uid"])
    complete = complete.drop_duplicates(
        subset=["source_group", "condition", "pcm_sha256"], keep="first"
    ).reset_index(drop=True)
    return complete, pair_count


def _l1_sample_weights(frame: pd.DataFrame, y: np.ndarray | None = None) -> np.ndarray:
    labels = (
        np.asarray([1 if str(value) == "tts" else 0 for value in frame["condition"]], dtype=int)
        if y is None
        else np.asarray(y, dtype=int)
    )
    groups = frame["source_group"].astype(str).to_numpy()
    weights = np.zeros(len(frame), dtype=float)
    total_rows = len(frame)
    for label in (0, 1):
        label_mask = labels == label
        label_groups = sorted(set(groups[label_mask]))
        if not label_groups:
            continue
        for group_id in label_groups:
            mask = label_mask & (groups == group_id)
            count = int(mask.sum())
            if count:
                weights[mask] = total_rows / (2.0 * len(label_groups) * count)
    return weights


def _l2_sample_weights(frame: pd.DataFrame) -> np.ndarray:
    groups = frame["source_group"].astype(str).to_numpy()
    pairs = frame["pair_uid"].astype(str).to_numpy()
    unique_groups = sorted(set(groups))
    weights = np.zeros(len(frame), dtype=float)
    for group_id in unique_groups:
        group_mask = groups == group_id
        group_pairs = sorted(set(pairs[group_mask]))
        if not group_pairs:
            continue
        for pair in group_pairs:
            mask = group_mask & (pairs == pair)
            count = int(mask.sum())
            if count:
                weights[mask] = len(frame) / (len(unique_groups) * len(group_pairs) * count)
    return weights


def _family_column_types(
    columns: Sequence[str], available: Iterable[str]
) -> tuple[list[str], list[str]]:
    available_set = set(available)
    selected = [column for column in columns if column in available_set]
    categorical = [column for column in selected if column in CATEGORICAL_BASE_FEATURES]
    numeric = [column for column in selected if column not in CATEGORICAL_BASE_FEATURES]
    return numeric, categorical


def _relative_map(protocol: Mapping[str, Any] | None) -> dict[str, str]:
    if not protocol:
        return dict(DEFAULT_RELATIVE_FIELDS)
    mapping = protocol.get("feature_variant_mapping", {}).get("relative_silence")
    if mapping is None:
        mapping = protocol.get("analysis", {}).get("feature_variant_mapping", {}).get(
            "relative_silence"
        )
    return dict(mapping or DEFAULT_RELATIVE_FIELDS)


def _l1_columns(
    schema: Mapping[str, Any],
    frame: pd.DataFrame,
    view: str,
    drop_family: str | None = None,
    relative: bool = False,
    protocol: Mapping[str, Any] | None = None,
) -> tuple[list[str], list[str], list[str]]:
    families = _family_columns(schema)
    if view not in L1_VIEW_FAMILIES:
        raise AuditInputError(f"unknown L1 view {view}")
    selected_families = list(L1_VIEW_FAMILIES[view])
    if drop_family is not None:
        selected_families = [name for name in selected_families if name != drop_family]
    columns = [column for family in selected_families for column in families[family]]
    if relative:
        mapping = _relative_map(protocol)
        columns = [mapping.get(column, column) for column in columns]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise AuditInputError(f"required L1 predictor columns missing: {missing}")
    numeric, categorical = _family_column_types(columns, frame.columns)
    return columns, numeric, categorical


def _l2_base_numeric_columns(schema: Mapping[str, Any], families: Sequence[str]) -> list[str]:
    specs = _family_columns(schema)
    return [
        column
        for family in families
        for column in specs[family]
        if column not in CATEGORICAL_BASE_FEATURES
    ]


def _l2_x(
    schema: Mapping[str, Any],
    frame: pd.DataFrame,
    view: str,
    drop_family: str | None = None,
    relative: bool = False,
    protocol: Mapping[str, Any] | None = None,
    context_only: bool = False,
) -> tuple[pd.DataFrame, list[str], list[str], list[str]]:
    families = _family_columns(schema)
    if context_only:
        selected_families: list[str] = []
    else:
        if view not in L2_VIEW_FAMILIES:
            raise AuditInputError(f"unknown L2 view {view}")
        selected_families = list(L2_VIEW_FAMILIES[view])
        if drop_family is not None:
            selected_families = [name for name in selected_families if name != drop_family]
    mapping = _relative_map(protocol) if relative else {}
    result: dict[str, Any] = {}
    numeric_names: list[str] = []
    categorical_names: list[str] = []
    required_families = selected_families
    for family in required_families:
        for base in families[family]:
            source = mapping.get(base, base)
            if base in CATEGORICAL_BASE_FEATURES:
                if family != "encoding":
                    continue
                natural_name, tts_name = f"natural__{source}", f"tts__{source}"
                if natural_name not in frame.columns or tts_name not in frame.columns:
                    raise AuditInputError(f"paired L2 inputs missing {natural_name}/{tts_name}")
                natural = frame[natural_name].astype(object)
                tts = frame[tts_name].astype(object)
                n_name, t_name = f"{base}__natural", f"{base}__tts"
                result[n_name] = natural
                result[t_name] = tts
                categorical_names.extend((n_name, t_name))
                continue
            natural_name, tts_name = f"natural__{source}", f"tts__{source}"
            if natural_name not in frame.columns or tts_name not in frame.columns:
                raise AuditInputError(f"paired L2 inputs missing {natural_name}/{tts_name}")
            n = pd.to_numeric(frame[natural_name], errors="coerce").to_numpy(dtype=float)
            t = pd.to_numeric(frame[tts_name], errors="coerce").to_numpy(dtype=float)
            n[~np.isfinite(n)] = np.nan
            t[~np.isfinite(t)] = np.nan
            d_name, m_name = f"{base}__T_minus_N", f"{base}__mean"
            delta = t - n
            mean = (t + n) / 2.0
            delta[~(np.isfinite(t) & np.isfinite(n))] = np.nan
            mean[~(np.isfinite(t) & np.isfinite(n))] = np.nan
            result[d_name] = delta
            result[m_name] = mean
            numeric_names.extend((d_name, m_name))
            if base in {"energy_rms_dbfs", "energy_active_rms_dbfs"}:
                ratio_name = f"{base}__log_T_over_N"
                ratio = (t - n) * (math.log(10.0) / 20.0)
                ratio[~(np.isfinite(t) & np.isfinite(n))] = np.nan
                result[ratio_name] = ratio
                numeric_names.append(ratio_name)
            elif (
                base in {"duration_s", "duration_active_s", "sample_rate_source_hz", "sample_rate_generator_hz",
                         "sample_rate_mux_hz", "sample_rate_scorer_hz",
                         "encoding_bitrate_bps", "encoding_intermediate_bitrate_bps"}
            ):
                ratio_name = f"{base}__log_T_over_N"
                valid = np.isfinite(n) & np.isfinite(t) & (n > 0.0) & (t > 0.0)
                ratio = np.full(len(frame), np.nan, dtype=float)
                ratio[valid] = np.log(t[valid] / n[valid])
                result[ratio_name] = ratio
                numeric_names.append(ratio_name)
    for context in L2_CONTEXT:
        result[context] = frame[context].astype(object)
        categorical_names.append(context)
    x = pd.DataFrame(result, index=frame.index)
    return x, numeric_names, categorical_names, required_families


def build_l2_frame(
    schema: Mapping[str, Any],
    scores: pd.DataFrame,
    joined_assets: pd.DataFrame,
    score_contract: str,
    contract_fingerprint: str,
    feature_layer: str,
    relative_fields: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    feature_cols = list(dict.fromkeys(
        _all_feature_columns(schema)
        + list((relative_fields or DEFAULT_RELATIVE_FIELDS).values())
    ))
    asset_index = joined_assets.set_index("audio_uid", drop=False)
    frame = scores.loc[
        (scores["score_contract"].astype(str) == str(score_contract))
        & (scores["contract_fingerprint"].astype(str) == str(contract_fingerprint))
        & (scores["feature_layer"].astype(str) == str(feature_layer))
    ].copy()
    valid_rows: list[dict[str, Any]] = []
    excluded: Counter[str] = Counter()
    for _, score in frame.iterrows():
        required_values = [score.get("C_n"), score.get("C_t"), score.get("delta_c")]
        if not all(pd.notna(value) for value in required_values):
            excluded["missing_score_arm_or_delta"] += 1
            continue
        natural_id, tts_id = str(score["natural_audio_uid"]), str(score["tts_audio_uid"])
        if natural_id not in asset_index.index or tts_id not in asset_index.index:
            excluded["audio_uid_not_in_assets"] += 1
            continue
        natural = asset_index.loc[natural_id]
        tts = asset_index.loc[tts_id]
        if isinstance(natural, pd.DataFrame) or isinstance(tts, pd.DataFrame):
            raise AuditInputError("joined asset audio_uid is not unique")
        if str(natural["condition"]) != "natural" or str(tts["condition"]) != "tts":
            excluded["arm_identity_mismatch"] += 1
            continue
        if str(natural["source_layer"]) != str(feature_layer) or str(tts["source_layer"]) != str(feature_layer):
            excluded["feature_layer_mismatch"] += 1
            continue
        if str(natural["pair_uid"]) != str(score["pair_uid"]) or str(tts["pair_uid"]) != str(score["pair_uid"]):
            excluded["pair_uid_mismatch"] += 1
            continue
        if (
            str(natural["source_group"]) != str(score["source_group"])
            or str(tts["source_group"]) != str(score["source_group"])
        ):
            excluded["source_group_mismatch"] += 1
            continue
        if pd.isna(natural["pcm_sha256"]) or pd.isna(tts["pcm_sha256"]):
            excluded["unknown_body_identity"] += 1
            continue
        delta = float(score["C_t"]) - float(score["C_n"])
        if not math.isclose(float(score["delta_c"]), delta, rel_tol=0.0, abs_tol=1e-9):
            raise AuditInputError(f"score arithmetic mismatch at {score['score_uid']}")
        row = {
            "score_uid": str(score["score_uid"]),
            "pair_uid": str(score["pair_uid"]),
            "source_group": str(score["source_group"]),
            "dataset": str(score["dataset"]),
            "language": str(score["language"]),
            "tts_provider": str(score["tts_provider"]),
            "tts_version": score["tts_version"],
            "tfg": str(score["tfg"]),
            "tfg_version": score["tfg_version"],
            "run": score["run"],
            "score_contract": str(score["score_contract"]),
            "contract_fingerprint": str(score["contract_fingerprint"]),
            "feature_layer": str(score["feature_layer"]),
            "natural_audio_uid": natural_id,
            "tts_audio_uid": tts_id,
            "C_n": float(score["C_n"]),
            "C_t": float(score["C_t"]),
            "delta_c": delta,
            "natural_file_sha256": natural["file_sha256"],
            "tts_file_sha256": tts["file_sha256"],
            "natural_pcm_sha256": natural["pcm_sha256"],
            "tts_pcm_sha256": tts["pcm_sha256"],
        }
        for auxiliary in ("B_n", "B_t", "Dmin_n", "Dmin_t", "D0_n", "D0_t"):
            row[auxiliary] = score.get(auxiliary)
        for column in feature_cols:
            row[f"natural__{column}"] = natural[column]
            row[f"tts__{column}"] = tts[column]
        valid_rows.append(row)
    result = pd.DataFrame(valid_rows)
    if result.empty:
        return result, {"input_score_rows": int(len(frame)), "eligible_score_rows": 0, "excluded": dict(excluded)}
    # Same pair/contract/context repetitions are retained and receive equal pair-level weight.
    if result["source_group"].isna().any():
        raise AuditInputError("eligible L2 rows contain a null source_group")
    summary = {
        "input_score_rows": int(len(frame)),
        "eligible_score_rows": int(len(result)),
        "excluded": dict(excluded),
        "groups": int(result["source_group"].nunique()),
        "pairs": int(result["pair_uid"].nunique()),
        "same_pcm_hash_both_conditions": _pcm_label_collisions(
            joined_assets.loc[joined_assets["source_layer"].astype(str) == str(feature_layer)]
        ),
    }
    return result, summary


def _fold_designs(
    raw_x: pd.DataFrame,
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    groups: Sequence[Any],
    fold_map: Mapping[str, int],
    diagnose_collinearity: bool = False,
) -> list[dict[str, Any]]:
    group_array = np.asarray([str(group) for group in groups], dtype=object)
    fold_ids = np.asarray([fold_map[str(group)] for group in group_array], dtype=int)
    result: list[dict[str, Any]] = []
    for fold in sorted(set(fold_ids.tolist())):
        test = np.flatnonzero(fold_ids == fold)
        train = np.flatnonzero(fold_ids != fold)
        if len(test) == 0:
            continue
        if len(train) == 0:
            raise AuditInputError("outer fold has no training rows")
        transformer = FoldTransformer(numeric_columns, categorical_columns)
        x_train, x_test = transformer.fit_transform_pair(raw_x.iloc[train], raw_x.iloc[test])
        diagnostics = dict(transformer.diagnostics)
        if diagnose_collinearity:
            diagnostics["training_design_rank"] = int(np.linalg.matrix_rank(x_train)) if x_train.size else 0
            diagnostics["training_design_columns"] = int(x_train.shape[1])
            raw_numeric = raw_x.iloc[train][list(numeric_columns)]
            corr = raw_numeric.corr(method="spearman", min_periods=3)
            high_pairs: list[dict[str, Any]] = []
            for i, left in enumerate(corr.columns):
                for right in corr.columns[i + 1 :]:
                    value = corr.loc[left, right]
                    if pd.notna(value) and abs(float(value)) >= 0.95:
                        high_pairs.append({"left": str(left), "right": str(right), "spearman": float(value)})
            diagnostics["high_abs_spearman_pairs_ge_0.95"] = high_pairs
        result.append(
            {
                "fold": int(fold),
                "train": train,
                "test": test,
                "x_train": x_train,
                "x_test": x_test,
                "feature_names": transformer.output_names,
                "preprocess_diagnostics": diagnostics,
            }
        )
    return result


def _fit_oof(
    designs: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    sample_weight: np.ndarray,
    groups: Sequence[Any],
    model_kind: str,
    seed: int,
) -> dict[str, Any]:
    n = len(y)
    predictions = np.full(n, np.nan, dtype=float)
    fold_output: list[dict[str, Any]] = []
    for design in designs:
        train, test = design["train"], design["test"]
        x_train = np.asarray(design["x_train"])
        if x_train.ndim == 2 and x_train.shape[1] == 0:
            if model_kind in {"logistic", "tree_classifier"}:
                if len(np.unique(y[train])) < 2:
                    raise AuditInputError(f"fold {design['fold']} training data has one L1 class")
                predictions[test] = float(np.average(y[train], weights=sample_weight[train]))
                fallback = "weighted_training_class_prior_no_features"
            elif model_kind in {"ridge", "tree_regressor", "weighted_mean"}:
                predictions[test] = float(np.average(y[train], weights=sample_weight[train]))
                fallback = "weighted_training_mean_no_features"
            else:
                raise AuditInputError(f"unknown model_kind {model_kind}")
            preprocess = dict(design["preprocess_diagnostics"])
            preprocess["zero_feature_model_fallback"] = fallback
            fold_output.append(
                {
                    "fold": int(design["fold"]),
                    "train_rows": int(len(train)),
                    "test_rows": int(len(test)),
                    "train_groups": int(len(set(str(groups[i]) for i in train))),
                    "test_groups": int(len(set(str(groups[i]) for i in test))),
                    "preprocess": preprocess,
                    "feature_names": list(design["feature_names"]),
                }
            )
            continue
        if model_kind == "logistic":
            if len(np.unique(y[train])) < 2:
                raise AuditInputError(f"fold {design['fold']} training data has one L1 class")
            model = LogisticRegression(
                penalty="l2",
                C=1.0,
                solver="liblinear",
                max_iter=2000,
                random_state=seed,
            )
            model.fit(
                design["x_train"],
                y[train],
                sample_weight=sample_weight[train],
            )
            predictions[test] = model.predict_proba(design["x_test"])[:, 1]
        elif model_kind == "tree_classifier":
            if len(np.unique(y[train])) < 2:
                raise AuditInputError(f"fold {design['fold']} training data has one L1 class")
            model = DecisionTreeClassifier(
                max_depth=2, min_samples_leaf=5, random_state=seed
            )
            model.fit(
                design["x_train"],
                y[train],
                sample_weight=sample_weight[train],
            )
            positive_index = int(np.flatnonzero(model.classes_ == 1)[0])
            predictions[test] = model.predict_proba(design["x_test"])[:, positive_index]
        elif model_kind == "ridge":
            model = Ridge(alpha=10.0, fit_intercept=True)
            model.fit(
                design["x_train"],
                y[train],
                sample_weight=sample_weight[train],
            )
            predictions[test] = model.predict(design["x_test"])
        elif model_kind == "tree_regressor":
            model = DecisionTreeRegressor(
                max_depth=2, min_samples_leaf=5, random_state=seed
            )
            model.fit(
                design["x_train"],
                y[train],
                sample_weight=sample_weight[train],
            )
            predictions[test] = model.predict(design["x_test"])
        elif model_kind == "weighted_mean":
            predictions[test] = float(np.average(y[train], weights=sample_weight[train]))
        else:
            raise AuditInputError(f"unknown model_kind {model_kind}")
        fold_output.append(
            {
                "fold": int(design["fold"]),
                "train_rows": int(len(train)),
                "test_rows": int(len(test)),
                "train_groups": int(len(set(str(groups[i]) for i in train))),
                "test_groups": int(len(set(str(groups[i]) for i in test))),
                "preprocess": design["preprocess_diagnostics"],
                "feature_names": list(design["feature_names"]),
            }
        )
    if np.any(~np.isfinite(predictions)):
        raise AuditInputError("OOF prediction array contains unfilled or non-finite values")
    return {"prediction": predictions, "folds": fold_output}


def _weighted_class_metrics(
    y: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> dict[str, float | None]:
    y = np.asarray(y, dtype=int)
    labels = (np.asarray(predictions, dtype=float) >= 0.5).astype(int)
    if len(np.unique(y)) < 2:
        auc: float | None = None
        ba: float | None = None
    else:
        auc = float(roc_auc_score(y, predictions, sample_weight=weights))
        ba = float(balanced_accuracy_score(y, labels, sample_weight=weights))
    return {"weighted_OOF_AUC": auc, "weighted_OOF_balanced_accuracy_at_0.5": ba}


def _weighted_regression_metrics(
    y: np.ndarray, predictions: np.ndarray, weights: np.ndarray
) -> dict[str, float | None]:
    y = np.asarray(y, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    if total <= 0.0:
        return {"weighted_OOF_R2": None, "weighted_OOF_MAE": None}
    mean = float(np.average(y, weights=weights))
    denominator = float(np.sum(weights * (y - mean) ** 2))
    mae = float(np.sum(weights * np.abs(y - predictions)) / total)
    r2 = None if denominator <= 1e-15 else float(1.0 - np.sum(weights * (y - predictions) ** 2) / denominator)
    return {"weighted_OOF_R2": r2, "weighted_OOF_MAE": mae}


def _group_metrics(
    y: np.ndarray,
    predictions: np.ndarray,
    weights: np.ndarray,
    groups: Sequence[Any],
    kind: str,
) -> list[dict[str, Any]]:
    group_array = np.asarray([str(g) for g in groups], dtype=object)
    result: list[dict[str, Any]] = []
    for group in sorted(set(group_array)):
        mask = group_array == group
        if kind == "classification":
            metrics = _weighted_class_metrics(y[mask], predictions[mask], weights[mask])
        else:
            metrics = _weighted_regression_metrics(y[mask], predictions[mask], weights[mask])
        result.append({"source_group": group, "rows": int(mask.sum()), **metrics})
    return result


def _l1_feature_diagnostics(
    schema: Mapping[str, Any], frame: pd.DataFrame, relative_fields: Mapping[str, str]
) -> dict[str, Any]:
    columns = list(dict.fromkeys(_all_feature_columns(schema) + list(relative_fields.values())))
    labels = frame["condition"].astype(str).to_numpy()
    weights = _l1_sample_weights(frame)
    diagnostics: dict[str, Any] = {}
    for column in columns:
        if column not in frame:
            continue
        series = frame[column]
        by_condition: dict[str, Any] = {}
        for condition in ("natural", "tts"):
            mask = labels == condition
            observed = series.iloc[np.flatnonzero(mask)]
            numeric = bool(pd.api.types.is_numeric_dtype(series))
            if numeric:
                values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                keep = mask & np.isfinite(values)
                w = weights[keep]
                x = values[keep]
                by_condition[condition] = {
                    "rows": int(mask.sum()),
                    "observed": int(keep.sum()),
                    "missing": int(mask.sum() - keep.sum()),
                    "weighted_mean": float(np.average(x, weights=w)) if len(x) and w.sum() > 0 else None,
                    "median": float(np.median(x)) if len(x) else None,
                    "sd": float(np.std(x, ddof=0)) if len(x) else None,
                    "minimum": float(np.min(x)) if len(x) else None,
                    "maximum": float(np.max(x)) if len(x) else None,
                }
            else:
                subset = series.iloc[np.flatnonzero(mask)]
                counts = subset.fillna(MISSING_CATEGORY).astype(str).value_counts().sort_index()
                by_condition[condition] = {
                    "rows": int(mask.sum()),
                    "observed": int(subset.notna().sum()),
                    "missing": int(subset.isna().sum()),
                    "category_counts": {str(k): int(v) for k, v in counts.items()},
                }
        item: dict[str, Any] = {"by_condition": by_condition}
        if pd.api.types.is_numeric_dtype(series):
            n_range = by_condition["natural"]
            t_range = by_condition["tts"]
            if n_range["minimum"] is not None and t_range["minimum"] is not None:
                lower = max(n_range["minimum"], t_range["minimum"])
                upper = min(n_range["maximum"], t_range["maximum"])
                union_lower = min(n_range["minimum"], t_range["minimum"])
                union_upper = max(n_range["maximum"], t_range["maximum"])
                intersects = lower <= upper
                overlap_width = max(0.0, upper - lower) if intersects else 0.0
                union_width = union_upper - union_lower
                item["observed_range_overlap"] = {
                    "status": "INTERSECTS" if intersects else "DISJOINT",
                    "natural_bounds": [n_range["minimum"], n_range["maximum"]],
                    "tts_bounds": [t_range["minimum"], t_range["maximum"]],
                    "intersection_bounds": [lower, upper] if intersects else None,
                    "intersection_width": overlap_width,
                    "combined_bounds": [union_lower, union_upper],
                    "intersection_fraction_of_combined_width": (
                        overlap_width / union_width if union_width > 0 else None
                    ),
                    "interpretation": "descriptive min/max interval intersection; not density overlap",
                }
            else:
                item["observed_range_overlap"] = {
                    "status": "UNAVAILABLE_MISSING_CONDITION_RANGE",
                    "natural_bounds": [n_range["minimum"], n_range["maximum"]],
                    "tts_bounds": [t_range["minimum"], t_range["maximum"]],
                    "intersection_bounds": None,
                    "intersection_width": None,
                    "combined_bounds": None,
                    "intersection_fraction_of_combined_width": None,
                    "interpretation": "descriptive min/max interval intersection; not density overlap",
                }
            paired = frame.pivot_table(
                index=["source_group", "pair_uid"], columns="condition", values=column, aggfunc="first"
            )
            if {"natural", "tts"}.issubset(paired.columns):
                paired = paired.dropna(subset=["natural", "tts"])
                paired["T_minus_N"] = pd.to_numeric(paired["tts"], errors="coerce") - pd.to_numeric(paired["natural"], errors="coerce")
                per_group = paired["T_minus_N"].groupby(level=0).mean().dropna()
                item["paired_group_equal_T_minus_N"] = {
                    "pairs": int(len(paired)),
                    "groups": int(len(per_group)),
                    "mean_group_difference": float(per_group.mean()) if len(per_group) else None,
                    "sd_across_group_means": float(per_group.std(ddof=1)) if len(per_group) > 1 else None,
                }
                if column == "silence_paper_leading_s":
                    y = (labels == "tts").astype(int)
                    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
                    keep = np.isfinite(values)
                    auc = float(roc_auc_score(y[keep], values[keep], sample_weight=weights[keep])) if len(set(y[keep])) == 2 else None
                    item["fixed_orientation_descriptive_AUC_TTS_positive"] = auc
                    item["auc_orientation"] = "raw leading-silence value; no max(AUC,1-AUC)"
        else:
            paired = frame.pivot_table(
                index=["source_group", "pair_uid"], columns="condition", values=column,
                aggfunc="first", dropna=False,
            )
            if {"natural", "tts"}.issubset(paired.columns):
                paired = paired.dropna(subset=["natural", "tts"])
                item["paired_category_agreement"] = {
                    "pairs": int(len(paired)),
                    "groups": int(paired.index.get_level_values(0).nunique()),
                    "agreement_fraction": float((paired["natural"].astype(str) == paired["tts"].astype(str)).mean()) if len(paired) else None,
                }
        diagnostics[column] = item
    return diagnostics


def _l1_missingness_model(
    schema: Mapping[str, Any], frame: pd.DataFrame, groups: np.ndarray,
    y: np.ndarray, weights: np.ndarray, fold_map: Mapping[str, int], seed: int,
) -> tuple[dict[str, Any], pd.DataFrame]:
    columns = _all_feature_columns(schema)
    x = pd.DataFrame(
        {f"{column}__missing": frame[column].isna().astype(float) for column in columns},
        index=frame.index,
    )
    numeric = list(x.columns)
    designs = _fold_designs(x, numeric, [], groups, fold_map)
    fitted = _fit_oof(designs, y, weights, groups, "logistic", seed)
    prediction = fitted["prediction"]
    model = {
        "view": "missingness_only",
        "model": "fixed LogisticRegression; only frozen predictor missingness indicators",
        **_weighted_class_metrics(y, prediction, weights),
        "per_group": _group_metrics(y, prediction, weights, groups, "classification"),
        "folds": fitted["folds"],
        "feature_columns": numeric,
    }
    oof = pd.DataFrame({
        "scope": "L1", "stratum": frame["source_layer"].astype(str), "model_id": "missingness_only",
        "audio_uid": frame["audio_uid"].astype(str),
        "pair_uid": frame["pair_uid"].astype(str), "source_group": groups,
        "dataset": frame["dataset"].astype(str), "condition": frame["condition"].astype(str),
        "y": y, "prediction": prediction, "sample_weight": weights,
        "fold": [fold_map[group] for group in groups],
    })
    return model, oof


def _l2_auxiliary_score_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    weights = _l2_sample_weights(frame)
    result: dict[str, Any] = {}
    for name, natural_name, tts_name in (
        ("B", "B_n", "B_t"), ("Dmin", "Dmin_n", "Dmin_t"), ("D0", "D0_n", "D0_t")
    ):
        if natural_name not in frame or tts_name not in frame:
            result[name] = {"status": "UNAVAILABLE_NOT_IN_SCORE_TABLE"}
            continue
        natural = pd.to_numeric(frame[natural_name], errors="coerce").to_numpy(dtype=float)
        tts = pd.to_numeric(frame[tts_name], errors="coerce").to_numpy(dtype=float)
        keep = np.isfinite(natural) & np.isfinite(tts)
        if not keep.any():
            result[name] = {"status": "UNAVAILABLE_NO_PAIRED_VALUES", "paired_rows": 0}
            continue
        delta = tts[keep] - natural[keep]
        w = weights[keep]
        result[name] = {
            "status": "DESCRIPTIVE_ONLY_NOT_A_PREDICTOR",
            "paired_rows": int(keep.sum()),
            "weighted_mean_T_minus_N": float(np.average(delta, weights=w)),
            "median_T_minus_N": float(np.median(delta)),
            "sd_T_minus_N": float(np.std(delta, ddof=0)),
        }
    return result


def _l2_missingness_inputs(
    schema: Mapping[str, Any], frame: pd.DataFrame, protocol: Mapping[str, Any] | None
) -> tuple[pd.DataFrame, list[str], list[str]]:
    columns = list(dict.fromkeys(_all_feature_columns(schema) + list(_relative_map(protocol).values())))
    result: dict[str, Any] = {}
    numeric: list[str] = []
    for column in columns:
        for arm in ("natural", "tts"):
            source = f"{arm}__{column}"
            if source not in frame:
                continue
            name = f"{source}__missing"
            result[name] = frame[source].isna().astype(float)
            numeric.append(name)
    for context in L2_CONTEXT:
        result[context] = frame[context].astype(object)
    categorical = list(L2_CONTEXT)
    x = pd.DataFrame(result, index=frame.index)
    return x, numeric, categorical


def _bootstrap_group_counts(
    groups: Sequence[Any], draws: int, seed: int
) -> tuple[np.ndarray, list[str]]:
    unique = sorted(set(str(group) for group in groups))
    if not unique or draws <= 0:
        return np.zeros((0, len(unique)), dtype=np.int16), unique
    index = {group: i for i, group in enumerate(unique)}
    rng = np.random.default_rng(seed)
    sample_indices = rng.integers(0, len(unique), size=(draws, len(unique)))
    counts = np.zeros((draws, len(unique)), dtype=np.int16)
    rows = np.arange(draws)[:, None]
    np.add.at(counts, (np.broadcast_to(rows, sample_indices.shape), sample_indices), 1)
    return counts, unique


def _classification_bootstrap(
    y: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    groups: Sequence[Any],
    count_matrix: np.ndarray,
    unique_groups: Sequence[str],
) -> dict[str, np.ndarray]:
    group_index = {group: i for i, group in enumerate(unique_groups)}
    gidx = np.asarray([group_index[str(group)] for group in groups], dtype=int)
    y = np.asarray(y, dtype=int)
    p = np.asarray(prediction, dtype=float)
    w = np.asarray(weights, dtype=float)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    if not len(pos) or not len(neg):
        nan = np.full(len(count_matrix), np.nan)
        return {"AUC": nan, "balanced_accuracy": nan}
    comparison = (p[pos, None] > p[None, neg]).astype(float)
    comparison += 0.5 * (p[pos, None] == p[None, neg])
    pair_group = np.zeros((len(unique_groups), len(unique_groups)), dtype=float)
    pair_values = (
        w[pos, None] * w[None, neg] * comparison
    )
    np.add.at(pair_group, (gidx[pos, None], gidx[None, neg]), pair_values)
    pos_group_weight = np.bincount(gidx[pos], weights=w[pos], minlength=len(unique_groups))
    neg_group_weight = np.bincount(gidx[neg], weights=w[neg], minlength=len(unique_groups))
    auc_num = np.einsum("bi,ij,bj->b", count_matrix, pair_group, count_matrix, optimize=True)
    pos_total = count_matrix @ pos_group_weight
    neg_total = count_matrix @ neg_group_weight
    auc = np.full(len(count_matrix), np.nan)
    valid_auc = (pos_total > 0) & (neg_total > 0)
    auc[valid_auc] = auc_num[valid_auc] / (pos_total[valid_auc] * neg_total[valid_auc])

    positive_correct = np.bincount(
        gidx[pos], weights=w[pos] * (p[pos] >= 0.5), minlength=len(unique_groups)
    )
    negative_correct = np.bincount(
        gidx[neg], weights=w[neg] * (p[neg] < 0.5), minlength=len(unique_groups)
    )
    pos_recall = np.divide(
        count_matrix @ positive_correct,
        pos_total,
        out=np.full(len(count_matrix), np.nan),
        where=pos_total > 0,
    )
    neg_recall = np.divide(
        count_matrix @ negative_correct,
        neg_total,
        out=np.full(len(count_matrix), np.nan),
        where=neg_total > 0,
    )
    return {"AUC": auc, "balanced_accuracy": (pos_recall + neg_recall) / 2.0}


def _regression_bootstrap(
    y: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray,
    groups: Sequence[Any],
    count_matrix: np.ndarray,
    unique_groups: Sequence[str],
) -> dict[str, np.ndarray]:
    group_index = {group: i for i, group in enumerate(unique_groups)}
    gidx = np.asarray([group_index[str(group)] for group in groups], dtype=int)
    y = np.asarray(y, dtype=float)
    pred = np.asarray(prediction, dtype=float)
    w = np.asarray(weights, dtype=float)
    n_groups = len(unique_groups)
    sw = np.bincount(gidx, weights=w, minlength=n_groups)
    sy = np.bincount(gidx, weights=w * y, minlength=n_groups)
    sy2 = np.bincount(gidx, weights=w * y * y, minlength=n_groups)
    abs_error = np.bincount(gidx, weights=w * np.abs(y - pred), minlength=n_groups)
    squared_error = np.bincount(gidx, weights=w * (y - pred) ** 2, minlength=n_groups)
    total_w = count_matrix @ sw
    total_y = count_matrix @ sy
    total_y2 = count_matrix @ sy2
    mae_sum = count_matrix @ abs_error
    sse = count_matrix @ squared_error
    denominator = total_y2 - np.divide(
        total_y * total_y,
        total_w,
        out=np.full_like(total_w, np.nan),
        where=total_w > 0,
    )
    mae = np.divide(mae_sum, total_w, out=np.full_like(total_w, np.nan), where=total_w > 0)
    r2 = np.divide(
        sse,
        denominator,
        out=np.full_like(denominator, np.nan),
        where=denominator > 1e-15,
    )
    r2 = 1.0 - r2
    return {"R2": r2, "MAE": mae}


def _ci_payload(values: np.ndarray, levels: Sequence[float]) -> dict[str, Any]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    result: dict[str, Any] = {"valid_draws": int(len(clean)), "total_draws": int(len(values))}
    if len(clean) == 0:
        result["intervals"] = {}
        return result
    result["median"] = float(np.median(clean))
    result["intervals"] = {}
    for level in levels:
        alpha = 1.0 - float(level)
        lo, hi = np.quantile(clean, [alpha / 2.0, 1.0 - alpha / 2.0])
        result["intervals"][f"{float(level):.12g}"] = [float(lo), float(hi)]
    return result


def _get_ci_levels(config: AuditConfig) -> list[float]:
    levels = [config.ordinary_coverage, config.adjusted_coverage]
    return sorted(set(float(level) for level in levels))


def _bootstrap_metrics_by_model(
    y: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    weights: np.ndarray,
    groups: Sequence[Any],
    kind: str,
    config: AuditConfig,
    seed_offset: int = 0,
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    count_matrix, unique = _bootstrap_group_counts(
        groups, config.bootstrap_draws, config.seed + seed_offset
    )
    raw: dict[str, dict[str, np.ndarray]] = {}
    summaries: dict[str, dict[str, Any]] = {}
    for name, pred in predictions.items():
        values = (
            _classification_bootstrap(y, pred, weights, groups, count_matrix, unique)
            if kind == "classification"
            else _regression_bootstrap(y, pred, weights, groups, count_matrix, unique)
        )
        raw[name] = values
        summaries[name] = {
            metric: _ci_payload(array, _get_ci_levels(config))
            for metric, array in values.items()
        }
    return summaries, raw


def _model_weights(groups: Sequence[Any], pair_ids: Sequence[Any]) -> np.ndarray:
    group_array = np.asarray([str(value) for value in groups], dtype=object)
    pair_array = np.asarray([str(value) for value in pair_ids], dtype=object)
    unique_groups = sorted(set(group_array))
    weights = np.zeros(len(group_array), dtype=float)
    for group in unique_groups:
        group_mask = group_array == group
        pairs = sorted(set(pair_array[group_mask]))
        for pair in pairs:
            mask = group_mask & (pair_array == pair)
            count = int(mask.sum())
            weights[mask] = len(group_array) / (len(unique_groups) * len(pairs) * count)
    return weights


def _resolve_config(
    schema: Mapping[str, Any], addendum: Mapping[str, Any] | None, protocol: Mapping[str, Any] | None
) -> AuditConfig:
    analysis = schema["analysis"]
    stable = (addendum or {}).get("stable_incremental_within_stratum", {})
    protocol_analysis = (protocol or {}).get("analysis", {})
    interval_cfg = (
        protocol_analysis.get("uncertainty", (protocol or {}).get("uncertainty", {}))
        if isinstance(protocol_analysis, Mapping) else (protocol or {}).get("uncertainty", {})
    )
    if not isinstance(interval_cfg, Mapping):
        interval_cfg = {}
    relative = _relative_map(protocol)
    ordinary = float(
        interval_cfg.get("ordinary_coverage", stable.get("ordinary_coverage", 0.95))
    )
    adjusted = float(
        interval_cfg.get(
            "seven_family_adjusted_coverage",
            stable.get("seven_family_adjusted_coverage", 1.0 - 0.05 / 7.0),
        )
    )
    return AuditConfig(
        seed=int(analysis.get("seed", 20261015)),
        min_groups=5,
        min_pairs=10,
        max_folds=5,
        bootstrap_draws=int(interval_cfg.get("bootstrap_draws", stable.get("bootstrap_draws", 2000))),
        permutation_draws=int(
            interval_cfg.get(
                "permutation_draws",
                int(analysis.get("permutation_L1", "199").split()[0])
                if isinstance(analysis.get("permutation_L1"), str)
                else analysis.get("permutation_L1", 199),
            )
        ),
        ordinary_coverage=ordinary,
        adjusted_coverage=adjusted,
        relative_fields=relative,
        cpu_threads=2,
    )


def _validate_protocol_is_final(protocol: Mapping[str, Any]) -> None:
    status = str(protocol.get("status", "")).upper()
    if "PENDING" in status or "DRAFT" in status or not status:
        raise AuditInputError(
            "formal analysis is blocked until the ROOT-bound final protocol has a non-pending status"
        )
    if not any(key in protocol for key in ("cohort_binding", "cohort", "inputs")):
        raise AuditInputError("final protocol lacks cohort/input binding")


def _check_protocol_input_bindings(
    protocol: Mapping[str, Any], input_paths: Mapping[str, str]
) -> dict[str, Any]:
    binding = protocol.get("input_bindings", protocol.get("inputs", {}))
    if not isinstance(binding, Mapping):
        raise AuditInputError("protocol input binding must be an object")
    checks: dict[str, Any] = {}
    for name, path in input_paths.items():
        expected = binding.get(name)
        if isinstance(expected, Mapping):
            expected_path = expected.get("path")
            if expected_path and str(expected_path) != str(path):
                raise AuditInputError(f"protocol path mismatch for {name}")
            expected_sha = expected.get("sha256")
            if expected_sha:
                import hashlib

                h = hashlib.sha256()
                with open(path, "rb") as stream:
                    for block in iter(lambda: stream.read(1 << 20), b""):
                        h.update(block)
                actual_sha = h.hexdigest()
                if actual_sha != expected_sha:
                    raise AuditInputError(f"protocol SHA-256 mismatch for {name}")
                checks[name] = {"path": path, "sha256": actual_sha, "status": "PASS"}
    return checks


def _l1_permutation_codes(n_groups: int, n_draws: int, seed: int) -> np.ndarray:
    if n_groups <= 0 or n_draws <= 0:
        return np.zeros((0, n_groups), dtype=np.int8)
    total_nonidentity = (1 << n_groups) - 1 if n_groups < 62 else None
    if total_nonidentity is not None and total_nonidentity <= n_draws:
        rows = [
            [(code >> bit) & 1 for bit in range(n_groups)]
            for code in range(1, 1 << n_groups)
        ]
        return np.asarray(rows, dtype=np.int8)
    rng = np.random.default_rng(seed)
    selected: set[bytes] = set()
    values: list[np.ndarray] = []
    while len(values) < n_draws:
        batch = rng.integers(0, 2, size=(max(16, n_draws), n_groups), dtype=np.int8)
        for row in batch:
            if not np.any(row):
                continue
            key = row.tobytes()
            if key not in selected:
                selected.add(key)
                values.append(row.copy())
                if len(values) == n_draws:
                    break
    return np.stack(values)


def _permute_l1_group_labels(
    y: np.ndarray, groups: Sequence[Any], unique_groups: Sequence[str], signs: np.ndarray
) -> np.ndarray:
    index = {group: i for i, group in enumerate(unique_groups)}
    result = np.asarray(y, dtype=int).copy()
    group_array = np.asarray([str(group) for group in groups], dtype=object)
    flip = {group: bool(signs[index[group]]) for group in unique_groups}
    for group, should_flip in flip.items():
        if should_flip:
            mask = group_array == group
            result[mask] = 1 - result[mask]
    return result


def _l1_permutation_test(
    datasets: Mapping[str, tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]],
    groups: Sequence[Any],
    observed: Mapping[str, float | None],
    config: AuditConfig,
) -> dict[str, Any]:
    unique_groups = sorted(set(str(group) for group in groups))
    signs = _l1_permutation_codes(
        len(unique_groups), config.permutation_draws, config.seed + 101
    )
    y0 = next(iter(datasets.values()))[1]
    nulls = {name: [] for name in datasets}
    for signs_row in signs:
        yp = _permute_l1_group_labels(y0, groups, unique_groups, signs_row)
        wp = _l1_sample_weights(
            pd.DataFrame({"condition": np.where(yp == 1, "tts", "natural"), "source_group": groups}),
            yp,
        )
        for name, (designs, _y, _weights, _g) in datasets.items():
            fitted = _fit_oof(designs, yp, wp, groups, "logistic", config.seed)
            metrics = _weighted_class_metrics(yp, fitted["prediction"], wp)
            nulls[name].append(metrics["weighted_OOF_AUC"])
    result: dict[str, Any] = {
        "status": "PASS",
        "method": "group-level N/T label swaps; same frozen folds; train-only preprocessing refit",
        "n_unique_groups": len(unique_groups),
        "permutations": int(len(signs)),
        "exact_enumeration": bool((1 << len(unique_groups)) - 1 <= config.permutation_draws)
        if len(unique_groups) < 62
        else False,
        "seed": config.seed + 101,
        "primary_statistic": "weighted_OOF_AUC; two-sided distance from 0.5",
        "views": {},
    }
    for name, values in nulls.items():
        arr = np.asarray(values, dtype=float)
        obs = observed[name]
        p_value = None
        if obs is not None and len(arr):
            p_value = float((1 + np.sum(np.abs(arr - 0.5) >= abs(float(obs) - 0.5))) / (1 + len(arr)))
        result["views"][name] = {
            "observed_auc": obs,
            "null_auc_mean": float(np.nanmean(arr)) if len(arr) else None,
            "null_auc_sd": float(np.nanstd(arr, ddof=1)) if len(arr) > 1 else None,
            "permutation_p_plus_one": p_value,
            "null_values": arr.tolist(),
        }
    return result


def _l2_exchangeability(
    frame: pd.DataFrame, weights: np.ndarray
) -> tuple[dict[str, list[str]], dict[str, Any]]:
    contexts = ["dataset", "tfg", "tts_provider", "score_contract", "feature_layer"]
    blocks: dict[str, list[str]] = defaultdict(list)
    rows_by_group = {
        str(group): group_frame.sort_values(contexts + ["pair_uid", "score_uid"], kind="mergesort")
        for group, group_frame in frame.groupby("source_group", sort=True)
    }
    signatures: dict[str, str] = {}
    for group, group_frame in rows_by_group.items():
        counts = Counter(tuple(str(row[column]) for column in contexts) for _, row in group_frame.iterrows())
        signature = json.dumps(
            sorted((list(context), count) for context, count in counts.items()),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        signatures[group] = signature
        blocks[signature].append(group)
    eligible_blocks = {signature: sorted(groups) for signature, groups in blocks.items() if len(groups) >= 2}
    movable = {group for groups in eligible_blocks.values() for group in groups}
    group_array = frame["source_group"].astype(str).to_numpy()
    movable_weight = float(np.asarray(weights)[np.isin(group_array, list(movable))].sum())
    total_weight = float(np.asarray(weights).sum())
    summary = {
        "groups": int(len(rows_by_group)),
        "movable_groups": int(len(movable)),
        "movable_rows": int(np.isin(group_array, list(movable)).sum()),
        "rows": int(len(frame)),
        "movable_weight_coverage": movable_weight / total_weight if total_weight > 0 else 0.0,
        "block_group_counts": sorted(len(value) for value in eligible_blocks.values()),
        "singleton_groups": sorted(set(rows_by_group) - movable),
    }
    return eligible_blocks, summary


def _permute_l2_target_blocks(
    frame: pd.DataFrame, y: np.ndarray, blocks: Mapping[str, Sequence[str]], rng: np.random.Generator
) -> np.ndarray:
    result = np.asarray(y, dtype=float).copy()
    group_array = frame["source_group"].astype(str).to_numpy()
    contexts = ["dataset", "tfg", "tts_provider", "score_contract", "feature_layer"]
    sorted_rows = {
        str(group): group_frame.sort_values(contexts + ["pair_uid", "score_uid"], kind="mergesort").index.to_numpy()
        for group, group_frame in frame.groupby("source_group", sort=True)
    }
    for block_groups in blocks.values():
        donor_order = rng.permutation(list(block_groups))
        for recipient, donor in zip(block_groups, donor_order):
            rec_idx = sorted_rows[recipient]
            donor_idx = sorted_rows[donor]
            if len(rec_idx) != len(donor_idx):
                raise AuditInputError("exchangeable L2 blocks have different row counts")
            # The block signature guarantees the same ordered context pattern.
            for rec_i, donor_i in zip(rec_idx, donor_idx):
                rec_row = frame.loc[rec_i]
                donor_row = frame.loc[donor_i]
                if any(str(rec_row[c]) != str(donor_row[c]) for c in contexts):
                    raise AuditInputError("exchangeable L2 row contexts do not align")
                result[frame.index.get_loc(rec_i)] = float(y[frame.index.get_loc(donor_i)])
    return result


def _l2_permutation_test(
    frame: pd.DataFrame,
    designs: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    weights: np.ndarray,
    observed_r2: float | None,
    config: AuditConfig,
) -> dict[str, Any]:
    blocks, summary = _l2_exchangeability(frame.reset_index(drop=True), weights)
    if summary["movable_groups"] < 5 or summary["movable_weight_coverage"] < 0.50:
        return {
            "status": "INCONCLUSIVE_INSUFFICIENT_EXCHANGEABLE_BLOCKS",
            "exchangeability": summary,
            "permutations": 0,
            "observed_r2": observed_r2,
        }
    frame0 = frame.reset_index(drop=True)
    rng = np.random.default_rng(config.seed + 202)
    null_values: list[float] = []
    y0 = np.asarray(y, dtype=float)
    for _ in range(config.permutation_draws):
        yp = _permute_l2_target_blocks(frame0, y0, blocks, rng)
        pred = _fit_oof(designs, yp, weights, frame0["source_group"].astype(str), "ridge", config.seed)
        metric = _weighted_regression_metrics(yp, pred["prediction"], weights)
        if metric["weighted_OOF_R2"] is not None:
            null_values.append(float(metric["weighted_OOF_R2"]))
    p_value = None
    if observed_r2 is not None and null_values:
        p_value = float((1 + np.sum(np.asarray(null_values) >= float(observed_r2))) / (1 + len(null_values)))
    return {
        "status": "PASS",
        "method": "whole-source delta-C target blocks permuted within identical context/count signatures",
        "seed": config.seed + 202,
        "exchangeability": summary,
        "permutations_requested": config.permutation_draws,
        "permutations_completed": len(null_values),
        "primary_statistic": "weighted_OOF_R2 of all-seven-plus-context Ridge",
        "observed_r2": observed_r2,
        "null_r2_mean": float(np.mean(null_values)) if null_values else None,
        "null_r2_sd": float(np.std(null_values, ddof=1)) if len(null_values) > 1 else None,
        "permutation_p_plus_one": p_value,
        "null_values": null_values,
    }


def _stable_family_label(
    adjusted_ci: Mapping[str, Any] | None,
    relative_gain: float | None,
    primary_gain: float | None,
    adjusted_level: float,
) -> str:
    if primary_gain is None or relative_gain is None or adjusted_ci is None:
        return "EVIDENCE_INCOMPLETE"
    interval = adjusted_ci.get("intervals", {}).get(f"{adjusted_level:.12g}")
    if interval is None:
        return "EVIDENCE_INCOMPLETE"
    if float(interval[0]) > 0.0 and primary_gain > 0.0 and relative_gain > 0.0:
        return "STABLE_INCREMENTAL_WITHIN_STRATUM"
    return "NO_STABLE_INCREMENTAL_EVIDENCE"


def _family_incremental(
    names: Sequence[str],
    full_name: str,
    dropped_names: Mapping[str, str],
    raw_bootstrap: Mapping[str, Mapping[str, np.ndarray]],
    point_metrics: Mapping[str, Mapping[str, float | None]],
    primary_metric: str,
    gain_direction: str,
    relative_full_name: str | None,
    relative_drop_names: Mapping[str, str] | None,
    adjusted_level: float,
    ordinary_level: float,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    full_draws = raw_bootstrap[full_name][primary_metric]
    for family in names:
        drop_name = dropped_names[family]
        drop_draws = raw_bootstrap[drop_name][primary_metric]
        if gain_direction == "higher":
            gain = full_draws - drop_draws
        else:
            gain = drop_draws - full_draws
        ordinary = _ci_payload(gain, [ordinary_level])
        adjusted = _ci_payload(gain, [adjusted_level])
        full_point = point_metrics.get(full_name, {}).get(primary_metric)
        drop_point = point_metrics.get(drop_name, {}).get(primary_metric)
        if full_point is None or drop_point is None:
            point_gain = None
        elif gain_direction == "higher":
            point_gain = float(full_point - drop_point)
        else:
            point_gain = float(drop_point - full_point)
        relative_gain = None
        if relative_full_name and relative_drop_names and family in relative_drop_names:
            rel_full = point_metrics.get(relative_full_name, {}).get(primary_metric)
            rel_drop = point_metrics.get(relative_drop_names[family], {}).get(primary_metric)
            if rel_full is not None and rel_drop is not None:
                relative_gain = (
                    float(rel_full - rel_drop)
                    if gain_direction == "higher"
                    else float(rel_drop - rel_full)
                )
        result[family] = {
            "primary_metric": primary_metric,
            "point_gain": point_gain,
            "ordinary_interval": ordinary,
            "seven_family_adjusted_interval": adjusted,
            "relative_silence_gain": relative_gain,
            "stable_label": _stable_family_label(
                adjusted, relative_gain, point_gain, adjusted_level
            ),
        }
    return result


def _engineering_family_incremental(
    names: Sequence[str],
    full_name: str,
    dropped_names: Mapping[str, str],
    point_metrics: Mapping[str, Mapping[str, float | None]],
    primary_metric: str,
    gain_direction: str,
    relative_full_name: str | None,
    relative_drop_names: Mapping[str, str] | None,
) -> dict[str, Any]:
    """Point contrasts for engineering smoke only; never attach inferential labels."""
    full_point = point_metrics.get(full_name, {}).get(primary_metric)
    relative_full = point_metrics.get(relative_full_name, {}).get(primary_metric) if relative_full_name else None
    output: dict[str, Any] = {}
    for family in names:
        drop_point = point_metrics.get(dropped_names[family], {}).get(primary_metric)
        if full_point is None or drop_point is None:
            point_gain = None
        elif gain_direction == "higher":
            point_gain = float(full_point - drop_point)
        else:
            point_gain = float(drop_point - full_point)
        relative_gain = None
        relative_drop_name = (relative_drop_names or {}).get(family)
        relative_drop = point_metrics.get(relative_drop_name, {}).get(primary_metric) if relative_drop_name else None
        if relative_full is not None and relative_drop is not None:
            relative_gain = (
                float(relative_full - relative_drop)
                if gain_direction == "higher"
                else float(relative_drop - relative_full)
            )
        output[family] = {
            "primary_metric": primary_metric,
            "point_gain": point_gain,
            "ordinary_interval": None,
            "seven_family_adjusted_interval": None,
            "relative_silence_gain": relative_gain,
            "stable_label": "NOT_EVALUATED_ENGINEERING_ONLY",
            "inference_status": "NOT_RUN_ENGINEERING_ONLY",
        }
    return output


def _lodo_partitions(
    frame: pd.DataFrame, config: AuditConfig
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build dataset holdouts after purging source groups shared across domains."""
    if "dataset" not in frame.columns:
        return [], [{"status": "NOT_TESTED", "reason": "dataset_column_unavailable"}]
    datasets = sorted(frame["dataset"].dropna().astype(str).unique().tolist())
    if len(datasets) < 2:
        return [], [{"status": "NOT_TESTED", "reason": "fewer_than_two_datasets"}]
    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    group_values = frame["source_group"].astype(str)
    dataset_values = frame["dataset"].astype(str)
    for dataset in datasets:
        test_mask = dataset_values.eq(dataset).to_numpy()
        test_groups = set(group_values[test_mask])
        train_mask = (~test_mask) & (~group_values.isin(test_groups).to_numpy())
        train_groups = set(group_values[train_mask])
        test_pairs = int(frame.loc[test_mask, "pair_uid"].nunique())
        train_pairs = int(frame.loc[train_mask, "pair_uid"].nunique())
        reason = None
        if len(test_groups) < config.min_groups or len(train_groups) < config.min_groups:
            reason = "below_minimum_independent_groups_after_purge"
        elif test_pairs < config.min_pairs or train_pairs < config.min_pairs:
            reason = "below_minimum_pairs_after_purge"
        else:
            context_columns = [column for column in ("tfg", "tts_provider") if column in frame.columns]
            if not context_columns:
                reason = "tfg_provider_context_unavailable"
            else:
                def pair_signatures(mask: np.ndarray) -> set[tuple[Any, ...]]:
                    subset = frame.loc[mask]
                    signatures: set[tuple[Any, ...]] = set()
                    for _, pair in subset.groupby("pair_uid", sort=True):
                        parts: list[tuple[str, ...]] = []
                        for column in context_columns:
                            observed = sorted(
                                {str(value) for value in pair[column].tolist() if pd.notna(value)}
                            )
                            parts.append(tuple(observed))
                        if all(parts):
                            signatures.add(tuple(parts))
                    return signatures

                common_context = pair_signatures(train_mask) & pair_signatures(test_mask)
                if not common_context:
                    reason = "no_shared_tfg_provider_context"
        if reason:
            skipped.append(
                {
                    "dataset": dataset,
                    "status": "NOT_TESTED",
                    "reason": reason,
                    "train_groups": len(train_groups),
                    "test_groups": len(test_groups),
                    "train_pairs": train_pairs,
                    "test_pairs": test_pairs,
                    "purged_shared_groups": len(test_groups & set(group_values[~test_mask])),
                }
            )
        else:
            eligible.append(
                {
                    "dataset": dataset,
                    "train_indices": np.flatnonzero(train_mask),
                    "test_indices": np.flatnonzero(test_mask),
                    "train_groups": sorted(train_groups),
                    "test_groups": sorted(test_groups),
                    "train_pairs": train_pairs,
                    "test_pairs": test_pairs,
                    "purged_shared_groups": len(test_groups & set(group_values[~test_mask])),
                }
            )
    return eligible, skipped


def _fit_external_split(
    train_x: pd.DataFrame,
    test_x: pd.DataFrame,
    numeric: Sequence[str],
    categorical: Sequence[str],
    y_train: np.ndarray,
    train_weights: np.ndarray,
    model_kind: str,
) -> np.ndarray:
    transformer = FoldTransformer(numeric, categorical)
    x_train, x_test = transformer.fit_transform_pair(train_x, test_x)
    if x_train.shape[1] == 0:
        raise AuditInputError("LODO predictor view has no estimable training columns")
    if model_kind == "logistic":
        model = LogisticRegression(
            penalty="l2", C=1.0, solver="liblinear", max_iter=2000, random_state=20261015
        )
        model.fit(x_train, y_train, sample_weight=train_weights)
        return model.predict_proba(x_test)[:, 1]
    if model_kind == "ridge":
        model = Ridge(alpha=10.0, fit_intercept=True)
        model.fit(x_train, y_train, sample_weight=train_weights)
        return model.predict(x_test)
    raise AuditInputError(f"unsupported external validation model: {model_kind}")


def _holdout_class_metrics(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    metrics = _weighted_class_metrics(y, prediction, weights)
    return {
        "weighted_holdout_AUC": metrics["weighted_OOF_AUC"],
        "weighted_holdout_balanced_accuracy_at_0.5": metrics["weighted_OOF_balanced_accuracy_at_0.5"],
    }


def _holdout_regression_metrics(y: np.ndarray, prediction: np.ndarray, weights: np.ndarray) -> dict[str, Any]:
    metrics = _weighted_regression_metrics(y, prediction, weights)
    return {
        "weighted_holdout_R2": metrics["weighted_OOF_R2"],
        "weighted_holdout_MAE": metrics["weighted_OOF_MAE"],
    }


def _lodo_l1(
    schema: Mapping[str, Any],
    protocol: Mapping[str, Any] | None,
    frame: pd.DataFrame,
    config: AuditConfig,
) -> dict[str, Any]:
    partitions, skipped = _lodo_partitions(frame, config)
    family_names = list(schema["families"].keys())
    output: dict[str, Any] = {
        "status": "NOT_TESTED",
        "method": "leave-one-dataset-out; shared source groups purged from training",
        "skipped_datasets": skipped,
        "domains": [],
        "family_direction": {},
    }
    if not partitions:
        return output
    y = np.asarray([1 if str(value) == "tts" else 0 for value in frame["condition"]], dtype=int)
    models = ["joint"] + [f"joint_drop_{family}" for family in family_names]
    models += ["joint_relative"] + [f"joint_relative_drop_{family}" for family in family_names]
    domain_results: list[dict[str, Any]] = []
    for part in partitions:
        train_idx, test_idx = part["train_indices"], part["test_indices"]
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        train_w = _l1_sample_weights(train, y[train_idx])
        test_w = _l1_sample_weights(test, y[test_idx])
        model_results: dict[str, Any] = {}
        for model_name in models:
            relative = "relative" in model_name
            drop_family = model_name.split("_drop_", 1)[1] if "_drop_" in model_name else None
            columns, numeric, categorical = _l1_columns(
                schema, frame, "joint", drop_family, relative, protocol
            )
            try:
                prediction = _fit_external_split(
                    train[columns], test[columns], numeric, categorical,
                    y[train_idx], train_w, "logistic"
                )
                model_results[model_name] = _holdout_class_metrics(
                    y[test_idx], prediction, test_w
                )
            except AuditInputError as exc:
                model_results[model_name] = {"status": "NON_IDENTIFIABLE", "reason": str(exc)}
        domain_results.append({
            "dataset": part["dataset"],
            "train_groups": len(part["train_groups"]),
            "test_groups": len(part["test_groups"]),
            "train_pairs": part["train_pairs"],
            "test_pairs": part["test_pairs"],
            "purged_shared_groups": part["purged_shared_groups"],
            "models": model_results,
        })
    output["domains"] = domain_results
    output["status"] = "PASS" if len(domain_results) >= 2 else "NOT_TESTED_LTWO_ELIGIBLE_DATASETS"
    for family in family_names:
        per_domain: list[dict[str, Any]] = []
        for domain in domain_results:
            models_at_domain = domain["models"]
            gains: dict[str, float | None] = {}
            for full_name, drop_name, label in (
                ("joint", f"joint_drop_{family}", "absolute"),
                ("joint_relative", f"joint_relative_drop_{family}", "relative_silence"),
            ):
                full = models_at_domain.get(full_name, {}).get("weighted_holdout_AUC")
                drop = models_at_domain.get(drop_name, {}).get("weighted_holdout_AUC")
                gains[label] = float(full - drop) if full is not None and drop is not None else None
            per_domain.append({"dataset": domain["dataset"], **gains})
        primary = [row["absolute"] for row in per_domain]
        relative = [row["relative_silence"] for row in per_domain]
        enough = output["status"] == "PASS" and all(value is not None for value in primary + relative)
        output["family_direction"][family] = {
            "per_domain_gains": per_domain,
            "same_positive_direction_across_domains": bool(
                enough and all(value > 0.0 for value in primary + relative)
            ) if enough else None,
            "interpretation": "direction check only; no cross-domain causal or population claim",
        }
    return output


def _lodo_l2(
    schema: Mapping[str, Any],
    protocol: Mapping[str, Any] | None,
    frame: pd.DataFrame,
    config: AuditConfig,
) -> dict[str, Any]:
    partitions, skipped = _lodo_partitions(frame, config)
    family_names = list(schema["families"].keys())
    output: dict[str, Any] = {
        "status": "NOT_TESTED",
        "method": "leave-one-dataset-out; shared source groups purged from training",
        "skipped_datasets": skipped,
        "domains": [],
        "family_direction": {},
    }
    if not partitions:
        return output
    y = frame["delta_c"].to_numpy(dtype=float)
    model_names = ["all_seven_plus_context"] + [
        f"all_seven_drop_{family}" for family in family_names
    ] + ["all_seven_relative_plus_context"] + [
        f"all_seven_relative_drop_{family}" for family in family_names
    ]
    domain_results: list[dict[str, Any]] = []
    for part in partitions:
        train_idx, test_idx = part["train_indices"], part["test_indices"]
        train, test = frame.iloc[train_idx], frame.iloc[test_idx]
        train_w = _l2_sample_weights(train)
        test_w = _l2_sample_weights(test)
        model_results: dict[str, Any] = {}
        for model_name in model_names:
            relative = "relative" in model_name
            drop_family = model_name.split("_drop_", 1)[1] if "_drop_" in model_name else None
            try:
                x, numeric, categorical, _ = _l2_x(
                    schema, frame, "all_seven_plus_context", drop_family,
                    relative, protocol
                )
                prediction = _fit_external_split(
                    x.iloc[train_idx], x.iloc[test_idx], numeric, categorical,
                    y[train_idx], train_w, "ridge"
                )
                model_results[model_name] = _holdout_regression_metrics(
                    y[test_idx], prediction, test_w
                )
            except AuditInputError as exc:
                model_results[model_name] = {"status": "NON_IDENTIFIABLE", "reason": str(exc)}
        domain_results.append({
            "dataset": part["dataset"],
            "train_groups": len(part["train_groups"]),
            "test_groups": len(part["test_groups"]),
            "train_pairs": part["train_pairs"],
            "test_pairs": part["test_pairs"],
            "purged_shared_groups": part["purged_shared_groups"],
            "models": model_results,
        })
    output["domains"] = domain_results
    output["status"] = "PASS" if len(domain_results) >= 2 else "NOT_TESTED_LTWO_ELIGIBLE_DATASETS"
    for family in family_names:
        per_domain: list[dict[str, Any]] = []
        for domain in domain_results:
            metrics = domain["models"]
            gains: dict[str, float | None] = {}
            for full_name, drop_name, label in (
                ("all_seven_plus_context", f"all_seven_drop_{family}", "absolute"),
                ("all_seven_relative_plus_context", f"all_seven_relative_drop_{family}", "relative_silence"),
            ):
                full = metrics.get(full_name, {}).get("weighted_holdout_MAE")
                drop = metrics.get(drop_name, {}).get("weighted_holdout_MAE")
                gains[label] = float(drop - full) if full is not None and drop is not None else None
            per_domain.append({"dataset": domain["dataset"], **gains})
        primary = [row["absolute"] for row in per_domain]
        relative = [row["relative_silence"] for row in per_domain]
        enough = output["status"] == "PASS" and all(value is not None for value in primary + relative)
        output["family_direction"][family] = {
            "per_domain_gains": per_domain,
            "same_positive_direction_across_domains": bool(
                enough and all(value > 0.0 for value in primary + relative)
            ) if enough else None,
            "interpretation": "direction check only; no cross-domain causal or population claim",
        }
    return output


def _l1_fit_stratum(
    schema: Mapping[str, Any],
    protocol: Mapping[str, Any] | None,
    frame: pd.DataFrame,
    pair_count: int,
    fold_map: Mapping[str, int],
    config: AuditConfig,
    layer: str,
    engineering_only: bool = False,
) -> tuple[dict[str, Any], list[pd.DataFrame]]:
    groups = frame["source_group"].astype(str).to_numpy()
    pairs = frame["pair_uid"].astype(str).to_numpy()
    y = np.asarray([1 if str(value) == "tts" else 0 for value in frame["condition"]], dtype=int)
    weights = _l1_sample_weights(frame, y)
    result: dict[str, Any] = {
        "layer": layer,
        "status": "PASS_ENGINEERING_ONLY" if engineering_only else "PASS",
        "rows": int(len(frame)),
        "complete_pairs": int(pair_count),
        "source_groups": int(len(set(groups))),
        "unique_audios": int(frame["audio_uid"].nunique()),
        "same_pcm_hash_both_conditions": _pcm_label_collisions(frame),
        "gate": {"min_groups": config.min_groups, "min_pairs": config.min_pairs},
        "models": {},
        "family_incremental": {},
        "permutation": None,
        "relative_silence_sensitivity": {"status": "NOT_RUN"},
        "feature_diagnostics": _l1_feature_diagnostics(schema, frame, config.relative_fields),
        "timing_seconds": {},
    }
    if len(set(groups)) < config.min_groups or pair_count < config.min_pairs:
        result["status"] = "DESCRIPTIVE_INCONCLUSIVE_BELOW_MINIMUM"
        return result, []
    model_preds: dict[str, np.ndarray] = {}
    bootstrap_predictions: dict[str, np.ndarray] = {}
    design_cache: dict[str, list[dict[str, Any]]] = {}
    oof_frames: list[pd.DataFrame] = []
    family_names = list(schema["families"].keys())
    started = time.perf_counter()

    def fit_variant(
        key: str,
        view: str,
        drop_family: str | None = None,
        relative: bool = False,
        model_kind: str = "logistic",
    ) -> tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]:
        columns, numeric, categorical = _l1_columns(
            schema, frame, view, drop_family, relative, protocol
        )
        raw_x = frame[columns].copy()
        design_key = f"{view}|{drop_family or 'full'}|{'relative' if relative else 'absolute'}"
        designs = _fold_designs(
            raw_x, numeric, categorical, groups, fold_map,
            diagnose_collinearity=(key == "joint"),
        )
        fitted = _fit_oof(designs, y, weights, groups, model_kind, config.seed)
        pred = fitted["prediction"]
        metrics = _weighted_class_metrics(y, pred, weights)
        out = {
            "view": view,
            "drop_family": drop_family,
            "variant": "relative_silence" if relative else "absolute_silence",
            "model": model_kind,
            **metrics,
            "per_group": _group_metrics(y, pred, weights, groups, "classification"),
            "folds": fitted["folds"],
            "feature_columns": columns,
            "weights_sum": float(weights.sum()),
        }
        prediction_frame = pd.DataFrame(
            {
                "scope": "L1",
                "stratum": layer,
                "model_id": key,
                "audio_uid": frame["audio_uid"].astype(str),
                "pair_uid": frame["pair_uid"].astype(str),
                "source_group": groups,
                "dataset": frame["dataset"].astype(str),
                "condition": frame["condition"].astype(str),
                "y": y,
                "prediction": pred,
                "sample_weight": weights,
                "fold": [fold_map[group] for group in groups],
            }
        )
        design_cache[design_key] = designs
        return out, pred, [prediction_frame]

    for view in ("metadata_only", "acoustic_only", "joint"):
        started_view = time.perf_counter()
        key = view
        out, pred, pieces = fit_variant(key, view)
        result["models"][key] = out
        model_preds[key] = pred
        bootstrap_predictions[key] = pred
        oof_frames.extend(pieces)
        result["timing_seconds"][key] = time.perf_counter() - started_view

    drop_names: dict[str, str] = {}
    for family in family_names:
        key = f"joint_drop_{family}"
        out, pred, pieces = fit_variant(key, "joint", drop_family=family)
        result["models"][key] = out
        bootstrap_predictions[key] = pred
        drop_names[family] = key
        oof_frames.extend(pieces)

    tree_out, tree_pred, tree_pieces = fit_variant(
        "joint_tree_depth2", "joint", model_kind="tree_classifier"
    )
    result["models"]["joint_tree_depth2"] = tree_out
    oof_frames.extend(tree_pieces)

    missing_out, missing_oof = _l1_missingness_model(
        schema, frame, groups, y, weights, fold_map, config.seed
    )
    result["models"]["missingness_only"] = missing_out
    oof_frames.append(missing_oof)

    relative_map = _relative_map(protocol)
    relative_columns = set(relative_map.values())
    missing_relative = sorted(relative_columns - set(frame.columns))
    if missing_relative:
        result["relative_silence_sensitivity"] = {
            "status": "CONTRACT_ERROR_RELATIVE_COLUMNS_MISSING",
            "missing_columns": missing_relative,
        }
        relative_drops: dict[str, str] = {}
    else:
        rel_names = ["acoustic_only_relative", "joint_relative"]
        relative_drops = {}
        relative_boot: dict[str, np.ndarray] = {}
        for key in rel_names:
            view = key.replace("_relative", "")
            out, pred, pieces = fit_variant(key, view, relative=True)
            result["models"][key] = out
            bootstrap_predictions[key] = pred
            relative_boot[key] = pred
            oof_frames.extend(pieces)
        for family in family_names:
            key = f"joint_relative_drop_{family}"
            out, pred, pieces = fit_variant(
                key, "joint", drop_family=family, relative=True
            )
            result["models"][key] = out
            bootstrap_predictions[key] = pred
            relative_drops[family] = key
            oof_frames.extend(pieces)
        result["relative_silence_sensitivity"] = {
            "status": "PASS",
            "same_fold_and_row_support": True,
            "required_alternate_columns": relative_map,
            "joint_relative_auc": result["models"]["joint_relative"]["weighted_OOF_AUC"],
        }

    relative_full = "joint_relative" if "joint_relative" in result["models"] else None
    point_metrics = {
        name: {
            "AUC": model.get("weighted_OOF_AUC"),
            "balanced_accuracy": model.get("weighted_OOF_balanced_accuracy_at_0.5"),
        }
        for name, model in result["models"].items()
    }
    if engineering_only:
        result["bootstrap"] = {"status": "NOT_RUN_ENGINEERING_ONLY", "draws": 0}
        result["family_incremental"] = _engineering_family_incremental(
            family_names, "joint", drop_names, point_metrics, "AUC", "higher",
            relative_full, relative_drops if relative_full else None,
        )
        result["permutation"] = {"status": "NOT_RUN_ENGINEERING_ONLY", "draws": 0}
        result["cross_domain_lodo"] = {"status": "NOT_RUN_ENGINEERING_ONLY"}
    else:
        boot_started = time.perf_counter()
        bootstrap_summary, bootstrap_raw = _bootstrap_metrics_by_model(
            y, bootstrap_predictions, weights, groups, "classification", config, seed_offset=17
        )
        result["bootstrap"] = {
            "draws": config.bootstrap_draws,
            "ordinary_coverage": config.ordinary_coverage,
            "seven_family_adjusted_coverage": config.adjusted_coverage,
            "models": bootstrap_summary,
            "conditioning": "frozen OOF folds and fitted predictions",
        }
        result["timing_seconds"]["bootstrap"] = time.perf_counter() - boot_started
        result["family_incremental"] = _family_incremental(
            family_names,
            "joint",
            drop_names,
            bootstrap_raw,
            point_metrics,
            "AUC",
            "higher",
            relative_full,
            relative_drops if relative_full else None,
            config.adjusted_coverage,
            config.ordinary_coverage,
        )
        # Only the three frozen L1 views enter the group-sign-swap null.
        perm_datasets: dict[str, tuple[list[dict[str, Any]], np.ndarray, np.ndarray, np.ndarray]] = {}
        for view in ("metadata_only", "acoustic_only", "joint"):
            columns, numeric, categorical = _l1_columns(schema, frame, view, protocol=protocol)
            designs = _fold_designs(frame[columns], numeric, categorical, groups, fold_map)
            perm_datasets[view] = (designs, y, weights, groups)
        perm_started = time.perf_counter()
        result["permutation"] = _l1_permutation_test(
            perm_datasets,
            groups,
            {view: result["models"][view]["weighted_OOF_AUC"] for view in perm_datasets},
            config,
        )
        result["timing_seconds"]["permutation"] = time.perf_counter() - perm_started
        lodo_started = time.perf_counter()
        result["cross_domain_lodo"] = _lodo_l1(schema, protocol, frame, config)
        result["timing_seconds"]["cross_domain_lodo"] = time.perf_counter() - lodo_started
    result["timing_seconds"]["total"] = time.perf_counter() - started
    return result, oof_frames


def _context_baseline_x(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str], list[str]]:
    x = pd.DataFrame(index=frame.index)
    for column in L2_CONTEXT:
        x[column] = frame[column].astype(object)
    return x, [], list(L2_CONTEXT)


def _l2_fit_stratum(
    schema: Mapping[str, Any],
    protocol: Mapping[str, Any] | None,
    frame: pd.DataFrame,
    input_summary: Mapping[str, Any],
    fold_map: Mapping[str, int],
    config: AuditConfig,
    score_contract: str,
    contract_fingerprint: str,
    feature_layer: str,
    engineering_only: bool = False,
) -> tuple[dict[str, Any], list[pd.DataFrame]]:
    y = frame["delta_c"].to_numpy(dtype=float)
    groups = frame["source_group"].astype(str).to_numpy()
    pairs = frame["pair_uid"].astype(str).to_numpy()
    weights = _l2_sample_weights(frame)
    key = f"{score_contract}|{contract_fingerprint}|{feature_layer}"
    result: dict[str, Any] = {
        "score_contract": score_contract,
        "contract_fingerprint": contract_fingerprint,
        "feature_layer": feature_layer,
        "stratum_key": key,
        "status": "PASS_ENGINEERING_ONLY" if engineering_only else "PASS",
        "rows": int(len(frame)),
        "pairs": int(frame["pair_uid"].nunique()),
        "groups": int(frame["source_group"].nunique()),
        "input": dict(input_summary),
        "models": {},
        "family_incremental": {},
        "relative_silence_sensitivity": {"status": "NOT_RUN"},
        "auxiliary_score_diagnostics": _l2_auxiliary_score_diagnostics(frame),
        "permutation": None,
        "timing_seconds": {},
    }
    if result["groups"] < config.min_groups or result["pairs"] < config.min_pairs:
        result["status"] = "DESCRIPTIVE_INCONCLUSIVE_BELOW_MINIMUM"
        return result, []

    started = time.perf_counter()
    raw_targets: dict[str, np.ndarray] = {}
    model_weight_predictions: dict[str, np.ndarray] = {}
    oof_frames: list[pd.DataFrame] = []
    model_designs: dict[str, list[dict[str, Any]]] = {}
    model_specs: dict[str, tuple[pd.DataFrame, list[str], list[str], str]] = {}

    def prepare_model(
        model_id: str,
        x: pd.DataFrame,
        numeric: Sequence[str],
        categorical: Sequence[str],
        kind: str,
    ) -> tuple[dict[str, Any], np.ndarray]:
        designs = _fold_designs(
            x, numeric, categorical, groups, fold_map,
            diagnose_collinearity=(model_id == "all_seven_plus_context"),
        )
        fitted = _fit_oof(designs, y, weights, groups, kind, config.seed)
        pred = fitted["prediction"]
        metrics = _weighted_regression_metrics(y, pred, weights)
        model_designs[model_id] = designs
        raw_targets[model_id] = pred
        model_weight_predictions[model_id] = pred
        model_specs[model_id] = (x, list(numeric), list(categorical), kind)
        out = {
            "model": kind,
            **metrics,
            "per_group": _group_metrics(y, pred, weights, groups, "regression"),
            "folds": fitted["folds"],
            "feature_columns": list(x.columns),
            "weights_sum": float(weights.sum()),
        }
        if model_id.startswith("baseline_"):
            out["mae_improvement_vs_training_mean"] = None
        return out, pred

    # Training-group-weighted mean baseline is calculated separately in each outer fold.
    mean_designs = [
        {
            "fold": int(fold),
            "train": np.flatnonzero(np.asarray([fold_map[g] for g in groups]) != fold),
            "test": np.flatnonzero(np.asarray([fold_map[g] for g in groups]) == fold),
            "x_train": np.zeros((int(np.sum(np.asarray([fold_map[g] for g in groups]) != fold)), 0)),
            "x_test": np.zeros((int(np.sum(np.asarray([fold_map[g] for g in groups]) == fold)), 0)),
            "feature_names": [],
            "preprocess_diagnostics": {},
        }
        for fold in sorted(set(fold_map[g] for g in groups))
        if np.any(np.asarray([fold_map[g] for g in groups]) == fold)
    ]
    mean_fitted = _fit_oof(mean_designs, y, weights, groups, "weighted_mean", config.seed)
    raw_targets["baseline_training_mean"] = mean_fitted["prediction"]
    model_weight_predictions["baseline_training_mean"] = mean_fitted["prediction"]
    result["models"]["baseline_training_mean"] = {
        "model": "outer-fold weighted training mean",
        **_weighted_regression_metrics(y, mean_fitted["prediction"], weights),
        "per_group": _group_metrics(y, mean_fitted["prediction"], weights, groups, "regression"),
        "folds": mean_fitted["folds"],
    }

    context_x, context_numeric, context_categorical = _context_baseline_x(frame)
    context_out, context_pred = prepare_model(
        "baseline_context_ridge", context_x, context_numeric, context_categorical, "ridge"
    )
    context_out["mae_improvement_vs_training_mean"] = (
        result["models"]["baseline_training_mean"]["weighted_OOF_MAE"] - context_out["weighted_OOF_MAE"]
        if context_out["weighted_OOF_MAE"] is not None
        else None
    )
    result["models"]["baseline_context_ridge"] = context_out
    context_out["folds"] = [
        {
            "fold": item["fold"],
            "train_rows": item["train_rows"],
            "test_rows": item["test_rows"],
            "train_groups": item["train_groups"],
            "test_groups": item["test_groups"],
            "preprocess": item["preprocess"],
            "feature_names": item["feature_names"],
        }
        for item in context_out["folds"]
    ]

    missing_x, missing_numeric, missing_categorical = _l2_missingness_inputs(
        schema, frame, protocol
    )
    missing_out, missing_pred = prepare_model(
        "missingness_only_plus_context",
        missing_x,
        missing_numeric,
        missing_categorical,
        "ridge",
    )
    mean_mae = result["models"]["baseline_training_mean"]["weighted_OOF_MAE"]
    context_mae = result["models"]["baseline_context_ridge"]["weighted_OOF_MAE"]
    missing_out["mae_improvement_vs_training_mean"] = (
        float(mean_mae - missing_out["weighted_OOF_MAE"])
        if mean_mae is not None and missing_out["weighted_OOF_MAE"] is not None else None
    )
    missing_out["mae_improvement_vs_context"] = (
        float(context_mae - missing_out["weighted_OOF_MAE"])
        if context_mae is not None and missing_out["weighted_OOF_MAE"] is not None else None
    )
    result["models"]["missingness_only_plus_context"] = missing_out

    for view in (
        "metadata_only_plus_context",
        "acoustic_only_plus_context",
        "all_seven_plus_context",
    ):
        x, numeric, categorical, _ = _l2_x(schema, frame, view, protocol=protocol)
        out, pred = prepare_model(view, x, numeric, categorical, "ridge")
        mean_mae = result["models"]["baseline_training_mean"]["weighted_OOF_MAE"]
        ctx_mae = result["models"]["baseline_context_ridge"]["weighted_OOF_MAE"]
        out["mae_improvement_vs_training_mean"] = (
            float(mean_mae - out["weighted_OOF_MAE"])
            if mean_mae is not None and out["weighted_OOF_MAE"] is not None
            else None
        )
        out["mae_improvement_vs_context"] = (
            float(ctx_mae - out["weighted_OOF_MAE"])
            if ctx_mae is not None and out["weighted_OOF_MAE"] is not None
            else None
        )
        result["models"][view] = out

    all_view = "all_seven_plus_context"
    family_names = list(schema["families"].keys())
    drop_names: dict[str, str] = {}
    for family in family_names:
        model_id = f"all_seven_drop_{family}"
        x, numeric, categorical, _ = _l2_x(
            schema, frame, "all_seven_plus_context", drop_family=family, protocol=protocol
        )
        out, _ = prepare_model(model_id, x, numeric, categorical, "ridge")
        full_mae = result["models"][all_view]["weighted_OOF_MAE"]
        out["mae_improvement_vs_full"] = (
            float(out["weighted_OOF_MAE"] - full_mae)
            if out["weighted_OOF_MAE"] is not None and full_mae is not None
            else None
        )
        result["models"][model_id] = out
        drop_names[family] = model_id

    tree_x, tree_num, tree_cat, _ = _l2_x(
        schema, frame, "all_seven_plus_context", protocol=protocol
    )
    tree_out, _ = prepare_model(
        "all_seven_tree_depth2", tree_x, tree_num, tree_cat, "tree_regressor"
    )
    result["models"]["all_seven_tree_depth2"] = tree_out

    relative_fields = _relative_map(protocol)
    relative_missing = sorted(
        f"{arm}__{column}"
        for column in relative_fields.values()
        for arm in ("natural", "tts")
        if f"{arm}__{column}" not in frame.columns
    )
    relative_drop_names: dict[str, str] = {}
    relative_full: str | None = None
    if relative_missing:
        result["relative_silence_sensitivity"] = {
            "status": "CONTRACT_ERROR_RELATIVE_COLUMNS_MISSING",
            "missing_columns": relative_missing,
        }
    else:
        for view in ("acoustic_only_plus_context", "all_seven_plus_context"):
            model_id = view.replace("_plus_context", "_relative_plus_context")
            x, numeric, categorical, _ = _l2_x(
                schema, frame, view, relative=True, protocol=protocol
            )
            out, _ = prepare_model(model_id, x, numeric, categorical, "ridge")
            result["models"][model_id] = out
            if view == "all_seven_plus_context":
                relative_full = model_id
        for family in family_names:
            model_id = f"all_seven_relative_drop_{family}"
            x, numeric, categorical, _ = _l2_x(
                schema,
                frame,
                "all_seven_plus_context",
                drop_family=family,
                relative=True,
                protocol=protocol,
            )
            out, _ = prepare_model(model_id, x, numeric, categorical, "ridge")
            result["models"][model_id] = out
            relative_drop_names[family] = model_id
        result["relative_silence_sensitivity"] = {
            "status": "PASS",
            "same_fold_and_row_support": True,
            "required_alternate_columns": relative_fields,
            "all_seven_mae": result["models"][relative_full]["weighted_OOF_MAE"],
        }

    all_predictions = {
        name: pred for name, pred in raw_targets.items()
    }
    point_metrics = {
        name: {
            "MAE": model.get("weighted_OOF_MAE"),
            "R2": model.get("weighted_OOF_R2"),
        }
        for name, model in result["models"].items()
    }
    if engineering_only:
        result["bootstrap"] = {"status": "NOT_RUN_ENGINEERING_ONLY", "draws": 0}
        result["family_incremental"] = _engineering_family_incremental(
            family_names, all_view, drop_names, point_metrics, "MAE", "lower",
            relative_full, relative_drop_names if relative_full else None,
        )
        result["permutation"] = {"status": "NOT_RUN_ENGINEERING_ONLY", "draws": 0}
        result["cross_domain_lodo"] = {"status": "NOT_RUN_ENGINEERING_ONLY"}
    else:
        boot_started = time.perf_counter()
        bootstrap_summary, bootstrap_raw = _bootstrap_metrics_by_model(
            y, all_predictions, weights, groups, "regression", config, seed_offset=31
        )
        result["bootstrap"] = {
            "draws": config.bootstrap_draws,
            "ordinary_coverage": config.ordinary_coverage,
            "seven_family_adjusted_coverage": config.adjusted_coverage,
            "models": bootstrap_summary,
            "conditioning": "frozen OOF folds and fitted predictions",
        }
        result["timing_seconds"]["bootstrap"] = time.perf_counter() - boot_started
        result["family_incremental"] = _family_incremental(
            family_names,
            all_view,
            drop_names,
            bootstrap_raw,
            point_metrics,
            "MAE",
            "lower",
            relative_full,
            relative_drop_names if relative_full else None,
            config.adjusted_coverage,
            config.ordinary_coverage,
        )
        perm_started = time.perf_counter()
        result["permutation"] = _l2_permutation_test(
            frame.reset_index(drop=True),
            model_designs[all_view],
            y,
            weights,
            result["models"][all_view]["weighted_OOF_R2"],
            config,
        )
        result["timing_seconds"]["permutation"] = time.perf_counter() - perm_started
        lodo_started = time.perf_counter()
        result["cross_domain_lodo"] = _lodo_l2(schema, protocol, frame, config)
        result["timing_seconds"]["cross_domain_lodo"] = time.perf_counter() - lodo_started
    for model_id, prediction in raw_targets.items():
        prediction_frame = pd.DataFrame(
            {
                "scope": "L2",
                "stratum": key,
                "model_id": model_id,
                "score_uid": frame["score_uid"].astype(str).to_numpy(),
                "pair_uid": pairs,
                "source_group": groups,
                "dataset": frame["dataset"].astype(str).to_numpy(),
                "tfg": frame["tfg"].astype(str).to_numpy(),
                "tts_provider": frame["tts_provider"].astype(str).to_numpy(),
                "score_contract": frame["score_contract"].astype(str).to_numpy(),
                "feature_layer": frame["feature_layer"].astype(str).to_numpy(),
                "y": y,
                "prediction": prediction,
                "sample_weight": weights,
                "fold": [fold_map[group] for group in groups],
            }
        )
        oof_frames.append(prediction_frame)
    result["timing_seconds"]["total"] = time.perf_counter() - started
    return result, oof_frames


def _protocol_stage_check(protocol: Mapping[str, Any]) -> None:
    status = str(protocol.get("status", "")).upper()
    if "PENDING" in status or "DRAFT" in status or not status:
        raise AuditInputError("final cohort/protocol binding is still pending; no formal fit allowed")
    if not any(key in protocol for key in ("cohort_binding", "cohort", "inputs")):
        raise AuditInputError("protocol must include final cohort/input binding before fit")


def run_analysis(
    schema: Mapping[str, Any],
    addendum: Mapping[str, Any],
    protocol: Mapping[str, Any],
    assets: pd.DataFrame,
    features: pd.DataFrame,
    scores: pd.DataFrame,
    allow_pending_for_synthetic: bool = False,
    engineering_only: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    """Run fixed OOF analyses on already-qualified tables; CLI guards real cohorts."""
    if not allow_pending_for_synthetic:
        _protocol_stage_check(protocol)
    if engineering_only:
        execution = protocol.get("execution", {})
        if not isinstance(execution, Mapping) or execution.get("stage_authorized") != "smoke_only":
            raise AuditInputError("protocol does not authorize an engineering smoke")
        if execution.get("full_analysis_authorized") is not False:
            raise AuditInputError("engineering smoke requires full analysis to remain unauthorized")
    config = _resolve_config(schema, addendum, protocol)
    validation = validate_input_tables(
        schema, assets, features, scores, relative_fields=config.relative_fields
    )
    joined = join_assets_features(schema, assets, features)
    # The asset manifest can legitimately lack decoded PCM hashes. Collision
    # checks therefore use the UID-validated feature identity after the join,
    # which fills those hashes without changing the missing-asset diagnostics.
    validation["same_pcm_hash_both_conditions"] = _pcm_label_collisions(joined)
    validation["same_pcm_hash_multiple_source_groups"] = _pcm_source_group_collisions(joined)
    validation["pcm_collision_hash_source"] = "uid_validated_joined_feature_pcm_sha256"
    source_group_pcm_collisions = validation["same_pcm_hash_multiple_source_groups"]
    if source_group_pcm_collisions:
        raise AuditInputError(
            "decoded PCM occurs in multiple source groups; stop before fitting and merge those source groups: "
            + json.dumps(source_group_pcm_collisions, ensure_ascii=False, sort_keys=True)
        )
    all_groups = set(assets["source_group"].dropna().astype(str))
    all_groups.update(scores["source_group"].dropna().astype(str))
    fold_map = make_group_fold_map(all_groups, config.max_folds)
    fold_manifest = pd.DataFrame(
        [{"source_group": group, "fold": fold_map[group]} for group in sorted(fold_map)]
    )
    results: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "status": "ENGINEERING_ONLY_SMOKE" if engineering_only else "ANALYSIS_COMPLETE",
        "analysis_stage": "smoke" if engineering_only else "full",
        "inference_status": "NOT_RUN_ENGINEERING_ONLY" if engineering_only else "FROZEN_PROTOCOL_INFERENCE",
        "history_scope": schema.get("history"),
        "validation": validation,
        "configuration": {
            "seed": config.seed,
            "max_folds": config.max_folds,
            "bootstrap_draws": 0 if engineering_only else config.bootstrap_draws,
            "permutation_draws": 0 if engineering_only else config.permutation_draws,
            "frozen_full_run_bootstrap_draws": config.bootstrap_draws,
            "frozen_full_run_permutation_draws": config.permutation_draws,
            "ordinary_coverage": config.ordinary_coverage,
            "seven_family_adjusted_coverage": config.adjusted_coverage,
            "cpu_threads": config.cpu_threads,
            "models": {
                "L1": "LogisticRegression(penalty=l2,C=1,solver=liblinear,max_iter=2000)",
                "L2": "Ridge(alpha=10,fit_intercept=True)",
                "nonlinear": "DecisionTree(max_depth=2,min_samples_leaf=5)",
            },
            "folds": "canonical source-group fold map shared across source layers/contracts/views",
        },
        "l1": {},
        "l2": {},
        "input_counts": {
            "assets": int(len(assets)),
            "features": int(len(features)),
            "scores": int(len(scores)),
            "groups": int(len(all_groups)),
        },
    }
    l1_oof: list[pd.DataFrame] = []
    for layer in sorted(set(joined["source_layer"].dropna().astype(str))):
        frame, pair_count = _complete_l1_pairs(joined, layer)
        if frame.empty:
            results["l1"][layer] = {
                "layer": layer,
                "status": "UNAVAILABLE_NO_COMPLETE_PAIRS",
                "complete_pairs": 0,
                "rows": 0,
            }
            continue
        result, rows = _l1_fit_stratum(
            schema, protocol, frame, pair_count, fold_map, config, layer,
            engineering_only=engineering_only,
        )
        results["l1"][layer] = result
        l1_oof.extend(rows)

    l2_oof: list[pd.DataFrame] = []
    contracts = scores[["score_contract", "contract_fingerprint", "feature_layer"]].drop_duplicates()
    for _, contract in contracts.sort_values(
        ["score_contract", "contract_fingerprint", "feature_layer"]
    ).iterrows():
        contract_name = str(contract["score_contract"])
        fingerprint = str(contract["contract_fingerprint"])
        layer = str(contract["feature_layer"])
        frame, summary = build_l2_frame(
            schema, scores, joined, contract_name, fingerprint, layer,
            relative_fields=config.relative_fields,
        )
        key = f"{contract_name}|{fingerprint}|{layer}"
        if frame.empty:
            results["l2"][key] = {
                "status": "UNAVAILABLE_NO_COMPLETE_SCORE_PAIRS",
                "score_contract": contract_name,
                "contract_fingerprint": fingerprint,
                "feature_layer": layer,
                "input": summary,
            }
            continue
        result, rows = _l2_fit_stratum(
            schema,
            protocol,
            frame,
            summary,
            fold_map,
            config,
            contract_name,
            fingerprint,
            layer,
            engineering_only=engineering_only,
        )
        results["l2"][key] = result
        l2_oof.extend(rows)
    return (
        results,
        pd.concat(l1_oof, ignore_index=True) if l1_oof else pd.DataFrame(),
        pd.concat(l2_oof, ignore_index=True) if l2_oof else pd.DataFrame(),
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if pd.isna(value) if not isinstance(value, (dict, list, tuple)) else False:
        return None
    return value
