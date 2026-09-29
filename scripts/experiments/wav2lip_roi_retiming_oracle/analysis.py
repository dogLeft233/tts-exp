from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import OracleError, file_sha256


def peak(curve: Sequence[float]) -> dict[str, Any]:
    values = np.asarray(curve, dtype=np.float64)
    if values.shape != (2 * config.VSHIFT + 1,) or not np.isfinite(values).all():
        raise OracleError(f"SyncNet curve must be finite with shape [31], got {values.shape}")
    order = np.sort(values, kind="stable")
    min_index = int(np.argmin(values))
    offset = config.VSHIFT - min_index
    gap = float(order[1] - order[0])
    return {
        "min_index": min_index,
        "offset": int(offset),
        "sync_d": float(order[0]),
        "sync_c": float(np.median(values) - order[0]),
        "minimum": float(order[0]),
        "second_minimum": float(order[1]),
        "peak_gap": gap,
        "clear": bool(gap > config.PEAK_GAP_THRESHOLD and abs(offset) < config.VSHIFT),
    }


def summarize(matrix: np.ndarray, rows: Sequence[int]) -> dict[str, Any]:
    value = np.asarray(matrix, dtype=np.float64)
    selected = [int(row) for row in rows]
    if value.ndim != 2 or value.shape[1] != 31 or not np.isfinite(value).all():
        raise OracleError(f"distance matrix is malformed: {value.shape}")
    if not selected or min(selected) < 0 or max(selected) >= value.shape[0]:
        raise OracleError(f"mask rows are outside distance matrix: {selected[:3]} / {value.shape[0]}")
    curve = np.mean(value[selected, :], axis=0, dtype=np.float64)
    evidence = peak(curve)
    evidence["rows"] = selected
    evidence["curve"] = [float(item) for item in curve]
    return evidence


def load_matrix(row: Mapping[str, Any]) -> np.ndarray:
    path = Path(str(row.get("matrix", "")))
    expected = str(row.get("matrix_sha256", ""))
    if not path.is_file() or not expected or file_sha256(path) != expected:
        raise OracleError(f"matrix binding is invalid: {path}")
    matrix = np.asarray(np.load(path, allow_pickle=False), dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 31 or matrix.shape[0] < 1 or not np.isfinite(matrix).all():
        raise OracleError(f"matrix is malformed: {path}")
    return matrix


def _cell_index(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], Mapping[str, Any]]:
    result: dict[tuple[str, str, str], Mapping[str, Any]] = {}
    for row in rows:
        key = (str(row.get("sample_id")), str(row.get("video_arm")), str(row.get("audio_arm")))
        if key in result:
            raise OracleError(f"duplicate score cell: {key}")
        result[key] = row
    return result


def _record_map(parent: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {str(row["sample_id"]): row for row in parent["records"]}


def _protocol_record(protocol: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    for row in protocol.get("records", []):
        if isinstance(row, Mapping) and str(row.get("sample_id")) == sample_id:
            return row
    raise OracleError(f"protocol record is missing: {sample_id}")


def _parent_score(parent: Mapping[str, Any], sample_id: str, video: str, audio: str) -> np.ndarray:
    row = parent["score_index"].get((sample_id, video, audio, False))
    if not isinstance(row, Mapping):
        raise OracleError(f"parent score cell is missing: {sample_id}/{video}/{audio}")
    matrix = parent["matrix_cache"].get(str(row.get("matrix_sha256")))
    if isinstance(matrix, np.ndarray):
        return np.asarray(matrix, dtype=np.float64)
    return load_matrix(row)


def _signature(matrix: np.ndarray, masks: Mapping[str, Any]) -> dict[str, Any]:
    common = summarize(matrix, masks["common_window_rows"])
    plus = summarize(matrix, masks["plus_rows"])
    minus = summarize(matrix, masks["minus_rows"])
    return {"common": common, "PLUS": plus, "MINUS": minus}


def _baseline_pass(signature: Mapping[str, Any]) -> bool:
    plus = signature["PLUS"]
    minus = signature["MINUS"]
    return bool(
        plus["clear"]
        and minus["clear"]
        and abs(int(plus["offset"]) - int(minus["offset"])) <= config.OFFSET_TOLERANCE_FRAMES
    )


def _timing_check(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    masks: Mapping[str, Any],
    expected_key: str,
) -> dict[str, Any]:
    expected_by_segment = {
        "B": {
            "PLUS": float(np.mean([masks["a_by_row"][str(row)] for row in masks["plus_rows"]])),
            "MINUS": float(np.mean([masks["a_by_row"][str(row)] for row in masks["minus_rows"]])),
        },
        "C_oracle": {
            "PLUS": float(-np.mean([masks["d_by_row"][str(row)] for row in masks["plus_rows"]])),
            "MINUS": float(-np.mean([masks["d_by_row"][str(row)] for row in masks["minus_rows"]])),
        },
        "O_oracle": {"PLUS": 0.0, "MINUS": 0.0},
    }
    result: dict[str, Any] = {}
    for segment in ("PLUS", "MINUS"):
        actual = float(left[segment]["offset"] - right[segment]["offset"])
        expected = expected_by_segment[expected_key][segment]
        result[segment] = {
            "actual": actual,
            "expected": expected,
            "error": actual - expected,
            "passes": bool(
                left[segment]["clear"]
                and right[segment]["clear"]
                and abs(actual - expected) <= config.OFFSET_TOLERANCE_FRAMES
            ),
        }
    result["passes"] = bool(result["PLUS"]["passes"] and result["MINUS"]["passes"])
    return result


def _metric_delta(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        "c": float(left["common"]["sync_c"] - right["common"]["sync_c"]),
        "d": float(left["common"]["sync_d"] - right["common"]["sync_d"]),
        "offset": int(left["common"]["offset"] - right["common"]["offset"]),
    }


def _bootstrap(values: Sequence[float], groups: Sequence[str]) -> dict[str, Any]:
    if len(values) != len(groups) or not values:
        raise OracleError("invalid bootstrap input")
    grouped: dict[str, list[float]] = defaultdict(list)
    for group, value in zip(groups, values, strict=True):
        if not np.isfinite(float(value)):
            raise OracleError("bootstrap value is not finite")
        grouped[str(group)].append(float(value))
    labels = sorted(grouped)
    means = {label: float(np.mean(grouped[label])) for label in labels}
    rng = np.random.default_rng(config.BOOTSTRAP_SEED)
    estimates = np.empty(config.BOOTSTRAP_DRAWS, dtype=np.float64)
    for index in range(config.BOOTSTRAP_DRAWS):
        sampled = rng.choice(labels, size=len(labels), replace=True)
        estimates[index] = float(np.mean([means[str(label)] for label in sampled]))
    return {
        "draws": config.BOOTSTRAP_DRAWS,
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy_default_rng_pcg64",
        "quantile_method": "linear",
        "record_count": len(values),
        "source_group_count": len(labels),
        "mean": float(statistics.fmean(float(value) for value in values)),
        "ci95": [
            float(np.quantile(estimates, 0.025, method="linear")),
            float(np.quantile(estimates, 0.975, method="linear")),
        ],
        "group_means": means,
    }


def _history_c_status(parent: Mapping[str, Any]) -> dict[str, bool]:
    result: dict[str, bool] = {}
    for row in parent["payloads"]["control"].get("per_record", []):
        if isinstance(row, Mapping):
            checks = row.get("checks")
            c = checks.get("C") if isinstance(checks, Mapping) else None
            if isinstance(c, Mapping) and isinstance(c.get("passes"), bool):
                result[str(row["sample_id"])] = bool(c["passes"])
    return result


def _comparison(old: np.ndarray, new: np.ndarray, sample_id: str, audio_arm: str) -> dict[str, Any]:
    if old.shape != new.shape:
        return {
            "sample_id": sample_id,
            "audio_arm": audio_arm,
            "shape_equal": False,
            "max_abs": None,
            "pass": False,
        }
    delta = np.abs(old - new)
    location = np.unravel_index(int(np.argmax(delta)), delta.shape)
    return {
        "sample_id": sample_id,
        "audio_arm": audio_arm,
        "shape_equal": True,
        "max_abs": float(np.max(delta)),
        "tolerance": config.MATRIX_DIFF_TOLERANCE,
        "argmax": {"row": int(location[0]), "column": int(location[1])},
        "old_value": float(old[location]),
        "new_value": float(new[location]),
        "pass": bool(np.max(delta) <= config.MATRIX_DIFF_TOLERANCE),
    }


def analyze(parent: Mapping[str, Any], protocol: Mapping[str, Any], score_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells = _cell_index(score_rows)
    expected = {
        (str(record["sample_id"]), video, audio)
        for record in parent["records"]
        for video, audio in config.CELL_SPECS
    }
    if set(cells) != expected:
        raise OracleError(f"new score cell set differs: {len(cells)}/{len(expected)}")
    records: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    history = _history_c_status(parent)
    for parent_record in parent["records"]:
        sample_id = str(parent_record["sample_id"])
        protocol_record = _protocol_record(protocol, sample_id)
        masks = protocol_record["masks"]
        new = {
            (video, audio): load_matrix(cells[(sample_id, video, audio)])
            for video, audio in config.CELL_SPECS
        }
        old_n = _parent_score(parent, sample_id, "G_N", "N")
        old_w = _parent_score(parent, sample_id, "G_N", "W")
        comparisons.extend(
            [
                _comparison(old_n, new["V_ID", "N"], sample_id, "N"),
                _comparison(old_w, new["V_ID", "W"], sample_id, "W"),
            ]
        )
        signatures = {key: _signature(value, masks) for key, value in new.items()}
        old_signatures = {
            "N": _signature(old_n, masks),
            "W": _signature(old_w, masks),
        }
        identity_reproduction = {
            "N": {
                "matrix_pass": comparisons[-2]["pass"],
                "offsets_equal": all(
                    signatures["V_ID", "N"][name]["offset"] == old_signatures["N"][name]["offset"]
                    for name in ("common", "PLUS", "MINUS")
                ),
                "clear_equal": all(
                    signatures["V_ID", "N"][name]["clear"] == old_signatures["N"][name]["clear"]
                    for name in ("common", "PLUS", "MINUS")
                ),
            },
            "W": {
                "matrix_pass": comparisons[-1]["pass"],
                "offsets_equal": all(
                    signatures["V_ID", "W"][name]["offset"] == old_signatures["W"][name]["offset"]
                    for name in ("common", "PLUS", "MINUS")
                ),
                "clear_equal": all(
                    signatures["V_ID", "W"][name]["clear"] == old_signatures["W"][name]["clear"]
                    for name in ("common", "PLUS", "MINUS")
                ),
            },
        }
        identity_pass = all(
            bool(item["matrix_pass"] and item["offsets_equal"] and item["clear_equal"])
            for item in identity_reproduction.values()
        )
        baseline = _baseline_pass(signatures["V_ID", "N"])
        checks = {
            "B": _timing_check(signatures["V_ID", "W"], signatures["V_ID", "N"], masks, "B"),
            "C_oracle": _timing_check(signatures["V_ORACLE", "N"], signatures["V_ID", "N"], masks, "C_oracle"),
            "O_oracle": _timing_check(signatures["V_ORACLE", "W"], signatures["V_ID", "N"], masks, "O_oracle"),
        }
        own = {
            "c": float(signatures["V_ORACLE", "W"]["common"]["sync_c"] - signatures["V_ID", "N"]["common"]["sync_c"]),
            "d": float(signatures["V_ID", "N"]["common"]["sync_d"] - signatures["V_ORACLE", "W"]["common"]["sync_d"]),
            "offset_agreement": bool(
                abs(signatures["V_ORACLE", "W"]["common"]["offset"] - signatures["V_ID", "N"]["common"]["offset"])
                <= config.OFFSET_TOLERANCE_FRAMES
            ),
        }
        damage = {
            "c": float(signatures["V_ORACLE", "W"]["common"]["sync_c"] - signatures["V_ORACLE", "N"]["common"]["sync_c"]),
            "d": float(signatures["V_ORACLE", "N"]["common"]["sync_d"] - signatures["V_ORACLE", "W"]["common"]["sync_d"]),
            "both_positive": bool(
                signatures["V_ORACLE", "W"]["common"]["sync_c"] > signatures["V_ORACLE", "N"]["common"]["sync_c"]
                and signatures["V_ORACLE", "N"]["common"]["sync_d"] > signatures["V_ORACLE", "W"]["common"]["sync_d"]
            ),
        }
        records.append(
            {
                "sample_id": sample_id,
                "source_group": str(parent_record["source_group"]),
                "historical_c_pass": history.get(sample_id),
                "identity_reproduction": identity_reproduction,
                "identity_pass": identity_pass,
                "baseline": baseline,
                "checks": checks,
                "own": own,
                "damage": damage,
                # Keep tuple keys for in-memory lookup above, but serialize
                # the public artifact with the same stable cell-key format
                # used by media and score manifests.
                "signatures": {
                    f"{video}__{audio}": value
                    for (video, audio), value in signatures.items()
                },
                "interpretation": "oracle 是已知像素重排参照，不等同于实际 G_W 生成响应",
            }
        )

    baseline_count = sum(bool(row["baseline"]) for row in records)
    timing_counts = {
        name: sum(bool(row["checks"][name]["passes"]) for row in records)
        for name in ("B", "C_oracle", "O_oracle")
    }
    groups = [str(row["source_group"]) for row in records]
    own_c = [float(row["own"]["c"]) for row in records]
    own_d = [float(row["own"]["d"]) for row in records]
    damage_c = [float(row["damage"]["c"]) for row in records]
    damage_d = [float(row["damage"]["d"]) for row in records]
    own_bootstrap = {"c": _bootstrap(own_c, groups), "d": _bootstrap(own_d, groups)}
    damage_bootstrap = {"c": _bootstrap(damage_c, groups), "d": _bootstrap(damage_d, groups)}
    own_gate = {
        "ci_lower_gt_negative_0_10": bool(
            own_bootstrap["c"]["ci95"][0] > -0.10 and own_bootstrap["d"]["ci95"][0] > -0.10
        ),
        "offset_agreement_count": sum(bool(row["own"]["offset_agreement"]) for row in records),
        "offset_agreement_pass": sum(bool(row["own"]["offset_agreement"]) for row in records) >= config.MIN_BASELINE_RECORDS,
    }
    own_gate["passes"] = bool(own_gate["ci_lower_gt_negative_0_10"] and own_gate["offset_agreement_pass"])
    damage_gate = {
        "ci_lower_gt_0_10": bool(
            damage_bootstrap["c"]["ci95"][0] > 0.10 and damage_bootstrap["d"]["ci95"][0] > 0.10
        ),
        "both_positive_count": sum(bool(row["damage"]["both_positive"]) for row in records),
        "both_positive_pass": sum(bool(row["damage"]["both_positive"]) for row in records) >= config.MIN_SUCCESS_RECORDS,
    }
    damage_gate["passes"] = bool(damage_gate["ci_lower_gt_0_10"] and damage_gate["both_positive_pass"])
    identity_pass = all(bool(row["identity_pass"]) for row in records)
    if not identity_pass:
        decision = "BASELINE_NOT_REPRODUCED"
    elif baseline_count < config.MIN_BASELINE_RECORDS or any(
        count < config.MIN_SUCCESS_RECORDS for count in timing_counts.values()
    ):
        decision = "ORACLE_TIMING_UNRESOLVED"
    elif not own_gate["passes"]:
        decision = "ORACLE_OWN_AUDIO_UNRESOLVED"
    elif not damage_gate["passes"]:
        decision = "ORACLE_DAMAGE_UNRESOLVED"
    else:
        decision = "ORACLE_CONTROL_SUPPORTED"
    return {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "decision": decision,
        "record_count": len(records),
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "score_cell_count": len(score_rows),
        "expected_score_cell_count": config.EXPECTED_SCORE_CELL_COUNT,
        "identity_reproduction": {
            "pass": identity_pass,
            "comparisons": comparisons,
            "max_abs_difference": float(max((item["max_abs"] or 0.0) for item in comparisons)),
        },
        "baseline_count": baseline_count,
        "baseline_minimum": config.MIN_BASELINE_RECORDS,
        "timing_counts": timing_counts,
        "timing_minimum": config.MIN_SUCCESS_RECORDS,
        "own_bootstrap": own_bootstrap,
        "own_gate": own_gate,
        "damage_bootstrap": damage_bootstrap,
        "damage_gate": damage_gate,
        "per_record": records,
        "causal_boundary": "V_ORACLE 是 G_N 的已知像素重定时参照；即使通过，也不能把父 G_W 控制或 replacement 效应解释为成立。",
        "parent_own_audio_retested": False,
        "oracle_own_audio_tested": True,
    }
