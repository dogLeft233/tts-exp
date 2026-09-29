from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, csv_write, finite_array, write_json


def score_row(model: str, sample_id: int, condition: str, c: Any, d: Any, source_group: str, source_path: str | None = None) -> dict[str, Any]:
    if model not in config.MODELS or condition not in config.CONDITIONS:
        raise ProtocolError(f"unknown score identity: {model}/{sample_id}/{condition}")
    c_value = float(c)
    d_value = float(d)
    if not math.isfinite(c_value) or not math.isfinite(d_value):
        raise ProtocolError(f"non-finite score: {model}/{sample_id}/{condition}")
    return {
        "model": model,
        "sample_id": int(sample_id),
        "condition": condition,
        "source_group": str(source_group),
        "sync_c": c_value,
        "sync_d": d_value,
        "background_b_hat": c_value + d_value,
        "source_path": source_path,
    }


def decompose_pair(natural: Mapping[str, Any], tts: Mapping[str, Any]) -> dict[str, Any]:
    if natural.get("condition") != "natural_raw" or tts.get("condition") != "tts_raw":
        raise ProtocolError("pair must be natural_raw followed by tts_raw")
    if natural.get("model") != tts.get("model") or int(natural["sample_id"]) != int(tts["sample_id"]):
        raise ProtocolError("pair identity mismatch")
    if str(natural.get("source_group")) != str(tts.get("source_group")):
        raise ProtocolError("pair source-group mismatch")
    c_n = float(natural["sync_c"])
    c_t = float(tts["sync_c"])
    d_n = float(natural["sync_d"])
    d_t = float(tts["sync_d"])
    b_n = float(natural["background_b_hat"])
    b_t = float(tts["background_b_hat"])
    gain_c = c_t - c_n
    gain_match = d_n - d_t
    gain_background = b_t - b_n
    residual = gain_c - gain_match - gain_background
    if abs(residual) > 1e-10:
        raise ProtocolError(f"decomposition identity failed: residual={residual}")
    return {
        "model": str(natural["model"]),
        "sample_id": int(natural["sample_id"]),
        "source_group": str(natural["source_group"]),
        "natural_c": c_n,
        "tts_c": c_t,
        "natural_d": d_n,
        "tts_d": d_t,
        "natural_b_hat": b_n,
        "tts_b_hat": b_t,
        "gain_c": gain_c,
        "gain_match": gain_match,
        "gain_background": gain_background,
        "decomposition_residual": residual,
        "c_positive": gain_c > 0.0,
    }


def group_pairs(pairs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in pairs:
        grouped[(str(row["model"]), str(row["source_group"]))].append(row)
    result: list[dict[str, Any]] = []
    metrics = ("gain_c", "gain_match", "gain_background")
    for (model, source_group), rows in sorted(grouped.items()):
        item: dict[str, Any] = {
            "model": model,
            "source_group": source_group,
            "record_count": len(rows),
            "sample_ids": [int(row["sample_id"]) for row in rows],
        }
        for metric in metrics:
            item[metric] = float(np.mean([float(row[metric]) for row in rows], dtype=np.float64))
        item["c_positive_count"] = int(sum(bool(row["c_positive"]) for row in rows))
        result.append(item)
    return result


def bootstrap_indices(group_count: int, *, seed: int = config.BOOTSTRAP_SEED, draws: int = config.BOOTSTRAP_DRAWS) -> np.ndarray:
    if group_count < 1 or draws < 1:
        raise ProtocolError("bootstrap dimensions must be positive")
    return np.random.default_rng(seed).integers(0, group_count, size=(draws, group_count), dtype=np.int64)


def _quantile(values: np.ndarray, q: float) -> float:
    try:
        return float(np.quantile(values, q, method="linear"))
    except TypeError:
        return float(np.quantile(values, q, interpolation="linear"))


def bootstrap_summary(
    group_values: Mapping[str, float],
    indices: np.ndarray,
    *,
    metric: str,
    bonferroni_comparisons: int = 4,
) -> dict[str, Any]:
    labels = sorted(group_values)
    if not labels:
        raise ProtocolError(f"no groups for {metric}")
    if bonferroni_comparisons < 1:
        raise ProtocolError("Bonferroni comparison count must be positive")
    index_array = np.asarray(indices, dtype=np.int64)
    if index_array.ndim != 2 or index_array.shape[1] != len(labels) or np.any(index_array < 0) or np.any(index_array >= len(labels)):
        raise ProtocolError(f"bootstrap indices do not match the group set for {metric}")
    values = np.asarray([float(group_values[label]) for label in labels], dtype=np.float64)
    sampled = values[index_array].mean(axis=1, dtype=np.float64)
    tail_probability = 0.05 / (2.0 * bonferroni_comparisons)
    bonferroni_interval = [_quantile(sampled, tail_probability), _quantile(sampled, 1.0 - tail_probability)]
    interval_name = "ci98_75" if bonferroni_comparisons == 4 else "ci98_333" if bonferroni_comparisons == 3 else "ci_bonferroni"
    return {
        "metric": metric,
        "mean": float(values.mean(dtype=np.float64)),
        "ci95": [_quantile(sampled, 0.025), _quantile(sampled, 0.975)],
        interval_name: bonferroni_interval,
        "ci_bonferroni": bonferroni_interval,
        "bonferroni_comparisons": int(bonferroni_comparisons),
        "bonferroni_tail_probability": tail_probability,
        "group_labels": labels,
        "group_means": {label: float(value) for label, value in zip(labels, values, strict=True)},
        "group_count": len(labels),
        "draws": int(indices.shape[0]),
        "seed": config.BOOTSTRAP_SEED,
        "rng": "numpy.default_rng(PCG64)",
        "quantile_method": "linear",
    }


def historical_analysis(pairs: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    group_count = len({str(row["source_group"]) for row in groups})
    if group_count != 45:
        raise ProtocolError(f"expected 45 source groups, found {group_count}")
    indices = bootstrap_indices(group_count)
    metrics = ("gain_c", "gain_match", "gain_background")
    summaries: dict[str, Any] = {}
    for model in config.MODELS:
        model_groups = [row for row in groups if row["model"] == model]
        values = {str(row["source_group"]): float(row["gain_c"]) for row in model_groups}
        summaries[f"{model}.gain_c"] = bootstrap_summary(values, indices, metric=f"{model}.gain_c")
        for metric in metrics[1:]:
            values = {str(row["source_group"]): float(row[metric]) for row in model_groups}
            summaries[f"{model}.{metric}"] = bootstrap_summary(values, indices, metric=f"{model}.{metric}")
    record_means: dict[str, Any] = {}
    for model in config.MODELS:
        rows = [row for row in pairs if row["model"] == model]
        record_means[model] = {
            "n": len(rows),
            "gain_c": float(np.mean([float(row["gain_c"]) for row in rows], dtype=np.float64)),
            "benefit_d": float(np.mean([float(row["gain_match"]) for row in rows], dtype=np.float64)),
            "gain_background": float(np.mean([float(row["gain_background"]) for row in rows], dtype=np.float64)),
            "c_positive": int(sum(bool(row["c_positive"]) for row in rows)),
        }
    return {
        "status": "complete",
        "estimand": "record_pair_then_source_group_mean_then_equal_group_bootstrap",
        "record_count": len(pairs) // 2,
        "pair_count": len(pairs),
        "source_group_count": group_count,
        "record_means": record_means,
        "group_summaries": summaries,
        "primary_components": ["gain_match", "gain_background"],
        "bonferroni": {"comparisons": 4, "interval": "98.75%", "tail_probability": 0.00625},
        "decomposition_identity_max_abs": float(max(abs(float(row["decomposition_residual"])) for row in pairs)),
    }


def _curve_metrics(matrix: np.ndarray, rows: Sequence[int], *, label: str) -> dict[str, Any]:
    value = finite_array(matrix, name=label, ndim=2)
    if value.shape[1] != config.LAG_COUNT:
        raise ProtocolError(f"{label} must have shape [T,31], got {value.shape}")
    selected = np.asarray(list(rows), dtype=np.int64)
    if selected.size < 1 or int(selected.min()) < 0 or int(selected.max()) >= value.shape[0]:
        raise ProtocolError(f"{label} support is out of range")
    # SyncNet's forward distances are float32.  Keep the official reduction order:
    # mean over time, then median/min over the 31 lag columns.
    curve = np.asarray(value[selected].astype(np.float32).mean(axis=0, dtype=np.float32), dtype=np.float64)
    min_index = int(np.argmin(curve))
    d_value = float(curve[min_index])
    b_value = float(np.median(curve))
    c_value = b_value - d_value
    d0 = float(curve[config.VSHIFT])
    half = d_value + c_value / 2.0
    if c_value <= 1e-6:
        width_ms: float | None = None
        width_status = "flat_curve"
    else:
        left = min_index
        right = min_index
        while left > 0 and float(curve[left - 1]) <= half:
            left -= 1
        while right + 1 < config.LAG_COUNT and float(curve[right + 1]) <= half:
            right += 1
        width_ms = float((right - left + 1) * 40.0)
        width_status = "censored" if left == 0 or right == config.LAG_COUNT - 1 else "complete"
    c5 = float(np.median(curve[10:21]) - np.min(curve[10:21]))
    return {
        "support_rows": [int(item) for item in selected],
        "support_count": int(selected.size),
        "curve": [float(item) for item in curve],
        "offsets": [config.VSHIFT - index for index in range(config.LAG_COUNT)],
        "min_index": min_index,
        "official_offset": config.VSHIFT - min_index,
        "sync_d": d_value,
        "sync_c": c_value,
        "background_b": b_value,
        "d0": d0,
        "search_gain_s": d0 - d_value,
        "c5": c5,
        "boundary_best": bool(min_index in (0, config.LAG_COUNT - 1)),
        "trough_width_ms": width_ms,
        "trough_width_status": width_status,
        "trough_ties": [int(index) for index, item in enumerate(curve) if float(item) == d_value],
    }


def equal_count_rows(interior_count: int, *, target_count: int) -> list[int]:
    if interior_count < 1 or target_count < 1 or target_count > interior_count:
        raise ProtocolError("equal-count target is invalid")
    if target_count == interior_count:
        return list(range(interior_count))
    result = np.floor(np.linspace(0, interior_count - 1, target_count)).astype(np.int64).tolist()
    if len(set(result)) != target_count:
        raise ProtocolError("equal-count rows contain duplicates")
    return [int(item) for item in result]


def pair_curve_metrics(natural_matrix: np.ndarray, tts_matrix: np.ndarray, *, sample_id: int, source_group: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    n_value = finite_array(natural_matrix, name=f"natural/{sample_id}", ndim=2)
    t_value = finite_array(tts_matrix, name=f"tts/{sample_id}", ndim=2)
    if n_value.shape[1] != config.LAG_COUNT or t_value.shape[1] != config.LAG_COUNT:
        raise ProtocolError("curve matrices must have 31 columns")
    endpoint_rows: list[dict[str, Any]] = []
    n_full = list(range(n_value.shape[0]))
    t_full = list(range(t_value.shape[0]))
    n_interior = list(range(config.VSHIFT, n_value.shape[0] - config.VSHIFT))
    t_interior = list(range(config.VSHIFT, t_value.shape[0] - config.VSHIFT))
    if len(n_interior) < 25 or len(t_interior) < 25:
        raise ProtocolError(f"INTERIOR support has fewer than 25 rows: {sample_id}")
    target = min(len(n_interior), len(t_interior))
    n_equal = [n_interior[index] for index in equal_count_rows(len(n_interior), target_count=target)]
    t_equal = [t_interior[index] for index in equal_count_rows(len(t_interior), target_count=target)]
    supports = {
        "FULL": (n_full, t_full),
        "INTERIOR": (n_interior, t_interior),
        "EQUAL_COUNT": (n_equal, t_equal),
    }
    for support, (n_rows, t_rows) in supports.items():
        n_metrics = _curve_metrics(n_value, n_rows, label=f"natural/{sample_id}/{support}")
        t_metrics = _curve_metrics(t_value, t_rows, label=f"tts/{sample_id}/{support}")
        for condition, metrics in (("natural_raw", n_metrics), ("tts_raw", t_metrics)):
            endpoint_rows.append({
                "model": "LeapTalk",
                "sample_id": int(sample_id),
                "source_group": str(source_group),
                "condition": condition,
                "support": support,
                **{key: value for key, value in metrics.items() if key not in {"curve", "support_rows", "offsets", "trough_ties"}},
                "curve": metrics["curve"],
                "support_rows": metrics["support_rows"],
                "offsets": metrics["offsets"],
                "trough_ties": metrics["trough_ties"],
            })
    return endpoint_rows, {
        "sample_id": int(sample_id),
        "source_group": str(source_group),
        "n_full_rows": len(n_full),
        "t_full_rows": len(t_full),
        "n_interior_rows": len(n_interior),
        "t_interior_rows": len(t_interior),
        "equal_count": target,
    }


def _endpoint_index(endpoints: Sequence[Mapping[str, Any]]) -> dict[tuple[int, str, str], Mapping[str, Any]]:
    result: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    for row in endpoints:
        key = (int(row["sample_id"]), str(row["condition"]), str(row["support"]))
        if key in result:
            raise ProtocolError(f"duplicate curve endpoint: {key}")
        result[key] = row
    return result


def curve_analysis(endpoints: Sequence[Mapping[str, Any]], source_groups: Mapping[int, str]) -> dict[str, Any]:
    if len({str(source_groups[int(sample_id)]) for sample_id in config.CURVE_SAMPLE_IDS}) != len(config.CURVE_SAMPLE_IDS):
        raise ProtocolError("curve queue must contain one distinct source group per sample")
    index = _endpoint_index(endpoints)
    expected = len(config.CURVE_SAMPLE_IDS) * len(config.CONDITIONS) * 3
    if len(index) != expected:
        raise ProtocolError(f"expected {expected} curve endpoint rows, found {len(index)}")
    paired: list[dict[str, Any]] = []
    for sample_id in config.CURVE_SAMPLE_IDS:
        group = str(source_groups[int(sample_id)])
        for support in ("FULL", "INTERIOR", "EQUAL_COUNT"):
            n = index[(sample_id, "natural_raw", support)]
            t = index[(sample_id, "tts_raw", support)]
            row = {
                "model": "LeapTalk",
                "sample_id": sample_id,
                "source_group": group,
                "support": support,
                "delta_c": float(t["sync_c"]) - float(n["sync_c"]),
                "gain_match": float(n["sync_d"]) - float(t["sync_d"]),
                "gain_background": float(t["background_b"]) - float(n["background_b"]),
                "delta_c5": float(t["c5"]) - float(n["c5"]),
                "delta_d0": float(t["d0"]) - float(n["d0"]),
                "delta_search_gain": float(t["search_gain_s"]) - float(n["search_gain_s"]),
                "delta_width_ms": None if n["trough_width_ms"] is None or t["trough_width_ms"] is None else float(t["trough_width_ms"]) - float(n["trough_width_ms"]),
                "offset_n": int(n["official_offset"]),
                "offset_t": int(t["official_offset"]),
                "boundary_n": bool(n["boundary_best"]),
                "boundary_t": bool(t["boundary_best"]),
            }
            row["decomposition_residual"] = row["delta_c"] - row["gain_match"] - row["gain_background"]
            if abs(float(row["decomposition_residual"])) > 1e-6:
                raise ProtocolError(f"curve decomposition identity failed: {sample_id}/{support}")
            paired.append(row)
    core_specs = (
        ("INTERIOR", "delta_c", "interior_delta_c"),
        ("INTERIOR", "gain_match", "interior_gain_match"),
    )
    full_by_id = {int(row["sample_id"]): row for row in paired if row["support"] == "FULL"}
    interior_by_id = {int(row["sample_id"]): row for row in paired if row["support"] == "INTERIOR"}
    core_values: dict[str, dict[str, float]] = {}
    for support, metric, name in core_specs:
        del support
        core_values[name] = {str(row["source_group"]): float(row[metric]) for row in interior_by_id.values()}
    core_values["interior_minus_full_delta_c"] = {
        str(interior_by_id[sample_id]["source_group"]): float(interior_by_id[sample_id]["delta_c"] - full_by_id[sample_id]["delta_c"])
        for sample_id in config.CURVE_SAMPLE_IDS
    }
    group_indices = bootstrap_indices(len(config.CURVE_SAMPLE_IDS))
    core_summaries = {
        name: bootstrap_summary(values, group_indices, metric=name, bonferroni_comparisons=3)
        for name, values in core_values.items()
    }
    return {
        "status": "complete",
        "record_count": len(config.CURVE_SAMPLE_IDS),
        "source_group_count": len(config.CURVE_SAMPLE_IDS),
        "endpoints_count": len(endpoints),
        "paired_count": len(paired),
        "paired": paired,
        "core_summaries": core_summaries,
        "exploratory_metrics": ["C_5", "D0", "S", "trough_width", "EQUAL_COUNT", "boundary_best", "offset"],
        "bonferroni": {"comparisons": 3, "interval": "98.333333%", "tail_probability": 1.0 / 120.0},
    }


def write_historical_tables(root: Path, cells: Sequence[Mapping[str, Any]], pairs: Sequence[Mapping[str, Any]], groups: Sequence[Mapping[str, Any]]) -> None:
    csv_write(root / "cells.csv", ("model", "sample_id", "source_group", "condition", "sync_c", "sync_d", "background_b_hat", "source_path"), cells)
    csv_write(root / "paired.csv", ("model", "sample_id", "source_group", "natural_c", "tts_c", "natural_d", "tts_d", "natural_b_hat", "tts_b_hat", "gain_c", "gain_match", "gain_background", "decomposition_residual", "c_positive"), pairs)
    csv_write(root / "groups.csv", ("model", "source_group", "record_count", "sample_ids", "gain_c", "gain_match", "gain_background", "c_positive_count"), groups)


def write_curve_tables(root: Path, endpoints: Sequence[Mapping[str, Any]], paired: Sequence[Mapping[str, Any]]) -> None:
    endpoint_rows = []
    for row in endpoints:
        item = dict(row)
        for key in ("curve", "support_rows", "offsets", "trough_ties"):
            item[key] = json_value(item.get(key))
        endpoint_rows.append(item)
    paired_rows = [{key: json_value(value) for key, value in row.items()} for row in paired]
    endpoint_fields = ("model", "sample_id", "source_group", "condition", "support", "support_count", "sync_d", "sync_c", "background_b", "d0", "search_gain_s", "c5", "official_offset", "min_index", "boundary_best", "trough_width_ms", "trough_width_status", "curve", "support_rows", "offsets", "trough_ties")
    paired_fields = tuple(sorted({key for row in paired_rows for key in row}))
    csv_write(root / "endpoints.csv", endpoint_fields, endpoint_rows)
    csv_write(root / "paired.csv", paired_fields, paired_rows)


def json_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        import json

        return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return value


def plot_outputs(root: Path, historical: Mapping[str, Any] | None, curve: Mapping[str, Any] | None) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except (ImportError, OSError, RuntimeError) as exc:  # pragma: no cover - environment-dependent fallback
        write_json(root / "figures_status.json", {"status": "UNAVAILABLE", "reason": f"{type(exc).__name__}: {exc}"})
        return []
    figure_dir = root / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    if historical is not None:
        fig, ax = plt.subplots(figsize=(7, 4))
        labels = list(config.MODELS)
        x = np.arange(len(labels), dtype=float)
        width = 0.35
        for shift, metric, title in ((-width / 2, "gain_match", "best-match component"), (width / 2, "gain_background", "background component")):
            values = [historical["group_summaries"][f"{model}.{metric}"]["mean"] for model in labels]
            ax.bar(x + shift, values, width, label=title)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(x, labels)
        ax.set_ylabel("TTS − natural component")
        ax.set_title("LRS3 historical Sync-C decomposition")
        ax.legend()
        fig.tight_layout()
        for suffix in (".png", ".svg"):
            target = figure_dir / f"historical_decomposition{suffix}"
            fig.savefig(target, dpi=160)
            paths.append(str(target))
        plt.close(fig)
    if curve is not None and curve.get("status") == "complete":
        endpoint_rows = curve.get("_endpoints", [])
        fig, axes = plt.subplots(3, 4, figsize=(14, 8), sharex=True, sharey=False)
        for axis, sample_id in zip(axes.flat, config.CURVE_SAMPLE_IDS, strict=True):
            for condition, color in (("natural_raw", "tab:blue"), ("tts_raw", "tab:orange")):
                rows = [row for row in endpoint_rows if int(row["sample_id"]) == sample_id and row["condition"] == condition and row["support"] == "INTERIOR"]
                if rows:
                    row = rows[0]
                    axis.plot(row["offsets"], row["curve"], color=color, label=condition.replace("_raw", ""))
            axis.axvline(0, color="0.7", linewidth=0.5)
            axis.set_title(str(sample_id))
            axis.set_xlabel("official offset (frames)")
        axes.flat[0].set_ylabel("distance")
        axes.flat[0].legend(fontsize=8)
        fig.suptitle("LeapTalk LRS3 INTERIOR distance curves")
        fig.tight_layout()
        for suffix in (".png", ".svg"):
            target = figure_dir / f"leaptalk_curves{suffix}"
            fig.savefig(target, dpi=160)
            paths.append(str(target))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 4))
        paired_rows = [row for row in curve["paired"] if row["support"] in {"FULL", "INTERIOR", "EQUAL_COUNT"}]
        positions = np.arange(len(config.CURVE_SAMPLE_IDS), dtype=float)
        for support, marker in (("FULL", "o"), ("INTERIOR", "s"), ("EQUAL_COUNT", "^")):
            values = [next(float(row["delta_c"]) for row in paired_rows if int(row["sample_id"]) == sample_id and row["support"] == support) for sample_id in config.CURVE_SAMPLE_IDS]
            ax.plot(positions, values, marker=marker, label=support)
        ax.axhline(0.0, color="black", linewidth=0.8)
        ax.set_xticks(positions, [str(item) for item in config.CURVE_SAMPLE_IDS], rotation=45)
        ax.set_ylabel("TTS − natural Sync-C")
        ax.set_title("Support sensitivity by LRS3 pair")
        ax.legend()
        fig.tight_layout()
        for suffix in (".png", ".svg"):
            target = figure_dir / f"support_sensitivity{suffix}"
            fig.savefig(target, dpi=160)
            paths.append(str(target))
        plt.close(fig)
    return paths
