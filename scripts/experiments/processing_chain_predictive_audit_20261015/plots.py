from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Mapping


def _adjusted_interval(item: Mapping[str, Any], level: float) -> Any:
    adjusted = item.get("seven_family_adjusted_interval")
    if not isinstance(adjusted, Mapping):
        return None
    intervals = adjusted.get("intervals")
    if not isinstance(intervals, Mapping):
        return None
    return intervals.get(f"{level:.12g}")


def _coverage_level(stratum: Mapping[str, Any]) -> float:
    bootstrap = stratum.get("bootstrap")
    if not isinstance(bootstrap, Mapping):
        return 0.9928571428571429
    return float(bootstrap.get("seven_family_adjusted_coverage", 0.9928571428571429))


def _plot_l1(results: Mapping[str, Any], path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows: list[tuple[str, str, float | None, list[float] | None]] = []
    for layer, stratum in sorted(results.get("l1", {}).items()):
        if not isinstance(stratum, Mapping):
            continue
        for family, item in sorted(stratum.get("family_incremental", {}).items()):
            point = item.get("point_gain")
            level = _coverage_level(stratum)
            ci = _adjusted_interval(item, level)
            rows.append((str(layer), str(family), point, ci))
    if not rows:
        return False
    layers = sorted({row[0] for row in rows})
    families = sorted({row[1] for row in rows})
    fig, axes = plt.subplots(len(layers), 1, figsize=(max(9, len(families) * 1.1), 3.4 * len(layers)), squeeze=False)
    for ax, layer in zip(axes[:, 0], layers):
        subset = {row[1]: row for row in rows if row[0] == layer}
        xs = list(range(len(families)))
        values = [subset.get(f, ("", f, None, None))[2] for f in families]
        vals = [0.0 if value is None else float(value) for value in values]
        lower, upper = [], []
        for family in families:
            ci = subset.get(family, ("", family, None, None))[3]
            point = subset.get(family, ("", family, None, None))[2]
            if ci and point is not None and len(ci) == 2:
                lower.append(max(0.0, float(point) - float(ci[0])))
                upper.append(max(0.0, float(ci[1]) - float(point)))
            else:
                lower.append(0.0)
                upper.append(0.0)
        ax.bar(xs, vals, color="#4778a8", yerr=[lower, upper], capsize=3)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(xs, families, rotation=25, ha="right")
        ax.set_ylabel("OOF AUC gain")
        layer_rows = [row for row in rows if row[0] == layer]
        missing = sum(row[3] is None for row in layer_rows)
        note = " — intervals unavailable" if missing == len(layer_rows) else (" — some intervals unavailable" if missing else "")
        ax.set_title(f"L1 family increment — {layer}{note}")
    fig.suptitle("Fixed-fold predictive family ablations (descriptive)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _plot_l2(results: Mapping[str, Any], path: Path) -> bool:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strata = [
        (name, row) for name, row in sorted(results.get("l2", {}).items())
        if isinstance(row, Mapping) and row.get("family_incremental")
    ]
    if not strata:
        return False
    families = sorted({family for _, row in strata for family in row["family_incremental"]})
    fig, axes = plt.subplots(len(strata), 1, figsize=(max(9, len(families) * 1.1), 3.4 * len(strata)), squeeze=False)
    for ax, (name, stratum) in zip(axes[:, 0], strata):
        increments = stratum["family_incremental"]
        values = [increments.get(f, {}).get("point_gain") for f in families]
        vals = [0.0 if value is None else float(value) for value in values]
        lower, upper = [], []
        for family in families:
            item = increments.get(family, {})
            level = _coverage_level(stratum)
            ci = _adjusted_interval(item, level)
            point = item.get("point_gain")
            if ci and point is not None and len(ci) == 2:
                lower.append(max(0.0, float(point) - float(ci[0])))
                upper.append(max(0.0, float(ci[1]) - float(point)))
            else:
                lower.append(0.0)
                upper.append(0.0)
        xs = list(range(len(families)))
        ax.bar(xs, vals, color="#aa7246", yerr=[lower, upper], capsize=3)
        ax.axhline(0, color="#333333", linewidth=0.8)
        ax.set_xticks(xs, families, rotation=25, ha="right")
        ax.set_ylabel("OOF MAE reduction")
        missing = sum(_adjusted_interval(item, _coverage_level(stratum)) is None for item in increments.values())
        note = " — intervals unavailable" if missing == len(increments) else (" — some intervals unavailable" if missing else "")
        ax.set_title(f"L2 family increment — {name}{note}")
    fig.suptitle("Fixed-fold predictive family ablations (descriptive)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def plot_analysis(results: Mapping[str, Any], output_dir: str | Path) -> list[dict[str, Any]]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for name, renderer in (("l1_family_increments.png", _plot_l1), ("l2_family_increments.png", _plot_l2)):
        target = output / name
        if renderer(results, target):
            artifacts.append({
                "path": str(target.resolve()),
                "bytes": target.stat().st_size,
                "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
                "status": "WRITTEN",
            })
    index = {"status": "PASS" if artifacts else "NO_PLOTTABLE_STRATA", "artifacts": artifacts}
    (output / "index.json").write_text(
        json.dumps(index, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return artifacts
