from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


REPRESENTATIVE_UNITS = {
    "energy_rms_dbfs": "dBFS",
    "silence_leading_s": "seconds",
    "sample_rate_source_hz": "Hz",
    "encoding_codec": "category",
    "band_ratio_0_250": "power ratio",
    "duration_s": "seconds",
    "timeline_audio_sample_count": "samples",
}
L2_FULL_VIEW = "all_seven_plus_context"
L2_BASELINE_VIEW = "baseline_training_mean"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _representatives(schema: Mapping[str, Any]) -> list[tuple[str, str, str]]:
    representatives: list[tuple[str, str, str]] = []
    for family, definition in schema["families"].items():
        columns = definition.get("acoustic") or definition.get("metadata") or []
        if not columns:
            raise ValueError(f"family {family!r} has no representative feature")
        column = str(columns[0])
        representatives.append((str(family), column, REPRESENTATIVE_UNITS.get(column, "native units")))
    return representatives


def _plot_feature_distributions(schema: Mapping[str, Any], features: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    representatives = _representatives(schema)
    layers = sorted(features["source_layer"].dropna().astype(str).unique())
    if not layers:
        raise ValueError("feature table has no source_layer rows")
    fig, axes = plt.subplots(
        len(layers), len(representatives),
        figsize=(3.0 * len(representatives), 2.8 * len(layers)),
        squeeze=False,
    )
    for row, layer in enumerate(layers):
        layer_frame = features.loc[features["source_layer"].astype(str) == layer]
        for col, (family, feature, unit) in enumerate(representatives):
            ax = axes[row, col]
            ax.set_title(f"{family}\n{feature}", fontsize=8)
            if feature not in layer_frame:
                ax.text(0.5, 0.5, "unavailable", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue
            condition_values = {
                condition: layer_frame.loc[layer_frame["condition"].astype(str) == condition, feature]
                for condition in ("natural", "tts")
            }
            missing = {condition: int(values.isna().sum()) for condition, values in condition_values.items()}
            totals = {condition: int(len(values)) for condition, values in condition_values.items()}
            if pd.api.types.is_numeric_dtype(layer_frame[feature]):
                plotted: list[np.ndarray] = []
                for condition in ("natural", "tts"):
                    numeric = pd.to_numeric(condition_values[condition], errors="coerce").to_numpy(dtype=float)
                    plotted.append(numeric[np.isfinite(numeric)])
                if any(values.size for values in plotted):
                    artists = ax.boxplot(plotted, labels=["N", "T"], patch_artist=True, showfliers=False)
                    for patch, color in zip(artists["boxes"], ("#4778a8", "#c77c4a")):
                        patch.set_facecolor(color)
                        patch.set_alpha(0.65)
                else:
                    ax.text(0.5, 0.5, "no finite values", ha="center", va="center", transform=ax.transAxes)
                ax.set_ylabel(unit, fontsize=8)
            else:
                categories = sorted({
                    str(value)
                    for values in condition_values.values()
                    for value in values.dropna().tolist()
                })
                if categories:
                    x = np.arange(len(categories), dtype=float)
                    width = 0.38
                    counts = {
                        condition: [int((condition_values[condition].astype(str) == category).sum()) for category in categories]
                        for condition in ("natural", "tts")
                    }
                    ax.bar(x - width / 2, counts["natural"], width, label="N", color="#4778a8")
                    ax.bar(x + width / 2, counts["tts"], width, label="T", color="#c77c4a")
                    ax.set_xticks(x, categories, rotation=35, ha="right", fontsize=7)
                    ax.legend(fontsize=7)
                else:
                    ax.text(0.5, 0.5, "no observed categories", ha="center", va="center", transform=ax.transAxes)
            ax.text(
                0.02, 0.98,
                f"missing N {missing['natural']}/{totals['natural']}\nmissing T {missing['tts']}/{totals['tts']}",
                ha="left", va="top", fontsize=7, transform=ax.transAxes,
            )
            ax.set_xlabel(layer if row == len(layers) - 1 else "", fontsize=8)
    fig.suptitle("Representative feature distributions by source layer and condition", fontsize=12)
    fig.text(
        0.5, 0.012,
        "Timeline sample count is tied to clip duration at fixed sample rate; it is not an independent dimension.",
        ha="center", fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.95))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_l1_views(analysis: Mapping[str, Any], oof: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    layers = sorted(analysis.get("l1", {}))
    if not layers:
        raise ValueError("analysis has no L1 strata")
    fig, axes = plt.subplots(len(layers), 2, figsize=(17, 5.2 * len(layers)), squeeze=False)
    for row, layer in enumerate(layers):
        summary = analysis["l1"][layer]
        models = summary.get("models", {})
        views = [name for name in sorted(models) if models[name].get("weighted_OOF_AUC") is not None]
        if not views:
            axes[row, 0].text(0.5, 0.5, "no L1 AUC views", ha="center", va="center", transform=axes[row, 0].transAxes)
            axes[row, 1].set_axis_off()
            continue
        metrics = summary.get("bootstrap", {})
        coverage = float(metrics.get("ordinary_coverage", 0.95))
        bootstrap_models = metrics.get("models", {}) if isinstance(metrics, Mapping) else {}
        auc_ax, roc_ax = axes[row, 0], axes[row, 1]
        y_positions = np.arange(len(views), dtype=float)
        for position, view in zip(y_positions, views):
            point = float(models[view]["weighted_OOF_AUC"])
            model_bootstrap = bootstrap_models.get(view, {})
            metric_bootstrap = model_bootstrap.get("AUC", {}) if isinstance(model_bootstrap, Mapping) else {}
            intervals = metric_bootstrap.get("intervals", {}) if isinstance(metric_bootstrap, Mapping) else {}
            ci = intervals.get(f"{coverage:.12g}") if isinstance(intervals, Mapping) else None
            if ci is not None and len(ci) == 2:
                xerr = [[max(0.0, point - float(ci[0]))], [max(0.0, float(ci[1]) - point)]]
                auc_ax.errorbar(point, position, xerr=xerr, fmt="o", color="#4778a8", capsize=2)
            else:
                auc_ax.plot(point, position, "o", color="#777777", markersize=4)
            rows = oof.loc[(oof["stratum"].astype(str) == str(layer)) & (oof["model_id"].astype(str) == view)]
            if rows.empty:
                continue
            if rows["audio_uid"].astype(str).duplicated().any():
                raise ValueError(f"duplicate OOF audio rows for {layer}/{view}")
            fpr, tpr, _ = roc_curve(
                rows["y"].to_numpy(dtype=int),
                rows["prediction"].to_numpy(dtype=float),
                sample_weight=rows["sample_weight"].to_numpy(dtype=float),
            )
            roc_ax.plot(fpr, tpr, linewidth=1.0, alpha=0.75, label=f"{view} ({point:.3f})")
        auc_ax.axvline(0.5, color="#555555", linestyle="--", linewidth=0.8)
        auc_ax.set_xlim(0.0, 1.0)
        auc_ax.set_yticks(y_positions, views, fontsize=7)
        auc_ax.invert_yaxis()
        auc_ax.set_xlabel(f"Weighted OOF AUC (ordinary {coverage:.1%} CI where available)")
        auc_ax.set_title(f"{layer}: view AUC")
        roc_ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=0.8)
        roc_ax.set_xlim(0.0, 1.0)
        roc_ax.set_ylim(0.0, 1.0)
        roc_ax.set_xlabel("False positive rate")
        roc_ax.set_ylabel("True positive rate")
        roc_ax.set_title(f"{layer}: weighted OOF ROC")
        roc_ax.legend(fontsize=6, loc="lower right", ncol=1)
    fig.suptitle("L1 saved-OOF views: AUC, intervals, and ROC", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def _plot_l2_delta_c(analysis: Mapping[str, Any], oof: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    strata = sorted(analysis.get("l2", {}).items())
    if not strata:
        raise ValueError("analysis has no L2 strata")
    fig, axes = plt.subplots(len(strata), 1, figsize=(7.5, 4.2 * len(strata)), squeeze=False)
    for row, (key, summary) in enumerate(strata):
        ax = axes[row, 0]
        rows = oof.loc[oof["stratum"].astype(str) == str(key)]
        baseline = rows.loc[rows["model_id"].astype(str) == L2_BASELINE_VIEW].sort_values("score_uid")
        full = rows.loc[rows["model_id"].astype(str) == L2_FULL_VIEW].sort_values("score_uid")
        if baseline.empty or full.empty:
            ax.text(0.5, 0.5, "required OOF view unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            continue
        if baseline["score_uid"].astype(str).tolist() != full["score_uid"].astype(str).tolist():
            raise ValueError(f"baseline/full score support differs for {key}")
        if not np.allclose(baseline["y"].to_numpy(dtype=float), full["y"].to_numpy(dtype=float), rtol=0.0, atol=1e-12):
            raise ValueError(f"baseline/full target values differ for {key}")
        actual = full["y"].to_numpy(dtype=float)
        baseline_pred = baseline["prediction"].to_numpy(dtype=float)
        full_pred = full["prediction"].to_numpy(dtype=float)
        ax.scatter(actual, baseline_pred, s=24, alpha=0.65, color="#777777", label="training-mean baseline")
        ax.scatter(actual, full_pred, s=28, alpha=0.75, color="#4778a8", label="all seven + context ridge")
        bounds = np.concatenate([actual, baseline_pred, full_pred])
        finite = bounds[np.isfinite(bounds)]
        low, high = float(finite.min()), float(finite.max())
        pad = max((high - low) * 0.05, 1e-6)
        ax.plot([low - pad, high + pad], [low - pad, high + pad], color="#444444", linestyle="--", linewidth=0.8)
        ax.set_xlim(low - pad, high + pad)
        ax.set_ylim(low - pad, high + pad)
        ax.set_xlabel("Actual delta_C")
        ax.set_ylabel("Saved OOF prediction of delta_C")
        ax.set_title(f"{summary.get('score_contract', key)} — {summary.get('feature_layer', '')}")
        ax.legend(fontsize=8)
    fig.suptitle("L2 actual delta_C versus fixed-fold OOF predictions", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output, dpi=160)
    plt.close(fig)


def create_plots(
    schema_path: Path,
    features_path: Path,
    analysis_path: Path,
    l1_oof_path: Path,
    l2_oof_path: Path,
    output_dir: Path,
) -> list[dict[str, Any]]:
    if output_dir.exists():
        raise FileExistsError(f"plot output must be fresh: {output_dir}")
    output_dir.mkdir(parents=True)
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    features = pd.read_csv(features_path, dtype={"audio_uid": str})
    l1_oof = pd.read_csv(l1_oof_path, dtype={"audio_uid": str, "pair_uid": str})
    l2_oof = pd.read_csv(l2_oof_path, dtype={"score_uid": str, "pair_uid": str})
    targets = [
        ("feature_distributions.png", lambda path: _plot_feature_distributions(schema, features, path)),
        ("l1_views_auc_roc.png", lambda path: _plot_l1_views(analysis, l1_oof, path)),
        ("l2_delta_c_oof.png", lambda path: _plot_l2_delta_c(analysis, l2_oof, path)),
    ]
    artifacts: list[dict[str, Any]] = []
    for name, renderer in targets:
        target = output_dir / name
        renderer(target)
        artifacts.append({"path": str(target.resolve()), "bytes": target.stat().st_size, "sha256": _sha256(target)})
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description="Create data-only figures from saved features, analysis, and OOF predictions.")
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--analysis", required=True, type=Path)
    parser.add_argument("--l1-oof", required=True, type=Path)
    parser.add_argument("--l2-oof", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args()
    if args.receipt.exists():
        raise FileExistsError(f"plot receipt must be fresh: {args.receipt}")
    artifacts = create_plots(args.schema, args.features, args.analysis, args.l1_oof, args.l2_oof, args.output_dir)
    input_paths = {
        "schema": args.schema,
        "features": args.features,
        "analysis": args.analysis,
        "l1_oof": args.l1_oof,
        "l2_oof": args.l2_oof,
    }
    receipt = {
        "schema": "processing_chain_predictive_audit_saved_plots_v1",
        "status": "PASS",
        "mode": "plots_only_no_model_fits",
        "inputs": {name: {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)} for name, path in input_paths.items()},
        "artifacts": artifacts,
    }
    args.receipt.parent.mkdir(parents=True, exist_ok=True)
    args.receipt.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(receipt, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
