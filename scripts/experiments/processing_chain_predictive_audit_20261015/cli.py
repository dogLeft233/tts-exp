from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from . import modeling
from .plots import plot_analysis


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _pin(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": _sha256(path)}


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _resolve_bound_inputs(
    protocol: Mapping[str, Any],
    protocol_path: Path,
    supplied: Mapping[str, str],
    input_binding: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Path], dict[str, Any]]:
    bindings = protocol.get("input_bindings", protocol.get("inputs", {}))
    cohort = protocol.get("cohort_binding", {})
    primary = cohort.get("primary", {}) if isinstance(cohort, Mapping) else {}
    if not isinstance(primary, Mapping):
        primary = {}
    feature_binding = protocol.get("feature_binding", primary.get("feature_binding"))
    if not isinstance(feature_binding, Mapping):
        feature_binding = primary.get("features")
    if not isinstance(feature_binding, Mapping):
        feature_binding = {}

    nested = {
        "assets_json": {
            "path": primary.get("asset_inventory_path"),
            "sha256": primary.get("asset_inventory_sha256"),
        },
        "paired_scores_csv": {
            "path": primary.get("paired_scores_path"),
            "sha256": primary.get("paired_scores_sha256"),
        },
        "features_csv": dict(feature_binding),
    }
    if not isinstance(bindings, Mapping):
        bindings = {}
    bound_inputs = (input_binding or {}).get("inputs", (input_binding or {}).get("input_bindings", {}))
    if not isinstance(bound_inputs, Mapping):
        bound_inputs = {}
    if input_binding is not None:
        required_names = {"assets_json", "features_csv", "paired_scores_csv"}
        if not required_names.issubset(bound_inputs):
            raise modeling.AuditInputError("smoke input binding must pin all three input tables")
        for name in required_names:
            ref = bound_inputs[name]
            if not isinstance(ref, Mapping) or not ref.get("path") or not ref.get("sha256"):
                raise modeling.AuditInputError(f"smoke input binding has an invalid {name} reference")
    if input_binding is not None:
        protocol_ref = input_binding.get("protocol")
        if not isinstance(protocol_ref, Mapping) or not protocol_ref.get("path") or not protocol_ref.get("sha256"):
            raise modeling.AuditInputError("smoke input binding must pin the frozen protocol path and SHA")
        expected_protocol = Path(str(protocol_ref["path"]))
        if not expected_protocol.is_absolute():
            expected_protocol = (Path(__file__).resolve().parents[3] / expected_protocol).resolve()
        else:
            expected_protocol = expected_protocol.resolve()
        if expected_protocol != protocol_path.resolve() or str(protocol_ref["sha256"]).lower() != _sha256(protocol_path):
            raise modeling.AuditInputError("smoke input binding does not match this frozen protocol")
    project_root = Path(str(protocol.get("project_root", Path(__file__).resolve().parents[3])))
    if not project_root.is_absolute():
        project_root = (protocol_path.parent / project_root).resolve()
    names = ("assets_json", "features_csv", "paired_scores_csv")
    resolved: dict[str, Path] = {}
    pins: dict[str, Any] = {}
    for name in names:
        protocol_binding = bindings.get(name)
        if not isinstance(protocol_binding, Mapping):
            protocol_binding = nested[name]
        binding = protocol_binding
        external_binding = bound_inputs.get(name)
        if isinstance(external_binding, Mapping):
            if isinstance(protocol_binding, Mapping) and protocol_binding.get("path") and protocol_binding.get("sha256"):
                protocol_path_value = Path(str(protocol_binding["path"]))
                if not protocol_path_value.is_absolute():
                    protocol_path_value = (project_root / protocol_path_value).resolve()
                else:
                    protocol_path_value = protocol_path_value.resolve()
                external_path_value = Path(str(external_binding.get("path", "")))
                if not external_path_value.is_absolute():
                    external_path_value = (project_root / external_path_value).resolve()
                else:
                    external_path_value = external_path_value.resolve()
                if (protocol_path_value != external_path_value
                        or str(protocol_binding["sha256"]).lower() != str(external_binding.get("sha256", "")).lower()):
                    raise modeling.AuditInputError(f"external input binding conflicts with frozen protocol for {name}")
            binding = external_binding
        if name == "features_csv" and isinstance(binding, Mapping):
            binding = {
                "path": binding.get("path", binding.get("features_path", binding.get("features_csv_path"))),
                "sha256": binding.get("sha256", binding.get("features_sha256", binding.get("features_csv_sha256"))),
            }
        if not isinstance(binding, Mapping) or not binding.get("path") or not binding.get("sha256"):
            raise modeling.AuditInputError(f"protocol lacks path/SHA binding for {name}")
        if name not in supplied or not supplied[name]:
            raise modeling.AuditInputError(f"CLI must supply the protocol-bound path for {name}")
        path = Path(supplied[name]).expanduser().resolve(strict=True)
        expected = Path(str(binding["path"]))
        if not expected.is_absolute():
            expected = (project_root / expected).resolve()
        else:
            expected = expected.resolve()
        if path != expected:
            raise modeling.AuditInputError(f"CLI path for {name} differs from frozen protocol path")
        actual_sha = _sha256(path)
        if actual_sha != str(binding["sha256"]).lower():
            raise modeling.AuditInputError(f"SHA-256 mismatch for frozen input {name}")
        resolved[name] = path
        pins[name] = _pin(path)
    if input_binding is not None:
        extra_refs = input_binding.get("features_manifest")
        if extra_refs is not None:
            if not isinstance(extra_refs, Mapping) or not extra_refs.get("path") or not extra_refs.get("sha256"):
                raise modeling.AuditInputError("feature_manifest binding must include path and SHA")
            manifest_path = Path(str(extra_refs["path"]))
            if not manifest_path.is_absolute():
                manifest_path = (project_root / manifest_path).resolve(strict=True)
            else:
                manifest_path = manifest_path.resolve(strict=True)
            if _sha256(manifest_path) != str(extra_refs["sha256"]).lower():
                raise modeling.AuditInputError("feature manifest SHA-256 mismatch")
            manifest = modeling.load_json(str(manifest_path))
            feature_record = manifest.get("features_csv", manifest.get("features")) if isinstance(manifest, Mapping) else None
            if isinstance(feature_record, Mapping):
                if (str(feature_record.get("sha256", "")).lower()
                        != str(pins["features_csv"]["sha256"]).lower()):
                    raise modeling.AuditInputError("feature manifest does not bind the selected feature CSV")
            pins["features_manifest"] = _pin(manifest_path)
        else:
            raise modeling.AuditInputError("smoke binding must include its feature manifest provenance")
        source_refs = input_binding.get("sources", {})
        if source_refs:
            if not isinstance(source_refs, Mapping):
                raise modeling.AuditInputError("input binding sources must be a name-to-path/SHA object")
            source_pins: dict[str, Any] = {}
            for name, reference in source_refs.items():
                if not isinstance(reference, Mapping) or not reference.get("path") or not reference.get("sha256"):
                    raise modeling.AuditInputError(f"input binding source {name} lacks path/SHA")
                source_path = Path(str(reference["path"]))
                if not source_path.is_absolute():
                    source_path = (project_root / source_path).resolve(strict=True)
                else:
                    source_path = source_path.resolve(strict=True)
                if _sha256(source_path) != str(reference["sha256"]).lower():
                    raise modeling.AuditInputError(f"input binding source SHA-256 mismatch: {name}")
                source_pins[str(name)] = _pin(source_path)
            pins["producer_sources"] = source_pins
    return resolved, pins


def _select_smoke_rows(
    protocol: Mapping[str, Any],
    assets: pd.DataFrame,
    features: pd.DataFrame,
    scores: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    execution = protocol.get("execution", {})
    cohort = protocol.get("cohort_binding", {})
    primary = cohort.get("primary", {}) if isinstance(cohort, Mapping) else {}
    if not isinstance(execution, Mapping) or not isinstance(primary, Mapping):
        raise modeling.AuditInputError("protocol smoke/cohort binding is malformed")
    pair_uids = [str(value) for value in execution.get("smoke_pair_uids", [])]
    source_groups = [str(value) for value in execution.get("smoke_source_groups", [])]
    if (execution.get("stage_authorized") != "smoke_only"
            or execution.get("full_analysis_authorized") is not False):
        raise modeling.AuditInputError("protocol does not authorize smoke-only execution")
    if (len(pair_uids) != len(set(pair_uids)) or len(pair_uids) != 12
            or len(source_groups) != len(set(source_groups)) or len(source_groups) != 6
            or int(execution.get("smoke_audio_count", -1)) != 24):
        raise modeling.AuditInputError("frozen smoke roster must bind 12 pairs, 6 groups, and 24 audios")
    selected_assets = assets.loc[assets["pair_uid"].astype(str).isin(pair_uids)].copy()
    if len(selected_assets) != 24 or selected_assets["audio_uid"].astype(str).nunique() != 24:
        raise modeling.AuditInputError("smoke selection did not resolve to exactly 24 unique assets")
    if set(selected_assets["pair_uid"].astype(str)) != set(pair_uids):
        raise modeling.AuditInputError("one or more frozen smoke pair UIDs are absent from assets")
    if set(selected_assets["source_group"].astype(str)) != set(source_groups):
        raise modeling.AuditInputError("asset source groups differ from the frozen smoke roster")
    if set(selected_assets["condition"].astype(str)) != {"natural", "tts"}:
        raise modeling.AuditInputError("smoke assets must contain exactly natural and TTS conditions")
    per_pair = selected_assets.groupby("pair_uid", dropna=False)
    if any(len(group) != 2 or set(group["condition"].astype(str)) != {"natural", "tts"}
           or group["source_group"].astype(str).nunique() != 1 for _, group in per_pair):
        raise modeling.AuditInputError("each smoke pair must have one natural and one TTS asset in one source group")
    group_membership = primary.get("group_membership", {})
    cohort_id = str(primary.get("cohort_id", ""))
    if isinstance(group_membership, Mapping) and group_membership and cohort_id:
        frozen_owner = {
            f"{cohort_id}:{str(member)}": str(group)
            for group, members in group_membership.items()
            for member in members
        }
        for pair_uid, rows in per_pair:
            expected_group = frozen_owner.get(str(pair_uid))
            actual_group = str(rows["source_group"].iloc[0])
            if expected_group is None or expected_group != actual_group:
                raise modeling.AuditInputError(f"smoke pair-to-source-group mapping differs: {pair_uid}")
    source_layer = str(primary.get("source_layer", ""))
    if not source_layer:
        raise modeling.AuditInputError("frozen primary cohort does not name its source layer")
    if set(selected_assets["source_layer"].astype(str)) != {source_layer}:
        raise modeling.AuditInputError("smoke asset layer differs from the frozen primary layer")

    selected_audio = set(selected_assets["audio_uid"].astype(str))
    selected_features = features.loc[features["audio_uid"].astype(str).isin(selected_audio)].copy()
    if len(selected_features) != 24 or selected_features["audio_uid"].astype(str).nunique() != 24:
        raise modeling.AuditInputError("smoke selection did not resolve to one feature row per audio")
    if set(selected_features["audio_uid"].astype(str)) != selected_audio:
        raise modeling.AuditInputError("smoke feature audio IDs do not exactly match the selected assets")

    selected_scores = scores.loc[scores["pair_uid"].astype(str).isin(pair_uids)].copy()
    if len(selected_scores) != 12 or selected_scores["pair_uid"].astype(str).nunique() != 12:
        raise modeling.AuditInputError("smoke selection must resolve to exactly one paired score row per pair")
    if set(selected_scores["pair_uid"].astype(str)) != set(pair_uids):
        raise modeling.AuditInputError("one or more frozen smoke pair UIDs are absent from scores")
    for field, expected in (
        ("score_contract", primary.get("score_contract")),
        ("contract_fingerprint", primary.get("contract_fingerprint")),
        ("feature_layer", source_layer),
    ):
        if expected is not None and set(selected_scores[field].astype(str)) != {str(expected)}:
            raise modeling.AuditInputError(f"smoke paired-score {field} differs from frozen primary")

    summary = {
        "status": "ENGINEERING_ONLY_SELECTED",
        "pair_uids": pair_uids,
        "source_groups": source_groups,
        "asset_audio_uids": sorted(selected_audio),
        "pairs": int(selected_scores["pair_uid"].nunique()),
        "assets": int(len(selected_assets)),
        "features": int(len(selected_features)),
        "scores": int(len(selected_scores)),
        "source_layer": source_layer,
        "inference": "ENGINEERING_ONLY_NO_BOOTSTRAP_NO_PERMUTATION_NO_SCIENTIFIC_CONCLUSION",
    }
    return selected_assets, selected_features, selected_scores, summary


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(modeling.json_safe(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the authorized stage of the observational processing-chain L1/L2 audit."
    )
    parser.add_argument("--stage", choices=("smoke", "full"), required=True)
    parser.add_argument("--schema", required=True, type=Path)
    parser.add_argument("--addendum", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--input-binding", type=Path, help="ROOT-created execution binding for derived feature inputs")
    parser.add_argument("--assets-json", required=True)
    parser.add_argument("--features-csv", required=True)
    parser.add_argument("--paired-scores-csv", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    protocol_path = args.protocol.expanduser().resolve(strict=True)
    schema_path = args.schema.expanduser().resolve(strict=True)
    addendum_path = args.addendum.expanduser().resolve(strict=True)
    output = args.output_dir.expanduser().absolute()
    protocol = modeling.load_json(str(protocol_path))
    if not isinstance(protocol, Mapping):
        raise modeling.AuditInputError("protocol must be a JSON object")
    modeling._validate_protocol_is_final(protocol)
    execution = protocol.get("execution", {})
    if not isinstance(execution, Mapping):
        raise modeling.AuditInputError("frozen protocol lacks a valid execution gate")
    if args.stage == "smoke":
        if (execution.get("stage_authorized") != "smoke_only"
                or execution.get("full_analysis_authorized") is not False):
            raise modeling.AuditInputError("protocol does not authorize the requested smoke stage")
        if args.input_binding is None:
            raise modeling.AuditInputError("smoke requires ROOT's separate feature/input binding")
    elif execution.get("full_analysis_authorized") is not True:
        raise modeling.AuditInputError("full analysis is not separately authorized by the frozen protocol")
    binding = None
    binding_path = None
    if args.input_binding is not None:
        binding_path = args.input_binding.expanduser().resolve(strict=True)
        binding = modeling.load_json(str(binding_path))
        if not isinstance(binding, Mapping):
            raise modeling.AuditInputError("input binding must be a JSON object")
        binding_status = str(binding.get("status", "")).upper()
        if binding_status not in {"BOUND", "PASS", "FROZEN"} and not (
            args.stage == "smoke" and not binding_status
        ):
            raise modeling.AuditInputError("input binding is not frozen/bound")
        if args.stage == "smoke" and binding.get("stage") != "smoke":
            raise modeling.AuditInputError("input binding stage must be exactly 'smoke'")
    input_paths, input_pins = _resolve_bound_inputs(
        protocol,
        protocol_path,
        {
            "assets_json": args.assets_json,
            "features_csv": args.features_csv,
            "paired_scores_csv": args.paired_scores_csv,
        },
        input_binding=binding,
    )
    for input_path in (schema_path, addendum_path):
        if not input_path.is_file():
            raise modeling.AuditInputError(f"required configuration file missing: {input_path}")
    if output.exists() or output.is_symlink():
        raise modeling.AuditInputError(f"output directory must be fresh: {output}")
    if output.parent.is_symlink() or not output.parent.is_dir():
        raise modeling.AuditInputError("output parent must exist and must not be a symlink")

    output.mkdir()
    started_utc = _utc_now()
    started = time.perf_counter()
    source_paths = {
        "modeling": Path(__file__).with_name("modeling.py"),
        "plots": Path(__file__).with_name("plots.py"),
        "cli": Path(__file__).with_name("cli.py"),
    }
    receipt: dict[str, Any] = {
        "status": "RUNNING",
        "analysis_stage": args.stage,
        "started_utc": started_utc,
        "schema": _pin(schema_path),
        "addendum": _pin(addendum_path),
        "protocol": _pin(protocol_path),
        "input_binding": _pin(binding_path) if binding_path is not None else None,
        "inputs": input_pins,
        "sources": {name: _pin(path) for name, path in source_paths.items()},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": _package_version("numpy"),
            "pandas": _package_version("pandas"),
            "scikit_learn": _package_version("scikit-learn"),
            "scipy": _package_version("scipy"),
            "matplotlib": _package_version("matplotlib"),
            "cpu_thread_limit": 2,
        },
    }
    _write_json(output / "run_receipt.json", receipt)
    try:
        schema = modeling.load_json(str(schema_path))
        addendum = modeling.load_json(str(addendum_path))
        assets, features, scores = modeling.load_input_tables(
            str(input_paths["assets_json"]),
            str(input_paths["features_csv"]),
            str(input_paths["paired_scores_csv"]),
        )
        full_input_counts = {
            "assets": int(len(assets)),
            "features": int(len(features)),
            "scores": int(len(scores)),
        }
        smoke_selection = None
        if args.stage == "smoke":
            assets, features, scores, smoke_selection = _select_smoke_rows(
                protocol, assets, features, scores
            )
            _write_json(output / "smoke_selection.json", smoke_selection)
            receipt["smoke_selection"] = smoke_selection
            receipt["full_input_counts"] = full_input_counts
        try:
            from threadpoolctl import threadpool_limits
        except ImportError as exc:
            raise modeling.AuditInputError("threadpoolctl is required to enforce the frozen CPU2 limit") from exc
        with threadpool_limits(limits=2):
            results, l1_oof, l2_oof = modeling.run_analysis(
                schema, addendum, protocol, assets, features, scores,
                engineering_only=(args.stage == "smoke"),
            )
        results["input_scope"] = smoke_selection or {
            "status": "FULL_COHORT",
            "pairs": int(scores["pair_uid"].nunique()),
            "assets": int(len(assets)),
            "features": int(len(features)),
            "scores": int(len(scores)),
        }
        _write_json(output / "analysis.json", results)
        l1_oof.to_csv(output / "l1_oof_predictions.csv", index=False, encoding="utf-8")
        l2_oof.to_csv(output / "l2_oof_predictions.csv", index=False, encoding="utf-8")
        all_groups = set(assets["source_group"].dropna().astype(str))
        all_groups.update(scores["source_group"].dropna().astype(str))
        folds = modeling.make_group_fold_map(all_groups, int(results["configuration"]["max_folds"]))
        pd.DataFrame(
            [{"source_group": group, "fold": folds[group]} for group in sorted(folds)]
        ).to_csv(output / "fold_map.csv", index=False, encoding="utf-8")
        plots = plot_analysis(results, output / "plots")
        receipt.update(
            {
                "status": "PASS",
                "finished_utc": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "outputs": {
                    name: _pin(output / name)
                    for name in (
                        "analysis.json",
                        "l1_oof_predictions.csv",
                        "l2_oof_predictions.csv",
                        "fold_map.csv",
                    )
                },
                "smoke_selection": _pin(output / "smoke_selection.json") if args.stage == "smoke" else None,
                "plot_artifacts": plots,
                "scope": (
                    "engineering-only OOF smoke; no bootstrap, permutations, inferential claims or scientific conclusion"
                    if args.stage == "smoke"
                    else "observational prediction only; no causal ranking"
                ),
            }
        )
        _write_json(output / "run_receipt.json", receipt)
        print(json.dumps({"status": "PASS", "output_dir": str(output.resolve())}, ensure_ascii=False))
        return 0
    except Exception as exc:
        receipt.update(
            {
                "status": "FAILED",
                "finished_utc": _utc_now(),
                "wall_seconds": time.perf_counter() - started,
                "error_type": type(exc).__name__,
                "error": str(exc)[:2000],
            }
        )
        _write_json(output / "run_receipt.json", receipt)
        print(f"processing-chain audit failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
