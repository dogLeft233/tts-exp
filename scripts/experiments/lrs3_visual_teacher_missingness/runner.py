from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from . import analysis, config
from .common import (
    ProtocolError,
    file_sha256,
    load_self_hashed,
    tree_manifest,
    write_json,
)


def _paths(run_id: str) -> config.RunPaths:
    root = config.run_root_for(run_id)
    root.mkdir(parents=True, exist_ok=True)
    return config.RunPaths(root)


def _assert_cached_support(current: dict[str, np.ndarray]) -> None:
    """Require the recomputed support to match H element-by-element."""
    with np.load(config.H / "support_indices.npz", allow_pickle=False) as cached:
        if set(cached.files) != set(current) or any(
            not np.array_equal(current[key], cached[key]) for key in current
        ):
            raise ProtocolError(
                "cached H support indices differ from recomputed support"
            )


def _check_fixed_inputs() -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    list[dict[str, Any]],
    list[str],
    dict[str, list[str]],
]:
    for name, expected in config.PARENT_HASHES.items():
        if file_sha256(config.PARENT / name) != expected:
            raise ProtocolError(f"parent hash changed: {config.PARENT / name}")
    for name, expected in config.LOCK_HASHES.items():
        if file_sha256(config.LOCK / name) != expected:
            raise ProtocolError(f"lock hash changed: {config.LOCK / name}")
    if file_sha256(config.REVIEW) != config.REVIEW_HASH:
        raise ProtocolError("artifact review hash changed")
    for name, expected in config.H_HASHES.items():
        if file_sha256(config.H / name) != expected:
            raise ProtocolError(f"cached H hash changed: {config.H / name}")
    files, _ = tree_manifest(config.PARENT / "records")
    feature_files, _ = tree_manifest(config.PARENT / "features")
    combined = sorted(files + feature_files, key=lambda row: row["path"])
    combined_hash = hashlib.sha256(
        "".join(f"{f['sha256']}  {f['path']}\n" for f in combined).encode()
    ).hexdigest()
    if (
        len(files) != 133
        or len(feature_files) != 399
        or combined_hash != config.TREE_HASH
    ):
        raise ProtocolError("record/feature tree manifest changed")
    h_protocol = load_self_hashed(config.H / "protocol.json")
    h_audit = load_self_hashed(config.H / "input_audit.json")
    h_payload = load_self_hashed(config.H / "records.json")
    h_analysis = load_self_hashed(config.H / "analysis.json")
    with np.load(config.H / "support_indices.npz", allow_pickle=False) as support:
        if set(support.files) != {"offsets", "real", "natural", "candidate"} or support[
            "offsets"
        ].shape != (116,):
            raise ProtocolError("cached H support denominator changed")
    records = list(h_payload.get("records", []))
    groups = list(h_payload.get("groups", []))
    split = {
        "fit_train": list(h_payload["split"]["fit_train"]),
        "fit_selection": list(h_payload["split"]["fit_selection"]),
    }
    if (
        len(records) != config.RECORD_COUNT
        or len(groups) != config.GROUP_COUNT
        or int(h_protocol.get("record_count", -1)) != config.RECORD_COUNT
    ):
        raise ProtocolError("cached H denominator changed")
    bound = {
        str(row["path"]): str(row["sha256"]) for row in h_audit.get("features", [])
    }
    if len(bound) != 399:
        raise ProtocolError("cached H feature audit is incomplete")
    for record in records:
        for path_value in record["feature_paths"].values():
            path = Path(str(path_value))
            try:
                rel = path.resolve().relative_to(config.REPO.resolve()).as_posix()
            except ValueError as exc:
                raise ProtocolError(f"feature path outside repository: {path}") from exc
            if rel not in bound or file_sha256(path) != bound[rel]:
                raise ProtocolError(f"feature binding changed: {path}")
    audit = {
        "parent_files": config.PARENT_HASHES,
        "lock_files": config.LOCK_HASHES,
        "review": {"path": str(config.REVIEW), "sha256": config.REVIEW_HASH},
        "cached_h": config.H_HASHES,
        "feature_tree_sha256": config.TREE_HASH,
    }
    return h_analysis, h_protocol, audit, records, groups, split


def prepare(paths: config.RunPaths) -> None:
    if (
        paths.protocol.exists()
        and paths.input_audit.exists()
        and paths.records.exists()
    ):
        protocol = load_self_hashed(paths.protocol)
        load_self_hashed(paths.input_audit)
        load_self_hashed(paths.records)
        _, _, _, records, groups, _ = _check_fixed_inputs()
        spec_root = (
            config.REPO / "openspec/changes/probe-lrs3-visual-dynamic-specificity"
        )
        code_hashes = {
            path.name: file_sha256(path)
            for path in sorted(Path(__file__).parent.glob("*.py"))
        }
        spec_hashes = {
            path.relative_to(spec_root).as_posix(): file_sha256(path)
            for path in sorted(spec_root.glob("*.md"))
            + sorted(spec_root.glob("specs/**/*.md"))
        }
        if (
            protocol.get("code_sha256") != code_hashes
            or protocol.get("spec_sha256") != spec_hashes
            or protocol.get("cached_h", {}).get("hashes") != config.H_HASHES
            or int(protocol.get("record_count", -1)) != len(records)
            or int(protocol.get("group_count", -1)) != len(groups)
        ):
            raise ProtocolError("resume identity changed; use a new run-id")
        return
    h_analysis, h_protocol, audit, records, groups, split = _check_fixed_inputs()
    code_paths = sorted(Path(__file__).parent.glob("*.py"))
    spec_root = config.REPO / "openspec/changes/probe-lrs3-visual-dynamic-specificity"
    spec_paths = sorted(spec_root.glob("*.md")) + sorted(
        spec_root.glob("specs/**/*.md")
    )
    protocol = {
        "schema_version": 2,
        "protocol_id": "lrs3_visual_dynamic_specificity",
        "status": "locked",
        "record_count": len(records),
        "group_count": len(groups),
        "groups": groups,
        "split": split,
        "rules": {
            "valid_fraction_min": config.VALID_FRACTION_MIN,
            "coverage_min": config.COVERAGE_MIN,
            "metrics": analysis.METRICS,
            "practical_threshold": analysis.PRACTICAL_THRESHOLD,
            "bootstrap_count": analysis.BOOTSTRAP_COUNT,
            "bootstrap_seed": analysis.BOOTSTRAP_SEED,
            "all_group_bounds": "23 groups equally weighted; missing values in [-1,1]",
        },
        "cached_h": {
            "root": str(config.H),
            "hashes": config.H_HASHES,
            "analysis_artifact_sha256": h_analysis["artifact_sha256"],
            "support_indices_sha256": config.H_HASHES["support_indices.npz"],
            "support_content_checked": True,
        },
        "source_hashes": audit,
        "code_sha256": {path.name: file_sha256(path) for path in code_paths},
        "spec_sha256": {
            path.relative_to(spec_root).as_posix(): file_sha256(path)
            for path in spec_paths
        },
        "zero_model_budget": True,
        "zero_video_score_budget": True,
    }
    write_json(paths.protocol, protocol)
    write_json(
        paths.input_audit,
        {
            **audit,
            "cached_h_protocol": h_protocol["artifact_sha256"],
            "records": len(records),
            "groups": len(groups),
        },
    )
    write_json(
        paths.records,
        {"schema_version": 1, "records": records, "groups": groups, "split": split},
    )


def analyze(paths: config.RunPaths) -> None:
    protocol = load_self_hashed(paths.protocol)
    payload = load_self_hashed(paths.records)
    history = load_self_hashed(config.H / "analysis.json")
    result = analysis.analyze_records(
        payload["records"], protocol["groups"], payload["split"], history
    )
    if result["observed_count"] != 115 or result["group_count"] != 23:
        raise ProtocolError("cached H observed denominator was not reproduced")
    legacy = result["legacy_reproduction"]
    if legacy["history_observed_count"] != 115 or legacy["max_abs_LU_error"] > 1e-10:
        raise ProtocolError("cached H missingness bound was not reproduced")
    observed = [row for row in result["records"] if row["observed"]]
    offsets = [0]
    real: list[int] = []
    natural: list[int] = []
    candidate: list[int] = []
    for row in observed:
        support = row["common_support"]
        real.extend(support["real_indices"])
        natural.extend(support["natural_indices"])
        candidate.extend(support["candidate_indices"])
        offsets.append(len(real))
    np.savez_compressed(
        paths.support_indices,
        offsets=np.asarray(offsets, dtype=np.int64),
        real=np.asarray(real, dtype=np.int64),
        natural=np.asarray(natural, dtype=np.int64),
        candidate=np.asarray(candidate, dtype=np.int64),
    )
    current = {
        "offsets": np.asarray(offsets, dtype=np.int64),
        "real": np.asarray(real, dtype=np.int64),
        "natural": np.asarray(natural, dtype=np.int64),
        "candidate": np.asarray(candidate, dtype=np.int64),
    }
    _assert_cached_support(current)
    result["support_indices_sha256"] = file_sha256(paths.support_indices)
    write_json(paths.analysis, {"protocol_id": protocol["protocol_id"], **result})


def finalize(paths: config.RunPaths) -> None:
    result = load_self_hashed(paths.analysis)
    validation = load_self_hashed(paths.validation)
    review = write_json(
        paths.review,
        {
            "schema_version": 1,
            "status": "self_reviewed",
            "checks": [
                "cached H hashes rechecked",
                "three-arm common support",
                "static/dynamic MSE identity",
                "fixed reversal controls",
                "23-group bounds and 21-group bootstrap",
                "zero model/video/score budget",
                "parent and Stage02 authorizations remain false",
            ],
            "analysis_sha256": result["artifact_sha256"],
            "validation_sha256": validation["artifact_sha256"],
            "decision": result["diagnostic_decision"],
        },
    )
    final = write_json(
        paths.final,
        {
            "schema_version": 1,
            "status": "complete",
            "engineering_decision": validation.get("engineering_decision"),
            "diagnostic_decision": result["diagnostic_decision"],
            "analysis_sha256": result["artifact_sha256"],
            "replacement_confirmed": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "parent_gate_repaired": False,
            "stage02_authorized": False,
            "review_sha256": review["artifact_sha256"],
            "validation_sha256": validation["artifact_sha256"],
            "model_calls": 0,
            "new_videos": 0,
            "new_scores": 0,
        },
    )
    paths.result.write_text(
        "# LRS3 visual dynamic specificity\n\n"
        + f"- diagnostic_decision: `{final['diagnostic_decision']}`\n- observed_subset: `{result['observed_subset']['record_count']}/{result['record_count']}` records, `{result['observed_subset']['group_count']}/{result['group_count']}` groups\n- timing gates: natural=`{result['observed_subset']['q_natural']['passes']}`, candidate=`{result['observed_subset']['q_candidate']['passes']}`\n- replacement_confirmed: `false`\n",
        encoding="utf-8",
    )


def run(run_id: str, stage: str, resume: bool) -> int:
    paths = _paths(run_id)
    try:
        if any(paths.root.iterdir()) and not resume:
            raise ProtocolError(f"existing output requires --resume: {paths.root}")
        prepare(paths)
        if stage == "prepare":
            return 0
        analyze(paths)
        if stage == "analyze":
            return 0
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.experiments.lrs3_visual_teacher_missingness.validate",
                "--run-root",
                str(paths.root),
            ],
            cwd=config.REPO,
            check=True,
        )
        finalize(paths)
        paths.root.joinpath("error.json").unlink(missing_ok=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        write_json(
            paths.root / "error.json",
            {
                "schema_version": 1,
                "status": "BLOCKED",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--stage", choices=("prepare", "analyze", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    return run(args.run_id, args.stage, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
