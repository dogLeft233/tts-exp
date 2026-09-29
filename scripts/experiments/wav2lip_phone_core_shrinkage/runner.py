from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments import wav2lip_probe_runtime as rt

from . import config, construct


def _fixed() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    for path, expected in (
        (config.P_DRIVERS, config.P_DRIVERS_SHA256),
        (config.P_CONTROL, config.P_CONTROL_SHA256),
        (config.D_DRIVERS, config.D_DRIVERS_SHA256),
        (config.D_MASKS, config.D_MASKS_SHA256),
        (config.WAV2LIP_CHECKPOINT, config.WAV2LIP_CHECKPOINT_SHA256),
        (config.SYNCNET_MODEL, config.SYNCNET_MODEL_SHA256),
    ):
        if not path.is_file() or rt.file_sha256(path) != expected:
            raise rt.ProtocolError(f"fixed asset changed or missing: {path}")
    return (
        rt.load_self(config.P_DRIVERS),
        rt.load_self(config.P_CONTROL),
        rt.read_json(config.D_DRIVERS),
    )


def _repo_key(path: Path) -> str:
    return path.resolve().relative_to(config.REPO.resolve()).as_posix()


def _hash_paths(paths: list[Path] | tuple[Path, ...]) -> dict[str, str]:
    return {_repo_key(path): rt.file_sha256(path) for path in paths}


def _identity_hashes() -> dict[str, dict[str, str]]:
    spec_root = (
        config.REPO / "openspec/changes/complete-wav2lip-phone-core-group-support"
    )
    code_paths = tuple(sorted(Path(__file__).parent.glob("*.py")))
    spec_paths = (spec_root / "proposal.md", spec_root / "design.md", config.SPEC)
    return {
        "code_sha256": _hash_paths(code_paths),
        "helper_sha256": _hash_paths(config.HELPER_PATHS),
        "spec_sha256": _hash_paths(spec_paths),
    }


def _rows(drivers: dict[str, Any]) -> list[dict[str, Any]]:
    rows = sorted(
        drivers.get("rows", []),
        key=lambda row: (str(row["source_group"]), str(row["sample_id"])),
    )
    groups = [str(row["source_group"]) for row in rows]
    if (
        len(rows) != 16
        or len(set(groups)) != 8
        or any(groups.count(group) != 2 for group in set(groups))
    ):
        raise rt.ProtocolError("cohort is not 16/8 with two records per group")
    return rows


def group_exposure(audit_rows: list[dict[str, Any]]) -> dict[str, bool]:
    """Aggregate record exposure at the frozen source_group unit."""
    result: dict[str, bool] = {}
    for item in audit_rows:
        group = str(item["source_group"])
        result[group] = result.get(group, False) or bool(
            item.get("record_exposed", False)
        )
    return result


def _decode_frames(path: Path) -> list[np.ndarray]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise rt.ProtocolError(f"replay video has no frames: {path}")
    return frames


def _audio_binding_equal(actual: dict[str, Any], reference: dict[str, Any]) -> bool:
    """Accept the fresh scorer's frozen PCM field and the parent field name."""
    actual_pcm = next(
        (
            actual.get(key)
            for key in (
                "source_pcm_sha256",
                "frozen_pcm_sha256",
                "extracted_pcm_sha256",
            )
            if actual.get(key)
        ),
        None,
    )
    reference_pcm = next(
        (
            reference.get(key)
            for key in (
                "source_pcm_sha256",
                "frozen_pcm_sha256",
                "extracted_pcm_sha256",
            )
            if reference.get(key)
        ),
        None,
    )
    return bool(
        actual.get("source_audio_sha256") == reference.get("source_audio_sha256")
        and actual_pcm
        and actual_pcm == reference_pcm
    )


def _parent_n_score(parent: dict[str, Any], sample_id: str) -> dict[str, Any]:
    """Select the parent's natural-video/natural-audio cell explicitly."""
    for row in parent["rows"]:
        if (
            str(row.get("sample_id")) == sample_id
            and str(row.get("video_arm")) == "N"
            and str(row.get("audio_arm")) == "N"
        ):
            return row.get("score", row)
    raise rt.ProtocolError(f"parent natural control missing: {sample_id}")


def _replay_plan(
    paths: config.RunPaths, parent: dict[str, Any], drivers: dict[str, Any]
) -> dict[str, Any]:
    parent_rows = sorted(
        (
            row
            for row in parent["rows"]
            if str(row.get("video_arm")) == "N" and str(row.get("audio_arm")) == "N"
        ),
        key=lambda row: (
            str(
                next(
                    item
                    for item in drivers["rows"]
                    if str(item["sample_id"]) == str(row["sample_id"])
                )["source_group"]
            ),
            str(row["sample_id"]),
        ),
    )[:2]
    driver_by_id = {str(row["sample_id"]): row for row in drivers["rows"]}
    rows = []
    for parent_row in parent_rows:
        sid = str(parent_row["sample_id"])
        source = driver_by_id[sid]
        score = parent_row.get("score", parent_row)
        output = str(
            (paths.root / "media" / "replay" / f"{sid}__N_REPLAY.mkv").resolve()
        )
        rows.append(
            {
                "sample_id": sid,
                "static_face": str(source["static_face"]["path"]),
                "source_audio": str(score["source_audio"]),
                "arms": {"N_REPLAY": str(source["arms"]["N"]["path"])},
                "outputs": {"N_REPLAY": output},
            }
        )
    if len(rows) != 2:
        raise rt.ProtocolError("expected two frozen N replay records")
    return rt.write_json(
        paths.root / "plans" / "replay.json",
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_phone_core_shrinkage",
            "rows": rows,
        },
    )


def _replay_rows(
    paths: config.RunPaths, parent: dict[str, Any], drivers: dict[str, Any]
) -> list[dict[str, Any]]:
    _replay_plan(paths, parent, drivers)
    generated = rt.run_gpu_plan(
        paths.root / "plans" / "replay.json",
        paths.root / "generation" / "replay.json",
        "wav2lip_phone_core_shrinkage",
        config.WAV2LIP_CHECKPOINT,
        config.FFMPEG,
        config.WAV2LIP_PYTHON,
    )
    rows = rt.score_videos(
        generated["rows"],
        paths.root / "scores" / "replay",
        config.SYNCNET_MODEL,
        "wav2lip_phone_core_shrinkage",
    )
    for row in rows:
        sid = str(row["sample_id"])
        actual = row["score"]
        reference = _parent_n_score(parent, sid)
        fresh_frames = _decode_frames(Path(str(actual["media"])))
        parent_frames = _decode_frames(Path(str(reference["media"])))
        fresh_matrix = np.asarray(
            np.load(actual["matrix"], allow_pickle=False), dtype=np.float64
        )
        parent_matrix = np.asarray(
            np.load(reference["matrix"], allow_pickle=False), dtype=np.float64
        )
        fresh_metrics = rt.score_metrics(fresh_matrix)
        parent_metrics = rt.score_metrics(parent_matrix)
        matrix_error = (
            float(np.max(np.abs(fresh_matrix - parent_matrix)))
            if fresh_matrix.shape == parent_matrix.shape
            else float("inf")
        )
        endpoint_error = (
            max(
                abs(a - b)
                for a, b in zip(
                    fresh_metrics["curve"], parent_metrics["curve"], strict=True
                )
            )
            if fresh_matrix.shape == parent_matrix.shape
            else float("inf")
        )
        replay = {
            "pixel_equal": bool(
                len(fresh_frames) == len(parent_frames)
                and all(
                    np.array_equal(a, b)
                    for a, b in zip(fresh_frames, parent_frames, strict=True)
                )
            ),
            "matrix_max_abs": matrix_error,
            "endpoint_max_abs": float(endpoint_error),
            "offset_equal": bool(fresh_metrics["offset"] == parent_metrics["offset"]),
            "pcm_equal": _audio_binding_equal(actual, reference),
            "passes": False,
        }
        replay["passes"] = bool(
            replay["pixel_equal"]
            and matrix_error <= 1e-4
            and endpoint_error <= 1e-6
            and replay["offset_equal"]
            and replay["pcm_equal"]
        )
        if not replay["passes"]:
            raise rt.ProtocolError(f"N replay control failed: {sid}")
        row["replay"] = replay
    return rows


def _parity_rows(
    paths: config.RunPaths, parent: dict[str, Any]
) -> list[dict[str, Any]]:
    selected = []
    references: dict[tuple[str, str], Path] = {}
    for row in parent["rows"]:
        arm = str(row.get("video_arm"))
        if not arm.startswith("PARITY_"):
            continue
        score = row.get("score", row)
        key = (str(row["sample_id"]), f"FRESH_{arm}")
        media = Path(str(score["media"])).resolve()
        selected.append(
            {
                "sample_id": str(row["sample_id"]),
                "arm": f"FRESH_{arm}",
                "output": str(media),
                "output_sha256": rt.file_sha256(media),
                "source_audio": str(Path(str(score["source_audio"])).resolve()),
            }
        )
        references[key] = Path(
            str(row.get("parity", {}).get("actual_matrix", score["matrix"]))
        ).resolve()
    if len(selected) != 2:
        raise rt.ProtocolError(f"expected two parity controls, got {len(selected)}")
    rows = rt.score_videos(
        selected,
        paths.root / "scores" / "parity",
        config.SYNCNET_MODEL,
        "wav2lip_phone_core_shrinkage",
    )
    for row in rows:
        key = (str(row["sample_id"]), str(row["video_arm"]))
        actual_path = Path(str(row["score"]["matrix"]))
        reference_path = references[key]
        actual = np.asarray(np.load(actual_path, allow_pickle=False), dtype=np.float64)
        reference = np.asarray(
            np.load(reference_path, allow_pickle=False), dtype=np.float64
        )
        actual_metrics = rt.score_metrics(actual)
        reference_metrics = rt.score_metrics(reference)
        matrix_error = (
            float(np.max(np.abs(actual - reference)))
            if actual.shape == reference.shape
            else float("inf")
        )
        endpoint_error = (
            max(
                abs(a - b)
                for a, b in zip(
                    actual_metrics["curve"], reference_metrics["curve"], strict=True
                )
            )
            if actual.shape == reference.shape
            else float("inf")
        )
        parity = {
            "reference_matrix": str(reference_path),
            "matrix_max_abs": matrix_error,
            "endpoint_max_abs": float(endpoint_error),
            "offset_equal": bool(
                actual_metrics["offset"] == reference_metrics["offset"]
            ),
            "passes": bool(
                actual.shape == reference.shape
                and matrix_error <= 1e-4
                and endpoint_error <= 1e-6
                and actual_metrics["offset"] == reference_metrics["offset"]
            ),
        }
        if not parity["passes"]:
            raise rt.ProtocolError(f"parity control failed: {key}")
        row["parity"] = parity
    return rows


def _prepare(paths: config.RunPaths, resume: bool) -> dict[str, Any]:
    p_drivers, _, d_payload = _fixed()
    if paths.protocol.is_file():
        if not resume:
            raise rt.ProtocolError("run exists; use --resume")
        existing = rt.load_self(paths.protocol)
        identity = _identity_hashes()
        if any(existing.get(name) != value for name, value in identity.items()):
            raise rt.ProtocolError("resume identity changed; use a new run-id")
        return existing
    masks_payload = rt.read_json(config.D_MASKS)
    masks = {str(item["mask_sha256"]): item for item in masks_payload["masks"]}
    d_rows = d_payload["drivers"]
    records = []
    audit_rows = []
    construction_degenerate = []
    for row in _rows(p_drivers):
        sid = str(row["sample_id"])
        p_mel_path = Path(str(row["arms"]["N"]["path"]))
        if rt.file_sha256(p_mel_path) != str(row["arms"]["N"]["sha256"]):
            raise rt.ProtocolError(f"natural mel changed: {sid}")
        natural = rt.load_mel(p_mel_path)
        try:
            cores = construct.selected_cores(sid, d_rows, masks)
        except rt.ProtocolError as exc:
            if "INPUT_DEGENERATE" not in str(exc):
                raise
            cores = []
            result = construct.construct_candidates(natural, cores)
            records.append(
                {
                    "sample_id": sid,
                    "source_group": str(row["source_group"]),
                    "static_face": row["static_face"],
                    "source_audio": row["natural_audio"],
                    "arms": {
                        "N": {
                            "path": str(p_mel_path.resolve()),
                            "sha256": rt.file_sha256(p_mel_path),
                        }
                    },
                    "construction": {"cores": cores, "status": "INPUT_DEGENERATE"},
                }
            )
            construction_degenerate.append(sid)
            audit_rows.append(
                {
                    "sample_id": sid,
                    "source_group": str(row["source_group"]),
                    "core_count": 0,
                    "status": "CONSTRUCTION_INPUT_DEGENERATE",
                    "record_exposed": False,
                    "exposed_rows": [],
                    "reason": str(exc),
                }
            )
            continue
        try:
            result = construct.construct_candidates(natural, cores)
        except rt.ProtocolError as exc:
            if "INPUT_DEGENERATE" in str(exc):
                records.append(
                    {
                        "sample_id": sid,
                        "source_group": str(row["source_group"]),
                        "static_face": row["static_face"],
                        "source_audio": row["natural_audio"],
                        "arms": {
                            "N": {
                                "path": str(p_mel_path.resolve()),
                                "sha256": rt.file_sha256(p_mel_path),
                            }
                        },
                        "construction": {"cores": cores, "status": "INPUT_DEGENERATE"},
                    }
                )
                construction_degenerate.append(sid)
                audit_rows.append(
                    {
                        "sample_id": sid,
                        "source_group": str(row["source_group"]),
                        "core_count": len(cores),
                        "status": "CONSTRUCTION_INPUT_DEGENERATE",
                        "record_exposed": False,
                        "reason": str(exc),
                    }
                )
                continue
            raise
        arms = {}
        for arm in config.ARMS:
            path = paths.root / "drivers" / f"{sid}__{arm}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, result[arm], allow_pickle=False)
            arms[arm] = {
                "path": str(path.resolve()),
                "sha256": rt.file_sha256(path),
                "shape": list(result[arm].shape),
                "dtype": str(result[arm].dtype),
            }
        records.append(
            {
                "sample_id": sid,
                "source_group": str(row["source_group"]),
                "static_face": row["static_face"],
                "source_audio": row["natural_audio"],
                "arms": arms,
                "construction": result["metadata"],
            }
        )
        audit_rows.append(
            {
                "sample_id": sid,
                "source_group": str(row["source_group"]),
                "core_count": len(cores),
                "core_columns": int(
                    sum(int(item["end"]) - int(item["start"]) for item in cores)
                ),
                "status": "GO",
                "record_exposed": bool(result["metadata"]["record_exposed"]),
                "exposed_rows": result["metadata"]["exposed_rows"],
                "positive_columns": result["metadata"]["positive_columns"],
                "changed_columns": result["metadata"]["changed_columns"],
                "u_row_perturbation_norms": result["metadata"][
                    "u_row_perturbation_norms"
                ],
            }
        )
    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in audit_rows:
        by_group.setdefault(str(item["source_group"]), []).append(item)
    group_support = group_exposure(audit_rows)
    unsupported_groups = sorted(
        group for group, exposed in group_support.items() if not exposed
    )
    if construction_degenerate:
        degenerate = [str(item["sample_id"]) for item in records]
        for item in audit_rows:
            item["status"] = "INPUT_DEGENERATE"
            item["group_exposed"] = False
            item["reason"] = "INPUT_DEGENERATE: numeric construction failure in queue"
    else:
        degenerate = [
            str(item["sample_id"])
            for item in records
            if str(item["source_group"]) in unsupported_groups
        ]
        for item in audit_rows:
            item["group_exposed"] = str(item["source_group"]) not in unsupported_groups
            if str(item["source_group"]) in unsupported_groups:
                item["status"] = "INPUT_DEGENERATE"
                item["reason"] = (
                    "INPUT_DEGENERATE: source group has no public U exposure"
                )
    bindings = {
        name: {"path": str(path.resolve()), "sha256": rt.file_sha256(path)}
        for name, path in {
            "proposal": config.PROPOSAL,
            "design": config.DESIGN,
            "spec": config.SPEC,
            "parent_drivers": config.P_DRIVERS,
            "parent_control": config.P_CONTROL,
            "d_drivers": config.D_DRIVERS,
            "mask_manifest": config.D_MASKS,
            "old_protocol": config.OLD_PROTOCOL,
            "old_input_audit": config.OLD_INPUT_AUDIT,
            "old_drivers": config.OLD_DRIVERS,
        }.items()
    }
    protocol = {
        "schema_version": 1,
        "protocol_id": "wav2lip_phone_core_shrinkage",
        "status": "complete" if degenerate else "locked",
        "run_id": paths.root.name.removeprefix("wav2lip_phone_core_shrinkage_"),
        "bindings": bindings,
        **_identity_hashes(),
        "records": [
            {"sample_id": row["sample_id"], "source_group": row["source_group"]}
            for row in records
        ],
        "record_count": 16,
        "source_group_count": 8,
        "budget": {"new_videos": 34, "new_scores": 36},
        "scientific_decision": "INPUT_DEGENERATE" if degenerate else None,
        "candidate_authorized": False,
        "exposure_gate": {
            "unit": "source_group",
            "unsupported_groups": unsupported_groups,
            "group_count": len(by_group),
        },
        "flags": {
            "replacement_confirmed": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "historical_shift_gate_repaired": False,
        },
    }
    rt.write_json(
        paths.input_audit,
        {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "status": "complete",
            "rows": audit_rows,
            "degenerate_ids": degenerate,
        },
    )
    rt.write_json(
        paths.drivers,
        {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "status": "complete",
            "rows": records,
        },
    )
    protocol = rt.write_json(paths.protocol, protocol)
    return protocol


def _controls(paths: config.RunPaths, resume: bool) -> dict[str, Any]:
    if (
        paths.controls.is_file()
        and paths.control_scores.is_file()
        and paths.control_validation.is_file()
        and resume
    ):
        old = rt.load_self(paths.controls)
        if "ci95" in old.get("anchor_damage", {}):
            return old
    value = rt.parent_control_gate(config.P_CONTROL, config.P_DRIVERS)
    _fixed()
    replay = _replay_rows(
        paths, rt.load_self(config.P_CONTROL), rt.load_self(paths.drivers)
    )
    parity = _parity_rows(paths, rt.load_self(config.P_CONTROL))
    fresh = replay + parity
    rt.write_json(
        paths.control_scores,
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_phone_core_shrinkage",
            "status": "complete",
            "rows": fresh,
            "count": len(fresh),
            "fresh_video_count": 2,
            "fresh_control_count": len(fresh),
            "evaluation_audio": "original_natural_pcm_only",
        },
    )
    value.update(
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_phone_core_shrinkage",
            "candidate_authorized": value["status"] == "PASS",
            "fresh_video_count": 2,
            "fresh_control_count": len(fresh),
            "fresh_control_rows": [str(row["video_arm"]) for row in fresh],
            "replay_pass": True,
            "parity_pass": True,
        }
    )
    controls = rt.write_json(paths.controls, value)
    from .validate import validate

    control_validation = validate(paths.root, stage="controls")
    if control_validation.get("status") != "PASS":
        raise rt.ProtocolError("independent control validator is not PASS")
    return controls


def _plan(paths: config.RunPaths, parent: dict[str, Any]) -> dict[str, Any]:
    parent_by_id = {str(row["sample_id"]): row for row in parent["rows"]}
    rows = []
    for row in rt.load_self(paths.drivers)["rows"]:
        sid = str(row["sample_id"])
        source = parent_by_id[sid]
        rows.append(
            {
                "sample_id": sid,
                "static_face": source["static_face"]["path"],
                "source_audio": source["natural_audio"]["path"],
                "arms": {
                    arm: row["arms"][arm]["path"]
                    for arm in ("PHONE_CORE", "GENERIC_CORE")
                },
                "outputs": {
                    arm: str((paths.root / "media" / f"{sid}__{arm}.mkv").resolve())
                    for arm in ("PHONE_CORE", "GENERIC_CORE")
                },
            }
        )
    return rt.write_json(
        paths.plan,
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_phone_core_shrinkage",
            "rows": rows,
        },
    )


def _candidates(paths: config.RunPaths, parent: dict[str, Any]) -> None:
    if rt.load_self(paths.controls).get("status") != "PASS":
        raise rt.ProtocolError("CONTROL_FAILED")
    if (
        not paths.control_validation.is_file()
        or rt.load_self(paths.control_validation).get("status") != "PASS"
    ):
        raise rt.ProtocolError("independent control validator is not PASS")
    control_validation = rt.load_self(paths.control_validation)
    if control_validation.get("control_sha256") != rt.file_sha256(
        paths.controls
    ) or control_validation.get("scores_sha256") != rt.file_sha256(
        paths.control_scores
    ):
        raise rt.ProtocolError("independent control validation binding changed")
    _plan(paths, parent)
    generation = rt.run_gpu_plan(
        paths.plan,
        paths.generation,
        "wav2lip_phone_core_shrinkage",
        config.WAV2LIP_CHECKPOINT,
        config.FFMPEG,
        config.WAV2LIP_PYTHON,
    )
    scores = rt.score_videos(
        generation["rows"],
        paths.root / "scores" / "candidates",
        config.SYNCNET_MODEL,
        "wav2lip_phone_core_shrinkage",
    )
    control_rows = rt.load_self(paths.control_scores)["rows"]
    rt.write_json(
        paths.scores,
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_phone_core_shrinkage",
            "status": "complete",
            "rows": scores + control_rows,
            "count": len(scores) + len(control_rows),
            "candidate_count": len(scores),
            "fresh_control_count": len(control_rows),
            "evaluation_audio": "original_natural_pcm_only",
        },
    )


def _finalize_degenerate(
    paths: config.RunPaths, validation: dict[str, Any], degenerate: list[str]
) -> None:
    audit = rt.load_self(paths.input_audit)
    zero_exposure = [
        str(row["sample_id"])
        for row in audit.get("rows", [])
        if not bool(row.get("record_exposed"))
    ]
    review = rt.write_json(
        paths.review,
        {
            "schema_version": 1,
            "status": "self_reviewed",
            "checks": [
                "independent degenerate-cohort reconstruction",
                "all 16 records retained",
                "no parameter retuning",
                "no candidate generation",
            ],
        },
    )
    actual_budget = {"new_videos": 0, "new_scores": 0}
    final = rt.write_json(
        paths.final,
        {
            "schema_version": 1,
            "status": "complete",
            "engineering_status": "GO",
            "scientific_decision": "INPUT_DEGENERATE",
            "validation_sha256": validation["artifact_sha256"],
            "review_sha256": review["artifact_sha256"],
            "zero_exposure_records": zero_exposure,
            "actual_budget": actual_budget,
            "replacement_confirmed": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "historical_shift_gate_repaired": False,
        },
    )
    paths.result.write_text(
        f"# Wav2Lip phone-core shrinkage probe\n\n- decision: `{final['scientific_decision']}`\n- retained records: `16/16`\n- degenerate records: `{len(degenerate)}`\n- zero-exposure records: `{', '.join(zero_exposure) if zero_exposure else 'none'}`\n- actual budget: `0 videos / 0 score cells`\n- next round: `independent source-group confirmation required`\n- replacement_confirmed: `false`\n",
        encoding="utf-8",
    )


def _natural(control: dict[str, Any], sid: str) -> dict[str, Any]:
    row = next(
        row
        for row in control["rows"]
        if str(row["sample_id"]) == sid
        and str(row["video_arm"]) == "N"
        and str(row["audio_arm"]) == "N"
    )
    _, _, matrix = rt.load_worker_arrays(row)
    return rt.score_metrics(matrix)


def _mechanism(records: list[dict[str, Any]]) -> dict[str, Any]:
    values = []
    for row in records:
        k0 = int(row["N"]["min_index"])
        values.append(
            {
                "C": row["PHONE_CORE"]["C"] - row["GENERIC_CORE"]["C"],
                "D": row["GENERIC_CORE"]["D"] - row["PHONE_CORE"]["D"],
                "A": row["GENERIC_CORE"]["curve"][k0] - row["PHONE_CORE"]["curve"][k0],
            }
        )
    groups = [str(row["source_group"]) for row in records]
    labels = sorted(set(groups))
    grouped = {
        label: {
            metric: float(
                np.mean(
                    [
                        value[metric]
                        for value, group in zip(values, groups, strict=True)
                        if group == label
                    ]
                )
            )
            for metric in ("C", "D", "A")
        }
        for label in labels
    }
    indices = rt.bootstrap_indices(labels)
    metrics = {
        metric: rt.grouped_stats([value[metric] for value in values], groups, indices)
        for metric in ("C", "D", "A")
    }
    return {
        "metrics": metrics,
        "group_means": grouped,
        "joint_positive_count": int(
            sum(
                all(grouped[label][metric] > 0 for metric in ("C", "D", "A"))
                for label in labels
            )
        ),
    }


def _analysis(paths: config.RunPaths, control: dict[str, Any]) -> dict[str, Any]:
    score_manifest = rt.load_self(paths.scores)
    score_map = {
        (str(row["sample_id"]), str(row["video_arm"])): row
        for row in score_manifest["rows"]
    }
    records = []
    for row in rt.load_self(paths.drivers)["rows"]:
        sid = str(row["sample_id"])
        item = {
            "sample_id": sid,
            "source_group": str(row["source_group"]),
            "N": _natural(control, sid),
        }
        for arm in ("PHONE_CORE", "GENERIC_CORE"):
            item[arm] = score_map[(sid, arm)]["score"]["full"]
        records.append(item)
    contrasts = {
        arm: rt.contrast_summary(records, arm) for arm in ("PHONE_CORE", "GENERIC_CORE")
    }
    mechanism = _mechanism(records)
    main_pass = rt.gain_pass(contrasts["PHONE_CORE"])
    mech_pass = bool(
        mechanism["joint_positive_count"] >= 7
        and all(
            mechanism["metrics"][metric]["ci99"][0] > 0 for metric in ("C", "D", "A")
        )
    )
    decision = (
        "NO_PHONE_CORE_GAIN_ESTABLISHED"
        if not main_pass
        else "PHONE_CORE_SPECIFIC_SIGNAL_TO_CONFIRM"
        if mech_pass
        else "NATURAL_GAIN_MECHANISM_UNRESOLVED"
    )
    return rt.write_json(
        paths.analysis,
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_phone_core_shrinkage",
            "status": "complete",
            "records": records,
            "contrasts": contrasts,
            "mechanism": mechanism,
            "main_pass": main_pass,
            "scientific_decision": decision,
            "replacement_confirmed": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "historical_shift_gate_repaired": False,
        },
    )


def _finalize(
    paths: config.RunPaths, validation: dict[str, Any], analysis: dict[str, Any]
) -> None:
    review = rt.write_json(
        paths.review,
        {
            "schema_version": 1,
            "status": "self_reviewed",
            "checks": [
                "mask identity and disjoint cores",
                "shared support and amplitude",
                "postclip norm gate",
                "independent validator",
            ],
        },
    )
    audit = rt.load_self(paths.input_audit)
    control = rt.load_self(paths.controls)
    zero_exposure = sorted(
        str(row["sample_id"])
        for row in audit.get("rows", [])
        if not bool(row.get("record_exposed"))
    )
    replay_videos = (
        len(rt.load_self(paths.root / "generation" / "replay.json")["rows"])
        if (paths.root / "generation" / "replay.json").is_file()
        else 0
    )
    actual_budget = {
        "new_videos": replay_videos + len(rt.load_self(paths.generation)["rows"]),
        "new_scores": len(rt.load_self(paths.scores)["rows"]),
    }
    final = rt.write_json(
        paths.final,
        {
            "schema_version": 1,
            "status": "complete",
            "engineering_status": "GO",
            "scientific_decision": analysis["scientific_decision"],
            "analysis_sha256": analysis["artifact_sha256"],
            "validation_sha256": validation["artifact_sha256"],
            "review_sha256": review["artifact_sha256"],
            "support_group_count": 8,
            "zero_exposure_records": zero_exposure,
            "old_offset_pass_count": control.get("old_offset_pass_count"),
            "matched_offset_pass_count": control.get("matched_offset_pass_count"),
            "actual_budget": actual_budget,
            "replacement_confirmed": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "historical_shift_gate_repaired": False,
        },
    )
    paths.result.write_text(
        f"# Wav2Lip phone-core shrinkage probe\n\n- decision: `{final['scientific_decision']}`\n- PHONE_CORE ΔC mean: `{analysis['contrasts']['PHONE_CORE']['metrics']['C']['mean']:.6f}`\n- mechanism joint-positive groups: `{analysis['mechanism']['joint_positive_count']}`\n- support: `8/8 source groups`; zero-exposure records: `{', '.join(zero_exposure) if zero_exposure else 'none'}`\n- historical offset controls: old=`{final['old_offset_pass_count']}/16`, matched=`{final['matched_offset_pass_count']}/16`; fresh replay/parity=`{control.get('replay_pass')}/{control.get('parity_pass')}`\n- actual budget: `{actual_budget['new_videos']} videos / {actual_budget['new_scores']} score cells`\n- next round: `independent source-group confirmation required; no waveform-head or replacement authorization`\n- replacement_confirmed: `false`\n",
        encoding="utf-8",
    )


def run(run_id: str, stage: str, resume: bool) -> int:
    paths = config.RunPaths(config.run_root_for(run_id))
    paths.root.mkdir(parents=True, exist_ok=True)
    try:
        protocol = _prepare(paths, resume)
        parent, control, _ = _fixed()
        if protocol.get("scientific_decision") == "INPUT_DEGENERATE":
            from .validate import validate

            validation = validate(paths.root)
            _finalize_degenerate(
                paths,
                validation,
                list(rt.load_self(paths.input_audit).get("degenerate_ids", [])),
            )
            paths.root.joinpath("error.json").unlink(missing_ok=True)
            return 0
        if stage == "prepare":
            return 0
        controls = _controls(paths, resume)
        if stage == "controls":
            return 0
        if controls["status"] != "PASS":
            return 1
        if stage in ("candidates", "analyze", "all") and (
            not paths.control_validation.is_file()
            or rt.load_self(paths.control_validation).get("status") != "PASS"
        ):
            raise rt.ProtocolError("independent control validator is not PASS")
        if stage == "analyze":
            analysis = _analysis(paths, control)
            from .validate import validate

            validation = validate(paths.root)
            _finalize(paths, validation, analysis)
            paths.root.joinpath("error.json").unlink(missing_ok=True)
            return 0
        _candidates(paths, parent)
        if stage == "candidates":
            return 0
        analysis = _analysis(paths, control)
        from .validate import validate

        validation = validate(paths.root)
        _finalize(paths, validation, analysis)
        paths.root.joinpath("error.json").unlink(missing_ok=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        rt.write_json(
            paths.root / "error.json",
            {
                "schema_version": 1,
                "status": "BLOCKED",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "controls", "candidates", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    return run(args.run_id, args.stage, args.resume)


if __name__ == "__main__":
    raise SystemExit(main())
