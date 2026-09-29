from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments import wav2lip_probe_runtime as rt

from . import config
from . import validator_math as vm


def _cores(sid: str, rows: list[dict], masks: dict[str, dict]) -> list[dict]:
    selected = set()
    for row in rows:
        if str(row["sample_id"]) != sid:
            continue
        current = set()
        for used in row["used_masks"]:
            item = masks.get(str(used["mask_sha256"]))
            if item is None:
                raise rt.ProtocolError("unknown mask")
            s, e = int(used["global_start_frame"]), int(used["global_end_frame"])
            if (s, e) != (
                int(item["natural_core_start_frame"]),
                int(item["natural_core_end_frame"]),
            ):
                raise rt.ProtocolError("mask coordinate mismatch")
            if str(item.get("label", "")) not in config.EXCLUDED_LABELS and e - s >= 5:
                current.add(
                    (str(used["mask_sha256"]), s, e, str(item.get("label", "")))
                )
        if not selected:
            selected = current
        elif selected != current:
            raise rt.ProtocolError("seed/condition mask set differs")
    result = [
        {"mask_sha256": key, "start": s, "end": e, "label": label}
        for key, s, e, label in sorted(
            selected, key=lambda value: (value[1], value[2], value[0])
        )
    ]
    if any(result[i + 1]["start"] < result[i]["end"] for i in range(len(result) - 1)):
        raise rt.ProtocolError("overlapping selected cores")
    return result


def _exposed_rows(cores: list[dict]) -> list[int]:
    chunks = []
    index = 0
    while True:
        start = int(index * 80.0 / 25.0)
        if start + 16 > 308:
            chunks.append(set(range(292, 308)))
            break
        chunks.append(set(range(start, start + 16)))
        index += 1
    positive = {
        int(core["start"]) + j
        for core in cores
        for j, value in enumerate(
            np.minimum(
                1.0,
                np.minimum(
                    np.arange(int(core["end"]) - int(core["start"]), dtype=np.float64)
                    / 2.0,
                    (
                        int(core["end"])
                        - int(core["start"])
                        - 1
                        - np.arange(
                            int(core["end"]) - int(core["start"]), dtype=np.float64
                        )
                    )
                    / 2.0,
                ),
            )
        )
        if value > 0
    }
    return [
        int(row)
        for row in config.U_ROWS
        if positive.intersection(set().union(*chunks[row : row + 5]))
    ]


def _build(mel: np.ndarray, cores: list[dict]) -> dict[str, np.ndarray | dict]:
    m = np.asarray(mel, dtype=np.float64)
    phone_r = np.zeros_like(m)
    generic_r = np.zeros_like(m)
    saved = []
    for core in cores:
        s, e = core["start"], core["end"]
        length = e - s
        j = np.arange(length, dtype=np.float64)
        w = np.minimum(1.0, np.minimum(j / 2.0, (length - 1 - j) / 2.0))
        mu = np.mean(m[:, s:e], axis=1, dtype=np.float64)
        phone_r[:, s:e] = w[None, :] * (mu[:, None] - m[:, s:e])
        saved.append((s, e, w))
    p = np.pad(m, ((0, 0), (2, 2)), mode="reflect")
    smooth = (
        p[:, 0:308] + 4 * p[:, 1:309] + 6 * p[:, 2:310] + 4 * p[:, 3:311] + p[:, 4:312]
    ) / 16.0
    for s, e, w in saved:
        generic_r[:, s:e] = w[None, :] * (smooth[:, s:e] - m[:, s:e])
    np_phone = np.linalg.norm(phone_r)
    np_generic = np.linalg.norm(generic_r)
    if np_phone <= 1e-12 or np_generic <= 1e-12:
        raise rt.ProtocolError("degenerate residual")
    generic_r *= np_phone / np_generic
    maximum = max(float(np.max(np.abs(phone_r))), float(np.max(np.abs(generic_r))))
    a = min(0.25, 0.5 / maximum)
    phone = np.clip(m + a * phone_r, -4, 4).astype(np.float32)
    generic = np.clip(m + a * generic_r, -4, 4).astype(np.float32)
    ratio = float(
        np.linalg.norm(generic.astype(np.float64) - m)
        / np.linalg.norm(phone.astype(np.float64) - m)
    )
    if not 0.95 <= ratio <= 1.05:
        raise rt.ProtocolError("postclip norm ratio invalid")
    return {"N": m.astype(np.float32), "PHONE_CORE": phone, "GENERIC_CORE": generic}


def _mechanism(records: list[dict]) -> dict:
    values = []
    for row in records:
        k = row["N"]["min_index"]
        values.append(
            {
                "C": row["PHONE_CORE"]["C"] - row["GENERIC_CORE"]["C"],
                "D": row["GENERIC_CORE"]["D"] - row["PHONE_CORE"]["D"],
                "A": row["GENERIC_CORE"]["curve"][k] - row["PHONE_CORE"]["curve"][k],
            }
        )
    groups = [str(row["source_group"]) for row in records]
    labels = sorted(set(groups))
    indices = vm.bootstrap_indices(labels)
    means = {
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
    metrics = {
        metric: vm.grouped_stats([value[metric] for value in values], groups, indices)
        for metric in ("C", "D", "A")
    }
    return {
        "metrics": metrics,
        "group_means": means,
        "joint_positive_count": int(
            sum(
                all(means[label][metric] > 0 for metric in ("C", "D", "A"))
                for label in labels
            )
        ),
    }


def _worker_metrics(score_row: dict, label: str) -> dict:
    visual, audio, cached = rt.load_worker_arrays(score_row, expected_rows=None)
    rebuilt = vm.matrix_from_embeddings(visual, audio)
    if (
        rebuilt.shape != cached.shape
        or float(np.max(np.abs(rebuilt.astype(np.float64) - cached.astype(np.float64))))
        > 1e-4
    ):
        raise rt.ProtocolError(f"independent SyncNet matrix mismatch: {label}")
    return vm.score_metrics(cached)


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


def _audio_binding_equal(actual: dict, reference: dict) -> bool:
    """Normalize fresh scorer and historical parent PCM field names."""
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


def _validate_replay_rows(rows: list[dict], parent_rows: list[dict]) -> None:
    parent_by_id = {
        (
            str(row["sample_id"]),
            str(row.get("video_arm")),
            str(row.get("audio_arm")),
        ): row.get("score", row)
        for row in parent_rows
    }
    replay_rows = [row for row in rows if str(row.get("video_arm")) == "N_REPLAY"]
    if len(replay_rows) != 2:
        raise rt.ProtocolError("expected two N replay score cells")
    for row in replay_rows:
        sid = str(row["sample_id"])
        actual = row["score"]
        reference = parent_by_id[(sid, "N", "N")]
        fresh_frames = _decode_frames(Path(str(actual["media"])))
        parent_frames = _decode_frames(Path(str(reference["media"])))
        fresh_matrix = np.asarray(
            np.load(actual["matrix"], allow_pickle=False), dtype=np.float64
        )
        parent_matrix = np.asarray(
            np.load(reference["matrix"], allow_pickle=False), dtype=np.float64
        )
        fresh_metrics = _worker_metrics(row, f"fresh-replay/{sid}")
        parent_metrics = vm.score_metrics(parent_matrix)
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
        computed = {
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
        computed["passes"] = bool(
            computed["pixel_equal"]
            and matrix_error <= 1e-4
            and endpoint_error <= 1e-6
            and computed["offset_equal"]
            and computed["pcm_equal"]
        )
        if row.get("replay") != computed or not computed["passes"]:
            raise rt.ProtocolError(f"independent N replay validation failed: {sid}")


def _validate_parity_rows(rows: list[dict], parent_rows: list[dict]) -> None:
    expected = {}
    for row in parent_rows:
        arm = str(row.get("video_arm"))
        if arm.startswith("PARITY_"):
            score = row.get("score", row)
            expected[(str(row["sample_id"]), f"FRESH_{arm}")] = (
                score,
                Path(
                    str(row.get("parity", {}).get("actual_matrix", score["matrix"]))
                ).resolve(),
            )
    parity_rows = [
        row for row in rows if str(row.get("video_arm", "")).startswith("FRESH_PARITY_")
    ]
    if len(expected) != 2 or len(parity_rows) != 2:
        raise rt.ProtocolError("expected two fresh parity score cells")
    for row in parity_rows:
        key = (str(row["sample_id"]), str(row["video_arm"]))
        reference_score, reference_path = expected.get(key, (None, None))
        if reference_score is None:
            raise rt.ProtocolError(f"unknown fresh parity cell: {key}")
        actual_path = Path(str(row["score"]["matrix"]))
        actual = np.asarray(np.load(actual_path, allow_pickle=False), dtype=np.float64)
        reference = np.asarray(
            np.load(reference_path, allow_pickle=False), dtype=np.float64
        )
        actual_metrics = _worker_metrics(row, f"fresh-parity/{key[0]}")
        reference_metrics = vm.score_metrics(reference)
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
        computed = {
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
        if row.get("parity") != computed or not computed["passes"]:
            raise rt.ProtocolError(f"independent fresh parity validation failed: {key}")


def _validate_fixed_bindings(protocol: dict) -> None:
    fixed = {
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
    }
    bindings = protocol.get("bindings", {})
    for name, path in fixed.items():
        binding = bindings.get(name)
        if not isinstance(binding, dict) or str(binding.get("path")) != str(
            path.resolve()
        ):
            raise rt.ProtocolError(f"fixed binding path mismatch: {name}")
        if not path.is_file() or str(binding.get("sha256")) != rt.file_sha256(path):
            raise rt.ProtocolError(f"fixed binding hash mismatch: {name}")
    for path, expected in (
        (config.P_DRIVERS, config.P_DRIVERS_SHA256),
        (config.P_CONTROL, config.P_CONTROL_SHA256),
        (config.D_DRIVERS, config.D_DRIVERS_SHA256),
        (config.D_MASKS, config.D_MASKS_SHA256),
        (config.OLD_PROTOCOL, config.OLD_PROTOCOL_SHA256),
        (config.OLD_INPUT_AUDIT, config.OLD_INPUT_AUDIT_SHA256),
        (config.OLD_DRIVERS, config.OLD_DRIVERS_SHA256),
        (config.WAV2LIP_CHECKPOINT, config.WAV2LIP_CHECKPOINT_SHA256),
        (config.SYNCNET_MODEL, config.SYNCNET_MODEL_SHA256),
    ):
        if not path.is_file() or rt.file_sha256(path) != expected:
            raise rt.ProtocolError(f"fixed asset changed or missing: {path}")


def _validate_old_driver_replay(saved: dict[str, Any]) -> int:
    """Recheck the fourteen successful rows from the prior phone-core run."""
    old = rt.load_self(config.OLD_DRIVERS)
    old_rows = {str(row["sample_id"]): row for row in old.get("rows", [])}
    current_rows = {str(row["sample_id"]): row for row in saved.get("rows", [])}
    if len(old_rows) != 16 or len(current_rows) != 16:
        raise rt.ProtocolError("old/current phone-core driver cohort is incomplete")
    successful = [
        row
        for row in old_rows.values()
        if all(arm in row.get("arms", {}) for arm in ("PHONE_CORE", "GENERIC_CORE"))
    ]
    if len(successful) != 14:
        raise rt.ProtocolError(
            f"expected fourteen successful old drivers, got {len(successful)}"
        )
    for old_row in successful:
        sid = str(old_row["sample_id"])
        current = current_rows.get(sid)
        if current is None:
            raise rt.ProtocolError(
                f"old driver sample missing in current cohort: {sid}"
            )
        for arm in ("PHONE_CORE", "GENERIC_CORE"):
            old_binding = old_row["arms"][arm]
            current_binding = current.get("arms", {}).get(arm)
            if not isinstance(current_binding, dict):
                raise rt.ProtocolError(f"current driver arm missing: {sid}/{arm}")
            old_path = Path(str(old_binding["path"]))
            current_path = Path(str(current_binding["path"]))
            if rt.file_sha256(old_path) != str(
                old_binding.get("sha256")
            ) or rt.file_sha256(current_path) != str(current_binding.get("sha256")):
                raise rt.ProtocolError(f"driver binding hash changed: {sid}/{arm}")
            if not np.array_equal(rt.load_mel(old_path), rt.load_mel(current_path)):
                raise rt.ProtocolError(f"old driver replay mismatch: {sid}/{arm}")
    return len(successful)


def _validate_exposure_contract(
    protocol: dict, audit: dict, saved: dict, expected: dict[str, tuple[str, list[int]]]
) -> None:
    audit_rows = audit.get("rows", [])
    if len(audit_rows) != 16 or len(saved.get("rows", [])) != 16:
        raise rt.ProtocolError("exposure audit must retain all 16 records")
    audit_by_id = {}
    for row in audit_rows:
        sid = str(row.get("sample_id"))
        if sid in audit_by_id:
            raise rt.ProtocolError(f"duplicate exposure audit row: {sid}")
        audit_by_id[sid] = row
    if set(audit_by_id) != set(expected):
        raise rt.ProtocolError("exposure audit cohort mismatch")
    groups = {sid: group for sid, (group, _) in expected.items()}
    supported = {group for sid, (group, rows) in expected.items() if rows}
    unsupported = sorted(set(groups.values()) - supported)
    gate = protocol.get("exposure_gate", {})
    if (
        gate.get("unit") != "source_group"
        or gate.get("group_count") != len(set(groups.values()))
        or sorted(gate.get("unsupported_groups", [])) != unsupported
    ):
        raise rt.ProtocolError("protocol exposure gate mismatch")
    for record in saved["rows"]:
        sid = str(record["sample_id"])
        group, exposed_rows = expected[sid]
        audit_row = audit_by_id[sid]
        expected_record_exposed = bool(exposed_rows)
        expected_group_exposed = group not in unsupported
        if (
            str(audit_row.get("source_group")) != group
            or bool(audit_row.get("record_exposed")) != expected_record_exposed
            or bool(audit_row.get("group_exposed")) != expected_group_exposed
            or [int(row) for row in audit_row.get("exposed_rows", [])] != exposed_rows
        ):
            raise rt.ProtocolError(f"exposure audit mismatch: {sid}")


def _validate_controls(paths: config.RunPaths) -> dict:
    control = rt.load_self(paths.controls)
    expected = vm.parent_control_gate(config.P_CONTROL, config.P_DRIVERS)
    for key, value in expected.items():
        if control.get(key) != value:
            raise rt.ProtocolError(f"parent control mismatch: {key}")
    score_manifest = rt.load_self(paths.control_scores)
    if (
        score_manifest.get("status") != "complete"
        or score_manifest.get("protocol_id") != "wav2lip_phone_core_shrinkage"
    ):
        raise rt.ProtocolError("control score manifest protocol/status mismatch")
    fresh_rows = list(score_manifest.get("rows", []))
    parent = rt.load_self(config.P_CONTROL)
    _validate_replay_rows(fresh_rows, parent["rows"])
    _validate_parity_rows(fresh_rows, parent["rows"])
    if (
        len(fresh_rows) != 4
        or score_manifest.get("fresh_video_count") != 2
        or control.get("fresh_video_count") != 2
        or control.get("fresh_control_count") != 4
    ):
        raise rt.ProtocolError("fresh control count mismatch")
    return rt.write_json(
        paths.control_validation,
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_phone_core_shrinkage",
            "status": "PASS",
            "independent": True,
            "control_pass": bool(control.get("status") == "PASS"),
            "control_sha256": rt.file_sha256(paths.controls),
            "scores_sha256": rt.file_sha256(paths.control_scores),
            "parent_control_sha256": rt.file_sha256(config.P_CONTROL),
            "fresh_control_count": len(fresh_rows),
            "checks": [
                "independent parent control reconstruction",
                "independent fresh replay/parity matrix validation",
                "fresh worker matrix validation",
            ],
        },
    )


def validate(root: Path, stage: str = "all") -> dict:
    paths = config.RunPaths(root)
    protocol = rt.load_self(paths.protocol)
    saved = rt.load_self(paths.drivers)
    groups = [str(row["source_group"]) for row in saved["rows"]]
    if (
        len(saved["rows"]) != 16
        or len(set(groups)) != 8
        or any(groups.count(group) != 2 for group in set(groups))
    ):
        raise rt.ProtocolError("cohort is not 16/8 with two records per group")
    _validate_fixed_bindings(protocol)
    old_driver_replay_count = _validate_old_driver_replay(saved)
    if stage == "prepare":
        return rt.write_json(
            paths.validation,
            {
                "schema_version": 1,
                "protocol_id": "wav2lip_phone_core_shrinkage",
                "status": "PASS",
                "independent": True,
                "stage": "prepare",
                "record_count": 16,
                "source_group_count": len(
                    {str(row["source_group"]) for row in saved["rows"]}
                ),
                "old_driver_replay_count": old_driver_replay_count,
                "candidate_authorized": False,
            },
        )
    if stage not in {"controls", "all"}:
        raise rt.ProtocolError(f"unknown validation stage: {stage}")
    if stage == "controls":
        return _validate_controls(paths)
    if protocol.get("scientific_decision") != "INPUT_DEGENERATE":
        _validate_controls(paths)
    p_drivers = rt.load_self(config.P_DRIVERS)
    d = rt.read_json(config.D_DRIVERS)
    masks = {
        str(item["mask_sha256"]): item for item in rt.read_json(config.D_MASKS)["masks"]
    }
    p_by_id = {str(row["sample_id"]): row for row in p_drivers["rows"]}
    control = rt.load_self(config.P_CONTROL)
    records = []
    if protocol.get("scientific_decision") != "INPUT_DEGENERATE":
        expected_exposure = {}
        for row in saved["rows"]:
            sid = str(row["sample_id"])
            expected_exposure[sid] = (
                str(row["source_group"]),
                _exposed_rows(_cores(sid, d["drivers"], masks)),
            )
        _validate_exposure_contract(
            protocol, rt.load_self(paths.input_audit), saved, expected_exposure
        )
    if protocol.get("scientific_decision") == "INPUT_DEGENERATE":
        reconstructed: dict[str, tuple[str, bool]] = {}
        construction_failures = []
        for row in saved["rows"]:
            sid = str(row["sample_id"])
            natural = rt.load_mel(Path(str(p_by_id[sid]["arms"]["N"]["path"])))
            try:
                _build(natural, _cores(sid, d["drivers"], masks))
            except rt.ProtocolError as exc:
                if "degenerate" not in str(exc).lower():
                    raise
                construction_failures.append(sid)
                reconstructed[sid] = (str(row["source_group"]), False)
            else:
                reconstructed[sid] = (
                    str(row["source_group"]),
                    bool(_exposed_rows(_cores(sid, d["drivers"], masks))),
                )
        if construction_failures:
            expected = [str(row["sample_id"]) for row in saved["rows"]]
        else:
            exposed_groups = {
                group for group, exposed in reconstructed.values() if exposed
            }
            unsupported_groups = {
                group
                for group, _ in reconstructed.values()
                if group not in exposed_groups
            }
            expected = [
                sid
                for sid, (group, _) in reconstructed.items()
                if group in unsupported_groups
            ]
        actual = list(rt.load_self(paths.input_audit).get("degenerate_ids", []))
        if expected != actual or not actual or len(reconstructed) != 16:
            raise rt.ProtocolError("degenerate cohort does not independently reproduce")
        return rt.write_json(
            paths.validation,
            {
                "schema_version": 1,
                "protocol_id": "wav2lip_phone_core_shrinkage",
                "status": "PASS",
                "independent": True,
                "engineering_decision": "GO",
                "scientific_decision": "INPUT_DEGENERATE",
                "degenerate_ids": actual,
                "old_driver_replay_count": old_driver_replay_count,
                "candidate_authorized": False,
            },
        )
    scores = rt.load_self(paths.scores)
    if len(scores["rows"]) != 36:
        raise rt.ProtocolError("missing phone-core scores")
    candidate_rows = [
        row
        for row in scores["rows"]
        if str(row["video_arm"]) in {"PHONE_CORE", "GENERIC_CORE"}
    ]
    fresh_rows = [
        row
        for row in scores["rows"]
        if str(row["video_arm"]) == "N_REPLAY"
        or str(row["video_arm"]).startswith("FRESH_PARITY_")
    ]
    if len(candidate_rows) != 32 or len(fresh_rows) != 4:
        raise rt.ProtocolError("phone-core candidate/control split mismatch")
    _validate_replay_rows(fresh_rows, control["rows"])
    _validate_parity_rows(fresh_rows, control["rows"])
    score_map = {
        (str(row["sample_id"]), str(row["video_arm"])): row for row in candidate_rows
    }
    for row in saved["rows"]:
        sid = str(row["sample_id"])
        natural = rt.load_mel(Path(str(p_by_id[sid]["arms"]["N"]["path"])))
        expected = _build(natural, _cores(sid, d["drivers"], masks))
        for arm in ("PHONE_CORE", "GENERIC_CORE"):
            actual = rt.load_mel(Path(str(row["arms"][arm]["path"])))
            if not np.array_equal(actual, expected[arm]):
                raise rt.ProtocolError(f"driver mismatch: {sid}/{arm}")
        nrow = next(
            item
            for item in control["rows"]
            if str(item["sample_id"]) == sid
            and str(item["video_arm"]) == "N"
            and str(item["audio_arm"]) == "N"
        )
        item = {
            "sample_id": sid,
            "source_group": str(row["source_group"]),
            "N": _worker_metrics(nrow, f"{sid}/N"),
        }
        for arm in ("PHONE_CORE", "GENERIC_CORE"):
            item[arm] = _worker_metrics(score_map[(sid, arm)], f"{sid}/{arm}")
        records.append(item)
    contrasts = {
        arm: vm.contrast_summary(records, arm) for arm in ("PHONE_CORE", "GENERIC_CORE")
    }
    mechanism = _mechanism(records)
    main_pass = vm.gain_pass(contrasts["PHONE_CORE"])
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
    analysis = rt.load_self(paths.analysis)
    if (
        analysis.get("contrasts") != contrasts
        or analysis.get("mechanism") != mechanism
        or analysis.get("scientific_decision") != decision
    ):
        raise rt.ProtocolError("analysis tamper or reconstruction mismatch")
    return rt.write_json(
        paths.validation,
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_phone_core_shrinkage",
            "status": "PASS",
            "independent": True,
            "engineering_decision": "GO",
            "scientific_decision": decision,
            "contrasts": contrasts,
            "mechanism": mechanism,
            "old_driver_replay_count": old_driver_replay_count,
            "candidate_authorized": False,
            "checks": [
                "independent masks and weights",
                "independent mel reconstruction",
                "independent SyncNet matrices",
                "shared grouped 99% bootstrap",
                "old fourteen-driver replay",
            ],
        },
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("prepare", "controls", "all"), default="all"
    )
    args = parser.parse_args()
    try:
        validate(args.run_root, stage=args.stage)
        return 0
    except Exception as exc:  # noqa: BLE001
        target = (
            config.RunPaths(args.run_root).control_validation
            if args.stage == "controls"
            else args.run_root / "validation.json"
        )
        rt.write_json(
            target,
            {
                "schema_version": 1,
                "protocol_id": "wav2lip_phone_core_shrinkage",
                "status": "FAIL",
                "engineering_decision": "BLOCKED",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
