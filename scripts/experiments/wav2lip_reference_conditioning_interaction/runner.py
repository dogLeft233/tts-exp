from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from . import config
from .common import (
    ProtocolError,
    file_sha256,
    gpu_lock,
    group_bootstrap,
    load_self_hashed,
    write_json,
)
from .media import delayed_audio, mux
from .score import ScoreEngine, domain_metrics, metrics, worker_arrays


def _paths(run_id: str) -> config.RunPaths:
    root = config.run_root_for(run_id)
    root.mkdir(parents=True, exist_ok=True)
    return config.RunPaths(root)


def _repo_key(path: Path) -> str:
    return path.resolve().relative_to(config.REPO.resolve()).as_posix()


def _hash_paths(paths: list[Path] | tuple[Path, ...]) -> dict[str, str]:
    return {_repo_key(path): file_sha256(path) for path in paths}


def _identity_hashes() -> dict[str, dict[str, str]]:
    spec_root = (
        config.REPO / "openspec/changes/complete-wav2lip-reference-matched-control"
    )
    code_paths = tuple(sorted(Path(__file__).parent.glob("*.py")))
    spec_paths = (
        spec_root / "proposal.md",
        spec_root / "design.md",
        spec_root / "specs/wav2lip-reference-matched-control/spec.md",
    )
    return {
        "code_sha256": _hash_paths(code_paths),
        "helper_sha256": _hash_paths(config.HELPER_PATHS),
        "spec_sha256": _hash_paths(spec_paths),
    }


def _fixed() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    fixed = {
        "drivers": config.DRIVERS,
        "control_scores": config.CONTROL_SCORES,
        "q_candidates": config.Q_CANDIDATES,
        "old_control_scores": config.OLD_CONTROL_SCORES,
        "old_protocol": config.OLD_PROTOCOL,
        "old_drivers": config.OLD_DRIVERS,
        "old_reference": config.OLD_REFERENCE,
        "checkpoint": config.WAV2LIP_CHECKPOINT,
        "syncnet": config.SYNCNET_MODEL,
    }
    for name, path in fixed.items():
        if not path.is_file() or file_sha256(path) != config.FIXED_HASHES[name]:
            raise ProtocolError(f"fixed asset changed or missing: {path}")
    return (
        load_self_hashed(config.DRIVERS),
        load_self_hashed(config.CONTROL_SCORES),
        load_self_hashed(config.Q_CANDIDATES),
        load_self_hashed(config.OLD_CONTROL_SCORES),
    )


def _frame46(video_path: Path, box_path: Path) -> np.ndarray:
    boxes = (
        json.loads(box_path.read_text(encoding="utf-8"))
        if box_path.suffix == ".json"
        else None
    )
    if not isinstance(boxes, list) or len(boxes) <= 46:
        raise ProtocolError(f"frame 46 box missing: {box_path}")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open source video: {video_path}")
    frame = None
    try:
        for _ in range(47):
            ok, value = capture.read()
            if not ok:
                raise ProtocolError(
                    f"source video has no zero-based frame 46: {video_path}"
                )
            frame = value
    finally:
        capture.release()
    assert frame is not None
    box = boxes[46]
    if not isinstance(box, list) or len(box) != 4:
        raise ProtocolError(f"invalid frame 46 box: {box}")
    x1, y1, x2, y2 = (int(value) for value in box)
    if not (0 <= x1 < x2 <= frame.shape[1] and 0 <= y1 < y2 <= frame.shape[0]):
        raise ProtocolError(f"invalid frame 46 box: {box}")
    return np.ascontiguousarray(
        cv2.resize(frame[y1:y2, x1:x2], (224, 224), interpolation=cv2.INTER_LINEAR)
    )


def _prepare(paths: config.RunPaths, resume: bool) -> dict[str, Any]:
    drivers, _parent_scores, _q, _old_scores = _fixed()
    rows = sorted(
        drivers.get("rows", []),
        key=lambda row: (str(row["source_group"]), str(row["sample_id"])),
    )
    old_refs = {
        str(row["sample_id"]): row
        for row in load_self_hashed(config.OLD_REFERENCE).get("rows", [])
    }
    groups = [str(row["source_group"]) for row in rows]
    if (
        len(rows) != config.RECORD_COUNT
        or len(set(groups)) != config.GROUP_COUNT
        or any(groups.count(group) != 2 for group in set(groups))
        or len(old_refs) != config.RECORD_COUNT
    ):
        raise ProtocolError("frozen cohort is not 16 records/8 groups")
    if paths.protocol.is_file():
        if not resume:
            raise ProtocolError(f"run exists; use --resume: {paths.root}")
        existing = load_self_hashed(paths.protocol)
        identity = _identity_hashes()
        if any(existing.get(name) != value for name, value in identity.items()):
            raise ProtocolError("resume identity changed; use a new run-id")
        expected_records = [
            {
                "sample_id": str(row["sample_id"]),
                "source_group": str(row["source_group"]),
            }
            for row in rows
        ]
        if (
            existing.get("configuration") != config.configuration()
            or existing.get("records") != expected_records
            or existing.get("cells") != ["F0_N", "F0_C", "F46_N", "F46_C"]
        ):
            raise ProtocolError(
                "resume protocol does not match frozen experiment identity"
            )
        load_self_hashed(paths.drivers)
        load_self_hashed(paths.reference)
        load_self_hashed(paths.input_audit)
        return existing
    out_rows, ref_rows = [], []
    for row in rows:
        sid = str(row["sample_id"])
        source_video, box = (
            Path(str(row["face_video"]["path"])),
            Path(str(row["box"]["path"])),
        )
        if file_sha256(source_video) != str(row["face_video"]["sha256"]) or file_sha256(
            box
        ) != str(row["box"]["sha256"]):
            raise ProtocolError(f"source binding changed: {sid}")
        for binding in (
            row["natural_audio"],
            row["static_face"],
            row["arms"]["N"],
            row["arms"]["CORRECT"],
        ):
            bound_path = Path(str(binding["path"]))
            if not bound_path.is_file() or file_sha256(bound_path) != str(
                binding["sha256"]
            ):
                raise ProtocolError(f"input binding changed: {sid}/{bound_path.name}")
        old_ref = old_refs[sid]["F46"]
        old_f46 = Path(str(old_ref["path"]))
        if file_sha256(old_f46) != str(old_ref["sha256"]):
            raise ProtocolError(f"historical F46 binding changed: {sid}")
        expected = _frame46(source_video, box)
        historical = np.asarray(np.load(old_f46, allow_pickle=False))
        if (
            expected.shape != (224, 224, 3)
            or expected.dtype != np.uint8
            or not np.array_equal(expected, historical)
        ):
            raise ProtocolError(
                f"F46 pixel binding differs from historical reference: {sid}"
            )
        f46_path = paths.root / "reference" / f"{sid}__F46.npy"
        if f46_path.exists() and not np.array_equal(
            np.asarray(np.load(f46_path, allow_pickle=False)), expected
        ):
            raise ProtocolError(f"existing F46 reference changed: {sid}")
        if not f46_path.exists():
            f46_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(f46_path, expected, allow_pickle=False)
        ref_rows.append(
            {
                "sample_id": sid,
                "source_group": str(row["source_group"]),
                "frame_index": 46,
                "source_video": row["face_video"],
                "box": row["box"],
                "F0": row["static_face"],
                "F46": {
                    "path": str(f46_path.resolve()),
                    "sha256": file_sha256(f46_path),
                    "shape": [224, 224, 3],
                    "dtype": "uint8",
                },
                "historical_F46": {
                    "path": str(old_f46.resolve()),
                    "sha256": str(old_ref["sha256"]),
                },
            }
        )
        out_rows.append(
            {
                "sample_id": sid,
                "source_group": str(row["source_group"]),
                "natural_audio": row["natural_audio"],
                "F0": row["static_face"],
                "F46": {
                    "path": str(f46_path.resolve()),
                    "sha256": file_sha256(f46_path),
                },
                "mel": {"N": row["arms"]["N"], "CORRECT": row["arms"]["CORRECT"]},
            }
        )
    protocol = {
        "schema_version": 1,
        "protocol_id": "wav2lip_reference_conditioning_interaction",
        "protocol_revision": "fixed_frame46_factorial_matched_control_v2",
        "status": "locked",
        "configuration": config.configuration(),
        **_identity_hashes(),
        "records": [
            {"sample_id": row["sample_id"], "source_group": row["source_group"]}
            for row in out_rows
        ],
        "record_count": config.RECORD_COUNT,
        "source_group_count": config.GROUP_COUNT,
        "cells": ["F0_N", "F0_C", "F46_N", "F46_C"],
        "fixed_anchor": "F0_N_k0_shared",
        "limits": {
            "fresh_videos": 36,
            "fresh_scores": 54,
            "replacement_confirmed": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "historical_shift_gate_repaired": False,
        },
    }
    audit = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "complete",
        "record_count": config.RECORD_COUNT,
        "source_group_count": config.GROUP_COUNT,
        "frame_index": 46,
        "rows": [
            {
                "sample_id": row["sample_id"],
                "source_video_sha256": row["source_video"]["sha256"],
                "box_sha256": row["box"]["sha256"],
                "F46_sha256": row["F46"]["sha256"],
                "historical_F46_sha256": row["historical_F46"]["sha256"],
            }
            for row in ref_rows
        ],
    }
    write_json(
        paths.reference,
        {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "status": "complete",
            "rows": ref_rows,
        },
    )
    write_json(paths.input_audit, audit)
    write_json(
        paths.drivers,
        {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "status": "complete",
            "rows": out_rows,
        },
    )
    return write_json(paths.protocol, protocol)


def _gpu(
    paths: config.RunPaths, label: str, rows: list[dict[str, Any]], resume: bool
) -> dict[str, Any]:
    plan = paths.root / "plans" / f"{label}.json"
    result = paths.root / "generation" / f"{label}.json"
    payload = {
        "schema_version": 1,
        "protocol_id": "wav2lip_reference_conditioning_interaction",
        "label": label,
        "rows": rows,
    }
    if plan.exists():
        existing_plan = load_self_hashed(plan)
        existing_body = dict(existing_plan)
        existing_body.pop("artifact_sha256", None)
        if existing_body != payload:
            raise ProtocolError(f"GPU plan changed: {plan}")
    else:
        write_json(plan, payload)
    if result.exists():
        if resume:
            return load_self_hashed(result)
        raise ProtocolError(f"GPU result exists; use --resume: {result}")
    log = paths.root / "logs" / f"{label}.gpu.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        str(config.WAV2LIP_PYTHON),
        "-m",
        "scripts.experiments.wav2lip_reference_conditioning_interaction.gpu_worker",
        "--plan",
        str(plan),
        "--result",
        str(result),
    ]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(config.REPO) + os.pathsep + env.get("PYTHONPATH", "")
    with gpu_lock(config.GPU_LOCK), log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(
            command,
            cwd=str(config.REPO),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0 or not result.exists():
        raise ProtocolError(f"GPU worker failed; see {log}")
    value = load_self_hashed(result)
    value.update({"command": command, "log": str(log.resolve())})
    return write_json(result, value)


def _plan(
    paths: config.RunPaths,
    drivers: dict[str, Any],
    reference: str,
    arm: str,
    label: str,
    subset: int | None = None,
    mel_arm: str = "N",
) -> list[dict[str, Any]]:
    rows = sorted(
        drivers["rows"],
        key=lambda row: (str(row["source_group"]), str(row["sample_id"])),
    )
    if subset is not None:
        rows = rows[:subset]
    return [
        {
            "sample_id": row["sample_id"],
            "static_face": row[reference]["path"],
            "arms": {arm: row["mel"][mel_arm]["path"]},
            "outputs": {
                arm: str(
                    (
                        paths.root
                        / "generation"
                        / label
                        / f"{row['sample_id']}__{arm}.video.mkv"
                    ).resolve()
                )
            },
        }
        for row in rows
    ]


def _score_generated(
    paths: config.RunPaths,
    generated: dict[str, Any],
    drivers: dict[str, Any],
    engine: ScoreEngine,
    label: str,
    audio_map: dict[str, Path] | None = None,
) -> list[dict[str, Any]]:
    by_id = {str(row["sample_id"]): row for row in drivers["rows"]}
    rows = []
    for item in generated["rows"]:
        sid, arm = str(item["sample_id"]), str(item["arm"])
        audio = (audio_map or {}).get(
            sid, Path(str(by_id[sid]["natural_audio"]["path"]))
        )
        media = paths.root / "media" / label / f"{sid}__{arm}.mkv"
        media_info = (
            mux(Path(str(item["video_only"])), audio, media)
            if not media.exists()
            else {
                "output": str(media.resolve()),
                "output_sha256": file_sha256(media),
                "video_only": item["video_only"],
                "audio": str(audio.resolve()),
            }
        )
        score = engine.score(
            media,
            audio,
            paths.root / "scores" / label / f"{sid}__{arm}",
            sample_id=sid,
            video_arm=arm,
            audio_arm="A_DELAY" if audio_map else "N",
        )
        rows.append(
            {
                "sample_id": sid,
                "source_group": str(by_id[sid]["source_group"]),
                "video_arm": arm,
                "audio_arm": "A_DELAY" if audio_map else "N",
                "media": media_info,
                "score": score,
                "fresh": True,
            }
        )
    return rows


def _parity_source_group(drivers: dict[str, Any], sample_id: str) -> str:
    """Parity fixtures are fixed inputs and may sit outside the 16-record cohort."""
    return next(
        (
            str(row["source_group"])
            for row in drivers["rows"]
            if str(row["sample_id"]) == str(sample_id)
        ),
        "parity",
    )


def _repeat_checks(rows: list[dict[str, Any]], repeat: list[dict[str, Any]]) -> bool:
    from scripts.experiments.lrs3_real_video_local_timing.media import (
        decode_video_frames,
    )

    ok = len(repeat) == 2
    for row in repeat:
        base = next(
            (
                item
                for item in rows
                if item["sample_id"] == row["sample_id"]
                and item["video_arm"] == "F46_N"
            ),
            None,
        )
        if base is None:
            ok = False
            continue
        a, b = (
            decode_video_frames(Path(str(base["media"]["output"]))),
            decode_video_frames(Path(str(row["media"]["output"]))),
        )
        ma = np.asarray(np.load(base["score"]["matrix"], allow_pickle=False))
        mb = np.asarray(np.load(row["score"]["matrix"], allow_pickle=False))
        pixel_equal = len(a) == len(b) and all(
            np.array_equal(x, y) for x, y in zip(a, b, strict=True)
        )
        row["repeat"] = {
            "pixel_equal": pixel_equal,
            "matrix_max_abs": float(np.max(np.abs(ma - mb)))
            if ma.shape == mb.shape
            else float("inf"),
            "endpoint_max_abs": max(
                (
                    abs(float(base["score"]["U"][key]) - float(row["score"]["U"][key]))
                    for key in ("C", "D")
                ),
                default=float("inf"),
            ),
            "offset_equal": int(base["score"]["U"]["offset"])
            == int(row["score"]["U"]["offset"]),
            "pcm_equal": base["score"].get("source_pcm_sha256")
            == row["score"].get("source_pcm_sha256"),
        }
        row["repeat"]["passes"] = bool(
            pixel_equal
            and row["repeat"]["matrix_max_abs"] <= 1e-4
            and row["repeat"]["endpoint_max_abs"] <= 1e-6
            and row["repeat"]["offset_equal"]
            and row["repeat"]["pcm_equal"]
        )
        ok = ok and row["repeat"]["passes"]
    return ok


def _historical_control(
    paths: config.RunPaths, old_scores: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    if paths.historical_control.exists():
        return load_self_hashed(paths.historical_control)
    rows = old_scores.get("rows", [])
    by_key = {
        (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row.get(
            "score", row
        )
        for row in rows
    }
    if len(by_key) < config.RECORD_COUNT * 2:
        raise ProtocolError("historical F46 control cells are missing")
    out, damages, groups, offset_pass = [], [], [], []
    for record in protocol["records"]:
        sid, group = str(record["sample_id"]), str(record["source_group"])
        natural = by_key[(sid, "F46_N", "N")]
        delayed = by_key[(sid, "F46_N", "A_DELAY")]
        visual_n, audio_n = worker_arrays(natural)
        visual_d, audio_d = worker_arrays(delayed)
        if not np.array_equal(visual_n, visual_d):
            raise ProtocolError(f"historical F46 visual embeddings differ: {sid}")
        n = domain_metrics(visual_n, audio_n, lag_start=config.NATURAL_LAG_START)
        matched = domain_metrics(visual_n, audio_d, lag_start=config.DELAY_LAG_START)
        legacy = domain_metrics(visual_n, audio_d, lag_start=config.NATURAL_LAG_START)
        offset_delta = int(matched["offset"]) - int(n["offset"])
        damage = float(legacy["curve"][n["min_index"]] - n["curve"][n["min_index"]])
        offset_pass.append(-6 <= offset_delta <= -4)
        damages.append(damage)
        groups.append(group)
        out.append(
            {
                "sample_id": sid,
                "source_group": group,
                "natural": n,
                "matched_delay": matched,
                "legacy_delay": legacy,
                "offset_difference": offset_delta,
                "offset_pass": bool(-6 <= offset_delta <= -4),
                "damage_at_natural_k0": damage,
            }
        )
    bootstrap = group_bootstrap(damages, groups, seed=20260909, draws=10_000)
    passed = bool(
        sum(offset_pass) >= 14
        and bootstrap["ci95"][0] > 0
        and bootstrap["group_positive_count"] >= 7
    )
    return write_json(
        paths.historical_control,
        {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "status": "complete",
            "control_pass": passed,
            "scientific_decision": "CONTROL_PASS" if passed else "CONTROL_FAILED",
            "offset_pass_count": int(sum(offset_pass)),
            "records": out,
            "legacy_damage": bootstrap,
            "lag_domains": {
                "natural": "q=r+j-15/offset=15-j",
                "matched_delay": "q=r+j-10/offset=10-j",
            },
        },
    )


def _replay_checks(f0: list[dict[str, Any]], parent: dict[str, Any]) -> bool:
    by_id = {
        str(row["sample_id"]): row
        for row in parent["rows"]
        if row.get("video_arm") == "N" and row.get("audio_arm") == "N"
    }
    from scripts.experiments.lrs3_real_video_local_timing.media import (
        decode_video_frames,
    )

    passed = len(f0) == 2
    for row in f0:
        sid = str(row["sample_id"])
        old = by_id.get(sid)
        if old is None:
            passed = False
            continue
        a = decode_video_frames(Path(str(row["media"]["output"])))
        b = decode_video_frames(Path(str(old["media"])))
        fresh = np.asarray(np.load(row["score"]["matrix"], allow_pickle=False))
        prior = np.asarray(np.load(old["matrix"], allow_pickle=False))
        equal = len(a) == len(b) and all(
            np.array_equal(x, y) for x, y in zip(a, b, strict=True)
        )
        row["replay"] = {
            "pixel_equal": equal,
            "matrix_max_abs": float(np.max(np.abs(fresh - prior)))
            if fresh.shape == prior.shape
            else float("inf"),
            "passes": bool(
                equal
                and fresh.shape == prior.shape
                and np.max(np.abs(fresh - prior)) <= 1e-4
            ),
        }
        passed = passed and row["replay"]["passes"]
    return passed


def _controls(
    paths: config.RunPaths,
    protocol: dict[str, Any],
    drivers: dict[str, Any],
    parent_scores: dict[str, Any],
    old_scores: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if paths.control_analysis.exists() and resume:
        control = load_self_hashed(paths.control_analysis)
        from .validate import validate

        validation = validate(paths.root, stage="controls")
        if validation.get("status") != "PASS":
            raise ProtocolError("independent control validator is not PASS")
        return control
    historical = _historical_control(paths, old_scores, protocol)
    if not historical.get("control_pass"):
        control = write_json(
            paths.control_analysis,
            {
                "schema_version": 1,
                "protocol_id": protocol["protocol_id"],
                "engineering_status": "GO",
                "control_pass": False,
                "scientific_decision": "CONTROL_FAILED",
                "historical_control": historical,
                "fresh_video_count": 0,
                "fresh_score_count": 0,
                "replacement_confirmed": False,
                "waveform_head_authorized": False,
                "generalization_established": False,
                "historical_shift_gate_repaired": False,
            },
        )
        from .validate import validate

        validation = validate(paths.root, stage="controls")
        if validation.get("status") != "PASS":
            raise ProtocolError("independent control validator is not PASS")
        return control
    engine = ScoreEngine()
    f0 = _score_generated(
        paths,
        _gpu(
            paths,
            "f0_replay",
            _plan(paths, drivers, "F0", "N_REPLAY", "f0_replay", 2, "N"),
            resume,
        ),
        drivers,
        engine,
        "f0_replay",
    )
    f46n = _score_generated(
        paths,
        _gpu(
            paths,
            "f46_n",
            _plan(paths, drivers, "F46", "F46_N", "f46_n", None, "N"),
            resume,
        ),
        drivers,
        engine,
        "f46_n",
    )
    repeats = _score_generated(
        paths,
        _gpu(
            paths,
            "f46_repeat",
            _plan(paths, drivers, "F46", "F46_N_REPEAT", "f46_repeat", 2, "N"),
            resume,
        ),
        drivers,
        engine,
        "f46_repeat",
    )
    delay_audio: dict[str, Path] = {}
    for row in drivers["rows"]:
        sid = str(row["sample_id"])
        target = paths.root / "audio" / "A_DELAY" / f"{sid}.wav"
        delayed_audio(Path(str(row["natural_audio"]["path"])), target)
        delay_audio[sid] = target
    delays = _score_generated(
        paths,
        {
            "rows": [
                {
                    "sample_id": row["sample_id"],
                    "arm": "F46_N",
                    "video_only": row["media"]["video_only"],
                }
                for row in f46n
            ]
        },
        drivers,
        engine,
        "f46_delay",
        delay_audio,
    )
    replay_pass, repeat_pass = (
        _replay_checks(f0, parent_scores),
        _repeat_checks(f46n, repeats),
    )
    parity = []
    for parent in [
        row
        for row in parent_scores["rows"]
        if str(row.get("video_arm", "")).startswith("PARITY_")
    ]:
        p = parent["score"]
        score = engine.score(
            Path(str(p["media"])),
            Path(str(p["source_audio"])),
            paths.root / "scores" / "parity" / str(parent["parity"]["label"]),
            sample_id=str(p["sample_id"]),
            video_arm=str(parent["video_arm"]),
            audio_arm="N",
            expected_rows=None,
        )
        actual = np.asarray(np.load(score["matrix"], allow_pickle=False))
        reference = np.asarray(
            np.load(parent["parity"]["reference_matrix"], allow_pickle=False)
        )
        actual_u, reference_u = (
            metrics(actual, list(config.U_ROWS)),
            metrics(reference, list(config.U_ROWS)),
        )
        endpoint = max(
            abs(float(actual_u[key]) - float(reference_u[key])) for key in ("C", "D")
        )
        source_group = _parity_source_group(drivers, str(p["sample_id"]))
        parity.append(
            {
                "sample_id": p["sample_id"],
                "source_group": source_group,
                "video_arm": p["video_arm"],
                "audio_arm": "N",
                "parity": {
                    "label": parent["parity"]["label"],
                    "matrix_max_abs": float(np.max(np.abs(actual - reference)))
                    if actual.shape == reference.shape
                    else float("inf"),
                    "endpoint_max_abs": endpoint,
                    "offset_equal": int(actual_u["offset"])
                    == int(reference_u["offset"]),
                    "passes": bool(
                        actual.shape == reference.shape
                        and np.max(np.abs(actual - reference)) <= 1e-4
                        and endpoint <= 1e-6
                        and int(actual_u["offset"]) == int(reference_u["offset"])
                    ),
                },
                "score": score,
            }
        )
    n_by = {str(row["sample_id"]): row["score"] for row in f46n}
    d_by = {str(row["sample_id"]): row["score"] for row in delays}
    offset_pass, damage, damage_groups = [], [], []
    for rec in protocol["records"]:
        sid = str(rec["sample_id"])
        n_visual, n_audio = worker_arrays(n_by[sid])
        d_visual, d_audio = worker_arrays(d_by[sid])
        if not np.array_equal(n_visual, d_visual):
            raise ProtocolError(
                f"fresh F46 visual embeddings differ after audio delay: {sid}"
            )
        n = domain_metrics(n_visual, n_audio, lag_start=config.NATURAL_LAG_START)
        matched = domain_metrics(n_visual, d_audio, lag_start=config.DELAY_LAG_START)
        legacy = domain_metrics(n_visual, d_audio, lag_start=config.NATURAL_LAG_START)
        offset_delta = int(matched["offset"]) - int(n["offset"])
        offset_pass.append(-6 <= offset_delta <= -4)
        damage.append(
            float(legacy["curve"][n["min_index"]] - n["curve"][n["min_index"]])
        )
        damage_groups.append(str(rec["source_group"]))
        d_by[sid]["domains"] = {
            "natural": n,
            "matched_delay": matched,
            "legacy_delay": legacy,
            "matched_offset_difference": offset_delta,
        }
    damage_stat = group_bootstrap(damage, damage_groups, seed=20260909, draws=10_000)
    delay_pass = bool(
        sum(offset_pass) >= 14
        and damage_stat["ci95"][0] > 0
        and damage_stat["group_positive_count"] >= 7
    )
    parity_pass = len(parity) == 2 and all(row["parity"]["passes"] for row in parity)
    control_pass = bool(replay_pass and repeat_pass and parity_pass and delay_pass)
    fresh = f0 + f46n + repeats + delays + parity
    write_json(
        paths.control_scores,
        {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "status": "complete",
            "rows": fresh,
            "count": len(fresh),
            "reused_f0_parent": True,
        },
    )
    control = write_json(
        paths.control_analysis,
        {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "engineering_status": "GO",
            "control_pass": control_pass,
            "scientific_decision": "CONTROL_PASS" if control_pass else "CONTROL_FAILED",
            "historical_control": historical,
            "f0_replay_pass": replay_pass,
            "f46_repeat_pass": repeat_pass,
            "parity_pass": parity_pass,
            "delay_control": {
                "offset_pass_count": int(sum(offset_pass)),
                "anchor_damage": damage_stat,
                "pass": delay_pass,
                "domains": {
                    "natural": "q=r+j-15/offset=15-j",
                    "matched_delay": "q=r+j-10/offset=10-j",
                },
            },
            "fresh_video_count": 20,
            "fresh_score_count": 38,
            "replacement_confirmed": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "historical_shift_gate_repaired": False,
        },
    )
    from .validate import validate

    validation = validate(paths.root, stage="controls")
    if validation.get("status") != "PASS":
        raise ProtocolError("independent control validator is not PASS")
    return control


def _candidates(
    paths: config.RunPaths, drivers: dict[str, Any], resume: bool
) -> dict[str, Any]:
    control = load_self_hashed(paths.control_analysis)
    if not control.get("control_pass"):
        raise ProtocolError("F46 controls failed; refusing F46_C generation")
    if (
        not paths.control_validation.is_file()
        or load_self_hashed(paths.control_validation).get("status") != "PASS"
    ):
        raise ProtocolError(
            "independent control validator is not PASS; refusing F46_C generation"
        )
    if paths.candidate_scores.exists() and resume:
        return load_self_hashed(paths.candidate_scores)
    engine = ScoreEngine()
    generated = _gpu(
        paths,
        "f46_c",
        _plan(paths, drivers, "F46", "F46_C", "f46_c", None, "CORRECT"),
        resume,
    )
    rows = _score_generated(paths, generated, drivers, engine, "f46_c")
    return write_json(
        paths.candidate_scores,
        {
            "schema_version": 1,
            "protocol_id": "wav2lip_reference_conditioning_interaction",
            "status": "complete",
            "rows": rows,
            "count": len(rows),
        },
    )


def _q_score(q: dict[str, Any], sid: str, arm: str) -> dict[str, Any]:
    for row in q["rows"]:
        if (
            str(row["sample_id"]) == sid
            and str(row["video_arm"]) == arm
            and str(row["audio_arm"]) == "N"
        ):
            return row["score"]
    raise ProtocolError(f"Q score missing: {sid}/{arm}")


def _parent_score(
    parent: dict[str, Any], sid: str, video: str, audio: str
) -> dict[str, Any]:
    for row in parent["rows"]:
        if (
            str(row.get("sample_id")) == sid
            and str(row.get("video_arm")) == video
            and str(row.get("audio_arm")) == audio
        ):
            return row.get("score", row)
    raise ProtocolError(f"parent score missing: {sid}/{video}/{audio}")


def _analysis(
    paths: config.RunPaths,
    protocol: dict[str, Any],
    drivers: dict[str, Any],
    parent: dict[str, Any],
    q: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    if paths.analysis.exists() and resume:
        return load_self_hashed(paths.analysis)
    control = load_self_hashed(paths.control_analysis)
    if not control.get("control_pass"):
        return write_json(
            paths.analysis,
            {
                "schema_version": 1,
                "protocol_id": protocol["protocol_id"],
                "engineering_status": "GO",
                "scientific_decision": "CONTROL_FAILED",
                "replacement_confirmed": False,
                "waveform_head_authorized": False,
                "generalization_established": False,
                "historical_shift_gate_repaired": False,
            },
        )
    fresh = load_self_hashed(paths.control_scores)
    candidates = load_self_hashed(paths.candidate_scores)
    f46n = {
        str(r["sample_id"]): r["score"]
        for r in fresh["rows"]
        if r["video_arm"] == "F46_N" and r["audio_arm"] == "N"
    }
    f46c = {str(r["sample_id"]): r["score"] for r in candidates["rows"]}
    groups, values, records = [], {"g0": [], "g1": [], "I": []}, []
    for rec in sorted(
        protocol["records"],
        key=lambda row: (str(row["source_group"]), str(row["sample_id"])),
    ):
        sid = str(rec["sample_id"])
        n0, c0, n1, c1 = (
            _parent_score(parent, sid, "N", "N"),
            _q_score(q, sid, "CORRECT"),
            f46n[sid],
            f46c[sid],
        )
        k = int(n0["U"]["min_index"])
        g0 = float(n0["U"]["curve"][k] - c0["U"]["curve"][k])
        g1 = float(n1["U"]["curve"][k] - c1["U"]["curve"][k])
        interaction = g1 - g0
        groups.append(str(rec["source_group"]))
        values["g0"].append(g0)
        values["g1"].append(g1)
        values["I"].append(interaction)
        records.append(
            {
                "sample_id": sid,
                "source_group": str(rec["source_group"]),
                "k0": k,
                "g0": g0,
                "g1": g1,
                "I": interaction,
                "F46_within_reference": {
                    "C": float(n1["U"]["C"] - c1["U"]["C"]),
                    "D": float(c1["U"]["D"] - n1["U"]["D"]),
                    "A": float(n1["U"]["curve"][k] - c1["U"]["curve"][k]),
                },
            }
        )
    inter = group_bootstrap(values["I"], groups)
    passed = bool(
        (inter["ci99"][0] > 0 or inter["ci99"][1] < 0)
        and abs(inter["mean"]) > 0.05
        and max(
            inter["group_positive_count"],
            config.GROUP_COUNT - inter["group_positive_count"],
        )
        >= 7
    )
    decision = (
        "REFERENCE_DEPENDENT_RESPONSE"
        if passed
        else "NO_REFERENCE_INTERACTION_ESTABLISHED"
    )
    return write_json(
        paths.analysis,
        {
            "schema_version": 1,
            "protocol_id": protocol["protocol_id"],
            "engineering_status": "GO",
            "scientific_decision": decision,
            "g0": group_bootstrap(values["g0"], groups),
            "g1": group_bootstrap(values["g1"], groups),
            "interaction": inter,
            "within_reference_negative": float(np.mean(values["g1"])) < 0,
            "records": records,
            "replacement_confirmed": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "historical_shift_gate_repaired": False,
        },
    )


def _finalize(paths: config.RunPaths) -> None:
    analysis = load_self_hashed(paths.analysis)
    validation = (
        load_self_hashed(paths.validation) if paths.validation.is_file() else {}
    )
    control = load_self_hashed(paths.control_analysis)
    candidate_count = 0
    if paths.candidate_scores.is_file():
        candidate_count = len(load_self_hashed(paths.candidate_scores).get("rows", []))
    actual_budget = {
        "new_videos": int(control.get("fresh_video_count", 0)) + candidate_count,
        "new_scores": int(control.get("fresh_score_count", 0)) + candidate_count,
    }
    historical_offset_pass_count = control.get("historical_control", {}).get(
        "offset_pass_count"
    )
    review = write_json(
        paths.review,
        {
            "schema_version": 1,
            "status": "self_reviewed",
            "checks": [
                "frame 46 fixed before scoring",
                "F0 anchor shared",
                "F46 matched delay gate before F46_C",
                "independent embeddings-to-matrix validation",
                "reference interaction not replacement",
            ],
            "decision": analysis.get("scientific_decision"),
        },
    )
    final = write_json(
        paths.final,
        {
            "schema_version": 1,
            "protocol_id": analysis["protocol_id"],
            "status": "complete" if validation.get("status") == "PASS" else "blocked",
            "engineering_status": "GO"
            if validation.get("status") == "PASS"
            else "BLOCKED",
            "scientific_decision": analysis.get("scientific_decision"),
            "analysis_sha256": file_sha256(paths.analysis),
            "validation_sha256": file_sha256(paths.validation)
            if paths.validation.is_file()
            else None,
            "review_sha256": review["artifact_sha256"],
            "historical_control_pass": bool(
                control.get("historical_control", {}).get("control_pass", False)
            ),
            "historical_f46_offset_pass_count": historical_offset_pass_count,
            "fresh_delay_offset_pass_count": control.get("delay_control", {}).get(
                "offset_pass_count"
            ),
            "actual_budget": actual_budget,
            "replacement_confirmed": False,
            "waveform_head_authorized": False,
            "generalization_established": False,
            "historical_shift_gate_repaired": False,
        },
    )
    interaction = analysis.get("interaction", {})
    paths.result.write_text(
        "# Reference conditioning interaction\n\n"
        + f"- decision: `{final['scientific_decision']}`\n- engineering_status: `{final['engineering_status']}`\n- historical F46 control offsets: `{final['historical_f46_offset_pass_count']}/16`; fresh matched-delay offsets passing: `{final['fresh_delay_offset_pass_count']}/16`\n- interaction mean / 99% CI: `{interaction.get('mean')}` / `{interaction.get('ci99')}`; group-positive count: `{interaction.get('group_positive_count')}/8`\n- actual budget: `{actual_budget['new_videos']} videos / {actual_budget['new_scores']} score cells`\n- replacement_confirmed: `false`\n",
        encoding="utf-8",
    )


def run(options: argparse.Namespace) -> int:
    paths = _paths(options.run_id)
    try:
        protocol = _prepare(paths, options.resume)
        _unused_drivers, parent, q, old = _fixed()
        local_drivers = load_self_hashed(paths.drivers)
        if options.stage == "prepare":
            paths.p("error.json").unlink(missing_ok=True)
            return 0
        if options.stage in ("controls", "all"):
            control = _controls(
                paths, protocol, local_drivers, parent, old, options.resume
            )
        elif paths.control_analysis.is_file():
            control = load_self_hashed(paths.control_analysis)
        else:
            raise ProtocolError(
                "requested stage has no control_analysis.json; run controls first"
            )
        if options.stage == "controls":
            paths.p("error.json").unlink(missing_ok=True)
            return 0
        if not control.get("control_pass"):
            if options.stage == "candidates":
                raise ProtocolError("control failed; candidate stage is not authorized")
            _analysis(paths, protocol, local_drivers, parent, q, options.resume)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.experiments.wav2lip_reference_conditioning_interaction.validate",
                    "--run-root",
                    str(paths.root),
                    "--stage",
                    "all",
                ],
                cwd=str(config.REPO),
                check=True,
            )
            _finalize(paths)
            paths.p("error.json").unlink(missing_ok=True)
            return 0
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.experiments.wav2lip_reference_conditioning_interaction.validate",
                "--run-root",
                str(paths.root),
                "--stage",
                "controls",
            ],
            cwd=str(config.REPO),
            check=True,
        )
        if options.stage in ("candidates", "all"):
            _candidates(paths, local_drivers, options.resume)
        if options.stage in ("analyze", "all"):
            if not paths.candidate_scores.is_file():
                raise ProtocolError(
                    "analyze is read-only and candidate_scores/manifest.json is missing"
                )
            _analysis(paths, protocol, local_drivers, parent, q, options.resume)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "scripts.experiments.wav2lip_reference_conditioning_interaction.validate",
                    "--run-root",
                    str(paths.root),
                    "--stage",
                    "all",
                ],
                cwd=str(config.REPO),
                check=True,
            )
            _finalize(paths)
            paths.p("error.json").unlink(missing_ok=True)
        elif options.stage == "candidates":
            paths.p("error.json").unlink(missing_ok=True)
        else:
            paths.p("error.json").unlink(missing_ok=True)
        return 0
    except Exception as exc:  # noqa: BLE001
        write_json(
            paths.p("error.json"),
            {
                "schema_version": 1,
                "protocol_id": "wav2lip_reference_conditioning_interaction",
                "status": "BLOCKED",
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "controls", "candidates", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    return run(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
