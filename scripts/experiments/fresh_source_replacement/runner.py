"""A-route runner: Wav2Lip smoke, generation, fixed-support scoring and analysis.

The runner consumes only the frozen P manifests. It never calls the Ditto
backend; Ditto artifacts, when present in the same run, are a separate model
branch owned by the remote-generation handoff.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.experiments.fresh_source_inputs.protocol import ProtocolError, read_json

try:
    from .common import (
    FPS,
    REPO_ROOT,
    SEEDS,
    SHIFT_SAMPLES,
    SYNCNET_MODEL,
    SYNCNET_MODEL_SHA256,
    VIDEO_FRAMES,
    WAV2LIP_CHECKPOINT,
    WAV2LIP_PYTHON,
    WAV2LIP_SCRIPT,
    ReplacementError,
    branch_dir,
    cell_key,
    decode_pcm,
    decode_video_raw,
    file_sha256,
    media_probe,
    mux_pcm_strict,
    normalize_video,
    read_pcm16,
    read_self_hashed,
    resolve_run_paths,
    seed_environment,
    sha256_bytes,
    validate_frozen_inputs,
    write_self_hashed,
    )
    from .scoring import analyze_model_cells, make_bootstrap_indices
except ImportError:  # ``python scripts/.../runner.py`` CLI invocation
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.experiments.fresh_source_replacement.common import (
        FPS,
        REPO_ROOT,
        SEEDS,
        SHIFT_SAMPLES,
        SYNCNET_MODEL,
        SYNCNET_MODEL_SHA256,
        VIDEO_FRAMES,
        WAV2LIP_CHECKPOINT,
        WAV2LIP_PYTHON,
        WAV2LIP_SCRIPT,
        ReplacementError,
        branch_dir,
        cell_key,
        decode_pcm,
        decode_video_raw,
        file_sha256,
        media_probe,
        mux_pcm_strict,
        normalize_video,
        read_pcm16,
        read_self_hashed,
        resolve_run_paths,
        seed_environment,
        sha256_bytes,
        validate_frozen_inputs,
        write_self_hashed,
    )
    from scripts.experiments.fresh_source_replacement.scoring import (
        analyze_model_cells,
        make_bootstrap_indices,
    )

MODEL = "wav2lip"
ARMS = ("N", "C")
DELAY_ARM = "DELAY"


def _status_from_cohort(run_root: Path) -> tuple[str, list[str]]:
    _root, shared = resolve_run_paths(run_root)
    path = shared / "cohort.json"
    try:
        cohort = read_json(path)
    except ProtocolError as exc:
        return "BLOCKED_NEW_SOURCE", [f"cannot read cohort.json: {exc}"]
    return str(cohort.get("status", "BLOCKED_NEW_SOURCE")), [str(value) for value in cohort.get("blockers", [])]


def _write_blocked_artifacts(run_root: Path, *, branch: str, status: str, reason: Sequence[str]) -> None:
    root = branch_dir(run_root, branch)
    write_self_hashed(root / "control_validation.json", {
        "schema_version": 1, "status": status, "controls_run": False, "reason": list(reason),
    })
    write_self_hashed(root / "analysis.json", {
        "schema_version": 1, "status": status, "model": MODEL, "formal_cells": 0,
        "expected_formal_cells": 50, "groups": [], "rows": [], "scientific_contrasts": [],
        "replacement_confirmed": False, "cross_generator_status": "BLOCKED_CROSS_GENERATOR",
        "reason": list(reason),
    })
    write_self_hashed(root / "validation.json", {
        "schema_version": 1, "status": "GO", "upstream_status": status, "errors": [],
        "scientific_cells": 0, "replacement_confirmed": False,
        "cross_generator_status": "BLOCKED_CROSS_GENERATOR",
    })
    write_self_hashed(root / "final.json", {
        "schema_version": 1, "status": status, "replacement_confirmed": False,
        "training_authorized": False, "generalization_established": False, "formal_cells": 0,
        "analysis_sha256": file_sha256(root / "analysis.json"),
        "validation_sha256": file_sha256(root / "validation.json"),
        "cross_generator_status": "BLOCKED_CROSS_GENERATOR",
        "reason": list(reason), "human_status": "pending",
        "note": "科学队列未打开；不能把未达到新来源分母的结果解释为阴性或阳性。",
    })


def run_blocked(run_root: Path, branch: str = "A") -> int:
    """Preserve the old blocked-path API used by the P-blocked tests."""

    status, blockers = _status_from_cohort(run_root)
    if status in {"GO", "COHORT_READY"}:
        raise SystemExit("this runner only handles the blocked upstream path")
    _write_blocked_artifacts(run_root, branch=branch, status="BLOCKED_UPSTREAM_NEW_SOURCE", reason=blockers)
    print(f"{branch} status=BLOCKED_UPSTREAM_NEW_SOURCE")
    return 0


def _input_bundle(run_root: Path, *, include_delay: bool = False) -> dict[str, Any]:
    try:
        bundle = validate_frozen_inputs(run_root, require_files=True, require_ready=True)
        return _attach_delay_bundle(bundle) if include_delay else bundle
    except ReplacementError:
        status, blockers = _status_from_cohort(run_root)
        if status not in {"GO", "COHORT_READY"}:
            _write_blocked_artifacts(run_root, branch="A", status="BLOCKED_UPSTREAM_NEW_SOURCE", reason=blockers)
        raise


def _attach_delay_bundle(bundle: Mapping[str, Any]) -> dict[str, Any]:
    """Bind C's immutable +5-frame PCM manifest for A DELAY generation.

    DELAY is a real generator arm, not a scorer-only reindex.  Validate its
    exact PCM/source-index construction here so Wav2Lip receives the frozen
    delayed waveform and strict mux can bind the same bytes later.
    """

    root = Path(bundle["run_root"])
    manifest_path = root / "run" / "C" / "delay_inputs.json"
    try:
        delay = read_self_hashed(manifest_path)
    except (OSError, ReplacementError) as exc:
        raise ReplacementError(f"DELAY manifest is unavailable: {manifest_path}: {exc}") from exc
    if delay.get("status") != "READY":
        raise ReplacementError(f"DELAY manifest is not READY: {delay.get('status')!r}")
    if int(delay.get("delay_samples", -1)) != SHIFT_SAMPLES or int(delay.get("delay_frames", -1)) != 5:
        raise ReplacementError("DELAY manifest is not the frozen +3200-sample/+5-frame contract")
    groups = delay.get("groups")
    if not isinstance(groups, list) or len(groups) != len(bundle["formal_records"]):
        raise ReplacementError("DELAY manifest does not contain exactly twelve formal groups")
    by_key = {
        (str(item.get("source_group")), str(item.get("sample_id"))): item
        for item in groups
        if isinstance(item, Mapping)
    }
    formal_keys = {
        (str(item["source_group"]), str(item["sample_id"]))
        for item in bundle["formal_records"]
    }
    updated_records: list[dict[str, Any]] = []
    for record in bundle["records"]:
        item = dict(record)
        if (str(item["source_group"]), str(item["sample_id"])) not in formal_keys:
            updated_records.append(item)
            continue
        key = (str(item["source_group"]), str(item["sample_id"]))
        entry = by_key.get(key)
        if not isinstance(entry, Mapping) or not isinstance(entry.get("delay"), Mapping):
            raise ReplacementError(f"DELAY manifest has no binding for {key[1]}")
        delayed = entry["delay"]
        delay_path = Path(str(delayed.get("path", ""))).expanduser().resolve()
        if not delay_path.is_file() or file_sha256(delay_path) != delayed.get("sha256"):
            raise ReplacementError(f"DELAY audio hash mismatch: {delay_path}")
        delayed_pcm, _delayed_values, delayed_params = read_pcm16(delay_path)
        if delayed_params["frame_count"] != int(item["natural_pcm_sample_count"]):
            raise ReplacementError(f"DELAY PCM length differs from natural audio: {key[1]}")
        if sha256_bytes(delayed_pcm) != delayed.get("pcm_sha256"):
            raise ReplacementError(f"DELAY PCM hash mismatch: {key[1]}")
        natural_pcm, natural_values, natural_params = read_pcm16(Path(str(item["_natural_audio_path"])))
        if delayed_params != natural_params:
            raise ReplacementError(f"DELAY PCM format differs from natural audio: {key[1]}")
        expected_pcm = b"\0" * (SHIFT_SAMPLES * 2) + natural_pcm[: -(SHIFT_SAMPLES * 2)]
        if delayed_pcm != expected_pcm:
            raise ReplacementError(f"DELAY PCM is not exact N[n-3200] with zero prefix: {key[1]}")
        map_binding = entry.get("source_index_map")
        map_path = Path(str(map_binding.get("path", ""))).expanduser().resolve() if isinstance(map_binding, Mapping) else None
        if map_path is None or not map_path.is_file():
            raise ReplacementError(f"DELAY source-index map is missing: {key[1]}")
        try:
            source_map = np.asarray(np.load(map_path, allow_pickle=False), dtype=np.int64)
        except (OSError, ValueError) as exc:
            raise ReplacementError(f"DELAY source-index map cannot be read: {map_path}") from exc
        expected_map = np.full(natural_values.size, -1, dtype=np.int64)
        expected_map[SHIFT_SAMPLES:] = np.arange(natural_values.size - SHIFT_SAMPLES, dtype=np.int64)
        if source_map.shape != expected_map.shape or not np.array_equal(source_map, expected_map):
            raise ReplacementError(f"DELAY source-index map mismatch: {key[1]}")
        item["_delay_audio_path"] = delay_path
        item["_delay_pcm_sha256"] = sha256_bytes(delayed_pcm)
        item["_delay_container_sha256"] = file_sha256(delay_path)
        item["_delay_source_index_map_path"] = map_path
        updated_records.append(item)
    updated = dict(bundle)
    updated["records"] = updated_records
    updated["formal_records"] = updated_records[: len(bundle["formal_records"])]
    updated["smoke_records"] = updated_records[len(bundle["formal_records"]):]
    updated["delay_inputs"] = delay
    updated["delay_inputs_path"] = manifest_path
    updated["delay_inputs_sha256"] = file_sha256(manifest_path)
    return updated


def _ingest_ditto_manifest(bundle: Mapping[str, Any], root: Path) -> dict[str, Any]:
    """Read-only audit of the available Ditto replacement manifest.

    A Wav2Lip result is never promoted to a two-generator conclusion merely
    because Ditto files happen to exist.  This artifact records the immutable
    Ditto evidence when its 14 N/C pairs pass the same frozen PCM/media checks;
    otherwise the terminal branch remains explicitly BLOCKED_CROSS_GENERATOR.
    """

    candidates = (
        root.parent / "replacement" / "ditto" / "replacement_manifest.json",
        root.parent / "remote_outputs" / "ditto" / "replacement_manifest.json",
    )
    source = next((path for path in candidates if path.is_file()), None)
    errors: list[str] = []
    if source is None:
        result = {
            "schema_version": 1,
            "model": "ditto",
            "status": "BLOCKED_CROSS_GENERATOR",
            "read_only": True,
            "record_count": 0,
            "errors": ["Ditto replacement manifest is missing"],
            "replacement_confirmed": False,
        }
        return write_self_hashed(root / "ditto_ingest.json", result)
    try:
        try:
            manifest = read_self_hashed(source)
        except ReplacementError:
            manifest = read_json(source)
    except (OSError, ProtocolError, ReplacementError) as exc:
        errors.append(f"cannot read Ditto manifest: {exc}")
        manifest = {}
    records = manifest.get("records") if isinstance(manifest, Mapping) else None
    frozen = {str(row["sample_id"]): row for row in bundle["records"]}
    seen: set[tuple[str, str]] = set()
    summaries: list[dict[str, Any]] = []
    if not isinstance(records, list):
        errors.append("Ditto manifest records is not a list")
        records = []
    for item in records:
        if not isinstance(item, Mapping):
            errors.append("Ditto manifest contains a non-object record")
            continue
        sample_id = str(item.get("sample_id", ""))
        condition = str(item.get("condition", "")).lower()
        arm = "N" if condition.startswith("natural") else "C" if condition.startswith(("direct", "candidate")) else ""
        record = frozen.get(sample_id)
        output = Path(str(item.get("output", ""))).expanduser().resolve()
        if not sample_id or arm not in ARMS or record is None:
            errors.append(f"Ditto record has unknown sample/condition: {sample_id}/{condition}")
            continue
        pair = (sample_id, arm)
        if pair in seen:
            errors.append(f"Ditto record is duplicated: {sample_id}/{arm}")
            continue
        seen.add(pair)
        if int(item.get("returncode", -1)) != 0:
            errors.append(f"Ditto record failed: {sample_id}/{arm}")
        if not output.is_file():
            errors.append(f"Ditto output is missing: {output}")
            continue
        output_sha = file_sha256(output)
        declared_sha = item.get("muxed_file_sha256") or item.get("output_sha256")
        if declared_sha and output_sha != declared_sha:
            errors.append(f"Ditto output hash mismatch: {sample_id}/{arm}")
        try:
            probe = media_probe(output)
            pcm_sha = sha256_bytes(decode_pcm(output))
        except (OSError, ReplacementError) as exc:
            errors.append(f"Ditto media cannot be decoded: {sample_id}/{arm}: {exc}")
            continue
        expected_pcm = str(record["natural_pcm_sha256"] if arm == "N" else (record.get("direct_audio") or {}).get("decoded_pcm_sha256"))
        if pcm_sha != expected_pcm:
            errors.append(f"Ditto PCM differs from frozen {arm}: {sample_id}/{arm}")
        if probe["frame_count"] < VIDEO_FRAMES or abs(float(probe["fps"]) - FPS) > 1e-6 or not probe.get("has_audio"):
            errors.append(f"Ditto media clock/audio contract failed: {sample_id}/{arm}")
        if item.get("audio_pcm_verified") is not True or item.get("video_stream_copy_verified") is not True:
            errors.append(f"Ditto manifest mux evidence is not verified: {sample_id}/{arm}")
        summaries.append({
            "sample_id": sample_id,
            "source_group": str(record["source_group"]),
            "arm": arm,
            "path": str(output),
            "sha256": output_sha,
            "pcm_sha256": pcm_sha,
            "frame_count": int(probe["frame_count"]),
            "fps": float(probe["fps"]),
        })
    expected_pairs = {(str(row["sample_id"]), arm) for row in bundle["records"] for arm in ARMS}
    if seen != expected_pairs:
        errors.append(f"Ditto pair denominator mismatch: expected {len(expected_pairs)}, got {len(seen)}")
    # A prior one-seed Ditto audit is useful provenance but is not the
    # two-generator endpoint specified for this run.  Require explicit
    # per-cell seed/repeat keys before allowing COMPLETE; never promote a
    # single N/C pair into a cross-generator comparison.
    formal_ids = {str(row["sample_id"]) for row in bundle["formal_records"]}
    expected_cells = {
        (sample_id, arm, seed, repeat)
        for sample_id in formal_ids
        for arm in ARMS
        for seed in SEEDS
        for repeat in ((0, 1) if arm == "N" and sample_id in {str(row["sample_id"]) for row in bundle["formal_records"][:2]} and seed == 42 else (0,))
    }
    declared_cells = {
        (str(item.get("sample_id", "")), "N" if str(item.get("condition", "")).lower().startswith("natural") else "C", int(item.get("seed", -1)), int(item.get("repeat_index", 0)))
        for item in records
        if isinstance(item, Mapping) and item.get("seed") is not None
    }
    if not expected_cells.issubset(declared_cells):
        errors.append("Ditto manifest lacks the frozen two-seed/repeat cell contract")
    result = {
        "schema_version": 1,
        "model": "ditto",
        "status": "COMPLETE" if not errors and len(summaries) == len(expected_pairs) else "BLOCKED_CROSS_GENERATOR",
        "read_only": True,
        "source_manifest": str(source.resolve()),
        "source_manifest_sha256": file_sha256(source),
        "record_count": len(summaries),
        "expected_record_count": len(expected_pairs),
        "records": summaries,
        "errors": errors,
        "replacement_confirmed": False,
    }
    return write_self_hashed(root / "ditto_ingest.json", result)


def _artifact_contract(bundle: Mapping[str, Any]) -> dict[str, Any]:
    spec_paths = [
        REPO_ROOT / "openspec/changes/probe-fresh-source-cross-generator/design.md",
        REPO_ROOT / "openspec/changes/probe-fresh-source-cross-generator/specs/fresh-source-replacement/spec.md",
    ]
    spec_manifest = [{"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)} for path in spec_paths if path.is_file()]
    code_paths = [
        Path(__file__),
        Path(__file__).with_name("common.py"),
        Path(__file__).with_name("scoring.py"),
        Path(__file__).with_name("validate.py"),
    ]
    code_manifest = [{"path": str(path.relative_to(REPO_ROOT)), "sha256": file_sha256(path)} for path in code_paths if path.is_file()]
    return {
        "schema_version": 1, "stage": "A", "model": MODEL,
        "inputs_sha256": bundle["inputs_sha256"], "cohort_sha256": bundle["cohort_sha256"],
        "spec_manifest": spec_manifest, "code_manifest": code_manifest,
        "wav2lip": {
            "python": str(WAV2LIP_PYTHON), "script": str(WAV2LIP_SCRIPT), "checkpoint": str(WAV2LIP_CHECKPOINT),
            "checkpoint_sha256": file_sha256(WAV2LIP_CHECKPOINT) if WAV2LIP_CHECKPOINT.is_file() else None,
            "fps": FPS, "batch_size": 4, "face_batch_size": 4, "static_reference": True, "nosmooth": True,
        },
        "syncnet": {"model": str(SYNCNET_MODEL), "model_sha256": SYNCNET_MODEL_SHA256, "batch_size": 20, "threads": 2},
        "fixed_support": {
            "frame_count": VIDEO_FRAMES,
            "seeds": list(SEEDS),
            "arms": list(ARMS),
            "delay_arm": DELAY_ARM,
            "delay_samples": SHIFT_SAMPLES,
            "delay_frames": 5,
        },
    }


def _execution_contract(bundle: Mapping[str, Any], root: Path, *, resume: bool) -> dict[str, Any]:
    path = root / "execution_contract.json"
    current = _artifact_contract(bundle)
    if path.is_file():
        prior = read_self_hashed(path)
        prior_body = dict(prior)
        prior_body.pop("artifact_sha256", None)
        if prior_body != current:
            raise ReplacementError("A execution contract changed; use a new run-id")
        return prior
    if resume:
        raise ReplacementError("--resume requested but A execution_contract.json is missing")
    return write_self_hashed(path, current)


def _cell_stem(sample_id: str, arm: str, seed: int, repeat_index: int) -> str:
    return f"{sample_id}__{arm}__seed{int(seed)}__repeat{int(repeat_index)}"


def _audio_for_record(record: Mapping[str, Any], arm: str) -> tuple[Path, str]:
    if arm == "N":
        path = Path(str(record["_natural_audio_path"]))
        pcm_sha = str(record["natural_pcm_sha256"])
    elif arm == "C":
        direct = record.get("direct_audio") or {}
        path = Path(str(record["_direct_audio_path"]))
        pcm_sha = str(direct.get("decoded_pcm_sha256") or "")
    elif arm == DELAY_ARM:
        path = Path(str(record.get("_delay_audio_path", "")))
        pcm_sha = str(record.get("_delay_pcm_sha256") or "")
        if not path.is_file() or not pcm_sha:
            raise ReplacementError("DELAY arm is not bound to a validated frozen delay input")
    else:
        raise ReplacementError(f"unsupported A arm: {arm}")
    return path, pcm_sha


def wav2lip_command(record: Mapping[str, Any], *, arm: str, seed: int, output: Path, boxes_output: Path | None = None) -> list[str]:
    audio, _pcm_sha = _audio_for_record(record, arm)
    command = [
        str(WAV2LIP_PYTHON), "-u", str(WAV2LIP_SCRIPT), "--checkpoint_path", str(WAV2LIP_CHECKPOINT),
        "--face", str(record["_reference_image_path"]), "--audio", str(audio), "--outfile", str(output),
        "--face_det_batch_size", "4", "--wav2lip_batch_size", "4", "--fps", str(FPS), "--nosmooth",
    ]
    if boxes_output is not None:
        command.extend(["--boxes_output", str(boxes_output)])
    return command


def _run_checked(command: Sequence[str], *, cwd: Path, env: Mapping[str, str], log_path: Path) -> None:
    cwd.mkdir(parents=True, exist_ok=True)
    (cwd / "temp").mkdir(parents=True, exist_ok=True)
    merged = os.environ.copy()
    merged.update({str(k): str(v) for k, v in env.items()})
    result = subprocess.run(list(command), cwd=str(cwd), env=merged, capture_output=True, text=True, check=False)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text((result.stdout or "") + ("\n[stderr]\n" + result.stderr if result.stderr else ""), encoding="utf-8")
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()[-3000:]
        raise ReplacementError(f"Wav2Lip failed ({result.returncode}): {detail}")


def _load_video_manifest(path: Path) -> dict[str, Any]:
    if path.is_file():
        return read_self_hashed(path)
    return {"schema_version": 1, "model": MODEL, "status": "INCOMPLETE", "cells": {}, "records": []}


def _write_video_manifest(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    cells = payload.get("cells", {})
    complete = bool(cells) and all(isinstance(item, Mapping) and item.get("status") == "complete" for item in cells.values())
    body = dict(payload)
    body["schema_version"] = 1
    body["status"] = "COMPLETE" if complete else "INCOMPLETE"
    body["cell_count"] = len(cells) if isinstance(cells, Mapping) else 0
    body["records"] = list(cells.values()) if isinstance(cells, Mapping) else []
    return write_self_hashed(path, body)


def _verify_existing_video_cell(item: Mapping[str, Any]) -> bool:
    if item.get("status") != "complete":
        return False
    path = Path(str(item.get("path", "")))
    if not path.is_file() or file_sha256(path) != item.get("sha256"):
        return False
    try:
        probe = media_probe(path)
        return probe["frame_count"] == VIDEO_FRAMES and abs(float(probe["fps"]) - FPS) < 1e-6 and bool(probe.get("has_audio"))
    except (ReplacementError, OSError):
        return False


def _generate_cell(
    bundle: Mapping[str, Any], record: Mapping[str, Any], *, arm: str, seed: int, repeat_index: int,
    root: Path, is_smoke: bool, existing: Mapping[str, Any] | None, resume: bool,
) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    key = cell_key(sample_id, MODEL, arm, seed, repeat_index)
    if resume and existing and _verify_existing_video_cell(existing):
        return dict(existing)
    if not WAV2LIP_PYTHON.is_file():
        raise ReplacementError(f"Wav2Lip Python is unavailable: {WAV2LIP_PYTHON}")
    if not WAV2LIP_SCRIPT.is_file() or not WAV2LIP_CHECKPOINT.is_file():
        raise ReplacementError("Wav2Lip script/checkpoint is unavailable")
    cell_root = root / "work" / _cell_stem(sample_id, arm, seed, repeat_index)
    label = "smoke" if is_smoke else "formal"
    raw = root / "raw" / MODEL / label / arm / f"{_cell_stem(sample_id, arm, seed, repeat_index)}.mp4"
    normalized = root / "normalized" / MODEL / label / f"{_cell_stem(sample_id, arm, seed, repeat_index)}.mkv"
    muxed = root / "videos" / MODEL / label / f"{_cell_stem(sample_id, arm, seed, repeat_index)}.mkv"
    boxes = cell_root / "boxes.json"
    # Upstream inference.py may return rc=0 after ffmpeg fails.  Ensure its
    # output parent exists, then the post-run existence check remains strict.
    raw.parent.mkdir(parents=True, exist_ok=True)
    boxes.parent.mkdir(parents=True, exist_ok=True)
    command = wav2lip_command(record, arm=arm, seed=seed, output=raw, boxes_output=boxes)
    _run_checked(command, cwd=cell_root, env=seed_environment(seed), log_path=root / "logs" / f"{_cell_stem(sample_id, arm, seed, repeat_index)}.log")
    if not raw.is_file():
        raise ReplacementError(f"Wav2Lip exited successfully but did not produce {raw}")
    norm = normalize_video(raw, normalized, frame_count=VIDEO_FRAMES)
    audio, pcm_sha = _audio_for_record(record, arm)
    mux = mux_pcm_strict(normalized, audio, muxed, expected_pcm_sha256=pcm_sha, expected_frame_count=VIDEO_FRAMES)
    return {
        "key": key, "sample_id": sample_id, "source_group": str(record["source_group"]), "model": MODEL,
        "arm": arm, "seed": int(seed), "repeat_index": int(repeat_index), "is_smoke": bool(is_smoke),
        "status": "complete", "path": str(muxed.resolve()), "sha256": file_sha256(muxed),
        "raw_path": str(raw.resolve()), "raw_sha256": file_sha256(raw),
        "normalized_path": str(normalized.resolve()), "normalized_sha256": file_sha256(normalized),
        "command": command, "seed_environment": seed_environment(seed), "normalization": norm, "mux": mux,
        "audio_path": str(audio.resolve()), "audio_pcm_sha256": pcm_sha, "frame_count": VIDEO_FRAMES, "fps": FPS,
        "fresh_forward": bool(arm == DELAY_ARM), "new_forward": bool(arm == DELAY_ARM),
        "natural_control_valid": True,
    }


def _generation_records(bundle: Mapping[str, Any], *, smoke: bool) -> list[tuple[Mapping[str, Any], bool]]:
    # The second smoke group is reserved for input/candidate diagnostics; A's
    # generation budget uses only the first smoke group.
    if smoke:
        return [(bundle["smoke_records"][0], True)]
    return [(record, False) for record in bundle["formal_records"]]


def run_smoke(run_root: Path, *, resume: bool = False, arms: Sequence[str] = ARMS) -> int:
    return _run_generation(run_root, smoke=True, resume=resume, arms=arms)


def run_generate(run_root: Path, *, resume: bool = False, arms: Sequence[str] = ARMS) -> int:
    return _run_generation(run_root, smoke=False, resume=resume, arms=arms)


def _run_generation(run_root: Path, *, smoke: bool, resume: bool, arms: Sequence[str]) -> int:
    try:
        requested_arms = tuple(str(arm) for arm in arms)
        allowed_arms = set(ARMS) | {DELAY_ARM}
        if any(arm not in allowed_arms for arm in requested_arms):
            raise ReplacementError(f"A generation accepts only N,C,DELAY arms; got {list(requested_arms)}")
        if smoke and DELAY_ARM in requested_arms:
            raise ReplacementError("DELAY has no smoke cohort; run formal generation after C delay_inputs is READY")
        bundle = _input_bundle(run_root, include_delay=DELAY_ARM in requested_arms)
        root = branch_dir(bundle["run_root"], "A")
        contract = _execution_contract(bundle, root, resume=resume)
        manifest_path = root / "videos.json"
        prior = _load_video_manifest(manifest_path)
        cells = dict(prior.get("cells", {})) if isinstance(prior.get("cells"), Mapping) else {}
        formal_ids = [str(row["sample_id"]) for row in bundle["formal_records"][:2]]
        for index, (record, is_smoke) in enumerate(_generation_records(bundle, smoke=smoke)):
            if len(requested_arms) < 2:
                arm_order = list(requested_arms)
            else:
                if DELAY_ARM in requested_arms:
                    arm_order = list(requested_arms)
                else:
                    arm_order = ["N", "C"] if index % 2 == 0 else ["C", "N"]
            for arm in arm_order:
                for seed in SEEDS[:1] if is_smoke else SEEDS:
                    repeat_indices = (0,)
                    if not is_smoke and str(record["sample_id"]) in formal_ids and arm == "N" and seed == 42:
                        repeat_indices = (0, 1)
                    for repeat_index in repeat_indices:
                        key = cell_key(str(record["sample_id"]), MODEL, arm, seed, repeat_index)
                        try:
                            cells[key] = _generate_cell(
                                bundle, record, arm=arm, seed=seed, repeat_index=repeat_index, root=root,
                                is_smoke=is_smoke, existing=cells.get(key), resume=resume,
                            )
                            if repeat_index == 1:
                                original = cells.get(cell_key(str(record["sample_id"]), MODEL, arm, seed, 0))
                                if original and cells[key].get("status") == "complete":
                                    left = decode_video_raw(Path(str(original["path"])))
                                    right = decode_video_raw(Path(str(cells[key]["path"])))
                                    if len(left) != len(right):
                                        raise ReplacementError("repeat decoded video byte length differs")
                                    diff = np.abs(np.frombuffer(left, dtype=np.uint8).astype(np.int16) - np.frombuffer(right, dtype=np.uint8).astype(np.int16))
                                    cells[key]["pixel_max_abs"] = int(diff.max()) if diff.size else 0
                                    cells[key]["pixel_mean_abs"] = float(diff.mean()) if diff.size else 0.0
                                    cells[key]["pixel_different_bytes"] = int(np.count_nonzero(diff))
                        except (ReplacementError, OSError, subprocess.SubprocessError) as exc:
                            cells[key] = {
                                "key": key, "sample_id": str(record["sample_id"]), "source_group": str(record["source_group"]),
                                "model": MODEL, "arm": arm, "seed": int(seed), "repeat_index": int(repeat_index),
                                "is_smoke": bool(is_smoke), "status": "failed", "error": str(exc),
                            }
        payload = {
            "schema_version": 1, "model": MODEL, "stage": "smoke" if smoke else "generate", "status": "INCOMPLETE",
            "run_root": str(bundle["run_root"]), "inputs_sha256": bundle["inputs_sha256"], "cohort_sha256": bundle["cohort_sha256"],
            "execution_contract_sha256": contract.get("artifact_sha256"),
            "delay_inputs_sha256": bundle.get("delay_inputs_sha256"),
            "arms": list(requested_arms),
            "expected_cells": (2 if smoke else 50) + (12 if DELAY_ARM in requested_arms and not smoke else 0),
            "cells": cells,
        }
        manifest = _write_video_manifest(manifest_path, payload)
        print(json.dumps({"stage": payload["stage"], "status": manifest["status"], "cells": manifest["cell_count"]}, ensure_ascii=False))
        return 0 if manifest["status"] == "COMPLETE" else 1
    except ReplacementError as exc:
        status, blockers = _status_from_cohort(run_root)
        if status not in {"GO", "COHORT_READY"}:
            _write_blocked_artifacts(run_root, branch="A", status="BLOCKED_UPSTREAM_NEW_SOURCE", reason=blockers)
            return 0
        root = branch_dir(run_root, "A")
        write_self_hashed(root / "generation_error.json", {"schema_version": 1, "status": "BLOCKED_INPUT_CONTRACT", "errors": [str(exc)]})
        print(f"A generation blocked: {exc}", file=sys.stderr)
        return 1


def _forward_fixed_support(
    model: Any,
    frames: list[np.ndarray],
    audio: np.ndarray,
    device: Any,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Run SyncNet heads on their independent supported windows.

    The older helper pairs visual and audio windows and truncates both to
    ``len(frames) - 5``.  A's lag support needs audio q through 144, while a
    140-frame video has only 135 visual windows, so the heads are evaluated
    independently.  No edge padding or synthetic audio rows are introduced.
    """

    import cv2
    import python_speech_features
    import torch

    if len(frames) < VIDEO_FRAMES:
        raise ReplacementError(f"video has fewer than {VIDEO_FRAMES} frames")
    mfcc = np.asarray(python_speech_features.mfcc(audio, 16_000), dtype=np.float32).T
    visual_count = VIDEO_FRAMES - 5
    audio_count = 1 + (mfcc.shape[1] - 20) // 4
    required_audio_rows = 115 - 1 + 15 + 1
    if audio_count < required_audio_rows:
        raise ReplacementError(
            f"audio does not support fixed lag domain: rows={audio_count}, need>={required_audio_rows}"
        )

    # A's canonical videos are 512x512.  SyncNet's fixed visual head consumes
    # 224x224 crops; this is the predeclared single resize applied to every
    # normalized frame, with no re-detection or condition-specific crop.
    resized_frames = [cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA) for frame in frames[:VIDEO_FRAMES]]
    image_stack = np.stack(resized_frames, axis=3)
    image_stack = np.transpose(np.expand_dims(image_stack, axis=0), (0, 3, 4, 1, 2))
    image_tensor = torch.from_numpy(image_stack.astype(np.float32, copy=False))
    audio_tensor = torch.from_numpy(mfcc[np.newaxis, np.newaxis, :, :].astype(np.float32, copy=False))
    visual_batches: list[torch.Tensor] = []
    audio_batches: list[torch.Tensor] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, visual_count, batch_size):
            stop = min(visual_count, start + batch_size)
            image_batch = torch.cat(
                [image_tensor[:, :, row : row + 5, :, :] for row in range(start, stop)], dim=0
            )
            visual_batches.append(model.forward_lip(image_batch.to(device)).detach().cpu().to(torch.float32))
        for start in range(0, audio_count, batch_size):
            stop = min(audio_count, start + batch_size)
            audio_batch = torch.cat(
                [audio_tensor[:, :, :, row * 4 : row * 4 + 20] for row in range(start, stop)], dim=0
            )
            audio_batches.append(model.forward_aud(audio_batch.to(device)).detach().cpu().to(torch.float32))
    visual = torch.cat(visual_batches, dim=0).numpy().astype(np.float32, copy=False)
    audio_embedding = torch.cat(audio_batches, dim=0).numpy().astype(np.float32, copy=False)
    if visual.ndim != 2 or audio_embedding.ndim != 2 or visual.shape[1] != audio_embedding.shape[1]:
        raise ReplacementError(f"unexpected embedding shapes: visual={visual.shape}, audio={audio_embedding.shape}")
    if not np.isfinite(visual).all() or not np.isfinite(audio_embedding).all():
        raise ReplacementError("SyncNet embeddings contain non-finite values")
    return visual, audio_embedding, {
        "visual_rows": int(visual.shape[0]),
        "audio_rows": int(audio_embedding.shape[0]),
        "input_resize": {"width": 224, "height": 224, "interpolation": "INTER_AREA", "source": "normalized_512"},
    }


def _score_cell(cell: Mapping[str, Any], record: Mapping[str, Any], *, output_dir: Path, scorer: Any) -> dict[str, Any]:
    from scripts.experiments.wav2lip_roi_peak_recheck.worker import _extract_media

    path = Path(str(cell["path"]))
    if not path.is_file() or file_sha256(path) != cell.get("sha256"):
        raise ReplacementError(f"video hash changed before scoring: {path}")
    arm = str(cell["arm"])
    audio, pcm_sha = _audio_for_record(record, arm)
    frames, pcm_audio, extraction = _extract_media(path, audio, output_dir / "extract")
    if extraction.get("media_pcm_sha256") != pcm_sha:
        raise ReplacementError(f"scoring media PCM differs from frozen {arm} PCM: {path}")
    visual, audio_embedding, support = _forward_fixed_support(
        scorer.model, frames, pcm_audio, scorer.device, scorer.batch_size
    )
    embedding_path = output_dir / "embeddings.npz"
    embedding_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(embedding_path, visual=np.asarray(visual, dtype=np.float32), audio=np.asarray(audio_embedding, dtype=np.float32))
    return {
        "key": str(cell["key"]), "sample_id": str(cell["sample_id"]), "source_group": str(cell["source_group"]),
        "model": MODEL, "arm": arm, "seed": int(cell["seed"]), "repeat_index": int(cell.get("repeat_index", 0)),
        "status": "complete", "media": str(path.resolve()), "media_sha256": file_sha256(path),
        "source_audio": str(audio.resolve()), "source_audio_sha256": file_sha256(audio), "pcm_sha256": pcm_sha,
        "embedding_path": str(embedding_path.resolve()), "embedding_sha256": file_sha256(embedding_path),
        "visual_shape": list(np.asarray(visual).shape), "audio_shape": list(np.asarray(audio_embedding).shape),
        "fixed_support": support,
        "dtype": "float32", "device": str(scorer.device), "batch_size": int(scorer.batch_size),
        "torch_threads": int(scorer.threads), "extraction": extraction,
    }


def run_score(
    run_root: Path, *, resume: bool = False, device: str = "cpu", threads: int = 2, batch_size: int = 20,
    include_smoke: bool = False,
) -> int:
    try:
        bundle = _input_bundle(run_root)
        root = branch_dir(bundle["run_root"], "A")
        videos = _load_video_manifest(root / "videos.json")
        if not isinstance(videos.get("cells"), Mapping):
            raise ReplacementError("videos.json has no cell map")
        if any(isinstance(cell, Mapping) and cell.get("arm") == DELAY_ARM for cell in videos["cells"].values()):
            bundle = _attach_delay_bundle(bundle)
        _execution_contract(bundle, root, resume=resume)
        score_path = root / "scores.json"
        prior = read_self_hashed(score_path) if score_path.is_file() else {"cells": {}}
        scored = dict(prior.get("cells", {})) if isinstance(prior.get("cells"), Mapping) else {}
        by_id = {str(row["sample_id"]): row for row in bundle["records"]}
        if not SYNCNET_MODEL.is_file() or file_sha256(SYNCNET_MODEL) != SYNCNET_MODEL_SHA256:
            raise ReplacementError(f"SyncNet model missing or hash changed: {SYNCNET_MODEL}")
        from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer

        scorer = SyncNetScorer(SYNCNET_MODEL, device=device, batch_size=batch_size, threads=threads)
        for key, cell in videos["cells"].items():
            if not isinstance(cell, Mapping) or cell.get("status") != "complete" or (cell.get("is_smoke") and not include_smoke):
                continue
            if resume and isinstance(scored.get(key), Mapping):
                prior_row = scored[key]
                emb = Path(str(prior_row.get("embedding_path", "")))
                if prior_row.get("status") == "complete" and emb.is_file() and file_sha256(emb) == prior_row.get("embedding_sha256") and prior_row.get("media_sha256") == cell.get("sha256"):
                    continue
            record = by_id.get(str(cell.get("sample_id")))
            if record is None:
                raise ReplacementError(f"score cell has unknown sample_id: {cell.get('sample_id')}")
            try:
                scored[key] = _score_cell(cell, record, output_dir=root / "embeddings" / key.replace("/", "__"), scorer=scorer)
            except (ReplacementError, OSError, ValueError) as exc:
                scored[key] = {"key": key, "status": "failed", "error": str(exc), "sample_id": cell.get("sample_id"), "arm": cell.get("arm"), "seed": cell.get("seed"), "repeat_index": cell.get("repeat_index", 0)}
        complete_count = sum(isinstance(row, Mapping) and row.get("status") == "complete" for row in scored.values())
        payload = {
            "schema_version": 1, "model": MODEL, "status": "COMPLETE" if complete_count else "INCOMPLETE",
            "inputs_sha256": bundle["inputs_sha256"], "videos_sha256": file_sha256(root / "videos.json"),
            "syncnet_model": str(SYNCNET_MODEL.resolve()), "syncnet_model_sha256": SYNCNET_MODEL_SHA256,
            "device": device, "batch_size": int(batch_size), "threads": int(threads), "cells": scored,
        }
        result = write_self_hashed(score_path, payload)
        print(json.dumps({"stage": "score", "status": result["status"], "complete_cells": complete_count}, ensure_ascii=False))
        return 0 if result["status"] == "COMPLETE" else 1
    except ReplacementError as exc:
        print(f"A score blocked: {exc}", file=sys.stderr)
        return 1


def _load_scored_cells(bundle: Mapping[str, Any], root: Path) -> dict[str, dict[str, Any]]:
    scores = read_self_hashed(root / "scores.json")
    cells: dict[str, dict[str, Any]] = {}
    for key, row in (scores.get("cells") or {}).items():
        if not isinstance(row, Mapping) or row.get("status") != "complete":
            continue
        emb_path = Path(str(row.get("embedding_path", "")))
        if not emb_path.is_file() or file_sha256(emb_path) != row.get("embedding_sha256"):
            raise ReplacementError(f"embedding hash mismatch: {emb_path}")
        with np.load(emb_path, allow_pickle=False) as data:
            if "visual" not in data or "audio" not in data:
                raise ReplacementError(f"embedding artifact lacks visual/audio: {emb_path}")
            item = dict(row)
            item["visual"] = np.asarray(data["visual"], dtype=np.float32)
            item["audio"] = np.asarray(data["audio"], dtype=np.float32)
        cells[str(key)] = item
    return cells


def run_analyze(run_root: Path, *, resume: bool = False) -> int:
    try:
        bundle = _input_bundle(run_root)
        root = branch_dir(bundle["run_root"], "A")
        contract = _execution_contract(bundle, root, resume=resume)
        cells = _load_scored_cells(bundle, root)
        analysis = analyze_model_cells(bundle["formal_records"], cells, model=MODEL, bootstrap_indices=make_bootstrap_indices(12))
        cross_generator = _ingest_ditto_manifest(bundle, root)
        analysis["inputs_sha256"] = bundle["inputs_sha256"]
        analysis["scores_sha256"] = file_sha256(root / "scores.json")
        analysis["videos_sha256"] = file_sha256(root / "videos.json") if (root / "videos.json").is_file() else None
        analysis["execution_contract_sha256"] = contract.get("artifact_sha256")
        analysis["cross_generator_status"] = cross_generator["status"]
        analysis["ditto_ingest_sha256"] = file_sha256(root / "ditto_ingest.json")
        written = write_self_hashed(root / "analysis.json", analysis)
        write_self_hashed(root / "control_validation.json", {
            "schema_version": 1, "status": "PASS" if analysis["controls"]["pass"] else "CONTROL_FAILED", "model": MODEL,
            "inputs_sha256": bundle["inputs_sha256"], "analysis_sha256": file_sha256(root / "analysis.json"),
            "controls": analysis["controls"], "replacement_confirmed": False,
        })
        print(json.dumps({"stage": "analyze", "status": written["status"], "formal_cells": written["formal_cells"]}, ensure_ascii=False))
        return 0
    except (ReplacementError, ValueError, OSError) as exc:
        print(f"A analysis blocked: {exc}", file=sys.stderr)
        return 1


def run_validate(run_root: Path, *, branch: str = "A") -> int:
    try:
        from .validate import validate
    except ImportError:  # direct script execution
        from scripts.experiments.fresh_source_replacement.validate import validate

    try:
        result = validate(Path(run_root).resolve(), branch=branch)
    except (KeyError, OSError, ReplacementError, TypeError, ValueError) as exc:
        print(f"A validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result.get("status") == "GO" else 1


def _parse_arms(value: str) -> tuple[str, ...]:
    arms = tuple(item.strip() for item in value.split(",") if item.strip())
    if not arms:
        raise argparse.ArgumentTypeError("--arms cannot be empty")
    return arms


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "generate", "score", "analyze", "validate", "tts-analyze"), default="analyze")
    parser.add_argument("--branch", choices=("A", "D"), default="A")
    parser.add_argument("--arms", type=_parse_arms, default=ARMS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args(argv)
    run_root = args.run_root.resolve()
    if args.stage == "smoke":
        return run_smoke(run_root, resume=args.resume, arms=args.arms)
    if args.stage == "generate":
        return run_generate(run_root, resume=args.resume, arms=args.arms)
    if args.stage == "score":
        return run_score(run_root, resume=args.resume, device=args.device, threads=args.threads, batch_size=args.batch_size, include_smoke=args.include_smoke)
    if args.stage == "analyze":
        return run_analyze(run_root, resume=args.resume)
    if args.stage == "validate":
        return run_validate(run_root, branch=args.branch)
    print("D/TTS analysis is owned by the deferred measurement branch", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
