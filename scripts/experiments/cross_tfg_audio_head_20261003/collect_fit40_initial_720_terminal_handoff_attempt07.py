#!/usr/bin/env python3
"""Collect a read-only, full 720-row terminal handoff for initial Fit40 ingest (attempt 07).

This collector verifies the frozen historical prefix and the completed live
queue, then emits the Stage-A preselection ingest index plus a separate
terminal-handoff manifest. It never runs the selector, initializes SPSA, or
starts model work. Historical child exit codes remain explicitly unavailable;
only the 86 live queue blocks require a captured numeric child rc of zero.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
from typing import Any, Mapping, Sequence

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.experiments.cross_tfg_audio_head_20261003 import (  # noqa: E402
    fit40_fixed_queue_controller_attempt03 as queue,
    fit_only_final_center_stagea_ingest_attempt05 as final_ingest,
    fit_only_stagea_ingest_attempt03 as ingest,
    run_fit40_fixed_queue_attempt05 as queue_runner,
)


SCHEMA = "fit40_initial_720_terminal_handoff_v1"
FAMILY_PLAN_REL = (
    ".codex/scratch/fit40_spsa_remote_family_attempt03/actual_readback_attempt01/"
    "fit40_three_model_family_plan_attempt03.json"
)
FAMILY_PLAN_BYTES = 2_584_596
FAMILY_PLAN_SHA256 = "9f7914033f9e7e89a60ae2618f7d429860d11de902a8400855fe57e37eb216a2"
FAMILY_PLAN_SELF_SHA256 = "b071434696b2c094180473677d484934d7e3d04c808b7915000c8cfc6c6abb0c"
HISTORICAL_INDEX_REL = (
    ".codex/scratch/fit40_queue_progress_index_attempt09/"
    "fit40_historical_terminal_prefix_index_attempt09.json"
)
HISTORICAL_INDEX_BYTES = 153_666
HISTORICAL_INDEX_SHA256 = "6949e516070ab7d1613f06c4521263f2f390ba6bbae306b37caedc699cb575dc"
HISTORICAL_INDEX_SELF_SHA256 = "0b9002b855ef07562de40b0e722e8eb3689d5ebe98d076a2b99f264900824364"
SCIENTIFIC_INPUT_SHA256 = "ae96def4c16fecabc6e49b2aeabea0a76a0308ea159fa6c7688f7908c62c7180"
COORDINATOR_SOURCE_BYTES = 3_982
COORDINATOR_SOURCE_SHA256 = "a448c29ab601abaed6aece0f3b5cc17082a1aa5440707f9e155bb69d5f2e9ac3"
COORDINATOR_CONFIG_BYTES = 2_097
COORDINATOR_CONFIG_SHA256 = "9b68effc703d44de4776b6c2dd6ea09b64a79fc5b628d09d468dfa509db91213"
HISTORICAL_BLOCKS = {
    "wav2lip_gan_bac9a81e": 23,
    "musetalk_fixed_portrait_attempt16": 6,
    "ditto_hubert_fixed_portrait_attempt18": 5,
}
EXPECTED_HISTORICAL_BLOCKS = 34
EXPECTED_QUEUE_BLOCKS = 86
EXPECTED_TOTAL_BLOCKS = 120
EXPECTED_ROWS = 720
EXPECTED_ROWS_PER_MODEL = 240
EXPECTED_NPZ_ARRAY_NAMES = {
    "query_frame_index", "query_timestamp_num", "query_timestamp_den",
    "track_frame_index", "valid_query_mask", "distance_by_query_lag",
    "lag_offsets_frames",
}
EXPECTED_SOURCE_MODULE_PINS = {
    "fit40_fixed_queue_controller_attempt03.py": (
        80_513, "dafe417e42cd7660b3b0c41731b30e10272f45ab4ead6911ea75c9f1fce1667a"
    ),
    "run_fit40_fixed_queue_attempt05.py": (
        25_969, "38e924bbccb54c04e9c7618502bdc57b38a71595babc71f3d9c51b646165de67"
    ),
    "fit_only_stagea_ingest_attempt03.py": (
        53_184, "22cd7f983b22babcae0d9c120f4720cb84a218ec4d2b84551d5e1668d23aeb95"
    ),
    "fit_only_final_center_stagea_ingest_attempt05.py": (
        106_167, "3384ebe67ed99569af6307c88746a0b16329897dda9935678677dcf80147e8ea"
    ),
}


class HandoffError(ValueError):
    """Terminal evidence is missing, stale, inconsistent, or not safely readable."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def self_hash(value: Mapping[str, Any], field: str) -> str:
    return digest(canonical({key: item for key, item in value.items() if key != field}))


def _is_sha(value: Any) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _read_regular(path: Path, label: str) -> tuple[bytes, dict[str, Any]]:
    try:
        info = path.lstat()
    except OSError as exc:
        raise HandoffError(f"missing {label}: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise HandoffError(f"{label} must be a regular non-symlink file: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise HandoffError(f"cannot read {label}: {path}: {exc}") from exc
    pin = {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": digest(raw)}
    return raw, pin


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    raw, pin = _read_regular(path, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"{label} is not valid UTF-8 JSON: {path}") from exc
    if not isinstance(value, dict):
        raise HandoffError(f"{label} must be a JSON object: {path}")
    return value, pin, raw


def _require_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise HandoffError(f"{label} path must be absolute: {path}")
    try:
        info = path.lstat()
    except OSError as exc:
        raise HandoffError(f"{label} directory is missing: {path}") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise HandoffError(f"{label} must be a regular non-symlink directory: {path}")
    return path.resolve(strict=True)


def _check_self(value: Mapping[str, Any], field: str, label: str) -> str:
    actual = self_hash(value, field)
    if value.get(field) != actual:
        raise HandoffError(f"{label} {field} mismatch")
    return actual


def _historical_rc_is_unknown(selected_record: Mapping[str, Any]) -> bool:
    """Keep historical detached-child rc unavailable; never infer it from PID state."""
    return (selected_record.get("historical_exit_code_available") is False
            and selected_record.get("historical_exit_code") is None)


def _verify_coordinator_source_config_files(
    coordinator_dir: Path, launch: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Verify the exact uploaded coordinator bytes and the launch-pinned config."""
    source_path = coordinator_dir / "queue_coordinator_attempt01.py"
    config_path = coordinator_dir / "queue_coordinator_config_attempt01.json"
    _, source_pin = _read_regular(source_path, "coordinator supervisor source")
    config, config_pin, _ = _read_json(config_path, "coordinator configuration")
    source_ref, config_ref = launch.get("coordinator_source"), launch.get("config")
    if (source_pin["bytes"] != COORDINATOR_SOURCE_BYTES
            or source_pin["sha256"] != COORDINATOR_SOURCE_SHA256
            or config_pin["bytes"] != COORDINATOR_CONFIG_BYTES
            or config_pin["sha256"] != COORDINATOR_CONFIG_SHA256
            or not isinstance(source_ref, Mapping)
            or dict(source_ref) != source_pin
            or not isinstance(config_ref, Mapping)
            or dict(config_ref) != config_pin):
        raise HandoffError("coordinator launch receipt does not bind the exact supervisor/configuration files")
    return source_pin, config_pin, config


def _require_exact_mapping(actual: Any, expected: Mapping[str, Any], label: str) -> None:
    if not isinstance(actual, Mapping) or dict(actual) != dict(expected):
        raise HandoffError(f"{label} differs from the pinned invocation")


def _expected_coordinator_argv(source_path: Path, config_path: Path) -> tuple[list[str], list[str]]:
    """Return the exec argv and Python's distinct ``sys.argv`` value."""
    source = str(source_path.resolve(strict=True))
    config = str(config_path.resolve(strict=True))
    return (["/usr/bin/python3", "-B", source, config], [source, config])


def _require_inside_output_leaf(path: Path, output_leaf: Path, label: str) -> Path:
    """Require an artifact to be a regular, non-symlink descendant of its block leaf."""
    try:
        root_info = output_leaf.lstat()
        if not stat.S_ISDIR(root_info.st_mode) or stat.S_ISLNK(root_info.st_mode):
            raise HandoffError(f"{label} output leaf is not a regular directory: {output_leaf}")
        # Keep the producer's lexical path under its exact selected output leaf,
        # then verify every component so an intermediate symlink cannot escape.
        relative = path.relative_to(output_leaf)
        if not relative.parts or any(part in ("", ".", "..") for part in relative.parts):
            raise HandoffError(f"{label} path is not a strict descendant of its output leaf")
        current = output_leaf
        for part in relative.parts:
            current = current / part
            info = current.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise HandoffError(f"{label} path contains a symlink: {current}")
        if not stat.S_ISREG(current.lstat().st_mode):
            raise HandoffError(f"{label} is not a regular file: {current}")
        resolved_root = output_leaf.resolve(strict=True)
        resolved_path = current.resolve(strict=True)
        resolved_path.relative_to(resolved_root)
        return current
    except HandoffError:
        raise
    except (OSError, ValueError) as exc:
        raise HandoffError(f"{label} is missing or outside its exact block output leaf") from exc


def _pinned_json(path_text: Any, expected_bytes: Any, expected_sha: Any,
                 expected_self: Any | None, label: str, *,
                 self_field: str = "self_sha256") -> tuple[dict[str, Any], dict[str, Any], bytes]:
    if (not isinstance(path_text, str) or not Path(path_text).is_absolute()
            or type(expected_bytes) is not int or expected_bytes <= 0 or not _is_sha(expected_sha)):
        raise HandoffError(f"{label} reference lacks absolute path/size/raw SHA")
    value, pin, raw = _read_json(Path(path_text), label)
    if pin["bytes"] != expected_bytes or pin["sha256"] != expected_sha:
        raise HandoffError(f"{label} raw bytes differ from the declared producer pin")
    if expected_self is not None and _check_self(value, self_field, label) != expected_self:
        raise HandoffError(f"{label} embedded self pin differs from expected value")
    return value, pin, raw


def _file_entry(file_id: str, role: str, declared_path: str, raw: bytes,
                *, pin_name: str | None = None) -> dict[str, Any]:
    if not Path(declared_path).is_absolute():
        raise HandoffError(f"producer-declared evidence path must be absolute: {declared_path}")
    path = Path(declared_path)
    actual, pin = _read_regular(path, role)
    if actual != raw:
        raise HandoffError(f"{role} changed between source read and pin table construction: {declared_path}")
    out = {
        "file_id": file_id, "role": role, "declared_path": declared_path,
        "local_path": pin["path"], "bytes": pin["bytes"], "sha256": pin["sha256"],
    }
    if pin_name is not None:
        out["pin_name"] = pin_name
    return out


def _add_file(files: list[dict[str, Any]], file_ids: set[str], file_id: str,
              role: str, declared_path: str, *, pin_name: str | None = None) -> dict[str, Any]:
    if file_id in file_ids:
        raise HandoffError(f"duplicate file ID: {file_id}")
    raw, _ = _read_regular(Path(declared_path), role)
    entry = _file_entry(file_id, role, declared_path, raw, pin_name=pin_name)
    if any(x["role"] == role and x["declared_path"] == declared_path for x in files):
        raise HandoffError(f"duplicate role/declaration mapping: {role} {declared_path}")
    files.append(entry)
    file_ids.add(file_id)
    return entry


def _load_jsonl(path: Path, expected_pin: Mapping[str, Any], label: str) -> tuple[list[dict[str, Any]], bytes]:
    raw, pin = _read_regular(path, label)
    if pin["bytes"] != expected_pin.get("bytes") or pin["sha256"] != expected_pin.get("sha256"):
        raise HandoffError(f"{label} differs from its frozen source plan pin")
    try:
        lines = raw.decode("utf-8").splitlines()
        values = [json.loads(line) for line in lines]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"{label} is malformed JSONL") from exc
    if any(not isinstance(item, dict) for item in values) or any(not line.strip() for line in lines):
        raise HandoffError(f"{label} must contain only nonblank JSON objects")
    return values, raw


def _load_exact_family(repo_root: Path, family_path: Path, family_sha256: str,
                       family_self_sha256: str) -> tuple[dict[str, Any], dict[str, Any], tuple[Any, ...]]:
    family, family_pin, _ = _pinned_json(
        str(family_path), FAMILY_PLAN_BYTES, family_sha256,
        family_self_sha256, "composed family plan", self_field="plan_sha256",
    )
    if (family.get("schema_version") != queue.PLAN_SCHEMA or family.get("status") != "FROZEN"
            or family.get("scientific_input_pins_sha256") != SCIENTIFIC_INPUT_SHA256
            or family.get("execution", {}).get("dev_or_confirm_access") is not False
            or family.get("execution", {}).get("candidate_selection") != "not performed"):
        raise HandoffError("composed family plan is not the pinned Fit-only, no-selection family")
    if family.get("plan_sha256") != self_hash(family, "plan_sha256"):
        raise HandoffError("composed family plan payload self hash mismatch")
    source_plans = family.get("execution", {}).get("source_model_plans")
    if (not isinstance(source_plans, list)
            or [x.get("model_id") for x in source_plans] != list(queue.MODEL_ORDER)):
        raise HandoffError("composed family source plans do not follow frozen W2L/Muse/Ditto order")
    bundles = queue.load_pinned_bundles(repo_root)
    if len(bundles) != 3:
        raise HandoffError("frozen queue bundle loader did not return exactly three models")
    family_input_pins = family.get("scientific_input_pins")
    if (not isinstance(family_input_pins, Mapping)
            or digest(canonical(family_input_pins)) != SCIENTIFIC_INPUT_SHA256
            or family_input_pins != bundles[0].plan.get("scientific_input_pins")
            or family.get("execution", {}).get("scientific_input_pins_sha256") != SCIENTIFIC_INPUT_SHA256):
        raise HandoffError("composed family changed the exact shared scientific input-pin object")
    for source, bundle in zip(source_plans, bundles):
        for field, pin in (("plan", bundle.spec.plan), ("planned_rows", bundle.spec.planned_rows)):
            ref = source.get(field)
            if (not isinstance(ref, Mapping) or ref.get("bytes") != pin.bytes
                    or ref.get("sha256") != pin.sha256):
                raise HandoffError(f"composed family {bundle.spec.model_id} {field} differs from the fixed queue pin")
            if not isinstance(ref.get("path"), str) or not Path(ref["path"]).is_absolute():
                raise HandoffError(f"composed family {bundle.spec.model_id} {field} path is not absolute")
            actual_raw, actual_pin = _read_regular(Path(ref["path"]),
                                                   f"composed family {bundle.spec.model_id} {field}")
            if actual_pin["bytes"] != pin.bytes or actual_pin["sha256"] != pin.sha256:
                raise HandoffError(f"composed family {bundle.spec.model_id} {field} file differs from its frozen source")
        if len(bundle.rows) != EXPECTED_ROWS_PER_MODEL:
            raise HandoffError(f"{bundle.spec.model_id} does not have exactly 240 frozen rows")
    family_rows = family.get("rows")
    family_execution = family.get("execution", {})
    if (not isinstance(family_rows, list) or len(family_rows) != EXPECTED_ROWS
            or len({row.get("row_id") for row in family_rows if isinstance(row, Mapping)}) != EXPECTED_ROWS
            or [x.get("model_id") for x in family.get("models", [])] != list(queue.MODEL_ORDER)
            or family.get("methods") != bundles[0].plan.get("methods")
            or family_execution.get("model_ids") != list(queue.MODEL_ORDER)
            or family_execution.get("planned_rows") != EXPECTED_ROWS
            or family_execution.get("stage") != "A"):
        raise HandoffError("composed family rows/models/methods differ from the exact frozen source plans")
    for bundle in bundles:
        family_model_rows = [row for row in family_rows if row.get("model_id") == bundle.spec.model_id]
        if family_model_rows != [dict(row) for row in bundle.rows]:
            raise HandoffError(f"composed family rows differ from the exact {bundle.spec.model_id} source plan")
    source_protocols = family.get("execution", {}).get("source_model_protocols")
    family_protocol = family.get("protocol")
    generators = family_protocol.get("generators_by_model") if isinstance(family_protocol, Mapping) else None
    if not isinstance(source_protocols, Mapping) or not isinstance(generators, Mapping):
        raise HandoffError("composed family omits per-model source protocols or generator projection")
    common_protocol = {key: value for key, value in family_protocol.items() if key != "generators_by_model"}
    for bundle in bundles:
        model_id = bundle.spec.model_id
        source_protocol = bundle.plan.get("protocol")
        if not isinstance(source_protocol, Mapping) or source_protocols.get(model_id) != source_protocol:
            raise HandoffError(f"composed family changed the frozen {model_id} source protocol")
        if generators.get(model_id) != source_protocol.get("generator"):
            raise HandoffError(f"composed family generator projection differs for {model_id}")
        shared_projection = {key: value for key, value in source_protocol.items() if key != "generator"}
        if shared_projection != common_protocol:
            raise HandoffError(f"composed family changed shared score protocol fields for {model_id}")
    return family, family_pin, bundles


def _verify_coordinator_time_provenance(
    start: Mapping[str, Any], launch: Mapping[str, Any], terminal: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate producer timestamps without imposing post-Popen receipt order.

    The detached launcher records ``launched_unix_seconds`` only after Popen
    and PID/start-tick capture. The coordinator can therefore have written its
    start receipt and begun its child before that launcher timestamp is stored.
    Keep all three clocks as separate observations; only require the coordinator
    child interval to follow the coordinator start receipt and be internally
    ordered.
    """
    fields = {
        "coordinator_started_unix_seconds": start.get("started_unix_seconds"),
        "launcher_recorded_unix_seconds": launch.get("launched_unix_seconds"),
        "child_started_unix_seconds": terminal.get("started_unix_seconds"),
        "child_ended_unix_seconds": terminal.get("ended_unix_seconds"),
    }
    if any(
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        for value in fields.values()
    ):
        raise HandoffError("coordinator time provenance contains a missing or non-finite timestamp")
    coordinator_started = float(fields["coordinator_started_unix_seconds"])
    child_started = float(fields["child_started_unix_seconds"])
    child_ended = float(fields["child_ended_unix_seconds"])
    if child_started < coordinator_started or child_ended < child_started:
        raise HandoffError("coordinator child time interval is inconsistent with its start receipt")
    return fields


def _verify_coordinator(repo_root: Path, coordinator_dir: Path, state_dir: Path,
                        receipt_dir: Path, history_pin: Mapping[str, Any],
                        history_self_sha256: str, import_pin: Mapping[str, Any],
                        import_self_sha256: str,
                        final_queue_state: Mapping[str, Any]) -> dict[str, Any]:
    launch, launch_pin, _ = _read_json(coordinator_dir / "detached_launch_receipt.json", "coordinator launch receipt")
    start, start_pin, _ = _read_json(coordinator_dir / "coordinator_start_receipt.json", "coordinator start receipt")
    terminal, terminal_pin, _ = _read_json(coordinator_dir / "coordinator_terminal_receipt.json", "coordinator terminal receipt")
    for value, field, expected_schema, label in (
        (launch, "self_sha256", "fit40_queue_detached_launch_receipt_v1", "coordinator launch receipt"),
        (start, "self_sha256", "fit40_queue_detached_coordinator_start_v1", "coordinator start receipt"),
        (terminal, "self_sha256", "fit40_queue_detached_coordinator_terminal_v1", "coordinator terminal receipt"),
    ):
        _check_self(value, field, label)
        if value.get("schema") != expected_schema:
            raise HandoffError(f"{label} schema mismatch")
    source_pin, config_pin, config = _verify_coordinator_source_config_files(coordinator_dir, launch)
    source_path = Path(source_pin["path"])
    config_path = Path(config_pin["path"])
    if (launch.get("status") != "PASS_COORDINATOR_STARTED"
            or launch.get("pid") != start.get("pid")
            or launch.get("start_ticks") != start.get("start_ticks")
            or type(launch.get("pid")) is not int or launch.get("pid") <= 0
            or type(launch.get("start_ticks")) is not int or launch.get("start_ticks") <= 0
            or launch.get("cwd") != str(repo_root.resolve(strict=True))
            or start.get("cwd") != str(repo_root.resolve(strict=True))
            or start.get("queue_state_dir") != str(state_dir.resolve(strict=True))
            or start.get("queue_receipt_dir") != str(receipt_dir.resolve(strict=True))
            or start.get("historical_index") != {
                "path": history_pin["path"], "bytes": history_pin["bytes"],
                "sha256": history_pin["sha256"], "self_sha256": history_self_sha256,
            }
            or start.get("historical_import") != {
                "path": import_pin["path"], "bytes": import_pin["bytes"],
                "sha256": import_pin["sha256"], "self_sha256": import_self_sha256,
            }
            or launch.get("historical_index_raw_sha256") != history_pin["sha256"]
            or launch.get("history_import_raw_sha256") != import_pin["sha256"]
            or start.get("model_launches_started") is not False
            or not isinstance(start.get("entry_source"), Mapping)):
        raise HandoffError("coordinator start/launch receipts do not bind the actual queue/historical inputs")
    # The coordinator executes the reviewed queue entry with the pinned remote
    # package path. Its raw source pin is compared to the source module loaded
    # by this collector, while argv and the separate terminal result bind the
    # exact invocation and final outcome.
    entry_source = start.get("entry_source")
    entry_argv = start.get("entry_argv")
    try:
        expected_entry_path = Path(queue_runner.__file__).resolve(strict=True)
        expected_entry_path.relative_to(repo_root.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise HandoffError("queue runner module is not loaded from the supplied project root") from exc
    expected_entry = str(expected_entry_path)
    expected_argv = [
        "/usr/bin/python3", "-B", expected_entry, "--mode", "run",
        "--repo-root", str(repo_root.resolve(strict=True)),
        "--historical-index", history_pin["path"],
        "--expected-index-sha256", history_pin["sha256"],
        "--state-dir", str(state_dir.resolve(strict=True)),
        "--receipt-dir", str(receipt_dir.resolve(strict=True)),
    ]
    entry_raw, entry_actual_pin = _read_regular(expected_entry_path, "pinned queue runner source")
    expected_env = {
        "PATH": "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        "HOME": "/home/wjj", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1",
    }
    # Python sets sys.argv[0] to the script path; the interpreter executable
    # and ``-B`` flag appear only in the detached launch receipt's argv.
    expected_supervisor_argv, expected_start_argv = _expected_coordinator_argv(
        source_path, config_path
    )
    expected_config = {
        "schema": "fit40_queue_detached_coordinator_config_v1",
        "runtime_dir": str(coordinator_dir.resolve(strict=True)),
        "cwd": str(repo_root.resolve(strict=True)),
        "entry_path": expected_entry,
        "entry_bytes": entry_actual_pin["bytes"],
        "entry_sha256": entry_actual_pin["sha256"],
        "entry_argv": expected_argv,
        "environment": expected_env,
        "index_path": history_pin["path"],
        "index_bytes": history_pin["bytes"],
        "index_sha256": history_pin["sha256"],
        "index_self_sha256": history_self_sha256,
        "import_path": import_pin["path"],
        "import_bytes": import_pin["bytes"],
        "import_sha256": import_pin["sha256"],
        "import_self_sha256": import_self_sha256,
        "state_dir": str(state_dir.resolve(strict=True)),
        "receipt_dir": str(receipt_dir.resolve(strict=True)),
    }
    _require_exact_mapping(config, expected_config, "coordinator configuration")
    if (not isinstance(entry_source, Mapping)
            or entry_source.get("path") != expected_entry
            or entry_source.get("bytes") != entry_actual_pin["bytes"]
            or entry_source.get("sha256") != entry_actual_pin["sha256"]
            or entry_actual_pin["sha256"] != EXPECTED_SOURCE_MODULE_PINS[Path(queue_runner.__file__).name][1]
            or not isinstance(entry_argv, list) or entry_argv != expected_argv
            or terminal.get("entry_argv") != expected_argv
            or start.get("coordinator_argv") != expected_start_argv
            or launch.get("argv") != expected_supervisor_argv
            or start.get("environment") != expected_env
            or launch.get("environment") != expected_env
            or launch.get("environment_sha256") != digest(canonical(expected_env))
            or launch.get("cwd") != str(repo_root.resolve(strict=True))
            or start.get("entry_argv") != expected_argv
            or start.get("entry_source") != {
                "path": expected_entry, "bytes": entry_actual_pin["bytes"],
                "sha256": entry_actual_pin["sha256"],
            }
            or launch.get("log_paths") != {
                "stdout": str((coordinator_dir / "coordinator_stdout.log").resolve(strict=True)),
                "stderr": str((coordinator_dir / "coordinator_stderr.log").resolve(strict=True)),
            }):
        raise HandoffError("coordinator did not execute the pinned source/config/argv/environment")
    time_provenance = _verify_coordinator_time_provenance(start, launch, terminal)
    result = terminal.get("entry_result")
    owner = result.get("owner") if isinstance(result, Mapping) else None
    if (terminal.get("coordinator_pid") != launch.get("pid")
            or terminal.get("coordinator_start_ticks") != launch.get("start_ticks")
            or type(terminal.get("entry_child_pid")) is not int
            or terminal.get("entry_child_pid") <= 0
            or type(terminal.get("entry_child_start_ticks")) is not int
            or terminal.get("entry_child_start_ticks") <= 0
            or type(terminal.get("entry_child_returncode")) is not int
            or terminal.get("entry_child_returncode") != 0
            or terminal.get("exception_sha256") is not None
            or type(terminal.get("exit_code")) is not int or terminal.get("exit_code") != 0
            or terminal.get("queue_complete") is not True
            or terminal.get("model_launches_started") is not True
            or not isinstance(result, Mapping)
            or result.get("schema") != "fit40_fixed_queue_run_attempt05_v1"
            or result.get("status") != "PASS_QUEUE_COMPLETE"
            or result.get("launches_models") is not True
            or not isinstance(owner, Mapping)
            or owner.get("pid") != terminal.get("entry_child_pid")
            or owner.get("start_ticks") != terminal.get("entry_child_start_ticks")):
        raise HandoffError("coordinator terminal receipt is not a clean exit-0 full-queue completion")
    if result.get("queue_state") != final_queue_state:
        raise HandoffError("coordinator result queue state differs from final read-only event replay")
    stdout_pin = terminal.get("entry_stdout")
    stderr_pin = terminal.get("entry_stderr")
    for name, pin, expected_name in (("entry stdout", stdout_pin, "entry05_stdout.json"),
                                     ("entry stderr", stderr_pin, "entry05_stderr.txt")):
        expected_path = str((coordinator_dir / expected_name).resolve(strict=True))
        if (not isinstance(pin, Mapping) or pin.get("path") != expected_path):
            raise HandoffError(f"coordinator terminal {name} pin is malformed")
        raw, observed = _read_regular(Path(pin["path"]), name)
        if observed["bytes"] != pin.get("bytes") or observed["sha256"] != pin.get("sha256"):
            raise HandoffError(f"coordinator terminal {name} file differs from its pin")
        if name == "entry stdout":
            try:
                stdout_result = json.loads(raw.decode("utf-8"))
            except Exception as exc:
                raise HandoffError("queue entry stdout is not a JSON result") from exc
            if stdout_result != result:
                raise HandoffError("queue entry stdout differs from coordinator terminal embedded result")
    return {
        "launch_receipt": launch_pin, "start_receipt": start_pin,
        "terminal_receipt": terminal_pin,
        "coordinator_source": source_pin, "coordinator_config": config_pin,
        "coordinator_pid": launch["pid"], "coordinator_start_ticks": launch["start_ticks"],
        "entry_child_pid": terminal["entry_child_pid"],
        "entry_child_start_ticks": terminal["entry_child_start_ticks"],
        "entry_child_returncode": terminal["entry_child_returncode"],
        "exit_code": terminal["exit_code"],
        "started_unix_seconds": terminal.get("started_unix_seconds"),
        "ended_unix_seconds": terminal.get("ended_unix_seconds"),
        "coordinator_time_provenance": time_provenance,
        "entry_argv": terminal["entry_argv"],
        "entry_stdout": stdout_pin,
        "entry_stderr": stderr_pin,
        "exception_sha256": terminal.get("exception_sha256"),
    }


def _verify_historical_block(
    model_id: str, block_index: int, historical_block: Mapping[str, Any],
    selected_record: Mapping[str, Any], bundle: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    expected_rows = list(bundle.rows[block_index * 6:(block_index + 1) * 6])
    expected_ids = [row["row_id"] for row in expected_rows]
    if not _historical_rc_is_unknown(selected_record):
        # Keep unavailable history status unknown. Never derive an rc from PID
        # absence or from a neighboring attempt record.
        raise HandoffError(f"historical {model_id}/{block_index} exit-code scope was rewritten")
    if (selected_record.get("block_start_index") != block_index * 6
            or selected_record.get("result_row_ids") != expected_ids
            or selected_record.get("result_row_statuses") != ["PASS"] * 6):
        raise HandoffError(f"historical {model_id}/{block_index} selected rows differ from the frozen plan")
    output_leaf = selected_record.get("leaf")
    artifacts = selected_record.get("artifact_pins")
    if not isinstance(output_leaf, str) or not Path(output_leaf).is_absolute() or not isinstance(artifacts, Mapping):
        raise HandoffError(f"historical {model_id}/{block_index} lacks actual output/artifact paths")
    artifact_files: dict[str, tuple[dict[str, Any], dict[str, Any], bytes]] = {}
    for label, filename in (("run_status", "control/run_status.json"),
                            ("result_manifest", "result_manifest.json"),
                            ("results_jsonl", "results.jsonl")):
        ref = artifacts.get(label)
        if not isinstance(ref, Mapping):
            raise HandoffError(f"historical {model_id}/{block_index} lacks {label} pin")
        path = Path(str(ref.get("path", "")))
        if path != Path(output_leaf) / filename:
            raise HandoffError(f"historical {model_id}/{block_index} {label} path is outside selected output leaf")
        expected_pin = {"bytes": ref.get("bytes"), "sha256": ref.get("sha256")}
        if (type(expected_pin["bytes"]) is not int or expected_pin["bytes"] <= 0
                or not _is_sha(expected_pin["sha256"])):
            raise HandoffError(f"historical {model_id}/{block_index} has malformed {label} file pin")
        if label == "results_jsonl":
            raw, pin = _read_regular(path, label)
            if pin["bytes"] != expected_pin["bytes"] or pin["sha256"] != expected_pin["sha256"]:
                raise HandoffError(f"historical {model_id}/{block_index} results bytes differ from historic pin")
            try:
                records = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
            except Exception as exc:
                raise HandoffError(f"historical {model_id}/{block_index} results JSONL is malformed") from exc
            if len(records) != 6 or [item.get("row_id") for item in records] != expected_ids:
                raise HandoffError(f"historical {model_id}/{block_index} ledger row order is not exact")
        else:
            obj, pin, raw = _read_json(path, label)
            if pin["bytes"] != expected_pin["bytes"] or pin["sha256"] != expected_pin["sha256"]:
                raise HandoffError(f"historical {model_id}/{block_index} {label} bytes differ from historic pin")
            if label == "result_manifest":
                if obj.get("manifest_sha256") != queue._self_hash(obj, "manifest_sha256"):
                    raise HandoffError(f"historical {model_id}/{block_index} manifest self hash mismatch")
                if (obj.get("status") != "BLOCK_COMPLETE" or obj.get("block_index") != block_index
                        or obj.get("block_start_index") != block_index * 6
                        or obj.get("block_rows") != 6 or obj.get("result_rows") != 6):
                    raise HandoffError(f"historical {model_id}/{block_index} manifest terminal scope mismatch")
            else:
                if (obj.get("status") != "BLOCK_COMPLETE" or obj.get("block_index") != block_index
                        or obj.get("block_start_index") != block_index * 6
                        or obj.get("block_rows") != 6 or obj.get("completed_rows") != 6):
                    raise HandoffError(f"historical {model_id}/{block_index} run status terminal scope mismatch")
        artifact_files[label] = (obj if label != "results_jsonl" else {}, pin, raw)
    records = [json.loads(line) for line in artifact_files["results_jsonl"][2].decode("utf-8").splitlines()]
    descriptor = bundle.plan["models"][0]
    protocol = ingest._score_protocol(bundle.plan)
    for record, planned in zip(records, expected_rows):
        if record.get("status") != "PASS":
            raise HandoffError(f"historical {model_id}/{block_index} contains a non-PASS row")
        ingest._validate_record_against_plan(record, planned, descriptor, protocol)
    status_obj, status_pin, _ = artifact_files["run_status"]
    manifest_obj, manifest_pin, _ = artifact_files["result_manifest"]
    ledger_pin = artifact_files["results_jsonl"][1]
    if (status_obj.get("run_id") != bundle.plan.get("run_id")
            or status_obj.get("plan_raw_sha256") != bundle.spec.plan.sha256
            or status_obj.get("planned_rows_sha256") != bundle.spec.planned_rows.sha256
            or manifest_obj.get("run_id") != bundle.plan.get("run_id")
            or manifest_obj.get("plan_raw_sha256") != bundle.spec.plan.sha256
            or manifest_obj.get("results_jsonl_sha256") != ledger_pin["sha256"]):
        raise HandoffError(f"historical {model_id}/{block_index} status/manifest lineage mismatch")
    proof = {
        "proof_origin": "HISTORICAL_PREFIX_TERMINAL",
        "model_id": model_id, "block_index": block_index,
        "block_start_index": block_index * 6, "row_ids": expected_ids,
        "historical_exit_code_available": False, "historical_exit_code": None,
        "selected_historical_pass_leaf": output_leaf,
        "run_status": status_pin, "result_manifest": manifest_pin,
        "results_jsonl": ledger_pin,
        "historical_end_unix": selected_record.get("historical_end_unix"),
    }
    return records, proof


def _verify_live_pass_event_output_pins(
    controller: Any, queue_runner: Any, claim: Mapping[str, Any],
    receipt: Mapping[str, Any], output_path: Path,
    process_receipt_pin: Mapping[str, Any], event_output_pins: Any,
) -> dict[str, Any]:
    """Reconstruct the exact proof shape emitted by FixedFitQueueController.finish_claim."""
    # finish_claim starts with {"process_receipt": process_pin}, then merges
    # _verify_pass_outputs and the queue runner's strict row-identity proof.
    proof = {"process_receipt": dict(process_receipt_pin)}
    proof.update(controller._verify_pass_outputs(claim, receipt, output_path))
    proof["strict_terminal_bindings"] = queue_runner._verify_strict_terminal_bindings(
        controller, claim, receipt, output_path
    )
    if proof != event_output_pins:
        raise HandoffError(
            "live queue PASS event does not bind the actual strict status/manifest/ledger/receipt bytes"
        )
    return proof


def _verify_new_queue_blocks(
    controller: Any, state_dir: Path, receipt_dir: Path,
    historical_counts: Mapping[str, int],
) -> tuple[dict[tuple[str, int], list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    events_dir = state_dir / "events"
    if not events_dir.is_dir() or events_dir.is_symlink():
        raise HandoffError("queue event directory is missing or symlinked")
    event_paths = sorted(events_dir.glob("*.json"))
    entries = list(events_dir.iterdir())
    if (len(event_paths) != 2 * EXPECTED_QUEUE_BLOCKS
            or len(entries) != len(event_paths)
            or any(path.suffix != ".json" or path.is_symlink() or not path.is_file() for path in entries)):
        raise HandoffError(f"expected {2 * EXPECTED_QUEUE_BLOCKS} claim/pass event files; found {len(event_paths)}")
    event_records: list[dict[str, Any]] = []
    events_by_seq: dict[int, tuple[dict[str, Any], dict[str, Any]]] = {}
    for path in event_paths:
        event, pin, _ = _read_json(path, "queue event")
        if event.get("schema") != queue.EVENT_SCHEMA or event.get("self_sha256") != queue._self_hash(event, "self_sha256"):
            raise HandoffError(f"queue event schema/self hash mismatch: {path}")
        seq = event.get("seq")
        if type(seq) is not int or seq <= 0 or seq in events_by_seq:
            raise HandoffError("queue event sequence is non-integer or duplicated")
        events_by_seq[seq] = (event, pin)
    if sorted(events_by_seq) != list(range(1, len(event_paths) + 1)):
        raise HandoffError("queue event sequence is not contiguous from one")
    pass_rows: dict[tuple[str, int], list[dict[str, Any]]] = {}
    claim_by_nonce: dict[str, dict[str, Any]] = {}
    pass_events: list[dict[str, Any]] = []
    for seq in sorted(events_by_seq):
        event, pin = events_by_seq[seq]
        event_records.append({"path": pin["path"], "bytes": pin["bytes"], "sha256": pin["sha256"],
                              "seq": seq, "event": event.get("event"), "self_sha256": event["self_sha256"]})
        if event.get("event") == "CLAIM":
            nonce = event.get("claim_nonce")
            if not isinstance(nonce, str) or nonce in claim_by_nonce:
                raise HandoffError("queue claim nonce is missing or duplicated")
            claim_by_nonce[nonce] = event
        elif event.get("event") == "BLOCK_PASS":
            nonce = event.get("claim_nonce")
            claim = claim_by_nonce.get(nonce)
            if claim is None:
                raise HandoffError("queue pass event has no prior claim")
            model_id, block_index = event.get("model_id"), event.get("block_index")
            key = (model_id, block_index)
            if key in pass_rows:
                raise HandoffError(f"queue repeats terminal block {key}")
            if (claim.get("model_id") != model_id or claim.get("block_index") != block_index
                    or claim.get("block_start_index") != event.get("block_start_index")
                    or claim.get("output_leaf") != event.get("output_leaf")
                    or event.get("failure_reason") is not None):
                raise HandoffError("queue claim/pass pair changes the claimed block identity")
            if (event.get("historical_prefix_index_sha256") != HISTORICAL_INDEX_SHA256
                    or claim.get("historical_prefix_index_sha256") != HISTORICAL_INDEX_SHA256):
                raise HandoffError("queue event is not bound to the pinned historical baseline")
            if (event.get("process_terminal") != {
                    "child_pid": event.get("process_terminal", {}).get("child_pid"),
                    "child_start_ticks": event.get("process_terminal", {}).get("child_start_ticks"),
                    "exit_code": 0,
                }
                    or type(event["process_terminal"].get("child_pid")) is not int
                    or type(event["process_terminal"].get("child_start_ticks")) is not int
                    or type(event["process_terminal"].get("exit_code")) is not int):
                raise HandoffError("live queue block lacks explicit numeric child PID/start ticks/rc=0")
            process_pin = event.get("verified_output_pins", {}).get("process_receipt")
            if not isinstance(process_pin, Mapping):
                raise HandoffError("live queue block lacks process receipt pin")
            process_path = _require_inside_output_leaf(
                Path(str(process_pin.get("path", ""))), receipt_dir, "live child process receipt"
            )
            receipt, observed_pin = controller._validate_process_receipt(claim, process_path)
            if (observed_pin["bytes"] != process_pin.get("bytes")
                    or observed_pin["sha256"] != process_pin.get("sha256")
                    or type(receipt.get("exit_code")) is not int
                    or receipt.get("exit_code") != 0
                    or receipt.get("terminal_status") != "EXITED"
                    or receipt.get("child_pid") != event["process_terminal"]["child_pid"]
                    or receipt.get("child_start_ticks") != event["process_terminal"]["child_start_ticks"]):
                raise HandoffError("live process terminal receipt differs from the PASS event's actual rc/PID")
            model_id = str(model_id)
            bundle = controller.bundles[model_id]
            output_path = Path(str(event.get("output_leaf", "")))
            proof = _verify_live_pass_event_output_pins(
                controller, queue_runner, claim, receipt, output_path,
                observed_pin, event.get("verified_output_pins"),
            )
            records, _ = _load_block_records(bundle, block_index, output_path / "results.jsonl")
            pass_rows[key] = records
            pass_events.append({"event": event, "event_pin": pin, "claim": claim,
                                "process_receipt": receipt, "process_pin": observed_pin,
                                "output_proof": proof})
        else:
            raise HandoffError(f"queue contains unsupported terminal/event state: {event.get('event')!r}")
    if len(claim_by_nonce) != EXPECTED_QUEUE_BLOCKS or len(pass_events) != EXPECTED_QUEUE_BLOCKS:
        raise HandoffError("live queue does not contain exactly 86 paired claim/PASS blocks")
    if set(claim_by_nonce) != {x["event"].get("claim_nonce") for x in pass_events}:
        raise HandoffError("a queue claim lacks exactly one terminal PASS")
    # Reuse the frozen read-only event-state machine, not an inferred count.
    state = controller._replay()
    expected_next = {model: EXPECTED_ROWS_PER_MODEL for model in queue.MODEL_ORDER}
    if (state.active_claim is not None or state.progress.keys() != expected_next.keys()
            or {key: value.next_start for key, value in state.progress.items()} != expected_next
            or any(value.paused for value in state.progress.values())
            or controller.lock_path.exists() or controller.lock_path.is_symlink()):
        raise HandoffError("replayed queue has active/paused work or is short of 240 rows for a model")
    status = queue_runner.classify_queue_run(controller.snapshot(),
                                             [item["event"] for item in pass_events])
    if status != "PASS_QUEUE_COMPLETE":
        raise HandoffError(f"frozen queue terminal classifier returned {status}")
    expected_new: set[tuple[str, int]] = set()
    for model_id in queue.MODEL_ORDER:
        for block_index in range(historical_counts[model_id], queue.BLOCKS_PER_MODEL):
            expected_new.add((model_id, block_index))
    if set(pass_rows) != expected_new:
        raise HandoffError("new queue terminal blocks are not the exact complement of the historic prefix")
    return pass_rows, event_records, pass_events


def _load_block_records(bundle: Any, block_index: int, ledger_path: Path) -> tuple[list[dict[str, Any]], bytes]:
    raw, _ = _read_regular(ledger_path, "block results JSONL")
    try:
        lines = raw.decode("utf-8").splitlines()
        records = [json.loads(line) for line in lines]
    except Exception as exc:
        raise HandoffError(f"block results JSONL is malformed: {ledger_path}") from exc
    expected = list(bundle.rows[block_index * 6:(block_index + 1) * 6])
    if (len(records) != 6 or any(not isinstance(row, dict) for row in records)
            or [row.get("row_id") for row in records] != [row.get("row_id") for row in expected]):
        raise HandoffError(f"block ledger does not contain the exact six ordered rows: {ledger_path}")
    descriptor = bundle.plan["models"][0]
    protocol = ingest._score_protocol(bundle.plan)
    for result, planned in zip(records, expected):
        if result.get("status") != "PASS":
            raise HandoffError(f"complete initial family requires PASS row: {result.get('row_id')}")
        ingest._validate_record_against_plan(result, planned, descriptor, protocol)
    return records, raw


def _validate_npz_and_frame_map(
    record: Mapping[str, Any], planned: Mapping[str, Any], bundle: Any,
    output_leaf: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    row_id = record.get("row_id")
    score = record.get("score")
    raw_ref = score.get("raw_matrix_npz") if isinstance(score, Mapping) else None
    frame_ref = score.get("primary_syncnet_input") if isinstance(score, Mapping) else None
    if not isinstance(row_id, str) or not isinstance(raw_ref, Mapping) or not isinstance(frame_ref, Mapping):
        raise HandoffError("PASS row lacks official raw matrix or source frame map reference")
    npz_path_text = raw_ref.get("path")
    npz_sha = raw_ref.get("file_sha256")
    map_path_text = frame_ref.get("frame_map_path")
    map_sha = frame_ref.get("frame_map_sha256")
    if (not isinstance(npz_path_text, str) or not Path(npz_path_text).is_absolute()
            or not _is_sha(npz_sha) or not isinstance(map_path_text, str)
            or not Path(map_path_text).is_absolute() or not _is_sha(map_sha)):
        raise HandoffError(f"row {row_id} has invalid absolute NPZ/frame-map references")
    npz_path = _require_inside_output_leaf(Path(npz_path_text), output_leaf, "raw matrix NPZ")
    map_path = _require_inside_output_leaf(Path(map_path_text), output_leaf, "official source frame map")
    npz_bytes, npz_pin = _read_regular(npz_path, "raw matrix NPZ")
    if npz_pin["sha256"] != npz_sha:
        raise HandoffError(f"row {row_id} raw NPZ full-file SHA differs from its score row")
    npz_file = ingest._PinnedFile(f"npz_{row_id}", "raw_matrix_npz", npz_path_text,
                                  str(npz_path.resolve(strict=True)), npz_pin["bytes"], npz_sha)
    arrays = ingest._load_arrays(npz_file, raw_ref)
    declared_arrays = raw_ref.get("arrays")
    if not isinstance(declared_arrays, Mapping) or set(declared_arrays) != EXPECTED_NPZ_ARRAY_NAMES:
        raise HandoffError(f"row {row_id} does not declare exactly the frozen seven raw arrays")
    array_refs: dict[str, dict[str, Any]] = {}
    for name, array in arrays.items():
        ref = declared_arrays.get(name)
        actual = {"dtype": array.dtype.str, "shape": list(array.shape),
                  "array_sha256": digest(array.tobytes(order="C"))}
        if not isinstance(ref, Mapping) or dict(ref) != actual:
            raise HandoffError(f"row {row_id} array metadata differs from exact decompressed bytes: {name}")
        array_refs[name] = actual
    map_bytes, map_pin = _read_regular(map_path, "official source frame map")
    if map_pin["sha256"] != map_sha:
        raise HandoffError(f"row {row_id} source frame map full-file SHA differs from score row")
    map_file = ingest._PinnedFile(f"frame_map_{row_id}", "official_source_frame_map", map_path_text,
                                  str(map_path.resolve(strict=True)), map_pin["bytes"], map_sha, row_id)
    try:
        frame_map = json.loads(map_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HandoffError(f"row {row_id} official source frame map is malformed JSON") from exc
    frames = frame_map.get("frames") if isinstance(frame_map, Mapping) else None
    if not isinstance(frames, list) or not frames:
        raise HandoffError(f"row {row_id} source frame map contains no actual frame records")
    source_frames = [item.get("source_clock_frame_index") if isinstance(item, Mapping) else None
                     for item in frames]
    if (any(type(value) is not int for value in source_frames)
            or any(right != left + 1 for left, right in zip(source_frames, source_frames[1:]))):
        raise HandoffError(f"row {row_id} source frame map is not the producer's contiguous source-clock window")
    clock = final_ingest.validate_official_source_frame_map(record, planned, arrays, map_file)
    return ({"path": npz_path_text, "bytes": npz_pin["bytes"], "sha256": npz_sha,
             "array_refs": array_refs},
            {"path": map_path_text, "bytes": map_pin["bytes"], "sha256": map_sha,
             "clock_validation": clock})


def build_terminal_handoff(
    *, repo_root: Path, family_path: Path, historical_index_path: Path,
    state_dir: Path, receipt_dir: Path, coordinator_dir: Path, output_dir: Path,
    expected_family_sha256: str = FAMILY_PLAN_SHA256,
    expected_family_self_sha256: str = FAMILY_PLAN_SELF_SHA256,
    expected_history_sha256: str = HISTORICAL_INDEX_SHA256,
    expected_history_self_sha256: str = HISTORICAL_INDEX_SELF_SHA256,
) -> dict[str, Any]:
    """Read all 720 exact source/terminal artifacts and create a sealed handoff.

    The routine raises before creating ``output_dir`` if any terminal/readback
    predicate fails. The two output files are create-only and are never opened
    for later mutation.
    """
    repo_root = _require_directory(repo_root, "source repository")
    state_dir = _require_directory(state_dir, "queue state")
    receipt_dir = _require_directory(receipt_dir, "queue receipt")
    coordinator_dir = _require_directory(coordinator_dir, "queue coordinator")
    if not family_path.is_absolute() or not historical_index_path.is_absolute() or not output_dir.is_absolute():
        raise HandoffError("family/index/output paths must be absolute")
    output_parent = _require_directory(output_dir.parent, "collector output parent")
    if output_dir.name in ("", ".", ".."):
        raise HandoffError("collector output leaf must have a fresh explicit name")
    output_dir = output_parent / output_dir.name
    if output_dir.exists() or output_dir.is_symlink():
        raise HandoffError(f"refusing to overwrite collector output namespace: {output_dir}")
    family, family_pin, bundles = _load_exact_family(
        repo_root, family_path, expected_family_sha256, expected_family_self_sha256
    )
    historical, history_pin, history_raw = _pinned_json(
        str(historical_index_path), HISTORICAL_INDEX_BYTES,
        expected_history_sha256, expected_history_self_sha256, "historical prefix index",
    )
    if historical.get("schema") != queue.HISTORICAL_INDEX_SCHEMA:
        raise HandoffError("historical prefix index schema mismatch")
    historical_seed = queue._load_historical_prefix_seed(historical_index_path, expected_history_sha256,
                                                         {x.spec.model_id: x for x in bundles})
    historical_counts = {model: HISTORICAL_BLOCKS[model] for model in queue.MODEL_ORDER}
    if sum(historical_counts.values()) != EXPECTED_HISTORICAL_BLOCKS:
        raise HandoffError("frozen history prefix count does not equal 34")
    controller = queue.FixedFitQueueController(
        bundles, state_dir, historical_prefix_index=historical_index_path,
        historical_prefix_index_sha256=expected_history_sha256,
    )
    import_receipt, import_pin, _ = _read_json(state_dir / "historical_prefix_import.json",
                                                "historical prefix import receipt")
    import_self = import_receipt.get("self_sha256")
    if (import_receipt.get("schema") != queue.HISTORICAL_IMPORT_SCHEMA
            or import_self != queue._self_hash(import_receipt, "self_sha256")
            or import_receipt.get("historical_index") != {
                "path": str(historical_index_path.resolve(strict=True)),
                "bytes": history_pin["bytes"], "sha256": history_pin["sha256"],
            }
            or import_receipt.get("next_starts") != dict(historical_seed.next_starts)):
        raise HandoffError("history import receipt does not bind exact prefix index and offsets")
    actual_rows_by_block, event_files, live_passes = _verify_new_queue_blocks(
        controller, state_dir, receipt_dir, historical_counts
    )
    final_state = controller.snapshot()
    coordinator_qc = _verify_coordinator(repo_root, coordinator_dir, state_dir, receipt_dir,
                                         history_pin, HISTORICAL_INDEX_SELF_SHA256,
                                         import_pin, import_self, final_state)

    # Gather exactly 120 terminal block records in model/plan order. Historic
    # receipts explicitly keep child rc unavailable; new block receipts must
    # link the queue's real process receipt with exit code zero.
    rows_by_model: dict[str, dict[str, Mapping[str, Any]]] = {}
    block_evidence: list[dict[str, Any]] = []
    records_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_id, bundle in zip(queue.MODEL_ORDER, bundles):
        hmodel = next((item for item in historical.get("models", []) if item.get("model_id") == model_id), None)
        if not isinstance(hmodel, Mapping):
            raise HandoffError(f"historical index omits model {model_id}")
        hblocks = hmodel.get("blocks")
        if not isinstance(hblocks, list) or len(hblocks) != historical_counts[model_id]:
            raise HandoffError(f"historical model {model_id} block prefix changed")
        model_records: list[dict[str, Any]] = []
        rows_lookup: dict[str, Mapping[str, Any]] = {}
        block_entries: list[dict[str, Any]] = []
        for block_index in range(queue.BLOCKS_PER_MODEL):
            if block_index < historical_counts[model_id]:
                hblock = hblocks[block_index]
                selected = next((x for x in hblock.get("attempts", [])
                                 if x.get("status") == "PASS_IMPORTABLE"
                                 and x.get("leaf") == hblock.get("selected_historical_pass_leaf")), None)
                if not isinstance(selected, Mapping):
                    raise HandoffError(f"historical block selection missing for {model_id}/{block_index}")
                block_records, proof = _verify_historical_block(model_id, block_index, hblock, selected, bundle)
                proof["attempt"] = selected.get("attempt")
                proof["historical_index_raw_sha256"] = history_pin["sha256"]
            else:
                block_records = actual_rows_by_block[(model_id, block_index)]
                live = next(x for x in live_passes
                            if x["event"].get("model_id") == model_id
                            and x["event"].get("block_index") == block_index)
                event = live["event"]
                proof = {
                    "proof_origin": "ACTUAL_QUEUE_CHILD_RC",
                    "model_id": model_id, "block_index": block_index,
                    "block_start_index": block_index * 6,
                    "row_ids": list(bundle.row_ids[block_index * 6:(block_index + 1) * 6]),
                    "claim_nonce": event["claim_nonce"],
                    "queue_event_seq": event["seq"],
                    "queue_event_self_sha256": event["self_sha256"],
                    "child_pid": live["process_receipt"]["child_pid"],
                    "child_start_ticks": live["process_receipt"]["child_start_ticks"],
                    "child_exit_code": live["process_receipt"]["exit_code"],
                    "process_receipt": live["process_pin"],
                    "run_status": event["verified_output_pins"]["run_status"],
                    "result_manifest": event["verified_output_pins"]["result_manifest"],
                    "results_jsonl": event["verified_output_pins"]["results_jsonl"],
                    "output_leaf": event["output_leaf"],
                    "strict_terminal_bindings": event["verified_output_pins"]["strict_terminal_bindings"],
                }
            model_records.extend(block_records)
            for row in block_records:
                row_id = row.get("row_id")
                if not isinstance(row_id, str) or row_id in rows_lookup:
                    raise HandoffError(f"duplicate/missing result row ID in {model_id}")
                rows_lookup[row_id] = row
            block_entries.append(proof)
        if (len(model_records) != EXPECTED_ROWS_PER_MODEL
                or tuple(row.get("row_id") for row in model_records) != bundle.row_ids
                or set(rows_lookup) != set(bundle.row_ids)):
            raise HandoffError(f"{model_id} actual result ledger does not exactly cover its frozen 240 rows")
        rows_by_model[model_id] = rows_lookup
        records_by_model[model_id] = model_records
        block_evidence.extend(block_entries)

    # Exact complete per-model plan outputs and source inputs form the normal
    # Stage-A index. No selector code is called by this collector.
    files: list[dict[str, Any]] = []
    file_ids: set[str] = set()
    input_pin_files: dict[str, str] = {}
    model_entries: list[dict[str, Any]] = []
    row_to_npz: dict[str, str] = {}
    row_to_frame_map: dict[str, str] = {}
    artifact_rows: list[dict[str, Any]] = []
    source_plan_entries = family["execution"]["source_model_plans"]
    input_pins: Mapping[str, Any] = family.get("scientific_input_pins", {})
    source_root = _require_directory(Path(source_plan_entries[0]["plan"]["path"]).parent,
                                    "shared source-plan project root")
    for pin_name, pin_data in sorted(input_pins.items()):
        if (not isinstance(pin_data, Mapping) or not isinstance(pin_data.get("path"), str)
                or type(pin_data.get("bytes")) is not int or pin_data["bytes"] <= 0
                or not _is_sha(pin_data.get("sha256"))):
            raise HandoffError(f"scientific input pin is malformed: {pin_name}")
        relative = Path(pin_data["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise HandoffError(f"scientific input pin path is not safe relative to staged source project: {pin_name}")
        candidate_path = source_root / relative
        candidate_path = _require_inside_output_leaf(candidate_path, source_root,
                                                     f"scientific input {pin_name}")
        declared = str(candidate_path)
        raw, actual = _read_regular(candidate_path, f"scientific input {pin_name}")
        if actual["bytes"] != pin_data["bytes"] or actual["sha256"] != pin_data["sha256"]:
            raise HandoffError(f"scientific input bytes differ from family plan pin: {pin_name}")
        file_id = f"input_pin:{pin_name}"
        _add_file(files, file_ids, file_id, "input_pin", declared, pin_name=pin_name)
        input_pin_files[pin_name] = file_id
    if not {"snapshot_index", "snapshot_step_030", "snapshot_step_060",
            "snapshot_step_090", "snapshot_step_120"} <= set(input_pin_files):
        raise HandoffError("materialized source inputs omit a frozen snapshot index/record")

    for source, bundle in zip(source_plan_entries, bundles):
        model_id = bundle.spec.model_id
        plan_ref, rows_ref = source["plan"], source["planned_rows"]
        plan_path, rows_path = Path(plan_ref["path"]), Path(rows_ref["path"])
        if (plan_path.is_symlink() or rows_path.is_symlink()
                or not plan_path.is_file() or not rows_path.is_file()):
            raise HandoffError(f"{model_id} source plan/rows path is not a regular file")
        plan_id, rows_id = f"plan:{model_id}", f"planned_rows:{model_id}"
        _add_file(files, file_ids, plan_id, "plan", str(plan_path))
        _add_file(files, file_ids, rows_id, "planned_rows", str(rows_path))
        block_index_entries: list[dict[str, Any]] = []
        for block_index in range(queue.BLOCKS_PER_MODEL):
            proof = next(item for item in block_evidence
                         if item["model_id"] == model_id and item["block_index"] == block_index)
            for role, key, file_role in (("run_status", "run_status", "run_status"),
                                         ("result_manifest", "result_manifest", "result_manifest"),
                                         ("results_jsonl", "results_jsonl", "results_jsonl")):
                ref = proof.get(key)
                if not isinstance(ref, Mapping):
                    raise HandoffError(f"terminal proof omits {role}: {model_id}/{block_index}")
                file_id = f"{role}:{model_id}:{block_index:02d}"
                _add_file(files, file_ids, file_id, file_role, str(ref["path"]))
                proof[f"{role}_file_id"] = file_id
            for record, planned in zip(
                    [r for r in records_by_model[model_id]
                     if r.get("row_id") in set(proof["row_ids"])],
                    bundle.rows[block_index * 6:(block_index + 1) * 6]):
                row_id = str(record["row_id"])
                if row_id != planned["row_id"]:
                    raise HandoffError(f"ledger/plan order changed inside {model_id}/{block_index}")
                output_leaf = proof.get("output_leaf", proof.get("selected_historical_pass_leaf"))
                if not isinstance(output_leaf, str) or not Path(output_leaf).is_absolute():
                    raise HandoffError(f"terminal proof omits the exact output leaf for {model_id}/{block_index}")
                npz_ref, map_ref = _validate_npz_and_frame_map(
                    record, planned, bundle, Path(output_leaf)
                )
                npz_id = f"raw_matrix_npz:{row_id}"
                map_id = f"official_source_frame_map:{row_id}"
                _add_file(files, file_ids, npz_id, "raw_matrix_npz", npz_ref["path"])
                _add_file(files, file_ids, map_id, "official_source_frame_map",
                          map_ref["path"], pin_name=row_id)
                row_to_npz[row_id] = npz_id
                row_to_frame_map[row_id] = map_id
                artifact_rows.append({"row_id": row_id, "model_id": model_id,
                                      "raw_matrix_npz_file_id": npz_id,
                                      "official_source_frame_map_file_id": map_id,
                                      "raw_matrix_npz": {k: npz_ref[k] for k in ("bytes", "sha256")},
                                      "raw_matrix_npz_arrays": npz_ref["array_refs"],
                                      "official_source_frame_map": {k: map_ref[k] for k in ("bytes", "sha256")},
                                      "clock_validation": map_ref["clock_validation"]})
            block_index_entries.append({
                "block_index": block_index,
                "run_status_file_id": proof["run_status_file_id"],
                "result_manifest_file_id": proof["result_manifest_file_id"],
                "results_file_id": proof["results_jsonl_file_id"],
            })
        model_entries.append({"model_id": model_id, "plan_file_id": plan_id,
                              "planned_rows_file_id": rows_id, "blocks": block_index_entries})

    all_sample_ids = [str(bundle.sample_ids[i]) for i in range(40)]
    ingest_index: dict[str, Any] = {
        "schema": ingest.SCHEMA,
        "status": "READY_FOR_LOCAL_INGEST_NOT_SELECTION",
        "scientific_input_pins_sha256": SCIENTIFIC_INPUT_SHA256,
        "expected_row_count": EXPECTED_ROWS,
        "model_ids": list(queue.MODEL_ORDER),
        "sample_ids": all_sample_ids,
        "input_pin_files": input_pin_files,
        "official_source_frame_map_file_ids": row_to_frame_map,
        "raw_matrix_npz_file_ids": row_to_npz,
        "files": files,
        "models": model_entries,
    }
    ingest_index["index_sha256"] = self_hash(ingest_index, "index_sha256")

    # Validate the generated preselector packet structurally and all raw clock
    # maps locally. This is transport/readiness QC only; do not call ingest_family.
    index_files = ingest._file_table(ingest_index)
    parsed_models = []
    for model_entry in model_entries:
        parsed = ingest._parse_model_family(model_entry, index_files)
        if parsed[0] != model_entry["model_id"] or parsed[6]:
            raise HandoffError(f"generated index has model parse errors for {model_entry['model_id']}: {parsed[6][:3]}")
        if len(parsed[5]) != EXPECTED_ROWS_PER_MODEL or any(row.get("status") != "PASS" for row in parsed[5]):
            raise HandoffError(f"generated index model {model_entry['model_id']} does not contain 240 PASS rows")
        parsed_models.append(parsed)
    if (len(row_to_npz) != EXPECTED_ROWS or len(row_to_frame_map) != EXPECTED_ROWS
            or len(set(row_to_npz.values())) != EXPECTED_ROWS
            or len(set(row_to_frame_map.values())) != EXPECTED_ROWS):
        raise HandoffError("full initial family does not have 720 unique seven-array NPZs and 720 unique source maps")

    coordinator_path = coordinator_dir / "coordinator_terminal_receipt.json"
    coord_raw, coord_pin = _read_regular(coordinator_path, "coordinator terminal receipt")
    handoff: dict[str, Any] = {
        "schema": SCHEMA,
        "status": "PASS_FULL_720_TERMINAL_READBACK_READY_FOR_SELECTION_INGEST",
        "selection_performed": False,
        "scientific_initialization_performed": False,
        "expected_rows": EXPECTED_ROWS,
        "verified_rows": EXPECTED_ROWS,
        "model_order": list(queue.MODEL_ORDER),
        "rows_by_model": {model: len(records_by_model[model]) for model in queue.MODEL_ORDER},
        "historical_prefix": {
            "index": history_pin, "index_self_sha256": HISTORICAL_INDEX_SELF_SHA256,
            "blocks": EXPECTED_HISTORICAL_BLOCKS,
            "blocks_by_model": dict(historical_counts),
            "child_exit_code_available": False,
            "child_exit_code": None,
            "rc_scope": historical.get("historical_process_rc_scope"),
        },
        "actual_queue": {
            "new_blocks": EXPECTED_QUEUE_BLOCKS,
            "new_rows": EXPECTED_QUEUE_BLOCKS * queue.BLOCK_ROWS,
            "event_count": final_state["event_count"],
            "last_event_sha256": final_state["last_event_sha256"],
            "next_start_by_model": final_state["next_start_by_model"],
            "paused_models": final_state["paused_models"],
            "active_claim": final_state["active_claim"],
            "process_rc_required": 0,
            "process_rc_verified_count": len(live_passes),
            "coordinator": coordinator_qc,
            "coordinator_terminal_receipt": coord_pin,
            "history_import_receipt": import_pin,
            "event_files": event_files,
        },
        "composed_family_plan": {**family_pin, "self_sha256": FAMILY_PLAN_SELF_SHA256,
                                 "plan_payload_sha256": family["plan_sha256"]},
        "scientific_input_pins_sha256": SCIENTIFIC_INPUT_SHA256,
        "initial_ingest_index": {
            "schema": ingest.SCHEMA,
            "status": ingest_index["status"],
            "index_sha256": ingest_index["index_sha256"],
            "expected_row_count": EXPECTED_ROWS,
            "files": len(files),
            "raw_matrix_npz_count": len(row_to_npz),
            "official_source_frame_map_count": len(row_to_frame_map),
        },
        "source_block_proofs": block_evidence,
        "row_artifact_bindings": artifact_rows,
        "source_code_pins": _source_code_pins(),
        "qualification_boundary": (
            "read-only actual terminal/NPZ/frame-map collection; selector and SPSA session not run"
        ),
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=False, exist_ok=False)
    index_path = output_dir / "initial_selector_ingest_index.json"
    handoff_path = output_dir / "terminal_handoff_manifest.json"
    index_raw = (json.dumps(ingest_index, sort_keys=True, indent=2,
                            ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    _write_exclusive(index_path, index_raw)
    index_local_pin = {
        "path": str(index_path.resolve(strict=True)), "bytes": len(index_raw),
        "sha256": digest(index_raw), "self_sha256": ingest_index["index_sha256"],
    }
    handoff["initial_ingest_index"].update(index_local_pin)
    handoff["handoff_sha256"] = self_hash(handoff, "handoff_sha256")
    handoff_raw = (json.dumps(handoff, sort_keys=True, indent=2,
                              ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    _write_exclusive(handoff_path, handoff_raw)
    _fsync_dir(output_dir)
    return {
        "status": handoff["status"],
        "index": {"path": str(index_path.resolve()), "bytes": len(index_raw),
                  "sha256": digest(index_raw), "self_sha256": ingest_index["index_sha256"]},
        "handoff": {"path": str(handoff_path.resolve()), "bytes": len(handoff_raw),
                    "sha256": digest(handoff_raw), "self_sha256": handoff["handoff_sha256"]},
        "counts": {"rows": EXPECTED_ROWS, "npz": len(row_to_npz),
                   "frame_maps": len(row_to_frame_map), "blocks": EXPECTED_TOTAL_BLOCKS,
                   "historical_blocks": EXPECTED_HISTORICAL_BLOCKS,
                   "actual_queue_blocks": EXPECTED_QUEUE_BLOCKS},
    }


def _source_code_pins() -> list[dict[str, Any]]:
    paths = (
        Path(__file__).resolve(),
        Path(queue.__file__).resolve(),
        Path(queue_runner.__file__).resolve(),
        Path(ingest.__file__).resolve(),
        Path(final_ingest.__file__).resolve(),
    )
    out = []
    for path in paths:
        raw, pin = _read_regular(path, "collector source closure")
        expected = EXPECTED_SOURCE_MODULE_PINS.get(path.name)
        if expected is not None and (pin["bytes"], pin["sha256"]) != expected:
            raise HandoffError(f"frozen helper source pin changed: {path.name}")
        out.append({"path": pin["path"], "bytes": pin["bytes"], "sha256": pin["sha256"]})
    return out


def _write_exclusive(path: Path, raw: bytes) -> None:
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--family-plan", type=Path, required=True)
    parser.add_argument("--historical-index", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--receipt-dir", type=Path, required=True)
    parser.add_argument("--coordinator-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = build_terminal_handoff(
            repo_root=args.repo_root, family_path=args.family_plan,
            historical_index_path=args.historical_index, state_dir=args.state_dir,
            receipt_dir=args.receipt_dir, coordinator_dir=args.coordinator_dir,
            output_dir=args.output_dir,
        )
    except Exception as exc:
        print(json.dumps({"status": "NOT_READY_OR_INVALID", "error": f"{type(exc).__name__}: {exc}"},
                         ensure_ascii=False, sort_keys=True, separators=(",", ":")), file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
