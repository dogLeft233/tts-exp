#!/usr/bin/env python3
"""Durably supervise exactly one SPSA pair per child process, up to 20 pairs.

This is orchestration only.  It does not create the initial selector receipt or
session, modify the selector, or evaluate a proposal itself.  Every pair launch
has a create-only intent and process-start receipt before waiting for its child.
An intent without a terminal receipt is read back and becomes UNKNOWN; it is
never replayed automatically.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence


PACKAGE = "scripts.experiments.cross_tfg_audio_head_20261003."
BOOTSTRAP_REL = "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_bootstrap_cli_attempt06.py"
SESSION_REL = "scripts/experiments/cross_tfg_audio_head_20261003/fit40_durable_spsa_session_attempt04.py"
BOOTSTRAP_PIN = (27_865, "d6daf57f9223898c34986dee15730f92a024aaf608f0acef40eea34295810ec0")
SESSION_PIN = (89_950, "9e7c6fe51f8426196770f9171a61c37c6444690f902f83466b1e0d5a703a6709")
DISPATCH_MOUNT_ROOT = Path("/dev/shm")
DISPATCH_ROOT_NAME = "fit40_joint_spsa_execution_recovery_attempt01"
COORDINATOR_SCHEMA = "fit40_spsa_twenty_pair_coordinator_v2"
START_INTENT_SCHEMA = "fit40_spsa_coordinator_start_intent_v2"
START_RECEIPT_SCHEMA = "fit40_spsa_coordinator_start_receipt_v2"
PAIR_INTENT_SCHEMA = "fit40_spsa_pair_launch_intent_v2"
PAIR_START_SCHEMA = "fit40_spsa_pair_process_start_v1"
PAIR_RECEIPT_SCHEMA = "fit40_spsa_pair_process_receipt_v1"
PAIR_READBACK_SCHEMA = "fit40_spsa_pair_unknown_readback_v1"
TERMINAL_STATUSES = {"COMPLETE_UPDATED", "NO_UPDATE", "UNKNOWN"}
MAX_PAIRS = 20
SESSION_MANIFEST_SCHEMA = "fit40_durable_spsa_session_manifest_v2"
RECOVERY_PREFLIGHT_SCHEMA = "fit40_spsa_pre_native_recovery_preflight_v1"
RECOVERY_PREFLIGHT_STATUS = "PASS_PRE_NATIVE_CONTROL_FAILURE_RECOVERY_AUTHORIZED"
RECOVERY_CONTROL_ROOT = Path(
    "/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01"
    "/job_control/fit40_spsa_twenty_pair_coordinator_attempt02")
RECOVERY_SESSION_ROOT = Path(
    "/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01"
    "/job_control/fit40_durable_spsa_session_attempt01")
RECOVERY_PROOF_PATH = Path(
    "/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01"
    "/job_control/fit40_spsa_recovery_preflight_attempt01/pre_native_recovery_preflight.json")
RECOVERY_NEW_COORDINATOR_ROOT = Path(
    "/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01"
    "/job_control/fit40_spsa_twenty_pair_coordinator_recovery_attempt01")
RECOVERY_DISPATCH_ROOT = Path("/dev/shm/fit40_joint_spsa_execution_attempt01")
RECOVERY_REPO_ROOT = Path("/dev/shm/fit40_queue_remote_package_attempt02/project")
RECOVERY_SESSION_ID = "fit40_spsa_00e52e562725_e6e2a096d641"
RECOVERY_COORDINATOR_ID = "fit40_twenty_pair_c4d6c2a7a5194e9fb5dbd0d1403f2400"
RECOVERY_PAIR_ID = "e5fa43c586f7665835da25c62d2e0acb3260f9662a2412baf913e2a688d73fc3"
RECOVERY_PLUS_PROPOSAL_ID = "e869a1bb8f667ee867ecfa7f8cc5a5c4cca7d1a218c7b8c3438df52bbc9cd9c8"
RECOVERY_MINUS_PROPOSAL_ID = "dab011a03e16c1b912c47f227e16443a8aef13afbeefd72391c734a0749178c8"
RECOVERY_FAMILY_PLAN = {
    "path": "/dev/shm/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/"
            "fit40_spsa_remote_family_composition_attempt03/fit40_three_model_family_plan_attempt03.json",
    "bytes": 2_584_596,
    "sha256": "9f7914033f9e7e89a60ae2618f7d429860d11de902a8400855fe57e37eb216a2",
    "self_sha256": "b071434696b2c094180473677d484934d7e3d04c808b7915000c8cfc6c6abb0c",
}
RECOVERY_PACKAGE = {
    "path": "/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/"
            "job_control/fit40_spsa_execution_package_attempt01/execution_package_manifest_attempt05.json",
    "bytes": 27_341,
    "sha256": "d0768e67664cbe4f8271f7f61b3a10f62619eaacbd0830c1eca3969a367e6711",
    "self_sha256": "7e6e10317245c7df02a34283762fa52985aa73adf98e66e2fc508fe8293e9601",
}
RECOVERY_INITIAL_SELECTOR_RECEIPT = {
    "path": "/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/"
            "job_control/fit40_initial_selector_receipt_attempt01.json",
    "bytes": 15_567,
    "sha256": "80eb055126e7114cb2129a8b72e1f8c0076d7948c34e3caf1abebd4b0962ef77",
    "self_sha256": "00e52e56272571f6626e3080e6141ee76c77ac5a5572bbddb8a00f0d36522003",
}
RECOVERY_READBACK_CAPTURE = {
    "path": ".codex/scratch/root_spsa_first_pair_actual_review_attempt01/remote_actual_readback.json",
    "bytes": 2_940_563,
    "sha256": "54dfe212d492d2ec115a5eb8c3990e1efd57ced49d592d5092d54f7001cee73d",
    "self_sha256": "897c1e599921e1ce6878649e0ef97cb820afdb4b6fe902a23c74ad891c62d28a",
    "root_qc_self_sha256": "b99f09a80749a788d729b176a019385a313580b629e090acd96649124c8b11bd",
}
RECOVERY_EXPECTED_FILES = (
    ("coordinator", "coordinator_manifest.json", 2400, "98a23b89df8a3e049a7d3a6b8b0d7e6ebc3d9c12efe53c21c34697913ccc33be"),
    ("coordinator", "coordinator_start_intent.json", 2155, "1f3b38590d10102d785fca691af74951995afa68d8816606bdddaf31c074fcbf"),
    ("coordinator", "coordinator_start_receipt.json", 3053, "0e723d86b260d8d1356698fec054004bc964c867139b2052b741452514e4c1cc"),
    ("coordinator", "pairs/pair_01/child.stderr", 85, "504ac626d8dd629b4b6f42a38968721c29b37eb8775c1069c85160e34894735a"),
    ("coordinator", "pairs/pair_01/child.stdout", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("coordinator", "pairs/pair_01/launch_intent.json", 6005, "ccea374bd5522fcf9ccf1182f056ea9448c1f6cd29d0c492fb852f0eaba0308f"),
    ("coordinator", "pairs/pair_01/packet/minus/plan.json", 202577, "aa75d210f09af406a5e9badfef4caa40d2f43906f10b9a2daf804717b4a84da6"),
    ("coordinator", "pairs/pair_01/packet/minus/planned_rows.jsonl", 57117, "5e90209b9b438718498d462b6a34c81a519af1adf914e22cab9b05e56c521aa3"),
    ("coordinator", "pairs/pair_01/packet/packet_index.json", 38378, "3e7350dd3d6e40262727e4b42328bb7522c2ee25c75064ea8fa1331389059b35"),
    ("coordinator", "pairs/pair_01/packet/plus/plan.json", 202524, "ea2144dad1a4380840e20e199f1acb0f7446528d3fc92c5c5ac81db912ab6ae4"),
    ("coordinator", "pairs/pair_01/packet/plus/planned_rows.jsonl", 57069, "67b8f0be86aaac9dee5241db605a63394d270debd152177134569033a7ba8dd5"),
    ("coordinator", "pairs/pair_01/process_receipt.json", 3693, "2d2469b5484a898ebfceda8c75881a88771c05860e47aea5760887fb3a140b34"),
    ("coordinator", "pairs/pair_01/process_start.json", 1164, "a1b6008a4d212a583f493174ccae6df5879ec789906f6cefa1c7e4ea1d8368cb"),
    ("coordinator", "process_logs/coordinator.stderr", 0, "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"),
    ("coordinator", "process_logs/coordinator.stdout", 567, "a1938ae98bf64319750739996b143b037f7402efaa1b48b8408176b7ea882fce"),
    ("session", "session_manifest.json", 1610799, "e2381e513efa23d0ecfe5d75d1d57e35224c29ae12c044db1126b7fb1c16a034"),
    ("session", "events/event_000000.json", 1685, "b2f477e90358dff974aeda6717d956536fc5704540ca242a51b4e585bc2babad"),
    ("session", "events/event_000001.json", 10014, "753aebe84afd8af97ca8463678c22057ee3d99eeb51538ad32296514df3d4764"),
)
RECOVERY_EXPECTED_SOURCES = (
    ("scripts/experiments/cross_tfg_audio_head_20261003/collect_fit40_initial_720_terminal_handoff_attempt08.py", 74897, "a83f4905b2765ba3637c3ac288be6e9729e5b39bc413032931ea90a83737a3a1"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/fit40_fixed_queue_controller_attempt03.py", 80513, "dafe417e42cd7660b3b0c41731b30e10272f45ab4ead6911ea75c9f1fce1667a"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_fixed_queue_attempt05.py", 25969, "38e924bbccb54c04e9c7618502bdc57b38a71595babc71f3d9c51b646165de67"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/fit_only_stagea_ingest_attempt03.py", 53184, "22cd7f983b22babcae0d9c120f4720cb84a218ec4d2b84551d5e1668d23aeb95"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/fit_only_final_center_stagea_ingest_attempt05.py", 106167, "3384ebe67ed99569af6307c88746a0b16329897dda9935678677dcf80147e8ea"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/fit_only_spsa_selection_attempt06.py", 66767, "dda610b985026be7f9328c479358e96c983a582c511964e8c6e72a6d398692e2"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/prepare_fit40_spsa_stagea_packets_attempt05.py", 47748, "cd5e2b80ab0bdd4f599d863468c687724898b2e6baad451a5acfe2123648bdae"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/fit40_durable_spsa_session_attempt04.py", 89950, "9e7c6fe51f8426196770f9171a61c37c6444690f902f83466b1e0d5a703a6709"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_point_dispatcher_attempt06.py", 84195, "871e4dbf63aa037ef0e8e0beab4c7b00dcf4a7c68b439b60b35d0dc72818a4c4"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_point_execution_attempt04.py", 67081, "7747052df91393f1cbf1138dc76523b69f888674d0ab6e07c7fa9cfadfcf1e46"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_center_dispatcher_attempt04.py", 67259, "a7cd2e11d1e057dd84715d2b6a20b4c6dc6a7017afb96ed25a6159621fccaa41"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_bootstrap_cli_attempt05.py", 27865, "7ba02294e9f16e7716a74919013b7b1ab16ab5c80d4fce2092fa4c41ff528aea"),
    ("scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_twenty_pair_coordinator_attempt05.py", 54963, "b17ad129895d3871381d1853907f92bab6c3bc1e5bedf7c62ffc7cc8a39b8427"),
)
RECOVERY_OLD_CHILDREN = ((660335, 6369467), (660380, 6370442))
RECOVERY_OLD_CHILD_ERROR = "bootstrap failed closed: ValueError: not enough values to unpack (expected 3, got 2)\n"


class CoordinatorError(RuntimeError):
    """Invalid source, durable state, or child execution evidence."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return _sha(_canonical(body))


def _seal(value: Mapping[str, Any], field: str) -> dict[str, Any]:
    result = dict(value)
    result.pop(field, None)
    result[field] = _sha(_canonical(result))
    return result


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _regular_bytes(path: Path, label: str, *, allow_symlink: bool = False) -> bytes:
    if not path.is_absolute() or (path.is_symlink() and not allow_symlink) or not path.is_file():
        raise CoordinatorError("%s must be an absolute regular file: %s" % (label, path))
    target = path.resolve(strict=True) if allow_symlink else path
    mode = target.stat(follow_symlinks=False).st_mode
    if not stat.S_ISREG(mode):
        raise CoordinatorError("%s is not a regular file: %s" % (label, path))
    return target.read_bytes()


def _file_pin(path: Path, *, allow_symlink: bool = False) -> dict[str, Any]:
    raw = _regular_bytes(path, "pinned file", allow_symlink=allow_symlink)
    resolved = path.resolve(strict=True)
    return {"path": str(resolved), "bytes": len(raw), "sha256": _sha(raw)}


def _self_file_pin(path: Path, label: str, field: str) -> dict[str, Any]:
    _value, _raw, pin = _read_self_json(path, label, field)
    return pin


def _read_self_json(path: Path, label: str, field: str) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    raw = _regular_bytes(path, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise CoordinatorError("%s is not UTF-8 JSON" % label) from exc
    if not isinstance(value, dict) or value.get(field) != _self_hash(value, field):
        raise CoordinatorError("%s self hash is missing or invalid" % label)
    pin = {"path": str(path.resolve(strict=True)), "bytes": len(raw),
           "sha256": _sha(raw), "self_sha256": value[field]}
    return value, raw, pin


def _recovery_tree(root: Path, label: str) -> tuple[set[str], set[str]]:
    if root.is_symlink() or not root.is_dir():
        raise CoordinatorError("%s root is missing or unsafe" % label)
    directories: set[str] = set()
    files: set[str] = set()
    for current, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
        current_path = Path(current)
        for name in list(dirnames):
            child = current_path / name
            if child.is_symlink() or not child.is_dir():
                raise CoordinatorError("%s contains an unsafe directory entry" % label)
            directories.add(child.relative_to(root).as_posix())
        for name in filenames:
            child = current_path / name
            if child.is_symlink() or not child.is_file():
                raise CoordinatorError("%s contains an unsafe file entry" % label)
            files.add(child.relative_to(root).as_posix())
    return directories, files


def _decode_recovery_json(raw: bytes, label: str, *, self_field: str | None = None) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise CoordinatorError("%s is not UTF-8 JSON" % label) from exc
    if not isinstance(value, dict):
        raise CoordinatorError("%s is not a JSON object" % label)
    if self_field is not None and value.get(self_field) != _self_hash(value, self_field):
        raise CoordinatorError("%s self hash is invalid" % label)
    return value


def _collect_pre_native_recovery_evidence(
    *, old_coordinator_dir: str | Path, session_dir: str | Path,
    old_dispatch_root: str | Path,
) -> dict[str, Any]:
    """Read-only proof that the first child failed before any native block dispatch."""
    old_coordinator = Path(old_coordinator_dir)
    session_root = Path(session_dir)
    old_dispatch = Path(old_dispatch_root)
    for actual, expected, label in (
        (old_coordinator, RECOVERY_CONTROL_ROOT, "old coordinator directory"),
        (session_root, RECOVERY_SESSION_ROOT, "durable session directory"),
        (old_dispatch, RECOVERY_DISPATCH_ROOT, "old native output directory"),
    ):
        if actual != expected or actual.is_symlink() or not actual.is_dir():
            raise CoordinatorError("%s does not match the exact prior attempt path" % label)

    control_expected_files = {relative for root_name, relative, _size, _sha in RECOVERY_EXPECTED_FILES
                              if root_name == "coordinator"}
    session_expected_files = {relative for root_name, relative, _size, _sha in RECOVERY_EXPECTED_FILES
                              if root_name == "session"}
    control_dirs, control_files = _recovery_tree(old_coordinator, "old coordinator capture")
    session_dirs, session_files = _recovery_tree(session_root, "durable session capture")
    dispatch_dirs, dispatch_files = _recovery_tree(old_dispatch, "old native output tree")
    if (control_files != control_expected_files
            or control_dirs != {"pairs", "pairs/pair_01", "pairs/pair_01/packet",
                                "pairs/pair_01/packet/minus", "pairs/pair_01/packet/plus",
                                "process_logs"}
            or session_files != session_expected_files or session_dirs != {"events"}
            or dispatch_dirs != {"pairs", "pairs/pair_01", "pairs/pair_01/dispatch_root"}
            or dispatch_files):
        raise CoordinatorError("captured coordinator/session/native tree differs from the sealed 18-file failure state")

    files: list[dict[str, Any]] = []
    raw_by_key: dict[tuple[str, str], bytes] = {}
    root_by_name = {"coordinator": old_coordinator, "session": session_root}
    for root_name, relative, expected_bytes, expected_sha in RECOVERY_EXPECTED_FILES:
        path = root_by_name[root_name] / relative
        raw = _regular_bytes(path, "recovery evidence file")
        pin = {"root": root_name, "relative_path": relative,
               "path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": _sha(raw)}
        if (len(raw), pin["sha256"]) != (expected_bytes, expected_sha):
            raise CoordinatorError("prior 18-file capture differs at %s" % path)
        files.append(pin)
        raw_by_key[(root_name, relative)] = raw

    source_files: list[dict[str, Any]] = []
    for relative, expected_bytes, expected_sha in RECOVERY_EXPECTED_SOURCES:
        path = RECOVERY_REPO_ROOT / relative
        raw = _regular_bytes(path, "old frozen source dependency")
        pin = {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": _sha(raw)}
        if (len(raw), pin["sha256"]) != (expected_bytes, expected_sha):
            raise CoordinatorError("old source closure changed at %s" % path)
        source_files.append({"relative_path": relative, **pin})

    def old_doc(root_name: str, relative: str, self_field: str) -> dict[str, Any]:
        return _decode_recovery_json(raw_by_key[(root_name, relative)], relative, self_field=self_field)

    coordinator = old_doc("coordinator", "coordinator_manifest.json", "coordinator_sha256")
    start_intent = old_doc("coordinator", "coordinator_start_intent.json", "intent_sha256")
    start_receipt = old_doc("coordinator", "coordinator_start_receipt.json", "receipt_sha256")
    pair_root = "pairs/pair_01/"
    pair_intent = old_doc("coordinator", pair_root + "launch_intent.json", "intent_sha256")
    child_start = old_doc("coordinator", pair_root + "process_start.json", "start_sha256")
    child_receipt = old_doc("coordinator", pair_root + "process_receipt.json", "receipt_sha256")
    packet_raw = raw_by_key[("coordinator", pair_root + "packet/packet_index.json")]
    packet = _decode_recovery_json(packet_raw, "pair packet index")
    session = old_doc("session", "session_manifest.json", "manifest_sha256")
    event0 = old_doc("session", "events/event_000000.json", "event_sha256")
    event1 = old_doc("session", "events/event_000001.json", "event_sha256")
    plus_plan = old_doc("coordinator", pair_root + "packet/plus/plan.json", "plan_sha256")
    minus_plan = old_doc("coordinator", pair_root + "packet/minus/plan.json", "plan_sha256")

    old_bootstrap = {
        "path": str(RECOVERY_REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_bootstrap_cli_attempt05.py"),
        "bytes": 27_865,
        "sha256": "7ba02294e9f16e7716a74919013b7b1ab16ab5c80d4fce2092fa4c41ff528aea",
    }
    old_session_module = {
        "path": str(RECOVERY_REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/fit40_durable_spsa_session_attempt04.py"),
        "bytes": 89_950,
        "sha256": "9e7c6fe51f8426196770f9171a61c37c6444690f902f83466b1e0d5a703a6709",
    }
    old_coordinator_source = {
        "path": str(RECOVERY_REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_twenty_pair_coordinator_attempt05.py"),
        "bytes": 54_963,
        "sha256": "b17ad129895d3871381d1853907f92bab6c3bc1e5bedf7c62ffc7cc8a39b8427",
    }
    session_manifest_pin = {"path": str(RECOVERY_SESSION_ROOT / "session_manifest.json"),
                            "bytes": 1_610_799,
                            "sha256": "e2381e513efa23d0ecfe5d75d1d57e35224c29ae12c044db1126b7fb1c16a034",
                            "self_sha256": "b0bf502c0531f5f07a4be41f6ff8bc632bfcdb386eea4106dea74835739c3f30"}
    expected_old_argv = [
        "/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/bin/python",
        str(RECOVERY_REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_twenty_pair_coordinator_attempt05.py"),
        "run", "--repo-root", str(RECOVERY_REPO_ROOT), "--session-dir", str(RECOVERY_SESSION_ROOT),
        "--coordinator-dir", str(RECOVERY_CONTROL_ROOT), "--dispatch-root", str(RECOVERY_DISPATCH_ROOT),
    ]
    if (coordinator.get("schema") != "fit40_spsa_twenty_pair_coordinator_v2"
            or coordinator.get("status") != "FROZEN_SOURCE_PINNED_COORDINATOR"
            or coordinator.get("coordinator_id") != RECOVERY_COORDINATOR_ID
            or coordinator.get("repo_root") != str(RECOVERY_REPO_ROOT)
            or coordinator.get("session_dir") != str(RECOVERY_SESSION_ROOT)
            or coordinator.get("dispatch_storage_root") != str(RECOVERY_DISPATCH_ROOT)
            or coordinator.get("bootstrap_pin") != old_bootstrap
            or coordinator.get("session_module_pin") != old_session_module
            or coordinator.get("coordinator_source_pin") != old_coordinator_source
            or coordinator.get("session_manifest_pin") != session_manifest_pin
            or coordinator.get("science_run_started_at_manifest_seal") is not False):
        raise CoordinatorError("prior coordinator/source/session manifest identity is not exact")
    if (Path(str(coordinator.get("repo_root", ""))) != RECOVERY_REPO_ROOT
            or Path(str(coordinator.get("session_dir", ""))) != RECOVERY_SESSION_ROOT):
        raise CoordinatorError("prior coordinator repository/session roots are not canonical")

    old_manifest_pin = {"path": str(RECOVERY_CONTROL_ROOT / "coordinator_manifest.json"),
                        "bytes": 2400, "sha256": "98a23b89df8a3e049a7d3a6b8b0d7e6ebc3d9c12efe53c21c34697913ccc33be",
                        "self_sha256": coordinator["coordinator_sha256"]}
    if (start_intent.get("schema") != "fit40_spsa_coordinator_start_intent_v2"
            or start_intent.get("coordinator_id") != RECOVERY_COORDINATOR_ID
            or start_intent.get("coordinator_manifest_pin") != old_manifest_pin
            or start_intent.get("coordinator_source_pin") != old_coordinator_source
            or start_intent.get("python_runtime") != coordinator.get("python_runtime")
            or start_intent.get("dispatch_storage_root") != str(RECOVERY_DISPATCH_ROOT)
            or start_intent.get("argv") != expected_old_argv):
        raise CoordinatorError("prior coordinator start intent does not match its exact source/runtime/paths")
    start_intent_pin = {"path": str(RECOVERY_CONTROL_ROOT / "coordinator_start_intent.json"),
                        "bytes": 2155, "sha256": "1f3b38590d10102d785fca691af74951995afa68d8816606bdddaf31c074fcbf",
                        "self_sha256": start_intent["intent_sha256"]}
    if (start_receipt.get("schema") != "fit40_spsa_coordinator_start_receipt_v2"
            or start_receipt.get("status") != "STARTED"
            or start_receipt.get("coordinator_id") != RECOVERY_COORDINATOR_ID
            or start_receipt.get("intent_pin") != start_intent_pin
            or start_receipt.get("coordinator_manifest_pin") != old_manifest_pin
            or start_receipt.get("coordinator_source_pin") != old_coordinator_source
            or start_receipt.get("python_runtime") != coordinator.get("python_runtime")
            or start_receipt.get("dispatch_storage_root") != str(RECOVERY_DISPATCH_ROOT)
            or start_receipt.get("argv") != expected_old_argv
            or start_receipt.get("child_pid") != RECOVERY_OLD_CHILDREN[0][0]
            or start_receipt.get("child_start_ticks") != RECOVERY_OLD_CHILDREN[0][1]):
        raise CoordinatorError("prior coordinator process-start receipt is not the captured process")

    family = _decode_recovery_json(
        _regular_bytes(Path(RECOVERY_FAMILY_PLAN["path"]), "prior family plan"),
        "prior composed family plan", self_field="plan_sha256")
    family_raw = _regular_bytes(Path(RECOVERY_FAMILY_PLAN["path"]), "prior family plan")
    family_pin = {"path": str(Path(RECOVERY_FAMILY_PLAN["path"]).resolve(strict=True)),
                  "bytes": len(family_raw), "sha256": _sha(family_raw),
                  "self_sha256": family.get("plan_sha256")}
    package_raw = _regular_bytes(Path(RECOVERY_PACKAGE["path"]), "prior execution package")
    package = _decode_recovery_json(package_raw, "prior execution package", self_field="manifest_sha256")
    package_pin = {"path": str(Path(RECOVERY_PACKAGE["path"]).resolve(strict=True)),
                   "bytes": len(package_raw), "sha256": _sha(package_raw),
                   "self_sha256": package.get("manifest_sha256")}
    if (family_pin != RECOVERY_FAMILY_PLAN or family.get("status") != "FROZEN"
            or package_pin != RECOVERY_PACKAGE or package.get("status") != "FROZEN"
            or package.get("composed_family_plan_pin") != family_pin
            or session.get("schema") != SESSION_MANIFEST_SCHEMA or session.get("status") != "OPEN"
            or session.get("session_id") != RECOVERY_SESSION_ID
            or session.get("initial_selection_status") != "NOT_ADMITTED"
            or session.get("metric_backend") != "torch_cpu"
            or not isinstance(session.get("initial_theta"), list)
            or len(session["initial_theta"]) != 9
            or session.get("spsa_seed_method_id") != "fit40_native_transition_head_v3_attempt11_step060"
            or session.get("composed_family_plan") != family_pin
            or session.get("execution_package_manifest") != package_pin
            or session.get("initial_selector_receipt") != RECOVERY_INITIAL_SELECTOR_RECEIPT):
        raise CoordinatorError("prior session, source family, or execution package identity changed")

    pair_pin = {"path": str(RECOVERY_CONTROL_ROOT / pair_root / "launch_intent.json"),
                "bytes": 6005, "sha256": "ccea374bd5522fcf9ccf1182f056ea9448c1f6cd29d0c492fb852f0eaba0308f",
                "self_sha256": pair_intent["intent_sha256"]}
    expected_pair_argv = [
        "/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/bin/python",
        str(RECOVERY_REPO_ROOT / "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_bootstrap_cli_attempt05.py"),
        "--repo-root", str(RECOVERY_REPO_ROOT), "run-pair",
        "--session-dir", str(RECOVERY_SESSION_ROOT),
        "--packet-dir", str(RECOVERY_CONTROL_ROOT / pair_root / "packet"),
        "--dispatch-root", str(RECOVERY_DISPATCH_ROOT / "pairs/pair_01/dispatch_root"),
    ]
    if (event0.get("event_type") != "SESSION_INITIALIZED" or event0.get("event_index") != 0
            or event0.get("session_id") != RECOVERY_SESSION_ID
            or event0.get("previous_event_sha256") is not None
            or event1.get("event_type") != "PAIR_ASKED" or event1.get("event_index") != 1
            or event1.get("session_id") != RECOVERY_SESSION_ID
            or event1.get("previous_event_sha256") != event0.get("event_sha256")
            or event1.get("pair_id") != RECOVERY_PAIR_ID or event1.get("iteration") != 0
            or not isinstance(event1.get("ask_pair"), Mapping)
            or pair_intent.get("schema") != PAIR_INTENT_SCHEMA
            or pair_intent.get("pair_index") != 1 or pair_intent.get("pair_id") != RECOVERY_PAIR_ID
            or pair_intent.get("session_id") != RECOVERY_SESSION_ID
            or pair_intent.get("coordinator_id") != RECOVERY_COORDINATOR_ID
            or pair_intent.get("bootstrap_pin") != old_bootstrap
            or pair_intent.get("session_manifest_pin") != {k: session_manifest_pin[k] for k in
                                                                ("path", "bytes", "sha256")}
            or pair_intent.get("coordinator_manifest_pin") != {k: old_manifest_pin[k] for k in
                                                                  ("path", "bytes", "sha256")}
            or pair_intent.get("proposal_pair", {}).get("pair_id") != RECOVERY_PAIR_ID
            or pair_intent.get("proposal_pair", {}).get("iteration") != 0
            or pair_intent.get("proposal_pair", {}).get("plus", {}).get("proposal_id") != RECOVERY_PLUS_PROPOSAL_ID
            or pair_intent.get("proposal_pair", {}).get("minus", {}).get("proposal_id") != RECOVERY_MINUS_PROPOSAL_ID
            or pair_intent.get("proposal_pair", {}).get("plus") != event1["ask_pair"].get("plus")
            or pair_intent.get("proposal_pair", {}).get("minus") != event1["ask_pair"].get("minus")
            or pair_intent.get("pair_sha256") != _sha(_canonical(pair_intent.get("proposal_pair", {})))
            or pair_intent.get("iteration") != 0
            or pair_intent.get("plus_proposal_id") != RECOVERY_PLUS_PROPOSAL_ID
            or pair_intent.get("minus_proposal_id") != RECOVERY_MINUS_PROPOSAL_ID
            or event1.get("session_id") != RECOVERY_SESSION_ID):
        raise CoordinatorError("old pair packet/intent does not bind the sole durable pending ask")
    if (pair_intent.get("argv") != expected_pair_argv
            or pair_intent.get("packet_dir") != str(RECOVERY_CONTROL_ROOT / pair_root / "packet")
            or pair_intent.get("dispatch_storage_root") != str(RECOVERY_DISPATCH_ROOT)
            or pair_intent.get("dispatch_root") != str(RECOVERY_DISPATCH_ROOT / "pairs/pair_01/dispatch_root")):
        raise CoordinatorError("old pair launch argv/output namespace differs from the captured failed attempt")

    if (packet.get("schema") != "fit40_spsa_stagea_execution_packet_v1"
            or packet.get("status") != "PLANS_ONLY_REQUIRES_BOUND_RUNTIME_AND_EXECUTION"
            or packet.get("execution_started") is not False or packet.get("model_calls_started") is not False
            or packet.get("planned_row_total") != 48
            or packet.get("scope", {}).get("pair_id") != RECOVERY_PAIR_ID
            or packet.get("scope", {}).get("source_seed_method_id") != "fit40_native_transition_head_v3_attempt11_step060"
            or packet.get("scope", {}).get("input_pins_sha256") != session.get("scientific_input_pins_sha256")
            or packet.get("parent_plan") != {k: family_pin[k] for k in ("path", "bytes", "sha256")}):
        raise CoordinatorError("prior proposal packet is not the exact 48-row pre-execution packet")
    side_docs = (("plus", plus_plan, RECOVERY_PLUS_PROPOSAL_ID, 202524,
                  "ea2144dad1a4380840e20e199f1acb0f7446528d3fc92c5c5ac81db912ab6ae4",
                  "248fa566dd3e6e84df23c10f407df774f5086c14271710b29b59d9de5d7796ba"),
                 ("minus", minus_plan, RECOVERY_MINUS_PROPOSAL_ID, 202577,
                  "aa75d210f09af406a5e9badfef4caa40d2f43906f10b9a2daf804717b4a84da6",
                  "b2debfcf52577a3b078bd4132b552639239d0ed9919562cdb6c11c0aabe3f6c9"))
    for side, plan, proposal_id, plan_bytes, plan_raw_sha, plan_self in side_docs:
        scope = plan.get("execution", {}).get("spsa_execution_scope")
        plan_entry = next((item for item in packet.get("plans", [])
                           if isinstance(item, Mapping) and item.get("label") == side), None)
        expected_plan_path = str(RECOVERY_CONTROL_ROOT / pair_root / "packet" / side / "plan.json")
        plan_raw = raw_by_key[("coordinator", pair_root + "packet/" + side + "/plan.json")]
        expected_rows_path = str(RECOVERY_CONTROL_ROOT / pair_root / "packet" / side / "planned_rows.jsonl")
        rows_rel = pair_root + "packet/" + side + "/planned_rows.jsonl"
        rows_raw = raw_by_key[("coordinator", rows_rel)]
        if (plan_entry is None or plan.get("plan_sha256") != plan_self
                or plan.get("run_id") != plan_entry.get("run_id")
                or plan_entry.get("plan") != {"path": expected_plan_path,
                                                "bytes": plan_bytes,
                                                "sha256": plan_raw_sha}
                or plan_entry.get("plan_self_sha256") != plan_self
                or plan_entry.get("planned_row_count") != 24
                or not isinstance(plan.get("rows"), list) or len(plan["rows"]) != 24
                or not isinstance(scope, Mapping)
                or scope.get("pair_id") != RECOVERY_PAIR_ID or scope.get("iteration") != 0
                or scope.get("side") != side or scope.get("proposal_id") != proposal_id
                or scope.get("source_seed_method_id") != "fit40_native_transition_head_v3_attempt11_step060"
                or scope.get("input_pins_sha256") != session.get("scientific_input_pins_sha256")
                or len(plan_raw) != plan_bytes or _sha(plan_raw) != plan_raw_sha):
            raise CoordinatorError("prior %s point plan is not the exact proposal identity in the captured packet" % side)
        expected_plan_rows = plan_entry.get("planned_rows")
        if (not isinstance(expected_plan_rows, Mapping)
                or expected_plan_rows.get("path") != expected_rows_path
                or expected_plan_rows.get("sha256") != _sha(rows_raw)
                or expected_plan_rows.get("bytes") != len(rows_raw)
                or plan_entry.get("planned_row_count") != len(plan["rows"])):
            raise CoordinatorError("prior %s planned-row file does not match its packet pin" % side)

    if (child_start.get("schema") != PAIR_START_SCHEMA or child_start.get("status") != "STARTED"
            or child_start.get("coordinator_id") != RECOVERY_COORDINATOR_ID
            or child_start.get("child_pid") != RECOVERY_OLD_CHILDREN[1][0]
            or child_start.get("child_start_ticks") != RECOVERY_OLD_CHILDREN[1][1]
            or child_start.get("argv") != pair_intent.get("argv")
            or child_receipt.get("schema") != PAIR_RECEIPT_SCHEMA
            or child_receipt.get("status") != "UNKNOWN"
            or child_receipt.get("child_exit_code") != 2
            or child_receipt.get("child_pid") != RECOVERY_OLD_CHILDREN[1][0]
            or child_receipt.get("child_start_ticks") != RECOVERY_OLD_CHILDREN[1][1]
            or child_receipt.get("pair_id") != RECOVERY_PAIR_ID or child_receipt.get("pair_index") != 1
            or child_receipt.get("iteration") != 0
            or child_receipt.get("plus_proposal_id") != RECOVERY_PLUS_PROPOSAL_ID
            or child_receipt.get("minus_proposal_id") != RECOVERY_MINUS_PROPOSAL_ID
            or child_receipt.get("coordinator_id") != RECOVERY_COORDINATOR_ID
            or child_receipt.get("relaunch_prohibited") is not True
            or child_receipt.get("packet_index_pin") is not None
            or child_receipt.get("launch_intent_pin") is not None
            or child_receipt.get("argv") != pair_intent.get("argv")):
        raise CoordinatorError("old child receipt is not the exact pre-dispatch rc=2 failure")
    process_start_pin = {"path": str(RECOVERY_CONTROL_ROOT / pair_root / "process_start.json"),
                         "bytes": 1164,
                         "sha256": "a1b6008a4d212a583f493174ccae6df5879ec789906f6cefa1c7e4ea1d8368cb",
                         "self_sha256": child_start["start_sha256"]}
    pair_intent_pin = {"path": str(RECOVERY_CONTROL_ROOT / pair_root / "launch_intent.json"),
                       "bytes": 6005,
                       "sha256": "ccea374bd5522fcf9ccf1182f056ea9448c1f6cd29d0c492fb852f0eaba0308f",
                       "self_sha256": pair_intent["intent_sha256"]}
    if (child_receipt.get("process_start_pin") != process_start_pin
            or child_receipt.get("intent_pin") != pair_intent_pin
            or child_receipt.get("started_utc") != child_start.get("started_utc")
            or child_receipt.get("ended_utc") is None):
        raise CoordinatorError("old child receipt does not bind its exact intent and observed start")
    stderr_raw = raw_by_key[("coordinator", pair_root + "child.stderr")]
    if stderr_raw.decode("utf-8", errors="replace") != RECOVERY_OLD_CHILD_ERROR:
        raise CoordinatorError("old child stderr is not the exact unpacking failure")
    coordinator_stdout_raw = raw_by_key[("coordinator", "process_logs/coordinator.stdout")]
    coordinator_result = _decode_recovery_json(coordinator_stdout_raw, "old coordinator stdout result")
    if (coordinator_result.get("status") != "UNKNOWN"
            or coordinator_result.get("completed_pairs") != 0
            or coordinator_result.get("stop_pair_index") != 1
            or not isinstance(coordinator_result.get("pairs"), list)
            or len(coordinator_result["pairs"]) != 1
            or coordinator_result["pairs"][0].get("pair_id") != RECOVERY_PAIR_ID
            or coordinator_result["pairs"][0].get("status") != "UNKNOWN"
            or coordinator_result["pairs"][0].get("receipt_pin") != {
                "path": str(RECOVERY_CONTROL_ROOT / pair_root / "process_receipt.json"),
                "bytes": 3693,
                "sha256": "2d2469b5484a898ebfceda8c75881a88771c05860e47aea5760887fb3a140b34",
                "self_sha256": child_receipt.get("receipt_sha256"),
            }):
        raise CoordinatorError("old coordinator terminal log does not match its failed first child")
    if raw_by_key[("coordinator", "process_logs/coordinator.stderr")]:
        raise CoordinatorError("old coordinator stderr was expected to be empty")
    stdout_pin = child_receipt.get("stdout_pin", {})
    stderr_pin = child_receipt.get("stderr_pin", {})
    if (stdout_pin.get("path") != str(RECOVERY_CONTROL_ROOT / pair_root / "child.stdout")
            or stdout_pin.get("bytes") != 0 or stdout_pin.get("sha256") != _sha(b"")
            or stderr_pin.get("path") != str(RECOVERY_CONTROL_ROOT / pair_root / "child.stderr")
            or stderr_pin.get("bytes") != len(stderr_raw) or stderr_pin.get("sha256") != _sha(stderr_raw)
            or not isinstance(pair_pin, Mapping)):
        raise CoordinatorError("captured child stdout/stderr pins do not match the rc=2 evidence")
    for pid, ticks in RECOVERY_OLD_CHILDREN:
        proc = Path("/proc") / str(pid)
        if proc.exists() or _proc_start_ticks(pid) is not None:
            raise CoordinatorError("a process identity from the failed first attempt is still observable")

    return {
        "schema": RECOVERY_PREFLIGHT_SCHEMA,
        "status": RECOVERY_PREFLIGHT_STATUS,
        "failure_scope": "PRE_NATIVE_CONTROL_FAILURE_ONLY",
        "prior_actual_readback_capture": dict(RECOVERY_READBACK_CAPTURE),
        "prior_coordinator_id": RECOVERY_COORDINATOR_ID,
        "prior_session_id": RECOVERY_SESSION_ID,
        "prior_pair_id": RECOVERY_PAIR_ID,
        "prior_child_exit_code": 2,
        "prior_child_error": RECOVERY_OLD_CHILD_ERROR.rstrip("\n"),
        "prior_coordinator_terminal_status": "UNKNOWN",
        "prior_child_processes_absent": [
            {"pid": pid, "start_ticks": ticks, "proc_entry_absent": True}
            for pid, ticks in RECOVERY_OLD_CHILDREN
        ],
        "prior_session_event_count": 2,
        "prior_session_event_types": ["SESSION_INITIALIZED", "PAIR_ASKED"],
        "prior_session_has_point_dispatch_intent": False,
        "prior_session_iteration": 0,
        "prior_session_pending_pair_id": RECOVERY_PAIR_ID,
        "prior_model_execution_started": False,
        "prior_model_calls_started": False,
        "prior_native_dispatch_tree_empty": True,
        "prior_native_dispatch_directories": ["pairs", "pairs/pair_01",
                                               "pairs/pair_01/dispatch_root"],
        "prior_native_dispatch_files": [],
        "prior_native_dispatch_root": str(RECOVERY_DISPATCH_ROOT),
        "prior_family_plan_pin": family_pin,
        "prior_execution_package_pin": package_pin,
        "prior_initial_selector_receipt_pin": dict(RECOVERY_INITIAL_SELECTOR_RECEIPT),
        "prior_packet_index_pin": {"path": str(RECOVERY_CONTROL_ROOT / pair_root / "packet/packet_index.json"),
                                    "bytes": len(packet_raw), "sha256": _sha(packet_raw)},
        "prior_proposal_ids": {"plus": RECOVERY_PLUS_PROPOSAL_ID, "minus": RECOVERY_MINUS_PROPOSAL_ID},
        "captured_18_files": files,
        "verified_prior_source_closure": source_files,
        "recovery_scope": "replay the exact durable pending ask only under a new coordinator/output namespace",
        "no_prior_failure_reinterpreted_as_success": True,
    }


def _seal_pre_native_recovery_preflight(
    *, old_coordinator_dir: str | Path, session_dir: str | Path,
    old_dispatch_root: str | Path, proof_out: str | Path,
) -> dict[str, Any]:
    target = Path(proof_out)
    if target != RECOVERY_PROOF_PATH or target.exists() or target.is_symlink():
        raise CoordinatorError("pre-native recovery proof must use its exact fresh persistent leaf")
    evidence = _collect_pre_native_recovery_evidence(
        old_coordinator_dir=old_coordinator_dir, session_dir=session_dir,
        old_dispatch_root=old_dispatch_root)
    for ancestor in (target.parent, *target.parent.parents):
        if ancestor.exists() and (ancestor.is_symlink() or not ancestor.is_dir()):
            raise CoordinatorError("pre-native recovery proof parent chain is unsafe")
    target.parent.mkdir(mode=0o700, exist_ok=True)
    document = {**evidence, "created_utc": _utc_now()}
    pin = _write_json_exclusive(target, document, "recovery_sha256")
    return {"status": RECOVERY_PREFLIGHT_STATUS, "recovery_preflight": pin,
            "evidence_file_count": len(evidence["captured_18_files"]),
            "verified_source_count": len(evidence["verified_prior_source_closure"]),
            "pair_id": evidence["prior_pair_id"]}


def _verify_pre_native_recovery_proof(
    proof_path: str | Path, *, session_dir: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = Path(proof_path)
    if path != RECOVERY_PROOF_PATH:
        raise CoordinatorError("coordinator recovery proof path is not the exact reviewed leaf")
    proof, _raw, pin = _read_self_json(path, "pre-native recovery proof", "recovery_sha256")
    if (proof.get("schema") != RECOVERY_PREFLIGHT_SCHEMA
            or proof.get("status") != RECOVERY_PREFLIGHT_STATUS
            or proof.get("failure_scope") != "PRE_NATIVE_CONTROL_FAILURE_ONLY"
            or proof.get("prior_session_id") != RECOVERY_SESSION_ID
            or Path(str(session_dir)).resolve(strict=True) != RECOVERY_SESSION_ROOT
            or proof.get("no_prior_failure_reinterpreted_as_success") is not True):
        raise CoordinatorError("recovery proof is not the exact prior pre-native control failure")
    current = _collect_pre_native_recovery_evidence(
        old_coordinator_dir=RECOVERY_CONTROL_ROOT,
        session_dir=RECOVERY_SESSION_ROOT,
        old_dispatch_root=RECOVERY_DISPATCH_ROOT)
    if any(proof.get(key) != value for key, value in current.items()):
        raise CoordinatorError("prior first-pair recovery evidence changed after preflight sealing")
    return proof, pin


def _verify_pre_native_recovery_pin_only(pin: Mapping[str, Any]) -> None:
    """Verify the already-qualified recovery receipt without replaying the old session."""
    path = Path(str(pin.get("path", "")))
    _value, _raw, actual = _read_self_json(path, "pre-native recovery proof", "recovery_sha256")
    if actual != dict(pin):
        raise CoordinatorError("sealed pre-native recovery proof changed")


def _write_exclusive(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise CoordinatorError("refusing to overwrite durable artifact: %s" % path) from exc
    try:
        with os.fdopen(fd, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_dir(path.parent)
    except Exception:
        raise


def _fsync_dir(path: Path) -> None:
    if os.name != "posix":
        return
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_json_exclusive(path: Path, value: Mapping[str, Any], field: str) -> dict[str, Any]:
    document = _seal(value, field)
    raw = (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    _write_exclusive(path, raw)
    return {"path": str(path.resolve(strict=True)), "bytes": len(raw),
            "sha256": _sha(raw), "self_sha256": document[field]}


def _source_closure(repo_root: Path) -> dict[str, dict[str, Any]]:
    observed: dict[str, dict[str, Any]] = {}
    for relative, expected in ((BOOTSTRAP_REL, BOOTSTRAP_PIN), (SESSION_REL, SESSION_PIN)):
        path = repo_root / relative
        pin = _file_pin(path)
        if (pin["bytes"], pin["sha256"]) != expected:
            raise CoordinatorError("frozen coordinator dependency changed: %s" % relative)
        observed[relative] = pin
    return observed


def _proc_start_ticks(pid: int, proc_root: Path = Path("/proc")) -> int | None:
    """Read Linux process starttime (stat field 22); None means unobservable."""
    stat_path = proc_root / str(pid) / "stat"
    try:
        raw = stat_path.read_text(encoding="ascii")
        # comm is parenthesized and may contain spaces or ')'; split after its final ')'.
        tail = raw[raw.rfind(")") + 2:].split()
        value = int(tail[19])  # fields after comm begin at field 3; field 22 is index 19.
        return value if value >= 0 else None
    except (OSError, ValueError, IndexError):
        return None


def _pid_matches(pid: Any, ticks: Any, proc_root: Path = Path("/proc")) -> bool:
    return (type(pid) is int and type(ticks) is int
            and _proc_start_ticks(pid, proc_root=proc_root) == ticks)


def _safe_directory(path: Path, *, create: bool) -> Path:
    if not path.is_absolute() or path.is_symlink():
        raise CoordinatorError("coordinator directory must be absolute and non-symlinked")
    if create:
        if path.exists():
            raise CoordinatorError("coordinator directory must be absent: %s" % path)
        path.mkdir(parents=False, exist_ok=False, mode=0o700)
        _fsync_dir(path.parent)
    elif not path.is_dir():
        raise CoordinatorError("coordinator directory is missing: %s" % path)
    return path.resolve(strict=True)


def _validate_dispatch_storage_root(value: str | Path, *, create: bool) -> Path:
    if not isinstance(value, (str, Path)):
        raise CoordinatorError("dispatch storage root must be a path")
    path = Path(value)
    if DISPATCH_MOUNT_ROOT.is_symlink() or not DISPATCH_MOUNT_ROOT.is_dir():
        raise CoordinatorError("dispatch mount root is missing or unsafe")
    expected_parent = DISPATCH_MOUNT_ROOT.resolve(strict=True)
    expected = expected_parent / DISPATCH_ROOT_NAME
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise CoordinatorError("dispatch storage root parent is missing or unsafe") from exc
    if (not path.is_absolute() or path.is_symlink()
            or path != expected or parent != expected_parent):
        raise CoordinatorError("dispatch storage root must be the dedicated /dev/shm execution namespace")
    if create:
        if path.exists():
            raise CoordinatorError("dispatch storage root must be absent for a fresh coordinator")
        path.mkdir(mode=0o700, exist_ok=False)
        _fsync_dir(path.parent)
        (path / "pairs").mkdir(mode=0o700, exist_ok=False)
        _fsync_dir(path)
    else:
        if path.is_symlink() or not path.is_dir():
            raise CoordinatorError("sealed dispatch storage root is missing or unsafe")
        pairs = path / "pairs"
        if pairs.is_symlink() or not pairs.is_dir():
            raise CoordinatorError("sealed dispatch storage pair namespace is missing or unsafe")
    resolved = path.resolve(strict=True)
    if resolved != expected:
        raise CoordinatorError("dispatch storage root resolves outside its exact namespace")
    return resolved


def _python_pin(executable: Path) -> dict[str, Any]:
    if not executable.is_absolute() or not executable.exists() or not executable.is_file():
        raise CoordinatorError("Python executable must be an absolute existing file")
    target = executable.resolve(strict=True)
    raw = target.read_bytes()
    return {"invoked_path": str(executable), "resolved_path": str(target),
            "bytes": len(raw), "sha256": _sha(raw)}


def _module_plain(value: Any) -> Any:
    # Reuse the session's canonical serializer when loaded; this fallback is for tests.
    if hasattr(value, "__dataclass_fields__"):
        return {key: _module_plain(getattr(value, key)) for key in value.__dataclass_fields__}
    if isinstance(value, Mapping):
        return {str(key): _module_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_module_plain(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        try:
            return _module_plain(value.item())
        except Exception:
            pass
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise CoordinatorError("proposal contains a non-canonical value: %s" % type(value).__name__)


def _read_child_json(raw: bytes) -> dict[str, Any] | None:
    if not raw:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception:
        return None
    return value if isinstance(value, dict) else None


def _tell_for_pair(session: Any, pair_id: str) -> dict[str, Any] | None:
    matches = [event for event in session.events
               if event.get("event_type") == "PAIR_TOLD" and event.get("pair_id") == pair_id]
    if len(matches) > 1:
        raise CoordinatorError("session replay has duplicate PAIR_TOLD events for a pair")
    return dict(matches[0]) if matches else None


def _readback_summary(session: Any, pair_id: str) -> dict[str, Any]:
    tell = _tell_for_pair(session, pair_id)
    pair = session.pending_pair
    pending = None
    if pair is not None:
        pending = {"pair_id": pair.pair_id, "iteration": pair.iteration,
                   "plus_proposal_id": pair.plus.proposal_id,
                   "minus_proposal_id": pair.minus.proposal_id}
    return {"session_id": session.manifest.get("session_id"),
            "controller_iteration": session.controller.iteration,
            "controller_halted": bool(session.halted),
            "pending_pair": pending,
            "event_count": len(session.events),
            "last_event_sha256": (session.events[-1].get("event_sha256") if session.events else None),
            "matching_pair_tell": tell}


class TwentyPairCoordinator:
    """Append-only pair supervisor. Child/session behavior is injectable for CPU tests."""

    def __init__(self, root: Path, manifest: Mapping[str, Any]):
        self.root = root
        self.manifest = dict(manifest)

    @classmethod
    def create(cls, *, coordinator_dir: str | Path, session_dir: str | Path,
               repo_root: str | Path, python_executable: str | Path,
               dispatch_root: str | Path,
               recovery_proof: str | Path) -> "TwentyPairCoordinator":
        coordinator_target = Path(coordinator_dir)
        if (coordinator_target != RECOVERY_NEW_COORDINATOR_ROOT
                or coordinator_target.exists() or coordinator_target.is_symlink()):
            raise CoordinatorError("recovery coordinator must use its exact fresh control namespace")
        repo = Path(repo_root).resolve(strict=True)
        session_input = Path(session_dir)
        if session_input.is_symlink():
            raise CoordinatorError("session root must not be a symlink")
        session_root = session_input.resolve(strict=True)
        if not session_root.is_dir():
            raise CoordinatorError("session root must be a real directory")
        _recovery_value, recovery_proof_pin = _verify_pre_native_recovery_proof(
            recovery_proof, session_dir=session_root)
        source_closure = _source_closure(repo)
        bootstrap = repo / BOOTSTRAP_REL
        runtime = _python_pin(Path(python_executable))
        session_manifest_path = session_root / "session_manifest.json"
        session_manifest, _raw, session_manifest_pin = _read_self_json(
            session_manifest_path, "durable session manifest", "manifest_sha256")
        if (session_manifest.get("schema") != SESSION_MANIFEST_SCHEMA
                or session_manifest.get("status") != "OPEN"):
            raise CoordinatorError("session manifest is not the sealed open SPSA session")
        dispatch_storage_root = _validate_dispatch_storage_root(dispatch_root, create=True)
        root = _safe_directory(Path(coordinator_dir), create=True)
        (root / "pairs").mkdir(mode=0o700)
        (root / "process_logs").mkdir(mode=0o700)
        _fsync_dir(root)
        manifest = _seal({
            "schema": COORDINATOR_SCHEMA,
            "status": "FROZEN_SOURCE_PINNED_COORDINATOR",
            "coordinator_id": "fit40_twenty_pair_" + uuid.uuid4().hex,
            "created_utc": _utc_now(),
            "repo_root": str(repo),
            "session_dir": str(session_root),
            "dispatch_storage_root": str(dispatch_storage_root),
            "session_id": session_manifest.get("session_id"),
            "session_manifest_pin": session_manifest_pin,
            "pre_native_recovery_proof_pin": recovery_proof_pin,
            "recovered_pending_pair_id": RECOVERY_PAIR_ID,
            "bootstrap_pin": source_closure[BOOTSTRAP_REL],
            "session_module_pin": source_closure[SESSION_REL],
            "coordinator_source_pin": _file_pin(Path(__file__).resolve(strict=True)),
            "python_runtime": runtime,
            "max_pairs": MAX_PAIRS,
            "launch_mode": "one_bootstrap_run_pair_child_per_pair",
            "science_run_started_at_manifest_seal": False,
        }, "coordinator_sha256")
        _write_json_exclusive(root / "coordinator_manifest.json", manifest, "coordinator_sha256")
        return cls(root, manifest)

    @classmethod
    def open(cls, coordinator_dir: str | Path) -> "TwentyPairCoordinator":
        root = _safe_directory(Path(coordinator_dir), create=False)
        value, _raw, _pin = _read_self_json(root / "coordinator_manifest.json",
                                           "coordinator manifest", "coordinator_sha256")
        if (value.get("schema") != COORDINATOR_SCHEMA
                or value.get("status") != "FROZEN_SOURCE_PINNED_COORDINATOR"
                or value.get("science_run_started_at_manifest_seal") is not False
                or value.get("max_pairs") != MAX_PAIRS):
            raise CoordinatorError("coordinator manifest schema/status/limit mismatch")
        observed_dispatch_root = _validate_dispatch_storage_root(
            value.get("dispatch_storage_root", ""), create=False)
        if str(observed_dispatch_root) != value.get("dispatch_storage_root"):
            raise CoordinatorError("coordinator manifest dispatch storage root is not canonical")
        return cls(root, value)

    def _verify_runtime_sources(self) -> tuple[Path, Path, Path]:
        repo = Path(self.manifest["repo_root"])
        closure = _source_closure(repo)
        if (closure[BOOTSTRAP_REL] != self.manifest.get("bootstrap_pin")
                or closure[SESSION_REL] != self.manifest.get("session_module_pin")
                or _file_pin(Path(__file__).resolve(strict=True)) != self.manifest.get("coordinator_source_pin")):
            raise CoordinatorError("source closure changed after coordinator creation")
        recovery_pin = self.manifest.get("pre_native_recovery_proof_pin")
        if (not isinstance(recovery_pin, Mapping)
                or self.manifest.get("recovered_pending_pair_id") != RECOVERY_PAIR_ID):
            raise CoordinatorError("coordinator does not bind the required pre-native recovery proof")
        _verify_pre_native_recovery_pin_only(recovery_pin)
        runtime = self.manifest.get("python_runtime")
        if not isinstance(runtime, Mapping):
            raise CoordinatorError("coordinator runtime pin is missing")
        executable = Path(str(runtime.get("invoked_path", "")))
        if _python_pin(executable) != dict(runtime):
            raise CoordinatorError("coordinator Python runtime changed")
        session = Path(self.manifest["session_dir"])
        session_value, _raw, session_pin = _read_self_json(
            session / "session_manifest.json", "durable session manifest", "manifest_sha256")
        if (session_value.get("session_id") != self.manifest.get("session_id")
                or session_pin != self.manifest.get("session_manifest_pin")):
            raise CoordinatorError("durable session manifest changed")
        dispatch_root = _validate_dispatch_storage_root(
            self.manifest.get("dispatch_storage_root", ""), create=False)
        if str(dispatch_root) != self.manifest.get("dispatch_storage_root"):
            raise CoordinatorError("sealed dispatch storage root changed")
        return repo, session, executable

    def _pair_dir(self, index: int) -> Path:
        return self.root / "pairs" / ("pair_%02d" % index)

    def _existing_prefix(self) -> tuple[int, str | None]:
        pair_root = self.root / "pairs"
        if pair_root.is_symlink() or not pair_root.is_dir():
            raise CoordinatorError("pair state directory is missing or unsafe")
        names = sorted(path.name for path in pair_root.iterdir())
        if any(not re.fullmatch(r"pair_[0-9]{2}", name) for name in names):
            raise CoordinatorError("unexpected entry in pair state directory")
        updated = 0
        unresolved: str | None = None
        for index, name in enumerate(names, 1):
            if name != "pair_%02d" % index:
                raise CoordinatorError("pair records are not a contiguous prefix")
            directory = pair_root / name
            if directory.is_symlink() or not directory.is_dir():
                raise CoordinatorError("pair record is not a real directory")
            intent_path, receipt_path = directory / "launch_intent.json", directory / "process_receipt.json"
            if not intent_path.is_file() or intent_path.is_symlink():
                raise CoordinatorError("pair directory has no valid durable intent")
            intent, _raw, _pin = _read_self_json(intent_path, "pair launch intent", "intent_sha256")
            if intent.get("pair_index") != index or intent.get("schema") != PAIR_INTENT_SCHEMA:
                raise CoordinatorError("pair launch intent identity mismatch")
            expected_dispatch_base = _validate_dispatch_storage_root(
                self.manifest.get("dispatch_storage_root", ""), create=False)
            expected_dispatch = (expected_dispatch_base / "pairs" / name / "dispatch_root")
            expected_packet = directory / "packet"
            expected_argv = [str(self.manifest["python_runtime"]["invoked_path"]),
                             str(Path(self.manifest["repo_root"]) / BOOTSTRAP_REL),
                             "--repo-root", self.manifest["repo_root"], "run-pair",
                             "--session-dir", self.manifest["session_dir"],
                             "--packet-dir", str(expected_packet),
                             "--dispatch-root", str(expected_dispatch)]
            if (intent.get("dispatch_storage_root") != str(expected_dispatch_base)
                    or intent.get("dispatch_root") != str(expected_dispatch)
                    or intent.get("packet_dir") != str(expected_packet)
                    or intent.get("argv") != expected_argv
                    or intent.get("pre_native_recovery_proof_pin")
                    != self.manifest.get("pre_native_recovery_proof_pin")):
                raise CoordinatorError("pair intent dispatch storage path differs from the sealed root/index")
            if not receipt_path.exists():
                readback_path = directory / "unknown_readback.json"
                if readback_path.exists():
                    readback, _raw, _pin = _read_self_json(
                        readback_path, "unknown pair readback", "readback_sha256")
                    if (readback.get("schema") != PAIR_READBACK_SCHEMA
                            or readback.get("status") != "UNKNOWN"
                            or readback.get("relaunch_prohibited") is not True):
                        raise CoordinatorError("existing unknown readback is malformed")
                unresolved = name
                break
            if expected_dispatch.is_symlink() or not expected_dispatch.is_dir():
                raise CoordinatorError("terminal pair receipt has no real dispatch output root")
            receipt, _raw, _pin = _read_self_json(receipt_path, "pair process receipt", "receipt_sha256")
            expected_intent_pin = {**_file_pin(intent_path), "self_sha256": intent["intent_sha256"]}
            if receipt.get("intent_pin") != expected_intent_pin:
                raise CoordinatorError("pair process receipt does not bind its exact launch intent")
            if (receipt.get("schema") != PAIR_RECEIPT_SCHEMA
                    or receipt.get("pair_index") != index
                    or receipt.get("pair_id") != intent.get("pair_id")
                    or receipt.get("iteration") != intent.get("iteration")
                    or type(receipt.get("child_exit_code")) not in (int, type(None))):
                raise CoordinatorError("pair process receipt schema or identity mismatch")
            start_pin = receipt.get("process_start_pin")
            if start_pin is not None:
                expected_start_path = (directory / "process_start.json").resolve(strict=True)
                if Path(str(start_pin.get("path", ""))).resolve(strict=True) != expected_start_path:
                    raise CoordinatorError("process start receipt is outside its immutable pair leaf")
                start, _start_raw, observed_start_pin = _read_self_json(
                    expected_start_path, "pair process start receipt", "start_sha256")
                if (observed_start_pin != dict(start_pin)
                        or start.get("schema") != PAIR_START_SCHEMA
                        or start.get("child_pid") != receipt.get("child_pid")
                        or start.get("child_start_ticks") != receipt.get("child_start_ticks")):
                    raise CoordinatorError("process receipt does not bind its parent-captured PID/start ticks")
            captured_raw: dict[str, bytes] = {}
            for role, leaf in (("stdout_pin", "child.stdout"), ("stderr_pin", "child.stderr")):
                output_pin = receipt.get(role)
                if not isinstance(output_pin, Mapping):
                    raise CoordinatorError("pair process receipt omits captured %s" % role)
                expected_output = directory / leaf
                if output_pin.get("path") is None:
                    if (receipt.get("status") != "UNKNOWN"
                            or expected_output.exists() or expected_output.is_symlink()
                            or output_pin.get("bytes") != 0
                            or output_pin.get("sha256") != _sha(b"")):
                        raise CoordinatorError("missing child output is only valid for an empty UNKNOWN receipt")
                    output_raw = b""
                else:
                    output_path = expected_output.resolve(strict=True)
                    if Path(str(output_pin.get("path", ""))).resolve(strict=True) != output_path:
                        raise CoordinatorError("captured child output is outside its immutable pair leaf")
                    output_raw = _regular_bytes(output_path, "captured child output")
                if (len(output_raw) != output_pin.get("bytes")
                        or _sha(output_raw) != output_pin.get("sha256")):
                    raise CoordinatorError("captured child output bytes changed")
                captured_raw[role] = output_raw
            parsed_child = _read_child_json(captured_raw["stdout_pin"])
            if parsed_child != receipt.get("child_result"):
                raise CoordinatorError("stored child result differs from its captured stdout bytes")
            if receipt.get("status") == "COMPLETE_UPDATED":
                if (type(receipt.get("child_exit_code")) is not int
                        or receipt["child_exit_code"] != 0
                        or type(receipt.get("child_pid")) is not int
                        or type(receipt.get("child_start_ticks")) is not int
                        or start_pin is None
                        or not isinstance(parsed_child, Mapping)
                        or parsed_child.get("status") != "PAIR_TOLD_UPDATED"
                        or parsed_child.get("tell_result", {}).get("status") != "UPDATED"):
                    raise CoordinatorError("UPDATED receipt lacks numeric child rc 0")
                updated += 1
            elif receipt.get("status") == "NO_UPDATE":
                if (receipt.get("child_exit_code") != 2 or start_pin is None
                        or type(receipt.get("child_pid")) is not int
                        or type(receipt.get("child_start_ticks")) is not int
                        or not isinstance(parsed_child, Mapping)
                        or parsed_child.get("status") != "PAIR_TOLD_NOT_UPDATED"
                        or parsed_child.get("tell_result", {}).get("status") != "NO_UPDATE"):
                    raise CoordinatorError("NO_UPDATE receipt lacks its exact terminal child evidence")
                return updated, name
            else:
                return updated, name
        return updated, unresolved

    def start_detached(self, *, python_executable: str | Path | None = None) -> dict[str, Any]:
        """Create start intent, launch this coordinator in a new session, then pin its PID."""
        if (self.root / "coordinator_start_intent.json").exists():
            raise CoordinatorError("detached start intent already exists; refusing duplicate start")
        repo, session, default_python = self._verify_runtime_sources()
        python = Path(python_executable).resolve(strict=True) if python_executable else default_python
        runtime = _python_pin(python)
        if runtime != self.manifest["python_runtime"]:
            raise CoordinatorError("detached Python runtime differs from the sealed coordinator runtime")
        argv = [str(python), str(Path(__file__).resolve(strict=True)), "run",
                "--repo-root", str(repo), "--session-dir", str(session),
                "--coordinator-dir", str(self.root),
                "--dispatch-root", str(self.manifest["dispatch_storage_root"])]
        source_pin = _file_pin(Path(__file__).resolve(strict=True))
        manifest_pin = _self_file_pin(self.root / "coordinator_manifest.json",
                                      "coordinator manifest", "coordinator_sha256")
        intent = _seal({"schema": START_INTENT_SCHEMA, "coordinator_id": self.manifest["coordinator_id"],
                        "created_utc": _utc_now(), "argv": argv,
                        "coordinator_source_pin": source_pin,
                        "coordinator_manifest_pin": manifest_pin,
                        "pre_native_recovery_proof_pin": self.manifest["pre_native_recovery_proof_pin"],
                        "python_runtime": runtime,
                        "dispatch_storage_root": self.manifest["dispatch_storage_root"]},
                       "intent_sha256")
        intent_pin = _write_json_exclusive(self.root / "coordinator_start_intent.json", intent,
                                           "intent_sha256")
        stdout_path, stderr_path = self.root / "process_logs" / "coordinator.stdout", self.root / "process_logs" / "coordinator.stderr"
        for path in (stdout_path, stderr_path):
            if path.exists() or path.is_symlink():
                raise CoordinatorError("detached log target already exists")
        started = _utc_now()
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                process = subprocess.Popen(argv, cwd=str(repo), stdin=subprocess.DEVNULL,
                                           stdout=stdout, stderr=stderr, close_fds=True,
                                           start_new_session=True)
                ticks = _proc_start_ticks(process.pid)
        except Exception as exc:
            receipt = {"schema": START_RECEIPT_SCHEMA, "status": "NOT_STARTED",
                       "coordinator_id": self.manifest["coordinator_id"],
                       "intent_pin": intent_pin, "started_utc": started,
                       "argv": argv, "coordinator_source_pin": source_pin,
                       "coordinator_manifest_pin": manifest_pin,
                       "pre_native_recovery_proof_pin": self.manifest["pre_native_recovery_proof_pin"],
                       "python_runtime": runtime,
                       "dispatch_storage_root": self.manifest["dispatch_storage_root"],
                       "child_pid": None, "child_start_ticks": None,
                       "error_type": type(exc).__name__, "error_text": str(exc)}
            _write_json_exclusive(self.root / "coordinator_start_receipt.json", receipt,
                                  "receipt_sha256")
            raise CoordinatorError("detached coordinator process did not start") from exc
        receipt = {"schema": START_RECEIPT_SCHEMA,
                   "status": "STARTED" if ticks is not None else "STARTED_TICKS_UNAVAILABLE",
                   "coordinator_id": self.manifest["coordinator_id"],
                   "intent_pin": intent_pin, "started_utc": started,
                   "argv": argv, "coordinator_source_pin": source_pin,
                   "coordinator_manifest_pin": manifest_pin,
                   "pre_native_recovery_proof_pin": self.manifest["pre_native_recovery_proof_pin"],
                   "python_runtime": runtime,
                   "dispatch_storage_root": self.manifest["dispatch_storage_root"],
                   "child_pid": process.pid, "child_start_ticks": ticks,
                   "stdout_path": str(stdout_path.resolve(strict=True)),
                   "stderr_path": str(stderr_path.resolve(strict=True))}
        _write_json_exclusive(self.root / "coordinator_start_receipt.json", receipt,
                              "receipt_sha256")
        return receipt

    def _load_session(self, session_loader: Callable[[], Any] | None = None) -> Any:
        if session_loader is not None:
            return session_loader()
        repo, session_dir, _python = self._verify_runtime_sources()
        bootstrap_path = repo / BOOTSTRAP_REL
        import importlib.util
        spec = importlib.util.spec_from_file_location("fit40_pinned_bootstrap_for_coordinator", bootstrap_path)
        if spec is None or spec.loader is None:
            raise CoordinatorError("cannot load the pinned bootstrap CLI")
        bootstrap = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bootstrap)
        modules = bootstrap._load_frozen_modules(repo)
        return modules["session"].DurableSPSASession.resume(session_dir=session_dir)

    def _make_pair_intent(self, *, index: int, pair: Any, session: Any,
                          repo: Path, session_dir: Path) -> dict[str, Any]:
        directory = self._pair_dir(index)
        if directory.exists() or directory.is_symlink():
            raise CoordinatorError("pair output leaf already exists")
        directory.mkdir(mode=0o700)
        _fsync_dir(directory)
        packet = directory / "packet"
        dispatch_base = Path(self.manifest["dispatch_storage_root"])
        dispatch = dispatch_base / "pairs" / ("pair_%02d" % index) / "dispatch_root"
        proposal_pair = {"pair_id": pair.pair_id, "iteration": pair.iteration,
                         "plus": _module_plain(pair.plus), "minus": _module_plain(pair.minus)}
        if session.pending_pair != pair:
            raise CoordinatorError("pair is not the exact durable pending ask")
        argv = [str(self.manifest["python_runtime"]["invoked_path"]),
                str(repo / BOOTSTRAP_REL), "--repo-root", str(repo), "run-pair",
                "--session-dir", str(session_dir), "--packet-dir", str(packet),
                "--dispatch-root", str(dispatch)]
        intent = {"schema": PAIR_INTENT_SCHEMA, "pair_index": index,
                  "coordinator_id": self.manifest["coordinator_id"],
                  "session_id": session.manifest.get("session_id"),
                  "iteration": pair.iteration, "pair_id": pair.pair_id,
                  "plus_proposal_id": pair.plus.proposal_id,
                  "minus_proposal_id": pair.minus.proposal_id,
                  "proposal_pair": proposal_pair,
                  "pair_sha256": _sha(_canonical(proposal_pair)),
                  "session_manifest_pin": _file_pin(session_dir / "session_manifest.json"),
                  "bootstrap_pin": _file_pin(repo / BOOTSTRAP_REL),
                  "coordinator_manifest_pin": _file_pin(self.root / "coordinator_manifest.json"),
                  "pre_native_recovery_proof_pin": self.manifest["pre_native_recovery_proof_pin"],
                  "argv": argv, "packet_dir": str(packet),
                  "dispatch_storage_root": str(dispatch_base),
                  "dispatch_root": str(dispatch),
                  "created_utc": _utc_now()}
        return _seal(intent, "intent_sha256")

    def _prepare_pair_dispatch_root(self, index: int) -> Path:
        base = _validate_dispatch_storage_root(
            self.manifest.get("dispatch_storage_root", ""), create=False)
        pairs_root = base / "pairs"
        if pairs_root.is_symlink() or not pairs_root.is_dir():
            raise CoordinatorError("dispatch pair namespace is missing or unsafe")
        pair_dir = pairs_root / ("pair_%02d" % index)
        if pair_dir.exists() or pair_dir.is_symlink():
            raise CoordinatorError("pair dispatch output leaf already exists")
        pair_dir.mkdir(mode=0o700, exist_ok=False)
        dispatch = pair_dir / "dispatch_root"
        dispatch.mkdir(mode=0o700, exist_ok=False)
        _fsync_dir(pair_dir)
        _fsync_dir(pairs_root)
        return dispatch.resolve(strict=True)

    def _child_command(self, argv: Sequence[str], directory: Path) -> dict[str, Any]:
        stdout_path, stderr_path = directory / "child.stdout", directory / "child.stderr"
        started = _utc_now()
        pid: int | None = None
        ticks: int | None = None
        return_code: int | None = None
        launch_pin: dict[str, Any] | None = None
        exception: dict[str, str] | None = None
        try:
            with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
                process = subprocess.Popen(list(argv), cwd=self.manifest["repo_root"],
                                           stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                           close_fds=True, start_new_session=True)
                pid = process.pid
                ticks = _proc_start_ticks(pid)
                launch = _seal({"schema": PAIR_START_SCHEMA, "status": "STARTED",
                                "coordinator_id": self.manifest["coordinator_id"],
                                "child_pid": pid, "child_start_ticks": ticks,
                                "started_utc": started, "argv": list(argv)}, "start_sha256")
                launch_pin = _write_json_exclusive(directory / "process_start.json", launch,
                                                   "start_sha256")
                return_code = process.wait()
        except Exception as exc:
            exception = {"type": type(exc).__name__, "message": str(exc)}
        if pid is not None and ticks is not None and _pid_matches(pid, ticks):
            return {"pid": pid, "start_ticks": ticks, "return_code": None,
                    "started_utc": started, "ended_utc": None, "launch_pin": launch_pin,
                    "stdout_path": str(stdout_path.resolve()) if stdout_path.exists() else None,
                    "stdout_bytes": None, "stdout_sha256": None,
                    "stderr_path": str(stderr_path.resolve()) if stderr_path.exists() else None,
                    "stderr_bytes": None, "stderr_sha256": None,
                    "stdout_json": None, "exception": exception, "running": True}
        ended = _utc_now()
        stdout_raw = stdout_path.read_bytes() if stdout_path.exists() and not stdout_path.is_symlink() else b""
        stderr_raw = stderr_path.read_bytes() if stderr_path.exists() and not stderr_path.is_symlink() else b""
        return {"pid": pid, "start_ticks": ticks, "return_code": return_code,
                "started_utc": started, "ended_utc": ended,
                "launch_pin": launch_pin,
                "stdout_path": str(stdout_path.resolve()) if stdout_path.exists() else None,
                "stdout_bytes": len(stdout_raw), "stdout_sha256": _sha(stdout_raw),
                "stderr_path": str(stderr_path.resolve()) if stderr_path.exists() else None,
                "stderr_bytes": len(stderr_raw), "stderr_sha256": _sha(stderr_raw),
                "stdout_json": _read_child_json(stdout_raw), "exception": exception,
                "running": False}

    def _unknown_from_existing_intent(self, index: int, session: Any) -> dict[str, Any]:
        directory = self._pair_dir(index)
        intent, _raw, intent_pin = _read_self_json(directory / "launch_intent.json",
                                                    "pair launch intent", "intent_sha256")
        start_path = directory / "process_start.json"
        if start_path.exists():
            start, _raw, start_pin = _read_self_json(start_path, "pair process start", "start_sha256")
            if (start.get("schema") != PAIR_START_SCHEMA
                    or start.get("status") != "STARTED"
                    or start.get("coordinator_id") != self.manifest["coordinator_id"]):
                raise CoordinatorError("existing process-start receipt identity is invalid")
            if start.get("child_pid") is not None and _pid_matches(
                    start.get("child_pid"), start.get("child_start_ticks")):
                return {"status": "RUNNING", "pair_index": index,
                        "pair_id": intent.get("pair_id"), "child_pid": start["child_pid"],
                        "child_start_ticks": start.get("child_start_ticks"),
                        "process_start_pin": start_pin, "relaunch_prohibited": True}
        readback = _readback_summary(session, str(intent.get("pair_id")))
        receipt_path = directory / "unknown_readback.json"
        body = {"schema": PAIR_READBACK_SCHEMA, "status": "UNKNOWN",
                "coordinator_id": self.manifest["coordinator_id"],
                "pair_index": index, "pair_id": intent.get("pair_id"),
                "intent_pin": intent_pin, "readback_utc": _utc_now(),
                "child_exit_code": None, "process_exit_unknown": True,
                "session_readback": readback, "relaunch_prohibited": True}
        pin = _write_json_exclusive(receipt_path, body, "readback_sha256")
        return {"status": "UNKNOWN", "pair_index": index,
                "pair_id": intent.get("pair_id"), "readback_pin": pin,
                "relaunch_prohibited": True}

    def run(self, *, session_loader: Callable[[], Any] | None = None,
            child_runner: Callable[[Sequence[str], Path], Mapping[str, Any]] | None = None,
            max_pairs: int = MAX_PAIRS) -> dict[str, Any]:
        if max_pairs != MAX_PAIRS:
            raise CoordinatorError("coordinator requires the full predeclared 20-pair limit")
        repo, session_dir, _python = self._verify_runtime_sources()
        session = self._load_session(session_loader)
        completed, unresolved = self._existing_prefix()
        if unresolved is not None:
            index = int(unresolved[-2:])
            if (self._pair_dir(index) / "process_receipt.json").exists():
                receipt, _raw, _pin = _read_self_json(self._pair_dir(index) / "process_receipt.json",
                                                      "pair process receipt", "receipt_sha256")
                return {"status": receipt.get("status"), "completed_pairs": completed,
                        "stop_pair_index": index, "reason": "prior terminal outcome is not UPDATED"}
            existing_readback = self._pair_dir(index) / "unknown_readback.json"
            if existing_readback.exists():
                readback, _raw, pin = _read_self_json(existing_readback,
                                                      "unknown pair readback", "readback_sha256")
                return {"status": readback.get("status"), "completed_pairs": completed,
                        "stop_pair_index": index, "readback_pin": pin,
                        "reason": "prior unresolved intent is permanently non-dispatchable"}
            result = self._unknown_from_existing_intent(index, session)
            return {"status": result["status"], "completed_pairs": completed,
                    "stop_pair_index": index, "readback": result}
        if session.controller.iteration != completed or session.halted or session.pending_pair is not None:
            raise CoordinatorError("session replay state does not match the exact completed coordinator prefix")
        # Cross-check every previously written COMPLETE receipt against the actual event-chain replay.
        tells = {event.get("pair_id"): event for event in session.events
                 if event.get("event_type") == "PAIR_TOLD"}
        for index in range(1, completed + 1):
            receipt, _raw, _pin = _read_self_json(self._pair_dir(index) / "process_receipt.json",
                                                   "pair process receipt", "receipt_sha256")
            event = tells.get(receipt.get("pair_id"))
            if (receipt.get("status") != "COMPLETE_UPDATED" or event is None
                    or event.get("pair_id") != receipt.get("pair_id")
                    or event.get("iteration") != receipt.get("iteration")
                    or event.get("tell_result", {}).get("status") != "UPDATED"):
                raise CoordinatorError("prior process receipt does not match deterministic session replay")
        results = []
        launcher = child_runner or self._child_command
        for index in range(completed + 1, MAX_PAIRS + 1):
            session = self._load_session(session_loader)
            if session.halted or session.controller.iteration != index - 1:
                raise CoordinatorError("session halted or iteration drifted before next pair")
            pair = session.pending_pair or session.ask_pair()
            if pair.iteration != index - 1:
                raise CoordinatorError("deterministic controller returned an unexpected pair iteration")
            directory = self._pair_dir(index)
            intent = self._make_pair_intent(index=index, pair=pair, session=session,
                                            repo=repo, session_dir=session_dir)
            intent_pin = _write_json_exclusive(directory / "launch_intent.json", intent,
                                               "intent_sha256")
            created_dispatch = self._prepare_pair_dispatch_root(index)
            if str(created_dispatch) != intent.get("dispatch_root"):
                raise CoordinatorError("created pair dispatch root differs from its durable intent")
            observed = dict(launcher(intent["argv"], directory))
            if observed.get("running") is True:
                return {"status": "RUNNING", "completed_pairs": index - 1,
                        "stop_pair_index": index, "child_pid": observed.get("pid"),
                        "child_start_ticks": observed.get("start_ticks"),
                        "reason": "child process is still alive; no duplicate launch"}
            session_after = self._load_session(session_loader)
            readback = _readback_summary(session_after, pair.pair_id)
            child_json = observed.get("stdout_json")
            tell = child_json.get("tell_result") if isinstance(child_json, Mapping) else None
            matching_tell = readback.get("matching_pair_tell")
            exact_updated = (
                type(observed.get("return_code")) is int
                and observed.get("return_code") == 0
                and type(observed.get("pid")) is int
                and type(observed.get("start_ticks")) is int
                and isinstance(child_json, Mapping)
                and child_json.get("status") == "PAIR_TOLD_UPDATED"
                and child_json.get("pair_id") == pair.pair_id
                and child_json.get("iteration") == pair.iteration
                and isinstance(tell, Mapping) and tell.get("status") == "UPDATED"
                and isinstance(matching_tell, Mapping)
                and matching_tell.get("iteration") == pair.iteration
                and matching_tell.get("tell_result", {}).get("status") == "UPDATED"
                and session_after.controller.iteration == index
                and session_after.pending_pair is None and not session_after.halted)
            no_update = (
                type(observed.get("return_code")) is int
                and observed.get("return_code") == 2
                and isinstance(child_json, Mapping)
                and child_json.get("status") == "PAIR_TOLD_NOT_UPDATED"
                and child_json.get("pair_id") == pair.pair_id
                and isinstance(tell, Mapping) and tell.get("status") == "NO_UPDATE"
                and isinstance(matching_tell, Mapping)
                and matching_tell.get("tell_result", {}).get("status") == "NO_UPDATE")
            status = "COMPLETE_UPDATED" if exact_updated else "NO_UPDATE" if no_update else "UNKNOWN"
            body = {"schema": PAIR_RECEIPT_SCHEMA, "status": status,
                    "coordinator_id": self.manifest["coordinator_id"],
                    "pair_index": index, "iteration": pair.iteration, "pair_id": pair.pair_id,
                    "plus_proposal_id": pair.plus.proposal_id,
                    "minus_proposal_id": pair.minus.proposal_id,
                    "intent_pin": intent_pin,
                    "process_start_pin": observed.get("launch_pin"),
                    "child_pid": observed.get("pid"),
                    "child_start_ticks": observed.get("start_ticks"),
                    "child_exit_code": observed.get("return_code"),
                    "started_utc": observed.get("started_utc"),
                    "ended_utc": observed.get("ended_utc"),
                    "argv": intent["argv"],
                    "stdout_pin": {"path": observed.get("stdout_path"),
                                   "bytes": observed.get("stdout_bytes"),
                                   "sha256": observed.get("stdout_sha256")},
                    "stderr_pin": {"path": observed.get("stderr_path"),
                                   "bytes": observed.get("stderr_bytes"),
                                   "sha256": observed.get("stderr_sha256")},
                    "child_result": child_json,
                    "session_readback": readback,
                    "exception": observed.get("exception"),
                    "relaunch_prohibited": True}
            receipt_pin = _write_json_exclusive(directory / "process_receipt.json", body,
                                                "receipt_sha256")
            results.append({"status": status, "pair_index": index, "pair_id": pair.pair_id,
                            "receipt_pin": receipt_pin})
            if status != "COMPLETE_UPDATED":
                return {"status": status, "completed_pairs": index - 1,
                        "stop_pair_index": index, "pairs": results}
        summary_path = self.root / "complete_summary.json"
        if summary_path.exists():
            summary, _raw, pin = _read_self_json(summary_path, "coordinator complete summary", "summary_sha256")
            if summary.get("status") != "COMPLETE_20_UPDATED" or summary.get("completed_pairs") != MAX_PAIRS:
                raise CoordinatorError("existing coordinator summary is not a valid complete record")
            return {"status": summary["status"], "completed_pairs": MAX_PAIRS,
                    "summary_pin": pin, "pairs": results}
        summary = {"schema": "fit40_spsa_twenty_pair_coordinator_summary_v1",
                   "status": "COMPLETE_20_UPDATED", "coordinator_id": self.manifest["coordinator_id"],
                   "completed_pairs": MAX_PAIRS,
                   "pair_receipt_pins": [item["receipt_pin"] for item in results],
                   "completed_utc": _utc_now()}
        summary_pin = _write_json_exclusive(self.root / "complete_summary.json", summary,
                                            "summary_sha256")
        return {"status": "COMPLETE_20_UPDATED", "completed_pairs": MAX_PAIRS,
                "summary_pin": summary_pin, "pairs": results}


def _start_detached(args: argparse.Namespace) -> int:
    coordinator = TwentyPairCoordinator.create(
        coordinator_dir=args.coordinator_dir, session_dir=args.session_dir,
        repo_root=args.repo_root, python_executable=args.python,
        dispatch_root=args.dispatch_root, recovery_proof=args.recovery_proof)
    receipt = coordinator.start_detached()
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0 if receipt.get("status") == "STARTED" else 2


def _seal_recovery_cli(args: argparse.Namespace) -> int:
    receipt = _seal_pre_native_recovery_preflight(
        old_coordinator_dir=args.old_coordinator_dir,
        session_dir=args.session_dir,
        old_dispatch_root=args.old_dispatch_root,
        proof_out=args.proof_out,
    )
    print(json.dumps(receipt, sort_keys=True, separators=(",", ":")))
    return 0


def _run_cli(args: argparse.Namespace) -> int:
    coordinator = TwentyPairCoordinator.open(args.coordinator_dir)
    if (Path(args.repo_root).resolve(strict=True) != Path(coordinator.manifest["repo_root"])
            or Path(args.session_dir).resolve(strict=True) != Path(coordinator.manifest["session_dir"])
            or str(_validate_dispatch_storage_root(args.dispatch_root, create=False))
            != coordinator.manifest.get("dispatch_storage_root")):
        raise CoordinatorError("run arguments differ from the sealed coordinator/session/dispatch roots")
    coordinator._verify_runtime_sources()
    # A detached child must match the parent-observed PID/start-tick receipt.
    start_path = coordinator.root / "coordinator_start_receipt.json"
    start_intent_path = coordinator.root / "coordinator_start_intent.json"
    deadline = time.monotonic() + 10.0
    while not start_path.is_file() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not start_path.is_file() or not start_intent_path.is_file():
        raise CoordinatorError("run requires a prior detached start receipt")
    start, _raw, _pin = _read_self_json(start_path, "coordinator start receipt", "receipt_sha256")
    intent, _raw, intent_pin = _read_self_json(
        start_intent_path, "coordinator start intent", "intent_sha256")
    manifest_pin = _self_file_pin(coordinator.root / "coordinator_manifest.json",
                                  "coordinator manifest", "coordinator_sha256")
    source_pin = _file_pin(Path(__file__).resolve(strict=True))
    runtime = coordinator.manifest.get("python_runtime")
    if not isinstance(runtime, Mapping):
        raise CoordinatorError("sealed coordinator Python runtime is missing")
    actual_runtime = _python_pin(Path(str(runtime.get("invoked_path", ""))))
    expected_argv = [str(actual_runtime["invoked_path"]),
                     str(Path(__file__).resolve(strict=True)), "run",
                     "--repo-root", str(Path(args.repo_root).resolve(strict=True)),
                     "--session-dir", str(Path(args.session_dir).resolve(strict=True)),
                     "--coordinator-dir", str(coordinator.root),
                     "--dispatch-root", str(coordinator.manifest["dispatch_storage_root"])]
    if (intent.get("schema") != START_INTENT_SCHEMA
            or intent.get("coordinator_id") != coordinator.manifest.get("coordinator_id")
            or intent.get("coordinator_id") != start.get("coordinator_id")
            or intent.get("coordinator_manifest_pin") != manifest_pin
            or intent.get("coordinator_source_pin") != source_pin
            or intent.get("pre_native_recovery_proof_pin")
            != coordinator.manifest.get("pre_native_recovery_proof_pin")
            or intent.get("python_runtime") != dict(runtime)
            or intent.get("dispatch_storage_root") != coordinator.manifest.get("dispatch_storage_root")
            or _python_pin(Path(str(runtime.get("invoked_path", "")))) != dict(runtime)
            or intent.get("argv") != expected_argv
            or start.get("schema") != START_RECEIPT_SCHEMA
            or start.get("status") != "STARTED"
            or start.get("coordinator_id") != intent.get("coordinator_id")
            or start.get("intent_pin") != intent_pin
            or start.get("coordinator_manifest_pin") != manifest_pin
            or start.get("coordinator_source_pin") != source_pin
            or start.get("pre_native_recovery_proof_pin")
            != coordinator.manifest.get("pre_native_recovery_proof_pin")
            or start.get("python_runtime") != dict(runtime)
            or start.get("dispatch_storage_root") != coordinator.manifest.get("dispatch_storage_root")
            or start.get("argv") != expected_argv
            or type(start.get("child_pid")) is not int
            or type(start.get("child_start_ticks")) is not int
            or start.get("child_pid") != os.getpid()
            or _proc_start_ticks(os.getpid()) != start.get("child_start_ticks")):
        raise CoordinatorError("run process does not match the parent-captured detached start receipt")
    result = coordinator.run()
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0 if result.get("status") == "COMPLETE_20_UPDATED" else 2


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    recovery = sub.add_parser("seal-recovery-preflight",
                              help="read-only verify the exact prior pre-native rc=2 control failure")
    recovery.add_argument("--old-coordinator-dir", type=Path, required=True)
    recovery.add_argument("--session-dir", type=Path, required=True)
    recovery.add_argument("--old-dispatch-root", type=Path, required=True)
    recovery.add_argument("--proof-out", type=Path, required=True)
    start = sub.add_parser("start-detached", help="create durable state and spawn the 20-pair coordinator")
    start.add_argument("--repo-root", type=Path, required=True)
    start.add_argument("--session-dir", type=Path, required=True)
    start.add_argument("--coordinator-dir", type=Path, required=True)
    start.add_argument("--dispatch-root", type=Path, required=True,
                       help="fresh /dev/shm root for pair model outputs; control files stay in coordinator-dir")
    start.add_argument("--recovery-proof", type=Path, required=True,
                       help="fresh self-pinned pre-native recovery receipt created by seal-recovery-preflight")
    start.add_argument("--python", type=Path, default=Path(sys.executable))
    run = sub.add_parser("run", help="continue only the parent-captured detached coordinator process")
    run.add_argument("--repo-root", type=Path, required=True)
    run.add_argument("--session-dir", type=Path, required=True)
    run.add_argument("--coordinator-dir", type=Path, required=True)
    run.add_argument("--dispatch-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "seal-recovery-preflight":
            return _seal_recovery_cli(args)
        if args.command == "start-detached":
            return _start_detached(args)
        if args.command == "run":
            return _run_cli(args)
        raise CoordinatorError("unsupported command")
    except Exception as exc:
        print("coordinator failed closed: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CoordinatorError", "TwentyPairCoordinator", "main"]
