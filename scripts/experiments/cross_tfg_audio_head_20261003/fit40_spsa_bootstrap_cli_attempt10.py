#!/usr/bin/env python3
"""Explicit entry point from a complete Fit40 handoff to durable SPSA.

This CLI never manufactures selector inputs. It verifies the collector08
terminal handoff, then exposes separate create-only steps for the initial
selector receipt and durable session. ``run-pair`` is an explicit one-pair
command; it is never started as a side effect of session creation or resume.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import stat
import sys
from typing import Any, Mapping, Sequence


PACKAGE = "scripts.experiments.cross_tfg_audio_head_20261003."
MODEL_ORDER = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
COLLECTOR_REL = "scripts/experiments/cross_tfg_audio_head_20261003/collect_fit40_initial_720_terminal_handoff_attempt08.py"
CENTER_DISPATCHER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_center_dispatcher_attempt07.py"
SOURCE_PINS = {
    COLLECTOR_REL: (74_897, "a83f4905b2765ba3637c3ac288be6e9729e5b39bc413032931ea90a83737a3a1"),
    "scripts/experiments/cross_tfg_audio_head_20261003/fit40_fixed_queue_controller_attempt03.py": (
        80_513, "dafe417e42cd7660b3b0c41731b30e10272f45ab4ead6911ea75c9f1fce1667a"),
    "scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_fixed_queue_attempt05.py": (
        25_969, "38e924bbccb54c04e9c7618502bdc57b38a71595babc71f3d9c51b646165de67"),
    "scripts/experiments/cross_tfg_audio_head_20261003/fit_only_stagea_ingest_attempt03.py": (
        53_184, "22cd7f983b22babcae0d9c120f4720cb84a218ec4d2b84551d5e1668d23aeb95"),
    "scripts/experiments/cross_tfg_audio_head_20261003/fit_only_final_center_stagea_ingest_attempt05.py": (
        106_167, "3384ebe67ed99569af6307c88746a0b16329897dda9935678677dcf80147e8ea"),
    "scripts/experiments/cross_tfg_audio_head_20261003/fit_only_spsa_selection_attempt06.py": (
        66_767, "dda610b985026be7f9328c479358e96c983a582c511964e8c6e72a6d398692e2"),
    "scripts/experiments/cross_tfg_audio_head_20261003/prepare_fit40_spsa_stagea_packets_attempt05.py": (
        47_748, "cd5e2b80ab0bdd4f599d863468c687724898b2e6baad451a5acfe2123648bdae"),
    "scripts/experiments/cross_tfg_audio_head_20261003/fit40_durable_spsa_session_attempt04.py": (
        89_950, "9e7c6fe51f8426196770f9171a61c37c6444690f902f83466b1e0d5a703a6709"),
    "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_point_dispatcher_attempt09.py": (
        109_249, "e63d51118a6e469a9731c1f3e23797808eb647f6a6609e4dabca1a4fbbe26ba5"),
    "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_point_execution_attempt07.py": (
        73_365, "d0e8f06c121ea52ca6ed695d8e5caf3fdd28a9a00edc1b1c35c648f9a9df2a2e"),
    CENTER_DISPATCHER_REL: (
        74_396, "39999903f12058091ff5e1488310e5219a0a00380c1dd0a704907ec929f17840"),
}
HANDOFF_SCHEMA = "fit40_initial_720_terminal_handoff_v1"
INGEST_SCHEMA = "fit40_stagea_selector_ingest_v1"
INITIAL_RECEIPT_SCHEMA = "fit40_initial_family_selector_receipt_v1"


class BootstrapError(RuntimeError):
    """A pinned input, terminal gate, or create-only operation failed."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    body = dict(value)
    body.pop(field, None)
    return _sha(_canonical(body))


def _read_regular(path: Path, label: str) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise BootstrapError(f"{label} must be an absolute non-symlink regular file: {path}")
    mode = path.stat(follow_symlinks=False).st_mode
    if not stat.S_ISREG(mode):
        raise BootstrapError(f"{label} is not a regular file: {path}")
    return path.read_bytes()


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular(path, label)
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise BootstrapError(f"{label} is not UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise BootstrapError(f"{label} must be a JSON object")
    return value, raw


def _check_pin(link: Mapping[str, Any], path: Path, raw: bytes, label: str,
               self_field: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink():
        raise BootstrapError(f"{label} path must be absolute and non-symlinked")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise BootstrapError(f"{label} path does not resolve") from exc
    observed = {"path": str(resolved), "bytes": len(raw), "sha256": _sha(raw)}
    if any(link.get(key) != observed[key] for key in observed):
        raise BootstrapError(f"{label} raw file pin mismatch")
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise BootstrapError(f"{label} JSON is malformed") from exc
    if not isinstance(value, dict) or link.get("self_sha256") != value.get(self_field):
        raise BootstrapError(f"{label} self-hash link is missing or mismatched")
    if _self_hash(value, self_field) != value.get(self_field):
        raise BootstrapError(f"{label} embedded self hash is invalid")
    return {**observed, "self_sha256": value[self_field]}


def _verify_source_closure(repo_root: Path) -> dict[str, dict[str, Any]]:
    root = repo_root.resolve(strict=True)
    if root.is_symlink() or not root.is_dir():
        raise BootstrapError("repository root must be a real directory")
    observed: dict[str, dict[str, Any]] = {}
    for relative, (expected_bytes, expected_sha) in SOURCE_PINS.items():
        path = root / relative
        raw = _read_regular(path, f"frozen source {relative}")
        pin = {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": _sha(raw)}
        if (pin["bytes"], pin["sha256"]) != (expected_bytes, expected_sha):
            raise BootstrapError(f"frozen source pin mismatch: {relative}")
        observed[relative] = pin
    return observed


def _verify_handoff_documents(
    handoff: Mapping[str, Any], index: Mapping[str, Any], family: Mapping[str, Any],
    *, handoff_pin: Mapping[str, Any], index_pin: Mapping[str, Any],
    family_pin: Mapping[str, Any], collector_pin: Mapping[str, Any],
) -> None:
    if (handoff.get("schema") != HANDOFF_SCHEMA
            or handoff.get("status") != "PASS_FULL_720_TERMINAL_READBACK_READY_FOR_SELECTION_INGEST"
            or handoff.get("selection_performed") is not False
            or handoff.get("scientific_initialization_performed") is not False
            or handoff.get("expected_rows") != 720 or handoff.get("verified_rows") != 720
            or handoff.get("rows_by_model") != {model: 240 for model in MODEL_ORDER}
            or handoff.get("scientific_input_pins_sha256") != index.get("scientific_input_pins_sha256")):
        raise BootstrapError("terminal handoff is not the exact complete 720-row preselection record")
    expected_index = handoff.get("initial_ingest_index")
    expected_family = handoff.get("composed_family_plan")
    if not isinstance(expected_index, Mapping) or not isinstance(expected_family, Mapping):
        raise BootstrapError("terminal handoff omits its actual ingest index or composed family pin")
    if any(expected_index.get(key) != index_pin.get(key) for key in ("path", "bytes", "sha256", "self_sha256")):
        raise BootstrapError("terminal handoff does not bind the supplied actual ingest index")
    if any(expected_family.get(key) != family_pin.get(key) for key in ("path", "bytes", "sha256", "self_sha256")):
        raise BootstrapError("terminal handoff does not bind its composed family plan")
    if (index.get("schema") != INGEST_SCHEMA
            or index.get("status") != "READY_FOR_LOCAL_INGEST_NOT_SELECTION"
            or index.get("expected_row_count") != 720
            or index.get("model_ids") != list(MODEL_ORDER)
            or len(index.get("sample_ids", [])) != 40
            or len(index.get("models", [])) != 3
            or len(index.get("raw_matrix_npz_file_ids", {})) != 720
            or len(index.get("official_source_frame_map_file_ids", {})) != 720
            or index.get("scientific_input_pins_sha256") != family.get("scientific_input_pins_sha256")):
        raise BootstrapError("initial ingest index is not the collector's complete 720-row packet")
    actual = handoff.get("actual_queue")
    historical = handoff.get("historical_prefix")
    proofs = handoff.get("source_block_proofs")
    if (not isinstance(actual, Mapping) or not isinstance(historical, Mapping)
            or not isinstance(proofs, list) or len(proofs) != 120
            or historical.get("blocks") != 34
            or historical.get("child_exit_code_available") is not False
            or historical.get("child_exit_code") is not None
            or actual.get("new_blocks") != 86 or actual.get("new_rows") != 516
            or actual.get("process_rc_required") != 0
            or actual.get("process_rc_verified_count") != 86
            or actual.get("paused_models") not in ([], {}) or actual.get("active_claim") is not None):
        raise BootstrapError("handoff queue terminal accounting is incomplete or paused")
    origins = [item.get("proof_origin") for item in proofs if isinstance(item, Mapping)]
    if origins.count("HISTORICAL_PREFIX_TERMINAL") != 34 or origins.count("ACTUAL_QUEUE_CHILD_RC") != 86:
        raise BootstrapError("historical and live block proof scopes do not total 34 + 86")
    if any(item.get("historical_exit_code_available") is not False
           or item.get("historical_exit_code") is not None
           for item in proofs if isinstance(item, Mapping)
           and item.get("proof_origin") == "HISTORICAL_PREFIX_TERMINAL"):
        raise BootstrapError("historical terminal proofs must retain unavailable child return codes")
    if any(type(item.get("child_exit_code")) is not int or item.get("child_exit_code") != 0
           for item in proofs if isinstance(item, Mapping)
           and item.get("proof_origin") == "ACTUAL_QUEUE_CHILD_RC"):
        raise BootstrapError("a live queue block does not have a verified zero child exit code")
    source_pins = handoff.get("source_code_pins")
    if not isinstance(source_pins, list):
        raise BootstrapError("handoff source-code pin table is absent")
    matches = [item for item in source_pins if isinstance(item, Mapping)
               and item.get("path") == collector_pin.get("path")]
    if (len(matches) != 1 or matches[0].get("bytes") != collector_pin.get("bytes")
            or matches[0].get("sha256") != collector_pin.get("sha256")):
        raise BootstrapError("handoff does not bind the reviewed collector08 source bytes")
    if (handoff_pin.get("bytes", 0) <= 0 or family_pin.get("bytes", 0) <= 0
            or index_pin.get("bytes", 0) <= 0):
        raise BootstrapError("handoff/index/family pin has an empty raw file")


def verify_terminal_handoff(handoff_path: str | Path, *, repo_root: str | Path) -> dict[str, Any]:
    """Read and validate a collector08 handoff and its exact index/family files."""
    root = Path(repo_root).resolve(strict=True)
    closure = _verify_source_closure(root)
    collector_pin = closure[COLLECTOR_REL]
    handoff_file = Path(handoff_path)
    handoff, handoff_raw = _read_json(handoff_file, "terminal handoff")
    handoff_self = handoff.get("handoff_sha256")
    if handoff_self != _self_hash(handoff, "handoff_sha256"):
        raise BootstrapError("terminal handoff self hash mismatch")
    handoff_pin = {"path": str(handoff_file.resolve(strict=True)),
                   "bytes": len(handoff_raw), "sha256": _sha(handoff_raw),
                   "self_sha256": handoff_self}
    index_ref = handoff.get("initial_ingest_index")
    family_ref = handoff.get("composed_family_plan")
    if not isinstance(index_ref, Mapping) or not isinstance(family_ref, Mapping):
        raise BootstrapError("terminal handoff omits pinned index/family source documents")
    index_path, family_path = Path(str(index_ref.get("path", ""))), Path(str(family_ref.get("path", "")))
    index, index_raw = _read_json(index_path, "initial ingest index")
    family, family_raw = _read_json(family_path, "composed family plan")
    index_pin = _check_pin(index_ref, index_path, index_raw, "initial ingest index", "index_sha256")
    family_pin = _check_pin(family_ref, family_path, family_raw, "composed family plan", "plan_sha256")
    _verify_handoff_documents(handoff, index, family, handoff_pin=handoff_pin,
                              index_pin=index_pin, family_pin=family_pin,
                              collector_pin=collector_pin)
    return {"handoff": handoff, "handoff_pin": handoff_pin,
            "index": index, "index_pin": index_pin,
            "family": family, "family_pin": family_pin,
            "source_closure": closure}


def _load_frozen_modules(repo_root: Path) -> dict[str, Any]:
    _verify_source_closure(repo_root)
    root_text = str(repo_root.resolve(strict=True))
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    modules = {
        "collector": importlib.import_module(PACKAGE + "collect_fit40_initial_720_terminal_handoff_attempt08"),
        "session": importlib.import_module(PACKAGE + "fit40_durable_spsa_session_attempt04"),
        "packet_builder": importlib.import_module(PACKAGE + "prepare_fit40_spsa_stagea_packets_attempt05"),
        "dispatcher": importlib.import_module(PACKAGE + "fit40_spsa_point_dispatcher_attempt09"),
    }
    closure = _verify_source_closure(repo_root)
    for role, relative in (("collector", COLLECTOR_REL),
                           ("session", "scripts/experiments/cross_tfg_audio_head_20261003/fit40_durable_spsa_session_attempt04.py"),
                           ("packet_builder", "scripts/experiments/cross_tfg_audio_head_20261003/prepare_fit40_spsa_stagea_packets_attempt05.py"),
                           ("dispatcher", "scripts/experiments/cross_tfg_audio_head_20261003/fit40_spsa_point_dispatcher_attempt09.py")):
        if Path(modules[role].__file__).resolve(strict=True) != Path(closure[relative]["path"]):
            raise BootstrapError(f"loaded {role} module origin differs from the pinned source closure")
    return modules


def _receipt_matches_terminal_inputs(
    receipt_path: Path, binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    receipt, raw = _read_json(receipt_path, "initial selector receipt")
    self_sha = receipt.get("receipt_sha256")
    if (receipt.get("schema") != INITIAL_RECEIPT_SCHEMA
            or receipt.get("status") != "INITIAL_720_SELECTOR_COMPLETE"
            or receipt.get("metric_backend") != "torch_cpu"
            or self_sha != _self_hash(receipt, "receipt_sha256")):
        raise BootstrapError("initial selector receipt schema/status/backend/self hash mismatch")
    for key, expected in (("initial_ingest_index", binding["index_pin"]),
                          ("composed_family_plan", binding["family_pin"])):
        actual = receipt.get(key)
        if not isinstance(actual, Mapping) or any(
            actual.get(field) != expected.get(field)
            for field in ("path", "bytes", "sha256", "self_sha256")
        ):
            raise BootstrapError(f"initial selector receipt has a different {key} pin")
    if receipt.get("scientific_input_pins_sha256") != binding["family"].get("scientific_input_pins_sha256"):
        raise BootstrapError("initial selector receipt input digest differs from the composed family")
    path = receipt_path.resolve(strict=True)
    pin = {"path": str(path), "bytes": len(raw), "sha256": _sha(raw), "self_sha256": self_sha}
    return receipt, pin


def _verify_execution_package(path: Path, family_pin: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    package, raw = _read_json(path, "execution package manifest")
    self_sha = package.get("manifest_sha256")
    if (package.get("schema") != "fit40_spsa_point_execution_package_manifest_v1"
            or package.get("status") != "FROZEN"
            or self_sha != _self_hash(package, "manifest_sha256")
            or package.get("composed_family_plan_pin") != dict(family_pin)):
        raise BootstrapError("execution package is not a self-pinned frozen package for this family")
    models = package.get("execution_sources_by_model")
    if not isinstance(models, Mapping) or set(models) != set(MODEL_ORDER):
        raise BootstrapError("execution package does not bind the exact three-model roster")
    pin = {"path": str(path.resolve(strict=True)), "bytes": len(raw),
           "sha256": _sha(raw), "self_sha256": self_sha}
    return package, pin


def _pin_for_path(path: Path) -> dict[str, Any]:
    raw = _read_regular(path, "output file")
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise BootstrapError("output receipt must be a JSON object")
    field = next((name for name in ("receipt_sha256", "manifest_sha256", "journal_sha256")
                  if name in value), None)
    if field is None:
        raise BootstrapError("output JSON lacks its recognized self-hash field")
    if value.get(field) != _self_hash(value, field):
        raise BootstrapError("output JSON self hash is invalid")
    return {"path": str(path.resolve(strict=True)), "bytes": len(raw),
            "sha256": _sha(raw), "self_sha256": value[field]}


def _source_pin(path: Path) -> dict[str, Any]:
    raw = _read_regular(path, "bootstrap CLI source")
    return {"path": str(path.resolve(strict=True)), "bytes": len(raw), "sha256": _sha(raw)}


def _cli_result(value: Mapping[str, Any]) -> None:
    print(json.dumps(dict(value), sort_keys=True, ensure_ascii=False, separators=(",", ":")))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal-initial-selector", help="run the full-720 torch_cpu selector and write its receipt")
    seal.add_argument("--terminal-handoff", type=Path, required=True)
    seal.add_argument("--receipt-out", type=Path, required=True)
    create = sub.add_parser("create-session", help="initialize a durable session from pinned full-720 inputs")
    create.add_argument("--terminal-handoff", type=Path, required=True)
    create.add_argument("--initial-selector-receipt", type=Path, required=True)
    create.add_argument("--execution-package", type=Path, required=True)
    create.add_argument("--session-dir", type=Path, required=True)
    create.add_argument("--session-id")
    resume = sub.add_parser("resume-verify", help="read-only deterministic replay of an existing session")
    resume.add_argument("--session-dir", type=Path, required=True)
    journal = sub.add_parser("seal-journal", help="create-only seal the actual complete 20-pair controller journal")
    journal.add_argument("--session-dir", type=Path, required=True)
    journal.add_argument("--journal-out", type=Path, required=True)
    run = sub.add_parser("run-pair", help="explicitly execute exactly one persisted plus/minus proposal pair")
    run.add_argument("--session-dir", type=Path, required=True)
    run.add_argument("--packet-dir", type=Path, required=True,
                     help="fresh pair packet leaf, or its existing immutable packet for readback")
    run.add_argument("--dispatch-root", type=Path, required=True,
                     help="pre-created local output root for model child receipts")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        repo_root = args.repo_root.resolve(strict=True)
        modules = _load_frozen_modules(repo_root)
        if args.command == "seal-initial-selector":
            binding = verify_terminal_handoff(args.terminal_handoff, repo_root=repo_root)
            receipt_pin = modules["session"].produce_initial_selector_receipt(
                binding["index_pin"]["path"], binding["family_pin"]["path"], args.receipt_out,
            )
            _cli_result({"status": "INITIAL_720_SELECTOR_COMPLETE",
                         "terminal_handoff": binding["handoff_pin"],
                         "initial_ingest_index": binding["index_pin"],
                         "composed_family_plan": binding["family_pin"],
                         "initial_selector_receipt": receipt_pin,
                         "bootstrap_cli": _source_pin(Path(__file__))})
            return 0
        if args.command == "create-session":
            binding = verify_terminal_handoff(args.terminal_handoff, repo_root=repo_root)
            receipt, receipt_pin = _receipt_matches_terminal_inputs(
                args.initial_selector_receipt, binding)
            package, package_pin = _verify_execution_package(
                args.execution_package, binding["family_pin"])
            session = modules["session"].DurableSPSASession.create(
                session_dir=args.session_dir,
                initial_index_path=binding["index_pin"]["path"],
                initial_receipt_path=receipt_pin["path"],
                composed_plan_path=binding["family_pin"]["path"],
                execution_package_path=package_pin["path"],
                session_id=args.session_id,
            )
            manifest_pin = _pin_for_path(session.root / "session_manifest.json")
            _cli_result({"status": "DURABLE_SESSION_OPEN",
                         "session_id": session.manifest["session_id"],
                         "terminal_handoff": binding["handoff_pin"],
                         "initial_selector_receipt": receipt_pin,
                         "execution_package": package_pin,
                         "session_manifest": manifest_pin,
                         "bootstrap_cli": _source_pin(Path(__file__))})
            return 0
        if args.command == "resume-verify":
            session = modules["session"].DurableSPSASession.resume(session_dir=args.session_dir)
            _cli_result({"status": "DURABLE_SESSION_REPLAYED_READ_ONLY",
                         "session_id": session.manifest["session_id"],
                         "event_count": len(session.events),
                         "pending_pair_id": (session.pending_pair.pair_id
                                             if session.pending_pair is not None else None),
                         "halted": session.halted,
                         "session_manifest": _pin_for_path(session.root / "session_manifest.json"),
                         "bootstrap_cli": _source_pin(Path(__file__))})
            return 0
        if args.command == "seal-journal":
            session_module = modules["session"]
            session = session_module.DurableSPSASession.resume(session_dir=args.session_dir)
            document = session.controller_journal_document()
            entries = document.get("entries") if isinstance(document, Mapping) else None
            if (not isinstance(document, Mapping)
                    or document.get("schema") != session_module.JOURNAL_SCHEMA
                    or document.get("status") != "COMPLETE_20_ASK_TELL_PAIRS"
                    or not isinstance(entries, list) or len(entries) != 20
                    or any(not isinstance(entry, Mapping)
                           or not isinstance(entry.get("tell_result"), Mapping)
                           or entry["tell_result"].get("status") != "UPDATED"
                           for entry in entries)):
                raise BootstrapError("session replay is not an actual complete 20-pair UPDATED journal")
            journal_pin = session.write_controller_journal_exclusive(args.journal_out)
            observed_pin = _pin_for_path(args.journal_out)
            if any(journal_pin.get(key) != observed_pin.get(key)
                   for key in ("path", "bytes", "sha256", "self_sha256")):
                raise BootstrapError("created controller journal differs from its create-only writer receipt")
            _cli_result({"status": "COMPLETE_20_ASK_TELL_PAIRS",
                         "session_id": session.manifest["session_id"],
                         "event_count": len(session.events),
                         "entry_count": len(entries),
                         "controller_journal": observed_pin,
                         "session_manifest": _pin_for_path(session.root / "session_manifest.json"),
                         "bootstrap_cli": _source_pin(Path(__file__))})
            return 0
        if args.command == "run-pair":
            session_module = modules["session"]
            session = session_module.DurableSPSASession.resume(session_dir=args.session_dir)
            pair = session.ask_pair()
            packet_dir = args.packet_dir
            packet_index = packet_dir / "packet_index.json"
            if packet_dir.exists() or packet_dir.is_symlink():
                if packet_dir.is_symlink() or not packet_dir.is_dir() or packet_index.is_symlink() or not packet_index.is_file():
                    raise BootstrapError("existing proposal packet is partial or unsafe; refusing to rebuild it")
            else:
                if any((proposal.proposal_id, proposal.side) in session.dispatch_intents
                       for proposal in (pair.plus, pair.minus)):
                    raise BootstrapError("persisted dispatch intent has no packet to read back; outcome is UNKNOWN")
                modules["packet_builder"].prepare_pair_packets(
                    base_plan_path=session.manifest["composed_family_plan"]["path"],
                    expected_base_plan_sha256=session.manifest["composed_family_plan"]["sha256"],
                    family_selection=session.inputs.family_selection,
                    controller=session.controller, pair=pair, output_dir=packet_dir,
                )
                if packet_index.is_symlink() or not packet_index.is_file():
                    raise BootstrapError("composer did not produce its exact pair packet index")
            dispatcher = modules["dispatcher"].PointBlockDispatcher(
                family_plan_pin=session.manifest["composed_family_plan"],
                initial_selector_receipt_pin=session.manifest["initial_selector_receipt"],
                execution_package_manifest_pin=session.manifest["execution_package_manifest"],
                dispatch_root=args.dispatch_root,
                point_outcome_factory=session_module.PointOutcome,
            )
            dispatcher.bind_pair_packet(pair, packet_index)
            tell = session.run_pending_pair(dispatcher)
            tell_status = getattr(tell, "status", None)
            updated = tell_status == "UPDATED"
            _cli_result({"status": "PAIR_TOLD_UPDATED" if updated else "PAIR_TOLD_NOT_UPDATED",
                         "session_id": session.manifest["session_id"],
                         "iteration": pair.iteration, "pair_id": pair.pair_id,
                         "tell_result": modules["session"]._plain(tell),
                         "event_count": len(session.events),
                         "bootstrap_cli": _source_pin(Path(__file__))})
            return 0 if updated else 2
        raise BootstrapError("unsupported command")
    except Exception as exc:
        print(f"bootstrap failed closed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
