"""Create the initial 720-row selector receipt and durable SPSA session log.

This module owns persistence/replay only. It does not render, score, train, or
select on partial data. A point executor is injected by the native dispatcher;
the durable wrapper writes the exact ask before dispatch and treats an
ambiguous readback as non-dispatchable.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import time
from io import BytesIO
from typing import Any, Callable, Mapping, Protocol, Sequence
import uuid

from scripts.experiments.cross_tfg_audio_head_20261003 import (
    fit40_native_point_score_adapter_attempt02 as native_point_score_adapter,
    fit_only_final_center_stagea_ingest_attempt07 as final_ingest,
    fit_only_spsa_selection_attempt06 as selector,
    fit_only_stagea_ingest_attempt03 as initial_ingest,
    prepare_fit40_spsa_stagea_packets_attempt05 as packet_builder,
)


INITIAL_RECEIPT_SCHEMA = "fit40_initial_family_selector_receipt_v1"
SESSION_SCHEMA = "fit40_durable_spsa_session_v2"
SESSION_MANIFEST_SCHEMA = "fit40_durable_spsa_session_manifest_v2"
EVENT_SCHEMA = "fit40_durable_spsa_event_v2"
POINT_ENTRY_SCHEMA = "fit40_durable_point_entry_v1"
JOURNAL_SCHEMA = "fit40_spsa_controller_journal_v1"
POINT_OUTCOMES = {"COMPLETE", "COMPLETE_WITH_FAILURES", "INCOMPLETE", "FAILED_BEFORE_RENDER"}
OFFICIAL_ARRAY_DTYPES = {
    "query_frame_index": "<i4",
    "query_timestamp_num": "<i8",
    "query_timestamp_den": "<i8",
    "track_frame_index": "<i4",
    "valid_query_mask": "|b1",
    "distance_by_query_lag": "<f4",
    "lag_offsets_frames": "<i4",
}


class DurableSessionError(RuntimeError):
    """A persistence, lineage, or replay invariant failed closed."""


class UnknownPointOutcome(DurableSessionError):
    """A launched point has no conclusive readback; it must not be relaunched."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    from hashlib import sha256
    return sha256(raw).hexdigest()


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes on POSIX; Windows has no directory fsync."""
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _hash_object(value: Mapping[str, Any], field: str, label: str) -> str:
    body = dict(value)
    declared = body.pop(field, None)
    actual = _sha(_canonical(body))
    if declared != actual:
        raise DurableSessionError(f"{label} {field} mismatch")
    return actual


def _plain(value: Any) -> Any:
    return final_ingest._plain(value)


def _regular_pin(path: str | Path, *, declared_path: str | None = None,
                 file_id: str, role: str) -> initial_ingest._PinnedFile:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise DurableSessionError(f"{role} must be a non-symlink regular file: {target}")
    resolved = target.resolve(strict=True)
    raw = resolved.read_bytes()
    return initial_ingest._PinnedFile(
        file_id, role, declared_path or str(resolved), str(resolved), len(raw), _sha(raw))


def _pin_link(pin: initial_ingest._PinnedFile, self_sha256: str) -> dict[str, Any]:
    return {"path": pin.declared_path, "bytes": pin.byte_count,
            "sha256": pin.sha256, "self_sha256": self_sha256}


def _write_exclusive(path: Path, raw: bytes) -> bool:
    """Create a file once; return False only for exact-byte idempotent recovery."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError:
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise DurableSessionError(f"refusing to replace non-identical durable file: {path}")
        return False
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(path.parent)
    except Exception:
        try:
            if path.exists() or path.is_symlink():
                path.unlink(missing_ok=True)
                _fsync_directory(path.parent)
        finally:
            raise
    return True


def _seal_document(document: Mapping[str, Any], field: str) -> dict[str, Any]:
    sealed = dict(document)
    sealed[field] = _sha(_canonical(sealed))
    return sealed


def _read_json(path: str | Path, label: str) -> tuple[dict[str, Any], bytes]:
    target = Path(path)
    if target.is_symlink() or not target.is_file():
        raise DurableSessionError(f"{label} must be a non-symlink regular file: {target}")
    raw = target.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:
        raise DurableSessionError(f"{label} is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise DurableSessionError(f"{label} must be a JSON object")
    return value, raw


def _file_entry(pin: initial_ingest._PinnedFile) -> dict[str, Any]:
    return {"file_id": pin.file_id, "role": pin.role,
            "declared_path": pin.declared_path, "local_path": pin.local_path,
            "bytes": pin.byte_count, "sha256": pin.sha256}


def _package_file_table(package: Mapping[str, Any], *,
                        fixed_pins: Sequence[initial_ingest._PinnedFile]) -> dict[str, initial_ingest._PinnedFile]:
    pins = list(fixed_pins)
    seen = {pin.file_id for pin in pins}
    source = package.get("execution_sources_by_model")
    if not isinstance(source, Mapping):
        raise DurableSessionError("execution package lacks its per-model source closure")
    external: list[tuple[str, Mapping[str, Any]]] = [
        ("shared_helper", package.get("shared_application_helper_pin")),
        ("point_runner", package.get("point_runner_pin")),
    ]
    for model_id, values in source.items():
        if not isinstance(values, Mapping):
            raise DurableSessionError(f"execution package source closure malformed for {model_id}")
        for role in ("launcher_pin", "runner_pin", "adapter_pin", "runtime_binding_pin"):
            external.append((f"{model_id}_{role}", values.get(role)))
        source_pins = values.get("source_pins")
        if not isinstance(source_pins, Mapping) or not source_pins:
            raise DurableSessionError(f"execution package source pins missing for {model_id}")
        for role, pin in source_pins.items():
            external.append((f"{model_id}_source_{role}", pin))
    for label, value in external:
        if not isinstance(value, Mapping):
            raise DurableSessionError(f"execution package pin is malformed: {label}")
        try:
            declared_path = value["path"]
            local = Path(declared_path)
            if not local.is_absolute() or local.is_symlink() or not local.is_file():
                raise DurableSessionError(f"execution package source path is absent/unsafe: {declared_path}")
            raw = local.read_bytes()
        except Exception as exc:
            if isinstance(exc, DurableSessionError):
                raise
            raise DurableSessionError(f"cannot read execution package source {label}: {exc}") from exc
        if (len(raw) != value.get("bytes") or _sha(raw) != value.get("sha256")):
            raise DurableSessionError(f"execution package source pin changed: {label}")
        file_id = f"exec-{_sha(label.encode())[:12]}-{value['sha256'][:16]}"
        if file_id in seen:
            continue
        seen.add(file_id)
        pins.append(initial_ingest._PinnedFile(
            file_id, label, declared_path, str(local.resolve(strict=True)), len(raw), _sha(raw)))
    return {pin.file_id: pin for pin in pins}


def produce_initial_selector_receipt(
    initial_index_path: str | Path,
    composed_family_plan_path: str | Path,
    receipt_path: str | Path,
) -> dict[str, Any]:
    """Re-evaluate the complete official 720 rows, then write the initial receipt.

    A complete but ``NOT_ADMITTED`` selector result is valid only when the
    selector's exact full-Fit fallback seed is serialized in the receipt.
    """
    index_pin = _regular_pin(initial_index_path, file_id="initial_ingest_index",
                             role="initial_ingest_index")
    index, index_raw = _read_json(index_pin.local_path, "initial ingest index")
    if _sha(index_raw) != index_pin.sha256:
        raise DurableSessionError("initial ingest index changed during receipt creation")
    index_self = final_ingest._hash_object(index, "index_sha256", "initial ingest index")
    nested_files = initial_ingest._file_table(index)
    final_ingest._verify_initial_preselector(index, nested_files)
    family_selection = initial_ingest.ingest_family(index_pin.local_path, metric_backend="torch_cpu")
    family = family_selection.family
    if (not family.complete_transport or family_selection.selection is None
            or family_selection.status not in final_ingest.ALLOWED_INITIAL_SELECTION
            or family_selection.spsa_seed_method_id is None
            or family.producer_failure_rows or family.input_artifact_failures
            or family.integrity_errors):
        raise DurableSessionError("initial selector did not verify a complete, clean 720-row family")
    selection = family_selection.selection
    if (selection.metric_backend != "torch_cpu"
            or selection.accounting.get("unique_official_score_rows") != 720
            or selection.accounting.get("observed_unique_rows") != 720
            or selection.accounting.get("failed_score_rows") != 0
            or selection.accounting.get("missing_score_rows") != 0
            or selection.accounting.get("unexpected_score_rows") != 0
            or selection.errors):
        raise DurableSessionError("initial selector did not produce a full clean canonical 720-row result")
    seed_id = family_selection.spsa_seed_method_id
    if ((family_selection.status == "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV"
         and selection.selected_method_id != seed_id)
            or (family_selection.status == "NOT_ADMITTED"
                and (selection.selected_method_id is not None
                     or selection.refinement_seed_method_id != seed_id))):
        raise DurableSessionError("initial SPSA seed is not the actual selection/fallback method")

    composed_pin = _regular_pin(composed_family_plan_path, file_id="composed_family_plan",
                                role="composed_family_plan")
    composed, composed_raw = _read_json(composed_pin.local_path, "composed family plan")
    composed_self = final_ingest._hash_object(composed, "plan_sha256", "composed family plan")
    final_ingest._verify_composed_family_plan(
        composed, composed_raw, composed_pin, index, nested_files, family_selection)
    if composed.get("scientific_input_pins_sha256") != family.input_pins_sha256:
        raise DurableSessionError("composed family plan does not bind the selector input digest")
    methods = {method["method_id"]: method for method in composed["methods"]}
    seed_method = methods.get(seed_id)
    seed_spec = next((spec for spec in family.candidates if spec.method_id == seed_id), None)
    if not isinstance(seed_method, Mapping) or seed_spec is None:
        raise DurableSessionError("actual selected/fallback seed is not a frozen snapshot method")
    source_lineage = packet_builder._seed_snapshot_lineage(family_selection, seed_method)
    source_pins = {
        "selector": _regular_pin(selector.__file__, file_id="selector_source", role="source_code"),
        "initial_ingest": _regular_pin(initial_ingest.__file__, file_id="initial_ingest_source", role="source_code"),
        "packet_builder": _regular_pin(packet_builder.__file__, file_id="packet_builder_source", role="source_code"),
        "final_consumer": _regular_pin(final_ingest.__file__, file_id="final_consumer_source", role="source_code"),
        "native_point_score_adapter": _regular_pin(
            native_point_score_adapter.__file__, file_id="native_point_score_adapter_source", role="source_code"),
        "durable_session": _regular_pin(__file__, file_id="durable_session_source", role="source_code"),
    }
    receipt = _seal_document({
        "schema": INITIAL_RECEIPT_SCHEMA,
        "status": "INITIAL_720_SELECTOR_COMPLETE",
        "metric_backend": "torch_cpu",
        "selection_status": family_selection.status,
        "selection": _plain(selection),
        "initial_ingest_index": _pin_link(index_pin, index_self),
        "initial_selector_index_sha256": index_pin.sha256,
        "initial_selector_index_self_sha256": index_self,
        "composed_family_plan": _pin_link(composed_pin, composed_self),
        "spsa_seed_method_id": seed_id,
        "scientific_input_pins_sha256": family.input_pins_sha256,
        "source_roster": {
            "sample_to_speaker": dict(family.roster.sample_to_speaker),
            "model_ids": list(family.roster.model_ids),
            "renderer_seed": family.roster.renderer_seed,
        },
        "reference_specs": [_plain(spec) for spec in family.reference_specs],
        "seed_snapshot_spec": _plain(seed_spec),
        "seed_snapshot_lineage": _plain(source_lineage),
        "source_code_pins": {
            role: {"path": pin.declared_path, "bytes": pin.byte_count, "sha256": pin.sha256}
            for role, pin in source_pins.items()
        },
    }, "receipt_sha256")
    raw = (json.dumps(receipt, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode("utf-8")
    target = Path(receipt_path)
    _write_exclusive(target, raw)
    target_pin = _regular_pin(target, file_id="initial_selector_receipt",
                              role="initial_selector_receipt")
    return {"path": target_pin.declared_path, "bytes": target_pin.byte_count,
            "sha256": target_pin.sha256, "self_sha256": receipt["receipt_sha256"]}


@dataclass(frozen=True)
class PointOutcome:
    """Proposal-side readback returned by the model-block dispatcher."""

    disposition: str  # ABSENT, UNKNOWN, or TERMINAL
    terminal_status: str | None = None
    point_entry: Mapping[str, Any] | None = None
    files: tuple[Mapping[str, Any], ...] = ()
    raw_npz_by_row: Mapping[str, Mapping[str, Any]] | None = None
    reason: str | None = None


class ProposalPointExecutor(Protocol):
    """Per-side executor; production implementation orchestrates three models."""

    def readback(self, proposal: selector.SPSAProposal) -> PointOutcome: ...

    def execute(self, proposal: selector.SPSAProposal) -> PointOutcome: ...


@dataclass(frozen=True)
class _VerifiedInputs:
    initial_index_pin: initial_ingest._PinnedFile
    initial_index_self: str
    initial_receipt_pin: initial_ingest._PinnedFile
    initial_receipt_self: str
    receipt: Mapping[str, Any]
    family_selection: initial_ingest.FamilySelection
    composed_pin: initial_ingest._PinnedFile
    composed_self: str
    composed: Mapping[str, Any]
    package_pin: initial_ingest._PinnedFile
    package_self: str
    package: Mapping[str, Any]
    execution_package: Mapping[str, Any]
    files: Mapping[str, initial_ingest._PinnedFile]
    seed_method: Mapping[str, Any]
    seed_lineage: Mapping[str, Any]


def _initial_outer_file_entries(index_pin: initial_ingest._PinnedFile,
                                receipt_pin: initial_ingest._PinnedFile,
                                composed_pin: initial_ingest._PinnedFile,
                                package_pin: initial_ingest._PinnedFile,
                                nested_files: Mapping[str, initial_ingest._PinnedFile],
                                package: Mapping[str, Any]) -> dict[str, initial_ingest._PinnedFile]:
    fixed = [index_pin, receipt_pin, composed_pin, package_pin,
             _regular_pin(__file__, file_id="durable_session_source", role="durable_session_source")]
    pins_by_id = {pin.file_id: pin for pin in nested_files.values()}
    for pin in fixed:
        previous = pins_by_id.get(pin.file_id)
        if previous is not None and previous != pin:
            raise DurableSessionError(f"outer file ID collides with the initial index: {pin.file_id}")
        pins_by_id[pin.file_id] = pin
    files = _package_file_table(package, fixed_pins=tuple(pins_by_id.values()))
    # Validate unique role/declaration identities exactly as the final ingest does.
    seen: set[tuple[str, str]] = set()
    for pin in files.values():
        key = (pin.role, pin.declared_path)
        if key in seen:
            raise DurableSessionError(f"duplicate outer file role/path in session closure: {key}")
        seen.add(key)
        initial_ingest._read_pinned(pin)
    return files


def _load_verified_inputs(initial_index_path: str | Path,
                          initial_receipt_path: str | Path,
                          composed_plan_path: str | Path,
                          execution_package_path: str | Path) -> _VerifiedInputs:
    index_pin = _regular_pin(initial_index_path, file_id="initial_ingest_index",
                             role="initial_ingest_index")
    index, index_raw = _read_json(index_pin.local_path, "initial ingest index")
    index_self = final_ingest._hash_object(index, "index_sha256", "initial ingest index")
    nested_files = initial_ingest._file_table(index)
    final_ingest._verify_initial_preselector(index, nested_files)
    family_selection = initial_ingest.ingest_family(index_pin.local_path, metric_backend="torch_cpu")
    family = family_selection.family
    if (not family.complete_transport or family_selection.selection is None
            or family_selection.status not in final_ingest.ALLOWED_INITIAL_SELECTION
            or family_selection.spsa_seed_method_id is None
            or family.producer_failure_rows or family.input_artifact_failures
            or family.integrity_errors):
        raise DurableSessionError("session initialization requires a complete, clean 720-row initial family")
    if (_sha(index_raw) != index_pin.sha256
            or family_selection.selection.metric_backend != "torch_cpu"
            or family_selection.selection.accounting.get("unique_official_score_rows") != 720
            or family_selection.selection.accounting.get("observed_unique_rows") != 720
            or family_selection.selection.accounting.get("failed_score_rows") != 0
            or family_selection.selection.accounting.get("missing_score_rows") != 0
            or family_selection.selection.errors):
        raise DurableSessionError("initial family failed canonical torch_cpu complete-720 accounting")

    receipt_pin = _regular_pin(initial_receipt_path, file_id="initial_selector_receipt",
                               role="initial_selector_receipt")
    receipt, receipt_raw = _read_json(receipt_pin.local_path, "initial selector receipt")
    receipt_self = final_ingest._hash_object(receipt, "receipt_sha256", "initial selector receipt")
    final_ingest._verify_receipt_link(receipt, index_pin=index_pin, index_self_sha256=index_self)
    expected_roster = {
        "sample_to_speaker": dict(family.roster.sample_to_speaker),
        "model_ids": list(family.roster.model_ids),
        "renderer_seed": family.roster.renderer_seed,
    }
    if (receipt.get("selection") != _plain(family_selection.selection)
            or receipt.get("selection_status") != family_selection.status
            or receipt.get("spsa_seed_method_id") != family_selection.spsa_seed_method_id
            or receipt.get("scientific_input_pins_sha256") != family.input_pins_sha256
            or receipt.get("initial_selector_index_sha256") != index_pin.sha256
            or receipt.get("initial_selector_index_self_sha256") != index_self
            or receipt.get("source_roster") != expected_roster
            or receipt.get("reference_specs") != [_plain(spec) for spec in family.reference_specs]
            or receipt.get("receipt_sha256") != receipt_self):
        raise DurableSessionError("initial selector receipt differs from actual complete-family replay")
    if _sha(receipt_raw) != receipt_pin.sha256:
        raise DurableSessionError("initial selector receipt changed while reading")
    expected_code = {
        "selector": selector.__file__,
        "initial_ingest": initial_ingest.__file__,
        "packet_builder": packet_builder.__file__,
        "final_consumer": final_ingest.__file__,
        "native_point_score_adapter": native_point_score_adapter.__file__,
        "durable_session": __file__,
    }
    code_pins = receipt.get("source_code_pins")
    if not isinstance(code_pins, Mapping) or set(code_pins) != set(expected_code):
        raise DurableSessionError("initial receipt does not bind the complete source-code closure")
    for role, actual_path in expected_code.items():
        expected_pin = _regular_pin(actual_path, file_id=role, role="source_code")
        declared = code_pins.get(role)
        if (not isinstance(declared, Mapping)
                or set(declared) != {"path", "bytes", "sha256"}
                or declared != {"path": expected_pin.declared_path,
                                "bytes": expected_pin.byte_count,
                                "sha256": expected_pin.sha256}):
            raise DurableSessionError(f"initial selector source-code pin changed: {role}")

    composed_pin = _regular_pin(composed_plan_path, file_id="composed_family_plan",
                                role="composed_family_plan")
    composed, composed_raw = _read_json(composed_pin.local_path, "composed family plan")
    composed_self = final_ingest._hash_object(composed, "plan_sha256", "composed family plan")
    final_ingest._verify_composed_family_plan(
        composed, composed_raw, composed_pin, index, nested_files, family_selection)
    if receipt.get("composed_family_plan") != _pin_link(composed_pin, composed_self):
        raise DurableSessionError("initial selector receipt does not link the exact composed family plan")
    source_methods = {item["method_id"]: item for item in composed["methods"]}
    seed_id = family_selection.spsa_seed_method_id
    seed_method = source_methods.get(seed_id)
    seed_spec = next((spec for spec in family.candidates if spec.method_id == seed_id), None)
    if not isinstance(seed_method, Mapping) or seed_spec is None:
        raise DurableSessionError("actual selected/fallback snapshot is not bound in the family")
    if (receipt.get("seed_snapshot_spec") != _plain(seed_spec)
            or receipt.get("seed_snapshot_lineage") != _plain(
                packet_builder._seed_snapshot_lineage(family_selection, seed_method))):
        raise DurableSessionError("receipt snapshot/head/constructor lineage differs from actual source snapshot")

    package_pin = _regular_pin(execution_package_path, file_id="execution_package_manifest",
                               role="execution_package_manifest")
    package, package_raw = _read_json(package_pin.local_path, "execution package manifest")
    package_self = final_ingest._hash_object(package, "manifest_sha256", "execution package manifest")
    files = _initial_outer_file_entries(index_pin, receipt_pin, composed_pin, package_pin,
                                        nested_files, package)
    execution_package = final_ingest._validate_execution_package_manifest(
        package, package_pin, composed, composed_pin, composed_self, files)
    if _sha(package_raw) != package_pin.sha256:
        raise DurableSessionError("execution package manifest changed during verification")
    return _VerifiedInputs(
        index_pin, index_self, receipt_pin, receipt_self, receipt, family_selection,
        composed_pin, composed_self, composed, package_pin, package_self, package,
        execution_package, files, dict(seed_method),
        packet_builder._seed_snapshot_lineage(family_selection, seed_method),
    )


def _point_entry_body(entry: Mapping[str, Any], files: Sequence[Mapping[str, Any]],
                      raw_npz_by_row: Mapping[str, Mapping[str, Any]] | None,
                      terminal_status: str) -> dict[str, Any]:
    if terminal_status not in POINT_OUTCOMES:
        raise DurableSessionError(f"unsupported terminal point status: {terminal_status}")
    if not isinstance(entry, Mapping) or not isinstance(entry.get("blocks"), list):
        raise DurableSessionError("terminal proposal point must contain native model blocks")
    return _seal_document({
        "schema": POINT_ENTRY_SCHEMA,
        "terminal_status": terminal_status,
        "point_entry": dict(entry),
        "file_entries": [dict(item) for item in files],
        "raw_npz_by_row": {key: dict(value) for key, value in (raw_npz_by_row or {}).items()},
    }, "point_entry_sha256")


class DurableSPSASession:
    """Create-only event store that replays the frozen ask/tell controller."""

    def __init__(self, root: Path, manifest: Mapping[str, Any], inputs: _VerifiedInputs,
                 controller: selector.SPSAController, events: list[dict[str, Any]],
                 point_entries: Mapping[tuple[str, str], Mapping[str, Any]],
                 *, poisoned: bool = False) -> None:
        self.root = root
        self.manifest = dict(manifest)
        self.inputs = inputs
        self.controller = controller
        self.events = events
        self.point_entries = dict(point_entries)
        self.used_npz_paths = {
            str(Path(pin.local_path).resolve(strict=True))
            for pin in inputs.files.values() if pin.role == "raw_matrix_npz"
        }
        self.dispatch_intents = {
            (event["proposal_id"], event["side"])
            for event in self.events if event.get("event_type") == "POINT_DISPATCH_INTENT"
        }
        self.native_support_derivations: dict[tuple[str, str], dict[str, Any]] = {}
        self.poisoned = poisoned

    @classmethod
    def create(
        cls, *, session_dir: str | Path, initial_index_path: str | Path,
        initial_receipt_path: str | Path, composed_plan_path: str | Path,
        execution_package_path: str | Path, session_id: str | None = None,
    ) -> "DurableSPSASession":
        inputs = _load_verified_inputs(initial_index_path, initial_receipt_path,
                                       composed_plan_path, execution_package_path)
        root = Path(session_dir)
        if root.exists() or root.is_symlink():
            raise DurableSessionError(f"new session path must be absent: {root}")
        root.mkdir(parents=True, exist_ok=False)
        _fsync_directory(root.parent)
        (root / "events").mkdir()
        (root / "point_entries").mkdir()
        _fsync_directory(root)
        actual_session_id = session_id or (
            f"fit40_spsa_{inputs.initial_receipt_self[:12]}_{uuid.uuid4().hex[:12]}")
        manifest = _seal_document({
            "schema": SESSION_MANIFEST_SCHEMA,
            "status": "OPEN",
            "session_id": actual_session_id,
            "metric_backend": "torch_cpu",
            "initial_ingest_index": _pin_link(inputs.initial_index_pin, inputs.initial_index_self),
            "initial_selector_receipt": _pin_link(inputs.initial_receipt_pin, inputs.initial_receipt_self),
            "composed_family_plan": _pin_link(inputs.composed_pin, inputs.composed_self),
            "execution_package_manifest": _pin_link(inputs.package_pin, inputs.package_self),
            "scientific_input_pins": dict(inputs.family_selection.family.input_pins),
            "scientific_input_pins_sha256": inputs.family_selection.family.input_pins_sha256,
            "spsa_seed_method_id": inputs.family_selection.spsa_seed_method_id,
            "initial_selection_status": inputs.family_selection.status,
            "initial_theta": list(next(
                spec.theta for spec in inputs.family_selection.family.candidates
                if spec.method_id == inputs.family_selection.spsa_seed_method_id)),
            "source_roster": {
                "sample_to_speaker": dict(inputs.family_selection.family.roster.sample_to_speaker),
                "model_ids": list(inputs.family_selection.family.roster.model_ids),
                "renderer_seed": inputs.family_selection.family.roster.renderer_seed,
            },
            "reference_specs": [_plain(spec) for spec in inputs.family_selection.family.reference_specs],
            "source_snapshot_lineage": _plain(inputs.seed_lineage),
        "source_file_table": [_file_entry(inputs.files[key]) for key in sorted(inputs.files)],
        }, "manifest_sha256")
        _write_exclusive(root / "session_manifest.json",
                         (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False,
                                     allow_nan=False) + "\n").encode("utf-8"))
        controller = initial_ingest.initialize_spsa(inputs.family_selection, metric_backend="torch_cpu")
        session = cls(root, manifest, inputs, controller, [], {})
        session._append_event("SESSION_INITIALIZED", {
            "controller_state": final_ingest.controller_state(controller),
            "selection_status": inputs.family_selection.status,
            "spsa_seed_method_id": inputs.family_selection.spsa_seed_method_id,
        })
        return session

    @classmethod
    def resume(
        cls, *, session_dir: str | Path, point_executor: ProposalPointExecutor | None = None,
    ) -> "DurableSPSASession":
        """Re-verify sources and deterministic replay; never infer a missing child as failed."""
        root = Path(session_dir)
        manifest, manifest_raw = _read_json(root / "session_manifest.json", "session manifest")
        manifest_self = _hash_object(manifest, "manifest_sha256", "session manifest")
        if (manifest.get("schema") != SESSION_MANIFEST_SCHEMA
                or manifest.get("status") != "OPEN"
                or manifest.get("metric_backend") != "torch_cpu"):
            raise DurableSessionError("session manifest is not an open canonical torch_cpu session")
        links = {key: value for key, value in manifest.items()
                 if key in {"initial_ingest_index", "initial_selector_receipt",
                            "composed_family_plan", "execution_package_manifest"}}
        for link in links.values():
            path = Path(link["path"])
            pin = _regular_pin(path, declared_path=link["path"], file_id="resume_pin", role="resume")
            if (pin.byte_count, pin.sha256) != (link["bytes"], link["sha256"]):
                raise DurableSessionError("session source pin changed before resume")
            value, _raw = _read_json(path, "session source"); self_sha = None
            if path == Path(manifest["initial_ingest_index"]["path"]):
                self_sha = final_ingest._hash_object(value, "index_sha256", "initial index")
            elif path == Path(manifest["initial_selector_receipt"]["path"]):
                self_sha = final_ingest._hash_object(value, "receipt_sha256", "initial receipt")
            elif path == Path(manifest["composed_family_plan"]["path"]):
                self_sha = final_ingest._hash_object(value, "plan_sha256", "composed plan")
            else:
                self_sha = final_ingest._hash_object(value, "manifest_sha256", "execution package")
            if self_sha != link["self_sha256"]:
                raise DurableSessionError("session source self hash changed before resume")
        inputs = _load_verified_inputs(
            manifest["initial_ingest_index"]["path"], manifest["initial_selector_receipt"]["path"],
            manifest["composed_family_plan"]["path"], manifest["execution_package_manifest"]["path"])
        if (inputs.initial_receipt_self != manifest["initial_selector_receipt"]["self_sha256"]
                or inputs.family_selection.spsa_seed_method_id != manifest["spsa_seed_method_id"]
                or inputs.family_selection.family.input_pins_sha256 != manifest["scientific_input_pins_sha256"]
                or inputs.package_self != manifest["execution_package_manifest"]["self_sha256"]
                or [_file_entry(inputs.files[key]) for key in sorted(inputs.files)]
                != manifest.get("source_file_table")
                or _sha(manifest_raw) != _sha((json.dumps(manifest, sort_keys=True, indent=2,
                                                           ensure_ascii=False, allow_nan=False) + "\n").encode())):
            raise DurableSessionError("session manifest does not match re-read full initial family")
        controller = initial_ingest.initialize_spsa(inputs.family_selection, metric_backend="torch_cpu")
        session = cls(root, manifest, inputs, controller, [], {})
        session._replay_events(point_executor=point_executor)
        return session

    @property
    def pending_pair(self) -> selector.SPSAProposalPair | None:
        return self.controller._pending

    @property
    def halted(self) -> bool:
        return self.controller.halted

    def _append_event(self, event_type: str, data: Mapping[str, Any]) -> dict[str, Any]:
        if self.poisoned:
            raise DurableSessionError("session has an uncommitted persistence error; reopen and replay it")
        index = len(self.events)
        previous = self.events[-1]["event_sha256"] if self.events else None
        event = _seal_document({
            "schema": EVENT_SCHEMA, "session_id": self.manifest["session_id"],
            "event_index": index, "event_type": event_type,
            "previous_event_sha256": previous, **dict(data),
        }, "event_sha256")
        path = self.root / "events" / f"event_{index:06d}.json"
        raw = (json.dumps(event, sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
        try:
            _write_exclusive(path, raw)
        except Exception:
            self.poisoned = True
            raise
        self.events.append(event)
        return event

    def _event_files(self) -> list[Path]:
        directory = self.root / "events"
        if directory.is_symlink() or not directory.is_dir():
            raise DurableSessionError("durable event directory is missing/unsafe")
        paths = sorted(directory.iterdir())
        expected = [f"event_{index:06d}.json" for index in range(len(paths))]
        if [path.name for path in paths] != expected or any(path.is_symlink() or not path.is_file() for path in paths):
            raise DurableSessionError("durable event files are missing, duplicated, or unsafe")
        return paths

    def _point_entry_files(self) -> list[Path]:
        directory = self.root / "point_entries"
        if directory.is_symlink() or not directory.is_dir():
            raise DurableSessionError("durable point-entry directory is missing/unsafe")
        paths = sorted(directory.iterdir())
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise DurableSessionError("durable point-entry directory contains an unsafe artifact")
        return paths

    @contextmanager
    def _writer_lock(self):
        """Serialize dispatch/tell writers; a crash leaves a fail-closed lock."""
        path = self.root / "writer.lock"
        lock_raw = _canonical({
            "schema": "fit40_durable_spsa_writer_lock_v1",
            "session_id": self.manifest["session_id"],
            "pid": os.getpid(),
            "created_unix_ns": time.time_ns(),
        }) + b"\n"
        try:
            descriptor = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError as exc:
            raise DurableSessionError(
                "another writer or an unreconciled crashed writer holds this session; do not dispatch"
            ) from exc
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(lock_raw)
                stream.flush()
                os.fsync(stream.fileno())
            _fsync_directory(self.root)
            yield
        finally:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != lock_raw:
                raise DurableSessionError("session writer lock changed; preserve it for manual reconciliation")
            path.unlink()
            _fsync_directory(self.root)

    @staticmethod
    def _validate_journal_chain_entry(entry: Mapping[str, Any]) -> None:
        body = dict(entry)
        declared = body.pop("entry_sha256", None)
        if declared != _sha(_canonical(body)):
            raise DurableSessionError("controller journal entry self hash mismatch")

    def _merge_point_files(self, entries: Sequence[Mapping[str, Any]]) -> None:
        if not isinstance(entries, Sequence):
            raise DurableSessionError("point entry file table is malformed")
        incoming = initial_ingest._file_table({"files": list(entries)})
        existing_by_id = dict(getattr(self, "_active_files", self.inputs.files))
        for file_id, pin in incoming.items():
            old = existing_by_id.get(file_id)
            if old is not None and old != pin:
                raise DurableSessionError(f"point file ID conflicts with an existing pin: {file_id}")
            if old is not None:
                initial_ingest._read_pinned(pin)
                continue
            duplicate = next((other for other in existing_by_id.values()
                              if (other.role, other.declared_path) == (pin.role, pin.declared_path)), None)
            if duplicate is not None:
                # A point terminal may repeat a session-level input under its
                # own deterministic file_id. Keep the original canonical ID
                # only when every content/path field is identical; the point
                # document retains its local alias for its own references.
                if ((duplicate.role, duplicate.declared_path, duplicate.local_path,
                     duplicate.byte_count, duplicate.sha256, duplicate.pin_name)
                        != (pin.role, pin.declared_path, pin.local_path,
                            pin.byte_count, pin.sha256, pin.pin_name)):
                    raise DurableSessionError("point file reuses role/path with different bytes or local path")
                initial_ingest._read_pinned(pin)
                continue
            initial_ingest._read_pinned(pin)
            existing_by_id[file_id] = pin
        # Store merged pins on the verified bundle without mutating the frozen base map.
        self._active_files = existing_by_id

    def _load_point_receipt(self, pin: Any) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
        if not isinstance(pin, Mapping) or set(pin) != {"path", "bytes", "sha256", "self_sha256"}:
            raise DurableSessionError("point entry pin must contain raw and canonical self hashes")
        path = Path(pin["path"])
        doc, raw = _read_json(path, "durable point entry")
        self_sha = _hash_object(doc, "point_entry_sha256", "durable point entry")
        if (len(raw), _sha(raw), self_sha) != (pin["bytes"], pin["sha256"], pin["self_sha256"]):
            raise DurableSessionError("durable point entry raw/self pin mismatch")
        if doc.get("schema") != POINT_ENTRY_SCHEMA:
            raise DurableSessionError("durable point entry schema differs")
        return doc, raw, dict(pin)

    def _validate_point_document(
        self, proposal: selector.SPSAProposal, doc: Mapping[str, Any],
    ) -> tuple[dict[str, initial_ingest._PinnedFile], dict[str, dict[str, Any]]]:
        """Recheck the point denominator and all claimed NPZ bytes on create and replay."""
        if (set(doc) != {"schema", "terminal_status", "point_entry", "file_entries",
                         "raw_npz_by_row", "point_entry_sha256"}
                or doc.get("schema") != POINT_ENTRY_SCHEMA
                or doc.get("terminal_status") not in POINT_OUTCOMES):
            raise DurableSessionError("durable point entry has an invalid terminal envelope")
        entry = doc.get("point_entry")
        if not isinstance(entry, Mapping):
            raise DurableSessionError("durable point entry omits its native point mapping")
        for key, expected in (
            ("point_id", proposal.method_id), ("method_id", proposal.method_id),
            ("point_kind", "spsa_proposal"), ("proposal_id", proposal.proposal_id),
            ("pair_id", proposal.pair_id), ("iteration", proposal.iteration),
            ("side", proposal.side),
        ):
            if entry.get(key) != expected:
                raise DurableSessionError(f"durable point entry {key} differs from its deterministic ask")
        blocks = entry.get("blocks")
        expected_models = list(self.inputs.family_selection.family.roster.model_ids)
        if (not isinstance(blocks, list)
                or [item.get("model_id") if isinstance(item, Mapping) else None for item in blocks]
                != expected_models):
            raise DurableSessionError("durable point blocks do not preserve the ordered three-model family")
        candidate_ids: set[str] = set()
        for block in blocks:
            row_ids = block.get("row_ids")
            if (not isinstance(row_ids, list) or len(row_ids) != len(proposal.sample_ids)
                    or any(not isinstance(value, str) or not value for value in row_ids)
                    or len(set(row_ids)) != len(row_ids)
                    or candidate_ids.intersection(row_ids)):
                raise DurableSessionError("durable point blocks do not contain exact unique proposal row IDs")
            candidate_ids.update(row_ids)
        if not isinstance(doc.get("file_entries"), list):
            raise DurableSessionError("durable point file table is malformed")
        try:
            point_files = initial_ingest._file_table({"files": doc["file_entries"]})
        except Exception as exc:
            raise DurableSessionError(f"durable point file table failed validation: {exc}") from exc
        raw_map = doc.get("raw_npz_by_row")
        if not isinstance(raw_map, Mapping):
            raise DurableSessionError("durable point NPZ map is malformed")
        if doc["terminal_status"] == "COMPLETE" and set(raw_map) != candidate_ids:
            raise DurableSessionError("complete point does not bind one official raw NPZ per candidate row")
        if not set(raw_map).issubset(candidate_ids):
            raise DurableSessionError("point NPZ map contains a row outside its planned candidate denominator")
        verified_npz: dict[str, dict[str, Any]] = {}
        local_npz_paths: set[str] = set()
        for row_id, npz_ref in raw_map.items():
            if (not isinstance(row_id, str) or not isinstance(npz_ref, Mapping)
                    or set(npz_ref) != {"path", "bytes", "sha256", "array_refs"}
                    or not isinstance(npz_ref.get("array_refs"), Mapping)):
                raise DurableSessionError(f"point raw NPZ link is malformed for {row_id}")
            matches = [pin for pin in point_files.values()
                       if pin.role == "raw_matrix_npz" and pin.declared_path == npz_ref["path"]
                       and pin.byte_count == npz_ref["bytes"] and pin.sha256 == npz_ref["sha256"]]
            if len(matches) != 1:
                raise DurableSessionError(f"point raw NPZ link does not resolve uniquely for {row_id}")
            pin = matches[0]
            try:
                local_path = str(Path(pin.local_path).resolve(strict=True))
            except Exception as exc:
                raise DurableSessionError(f"candidate NPZ local target is unavailable for {row_id}") from exc
            if local_path in self.used_npz_paths or local_path in local_npz_paths:
                raise DurableSessionError(
                    f"candidate raw NPZ path is reused across an initial/prior/current score row: {row_id}")
            local_npz_paths.add(local_path)
            arrays = self._read_and_validate_npz(pin)
            if arrays != dict(npz_ref["array_refs"]):
                raise DurableSessionError(f"point NPZ array references differ from actual bytes for {row_id}")
            verified_npz[row_id] = dict(npz_ref)
        return point_files, verified_npz, local_npz_paths

    @staticmethod
    def _read_and_validate_npz(pin: initial_ingest._PinnedFile) -> dict[str, dict[str, Any]]:
        try:
            import numpy as np
            raw = initial_ingest._read_pinned(pin)
            with np.load(BytesIO(raw), allow_pickle=False) as archive:
                if set(archive.files) != set(OFFICIAL_ARRAY_DTYPES):
                    raise DurableSessionError("point score NPZ member set differs from the seven-array contract")
                arrays = {name: np.asarray(archive[name]) for name in OFFICIAL_ARRAY_DTYPES}
        except DurableSessionError:
            raise
        except Exception as exc:
            raise DurableSessionError(f"cannot read point score NPZ {pin.file_id}: {exc}") from exc
        normalized = {}
        query_count = None
        for name, dtype in OFFICIAL_ARRAY_DTYPES.items():
            value = arrays[name]
            rank = 2 if name == "distance_by_query_lag" else 1
            if value.dtype.str != dtype or value.ndim != rank:
                raise DurableSessionError(f"point score NPZ dtype/rank differs for {name}")
            value = np.ascontiguousarray(value)
            if name == "distance_by_query_lag":
                if value.shape[1:] != (31,) or not np.isfinite(value).all():
                    raise DurableSessionError("point SyncNet matrix must be finite [N,31]")
                query_count = int(value.shape[0])
            normalized[name] = value
        if (not query_count or query_count < 1
                or any(value.shape[0] != query_count for name, value in normalized.items()
                       if name != "lag_offsets_frames")
                or normalized["lag_offsets_frames"].tolist() != list(range(-15, 16))
                or np.any(normalized["query_timestamp_den"] <= 0)):
            raise DurableSessionError("point score NPZ clocks, query counts, or lag axis are invalid")
        return {
            name: {
                "dtype": value.dtype.str,
                "shape": list(value.shape),
                "array_sha256": _sha(value.tobytes(order="C")),
            }
            for name, value in normalized.items()
        }

    def _empty_evaluation(self, proposal: selector.SPSAProposal) -> selector.ProposalEvaluation:
        return selector.ProposalEvaluation(
            proposal.proposal_id, proposal.pair_id, proposal.side, proposal.method_id,
            proposal.theta, proposal.theta_sha256, proposal.artifact_sha256,
            proposal.binding_sha256, proposal.input_pins_sha256, proposal.batch_id, (),
        )

    def _validate_evaluation_rows(self, evaluation: selector.ProposalEvaluation,
                                  proposal: selector.SPSAProposal) -> None:
        rows = evaluation.rows
        family = self.inputs.family_selection.family
        expected_ids = {"identity", "rms_fixed", proposal.method_id}
        model_ids = tuple(family.roster.model_ids)
        expected_contexts = {(sample, model) for sample in proposal.sample_ids for model in model_ids}
        if len(rows) != len(proposal.sample_ids) * 3 * len(model_ids):
            raise DurableSessionError("proposal evaluation must contain 8 clips × 3 models × 3 methods")
        if len({row.row_id for row in rows}) != len(rows):
            raise DurableSessionError("proposal evaluation repeats a row ID")
        if {row.method_id for row in rows} != expected_ids:
            raise DurableSessionError("proposal evaluation method roster differs from raw identity/rms_fixed/candidate")
        if any(row.status != "PASS" or row.input_pins_sha256 != family.input_pins_sha256
               for row in rows):
            raise DurableSessionError("proposal evaluation includes failed rows or different source pins")
        samples = set(proposal.sample_ids)
        if {row.sample_id for row in rows} != samples:
            raise DurableSessionError("proposal evaluation does not use the exact deterministic eight-clip batch")
        if any(row.sample_id not in samples or row.seed != selector.RENDERER_SEED
               for row in rows):
            raise DurableSessionError("proposal evaluation changed the fixed Fit source/seed scope")
        models = set(model_ids)
        if {row.model_id for row in rows} != models:
            raise DurableSessionError("proposal evaluation does not contain the exact three-model family")
        for method_id in expected_ids:
            method_rows = [row for row in rows if row.method_id == method_id]
            if (len(method_rows) != len(expected_contexts)
                    or {(row.sample_id, row.model_id) for row in method_rows} != expected_contexts):
                raise DurableSessionError(f"proposal evaluation row count differs for method {method_id}")
        for reference_id in ("identity", "rms_fixed"):
            replayed = [row for row in rows if row.method_id == reference_id]
            expected = [row for row in family.rows if row.method_id == reference_id
                        and row.sample_id in samples]
            if (selector._baseline_fingerprint(replayed, reference_id)
                    != selector._baseline_fingerprint(expected, reference_id)):
                raise DurableSessionError(f"proposal {reference_id} rows differ from the original full-family baseline")
        identity_by_context = {(row.sample_id, row.model_id): row for row in family.rows
                               if row.method_id == "identity" and row.sample_id in samples}
        candidate_rows = [row for row in rows if row.method_id == proposal.method_id]
        if any(row.artifact_sha256 != proposal.artifact_sha256
               or row.binding_sha256 != proposal.binding_sha256 for row in candidate_rows):
            raise DurableSessionError("candidate score rows do not bind the asked theta/artifact")
        for row in candidate_rows:
            parent = identity_by_context.get((row.sample_id, row.model_id))
            if parent is None or any((getattr(row, name) != getattr(parent, name)) for name in (
                "split", "speaker_id", "face_id", "seed", "natural_pcm_sha256",
                "source_video_sha256", "source_face_sha256", "input_pins_sha256",
                "lag_offsets_frames", "anchor_column", "source_media_kind",
            )):
                raise DurableSessionError("candidate rows changed the original source/audio/query-clock identity")
            # The final consumer validates candidate query keys against that
            # candidate's own pinned producer frame map and the original PCM
            # clock. Tracking coverage can differ from identity; selector06
            # alone decides the common FULL support used for admission.

    @staticmethod
    def _validate_evaluation_matches(evaluation: selector.ProposalEvaluation,
                                     proposal: selector.SPSAProposal) -> bool:
        return (evaluation.proposal_id == proposal.proposal_id
                and evaluation.pair_id == proposal.pair_id and evaluation.side == proposal.side
                and evaluation.method_id == proposal.method_id and tuple(evaluation.theta) == proposal.theta
                and evaluation.theta_sha256 == proposal.theta_sha256
                and evaluation.artifact_sha256 == proposal.artifact_sha256
                and evaluation.binding_sha256 == proposal.binding_sha256
                and evaluation.input_pins_sha256 == proposal.input_pins_sha256
                and evaluation.batch_id == proposal.batch_id)

    def _append_point_outcome(self, proposal: selector.SPSAProposal,
                              outcome: PointOutcome) -> Mapping[str, Any]:
        if outcome.disposition != "TERMINAL" or outcome.terminal_status not in POINT_OUTCOMES:
            raise DurableSessionError("only an explicit terminal point can be persisted")
        if (not isinstance(outcome.point_entry, Mapping)
                or not isinstance(outcome.files, (tuple, list))
                or not isinstance(outcome.raw_npz_by_row, Mapping)):
            raise DurableSessionError("terminal point outcome lacks exact entry/file table")
        normalized_files = tuple(dict(item) for item in outcome.files)
        body = _point_entry_body(outcome.point_entry, normalized_files, outcome.raw_npz_by_row,
                                 outcome.terminal_status)
        key = (proposal.proposal_id, proposal.side)
        existing = self.point_entries.get(key)
        if existing is not None:
            if (existing.get("point_doc") != body
                    or existing.get("event", {}).get("terminal_status") != outcome.terminal_status):
                raise DurableSessionError("duplicate point terminal changed its raw/self evidence")
            return existing
        if key not in self.dispatch_intents:
            raise DurableSessionError("terminal point has no durable pre-dispatch intent")
        point_files, verified_npz, local_npz_paths = self._validate_point_document(proposal, body)
        point_dir = self.root / "point_entries"
        safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", proposal.method_id)
        point_path = point_dir / f"{proposal.iteration:02d}_{proposal.side}_{safe_id}.json"
        raw = (json.dumps(body, sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
        try:
            _write_exclusive(point_path, raw)
        except Exception:
            self.poisoned = True
            raise
        point_pin = {"path": str(point_path.resolve()), "bytes": len(raw),
                     "sha256": _sha(raw), "self_sha256": body["point_entry_sha256"]}
        expected_leaf = f"{proposal.iteration:02d}_{proposal.side}_{safe_id}.json"
        if point_path.name != expected_leaf:
            raise DurableSessionError("durable point receipt filename is not canonical")
        try:
            self._merge_point_files(normalized_files)
            self._append_event("POINT_TERMINAL", {
                "proposal_id": proposal.proposal_id, "point_id": proposal.method_id,
                "pair_id": proposal.pair_id, "iteration": proposal.iteration,
                "side": proposal.side, "terminal_status": outcome.terminal_status,
                "reason": outcome.reason, "point_entry": dict(outcome.point_entry),
                "point_entry_pin": point_pin,
            })
        except Exception:
            self.poisoned = True
            raise
        self.used_npz_paths.update(local_npz_paths)
        record = {"point_doc": body, "point_pin": point_pin,
                  "event": self.events[-1]}
        self.point_entries[key] = record
        return record

    def _append_dispatch_intent(self, proposal: selector.SPSAProposal) -> None:
        key = (proposal.proposal_id, proposal.side)
        if key in self.dispatch_intents:
            raise UnknownPointOutcome(
                f"{proposal.method_id} has a durable launch intent without terminal readback")
        self._append_event("POINT_DISPATCH_INTENT", {
            "proposal_id": proposal.proposal_id, "point_id": proposal.method_id,
            "pair_id": proposal.pair_id, "iteration": proposal.iteration,
            "side": proposal.side, "proposal": _plain(proposal),
        })
        self.dispatch_intents.add(key)

    def ask_pair(self) -> selector.SPSAProposalPair:
        with self._writer_lock():
            return self._ask_pair_locked()

    def _ask_pair_locked(self) -> selector.SPSAProposalPair:
        if self.poisoned or self.controller.halted:
            raise DurableSessionError("session cannot ask after a persistence error or controller halt")
        if self.controller._pending is not None:
            return self.controller._pending
        if self.events and self.events[-1].get("event_type") == "PAIR_ASKED":
            raise DurableSessionError("in-memory pending pair was lost; reopen session to replay it")
        pair = self.controller.ask()
        try:
            self._append_event("PAIR_ASKED", {
                "iteration": pair.iteration, "pair_id": pair.pair_id,
                "ask_pair": _plain(pair),
                "controller_state_after": final_ingest.controller_state(self.controller),
            })
        except Exception:
            self.poisoned = True
            raise
        return pair

    def _point_outcome_for(self, proposal: selector.SPSAProposal,
                           executor: ProposalPointExecutor) -> Mapping[str, Any]:
        key = (proposal.proposal_id, proposal.side)
        if key in self.point_entries:
            return self.point_entries[key]
        readback = executor.readback(proposal)
        if not isinstance(readback, PointOutcome):
            raise DurableSessionError("executor readback must return PointOutcome")
        if readback.disposition == "UNKNOWN":
            raise UnknownPointOutcome(
                f"{proposal.method_id} outcome is unknown; preserve pending ask and read back again")
        if readback.disposition == "TERMINAL":
            if key not in self.dispatch_intents:
                raise DurableSessionError("executor reports a terminal point without a durable dispatch intent")
            return self._append_point_outcome(proposal, readback)
        if readback.disposition != "ABSENT":
            raise DurableSessionError("executor readback disposition must be ABSENT/UNKNOWN/TERMINAL")
        if key in self.dispatch_intents:
            raise UnknownPointOutcome(
                f"{proposal.method_id} has a durable launch intent but no terminal readback; do not relaunch")
        self._append_dispatch_intent(proposal)
        outcome = executor.execute(proposal)
        if not isinstance(outcome, PointOutcome):
            raise DurableSessionError("executor execute must return PointOutcome")
        if outcome.disposition == "UNKNOWN" or outcome.disposition == "ABSENT":
            raise UnknownPointOutcome(
                f"{proposal.method_id} dispatch has no terminal readback; do not duplicate execution")
        if outcome.disposition != "TERMINAL":
            raise DurableSessionError("executor returned unsupported point disposition")
        return self._append_point_outcome(proposal, outcome)

    def _evaluation_for_point(self, proposal: selector.SPSAProposal,
                              terminal: Mapping[str, Any]) -> selector.ProposalEvaluation:
        if terminal["event"].get("terminal_status") != "COMPLETE":
            return self._empty_evaluation(proposal)
        evaluation = self._load_official_evaluation(proposal, terminal["point_doc"])
        if not self._validate_evaluation_matches(evaluation, proposal):
            raise DurableSessionError("official raw-row evaluation differs from exact persisted ask/theta/pins")
        self._validate_evaluation_rows(evaluation, proposal)
        npz_identity = self._evaluation_npz_identity(proposal, terminal)
        candidate_ids = {row.row_id for row in evaluation.rows if row.method_id == proposal.method_id}
        if set(npz_identity["candidate_row_ids"]) != candidate_ids:
            raise DurableSessionError("candidate NPZ references do not cover the exact candidate evaluation rows")
        return evaluation

    def _load_official_evaluation(
        self, proposal: selector.SPSAProposal, point_doc: Mapping[str, Any],
    ) -> selector.ProposalEvaluation:
        """Rebuild tell rows from pinned Stage-A JSONL and NPZ bytes, never executor scalars."""
        entry = point_doc.get("point_entry")
        if not isinstance(entry, Mapping):
            raise DurableSessionError("completed point receipt lacks its execution entry")
        file_table: Mapping[str, initial_ingest._PinnedFile] = getattr(
            self, "_active_files", self.inputs.files)
        try:
            plan_pin = initial_ingest._required_file(file_table, entry.get("plan_file_id"), "plan")
            planned_pin = initial_ingest._required_file(
                file_table, entry.get("planned_rows_file_id"), "planned_rows")
            plan_raw = initial_ingest._read_pinned(plan_pin)
            plan = json.loads(plan_raw.decode("utf-8"))
            planned_rows = initial_ingest._parse_jsonl(
                initial_ingest._read_pinned(planned_pin), "proposal planned rows")
        except Exception as exc:
            raise DurableSessionError(f"cannot load the byte-pinned proposal plan/rows: {exc}") from exc
        if not isinstance(plan, Mapping):
            raise DurableSessionError("proposal point plan is not a JSON object")
        try:
            plan_self = final_ingest._hash_object(plan, "plan_sha256", "proposal point plan")
        except Exception as exc:
            raise DurableSessionError(f"proposal point plan self hash failed: {exc}") from exc
        if (plan.get("schema_version") != initial_ingest.PLAN_SCHEMA
                or plan.get("status") != "FROZEN"
                or plan.get("run_id") != entry.get("run_id")
                or plan.get("rows") != planned_rows
                or len(planned_rows) != len(proposal.sample_ids) * 3
                or plan.get("scientific_input_pins_sha256")
                != self.inputs.family_selection.family.input_pins_sha256):
            raise DurableSessionError("proposal plan is not the exact frozen 8×3 Fit derivation")
        execution = plan.get("execution")
        scope = execution.get("spsa_execution_scope") if isinstance(execution, Mapping) else None
        parent_pin = execution.get("spsa_execution_parent_plan") if isinstance(execution, Mapping) else None
        # The frozen point-plan producer and final-center consumer bind the
        # composed family with this exact three-field raw file link.  Its
        # canonical plan self hash is independently revalidated when the
        # composed plan is loaded into `self.inputs`.
        expected_parent = {
            "path": self.inputs.composed_pin.declared_path,
            "bytes": self.inputs.composed_pin.byte_count,
            "sha256": self.inputs.composed_pin.sha256,
        }
        if parent_pin != expected_parent or not isinstance(scope, Mapping):
            raise DurableSessionError("proposal plan does not bind the verified composed family and scope")
        expected_scope = {
            "scope_kind": "spsa_pair_proposal", "iteration": proposal.iteration,
            "pair_id": proposal.pair_id, "proposal_id": proposal.proposal_id,
            "side": proposal.side, "batch_id": proposal.batch_id,
            "sample_ids": list(proposal.sample_ids), "source_seed_method_id": self.inputs.family_selection.spsa_seed_method_id,
            "method_id": proposal.method_id, "artifact_sha256": proposal.artifact_sha256,
            "binding_sha256": proposal.binding_sha256,
            "input_pins_sha256": proposal.input_pins_sha256,
        }
        if any(scope.get(key) != value for key, value in expected_scope.items()):
            raise DurableSessionError("proposal plan scope does not match the persisted SPSA ask")
        if scope.get("run_id") != plan.get("run_id"):
            raise DurableSessionError("proposal scope run ID differs from its point plan")
        if entry.get("point_id") != proposal.method_id or entry.get("method_id") != proposal.method_id:
            raise DurableSessionError("proposal execution entry method differs from the exact ask")
        point_methods = plan.get("methods")
        point_method = next((item for item in point_methods
                             if isinstance(item, Mapping)
                             and item.get("method_id") == proposal.method_id), None)
        if (not isinstance(point_methods, list) or len(point_methods) != 1
                or not isinstance(point_method, Mapping)
                or not isinstance(point_method.get("config"), Mapping)):
            raise DurableSessionError("proposal candidate method is absent from its point plan")
        config = point_method["config"]
        expected_config = {
            "method": "fit40_spsa_proposal", "schema": "fit40_spsa_proposal_transform_v1",
            "proposal_id": proposal.proposal_id, "pair_id": proposal.pair_id,
            "iteration": proposal.iteration, "side": proposal.side,
            "method_id": proposal.method_id, "theta": list(proposal.theta),
            "theta_sha256": proposal.theta_sha256,
            "artifact_sha256": proposal.artifact_sha256,
            "binding_sha256": proposal.binding_sha256,
            "input_pins_sha256": proposal.input_pins_sha256,
            "batch_id": proposal.batch_id, "sample_ids": list(proposal.sample_ids),
            "direction": list(proposal.direction), "c_k": proposal.c_k,
        }
        if (point_method.get("config_sha256") != _sha(_canonical(config))
                or point_method.get("kind") != packet_builder.PROPOSAL_METHOD_KIND
                or any(config.get(key) != value for key, value in expected_config.items())):
            raise DurableSessionError("proposal method config does not bind exact theta/point provenance")
        for key, value in self.inputs.seed_lineage.items():
            if config.get(key) != value:
                raise DurableSessionError(f"proposal point config changed frozen snapshot lineage: {key}")
        if plan_self != plan.get("plan_sha256"):
            raise DurableSessionError("proposal plan hash changed after readback")

        candidate_plan_rows = [row for row in planned_rows if row.get("method_id") == proposal.method_id]
        expected_order = [(sample, model) for sample in proposal.sample_ids
                          for model in self.inputs.family_selection.family.roster.model_ids]
        if (len(candidate_plan_rows) != len(expected_order)
                or [(row.get("sample_id"), row.get("model_id")) for row in candidate_plan_rows]
                != expected_order
                or any(row.get("method_config_sha256") != point_method.get("config_sha256")
                       or row.get("spsa_point_binding_sha256") != proposal.binding_sha256
                       or row.get("score_audio") != row.get("natural_pcm")
                       for row in candidate_plan_rows)):
            raise DurableSessionError("candidate planned rows changed order, method config, or original-audio score source")
        plan_by_id = {row.get("row_id"): row for row in candidate_plan_rows}
        current_npz_paths = {
            ref.get("path") for ref in point_doc.get("raw_npz_by_row", {}).values()
            if isinstance(ref, Mapping) and isinstance(ref.get("path"), str)
        }
        npz_by_declared: dict[str, initial_ingest._PinnedFile] = {}
        used_npz_ids: set[str] = set()
        for pin in file_table.values():
            if pin.role != "raw_matrix_npz":
                continue
            if pin.declared_path in npz_by_declared:
                raise DurableSessionError("raw score NPZ declaration is ambiguous in the active file table")
            npz_by_declared[pin.declared_path] = pin
            local_path = str(Path(pin.local_path).resolve(strict=True))
            if local_path in self.used_npz_paths and pin.declared_path not in current_npz_paths:
                used_npz_ids.add(pin.file_id)
        try:
            used_frame_map_ids: set[str] = set()
            current_key = (proposal.proposal_id, proposal.side)
            for point_key, prior_record in self.point_entries.items():
                if point_key == current_key:
                    continue
                prior_doc = prior_record.get("point_doc") if isinstance(prior_record, Mapping) else None
                prior_entries = prior_doc.get("file_entries") if isinstance(prior_doc, Mapping) else None
                if isinstance(prior_entries, list):
                    prior_files = initial_ingest._file_table({"files": prior_entries})
                    used_frame_map_ids.update(
                        pin.file_id for pin in prior_files.values()
                        if pin.role == "official_source_frame_map")
            converted_rows, verified_block_pins = final_ingest._validate_point_blocks(
                entry, plan, plan_pin, planned_rows, planned_pin,
                self.inputs.composed_pin, self.inputs.composed_self,
                file_table, point_method, proposal.input_pins_sha256,
                npz_by_declared, used_npz_ids, self.inputs.execution_package,
                used_frame_map_ids,
                outer_npz_by_row=point_doc.get("raw_npz_by_row"),
            )
        except Exception as exc:
            raise DurableSessionError(
                f"proposal point failed independent final-consumer block/receipt validation: {exc}") from exc
        derivations_by_row = {
            row_id: derivation
            for block_pin in verified_block_pins
            for row_id, derivation in block_pin.get("native_support_derivations", {}).items()
        }
        expected_candidate_ids = {row["row_id"] for row in candidate_plan_rows}
        if set(derivations_by_row) != expected_candidate_ids:
            raise DurableSessionError("verified native support derivations do not cover exact candidate rows")
        self.native_support_derivations[(proposal.proposal_id, proposal.side)] = dict(sorted(derivations_by_row.items()))
        records_by_id = {row.row_id: row for row in converted_rows}
        if set(records_by_id) != set(plan_by_id):
            raise DurableSessionError("verified official score rows do not cover every planned candidate")
        samples = set(proposal.sample_ids)
        baseline = [row for row in self.inputs.family_selection.family.rows
                    if row.sample_id in samples and row.method_id in {"identity", "rms_fixed"}]
        candidate = [records_by_id[row["row_id"]] for row in candidate_plan_rows]
        evaluation = selector.ProposalEvaluation(
            proposal_id=proposal.proposal_id, pair_id=proposal.pair_id, side=proposal.side,
            method_id=proposal.method_id, theta=proposal.theta,
            theta_sha256=proposal.theta_sha256, artifact_sha256=proposal.artifact_sha256,
            binding_sha256=proposal.binding_sha256,
            input_pins_sha256=proposal.input_pins_sha256, batch_id=proposal.batch_id,
            rows=tuple((*baseline, *candidate)),
        )
        return evaluation

    def run_pending_pair(self, executor: ProposalPointExecutor) -> selector.SPSATellResult:
        """Resolve one persisted pair; unknown child outcomes are never retried."""
        with self._writer_lock():
            pair = self._ask_pair_locked()
            # A terminal failure on either side ends this pair.  In particular,
            # after reopening a session do not mistake the still-absent sibling
            # for permission to dispatch it.
            terminal_failure = any(
                record is not None and record["event"].get("terminal_status") != "COMPLETE"
                for record in (
                    self.point_entries.get((pair.plus.proposal_id, "plus")),
                    self.point_entries.get((pair.minus.proposal_id, "minus")),
                )
            )
            if not terminal_failure:
                for proposal in (pair.plus, pair.minus):
                    key = (proposal.proposal_id, proposal.side)
                    if key in self.point_entries:
                        if self.point_entries[key]["event"].get("terminal_status") != "COMPLETE":
                            break
                        continue
                    self._point_outcome_for(proposal, executor)
                    if self.point_entries[key]["event"].get("terminal_status") != "COMPLETE":
                        # Preserve this terminal failure and do not launch the
                        # sibling. The controller records NO_UPDATE and halts.
                        break
            plus_record = self.point_entries.get((pair.plus.proposal_id, "plus"))
            minus_record = self.point_entries.get((pair.minus.proposal_id, "minus"))
            if (plus_record is None or minus_record is None) and not any(
                record is not None and record["event"].get("terminal_status") != "COMPLETE"
                for record in (plus_record, minus_record)
            ):
                raise UnknownPointOutcome("pending pair lacks complete terminal readback for both sides")
            plus_eval = (self._evaluation_for_point(pair.plus, plus_record)
                         if plus_record and plus_record["event"].get("terminal_status") == "COMPLETE"
                         else self._empty_evaluation(pair.plus))
            minus_eval = (self._evaluation_for_point(pair.minus, minus_record)
                          if minus_record and minus_record["event"].get("terminal_status") == "COMPLETE"
                          else self._empty_evaluation(pair.minus))
            tell = self.controller.tell(plus_eval, minus_eval)
            journal_entry = self._make_journal_entry(pair, plus_record, minus_record,
                                                     plus_eval, minus_eval, tell)
            self._append_event("PAIR_TOLD", {
                "iteration": pair.iteration, "pair_id": pair.pair_id,
                "tell_result": _plain(tell),
                "controller_state_after": final_ingest.controller_state(self.controller),
                "journal_entry": journal_entry,
            })
            return tell

    def _make_journal_entry(self, pair: selector.SPSAProposalPair,
                            plus_record: Mapping[str, Any] | None,
                            minus_record: Mapping[str, Any] | None,
                            plus_eval: selector.ProposalEvaluation,
                            minus_eval: selector.ProposalEvaluation,
                            tell: selector.SPSATellResult) -> dict[str, Any]:
        point_outcomes = {
            "plus": self._point_outcome_status(plus_record),
            "minus": self._point_outcome_status(minus_record),
        }
        if ("NOT_DISPATCHED_AFTER_TERMINAL_FAILURE" in point_outcomes.values()
                and not any(value in POINT_OUTCOMES - {"COMPLETE"}
                            for value in point_outcomes.values())):
            raise DurableSessionError("an unlaunched side requires an explicit sibling terminal failure")
        entries = [event["journal_entry"] for event in self.events
                   if event.get("event_type") == "PAIR_TOLD"]
        previous = entries[-1]["entry_sha256"] if entries else None
        body = {
            "schema": "fit40_spsa_journal_entry_v1",
            "entry_index": pair.iteration,
            "iteration": pair.iteration,
            "previous_entry_sha256": previous,
            "pair_id": pair.pair_id,
            "ask_pair": _plain(pair),
            "plus_point_id": pair.plus.method_id,
            "minus_point_id": pair.minus.method_id,
            "plus_point_entry_pin": plus_record.get("point_pin") if plus_record else None,
            "minus_point_entry_pin": minus_record.get("point_pin") if minus_record else None,
            "point_outcomes": point_outcomes,
            "plus_evaluation": self._evaluation_record(pair.plus, plus_eval, plus_record),
            "minus_evaluation": self._evaluation_record(pair.minus, minus_eval, minus_record),
            "tell_result": _plain(tell),
            "controller_state_after": final_ingest.controller_state(self.controller),
        }
        return _seal_document(body, "entry_sha256")

    @staticmethod
    def _point_outcome_status(record: Mapping[str, Any] | None) -> str:
        if record is None:
            return "NOT_DISPATCHED_AFTER_TERMINAL_FAILURE"
        event = record.get("event")
        status = event.get("terminal_status") if isinstance(event, Mapping) else None
        if status not in POINT_OUTCOMES:
            raise DurableSessionError("stored proposal point has an unknown terminal status")
        return status

    def _evaluation_record(self, proposal: selector.SPSAProposal,
                           evaluation: selector.ProposalEvaluation,
                           point_record: Mapping[str, Any] | None) -> dict[str, Any]:
        refs = self._baseline_npz_identity(proposal.sample_ids)
        candidate = (self._evaluation_npz_identity(proposal, point_record)
                     if point_record and point_record["event"].get("terminal_status") == "COMPLETE"
                     else {"point_entry_pin": point_record.get("point_pin") if point_record else None,
                           "candidate_row_ids": [], "candidate_raw_npz_by_row": {}})
        derivations = (self.native_support_derivations.get((proposal.proposal_id, proposal.side), {})
                       if point_record and point_record["event"].get("terminal_status") == "COMPLETE"
                       else {})
        if point_record and point_record["event"].get("terminal_status") == "COMPLETE":
            if not derivations or set(derivations) != set(candidate.get("candidate_row_ids", [])):
                raise DurableSessionError("complete point journal lacks exact native support derivation receipts")
        candidate["native_support_derivations_by_row"] = derivations
        return {
            "proposal": _plain(proposal),
            "proposal_evaluation": {
                "proposal_id": evaluation.proposal_id, "pair_id": evaluation.pair_id,
                "side": evaluation.side, "method_id": evaluation.method_id,
                "theta": list(evaluation.theta), "theta_sha256": evaluation.theta_sha256,
                "artifact_sha256": evaluation.artifact_sha256,
                "binding_sha256": evaluation.binding_sha256,
                "input_pins_sha256": evaluation.input_pins_sha256,
                "batch_id": evaluation.batch_id,
                "row_ids": [row.row_id for row in evaluation.rows],
            },
            "baseline_npz_by_row": refs,
            "candidate_point_evidence": candidate,
        }

    def _baseline_npz_identity(self, sample_ids: Sequence[str]) -> dict[str, Any]:
        selected = set(sample_ids)
        family = self.inputs.family_selection.family
        target_rows = {row.row_id for row in family.rows
                       if row.sample_id in selected
                       and row.method_id in {"identity", "rms_fixed"}}
        # The initial selector binding stores every original NPZ file identity.
        bindings = {item["row_id"]: item for item in family.score_npz_bindings
                    if item.get("row_id") in target_rows}
        if set(bindings) != target_rows:
            raise DurableSessionError("initial baseline NPZ evidence is incomplete for the exact proposal batch")
        out = {}
        for row_id, item in bindings.items():
            out[row_id] = {
                "path": item["declared_path"], "bytes": item["bytes"],
                "sha256": item["sha256"], "file_id": item["file_id"],
            }
        return dict(sorted(out.items()))

    def _replay_events(self, *, point_executor: ProposalPointExecutor | None) -> None:
        previous_event = None
        terminal: dict[tuple[str, str], Mapping[str, Any]] = {}
        expected_journal_previous = None
        for index, path in enumerate(self._event_files()):
            event, _raw = _read_json(path, f"durable event {index}")
            event_self = _hash_object(event, "event_sha256", f"durable event {index}")
            if (event.get("schema") != EVENT_SCHEMA or event.get("event_index") != index
                    or event.get("session_id") != self.manifest["session_id"]
                    or event.get("previous_event_sha256") != previous_event):
                raise DurableSessionError(f"durable event hash chain mismatch at {index}")
            if index == 0:
                if (event.get("event_type") != "SESSION_INITIALIZED"
                        or event.get("controller_state") != final_ingest.controller_state(self.controller)):
                    raise DurableSessionError("initial controller state differs from deterministic initialization")
            elif event.get("event_type") == "PAIR_ASKED":
                pair = self.controller.ask()
                if (event.get("ask_pair") != _plain(pair)
                        or event.get("controller_state_after") != final_ingest.controller_state(self.controller)):
                    raise DurableSessionError("persisted pair/RNG/pending state differs from deterministic ask")
            elif event.get("event_type") == "POINT_DISPATCH_INTENT":
                pair = self.controller._pending
                if pair is None:
                    raise DurableSessionError("dispatch intent has no pending deterministic ask")
                side = event.get("side")
                proposal = pair.plus if side == "plus" else pair.minus if side == "minus" else None
                key = (proposal.proposal_id, side) if proposal is not None else None
                if (proposal is None or key in self.dispatch_intents
                        or event.get("proposal_id") != proposal.proposal_id
                        or event.get("point_id") != proposal.method_id
                        or event.get("pair_id") != proposal.pair_id
                        or event.get("iteration") != proposal.iteration
                        or event.get("proposal") != _plain(proposal)):
                    raise DurableSessionError("dispatch intent differs from the pending proposal or is duplicated")
                self.dispatch_intents.add(key)
            elif event.get("event_type") == "POINT_TERMINAL":
                pair = self.controller._pending
                if pair is None:
                    raise DurableSessionError("point terminal event has no pending ask")
                side = event.get("side")
                proposal = pair.plus if side == "plus" else pair.minus if side == "minus" else None
                if proposal is None or proposal.proposal_id != event.get("proposal_id"):
                    raise DurableSessionError("point terminal event differs from pending proposal")
                key = (proposal.proposal_id, side)
                if key not in self.dispatch_intents:
                    raise DurableSessionError("point terminal event has no durable pre-dispatch intent")
                doc, _raw, pin = self._load_point_receipt(event.get("point_entry_pin"))
                if (doc.get("point_entry") != event.get("point_entry")
                        or doc.get("terminal_status") != event.get("terminal_status")):
                    raise DurableSessionError("point entry changed relative to terminal event")
                point_files, _verified_npz, local_npz_paths = self._validate_point_document(proposal, doc)
                safe_id = re.sub(r"[^A-Za-z0-9_.-]", "_", proposal.method_id)
                expected_leaf = f"{proposal.iteration:02d}_{proposal.side}_{safe_id}.json"
                expected_path = (self.root / "point_entries" / expected_leaf).resolve(strict=True)
                if Path(pin["path"]).resolve(strict=True) != expected_path:
                    raise DurableSessionError("point-entry receipt is outside its canonical session path")
                if key in terminal:
                    raise DurableSessionError("duplicate proposal-side terminal event")
                self._merge_point_files(doc["file_entries"])
                self.used_npz_paths.update(local_npz_paths)
                record = {"event": event, "point_doc": doc, "point_pin": pin}
                terminal[key] = record
                self.point_entries[key] = record
            elif event.get("event_type") == "PAIR_TOLD":
                pair = self.controller._pending
                if pair is None or pair.pair_id != event.get("pair_id"):
                    raise DurableSessionError("persisted tell has no matching deterministic pending pair")
                entry = event.get("journal_entry")
                if not isinstance(entry, Mapping):
                    raise DurableSessionError("tell event omits its entry")
                self._validate_journal_chain_entry(entry)
                plus_record = terminal.get((pair.plus.proposal_id, "plus"))
                minus_record = terminal.get((pair.minus.proposal_id, "minus"))
                outcomes = entry.get("point_outcomes")
                expected_outcomes = {
                    "plus": self._point_outcome_status(plus_record),
                    "minus": self._point_outcome_status(minus_record),
                }
                if outcomes != expected_outcomes:
                    raise DurableSessionError("journal terminal/not-dispatched side status differs from actual readbacks")
                plus = self._replay_evaluation(
                    pair.plus, terminal, point_executor,
                    allow_missing=(outcomes["plus"] == "NOT_DISPATCHED_AFTER_TERMINAL_FAILURE"))
                minus = self._replay_evaluation(
                    pair.minus, terminal, point_executor,
                    allow_missing=(outcomes["minus"] == "NOT_DISPATCHED_AFTER_TERMINAL_FAILURE"))
                tell = self.controller.tell(plus, minus)
                expected_entry = self._make_journal_entry(
                    pair, plus_record, minus_record, plus, minus, tell)
                if (dict(entry) != expected_entry
                        or entry.get("previous_entry_sha256") != expected_journal_previous
                        or event.get("controller_state_after") != final_ingest.controller_state(self.controller)):
                    raise DurableSessionError("journal tell, raw evidence, state, or hash chain does not replay exactly")
                expected_journal_previous = entry["entry_sha256"]
            else:
                raise DurableSessionError(f"unknown durable event type: {event.get('event_type')}")
            self.events.append(event)
            previous_event = event_self
        if not self.events:
            raise DurableSessionError("session is missing its initialization event")
        expected_point_paths = {
            Path(record["point_pin"]["path"]).resolve(strict=True)
            for record in terminal.values()
        }
        actual_point_paths = {path.resolve(strict=True) for path in self._point_entry_files()}
        if actual_point_paths != expected_point_paths:
            raise DurableSessionError(
                "point-entry files do not exactly match event-referenced terminal receipts; preserve for reconciliation")

    def _replay_evaluation(self, proposal: selector.SPSAProposal,
                           terminal: Mapping[tuple[str, str], Mapping[str, Any]],
                           executor: ProposalPointExecutor | None, *,
                           allow_missing: bool = False) -> selector.ProposalEvaluation:
        record = terminal.get((proposal.proposal_id, proposal.side))
        if record is None:
            if allow_missing:
                return self._empty_evaluation(proposal)
            raise DurableSessionError("tell event references a proposal side without terminal readback")
        if record["event"].get("terminal_status") != "COMPLETE":
            return self._empty_evaluation(proposal)
        # The durable point document and its pinned raw NPZs are the replay
        # source.  Never depend on a live executor callback to reconstruct an
        # already terminal scientific evaluation.
        return self._evaluation_for_point(proposal, record)

    def _evaluation_npz_identity(self, proposal: selector.SPSAProposal,
                                 record: Mapping[str, Any]) -> dict[str, Any]:
        point_doc = record["point_doc"]
        raw_map = point_doc.get("raw_npz_by_row")
        expected_count = len(proposal.sample_ids) * len(self.inputs.family_selection.family.roster.model_ids)
        if not isinstance(raw_map, Mapping) or len(raw_map) != expected_count:
            raise DurableSessionError("point terminal receipt lacks the exact candidate NPZ denominator")
        file_table = getattr(self, "_active_files", self.inputs.files)
        checked: dict[str, dict[str, Any]] = {}
        for row_id, raw_pin in raw_map.items():
            if not isinstance(row_id, str) or not isinstance(raw_pin, Mapping):
                raise DurableSessionError("candidate NPZ evidence is malformed")
            if set(raw_pin) != {"path", "bytes", "sha256", "array_refs"}:
                raise DurableSessionError("candidate NPZ reference fields differ")
            matches = [pin for pin in file_table.values()
                       if pin.role == "raw_matrix_npz" and pin.declared_path == raw_pin["path"]
                       and pin.byte_count == raw_pin["bytes"] and pin.sha256 == raw_pin["sha256"]]
            if len(matches) != 1:
                raise DurableSessionError(f"candidate NPZ pin does not resolve uniquely: {row_id}")
            initial_ingest._read_pinned(matches[0])
            checked[row_id] = dict(raw_pin)
        return {"point_entry_pin": record["point_pin"],
                "candidate_row_ids": sorted(checked),
                "candidate_raw_npz_by_row": dict(sorted(checked.items()))}

    def controller_journal_document(self) -> dict[str, Any]:
        entries = [event["journal_entry"] for event in self.events
                   if event.get("event_type") == "PAIR_TOLD"]
        complete = (self.controller.iteration == selector.SPSA_ITERATIONS
                    and self.controller._pending is None and not self.controller.halted
                    and len(entries) == selector.SPSA_ITERATIONS
                    and all(item["tell_result"].get("status") == "UPDATED" for item in entries))
        initial_event = self.events[0]
        document = {
            "schema": JOURNAL_SCHEMA,
            "status": "COMPLETE_20_ASK_TELL_PAIRS" if complete else "INCOMPLETE_OR_HALTED",
            "metric_backend": "torch_cpu",
            "initial_ingest_index": self.manifest["initial_ingest_index"],
            "initial_selector_receipt": self.manifest["initial_selector_receipt"],
            "composed_family_plan": self.manifest["composed_family_plan"],
            "scientific_input_pins_sha256": self.inputs.family_selection.family.input_pins_sha256,
            "scientific_input_pins": dict(self.inputs.family_selection.family.input_pins),
            "seed_method_id": self.inputs.family_selection.spsa_seed_method_id,
            "candidate_artifact_sha256": self.controller.candidate_artifact_sha256,
            "initial_theta": list(self.controller.theta_initial),
            "rng_seed": selector.SPSA_RNG_SEED,
            "initial_controller_state": initial_event["controller_state"],
            "source_snapshot_lineage": _plain(self.inputs.seed_lineage),
            "initial_selection_status": self.inputs.family_selection.status,
            "entries": entries,
        }
        return _seal_document(document, "journal_sha256")

    def write_controller_journal_exclusive(self, path: str | Path) -> dict[str, Any]:
        document = self.controller_journal_document()
        if document["status"] != "COMPLETE_20_ASK_TELL_PAIRS":
            raise DurableSessionError("only an actual complete 20-pair journal can be sealed as terminal")
        target = Path(path)
        raw = (json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False,
                          allow_nan=False) + "\n").encode("utf-8")
        _write_exclusive(target, raw)
        pin = _regular_pin(target, file_id="controller_journal", role="controller_journal")
        return {"path": pin.declared_path, "bytes": pin.byte_count, "sha256": pin.sha256,
                "self_sha256": document["journal_sha256"]}


def _evaluation_matches(evaluation: selector.ProposalEvaluation,
                        proposal: selector.SPSAProposal) -> bool:
    return DurableSPSASession._validate_evaluation_matches(evaluation, proposal)


__all__ = [
    "DurableSPSASession", "DurableSessionError", "INITIAL_RECEIPT_SCHEMA",
    "JOURNAL_SCHEMA", "PointOutcome", "ProposalPointExecutor",
    "UnknownPointOutcome", "produce_initial_selector_receipt",
]
