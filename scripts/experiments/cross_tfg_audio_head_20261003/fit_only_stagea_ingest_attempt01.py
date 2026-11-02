"""Verify three native Stage-A Fit40 score families before selector06 use.

This CPU-side reader does not render, score, train, or access a remote host.
It verifies the producer plan and 40 six-row block ledgers for each model,
then hashes each whole NPZ at its pinned local readback path before loading
the seven official arrays and adapting them with selector06.  Producer paths
remain in the source records; an explicit file table maps those original paths
to local readback files.

The index schema is ``fit40_stagea_selector_ingest_v1``.  It contains:

* ``models``: exactly three entries, each with pinned ``plan`` and
  ``planned_rows`` file IDs and exactly 40 block entries.  Each block entry
  names pinned ``run_status``, ``result_manifest`` and ``results_jsonl`` IDs.
* ``files``: exact ``file_id``, ``role``, ``declared_path``, ``local_path``,
  ``bytes`` and ``sha256`` records.  Roles are plan, planned_rows,
  result_manifest, results_jsonl, raw_matrix_npz, or input_pin.  Snapshot
  input pins additionally carry ``pin_name``.
* ``input_pin_files``: pin-name to input_pin file-ID mappings for the snapshot
  set index and four unselected snapshot records.

The complete 720-row denominator is always formed from the three frozen
240-row plans. Missing result rows stay missing and block selection; producer
failures stay failed rows. No synthetic score rows are invented.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import struct
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.cross_tfg_audio_head_20261003.fit_only_spsa_selection_attempt06 import (
    FitRoster,
    FitScoreRow,
    FitSelectionError,
    MethodSpec,
    SnapshotSpec,
    SPSAController,
    canonical_input_pins_sha256,
    full_fit_candidate_binding_sha256,
    official_stagea_score_row,
    select_snapshot_family,
)


SCHEMA = "fit40_stagea_selector_ingest_v1"
PLAN_SCHEMA = "cross-tfg-audio-eval-v1"
RESULT_SCHEMA = "cross-tfg-audio-eval-v1"
STAGEA_ROW_SCHEMA = "cross_tfg_audio_head_stagea_v1"
METHOD_IDS = (
    "identity",
    "rms_fixed",
    "fit40_native_transition_head_v3_attempt11_step030",
    "fit40_native_transition_head_v3_attempt11_step060",
    "fit40_native_transition_head_v3_attempt11_step090",
    "fit40_native_transition_head_v3_attempt11_step120",
)
REFERENCE_METHOD_IDS = ("identity", "rms_fixed")
SNAPSHOT_STEPS = (30, 60, 90, 120)
SAMPLES_PER_MODEL = 40
ROWS_PER_MODEL = 240
BLOCK_ROWS = 6
BLOCK_COUNT = 40
EXPECTED_TOTAL_ROWS = 720
RENDERER_SEED = 20261003
ARRAY_NAMES = frozenset({
    "query_frame_index",
    "query_timestamp_num",
    "query_timestamp_den",
    "track_frame_index",
    "valid_query_mask",
    "distance_by_query_lag",
    "lag_offsets_frames",
})


class IngestError(ValueError):
    """Invalid index or source plan: no selector result is produced."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(data: bytes) -> str:
    return sha256(data).hexdigest()


def _is_sha(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(ch in "0123456789abcdef" for ch in value))


def _self_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    actual = _digest(_canonical(unsigned))
    if not _is_sha(declared) or declared != actual:
        raise IngestError(f"{field} self-hash mismatch")
    return actual


def _payload_hash(value: Mapping[str, Any], field: str) -> str:
    unsigned = dict(value)
    declared = unsigned.pop(field, None)
    actual = _digest(_canonical(unsigned))
    if not _is_sha(declared) or declared != actual:
        raise IngestError(f"{field} payload hash mismatch")
    return actual


def _read_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # retain a concise boundary error
        raise IngestError(f"cannot read JSON {path}: {type(exc).__name__}: {exc}") from exc
    if not isinstance(value, dict):
        raise IngestError(f"JSON document is not an object: {path}")
    return value, raw


@dataclass(frozen=True)
class _PinnedFile:
    file_id: str
    role: str
    declared_path: str
    local_path: str
    byte_count: int
    sha256: str
    pin_name: str | None = None


@dataclass(frozen=True)
class FamilyInputs:
    rows: tuple[FitScoreRow, ...]
    roster: FitRoster
    input_pins: Mapping[str, Mapping[str, Any]]
    input_pins_sha256: str
    score_protocol_sha256: str
    reference_specs: tuple[MethodSpec, MethodSpec]
    candidates: tuple[SnapshotSpec, SnapshotSpec, SnapshotSpec, SnapshotSpec]
    expected_rows: int
    actual_rows: int
    missing_row_ids: tuple[str, ...]
    producer_failure_rows: tuple[str, ...]
    input_artifact_failures: tuple[str, ...]
    verified_npz_files: int
    score_npz_bindings: tuple[Mapping[str, Any], ...]
    source_file_pins: tuple[Mapping[str, Any], ...]
    integrity_errors: tuple[str, ...]

    @property
    def complete_transport(self) -> bool:
        return (not self.integrity_errors and not self.input_artifact_failures
                and not self.missing_row_ids and self.actual_rows == self.expected_rows)


@dataclass(frozen=True)
class FamilySelection:
    family: FamilyInputs
    selection: Any | None
    status: str
    spsa_seed_method_id: str | None
    errors: tuple[str, ...]


def _file_table(index: Mapping[str, Any]) -> dict[str, _PinnedFile]:
    raw_files = index.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise IngestError("index files must be a non-empty list")
    out: dict[str, _PinnedFile] = {}
    declared_roles: set[tuple[str, str]] = set()
    for item in raw_files:
        if not isinstance(item, Mapping):
            raise IngestError("index file entry is not an object")
        file_id, role = item.get("file_id"), item.get("role")
        declared_path, local_path = item.get("declared_path"), item.get("local_path")
        size, digest = item.get("bytes"), item.get("sha256")
        if (not isinstance(file_id, str) or not file_id or file_id in out
                or not isinstance(role, str) or not role
                or not isinstance(declared_path, str) or not declared_path
                or not isinstance(local_path, str) or not local_path
                or type(size) is not int or size < 0 or not _is_sha(digest)):
            raise IngestError("index file pin has malformed/duplicate identity or byte pin")
        pair = (role, declared_path)
        if pair in declared_roles:
            raise IngestError(f"duplicate role/declaration mapping: {pair}")
        declared_roles.add(pair)
        out[file_id] = _PinnedFile(file_id, role, declared_path, local_path, size,
                                   digest, item.get("pin_name"))
    return out


def _read_pinned(file_pin: _PinnedFile) -> bytes:
    path = Path(file_pin.local_path)
    if not path.is_absolute():
        raise IngestError(f"local readback path must be absolute: {file_pin.file_id}")
    if path.is_symlink():
        raise IngestError(f"local readback file cannot be a symlink: {file_pin.file_id}")
    try:
        raw = path.read_bytes()
    except Exception as exc:
        raise IngestError(f"cannot read pinned file {file_pin.file_id}: {exc}") from exc
    if len(raw) != file_pin.byte_count or _digest(raw) != file_pin.sha256:
        raise IngestError(f"local readback differs from exact file pin: {file_pin.file_id}")
    return raw


def _required_file(files: Mapping[str, _PinnedFile], file_id: Any, role: str) -> _PinnedFile:
    if not isinstance(file_id, str) or file_id not in files:
        raise IngestError(f"missing file ID for required role {role}")
    result = files[file_id]
    if result.role != role:
        raise IngestError(f"file {file_id} role is {result.role}, expected {role}")
    return result


def _parse_jsonl(raw: bytes, label: str) -> list[dict[str, Any]]:
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise IngestError(f"{label} is not UTF-8") from exc
    if any(not line.strip() for line in lines):
        raise IngestError(f"{label} contains a blank line")
    parsed: list[dict[str, Any]] = []
    for number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except Exception as exc:
            raise IngestError(f"{label} line {number} is invalid JSON") from exc
        if not isinstance(value, dict):
            raise IngestError(f"{label} line {number} is not an object")
        parsed.append(value)
    return parsed


def _row_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row.get(k) for k in (
        "row_id", "split", "sample_id", "speaker_id", "model_id", "face_id",
        "seed", "method_id", "method_config_sha256",
    ))


def _model_descriptor(plan: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    descriptors = plan.get("models")
    if not isinstance(descriptors, list) or len(descriptors) != 1:
        raise IngestError("Stage-A plan must contain exactly one model descriptor")
    descriptor = descriptors[0]
    if not isinstance(descriptor, Mapping) or descriptor.get("model_id") != model_id:
        raise IngestError("Stage-A model descriptor does not match its index model_id")
    if (not isinstance(descriptor.get("audio_frontend"), str)
            or not descriptor["audio_frontend"]
            or not _is_sha(descriptor.get("model_sha256"))
            or not _is_sha(descriptor.get("score_model_sha256"))):
        raise IngestError("Stage-A model descriptor lacks frontend/model/checkpoint identity pins")
    generator = descriptor.get("generator")
    if isinstance(generator, Mapping):
        checkpoint_sha = generator.get("checkpoint_sha256")
        if checkpoint_sha is not None and not _is_sha(checkpoint_sha):
            raise IngestError("Stage-A generator descriptor has an invalid checkpoint SHA-256")
    return descriptor


def _score_protocol(plan: Mapping[str, Any]) -> Mapping[str, Any]:
    protocol = plan.get("protocol")
    syncnet = protocol.get("syncnet") if isinstance(protocol, Mapping) else None
    roi = protocol.get("roi") if isinstance(protocol, Mapping) else None
    expected_lags = list(range(-15, 16))
    if not isinstance(syncnet, Mapping) or not isinstance(roi, Mapping):
        raise IngestError("Stage-A plan lacks its frozen SyncNet/ROI protocol")
    if (syncnet.get("anchor_column") != 15
            or syncnet.get("zero_lag_column") != 15
            or syncnet.get("lag_offsets_frames") != expected_lags):
        raise IngestError("Stage-A plan score anchor/zero-lag/lag axis differs from 15/-15..15")
    if (syncnet.get("score_audio_estimand") != "native_roi_original_pcm16"
            or roi.get("mode") != "native_official"
            or roi.get("estimand") != "native_roi_original_pcm16"):
        raise IngestError("Stage-A plan does not use the frozen native original-PCM score protocol")
    if (not isinstance(syncnet.get("score_model_id"), str)
            or not syncnet["score_model_id"]
            or not _is_sha(syncnet.get("score_model_sha256"))
            or not isinstance(syncnet.get("source_sha256"), Mapping)
            or not syncnet["source_sha256"]
            or any(not isinstance(key, str) or not _is_sha(value)
                   for key, value in syncnet["source_sha256"].items())
            or not isinstance(syncnet.get("primary_audio_input"), Mapping)
            or not isinstance(syncnet.get("score_reduction"), Mapping)):
        raise IngestError("Stage-A SyncNet protocol lacks model/source/audio/reduction identity")
    if (not isinstance(syncnet["primary_audio_input"].get("code_sha256"), str)
            or not _is_sha(syncnet["primary_audio_input"].get("code_sha256"))
            or not isinstance(syncnet["primary_audio_input"].get("ffmpeg_sha256"), str)
            or not _is_sha(syncnet["primary_audio_input"].get("ffmpeg_sha256"))):
        raise IngestError("Stage-A primary audio input lacks its source code pin")
    return syncnet


def _source_identity(row: Mapping[str, Any]) -> tuple[Any, ...]:
    natural = row.get("natural_pcm")
    face = row.get("source_face")
    video = row.get("source_video")
    if not all(isinstance(x, Mapping) for x in (natural, face, video)):
        raise IngestError("planned row lacks natural/source media metadata")
    return (
        row.get("sample_id"), row.get("speaker_id"), row.get("face_id"),
        natural.get("pcm_sha256"), row.get("source_media_kind"),
        video.get("file_sha256"), face.get("file_sha256"),
    )


def _validate_plan(plan: Mapping[str, Any], raw_plan: bytes, planned_rows_raw: bytes,
                   model_id: str, descriptor: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    if plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "FROZEN":
        raise IngestError(f"{model_id} plan is not the frozen Stage-A producer plan")
    _payload_hash(plan, "plan_sha256")
    if plan.get("scientific_input_pins_sha256") != _digest(_canonical(plan.get("scientific_input_pins"))):
        raise IngestError(f"{model_id} plan scientific input pins digest is invalid")
    if not isinstance(plan.get("scientific_input_pins"), Mapping):
        raise IngestError(f"{model_id} plan lacks scientific input pins")
    if canonical_input_pins_sha256(plan["scientific_input_pins"]) != plan["scientific_input_pins_sha256"]:
        raise IngestError(f"{model_id} plan input-pin canonicalization differs from selector contract")
    if type(plan.get("rows")) is not list or len(plan["rows"]) != ROWS_PER_MODEL:
        raise IngestError(f"{model_id} plan must contain exactly 240 physical rows")
    if plan.get("run_id") is None:
        raise IngestError(f"{model_id} plan lacks run_id")
    _score_protocol(plan)
    planned_rows = _parse_jsonl(planned_rows_raw, f"{model_id} planned_rows")
    if len(planned_rows) != ROWS_PER_MODEL or planned_rows != plan["rows"]:
        raise IngestError(f"{model_id} planned_rows JSONL differs from the sealed plan rows")
    methods = plan.get("methods")
    if not isinstance(methods, list) or len(methods) != len(METHOD_IDS):
        raise IngestError(f"{model_id} plan must define the exact six methods")
    method_map: dict[str, dict[str, Any]] = {}
    for method in methods:
        if not isinstance(method, Mapping):
            raise IngestError(f"{model_id} plan has a malformed method record")
        method_id = method.get("method_id")
        config = method.get("config")
        config_sha = method.get("config_sha256")
        if (method_id not in METHOD_IDS or method_id in method_map
                or not isinstance(config, Mapping)
                or config_sha != _digest(_canonical(config))):
            raise IngestError(f"{model_id} method config hash/identity mismatch")
        method_map[method_id] = dict(method)
    if tuple(m["method_id"] for m in methods) != METHOD_IDS:
        raise IngestError(f"{model_id} plan method order differs from the frozen six-method family")
    model_ids_in_rows = {row.get("model_id") for row in planned_rows}
    if model_ids_in_rows != {model_id}:
        raise IngestError(f"{model_id} planned rows contain a different model identity")
    expected_method_config = {mid: method_map[mid]["config_sha256"] for mid in METHOD_IDS}
    seen_context_methods: set[tuple[str, str]] = set()
    for row in planned_rows:
        if (row.get("schema_version") not in (None, STAGEA_ROW_SCHEMA)
                or row.get("split") != "fit" or row.get("seed") != RENDERER_SEED
                or row.get("method_id") not in expected_method_config
                or row.get("method_config_sha256") != expected_method_config[row["method_id"]]):
            raise IngestError(f"{model_id} planned row has wrong split/seed/method config")
        if (row.get("audio_frontend") != descriptor.get("audio_frontend")
                or row.get("checkpoint_sha256") is not None
                and row.get("checkpoint_sha256") not in {
                    descriptor.get("model_sha256"),
                    (descriptor.get("generator") or {}).get("checkpoint_sha256")
                    if isinstance(descriptor.get("generator"), Mapping) else None,
                }):
            raise IngestError(f"{model_id} planned row frontend/checkpoint differs from its model descriptor")
        key = (row.get("sample_id"), row.get("method_id"))
        if key in seen_context_methods:
            raise IngestError(f"{model_id} plan duplicates sample/method row {key}")
        seen_context_methods.add(key)
    if len(seen_context_methods) != SAMPLES_PER_MODEL * len(METHOD_IDS):
        raise IngestError(f"{model_id} plan does not cover all 40×6 method slots")
    if type(plan.get("generation_seed_family")) not in (dict, list):
        raise IngestError(f"{model_id} plan lacks seed-family metadata")
    return planned_rows, method_map


def _load_snapshot_points(
    index: Mapping[str, Any], files: Mapping[str, _PinnedFile],
    input_pins: Mapping[str, Mapping[str, Any]], method_map: Mapping[str, Mapping[str, Any]],
) -> tuple[SnapshotSpec, SnapshotSpec, SnapshotSpec, SnapshotSpec]:
    pin_files = index.get("input_pin_files")
    if not isinstance(pin_files, Mapping):
        raise IngestError("index must pin the snapshot index and each snapshot record")
    required_pin_names = ("snapshot_index", "snapshot_step_030", "snapshot_step_060",
                          "snapshot_step_090", "snapshot_step_120")
    contents: dict[str, bytes] = {}
    for pin_name in required_pin_names:
        pin = input_pins.get(pin_name)
        if not isinstance(pin, Mapping):
            raise IngestError(f"common scientific pins lack {pin_name}")
        file_pin = _required_file(files, pin_files.get(pin_name), "input_pin")
        if (file_pin.pin_name != pin_name or file_pin.declared_path != pin.get("path")
                or file_pin.byte_count != pin.get("bytes") or file_pin.sha256 != pin.get("sha256")):
            raise IngestError(f"materialized input pin differs from common plan: {pin_name}")
        contents[pin_name] = _read_pinned(file_pin)
    snapshot_index = json.loads(contents["snapshot_index"].decode("utf-8"))
    if not isinstance(snapshot_index, dict) or snapshot_index.get("schema") != "cross_tfg_fit40_snapshot_set_index_v1":
        raise IngestError("snapshot input pin has unexpected schema")
    _payload_hash(snapshot_index, "index_sha256")
    if snapshot_index.get("status") != "EXTRACTED_UNSELECTED_NATIVE_ENDPOINT_UNBOUND":
        raise IngestError("snapshot set is not the frozen unselected Fit-only extraction")
    entries = snapshot_index.get("snapshots")
    if not isinstance(entries, list) or len(entries) != 4:
        raise IngestError("snapshot index must contain exactly four predeclared snapshots")
    expected_methods = [METHOD_IDS[2], METHOD_IDS[3], METHOD_IDS[4], METHOD_IDS[5]]
    index_pin_path = PurePosixPath(str(input_pins["snapshot_index"].get("path", "")))
    if (index_pin_path.is_absolute() or not index_pin_path.parts
            or ".." in index_pin_path.parts):
        raise IngestError("snapshot index pin path must be a safe project-relative path")
    specs: list[SnapshotSpec] = []
    for entry, step, method_id, pin_name in zip(entries, SNAPSHOT_STEPS, expected_methods,
                                                 required_pin_names[1:]):
        if (not isinstance(entry, Mapping) or entry.get("method_id") != method_id
                or entry.get("source_optimizer_step") != step):
            raise IngestError("snapshot index ordering/method/step differs from the fixed family")
        raw = contents[pin_name]
        item = json.loads(raw.decode("utf-8"))
        pin = input_pins[pin_name]
        entry_path = PurePosixPath(str(entry.get("path", "")))
        pin_path = PurePosixPath(str(pin.get("path", "")))
        if (entry_path.is_absolute() or not entry_path.parts or ".." in entry_path.parts
                or pin_path.is_absolute() or not pin_path.parts or ".." in pin_path.parts
                or (index_pin_path.parent / entry_path).as_posix() != pin_path.as_posix()
                or entry.get("bytes") != pin.get("bytes")
                or entry.get("sha256") != pin.get("sha256")):
            raise IngestError(f"snapshot index file reference differs from the materialized pin: {method_id}")
        if (not isinstance(item, dict) or item.get("schema") != "cross_tfg_fit40_parameter_snapshot_v1"
                or _digest(raw) != pin.get("sha256") or item.get("method_id") != method_id
                or item.get("source_optimizer_step") != step
                or item.get("snapshot_sha256") != entry.get("snapshot_sha256")
                or item.get("training_family") != "fit40_native_transition_head_v3_attempt11"
                or item.get("selection_status") != "NOT_SELECTED"
                or item.get("optimizer_step_is_selected") is not False):
            raise IngestError(f"snapshot record identity/status is not source-pinned: {method_id}")
        param = item.get("parameter")
        if not isinstance(param, Mapping) or param.get("shape") != [3, 3]:
            raise IngestError(f"snapshot does not contain the expected 3×3 head parameter: {method_id}")
        values = param.get("values")
        if (not isinstance(values, list) or len(values) != 3
                or any(not isinstance(row, list) or len(row) != 3 for row in values)):
            raise IngestError(f"snapshot theta has the wrong shape: {method_id}")
        theta = tuple(float(value) for row in values for value in row)
        if not all(math.isfinite(value) for value in theta):
            raise IngestError(f"snapshot theta has non-finite entries: {method_id}")
        little_endian = struct.pack("<9f", *theta)
        if _digest(little_endian) != param.get("little_endian_f32_sha256"):
            raise IngestError(f"snapshot theta does not match its float32 payload pin: {method_id}")
        config = method_map[method_id].get("config", {})
        if (config.get("method") != "precomputed_fit_snapshot_waveform"
                or config.get("source_optimizer_step") != step
                or config.get("selection_status") != "NOT_SELECTED"
                or config.get("snapshot_sha256") != item.get("snapshot_sha256")):
            raise IngestError(f"Stage-A method config does not bind snapshot artifact: {method_id}")
        artifact_sha = item.get("snapshot_sha256")
        binding_sha = full_fit_candidate_binding_sha256(
            method_id=method_id, artifact_sha256=artifact_sha,
            input_pins_sha256=canonical_input_pins_sha256(input_pins),
            source_step=step, theta=theta, point_kind="snapshot",
        )
        specs.append(SnapshotSpec(method_id, artifact_sha, binding_sha, step, theta, "snapshot"))
    return tuple(specs)  # type: ignore[return-value]


def _reference_specs(method_map: Mapping[str, Mapping[str, Any]], input_sha: str) -> tuple[MethodSpec, MethodSpec]:
    out: list[MethodSpec] = []
    for method_id in REFERENCE_METHOD_IDS:
        method = method_map[method_id]
        config = method["config"]
        if method_id == "identity" and config.get("method") != "identity":
            raise IngestError("identity method config does not identify identity")
        if method_id == "rms_fixed" and config.get("method") != "rms":
            raise IngestError("rms_fixed method config does not identify RMS")
        artifact_sha = method["config_sha256"]
        binding_sha = _digest(_canonical({
            "schema": "fit40_reference_method_binding_v1",
            "method_id": method_id,
            "artifact_sha256": artifact_sha,
            "input_pins_sha256": input_sha,
            "renderer_seed": RENDERER_SEED,
        }))
        out.append(MethodSpec(method_id, artifact_sha, binding_sha))
    return out[0], out[1]


def _load_arrays(npz_pin: _PinnedFile, raw_ref: Mapping[str, Any]) -> dict[str, np.ndarray]:
    if raw_ref.get("path") != npz_pin.declared_path:
        raise IngestError("declared producer NPZ path does not match the explicit local restore map")
    if raw_ref.get("file_sha256") != npz_pin.sha256:
        raise IngestError("whole-NPZ producer SHA differs from the readback mapping")
    raw = _read_pinned(npz_pin)
    if _digest(raw) != raw_ref.get("file_sha256"):
        raise IngestError("whole-NPZ bytes differ from the producer's raw_matrix_npz pin")
    try:
        with np.load(BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != ARRAY_NAMES:
                raise IngestError("raw Stage-A NPZ member set differs from the seven-array contract")
            arrays = {name: np.array(archive[name], copy=True) for name in archive.files}
    except IngestError:
        raise
    except Exception as exc:
        raise IngestError(f"cannot load verified NPZ: {type(exc).__name__}: {exc}") from exc
    return arrays


def _validate_record_against_plan(
    record: Mapping[str, Any], plan_row: Mapping[str, Any],
    descriptor: Mapping[str, Any], syncnet_protocol: Mapping[str, Any],
) -> None:
    if record.get("schema_version") != STAGEA_ROW_SCHEMA:
        raise IngestError("result ledger row has an unexpected Stage-A row schema")
    identity = (record.get("row_id"), record.get("split"), record.get("sample_id"),
                record.get("speaker_id"), record.get("model_id"), record.get("face_id"),
                record.get("seed"), record.get("method_id"), record.get("method_config_sha256"))
    if identity != _row_identity(plan_row):
        raise IngestError(f"result ledger row identity/config differs from plan: {record.get('row_id')}")
    planned_checkpoint = plan_row.get("checkpoint_sha256")
    declared_checkpoint_sha = descriptor.get("model_sha256")
    generator = descriptor.get("generator")
    if isinstance(generator, Mapping) and generator.get("checkpoint_sha256") is not None:
        declared_checkpoint_sha = generator.get("checkpoint_sha256")
    if (record.get("audio_frontend") != descriptor.get("audio_frontend")
            or record.get("audio_frontend") != plan_row.get("audio_frontend")
            or record.get("checkpoint_sha256") != planned_checkpoint
            or (record.get("checkpoint_sha256") is not None
                and record.get("checkpoint_sha256") != declared_checkpoint_sha)):
        raise IngestError(f"result frontend/checkpoint/score-model identity differs from plan: {record.get('row_id')}")
    natural = plan_row.get("natural_pcm")
    face = plan_row.get("source_face")
    video = plan_row.get("source_video")
    if (not isinstance(natural, Mapping) or not isinstance(face, Mapping)
            or not isinstance(video, Mapping)):
        raise IngestError("planned row lacks source identity fields")
    if descriptor.get("score_model_sha256") != syncnet_protocol.get("score_model_sha256"):
        raise IngestError(f"plan descriptor and SyncNet score model pin disagree: {record.get('row_id')}")
    if record.get("status") == "PASS":
        score = record.get("score")
        audio_input = syncnet_protocol.get("primary_audio_input")
        primary_input = score.get("primary_syncnet_input") if isinstance(score, Mapping) else None
        if (not isinstance(score, Mapping)
                or score.get("score_model_sha256") != syncnet_protocol.get("score_model_sha256")
                or score.get("preprocessing_sha256") != audio_input.get("code_sha256")
                or not isinstance(primary_input, Mapping)
                or primary_input.get("audio_codec") != audio_input.get("codec")
                or primary_input.get("channels") != audio_input.get("channels")
                or primary_input.get("sample_rate_hz") != audio_input.get("sample_rate_hz")):
            raise IngestError(f"result score model/audio path differs from frozen protocol: {record.get('row_id')}")
    if (record.get("natural_pcm_sha256") != natural.get("pcm_sha256")
            or record.get("score_audio_sha256") != natural.get("pcm_sha256")
            or record.get("source_media_kind") != plan_row.get("source_media_kind")
            or record.get("source_video_sha256") != video.get("file_sha256")
            or record.get("source_face_sha256") != face.get("file_sha256")):
        raise IngestError(f"result ledger row source identity differs from plan: {record.get('row_id')}")


def _validate_block_counts(manifest: Mapping[str, Any], records: Sequence[Mapping[str, Any]], model_id: str) -> None:
    expected: dict[tuple[str, str], dict[str, int]] = {}
    aggregate = {"PASS": 0, "FAILED": 0, "INCOMPLETE": 0}
    for row in records:
        status = row.get("status")
        method = row.get("method_id")
        if status not in aggregate:
            raise IngestError(f"block row has unsupported producer status: {status}")
        key = (method, model_id)
        counts = expected.setdefault(key, {"PASS": 0, "FAILED": 0, "INCOMPLETE": 0})
        counts[status] += 1
        aggregate[status] += 1
    raw_counts = manifest.get("counts_by_model_method")
    if not isinstance(raw_counts, list):
        raise IngestError("block manifest lacks counts_by_model_method")
    actual: dict[tuple[str, str], dict[str, int]] = {}
    for item in raw_counts:
        if not isinstance(item, Mapping):
            raise IngestError("block manifest counts record is malformed")
        key = (item.get("method_id"), item.get("model_id"))
        counts = item.get("status_counts")
        if key in actual or not isinstance(counts, Mapping):
            raise IngestError("block manifest counts contain duplicate/malformed method entry")
        actual[key] = {name: counts.get(name, 0) for name in ("PASS", "FAILED", "INCOMPLETE")}
        if item.get("planned_rows") != 1 or item.get("result_rows") != expected.get(key, {}).get("PASS", 0) + expected.get(key, {}).get("FAILED", 0) + expected.get(key, {}).get("INCOMPLETE", 0):
            raise IngestError("block method result counts differ from ledger rows")
    # A one-row block may omit a method slot only if the actual producer emits
    # a result row; missing ledger rows are checked against the frozen plan.
    if set(actual) != {(mid, model_id) for mid in METHOD_IDS}:
        raise IngestError("block manifest method/model counts differ from the six-method plan")
    for key in actual:
        counts = expected.get(key, {"PASS": 0, "FAILED": 0, "INCOMPLETE": 0})
        if actual[key] != counts:
            raise IngestError(f"block status counts disagree with its ledger for {key}")
    manifest_aggregate = manifest.get("status_counts")
    if (not isinstance(manifest_aggregate, Mapping)
            or {key: manifest_aggregate.get(key, 0) for key in aggregate} != aggregate):
        raise IngestError("block aggregate status counts disagree with ledger rows")


def _parse_model_family(
    model_entry: Mapping[str, Any], files: Mapping[str, _PinnedFile],
) -> tuple[str, Mapping[str, Any], bytes, list[dict[str, Any]], dict[str, dict[str, Any]],
           list[dict[str, Any]], list[str], set[str]]:
    model_id = model_entry.get("model_id")
    if not isinstance(model_id, str) or not model_id:
        raise IngestError("model entry lacks an exact model_id")
    plan_pin = _required_file(files, model_entry.get("plan_file_id"), "plan")
    planned_pin = _required_file(files, model_entry.get("planned_rows_file_id"), "planned_rows")
    plan_raw = _read_pinned(plan_pin)
    planned_raw = _read_pinned(planned_pin)
    try:
        plan = json.loads(plan_raw.decode("utf-8"))
    except Exception as exc:
        raise IngestError(f"invalid model plan JSON: {model_id}") from exc
    if not isinstance(plan, Mapping):
        raise IngestError(f"model plan is not an object: {model_id}")
    descriptor = _model_descriptor(plan, model_id)
    syncnet_protocol = _score_protocol(plan)
    if descriptor.get("score_model_sha256") != syncnet_protocol.get("score_model_sha256"):
        raise IngestError(f"model descriptor and SyncNet protocol score-model pins differ: {model_id}")
    planned_rows, method_map = _validate_plan(plan, plan_raw, planned_raw, model_id, descriptor)
    blocks = model_entry.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != BLOCK_COUNT:
        raise IngestError(f"{model_id} must provide exactly forty block entries")
    if [x.get("block_index") if isinstance(x, Mapping) else None for x in blocks] != list(range(BLOCK_COUNT)):
        raise IngestError(f"{model_id} block entries are not the ordered 0..39 family")
    records_all: list[dict[str, Any]] = []
    errors: list[str] = []
    manifest_ids_used: set[str] = set()
    ledger_ids_used: set[str] = set()
    for block_index, block in enumerate(blocks):
        status_pin = _required_file(files, block.get("run_status_file_id"), "run_status")
        manifest_pin = _required_file(files, block.get("result_manifest_file_id"), "result_manifest")
        ledger_pin = _required_file(files, block.get("results_file_id"), "results_jsonl")
        if (status_pin.file_id in manifest_ids_used or manifest_pin.file_id in manifest_ids_used
                or ledger_pin.file_id in ledger_ids_used):
            raise IngestError(f"{model_id} repeats a status, manifest or results file ID")
        manifest_ids_used.add(status_pin.file_id)
        manifest_ids_used.add(manifest_pin.file_id)
        ledger_ids_used.add(ledger_pin.file_id)
        status_raw = _read_pinned(status_pin)
        manifest_raw = _read_pinned(manifest_pin)
        ledger_raw = _read_pinned(ledger_pin)
        try:
            run_status = json.loads(status_raw.decode("utf-8"))
        except Exception as exc:
            raise IngestError(f"invalid run status JSON: {model_id}/{block_index}") from exc
        if not isinstance(run_status, Mapping):
            raise IngestError(f"run status is not an object: {model_id}/{block_index}")
        try:
            manifest = json.loads(manifest_raw.decode("utf-8"))
        except Exception as exc:
            raise IngestError(f"invalid block manifest JSON: {model_id}/{block_index}") from exc
        if not isinstance(manifest, Mapping):
            raise IngestError(f"block manifest is not an object: {model_id}/{block_index}")
        _payload_hash(manifest, "manifest_sha256")
        if (manifest.get("schema_version") != RESULT_SCHEMA
                or manifest.get("run_id") != plan.get("run_id")
                or manifest.get("plan_raw_sha256") != plan_pin.sha256
                or manifest.get("plan_sha256") != plan.get("plan_sha256")
                or manifest.get("parent_planned_rows") != ROWS_PER_MODEL
                or manifest.get("planned_rows") != ROWS_PER_MODEL
                or manifest.get("block_index") != block_index
                or manifest.get("block_start_index") != block_index * BLOCK_ROWS
                or manifest.get("block_rows") != BLOCK_ROWS
                or manifest.get("status") != "BLOCK_COMPLETE"
                or manifest.get("result_rows") < 0
                or manifest.get("result_rows") > BLOCK_ROWS
                or manifest.get("results_jsonl_sha256") != _digest(ledger_raw)):
            raise IngestError(f"block manifest lineage/counts do not bind source bytes: {model_id}/{block_index}")
        if (run_status.get("status") != "BLOCK_COMPLETE"
                or run_status.get("run_id") != plan.get("run_id")
                or run_status.get("plan_path") != plan_pin.declared_path
                or run_status.get("plan_raw_sha256") != plan_pin.sha256
                or run_status.get("planned_rows_path") != planned_pin.declared_path
                or run_status.get("planned_rows_sha256") != planned_pin.sha256
                or run_status.get("planned_rows") != ROWS_PER_MODEL
                or run_status.get("completed_rows") != BLOCK_ROWS
                or run_status.get("block_index") != block_index
                or run_status.get("block_start_index") != block_index * BLOCK_ROWS
                or run_status.get("block_rows") != BLOCK_ROWS
                or run_status.get("result_manifest_sha256") != _digest(manifest_raw)
                or run_status.get("status_counts") != manifest.get("status_counts")):
            raise IngestError(f"run status does not confirm terminal block lineage: {model_id}/{block_index}")
        records = _parse_jsonl(ledger_raw, f"{model_id} block {block_index} results")
        if len(records) != manifest.get("result_rows"):
            raise IngestError(f"block results count differs from manifest: {model_id}/{block_index}")
        _validate_block_counts(manifest, records, model_id)
        expected_slice = planned_rows[block_index * BLOCK_ROWS:(block_index + 1) * BLOCK_ROWS]
        expected_by_id = {row["row_id"]: (idx, row) for idx, row in enumerate(expected_slice)}
        prior = -1
        for record in records:
            row_id = record.get("row_id")
            if row_id not in expected_by_id:
                errors.append(f"unexpected planned row in {model_id} block {block_index}: {row_id}")
                continue
            ordinal, plan_row = expected_by_id[row_id]
            if ordinal <= prior:
                errors.append(f"block row order differs from sealed plan: {model_id}/{row_id}")
                continue
            prior = ordinal
            try:
                _validate_record_against_plan(record, plan_row, descriptor, syncnet_protocol)
            except IngestError as exc:
                errors.append(f"{model_id}/{row_id}: {exc}")
                continue
            records_all.append(record)
    return (model_id, plan, plan_raw, planned_rows, method_map, records_all,
            errors, manifest_ids_used | ledger_ids_used)


def ingest_family(index_path: str | Path, *, metric_backend: str = "torch_cpu") -> FamilySelection:
    """Verify and adapt a complete three-model 720-slot source index.

    Partial or failed producer evidence can only yield a non-admitted selector
    result. Integrity failures are listed and no selection is attempted.
    """
    index_path = Path(index_path)
    index, index_raw = _read_json(index_path)
    if index.get("schema") != SCHEMA:
        raise IngestError("index schema is not fit40_stagea_selector_ingest_v1")
    _self_hash(index, "index_sha256")
    if index.get("status") != "READY_FOR_LOCAL_INGEST_NOT_SELECTION":
        raise IngestError("index is not the sealed preselection readback packet")
    files = _file_table(index)
    model_entries = index.get("models")
    if not isinstance(model_entries, list) or len(model_entries) != 3:
        raise IngestError("index must contain exactly three model families")
    family_parts = [_parse_model_family(entry, files) for entry in model_entries]
    model_ids = tuple(part[0] for part in family_parts)
    if len(set(model_ids)) != 3:
        raise IngestError("three model families must have distinct model IDs")
    plan_by_model = {part[0]: part[1] for part in family_parts}
    planned_by_model = {part[0]: part[3] for part in family_parts}
    methods_by_model = {part[0]: part[4] for part in family_parts}
    records_by_model = {part[0]: part[5] for part in family_parts}
    integrity_errors = [err for part in family_parts for err in part[6]]
    reference_model = model_ids[0]
    reference_plan = plan_by_model[reference_model]
    input_pins = reference_plan.get("scientific_input_pins")
    input_sha = canonical_input_pins_sha256(input_pins)
    if (input_sha != reference_plan.get("scientific_input_pins_sha256")
            or index.get("scientific_input_pins_sha256") != input_sha):
        raise IngestError("index/common plan scientific input pin identity differs")
    source_identity_by_sample: dict[str, tuple[Any, ...]] = {}
    score_protocol_hashes: set[str] = set()
    for model_id in model_ids:
        plan = plan_by_model[model_id]
        score_protocol_hashes.add(_digest(_canonical(_score_protocol(plan))))
        if (plan.get("scientific_input_pins_sha256") != input_sha
                or plan.get("scientific_input_pins") != input_pins):
            integrity_errors.append(f"{model_id} does not share exact scientific input pins")
        if tuple(m["method_id"] for m in plan.get("methods", [])) != METHOD_IDS:
            integrity_errors.append(f"{model_id} method roster differs from the frozen six methods")
        if methods_by_model[model_id] != methods_by_model[reference_model]:
            integrity_errors.append(f"{model_id} method configs differ from the common family")
        if len(planned_by_model[model_id]) != ROWS_PER_MODEL:
            integrity_errors.append(f"{model_id} plan row count differs from 240")
        by_sample: dict[str, tuple[Any, ...]] = {}
        for row in planned_by_model[model_id]:
            identity = _source_identity(row)
            sample = row.get("sample_id")
            prior = by_sample.setdefault(sample, identity)
            if prior != identity:
                integrity_errors.append(f"{model_id} planned source identity differs across methods: {sample}")
        if not source_identity_by_sample:
            source_identity_by_sample = by_sample
        elif by_sample != source_identity_by_sample:
            integrity_errors.append(f"{model_id} source roster/input identity differs across models")
    if len(score_protocol_hashes) != 1:
        integrity_errors.append("model plans do not share one exact SyncNet score protocol")
    score_protocol_sha = next(iter(score_protocol_hashes)) if len(score_protocol_hashes) == 1 else ""
    sample_to_speaker = {sample: identity[1] for sample, identity in source_identity_by_sample.items()}
    roster = FitRoster(sample_to_speaker, model_ids)
    expected_row_ids: set[str] = set()
    for model_id in model_ids:
        for row in planned_by_model[model_id]:
            if row.get("row_id") in expected_row_ids:
                integrity_errors.append(f"duplicate planned row ID: {row.get('row_id')}")
            expected_row_ids.add(row.get("row_id"))
    if len(expected_row_ids) != EXPECTED_TOTAL_ROWS:
        integrity_errors.append(f"expected 720 unique plan row IDs, got {len(expected_row_ids)}")
    if index.get("expected_row_count") != EXPECTED_TOTAL_ROWS:
        integrity_errors.append("index does not retain the frozen 720-row family denominator")
    input_pin_files = index.get("input_pin_files")
    if not isinstance(input_pin_files, Mapping):
        raise IngestError("index lacks materialized snapshot pin references")
    try:
        candidates = _load_snapshot_points(index, files, input_pins,
                                          methods_by_model[reference_model])
        references = _reference_specs(methods_by_model[reference_model], input_sha)
    except Exception as exc:
        raise IngestError(f"snapshot/reference specs could not be bound: {exc}") from exc
    raw_by_declared: dict[str, _PinnedFile] = {}
    for pin in files.values():
        if pin.role == "raw_matrix_npz":
            if pin.declared_path in raw_by_declared:
                integrity_errors.append(f"duplicate NPZ declared path mapping: {pin.declared_path}")
            raw_by_declared[pin.declared_path] = pin
    used_npz_ids: set[str] = set()
    verified_npz_ids: set[str] = set()
    artifact_failures: list[str] = []
    converted: list[FitScoreRow] = []
    producer_failures: list[str] = []
    score_npz_bindings: list[Mapping[str, Any]] = []
    observed_row_ids: set[str] = set()
    plan_lookup: dict[str, Mapping[str, Any]] = {
        row["row_id"]: row for model_id in model_ids for row in planned_by_model[model_id]
    }
    spec_by_method: dict[str, MethodSpec] = {s.method_id: s for s in references}
    spec_by_method.update({s.method_id: s for s in candidates})
    for model_id in model_ids:
        for record in records_by_model[model_id]:
            row_id = record.get("row_id")
            if row_id not in plan_lookup:
                integrity_errors.append(f"observed record is outside the 720 planned rows: {row_id}")
                continue
            if row_id in observed_row_ids:
                integrity_errors.append(f"duplicate observed row ID: {row_id}")
                continue
            observed_row_ids.add(row_id)
            spec = spec_by_method.get(record.get("method_id"))
            if spec is None:
                integrity_errors.append(f"observed row method is outside the frozen six: {row_id}")
                continue
            status = record.get("status")
            if status not in {"PASS", "FAILED", "INCOMPLETE"}:
                integrity_errors.append(f"unsupported producer row status: {row_id}/{status}")
                continue
            arrays: dict[str, np.ndarray] = {}
            if status == "PASS":
                score = record.get("score")
                raw_ref = score.get("raw_matrix_npz") if isinstance(score, Mapping) else None
                declared = raw_ref.get("path") if isinstance(raw_ref, Mapping) else None
                npz_pin = raw_by_declared.get(declared)
                if npz_pin is None:
                    artifact_failures.append(f"{row_id}: no exact local NPZ mapping for declared path {declared!r}")
                    continue
                if npz_pin.file_id in used_npz_ids:
                    integrity_errors.append(f"NPZ file mapping reused by multiple score rows: {npz_pin.file_id}")
                    continue
                used_npz_ids.add(npz_pin.file_id)
                score_npz_bindings.append({
                    "row_id": row_id,
                    "model_id": model_id,
                    "file_id": npz_pin.file_id,
                    "declared_path": npz_pin.declared_path,
                    "local_path": npz_pin.local_path,
                    "bytes": npz_pin.byte_count,
                    "sha256": npz_pin.sha256,
                })
                try:
                    arrays = _load_arrays(npz_pin, raw_ref)
                    converted.append(official_stagea_score_row(
                        record, arrays, artifact_sha256=spec.artifact_sha256,
                        binding_sha256=spec.binding_sha256, input_pins_sha256=input_sha,
                        expected_anchor_column=15,
                    ))
                    verified_npz_ids.add(npz_pin.file_id)
                except (IngestError, FitSelectionError, Exception) as exc:
                    artifact_failures.append(f"{row_id}: {type(exc).__name__}: {exc}")
                    continue
            else:
                producer_failures.append(row_id)
                try:
                    converted.append(official_stagea_score_row(
                        record, {}, artifact_sha256=spec.artifact_sha256,
                        binding_sha256=spec.binding_sha256, input_pins_sha256=input_sha,
                        expected_anchor_column=15,
                    ))
                except Exception as exc:
                    integrity_errors.append(f"failed producer row is malformed {row_id}: {exc}")
    unused_npz = sorted(set(pin.file_id for pin in raw_by_declared.values()) - used_npz_ids)
    if unused_npz:
        artifact_failures.extend(f"unreferenced NPZ file pin: {file_id}" for file_id in unused_npz)
    missing_row_ids = tuple(sorted(expected_row_ids - observed_row_ids))
    family = FamilyInputs(
        rows=tuple(converted), roster=roster, input_pins=input_pins,
        input_pins_sha256=input_sha, score_protocol_sha256=score_protocol_sha,
        reference_specs=references, candidates=candidates,
        expected_rows=EXPECTED_TOTAL_ROWS, actual_rows=len(observed_row_ids),
        missing_row_ids=missing_row_ids, producer_failure_rows=tuple(producer_failures),
        input_artifact_failures=tuple(artifact_failures),
        verified_npz_files=len(verified_npz_ids),
        score_npz_bindings=tuple(sorted(score_npz_bindings, key=lambda item: str(item["row_id"]))),
        source_file_pins=tuple({
            "file_id": item.file_id, "role": item.role, "declared_path": item.declared_path,
            "local_path": item.local_path, "bytes": item.byte_count, "sha256": item.sha256,
        } for item in sorted(files.values(), key=lambda f: f.file_id)),
        integrity_errors=tuple(dict.fromkeys(integrity_errors)),
    )
    if family.integrity_errors or family.input_artifact_failures:
        return FamilySelection(family, None, "CONTROL_FAILED_INPUT_EVIDENCE", None,
                               family.integrity_errors + family.input_artifact_failures)
    try:
        selection = select_snapshot_family(
            family.rows, roster=family.roster, reference_specs=family.reference_specs,
            candidates=family.candidates, expected_input_pins_sha256=family.input_pins_sha256,
            metric_backend=metric_backend,
        )
    except Exception as exc:
        return FamilySelection(family, None, "CONTROL_FAILED_SELECTOR_EXCEPTION", None,
                               (f"{type(exc).__name__}: {exc}",))
    if selection.status == "CONTROL_FAILED_MISSING_FIT_EVIDENCE":
        return FamilySelection(family, selection, selection.status, None, tuple(selection.errors))
    seed_method_id = selection.selected_method_id or selection.refinement_seed_method_id
    if not family.complete_transport:
        return FamilySelection(family, selection, "CONTROL_FAILED_INCOMPLETE_720_FAMILY", None,
                               ("full 720-row transport is incomplete",))
    return FamilySelection(family, selection, selection.status, seed_method_id, tuple(selection.errors))


def initialize_spsa(selection: FamilySelection, *, metric_backend: str = "torch_cpu") -> SPSAController:
    """Create the CPU SPSA controller only from a complete verified 720 family."""
    family = selection.family
    result = selection.selection
    if (result is None or not family.complete_transport
            or selection.status not in {"ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV", "NOT_ADMITTED"}
            or selection.spsa_seed_method_id is None):
        raise IngestError("SPSA requires a complete verified 720-row selector family and a bound seed")
    if metric_backend != "torch_cpu" or result.metric_backend != "torch_cpu":
        raise IngestError("SPSA initialization requires canonical torch_cpu selection")
    if (result.accounting.get("unique_official_score_rows") != EXPECTED_TOTAL_ROWS
            or result.accounting.get("observed_unique_rows") != EXPECTED_TOTAL_ROWS
            or result.accounting.get("failed_score_rows") != 0
            or result.accounting.get("missing_score_rows") != 0
            or result.errors):
        raise IngestError("SPSA initialization blocked by incomplete or failed 720-row accounting")
    seed = next((spec for spec in family.candidates
                 if spec.method_id == selection.spsa_seed_method_id), None)
    if seed is None:
        raise IngestError("selector seed is not one of the four actual Fit snapshots")
    return SPSAController(
        initial_theta=seed.theta,
        input_pins=family.input_pins,
        candidate_artifact_sha256=seed.artifact_sha256,
        roster=family.roster,
        reference_specs=family.reference_specs,
        metric_backend=metric_backend,
    )


__all__ = ["SCHEMA", "FamilyInputs", "FamilySelection", "IngestError",
           "ingest_family", "initialize_spsa"]
