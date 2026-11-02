"""Package and apply an admitted Fit-only SPSA center as a portable WAV head.

This is separate from ``portable_fit_snapshot_apply_attempt03`` because that
module intentionally accepts only explicitly unselected training snapshots.
The selected-center path reuses its pinned ``FitSnapshotTransform`` code but
never relabels the selected point as a training checkpoint or step-120 result.
Application remains waveform-only and does not admit Dev/Confirm or qualify a
native TFG renderer.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import struct
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

ARTIFACT_SCHEMA = "cross_tfg_v3_selected_center_application_artifact_v1"
ARTIFACT_STATUS = "SELECTED_FIT_CENTER_APPLICATION_ONLY_NOT_FINAL_DEV"
RECEIPT_SCHEMA = "cross_tfg_v3_portable_selected_center_wav_application_v1"
RECEIPT_STATUS = "PASS_SELECTED_FIT_CENTER_APPLICATION_ONLY_NOT_FINAL_DEV"
FINAL_RECEIPT_SCHEMA = "fit40_final_center_selector_receipt_v1"
FINAL_RECEIPT_STATUS = "COMPLETE_FIT_ONLY_FINAL_CENTER_SELECTION_NOT_FINAL_DEV"
FINAL_SELECTION_STATUS = "ADMITTED_FIT_CANDIDATE_NOT_FINAL_DEV"
FINAL_INDEX_SCHEMA = "fit40_final_center_stagea_ingest_v1"
CENTER_TERMINAL_SCHEMA = "fit40_spsa_center_dispatch_terminal_v1"
MAX_PACKAGE_BYTES = 128 * 1024 * 1024
PORTABLE_APPLY_PIN = {
    "path": "scripts/audio_enhancement/portable_fit_snapshot_apply_attempt03.py",
    "bytes": 32995,
    "sha256": "2e8f0db6ef37a673f7fcadd7ed26a2375eeb94c3a3534e0b0c706bd4338c5fa4",
}
FINAL_INGEST_PIN = {
    "path": "scripts/experiments/cross_tfg_audio_head_20261003/fit_only_final_center_stagea_ingest_attempt07.py",
    "bytes": 110499,
    "sha256": "fea1467893fc1bb611672895e8490e3681be83504b5fb7b9dbb36cd3d0c33ef3",
}
FINAL_INGEST_DEPENDENCY_SOURCE_PINS = {
    "selector": {
        "path": "scripts/experiments/cross_tfg_audio_head_20261003/fit_only_spsa_selection_attempt06.py",
        "bytes": 66767,
        "sha256": "dda610b985026be7f9328c479358e96c983a582c511964e8c6e72a6d398692e2",
        "module_name": "scripts.experiments.cross_tfg_audio_head_20261003.fit_only_spsa_selection_attempt06",
    },
    "initial_ingest": {
        "path": "scripts/experiments/cross_tfg_audio_head_20261003/fit_only_stagea_ingest_attempt03.py",
        "bytes": 53184,
        "sha256": "22cd7f983b22babcae0d9c120f4720cb84a218ec4d2b84551d5e1668d23aeb95",
        "module_name": "scripts.experiments.cross_tfg_audio_head_20261003.fit_only_stagea_ingest_attempt03",
    },
    "packet_builder": {
        "path": "scripts/experiments/cross_tfg_audio_head_20261003/prepare_fit40_spsa_stagea_packets_attempt05.py",
        "bytes": 47748,
        "sha256": "cd5e2b80ab0bdd4f599d863468c687724898b2e6baad451a5acfe2123648bdae",
        "module_name": "scripts.experiments.cross_tfg_audio_head_20261003.prepare_fit40_spsa_stagea_packets_attempt05",
    },
    "native_point_score_adapter": {
        "path": "scripts/experiments/cross_tfg_audio_head_20261003/fit40_native_point_score_adapter_attempt02.py",
        "bytes": 20147,
        "sha256": "b4132a55da20f46338452885ea5ba7dc80ac94d4c6478756ead30cc333bc72c6",
        "module_name": "scripts.experiments.cross_tfg_audio_head_20261003.fit40_native_point_score_adapter_attempt02",
    },
}
WRAPPER_SOURCE_REL = "scripts/audio_enhancement/portable_selected_fit_center_attempt06.py"

# These modules are intentionally loaded only after their exact source files
# have been checked under the caller's bound source root.
portable: Any = None
final_ingest: Any = None


class SelectedCenterError(RuntimeError):
    """Selected-center package or application did not satisfy its evidence contract."""


def _read_raw(path: Path, label: str, limit: int = MAX_PACKAGE_BYTES) -> bytes:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise SelectedCenterError(f"{label} must be an absolute regular non-symlink file")
    info = path.stat(follow_symlinks=False)
    if info.st_size <= 0 or info.st_size > limit:
        raise SelectedCenterError(f"{label} is empty or exceeds its bounded read limit")
    raw = path.read_bytes()
    if len(raw) != info.st_size:
        raise SelectedCenterError(f"{label} changed during read")
    return raw


def _load_doc(path: Path, label: str, limit: int = MAX_PACKAGE_BYTES) -> tuple[dict[str, Any], bytes]:
    raw = _read_raw(path, label, limit)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectedCenterError(f"{label} is not UTF-8 JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SelectedCenterError(f"{label} must be a JSON object")
    return value, raw


def _file_ref(path: Path, raw: bytes, value: Mapping[str, Any], self_field: str) -> dict[str, Any]:
    return {
        "path": str(path.resolve(strict=True)), "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "self_sha256": value.get(self_field),
        "raw_utf8": raw.decode("utf-8"),
    }


def _verify_embedded_ref(ref: Any, label: str, self_field: str) -> tuple[dict[str, Any], bytes]:
    if not isinstance(ref, Mapping):
        raise SelectedCenterError(f"{label} ref is missing")
    raw_text = ref.get("raw_utf8")
    if not isinstance(raw_text, str):
        raise SelectedCenterError(f"{label} does not retain exact UTF-8 source bytes")
    raw = raw_text.encode("utf-8")
    if (len(raw) != ref.get("bytes")
            or hashlib.sha256(raw).hexdigest() != ref.get("sha256")):
        raise SelectedCenterError(f"{label} raw-byte pin mismatch")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectedCenterError(f"{label} embedded JSON is invalid: {exc}") from exc
    if not isinstance(doc, dict) or doc.get(self_field) != ref.get("self_sha256"):
        raise SelectedCenterError(f"{label} self-hash value mismatch")
    if not portable._self_hash_matches(doc, self_field):
        raise SelectedCenterError(f"{label} self-hash verification failed")
    return doc, raw


def _hash_document(doc: Mapping[str, Any], field: str) -> str:
    return hashlib.sha256(portable.canonical_bytes(
        {key: value for key, value in doc.items() if key != field})).hexdigest()


def _is_sha256(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(character in "0123456789abcdef" for character in value))


def _source_path(root: Path, pin: Mapping[str, Any], label: str) -> tuple[Path, bytes]:
    relative = PurePosixPath(str(pin.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise SelectedCenterError(f"invalid pinned source path for {label}")
    path = root.joinpath(*relative.parts)
    cursor = root
    for component in relative.parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise SelectedCenterError(f"pinned source path traverses a symlink: {label}")
    raw = _read_raw(path, f"pinned {label} source", 2 * 1024 * 1024)
    if (len(raw), hashlib.sha256(raw).hexdigest()) != (pin.get("bytes"), pin.get("sha256")):
        raise SelectedCenterError(f"pinned {label} source differs from its frozen bytes/SHA")
    return path.resolve(strict=True), raw


def _load_bound_modules(root: Path) -> tuple[Any, Any]:
    """Check both wrapper dependencies before importing either source module."""
    global portable, final_ingest
    if root.is_symlink() or not root.is_dir():
        raise SelectedCenterError("source root must be an existing real directory")
    root = root.resolve(strict=True)
    pins = (("portable", PORTABLE_APPLY_PIN, "_codex_selected_center_portable_apply_attempt03"),
            ("final_ingest", FINAL_INGEST_PIN, "_codex_selected_center_final_ingest_attempt07"))
    checked: list[tuple[str, Path, bytes]] = []
    for attribute, pin, _module_name in pins:
        path, raw = _source_path(root, pin, attribute)
        checked.append((attribute, path, raw))
    dependency_paths: dict[str, tuple[Path, bytes]] = {}
    for role, pin in FINAL_INGEST_DEPENDENCY_SOURCE_PINS.items():
        path, raw = _source_path(root, pin, f"final-ingest dependency {role}")
        dependency_paths[role] = (path, raw)
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    for package_name, expected_dir in (("scripts", root / "scripts"),
                                       ("scripts.experiments", root / "scripts" / "experiments"),
                                       ("scripts.audio_enhancement", root / "scripts" / "audio_enhancement")):
        package = sys.modules.get(package_name)
        if package is None:
            continue
        package_paths = getattr(package, "__path__", None)
        resolved_paths = {Path(str(item)).resolve(strict=True) for item in package_paths or ()}
        if resolved_paths != {expected_dir.resolve(strict=True)}:
            raise SelectedCenterError(f"already-loaded package {package_name} is outside the pinned source root")
    for role, pin in FINAL_INGEST_DEPENDENCY_SOURCE_PINS.items():
        existing = sys.modules.get(pin["module_name"])
        if existing is not None:
            expected_path, expected_raw = dependency_paths[role]
            actual_path = Path(str(getattr(existing, "__file__", ""))).resolve(strict=True)
            actual_raw = _read_raw(actual_path, f"loaded final-ingest dependency {role}", 2 * 1024 * 1024)
            if actual_path != expected_path or actual_raw != expected_raw:
                raise SelectedCenterError(f"loaded final-ingest dependency {role} differs from its source pin")
    loaded: dict[str, Any] = {}
    for (attribute, _pin, module_name), (_checked_attribute, path, raw) in zip(pins, checked):
        existing = sys.modules.get(module_name)
        if existing is not None:
            existing_path = Path(str(getattr(existing, "__file__", ""))).resolve(strict=True)
            if existing_path != path:
                raise SelectedCenterError(f"{attribute} module cache points outside its pinned source")
            module = existing
        else:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                raise SelectedCenterError(f"cannot load exact pinned {attribute} module")
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except BaseException:
                sys.modules.pop(module_name, None)
                raise
        loaded[attribute] = module
        actual_path = Path(str(getattr(module, "__file__", ""))).resolve(strict=True)
        after = _read_raw(path, f"pinned {attribute} source after import", 2 * 1024 * 1024)
        if (actual_path != path or after != raw):
            raise SelectedCenterError(f"{attribute} source changed or loaded from an unexpected path")
    portable = loaded["portable"]
    final_ingest = loaded["final_ingest"]
    for role, attribute in (("selector", "selector"),
                            ("initial_ingest", "initial_ingest"),
                            ("packet_builder", "packet_builder"),
                            ("native_point_score_adapter", "native_point_score_adapter")):
        pin = FINAL_INGEST_DEPENDENCY_SOURCE_PINS[role]
        dependency = getattr(final_ingest, attribute, None)
        if dependency is None:
            raise SelectedCenterError(f"pinned final-ingest module lacks required dependency {role}")
        expected_path, expected_raw = dependency_paths[role]
        actual_path = Path(str(getattr(dependency, "__file__", ""))).resolve(strict=True)
        actual_raw = _read_raw(actual_path, f"loaded final-ingest dependency {role}", 2 * 1024 * 1024)
        if actual_path != expected_path or actual_raw != expected_raw:
            raise SelectedCenterError(f"final-ingest dependency {role} loaded from an unpinned source")
    return portable, final_ingest


def _resolve_source_root(root: str | Path) -> Path:
    path = Path(root)
    if path.is_symlink() or not path.is_dir():
        raise SelectedCenterError("source root must be an existing real directory")
    return path.resolve(strict=True)


def _wrapper_source_pin(root: Path) -> dict[str, Any]:
    """Bind this wrapper by portable repository-relative identity.

    The resolved origin is checked to stay inside the declared source root but
    is intentionally omitted from the binding: a valid bundle may be applied
    from a different checkout path on another host.
    """
    expected = root.joinpath(*PurePosixPath(WRAPPER_SOURCE_REL).parts).resolve(strict=True)
    actual_path = Path(__file__)
    if actual_path.is_symlink() or actual_path.resolve(strict=True) != expected:
        raise SelectedCenterError("selected-center wrapper __file__ is outside its declared source root")
    raw = _read_raw(expected, "selected-center wrapper source", 2 * 1024 * 1024)
    return {
        "path": WRAPPER_SOURCE_REL,
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _selected_center_from_evidence(final_index: Mapping[str, Any],
                                   selector_receipt: Mapping[str, Any],
                                   center_terminal: Mapping[str, Any],
                                   point_plan: Mapping[str, Any],
                                   seed_artifact: Mapping[str, Any], *,
                                   terminal_file_ref: Mapping[str, Any],
                                   plan_file_ref: Mapping[str, Any]) -> dict[str, Any]:
    if (selector_receipt.get("schema") != FINAL_RECEIPT_SCHEMA
            or selector_receipt.get("status") != FINAL_RECEIPT_STATUS
            or selector_receipt.get("selection_status") != FINAL_SELECTION_STATUS
            or selector_receipt.get("final_dev_or_confirm_admission_performed") is not False):
        raise SelectedCenterError("selector receipt is not an admitted Fit-only center selection")
    selection = selector_receipt.get("final_selection")
    expected_accounting = {
        "initial_rows_verified": 720,
        "proposal_rows_verified": 960,
        "center_rows_verified": 480,
        "final_selection_rows": 720,
        "unique_candidate_npz_files_verified": 1440,
    }
    selection_accounting = selection.get("accounting") if isinstance(selection, Mapping) else None
    if (selector_receipt.get("metric_backend") != "torch_cpu"
            or not isinstance(selection, Mapping)
            or selection.get("status") != FINAL_SELECTION_STATUS
            or selection.get("metric_backend") != "torch_cpu"
            or selector_receipt.get("accounting") != expected_accounting
            or any(type(selector_receipt.get("accounting", {}).get(key)) is not int
                   for key in expected_accounting)
            or not isinstance(selection_accounting, Mapping)
            or any(type(selection_accounting.get(key)) is not int for key in (
                "unique_official_score_rows", "observed_unique_rows",
                "failed_score_rows", "missing_score_rows"))
            or selection_accounting.get("unique_official_score_rows") != 720
            or selection_accounting.get("observed_unique_rows") != 720
            or selection_accounting.get("failed_score_rows") != 0
            or selection_accounting.get("missing_score_rows") != 0):
        raise SelectedCenterError("selector receipt lacks canonical Torch CPU selection and complete 720-row accounting")
    scientific_input_pin = selector_receipt.get("scientific_input_pins_sha256")
    if not _is_sha256(scientific_input_pin):
        raise SelectedCenterError("selector receipt lacks the canonical shared scientific-input pin")
    selected_id = selection.get("selected_method_id") if isinstance(selection, Mapping) else None
    if not isinstance(selected_id, str) or not selected_id:
        raise SelectedCenterError("selector receipt lacks one admitted selected center method")
    specs = selector_receipt.get("center_specs")
    matching_specs = [spec for spec in specs if isinstance(spec, Mapping)
                      and spec.get("method_id") == selected_id] if isinstance(specs, list) else []
    if len(matching_specs) != 1:
        raise SelectedCenterError("selected method does not resolve to exactly one verified center spec")
    spec = dict(matching_specs[0])
    step = spec.get("source_step")
    theta = spec.get("theta")
    canonical_theta_sha = hashlib.sha256(portable.canonical_bytes(list(theta))).hexdigest() \
        if isinstance(theta, (list, tuple)) else None
    if (spec.get("point_kind") != "spsa_center" or type(step) is not int
            or step not in (5, 10, 15, 20) or spec.get("method_id") != selected_id
            or selected_id != f"fit40_d0_spsa_center_k{step:02d}"
            or not isinstance(theta, (list, tuple)) or len(theta) != 9
            or any(type(x) not in (int, float) or not math.isfinite(float(x)) for x in theta)):
        raise SelectedCenterError("selected center spec has invalid point kind, source step, or theta")
    if (not _is_sha256(canonical_theta_sha)
            or not _is_sha256(spec.get("artifact_sha256"))
            or not _is_sha256(spec.get("binding_sha256"))):
        raise SelectedCenterError("selected center spec theta/artifact bindings differ from canonical values")
    if (final_index.get("schema") != FINAL_INDEX_SCHEMA
            or final_index.get("selection_status") != "NOT_SELECTED"
            or final_index.get("scientific_input_pins_sha256") != scientific_input_pin
            or not portable._self_hash_matches(final_index, "index_sha256")):
        raise SelectedCenterError("final-center index schema, pending selector state, or self-hash is invalid")
    centers = final_index.get("center_points")
    points = [entry for entry in centers if isinstance(entry, Mapping)
              and entry.get("method_id") == selected_id] if isinstance(centers, list) else []
    if (len(points) != 1 or points[0].get("point_kind") != "spsa_center"
            or points[0].get("point_id") != selected_id or points[0].get("source_step") != step):
        raise SelectedCenterError("final-center index does not bind the selected center point")
    files = final_index.get("files")
    file_entries = [entry for entry in files if isinstance(entry, Mapping)] if isinstance(files, list) else []
    file_ids = [entry.get("file_id") for entry in file_entries]
    if (len(file_entries) != len(files or []) or any(not isinstance(x, str) or not x for x in file_ids)
            or len(set(file_ids)) != len(file_ids)):
        raise SelectedCenterError("final-center index file table has malformed or duplicate file IDs")
    by_id = {entry.get("file_id"): entry for entry in file_entries}
    terminal_ref = by_id.get(points[0].get("dispatch_terminal_file_id"))
    plan_ref = by_id.get(points[0].get("plan_file_id"))
    if (not isinstance(terminal_ref, Mapping) or terminal_ref.get("role") != "center_dispatch_terminal"
            or not isinstance(plan_ref, Mapping) or plan_ref.get("role") != "plan"):
        raise SelectedCenterError("index lacks exact selected terminal or center plan file entries")
    if (center_terminal.get("schema") != CENTER_TERMINAL_SCHEMA
            or center_terminal.get("status") != "COMPLETE"
            or center_terminal.get("point_kind") != "spsa_center"
            or center_terminal.get("method_id") != selected_id
            or center_terminal.get("point_id") != selected_id
            or center_terminal.get("source_step") != step
            or center_terminal.get("candidate_rows") != 120
            or center_terminal.get("planned_candidate_rows") != 120
            or not portable._self_hash_matches(center_terminal, "terminal_sha256")):
        raise SelectedCenterError("selected terminal is not the complete 120-row center")
    terminal_entry = center_terminal.get("point_entry")
    if (not isinstance(terminal_entry, Mapping)
            or terminal_entry.get("method_id") != selected_id
            or terminal_entry.get("point_id") != selected_id
            or terminal_entry.get("source_step") != step
            or terminal_entry.get("point_kind") != "spsa_center"
            or len(center_terminal.get("raw_npz_by_row", {})) != 120):
        raise SelectedCenterError("selected terminal's point entry or row denominator differs")
    point_plan_methods = point_plan.get("methods")
    method_matches = [item for item in point_plan_methods if isinstance(item, Mapping)
                      and item.get("method_id") == selected_id] if isinstance(point_plan_methods, list) else []
    if (len(method_matches) != 1 or point_plan.get("schema_version") != "cross-tfg-audio-eval-v1"
            or point_plan.get("status") != "FROZEN"
            or not portable._self_hash_matches(point_plan, "plan_sha256")):
        raise SelectedCenterError("selected point plan lacks its unique frozen method or valid self-hash")
    method = dict(method_matches[0])
    config = method.get("config")
    if (not isinstance(config, Mapping) or method.get("kind") != "fit40_spsa_center"
            or method.get("checkpoint_sha256") is not None
            or method.get("config_sha256") != hashlib.sha256(portable.canonical_bytes(config)).hexdigest()
            or config.get("method") != "fit40_spsa_center"
            or config.get("schema") != "fit40_spsa_center_transform_v1"
            or config.get("point_kind") != "spsa_center"
            or config.get("method_id") != selected_id
            or config.get("source_step") != step
            or config.get("theta") != list(theta)
            or config.get("theta_sha256") != canonical_theta_sha
            or config.get("source_theta_sha256") != hashlib.sha256(
                portable.canonical_bytes(config.get("source_theta"))).hexdigest()
            or config.get("artifact_sha256") != spec.get("artifact_sha256")
            or config.get("binding_sha256") != spec.get("binding_sha256")
            or config.get("input_pins_sha256") != scientific_input_pin
            or point_plan.get("scientific_input_pins_sha256") != scientific_input_pin):
        raise SelectedCenterError("selected center plan method does not exactly match selector spec")
    try:
        expected_binding = final_ingest.selector.full_fit_candidate_binding_sha256(
            method_id=selected_id,
            artifact_sha256=spec["artifact_sha256"],
            input_pins_sha256=scientific_input_pin,
            source_step=step,
            theta=theta,
            point_kind="spsa_center",
        )
    except (AttributeError, KeyError, TypeError, ValueError) as exc:
        raise SelectedCenterError("pinned selector cannot validate the selected center binding") from exc
    if spec.get("binding_sha256") != expected_binding:
        raise SelectedCenterError("selected center spec binding differs from the pinned selector formula")
    seed_snapshot = seed_artifact.get("snapshot_record")
    seed_id = config.get("source_seed_method_id")
    seed_step = config.get("source_snapshot_step")
    if (not isinstance(seed_snapshot, Mapping)
            or seed_artifact.get("schema") != portable.ARTIFACT_SCHEMA
            or seed_artifact.get("status") != portable.ARTIFACT_STATUS
            or not portable._self_hash_matches(seed_artifact, "artifact_sha256")
            or seed_id != seed_snapshot.get("method_id")
            or seed_step != seed_snapshot.get("source_optimizer_step")
            or seed_snapshot.get("snapshot_sha256") != spec.get("artifact_sha256")
            or config.get("source_snapshot_artifact_sha256") != seed_snapshot.get("snapshot_sha256")
            or config.get("artifact_sha256") != seed_snapshot.get("snapshot_sha256")):
        raise SelectedCenterError("selected center seed lineage does not match the sealed source artifact")
    source_record_pin = config.get("source_snapshot_record")
    source_index_pin = config.get("source_snapshot_index")
    seed_record_pin = seed_artifact.get("snapshot_record_pin")
    seed_index_pin = seed_artifact.get("snapshot_index_pin")
    if (not isinstance(source_record_pin, Mapping) or not isinstance(source_index_pin, Mapping)
            or not isinstance(seed_record_pin, Mapping) or not isinstance(seed_index_pin, Mapping)
            or not isinstance(source_record_pin.get("path"), str) or not source_record_pin.get("path")
            or not isinstance(source_index_pin.get("path"), str) or not source_index_pin.get("path")
            or any(source_record_pin.get(key) != seed_record_pin.get(key)
                   for key in ("bytes", "sha256", "self_sha256"))
            or any(source_index_pin.get(key) != seed_index_pin.get(key)
                   for key in ("bytes", "sha256", "self_sha256"))
            or config.get("source_snapshot_record_raw_sha256") != seed_record_pin.get("sha256")
            or config.get("source_snapshot_index_raw_sha256") != seed_index_pin.get("sha256")
            or source_record_pin.get("self_sha256") != seed_snapshot.get("snapshot_sha256")):
        raise SelectedCenterError("selected center source snapshot record/index raw and self pins differ")
    if (not isinstance(terminal_entry.get("run_id"), str)
            or terminal_entry.get("run_id") != point_plan.get("run_id")):
        raise SelectedCenterError("selected center terminal and point plan run identities differ")
    _verify_index_file_bindings(
        final_index, points[0], center_terminal, point_plan, terminal_file_ref, plan_file_ref)
    seed_parameter = seed_snapshot.get("parameter")
    seed_values = seed_parameter.get("values") if isinstance(seed_parameter, Mapping) else None
    if (not isinstance(seed_values, list) or len(seed_values) != 3
            or any(not isinstance(row, list) or len(row) != 3 for row in seed_values)):
        raise SelectedCenterError("original seed snapshot parameter is not a 3x3 tensor")
    seed_theta_flat = [value for row in seed_values for value in row]
    if (config.get("source_theta") != seed_theta_flat
            or config.get("source_theta_sha256") != hashlib.sha256(
                portable.canonical_bytes(seed_theta_flat)).hexdigest()):
        raise SelectedCenterError("selected center source theta does not match the flattened original seed snapshot")
    return {
        "method_id": selected_id,
        "point_kind": "spsa_center",
        "source_step": step,
        "theta": [float(x) for x in theta],
        "theta_sha256": canonical_theta_sha,
        "artifact_sha256": spec.get("artifact_sha256"),
        "binding_sha256": spec.get("binding_sha256"),
        "input_pins_sha256": scientific_input_pin,
        "config_sha256": method.get("config_sha256"),
        "source_seed_method_id": seed_id,
        "source_snapshot_step": seed_step,
        "source_snapshot_artifact_sha256": seed_snapshot.get("snapshot_sha256"),
    }


def _verify_index_file_bindings(final_index: Mapping[str, Any], point_entry: Mapping[str, Any],
                                terminal: Mapping[str, Any], point_plan: Mapping[str, Any],
                                terminal_file_ref: Mapping[str, Any],
                                plan_file_ref: Mapping[str, Any]) -> None:
    """Bind embedded terminal/plan bytes to the final index's exact file entries."""
    if (terminal_file_ref.get("self_sha256") != terminal.get("terminal_sha256")
            or plan_file_ref.get("self_sha256") != point_plan.get("plan_sha256")):
        raise SelectedCenterError("selected terminal/plan embedded self pins differ from their source refs")
    files = final_index.get("files")
    if not isinstance(files, list):
        raise SelectedCenterError("final-center index lacks its file table")
    file_ids = [item.get("file_id") for item in files if isinstance(item, Mapping)]
    if len(file_ids) != len(files) or len(set(file_ids)) != len(file_ids):
        raise SelectedCenterError("final-center index file table has duplicate or malformed entries")
    refs = {item.get("file_id"): item for item in files if isinstance(item, Mapping)}

    def check(entry: Any, ref: Mapping[str, Any], expected_id: Any, role: str) -> Mapping[str, Any]:
        if not isinstance(entry, Mapping) or not isinstance(expected_id, str):
            raise SelectedCenterError(f"final-center index lacks the selected {role} file entry")
        declared_path = entry.get("declared_path", entry.get("path"))
        if (entry.get("file_id") != expected_id or entry.get("role") != role
                or not isinstance(declared_path, str)
                or declared_path != ref.get("path")
                or entry.get("bytes") != ref.get("bytes")
                or entry.get("sha256") != ref.get("sha256")
                or not isinstance(ref.get("raw_utf8"), str)):
            raise SelectedCenterError(f"selected {role} raw bytes are not bound by the final index file entry")
        raw = ref["raw_utf8"].encode("utf-8")
        if (len(raw) != entry.get("bytes")
                or hashlib.sha256(raw).hexdigest() != entry.get("sha256")):
            raise SelectedCenterError(f"selected {role} embedded bytes differ from the final index entry")
        return entry

    terminal_index_entry = check(
        refs.get(point_entry.get("dispatch_terminal_file_id")), terminal_file_ref,
        point_entry.get("dispatch_terminal_file_id"), "center_dispatch_terminal")
    plan_index_entry = check(
        refs.get(point_entry.get("plan_file_id")), plan_file_ref,
        point_entry.get("plan_file_id"), "plan")
    _ = terminal_index_entry
    terminal_point = terminal.get("point_entry")
    if (not isinstance(terminal_point, Mapping)
            or terminal_point.get("plan_file_id") != plan_index_entry.get("file_id")):
        raise SelectedCenterError("center terminal does not reference the indexed point plan")
    terminal_files = terminal.get("file_entries")
    if not isinstance(terminal_files, list):
        raise SelectedCenterError("center terminal lacks its source file table")
    plan_matches = [item for item in terminal_files if isinstance(item, Mapping)
                    and item.get("file_id") == plan_index_entry.get("file_id")]
    if len(plan_matches) != 1:
        raise SelectedCenterError("center terminal does not retain exactly one indexed plan entry")
    embedded_plan_entry = plan_matches[0]
    for key in ("file_id", "role", "declared_path", "bytes", "sha256"):
        if embedded_plan_entry.get(key) != plan_index_entry.get(key):
            raise SelectedCenterError("center terminal plan entry differs from the final index plan entry")


def _load_portable_module(root: Path) -> Any:
    return _load_bound_modules(root)[0]


def seal_selected_center_artifact(*, final_center_index_path: str | Path,
                                  selector_receipt_path: str | Path,
                                  seed_artifact_path: str | Path,
                                  source_root: str | Path,
                                  output_artifact_path: str | Path) -> dict[str, Any]:
    root = _resolve_source_root(source_root)
    _load_bound_modules(root)
    index_path = Path(final_center_index_path)
    receipt_path = Path(selector_receipt_path)
    seed_path = Path(seed_artifact_path)
    output_path = Path(output_artifact_path)
    index, index_raw = _load_doc(index_path, "final center index")
    receipt, receipt_raw = _load_doc(receipt_path, "final selector receipt")
    seed_artifact, seed_raw = _load_doc(seed_path, "provisional seed artifact")
    if (not output_path.is_absolute() or output_path.exists() or output_path.is_symlink()
            or output_path.parent.is_symlink() or not output_path.parent.is_dir()):
        raise SelectedCenterError("selected artifact output must be a new absolute path under an existing real directory")

    # Re-run the frozen complete consumer on the actual index and compare its
    # complete receipt.  This is the source of selection truth, not a caller ID.
    verified = final_ingest.ingest_final_center_family(index_path, metric_backend="torch_cpu")
    expected_receipt = final_ingest.selection_evidence_document(verified)
    if receipt != expected_receipt:
        raise SelectedCenterError("selector receipt does not byte-semantically match fresh complete-center ingestion")
    if receipt.get("selection_status") != FINAL_SELECTION_STATUS:
        raise SelectedCenterError("final selector did not admit a Fit candidate")

    selected_id = receipt["final_selection"]["selected_method_id"]
    index_files = {entry.get("file_id"): entry for entry in index.get("files", [])
                   if isinstance(entry, Mapping)}
    points = [item for item in index.get("center_points", [])
              if isinstance(item, Mapping) and item.get("method_id") == selected_id]
    if len(points) != 1:
        raise SelectedCenterError("final index does not contain one selected center point")
    point = points[0]
    terminal_entry = index_files.get(point.get("dispatch_terminal_file_id"))
    plan_entry = index_files.get(point.get("plan_file_id"))
    if not isinstance(terminal_entry, Mapping) or not isinstance(plan_entry, Mapping):
        raise SelectedCenterError("selected center terminal or plan is missing from final index file table")
    terminal_path = Path(str(terminal_entry.get("declared_path", terminal_entry.get("path", ""))))
    plan_path = Path(str(plan_entry.get("declared_path", plan_entry.get("path", ""))))
    terminal, terminal_raw = _load_doc(terminal_path, "selected center terminal")
    plan, plan_raw = _load_doc(plan_path, "selected center point plan")
    if (terminal_entry.get("role") != "center_dispatch_terminal"
            or terminal_entry.get("sha256") != hashlib.sha256(terminal_raw).hexdigest()
            or terminal_entry.get("bytes") != len(terminal_raw)
            or plan_entry.get("role") != "plan"
            or plan_entry.get("sha256") != hashlib.sha256(plan_raw).hexdigest()
            or plan_entry.get("bytes") != len(plan_raw)):
        raise SelectedCenterError("selected terminal/plan raw file identity differs from the final index")
    selected = _selected_center_from_evidence(
        index, receipt, terminal, plan, seed_artifact,
        terminal_file_ref=_file_ref(terminal_path, terminal_raw, terminal, "terminal_sha256"),
        plan_file_ref=_file_ref(plan_path, plan_raw, plan, "plan_sha256"))
    seed_checked, seed_checked_raw = portable._load_artifact(seed_path)
    if seed_checked_raw != seed_raw:
        raise SelectedCenterError("seed artifact changed during verification")
    if selected["source_snapshot_artifact_sha256"] != seed_checked["snapshot_record"]["snapshot_sha256"]:
        raise SelectedCenterError("selected center does not descend from the supplied sealed source snapshot")

    evidence = {
        "final_center_index": _file_ref(index_path, index_raw, index, "index_sha256"),
        "selector_receipt": _file_ref(receipt_path, receipt_raw, receipt, "receipt_sha256"),
        "center_terminal": _file_ref(terminal_path, terminal_raw, terminal, "terminal_sha256"),
        "point_plan": _file_ref(plan_path, plan_raw, plan, "plan_sha256"),
        "seed_application_artifact": _file_ref(seed_path, seed_raw, seed_artifact, "artifact_sha256"),
    }
    wrapper_source_pin = _wrapper_source_pin(root)
    payload = {
        "schema": ARTIFACT_SCHEMA,
        "status": ARTIFACT_STATUS,
        "selection_status": FINAL_SELECTION_STATUS,
        "native_tfg_qualification_claim": False,
        "final_dev_or_confirm_admission_performed": False,
        "selected_center": selected,
        "evidence": evidence,
        "source_code_pins": seed_artifact.get("source_code_pins"),
        "portable_apply_source_pin": dict(PORTABLE_APPLY_PIN),
        "wrapper_source_pin": wrapper_source_pin,
        "final_ingest_source_pin": dict(FINAL_INGEST_PIN),
        "final_ingest_dependency_source_pins": copy.deepcopy(FINAL_INGEST_DEPENDENCY_SOURCE_PINS),
        "waveform_contract": {
            "input": "finite mono 16 kHz PCM; no resampling",
            "output": "mono FLOAT_32 at 16 kHz; exact sample count preserved",
            "global_projected_log_magnitude_mask_bound_db": 3.0,
            "global_rms_target_from_pinned_constructor": True,
            "peak_limiter": False,
        },
        "scope": "selected_fit_center_waveform_application_only; no Dev/Confirm admission or TFG qualification",
    }
    artifact = dict(payload)
    artifact["artifact_sha256"] = hashlib.sha256(portable.canonical_bytes(payload)).hexdigest()
    if _wrapper_source_pin(root) != wrapper_source_pin:
        raise SelectedCenterError("selected-center wrapper source changed during artifact verification")
    raw = json.dumps(artifact, sort_keys=True, ensure_ascii=False, indent=2,
                     allow_nan=False).encode("utf-8") + b"\n"
    portable._write_exclusive_json(output_path, raw, "selected-center application artifact")
    return {"path": str(output_path.resolve(strict=True)), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "artifact_sha256": artifact["artifact_sha256"],
            "status": ARTIFACT_STATUS, "selected_method_id": selected["method_id"],
            "source_step": selected["source_step"], "final_dev_or_confirm_admission_performed": False}


def _load_selected_artifact(path: Path) -> tuple[dict[str, Any], bytes, dict[str, Any]]:
    artifact, raw = _load_doc(path, "selected-center application artifact")
    if (artifact.get("schema") != ARTIFACT_SCHEMA or artifact.get("status") != ARTIFACT_STATUS
            or artifact.get("selection_status") != FINAL_SELECTION_STATUS
            or artifact.get("native_tfg_qualification_claim") is not False
            or artifact.get("final_dev_or_confirm_admission_performed") is not False
            or not portable._self_hash_matches(artifact, "artifact_sha256")):
        raise SelectedCenterError("selected-center artifact schema/status/self-hash is invalid")
    evidence = artifact.get("evidence")
    if not isinstance(evidence, Mapping):
        raise SelectedCenterError("selected-center artifact lacks evidence refs")
    if artifact.get("portable_apply_source_pin") != PORTABLE_APPLY_PIN:
        raise SelectedCenterError("selected-center artifact names a different portable application source")
    wrapper_pin = artifact.get("wrapper_source_pin")
    if (not isinstance(wrapper_pin, Mapping)
            or wrapper_pin.get("path") != WRAPPER_SOURCE_REL
            or not isinstance(wrapper_pin.get("bytes"), int)
            or not isinstance(wrapper_pin.get("sha256"), str)):
        raise SelectedCenterError("selected-center artifact lacks its wrapper source pin")
    if artifact.get("final_ingest_source_pin") != FINAL_INGEST_PIN:
        raise SelectedCenterError("selected-center artifact names a different pinned final-center ingest source")
    if artifact.get("final_ingest_dependency_source_pins") != FINAL_INGEST_DEPENDENCY_SOURCE_PINS:
        raise SelectedCenterError("selected-center artifact names different final-ingest dependency sources")
    docs = {}
    raws = {}
    for key, field in (("final_center_index", "index_sha256"),
                       ("selector_receipt", "receipt_sha256"),
                       ("center_terminal", "terminal_sha256"),
                       ("point_plan", "plan_sha256"),
                       ("seed_application_artifact", "artifact_sha256")):
        docs[key], raws[key] = _verify_embedded_ref(evidence.get(key), key, field)
    selected = _selected_center_from_evidence(
        docs["final_center_index"], docs["selector_receipt"], docs["center_terminal"],
        docs["point_plan"], docs["seed_application_artifact"],
        terminal_file_ref=evidence["center_terminal"], plan_file_ref=evidence["point_plan"])
    if selected != artifact.get("selected_center"):
        raise SelectedCenterError("selected-center summary differs from its bound evidence")
    seed = docs["seed_application_artifact"]
    if (seed.get("schema") != portable.ARTIFACT_SCHEMA
            or seed.get("status") != portable.ARTIFACT_STATUS
            or seed.get("native_tfg_qualification_claim") is not False
            or not portable._self_hash_matches(seed, "artifact_sha256")):
        raise SelectedCenterError("embedded source application artifact self-hash is invalid")
    snapshot = seed.get("snapshot_record")
    index = seed.get("snapshot_index")
    if not isinstance(snapshot, Mapping) or not isinstance(index, Mapping):
        raise SelectedCenterError("embedded seed artifact lacks its source snapshot/index")
    portable._verify_snapshot(snapshot)
    snapshot_text = seed.get("snapshot_record_raw_utf8")
    index_text = seed.get("snapshot_index_raw_utf8")
    snapshot_pin = seed.get("snapshot_record_pin")
    index_pin = seed.get("snapshot_index_pin")
    if (not isinstance(snapshot_text, str) or not isinstance(index_text, str)
            or not isinstance(snapshot_pin, Mapping) or not isinstance(index_pin, Mapping)):
        raise SelectedCenterError("embedded seed artifact lacks exact source snapshot/index bytes")
    snapshot_raw = snapshot_text.encode("utf-8")
    index_raw = index_text.encode("utf-8")
    try:
        decoded_snapshot = json.loads(snapshot_raw.decode("utf-8"))
        decoded_index = json.loads(index_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SelectedCenterError("embedded seed snapshot/index JSON is invalid") from exc
    if (decoded_snapshot != snapshot or decoded_index != index
            or snapshot_pin.get("bytes") != len(snapshot_raw)
            or snapshot_pin.get("sha256") != hashlib.sha256(snapshot_raw).hexdigest()
            or snapshot_pin.get("self_sha256") != snapshot.get("snapshot_sha256")
            or index_pin.get("bytes") != len(index_raw)
            or index_pin.get("sha256") != hashlib.sha256(index_raw).hexdigest()
            or index_pin.get("self_sha256") != index.get("index_sha256")):
        raise SelectedCenterError("embedded source snapshot/index raw bytes do not match their pins")
    portable._verify_index(index, snapshot, snapshot_pin["sha256"], snapshot_pin["bytes"])
    return artifact, raw, docs


def _verify_selection_against_current_files(artifact: Mapping[str, Any],
                                            evidence_docs: Mapping[str, Any],
                                            final_center_index_path: Path | None,
                                            selector_receipt_path: Path | None) -> None:
    if final_center_index_path is None and selector_receipt_path is None:
        # The selected artifact embeds the exact source bytes.  This mode is
        # useful after moving the portable bundle; sealing already reran the
        # complete selector against the original 720-row index.
        return
    if final_center_index_path is None or selector_receipt_path is None:
        raise SelectedCenterError("fresh selection replay requires both index and selector receipt")
    index_raw = _read_raw(final_center_index_path, "current final-center index")
    receipt_raw = _read_raw(selector_receipt_path, "current selector receipt")
    index_ref = artifact["evidence"]["final_center_index"]
    receipt_ref = artifact["evidence"]["selector_receipt"]
    # A restore location may differ from the original declared path; exact raw
    # bytes and the embedded original path pin are the evidence identity.
    if (index_raw.decode("utf-8") != index_ref["raw_utf8"]
            or receipt_raw.decode("utf-8") != receipt_ref["raw_utf8"]):
        raise SelectedCenterError("current final-center index/selector receipt differ from packaged bytes")
    verified = final_ingest.ingest_final_center_family(
        final_center_index_path, metric_backend="torch_cpu")
    expected_receipt = final_ingest.selection_evidence_document(verified)
    if expected_receipt != evidence_docs["selector_receipt"]:
        raise SelectedCenterError("fresh complete-center replay differs from packaged selector receipt")
    if evidence_docs["final_center_index"].get("index_sha256") != verified_index_self(evidence_docs["final_center_index"]):
        raise SelectedCenterError("packaged final-center index self pin is invalid")


def verified_index_self(index: Mapping[str, Any]) -> str:
    declared = index.get("index_sha256")
    if not isinstance(declared, str) or _hash_document(index, "index_sha256") != declared:
        raise SelectedCenterError("final-center index canonical self-hash mismatch")
    return declared


def apply_selected_artifact(*, artifact_path: str | Path,
                            final_center_index_path: str | Path | None,
                            selector_receipt_path: str | Path | None,
                            source_root: str | Path,
                            input_wav: str | Path,
                            output_wav: str | Path,
                            receipt_path: str | Path) -> dict[str, Any]:
    root = _resolve_source_root(source_root)
    _load_bound_modules(root)
    artifact_path = Path(artifact_path)
    artifact, artifact_raw, docs = _load_selected_artifact(artifact_path)
    actual_wrapper_pin = _wrapper_source_pin(root)
    if artifact.get("wrapper_source_pin") != actual_wrapper_pin:
        raise SelectedCenterError("selected-center wrapper bytes/path differ from the sealed artifact source pin")
    _verify_selection_against_current_files(
        artifact, docs,
        Path(final_center_index_path) if final_center_index_path is not None else None,
        Path(selector_receipt_path) if selector_receipt_path is not None else None)
    seed_artifact = docs["seed_application_artifact"]
    code_pins = artifact.get("source_code_pins")
    if code_pins != seed_artifact.get("source_code_pins"):
        raise SelectedCenterError("selected artifact source-code pins differ from seed envelope")
    import numpy as np
    import torch

    transform_class, audio_io, actual_code_pins = portable._load_runtime_modules(root, code_pins)
    input_path = Path(input_wav)
    if not input_path.is_absolute() or input_path.is_symlink() or not input_path.is_file():
        raise SelectedCenterError("input WAV must be an absolute regular non-symlink file")
    input_raw = _read_raw(input_path, "input WAV", portable.MAX_WAV_BYTES)
    input_sha = hashlib.sha256(input_raw).hexdigest()
    output_path = portable._safe_output_path(Path(output_wav), input_path=input_path, label="output WAV")
    receipt_out = portable._safe_output_path(Path(receipt_path), input_path=input_path, label="application receipt")
    if output_path.resolve(strict=False) == receipt_out.resolve(strict=False):
        raise SelectedCenterError("output WAV and application receipt must be distinct")
    try:
        natural, read_info = audio_io.read_canonical_wav(input_path, target_rate=16000)
    except Exception as exc:
        raise SelectedCenterError(f"input WAV decode failed: {type(exc).__name__}: {exc}") from exc
    if (read_info.file_sha256 != input_sha or read_info.sample_rate != 16000
            or read_info.source_sample_rate != 16000 or read_info.source_channels != 1
            or read_info.frames <= 0 or natural.ndim != 1 or not np.isfinite(natural).all()):
        raise SelectedCenterError("input must be finite mono 16 kHz PCM without resampling")
    seed_snapshot = seed_artifact["snapshot_record"]
    snapshot = copy.deepcopy(seed_snapshot)
    theta = artifact["selected_center"]["theta"]
    if len(theta) != 9 or any(not math.isfinite(float(value)) for value in theta):
        raise SelectedCenterError("selected artifact theta is malformed")
    theta32 = [float(struct.unpack("<f", struct.pack("<f", float(value)))[0]) for value in theta]
    snapshot["parameter"]["values"] = [theta32[offset:offset + 3] for offset in range(0, 9, 3)]
    theta_raw = struct.pack("<9f", *theta32)
    theta_sha = hashlib.sha256(theta_raw).hexdigest()
    snapshot["parameter"]["little_endian_f32_sha256"] = theta_sha
    if (theta_sha != portable.sha256_bytes(theta_raw)
            or artifact["selected_center"].get("source_snapshot_artifact_sha256")
            != seed_snapshot.get("snapshot_sha256")):
        raise SelectedCenterError("selected runtime theta/source seed binding is inconsistent")
    constructor = snapshot["head_constructor"]
    if (constructor.get("stft_center") is True and constructor.get("stft_pad_mode") == "reflect"
            and read_info.frames < int(constructor["n_fft"]) // 2 + 1):
        raise SelectedCenterError("input is too short for the pinned centered reflect STFT")
    head_path = root.joinpath(*PurePosixPath(portable.SOURCE_PINS["head"][0]).parts)
    baselines_path = root.joinpath(*PurePosixPath(portable.SOURCE_PINS["baselines"][0]).parts)
    transform = transform_class(snapshot=snapshot, head_module_path=head_path,
                                head_module_sha256=portable.SOURCE_PINS["head"][2],
                                device="cpu", baselines_path=baselines_path)
    hook_state: dict[str, Any] = {}

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        if not isinstance(output, dict):
            raise SelectedCenterError("pinned head emitted no diagnostics mapping")
        mask = output.get("projected_log_magnitude_mask_db")
        if mask is None or not bool(torch.isfinite(mask).all().item()):
            raise SelectedCenterError("pinned head emitted a missing/non-finite projected mask")
        hook_state.update({
            "preprojection_max_abs_mask_db": float(output["preprojection_max_abs_mask_db"].detach().cpu().item()),
            "postprojection_max_abs_mask_db": float(output["postprojection_max_abs_mask_db"].detach().cpu().item()),
            "valid_transition_count": int(output["valid_transition_mask"].sum().detach().cpu().item()),
            "stft_frame_count": int(mask.shape[-1]),
        })

    hook = transform.head.register_forward_hook(capture)
    try:
        output, transform_summary = transform.transform(natural)
    except Exception as exc:
        raise SelectedCenterError(f"selected-center waveform transform failed: {type(exc).__name__}: {exc}") from exc
    finally:
        hook.remove()
    output = np.ascontiguousarray(output, dtype=np.float32)
    output_rms = float(np.sqrt(np.mean(output.astype(np.float64) ** 2, dtype=np.float64)))
    target_rms = float(constructor["target_rms"])
    baseline = transform_summary.get("canonical_rms_baseline")
    if (output.shape != natural.shape or not np.isfinite(output).all() or not hook_state
            or not isinstance(baseline, Mapping)
            or not math.isfinite(output_rms)
            or abs(output_rms - float(baseline.get("output_rms", float("nan")))) > 1e-6
            or abs(output_rms - target_rms) > float(constructor["baseline_match_atol"]) + 1e-6
            or hook_state["postprojection_max_abs_mask_db"] > 3.0 + 1e-5):
        raise SelectedCenterError("output violates the frozen sample-clock, global RMS, or 3 dB contract")
    temp_path = output_path.with_name(f".{output_path.name}.{os.urandom(8).hex()}.pending.wav")
    if temp_path.exists() or temp_path.is_symlink():
        raise SelectedCenterError("temporary WAV name collision")
    audio_io.write_float32_wav(temp_path, output, sample_rate=16000,
                               peak_ceiling=None, overwrite=False)
    os.link(temp_path, output_path)
    temp_path.unlink()
    output_raw = _read_raw(output_path, "written output WAV", portable.MAX_WAV_BYTES)
    reread, reread_info = audio_io.read_canonical_wav(output_path, target_rate=16000)
    pcm_sha = hashlib.sha256(output.astype("<f4", copy=False).tobytes()).hexdigest()
    if (reread_info.frames != read_info.frames or reread_info.pcm_float32_sha256 != pcm_sha
            or not np.array_equal(reread, output)):
        raise SelectedCenterError("written FLOAT_32 WAV failed exact PCM/source-clock readback")
    if bool(torch.cuda.is_initialized()):
        raise SelectedCenterError("CPU-only selected application unexpectedly initialized CUDA")
    selected = artifact["selected_center"]
    receipt_body = {
        "schema": RECEIPT_SCHEMA,
        "status": RECEIPT_STATUS,
        "selection_status": FINAL_SELECTION_STATUS,
        "native_tfg_qualification_claim": False,
        "final_dev_or_confirm_admission_performed": False,
        "selected_artifact": {"path": str(artifact_path.resolve(strict=True)),
                              "bytes": len(artifact_raw),
                              "sha256": hashlib.sha256(artifact_raw).hexdigest(),
                              "artifact_sha256": artifact["artifact_sha256"]},
        "selected_center": selected,
        "evidence_pins": {key: {name: value for name, value in ref.items() if name != "raw_utf8"}
                          for key, ref in artifact["evidence"].items()},
        "portable_apply_source_pin": dict(PORTABLE_APPLY_PIN),
        "wrapper_source_pin": actual_wrapper_pin,
        "final_ingest_source_pin": dict(FINAL_INGEST_PIN),
        "final_ingest_dependency_source_pins": copy.deepcopy(FINAL_INGEST_DEPENDENCY_SOURCE_PINS),
        "seed_lineage": {"method_id": selected["source_seed_method_id"],
                         "source_snapshot_step": selected["source_snapshot_step"],
                         "source_snapshot_artifact_sha256": selected["source_snapshot_artifact_sha256"]},
        "input_wav": {"path": str(input_path.resolve(strict=True)), "bytes": len(input_raw),
                      "sha256": input_sha, "pcm_float32_sha256": read_info.pcm_float32_sha256,
                      "sample_count": int(read_info.frames), "sample_rate_hz": 16000,
                      "channels": 1, "source_encoding": read_info.source_encoding},
        "output_wav": {"path": str(output_path.resolve(strict=True)), "bytes": len(output_raw),
                       "sha256": hashlib.sha256(output_raw).hexdigest(),
                       "pcm_float32_sha256": pcm_sha,
                       "sample_count": int(reread_info.frames), "sample_rate_hz": 16000,
                       "channels": 1, "encoding": "FLOAT_32"},
        "clock": {"sample_rate_hz": 16000, "input_samples": int(read_info.frames),
                  "output_samples": int(reread_info.frames), "preserved": True},
        "theta": [list(theta32[i:i + 3]) for i in range(0, 9, 3)],
        "theta_little_endian_f32_sha256": theta_sha,
        "constructor_sha256": snapshot["head_constructor_sha256"],
        "runtime_source_pins": actual_code_pins,
        "rms": {"input_rms": float(baseline["input_rms"]),
                "target_rms": target_rms, "output_rms": output_rms,
                "output_rms_error": abs(output_rms - target_rms), "peak_ceiling": None,
                "peak_limiter_applied": False},
        "projection": {**hook_state, "bound_db": 3.0, "within_bound": True,
                       "projection_is_global_scale": True},
        "runtime": {"python_version": platform.python_version(),
                    "python_executable": str(Path(sys.executable).resolve()),
                    "torch_version": str(torch.__version__), "numpy_version": str(np.__version__),
                    "device": "cpu", "torch_num_threads": int(torch.get_num_threads()),
                    "torch_num_interop_threads": int(torch.get_num_interop_threads()),
                    "cuda_initialized": bool(torch.cuda.is_initialized())},
        "scope": "selected_fit_center_waveform_application_only; no TFG render/score or Dev/Confirm admission",
    }
    receipt = dict(receipt_body)
    receipt["self_sha256"] = hashlib.sha256(portable.canonical_bytes(receipt_body)).hexdigest()
    receipt_raw = json.dumps(receipt, sort_keys=True, ensure_ascii=False,
                             indent=2, allow_nan=False).encode("utf-8") + b"\n"
    portable._write_exclusive_json(receipt_out, receipt_raw, "selected-center application receipt")
    return {"path": str(receipt_out.resolve(strict=True)), "bytes": len(receipt_raw),
            "sha256": hashlib.sha256(receipt_raw).hexdigest(),
            "self_sha256": receipt["self_sha256"], "status": RECEIPT_STATUS,
            "selected_method_id": selected["method_id"], "output_wav": receipt["output_wav"]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    seal = sub.add_parser("seal-selected", help="reverify final selection and seal selected center evidence")
    seal.add_argument("--final-center-index", type=Path, required=True)
    seal.add_argument("--selector-receipt", type=Path, required=True)
    seal.add_argument("--seed-artifact", type=Path, required=True)
    seal.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    seal.add_argument("--output-artifact", type=Path, required=True)
    apply = sub.add_parser("apply-selected", help="apply a sealed selected center to mono 16 kHz PCM")
    apply.add_argument("--artifact", type=Path, required=True)
    apply.add_argument("--final-center-index", type=Path,
                       help="optional exact-byte replay against the original Fit tree")
    apply.add_argument("--selector-receipt", type=Path,
                       help="required together with --final-center-index when supplied")
    apply.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[2])
    apply.add_argument("--input-wav", type=Path, required=True)
    apply.add_argument("--output-wav", type=Path, required=True)
    apply.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "seal-selected":
            result = seal_selected_center_artifact(
                final_center_index_path=args.final_center_index,
                selector_receipt_path=args.selector_receipt,
                seed_artifact_path=args.seed_artifact,
                source_root=args.source_root,
                output_artifact_path=args.output_artifact)
        else:
            result = apply_selected_artifact(
                artifact_path=args.artifact,
                final_center_index_path=args.final_center_index,
                selector_receipt_path=args.selector_receipt,
                source_root=args.source_root,
                input_wav=args.input_wav,
                output_wav=args.output_wav,
                receipt_path=args.receipt)
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED",
                          "error": f"{type(exc).__name__}: {exc}"},
                         sort_keys=True, ensure_ascii=False), file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
