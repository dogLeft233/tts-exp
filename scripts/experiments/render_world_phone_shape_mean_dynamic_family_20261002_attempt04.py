#!/usr/bin/env python3
"""Render sealed WORLD phone-shape factorial family outputs (attempt 02).

This renderer consumes only the independent consumer's sealed JSON/JSONL/NPZ
outputs. It does not read source waveforms, videos, scores, or model artifacts.
The root-approved input binding is required and must bind the complete consumer
output manifest, source code, analysis plan, physical manifest, and design.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


BINDING_SCHEMA = "ROOT_APPROVED_WORLD_PHONE_SHAPE_FIGURE_INPUT_V2"
BINDING_PAYLOAD_FIELD = "payload_sha256"
FAMILY_SIZE = 11_776
GLOBAL_SIZE = 3_456
LOCAL_PHONE_SIZE = 8_192
LOCAL_OVERALL_SIZE = 128
SPEAKERS = 40
CLIPS = 80
CONTEXTS = 160
BOOTSTRAP_DRAWS = 20_000
BOOTSTRAP_SEED = 20_260_930
DIRECTIONS = ("tts_to_natural", "natural_to_tts")
AUDIO_DOMAINS = ("A_RAW", "A_RT")
GEOMETRIES = ("raw", "unit")
SUPPORTS = ("full", "valid", "guard20", "guard15")
METRICS = ("C_fixed_anchor", "B_median", "D_min", "D0", "search_gain", "optimal_offset_frames")
CONTRASTS = (
    "mean", "trajectory", "joint", "grid_cycle", "mean_minus_trajectory",
    "mean_beyond_grid", "trajectory_beyond_grid", "joint_beyond_grid", "factorial_interaction",
)
LOCAL_TREATMENTS = ("q10", "q01", "q11", "SELF_GRID")
PHONE_LABELS = (
    "a˥˩", "i˥", "i˥˩", "i˧˥", "j", "n", "o˥˩", "o˧˥",
    "t", "ts", "tɕ", "w", "ŋ", "ɕ", "ʂ", "ʈʂ",
)
PHONE_PHASES = ("whole", "begin", "middle", "end")
GLOBAL_ESTIMANDS = ("observed", "time_excess")
LOCAL_ESTIMANDS = ("observed", "time_excess")
GLOBAL_CONTEXT_FILE = "global_context_index.jsonl"
RF_CONTEXT_FILE = "receptive_field_context_index.jsonl"
ENDPOINTS_FILE = "family_endpoints.jsonl"
RECONSTRUCTION_FILE = "family_reconstruction.npz"
SCHEMA_FILE = "analysis_schema.json"
MANIFEST_FILE = "independent_output_manifest.json"

PALETTE = {
    ("tts_to_natural", "A_RAW"): "#2369a1",
    ("tts_to_natural", "A_RT"): "#6d9ec5",
    ("natural_to_tts", "A_RAW"): "#be5b31",
    ("natural_to_tts", "A_RT"): "#d89575",
}
GEOMETRY_MARKER = {"raw": "o", "unit": "s"}


class RenderError(RuntimeError):
    """Raised when an input is not exactly bound or family-complete as declared."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot read JSON {path}: {type(exc).__name__}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8", newline="") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    item = json.loads(line)
                    if not isinstance(item, dict):
                        raise RenderError(f"{path}:{line_number} is not a JSON object")
                    rows.append(item)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RenderError(f"cannot read JSONL {path}: {type(exc).__name__}: {exc}") from exc
    return rows


def _sha_text(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
        raise RenderError(f"{label} must be a lowercase 64-character SHA-256")
    return value


def _resolve_path(value: Any, base: Path, label: str, *, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RenderError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    try:
        return path.resolve(strict=must_exist)
    except OSError as exc:
        raise RenderError(f"{label} path does not resolve: {path}") from exc


def _safe_child(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise RenderError(f"{label} must be a nonempty relative path")
    candidate = (root / relative).resolve(strict=True)
    try:
        candidate.relative_to(root.resolve(strict=True))
    except ValueError as exc:
        raise RenderError(f"{label} escapes the sealed output root: {relative}") from exc
    return candidate


def _fingerprint(value: dict[str, Any], excluded_field: str) -> str:
    return hashlib.sha256(_canonical({key: item for key, item in value.items() if key != excluded_field})).hexdigest()


def _expected_family() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for contrast in CONTRASTS:
        for direction in DIRECTIONS:
            for geometry in GEOMETRIES:
                for support in SUPPORTS:
                    for metric in METRICS:
                        for audio_domain in AUDIO_DOMAINS:
                            for estimand in GLOBAL_ESTIMANDS:
                                key = ("global", contrast, direction, geometry, support, metric, audio_domain, estimand)
                                entries.append({
                                    "key": key, "family": "global", "contrast": contrast,
                                    "direction": direction, "geometry": geometry, "support": support,
                                    "metric": metric, "audio_domain": audio_domain, "estimand": estimand,
                                })
    for treatment in LOCAL_TREATMENTS:
        for direction in DIRECTIONS:
            for geometry in GEOMETRIES:
                for support in SUPPORTS:
                    for phone in PHONE_LABELS:
                        for phase in PHONE_PHASES:
                            for estimand in LOCAL_ESTIMANDS:
                                key = ("local", treatment, direction, geometry, support, phone, phase, estimand)
                                entries.append({
                                    "key": key, "family": "local_phone", "treatment": treatment,
                                    "direction": direction, "geometry": geometry, "support": support,
                                    "phone": phone, "phase": phase, "estimand": estimand,
                                })
    for treatment in LOCAL_TREATMENTS:
        for direction in DIRECTIONS:
            for geometry in GEOMETRIES:
                for support in SUPPORTS:
                    for estimand in LOCAL_ESTIMANDS:
                        key = ("overall", treatment, direction, geometry, support, estimand)
                        entries.append({
                            "key": key, "family": "local_overall", "treatment": treatment,
                            "direction": direction, "geometry": geometry, "support": support,
                            "estimand": estimand,
                        })
    counts = {
        "global": sum(row["family"] == "global" for row in entries),
        "local_phone": sum(row["family"] == "local_phone" for row in entries),
        "local_overall": sum(row["family"] == "local_overall" for row in entries),
    }
    if len(entries) != FAMILY_SIZE or len({entry["key"] for entry in entries}) != FAMILY_SIZE:
        raise RenderError(f"family construction is malformed: {len(entries)} entries")
    if counts != {"global": GLOBAL_SIZE, "local_phone": LOCAL_PHONE_SIZE, "local_overall": LOCAL_OVERALL_SIZE}:
        raise RenderError(f"frozen family cardinalities differ: {counts}")
    return entries


def _verify_binding(binding_path: Path, expected_raw_sha256: str, source_path: Path) -> dict[str, Any]:
    expected_raw_sha256 = _sha_text(expected_raw_sha256, "input binding raw SHA")
    observed_raw_sha256 = _sha_file(binding_path)
    if observed_raw_sha256 != expected_raw_sha256:
        raise RenderError("root input binding raw SHA does not match the command-line seal")
    binding = _read_json(binding_path)
    if not isinstance(binding, dict):
        raise RenderError("root input binding must be a JSON object")
    if binding.get("schema") != BINDING_SCHEMA or binding.get("status") != "FROZEN":
        raise RenderError("input binding is not a frozen root-approved figure binding")
    payload_sha = _fingerprint(binding, BINDING_PAYLOAD_FIELD)
    if _sha_text(binding.get(BINDING_PAYLOAD_FIELD), "binding payload SHA") != payload_sha:
        raise RenderError("input binding canonical payload SHA is invalid")
    renderer = binding.get("renderer")
    if not isinstance(renderer, dict):
        raise RenderError("binding lacks renderer source provenance")
    renderer_path = _resolve_path(renderer.get("source_path"), binding_path.parent, "renderer source")
    current_path = source_path.resolve(strict=True)
    if renderer_path != current_path:
        raise RenderError("binding renderer source path does not identify this file")
    renderer_sha = _sha_text(renderer.get("source_sha256"), "renderer source SHA")
    if _sha_file(current_path) != renderer_sha:
        raise RenderError("renderer source differs from its frozen binding")
    return {
        "binding": binding,
        "raw_sha256": observed_raw_sha256,
        "payload_sha256": payload_sha,
        "renderer_source_sha256": renderer_sha,
    }


def _validate_binding_contract(binding: dict[str, Any], binding_path: Path) -> tuple[Path, dict[str, Any]]:
    analysis = binding.get("analysis_output")
    expected = binding.get("expected")
    if not isinstance(analysis, dict) or not isinstance(expected, dict):
        raise RenderError("binding must contain analysis_output and expected objects")
    output_root = _resolve_path(analysis.get("root"), binding_path.parent, "analysis output root")
    if not output_root.is_dir():
        raise RenderError("analysis output root is not a directory")
    manifest_rel = analysis.get("manifest_relative_path", MANIFEST_FILE)
    manifest_path = _safe_child(output_root, manifest_rel, "independent output manifest")
    manifest_sha = _sha_text(analysis.get("manifest_raw_sha256"), "analysis manifest raw SHA")
    if _sha_file(manifest_path) != manifest_sha:
        raise RenderError("independent output manifest raw SHA differs from the root binding")
    required_expected = {
        "consumer_source_path", "consumer_source_sha256", "execution_plan_sha256", "physical_manifest_sha256",
        "design_sha256", "metric_addendum_sha256", "phone_inventory_sha256",
        "planned_family_endpoint_count", "planned_global_endpoint_count",
        "planned_local_phone_endpoint_count", "planned_local_overall_endpoint_count",
        "bootstrap_draws", "bootstrap_seed",
    }
    if set(expected) != required_expected:
        raise RenderError(f"binding expected metadata keys differ: {sorted(set(expected) ^ required_expected)}")
    consumer_source_path = _resolve_path(expected["consumer_source_path"], binding_path.parent, "family consumer source")
    consumer_source_sha = _sha_text(expected["consumer_source_sha256"], "binding expected consumer_source_sha256")
    if _sha_file(consumer_source_path) != consumer_source_sha:
        raise RenderError("family consumer source file does not match its root-approved SHA")
    for field in (
        "execution_plan_sha256", "physical_manifest_sha256",
        "design_sha256", "metric_addendum_sha256", "phone_inventory_sha256",
    ):
        _sha_text(expected[field], f"binding expected {field}")
    numeric_expected = {
        "planned_family_endpoint_count": FAMILY_SIZE,
        "planned_global_endpoint_count": GLOBAL_SIZE,
        "planned_local_phone_endpoint_count": LOCAL_PHONE_SIZE,
        "planned_local_overall_endpoint_count": LOCAL_OVERALL_SIZE,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
    }
    for field, wanted in numeric_expected.items():
        if expected.get(field) != wanted:
            raise RenderError(f"binding {field} is {expected.get(field)!r}, expected {wanted}")
    return output_root, {"manifest_path": manifest_path, "manifest_raw_sha256": manifest_sha, "expected": expected}


def _verify_output_tree(root: Path, manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    outputs = manifest.get("output_files")
    if not isinstance(outputs, list):
        raise RenderError("independent output manifest lacks output_files list")
    expected: dict[str, dict[str, Any]] = {}
    for item in outputs:
        if not isinstance(item, dict) or set(item) != {"path", "sha256", "bytes"}:
            raise RenderError("malformed independent output file entry")
        rel = str(item["path"])
        _sha_text(item["sha256"], f"output file {rel} SHA")
        if rel in expected:
            raise RenderError(f"duplicate output file path: {rel}")
        expected[rel] = item
    if MANIFEST_FILE in expected:
        raise RenderError("self-referential output manifest entry is not permitted")
    actual_files: set[str] = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise RenderError(f"symbolic link inside sealed analysis output: {path}")
        if path.is_file() and path.resolve(strict=True) != (root / MANIFEST_FILE).resolve(strict=True):
            actual_files.add(path.relative_to(root).as_posix())
    if actual_files != set(expected):
        missing = sorted(set(expected) - actual_files)[:5]
        extra = sorted(actual_files - set(expected))[:5]
        raise RenderError(f"analysis output tree differs from its manifest; missing={missing}, extra={extra}")
    verified: dict[str, dict[str, Any]] = {}
    for rel, item in expected.items():
        path = _safe_child(root, rel, f"sealed output {rel}")
        size = path.stat().st_size
        if size != int(item["bytes"]):
            raise RenderError(f"sealed output size differs: {rel}: {size} != {item['bytes']}")
        observed_sha = _sha_file(path)
        if observed_sha != item["sha256"]:
            raise RenderError(f"sealed output SHA differs: {rel}")
        verified[rel] = {"path": str(path), "bytes": size, "sha256": observed_sha}
    return verified


def _parse_family_key(raw: Any, index: int) -> tuple[Any, ...]:
    if not isinstance(raw, str):
        raise RenderError(f"family endpoint {index} has no serialized endpoint_key")
    try:
        key = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RenderError(f"family endpoint {index} key is invalid JSON") from exc
    if not isinstance(key, list):
        raise RenderError(f"family endpoint {index} key is not an array")
    return tuple(key)


def _verify_report(root: Path, manifest_path: Path, manifest: dict[str, Any], expected: dict[str, Any], verified_outputs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    if manifest.get("schema_version") != 1 or manifest.get("status") != "ANALYSIS_COMPLETE":
        raise RenderError("consumer output manifest is not ANALYSIS_COMPLETE schema version 1")
    if manifest.get("planned_family_endpoint_count") != FAMILY_SIZE:
        raise RenderError("consumer manifest planned family count differs from 11,776")
    if manifest.get("consumer_source_sha256") != expected["consumer_source_sha256"]:
        raise RenderError("consumer source SHA differs from the root-approved input binding")
    for field in ("execution_plan_sha256", "physical_manifest_sha256", "design_sha256", "metric_addendum_sha256", "phone_inventory_sha256"):
        if manifest.get(field) != expected[field]:
            raise RenderError(f"consumer manifest {field} differs from the root-approved input binding")
    if manifest.get("context_count") != CONTEXTS or manifest.get("unique_clip_count") != CLIPS or manifest.get("speaker_count") != SPEAKERS:
        raise RenderError("consumer manifest context/clip/speaker counts differ from the frozen design")
    if manifest.get("bootstrap_indices_shape") != [BOOTSTRAP_DRAWS, SPEAKERS] or manifest.get("bootstrap_indices_dtype") != "int16":
        raise RenderError("consumer manifest speaker bootstrap index shape/dtype differs")
    if manifest.get("bootstrap_counts_dtype") != "uint8" or manifest.get("bootstrap_max_t_draw_count") != BOOTSTRAP_DRAWS:
        raise RenderError("consumer manifest bootstrap count/max-t metadata differs")
    if manifest.get("bootstrap_no_draws_by_endpoint_matrix") is not True:
        raise RenderError("consumer manifest does not attest to the frozen shared bootstrap design")
    if manifest.get("planned_family_endpoint_count") != expected["planned_family_endpoint_count"]:
        raise RenderError("consumer output family count differs from binding")
    critical = manifest.get("family_critical_99_max_t")
    if not isinstance(critical, (int, float)) or not math.isfinite(float(critical)) or float(critical) <= 0:
        raise RenderError("consumer manifest has an invalid shared 99% max-|t| critical value")
    family_status = manifest.get("family_support_status")
    if family_status not in {"COMPLETE", "INCOMPLETE_SUPPORT"}:
        raise RenderError("consumer family support status is not recognized")

    schema = _read_json(_safe_child(root, SCHEMA_FILE, "analysis schema"))
    if not isinstance(schema, dict):
        raise RenderError("analysis schema is not an object")
    schema_expected = {
        "planned_family_endpoint_count": FAMILY_SIZE,
        "planned_global_endpoint_count": GLOBAL_SIZE,
        "planned_local_phone_endpoint_count": LOCAL_PHONE_SIZE,
        "planned_local_overall_endpoint_count": LOCAL_OVERALL_SIZE,
        "direction_order": list(DIRECTIONS), "cell_order": ["q00", "q10", "q01", "q11", "SELF_GRID"],
        "audio_domains": list(AUDIO_DOMAINS), "supports": list(SUPPORTS),
        "geometries": list(GEOMETRIES), "metrics": list(METRICS), "contrasts": list(CONTRASTS),
        "local_treatments": list(LOCAL_TREATMENTS), "phone_labels_cal_only": list(PHONE_LABELS),
        "phone_phases": list(PHONE_PHASES),
    }
    for field, wanted in schema_expected.items():
        if schema.get(field) != wanted:
            raise RenderError(f"analysis schema {field} differs from the frozen family")
    bootstrap = schema.get("speaker_bootstrap")
    if not isinstance(bootstrap, dict) or bootstrap.get("draws") != BOOTSTRAP_DRAWS or bootstrap.get("seed") != BOOTSTRAP_SEED or bootstrap.get("generator") != "PCG64":
        raise RenderError("analysis schema bootstrap seed/draws/generator differ")
    if schema.get("time_null", {}).get("draws") != 100:
        raise RenderError("analysis schema time-null draw count differs from the sealed family")

    endpoint_rows = _read_jsonl(_safe_child(root, ENDPOINTS_FILE, "family endpoint rows"))
    expected_entries = _expected_family()
    if len(endpoint_rows) != FAMILY_SIZE:
        raise RenderError(f"family endpoint file has {len(endpoint_rows)} rows, expected {FAMILY_SIZE}")
    status_counts = {"estimable": 0, "INCOMPLETE_SUPPORT": 0}
    indexed_rows: list[dict[str, Any]] = []
    for index, (row, expected_entry) in enumerate(zip(endpoint_rows, expected_entries, strict=True)):
        if row.get("planned_family_index") != index:
            raise RenderError(f"family row order/index differs at {index}")
        if _parse_family_key(row.get("endpoint_key"), index) != expected_entry["key"]:
            raise RenderError(f"family endpoint key differs from frozen _family_entries order at {index}")
        for field, value in expected_entry.items():
            if field == "key":
                continue
            if row.get(field) != value:
                raise RenderError(f"family row {index} {field} differs from planned family")
        status = row.get("status")
        if status not in status_counts:
            raise RenderError(f"family endpoint has unexpected status at {index}: {status!r}")
        status_counts[status] += 1
        if row.get("family_critical_99_max_t") is None or not math.isclose(float(row["family_critical_99_max_t"]), float(critical), rel_tol=0, abs_tol=2e-11):
            raise RenderError(f"family endpoint {index} does not bind the shared max-|t| critical value")
        estimate = row.get("estimate_speaker_equal_mean")
        se = row.get("bootstrap_se")
        ci = row.get("ci99_simultaneous_max_t")
        if status == "estimable":
            if estimate is None or se is None or ci is None or not isinstance(ci, list) or len(ci) != 2:
                raise RenderError(f"estimable endpoint {index} lacks its estimate, SE, or simultaneous interval")
            estimate, se = float(estimate), float(se)
            low, high = float(ci[0]), float(ci[1])
            if not all(math.isfinite(value) for value in (estimate, se, low, high)) or se < 0 or low > estimate or estimate > high:
                raise RenderError(f"estimable endpoint {index} has invalid statistics")
        elif estimate is not None or se is not None or ci is not None:
            raise RenderError(f"incomplete endpoint {index} must keep estimate/SE/CI null")
        indexed_rows.append(row)
    if manifest.get("estimable_family_endpoint_count") != status_counts["estimable"]:
        raise RenderError("manifest estimable endpoint count differs from row statuses")
    if manifest.get("unestimable_family_endpoint_count") != status_counts["INCOMPLETE_SUPPORT"]:
        raise RenderError("manifest incomplete endpoint count differs from row statuses")
    expected_family_status = "COMPLETE" if status_counts["INCOMPLETE_SUPPORT"] == 0 else "INCOMPLETE_SUPPORT"
    if family_status != expected_family_status:
        raise RenderError("manifest family support status differs from all 11,776 endpoint statuses")

    support_status_counts = {
        support: {
            status: sum(row["support"] == support and row["status"] == status for row in indexed_rows)
            for status in ("estimable", "INCOMPLETE_SUPPORT")
        }
        for support in SUPPORTS
    }

    global_rows = [row for row in indexed_rows if row["family"] == "global"]
    local_phone_rows = [row for row in indexed_rows if row["family"] == "local_phone"]
    local_overall_rows = [row for row in indexed_rows if row["family"] == "local_overall"]
    if (len(global_rows), len(local_phone_rows), len(local_overall_rows)) != (GLOBAL_SIZE, LOCAL_PHONE_SIZE, LOCAL_OVERALL_SIZE):
        raise RenderError("family endpoint category counts do not match the frozen counts")

    rf_rows = _read_jsonl(_safe_child(root, RF_CONTEXT_FILE, "receptive-field context index"))
    if len(rf_rows) != CONTEXTS:
        raise RenderError("receptive-field context index does not retain all 160 contexts")
    context_ids: set[tuple[str, str]] = set()
    short_ids: set[tuple[str, str]] = set()
    guard20_empty_ids: set[tuple[str, str]] = set()
    short_samples: set[str] = set()
    for row in rf_rows:
        sample = str(row.get("sample_id", ""))
        direction = str(row.get("direction", ""))
        if not sample or direction not in DIRECTIONS or (sample, direction) in context_ids:
            raise RenderError("receptive-field context index has a missing/duplicate context identity")
        context_ids.add((sample, direction))
        query_count = row.get("syncnet_query_count")
        if not isinstance(query_count, int) or query_count < 0:
            raise RenderError("receptive-field context index lacks a valid SyncNet query count")
        if query_count < 40:
            short_ids.add((sample, direction))
            short_samples.add(sample)
        if query_count <= 40:
            # Frozen support rule yields an empty guard20 interval at n <= 40.
            guard20_empty_ids.add((sample, direction))
    if len({sample for sample, _ in context_ids}) != CLIPS:
        raise RenderError("receptive-field context index does not retain all 80 clips")

    with np.load(_safe_child(root, RECONSTRUCTION_FILE, "family reconstruction NPZ"), allow_pickle=False) as archive:
        required_npz = {
            "speaker_order_unicode", "endpoint_key_unicode", "estimable_mask_uint8", "speaker_values_float64",
            "clip_values_by_endpoint_estimand_float64", "clip_time_null_means_float64",
            "bootstrap_indices_int16", "bootstrap_counts_uint8", "family_max_t_draws_float64",
        }
        if set(archive.files) != required_npz:
            raise RenderError("family reconstruction NPZ keys differ from the consumer schema")
        arrays = {name: archive[name].copy() for name in sorted(required_npz)}
    expected_dtypes_shapes = {
        "speaker_order_unicode": (np.dtype("<U16"), (SPEAKERS,)),
        "endpoint_key_unicode": (np.dtype("<U512"), (FAMILY_SIZE,)),
        "estimable_mask_uint8": (np.dtype("uint8"), (FAMILY_SIZE,)),
        "speaker_values_float64": (np.dtype("float64"), (FAMILY_SIZE, SPEAKERS)),
        "clip_values_by_endpoint_estimand_float64": (np.dtype("float64"), (FAMILY_SIZE, CLIPS)),
        "clip_time_null_means_float64": (np.dtype("float64"), (FAMILY_SIZE, CLIPS)),
        "bootstrap_indices_int16": (np.dtype("int16"), (BOOTSTRAP_DRAWS, SPEAKERS)),
        "bootstrap_counts_uint8": (np.dtype("uint8"), (BOOTSTRAP_DRAWS, SPEAKERS)),
        "family_max_t_draws_float64": (np.dtype("float64"), (BOOTSTRAP_DRAWS,)),
    }
    for name, (dtype, shape) in expected_dtypes_shapes.items():
        if arrays[name].dtype != dtype or arrays[name].shape != shape:
            raise RenderError(f"family reconstruction {name} dtype/shape is {arrays[name].dtype}/{arrays[name].shape}, expected {dtype}/{shape}")
    keys_from_rows = np.asarray([row["endpoint_key"] for row in indexed_rows], dtype="U512")
    if not np.array_equal(arrays["endpoint_key_unicode"], keys_from_rows):
        raise RenderError("family reconstruction endpoint key array differs from endpoint JSONL")
    mask = arrays["estimable_mask_uint8"]
    expected_mask = np.asarray([row["status"] == "estimable" for row in indexed_rows], dtype=np.uint8)
    if not np.array_equal(mask, expected_mask) or _array_sha(mask) != manifest.get("family_mask_sha256"):
        raise RenderError("family reconstruction mask differs from endpoint statuses or manifest SHA")
    if int(mask.sum()) != status_counts["estimable"]:
        raise RenderError("family reconstruction mask count differs from endpoint rows")
    if not np.isfinite(arrays["family_max_t_draws_float64"]).all():
        raise RenderError("family max-|t| bootstrap draws contain nonfinite values")
    reconstructed_critical = float(np.quantile(arrays["family_max_t_draws_float64"], 0.99, method="linear"))
    if not math.isclose(reconstructed_critical, float(critical), rel_tol=2e-11, abs_tol=2e-11):
        raise RenderError("manifest 99% family critical value differs from the sealed max-|t| draw array")
    indices = arrays["bootstrap_indices_int16"]
    counts = arrays["bootstrap_counts_uint8"]
    if (indices < 0).any() or (indices >= SPEAKERS).any() or not np.all(counts.sum(axis=1, dtype=np.int64) == SPEAKERS):
        raise RenderError("bootstrap index/count arrays violate the 40-speaker resampling contract")
    regenerated_counts = np.stack([np.bincount(row, minlength=SPEAKERS) for row in indices]).astype(np.uint8)
    if not np.array_equal(regenerated_counts, counts):
        raise RenderError("bootstrap count matrix does not match the sealed index draws")

    # Check FULL/VALID pairs directly; they remain separate planned family entries.
    by_non_support: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for row in indexed_rows:
        key = tuple(_parse_family_key(row["endpoint_key"], int(row["planned_family_index"])))
        without_support = key[:4] + key[5:]
        by_non_support.setdefault(without_support, {})[key[4]] = row
    full_valid_pairs = 0
    duplicate_estimable = 0
    duplicate_incomplete = 0
    for supports in by_non_support.values():
        if "full" not in supports or "valid" not in supports:
            continue
        full_valid_pairs += 1
        left, right = supports["full"], supports["valid"]
        fields = ("status", "estimate_speaker_equal_mean", "bootstrap_se", "ci99_simultaneous_max_t")
        if all(left.get(field) == right.get(field) for field in fields):
            if left["status"] == "estimable":
                duplicate_estimable += 1
            else:
                duplicate_incomplete += 1
    if full_valid_pairs != 2_944:
        raise RenderError(f"FULL/VALID planned pair count differs: {full_valid_pairs}")

    return {
        "manifest_sha256": _sha_file(manifest_path),
        "manifest_payload_sha256": _fingerprint(manifest, "manifest_sha256") if "manifest_sha256" in manifest else None,
        "output_file_count": len(verified_outputs),
        "status_counts": status_counts,
        "support_status_counts": support_status_counts,
        "critical": float(critical),
        "family_support_status": family_status,
        "global_rows": global_rows,
        "local_phone_rows": local_phone_rows,
        "local_overall_rows": local_overall_rows,
        "short_context_count": len(short_ids),
        "short_clip_count": len(short_samples),
        "guard20_empty_context_count": len(guard20_empty_ids),
        "full_valid_planned_pairs": full_valid_pairs,
        "full_valid_equal_estimable_pairs": duplicate_estimable,
        "full_valid_equal_incomplete_pairs": duplicate_incomplete,
        "reconstruction_npz_sha256": verified_outputs[RECONSTRUCTION_FILE]["sha256"],
        "endpoints_sha256": verified_outputs[ENDPOINTS_FILE]["sha256"],
        "analysis_schema_sha256": verified_outputs[SCHEMA_FILE]["sha256"],
        "global_context_sha256": verified_outputs[GLOBAL_CONTEXT_FILE]["sha256"],
        "rf_context_sha256": verified_outputs[RF_CONTEXT_FILE]["sha256"],
    }


def _csv_record(row: dict[str, Any]) -> dict[str, Any]:
    fields = (
        "planned_family_index", "endpoint_key", "family", "contrast", "treatment", "direction", "geometry",
        "support", "metric", "audio_domain", "estimand", "phone", "phase", "status",
        "estimate_speaker_equal_mean", "bootstrap_se", "family_critical_99_max_t",
    )
    record = {field: row.get(field) for field in fields}
    ci = row.get("ci99_simultaneous_max_t")
    record["ci99_lower"] = ci[0] if isinstance(ci, list) and len(ci) == 2 else None
    record["ci99_upper"] = ci[1] if isinstance(ci, list) and len(ci) == 2 else None
    return record


CSV_FIELDS = (
    "planned_family_index", "endpoint_key", "family", "contrast", "treatment", "direction", "geometry",
    "support", "metric", "audio_domain", "estimand", "phone", "phase", "status",
    "estimate_speaker_equal_mean", "bootstrap_se", "ci99_lower", "ci99_upper", "family_critical_99_max_t",
)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(_csv_record(row))
            count += 1
    return count


def _ci(row: dict[str, Any]) -> tuple[float, float] | None:
    raw = row.get("ci99_simultaneous_max_t")
    if not isinstance(raw, list) or len(raw) != 2:
        return None
    return float(raw[0]), float(raw[1])


def _forest_style(ax: Any) -> None:
    ax.axvline(0.0, color="#303744", linewidth=0.9, linestyle="--", zorder=1)
    ax.grid(axis="x", color="#e4e8ee", linewidth=0.65)
    ax.set_axisbelow(True)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)


def _global_forest(path: Path, rows: list[dict[str, Any]], critical: float, short_contexts: int, guard20_incomplete: int, full_valid_equal: int) -> dict[str, Any]:
    plotted: dict[str, list[dict[str, Any]]] = {}
    for contrast in CONTRASTS:
        selected = [row for row in rows if row["contrast"] == contrast and row["metric"] == "C_fixed_anchor"]
        expected_count = len(DIRECTIONS) * len(AUDIO_DOMAINS) * len(GEOMETRIES) * len(SUPPORTS) * len(GLOBAL_ESTIMANDS)
        if len(selected) != expected_count:
            raise RenderError(f"primary forest contrast {contrast} has {len(selected)} endpoints, expected {expected_count}")
        plotted[contrast] = selected
    finite_intervals = [
        bound
        for rows_for_contrast in plotted.values()
        for row in rows_for_contrast
        for bound in (_ci(row) or ())
    ]
    left = min([0.0, *finite_intervals])
    right = max([0.0, *finite_intervals])
    margin = max((right - left) * 0.06, 0.002)
    xlim = (left - margin, right + margin)
    legend_handles = [
        Line2D([0], [0], color=color, marker="o", linestyle="none", label=f"{direction} · {audio}")
        for (direction, audio), color in PALETTE.items()
    ] + [
        Line2D([0], [0], color="#303744", marker=GEOMETRY_MARKER[geometry], linestyle="none", label=f"{geometry} geometry")
        for geometry in GEOMETRIES
    ] + [
        Line2D([0], [0], color="#303744", marker="o", markerfacecolor=("#303744" if estimand == "observed" else "white"), linestyle="none", label=estimand)
        for estimand in GLOBAL_ESTIMANDS
    ]
    with PdfPages(path, metadata={"Title": "WORLD phone-shape factorial family: fixed-anchor global forest", "Author": "TTS-TFG analysis"}) as pdf:
        for contrast in CONTRASTS:
            fig, axes = plt.subplots(4, 1, figsize=(14.0, 17.0), sharex=True, constrained_layout=False)
            rows_for_contrast = plotted[contrast]
            support_count = 0
            for ax, support in zip(axes, SUPPORTS, strict=True):
                panel = [row for row in rows_for_contrast if row["support"] == support]
                panel.sort(key=lambda row: (
                    DIRECTIONS.index(row["direction"]), AUDIO_DOMAINS.index(row["audio_domain"]),
                    GEOMETRIES.index(row["geometry"]), GLOBAL_ESTIMANDS.index(row["estimand"]),
                ))
                if len(panel) != 16:
                    raise RenderError(f"forest panel {contrast}/{support} has {len(panel)} rows, expected 16")
                support_count += len(panel)
                labels: list[str] = []
                for y, row in enumerate(panel):
                    direction = row["direction"]
                    audio = row["audio_domain"]
                    geometry = row["geometry"]
                    estimand = row["estimand"]
                    label = f"{direction} · {audio} · {geometry} · {estimand}"
                    labels.append(label)
                    color = PALETTE[(direction, audio)]
                    marker = GEOMETRY_MARKER[geometry]
                    if row["status"] == "estimable":
                        estimate = float(row["estimate_speaker_equal_mean"])
                        low, high = _ci(row) or (math.nan, math.nan)
                        ax.errorbar(
                            estimate, y, xerr=[[estimate - low], [high - estimate]],
                            fmt=marker, color=color, ecolor=color, elinewidth=0.8,
                            capsize=1.8, markersize=3.8,
                            markerfacecolor=(color if estimand == "observed" else "white"),
                            markeredgewidth=0.8, zorder=3,
                        )
                    else:
                        ax.text(0.005, y, "INCOMPLETE_SUPPORT", transform=ax.get_yaxis_transform(),
                                va="center", ha="left", fontsize=5.6, color="#777777")
                ax.set_yticks(np.arange(len(labels)), labels, fontsize=6.1)
                ax.invert_yaxis()
                ax.set_xlim(*xlim)
                _forest_style(ax)
                ax.set_title(f"{support} · all planned endpoints retained", loc="left", fontsize=9.2, pad=4)
                if support == "guard15":
                    ax.set_xlabel("Change in fixed-anchor SyncNet C (99% simultaneous max-|t| CI)", fontsize=9)
            if support_count != 64:
                raise RenderError(f"forest contrast {contrast} did not retain all 64 support rows")
            fig.suptitle(
                f"{contrast} · C_fixed_anchor\n11,776-endpoint shared speaker-bootstrap family; critical={critical:.6g}",
                x=0.06, y=0.988, ha="left", fontsize=13, fontweight="bold",
            )
            fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.55, 0.967),
                       ncol=4, frameon=False, fontsize=7.1)
            fig.text(
                0.06, 0.008,
                f"All 9 frozen global contrasts; 80 clips / 40 speakers; shared 20,000-draw PCG64 max-|t| 99% intervals. "
                f"Short contexts with query_count < 40 retained: {short_contexts}; guard20 endpoints marked INCOMPLETE_SUPPORT: {guard20_incomplete}. "
                f"FULL and VALID remain separate planned supports; exact equal estimable pairs: {full_valid_equal}. "
                "Open markers denote time_excess = observed minus the mean full-A time-permutation value.",
                fontsize=7.0, color="#505a68",
            )
            fig.subplots_adjust(left=0.25, right=0.98, top=0.925, bottom=0.055, hspace=0.48)
            pdf.savefig(fig, facecolor="white")
            plt.close(fig)
    return {"path": str(path), "pages": len(CONTRASTS), "points": len(rows) // len(METRICS) if len(METRICS) else 0, "xlim": list(xlim)}


def _guard15_primary_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select and validate the prospectively readable 144-endpoint overview."""
    selected = [
        row for row in rows
        if row.get("metric") == "C_fixed_anchor" and row.get("support") == "guard15"
    ]
    expected_count = len(CONTRASTS) * len(DIRECTIONS) * len(AUDIO_DOMAINS) * len(GEOMETRIES) * len(GLOBAL_ESTIMANDS)
    if len(selected) != expected_count:
        raise RenderError(f"guard15 C_fixed_anchor overview has {len(selected)} endpoints, expected {expected_count}")
    expected_keys = {
        (contrast, direction, geometry, audio_domain, estimand)
        for contrast in CONTRASTS
        for direction in DIRECTIONS
        for geometry in GEOMETRIES
        for audio_domain in AUDIO_DOMAINS
        for estimand in GLOBAL_ESTIMANDS
    }
    observed_keys = {
        (row.get("contrast"), row.get("direction"), row.get("geometry"), row.get("audio_domain"), row.get("estimand"))
        for row in selected
    }
    if observed_keys != expected_keys:
        raise RenderError("guard15 C_fixed_anchor overview endpoint keys differ from the frozen 144-cell design")
    if any(row.get("status") != "estimable" for row in selected):
        raise RenderError("guard15 C_fixed_anchor overview requires all 144 endpoints estimable; no rows are omitted")
    return sorted(selected, key=lambda row: (
        CONTRASTS.index(row["contrast"]), GEOMETRIES.index(row["geometry"]),
        DIRECTIONS.index(row["direction"]), AUDIO_DOMAINS.index(row["audio_domain"]),
        GLOBAL_ESTIMANDS.index(row["estimand"]),
    ))


def _guard15_overview(png_path: Path, svg_path: Path, rows: list[dict[str, Any]], critical: float) -> dict[str, Any]:
    """Plot all 144 predeclared guard15 primary endpoints without selection."""
    expected_count = len(DIRECTIONS) * len(AUDIO_DOMAINS) * len(GLOBAL_ESTIMANDS)
    xlimits: dict[str, tuple[float, float]] = {}
    for geometry in GEOMETRIES:
        geometry_rows = [row for row in rows if row["geometry"] == geometry]
        if len(geometry_rows) != expected_count * len(CONTRASTS):
            raise RenderError(f"guard15 overview {geometry} column has the wrong endpoint count")
        bounds = [bound for row in geometry_rows for bound in (_ci(row) or ())]
        if len(bounds) != 2 * len(geometry_rows) or not all(math.isfinite(value) for value in bounds):
            raise RenderError(f"guard15 overview {geometry} column has a missing/nonfinite simultaneous interval")
        left, right = min([0.0, *bounds]), max([0.0, *bounds])
        margin = max((right - left) * 0.10, 0.002)
        xlimits[geometry] = (left - margin, right + margin)

    fig, axes = plt.subplots(len(CONTRASTS), len(GEOMETRIES), figsize=(17.5, 17.0), sharex="col")
    label_map = {
        ("tts_to_natural", "A_RAW", "observed"): "T→N · RAW · obs",
        ("tts_to_natural", "A_RAW", "time_excess"): "T→N · RAW · excess",
        ("tts_to_natural", "A_RT", "observed"): "T→N · RT · obs",
        ("tts_to_natural", "A_RT", "time_excess"): "T→N · RT · excess",
        ("natural_to_tts", "A_RAW", "observed"): "N→T · RAW · obs",
        ("natural_to_tts", "A_RAW", "time_excess"): "N→T · RAW · excess",
        ("natural_to_tts", "A_RT", "observed"): "N→T · RT · obs",
        ("natural_to_tts", "A_RT", "time_excess"): "N→T · RT · excess",
    }
    for contrast_i, contrast in enumerate(CONTRASTS):
        for geometry_i, geometry in enumerate(GEOMETRIES):
            ax = axes[contrast_i, geometry_i]
            panel = [row for row in rows if row["contrast"] == contrast and row["geometry"] == geometry]
            panel.sort(key=lambda row: (
                DIRECTIONS.index(row["direction"]), AUDIO_DOMAINS.index(row["audio_domain"]),
                GLOBAL_ESTIMANDS.index(row["estimand"]),
            ))
            if len(panel) != expected_count:
                raise RenderError(f"guard15 overview panel {contrast}/{geometry} has {len(panel)} rows")
            labels: list[str] = []
            for y, row in enumerate(panel):
                direction, audio, estimand = row["direction"], row["audio_domain"], row["estimand"]
                estimate = float(row["estimate_speaker_equal_mean"])
                interval = _ci(row)
                if interval is None:
                    raise RenderError("guard15 overview endpoint is missing its shared-family 99% interval")
                low, high = interval
                if not all(math.isfinite(value) for value in (estimate, low, high)) or low > estimate or estimate > high:
                    raise RenderError("guard15 overview endpoint has invalid estimate/interval values")
                color = PALETTE[(direction, audio)]
                marker = "o" if estimand == "observed" else "s"
                ax.errorbar(
                    estimate, y, xerr=[[estimate - low], [high - estimate]],
                    fmt=marker, color=color, ecolor=color, elinewidth=0.75,
                    capsize=1.5, markersize=3.3,
                    markerfacecolor=(color if estimand == "observed" else "white"),
                    markeredgewidth=0.75, zorder=3,
                )
                labels.append(label_map[(direction, audio, estimand)])
            ax.set_yticks(np.arange(len(labels)), labels, fontsize=6.7)
            ax.invert_yaxis()
            ax.set_xlim(*xlimits[geometry])
            _forest_style(ax)
            ax.set_title(f"{contrast} · {geometry}", loc="left", fontsize=8.7, pad=3)
            if contrast_i == len(CONTRASTS) - 1:
                ax.set_xlabel(f"Δ fixed-anchor SyncNet C · {geometry} units", fontsize=8.3)

    legend_handles = [
        Line2D([0], [0], color=color, marker="o", linestyle="none", label=f"{direction} · {audio}")
        for (direction, audio), color in PALETTE.items()
    ] + [
        Line2D([0], [0], color="#303744", marker="o", markerfacecolor="#303744", linestyle="none", label="observed"),
        Line2D([0], [0], color="#303744", marker="s", markerfacecolor="white", linestyle="none", label="time_excess"),
    ]
    fig.suptitle(
        f"Guard15 · C_fixed_anchor · all 9 global contrasts\n"
        f"144 endpoints · shared 11,776-endpoint speaker-bootstrap family · critical={critical:.5g}",
        x=0.07, y=0.994, ha="left", fontsize=13, fontweight="bold",
    )
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.70, 0.988),
               ncol=3, frameon=False, fontsize=7.0)
    fig.text(
        0.07, 0.012,
        "Guard15 was selected in advance for readable coverage with all 40 speakers. These intervals remain in the full shared 11,776-endpoint family; "
        "they are not a separate family. Guard20 incomplete endpoints remain in the full PDFs/CSVs and are not replaced by guard15. "
        "Raw and unit coordinates use separate x scales, shared within each geometry column. Bars are simultaneous 99% max-|t| intervals.",
        fontsize=7.2, color="#505a68",
    )
    fig.subplots_adjust(left=0.205, right=0.99, top=0.955, bottom=0.055, hspace=0.52, wspace=0.20)
    fig.savefig(png_path, dpi=220, facecolor="white", metadata={"Title": "Guard15 C_fixed_anchor overview"})
    fig.savefig(svg_path, facecolor="white", metadata={"Title": "Guard15 C_fixed_anchor overview", "Creator": "TTS-TFG analysis"})
    plt.close(fig)
    return {
        "png": {"path": str(png_path), "endpoints": len(rows), "xlimits_by_geometry": {key: list(value) for key, value in xlimits.items()}},
        "svg": {"path": str(svg_path), "endpoints": len(rows)},
        "support": "guard15", "metric": "C_fixed_anchor", "contrasts": list(CONTRASTS),
        "shared_family_size": FAMILY_SIZE, "shared_family_critical_99_max_t": critical,
        "shared_family_not_reestimated": True,
        "guard15_selection_note": "Prospectively selected for readable coverage with all 40 speakers; its intervals remain in the full shared family.",
        "guard20_note": "Guard20 incomplete endpoints remain in the full PDFs/CSVs and are not replaced by guard15.",
    }


def _local_heatmaps(path: Path, rows: list[dict[str, Any]], short_contexts: int, guard20_incomplete: int, full_valid_equal: int) -> dict[str, Any]:
    lookup: dict[tuple[str, str, str, str, str, str, str], dict[str, Any]] = {}
    finite_values: list[float] = []
    for row in rows:
        key = (row["treatment"], row["direction"], row["geometry"], row["support"], row["phone"], row["phase"], row["estimand"])
        if key in lookup:
            raise RenderError(f"duplicate local phone endpoint key: {key}")
        lookup[key] = row
        if row["status"] == "estimable":
            finite_values.append(float(row["estimate_speaker_equal_mean"]))
    if len(lookup) != LOCAL_PHONE_SIZE:
        raise RenderError(f"local phone heatmap expected {LOCAL_PHONE_SIZE} cells, got {len(lookup)}")
    vmax = max((abs(value) for value in finite_values), default=0.0)
    if not math.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("#e5e8ec")
    page_count = len(DIRECTIONS) * len(GEOMETRIES) * len(SUPPORTS) * len(LOCAL_ESTIMANDS)
    with PdfPages(path, metadata={"Title": "WORLD phone-local factorial family summaries", "Author": "TTS-TFG analysis"}) as pdf:
        for direction in DIRECTIONS:
            for geometry in GEOMETRIES:
                for support in SUPPORTS:
                    for estimand in LOCAL_ESTIMANDS:
                        fig, axes = plt.subplots(2, 2, figsize=(14.0, 12.0), constrained_layout=False)
                        image = None
                        for ax, treatment in zip(axes.flat, LOCAL_TREATMENTS, strict=True):
                            grid = np.full((len(PHONE_LABELS), len(PHONE_PHASES)), np.nan, dtype=np.float64)
                            status_grid = np.full(grid.shape, "INCOMPLETE_SUPPORT", dtype=object)
                            for phone_i, phone in enumerate(PHONE_LABELS):
                                for phase_i, phase in enumerate(PHONE_PHASES):
                                    row = lookup[(treatment, direction, geometry, support, phone, phase, estimand)]
                                    status_grid[phone_i, phase_i] = row["status"]
                                    if row["status"] == "estimable":
                                        grid[phone_i, phase_i] = float(row["estimate_speaker_equal_mean"])
                                        ci = _ci(row)
                                        if ci is not None and (ci[0] > 0.0 or ci[1] < 0.0):
                                            status_grid[phone_i, phase_i] = "CI_EXCLUDES_ZERO"
                            image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, norm=norm, aspect="auto", interpolation="nearest")
                            ax.set_xticks(np.arange(len(PHONE_PHASES)), PHONE_PHASES, fontsize=8)
                            ax.set_yticks(np.arange(len(PHONE_LABELS)), PHONE_LABELS, fontsize=8)
                            ax.set_title(f"{treatment} · local response relative to q00", fontsize=10, loc="left")
                            ax.set_xlabel("Phone phase")
                            for spine in ax.spines.values():
                                spine.set_visible(False)
                            ax.set_xticks(np.arange(-0.5, len(PHONE_PHASES), 1), minor=True)
                            ax.set_yticks(np.arange(-0.5, len(PHONE_LABELS), 1), minor=True)
                            ax.grid(which="minor", color="white", linewidth=1.2)
                            ax.tick_params(which="minor", bottom=False, left=False)
                            for phone_i in range(len(PHONE_LABELS)):
                                for phase_i in range(len(PHONE_PHASES)):
                                    if status_grid[phone_i, phase_i] != "estimable":
                                        if status_grid[phone_i, phase_i] == "CI_EXCLUDES_ZERO":
                                            ax.scatter([phase_i], [phone_i], marker="o", s=11, c="#202733", linewidths=0.3, zorder=4)
                                        else:
                                            ax.text(phase_i, phone_i, "N/A", ha="center", va="center", fontsize=6.5, color="#4d5661")
                        if image is None:
                            raise RenderError("local heatmap has no plotted panels")
                        fig.suptitle(
                            f"Phone-local descriptive response · {direction} · {geometry} · {support} · {estimand}",
                            x=0.06, y=0.982, ha="left", fontsize=13, fontweight="bold",
                        )
                        fig.subplots_adjust(left=0.105, right=0.86, top=0.90, bottom=0.095, wspace=0.28, hspace=0.32)
                        cax = fig.add_axes([0.892, 0.205, 0.018, 0.60])
                        cbar = fig.colorbar(image, cax=cax)
                        cbar.set_label("Equal-speaker mean local response (SyncNet C units)")
                        fig.text(
                            0.06, 0.012,
                            f"16 frozen phone labels × 4 phases; four predefined treatments; A_RAW only. "
                            "All planned 80 clips / 40 speakers remain in the denominator. "
                            f"Missing/incomplete cells are retained (N/A); short query_count<40 contexts: {short_contexts}; "
                            f"guard20 endpoints marked INCOMPLETE_SUPPORT: {guard20_incomplete}. "
                            f"Black dots mark cells whose shared-family 99% interval excludes zero. "
                            f"FULL/VALID exact equal estimable pairs across family: {full_valid_equal}. "
                            "An empty short-T guard20 endpoint is not rescued with guard15 or 39 speakers. "
                            "These receptive-field-weighted local summaries are descriptive, not phone-specific causal estimates. "
                            "Point estimates shown; simultaneous 99% family intervals are in the CSV supplement.",
                            fontsize=7.3, color="#505a68",
                        )
                        pdf.savefig(fig, facecolor="white")
                        plt.close(fig)
    return {"path": str(path), "pages": page_count, "heatmaps": page_count * len(LOCAL_TREATMENTS), "cells": len(rows), "global_symmetric_color_limit": vmax}


def _write_output_manifest(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("*")):
        if path.is_symlink():
            raise RenderError("renderer output must not contain symlinks")
        if path.is_file() and path.name != "render_manifest.json":
            inventory.append({
                "path": path.relative_to(output_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": _sha_file(path),
            })
    manifest = {"schema_version": 1, **payload, "output_files": inventory}
    manifest["payload_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
    path = output_dir / "render_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "raw_sha256": _sha_file(path), "payload_sha256": manifest["payload_sha256"], "file_count": len(inventory)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True, help="Root-approved frozen figure input binding JSON")
    parser.add_argument("--binding-raw-sha256", required=True, help="Exact raw-byte SHA-256 of --binding")
    parser.add_argument("--output-dir", type=Path, required=True, help="New, empty output directory; existing paths are rejected")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    source_path = Path(__file__).resolve(strict=True)
    binding_path = args.binding.resolve(strict=True)
    binding_check = _verify_binding(binding_path, args.binding_raw_sha256, source_path)
    binding = binding_check["binding"]
    output_root, analysis_info = _validate_binding_contract(binding, binding_path)
    manifest = _read_json(analysis_info["manifest_path"])
    if not isinstance(manifest, dict):
        raise RenderError("independent output manifest is not an object")
    verified_outputs = _verify_output_tree(output_root, manifest)
    report = _verify_report(output_root, analysis_info["manifest_path"], manifest, analysis_info["expected"], verified_outputs)
    guard15_rows = _guard15_primary_rows(report["global_rows"])

    output_dir = args.output_dir.resolve(strict=False)
    if output_dir.exists():
        raise RenderError(f"output directory already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=False)
    try:
        endpoints = _read_jsonl(_safe_child(output_root, ENDPOINTS_FILE, "family endpoint rows"))
        all_csv_count = _write_csv(output_dir / "family_endpoints_all.csv", endpoints)
        global_csv_count = _write_csv(output_dir / "global_all_metrics.csv", report["global_rows"])
        local_phone_csv_count = _write_csv(output_dir / "local_phone_endpoints.csv", report["local_phone_rows"])
        local_overall_csv_count = _write_csv(output_dir / "local_overall_endpoints.csv", report["local_overall_rows"])
        if (all_csv_count, global_csv_count, local_phone_csv_count, local_overall_csv_count) != (FAMILY_SIZE, GLOBAL_SIZE, LOCAL_PHONE_SIZE, LOCAL_OVERALL_SIZE):
            raise RenderError("CSV supplements do not preserve the full planned family counts")
        global_pdf = _global_forest(
            output_dir / "global_C_fixed_anchor_forest.pdf", report["global_rows"], report["critical"],
            report["short_context_count"], report["support_status_counts"]["guard20"]["INCOMPLETE_SUPPORT"],
            report["full_valid_equal_estimable_pairs"],
        )
        local_pdf = _local_heatmaps(
            output_dir / "local_phone_phase_heatmaps.pdf", report["local_phone_rows"],
            report["short_context_count"], report["support_status_counts"]["guard20"]["INCOMPLETE_SUPPORT"],
            report["full_valid_equal_estimable_pairs"],
        )
        overview = _guard15_overview(
            output_dir / "guard15_C_fixed_anchor_overview.png",
            output_dir / "guard15_C_fixed_anchor_overview.svg",
            guard15_rows,
            report["critical"],
        )
        input_record = {
            "binding_path": str(binding_path), "binding_raw_sha256": binding_check["raw_sha256"],
            "binding_payload_sha256": binding_check["payload_sha256"],
            "analysis_output_root": str(output_root),
            "analysis_manifest_path": str(analysis_info["manifest_path"]),
            "analysis_manifest_raw_sha256": analysis_info["manifest_raw_sha256"],
            "consumer_source_sha256": manifest["consumer_source_sha256"],
            "execution_plan_sha256": manifest["execution_plan_sha256"],
            "physical_manifest_sha256": manifest["physical_manifest_sha256"],
            "design_sha256": manifest["design_sha256"],
            "metric_addendum_sha256": manifest["metric_addendum_sha256"],
            "phone_inventory_sha256": manifest["phone_inventory_sha256"],
            "family_reconstruction_npz_sha256": report["reconstruction_npz_sha256"],
            "family_endpoints_sha256": report["endpoints_sha256"],
            "verified_output_file_count": report["output_file_count"],
        }
        payload = {
            "status": "RENDERED",
            "renderer_source_path": str(source_path),
            "renderer_source_sha256": binding_check["renderer_source_sha256"],
            "input": input_record,
            "analysis": {
                "family_endpoint_count": FAMILY_SIZE,
                "family_status_counts": report["status_counts"],
                "family_support_status": report["family_support_status"],
                "support_status_counts": report["support_status_counts"],
                "shared_family_critical_99_max_t": report["critical"],
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "bootstrap_seed": BOOTSTRAP_SEED,
                "short_context_query_count_below_40": report["short_context_count"],
                "short_clip_count_query_count_below_40": report["short_clip_count"],
                "guard20_empty_context_count_query_count_at_most_40": report["guard20_empty_context_count"],
                "full_valid_planned_endpoint_pairs": report["full_valid_planned_pairs"],
                "full_valid_exact_equal_estimable_pairs": report["full_valid_equal_estimable_pairs"],
                "full_valid_equal_incomplete_pairs": report["full_valid_equal_incomplete_pairs"],
                "full_valid_note": "FULL and VALID are separate planned endpoints; exact equal values are reported and are not counted as independent replications.",
                "local_note": "Local phone-phase results are descriptive receptive-field-weighted summaries; they are not phone-specific causal estimates.",
                "incomplete_note": "INCOMPLETE_SUPPORT endpoints remain in the 11,776-row family and 80-clip/40-speaker denominator; empty short-T guard20 endpoints are not rescued with guard15 or 39 speakers.",
                "support_duplicate_note": "FULL and VALID remain distinct planned support endpoints; exact equal pairs are counted and are not independent replications.",
            },
            "rendered_figures": {
                "global_forest": global_pdf,
                "local_heatmaps": local_pdf,
                "guard15_primary_overview": overview,
            },
            "guard15_overview": {
                "endpoint_count": len(guard15_rows),
                "support": "guard15",
                "metric": "C_fixed_anchor",
                "global_contrast_count": len(CONTRASTS),
                "complete_speaker_count": SPEAKERS,
                "shared_family_size": FAMILY_SIZE,
                "interval_family_independent": False,
                "guard20_incomplete_retained_in_full_outputs": True,
            },
            "csv_rows": {
                "family_endpoints_all": all_csv_count,
                "global_all_metrics": global_csv_count,
                "local_phone": local_phone_csv_count,
                "local_overall": local_overall_csv_count,
            },
        }
        receipt = _write_output_manifest(output_dir, payload)
    except Exception:
        # Leave the failed, non-scientific render attempt intact for diagnosis.
        raise
    print(json.dumps({"status": "RENDERED", "output_dir": str(output_dir), **receipt}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
