"""CPU-only, score-free conditioning diagnostics for the frozen V3 head.

This module is intentionally separate from SPSA, Stage-A scoring, video
rendering, and the fitted objective. It computes source-clock positive and
negative gates from each original Fit WAV, then measures the empirical linear
map from the nine stored dB parameters to V3's smoothed pre-common-mode mask.
Each parameter weights its own one-hot non-reference band; the explicit
reference band has fixed coefficient zero. The Gram matrix is formed before
common-mode removal, radial tanh projection, ISTFT, waveform RMS matching, or
any video/SyncNet operation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence


SCHEMA = "cross_tfg_fit40_head_basis_conditioning_diagnostic_v2"
REQUEST_SCHEMA = "cross_tfg_fit40_head_basis_conditioning_request_v1"
HEAD_RELATIVE = Path("scripts/audio_enhancement/shared_recipient_transition_head_v3.py")
AUDIO_IO_RELATIVE = Path("scripts/audio_enhancement/audio_io.py")
EXPECTED_SAMPLE_COUNT = 40
EXPECTED_SNAPSHOT_STEP = 120
COMPONENTS = ("static", "positive_flux", "negative_flux")


class ConditioningError(RuntimeError):
    """Raised when a source pin or diagnostic invariant fails."""


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def self_hash(value: Mapping[str, Any], field: str) -> str:
    return sha256_bytes(canonical_json_bytes({key: item for key, item in value.items()
                                              if key != field}))


def _finite_vector(values: Any, label: str) -> Any:
    import numpy as np

    vector = np.asarray(values, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0 or not np.isfinite(vector).all():
        raise ConditioningError("%s must be a nonempty finite vector" % label)
    return vector


def population_summary(values: Any) -> dict[str, Optional[float]]:
    """Return population mean/std/CV; CV is undefined for a zero mean."""
    import numpy as np

    vector = _finite_vector(values, "summary input")
    mean = float(vector.mean(dtype=np.float64))
    std = float(vector.std(dtype=np.float64, ddof=0))
    return {"mean": mean, "population_std": std,
            "coefficient_of_variation": (std / mean if mean != 0.0 else None)}


def population_correlation(left: Any, right: Any) -> Optional[float]:
    """Population Pearson correlation, or None for a constant vector."""
    import numpy as np

    x = _finite_vector(left, "correlation left input")
    y = _finite_vector(right, "correlation right input")
    if x.shape != y.shape:
        raise ConditioningError("correlation vectors have different shapes")
    xc = x - x.mean(dtype=np.float64)
    yc = y - y.mean(dtype=np.float64)
    x_energy = float(np.mean(xc * xc, dtype=np.float64))
    y_energy = float(np.mean(yc * yc, dtype=np.float64))
    if x_energy == 0.0 or y_energy == 0.0:
        return None
    covariance = float(np.mean(xc * yc, dtype=np.float64))
    return covariance / math.sqrt(x_energy * y_energy)


def build_preprojection_design(
    band_basis: Any,
    reference_band_index: int,
    positive_gate: Any,
    negative_gate: Any,
    smooth: Callable[[Any], Any],
) -> Any:
    """Build the exact 9-column derivative of V3's smoothed mask.

    Columns follow the frozen row-major parameter layout: three temporal
    components (static, positive_flux, negative_flux), each crossed with the
    three non-reference bands in ascending band index.  A column is the
    corresponding one-hot band multiplied by its source-clock gate and passed
    through the head's linear mask smoother. The fixed reference band's
    coefficient is zero; it is not subtracted from each trainable band's
    column. The common-mode and nonlinear radial projection are intentionally
    omitted.
    """
    import numpy as np

    basis = np.asarray(band_basis, dtype=np.float32)
    pos = _finite_vector(positive_gate, "positive gate").astype(np.float32)
    neg = _finite_vector(negative_gate, "negative gate").astype(np.float32)
    if basis.ndim != 2 or basis.shape[0] != 4 or basis.shape[1] < 1:
        raise ConditioningError("band basis must have shape [4, frequency_bins]")
    if not np.isfinite(basis).all() or not np.all((basis == 0.0) | (basis == 1.0)):
        raise ConditioningError("band basis must be finite and one-hot")
    if not 0 <= reference_band_index < 4:
        raise ConditioningError("reference band index is outside four bands")
    if pos.shape != neg.shape:
        raise ConditioningError("positive and negative gates have different clocks")
    if np.any(pos < 0.0) or np.any(neg < 0.0):
        raise ConditioningError("gates must be nonnegative")
    if not np.all(basis.sum(axis=0) == 1.0):
        raise ConditioningError("each frequency bin must belong to exactly one band")

    free_bands = [index for index in range(4) if index != reference_band_index]
    temporal = (("static", np.ones(pos.shape, dtype=np.float32)),
                ("positive_flux", pos), ("negative_flux", neg))
    columns = []
    labels = []
    for component, gate in temporal:
        for band_index in free_bands:
            parameter_band = basis[band_index]
            plane = parameter_band[:, None] * gate[None, :]
            smoothed = np.asarray(smooth(plane), dtype=np.float32)
            if smoothed.shape != plane.shape or not np.isfinite(smoothed).all():
                raise ConditioningError("smoother changed design shape or produced nonfinite values")
            columns.append(smoothed.reshape(-1))
            labels.append({"parameter_index": len(labels), "component": component,
                           "band_index": band_index, "reference_band_index": reference_band_index})
    design = np.stack(columns, axis=1)
    if design.shape != (basis.shape[1] * pos.size, 9):
        raise ConditioningError("empirical design is not [frequency_bins*frames, 9]")
    return design, labels


def design_gram_summary(gram: Any) -> dict[str, Any]:
    """Summarize the pooled 9x9 Gram and normalized-column geometry."""
    import numpy as np

    matrix = np.asarray(gram, dtype=np.float64)
    if matrix.shape != (9, 9) or not np.isfinite(matrix).all():
        raise ConditioningError("design Gram must be a finite 9x9 matrix")
    symmetric = (matrix + matrix.T) * 0.5
    eigenvalues = np.linalg.eigvalsh(symmetric)
    largest = max(float(eigenvalues[-1]), 0.0)
    tol = max(1.0, largest) * np.finfo(np.float64).eps * 9 * 64
    singular_values = np.sqrt(np.maximum(eigenvalues[::-1], 0.0))
    rank = int(np.count_nonzero(eigenvalues > tol))
    column_norms = np.sqrt(np.maximum(np.diag(symmetric), 0.0))
    denominator = column_norms[:, None] * column_norms[None, :]
    normalized = np.zeros((9, 9), dtype=np.float64)
    valid = denominator > 0.0
    normalized[valid] = symmetric[valid] / denominator[valid]
    normalized_eigenvalues = np.linalg.eigvalsh((normalized + normalized.T) * 0.5)
    normalized_singular_values = np.sqrt(np.maximum(normalized_eigenvalues[::-1], 0.0))
    normalized_rank = int(np.count_nonzero(normalized_eigenvalues > 9 * np.finfo(np.float64).eps * 64))
    positive_values = singular_values[singular_values > math.sqrt(tol)]
    normalized_positive = normalized_singular_values[
        normalized_singular_values > math.sqrt(9 * np.finfo(np.float64).eps * 64)
    ]
    return {
        "gram": symmetric.tolist(),
        "column_norms": column_norms.tolist(),
        "column_correlation_from_normalized_gram": normalized.tolist(),
        "singular_values": singular_values.tolist(),
        "effective_rank": rank,
        "condition_number_nonzero": (float(positive_values[0] / positive_values[-1])
                                      if positive_values.size else None),
        "normalized_column_singular_values": normalized_singular_values.tolist(),
        "normalized_effective_rank": normalized_rank,
        "normalized_condition_number_nonzero": (
            float(normalized_positive[0] / normalized_positive[-1])
            if normalized_positive.size else None),
        "rank_tolerance_on_gram_eigenvalue": float(tol),
    }


def _file_pin(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ConditioningError("symlink file is not allowed: %s" % path)
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ConditioningError("expected a regular file: %s" % path)
    raw = resolved.read_bytes()
    return {"path": str(resolved), "bytes": len(raw), "sha256": sha256_bytes(raw)}


def _read_pinned_json(pin: Mapping[str, Any], label: str,
                      self_field: Optional[str] = None) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(pin, Mapping):
        raise ConditioningError("%s pin must be an object" % label)
    path_text = pin.get("path")
    if not isinstance(path_text, str) or not Path(path_text).is_absolute():
        raise ConditioningError("%s path must be absolute" % label)
    path = Path(path_text)
    actual = _file_pin(path)
    if actual["bytes"] != pin.get("bytes") or actual["sha256"] != pin.get("sha256"):
        raise ConditioningError("%s raw pin mismatch" % label)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ConditioningError("%s is not UTF-8 JSON" % label) from exc
    if not isinstance(value, dict):
        raise ConditioningError("%s root must be an object" % label)
    if self_field is not None:
        declared = value.get(self_field)
        if not isinstance(declared, str) or self_hash(value, self_field) != declared:
            raise ConditioningError("%s canonical self hash mismatch" % label)
        if pin.get("self_sha256") != declared:
            raise ConditioningError("%s supplied self hash mismatch" % label)
    return value, actual


def _load_pinned_module(name: str, path: Path, pin: Mapping[str, Any]) -> Any:
    actual = _file_pin(path)
    if (actual["bytes"], actual["sha256"]) != (pin.get("bytes"), pin.get("sha256")):
        raise ConditioningError("%s source pin mismatch" % name)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ConditioningError("cannot load pinned source %s" % name)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    if Path(str(module.__file__)).resolve(strict=True) != path.resolve(strict=True):
        raise ConditioningError("%s loaded from an unexpected source path" % name)
    return module


def _gate_and_design_for_pcm(head: Any, head_module: Any, samples: Any,
                             torch: Any) -> tuple[dict[str, Any], Any, list[dict[str, Any]]]:
    import numpy as np

    source = torch.as_tensor(np.ascontiguousarray(samples, dtype=np.float32), device="cpu")
    if source.ndim != 1 or source.numel() <= head.n_fft // 2:
        raise ConditioningError("source PCM does not satisfy the head's minimum centered-reflect STFT length")
    if not bool(torch.isfinite(source).all()):
        raise ConditioningError("source PCM has nonfinite samples")
    spectrum = torch.stft(
        source, n_fft=head.n_fft, hop_length=head.hop_length, win_length=head.win_length,
        window=head.stft_window.to(dtype=source.dtype, device="cpu"), center=head.stft_center,
        pad_mode=head.stft_pad_mode, normalized=False, onesided=True, return_complex=True,
    )
    flux = head._source_clock_flux(spectrum)
    positive, positive_threshold, positive_count, positive_support = head._make_gate(
        flux["positive_relative_flux"], flux["valid_transition_mask"], head.positive_flux_quantile)
    negative, negative_threshold, negative_count, negative_support = head._make_gate(
        flux["negative_relative_flux"], flux["valid_transition_mask"], head.negative_flux_quantile)
    if positive_count != negative_count or positive_count != int(flux["valid_transition_mask"].sum().item()):
        raise ConditioningError("head-reported positive/negative support counts disagree")
    if positive_support != negative_support:
        raise ConditioningError("head reports different polarity support presence")

    positive_np = positive.detach().cpu().numpy().astype(np.float32, copy=True)
    negative_np = negative.detach().cpu().numpy().astype(np.float32, copy=True)
    valid_np = flux["valid_transition_mask"].detach().cpu().numpy().astype(bool, copy=True)

    def smooth(plane: Any) -> Any:
        value = torch.as_tensor(np.ascontiguousarray(plane, dtype=np.float32), dtype=torch.float32)
        value = head_module._smooth_mask_axis(
            value, head.mask_smoothing_frequency_bins, "frequency", head.mask_smoothing_pad_mode)
        value = head_module._smooth_mask_axis(
            value, head.mask_smoothing_time_frames, "time", head.mask_smoothing_pad_mode)
        return value.detach().cpu().numpy()

    design, labels = build_preprojection_design(
        head.band_basis.detach().cpu().numpy(), head.reference_band_index,
        positive_np, negative_np, smooth)
    details = {
        "positive_gate": positive_np,
        "negative_gate": negative_np,
        "valid_transition_mask": valid_np,
        "positive_threshold": float(positive_threshold.detach().cpu().item()),
        "negative_threshold": float(negative_threshold.detach().cpu().item()),
        "positive_support": bool(positive_support),
        "negative_support": bool(negative_support),
        "stft_frame_count": int(positive_np.size),
        "valid_transition_count": int(valid_np.sum()),
    }
    return details, design, labels


def _dc_dominance(summary: Mapping[str, Optional[float]]) -> dict[str, Optional[float]]:
    mean = summary["mean"]
    std = summary["population_std"]
    if mean is None or std is None:
        return {"dc_energy_fraction": None, "variance_to_dc_energy": None}
    dc_energy = float(mean) ** 2
    variance = float(std) ** 2
    total = dc_energy + variance
    return {
        "dc_energy_fraction": (dc_energy / total if total > 0.0 else None),
        "variance_to_dc_energy": (variance / dc_energy if dc_energy > 0.0 else None),
    }


def run_analysis(request_path: Path, output_dir: Path) -> dict[str, Any]:
    import numpy as np
    import torch

    # Set both CPU thread pools before importing the pinned model source or
    # constructing the head.  In a fresh process this keeps the diagnostic's
    # execution contract deterministic and prevents torch from initializing
    # its inter-op pool under a host-default setting.
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as exc:
        raise ConditioningError("fresh process could not set one interop thread") from exc

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ConditioningError("CPU diagnostic requires CUDA_VISIBLE_DEVICES to be empty")
    if torch.cuda.is_initialized():
        raise ConditioningError("CUDA was initialized before the CPU-only diagnostic")
    if output_dir.exists() or output_dir.is_symlink():
        raise ConditioningError("output directory must be fresh")
    if output_dir.parent.is_symlink() or not output_dir.parent.is_dir():
        raise ConditioningError("output parent must be an existing real directory")
    request_pin = _file_pin(request_path)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    if (not isinstance(request, dict) or request.get("schema") != REQUEST_SCHEMA
            or request.get("status") != "READY_FOR_REVIEW_DIAGNOSTIC_NOT_RUN"
            or request.get("request_sha256") != self_hash(request, "request_sha256")
            or request.get("expected_fit_wav_count") != EXPECTED_SAMPLE_COUNT):
        raise ConditioningError("request schema/status/self hash is invalid")

    repo_root_text = request.get("project_root")
    if not isinstance(repo_root_text, str) or not Path(repo_root_text).is_absolute():
        raise ConditioningError("project root must be absolute")
    repo_root = Path(repo_root_text).resolve(strict=True)
    manifest, manifest_pin = _read_pinned_json(request["input_bundle_manifest"], "Fit input bundle manifest")
    source_manifest, source_manifest_pin = _read_pinned_json(
        request["fit_source_manifest"], "Fit source manifest")
    snapshot_index, snapshot_index_pin = _read_pinned_json(
        request["snapshot_index"], "snapshot index")
    snapshot, snapshot_pin = _read_pinned_json(request["snapshot_record"], "step120 snapshot")
    mask_receipt, mask_receipt_pin = _read_pinned_json(
        request["mask_gate07_receipt"], "mask-gates07 receipt", self_field="self_sha256")
    if (manifest.get("manifest_sha256") != request["input_bundle_manifest"].get("self_sha256")
            or self_hash(manifest, "manifest_sha256") != manifest.get("manifest_sha256")
            or source_manifest.get("schema") != "cross_tfg_v5_fit40_source_transfer_manifest_adr013_v1"
            or source_manifest.get("status") != "PASS_FIT_ONLY_SOURCES_MATERIALIZED_NO_TREATMENT_EXECUTED"):
        raise ConditioningError("frozen Fit manifests are not the expected source-only inputs")
    if (snapshot_index.get("schema") != "cross_tfg_fit40_snapshot_set_index_v1"
            or snapshot_index.get("status") != "EXTRACTED_UNSELECTED_NATIVE_ENDPOINT_UNBOUND"
            or self_hash(snapshot_index, "index_sha256") != snapshot_index.get("index_sha256")):
        raise ConditioningError("snapshot index schema/status/self hash is invalid")
    if (mask_receipt.get("schema") != "cross_tfg_fit_snapshot_mask_gate_diagnostic_v1"
            or mask_receipt.get("status") != "PASS_SOURCE_PINNED_MASK_AND_GATE_DIAGNOSTIC_ONLY"):
        raise ConditioningError("mask-gates07 receipt is not the expected prior diagnostic")
    if (snapshot.get("schema") != "cross_tfg_fit40_parameter_snapshot_v1"
            or snapshot.get("source_optimizer_step") != EXPECTED_SNAPSHOT_STEP
            or snapshot.get("optimizer_step_is_selected") is not False
            or snapshot.get("selection_status") != "NOT_SELECTED_FIT_NATIVE_SCORING_NOT_RUN"):
        raise ConditioningError("snapshot must remain the unselected Fit source step 120")
    if snapshot.get("head_schema") != "cross_tfg_shared_recipient_transition_head_v3":
        raise ConditioningError("snapshot head schema differs from frozen V3")
    if self_hash(snapshot, "snapshot_sha256") != snapshot.get("snapshot_sha256"):
        raise ConditioningError("snapshot record canonical self hash is invalid")
    index_matches = [entry for entry in snapshot_index.get("snapshots", [])
                     if isinstance(entry, Mapping)
                     and entry.get("source_optimizer_step") == EXPECTED_SNAPSHOT_STEP]
    if len(index_matches) != 1:
        raise ConditioningError("snapshot index must contain exactly one step120 record")
    index_entry = index_matches[0]
    if (index_entry.get("path") != Path(request["snapshot_record"]["path"]).name
            or index_entry.get("bytes") != snapshot_pin["bytes"]
            or index_entry.get("sha256") != snapshot_pin["sha256"]
            or index_entry.get("snapshot_sha256") != snapshot.get("snapshot_sha256")
            or index_entry.get("method_id") != snapshot.get("method_id")):
        raise ConditioningError("step120 snapshot record differs from its frozen index entry")
    constructor = snapshot.get("head_constructor")
    if not isinstance(constructor, Mapping) or sha256_bytes(canonical_json_bytes(constructor)) != snapshot.get("head_constructor_sha256"):
        raise ConditioningError("snapshot head constructor SHA is invalid")
    parameter = snapshot.get("parameter")
    if (not isinstance(parameter, Mapping) or parameter.get("shape") != [3, 3]
            or parameter.get("temporal_components") != list(COMPONENTS)
            or len(parameter.get("values", [])) != 3):
        raise ConditioningError("frozen snapshot parameter shape/order differs from V3")
    theta = [value for row in parameter["values"] for value in row]
    if len(theta) != 9 or not all(math.isfinite(float(value)) for value in theta):
        raise ConditioningError("frozen step120 theta must contain nine finite values")
    if sha256_bytes(np.asarray(theta, dtype="<f4").tobytes()) != parameter.get("little_endian_f32_sha256"):
        raise ConditioningError("step120 theta does not match its exact little-endian float32 pin")

    expected_head_member = next((item for item in manifest["members"]
                                 if item.get("role") == "trained_head_module"), None)
    expected_audio_io = request.get("audio_io_source")
    if not isinstance(expected_head_member, Mapping) or not isinstance(expected_audio_io, Mapping):
        raise ConditioningError("input bundle omits the pinned head or canonical WAV reader")
    if expected_head_member.get("source_repo_relative") != str(HEAD_RELATIVE):
        raise ConditioningError("head source path differs from the exact input bundle member")
    index_member = next((item for item in manifest["members"]
                         if item.get("role") == "snapshot_set_index"), None)
    snapshot_member = next((item for item in manifest["members"]
                            if item.get("role") == "fit_snapshot_record"
                            and item.get("source_repo_relative", "").endswith("snapshot_step120.json")), None)
    if (not isinstance(index_member, Mapping) or not isinstance(snapshot_member, Mapping)
            or index_member.get("bytes") != snapshot_index_pin["bytes"]
            or index_member.get("sha256") != snapshot_index_pin["sha256"]
            or snapshot_member.get("bytes") != snapshot_pin["bytes"]
            or snapshot_member.get("sha256") != snapshot_pin["sha256"]):
        raise ConditioningError("snapshot index/record differ from the exact staged input bundle")
    source_manifest_member = next((item for item in manifest["members"]
                                   if item.get("role") == "fit_source_manifest"), None)
    source_manifest_rel = str(Path(request["fit_source_manifest"]["path"]).relative_to(repo_root))
    if (not isinstance(source_manifest_member, Mapping)
            or source_manifest_member.get("bytes") != source_manifest_pin["bytes"]
            or source_manifest_member.get("sha256") != source_manifest_pin["sha256"]
            or source_manifest_member.get("source_repo_relative") != source_manifest_rel):
        raise ConditioningError("Fit source-manifest pin differs from the bundle record")
    if expected_audio_io.get("path") != str((repo_root / AUDIO_IO_RELATIVE).resolve(strict=True)):
        raise ConditioningError("canonical WAV reader path differs from the pinned project source")
    head_path = repo_root / HEAD_RELATIVE
    audio_io_path = repo_root / AUDIO_IO_RELATIVE
    head_module = _load_pinned_module("fit40_head_basis_conditioning_v3", head_path, {
        "bytes": expected_head_member["bytes"], "sha256": expected_head_member["sha256"]})
    audio_io_module = _load_pinned_module("fit40_head_basis_conditioning_audio_io", audio_io_path,
                                          expected_audio_io)
    head = head_module.SharedRecipientTransitionHeadV3(**dict(constructor))
    with torch.no_grad():
        head.band_contrasts_db.copy_(torch.as_tensor(theta, dtype=torch.float32).reshape(3, 3))
    head.eval()

    wav_members = [item for item in manifest["members"] if item.get("role") == "fit_natural_audio"]
    source_rows = source_manifest.get("rows")
    if len(wav_members) != EXPECTED_SAMPLE_COUNT or not isinstance(source_rows, list) or len(source_rows) != EXPECTED_SAMPLE_COUNT:
        raise ConditioningError("the frozen source inputs must contain exactly 40 Fit WAVs and rows")
    source_by_id = {row.get("sample_id"): row for row in source_rows if isinstance(row, Mapping)}
    if len(source_by_id) != EXPECTED_SAMPLE_COUNT:
        raise ConditioningError("Fit source manifest sample IDs are not unique")

    per_sample = []
    pooled_pos_all = []
    pooled_neg_all = []
    pooled_pos_valid = []
    pooled_neg_valid = []
    mean_pairs = []
    pooled_gram = np.zeros((9, 9), dtype=np.float64)
    pooled_cells = 0
    labels: Optional[list[dict[str, Any]]] = None
    all_row_ids: set[str] = set()
    mask_sample = request.get("mask_gate07_sample_id")
    mask_arrays_path = Path(str(request.get("mask_gate07_arrays_path", "")))
    mask_arrays_actual = _file_pin(mask_arrays_path)
    if (mask_arrays_actual["bytes"] != request.get("mask_gate07_arrays_bytes")
            or mask_arrays_actual["sha256"] != request.get("mask_gate07_arrays_sha256")):
        raise ConditioningError("existing mask-gates07 NPZ raw pin mismatch")
    declared_mask_arrays = mask_receipt.get("diagnostic_arrays")
    if (not isinstance(declared_mask_arrays, Mapping)
            or declared_mask_arrays.get("bytes") != mask_arrays_actual["bytes"]
            or declared_mask_arrays.get("sha256") != mask_arrays_actual["sha256"]):
        raise ConditioningError("mask-gates07 receipt does not bind the supplied arrays")
    baseline_mask_arrays = np.load(mask_arrays_path, allow_pickle=False)
    mask_compare = None

    for member in wav_members:
        relative = member.get("source_repo_relative")
        archive_path = member.get("archive_path")
        sample_id = Path(str(archive_path)).stem
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ConditioningError("Fit WAV path is not safe and repository-relative")
        row = source_by_id.get(sample_id)
        if not isinstance(row, Mapping) or row.get("sample_id") != sample_id:
            raise ConditioningError("Fit WAV has no exact source-manifest row")
        if row.get("source_member_bytes") != member.get("bytes") or row.get("source_member_sha256") != member.get("sha256"):
            raise ConditioningError("bundle and Fit source manifest disagree on raw WAV pins")
        path = repo_root / Path(relative)
        raw_pin = _file_pin(path)
        if raw_pin["bytes"] != member["bytes"] or raw_pin["sha256"] != member["sha256"]:
            raise ConditioningError("original Fit WAV differs from its frozen source pin: %s" % sample_id)
        samples, info = audio_io_module.read_canonical_wav(path, target_rate=16000)
        if (info.frames != row.get("sample_count") or info.sample_rate != row.get("sample_rate_hz")
                or info.source_channels != row.get("channels") or info.source_encoding != row.get("source_encoding")
                or info.pcm_float32_sha256 != row.get("canonical_float32_pcm_sha256")
                or info.file_sha256 != member["sha256"]):
            raise ConditioningError("canonical PCM/clock differs from frozen source row: %s" % sample_id)
        details, design, current_labels = _gate_and_design_for_pcm(head, head_module, samples, torch)
        if labels is None:
            labels = current_labels
        elif labels != current_labels:
            raise ConditioningError("design column order changed across Fit inputs")
        gram = design.astype(np.float64).T @ design.astype(np.float64)
        pooled_gram += gram
        pooled_cells += int(design.shape[0])

        valid = details["valid_transition_mask"]
        pos_all = population_summary(details["positive_gate"])
        neg_all = population_summary(details["negative_gate"])
        pos_valid = population_summary(details["positive_gate"][valid]) if bool(valid.any()) else None
        neg_valid = population_summary(details["negative_gate"][valid]) if bool(valid.any()) else None
        pos_valid_values = details["positive_gate"][valid]
        neg_valid_values = details["negative_gate"][valid]
        if bool(valid.any()):
            pooled_pos_valid.extend(pos_valid_values.astype(np.float64).tolist())
            pooled_neg_valid.extend(neg_valid_values.astype(np.float64).tolist())
        pooled_pos_all.extend(details["positive_gate"].astype(np.float64).tolist())
        pooled_neg_all.extend(details["negative_gate"].astype(np.float64).tolist())
        if pos_valid is not None and neg_valid is not None:
            mean_pairs.append([pos_valid["mean"], neg_valid["mean"]])
        per_sample.append({
            "sample_id": sample_id,
            "speaker_id": row.get("speaker_id"),
            "row_id": row.get("row_id"),
            "source_wav": raw_pin,
            "source_pcm_float32_sha256": info.pcm_float32_sha256,
            "sample_count": int(info.frames),
            "sample_rate_hz": int(info.sample_rate),
            "stft_frame_count": details["stft_frame_count"],
            "valid_transition_count": details["valid_transition_count"],
            "gate_thresholds": {"positive": details["positive_threshold"],
                                "negative": details["negative_threshold"]},
            "positive_gate_all_stft_frames": {**pos_all, **_dc_dominance(pos_all)},
            "negative_gate_all_stft_frames": {**neg_all, **_dc_dominance(neg_all)},
            "positive_gate_valid_transitions": ({**pos_valid, **_dc_dominance(pos_valid)} if pos_valid else None),
            "negative_gate_valid_transitions": ({**neg_valid, **_dc_dominance(neg_valid)} if neg_valid else None),
            "positive_negative_gate_correlation_all_frames": population_correlation(
                details["positive_gate"], details["negative_gate"]),
            "positive_negative_gate_correlation_valid_transitions": (
                population_correlation(pos_valid_values, neg_valid_values) if bool(valid.any()) else None),
            "design_cell_count_frequency_times_stft_frames": int(design.shape[0]),
            "design_gram": design_gram_summary(gram),
        })
        if not isinstance(row.get("row_id"), str) or row["row_id"] in all_row_ids:
            raise ConditioningError("Fit row ID is absent or duplicated")
        all_row_ids.add(row["row_id"])

        if sample_id == mask_sample:
            mask_source = mask_receipt.get("source_audio", {})
            mask_source_file = mask_source.get("file", {}) if isinstance(mask_source, Mapping) else {}
            mask_sample_pcm = mask_source.get("pcm_float32_sha256") if isinstance(mask_source, Mapping) else None
            if mask_sample_pcm != info.pcm_float32_sha256:
                raise ConditioningError("mask-gates07 sample PCM differs from the current Fit source")
            if (mask_source_file.get("bytes") != raw_pin["bytes"]
                    or mask_source_file.get("sha256") != raw_pin["sha256"]
                    or mask_source.get("sample_count") != info.frames):
                raise ConditioningError("mask-gates07 source WAV file/clock differs from the Fit input")
            for key, local in (("positive_gate", details["positive_gate"]),
                               ("negative_gate", details["negative_gate"]),
                               ("valid_transition_mask", details["valid_transition_mask"])):
                if key not in baseline_mask_arrays or not np.array_equal(baseline_mask_arrays[key], local):
                    raise ConditioningError("mask-gates07 %s differs for shared source PCM" % key)
            mask_sample = {"sample_id": sample_id,
                           "source_pcm_float32_sha256": info.pcm_float32_sha256,
                           "gate_arrays_exactly_match_mask_gates07": True,
                           "mask07_method_id": mask_receipt.get("method", {}).get("method_id"),
                           "mask07_theta_sha256": mask_receipt.get("method", {}).get("theta_sha256"),
                           "comparison_scope": "source-derived gate arrays; theta does not enter source-clock gate construction"}

    if len(per_sample) != EXPECTED_SAMPLE_COUNT or len(all_row_ids) != EXPECTED_SAMPLE_COUNT:
        raise ConditioningError("not all 40 planned Fit inputs were retained")
    if mask_sample is not None and not isinstance(mask_sample, Mapping):
        raise ConditioningError("mask-gates07 shared sample was not found among the 40 Fit inputs")
    if labels is None:
        raise ConditioningError("no design columns were built")

    pooled_pair = np.column_stack((np.asarray(pooled_pos_valid), np.asarray(pooled_neg_valid)))
    pooled_all_pair = np.column_stack((np.asarray(pooled_pos_all), np.asarray(pooled_neg_all)))
    pooled_summary = {
        "all_stft_frames": {
            "positive": {**population_summary(pooled_all_pair[:, 0]),
                         **_dc_dominance(population_summary(pooled_all_pair[:, 0]))},
            "negative": {**population_summary(pooled_all_pair[:, 1]),
                         **_dc_dominance(population_summary(pooled_all_pair[:, 1]))},
            "positive_negative_correlation": population_correlation(pooled_all_pair[:, 0], pooled_all_pair[:, 1]),
        },
        "valid_transitions": {
            "positive": {**population_summary(pooled_pair[:, 0]),
                         **_dc_dominance(population_summary(pooled_pair[:, 0]))},
            "negative": {**population_summary(pooled_pair[:, 1]),
                         **_dc_dominance(population_summary(pooled_pair[:, 1]))},
            "positive_negative_correlation": population_correlation(pooled_pair[:, 0], pooled_pair[:, 1]),
        },
        "across_sample_valid_gate_means_correlation": population_correlation(
            np.asarray(mean_pairs, dtype=np.float64)[:, 0], np.asarray(mean_pairs, dtype=np.float64)[:, 1]),
    }
    packet = {
        "schema": SCHEMA,
        "status": "PASS_FIT_SOURCE_GATE_AND_LINEAR_DESIGN_DIAGNOSTIC_ONLY",
        "scope": "40 original Fit WAVs; frozen V3 source-clock gates and smoothed linear mask design only; no waveform application, audio/video output, TFG scoring, selector, SPSA controller, or model objective update",
        "inputs": {
            "request": {**request_pin, "request_sha256": request["request_sha256"]},
            "input_bundle_manifest": {**manifest_pin, "self_sha256": manifest.get("manifest_sha256")},
            "fit_source_manifest": source_manifest_pin,
            "snapshot_index": {**snapshot_index_pin, "self_sha256": snapshot_index.get("index_sha256")},
            "snapshot_record": {**snapshot_pin, "self_sha256": snapshot.get("snapshot_sha256")},
            "head_source": _file_pin(head_path),
            "audio_io_source": _file_pin(audio_io_path),
            "mask_gates07_receipt": {**mask_receipt_pin, "self_sha256": mask_receipt["self_sha256"]},
            "mask_gates07_arrays": mask_arrays_actual,
        },
        "snapshot": {
            "method_id": snapshot.get("method_id"),
            "source_optimizer_step": snapshot.get("source_optimizer_step"),
            "selection_status": snapshot.get("selection_status"),
            "optimizer_step_is_selected": snapshot.get("optimizer_step_is_selected"),
            "head_constructor_sha256": snapshot.get("head_constructor_sha256"),
            "head_schema": snapshot.get("head_schema"),
            "parameter_shape": parameter.get("shape"),
            "parameter_order": {"temporal_components": list(COMPONENTS),
                                "non_reference_bands": parameter.get("non_reference_band_order"),
                                "reference_band_index": constructor.get("reference_band_index")},
            "parameter_values_little_endian_f32_sha256": parameter.get("little_endian_f32_sha256"),
        },
        "runtime": {"python_version": platform.python_version(), "torch_version": str(torch.__version__),
                    "numpy_version": str(np.__version__), "device": "cpu",
                    "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
                    "cuda_visible_devices": "", "cuda_initialized": bool(torch.cuda.is_initialized()),
                    "optimizer_or_selector_called": False, "waveform_output_written": False,
                    "video_or_syncnet_called": False},
        "design_definition": {
            "row_unit": "one (frequency_bin, STFT_frame) cell, concatenated across all 40 WAVs",
            "column_order": labels,
            "matrix_definition": "d(smoothed raw log-magnitude mask dB)/d(theta[9]); each non-reference parameter multiplies its own one-hot band basis and source-clock gate, followed by exact V3 linear frequency/time smoothing",
            "included": ["four-band one-hot basis with fixed zero-coefficient reference band", "actual positive/negative V3 gates", "exact configured frequency and time mask smoothing"],
            "reference_semantics": "fixed reference weight is zero; trainable non-reference columns are one-hot band indicators and do not subtract the reference-band indicator",
            "excluded": ["frequency common-mode policy", "global radial tanh mask projection", "ISTFT", "RMS normalization", "peak policy", "audio/video scoring"],
            "objective_or_basis_changed": False,
            "parameter_count": 9,
        },
        "fit_denominator": {"planned": EXPECTED_SAMPLE_COUNT, "analyzed": len(per_sample),
                            "missing": 0, "failed": 0,
                            "sample_ids": [row["sample_id"] for row in per_sample]},
        "pooled_gate_statistics": pooled_summary,
        "pooled_design": {"frequency_time_cells": pooled_cells,
                          "conditioning": design_gram_summary(pooled_gram)},
        "mask_gates07_relation": mask_sample,
        "per_sample": per_sample,
        "claim_limits": {"fit_only_diagnostic": True, "no_dev_or_confirmation_data": True,
                         "no_new_tfg_score": True, "no_method_selection_or_objective_update": True,
                         "not_a_spsa_result": True},
    }
    packet["receipt_sha256"] = self_hash(packet, "receipt_sha256")
    output_dir.mkdir(mode=0o700)
    receipt_path = output_dir / "fit40_head_basis_conditioning_receipt_attempt01.json"
    raw = (json.dumps(packet, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
    with receipt_path.open("xb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())
    actual = _file_pin(receipt_path)
    return {"receipt_path": actual["path"], "receipt_bytes": actual["bytes"],
            "receipt_sha256": actual["sha256"], "receipt_self_sha256": packet["receipt_sha256"]}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        result = run_analysis(args.request, args.output_dir)
    except Exception as exc:
        print(json.dumps({"status": "FAILED_CLOSED", "error_type": type(exc).__name__,
                          "error": str(exc)}, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps({"status": "PASS_FIT_SOURCE_GATE_AND_LINEAR_DESIGN_DIAGNOSTIC_ONLY", **result},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
