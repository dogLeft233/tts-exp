"""Source-only Fit40 scorer for frozen-theta post-projection controls.

Only the standard library is imported at module load. The caller supplies the
already qualified head, native Wav2Lip/SyncNet runtime, Fit40 tensors, and
actual lineage pins. This helper creates no optimizer, writes no file, changes
no theta, and encodes no video. It retains every 31-lag by 27-query matrix for
80 same-theta references and 240 controls.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

SCHEMA = "cross_tfg_fit40_post_projection_fullquery_score_diagnostic_v1"
CONTROL_SCHEMA = "cross_tfg_fit40_post_projection_control_diagnostic_v2"
GATE_SCHEMA = "cross_tfg_fit40_gate_dc_same_theta_score_v1"
FIT_ROWS = 40
BRANCHES = ("dynamic", "gate_dc")
CONTROLS = ("post_proj_mean", "post_proj_roll_plus64", "post_proj_roll_minus64")
QUERY_COUNT = 27
LAG_COUNT = 31
LAG_RADIUS = 15
VIDEO_FRAMES = 32
SAMPLE_RATE = 16_000
VIDEO_FPS_NUM = 25
VIDEO_FPS_DEN = 1
D0_HOP = 128
D0_WIN = 512
W2L_HOP = 200
W2L_N_FFT = 800
W2L_MEL_WIDTH = 16
SAME_HOST_SCALAR_ATOL = 1e-5  # D0 runner's existing zero-head curve/visual tolerance
LINEAGE_ROLES = (
    "completion", "checkpoint", "training_record", "runtime_source_pins",
    "head_config", "strict_backward_qualification", "fit40_gate_dc_control",
    "native_geometry_manifest", "post_projection_controls",
)
SOURCE_ROLES = (
    "d0_runner", "d0_head", "native_adapter", "v4_runtime", "v2_runtime",
    "wav2lip_frontend", "post_projection_controls", "score_adapter",
)


class ScoreDiagnosticError(RuntimeError):
    """The actual Fit40, source-clock, or native-score contract is incomplete."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _require_sha(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ScoreDiagnosticError(f"{label} must be a lowercase full SHA-256")
    return value


def _pin(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "bytes", "sha256"}:
        raise ScoreDiagnosticError(f"{label} must contain exactly path/bytes/sha256")
    if not isinstance(value["path"], str) or not value["path"] or type(value["bytes"]) is not int or value["bytes"] <= 0:
        raise ScoreDiagnosticError(f"{label} path or byte count is invalid")
    return {"path": value["path"], "bytes": value["bytes"], "sha256": _require_sha(value["sha256"], label + ".sha256")}


def _tensor_bytes(value: Any, torch: Any) -> bytes:
    if value is None or not hasattr(value, "detach"):
        raise ScoreDiagnosticError("expected a tensor in the native score path")
    array = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    return array.astype("<f4", copy=False).tobytes(order="C")


def _tensor_sha(value: Any, torch: Any) -> str:
    return _sha(_tensor_bytes(value, torch))


def _number(value: Any, label: str) -> float:
    if hasattr(value, "detach"):
        value = value.detach().item()
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ScoreDiagnosticError(f"{label} must be a finite numeric scalar")
    return float(value)


def _parameter_payload(head: Any, torch: Any) -> dict[str, Any]:
    values = head.band_contrasts_db.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().astype("<f4", copy=False)
    if list(values.shape) != [3, 3]:
        raise ScoreDiagnosticError("frozen D0 parameter is no longer exactly [3,3]")
    return {
        "name": "band_contrasts_db", "shape": [3, 3], "dtype": "float32",
        "temporal_components": ["static", "positive_flux", "negative_flux"],
        "non_reference_band_order": [0, 1, 2], "values": values.tolist(),
    }


def _parameter_sha(head: Any, torch: Any) -> str:
    return _sha(_canonical(_parameter_payload(head, torch)))


def _fit_identity(row: Mapping[str, Any]) -> tuple[str, str, int]:
    natural = row.get("natural_pcm")
    if not isinstance(natural, Mapping):
        raise ScoreDiagnosticError("Fit row has no actual natural_pcm identity")
    pcm_sha = _require_sha(natural.get("pcm_sha256"), "natural PCM hash")
    file_sha = _require_sha(natural.get("file_sha256"), "natural WAV file hash")
    count = natural.get("sample_count")
    if type(count) is not int or count <= 0:
        raise ScoreDiagnosticError("natural PCM sample_count is invalid")
    return pcm_sha, file_sha, count


def _lineage_maps(lineage: Mapping[str, Any], sources: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(lineage, Mapping) or not isinstance(sources, Mapping):
        raise ScoreDiagnosticError("actual lineage and source pins are required")
    return (
        {role: _pin(lineage.get(role), f"lineage.{role}") for role in LINEAGE_ROLES},
        {role: _pin(sources.get(role), f"sources.{role}") for role in SOURCE_ROLES},
    )


def _fit_rows(rows: Sequence[Mapping[str, Any]], expected_ids: Sequence[str]) -> dict[str, Mapping[str, Any]]:
    if len(rows) != FIT_ROWS or len(expected_ids) != FIT_ROWS:
        raise ScoreDiagnosticError("the diagnostic denominator is the frozen 40-row Fit cohort")
    actual = [str(row.get("row_id")) for row in rows]
    expected = [str(value) for value in expected_ids]
    if actual != expected or len(set(actual)) != FIT_ROWS:
        raise ScoreDiagnosticError("Fit row order or identity differs from the actual frozen roster")
    for row in rows:
        _fit_identity(row)
        if not isinstance(row.get("sample_id"), str) or not isinstance(row.get("speaker_id"), str):
            raise ScoreDiagnosticError("Fit row lacks its sample/speaker identity")
    return {str(row["row_id"]): row for row in rows}


def _gate_rows(rows: Sequence[Mapping[str, Any]], row_ids: Sequence[str], theta_sha: str) -> dict[tuple[str, str], Mapping[str, Any]]:
    expected = [(row_id, mode) for row_id in row_ids for mode in BRANCHES]
    indexed: dict[tuple[str, str], Mapping[str, Any]] = {}
    observed = []
    for row in rows:
        if not isinstance(row, Mapping) or row.get("schema") != GATE_SCHEMA:
            raise ScoreDiagnosticError("same-theta gate ledger has an unexpected producer schema")
        row_id, mode = str(row.get("row_id")), row.get("gate_mode")
        key = (row_id, str(mode))
        _require_sha(row.get("source_pcm_float32_sha256"), "GateDC natural PCM hash")
        if key in indexed or mode not in BRANCHES:
            raise ScoreDiagnosticError("same-theta gate ledger duplicates or mislabels a branch")
        if (
            row.get("same_frozen_parameter_sha256") != theta_sha
            or row.get("gate_dc_control_is_training_objective") is not False
            or type(row.get("source_sample_count")) is not int
            or not isinstance(row.get("candidate_waveform_float32_sha256"), str)
            or len(row["candidate_waveform_float32_sha256"]) != 64
            or not isinstance(row.get("score_metrics"), Mapping)
        ):
            raise ScoreDiagnosticError("same-theta gate ledger lacks source/theta/scope metrics")
        indexed[key] = row
        observed.append(key)
    if len(rows) != 80 or observed != expected:
        raise ScoreDiagnosticError("same-theta gate ledger is not the ordered 40 x 2 denominator")
    return indexed


def _control_rows(result: Mapping[str, Any], row_ids: Sequence[str], theta_sha: str, torch: Any) -> tuple[dict[tuple[str, str, str], Mapping[str, Any]], Mapping[str, Any]]:
    if (
        result.get("schema") != CONTROL_SCHEMA
        or result.get("frozen_parameter_sha256") != theta_sha
        or result.get("row_count") != 40 or result.get("record_count") != 240 or result.get("output_count") != 240
        or result.get("base_branches") != list(BRANCHES) or result.get("controls_per_branch") != list(CONTROLS)
        or result.get("no_reprojection") is not True or result.get("no_gradients") is not True
        or result.get("no_optimizer") is not True or result.get("no_render_or_score") is not True
        or result.get("score_query_filtering_for_seams") is not False
        or result.get("full_fit40_source_rows_preserved") is not True
    ):
        raise ScoreDiagnosticError("post-projection helper result is not the exact no-score 240-control artifact")
    records, outputs = result.get("records"), result.get("outputs_float32_tensors")
    if not isinstance(records, Sequence) or len(records) != 240 or not isinstance(outputs, Mapping) or len(outputs) != 240:
        raise ScoreDiagnosticError("post-projection artifact is missing control records or full-length waveforms")
    expected = [(row_id, mode, name) for row_id in row_ids for mode in BRANCHES for name in CONTROLS]
    indexed, observed = {}, []
    for record in records:
        if not isinstance(record, Mapping) or record.get("schema") != CONTROL_SCHEMA:
            raise ScoreDiagnosticError("post-projection control row has an unexpected schema")
        key = (str(record.get("row_id")), str(record.get("base_gate_mode")), str(record.get("control")))
        tensor_key = "::".join(key)
        if key in indexed or tensor_key not in outputs:
            raise ScoreDiagnosticError("post-projection control key is duplicated or has no output PCM")
        pcm, mask, stft = record.get("pcm"), record.get("post_projection_mask"), record.get("stft")
        if (
            record.get("same_frozen_parameter_sha256") != theta_sha
            or record.get("base_projection_reconstruction_exact") is not True
            or record.get("optimizer_created_or_stepped") is not False
            or record.get("scoring_or_rendering_performed") is not False
            or record.get("score_queries_filtered_for_seam") is not False
            or not isinstance(pcm, Mapping) or pcm.get("output_exact_full_source_length") is not True
            or not isinstance(mask, Mapping) or mask.get("frequency_bin_count") != 257
            or mask.get("reprojection_applied") is not False
            or not isinstance(stft, Mapping) or not isinstance(stft.get("phase_semantics"), Mapping)
            or int(outputs[tensor_key].numel()) != int(record.get("source_sample_count", -1))
            or _tensor_sha(outputs[tensor_key], torch) != pcm.get("output_pcm_float32_sha256")
        ):
            raise ScoreDiagnosticError(f"post-projection source/theta/PCM/phase contract differs: {tensor_key}")
        indexed[key] = record
        observed.append(key)
    if observed != expected:
        raise ScoreDiagnosticError("post-projection control rows are not ordered Fit40 x 2 x 3")
    return indexed, outputs


def _make_control_mask(mask: Any, name: str, torch: Any) -> Any:
    if name == "post_proj_mean":
        mean = mask.to(dtype=torch.float64).mean(dim=1, keepdim=True).to(dtype=mask.dtype)
        return mean.expand_as(mask)
    if name == "post_proj_roll_plus64":
        return torch.roll(mask, shifts=64, dims=1)
    if name == "post_proj_roll_minus64":
        return torch.roll(mask, shifts=-64, dims=1)
    raise ScoreDiagnosticError(f"unknown predeclared post-projection control: {name}")


def _chunk_support(mel_start: int, sample_count: int) -> list[int]:
    if type(mel_start) is not int or mel_start < 0:
        raise ScoreDiagnosticError("native context has an invalid Wav2Lip mel start")
    # Each mel value uses a centered 800-sample STFT frame; pre-emphasis adds
    # one sample of left support. Clip only the declared window padding edges.
    lo = max(0, mel_start * W2L_HOP - W2L_N_FFT // 2 - 1)
    last_center = (mel_start + W2L_MEL_WIDTH - 1) * W2L_HOP
    hi = min(sample_count, last_center + W2L_N_FFT // 2)
    if lo >= hi:
        raise ScoreDiagnosticError("Wav2Lip mel chunk has empty full-source support")
    return [lo, hi]


def _query_supports(context: Any, sample_count: int) -> list[dict[str, Any]]:
    context.validate()
    starts = tuple(int(value) for value in context.mel_starts)
    frames = tuple(int(value) for value in context.source_video_frame_indices)
    if (
        context.frame_count != VIDEO_FRAMES or context.query_count != QUERY_COUNT
        or len(starts) != VIDEO_FRAMES or len(frames) != VIDEO_FRAMES
        or int(context.expected_sample_count) != sample_count
        or frames != tuple(range(int(context.query_start_frame), int(context.query_start_frame) + VIDEO_FRAMES))
    ):
        raise ScoreDiagnosticError("native context differs from the fixed 32-frame/27-query original clock")
    records = []
    for query in range(QUERY_COUNT):
        q_starts, q_frames = starts[query:query + 5], frames[query:query + 5]
        if len(q_starts) != 5 or len(q_frames) != 5:
            raise ScoreDiagnosticError("query must use exactly five native video frames")
        intervals = [_chunk_support(start, sample_count) for start in q_starts]
        records.append({
            "query_index": query,
            "query_center_source_clock_frame": int(context.query_start_frame) + query,
            "five_video_source_clock_frames": list(q_frames),
            "five_wav2lip_mel_starts": list(q_starts),
            "per_frame_mel_sample_support_half_open": intervals,
            "union_mel_sample_support_half_open": [min(x[0] for x in intervals), max(x[1] for x in intervals)],
            "support_basis": "official 16-bin chunks; centered 800-sample Wav2Lip STFT; one-sample pre-emphasis left extension; clipped to full source length",
        })
    return records


def _frame_range_samples(frame_range: Any, frame_count: int, sample_count: int) -> list[int] | None:
    if not isinstance(frame_range, (tuple, list)) or len(frame_range) != 2:
        return None
    first, stop = int(frame_range[0]), int(frame_range[1])
    if not 0 <= first < stop <= frame_count:
        raise ScoreDiagnosticError("roll mask frame interval is outside the source-clock STFT")
    lo = max(0, first * D0_HOP - D0_WIN // 2)
    hi = min(sample_count, (stop - 1) * D0_HOP + D0_WIN // 2)
    return [lo, hi]


def _overlap(left: Sequence[int], right: Sequence[int]) -> int:
    return max(0, min(int(left[1]), int(right[1])) - max(int(left[0]), int(right[0])))


def _seam_summary(record: Mapping[str, Any], queries: Sequence[Mapping[str, Any]], stft_frames: int, sample_count: int) -> dict[str, Any]:
    roll = record.get("roll_contract")
    if not isinstance(roll, Mapping) or roll.get("is_circular_roll") is not True:
        return {"applicable": False, "query_indices_near_seam": [], "query_rows_removed": 0}
    near = roll.get("join_neighborhood_frame_range_half_open_approximate")
    dest = roll.get("wrap_destination_frame_range_half_open")
    donor = roll.get("wrap_source_frame_range_half_open")
    seam_samples = _frame_range_samples(near, stft_frames, sample_count)
    dest_samples = _frame_range_samples(dest, stft_frames, sample_count)
    donor_samples = _frame_range_samples(donor, stft_frames, sample_count)
    overlaps = [_overlap(q["union_mel_sample_support_half_open"], seam_samples) for q in queries]
    dest_overlaps = [_overlap(q["union_mel_sample_support_half_open"], dest_samples) for q in queries]
    return {
        "applicable": True,
        "signed_roll_frames": int(roll["signed_frames"]),
        "signed_roll_milliseconds": int(roll["signed_milliseconds"]),
        "join_boundary_before_output_frame": int(roll["join_boundary_before_output_frame_index"]),
        "approx_join_neighborhood_mask_frames_half_open": list(near),
        "join_neighborhood_after_512_sample_stft_window_propagation_half_open": seam_samples,
        "wrap_destination_mask_frames_half_open": list(dest),
        "wrap_destination_after_stft_window_propagation_half_open": dest_samples,
        "wrap_source_mask_donor_frames_half_open": list(donor),
        "wrap_source_mask_donor_interval_not_audio_remapping": donor_samples,
        "query_indices_with_mel_support_intersecting_join_neighborhood": [i for i, amount in enumerate(overlaps) if amount > 0],
        "query_overlap_samples_with_join_neighborhood": overlaps,
        "query_overlap_samples_with_wrap_destination": dest_overlaps,
        "all_27_queries_scored": True,
        "query_rows_removed": 0,
        "interpretation_limit": "secondary support flag only; the mask roll changes values outside the seam and residual-RMS/baseline correction is global",
    }


def _distance_matrix(score: Mapping[str, Any], context: Any, adapter: Any, torch: Any) -> tuple[Any, Any, Any, Any]:
    visual = score.get("visual_embeddings_float32")
    centers = score.get("audio_center_embeddings_float32")
    curve = score.get("curve_float32")
    if visual is None or centers is None or curve is None or tuple(visual.shape) != (27, 1024) or tuple(centers.shape) != (27, 1024) or tuple(curve.shape) != (31,):
        raise ScoreDiagnosticError("native scorer must return 27 visual/audio rows and one 31-lag curve")
    if not torch.isfinite(visual).all().item() or not torch.isfinite(centers).all().item() or not torch.isfinite(curve).all().item():
        raise ScoreDiagnosticError("native score output contains non-finite values")
    audio = context.audio_embeddings
    if audio.device != visual.device:
        audio = audio.to(device=visual.device)
    first = int(context.query_start_frame)
    if not torch.equal(audio[first:first + 27], centers):
        raise ScoreDiagnosticError("audio centers differ from the unchanged natural-audio embedding sequence")
    per_query = []
    for query in range(27):
        center = first + query
        candidates = audio[center - 15:center + 16]
        if tuple(candidates.shape) != (31, 1024):
            raise ScoreDiagnosticError("frozen full-audio window lacks one of the 31 unpadded lags")
        per_query.append(adapter.F.pairwise_distance(visual[query:query + 1].expand(31, -1), candidates))
    matrix = torch.stack(per_query, dim=1)
    if tuple(matrix.shape) != (31, 27) or not torch.isfinite(matrix).all().item() or not torch.equal(matrix.mean(dim=1), curve):
        raise ScoreDiagnosticError("full 31x27 matrix does not exactly reproduce the frozen lag curve")
    return matrix, visual, centers, curve


def _metrics(score: Mapping[str, Any], v4: Any, functional: Any, torch: Any) -> tuple[dict[str, float], Any, Any]:
    curve = score["curve_float32"]
    visual = score["visual_embeddings_float32"]
    centers = score["audio_center_embeddings_float32"]
    metric_tensors = v4._curve_metrics(torch, curve)
    norms, _, cosine = v4._embedding_metrics(torch, functional, visual, centers)
    values = {name: _number(value, name) for name, value in metric_tensors.items()}
    values["mean_zero_lag_unit_cosine"] = _number(cosine.mean(), "mean zero-lag cosine")
    values["mean_visual_embedding_l2"] = _number(norms.mean(), "mean visual embedding norm")
    return values, cosine, norms


def _compare_reference(actual: Mapping[str, float], gate_row: Mapping[str, Any]) -> dict[str, Any]:
    expected = gate_row.get("score_metrics")
    if not isinstance(expected, Mapping) or any(key not in expected for key in actual):
        raise ScoreDiagnosticError("original GateDC ledger omits a scalar metric used by the same-host comparison")
    deltas = {key: abs(actual[key] - _number(expected[key], "reference " + key)) for key in actual}
    return {
        "absolute_tolerance": SAME_HOST_SCALAR_ATOL,
        "relative_tolerance": 0.0,
        "tolerance_source": "D0 runner's existing curve31 and visual embedding zero-head same-host gate",
        "per_metric_absolute_delta": deltas,
        "max_absolute_delta": max(deltas.values(), default=0.0),
        "within_existing_tolerance": all(delta <= SAME_HOST_SCALAR_ATOL for delta in deltas.values()),
    }


def score_fit40_post_projection_controls(
    *, rows: Sequence[Mapping[str, Any]], expected_row_ids: Sequence[str],
    sources: Mapping[str, Any], baseline_waveforms: Mapping[str, Any], head: Any,
    expected_parameter_sha256: str, gate_ledger_rows: Sequence[Mapping[str, Any]],
    post_projection_result: Mapping[str, Any], contexts: Mapping[str, Any],
    native_geometries: Mapping[str, Any], adapter: Any, v4_runtime: Any,
    v3_v4_adapter: Any, torch: Any, functional: Any,
    lineage_pins: Mapping[str, Any], source_pins: Mapping[str, Any],
    runtime_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Return JSON-safe receipt/records and in-memory full-query score arrays.

    The 80 reference waveforms are freshly replayed at the same frozen theta
    and must hash-match the original GateDC ledger. The 240 projected-control
    waveforms are consumed exactly from the post-projection helper. All rows
    use the original natural-audio SyncNet embedding cache and unchanged
    source-clock query/context. Per-query seam overlap is secondary metadata;
    it never removes any of the 27 primary queries.
    """
    if not isinstance(runtime_identity, Mapping):
        raise ScoreDiagnosticError("actual runtime identity is required")
    lineage, pinned_sources = _lineage_maps(lineage_pins, source_pins)
    row_map = _fit_rows(rows, expected_row_ids)
    row_ids = [str(value) for value in expected_row_ids]
    theta_sha = _parameter_sha(head, torch)
    if _require_sha(expected_parameter_sha256, "checkpoint theta SHA") != theta_sha:
        raise ScoreDiagnosticError("loaded head theta differs from the completed checkpoint")
    gate_map = _gate_rows(gate_ledger_rows, row_ids, theta_sha)
    control_map, control_waveforms = _control_rows(post_projection_result, row_ids, theta_sha, torch)
    for label, mapping in (("source", sources), ("baseline", baseline_waveforms), ("context", contexts), ("geometry", native_geometries)):
        if not isinstance(mapping, Mapping) or set(map(str, mapping)) != set(row_ids):
            raise ScoreDiagnosticError(f"{label} map differs from the exact Fit40 roster")
    if getattr(adapter, "torch", None) is not torch:
        raise ScoreDiagnosticError("native adapter and caller must share the already-qualified torch runtime")

    records: list[dict[str, Any]] = []
    matrices, curves, cosines, norms = [], [], [], []
    reference_index: dict[tuple[str, str], int] = {}
    reference_ok: list[bool] = []
    reference_failures: list[dict[str, Any]] = []
    control_deltas, seam_masks = [], []
    control_mask_cache: dict[tuple[str, str, str], Any] = {}
    theta_before = _parameter_sha(head, torch)

    def score_one(row_id: str, mode: str, waveform: Any, *, role: str, control_name: str | None, gate_row: Mapping[str, Any], control_record: Mapping[str, Any] | None) -> int:
        row = row_map[row_id]
        source_sha, source_file_sha, sample_count = _fit_identity(row)
        source, baseline = sources[row_id], baseline_waveforms[row_id]
        if (
            source.dtype != torch.float32 or baseline.dtype != torch.float32 or waveform.dtype != torch.float32
            or source.ndim != 1 or baseline.shape != source.shape or waveform.shape != source.shape
            or int(source.numel()) != sample_count or int(waveform.numel()) != sample_count
            or not torch.isfinite(source).all().item() or not torch.isfinite(baseline).all().item()
            or not torch.isfinite(waveform).all().item() or _tensor_sha(source, torch) != source_sha
        ):
            raise ScoreDiagnosticError(f"{row_id}/{mode}/{control_name}: original full-length float32 PCM contract differs")
        context = contexts[row_id]
        query_support = _query_supports(context, sample_count)
        score = v3_v4_adapter.score_with_v4_native_path(
            v4_runtime=v4_runtime, v4_adapter=adapter, context=context,
            geometry_record=native_geometries[row_id], enhanced_audio=waveform,
        )
        matrix, visual, audio_center, curve = _distance_matrix(score, context, adapter, torch)
        score_metrics, cosine, visual_norm = _metrics(score, v4_runtime, functional, torch)
        if tuple(score.get("rendered_crop_shape", ())) != (32, 3, 224, 224):
            raise ScoreDiagnosticError("score path did not use the frozen 32 native SyncNet crops")
        scored_pcm_sha = _tensor_sha(waveform, torch)
        ref_check = None
        if role == "same_theta_reference":
            if scored_pcm_sha != gate_row.get("candidate_waveform_float32_sha256"):
                raise ScoreDiagnosticError(f"same-theta reference PCM differs from actual GateDC ledger: {row_id}/{mode}")
            if gate_row.get("source_pcm_float32_sha256") != source_sha or gate_row.get("source_sample_count") != sample_count:
                raise ScoreDiagnosticError(f"same-theta reference source differs from actual Fit40 PCM: {row_id}/{mode}")
            ref_check = _compare_reference(score_metrics, gate_row)
            reference_ok.append(ref_check["within_existing_tolerance"])
            if not ref_check["within_existing_tolerance"]:
                reference_failures.append({"row_id": row_id, "mode": mode, "reason": "SCALAR_REFERENCE_REPLAY_OUTSIDE_EXISTING_TOLERANCE"})
        else:
            if control_record is None or control_name is None:
                raise ScoreDiagnosticError("control score row has no post-projection record")
            baseline_sha = _tensor_sha(baseline, torch)
            if (
                control_record.get("row_id") != row_id or control_record.get("base_gate_mode") != mode
                or control_record.get("control") != control_name
                or control_record.get("input_source_pcm_float32_sha256") != source_sha
                or control_record.get("source_sample_count") != sample_count
                or control_record.get("canonical_rms_baseline_pcm_float32_sha256") != baseline_sha
                or control_record.get("same_frozen_parameter_sha256") != theta_sha
            ):
                raise ScoreDiagnosticError(f"control score row source/branch/theta differs: {row_id}/{mode}/{control_name}")
            mask_pin = control_record["post_projection_mask"].get("mask_float32_sha256")
            if _tensor_sha(control_mask_cache[(row_id, mode, control_name)], torch) != mask_pin:
                raise ScoreDiagnosticError(f"control mask differs from actual post-projection record: {row_id}/{mode}/{control_name}")

        record = {
            "schema": SCHEMA,
            "record_id": f"{row_id}::{mode}::{control_name or 'original'}",
            "role": role,
            "row_id": row_id,
            "sample_id": str(row["sample_id"]),
            "speaker_id": str(row["speaker_id"]),
            "base_gate_mode": mode,
            "control": control_name,
            "reference_record_id": f"{row_id}::{mode}::original",
            "natural_source_pcm_float32_sha256": source_sha,
            "natural_source_file_sha256": source_file_sha,
            "scored_input_pcm_float32_sha256": scored_pcm_sha,
            "source_sample_rate_hz": SAMPLE_RATE,
            "source_sample_count": sample_count,
            "source_clock_fps": {"numerator": VIDEO_FPS_NUM, "denominator": VIDEO_FPS_DEN},
            "source_video_sha256": str(context.source_video_sha256),
            "query_start_frame": int(context.query_start_frame),
            "query_count": QUERY_COUNT,
            "lag_offsets_frames": list(range(-LAG_RADIUS, LAG_RADIUS + 1)),
            "rendered_native_crop_shape": list(score["rendered_crop_shape"]),
            "distance_matrix_shape_lag_by_query": list(matrix.shape),
            "primary_queries_scored": list(range(QUERY_COUNT)),
            "score_metrics": score_metrics,
            "reference_replay_check": ref_check,
            "query_audio_support": query_support,
            "post_projection_record_sha256": _sha(_canonical(dict(control_record))) if control_record is not None else None,
            "post_projection_mask_summary": None,
        }
        if control_record is None:
            record["seam_sensitivity_secondary"] = None
        else:
            mask_summary = control_record["post_projection_mask"]
            record["post_projection_mask_summary"] = {
                "mask_float32_sha256": mask_summary["mask_float32_sha256"],
                "per_bin_mean_close_all_257": mask_summary["mean_close_all_bins"],
                "per_bin_mean_close_count": mask_summary["mean_close_bin_count"],
                "max_per_bin_mean_abs_error_db": mask_summary["max_per_bin_mean_abs_error_db"],
                "distribution_exactly_preserved_all_bins": mask_summary["distribution_exactly_preserved_all_bins"],
                "distribution_exactly_preserved_bin_count": mask_summary["distribution_exactly_preserved_bin_count"],
                "phase_semantics": control_record["stft"]["phase_semantics"],
                "reprojection_applied": False,
            }
            record["seam_sensitivity_secondary"] = _seam_summary(
                control_record, query_support, int(control_record["stft"]["frame_count"]), sample_count
            )
        records.append(record)
        matrices.append(matrix.detach().to(device="cpu", dtype=torch.float32).contiguous())
        curves.append(curve.detach().to(device="cpu", dtype=torch.float32).contiguous())
        cosines.append(cosine.detach().to(device="cpu", dtype=torch.float32).contiguous())
        norms.append(visual_norm.detach().to(device="cpu", dtype=torch.float32).contiguous())
        return len(records) - 1

    # Baselines are replayed first so each control is compared to this exact
    # row/mode matrix. Original PCM hashes must match the historical 80 rows.
    with torch.no_grad():
        for row_id in row_ids:
            source, baseline = sources[row_id], baseline_waveforms[row_id]
            source_sha, _, sample_count = _fit_identity(row_map[row_id])
            if int(source.numel()) != sample_count or _tensor_sha(source, torch) != source_sha:
                raise ScoreDiagnosticError(f"source tensor differs from exact original PCM: {row_id}")
            for mode in BRANCHES:
                gate_row = gate_map[(row_id, mode)]
                original = head(source, baseline, gate_mode=mode)
                waveform = original.get("output_pcm_float32")
                if original.get("gate_mode") != mode or int(original.get("source_sample_count", -1)) != sample_count:
                    raise ScoreDiagnosticError(f"frozen head replay changed branch or source clock: {row_id}/{mode}")
                if _tensor_sha(waveform, torch) != gate_row["candidate_waveform_float32_sha256"]:
                    raise ScoreDiagnosticError(f"frozen theta replay differs from original Fit40 waveform: {row_id}/{mode}")
                reference_index[(row_id, mode)] = score_one(
                    row_id, mode, waveform, role="same_theta_reference", control_name=None,
                    gate_row=gate_row, control_record=None,
                )

        for row_id in row_ids:
            source, baseline = sources[row_id], baseline_waveforms[row_id]
            _, _, sample_count = _fit_identity(row_map[row_id])
            for mode in BRANCHES:
                original = head(source, baseline, gate_mode=mode)
                original_mask = original.get("projected_log_magnitude_mask_db")
                if original_mask is None or tuple(original_mask.shape)[0] != 257 or not torch.isfinite(original_mask).all().item():
                    raise ScoreDiagnosticError(f"replayed post-projection mask is invalid: {row_id}/{mode}")
                gate_row = gate_map[(row_id, mode)]
                for name in CONTROLS:
                    key = (row_id, mode, name)
                    control_record = control_map[key]
                    mask = _make_control_mask(original_mask, name, torch)
                    control_mask_cache[key] = mask
                    waveform = control_waveforms["::".join(key)]
                    if _tensor_sha(mask, torch) != control_record["post_projection_mask"]["mask_float32_sha256"]:
                        raise ScoreDiagnosticError(f"post-projection mask reconstruction differs: {row_id}/{mode}/{name}")
                    if int(waveform.numel()) != sample_count or _tensor_sha(waveform, torch) != control_record["pcm"]["output_pcm_float32_sha256"]:
                        raise ScoreDiagnosticError(f"post-projection PCM differs from its helper record: {row_id}/{mode}/{name}")
                    index = score_one(
                        row_id, mode, waveform, role="post_projection_control", control_name=name,
                        gate_row=gate_row, control_record=control_record,
                    )
                    ref = matrices[reference_index[(row_id, mode)]]
                    delta = matrices[index] - ref
                    # Do not subset the stored matrix. The boolean seam mask
                    # only marks secondary summaries for later analysis.
                    control_deltas.append(delta.contiguous())
                    near = records[index]["seam_sensitivity_secondary"]
                    chosen = set(near.get("query_indices_with_mel_support_intersecting_join_neighborhood", [])) if near else set()
                    seam_masks.append(torch.tensor([query in chosen for query in range(QUERY_COUNT)], dtype=torch.bool))

    if len(records) != 320 or len(matrices) != 320 or len(control_deltas) != 240 or len(reference_ok) != 80:
        raise ScoreDiagnosticError("score-only diagnostic lost an original or control row")
    if _parameter_sha(head, torch) != theta_before or theta_before != theta_sha:
        raise ScoreDiagnosticError("D0 theta changed during no-update scoring")

    arrays = {
        "distance_matrices_float32": torch.stack(matrices, dim=0).contiguous(),
        "curves_float32": torch.stack(curves, dim=0).contiguous(),
        "zero_lag_unit_cosines_float32": torch.stack(cosines, dim=0).contiguous(),
        "visual_embedding_norms_float32": torch.stack(norms, dim=0).contiguous(),
        "control_minus_same_theta_reference_distance_matrices_float32": torch.stack(control_deltas, dim=0).contiguous(),
        "seam_query_mask_bool": torch.stack(seam_masks, dim=0).contiguous(),
    }
    array_manifest = {}
    for name, value in arrays.items():
        cpu = value.detach().to(device="cpu").contiguous()
        if str(cpu.dtype) == "torch.bool":
            raw = cpu.numpy().astype("uint8", copy=False).tobytes(order="C")
            dtype = "bool"
        else:
            raw = _tensor_bytes(cpu, torch)
            dtype = "float32"
        array_manifest[name] = {"shape": [int(x) for x in cpu.shape], "dtype": dtype, "byte_count": len(raw), "sha256": _sha(raw)}
    reference_pass = all(reference_ok)
    receipt = {
        "schema": SCHEMA,
        "status": "BUILT_FIT_ONLY_320_FULLQUERY_SCORE_DIAGNOSTIC_NOT_A_QUALIFICATION_RESULT",
        "fit_row_count": FIT_ROWS,
        "same_theta_reference_count": 80,
        "post_projection_control_count": 240,
        "scored_record_count": len(records),
        "row_order": row_ids,
        "branch_order": list(BRANCHES),
        "control_order": list(CONTROLS),
        "lag_offsets_frames": list(range(-LAG_RADIUS, LAG_RADIUS + 1)),
        "query_count": QUERY_COUNT,
        "distance_matrix_definition": "31 unpadded natural-audio lag embeddings by all 27 unchanged source-clock visual queries; mean across all queries exactly reproduces the existing curve31 scorer output",
        "same_theta_reference_replay": {
            "status": "PASS_WITHIN_EXISTING_SAME_HOST_TOLERANCE" if reference_pass else "MISMATCH_RETAINED_NO_QUALIFICATION",
            "original_pcm_hashes_exact": True,
            "scalar_metric_atol": SAME_HOST_SCALAR_ATOL,
            "scalar_metric_rtol": 0.0,
            "tolerance_source": "existing D0 runner zero-head native curve31 and visual embedding check",
            "failed_row_count": len(reference_failures),
            "failed_rows": reference_failures,
        },
        "frozen_parameter_sha256": theta_sha,
        "runtime_identity": dict(runtime_identity),
        "lineage_pins": lineage,
        "source_pins": pinned_sources,
        "scope": {
            "fit_only": True,
            "native_wav2lip_and_syncnet_forwards": True,
            "full_video_file_encoded": False,
            "optimizer_created_or_stepped_by_this_helper": False,
            "head_theta_changed": False,
            "original_natural_audio_embedding_reference": True,
            "full_27_queries_scored_for_every_record": True,
            "seam_sensitivity_is_secondary": True,
            "queries_removed_for_seam": 0,
            "candidate_selected": False,
            "development_or_confirmation_opened": False,
            "api_called": False,
        },
        "interpretation_limit": "GateDC is a functional ablation under the global radial projection. Per-query arrays and seam overlaps are descriptive, not causal-locality evidence; fixed-RMS normalization and baseline correction remain global.",
        "arrays": array_manifest,
    }
    return {"receipt": receipt, "records": records, "arrays": arrays}


__all__ = ["SCHEMA", "score_fit40_post_projection_controls"]
