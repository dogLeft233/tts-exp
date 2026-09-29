"""Instrument calibration for the repaired phone/TFG experiment.

The calibration code is intentionally independent of the E_SEEN TFG scores.
It checks the teacher frontend, the PCM/STFT identity path, and whether the
MFA timing measurement reacts to controls with known timing changes.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts.experiments.phone_separability_enhancement.audio import BandGainRenderer, make_protected_mask
from scripts.experiments.phone_separability_enhancement.teacher import load_frozen_teacher, phone_margins_from_hidden, processor_parity

from .audio_contract import export_safe_pcm
from .assets import read_pcm16
from .config import file_sha256, proxy_environment, resource_decision, write_json
from .mfa import mfa_batch_attempt
from .quality import audit_timing, monotonic_phone_match, construct_local_boundary_control


def _zero_padded_shift(pcm: np.ndarray, samples: int) -> np.ndarray:
    """Shift a waveform without circular wraparound or length changes."""

    source = np.asarray(pcm, dtype=np.int16).reshape(-1)
    offset = int(samples)
    output = np.zeros_like(source)
    if offset >= 0:
        if offset < source.size:
            output[offset:] = source[: source.size - offset]
    else:
        advance = -offset
        if advance < source.size:
            output[: source.size - advance] = source[advance:]
    return output


def _speech_tokens(tokens: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [dict(token) for token in tokens if bool(token.get("speech", not token.get("silence", False)))]


def _global_shift_result(source: Sequence[Mapping[str, Any]], target: Sequence[Mapping[str, Any]], *, expected_ms: float, tolerance_ms: float = 25.0) -> dict[str, Any]:
    matching = monotonic_phone_match(_speech_tokens(source), _speech_tokens(target))
    shifts = []
    for row in matching.get("matches", []):
        target_index = row.get("target_index")
        if target_index is None:
            continue
        source_token = _speech_tokens(source)[int(row["source_index"])]
        target_token = _speech_tokens(target)[int(target_index)]
        shifts.append((float(target_token["start_s"]) - float(source_token["start_s"])) * 1000.0)
    observed = float(np.median(shifts)) if shifts else None
    return {
        "expected_shift_ms": float(expected_ms),
        "observed_shift_ms": observed,
        "matched_count": int(len(shifts)),
        "edit_rate": float(matching.get("edit_rate", 1.0)),
        "detected": bool(observed is not None and observed >= float(expected_ms) - float(tolerance_ms) and matching.get("matched_count", 0) > 0),
    }


def _local_shift_result(source: Sequence[Mapping[str, Any]], target: Sequence[Mapping[str, Any]], control: Mapping[str, Any], *, min_shift_ms: float = 20.0) -> dict[str, Any]:
    source_speech = _speech_tokens(source)
    target_speech = _speech_tokens(target)
    matching = monotonic_phone_match(source_speech, target_speech)
    left_source_index = int(control["boundary"]["token_left"])
    right_source_index = int(control["boundary"]["token_right"])
    source_to_speech = {int(token.get("token_index", index)): index for index, token in enumerate(source_speech)}
    left_speech_index = source_to_speech.get(left_source_index)
    right_speech_index = source_to_speech.get(right_source_index)
    rows = {int(row["source_index"]): row for row in matching.get("matches", [])}
    left_match = rows.get(left_speech_index) if left_speech_index is not None else None
    right_match = rows.get(right_speech_index) if right_speech_index is not None else None
    observed = None
    if left_match and right_match:
        observed = (float(target_speech[int(left_match["target_index"])]["end_s"]) - float(source_speech[left_speech_index]["end_s"])) * 1000.0
    return {
        "expected_shift_ms": float(control["boundary"]["shift_ms"]),
        "observed_shift_ms": observed,
        "matched_left": bool(left_match),
        "matched_right": bool(right_match),
        "edit_rate": float(matching.get("edit_rate", 1.0)),
        "detected": bool(observed is not None and observed >= float(min_shift_ms) and matching.get("matched_count", 0) > 0),
    }


def _timing_inputs(registry: Mapping[str, Any], *, count: int, sample_rate: int, local_shift_ms: float) -> list[dict[str, Any]]:
    """Build a deterministic FIT candidate pool for fresh-MFA selection.

    The old implementation stopped after finding ``count`` rows whose parent
    TextGrid happened to contain two long adjacent phones.  That is not a
    valid fixture contract: the control is measured against this run's fresh
    MFA reference, whose phone durations can differ from the parent TextGrid.
    Keep the parent control only as a diagnostic fallback and let the caller
    select the final fixed denominator from fresh MFA tokens.
    """
    fit_rows = sorted((dict(row) for row in registry.get("rows", []) if str(row.get("analysis_split")) == "fit"), key=lambda row: str(row["pair_id"]))
    rows: list[dict[str, Any]] = []
    for row in fit_rows:
        pcm, _ = read_pcm16(row["natural"]["audio_path"], sample_rate=sample_rate)
        parent_local = None
        try:
            parent_local = construct_local_boundary_control(pcm, row["natural"]["tokens"], sample_rate=sample_rate, shift_ms=local_shift_ms)
        except (ValueError, IndexError):
            # The parent fixture is not the measurement reference.  Retain
            # the row so fresh MFA can decide whether it is structurally
            # eligible, while recording that no parent fallback is available.
            parent_local = None
        rows.append({
            "pair_id": str(row["pair_id"]),
            "transcript": str(row.get("transcript", "")),
            "pcm": pcm.copy(),
            "parent_tokens": [dict(token) for token in row["natural"]["tokens"]],
            "parent_local_control": parent_local,
        })
    if len(rows) < int(count):
        raise ValueError(f"timing calibration requires at least {count} FIT rows, got {len(rows)}")
    return rows


def _fresh_local_selection(
    pool: Sequence[Mapping[str, Any]],
    reference_rows: Sequence[Mapping[str, Any]],
    *,
    count: int,
    sample_rate: int,
    local_shift_ms: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select the fixed local-control denominator from fresh MFA tokens.

    Eligibility is structural only: the first deterministic FIT rows whose
    fresh MFA phone tier contains two adjacent speech intervals of at least
    160 ms are selected.  No control result or downstream score participates
    in this selection.
    """

    reference_by_id = {str(row["pair_id"]): row for row in reference_rows}
    selected: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    complete_reference_count = 0
    for item in pool:
        pair_id = str(item["pair_id"])
        reference = reference_by_id.get(pair_id)
        if not reference or reference.get("status") != "COMPLETE":
            rejected.append({"pair_id": pair_id, "reason": "FRESH_REFERENCE_MISSING"})
            continue
        complete_reference_count += 1
        try:
            local = construct_local_boundary_control(
                np.asarray(item["pcm"], dtype=np.int16),
                reference.get("tokens", []),
                sample_rate=sample_rate,
                shift_ms=local_shift_ms,
            )
        except (ValueError, IndexError) as exc:
            rejected.append({"pair_id": pair_id, "reason": f"{type(exc).__name__}:{exc}"})
            continue
        selected.append({
            "pair_id": pair_id,
            "transcript": str(item.get("transcript", "")),
            "pcm": np.asarray(item["pcm"], dtype=np.int16).copy(),
            "parent_tokens": [dict(token) for token in item.get("parent_tokens", [])],
            "reference_tokens": [dict(token) for token in reference.get("tokens", [])],
            "local_control": local,
            "control_source": "fresh_mfa_reference",
        })
        if len(selected) == int(count):
            break
    return selected, {
        "pool_count": len(pool),
        "fresh_reference_complete_count": complete_reference_count,
        "fresh_local_eligible_count": len(selected),
        "selected_pair_ids": [str(item["pair_id"]) for item in selected],
        "rejected_prefix": rejected[:32],
        "selection_rule": "first FIT rows by pair_id with fresh MFA adjacent speech spans >=160ms",
    }


def run_timing_calibration(config: Mapping[str, Any], root: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    quality = config.get("quality", {})
    count = int(quality.get("calibration_sentences", 8))
    sample_rate = int(config.get("audio", {}).get("sample_rate", 16000))
    global_shift_ms = float(quality.get("calibration_global_shift_ms", 80.0))
    local_shift_ms = float(quality.get("calibration_local_shift_ms", 40.0))
    pool = _timing_inputs(registry, count=count, sample_rate=sample_rate, local_shift_ms=local_shift_ms)
    pool_rows = [{"pair_id": item["pair_id"], "transcript": item["transcript"], "pcm": item["pcm"]} for item in pool]
    reference_pool = mfa_batch_attempt(config, root, rows=pool_rows, condition="fit_calibration_reference_pool")
    inputs, selection = _fresh_local_selection(
        pool,
        reference_pool.get("rows", []),
        count=count,
        sample_rate=sample_rate,
        local_shift_ms=local_shift_ms,
    )
    # Keep the corpus composition identical across all whole-corpus controls.
    # MFA may estimate corpus/speaker adaptation statistics, so aligning only
    # the selected eight rows would make N_REPEAT test a corpus change rather
    # than a repeated measurement of the same instrument.
    selected_by_id = {str(item["pair_id"]): item for item in inputs}
    repeat_rows = list(pool_rows)
    global_rows = [
        {**row, "pcm": _zero_padded_shift(row["pcm"], int(round(global_shift_ms * sample_rate / 1000.0)))}
        for row in pool_rows
    ]
    local_rows = []
    for item in pool:
        selected = selected_by_id.get(str(item["pair_id"]))
        local_rows.append({
            "pair_id": item["pair_id"],
            "transcript": item["transcript"],
            "pcm": selected["local_control"]["pcm"] if selected is not None else item["pcm"],
        })
    payloads: dict[str, Any] = {
        # Compare this runner's own fresh alignment with a repeated alignment;
        # historical parent boundaries are not an instrument reference.
        "reference": reference_pool,
        "repeat": mfa_batch_attempt(config, root, rows=repeat_rows, condition="fit_calibration_repeat_pool"),
        "global_shift": mfa_batch_attempt(config, root, rows=global_rows, condition="fit_calibration_global_shift_pool"),
        "local_shift": mfa_batch_attempt(config, root, rows=local_rows, condition="fit_calibration_local_shift_pool"),
    }
    metadata: list[dict[str, Any]] = []
    for item in inputs:
        local_control = item["local_control"]
        metadata.append({"pair_id": item["pair_id"], "parent_tokens": item["parent_tokens"], "reference_tokens": item["reference_tokens"], "local_control": local_control, "local_error": None, "control_source": item["control_source"]})
    by_condition = {name: {str(row["pair_id"]): row for row in payload.get("rows", [])} for name, payload in payloads.items()}
    rows_out: list[dict[str, Any]] = []
    repeat_pass = 0
    global_pass = 0
    local_pass = 0
    for item in metadata:
        pair_id = str(item["pair_id"])
        reference = item.get("reference_tokens")
        repeat = by_condition["repeat"].get(pair_id)
        shifted = by_condition["global_shift"].get(pair_id)
        local = by_condition["local_shift"].get(pair_id)
        row: dict[str, Any] = {"pair_id": pair_id, "local_control_error": item.get("local_error")}
        if reference is not None and repeat and repeat.get("status") == "COMPLETE":
            row["repeat"] = audit_timing(reference, repeat.get("tokens", []), max_edit_rate=float(quality.get("timing_max_edit_rate", 0.05)), min_coverage=float(quality.get("timing_min_coverage", 0.90)), median_error_ms=float(quality.get("timing_median_ms", 20.0)), p95_error_ms=float(quality.get("timing_p95_ms", 40.0)), pause_min_ms=float(quality.get("pause_min_ms", 50.0)), pause_iou_min=float(quality.get("pause_iou_min", 0.50)), edge_error_ms=float(quality.get("pause_edge_ms", 20.0)))
        else:
            row["repeat"] = {"timing_pass": False, "status": "FRESH_REFERENCE_MISSING" if reference is None else "MFA_INCOMPLETE"}
        if reference is not None and shifted and shifted.get("status") == "COMPLETE":
            row["global_shift"] = _global_shift_result(reference, shifted.get("tokens", []), expected_ms=global_shift_ms)
        else:
            row["global_shift"] = {"detected": False, "status": "FRESH_REFERENCE_MISSING" if reference is None else "MFA_INCOMPLETE"}
        if reference is not None and local and local.get("status") == "COMPLETE" and item.get("local_control") is not None:
            row["local_shift"] = _local_shift_result(reference, local.get("tokens", []), item["local_control"], min_shift_ms=float(quality.get("calibration_local_detect_min_ms", 20.0)))
        else:
            row["local_shift"] = {"detected": False, "status": "FRESH_REFERENCE_MISSING" if reference is None else "MFA_INCOMPLETE"}
        row["control_source"] = item.get("control_source")
        row["local_control_error"] = item.get("local_error")
        repeat_pass += int(bool(row["repeat"].get("timing_pass", False)))
        global_pass += int(bool(row["global_shift"].get("detected", False)))
        local_pass += int(bool(row["local_shift"].get("detected", False) and not item.get("local_error")))
        rows_out.append(row)
    expected = len(metadata)
    status = "PASS" if expected == count and repeat_pass == count and global_pass >= max(0, count - 1) and local_pass >= max(0, int(np.ceil(0.75 * count))) else "UNCALIBRATED"
    return {
        "status": status,
        "expected_sentences": expected,
        "requested_sentences": count,
        "selection": selection,
        "thresholds": {"repeat_pass": f"{count}/{count}", "global_shift_pass_min": max(0, count - 1), "local_shift_pass_min": max(0, int(np.ceil(0.75 * count))), "global_shift_ms": global_shift_ms, "local_shift_ms": local_shift_ms},
        "counts": {"repeat_pass": repeat_pass, "global_shift_detected": global_pass, "local_shift_detected": local_pass},
        "rows": rows_out,
        "mfa": {name: {key: value for key, value in payload.items() if key != "rows"} for name, payload in payloads.items()},
    }


def run_teacher_calibration(config: Mapping[str, Any], root: Path, registry: Mapping[str, Any], support: Mapping[str, Any]) -> dict[str, Any]:
    fit_rows = sorted((row for row in registry.get("rows", []) if str(row.get("analysis_split")) == "fit"), key=lambda row: str(row["pair_id"]))
    if not fit_rows:
        return {"status": "ENGINEERING_FAILURE", "reason": "NO_FIT_ROW"}
    gate = resource_decision(config, stage="calibrate", gpu_required=True)
    write_json(root / "01_calibration/resource_preflight.json", gate)
    if gate["decision"] != "PASS":
        return {"status": "RESOURCE_WAIT", "reason": gate["reasons"], "resource": gate["snapshot"]}
    device = str(config.get("runtime", {}).get("device", config.get("models", {}).get("device", "cuda:0")))
    model_cfg = config.get("models", {}).get("primary", {})
    try:
        with proxy_environment(config.get("runtime", {}).get("proxy")):
            teacher = load_frozen_teacher(model_cfg, device=device, proxy=config.get("runtime", {}).get("proxy"), allow_download=bool(config.get("runtime", {}).get("allow_model_download", True)))
        pcm, _ = read_pcm16(fit_rows[0]["natural"]["audio_path"])
        waveform = np.asarray(pcm, dtype=np.float32) / 32768.0
        parity = processor_parity(teacher, waveform)
        import torch

        value = torch.as_tensor(waveform, dtype=torch.float32, device=device).requires_grad_(True)
        hidden, frame_times, _ = teacher.encode(value, layer=int(model_cfg.get("layer", 6)))
        centroids = support.get("mixed_centroids", {}).get("hubert", {})
        loss, phone_rows = phone_margins_from_hidden(hidden, frame_times, fit_rows[0]["natural"]["tokens"], centroids, view="core")
        if not loss.requires_grad:
            loss = hidden.square().mean()
        loss.backward()
        parameter_grads = [parameter.grad for parameter in teacher.model.parameters() if parameter.grad is not None]
        gradient = {"status": "PASS" if value.grad is not None and torch.isfinite(value.grad).all() and not parameter_grads else "FAIL", "waveform_grad_norm": float(torch.linalg.vector_norm(value.grad).detach().cpu()) if value.grad is not None else None, "teacher_parameter_grads": len(parameter_grads), "phone_tokens": len(phone_rows)}
        result = {"status": "PASS" if parity.get("pass") and gradient.get("status") == "PASS" else "FAIL", "processor_parity": parity, "gradient": gradient, "teacher": {"key": teacher.key, "revision": teacher.revision, "frontend": teacher.frontend, "processor": teacher.processor_info}}
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result
    except (ImportError, OSError, RuntimeError, ValueError, KeyError) as exc:
        return {"status": "DEPENDENCY_BLOCKED", "reason": f"{type(exc).__name__}:{exc}"}


def run_pcm_calibration(config: Mapping[str, Any], root: Path, registry: Mapping[str, Any]) -> dict[str, Any]:
    fit_rows = sorted((row for row in registry.get("rows", []) if str(row.get("analysis_split")) == "fit"), key=lambda row: str(row["pair_id"]))
    if not fit_rows:
        return {"status": "ENGINEERING_FAILURE", "reason": "NO_FIT_ROW"}
    pcm, _ = read_pcm16(fit_rows[0]["natural"]["audio_path"])
    protection = make_protected_mask(fit_rows[0]["natural"]["tokens"], pcm.size)
    try:
        import torch

        renderer = BandGainRenderer(sample_rate=int(config.get("audio", {}).get("sample_rate", 16000)), n_fft=int(config.get("audio", {}).get("n_fft", 512)), win_length=int(config.get("audio", {}).get("win_length", 512)), hop_length=int(config.get("audio", {}).get("hop_length", 128)), n_bands=int(config.get("audio", {}).get("n_bands", 24)), max_gain_db=float(config.get("audio", {}).get("max_gain_db", 6.0)))
        waveform = torch.as_tensor(pcm.astype(np.float32) / 32768.0)
        features, _ = renderer.band_features(waveform)
        zero, render_meta = renderer.render(waveform, torch.zeros((1, renderer.n_bands, features.shape[-1])), torch.as_tensor(protection["mask"]))
        exported, pcm_meta = export_safe_pcm(zero, pcm, protection["protected"], max_residual_ratio=float(config["audio"]["max_residual_energy_ratio"]), min_snr_db=float(config["audio"]["min_snr_db"]), max_rms_change_db=float(config["audio"]["max_rms_change_db"]))
        passed = bool(np.array_equal(exported, pcm))
        return {"status": "PASS" if passed else "FAIL", "render": render_meta, "pcm": pcm_meta, "pcm_identity": passed}
    except (ImportError, RuntimeError, ValueError) as exc:
        return {"status": "FAIL", "reason": f"{type(exc).__name__}:{exc}"}


def fit_feature_stats(config: Mapping[str, Any], registry: Mapping[str, Any]) -> dict[str, Any]:
    """Estimate and freeze FIT-only 24-band log-power normalization."""

    import torch

    audio = config.get("audio", {})
    renderer = BandGainRenderer(sample_rate=int(audio.get("sample_rate", 16000)), n_fft=int(audio.get("n_fft", 512)), win_length=int(audio.get("win_length", 512)), hop_length=int(audio.get("hop_length", 128)), n_bands=int(audio.get("n_bands", 24)), max_gain_db=float(audio.get("max_gain_db", 6.0)))
    sums = None
    squares = None
    count = 0
    for row in sorted((item for item in registry.get("rows", []) if str(item.get("analysis_split")) == "fit"), key=lambda item: str(item["pair_id"])):
        pcm, _ = read_pcm16(row["natural"]["audio_path"], sample_rate=int(audio.get("sample_rate", 16000)))
        waveform = torch.as_tensor(pcm.astype(np.float32) / 32768.0)
        features, _ = renderer.band_features(waveform)
        values = features[0].detach().cpu().numpy().astype(np.float64)
        sums = values.sum(axis=1) if sums is None else sums + values.sum(axis=1)
        squares = (values * values).sum(axis=1) if squares is None else squares + (values * values).sum(axis=1)
        count += int(values.shape[1])
    if not count or sums is None or squares is None:
        raise ValueError("NO_FIT_FEATURE_FRAMES")
    mean = sums / float(count)
    variance = np.maximum(squares / float(count) - mean * mean, 1e-6)
    std = np.maximum(np.sqrt(variance), 1e-6)
    return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist(), "n_frames": int(count), "feature_type": "24_band_log_power", "fit_split_only": True, "floor": 1e-6}


__all__ = ["fit_feature_stats", "run_pcm_calibration", "run_teacher_calibration", "run_timing_calibration"]
