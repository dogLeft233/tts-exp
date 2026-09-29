#!/usr/bin/env python3
"""Run the fixed four-pair WORLD/DIO repair pilot.

This entry point is intentionally separate from the historical v1 runner.  It
copies only the four frozen pilot records from the parent run, audits the old
parameter/WAV artifacts, generates the v2 audio once, and stops before any
Wav2Lip or SyncNet work.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tts_f0_swap as core  # noqa: E402


PROTOCOL_ID = core.REPAIR_PROTOCOL_ID
PARENT_DEFAULT = REPO / "runs/tts_f0_swap_f0spec_audit2"
FIXED_RECORDS = {
    "aishell1_test_400__BAC009S0765W0312": (26, "S0765"),
    "aishell1_test_400__BAC009S0770W0414": (89, "S0770"),
    "aishell1_test_400__BAC009S0901W0487": (149, "S0901"),
    "aishell1_test_400__BAC009S0906W0401": (188, "S0906"),
}


class RepairError(RuntimeError):
    """A frozen repair protocol or artifact is invalid."""


def _sha256(path: Path) -> str:
    return core.file_sha256(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    core.write_json(path, payload, self_hash=True)


def _read_json(path: Path, *, self_hash: bool = False) -> dict[str, Any]:
    value = core.read_json(path)
    if not isinstance(value, dict):
        raise RepairError(f"expected JSON object: {path}")
    if self_hash:
        return core.verify_json(path, self_hash=True)
    return value


def run_root(run_id: str) -> Path:
    return REPO / "runs" / f"tts_f0_swap_repair_{core.run_id_valid(run_id)}"


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO), capture_output=True, text=True, check=False, timeout=10)
        return completed.stdout.strip() or None
    except Exception:
        return None


def _load_parent(parent_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    inputs_path = parent_root / "inputs.json"
    pilot_path = parent_root / "pilot_qc.json"
    if not inputs_path.is_file() or not pilot_path.is_file():
        raise RepairError(f"parent run is incomplete: {parent_root}")
    inputs = _read_json(inputs_path, self_hash=True)
    pilot = _read_json(pilot_path, self_hash=True)
    records = {str(row.get("paired_key")): dict(row) for row in inputs.get("records", []) if row.get("role") == "pilot"}
    if set(records) != set(FIXED_RECORDS):
        raise RepairError(f"INPUT_MISMATCH: parent pilot keys {sorted(records)}")
    selected: list[dict[str, Any]] = []
    for key in sorted(FIXED_RECORDS):
        record = records[key]
        expected_sample, expected_speaker = FIXED_RECORDS[key]
        if int(record.get("sample_id", -1)) != expected_sample or str(record.get("speaker_id")) != expected_speaker:
            raise RepairError(f"INPUT_MISMATCH: {key} sample/speaker differs from frozen table")
        for field in ("natural_audio", "tts_audio", "natural_textgrid", "tts_textgrid"):
            path = Path(str(record.get(field, "")))
            if not path.is_file():
                raise RepairError(f"INPUT_MISSING: {key}/{field}: {path}")
        for field, hash_field in (("natural_audio", "natural_container_sha256"), ("tts_audio", "tts_container_sha256"), ("natural_textgrid", "natural_textgrid_sha256"), ("tts_textgrid", "tts_textgrid_sha256")):
            path = Path(str(record[field]))
            recorded = str(record.get(hash_field, ""))
            if recorded and _sha256(path) != recorded:
                raise RepairError(f"INPUT_HASH_MISMATCH: {key}/{field}")
        selected.append(record)
    return {"inputs": inputs, "pilot": pilot, "inputs_sha256": _sha256(inputs_path), "pilot_sha256": _sha256(pilot_path)}, selected


def _f0_delta(target: np.ndarray, original: np.ndarray, mask: np.ndarray) -> np.ndarray:
    result = np.full(original.shape, np.nan, dtype=np.float64)
    good = mask & (target > 0.0) & (original > 0.0)
    result[good] = 12.0 * np.log2(target[good] / original[good])
    return result


def _old_parameter_audit(parent_root: Path, records: list[dict[str, Any]], pilot: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute old mean/dose quantities without changing the old run."""

    results: list[dict[str, Any]] = []
    old_manifests = {str(item.get("paired_key")): item for item in pilot.get("manifests", []) if isinstance(item, Mapping)}
    # The detailed failure reasons are copied from the parent CSV verbatim.
    qc_path = parent_root / "audio_qc.csv"
    qc_rows: list[dict[str, Any]] = []
    if qc_path.is_file():
        with qc_path.open(encoding="utf-8", newline="") as handle:
            qc_rows = list(csv.DictReader(handle))
    qc_by_key = {(str(row.get("paired_key")), str(row.get("receiver"))): row for row in qc_rows}
    for record in records:
        key = str(record["paired_key"])
        manifest = old_manifests.get(key, {})
        parameter_path = parent_root / "parameters" / f"{key}.npz"
        if not parameter_path.is_file():
            raise RepairError(f"old parameter missing: {parameter_path}")
        with np.load(parameter_path, allow_pickle=False) as arrays:
            for receiver in ("N", "T"):
                original = np.asarray(arrays[f"{receiver}_f0"], dtype=np.float64)
                target = np.asarray(arrays[f"{receiver}_target_CONTOUR"], dtype=np.float64)
                weight = np.asarray(arrays[f"{receiver}_weight"], dtype=np.float64)
                valid = np.asarray(arrays[f"{receiver}_valid"], dtype=bool)
                voiced = original > 0.0
                delta = _f0_delta(target, original, voiced)
                weighted_support = valid & (weight > 0.0) & np.isfinite(delta)
                strong_support = valid & (weight >= 0.5) & np.isfinite(delta)
                old_weighted_mean = float(np.sum(weight[weighted_support] * delta[weighted_support]) / np.sum(weight[weighted_support])) if np.any(weighted_support) else None
                old_weighted_dose = float(np.sqrt(np.sum(weight[weighted_support] * np.square(delta[weighted_support])) / np.sum(weight[weighted_support]))) if np.any(weighted_support) else None
                v2_dose = float(np.sqrt(np.mean(np.square(delta[strong_support])))) if np.any(strong_support) else None
                audio_stats: dict[str, Any] = {}
                audio_entry = manifest.get("audio", {}).get(receiver, {}) if isinstance(manifest, Mapping) else {}
                for arm in ("RAW", "ID", "LEVEL", "CONTOUR"):
                    item = audio_entry.get(arm, {}) if isinstance(audio_entry, Mapping) else {}
                    path = Path(str(item.get("path", "")))
                    if path.is_file():
                        values, meta = core.decode_audio(path)
                        audio_stats[arm] = {"path": str(path.resolve()), "sha256": _sha256(path), "sample_count": int(values.size), "peak": float(np.max(np.abs(values), initial=0.0)), "rms": core.rms(values), "manifest_sha256": item.get("container_sha256")}
                    else:
                        audio_stats[arm] = {"status": "MISSING", "path": str(path)}
                results.append({"paired_key": key, "receiver": receiver, "parameter_path": str(parameter_path.resolve()), "parameter_sha256": _sha256(parameter_path), "old_ordinary_mean_delta_st": float(np.nanmean(delta[voiced])) if np.any(np.isfinite(delta[voiced])) else None, "old_weighted_mean_delta_st": old_weighted_mean, "old_weighted_dose_rms_st": old_weighted_dose, "v2_definition_dose_rms_st": v2_dose, "voiced_frames": int(np.sum(voiced)), "valid_frames": int(np.sum(valid)), "audio": audio_stats, "old_qc": qc_by_key.get((key, receiver), {}), "old_manifest_parameter_sha256": manifest.get("parameter_sha256") if isinstance(manifest, Mapping) else None})
    return {"parent_run": str(parent_root.resolve()), "parent_pilot_qc_sha256": _sha256(parent_root / "pilot_qc.json"), "records": results, "old_gate": pilot.get("gate"), "audit_findings": ["old identity_error field represented the taper-weighted mean; v2 records ordinary and weighted means separately"]}


def _write_csv(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({str(key) for row in rows for key in row})
    if "paired_key" in fields:
        fields.remove("paired_key")
        fields.insert(0, "paired_key")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _repair_inputs(parent: Mapping[str, Any], records: list[dict[str, Any]], parent_root: Path) -> dict[str, Any]:
    inputs = parent["inputs"]
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "parent_run": str(parent_root.resolve()),
        "parent_inputs_sha256": parent["inputs_sha256"],
        "parent_pilot_qc_sha256": parent["pilot_sha256"],
        "fixed_pair_count": 4,
        "fixed_keys": [str(row["paired_key"]) for row in records],
        "records": records,
        "formal_keys": [str(key) for key in inputs.get("formal_keys", [])],
        "formal_record_count": int(inputs.get("formal_pair_count", 0)),
        "selection": "copied verbatim from parent pilot; no score-based reselection",
        "execution_scope": "audio-only; no Wav2Lip/SyncNet",
    }


def _protocol(root: Path, parent: Mapping[str, Any], inputs: Mapping[str, Any], parent_root: Path) -> dict[str, Any]:
    runner = Path(__file__).resolve()
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "formula_version": "ordinary_mean_preserving_v2",
        "measurement_contract": "H_source,H_RAW,H_ID,H_LEVEL,H_CONTOUR,D_RAW,D_ID,D_LEVEL,D_CONTOUR; exact_grid_no_interp",
        "parent_run": str(parent_root.resolve()),
        "parent_inputs_sha256": parent["inputs_sha256"],
        "parent_pilot_qc_sha256": parent["pilot_sha256"],
        "fixed_keys": list(inputs["fixed_keys"]),
        "thresholds": {"min_effective_coverage": core.MIN_EFFECTIVE_COVERAGE, "min_effective_phones": core.MIN_EFFECTIVE_PHONES, "min_voiced_seconds": core.MIN_VOICED_SECONDS, "ordinary_identity_abs_st": 1e-8, "id_mask_mismatch": 0.10, "id_coverage": 0.80, "candidate_mask_mismatch": 0.05, "candidate_coverage": 0.80, "pilot_required_pairs": 3, "pilot_denominator": 4},
        "world": {"sample_rate": core.SAMPLE_RATE, "frame_period_ms": core.FRAME_PERIOD_MS, "f0_floor": core.F0_FLOOR, "f0_ceil": core.F0_CEIL, "pyworld_expected": "0.3.5"},
        "runner": str(runner),
        "runner_sha256": _sha256(runner),
        "core_runner": str(Path(core.__file__).resolve()),
        "core_runner_sha256": _sha256(Path(core.__file__).resolve()),
        "spec": str(core.REPAIR_SPEC_PATH.resolve()),
        "spec_sha256": _sha256(core.REPAIR_SPEC_PATH) if core.REPAIR_SPEC_PATH.is_file() else None,
        "runtime": {"python": platform.python_version(), "platform": platform.platform(), "git": _git_commit(), "created_at": datetime.now(timezone.utc).isoformat()},
        "gpu_stages_run": 0,
    }


def _run_audio(root: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    paths = core.RunPaths(root)
    manifests: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []
    for record in records:
        try:
            result = core.build_audio_for_pair(record, paths, require_measurement=True)
            result.pop("world", None)
            manifests.append(result)
            qc_rows.extend(result.get("qc", []))
            diagnostic_rows.extend(result.get("diagnostics", []))
        except Exception as exc:  # keep all four fixed pairs in denominator
            failures.append({"paired_key": record.get("paired_key"), "status": "PAIR_ERROR", "error": f"{type(exc).__name__}:{exc}"})
            qc_rows.extend({"paired_key": record.get("paired_key"), "sample_id": record.get("sample_id"), "speaker_id": record.get("speaker_id"), "receiver": receiver, "status": "PAIR_ERROR", "reasons": f"{type(exc).__name__}:{exc}"} for receiver in ("N", "T"))
    audio_manifest: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "complete" if not failures else "incomplete",
        "expected_pair_count": len(records),
        "pair_count": len(manifests),
        "fixed_keys": [str(row["paired_key"]) for row in records],
        "manifests": manifests,
        "failures": failures,
        "qc_row_count": len(qc_rows),
        "diagnostics_row_count": len(diagnostic_rows),
        "audio_backend": core._audio_backend_info(),
    }
    _write_json(root / "audio_manifest.json", audio_manifest)
    _write_csv(root / "audio_qc.csv", qc_rows)
    _write_csv(root / "diagnostics.csv", diagnostic_rows)
    gate = core.pilot_gate(audio_manifest, expected_keys=[str(row["paired_key"]) for row in records])
    audio_manifest["gate"] = gate
    audio_manifest["audio_qc_sha256"] = _sha256(root / "audio_qc.csv")
    audio_manifest["diagnostics_sha256"] = _sha256(root / "diagnostics.csv")
    _write_json(root / "audio_manifest.json", audio_manifest)
    return audio_manifest


def _svg_series(values: np.ndarray, time: np.ndarray, *, color: str, width: float, height: float, left: float, top: float, x0: float, x1: float, y0: float, y1: float, positive_only: bool = False) -> list[str]:
    """Return compact SVG polylines, breaking lines at missing/unvoiced frames."""

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    time = np.asarray(time, dtype=np.float64).reshape(-1)
    if values.shape != time.shape:
        return []
    valid = np.isfinite(values) & np.isfinite(time)
    if positive_only:
        valid &= values > 0.0
    if not np.any(valid) or not np.isfinite([x0, x1, y0, y1]).all() or x1 <= x0 or y1 <= y0:
        return []
    indices = np.flatnonzero(valid)
    chunks = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    lines: list[str] = []
    for chunk in chunks:
        if chunk.size < 2:
            continue
        # Keep SVG sizes bounded while preserving all short pilot trajectories.
        stride = max(1, int(np.ceil(chunk.size / 1200)))
        chunk = chunk[::stride]
        points = []
        for index in chunk:
            x = left + (float(time[index]) - x0) / (x1 - x0) * width
            y = top + height - (float(values[index]) - y0) / (y1 - y0) * height
            points.append(f"{x:.2f},{y:.2f}")
        if len(points) >= 2:
            lines.append(f'<polyline fill="none" stroke="{html.escape(color)}" stroke-width="1.4" points="{" ".join(points)}"/>')
    return lines


def _write_diagnostic_plots(root: Path, audio: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Write one lightweight, dependency-free SVG trajectory plot per receiver."""

    plot_dir = root / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[dict[str, Any]] = []
    colors = {"H_source": "#111111", "H_RAW": "#1f77b4", "D_RAW": "#17becf", "H_ID": "#9467bd", "D_ID": "#2ca02c", "target": "#d62728", "D_CONTOUR": "#ff7f0e"}
    for item in audio.get("manifests", []):
        pair = str(item.get("paired_key"))
        parameter_path = Path(str(item.get("parameter_path", "")))
        if not parameter_path.is_file():
            continue
        with np.load(parameter_path, allow_pickle=False) as arrays:
            for receiver in ("N", "T"):
                prefix = receiver
                time = np.asarray(arrays[f"{prefix}_time"], dtype=np.float64)
                trajectories = {
                    name: np.asarray(arrays[f"{prefix}_{name}_f0"], dtype=np.float64)
                    for name in ("H_source", "H_RAW", "D_RAW", "H_ID", "D_ID", "D_CONTOUR")
                }
                target = np.asarray(arrays[f"{prefix}_target_CONTOUR"], dtype=np.float64)
                weight = np.asarray(arrays[f"{prefix}_weight"], dtype=np.float64)
                semitone = lambda f0: np.where(f0 > 0.0, 12.0 * np.log2(np.maximum(f0, 1e-12)), np.nan)
                top_values = [values[values > 0.0] for values in trajectories.values() if np.any(values > 0.0)]
                if not top_values:
                    continue
                top_concat = np.concatenate(top_values)
                f0_max = max(100.0, float(np.nanmax(top_concat)) * 1.05)
                st_values = [semitone(trajectories["D_ID"]), semitone(target), semitone(trajectories["D_CONTOUR"])]
                st_concat = np.concatenate([values[np.isfinite(values)] for values in st_values if np.any(np.isfinite(values))])
                st_min, st_max = float(np.nanmin(st_concat)) - 1.0, float(np.nanmax(st_concat)) + 1.0
                x0, x1 = float(time[0]), float(time[-1]) if time.size else 1.0
                if x1 <= x0:
                    x1 = x0 + 1.0
                width, height, left, right, top, middle, bottom = 1120.0, 420.0, 80.0, 30.0, 52.0, 272.0, 52.0
                total_h = top + middle + bottom + 80.0
                parts = [
                    '<?xml version="1.0" encoding="UTF-8"?>',
                    f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width + left + right)}" height="{int(total_h)}" viewBox="0 0 {int(width + left + right)} {int(total_h)}">',
                    '<rect width="100%" height="100%" fill="white"/>',
                    f'<text x="{left}" y="24" font-family="sans-serif" font-size="16">{html.escape(pair)} / {receiver}</text>',
                    f'<text x="8" y="{top + 12}" font-family="sans-serif" font-size="12">F0 (Hz)</text>',
                    f'<text x="8" y="{top + middle + 12}" font-family="sans-serif" font-size="12">semitone / w</text>',
                    f'<rect x="{left}" y="{top}" width="{width}" height="{middle}" fill="#fafafa" stroke="#888"/>',
                    f'<rect x="{left}" y="{top + middle + 36}" width="{width}" height="{bottom}" fill="#fafafa" stroke="#888"/>',
                ]
                for name in ("H_source", "H_RAW", "D_RAW", "H_ID", "D_ID"):
                    parts.extend(_svg_series(trajectories[name], time, color=colors[name], width=width, height=middle, left=left, top=top, x0=x0, x1=x1, y0=0.0, y1=f0_max, positive_only=True))
                contour_top = top + middle + 36
                for name, values in (("D_ID", semitone(trajectories["D_ID"])), ("target", semitone(target)), ("D_CONTOUR", semitone(trajectories["D_CONTOUR"]))):
                    parts.extend(_svg_series(values, time, color=colors[name], width=width, height=bottom, left=left, top=contour_top, x0=x0, x1=x1, y0=st_min, y1=st_max))
                parts.extend(_svg_series(weight, time, color="#777777", width=width, height=bottom, left=left, top=contour_top, x0=x0, x1=x1, y0=st_min, y1=st_max))
                legend = [("Hsrc", colors["H_source"]), ("Hraw", colors["H_RAW"]), ("Draw", colors["D_RAW"]), ("Hid", colors["H_ID"]), ("Did", colors["D_ID"]), ("target", colors["target"]), ("Dcontour", colors["D_CONTOUR"]), ("w", "#777777")]
                for index, (label, color) in enumerate(legend):
                    x = left + (index % 4) * 180
                    y = total_h - 30 + (index // 4) * 16
                    parts.append(f'<line x1="{x}" y1="{y - 4}" x2="{x + 22}" y2="{y - 4}" stroke="{color}" stroke-width="2"/>')
                    parts.append(f'<text x="{x + 28}" y="{y}" font-family="sans-serif" font-size="11">{label}</text>')
                parts.append('</svg>')
                path = plot_dir / f"{pair}_{receiver}.svg"
                path.write_text("\n".join(parts), encoding="utf-8")
                outputs.append({"paired_key": pair, "receiver": receiver, "path": str(path.resolve()), "sha256": _sha256(path)})
    return outputs


def _validate(root: Path, parent: Mapping[str, Any], records: list[dict[str, Any]], audio: Mapping[str, Any], old_audit: Mapping[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    try:
        protocol = _read_json(root / "protocol.json", self_hash=True)
        inputs = _read_json(root / "inputs.json", self_hash=True)
        audio_check = _read_json(root / "audio_manifest.json", self_hash=True)
        if protocol.get("protocol_id") != PROTOCOL_ID or inputs.get("protocol_id") != PROTOCOL_ID or audio_check.get("protocol_id") != PROTOCOL_ID:
            errors.append("protocol_id mismatch")
        if list(inputs.get("fixed_keys", [])) != [str(row["paired_key"]) for row in records]:
            errors.append("fixed key list mismatch")
        if int(audio_check.get("expected_pair_count", -1)) != 4:
            errors.append("pilot denominator is not four")
        if len(audio_check.get("manifests", [])) + len(audio_check.get("failures", [])) != 4:
            errors.append("fixed pair denominator lost a failure")
        for item in audio_check.get("manifests", []):
            if set(item.get("audio", {})) != {"N", "T"}:
                errors.append(f"receiver arms missing: {item.get('paired_key')}")
            for receiver in ("N", "T"):
                if set(item.get("audio", {}).get(receiver, {})) != {"RAW", "ID", "LEVEL", "CONTOUR"}:
                    errors.append(f"audio arms missing: {item.get('paired_key')}/{receiver}")
                for arm, meta in item.get("audio", {}).get(receiver, {}).items():
                    path = Path(str(meta.get("path", "")))
                    if not path.is_file() or _sha256(path) != str(meta.get("container_sha256", "")):
                        errors.append(f"audio hash mismatch: {item.get('paired_key')}/{receiver}_{arm}")
            parameter_path = Path(str(item.get("parameter_path", "")))
            if not parameter_path.is_file() or _sha256(parameter_path) != str(item.get("parameter_sha256", "")):
                errors.append(f"parameter hash mismatch: {item.get('paired_key')}")
        if (root / "videos").exists() or (root / "scores.csv").exists() or (root / "scores_manifest.json").exists():
            errors.append("GPU/video/score artifact exists in audio-only repair run")
        if not (root / "audio_qc.csv").is_file() or not (root / "diagnostics.csv").is_file():
            errors.append("diagnostic CSV missing")
        plots = audio_check.get("diagnostic_plots", [])
        expected_plots = 8
        if len(plots) != expected_plots:
            errors.append(f"diagnostic plot count mismatch: expected {expected_plots}, got {len(plots)}")
        for plot in plots:
            path = Path(str(plot.get("path", "")))
            if not path.is_file() or _sha256(path) != str(plot.get("sha256", "")):
                errors.append(f"diagnostic plot hash mismatch: {plot.get('paired_key')}/{plot.get('receiver')}")
        # Check the new implementation has no interpolation fallback.
        if "np.interp" in Path(core.__file__).read_text(encoding="utf-8"):
            errors.append("np.interp remains in F0 repair runner")
        if audio.get("gate", {}).get("pilot_pair_count") != 4:
            errors.append("gate denominator mismatch")
        if audio.get("gate", {}).get("required_pairs") != 3:
            errors.append("gate continuation threshold mismatch")
        if old_audit.get("records") is None or len(old_audit["records"]) != 8:
            errors.append("old audit does not contain eight receiver rows")
    except Exception as exc:  # preserve readable validation artifact
        errors.append(f"validation exception: {type(exc).__name__}:{exc}")
    payload = {"schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS" if not errors else "FAIL", "errors": errors, "warnings": warnings, "independent_recompute": "PENDING", "fixed_pair_count": 4, "qc_receiver_count": len(audio.get("manifests", [])) * 2, "checked_at": datetime.now(timezone.utc).isoformat()}
    _write_json(root / "validation.json", payload)
    return payload


def _report(root: Path, parent: Mapping[str, Any], old_audit: Mapping[str, Any], audio: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    rows = [row for item in audio.get("manifests", []) for row in item.get("qc", [])]
    pass_count = sum(str(row.get("status")) == "PASS" for row in rows)
    gate = audio.get("gate", {})
    lines = ["# tts_f0_swap_repair_v2", "", "本轮只验证固定 4 对 F0 音频操作；没有生成视频，也没有调用 Wav2Lip/SyncNet。", "", f"- 固定 pair：4；接收方 QC 行：{len(rows)}；接收方通过：{pass_count}/8", f"- Pilot gate：{gate.get('status')}（{gate.get('passed_pairs', 0)}/{gate.get('pilot_pair_count', 4)} 对；继续门槛 {gate.get('required_pairs', 3)}）", f"- validation：{validation.get('status')}（errors={len(validation.get('errors', []))}）", f"- 轨迹图：{len(audio.get('diagnostic_plots', []))} 个 SVG（每个已完成接收方一张）", ""]
    lines.extend(["## 旧产物公式审计", "", "旧 run 保持只读；旧字段中的恒等式是 taper 加权均值，下面同时保存普通均值和加权均值。"])
    for row in old_audit.get("records", []):
        lines.append(f"- {row['paired_key']} / {row['receiver']}: old mean(Δs[V])={row.get('old_ordinary_mean_delta_st')!s}, old weighted mean={row.get('old_weighted_mean_delta_st')!s}, v2 dose={row.get('v2_definition_dose_rms_st')!s}")
    lines.extend(["", "## 新诊断判读", ""])
    for item in audio.get("manifests", []):
        lines.append(f"- {item.get('paired_key')}: " + "; ".join(f"{row.get('receiver')}={row.get('status')} ({row.get('reasons', '')})" for row in item.get("qc", [])))
        for row in item.get("diagnostics", []):
            if row.get("comparison") in {"H_RAW->D_RAW", "D_RAW->D_ID"}:
                lines.append(f"  - {row.get('receiver')} {row.get('comparison')}: coverage={row.get('coverage')}, mask_mismatch={row.get('mask_mismatch')}, lost={row.get('lost_voiced')}, gained={row.get('gained_voiced')}")
    lines.extend(["", "RAW→D_RAW 的比较反映重合成前的检测器分歧；D_RAW→D_ID 反映重合成后新增变化。只有诊断计数同时存在时才作这一区分，不能把自动 F0 直接当作听感结论。", "", f"继续条件：{'达到' if gate.get('status') == 'PASS' else '未达到'} 3/4 pair 双向通过；本轮不回答 F0 是否解释 TFG 上的 TTS 增益。", ""])
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")


def _run_independent_recompute(root: Path) -> dict[str, Any]:
    checker = Path(__file__).with_name("tts_f0_swap_recompute.py")
    completed = subprocess.run([sys.executable, str(checker), "--repair-run-dir", str(root)], cwd=str(REPO), capture_output=True, text=True, check=False, timeout=300)
    output_path = root / "recompute.json"
    if not output_path.is_file():
        raise RepairError(f"independent recompute did not write {output_path}: {completed.stderr[-1000:]}")
    result = _read_json(output_path, self_hash=False)
    if result.get("status") != "PASS":
        raise RepairError(f"independent recompute failed: {result.get('reason', result.get('status'))}")
    return result


def run(run_id: str, *, parent_root: Path = PARENT_DEFAULT) -> dict[str, Any]:
    root = run_root(run_id)
    root.mkdir(parents=True, exist_ok=True)
    parent, records = _load_parent(parent_root)
    inputs = _repair_inputs(parent, records, parent_root)
    protocol = _protocol(root, parent, inputs, parent_root)
    protocol_path = root / "protocol.json"
    if protocol_path.is_file():
        existing = _read_json(protocol_path, self_hash=True)
        if existing.get("runner_sha256") != protocol["runner_sha256"] or existing.get("core_runner_sha256") != protocol["core_runner_sha256"] or existing.get("spec_sha256") != protocol["spec_sha256"] or existing.get("parent_inputs_sha256") != protocol["parent_inputs_sha256"]:
            raise RepairError("existing repair run has stale protocol; use a new run-id")
    else:
        _write_json(protocol_path, protocol)
    if not (root / "inputs.json").is_file():
        _write_json(root / "inputs.json", inputs)
    else:
        existing_inputs = _read_json(root / "inputs.json", self_hash=True)
        if existing_inputs.get("fixed_keys") != inputs.get("fixed_keys") or existing_inputs.get("parent_inputs_sha256") != inputs.get("parent_inputs_sha256"):
            raise RepairError("existing repair inputs differ from frozen parent")
    old_path = root / "old_audit.json"
    if old_path.is_file():
        old_audit = _read_json(old_path, self_hash=True)
    else:
        old_audit = _old_parameter_audit(parent_root, records, parent["pilot"])
        _write_json(old_path, old_audit)
    audio = _run_audio(root, records)
    audio["diagnostic_plots"] = _write_diagnostic_plots(root, audio)
    _write_json(root / "audio_manifest.json", audio)
    validation = _validate(root, parent, records, audio, old_audit)
    independent = _run_independent_recompute(root)
    validation["independent_recompute"] = "PASS"
    validation["independent_recompute_path"] = str((root / "recompute.json").resolve())
    validation["independent_recompute_sha256"] = _sha256(root / "recompute.json")
    _write_json(root / "validation.json", validation)
    _report(root, parent, old_audit, audio, validation)
    return {"run_root": str(root.resolve()), "protocol": protocol, "old_audit": old_audit, "audio": audio, "validation": validation, "independent_recompute": independent, "gate": audio.get("gate")}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--parent-run", type=Path, default=PARENT_DEFAULT)
    args = parser.parse_args(argv)
    try:
        result = run(args.run_id, parent_root=args.parent_run.resolve())
    except Exception as exc:  # leave a useful error in the terminal and nonzero status
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
