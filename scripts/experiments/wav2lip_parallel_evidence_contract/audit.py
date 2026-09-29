from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .common import ProtocolError, gain, group_bootstrap, load_worker, matrix_from_embeddings, metrics, read_json


def _rows(path: Path) -> list[dict[str, Any]]:
    value = read_json(path).get("rows")
    if not isinstance(value, list): raise ProtocolError(f"rows missing: {path}")
    return value


def _records() -> list[dict[str, str]]:
    rows = read_json(config.A / "protocol.json")["records"]
    return sorted(({"sample_id": str(row["sample_id"]), "source_group": str(row["source_group"])} for row in rows), key=lambda row: (row["source_group"], row["sample_id"]))


def _score_map(rows: list[dict[str, Any]], *, arms: set[str] | None = None, audio: set[str] | None = None) -> dict[tuple[str, str, str], dict[str, Any]]:
    result = {}
    for row in rows:
        key = (str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"]))
        if arms is not None and key[1] not in arms: continue
        if audio is not None and key[2] not in audio: continue
        result[key] = row
    return result


def audit_a(records: list[dict[str, str]]) -> dict[str, Any]:
    parent = _score_map(_rows(config.P / "control_scores/manifest.json"), arms={"N"}, audio={"N"})
    candidates = _score_map(_rows(config.A / "candidate_scores/manifest.json"), arms={"SMOOTH", "SHARP"}, audio={"N"})
    rows = []
    for rec in records:
        sid = rec["sample_id"]; base_row = parent[(sid, "N", "N")]
        v, a, producer = load_worker(base_row); natural = matrix_from_embeddings(v, a); base_m = metrics(natural)
        row = {"sample_id": sid, "source_group": rec["source_group"], "matrix_reproduction_max_abs": {"N": float(np.max(np.abs(natural - producer)))}}
        for arm in ("SMOOTH", "SHARP"):
            vv, aa, pp = load_worker(candidates[(sid, arm, "N")]); matrix = matrix_from_embeddings(vv, aa); row[arm] = metrics(matrix); row["matrix_reproduction_max_abs"][arm] = float(np.max(np.abs(matrix - pp)))
        row["N"] = base_m; row["gain"] = {arm: gain(row[arm], base_m) for arm in ("SMOOTH", "SHARP")}; rows.append(row)
    result = {"records": rows, "contrasts": {arm: {metric: group_bootstrap([{**row, metric: row["gain"][arm][metric]} for row in rows], metric) for metric in ("C", "D", "A")} for arm in ("SMOOTH", "SHARP")}}
    return result


def _curve_from_embeddings(visual: np.ndarray, audio: np.ndarray, start: int) -> np.ndarray:
    values = []
    for r in range(30, 58):
        vals = []
        for j in range(31):
            q = r + j - start
            if q < 0 or q >= audio.shape[0]: vals.append(np.nan)
            else: vals.append(float(np.sqrt(np.sum((visual[r] - audio[q] + 1e-6) ** 2, dtype=np.float64))))
        values.append(vals)
    array = np.asarray(values, dtype=np.float64)
    if not np.isfinite(array).all(): raise ProtocolError("B support contains an invalid q")
    return np.mean(array, axis=0, dtype=np.float64)


def audit_b(records: list[dict[str, str]]) -> dict[str, Any]:
    rows = _rows(config.P / "control_scores/manifest.json")
    pmap = _score_map(rows, arms={"N"}, audio={"N", "A_DELAY"})
    out = []
    old_pass = 0; matched_pass = 0
    for rec in records:
        sid = rec["sample_id"]; nr = pmap[(sid, "N", "N")]; dr = pmap[(sid, "N", "A_DELAY")]
        vn, an, pn = load_worker(nr); vd, ad, pd = load_worker(dr)
        if np.max(np.abs(vn - vd)) > 1e-4: raise ProtocolError(f"B visual embeddings differ: {sid}")
        b = _curve_from_embeddings(vn, an, 15); t = _curve_from_embeddings(vn, ad, 10); legacy = _curve_from_embeddings(vn, ad, 15)
        natural_offset = 15 - int(np.argmin(b))
        old_off = 15 - int(np.argmin(legacy)); new_off = 10 - int(np.argmin(t))
        old_pass += int(-6 <= (old_off - natural_offset) <= -4)
        matched_pass += int(-6 <= (new_off - natural_offset) <= -4)
        out.append({"sample_id": sid, "source_group": rec["source_group"], "legacy_offset": old_off, "matched_offset": new_off, "natural_peak_index": int(np.argmin(b)), "delay_peak_index": int(np.argmin(t)), "legacy_curve": legacy.tolist(), "matched_curve": t.tolist(), "legacy_damage_at_natural_k0": float(legacy[int(np.argmin(b))] - b[int(np.argmin(b))]), "embedding_visual_max_abs": float(np.max(np.abs(vn - vd))), "producer_matrix_max_abs": {"natural": float(np.max(np.abs(matrix_from_embeddings(vn, an) - pn))), "delay": float(np.max(np.abs(matrix_from_embeddings(vd, ad) - pd)))}})
    damages = [row["legacy_damage_at_natural_k0"] for row in out]
    group_values: dict[str, list[float]] = {}
    for row in out: group_values.setdefault(row["source_group"], []).append(row["legacy_damage_at_natural_k0"])
    labels = sorted(group_values); means = np.asarray([np.mean(group_values[x]) for x in labels]); rng = np.random.Generator(np.random.PCG64(20260909)); index = rng.integers(0, 8, size=(10_000, 8)); estimates = means[index].mean(axis=1)
    damage = {"mean": float(np.mean(damages)), "ci95": [float(np.quantile(estimates, .025, method="linear")), float(np.quantile(estimates, .975, method="linear"))], "group_labels": labels, "group_means": {x: float(y) for x, y in zip(labels, means, strict=True)}, "group_positive_count": int(np.sum(means > 0)), "seed": 20260909, "draws": 10_000, "indices": index.tolist()}
    return {"records": out, "legacy_offset_pass_count": old_pass, "matched_offset_pass_count": matched_pass, "damage": damage, "old_control_pass": bool(old_pass >= 14 and damage["ci95"][0] > 0 and damage["group_positive_count"] >= 7), "f46_boundary_restored": bool(old_pass == 16), "f46_control_as_spec": "PASS" if matched_pass >= 14 and damage["ci95"][0] > 0 and damage["group_positive_count"] >= 7 else "FAIL"}


def _k_for(sample_id: str, driver_rows: list[dict[str, Any]], masks: dict[str, dict[str, Any]]) -> list[int]:
    all_sets = []
    for row in driver_rows:
        if str(row["sample_id"]) != sample_id: continue
        columns: set[int] = set()
        for used in row.get("used_masks", []):
            mask = masks.get(str(used["mask_sha256"]))
            if mask is None: raise ProtocolError(f"unknown mask: {sample_id}")
            # `core_start/core_end` are phoneme-local prediction coordinates.
            # The mask manifest is bound to the natural/global mel coordinates.
            start = int(used["global_start_frame"]); end = int(used["global_end_frame"])
            if start != int(mask["natural_core_start_frame"]) or end != int(mask["natural_core_end_frame"]): raise ProtocolError(f"mask core identity mismatch: {sample_id}")
            columns.update(range(start, end))
        all_sets.append(columns)
    if not all_sets or any(columns != all_sets[0] for columns in all_sets): raise ProtocolError(f"C K differs across seeds/conditions: {sample_id}")
    return sorted(all_sets[0])


def _load_matrix_row(row: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    v, a, producer = load_worker(row); return matrix_from_embeddings(v, a), v, a


def _stats(values: list[dict[str, Any]], field: str) -> dict[str, Any]:
    grouped = {str(row["source_group"]): float(row[field]) for row in values}
    labels = sorted(grouped)
    if len(labels) != 8: raise ProtocolError("expected 8 group values")
    means = np.asarray([grouped[label] for label in labels], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(config.BOOTSTRAP_SEED))
    indices = rng.integers(0, 8, size=(config.BOOTSTRAP_DRAWS, 8))
    estimates = means[indices].mean(axis=1)
    return {"mean": float(means.mean()), "ci99": [float(np.quantile(estimates, .005, method="linear")), float(np.quantile(estimates, .995, method="linear"))], "group_labels": labels, "group_means": grouped, "group_positive_count": int(np.sum(means > 0)), "seed": config.BOOTSTRAP_SEED, "draws": config.BOOTSTRAP_DRAWS, "indices": indices.tolist()}


def audit_c(records: list[dict[str, str]]) -> dict[str, Any]:
    pmap = _score_map(_rows(config.P / "control_scores/manifest.json"), arms={"N"}, audio={"N"})
    qmap = _score_map(_rows(config.Q / "candidate_scores/manifest.json"), arms={"CORRECT", "WRONG", "SHUFFLE"}, audio={"N"})
    drivers = read_json(config.D_FILES["drivers"])["drivers"]; masks = {str(row["mask_sha256"]): row for row in read_json(config.D_FILES["mask_manifest"])["masks"]}; supports = __import__("scripts.experiments.wav2lip_parallel_evidence_contract.common", fromlist=["supports"]).supports()
    per = []
    for rec in records:
        sid = rec["sample_id"]; k = _k_for(sid, drivers, masks); exp = {r: float(len(set(supports[r].tolist()).intersection(k)) / len(supports[r])) for r in supports}; matrices = {}; visuals = {}
        n, vn, _ = _load_matrix_row(pmap[(sid, "N", "N")]); matrices["N"] = n; visuals["N"] = vn
        for arm in ("CORRECT", "WRONG", "SHUFFLE"):
            m, v, _ = _load_matrix_row(qmap[(sid, arm, "N")]); matrices[arm] = m; visuals[arm] = v
        k0 = int(np.argmin(np.mean(n[list(range(30, 58))], axis=0))); row = {"sample_id": sid, "source_group": rec["source_group"], "K": k, "K_count": len(k), "exposure": exp, "k0": k0, "matrix_max_abs": {}, "zero_exposure_rows": [], "zero_exposure_violations": []}
        d = {}
        for arm in ("CORRECT", "WRONG", "SHUFFLE"):
            delta = n[:, k0] - matrices[arm][:, k0]; d[arm] = delta[list(range(30, 58))].tolist(); row[arm] = {"d_mean": float(np.mean(d[arm])), "d_rows": d[arm]}
            row["matrix_max_abs"][arm] = float(np.max(np.abs(matrices[arm] - matrices["N"])))
        for r in range(30, 58):
            if exp[r] == 0.0:
                row["zero_exposure_rows"].append(r)
                visual_delta = float(np.max(np.abs(visuals["N"][r] - visuals["CORRECT"][r])))
                matrix_delta = float(np.max(np.abs(matrices["N"][r] - matrices["CORRECT"][r])))
                if visual_delta > 1e-4 or matrix_delta > 1e-4:
                    row["zero_exposure_violations"].append({"row": r, "visual_max_abs": visual_delta, "matrix_max_abs": matrix_delta})
        row["exposure_min"] = min(exp.values()); row["exposure_max"] = max(exp.values()); per.append(row)
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in per: groups.setdefault(row["source_group"], []).append(row)
    stats = {}
    for arm in ("CORRECT", "WRONG", "SHUFFLE"):
        group_rows = []; identifiable = True
        for group, items in sorted(groups.items()):
            beta_num = 0.0; beta_den = 0.0; local_num = 0.0; local_den = 0.0
            for item in items:
                e = np.asarray([item["exposure"][str(r)] if str(r) in item["exposure"] else item["exposure"][r] for r in range(30, 58)], dtype=np.float64); d = np.asarray(item[arm]["d_rows"], dtype=np.float64)
                ec = e - np.mean(e); dc = d - np.mean(d); beta_num += float(np.mean(ec * dc)); beta_den += float(np.mean(ec * ec)); local_num += float(np.sum(e * d)); local_den += float(np.sum(e))
            if beta_den <= 1e-12 or local_den <= 1e-12: identifiable = False
            group_rows.append({"source_group": group, "beta": beta_num / beta_den if beta_den > 1e-12 else None, "local_gain": local_num / local_den if local_den > 1e-12 else None})
        if not identifiable: stats[arm] = {"identifiable": False, "groups": group_rows}
        else: stats[arm] = {"identifiable": True, "groups": group_rows, "beta": _stats([{"source_group": row["source_group"], "value": row["beta"]} for row in group_rows], "value"), "local_gain": _stats([{"source_group": row["source_group"], "value": row["local_gain"]} for row in group_rows], "value")}
    return {"records": per, "stats": stats, "K_recomputed": True, "zero_exposure_checked": True, "zero_exposure_violation_count": int(sum(len(row["zero_exposure_violations"]) for row in per)), "parent_full_U_delta_A": float(np.mean([row["CORRECT"]["d_mean"] for row in per]))}


def run_audit() -> dict[str, Any]:
    records = _records(); a = audit_a(records); b = audit_b(records); c = audit_c(records)
    contract = {"A": ["old validator reused producer statistics; independent matrix/stat recomputation performed"], "B": ["legacy helper uses offset=15-j for delayed audio; matched domain recomputed with 10-j", "old 13/16 gate is not the matched-domain result"], "C": ["old validator reused producer supports/row_supports", "old joint-positive rule was marginal counts; independent same-group conjunction used", "zero-exposure rows checked one by one"], "final_binding": ["old B final does not bind independent validation"]}
    return {"schema_version": 1, "record_count": len(records), "group_count": 8, "A": a, "B": b, "C": c, "contract_violations": contract, "replacement_confirmed": False, "waveform_head_authorized": False, "generalization_established": False, "historical_shift_gate_repaired": False, "f46_candidate_created": False}
