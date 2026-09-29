"""Independent PCM, timing, content, and blind-listening contracts."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _speech(token: Mapping[str, Any]) -> bool:
    return bool(token.get("speech", not token.get("silence", False)))


def pcm_contract(source: np.ndarray, candidate: np.ndarray, protected: np.ndarray, *, max_residual_ratio: float = 0.01, min_snr_db: float = 20.0, max_rms_change_db: float = 1.0) -> dict[str, Any]:
    source = np.asarray(source, dtype=np.int16).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.int16).reshape(-1)
    protected = np.asarray(protected, dtype=bool).reshape(-1)
    if source.shape != candidate.shape or source.shape != protected.shape:
        raise ValueError("PCM contract arrays have different shapes")
    editable = ~protected
    source_f = source.astype(np.float64)
    candidate_f = candidate.astype(np.float64)
    residual = candidate_f - source_f
    residual_energy = float(np.sum(residual[editable] ** 2)) if np.any(editable) else 0.0
    signal_energy = float(np.sum(source_f[editable] ** 2)) if np.any(editable) else 0.0
    ratio = residual_energy / max(signal_energy, 1e-8)
    no_editable_support = not editable.any()
    identity = bool(residual_energy == 0.0 and not no_editable_support)
    snr = None if identity or no_editable_support else 10.0 * math.log10(max(signal_energy, 1e-8) / max(residual_energy, 1e-8))
    rms_source = float(np.sqrt(np.mean(source_f[editable] ** 2))) if editable.any() else 0.0
    rms_candidate = float(np.sqrt(np.mean(candidate_f[editable] ** 2))) if editable.any() else 0.0
    rms_change = 20.0 * math.log10(max(rms_candidate, 1e-12) / max(rms_source, 1e-12)) if editable.any() else 0.0
    source_saturated = np.abs(source.astype(np.int32)) >= 32767
    candidate_saturated = np.abs(candidate.astype(np.int32)) >= 32767
    new_saturation = int(np.count_nonzero(candidate_saturated & ~source_saturated))
    pass_snr = identity or (not no_editable_support and snr is not None and snr >= min_snr_db)
    return {"same_length": True, "protected_pcm_equal": bool(np.array_equal(source[protected], candidate[protected])), "residual_energy_ratio": float(ratio), "snr_db": snr, "identity": identity, "reason": "IDENTITY" if identity else ("NO_EDITABLE_SUPPORT" if no_editable_support else None), "rms_change_db": float(rms_change), "source_saturated_samples": int(source_saturated.sum()), "candidate_saturated_samples": int(candidate_saturated.sum()), "new_saturated_samples": new_saturation, "pass": bool(np.array_equal(source[protected], candidate[protected]) and not no_editable_support and ratio <= max_residual_ratio and pass_snr and abs(rms_change) <= max_rms_change_db and new_saturation == 0)}


def t0_contract(source: np.ndarray, candidate: np.ndarray, protected: np.ndarray, **kwargs: Any) -> dict[str, Any]:
    return pcm_contract(source, candidate, protected, **kwargs)


def pause_spans(tokens: Sequence[Mapping[str, Any]], *, min_ms: float = 50.0) -> list[tuple[float, float]]:
    return [(float(token["start_s"]), float(token["end_s"])) for token in tokens if not _speech(token) and (float(token["end_s"]) - float(token["start_s"])) * 1000.0 >= float(min_ms)]


def _iou(left: tuple[float, float], right: tuple[float, float]) -> float:
    overlap = max(0.0, min(left[1], right[1]) - max(left[0], right[0]))
    union = max(left[1], right[1]) - min(left[0], right[0])
    return overlap / union if union > 0 else 0.0


def _ordered_matches(source: Sequence[Mapping[str, Any]], target: Sequence[Mapping[str, Any]]) -> tuple[list[tuple[int, int]], bool]:
    """Maximum-cardinality monotonic label matching; ties are marked ambiguous."""
    n, m = len(source), len(target)
    score = np.zeros((n + 1, m + 1), dtype=np.int32)
    paths: list[list[list[tuple[tuple[int, int], ...]]]] = [[[] for _ in range(m + 1)] for _ in range(n + 1)]
    paths[0][0] = [()]
    for i in range(n + 1):
        for j in range(m + 1):
            if i == 0 and j == 0:
                continue
            candidates: list[tuple[int, tuple[tuple[int, int], ...]]] = []
            if i:
                candidates.extend((score[i - 1, j], path) for path in paths[i - 1][j])
            if j:
                candidates.extend((score[i, j - 1], path) for path in paths[i][j - 1])
            if i and j and str(source[i - 1].get("label", "")) == str(target[j - 1].get("label", "")):
                candidates.extend((score[i - 1, j - 1] + 1, path + ((i - 1, j - 1),)) for path in paths[i - 1][j - 1])
            best = max((value for value, _ in candidates), default=0)
            score[i, j] = best
            unique = {path for value, path in candidates if value == best}
            paths[i][j] = sorted(unique)[:2]
    final = paths[n][m]
    return list(final[0]) if final else [], len(final) > 1


def monotonic_phone_match(source_tokens: Sequence[Mapping[str, Any]], target_tokens: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source = [token for token in source_tokens if _speech(token)]
    target = [token for token in target_tokens if _speech(token)]
    pairs, ambiguous = _ordered_matches(source, target)
    rows = [{"source_index": int(i), "target_index": int(j), "label": str(source[i].get("label", "")), "start_error_ms": abs(float(source[i]["start_s"]) - float(target[j]["start_s"])) * 1000.0, "end_error_ms": abs(float(source[i]["end_s"]) - float(target[j]["end_s"])) * 1000.0} for i, j in pairs]
    matched = len(rows)
    # A matched-only score hides insertions (e.g. [a,b] -> [a,x,b]).  Compute
    # the standard Levenshtein distance on the ordered phone labels and report
    # S/D/I separately while retaining the boundary matches above.
    source_labels = [str(token.get("label", "")) for token in source]
    target_labels = [str(token.get("label", "")) for token in target]
    dp = np.zeros((len(source_labels) + 1, len(target_labels) + 1), dtype=np.int32)
    op: list[list[str]] = [["" for _ in range(len(target_labels) + 1)] for _ in range(len(source_labels) + 1)]
    for i in range(len(source_labels) + 1):
        dp[i, 0] = i
        if i:
            op[i][0] = "D"
    for j in range(len(target_labels) + 1):
        dp[0, j] = j
        if j:
            op[0][j] = "I"
    for i in range(1, len(source_labels) + 1):
        for j in range(1, len(target_labels) + 1):
            candidates = [(dp[i - 1, j] + 1, "D"), (dp[i, j - 1] + 1, "I"), (dp[i - 1, j - 1] + int(source_labels[i - 1] != target_labels[j - 1]), "S" if source_labels[i - 1] != target_labels[j - 1] else "M")]
            dp[i, j], op[i][j] = min(candidates, key=lambda item: (item[0], {"M": 0, "S": 1, "D": 2, "I": 3}[item[1]]))
    substitutions = deletions = insertions = 0
    i, j = len(source_labels), len(target_labels)
    while i or j:
        action = op[i][j]
        if action == "M":
            i -= 1; j -= 1
        elif action == "S":
            substitutions += 1; i -= 1; j -= 1
        elif action == "D":
            deletions += 1; i -= 1
        elif action == "I":
            insertions += 1; j -= 1
        else:
            break
    distance = substitutions + deletions + insertions
    edit_rate = distance / max(len(source_labels), 1)
    return {"matches": rows, "source_count": len(source), "target_count": len(target), "matched_count": matched, "substitutions": substitutions, "deletions": deletions, "insertions": insertions, "levenshtein_distance": distance, "edit_rate": float(edit_rate), "ambiguous": bool(ambiguous)}


def monotonic_pause_match(source: Sequence[tuple[float, float]], target: Sequence[tuple[float, float]], *, iou_min: float = 0.5, edge_error_ms: float = 20.0) -> list[dict[str, Any]]:
    # Dynamic programming maximizes valid, order-preserving matches.  A
    # crossing match is never allowed, even when it has a higher IoU.
    n, m = len(source), len(target)
    dp = np.zeros((n + 1, m + 1), dtype=np.int32)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            valid = _iou(source[i - 1], target[j - 1]) >= iou_min and abs(source[i - 1][0] - target[j - 1][0]) * 1000 <= edge_error_ms and abs(source[i - 1][1] - target[j - 1][1]) * 1000 <= edge_error_ms
            dp[i, j] = max(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1] + int(valid))
    matches: list[dict[str, Any]] = []
    i, j = n, m
    while i and j:
        valid = _iou(source[i - 1], target[j - 1]) >= iou_min and abs(source[i - 1][0] - target[j - 1][0]) * 1000 <= edge_error_ms and abs(source[i - 1][1] - target[j - 1][1]) * 1000 <= edge_error_ms
        if valid and dp[i, j] == dp[i - 1, j - 1] + 1:
            matches.append({"source_index": i - 1, "target_index": j - 1, "iou": _iou(source[i - 1], target[j - 1]), "start_error_ms": abs(source[i - 1][0] - target[j - 1][0]) * 1000, "end_error_ms": abs(source[i - 1][1] - target[j - 1][1]) * 1000, "pass": True})
            i -= 1
            j -= 1
        elif dp[i - 1, j] >= dp[i, j - 1]:
            i -= 1
        else:
            j -= 1
    matches.reverse()
    matched_source = {row["source_index"] for row in matches}
    matched_target = {row["target_index"] for row in matches}
    matches.extend({"source_index": index, "target_index": None, "pass": False, "missing": True} for index in range(n) if index not in matched_source)
    matches.extend({"source_index": None, "target_index": index, "pass": False, "new": True} for index in range(m) if index not in matched_target)
    return matches


def audit_timing(source_tokens: Sequence[Mapping[str, Any]], candidate_tokens: Sequence[Mapping[str, Any]], *, pause_min_ms: float = 50.0, pause_iou_min: float = 0.5, edge_error_ms: float = 20.0, max_edit_rate: float = 0.05, min_coverage: float = 0.90, median_error_ms: float = 20.0, p95_error_ms: float = 40.0) -> dict[str, Any]:
    matching = monotonic_phone_match(source_tokens, candidate_tokens)
    errors = [value for row in matching["matches"] for value in (row.get("start_error_ms"), row.get("end_error_ms")) if value is not None]
    pauses = monotonic_pause_match(pause_spans(source_tokens, min_ms=pause_min_ms), pause_spans(candidate_tokens, min_ms=pause_min_ms), iou_min=pause_iou_min, edge_error_ms=edge_error_ms)
    median = float(np.median(errors)) if errors else None
    p95 = float(np.quantile(errors, 0.95)) if errors else None
    passed = bool(not matching["ambiguous"] and matching["source_count"] > 0 and matching["target_count"] > 0 and matching["edit_rate"] <= max_edit_rate and matching["matched_count"] / max(matching["source_count"], 1) >= min_coverage and median is not None and p95 is not None and median <= median_error_ms and p95 <= p95_error_ms and all(bool(row.get("pass", False)) for row in pauses))
    return {**matching, "boundary_errors_ms": errors, "boundary_median_ms": median, "boundary_p95_ms": p95, "pauses": pauses, "timing_pass": passed}


def construct_local_boundary_control(pcm: np.ndarray, tokens: Sequence[Mapping[str, Any]], *, sample_rate: int = 16_000, shift_ms: float = 40.0) -> dict[str, Any]:
    source = np.asarray(pcm, dtype=np.int16).reshape(-1)
    selected = None
    for index in range(len(tokens) - 1):
        left, right = tokens[index], tokens[index + 1]
        if _speech(left) and _speech(right) and float(left["end_s"]) - float(left["start_s"]) >= 0.16 and float(right["end_s"]) - float(right["start_s"]) >= 0.16:
            selected = (index, index + 1)
            break
    if selected is None:
        raise ValueError("no qualifying adjacent speech spans for local boundary control")
    left_index, right_index = selected
    a = int(round(float(tokens[left_index]["start_s"]) * sample_rate))
    b = int(round(float(tokens[left_index]["end_s"]) * sample_rate))
    c = int(round(float(tokens[right_index]["end_s"]) * sample_rate))
    delta = int(round(float(shift_ms) / 1000.0 * sample_rate))
    left_part = source[a:b]
    right_part = source[b:c]
    if len(left_part) + delta <= 1 or len(right_part) - delta <= 1:
        raise ValueError("local boundary control cannot preserve positive spans")
    def resample(values: np.ndarray, length: int) -> np.ndarray:
        old = np.linspace(0.0, 1.0, num=len(values), endpoint=True)
        new = np.linspace(0.0, 1.0, num=length, endpoint=True)
        return np.rint(np.interp(new, old, values.astype(np.float64))).astype(np.int16)
    edited = source.copy()
    edited[a:c] = np.concatenate([resample(left_part, len(left_part) + delta), resample(right_part, len(right_part) - delta)])
    return {"pcm": edited, "boundary": {"token_left": left_index, "token_right": right_index, "a": a, "b": b, "c": c, "expected_boundary": b + delta, "shift_samples": delta, "shift_ms": float(shift_ms)}, "outside_identity": bool(np.array_equal(source[:a], edited[:a]) and np.array_equal(source[c:], edited[c:]))}


def detect_local_boundary_shift(original_boundary: int, candidate_tokens: Sequence[Mapping[str, Any]], *, left_index: int, right_index: int, sample_rate: int = 16_000, min_shift_ms: float = 20.0, expected_direction: int = 1) -> dict[str, Any]:
    if left_index >= len(candidate_tokens) or right_index >= len(candidate_tokens):
        return {"detected": False, "reason": "MISSING_BOUNDARY"}
    observed = int(round(float(candidate_tokens[left_index]["end_s"]) * sample_rate))
    shift_ms = (observed - int(original_boundary)) * 1000.0 / sample_rate
    return {"observed_boundary": observed, "shift_ms": float(shift_ms), "detected": bool(expected_direction * shift_ms >= float(min_shift_ms))}


def blind_pack(rows: Sequence[Mapping[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    manifest = []
    for index, row in enumerate(rows):
        alias = hashlib.sha256(f"blind|{index}|{row.get('pair_id')}|{row.get('arm')}".encode()).hexdigest()[:12]
        manifest.append({"alias": alias, "pair_id": row.get("pair_id"), "arm": row.get("arm"), "path": row.get("path"), "anonymous": True})
    (target / "ratings_template.csv").write_text("alias,clarity,naturalness,content_error,pause_position,notes\n" + "\n".join(f"{row['alias']},,,," for row in manifest) + "\n", encoding="utf-8")
    return {"status": "HUMAN_NOT_ASSESSED", "manifest": manifest, "ratings": str((target / "ratings_template.csv").resolve())}


__all__ = ["audit_timing", "blind_pack", "construct_local_boundary_control", "detect_local_boundary_shift", "monotonic_pause_match", "monotonic_phone_match", "pause_spans", "pcm_contract", "t0_contract"]
