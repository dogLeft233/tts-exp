#!/usr/bin/env python3
"""Pure measurements for the TTS visual-timing experiment.

This module intentionally has no OpenCV, MediaPipe, MFA, Torch, or model
imports.  It is the small, deterministic seam used by the runner and by the
independent checker.  Inputs are ordinary numpy arrays and JSON-compatible
records, which makes the scientific rules testable without a GPU.
"""

from __future__ import annotations

import functools
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


REQUIRED_LANDMARKS = (13, 14, 61, 291, 33, 263)
FPS = 25.0
EVENT_PENALTY_S = 0.240
EVENT_MATCH_CAP_S = 0.240
MIN_VALID_FRACTION = 0.95
MAX_INVALID_RUN = 5
LOW_MOTION_RANGE = 0.015
WORD_SUPPORT_COVERAGE = 0.80
SUPPORT_EPSILON_S = 1e-6
V2_WARP_KNOT_FRACTIONS = (0.0, 0.10, 0.20, 0.80, 0.90, 1.0)
V2_WARP_KNOT_VALUES = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)


def _as_finite_array(value: Any, *, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if ndim is not None and array.ndim != ndim:
        raise ValueError("array has an unexpected rank")
    return array


def normalized_to_pixels(landmarks: Any, width: float, height: float) -> np.ndarray:
    """Convert MediaPipe normalized xy coordinates to pixel coordinates."""

    points = _as_finite_array(landmarks)
    if points.ndim != 3 or points.shape[1:] != (478, 2):
        raise ValueError("landmarks must have shape (frames, 478, 2)")
    if not math.isfinite(float(width)) or not math.isfinite(float(height)) or width <= 0 or height <= 0:
        raise ValueError("frame dimensions must be positive")
    result = points.copy()
    result[:, :, 0] *= float(width)
    result[:, :, 1] *= float(height)
    return result


def _invalid_run(values: np.ndarray) -> int:
    longest = current = 0
    for item in np.asarray(values, dtype=bool):
        if not bool(item):
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return int(longest)


def _median_three(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Median smooth only complete valid 3-frame neighborhoods."""

    source = np.asarray(values, dtype=np.float64).copy()
    output = source.copy()
    for index in range(1, len(output) - 1):
        if valid[index - 1] and valid[index] and valid[index + 1]:
            output[index] = float(np.median(source[index - 1 : index + 2]))
    return output


def _event_candidates(aperture: np.ndarray, valid: np.ndarray, timestamps: np.ndarray, *, p10: Optional[float] = None, p90: Optional[float] = None) -> List[Dict[str, Any]]:
    finite = valid & np.isfinite(aperture) & np.isfinite(timestamps)
    if not finite.any():
        return []
    p10 = float(np.percentile(aperture[finite], 10)) if p10 is None else float(p10)
    p90 = float(np.percentile(aperture[finite], 90)) if p90 is None else float(p90)
    spread = p90 - p10
    if spread < LOW_MOTION_RANGE:
        return []
    z = np.full(len(aperture), np.nan, dtype=np.float64)
    z[finite] = (aperture[finite] - p10) / spread
    candidates: List[Dict[str, Any]] = []
    index = 0
    while index < len(z):
        if not finite[index] or not (z[index] <= 0.25):
            index += 1
            continue
        start = index
        while index + 1 < len(z) and finite[index + 1] and z[index + 1] <= 0.25:
            index += 1
        end = index
        length = end - start + 1
        if 1 <= length <= 8:
            left = max(0, start - int(round(0.200 * FPS)))
            right = min(len(z), end + int(round(0.200 * FPS)) + 1)
            # A candidate is a real valley only when its complete neighborhood
            # is valid and contains an open-mouth frame on each side.
            neighborhood = finite[left:right]
            if neighborhood.all():
                left_open = np.any(z[left:start] >= 0.60)
                right_open = np.any(z[end + 1 : right] >= 0.60)
                if left_open and right_open:
                    values = aperture[start : end + 1]
                    minimum = float(np.min(values))
                    minima = [i for i in range(start, end + 1) if aperture[i] == minimum]
                    # Keep the plateau position as a real-valued frame coordinate.
                    # The display index remains nearest-half-up, while timing uses
                    # the arithmetic midpoint of the first/last minimum PTS.
                    position_frames = 0.5 * (minima[0] + minima[-1])
                    chosen = int(math.floor(position_frames + 0.5))
                    candidates.append(
                        {
                            "index": chosen,
                            "event_position_frames": float(position_frames),
                            "time_s": float(0.5 * (timestamps[minima[0]] + timestamps[minima[-1]])),
                            "aperture": float(aperture[chosen]),
                            "start_index": int(start),
                            "end_index": int(end),
                            "z": float(z[chosen]),
                        }
                    )
        index += 1
    # Candidates closer than 200 ms are one event; retain the smaller valley.
    selected: List[Dict[str, Any]] = []
    for candidate in candidates:
        if selected and candidate["time_s"] - selected[-1]["time_s"] < 0.200:
            previous = selected[-1]
            if (candidate["aperture"], candidate["time_s"]) < (previous["aperture"], previous["time_s"]):
                selected[-1] = candidate
        else:
            selected.append(candidate)
    return selected


def aperture_events(
    landmarks: Any,
    valid: Any,
    timestamps_s: Any,
    *,
    width: float,
    height: float,
    coordinate_system: str = "normalized",
) -> Dict[str, Any]:
    """Extract normalized lip-opening low-valley events from pure video data."""

    raw = _as_finite_array(landmarks)
    if raw.ndim != 3 or raw.shape[1:] != (478, 2):
        raise ValueError("landmarks must have shape (frames, 478, 2)")
    valid_array = np.asarray(valid, dtype=bool)
    timestamps = _as_finite_array(timestamps_s, ndim=1)
    if valid_array.shape != (raw.shape[0],) or timestamps.shape != (raw.shape[0],):
        raise ValueError("valid and timestamps must have one value per frame")
    if len(timestamps) > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError("timestamps must be strictly increasing")
    points = raw.copy()
    if coordinate_system == "normalized":
        # Invalid frames may be NaN.  Scaling them is harmless and keeps the
        # original validity mask authoritative.
        points[:, :, 0] *= float(width)
        points[:, :, 1] *= float(height)
    elif coordinate_system != "pixel":
        raise ValueError("coordinate_system must be normalized or pixel")
    required = np.asarray(REQUIRED_LANDMARKS, dtype=np.int64)
    finite_required = np.isfinite(points[:, required, :]).all(axis=(1, 2))
    frame_valid = valid_array & finite_required
    if len(frame_valid) == 0:
        return {"status": "DETECTION_FAILURE", "reason": "NO_FRAMES", "valid_fraction": 0.0, "events": []}
    valid_fraction = float(frame_valid.mean())
    longest_invalid = _invalid_run(frame_valid)
    aperture = np.full(len(points), np.nan, dtype=np.float64)
    if frame_valid.any():
        upper = points[frame_valid, 13]
        lower = points[frame_valid, 14]
        left_eye = points[frame_valid, 33]
        right_eye = points[frame_valid, 263]
        denominator = np.linalg.norm(right_eye - left_eye, axis=1)
        usable = np.isfinite(denominator) & (denominator > 1e-8)
        frame_indices = np.flatnonzero(frame_valid)
        aperture[frame_indices[usable]] = np.linalg.norm(lower[usable] - upper[usable], axis=1) / denominator[usable]
        frame_valid[frame_indices[~usable]] = False
    result: Dict[str, Any] = {
        "status": "MEASURABLE",
        "reason": None,
        "valid_fraction": float(frame_valid.mean()),
        "longest_invalid_run": int(_invalid_run(frame_valid)),
        "frame_count": int(len(points)),
        "coordinate_system": coordinate_system,
        "required_landmarks": list(REQUIRED_LANDMARKS),
        "aperture": [None if not math.isfinite(float(value)) else float(value) for value in aperture],
        "timestamps_s": [float(value) for value in timestamps],
        "events": [],
    }
    if result["valid_fraction"] < MIN_VALID_FRACTION or result["longest_invalid_run"] > MAX_INVALID_RUN:
        result.update(status="DETECTION_FAILURE", reason="VALIDITY_GATE")
        return result
    finite = frame_valid & np.isfinite(aperture)
    if not finite.any():
        result.update(status="DETECTION_FAILURE", reason="NO_VALID_APERTURE")
        return result
    p10 = float(np.percentile(aperture[finite], 10))
    p90 = float(np.percentile(aperture[finite], 90))
    spread = p90 - p10
    result.update(p10=p10, p90=p90, dynamic_range=float(spread))
    if spread < LOW_MOTION_RANGE:
        result.update(status="LOW_MOTION", reason="P90_P10_BELOW_0_015")
        return result
    smooth = _median_three(aperture, finite)
    events = _event_candidates(smooth, finite, timestamps, p10=p10, p90=p90)
    z = np.full(len(smooth), np.nan, dtype=np.float64)
    z[finite] = (smooth[finite] - p10) / spread
    result["smoothed_aperture"] = [None if not math.isfinite(float(value)) else float(value) for value in smooth]
    result["z"] = [None if not math.isfinite(float(value)) else float(value) for value in z]
    result["events"] = events
    return result


def _event_time(item: Any) -> float:
    if isinstance(item, Mapping):
        value = item.get("time_s", item.get("time"))
    else:
        value = item
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("event time must be finite")
    return value


def event_distance(reference: Sequence[Any], prediction: Sequence[Any]) -> Dict[str, Any]:
    """Ordered one-to-one event edit distance in seconds.

    Match, delete-reference, delete-prediction tie order is explicit in the
    backtrace.  This retains the fixed reference denominator when prediction
    contains no events.
    """

    ref = [_event_time(item) for item in reference]
    pred = [_event_time(item) for item in prediction]
    if not ref and not pred:
        raise ValueError("two empty event sets are not scorable")
    n, m = len(ref), len(pred)
    dp = np.zeros((n + 1, m + 1), dtype=np.float64)
    choices: List[List[str]] = [["" for _ in range(m + 1)] for _ in range(n + 1)]
    for i in range(1, n + 1):
        dp[i, 0] = dp[i - 1, 0] + EVENT_PENALTY_S
        choices[i][0] = "delete_reference"
    for j in range(1, m + 1):
        dp[0, j] = dp[0, j - 1] + EVENT_PENALTY_S
        choices[0][j] = "delete_prediction"
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            match = dp[i - 1, j - 1] + 2.0 * min(abs(ref[i - 1] - pred[j - 1]), EVENT_MATCH_CAP_S)
            delete_ref = dp[i - 1, j] + EVENT_PENALTY_S
            delete_pred = dp[i, j - 1] + EVENT_PENALTY_S
            values = (match, delete_ref, delete_pred)
            best = min(values)
            # The order implements the protocol's tie rule.
            if math.isclose(match, best, rel_tol=0.0, abs_tol=1e-12):
                choice = "match"
            elif math.isclose(delete_ref, best, rel_tol=0.0, abs_tol=1e-12):
                choice = "delete_reference"
            else:
                choice = "delete_prediction"
            dp[i, j] = best
            choices[i][j] = choice
    matches: List[Dict[str, Any]] = []
    missing_reference = 0
    extra_prediction = 0
    i, j = n, m
    while i or j:
        choice = choices[i][j]
        if choice == "match":
            matches.append({"reference_index": i - 1, "prediction_index": j - 1, "error_s": abs(ref[i - 1] - pred[j - 1])})
            i -= 1
            j -= 1
        elif choice == "delete_reference":
            missing_reference += 1
            i -= 1
        elif choice == "delete_prediction":
            extra_prediction += 1
            j -= 1
        else:  # boundary choices
            if i:
                missing_reference += 1
                i -= 1
            elif j:
                extra_prediction += 1
                j -= 1
    matches.reverse()
    denominator = n + m
    return {
        "distance_s": float(dp[n, m] / denominator),
        "distance_ms": float(1000.0 * dp[n, m] / denominator),
        "reference_count": n,
        "prediction_count": m,
        "matched_count": len(matches),
        "missing_reference_count": int(missing_reference),
        "extra_prediction_count": int(extra_prediction),
        "missing_reference_rate": float(missing_reference / n) if n else None,
        "extra_prediction_rate": float(extra_prediction / m) if m else None,
        "matches": matches,
    }


def local_warp_indices(
    frame_count: int,
    fps: float,
    delta_s: float,
    *,
    protocol: str = "tts_visual_timing_v1",
) -> Tuple[List[int], Dict[str, Any]]:
    """Return a registered monotone piecewise local frame map.

    ``v1`` keeps its historical knots for reproducibility.  ``v2`` is the
    single fixed repair candidate from the repair spec; callers must select it
    explicitly instead of mutating module globals.
    """

    if frame_count < 2 or fps <= 0 or not math.isfinite(float(delta_s)):
        raise ValueError("invalid frame map arguments")
    length = float(frame_count) / float(fps)
    if protocol == "tts_visual_timing_v1":
        fractions = (0.0, 0.20, 0.35, 0.65, 0.80, 1.0)
        values = (0.0, 0.0, 1.0, 1.0, 0.0, 0.0)
    elif protocol == "tts_visual_timing_v2":
        fractions = V2_WARP_KNOT_FRACTIONS
        values = V2_WARP_KNOT_VALUES
    else:
        raise ValueError("unknown timing protocol: %s" % protocol)
    knots_x = np.asarray(tuple(float(value) * length for value in fractions), dtype=np.float64)
    knots_y = np.asarray(values, dtype=np.float64)
    output_times = np.arange(frame_count, dtype=np.float64) / float(fps)
    w = np.interp(output_times, knots_x, knots_y)
    source_positions = output_times + float(delta_s) * w
    source_positions[0] = 0.0
    source_positions[-1] = float(frame_count - 1) / float(fps)
    selected = np.floor(source_positions * float(fps) + 0.5).astype(np.int64)
    if int(selected[0]) != 0 or int(selected[-1]) != frame_count - 1 or np.any(selected < 0) or np.any(selected >= frame_count) or np.any(np.diff(selected) < 0):
        raise ValueError("local frame map is not legal")
    displacement = selected - np.arange(frame_count, dtype=np.int64)
    return [int(value) for value in selected], {
        "fps": float(fps),
        "frame_count": int(frame_count),
        "duration_s": length,
        "delta_s": float(delta_s),
        "protocol": protocol,
        "positive_reads_future": True,
        "knots": [[float(x), float(y)] for x, y in zip(knots_x, knots_y)],
        "actual_displacement_frames": [int(value) for value in displacement],
    }


def _normal(value: Any) -> str:
    return str(value or "").strip().lower()


def _usable_token(token: Mapping[str, Any]) -> bool:
    return bool(token.get("label")) and not bool(token.get("silence")) and not bool(token.get("unknown")) and float(token.get("end_s", 0.0)) > float(token.get("start_s", 0.0))


def _word_groups(tokens: Sequence[Mapping[str, Any]]) -> Optional[List[Tuple[str, List[Mapping[str, Any]]]]]:
    groups: Dict[int, List[Mapping[str, Any]]] = {}
    # Silence/unknown intervals are expected to have no word annotation in
    # MFA output.  Validate the word tier only for usable speech phones.
    # Dropping a usable phone with no word would hide a real alignment error,
    # so that case remains unsupported.
    usable_tokens = [token for token in tokens if _usable_token(token)]
    for token in usable_tokens:
        word_index = token.get("word_index")
        word_label = token.get("word_normalized")
        if word_index is None or not word_label:
            return None
        groups.setdefault(int(word_index), []).append(token)
    ordered = sorted(groups.items(), key=lambda item: item[0])
    return [(_normal(items[0].get("word_normalized", items[0].get("word_label", "")) if items else ""), items) for _, items in ordered]


def continuous_support_blocks(
    segments: Sequence[Mapping[str, Any]],
    *,
    tolerance_s: float = SUPPORT_EPSILON_S,
) -> List[Dict[str, Any]]:
    """Group adjacent phone maps into continuous N/T support blocks.

    A block is continuous only when both clocks share the same endpoint.  The
    individual phone affine segments are retained; the block is metadata for
    support selection and never replaces the piecewise map.
    """

    if tolerance_s < 0 or not math.isfinite(float(tolerance_s)):
        raise ValueError("tolerance_s must be finite and non-negative")
    normalized: List[Dict[str, Any]] = []
    previous_n_end: Optional[float] = None
    previous_t_end: Optional[float] = None
    for index, raw in enumerate(segments):
        n0 = float(raw["n_start_s"]); n1 = float(raw["n_end_s"])
        t0 = float(raw["t_start_s"]); t1 = float(raw["t_end_s"])
        values = (n0, n1, t0, t1)
        if not all(math.isfinite(value) for value in values) or n1 <= n0 or t1 <= t0:
            raise ValueError("invalid phone segment at index %d" % index)
        if previous_n_end is not None and (n0 < previous_n_end - tolerance_s or t0 < previous_t_end - tolerance_s):
            raise ValueError("phone segments are not monotone")
        if previous_n_end is not None:
            if n0 > previous_n_end + tolerance_s or t0 > previous_t_end + tolerance_s:
                # A real gap starts a new block.
                pass
            else:
                # Absorb only floating-point endpoint noise.  The original
                # segment boundaries remain in ``segments`` for mapping.
                if abs(n0 - previous_n_end) <= tolerance_s:
                    n0 = previous_n_end
                if abs(t0 - previous_t_end) <= tolerance_s:
                    t0 = previous_t_end
        item = dict(raw)
        item.update({"n_start_s": n0, "n_end_s": n1, "t_start_s": t0, "t_end_s": t1, "segment_index": index})
        normalized.append(item)
        previous_n_end, previous_t_end = n1, t1
    blocks: List[Dict[str, Any]] = []
    for item in normalized:
        if not blocks:
            blocks.append({"n_start_s": item["n_start_s"], "n_end_s": item["n_end_s"], "t_start_s": item["t_start_s"], "t_end_s": item["t_end_s"], "segment_indices": [int(item["segment_index"])]})
            continue
        block = blocks[-1]
        n_adjacent = abs(float(item["n_start_s"]) - float(block["n_end_s"])) <= tolerance_s
        t_adjacent = abs(float(item["t_start_s"]) - float(block["t_end_s"])) <= tolerance_s
        if n_adjacent and t_adjacent:
            block["n_end_s"] = item["n_end_s"]
            block["t_end_s"] = item["t_end_s"]
            block["segment_indices"].append(int(item["segment_index"]))
        else:
            blocks.append({"n_start_s": item["n_start_s"], "n_end_s": item["n_end_s"], "t_start_s": item["t_start_s"], "t_end_s": item["t_end_s"], "segment_indices": [int(item["segment_index"])]})
    return blocks


def _lcs_pairs(left: Sequence[Mapping[str, Any]], right: Sequence[Mapping[str, Any]]) -> List[Tuple[Mapping[str, Any], Mapping[str, Any]]]:
    labels_left = [_normal(item.get("normalized", item.get("label"))) for item in left]
    labels_right = [_normal(item.get("normalized", item.get("label"))) for item in right]

    @functools.lru_cache(maxsize=None)
    def solve(i: int, j: int) -> Tuple[Tuple[int, int], ...]:
        if i >= len(left) or j >= len(right):
            return ()
        candidates: List[Tuple[Tuple[int, int], ...]] = [solve(i + 1, j), solve(i, j + 1)]
        if labels_left[i] == labels_right[j] and labels_left[i]:
            candidates.append(((i, j),) + solve(i + 1, j + 1))
        max_len = max(len(item) for item in candidates)
        best = [item for item in candidates if len(item) == max_len]
        return min(best, key=lambda item: tuple(item))

    pairs = solve(0, 0)
    return [(left[i], right[j]) for i, j in pairs]


def audio_time_map(n_tokens: Sequence[Mapping[str, Any]], t_tokens: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Construct a strict word-ordinal, phone-LCS piecewise affine map N→T."""

    n_words = _word_groups(n_tokens)
    t_words = _word_groups(t_tokens)
    if n_words is None or t_words is None:
        return {"status": "ALIGNMENT_UNSUPPORTED", "reason": "WORDS_TIER_REQUIRED", "segments": [], "matched_phone_count": 0}
    if len(n_words) != len(t_words) or any(n_words[i][0] != t_words[i][0] for i in range(min(len(n_words), len(t_words)))):
        return {"status": "ALIGNMENT_UNSUPPORTED", "reason": "WORD_ORDINAL_MISMATCH", "segments": [], "matched_phone_count": 0}
    pairs: List[Tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    matched_words = 0
    for (n_label, n_group), (t_label, t_group) in zip(n_words, t_words):
        if n_label != t_label:
            continue
        word_pairs = _lcs_pairs(n_group, t_group)
        if word_pairs:
            matched_words += 1
            pairs.extend(word_pairs)
    segments: List[Dict[str, Any]] = []
    for n_token, t_token in pairs:
        n0, n1 = float(n_token["start_s"]), float(n_token["end_s"])
        t0, t1 = float(t_token["start_s"]), float(t_token["end_s"])
        if n1 <= n0 or t1 <= t0:
            continue
        segments.append({"n_start_s": n0, "n_end_s": n1, "t_start_s": t0, "t_end_s": t1, "slope": (t1 - t0) / (n1 - n0), "phone": _normal(n_token.get("normalized", n_token.get("label")))})
    def coverage(tokens: Sequence[Mapping[str, Any]], side: str) -> float:
        usable = [(float(item["start_s"]), float(item["end_s"])) for item in tokens if _usable_token(item)]
        total = sum(max(0.0, right - left) for left, right in usable)
        covered = sum(max(0.0, float(item[f"{side}_end_s"]) - float(item[f"{side}_start_s"])) for item in segments)
        return float(covered / total) if total > 0 else 0.0
    n_coverage = coverage(n_tokens, "n")
    t_coverage = coverage(t_tokens, "t")
    blocks: List[Dict[str, Any]] = []
    block_error: Optional[str] = None
    if segments:
        try:
            blocks = continuous_support_blocks(segments)
        except ValueError as exc:
            block_error = str(exc)
    status = "COMPLETE" if segments and block_error is None and n_coverage >= WORD_SUPPORT_COVERAGE and t_coverage >= WORD_SUPPORT_COVERAGE else "INSUFFICIENT_COVERAGE"
    reason = None if status == "COMPLETE" else ("INVALID_CONTINUOUS_SUPPORT" if block_error else "MATCHED_SPEECH_COVERAGE_BELOW_0_80")
    return {"status": status, "reason": reason, "segments": segments, "support_blocks": blocks, "matched_phone_count": len(segments), "matched_word_count": matched_words, "n_speech_coverage": n_coverage, "t_speech_coverage": t_coverage}


def map_time(value_s: float, segments: Sequence[Mapping[str, Any]], *, direction: str) -> Optional[float]:
    """Map one time through a piecewise map; unsupported gaps return None."""

    value = float(value_s)
    for segment in segments:
        if direction == "n_to_t":
            left, right = float(segment["n_start_s"]), float(segment["n_end_s"])
            if left <= value < right or (value == right and right == max(float(item["n_end_s"]) for item in segments)):
                ratio = (value - left) / (right - left)
                return float(segment["t_start_s"] + ratio * (float(segment["t_end_s"]) - float(segment["t_start_s"])))
        elif direction == "t_to_n":
            left, right = float(segment["t_start_s"]), float(segment["t_end_s"])
            if left <= value < right or (value == right and right == max(float(item["t_end_s"]) for item in segments)):
                ratio = (value - left) / (right - left)
                return float(segment["n_start_s"] + ratio * (float(segment["n_end_s"]) - float(segment["n_start_s"])))
        else:
            raise ValueError("direction must be n_to_t or t_to_n")
    return None


def _event_position_frames(event: Mapping[str, Any]) -> float:
    value = event.get("event_position_frames", event.get("index"))
    if value is None or not math.isfinite(float(value)):
        raise ValueError("event has no finite frame position")
    return float(value)


def _inverse_frame_time(source_position: float, mapping: Sequence[int], fps: float) -> Optional[float]:
    """Invert an output-frame→source-frame map without nearest-frame fallback."""

    if not mapping or fps <= 0 or not math.isfinite(float(source_position)):
        return None
    exact = [index for index, value in enumerate(mapping) if int(value) == int(math.floor(source_position)) and abs(source_position - math.floor(source_position)) <= 1e-9]
    if exact:
        return float(0.5 * (exact[0] + exact[-1]) / fps)
    left = int(math.floor(source_position)); right = int(math.ceil(source_position))
    if left == right or left < 0 or right >= len(mapping):
        return None
    left_positions = [index for index, value in enumerate(mapping) if int(value) == left]
    right_positions = [index for index, value in enumerate(mapping) if int(value) == right]
    if not left_positions or not right_positions:
        return None
    left_time = 0.5 * (left_positions[0] + left_positions[-1]) / fps
    right_time = 0.5 * (right_positions[0] + right_positions[-1]) / fps
    ratio = float(source_position - left) / float(right - left)
    return float(left_time + ratio * (right_time - left_time))


def _source_position_at_output(event: Mapping[str, Any], mapping: Sequence[int]) -> Optional[float]:
    position = _event_position_frames(event)
    if not mapping:
        return None
    left = int(math.floor(position)); right = int(math.ceil(position))
    if left < 0 or right >= len(mapping):
        return None
    if left == right:
        return float(mapping[left])
    return float(mapping[left]) + (float(position) - left) * float(int(mapping[right]) - int(mapping[left]))


def _filter_events_by_source_interval(
    events: Sequence[Mapping[str, Any]],
    mapping: Sequence[int],
    source_interval: Tuple[float, float],
    *,
    fps: float = FPS,
) -> List[Dict[str, Any]]:
    left, right = map(float, source_interval)
    result: List[Dict[str, Any]] = []
    for event in events:
        source_position = _source_position_at_output(event, mapping)
        if source_position is not None and left <= source_position / float(fps) < right:
            result.append(dict(event))
    return result


def _expected_control_events(
    reference_events: Sequence[Mapping[str, Any]],
    mapping: Sequence[int],
    fps: float,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    expected: List[Dict[str, Any]] = []
    unsupported: List[int] = []
    for ref_index, event in enumerate(reference_events):
        source_position = _event_position_frames(event)
        expected_time = _inverse_frame_time(source_position, mapping, fps)
        if expected_time is None:
            unsupported.append(ref_index)
            continue
        expected.append({"reference_index": ref_index, "source_position_frames": source_position, "time_s": expected_time})
    return expected, unsupported


def evaluate_calibration_record(
    reference_events: Sequence[Mapping[str, Any]],
    controls: Mapping[str, Mapping[str, Any]],
    *,
    source_interval: Tuple[float, float],
    fps: float = FPS,
    recovery_tolerance_s: float = 0.040,
    repeat_limit_ms: float = 20.0,
    error_limit_ms: float = 40.0,
    recovery_min: float = 0.80,
) -> Dict[str, Any]:
    """Independently compute one v2 calibration record.

    ``controls`` maps an arm name to ``events``, ``mapping`` (output frame to
    source frame), ``status`` and ``delta_s``.  Every numerical value used by
    the five gates is returned so the runner and checker can compare evidence
    rather than trusting a saved boolean.
    """

    references = [dict(item) for item in reference_events]
    if len(references) < 2:
        return {"status": "INSUFFICIENT_SUPPORT", "reason": "REFERENCE_EVENT_COUNT_LT_2", "reference_events": references, "arms": {}, "checks": {}}
    arms: Dict[str, Any] = {}
    for arm, raw in controls.items():
        item = dict(raw)
        mapping = [int(value) for value in item.get("mapping", [])]
        observed = [dict(value) for value in item.get("events", [])]
        source_supported = _filter_events_by_source_interval(observed, mapping, source_interval, fps=fps) if mapping else []
        expected, unsupported = _expected_control_events(references, mapping, fps) if mapping else ([], list(range(len(references))))
        # Keep a reference-index map in the expected records.  The matching
        # distance is ordered and deterministic, so direction is based on the
        # original R event for each successfully recovered reference.
        recovery: Dict[str, Any]
        if not expected:
            recovery = {"recovery_rate": 0.0, "matched_count": 0, "median_abs_error_ms": None, "median_signed_shift_ms": None, "distance_ms": None, "matches": []}
        else:
            distance = event_distance(expected, source_supported) if source_supported or expected else None
            matches = distance.get("matches", []) if distance else []
            good = [match for match in matches if float(match["error_s"]) <= recovery_tolerance_s]
            signed: List[float] = []
            for match in good:
                expected_item = expected[int(match["reference_index"])]
                reference_index = int(expected_item["reference_index"])
                original_time = float(references[reference_index]["time_s"])
                observed_time = float(source_supported[int(match["prediction_index"])]["time_s"])
                signed.append(observed_time - original_time)
            recovery = {
                "recovery_rate": float(len(good) / len(references)),
                "matched_count": len(good),
                "median_abs_error_ms": float(np.median([float(match["error_s"]) for match in good]) * 1000.0) if good else None,
                "median_signed_shift_ms": float(np.median(signed) * 1000.0) if signed else None,
                "distance_ms": float(distance["distance_ms"]) if distance else None,
                "matches": [dict(match, reference_index=int(expected[int(match["reference_index"])]["reference_index"])) for match in good],
            }
        real_distance = event_distance(references, source_supported) if source_supported or references else None
        expected_delta = item.get("delta_s")
        expected_sign = None
        observed_shift = recovery.get("median_signed_shift_ms")
        if expected_delta is not None and abs(float(expected_delta)) > 1e-12:
            expected_sign = -1 if float(expected_delta) > 0 else 1
        arm_result = {
            "status": item.get("status"),
            "delta_s": None if expected_delta is None else float(expected_delta),
            "mapping": mapping,
            "source_interval_s": [float(source_interval[0]), float(source_interval[1])],
            "observed_events": observed,
            "supported_observed_events": source_supported,
            "expected_events": expected,
            "unsupported_reference_indices": unsupported,
            "recovery": recovery,
            "distance_to_real_ms": float(real_distance["distance_ms"]) if real_distance else None,
            "observed_direction_sign": expected_sign,
            "direction_value_ms": observed_shift,
            "direction_ok": bool(expected_sign is None or (observed_shift is not None and observed_shift * expected_sign > 0)),
        }
        arms[str(arm)] = arm_result
    repeat = arms.get("REPEAT", {})
    warp_names = ("LOCAL_+80", "LOCAL_-80", "LOCAL_+160", "LOCAL_-160")
    repeat_distance = repeat.get("distance_to_real_ms")
    repeat_ok = repeat_distance is not None and float(repeat_distance) <= float(repeat_limit_ms)
    warp_ok = all(
        arms.get(name, {}).get("recovery", {}).get("recovery_rate") is not None
        and float(arms[name]["recovery"]["recovery_rate"]) >= float(recovery_min)
        and arms[name]["recovery"].get("median_abs_error_ms") is not None
        and float(arms[name]["recovery"]["median_abs_error_ms"]) <= float(error_limit_ms)
        for name in warp_names
    )
    sign_ok = all(bool(arms.get(name, {}).get("direction_ok")) for name in warp_names)
    order_ok = all(
        arms.get(large, {}).get("distance_to_real_ms") is not None
        and arms.get(small, {}).get("distance_to_real_ms") is not None
        and float(arms[large]["distance_to_real_ms"]) > float(arms[small]["distance_to_real_ms"]) > float(repeat_distance)
        for large, small in (("LOCAL_+160", "LOCAL_+80"), ("LOCAL_-160", "LOCAL_-80"))
    )
    frozen = arms.get("FROZEN", {})
    frozen_status = str(frozen.get("status"))
    frozen_ok = frozen_status == "LOW_MOTION" and not frozen.get("observed_events")
    checks = {
        "repeat_e_le_20ms": bool(repeat_ok),
        "warp_recovery_and_error": bool(warp_ok),
        "sign_correct": bool(sign_ok),
        "monotone_error": bool(order_ok),
        "frozen_not_perfect": bool(frozen_ok),
    }
    passed = bool(len(arms) >= 6 and all(checks.values()))
    return {
        "status": "PASS" if passed else "FAIL",
        "reason": None if passed else "CALIBRATION_GATE_FAILED",
        "reference_events": references,
        "source_interval_s": [float(source_interval[0]), float(source_interval[1])],
        "arms": arms,
        "checks": checks,
        "passed": passed,
    }


def group_summary(rows: Iterable[Mapping[str, Any]], value_key: str) -> Dict[str, Any]:
    grouped: Dict[str, List[float]] = {}
    for row in rows:
        value = row.get(value_key)
        if value is None or not math.isfinite(float(value)):
            continue
        grouped.setdefault(str(row.get("source_group", row.get("sample_id", "unknown"))), []).append(float(value))
    values = {key: float(np.mean(item)) for key, item in sorted(grouped.items()) if item}
    return {"group_count": len(values), "group_values": values, "mean": float(np.mean(list(values.values()))) if values else None, "positive_group_count": int(sum(value > 0 for value in values.values()))}


def bootstrap_mean(values: Mapping[str, float], *, draws: int = 20_000, seed: int = 20260920, alpha: float = 0.0166666666666667) -> Dict[str, Any]:
    labels = sorted(values)
    if not labels:
        return {"status": "NOT_ESTIMABLE", "group_count": 0, "mean": None, "ci": [None, None], "draws": draws, "seed": seed}
    data = np.asarray([float(values[label]) for label in labels], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(int(seed)))
    indices = rng.integers(0, len(data), size=(int(draws), len(data)), dtype=np.int64)
    means = data[indices].mean(axis=1)
    # Keep the exact resampling table in the result.  This is deliberately
    # redundant with ``seed``: a later checker can recompute the CI without
    # depending on a future NumPy implementation of PCG64.  Twelve groups ×
    # 20k draws is small compared with the media artifacts and makes the
    # statistical denominator auditable.
    return {"status": "COMPLETE", "group_count": len(labels), "group_labels": labels, "group_values": {label: float(value) for label, value in zip(labels, data)}, "mean": float(data.mean()), "ci": [float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1.0 - alpha / 2))], "ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))], "draws": int(draws), "seed": int(seed), "rng": "numpy.random.Generator(PCG64)", "draw_indices": indices.tolist()}


def decide(values: Mapping[str, float], *, threshold: float = 0.0, minimum_mean: float = 0.0, min_groups: int = 8, draws: int = 20_000, alpha: float = 0.0166666666666667) -> Dict[str, Any]:
    summary = bootstrap_mean(values, draws=draws, alpha=alpha)
    if summary["group_count"] < int(min_groups):
        summary["science"] = "INSUFFICIENT_SUPPORT"
    elif summary["mean"] is not None and summary["mean"] >= minimum_mean and summary["ci"][0] > threshold:
        summary["science"] = "SUPPORTED"
    else:
        summary["science"] = "NO_CLEAR_SUPPORT"
    return summary
