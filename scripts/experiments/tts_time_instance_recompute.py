"""Independent checks for the TTS temporal-identifiability experiment.

This file intentionally does not import ``tts_time_instance``.  It rebuilds
the distance, lag, phone matching, phase sampling, and bootstrap operations
from the frozen run artifacts so that a producer-side PASS cannot validate
itself merely by reading its own status fields.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SAMPLE_IDS = tuple(range(151, 163))
DIM = 1024
VSHIFT = 15
BOOTSTRAP_SEED = 20260917
BOOTSTRAP_DRAWS = 20_000
PRIMARY_ALPHA = 0.05 / 4.0
PRIMARY_DELTAS = (-5, -4, -3, -2, 2, 3, 4, 5)
SILENCE = {"", "sp", "sil", "silence"}
UNKNOWN = {"spn", "<unk>", "unk", "unknown", "oov", "<oov>"}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_distance(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64)
    b = np.asarray(right, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("nonfinite or shape-mismatched distance input")
    return float(np.sqrt(np.sum((a - b) ** 2, dtype=np.float64)))


def official_matrix(video: np.ndarray, audio: np.ndarray) -> np.ndarray:
    """Pure NumPy equivalent of float32 pairwise_distance(eps=1e-6)."""

    v = np.asarray(video, dtype=np.float32)
    a = np.asarray(audio, dtype=np.float32)
    n = min(len(v), len(a))
    padded = np.pad(a[:n], ((VSHIFT, VSHIFT), (0, 0)))
    rows = []
    for i in range(n):
        delta = v[i : i + 1] - padded[i : i + 31]
        rows.append(np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float32) + np.float32(1e-12)))
    return np.asarray(rows, dtype=np.float64)


def interpolate(value: np.ndarray, position: float) -> np.ndarray | None:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or not math.isfinite(position) or position < 0.0 or position > len(array) - 1:
        return None
    left = math.floor(position)
    right = min(left + 1, len(array) - 1)
    alpha = position - left
    result = (1.0 - alpha) * array[left] + alpha * array[right]
    return result if np.isfinite(result).all() else None


def unit(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    norm = float(np.linalg.norm(value))
    if norm <= 0.0 or not math.isfinite(norm):
        return None
    return np.asarray(value, dtype=np.float64) / norm


def bootstrap(values: Mapping[str, float], metric: str, minimum: int) -> dict[str, Any]:
    labels = sorted(str(key) for key in values)
    if len(labels) < minimum:
        return {"status": "NOT_ESTIMABLE", "metric": metric, "group_count": len(labels), "required_group_count": minimum}
    array = np.asarray([float(values[label]) for label in labels], dtype=np.float64)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    indices = rng.integers(0, len(labels), size=(BOOTSTRAP_DRAWS, len(labels)), dtype=np.int64)
    means = array[indices].mean(axis=1, dtype=np.float64)
    q = lambda p: float(np.quantile(means, p, method="linear"))
    lo95, hi95 = q(0.025), q(0.975)
    lo_bc, hi_bc = q(PRIMARY_ALPHA / 2.0), q(1.0 - PRIMARY_ALPHA / 2.0)
    return {"status": "COMPLETE", "metric": metric, "mean": float(array.mean()), "ci95": [lo95, hi95], "ci98_75_bonferroni": [lo_bc, hi_bc], "group_count": len(labels), "group_values": {label: float(v) for label, v in zip(labels, array, strict=True)}, "positive_group_count": int(np.sum(array > 0)), "draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED}


def calibrate_a(video: np.ndarray, audio: np.ndarray, support: Sequence[int]) -> tuple[int | None, dict[str, Any]]:
    usable = [int(i) for i in support if 0 <= int(i) < min(len(video), len(audio))]
    folds = {0: [i for i in usable if (i // 25) % 2 == 0], 1: [i for i in usable if (i // 25) % 2 == 1]}
    if len(folds[0]) < 10 or len(folds[1]) < 10:
        return None, {"status": "COVERAGE_LOW", "fold_counts": {str(k): len(v) for k, v in folds.items()}}
    details: dict[str, Any] = {"fold_counts": {str(k): len(v) for k, v in folds.items()}}
    fold_k: dict[str, int] = {}
    for test_fold in (0, 1):
        train = folds[1 - test_fold]
        candidates = []
        for k in range(-VSHIFT, VSHIFT + 1):
            values = [strict_distance(video[i], audio[i + k]) for i in train if 0 <= i + k < len(audio)]
            if values:
                candidates.append((float(np.mean(values)), abs(k), k))
        chosen = min(candidates)
        fold_k[str(test_fold)] = int(chosen[2])
        details[f"test_fold_{test_fold}"] = {"k0": int(chosen[2]), "train_count": len(train)}
    candidates = []
    for k in range(-VSHIFT, VSHIFT + 1):
        values = [strict_distance(video[i], audio[i + k]) for i in usable if 0 <= i + k < len(audio)]
        if values:
            candidates.append((float(np.mean(values)), abs(k), k))
    chosen = min(candidates)
    details.update({"status": "COMPLETE", "k0": int(chosen[2]), "cross_fold_k": fold_k})
    return int(chosen[2]), details


def calibration_support(rows: Sequence[int], k: int) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for index in rows:
        visual_start = 0.04 * (int(index) - int(k))
        intervals.append((visual_start, visual_start + 0.2))
        for lag in range(-VSHIFT, VSHIFT + 1):
            audio_start = 0.04 * (int(index) + lag)
            intervals.append((audio_start, audio_start + 0.215))
    return intervals


def support_clear(time_s: float, intervals: Sequence[tuple[float, float]]) -> bool:
    query_intervals = ((float(time_s) - 0.1075, float(time_s) + 0.1075), (float(time_s) - 0.1, float(time_s) + 0.1))
    for query_start, query_end in query_intervals:
        for start, end in intervals:
            gap = max(float(start) - query_end, query_start - float(end), 0.0)
            if gap < 0.24:
                return False
    return True


def parse_grid(path: Path) -> list[dict[str, Any]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    interval = re.compile(r"intervals\s*\[\s*\d+\s*\]\s*:\s*xmin\s*=\s*([0-9.eE+-]+).*?xmax\s*=\s*([0-9.eE+-]+).*?text\s*=\s*\"(.*?)\"", flags=re.IGNORECASE | re.DOTALL)

    def parse_tier(name: str) -> list[dict[str, Any]]:
        marker = re.search(rf'name\s*=\s*"{re.escape(name)}"', raw, flags=re.IGNORECASE)
        if marker is None:
            return []
        tail = raw[marker.end() :]
        next_tier = re.search(r'item\s*\[\s*\d+\s*\]\s*:', tail, flags=re.IGNORECASE)
        section = tail[: next_tier.start()] if next_tier else tail
        result = []
        for match in interval.finditer(section):
            start, end, label = float(match.group(1)), float(match.group(2)), match.group(3).strip()
            if not math.isfinite(start) or not math.isfinite(end) or end <= start:
                continue
            normalized = label.lower()
            result.append({"index": len(result), "label": label, "normalized": normalized, "start_s": start, "end_s": end, "silence": normalized in SILENCE, "unknown": normalized in UNKNOWN})
        return result

    result = parse_tier("phones")
    if not result or not any(not row["silence"] and not row["unknown"] for row in result):
        raise ValueError(f"no usable phones: {path}")
    words = parse_tier("words")
    for token in result:
        midpoint = 0.5 * (float(token["start_s"]) + float(token["end_s"]))
        owner = next((word for word in words if float(word["start_s"]) <= midpoint < float(word["end_s"]) and not word["silence"] and word["label"]), None)
        token["word_index"] = int(owner["index"]) if owner is not None else None
        token["word_normalized"] = str(owner["normalized"]) if owner is not None else None
    return result


def token_at(tokens: Sequence[Mapping[str, Any]], time_s: float) -> Mapping[str, Any] | None:
    return next((token for token in tokens if float(token["start_s"]) <= time_s < float(token["end_s"])), None)


def event(tokens: Sequence[Mapping[str, Any]], left_s: float, right_s: float) -> str:
    left, right = token_at(tokens, left_s), token_at(tokens, right_s)
    if left is None or right is None or left["silence"] or right["silence"] or left["unknown"] or right["unknown"]:
        return "EXCLUDED"
    if left["index"] == right["index"]:
        return "SAME"
    if left["normalized"] != right["normalized"]:
        return "CROSS"
    return "SAME_LABEL_DIFFERENT_OCCURRENCE"


def match(tokens_by_arm: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    speech = {arm: [t for t in tokens if not t["silence"] and not t["unknown"]] for arm, tokens in tokens_by_arm.items()}
    arms = list(speech)
    if not arms:
        return []
    has_words = all(all(token.get("word_index") is not None and token.get("word_normalized") for token in tokens) for tokens in speech.values())
    if has_words:
        groups: dict[str, list[list[dict[str, Any]]]] = {}
        for arm in arms:
            grouped: list[list[dict[str, Any]]] = []
            for token in speech[arm]:
                if not grouped or grouped[-1][0].get("word_index") != token.get("word_index"):
                    grouped.append([])
                grouped[-1].append(dict(token))
            groups[arm] = grouped
        cursors = {arm: 0 for arm in arms}
        result = []
        for anchor_group in groups[arms[0]]:
            label = anchor_group[0].get("word_normalized")
            chosen = {arms[0]: anchor_group}
            ok = True
            for arm in arms[1:]:
                found = next((idx for idx in range(cursors[arm], len(groups[arm])) if groups[arm][idx][0].get("word_normalized") == label), None)
                if found is None:
                    ok = False
                    break
                cursors[arm] = found + 1
                chosen[arm] = groups[arm][found]
            if not ok:
                continue
            phone_cursors = {arm: 0 for arm in arms}
            for anchor in anchor_group:
                row = {"occurrence": len(result), arms[0]: anchor}
                phone_ok = True
                for arm in arms[1:]:
                    found = next((idx for idx in range(phone_cursors[arm], len(chosen[arm])) if chosen[arm][idx]["normalized"] == anchor["normalized"]), None)
                    if found is None:
                        phone_ok = False
                        break
                    phone_cursors[arm] = found + 1
                    row[arm] = chosen[arm][found]
                if phone_ok:
                    result.append(row)
        return result
    cursor = {arm: 0 for arm in arms}
    result = []
    for anchor in speech[arms[0]]:
        row = {"occurrence": len(result), arms[0]: anchor}
        ok = True
        for arm in arms[1:]:
            found = next((i for i in range(cursor[arm], len(speech[arm])) if speech[arm][i]["normalized"] == anchor["normalized"]), None)
            if found is None:
                ok = False
                break
            cursor[arm] = found + 1
            row[arm] = speech[arm][found]
        if ok:
            result.append(row)
    return result


def independent_a(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    inputs = load_json(root / "inputs.json")
    by_id = {(int(row["sample_id"]), str(row["arm"])): row for row in inputs["A_features"]}
    rank_values: dict[str, float] = {}
    event_values: dict[str, float] = {}
    replay_max = 0.0
    replay_failures = []
    pair_rows = list(csv.DictReader((root / "A" / "time_pairs.csv").open(encoding="utf-8")))
    for sid in SAMPLE_IDS:
        states = {}
        for arm in ("N", "T"):
            row = by_id[(sid, arm)]
            video = np.asarray(np.load(row["visual"], allow_pickle=False), dtype=np.float32)
            audio = np.asarray(np.load(row["audio"], allow_pickle=False), dtype=np.float32)
            matrix = official_matrix(video, audio)
            cached = np.asarray(np.load(row["historical_matrix"], allow_pickle=False), dtype=np.float32)
            error = float(np.max(np.abs(matrix - cached))) if matrix.shape == cached.shape else float("inf")
            replay_max = max(replay_max, error)
            if error > 1e-4:
                replay_failures.append(f"{sid}/{arm}")
            support = list(range(30, min(len(video), len(audio)) - 30))
            k0, detail = calibrate_a(video, audio, support)
            states[arm] = (video, audio, support, k0, detail)
        if states["N"][3] is not None and states["T"][3] is not None:
            values = {}
            for arm, (video, audio, support, k0, _) in states.items():
                wins = []
                for i in support:
                    d0 = strict_distance(video[i], audio[i + int(k0)])
                    for delta in PRIMARY_DELTAS:
                        j = i + int(k0) + delta
                        if 0 <= j < len(audio):
                            dd = strict_distance(video[i], audio[j])
                            wins.append(1.0 if dd > d0 else 0.0 if dd < d0 else 0.5)
                values[arm] = float(np.mean(wins)) if wins else float("nan")
            if all(math.isfinite(v) for v in values.values()):
                rank_values[str(inputs["records"][sid - 151]["source_group"])] = values["T"] - values["N"]
        # Rebuild A_event directly from the producer's pair table and grids.
        records = {"N": [r for r in pair_rows if int(r["sample_id"]) == sid and r["arm"] == "N"], "T": [r for r in pair_rows if int(r["sample_id"]) == sid and r["arm"] == "T"]}
        if not records["N"] or not records["T"]:
            continue
        try:
            tokens = {arm: parse_grid(root / "A" / "A_textgrids" / f"{sid}_{arm}_audio.TextGrid") for arm in ("N", "T")}
        except (OSError, ValueError):
            continue
        by_h: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
        for arm in ("N", "T"):
            for row in records[arm]:
                if str(row.get("primary", "true")).strip().lower() not in {"1", "true", "yes"}:
                    continue
                category = event(tokens[arm], float(row["center0_s"]), float(row["center_delta_s"]))
                if category in {"SAME", "CROSS"}:
                    by_h[int(row["h"])][f"{arm}_{category}"].append(float(row["w_raw"]))
        qualified = []
        for cells in by_h.values():
            if all(len(cells.get(f"{arm}_{cat}", [])) >= 5 for arm in ("N", "T") for cat in ("SAME", "CROSS")):
                qualified.append(float(np.mean(cells["T_CROSS"]) - np.mean(cells["N_CROSS"])) - float(np.mean(cells["T_SAME"]) - np.mean(cells["N_SAME"])))
        if qualified:
            event_values[str(inputs["records"][sid - 151]["source_group"])] = float(np.mean(qualified))
    expected_rank = expected.get("a1", {}).get("summary", {}).get("group_values", {})
    rank_diff = {key: abs(float(value) - rank_values.get(key, float("nan"))) for key, value in expected_rank.items()}
    rank_pass = bool(rank_diff) and all(math.isfinite(v) and v <= 1e-10 for v in rank_diff.values())
    expected_event = expected.get("a_event", {}).get("summary", {}).get("group_values", {})
    event_diff = {key: abs(float(value) - event_values.get(key, float("nan"))) for key, value in expected_event.items()}
    event_pass = not expected_event or all(math.isfinite(v) and v <= 1e-10 for v in event_diff.values())
    return {"status": "PASS" if replay_max <= 1e-4 and rank_pass and event_pass else "FAIL", "replay": {"status": "PASS" if not replay_failures else "FAIL", "max_abs_error": replay_max, "failures": replay_failures}, "A_rank": {"status": "PASS" if rank_pass else "FAIL", "max_group_difference": max(rank_diff.values()) if rank_diff else None, "recomputed_group_values": rank_values}, "A_event": {"status": "PASS" if event_pass else "FAIL", "max_group_difference": max(event_diff.values()) if event_diff else None, "recomputed_group_values": event_values}}


def independent_b(root: Path, expected: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = root / "B" / "feature_manifest.json"
    if not manifest_path.is_file() or load_json(manifest_path).get("status") != "COMPLETE":
        return {"status": "INCOMPLETE", "reason": "feature manifest is not COMPLETE"}
    manifest = load_json(manifest_path)
    hash_failures = []
    for row in manifest.get("rows", []):
        for key in ("visual", "audio_feature"):
            path = Path(row[key])
            expected_hash = row["visual_sha256"] if key == "visual" else row["audio_feature_sha256"]
            if not path.is_file() or sha256(path) != expected_hash:
                hash_failures.append(str(path))
    if hash_failures:
        return {"status": "FAIL", "reason": "feature hash mismatch", "failures": hash_failures}
    grids = root / "B" / "B_textgrids"
    record_values: dict[str, dict[str, float]] = {}
    details = []
    for sid in SAMPLE_IDS:
        paths = {arm: grids / f"{sid}_{arm}.TextGrid" for arm in ("N", "T1", "T2")}
        if not all(path.is_file() for path in paths.values()):
            details.append({"sample_id": sid, "status": "ALIGNMENT_MISSING"})
            continue
        try:
            tokens = {arm: parse_grid(paths[arm]) for arm in paths}
            occurrences = match(tokens)
        except (OSError, ValueError) as exc:
            details.append({"sample_id": sid, "status": "ALIGNMENT_INVALID", "reason": str(exc)})
            continue
        features = {}
        for arm in ("N", "T1", "T2"):
            base = root / "B" / "features" / "main" / str(sid)
            features[arm] = (np.asarray(np.load(base / f"V_{arm}.npy", allow_pickle=False), dtype=np.float32), np.asarray(np.load(base / f"A_{arm}.npy", allow_pickle=False), dtype=np.float32))
        n_cal = max(1, math.ceil(0.2 * len(occurrences)))
        calibration_occ, eval_occ = occurrences[:n_cal], occurrences[n_cal:]
        k_by_arm = {}
        calibration_rows_by_arm = {}
        for arm in ("N", "T1", "T2"):
            rows = []
            video, audio = features[arm]
            for occ in calibration_occ:
                token = occ[arm]
                center = 0.5 * (float(token["start_s"]) + float(token["end_s"]))
                index = math.floor((center - 0.1075) / 0.04 + 0.5)
                if 0 <= index < len(video) and 0 <= index - VSHIFT < len(audio) and 0 <= index + VSHIFT < len(audio):
                    rows.append(index)
            rows = sorted(set(rows))
            calibration_rows_by_arm[arm] = rows
            if len(rows) < 5:
                k_by_arm[arm] = None
                continue
            candidates = []
            for k in range(-VSHIFT, VSHIFT + 1):
                ds = [strict_distance(video[i], audio[i + k]) for i in rows]
                candidates.append((float(np.mean(ds)), abs(k), k))
            k_by_arm[arm] = int(min(candidates)[2])
        if any(k_by_arm[arm] is None for arm in k_by_arm) or len(eval_occ) < 8:
            details.append({"sample_id": sid, "status": "ALIGNMENT_INSUFFICIENT", "matched": len(occurrences), "k": k_by_arm})
            continue
        exclusion = {arm: calibration_support(calibration_rows_by_arm[arm], int(k_by_arm[arm])) for arm in ("N", "T1", "T2")}
        cells: dict[str, list[float]] = defaultdict(list)
        valid_occurrences = []
        identity_errors: list[float] = []
        raw_identity_errors: list[float] = []
        for occ in eval_occ:
            query_values = []
            occurrence_valid = True
            for phase in (0.25, 0.5, 0.75):
                vectors = {"v": {}, "a": {}}
                raw_vectors = {"v": {}, "a": {}}
                valid = True
                for arm in ("N", "T1", "T2"):
                    token = occ[arm]
                    t = float(token["start_s"]) + phase * (float(token["end_s"]) - float(token["start_s"]))
                    u = (t - 0.1075) / 0.04
                    video, audio = features[arm]
                    vv = unit(interpolate(video, u - int(k_by_arm[arm])))
                    aa = unit(interpolate(audio, u))
                    if vv is None or aa is None or not (0.24 <= t <= (len(audio) - 1) * 0.04) or not support_clear(t, exclusion[arm]):
                        valid = False
                        break
                    vectors["v"][arm], vectors["a"][arm] = vv, aa
                    raw_vectors["v"][arm] = interpolate(video, u - int(k_by_arm[arm]))
                    raw_vectors["a"][arm] = interpolate(audio, u)
                if not valid:
                    occurrence_valid = False
                    break
                row = {}
                for vi in ("N", "T1", "T2"):
                    for aj in ("N", "T1", "T2"):
                        row[f"{vi}{aj}"] = float(np.sum((vectors["v"][vi] - vectors["a"][aj]) ** 2))
                interaction = 0.5 * (row["T1T2"] + row["T2T1"] - row["T1T1"] - row["T2T2"])
                dot = float(np.dot(vectors["v"]["T1"] - vectors["v"]["T2"], vectors["a"]["T1"] - vectors["a"]["T2"]))
                raw_interaction = 0.5 * (sum(float(np.sum((raw_vectors["v"][vi] - raw_vectors["a"][aj]) ** 2)) for vi, aj in (("T1", "T2"), ("T2", "T1"))) - sum(float(np.sum((raw_vectors["v"][vi] - raw_vectors["a"][aj]) ** 2)) for vi, aj in (("T1", "T1"), ("T2", "T2"))))
                raw_dot = float(np.dot(raw_vectors["v"]["T1"] - raw_vectors["v"]["T2"], raw_vectors["a"]["T1"] - raw_vectors["a"]["T2"]))
                identity_errors.append(abs(interaction - dot))
                raw_identity_errors.append(abs(raw_interaction - raw_dot))
                query_values.append(row)
            if occurrence_valid and len(query_values) == 3:
                valid_occurrences.append(occ)
                for key in query_values[0]:
                    cells[key].append(float(np.mean([row[key] for row in query_values])))
        if len(valid_occurrences) < 8:
            details.append({"sample_id": sid, "status": "COMMON_SUPPORT_LOW", "valid_occurrences": len(valid_occurrences), "identity_max_abs_error": max(identity_errors, default=None), "raw_identity_max_abs_error": max(raw_identity_errors, default=None)})
            continue
        means = {key: float(np.mean(value)) for key, value in cells.items()}
        record = {"sample_id": sid, "I_TT": 0.5 * (means["T1T2"] + means["T2T1"] - means["T1T1"] - means["T2T2"]), "X": means["NN"] - 0.5 * (means["T1N"] + means["T2N"])}
        group = str(load_json(root / "inputs.json")["records"][sid - 151]["source_group"])
        record_values[group] = {"I_TT": float(record["I_TT"]), "X": float(record["X"])}
        details.append({"sample_id": sid, "status": "COMPLETE", "valid_occurrences": len(valid_occurrences), "identity_max_abs_error": max(identity_errors, default=None), "raw_identity_max_abs_error": max(raw_identity_errors, default=None), **record})
    expected_records = {str(row["source_group"]): row for row in expected.get("record_stats", []) if row.get("status") == "COMPLETE"}
    differences = {}
    for group, row in expected_records.items():
        if group in record_values:
            differences[group] = {key: abs(float(row[key]) - record_values[group][key]) for key in ("I_TT", "X")}
    pass_values = bool(differences) and all(value <= 1e-10 for row in differences.values() for value in row.values())
    identity_pass = all(float(row.get("identity_max_abs_error") or 0.0) <= 1e-10 and float(row.get("raw_identity_max_abs_error") or 0.0) <= 1e-8 for row in details if row.get("status") == "COMPLETE")
    return {"status": "PASS" if pass_values and identity_pass else "INCOMPLETE", "record_values": record_values, "max_difference": max((value for row in differences.values() for value in row.values()), default=None), "identity_pass": identity_pass, "details": details, "B_instance": bootstrap({g: v["I_TT"] for g, v in record_values.items()}, "B_instance", 8), "B_transfer": bootstrap({g: v["X"] for g, v in record_values.items()}, "B_transfer", 8)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.run_root.resolve()
    expected_a = load_json(root / "A" / "a_summary.json") if (root / "A" / "a_summary.json").is_file() else {}
    expected_b = load_json(root / "B" / "b_summary.json") if (root / "B" / "b_summary.json").is_file() else {}
    result = {"schema_version": 1, "status": "INCOMPLETE", "A": independent_a(root, expected_a), "B": independent_b(root, expected_b)}
    result["status"] = "PASS" if result["A"]["status"] == "PASS" and result["B"].get("status") in {"PASS", "INCOMPLETE"} else "FAIL"
    (root / "independent_recompute.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
