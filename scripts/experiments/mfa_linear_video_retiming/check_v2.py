"""Independent v2 search audit; no producer scorer or selector is imported."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .common import ProtocolError, canonical_json_sha256, file_sha256, verify_json
from .check import rebuild_q_independently, np_distance_matrix, np_metrics


def _proxy_score(visual: np.ndarray, audio: np.ndarray, q: np.ndarray, rows: np.ndarray) -> dict[str, Any]:
    coordinate = q[rows + 2] - 2
    low = np.floor(coordinate).astype(np.int64)
    high = np.ceil(coordinate).astype(np.int64)
    if np.any(low < 0) or np.any(high >= visual.shape[0]):
        raise ProtocolError("CHECK_PROXY_SUPPORT_OUT_OF_BOUNDS")
    weight = (coordinate - low).astype(np.float32)[:, None]
    v = (np.float32(1) - weight) * visual[low] + weight * visual[high]
    curve_rows = []
    for lag in range(-15, 16):
        if np.any(rows + lag < 0) or np.any(rows + lag >= audio.shape[0]):
            raise ProtocolError("CHECK_PROXY_AUDIO_SUPPORT_OUT_OF_BOUNDS")
        d = v - audio[rows + lag] + np.float32(1e-6)
        curve_rows.append(np.sqrt(np.sum(np.square(d, dtype=np.float32), axis=1, dtype=np.float32), dtype=np.float32))
    curve = np.mean(np.stack(curve_rows, axis=1), axis=0, dtype=np.float32)
    minimum = float(np.min(curve))
    index = int(np.argmin(curve))
    return {"sync_c": float(np.median(curve) - minimum), "sync_d": minimum,
            "d0": float(curve[15]), "offset": 15 - index, "curve": curve}


def _reasons(row: Mapping[str, Any], baseline: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[str]:
    if row.get("score_kind") != "true_fixed_crop":
        return ["NOT_TRUE_FORWARD"]
    m = row["metrics"]
    reasons = []
    if float(m["sync_c"]) < float(baseline["sync_c"]) - float(cfg["noninferiority_sync_c"]):
        reasons.append("SYNC_C_BELOW_FLOOR")
    if float(m["sync_d"]) > float(baseline["sync_d"]) + float(cfg["noninferiority_sync_d"]):
        reasons.append("SYNC_D_ABOVE_CEILING")
    if float(m["d0"]) > float(baseline["d0"]) - float(cfg["minimum_d0_improvement"]):
        reasons.append("D0_GAIN_TOO_SMALL")
    if abs(int(m["offset"])) > 1:
        reasons.append("OFFSET_OUTSIDE_ONE_FRAME")
    if not m.get("local_windows") or not baseline.get("local_windows"):
        reasons.append("LOCAL_WINDOWS_MISSING")
    else:
        if float(m["local_median_d0"]) > float(baseline["local_median_d0"]):
            reasons.append("LOCAL_D0_WORSE")
        if float(m["local_abs_offset_q90"]) > float(baseline["local_abs_offset_q90"]):
            reasons.append("LOCAL_OFFSET_WORSE")
    return reasons


def _shortlist_independently(rows: list[Mapping[str, Any]], support: np.ndarray,
                             exclude: set[str]) -> list[str]:
    pool = [row for row in rows if row["map_sha256"] not in exclude]
    def d0_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        score = row["proxy_metrics"]
        return (float(score["d0"]), abs(int(score["offset"])), -float(score["sync_c"]),
                float(row["regularization"]), str(row["map_sha256"]))
    def c_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
        score = row["proxy_metrics"]
        return (abs(int(score["offset"])) > 1, -float(score["sync_c"]), float(score["d0"]),
                float(row["regularization"]), str(row["map_sha256"]))
    chosen = []
    seen = set()
    for ordered, limit in ((sorted(pool, key=d0_key), 8), (sorted(pool, key=c_key), 16)):
        for row in ordered:
            if len(chosen) == limit:
                break
            if row["map_sha256"] not in seen:
                chosen.append(row)
                seen.add(row["map_sha256"])
    coordinates = {row["map_sha256"]: np.asarray(row["map"]["delta"], dtype=np.float32)[support] for row in pool}
    while len(chosen) < 25:
        remaining = [row for row in pool if row["map_sha256"] not in seen]
        if not remaining:
            break
        row = min(remaining, key=lambda candidate: (
            -min(float(np.mean(np.abs(coordinates[candidate["map_sha256"]] - coordinates[old["map_sha256"]]), dtype=np.float32))
                 for old in chosen), *d0_key(candidate)))
        chosen.append(row)
        seen.add(row["map_sha256"])
    return [row["map_sha256"] for row in chosen]


def verify_search_v2(run_dir: Path, frozen: Mapping[str, Any], config: Mapping[str, Any]) -> list[dict[str, Any]]:
    output = []
    for record in frozen["records"]:
        sid = str(record["sample_id"])
        base = run_dir / "03_search" / sid
        calibration_path, state_path = base / "state.json", base / "state_v2.json"
        calibration, state, result = (verify_json(path, self_hash=True) for path in
                                      (calibration_path, state_path, base / "result.json"))
        if result.get("protocol") != config["protocol"] or state.get("protocol") != config["protocol"]:
            raise ProtocolError(f"CHECK_V2_PROTOCOL_MISMATCH:{sid}")
        if state["calibration_state_sha256"] != file_sha256(calibration_path):
            raise ProtocolError(f"CHECK_V2_CALIBRATION_HASH_MISMATCH:{sid}")
        request_hash = canonical_json_sha256({"request": calibration["request_fingerprint"],
                                              "search": config["search"],
                                              "calibration_sha256": file_sha256(calibration_path)})
        if request_hash != state.get("request_hash"):
            raise ProtocolError(f"CHECK_V2_REQUEST_HASH_MISMATCH:{sid}")
        if int(state["proxy_attempted"]) > int(config["search"]["proxy_unique_maps"]) or int(state["true_attempted"]) > 96:
            raise ProtocolError(f"CHECK_V2_BUDGET_EXCEEDED:{sid}")
        with np.load(calibration["baseline_candidate"]["embedding_path"], allow_pickle=False) as handle:
            visual_original = np.asarray(handle["visual"], dtype=np.float32)
        with np.load(calibration["natural_audio_embedding_path"], allow_pickle=False) as handle:
            audio = np.asarray(handle["audio"], dtype=np.float32)
        support = np.asarray(calibration["frozen_support"], dtype=np.int64)
        proxy_rows = {}
        for ref in state["proxy_chunks"]:
            path = base / ref["name"]
            if file_sha256(path) != ref["sha256"]:
                raise ProtocolError(f"CHECK_V2_PROXY_CHUNK_HASH_MISMATCH:{sid}")
            chunk = verify_json(path, self_hash=True)
            if len(chunk["rows"]) != int(ref["count"]):
                raise ProtocolError(f"CHECK_V2_PROXY_CHUNK_COUNT_MISMATCH:{sid}")
            for row in chunk["rows"]:
                key = str(row["map_sha256"])
                if row.get("score_kind") != "proxy_embedding" or "metrics" in row or "embedding_path" in row or key in proxy_rows:
                    raise ProtocolError(f"CHECK_V2_PROXY_ROLE_INVALID:{sid}:{key}")
                q = rebuild_q_independently(row["map"])
                actual = _proxy_score(visual_original, audio, q, support)
                saved = row["proxy_metrics"]
                for name in ("sync_c", "sync_d", "d0"):
                    if abs(float(actual[name]) - float(saved[name])) > 1e-4:
                        raise ProtocolError(f"CHECK_V2_PROXY_METRIC_MISMATCH:{sid}:{key}:{name}")
                if int(actual["offset"]) != int(saved["offset"]):
                    raise ProtocolError(f"CHECK_V2_PROXY_OFFSET_MISMATCH:{sid}:{key}")
                if not np.allclose(actual["curve"], saved["curve"], atol=1e-4, rtol=0):
                    raise ProtocolError(f"CHECK_V2_PROXY_CURVE_MISMATCH:{sid}:{key}")
                proxy_rows[key] = row
        if len(proxy_rows) != int(state["proxy_completed"]):
            raise ProtocolError(f"CHECK_V2_PROXY_COUNT_MISMATCH:{sid}")
        stages = {label: sum(row.get("stage") == label for row in proxy_rows.values()) for label in ("A", "B", "LOCAL")}
        if (stages["A"] != int(state["stage_a_proxy_count"])
                or stages["B"] + stages["LOCAL"] != int(state["stage_b_proxy_count"])
                or sum(stages.values()) != len(proxy_rows)):
            raise ProtocolError(f"CHECK_V2_PROXY_STAGE_COUNT_MISMATCH:{sid}")
        if float(state["elapsed_search_seconds"]) > float(config["search"]["max_seconds_per_sample"]):
            raise ProtocolError(f"CHECK_V2_SEARCH_TIME_EXCEEDED:{sid}")
        true_rows = state["true_candidates"]
        if int(state["true_completed"]) != len(true_rows) - 1:
            raise ProtocolError(f"CHECK_V2_TRUE_COUNT_MISMATCH:{sid}")
        for key, row in true_rows.items():
            if row.get("score_kind") != "true_fixed_crop" or str(row["map_sha256"]) != key:
                raise ProtocolError(f"CHECK_V2_TRUE_ROLE_INVALID:{sid}:{key}")
            rebuild_q_independently(row["map"])
            if file_sha256(row["embedding_path"]) != row["embedding_sha256"]:
                raise ProtocolError(f"CHECK_V2_TRUE_EMBEDDING_HASH_MISMATCH:{sid}:{key}")
            with np.load(row["embedding_path"], allow_pickle=False) as handle:
                visual = np.asarray(handle["visual"], dtype=np.float32)
            n = int(calibration["baseline_candidate"]["metrics"]["conservative_window_count"])
            matrix = np_distance_matrix(visual, audio, n)
            actual = np_metrics(matrix, support)
            for name in ("sync_c", "sync_d", "d0", "local_median_d0", "local_abs_offset_q90"):
                if abs(float(actual[name]) - float(row["metrics"][name])) > 1e-4:
                    raise ProtocolError(f"CHECK_V2_TRUE_METRIC_MISMATCH:{sid}:{key}:{name}")
            if int(actual["offset"]) != int(row["metrics"]["offset"]):
                raise ProtocolError(f"CHECK_V2_TRUE_OFFSET_MISMATCH:{sid}:{key}")
            if not np.allclose(actual["curve"], row["metrics"]["curve"], atol=1e-4, rtol=0):
                raise ProtocolError(f"CHECK_V2_TRUE_CURVE_MISMATCH:{sid}:{key}")
            local_saved = row["metrics"].get("local_windows", [])
            expected_chunks = [support[index:index + 25] for index in range(0, len(support), 25) if len(support[index:index + 25]) == 25]
            if len(local_saved) != len(expected_chunks):
                raise ProtocolError(f"CHECK_V2_LOCAL_WINDOW_COUNT_MISMATCH:{sid}:{key}")
            for saved_window, chunk in zip(local_saved, expected_chunks, strict=True):
                curve = np.mean(matrix[chunk], axis=0, dtype=np.float32)
                if (int(saved_window["row_start"]) != int(chunk[0]) or int(saved_window["row_stop_exclusive"]) != int(chunk[-1] + 1)
                        or abs(float(saved_window["d0"]) - float(curve[15])) > 1e-4
                        or int(saved_window["offset"]) != 15 - int(np.argmin(curve))):
                    raise ProtocolError(f"CHECK_V2_LOCAL_WINDOW_MISMATCH:{sid}:{key}")
        identity_key = calibration["baseline_candidate"]["map_sha256"]
        roots = {key for key, row in true_rows.items() if key == identity_key or row.get("stage") == "ROOT"}
        a_rows = [row for row in proxy_rows.values() if row.get("stage") == "A"]
        expected_a = _shortlist_independently(a_rows, support, roots)
        if state["shortlist_a"] != expected_a:
            raise ProtocolError(f"CHECK_V2_SHORTLIST_A_MISMATCH:{sid}")
        after_a = roots | set(state["shortlist_a"])
        b_rows = [row for row in proxy_rows.values() if row.get("stage") == "B"]
        expected_b = _shortlist_independently(b_rows, support, after_a)
        if state["shortlist_b"] != expected_b:
            raise ProtocolError(f"CHECK_V2_SHORTLIST_B_MISMATCH:{sid}")
        baseline = true_rows[identity_key]["metrics"]
        expected_gate = {key: _reasons(row, baseline, config["search"]) for key, row in true_rows.items()}
        if state.get("final_gate") != expected_gate:
            raise ProtocolError(f"CHECK_V2_FINAL_GATE_MISMATCH:{sid}")
        eligible = [row for key, row in true_rows.items() if key != identity_key and not _reasons(row, baseline, config["search"])]
        if eligible:
            minimum = min(float(row["metrics"]["d0"]) for row in eligible)
            subset = [row for row in eligible if float(row["metrics"]["d0"]) <= minimum + float(config["search"]["final_d0_slack"])]
            selected = min(subset, key=lambda row: (-float(row["metrics"]["sync_c"]), float(row["regularization"]), str(row["map_sha256"])))
        else:
            selected = true_rows[identity_key]
        if result["selected"]["map_sha256"] != selected["map_sha256"] or state["selected_map_sha256"] != selected["map_sha256"]:
            raise ProtocolError(f"CHECK_V2_SELECTION_MISMATCH:{sid}")
        if result["selected"].get("score_kind") != "true_fixed_crop":
            raise ProtocolError(f"CHECK_V2_PROXY_SELECTED_AS_TRUE:{sid}")
        global_rows = [r for r in true_rows.values() if str(r.get("label", "")).startswith("global_seed_")]
        if len(global_rows) != 6:
            raise ProtocolError(f"CHECK_V2_GLOBAL_ROOT_COUNT:{sid}")
        global_best = min(global_rows, key=lambda r: (float(r["metrics"]["d0"]), abs(int(r["metrics"]["offset"])),
                                                     -float(r["metrics"]["sync_c"]), float(r["regularization"]), str(r["map_sha256"])))
        if result["global_control"]["map_sha256"] != global_best["map_sha256"]:
            raise ProtocolError(f"CHECK_V2_GLOBAL_SELECTION_MISMATCH:{sid}")
        output.append({"sample_id": sid, "proxy_count": len(proxy_rows), "true_count": len(true_rows),
                       "selected_map_sha256": selected["map_sha256"], "status": "PASS"})
    return output
