"""Two-stage embedding beam search with true SyncNet reranking."""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .common import ProtocolError, canonical_json_sha256, file_sha256, verify_json, write_json
from .retime import build_map, global_seed, validate_map, render_map
from .scorer import calibration_fingerprint, crop_frames, forward_video, load_frozen_syncnet, read_video_frames, score_embeddings
from .search_worker import _load_embedding, _save_embeddings
from .surrogate import score_proxy


def final_reasons(candidate: Mapping[str, Any], baseline: Mapping[str, Any], cfg: Mapping[str, Any]) -> list[str]:
    if candidate.get("score_kind") != "true_fixed_crop":
        return ["NOT_TRUE_FORWARD"]
    m = candidate["metrics"]
    reasons = []
    for name in ("sync_c", "sync_d", "d0"):
        if not np.isfinite(float(m[name])):
            reasons.append(f"NONFINITE_{name.upper()}")
    if reasons:
        return reasons
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


def select_final(rows: Sequence[Mapping[str, Any]], baseline: Mapping[str, Any], cfg: Mapping[str, Any]) -> Mapping[str, Any] | None:
    eligible = [row for row in rows if row.get("label") != "identity" and not final_reasons(row, baseline, cfg)]
    if not eligible:
        return None
    best_d0 = min(float(row["metrics"]["d0"]) for row in eligible)
    close = [row for row in eligible if float(row["metrics"]["d0"]) <= best_d0 + float(cfg["final_d0_slack"])]
    return min(close, key=lambda row: (-float(row["metrics"]["sync_c"]), float(row["regularization"]), str(row["map_sha256"])))


def _d0_key(row: Mapping[str, Any], *, proxy: bool) -> tuple[Any, ...]:
    m = row["proxy_metrics" if proxy else "metrics"]
    return (float(m["d0"]), abs(int(m["offset"])), -float(m["sync_c"]), float(row["regularization"]), str(row["map_sha256"]))


def _c_key(row: Mapping[str, Any], *, proxy: bool) -> tuple[Any, ...]:
    m = row["proxy_metrics" if proxy else "metrics"]
    return (abs(int(m["offset"])) > 1, -float(m["sync_c"]), float(m["d0"]), float(row["regularization"]), str(row["map_sha256"]))


def _distance(left: Mapping[str, Any], right: Mapping[str, Any], support: np.ndarray) -> float:
    a = np.asarray(left["map"]["delta"], dtype=np.float32)
    b = np.asarray(right["map"]["delta"], dtype=np.float32)
    return float(np.mean(np.abs(a[support] - b[support]), dtype=np.float32))


def choose_routes(rows: Sequence[Mapping[str, Any]], support: np.ndarray, *, counts: tuple[int, int, int], proxy: bool,
                  exclude: set[str] | None = None) -> list[Mapping[str, Any]]:
    pool = [r for r in rows if str(r["map_sha256"]) not in (exclude or set()) and r.get("proxy_metrics" if proxy else "metrics") is not None]
    chosen: list[Mapping[str, Any]] = []
    seen: set[str] = set()

    def add(row: Mapping[str, Any]) -> None:
        key = str(row["map_sha256"])
        if key not in seen:
            seen.add(key)
            chosen.append(row)

    for row in sorted(pool, key=lambda r: _d0_key(r, proxy=proxy)):
        if len(chosen) >= counts[0]:
            break
        add(row)
    target = counts[0] + counts[1]
    for row in sorted(pool, key=lambda r: _c_key(r, proxy=proxy)):
        if len(chosen) >= target:
            break
        add(row)
    target += counts[2]
    # The archive has up to 20k maps. Compute coverage distances in batches
    # instead of rebuilding two NumPy arrays for every pair in Python.
    coordinates = np.stack([np.asarray(row["map"]["delta"], dtype=np.float32)[support] for row in pool]) if pool else np.empty((0, len(support)), dtype=np.float32)
    index_by_sha = {str(row["map_sha256"]): index for index, row in enumerate(pool)}
    nearest = np.full(len(pool), np.inf, dtype=np.float32)
    for row in chosen:
        index = index_by_sha[str(row["map_sha256"])]
        nearest = np.minimum(nearest, np.mean(np.abs(coordinates - coordinates[index]), axis=1, dtype=np.float32))
    while len(chosen) < target:
        remaining = [index for index, row in enumerate(pool) if str(row["map_sha256"]) not in seen]
        if not remaining:
            break
        if chosen:
            index = min(remaining, key=lambda i: (-float(nearest[i]), *_d0_key(pool[i], proxy=proxy)))
        else:
            index = min(remaining, key=lambda i: _d0_key(pool[i], proxy=proxy))
        row = pool[index]
        add(row)
        nearest = np.minimum(nearest, np.mean(np.abs(coordinates - coordinates[index]), axis=1, dtype=np.float32))
    return chosen


def propose_maps(parent: Mapping[str, Any], step: float) -> list[tuple[str, dict[str, Any]]]:
    mapping = parent["map"]
    positions = mapping["knot_positions"]
    current = np.asarray(mapping["knot_deltas"], dtype=np.float64)
    proposals: list[tuple[str, dict[str, Any]]] = []
    moves: list[tuple[str, np.ndarray]] = []
    for width, label in ((1, "single"), (2, "pair"), (4, "four")):
        for index in range(max(0, len(current) - width + 1)):
            for sign in (-1, 1):
                values = current.copy()
                values[index:index + width] += sign * step
                moves.append((f"{label}_{index}_{sign:+d}_{step:g}", values))
    if len(current):
        profile = np.asarray(global_seed(mapping["frame_count"], mapping["valid_frame_count"], 1)["knot_deltas"], dtype=np.float64)
        for sign in (-1, 1):
            moves.append((f"global_{sign:+d}_{step:g}", current + sign * step * profile))
    for label, values in moves:
        values = np.rint(values * 4.0) / 4.0
        if np.any(np.abs(values) > 3.0 + 1e-12):
            continue
        try:
            item = build_map(mapping["frame_count"], mapping["valid_frame_count"], values, positions=positions)
            validate_map(item)
        except ProtocolError:
            continue
        proposals.append((label, item))
    return proposals


def _iter_round(parents: Sequence[Mapping[str, Any]], step: float):
    # Interleave parents to prevent one early parent from spending a phase budget.
    groups = [propose_maps(parent, step) for parent in parents]
    for index in range(max((len(group) for group in groups), default=0)):
        for parent, group in zip(parents, groups, strict=True):
            if index < len(group):
                label, mapping = group[index]
                yield str(parent["map_sha256"]), label, mapping


def _load_proxy_chunks(state: Mapping[str, Any], base: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for ref in state.get("proxy_chunks", []):
        path = base / ref["name"]
        if file_sha256(path) != ref["sha256"]:
            raise ProtocolError("PROXY_CHUNK_HASH_MISMATCH")
        chunk = verify_json(path, self_hash=True)
        for row in chunk["rows"]:
            key = row["map_sha256"]
            if key in rows:
                raise ProtocolError("DUPLICATE_PROXY_MAP")
            rows[key] = row
    return rows


def _flush_proxy(state: dict[str, Any], rows: list[dict[str, Any]], pending: list[dict[str, Any]], state_path: Path) -> None:
    if not pending:
        return
    name = f"proxy_chunks/chunk_{len(state['proxy_chunks']):05d}.json"
    path = state_path.parent / name
    write_json(path, {"schema_version": 2, "rows": pending}, self_hash=True)
    state["proxy_chunks"].append({"name": name, "sha256": file_sha256(path), "count": len(pending)})
    state["proxy_completed"] = int(state["proxy_completed"]) + len(pending)
    for row in pending:
        rows.append(row)
    pending.clear()
    write_json(state_path, state, self_hash=True)


def search_record_v2(request: Mapping[str, Any]) -> dict[str, Any]:
    calibration_path = Path(request["state_path"])
    calibration = verify_json(calibration_path, self_hash=True)
    if calibration.get("status") != "CALIBRATED" or calibration.get("request_fingerprint") != request["request_fingerprint"]:
        raise ProtocolError("V2_CALIBRATION_BINDING_FAILED")
    search_cfg = request["search"]
    state_path = calibration_path.with_name("state_v2.json")
    original_visual = _load_embedding(calibration["baseline_candidate"]["embedding_path"], calibration["baseline_candidate"]["embedding_sha256"])["visual"]
    audio = _load_embedding(calibration["natural_audio_embedding_path"], calibration["natural_audio_embedding_sha256"])["audio"]
    support = np.asarray(calibration["frozen_support"], dtype=np.int64)
    identity = {**calibration["baseline_candidate"], "score_kind": "true_fixed_crop"}
    baseline = identity["metrics"]
    proxy_identity = score_proxy(original_visual, audio, identity["map"], support)
    if (max(abs(float(proxy_identity[key]) - float(baseline[key])) for key in ("sync_c", "sync_d", "d0")) > 1e-4
            or int(proxy_identity["offset"]) != int(baseline["offset"])):
        raise ProtocolError("PROXY_IDENTITY_PARITY_FAILED")
    request_hash = canonical_json_sha256({"request": request["request_fingerprint"], "search": search_cfg,
                                         "calibration_sha256": file_sha256(calibration_path)})
    if state_path.exists():
        state = verify_json(state_path, self_hash=True)
        if state.get("request_hash") != request_hash:
            raise ProtocolError("V2_SEARCH_STATE_FINGERPRINT_MISMATCH")
        proxy = _load_proxy_chunks(state, state_path.parent)
        if len(proxy) != int(state["proxy_completed"]):
            raise ProtocolError("V2_PROXY_COUNT_MISMATCH")
        if state.get("active_search_started_at") is not None:
            # A crashed process cannot prove when it stopped. Conservatively
            # charge all elapsed wall time until this resume.
            state["elapsed_search_seconds"] = float(state["elapsed_search_seconds"]) + max(0.0, time.time() - float(state["active_search_started_at"]))
        for row in state["true_candidates"].values():
            if file_sha256(row["embedding_path"]) != row["embedding_sha256"]:
                raise ProtocolError("V2_TRUE_EMBEDDING_HASH_MISMATCH")
    else:
        state = {"schema_version": 2, "protocol": "mfa_linear_video_retiming_v2", "request_hash": request_hash,
                 "calibration_state_path": str(calibration_path.resolve()), "calibration_state_sha256": file_sha256(calibration_path),
                 "scorer_fingerprint": calibration["scorer_fingerprint"], "status": "SEARCHING", "phase": "roots",
                 "proxy_chunks": [], "proxy_attempted": 0, "proxy_completed": 0, "true_attempted": 0,
                 "true_completed": 0, "true_failed": 0, "true_candidates": {identity["map_sha256"]: identity},
                 "root_archive": [], "beam": [], "shortlist_a": [], "shortlist_b": [],
                 "stage_a_proxy_count": 0, "stage_b_proxy_count": 0, "elapsed_search_seconds": 0.0,
                 "proxy_scoring_seconds": 0.0, "true_forward_seconds": 0.0,
                 "cursor": {"phase": "roots", "step": 0, "round": 0}, "inflight": None}
        proxy = {}
        write_json(state_path, state, self_hash=True)
    if state.get("status") == "SEARCH_COMPLETE" and Path(request["result_path"]).exists():
        return verify_json(request["result_path"], self_hash=True)

    scorer, torch, model_meta = load_frozen_syncnet(request["syncnet_root"], request["syncnet_model"], device="cuda")
    expected = calibration_fingerprint(model_meta, score_box=request["score_box_xyxy"], support=support,
                                       legacy_score_worker=request["legacy_score_worker"])
    if expected != calibration["scorer_fingerprint"] or model_meta["checkpoint_sha256"] != request["expected_syncnet_sha256"]:
        raise ProtocolError("V2_SCORER_FINGERPRINT_MISMATCH")
    frames, _ = read_video_frames(request["video_m"])
    if file_sha256(request["video_m"]) != calibration["original_M_video_sha256"] or hashlib.sha256(np.ascontiguousarray(frames).tobytes()).hexdigest() != calibration["original_M_frame_sha256"]:
        raise ProtocolError("V2_ORIGINAL_M_VIDEO_CHANGED")
    started = time.monotonic()
    state["active_search_started_at"] = time.time()
    write_json(state_path, state, self_hash=True)
    # A 30-second tail covers the last model forward, atomic state writes and
    # final selection, so the recorded wall time stays below the hard ceiling.
    deadline = started + max(0.0, float(search_cfg["max_seconds_per_sample"]) - float(state["elapsed_search_seconds"]) - 30.0)
    proxy_deadline = deadline
    pending: list[dict[str, Any]] = []

    def save() -> None:
        _flush_proxy(state, [], pending, state_path)
        state["elapsed_search_seconds"] = float(state["elapsed_search_seconds"]) + time.monotonic() - started
        write_json(state_path, state, self_hash=True)

    def proxy_one(mapping: Mapping[str, Any], *, stage: str, parent: str | None, move: str) -> dict[str, Any] | None:
        key = str(mapping["map_sha256"])
        if key in proxy:
            return proxy[key]
        limit = int(search_cfg["stage_a_proxy_maps"] if stage == "A" else search_cfg["stage_b_proxy_maps"])
        count_key = "stage_a_proxy_count" if stage == "A" else "stage_b_proxy_count"
        if int(state[count_key]) >= limit or int(state["proxy_attempted"]) >= int(search_cfg["proxy_unique_maps"]) or time.monotonic() >= min(deadline, proxy_deadline):
            return None
        state["proxy_attempted"] += 1
        state[count_key] += 1
        write_json(state_path, state, self_hash=True)
        score_started = time.monotonic()
        metrics = score_proxy(original_visual, audio, mapping, support)
        state["proxy_scoring_seconds"] = float(state.get("proxy_scoring_seconds", 0.0)) + time.monotonic() - score_started
        geometry = validate_map(mapping)
        row = {"candidate_id": key, "map_sha256": key, "map": dict(mapping), "proxy_metrics": metrics,
               "score_kind": "proxy_embedding", "stage": stage, "parent": parent, "move": move,
               "regularization": geometry["regularization"]}
        proxy[key] = row
        pending.append(row)
        if len(pending) >= int(search_cfg["proxy_chunk_size"]):
            _flush_proxy(state, [], pending, state_path)
        return row

    def true_one(mapping: Mapping[str, Any], *, stage: str, label: str) -> dict[str, Any] | None:
        key = str(mapping["map_sha256"])
        if key in state["true_candidates"]:
            return state["true_candidates"][key]
        if int(state["true_attempted"]) >= int(search_cfg["true_candidate_forwards"]) or time.monotonic() >= deadline:
            return None
        state["true_attempted"] += 1
        state["inflight"] = key
        write_json(state_path, state, self_hash=True)
        forward_started = time.monotonic()
        try:
            audit = validate_map(mapping, frame_count=frames.shape[0], valid_frame_count=request["valid_frame_count"])
            rendered, render_audit = render_map(frames, mapping)
            visual = forward_video(crop_frames(rendered, request["score_box_xyxy"]), scorer, torch, batch_size=int(request["batch_size"]))
            metrics = score_embeddings(visual, audio, frame_count=frames.shape[0],
                                       valid_frame_count=int(request["valid_frame_count"]),
                                       audio_sample_count=int(calibration["natural_audio"]["sample_count"]), torch=torch)
            embedding_path = Path(request["embedding_dir"]) / f"v2_true_{key}.npz"
            embedding_sha = _save_embeddings(embedding_path, visual=visual)
            metrics.pop("distance_matrix", None)
            row = {"candidate_id": key, "label": label, "stage": stage, "score_kind": "true_fixed_crop",
                   "map": dict(mapping), "map_sha256": key, "metrics": metrics,
                   "regularization": audit["regularization"], "rendered_frames_sha256": hashlib.sha256(np.ascontiguousarray(rendered).tobytes()).hexdigest(),
                   "embedding_path": str(embedding_path.resolve()), "embedding_sha256": embedding_sha,
                   "scorer_metadata": model_meta, "render_audit": {"blended_fraction": render_audit["blended_fraction"]}}
            state["true_candidates"][key] = row
            state["true_completed"] += 1
            state["true_forward_seconds"] = float(state.get("true_forward_seconds", 0.0)) + time.monotonic() - forward_started
            state["inflight"] = None
            write_json(state_path, state, self_hash=True)
            return row
        except Exception:
            state["true_failed"] += 1
            state["true_forward_seconds"] = float(state.get("true_forward_seconds", 0.0)) + time.monotonic() - forward_started
            state["inflight"] = None
            write_json(state_path, state, self_hash=True)
            raise

    roots = [("identity", identity["map"])] + [(f"global_seed_{amount:+d}", global_seed(frames.shape[0], int(request["valid_frame_count"]), amount)) for amount in (-3, -2, -1, 1, 2, 3)]
    for label, mapping in roots:
        row = proxy_one(mapping, stage="A", parent=None, move=label)
        if row is None:
            break
        if mapping["map_sha256"] not in state["root_archive"]:
            state["root_archive"].append(mapping["map_sha256"])
        if label != "identity":
            true_one(mapping, stage="ROOT", label=label)
    _flush_proxy(state, [], pending, state_path)
    if len(state["root_archive"]) != 7:
        save()
        raise ProtocolError("V2_GLOBAL_ROOTS_INCOMPLETE")

    def beam_phase(phase: str, parents: list[Mapping[str, Any]], stage: str) -> None:
        nonlocal proxy_deadline
        if state["phase"] not in {phase, "roots", "shortlist_a_done", "shortlist_b_done"}:
            return
        state["phase"] = phase
        # Protect both true shortlists and local true refinement from the
        # potentially expensive archive-wide proxy search.
        phase_seconds = float(search_cfg["stage_a_wall_deadline_seconds" if stage == "A" else "stage_b_wall_deadline_seconds"])
        proxy_deadline = min(deadline, started + max(0.0, phase_seconds - float(state["elapsed_search_seconds"])))
        beam = parents
        restart_keys = {str(row["map_sha256"]) for row in parents}
        for step_index, step in enumerate(search_cfg["steps"]):
            for round_index in range(int(search_cfg["rounds_per_step"])):
                cursor = state["cursor"]
                if cursor.get("phase") == phase and (step_index, round_index) < (int(cursor.get("step", 0)), int(cursor.get("round", 0))):
                    beam = [proxy[key] for key in state.get("beam", []) if key in proxy]
                    continue
                if time.monotonic() >= proxy_deadline:
                    state.setdefault("phase_time_limits", {})[phase] = "TIME_RESERVED_FOR_TRUE_RERANK"
                    state["phase"] = phase + "_done"
                    write_json(state_path, state, self_hash=True)
                    return
                for parent_key, move, mapping in _iter_round(beam, float(step)):
                    if proxy_one(mapping, stage=stage, parent=parent_key, move=move) is None:
                        break
                _flush_proxy(state, [], pending, state_path)
                available = ([row for row in proxy.values() if row["stage"] == "B" or row["map_sha256"] in restart_keys]
                             if stage == "B" else list(proxy.values()))
                beam = choose_routes(available, support, counts=(8, 8, 8), proxy=True)
                state["beam"] = [row["map_sha256"] for row in beam]
                state["cursor"] = {"phase": phase, "step": step_index, "round": round_index + 1}
                write_json(state_path, state, self_hash=True)
                if int(state["stage_a_proxy_count" if stage == "A" else "stage_b_proxy_count"]) >= int(search_cfg["stage_a_proxy_maps" if stage == "A" else "stage_b_proxy_maps"]):
                    state["phase"] = phase + "_done"
                    write_json(state_path, state, self_hash=True)
                    return
        state["phase"] = phase + "_done"
        write_json(state_path, state, self_hash=True)

    if state["phase"] in {"roots", "stage_a"}:
        beam_phase("stage_a", [proxy[key] for key in state["root_archive"]], "A")
    if state["phase"] == "stage_a_done":
        shortlist = choose_routes(list(proxy.values()), support, counts=(8, 8, 9), proxy=True,
                                  exclude=set(state["true_candidates"]))
        state["shortlist_a"] = [row["map_sha256"] for row in shortlist]
        write_json(state_path, state, self_hash=True)
        for row in shortlist:
            if true_one(row["map"], stage="SHORTLIST_A", label="shortlist_a") is None:
                break
        state["phase"] = "shortlist_a_done"
        write_json(state_path, state, self_hash=True)
    if state["phase"] in {"shortlist_a_done", "stage_b"}:
        true_rows = list(state["true_candidates"].values())
        restart = [proxy[key] for key in state["root_archive"]]
        for row in sorted(true_rows, key=lambda r: _d0_key(r, proxy=False))[:4] + sorted(true_rows, key=lambda r: _c_key(r, proxy=False))[:4]:
            if row["map_sha256"] in proxy and row["map_sha256"] not in {r["map_sha256"] for r in restart}:
                restart.append(proxy[row["map_sha256"]])
        beam_phase("stage_b", restart, "B")
    if state["phase"] == "stage_b_done":
        shortlist = choose_routes([r for r in proxy.values() if r["stage"] == "B"], support, counts=(8, 8, 9), proxy=True,
                                  exclude=set(state["true_candidates"]))
        state["shortlist_b"] = [row["map_sha256"] for row in shortlist]
        write_json(state_path, state, self_hash=True)
        for row in shortlist:
            if true_one(row["map"], stage="SHORTLIST_B", label="shortlist_b") is None:
                break
        state["phase"] = "shortlist_b_done"
        write_json(state_path, state, self_hash=True)
    if state["phase"] in {"shortlist_b_done", "local"}:
        state["phase"] = "local"
        proxy_deadline = deadline
        for round_index in range(int(state.get("local_round", 0)), 4):
            actual = list(state["true_candidates"].values())
            starts = sorted(actual, key=lambda r: _d0_key(r, proxy=False))[:2] + sorted(actual, key=lambda r: _c_key(r, proxy=False))[:2]
            proposals = []
            for step in (0.25, 0.5, 1.0):
                proposals.extend(_iter_round(starts, step))
            local = {}
            unscored = {}
            for parent_key, move, mapping in proposals:
                key = mapping["map_sha256"]
                if key in state["true_candidates"] or key in local or key in unscored:
                    continue
                row = proxy_one(mapping, stage="LOCAL", parent=parent_key, move=move)
                if row is not None:
                    local[key] = row
                else:
                    unscored[key] = {"map": mapping, "map_sha256": key, "proxy_metrics": None}
            _flush_proxy(state, [], pending, state_path)
            shortlisted = choose_routes(list(local.values()), support, counts=(4, 3, 3), proxy=True)
            if len(shortlisted) < 10:
                shortlisted.extend(sorted(unscored.values(), key=lambda row: row["map_sha256"])[:10 - len(shortlisted)])
            for row in shortlisted:
                if true_one(row["map"], stage=f"LOCAL_{round_index}", label="local_refine") is None:
                    break
            state["local_round"] = round_index + 1
            write_json(state_path, state, self_hash=True)
            if not shortlisted or time.monotonic() >= deadline:
                break
        state["phase"] = "local_done"
    true_rows = list(state["true_candidates"].values())
    selected = select_final(true_rows, baseline, search_cfg) or identity
    globals_true = [r for r in true_rows if str(r.get("label", "")).startswith("global_seed_")]
    global_control = min(globals_true, key=lambda r: _d0_key(r, proxy=False)) if globals_true else identity
    best_d0 = min(true_rows, key=lambda r: _d0_key(r, proxy=False))
    best_c = min(true_rows, key=lambda r: (-float(r["metrics"]["sync_c"]), float(r["metrics"]["d0"]), str(r["map_sha256"])))
    aligned = [r for r in true_rows if abs(int(r["metrics"]["offset"])) <= 1]
    best_aligned = min(aligned, key=lambda r: _c_key(r, proxy=False)) if aligned else None
    termination = ("COMPLETE" if state["phase"] == "local_done" and time.monotonic() < deadline
                   and not state.get("phase_time_limits") else "BUDGET_LIMITED")
    if selected["map_sha256"] == identity["map_sha256"]:
        selection_status = "NO_ACCEPTABLE_WARP"
    elif float(selected["metrics"]["sync_c"]) >= float(baseline["sync_c"]) + float(search_cfg["c_gain_label_threshold"]):
        selection_status = "ALIGNMENT_AND_C_GAIN"
    else:
        selection_status = "ALIGNMENT_GAIN"
    state["final_gate"] = {r["map_sha256"]: final_reasons(r, baseline, search_cfg) for r in true_rows}
    state["status"] = "SEARCH_COMPLETE" if termination == "COMPLETE" else "BUDGET_LIMITED"
    state["termination"] = termination
    state["selection_status"] = selection_status
    state["selected_map_sha256"] = selected["map_sha256"]
    state["elapsed_search_seconds"] = float(state["elapsed_search_seconds"]) + time.monotonic() - started
    state["active_search_started_at"] = None
    write_json(state_path, state, self_hash=True)
    result = {"schema_version": 2, "mode": "search_v2", "protocol": "mfa_linear_video_retiming_v2",
              "engine": "embedding_beam_v2", "status": "BUDGET_LIMITED" if termination != "COMPLETE" else selection_status,
              "termination": termination, "selection_status": selection_status,
              "sample_id": str(request["sample_id"]), "paired_key": str(request["paired_key"]), "portrait_id": str(request["portrait_id"]),
              "baseline": baseline, "selected": selected, "global_control": global_control,
              "best_attempt": best_aligned or best_d0, "best_d0": best_d0, "best_c": best_c, "best_aligned": best_aligned,
              "state_path": str(calibration_path.resolve()), "state_v2_path": str(state_path.resolve()),
              "state_v2_sha256": file_sha256(state_path),
              "search_result": {"actual_score_calls": int(state["true_attempted"]),
                                "true_completed": int(state["true_completed"]), "proxy_unique_maps": int(state["proxy_completed"]),
                                "candidate_count": len(true_rows), "elapsed_search_seconds": float(state["elapsed_search_seconds"]),
                                "proxy_scoring_seconds": float(state.get("proxy_scoring_seconds", 0.0)),
                                "true_forward_seconds": float(state.get("true_forward_seconds", 0.0))}}
    write_json(request["result_path"], result, self_hash=True)
    return result
