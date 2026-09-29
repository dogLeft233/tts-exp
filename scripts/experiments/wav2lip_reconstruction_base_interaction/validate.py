from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.experiments import wav2lip_probe_runtime as rt
from scripts.experiments import wav2lip_probe_validator_common as vc

from . import config


def _factor(sid: str, natural: np.ndarray, d_rows: list[dict], masks: dict[str, dict]) -> dict[str, np.ndarray]:
    rows = [row for row in d_rows if str(row["sample_id"]) == sid and str(row.get("condition")) in config.CONDITIONS]
    if len(rows) != 9: raise rt.ProtocolError("wrong D factor count")
    sets = []
    for row in rows:
        current = set()
        for used in row["used_masks"]:
            item = masks.get(str(used["mask_sha256"])); s, e = int(used["global_start_frame"]), int(used["global_end_frame"])
            if item is None or (s, e) != (int(item["natural_core_start_frame"]), int(item["natural_core_end_frame"])): raise rt.ProtocolError("D mask identity mismatch")
            current.update(range(s, e))
        sets.append(current)
    if any(value != sets[0] for value in sets[1:]): raise rt.ProtocolError("D K differs")
    by = {(int(row["seed"]), str(row["condition"])): np.asarray(np.load(Path(str(row["path"])), allow_pickle=False), dtype=np.float64) for row in rows}
    if any(value.shape != (80, 308) for value in by.values()): raise rt.ProtocolError("D array shape mismatch")
    seeds = (20260901, 20260902, 20260903); m = np.asarray(natural, dtype=np.float64); z = np.mean(np.stack([by[(seed, "NAT_ONLY")] for seed in seeds]), axis=0); c = np.mean(np.stack([by[(seed, "PAIRED_TTS")] - by[(seed, "NAT_ONLY")] for seed in seeds]), axis=0); w = np.mean(np.stack([by[(seed, "SAME_PHONE_WRONG_INSTANCE")] - by[(seed, "NAT_ONLY")] for seed in seeds]), axis=0); k = sorted(sets[0]); outside = np.ones(308, dtype=bool); outside[k] = False
    if np.max(np.abs(c[:, outside])) > 1e-6 or np.max(np.abs(w[:, outside])) > 1e-6 or np.max(np.abs((z - m)[:, outside])) > 1e-6: raise rt.ProtocolError("factor residual outside K")
    w *= np.linalg.norm(c[:, k]) / np.linalg.norm(w[:, k]); a = min(0.25, 0.5 / max(float(np.max(np.abs(c))), float(np.max(np.abs(w))))); c *= a; w *= a; base_residual = z - m; b = min(0.25, 0.5 / float(np.max(np.abs(base_residual)))); base = np.clip(m + b * base_residual, -4, 4)
    return {"N": m.astype(np.float32), "N_CONTENT": np.clip(m + c, -4, 4).astype(np.float32), "BASE": base.astype(np.float32), "BASE_CONTENT": np.clip(base + c, -4, 4).astype(np.float32), "BASE_WRONG": np.clip(base + w, -4, 4).astype(np.float32)}


def _pair(records: list[dict], left: str, right: str) -> dict:
    temp = [{"sample_id": row["sample_id"], "source_group": row["source_group"], "N": row[right], "LEFT": row[left]} for row in records]; return vc.contrast_summary(temp, "LEFT")


def _interaction(records: list[dict]) -> dict:
    values = []; groups = [str(row["source_group"]) for row in records]
    for row in records:
        k = row["N"]["min_index"]; values.append((row["BASE"]["curve"][k] - row["BASE_CONTENT"]["curve"][k]) - (row["N"]["curve"][k] - row["N_CONTENT"]["curve"][k]))
    labels = sorted(set(groups)); indices = vc.bootstrap_indices(labels); stats = vc.grouped_stats(values, groups, indices); joint = int(sum(float(np.mean([value for value, group in zip(values, groups, strict=True) if group == label])) > 0 for label in labels)); return {"values": values, "stats": stats, "joint_positive_count": joint, "signal": bool(stats["mean"] > 0.05 and stats["ci99"][0] > 0 and joint >= 7)}


def _worker_metrics(score_row: dict, label: str) -> dict:
    visual, audio, cached = rt.load_worker_arrays(score_row)
    rebuilt = vc.matrix_from_embeddings(visual, audio)
    if rebuilt.shape != cached.shape or float(np.max(np.abs(rebuilt.astype(np.float64) - cached.astype(np.float64)))) > 1e-4:
        raise rt.ProtocolError(f"independent SyncNet matrix mismatch: {label}")
    return vc.score_metrics(cached)


def validate(root: Path) -> dict:
    paths = config.RunPaths(root); rt.load_self(paths.protocol); saved = rt.load_self(paths.drivers); scores = rt.load_self(paths.scores); p = rt.load_self(config.P_DRIVERS); control = rt.load_self(config.P_CONTROL); q = rt.load_self(config.Q_SCORES); d = rt.read_json(config.D_DRIVERS); masks = {str(item["mask_sha256"]): item for item in rt.read_json(config.D_MASKS)["masks"]}; p_by_id = {str(row["sample_id"]): row for row in p["rows"]}; score_map = {(str(row["sample_id"]), str(row["video_arm"])): row for row in scores["rows"]}; q_map = {str(row["sample_id"]): row for row in q["rows"] if str(row["video_arm"]) == "CORRECT" and str(row["audio_arm"]) == "N"}
    if rt.file_sha256(config.D_RECONSTRUCTION) != config.D_RECONSTRUCTION_SHA256:
        raise rt.ProtocolError("D reconstruction binding changed")
    if len(saved["rows"]) != 16 or len(scores["rows"]) != 52: raise rt.ProtocolError("E4 record or score count mismatch")
    candidate_rows = [row for row in scores["rows"] if str(row["video_arm"]) in {"BASE", "BASE_CONTENT", "BASE_WRONG"}]
    fresh_rows = [row for row in scores["rows"] if str(row["video_arm"]).startswith("FRESH_")]
    if len(candidate_rows) != 48 or len(fresh_rows) != 4:
        raise rt.ProtocolError("E4 candidate/control score split mismatch")
    if {str(row["video_arm"]) for row in fresh_rows} != {"FRESH_N_REPEAT", "FRESH_PARITY_V_N", "FRESH_PARITY_V_DELAY_200"}:
        raise rt.ProtocolError("E4 fresh control cells missing")
    vc.validate_fresh_controls(fresh_rows, rt.load_self(config.P_CONTROL)["rows"])
    score_map = {(str(row["sample_id"]), str(row["video_arm"])): row for row in candidate_rows}
    if len(score_map) != 48:
        raise rt.ProtocolError("E4 duplicate or missing candidate score cell")
    records = []
    for row in saved["rows"]:
        sid = str(row["sample_id"]); expected = _factor(sid, rt.load_mel(Path(str(p_by_id[sid]["arms"]["N"]["path"]))), d["drivers"], masks)
        for arm in ("N", "N_CONTENT", "BASE", "BASE_CONTENT", "BASE_WRONG"):
            actual = rt.load_mel(Path(str(row["arms"][arm]["path"])))
            if not np.array_equal(actual, expected[arm]): raise rt.ProtocolError(f"factor array mismatch: {sid}/{arm}")
        nrow = next(item for item in control["rows"] if str(item["sample_id"]) == sid and str(item["video_arm"]) == "N" and str(item["audio_arm"]) == "N"); qrow = q_map[sid]; item = {"sample_id": sid, "source_group": str(row["source_group"]), "N": _worker_metrics(nrow, f"{sid}/N"), "N_CONTENT": _worker_metrics(qrow, f"{sid}/N_CONTENT")}
        for arm in ("BASE", "BASE_CONTENT", "BASE_WRONG"):
            item[arm] = _worker_metrics(score_map[(sid, arm)], f"{sid}/{arm}")
        records.append(item)
    pairs = {(left, right): _pair(records, left, right) for left, right in (("BASE", "N"), ("BASE_CONTENT", "N"), ("BASE_WRONG", "N"), ("BASE_CONTENT", "BASE"), ("BASE_CONTENT", "BASE_WRONG"))}; interaction = _interaction(records); main_pass = vc.gain_pass(pairs[("BASE_CONTENT", "N")]); mechanism_pass = bool(interaction["signal"] and vc.gain_pass(pairs[("BASE_CONTENT", "BASE_WRONG")])) ; decision = "NO_BASE_CONTENT_GAIN_ESTABLISHED" if not main_pass else "BASE_DEPENDENT_CONTENT_SIGNAL_TO_CONFIRM" if mechanism_pass else "NATURAL_GAIN_MECHANISM_UNRESOLVED"
    analysis = rt.load_self(paths.analysis)
    if analysis.get("contrasts") != {f"{left}_vs_{right}": value for (left, right), value in pairs.items()} or analysis.get("interaction") != interaction or analysis.get("scientific_decision") != decision: raise rt.ProtocolError("E4 analysis mismatch or tamper")
    return rt.write_json(paths.validation, {"schema_version": 1, "protocol_id": "wav2lip_reconstruction_base_interaction", "status": "PASS", "independent": True, "engineering_decision": "GO", "scientific_decision": decision, "contrasts": analysis["contrasts"], "interaction": interaction, "candidate_authorized": False, "checks": ["independent D seed fusion", "independent factorial construction", "independent SyncNet matrix metrics", "absolute and interaction decisions separated"]})


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--run-root", type=Path, required=True); args = parser.parse_args()
    try: validate(args.run_root); return 0
    except Exception as exc: rt.write_json(args.run_root / "validation.json", {"schema_version": 1, "protocol_id": "wav2lip_reconstruction_base_interaction", "status": "FAIL", "engineering_decision": "BLOCKED", "error": f"{type(exc).__name__}: {exc}"}); return 1


if __name__ == "__main__": raise SystemExit(main())
