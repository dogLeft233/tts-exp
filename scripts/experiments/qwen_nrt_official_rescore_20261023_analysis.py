"""CPU-only paired analysis of the frozen 40-speaker official rescore.

Consumes all planned score receipts, retaining incomplete tuples and failures.
Never selects a track, changes media, or starts a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

ROLES = ("N", "R", "T1", "T2", "T3")
CONTRASTS = ("T-N", "T-R", "R-N")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def exclusive_json(path, obj):
    with Path(path).open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def summarize(values, seed=20261023, replicates=10000):
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1 or not np.isfinite(x).all():
        raise ValueError("Finite, one-dimensional speaker contrasts required")
    if len(x) == 0:
        return {"n": 0, "mean": None, "ci95": None, "ci97_5": None,
                "positive": 0, "negative": 0, "zero": 0, "values": []}
    idx = np.random.Generator(np.random.PCG64(seed)).integers(0, len(x), (replicates, len(x)))
    boot = x[idx].mean(axis=1)
    return {"n": len(x), "mean": float(x.mean()),
            "ci95": np.quantile(boot, [0.025, 0.975]).tolist(),
            "ci97_5": np.quantile(boot, [0.0125, 0.9875]).tolist(),
            "positive": int((x > 0).sum()), "negative": int((x < 0).sum()),
            "zero": int((x == 0).sum()), "values": x.tolist()}


def contrasts(roles, t_roles=("T1", "T2", "T3")):
    if not t_roles:
        raise ValueError("No qualifying T draw; do not impute")
    t = float(np.mean([roles[r] for r in t_roles]))
    return {"T-N": t - roles["N"], "T-R": t - roles["R"],
            "R-N": roles["R"] - roles["N"]}


def analyze(run: Path, source: Path, protocol: dict):
    old = read(source / "analysis.json")
    videos = read(source / "video_manifest.json")["records"]
    plan = {(r["sample_id"], r["role"]): r for r in videos}
    if len(plan) != 200:
        raise ValueError("Expected exactly 200 distinct planned cells")
    metrics = {"C": "sync_c", "D": "sync_d", "offset": "av_offset"}
    scores, failures, unplanned = {}, [], []
    for path in sorted((run / "scores").glob("*.json")):
        row = read(path)
        key = (row["sample_id"], row["role"])
        if key not in plan:
            unplanned.append(str(path))
            continue
        if key in scores:
            raise ValueError(f"Duplicate score {key}")
        # Score presence is explicit; no status-string-dependent silent loss.
        row["finite_first_track"] = row.get("status") == "COMPLETE" and all(isinstance(row.get(f), (int, float))
                                        and np.isfinite(row[f]) for f in metrics.values())
        scores[key] = row
    if unplanned:
        raise ValueError(f"Unplanned score files: {unplanned}")
    for key, planned in sorted(plan.items()):
        if key not in scores or not scores[key]["finite_first_track"]:
            failures.append({"sample_id": key[0], "role": key[1],
                             "status": scores.get(key, {}).get("status", "MISSING_RECEIPT"),
                             "cohort": planned["cohort"]})
    records, incomplete = [], []
    for prior in sorted(old["records"], key=lambda r: r["sample_id"]):
        sid = prior["sample_id"]
        missing = [r for r in ROLES if (sid, r) not in scores or not scores[sid, r]["finite_first_track"]]
        if missing:
            incomplete.append({"sample_id": sid, "speaker_id": prior["speaker_id"],
                               "cohort": prior["cohort"], "missing_roles": missing})
            continue
        qualified = [r for r in ROLES[2:] if prior["draw_reliability"][r]["qualified"]]
        row = {"sample_id": sid, "speaker_id": prior["speaker_id"], "cohort": prior["cohort"],
               "qualified_t_roles": qualified, "official": {}, "official_contrasts": {},
               "qualified_contrasts": {}, "old_common_contrasts": {}, "protocol_changes_C": {},
               "track_counts": {r: len(scores[sid, r].get("tracks", [])) for r in ROLES}}
        for metric, field in metrics.items():
            role_values = {r: float(scores[sid, r][field]) for r in ROLES}
            row["official"][metric] = role_values
            row["official_contrasts"][metric] = contrasts(role_values)
            row["qualified_contrasts"][metric] = contrasts(role_values, qualified) if qualified else None
        for policy, vals in prior["policies"].items():
            previous = contrasts({r: float(vals[r]["C"]) for r in ROLES})
            row["protocol_changes_C"][policy] = {
                k: row["official_contrasts"]["C"][k] - previous[k] for k in CONTRASTS}
            if policy == "common":
                row["old_common_contrasts"] = previous
        records.append(row)
    if len({r["speaker_id"] for r in records}) != len(records):
        raise ValueError("Speakers must be independent across tuples")
    cohorts = {}
    for name in ("new30", "all40", "pilot10"):
        rs = [r for r in records if name == "all40" or r["cohort"] == name]
        block = {"n": len(rs), "speaker_ids": [r["speaker_id"] for r in rs],
                 "metrics": {}, "qualified_supplement": {}, "protocol_changes_C": {},
                 "per_seed_C": {}}
        for metric in metrics:
            block["metrics"][metric] = {
                "role_means": {r: float(np.mean([x["official"][metric][r] for x in rs]))
                               if rs else None for r in ROLES},
                "contrasts": {k: summarize([x["official_contrasts"][metric][k] for x in rs],
                                             protocol["bootstrap"]["seed"], protocol["bootstrap"]["replicates"])
                              for k in CONTRASTS}}
            block["qualified_supplement"][metric] = {
                k: summarize([x["qualified_contrasts"][metric][k] for x in rs
                              if x["qualified_contrasts"][metric] is not None],
                             protocol["bootstrap"]["seed"], protocol["bootstrap"]["replicates"])
                for k in CONTRASTS}
        for policy in ("common", "native_full", "equal_count_spread"):
            block["protocol_changes_C"][policy] = {
                k: summarize([x["protocol_changes_C"][policy][k] for x in rs],
                             protocol["bootstrap"]["seed"], protocol["bootstrap"]["replicates"])
                for k in CONTRASTS}
        for role in ROLES[2:]:
            block["per_seed_C"][role] = {
                k: summarize([contrasts(x["official"]["C"], (role,))[k] for x in rs],
                             protocol["bootstrap"]["seed"], protocol["bootstrap"]["replicates"])
                for k in ("T-N", "T-R")}
        cohorts[name] = block
    return {"status": "ANALYZED_COMPLETE_COVERAGE" if not failures else "INCOMPLETE_COVERAGE_DESCRIPTIVE",
            "analysis_protocol_sha256": sha(run / "analysis_protocol.json"),
            "source_analysis_sha256": sha(source / "analysis.json"),
            "source_video_manifest_sha256": sha(source / "video_manifest.json"),
            "analysis_source_sha256": sha(__file__), "protocol": protocol,
            "coverage": {"planned_cells": len(plan), "score_receipts": len(scores),
                         "finite_cells": sum(r["finite_first_track"] for r in scores.values()),
                         "complete_speakers": len(records), "failures": failures,
                         "incomplete_speakers": incomplete,
                         "multi_track_cells": [{"sample_id": k[0], "role": k[1],
                                                "track_count": len(v.get("tracks", []))}
                                               for k, v in sorted(scores.items()) if len(v.get("tracks", [])) > 1]},
            "records": records, "cohorts": cohorts,
            "interpretation": "Same-media evaluation-chain comparison; first printed official track. No human quality or pure LM causal inference; no equivalence claim from intervals spanning zero."}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    args = p.parse_args()
    protocol = read(args.run / "analysis_protocol.json")
    result = analyze(args.run, args.source, protocol)
    exclusive_json(args.run / "analysis.json", result)
    with (args.run / "speaker_contrasts.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample_id", "speaker_id", "cohort", "T-N", "T-R", "R-N", "official_minus_old_common_T-N", "official_minus_old_common_T-R"])
        for r in result["records"]:
            writer.writerow([r["sample_id"], r["speaker_id"], r["cohort"],
                             *[r["official_contrasts"]["C"][k] for k in CONTRASTS],
                             *[r["protocol_changes_C"]["common"][k] for k in ("T-N", "T-R")]])
    print(json.dumps({"status": result["status"], "coverage": result["coverage"],
                      "new30_C": result["cohorts"]["new30"]["metrics"]["C"]["contrasts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
