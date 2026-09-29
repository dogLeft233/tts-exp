"""Frozen annotation-matched negative-pool intervention on cached embeddings.

No rendering, training, embedding interpolation, or evaluation-lag selection.
The event margin is explicitly distinct from official SyncNet confidence.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np

from scripts.experiments.tts_independent_visual import OUT, PARENT, ROOT, read, sha, write
from scripts.experiments.tts_pcm_residual import cluster
from scripts.experiments.tts_phone_video_transfer import phones
from scripts.experiments.tts_prepost_mel import BASE as SOURCE

BASE = ROOT / "runs/tts_negative_pool_20260926"
ARMS = ("N", "T")
PHASES = (0.25, 0.5, 0.75)
ACENTER = 0.1075  # 20 MFCC frames, 10 ms stride, 25 ms frame width.
KINDS = ("same_occurrence", "same_phone_other", "different_phone", "silence_gap")


def freeze():
    source = read(SOURCE / "protocol.json")
    calibration = [r for r in read(OUT / "protocol.json")["rows"] if r["split"] == "calibration"]
    assert not {r["id"] for r in calibration} & {r["id"] for r in source["rows"]}
    offsets = []
    for r in calibration:
        path = PARENT / "scores" / (r["id"] + ".json")
        score = read(path)
        offsets.append({"id": r["id"], "speaker": r["speaker"], "sha256": sha(path),
                        **{a: score["cells"][a + "_source"]["20"]["offset"] for a in ARMS}})
    median = float(np.median([r[a] for r in offsets for a in ARMS]))
    assert median.is_integer()
    p = {
        "protocol": "tts_negative_pool_v1", "source_sha256": sha(SOURCE / "protocol.json"),
        "annotation_manifest_sha256": sha(OUT / "phone_transfer/manifest.json"),
        "split_sha256": sha(OUT / "protocol.json"), "calibration": offsets,
        "fixed_audio_minus_visual_index": int(-median), "phases": list(PHASES),
        "audio_window_center_seconds": ACENTER, "visual_window_center_seconds": 0.08,
        "sampling": "nearest grid half-up, <=20ms error and same interval; both durations >=80ms; no interpolation",
        "query": "common speech occurrence/phase, visual guard20 in both arms",
        "donor": "common speech/gap occurrence/phase; absolute audio index displacement 3..15 in BOTH arms; exclude own node",
        "gaps": "positive gaps between consecutive speech occurrences; no leading/trailing-gap donors",
        "support": "query >=5 common donors; utterance >=8 queries and >=5 distinct speech occurrences; image-invariant support",
        "native_pool": "on identical queries, audio indices positive_index + [-15..-3,3..15], equal weight; no zero padding",
        "matched_pool": "equal weight per common donor occurrence/phase, then equal query weight; duplicate nearest indices retained and counted",
        "primary": "raw AV mean-negative minus fixed-positive margin: T-N native, matched and matched-minus-native difference in differences",
        "secondary": "conditional rank, unit-norm geometry, A/A and V/V distances; same occurrence/same phone other/different phone/gap strata",
        "statistics": "3 images mean within utterance; utterance means within speaker; 20000 bootstrap seed20260926; 95/99 CI",
        "limitations": "annotation event identity not physical truth; phone not viseme; 200ms windows cross phones; calibration IDs disjoint but same speakers; common static-image lag transfer assumed",
        "source_code_sha256": sha(__file__),
    }
    if (BASE / "protocol.json").exists():
        assert read(BASE / "protocol.json") == p
    write(BASE / "protocol.json", p)
    print("frozen k", -median, "calibration", len(calibration))


def category(query, donor):
    if donor["phone"] == "__gap__":
        return "silence_gap"
    if query["event"] == donor["event"]:
        return "same_occurrence"
    if query["phone"] == donor["phone"]:
        return "same_phone_other"
    return "different_phone"


def make_support(row, grids, lengths, k):
    ph = {a: phones(grids[a]["path"]) for a in ARMS}
    assert [p[0] for p in ph["N"]] == [p[0] for p in ph["T"]]
    events = []
    for n, p in enumerate(ph["N"]):
        events.append({"event": f"phone_{n}", "phone": p[0], "span": {a: list(ph[a][n][1:]) for a in ARMS}})
        if n + 1 < len(ph["N"]):
            events.append({"event": f"gap_{n}", "phone": "__gap__", "span": {a: [ph[a][n][2], ph[a][n + 1][1]] for a in ARMS}})
    nodes, rejected = [], Counter()
    for e in events:
        if any(e["span"][a][1] - e["span"][a][0] < 0.08 - 1e-9 for a in ARMS):
            rejected["event_duration"] += 1
            continue
        for phase in PHASES:
            indices, errors = {}, {}
            valid = True
            for a in ARMS:
                lo, hi = e["span"][a]
                target = lo + phase * (hi - lo)
                j = int(np.floor((target - ACENTER) * 25 + 0.5))
                time = j / 25 + ACENTER
                indices[a], errors[a] = j, time - target
                valid &= lo <= time < hi and abs(time - target) <= 0.02000001
                valid &= 0 <= j < lengths[a] and 0 <= j - k < lengths[a]
            if valid:
                nodes.append({**e, "phase": phase, "j": indices, "errors": errors})
            else:
                rejected["node_grid_or_extent"] += 1
    queries = []
    for qn, q in enumerate(nodes):
        if q["phone"] == "__gap__":
            continue
        if not all(20 <= q["j"][a] - k < lengths[a] - 20 for a in ARMS):
            rejected["query_guard"] += 1
            continue
        donors = [dn for dn, d in enumerate(nodes) if dn != qn and all(3 <= abs(d["j"][a] - q["j"][a]) <= 15 for a in ARMS)]
        if len(donors) < 5:
            rejected["query_donors"] += 1
            continue
        assert all(0 <= q["j"][a] - 15 and q["j"][a] + 15 < lengths[a] for a in ARMS)
        queries.append({"node": qn, "donors": donors})
    occurrences = {nodes[q["node"]]["event"] for q in queries}
    return {"id": row["id"], "speaker": row["speaker"], "nodes": nodes, "queries": queries,
            "events": events, "rejected": dict(rejected), "lengths": lengths,
            "n_occurrences": len(occurrences), "eligible": len(queries) >= 8 and len(occurrences) >= 5}


def support():
    p = read(BASE / "protocol.json")
    source = read(SOURCE / "protocol.json")
    assert sha(SOURCE / "protocol.json") == p["source_sha256"]
    annotation = {r["id"]: r for r in read(OUT / "phone_transfer/manifest.json")["rows"]}
    rows, assets = [], {}
    for r in source["rows"]:
        sid = r["id"]
        grids = annotation[sid]["textgrids"]
        for g in grids.values():
            assert sha(g["path"]) == g["sha256"]
            assets[g["path"]] = g["sha256"]
        lens = []
        af = PARENT / "features" / sid / "features.npz"
        for im in source["images"]:
            f = SOURCE / "scores" / im["id"] / f"{sid}.json"
            receipt = read(f)
            vf = f.with_name(sid + "_visual.npz")
            for path, expected in [(af, receipt["audio_features_sha256"]), (vf, receipt["visual_sha256"]), (f.with_suffix(".npz"), receipt["matrix_sha256"])]:
                assert sha(path) == expected
                assets[str(path)] = expected
            assets[str(f)] = sha(f)
            m = np.load(f.with_suffix(".npz"))
            lens.append({a: len(m[a]) for a in ARMS})
        assert lens[0] == lens[1] == lens[2]
        rows.append(make_support(r, grids, lens[0], p["fixed_audio_minus_visual_index"]))
    write(BASE / "support.json", {"rows": rows, "assets": assets})
    print("support", sum(r["eligible"] for r in rows), "/", len(rows),
          "queries", sum(len(r["queries"]) for r in rows if r["eligible"]))


def dist(x, y):
    # Official epsilon direction with double accumulation for independent replay.
    return np.sqrt(np.sum((x.astype(np.float64) - y.astype(np.float64) + 1e-6) ** 2, axis=-1))


def norm(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def rank(neg, pos):
    return float(np.mean((neg > pos) + 0.5 * (neg == pos)))


def annotate_raw(s, q, j, arm):
    t = j / 25 + ACENTER
    for e in s["events"]:
        if e["span"][arm][0] <= t < e["span"][arm][1]:
            return category(q, e)
    return "silence_gap"


def evaluate():
    p = read(BASE / "protocol.json")
    source, support = read(SOURCE / "protocol.json"), read(BASE / "support.json")
    k = p["fixed_audio_minus_visual_index"]
    cells, replay_max = [], 0.0
    for s in support["rows"]:
        sid = s["id"]
        audio = np.load(PARENT / "features" / sid / "features.npz")
        for im in source["images"]:
            file = SOURCE / "scores" / im["id"] / f"{sid}.json"
            matrices = np.load(file.with_suffix(".npz"))
            visual = np.load(file.with_name(sid + "_visual.npz"))
            cell = {"id": sid, "speaker": s["speaker"], "image": im["id"], "eligible": s["eligible"], "arms": {}}
            for arm in ARMS:
                n = len(matrices[arm])
                a, v = audio["a_" + arm + "_source"][:n], visual[arm][:n]
                pad = np.pad(a, ((15, 15), (0, 0)))
                rebuilt = np.stack([dist(v, pad[d:d+n]) for d in range(31)], 1)
                replay_max = max(replay_max, float(np.max(np.abs(rebuilt - matrices[arm]))))
                official = {}
                for scope, ix in [("full", np.arange(20, n-20)), ("queries", np.array([s["nodes"][q["node"]]["j"][arm] - k for q in s["queries"]], dtype=int))]:
                    if len(ix):
                        curve = rebuilt[ix].mean(0)
                        negative_columns = [z for z in range(31) if abs(z-15-k)>=3]
                        official[scope] = {"C": float(np.median(curve)-curve.min()), "B": float(np.median(curve)), "D": float(curve.min()), "fixed_D": float(curve[15+k]),
                                           "fixed_median_margin": float(np.median(curve)-curve[15+k]),
                                           "fixed_mean_official_margin": float(curve[negative_columns].mean()-curve[15+k])}
                records = []
                if s["eligible"]:
                    for q in s["queries"]:
                        node = s["nodes"][q["node"]]
                        j = node["j"][arm]
                        i = j - k
                        native = j + np.r_[np.arange(-15, -2), np.arange(3, 16)]
                        matched = np.array([s["nodes"][dn]["j"][arm] for dn in q["donors"]])
                        kinds = [category(node, s["nodes"][dn]) for dn in q["donors"]]
                        record = {"node": q["node"], "pools": {}}
                        for pool, inds, labels in [("native", native, [annotate_raw(s, node, int(d), arm) for d in native]), ("matched", matched, kinds)]:
                            data = {"count": len(inds), "time_abs": float(np.mean(abs(inds-j))/25), "composition": {c: labels.count(c)/len(labels) for c in KINDS}, "metrics": {}}
                            for geometry, av, vv in [("raw", a, v), ("unit", norm(a), norm(v))]:
                                neg, pos = dist(vv[i], av[inds]), float(dist(vv[i], av[j]))
                                aa = dist(av[j], av[inds])
                                # indices are valid for both modalities by construction.
                                vinds = inds - k
                                vdist = dist(vv[i], vv[vinds])
                                m = {"positive": pos, "negative": float(neg.mean()), "margin": float(neg.mean()-pos), "rank": rank(neg, pos), "AA": float(aa.mean()), "VV": float(vdist.mean())}
                                data["metrics"][geometry] = m
                                data.setdefault("strata", {})[geometry] = {
                                    c: {"n": int(sum(z == c for z in labels)), "margin": float(neg[np.array(labels)==c].mean()-pos), "rank": rank(neg[np.array(labels)==c],pos), "AA": float(aa[np.array(labels)==c].mean()), "VV": float(vdist[np.array(labels)==c].mean())}
                                    for c in KINDS if c in labels}
                            record["pools"][pool] = data
                        records.append(record)
                cell["arms"][arm] = {"official": official, "queries": records, "audio_norm": float(np.linalg.norm(a,axis=1).mean()), "visual_norm": float(np.linalg.norm(v,axis=1).mean())}
            cells.append(cell)
        print("evaluated", sid, s["eligible"], len(s["queries"]), flush=True)
    assert replay_max < 1e-5
    write(BASE / "scores.json", cells)
    write(BASE / "replay.json", {"max_abs_error": replay_max, "cells": len(cells)*2, "epsilon": 1e-6})


def summarize():
    cells = read(BASE / "scores.json")
    scalars = {}
    for c in cells:
        key = (c["id"], c["speaker"])
        out = scalars.setdefault(key, {})
        arms = c["arms"]
        for scope in ("full", "queries"):
            if scope == "queries" and not c["eligible"]:
                continue
            for m in ("C", "B", "D", "fixed_D", "fixed_median_margin", "fixed_mean_official_margin"):
                value = arms["T"]["official"][scope][m] - arms["N"]["official"][scope][m]
                out.setdefault("official_"+scope+"_"+m, []).append(value)
        if not c["eligible"]:
            continue
        for geometry in ("raw", "unit"):
            for m in ("positive", "negative", "margin", "rank", "AA", "VV"):
                effects = {}
                for pool in ("native", "matched"):
                    means = {a: np.mean([q["pools"][pool]["metrics"][geometry][m] for q in arms[a]["queries"]]) for a in ARMS}
                    for a in ARMS:
                        out.setdefault(f"level_{a}_{geometry}_{pool}_{m}", []).append(means[a])
                    effects[pool] = means["T"]-means["N"]
                    out.setdefault(f"{geometry}_{pool}_{m}", []).append(effects[pool])
                out.setdefault(f"{geometry}_did_{m}", []).append(effects["matched"]-effects["native"])
            for cat in KINDS:
                for metric in ("margin", "rank", "AA", "VV"):
                    valid = [qn for qn,q in enumerate(arms["N"]["queries"]) if q["pools"]["matched"]["strata"][geometry].get(cat,{}).get("n",0)>=3]
                    if len(valid)>=5:
                        means = {a: np.mean([arms[a]["queries"][qn]["pools"]["matched"]["strata"][geometry][cat][metric] for qn in valid]) for a in ARMS}
                        out.setdefault(f"{geometry}_stratum_{cat}_{metric}", []).append(means["T"]-means["N"])
        for pool in ("native", "matched"):
            for category_name in KINDS:
                means = {a: np.mean([q["pools"][pool]["composition"][category_name] for q in arms[a]["queries"]]) for a in ARMS}
                for a in ARMS:
                    out.setdefault(f"level_{a}_composition_{pool}_{category_name}", []).append(means[a])
                out.setdefault(f"composition_{pool}_{category_name}", []).append(means["T"]-means["N"])
            out.setdefault(f"time_{pool}",[]).append(np.mean([q["pools"][pool]["time_abs"] for q in arms["T"]["queries"]])-np.mean([q["pools"][pool]["time_abs"] for q in arms["N"]["queries"]]))
    names = sorted({name for values in scalars.values() for name in values})
    result = {}
    for name in names:
        selected = [(key, np.mean(values[name])) for key,values in scalars.items() if name in values]
        assert all(len(scalars[key][name]) == 3 for key,_ in selected)
        result[name] = cluster([v for _,v in selected], [key[1] for key,_ in selected])
    write(BASE / "analysis.json", result)
    write(BASE / "utterance_effects.json", [{"id": k[0], "speaker": k[1], "effects": {name:float(np.mean(v)) for name,v in vals.items()}} for k,vals in scalars.items()])
    for name in ("official_full_C", "official_queries_C", "raw_native_margin", "raw_matched_margin", "raw_did_margin", "raw_matched_rank", "unit_matched_margin", "unit_matched_rank"):
        print(name, result[name]["speaker_mean"], result[name]["speaker_ci95"])


def sensitivity():
    global BASE
    main = BASE
    p = read(main / "protocol.json")
    for delta in (-1, 1):
        BASE = main / f"lag_{delta:+d}"
        q = dict(p)
        q["fixed_audio_minus_visual_index"] += delta
        q["sensitivity_delta"] = delta
        q["primary_protocol_sha256"] = sha(main / "protocol.json")
        write(BASE / "protocol.json", q)
        support()
        evaluate()
        summarize()
    BASE = main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "support", "evaluate", "summarize", "sensitivity"))
    globals()[parser.parse_args().stage]()
