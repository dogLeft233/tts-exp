"""CPU audit and frozen native SyncNet boundary scoring intervention.

No model inference, audio edits, lag tuning, or embedding normalization.
The archived extraction frontend is retained; only joint length and scoring change.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "runs/tts_native_boundary_audit_20260926"
STATIC = ROOT / "runs/tts_chinese_instance_primed_20260926"
DYNAMIC = ROOT / "runs/tts_pcm_residual_20260926"
DITTO = ROOT / "runs/tts_clock_cross_20260926"
LAGS = np.arange(-15, 16)
SEED = 20260926


def read(p):
    return json.loads(Path(p).read_text())


def write(p, x):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(x, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def sha(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def array(desc):
    p = Path(desc["path"])
    if p.suffix == ".npz":
        with np.load(p) as z:
            a = z[desc["key"]].copy()
    else:
        a = np.load(p)
    return a[desc.get("start", 0):]


def descriptor(p, key=None, start=0):
    d = {"path": str(p)}
    if key is not None:
        d["key"] = key
    if start:
        d["start"] = int(start)
    return d


def audit():
    """Read metadata/shapes/hashes only; do not evaluate new score effects."""
    records, sources = [], {}

    def track(p, expected=None):
        p = str(p)
        if p not in sources:
            sources[p] = sha(p)
        if expected is not None and sources[p] != expected:
            raise ValueError(f"source hash mismatch: {p}")

    def add(panel, sid, speaker, arm, split, image, channel, v, a, m, frames,
            samples, start, meta):
        for d in (v, a, m):
            track(d["path"])
        nv, na, old = len(array(v)), len(array(a)), len(array(m))
        joint = min(frames, samples // 640 - start) - 5
        if joint < 1 or joint > min(nv, na):
            raise ValueError(f"joint features unavailable {panel}/{sid}/{arm}: {joint}/{nv}/{na}")
        key = f"{panel}/{image}/{sid}/{arm}_{channel}"
        records.append(dict(key=key, panel=panel, id=str(sid), speaker=speaker,
                            arm=arm, split=split, image=image, channel=channel,
                            visual=v, audio=a, old_matrix=m, frames=frames,
                            sample_count=samples, track_start=start, n_visual=nv,
                            n_audio=na, old_L=old, joint_L=joint, metadata=meta))

    p = STATIC / "protocol.json"
    track(p)
    protocol = read(p)
    for row in protocol["rows"]:
        sid = row["id"]
        ap = STATIC / "audio_features" / f"{sid}.npz"
        am = read(ap.with_suffix(".json"))
        track(ap.with_suffix(".json")); track(ap, am["sha256"])
        for arm in ("N", "C", "Q1", "Q2"):
            track(am["audio"][arm]["path"], am["audio"][arm]["sha256"])
        for image in ("3", "6", "9"):
            if image != "3" and sid not in protocol["sensitivity_ids"]:
                continue
            for arm in ("N", "C", "Q1", "Q2"):
                mp = STATIC / "scores" / image / sid / f"{arm}.npz"
                info = read(mp.with_suffix(".json"))
                track(mp.with_suffix(".json")); track(mp, info["sha256"])
                vm = info["video"].get("visual_metadata", {"frame_count": info["video"]["frames"], "source": "archived render receipt"}) if isinstance(info["video"], dict) else info["visual_metadata"]
                add("static", sid, row["speaker"], arm, row["split"], image,
                    "source", descriptor(mp, "visual"), descriptor(ap, arm),
                    descriptor(mp, "matrix"), vm["frame_count"],
                    am["metadata"][arm]["sample_count"], 0,
                    {"score_receipt": str(mp.with_suffix(".json")), "frontend": vm})
    p = DYNAMIC / "protocol.json"
    track(p)
    for row in read(p)["rows"]:
        sid = row["id"]
        fp = DYNAMIC / "features" / sid / "features.npz"
        mp = DYNAMIC / "scores" / f"{sid}.npz"
        info = read(mp.with_suffix(".json"))
        track(mp.with_suffix(".json")); track(fp, info["features_sha256"])
        track(mp, info["matrices_sha256"])
        rp = DYNAMIC / "pcm" / sid / "receipt.json"
        track(rp); receipt = read(rp)
        for arm in ("N", "T"):
            cell = row["cells"][arm]
            vm = info["replay"][arm]["video_meta"]
            for channel in ("source", "legacy"):
                pcm = receipt[f"{arm}_{channel}"]
                track(pcm["path"], pcm["sha256"])
                start = cell["track_start"] if channel == "source" else 0
                add("dynamic", sid, row["speaker"], arm, "historical", "dynamic",
                    channel, descriptor(fp, f"v_{arm}"),
                    descriptor(fp, f"a_{arm}_{channel}", start),
                    descriptor(mp, f"{arm}_{channel}"), vm["frame_count"],
                    sf.info(pcm["path"]).frames, start,
                    {"score_receipt": str(mp.with_suffix(".json")), "pcm": pcm,
                     "frontend": vm, "old_rounded_C": cell["old_C"],
                     "old_rounded_D": cell["old_D"]})
    p = DITTO / "protocol.json"
    track(p)
    for row in read(p)["rows"]:
        sid, arm = row["id"], row["arm"]
        rp = ROOT / "runs/ditto_timing_rate_v1/receipts" / str(sid) / f"{arm}_O.json"
        track(rp); canonical = read(rp)
        sp = DITTO / "scores" / f"{sid}_{arm}.json"
        track(sp); info = read(sp)
        track(row["visual"], row["visual_sha256"])
        track(row["legacy_audio"], row["legacy_audio_sha256"])
        track(info["matrix_path"], info["matrix_sha256"])
        for channel in ("source", "legacy"):
            pcm = row[f"{channel}_pcm"]
            track(pcm, row[f"{channel}_pcm_sha256"])
            ap = DITTO / "features" / str(sid) / arm / "source.npy" if channel == "source" else Path(row["legacy_audio"])
            add("ditto", sid, row["speaker"], arm, "historical", "dynamic",
                channel, descriptor(row["visual"]), descriptor(ap),
                descriptor(info["matrix_path"], channel), canonical["target_M"],
                sf.info(pcm).frames, 0,
                {"score_receipt": str(sp), "canonical_receipt": str(rp),
                 "source_track_start_already_cut": row["track_start"], "pcm": pcm})
    summary = {}
    for panel in ("dynamic", "static", "ditto"):
        rs = [r for r in records if r["panel"] == panel]
        summary[panel] = {
            "cells": len(rs), "ids": len(set(r["id"] for r in rs)),
            "speakers": len(set(r["speaker"] for r in rs)),
            "joint_minus_old": dict(Counter(str(r["joint_L"] - r["old_L"]) for r in rs)),
            "nonzero_track_start": sum(r["track_start"] != 0 for r in rs),
            "length_range": [min(r["joint_L"] for r in rs), max(r["joint_L"] for r in rs)],
        }
    write(OUT / "audit.json", {"records": records, "summary": summary})
    write(OUT / "source_hashes.json", sources)
    print(json.dumps(summary, ensure_ascii=False))


def freeze():
    p = OUT / "protocol.json"
    if p.exists():
        raise ValueError("frozen protocol already exists")
    write(p, {
        "status": "frozen_before_new_effects", "seed": SEED, "bootstrap": 20000,
        "main": "dynamic74 source PCM native T-N; static74 eval image3 four-arm replication",
        "descriptive": "Ditto47 source and legacy, one speaker; no inferential CI",
        "frontend": "Archived frontend; official score formula and joint length. Not full official pipeline equivalence.",
        "lengths": {"old": "Exact archived per-cell matrix length",
                    "joint": "min(decoded visual frame count, floor(PCM samples/640)-track_start)-5; both embeddings truncate before sentinel padding"},
        "score": "float32 pairwise_distance eps1e-6; lag curve float64 means; B=median31, D=min31, C=B-D; offset=-best_lag",
        "precision": "float64 lag reduction follows archived reports; official torch float32 reduction separately checked numerically",
        "policies": ["guard0", "valid", "guard15", "guard20"],
        "valid": "For each lag s average only i with 0<=i+s<L; L-abs(s) rows. Lag-dependent query support.",
        "regions": "L>40: I rows20:L-20; E valid edge rows; P invalid audio rows; counts L-40,40-abs(s),abs(s). P curve zero at s0 where weight0.",
        "equal_valid_regions": "Secondary: equal1/3 weights for valid head20/interior/tail20 lag curves; fixed and not optimized",
        "shapley": "For guard0 and valid, f(curves_arm,weights_arm), 4cells NN/TN/NT/TT; curve=.5[(TN-NN)+(TT-NT)], weight=.5[(NT-NN)+(TT-TN)] for B,D,C. valid weights delete P then normalize perlag.",
        "support": "Primary each panel common all arms, channels and both lengths L>40; static cal and images6/9 excluded scoring. Maximum support per comparison/length/policy supplemental. List every excluded ID and row counts. No CER filter.",
        "static_instances": "Compute each Qj-N score and Shapley separately, then within-record average. QminusC on same four-arm support.",
        "statistics": "Within-speaker record means, then speakers equal; paired differences; PCG64 seed20260926 20000 draws;99CI and95CI; Ditto point estimates only",
        "interpretation": "Scoring intervention and exact arithmetic attribution, not physical AV truth, biology, causal percentage or speech-only endpoint. Guard does not identify silence.",
        "audit_sha256": sha(OUT / "audit.json"), "source_manifest_sha256": sha(OUT / "source_hashes.json"),
        "code_sha256_at_freeze": sha(Path(__file__)),
    })


def metric(curve):
    z = np.asarray(curve, dtype=np.float64)
    j = int(z.argmin())
    b, d = float(np.median(z)), float(z[j])
    return {"C": b-d, "B": b, "D": d, "offset": 15-j,
            "best_lag": j-15, "curve": z.tolist()}


def summarize_matrix(m):
    n = len(m)
    i = np.arange(n)[:, None]
    valid = (i + LAGS >= 0) & (i + LAGS < n)
    curves = {"guard0": m.mean(0),
              "valid": (m * valid).sum(0) / valid.sum(0)}
    for g in (15, 20):
        if n > 2*g:
            curves[f"guard{g}"] = m[g:-g].mean(0)
    result = {"L": n, "valid_counts": valid.sum(0).tolist(),
              "metrics": {k: metric(z) for k, z in curves.items()}}
    if n <= 40:
        return result
    masks = {
        "I": np.broadcast_to((i >= 20) & (i < n-20), m.shape),
        "E": valid & ((i < 20) | (i >= n-20)), "P": ~valid,
        "H": valid & (i < 20), "T": valid & (i >= n-20),
    }
    region = {}
    for name, mask in masks.items():
        count = mask.sum(0)
        z = (m*mask).sum(0) / np.maximum(count, 1)
        region[name] = {"counts": count.tolist(), "curve": z.tolist(),
                        "metrics": metric(z) if name != "P" else None}
    z = np.array([region[k]["curve"] for k in ("I", "E", "P")])
    counts = np.array([region[k]["counts"] for k in ("I", "E", "P")])
    weights = {"guard0": counts/n,
               "valid": np.array([counts[0], counts[1], np.zeros(31)]) / valid.sum(0)}
    result["regions"] = region
    result["weights"] = {k: w.tolist() for k, w in weights.items()}
    result["closure"] = {k: float(np.max(np.abs((z*w).sum(0)-curves[k]))) for k, w in weights.items()}
    equal = np.mean([region[k]["curve"] for k in ("H", "I", "T")], 0)
    result["metrics"]["equal_valid_regions"] = metric(equal)
    return result


def score():
    import torch
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
    torch.set_num_threads(2)
    protocol = read(OUT / "protocol.json")
    assert sha(OUT / "audit.json") == protocol["audit_sha256"]
    hashes = read(OUT / "source_hashes.json")
    for p, h in hashes.items():
        assert sha(p) == h, p
    rows = read(OUT / "audit.json")["records"]
    selected = [r for r in rows if r["panel"] != "static" or (r["split"] == "evaluation" and r["image"] == "3")]
    replay_max = reduction_max = 0.
    for index, row in enumerate(selected):
        dest = OUT / "cells" / (row["key"] + ".json")
        if dest.exists():
            prior = read(dest)
            replay_max = max(replay_max, prior["replay_max_error"])
            reduction_max = max(reduction_max, prior["torch_reduction_max_error"])
            continue
        v, a, archived = array(row["visual"]), array(row["audio"]), array(row["old_matrix"])
        matrices, scores = {}, {}
        for version in ("old", "joint"):
            n = row[f"{version}_L"]
            m = SyncNetEngine.distance_matrix(v[:n], a[:n])
            matrices[version] = m
            scores[version] = summarize_matrix(m)
        replay = float(np.max(np.abs(matrices["old"]-archived)))
        assert replay < 1e-4, (row["key"], replay)
        reductions = []
        for m in matrices.values():
            official = torch.from_numpy(m.astype(np.float32)).mean(0).numpy()
            ours = m.mean(0)
            reductions.append(float(np.max(np.abs(official-ours))))
            reductions.extend(abs(metric(official)[k]-metric(ours)[k]) for k in ("B", "D", "C"))
        rp = max(reductions)
        replay_max, reduction_max = max(replay_max, replay), max(reduction_max, rp)
        dest.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest.with_suffix(".npz"), **matrices)
        write(dest, {"key": row["key"], "versions": scores, "matrix_sha256": sha(dest.with_suffix(".npz")),
                     "replay_max_error": replay, "torch_reduction_max_error": rp})
        if (index+1) % 100 == 0:
            print(f"CPU scored {index+1}/{len(selected)}", flush=True)
    write(OUT / "scoring_validation.json", {"cells": len(selected), "archived_matrix_max_error": replay_max,
          "official_float32_vs_float64_reduction_max_error": reduction_max, "source_hashes_verified": len(hashes), "GPU_used": False})
    print("score complete", len(selected), replay_max, reduction_max)


def stats(values, speakers):
    x = np.asarray(values, dtype=float)
    names = sorted(set(speakers))
    y = np.array([x[np.array(speakers) == name].mean(0) for name in names])
    result = {"n": len(x), "speakers": len(names), "mean": y.mean(0).tolist(),
              "per_speaker": {n: v.tolist() for n, v in zip(names, y)},
              "utterance_mean": x.mean(0).tolist()}
    if len(names) > 1:
        ix = np.random.default_rng(SEED).integers(len(names), size=(20000, len(names)))
        boot = y[ix].mean(1)
        result["ci99"] = np.quantile(boot, [.005, .995], axis=0).tolist()
        result["ci95"] = np.quantile(boot, [.025, .975], axis=0).tolist()
    else:
        result["ci99"] = result["ci95"] = None
    return result


def shapley(n, t, policy):
    z = [np.array([x["regions"][r]["curve"] for r in ("I", "E", "P")]) for x in (n, t)]
    w = [np.asarray(x["weights"][policy]) for x in (n, t)]
    cells = {f"{a}{b}": metric((z[a]*w[b]).sum(0)) for a in (0, 1) for b in (0, 1)}
    out = {"cells": cells}
    for k in ("B", "D", "C"):
        f00, f10, f01, f11 = [cells[c][k] for c in ("00", "10", "01", "11")]
        out[k] = {"curve": .5*((f10-f00)+(f11-f01)),
                  "weight": .5*((f01-f00)+(f11-f10)), "total": f11-f00}
    out["closure"] = max(abs(out[k]["curve"]+out[k]["weight"]-out[k]["total"]) for k in ("B", "D", "C"))
    return out


def analyze():
    rows = read(OUT / "audit.json")["records"]
    selected = [r for r in rows if r["panel"] != "static" or (r["split"] == "evaluation" and r["image"] == "3")]
    cells = {r["key"]: read(OUT / "cells" / (r["key"] + ".json"))["versions"] for r in selected}
    panels, supports, pairs, interventions = {}, {}, [], {}
    for panel in ("dynamic", "static", "ditto"):
        rs = [r for r in selected if r["panel"] == panel]
        ids = sorted(set(r["id"] for r in rs))
        byid = {sid: [r for r in rs if r["id"] == sid] for sid in ids}
        common = [sid for sid in ids if all(min(r["old_L"], r["joint_L"]) > 40 for r in byid[sid])]
        supports[panel] = {"input_ids": ids, "common_ids": common,
                          "excluded": [{"id": sid, "cells": [{k: r[k] for k in ("arm", "channel", "old_L", "joint_L")} for r in byid[sid]]} for sid in ids if sid not in common],
                          "support_by_length": {ver: [sid for sid in ids if all(r[ver+"_L"] > 40 for r in byid[sid])] for ver in ("old", "joint")}}
        lookup = {(r["id"], r["arm"], r["channel"]): r for r in rs}
        comparisons = {"T-N": (["T"], ["N"])} if panel != "static" else {"C-N": (["C"], ["N"]), "Q-N": (["Q1", "Q2"], ["N"]), "Q-C": (["Q1", "Q2"], ["C"])}
        channels = ("source",) if panel == "static" else ("source", "legacy")
        result = {}
        for channel in channels:
            for comparison, (ts, ns) in comparisons.items():
                for version in ("old", "joint"):
                    def cell(sid, arm):
                        return cells[lookup[sid, arm, channel]["key"]][version]
                    for policy in ("guard0", "valid", "guard15", "guard20", "equal_valid_regions"):
                        maxi = [sid for sid in ids if all(policy in cell(sid, arm)["metrics"] for arm in ts+ns)]
                        for support, ss in (("common", common), ("maximal", maxi)):
                            if not ss:
                                continue
                            speakers = [byid[sid][0]["speaker"] for sid in ss]
                            d = {}
                            for k in ("C", "B", "D", "best_lag"):
                                nv = [np.mean([cell(sid, a)["metrics"][policy][k] for a in ns]) for sid in ss]
                                tv = [np.mean([cell(sid, a)["metrics"][policy][k] for a in ts]) for sid in ss]
                                d[k] = {"N": stats(nv, speakers), "T": stats(tv, speakers),
                                        "delta": stats(np.array(tv)-nv, speakers)}
                            d["ids"] = ss
                            result[f"{channel}/{comparison}/{version}/{policy}/{support}"] = d
                    # All policy and length interventions use exactly the common records.
                    for base, target in (("guard0", "valid"), ("guard0", "guard15"), ("guard0", "guard20"), ("valid", "guard15"), ("valid", "guard20"), ("valid", "equal_valid_regions")):
                        vv = {}
                        for k in ("C", "B", "D"):
                            values = []
                            for sid in common:
                                values.append(np.mean([cell(sid,a)["metrics"][target][k]-cell(sid,a)["metrics"][base][k] for a in ts])-np.mean([cell(sid,a)["metrics"][target][k]-cell(sid,a)["metrics"][base][k] for a in ns]))
                            vv[k] = stats(values, [byid[sid][0]["speaker"] for sid in common])
                        interventions["/".join((panel, channel, comparison, version, target+"-"+base))] = vv
                    if comparison == "Q-C":
                        continue
                    # Four cells are only defined where the interior region exists.
                    for sid in common:
                        for ta in ts:
                            for na in ns:
                                for policy in ("guard0", "valid"):
                                    sh = shapley(cell(sid, na), cell(sid, ta), policy)
                                    pairs.append({"panel": panel, "id": sid, "speaker": byid[sid][0]["speaker"],
                                                  "channel": channel, "comparison": comparison,
                                                  "N_arm": na, "T_arm": ta, "version": version, "policy": policy, **sh})
                for policy in ("guard0", "valid", "guard15", "guard20", "equal_valid_regions"):
                    vv = {}
                    for k in ("C", "B", "D"):
                        values = []
                        for sid in common:
                            def change(a):
                                cc = cells[lookup[sid,a,channel]["key"]]
                                return cc["joint"]["metrics"][policy][k]-cc["old"]["metrics"][policy][k]
                            values.append(np.mean([change(a) for a in ts])-np.mean([change(a) for a in ns]))
                        vv[k] = stats(values, [byid[sid][0]["speaker"] for sid in common])
                    interventions["/".join((panel, channel, comparison, "joint-old", policy))] = vv
        panels[panel] = result
    sh_summary = {}
    for group in sorted(set((p["panel"], p["channel"], p["comparison"], p["version"], p["policy"]) for p in pairs)):
        pp = [p for p in pairs if tuple(p[k] for k in ("panel", "channel", "comparison", "version", "policy")) == group]
        ids = sorted(set(p["id"] for p in pp))
        per = {sid: [p for p in pp if p["id"] == sid] for sid in ids}
        sp = [per[sid][0]["speaker"] for sid in ids]
        sh_summary["/".join(group)] = {
            k: {effect: stats([np.mean([p[k][effect] for p in per[sid]]) for sid in ids], sp)
                for effect in ("total", "curve", "weight")} for k in ("B", "D", "C")}
    # Arm/region curves and weights averaged after per-record metrics, not before C.
    region_summary = {}
    for panel, support in supports.items():
        for channel in (("source",) if panel == "static" else ("source", "legacy")):
            for arm in (("N", "C", "Q1", "Q2") if panel == "static" else ("N", "T")):
                rr = [r for r in selected if r["panel"] == panel and r["channel"] == channel and r["arm"] == arm and r["id"] in support["common_ids"]]
                sp = [r["speaker"] for r in rr]
                for ver in ("old", "joint"):
                    cc = [cells[r["key"]][ver] for r in rr]
                    key = "/".join((panel, channel, arm, ver))
                    region_summary[key] = {"L": stats([c["L"] for c in cc], sp), "regions": {}}
                    for reg in ("I", "E", "P", "H", "T"):
                        z = [c["regions"][reg]["curve"] for c in cc]
                        rg = {"curve": stats(z, sp), "counts": stats([c["regions"][reg]["counts"] for c in cc], sp)}
                        if reg != "P":
                            rg["B_D_C"] = {k: stats([c["regions"][reg]["metrics"][k] for c in cc], sp) for k in ("B", "D", "C")}
                        # Positive/negative descriptive distances use each arm's own full-curve argmin/median rank lag; no physical truth claim.
                        for policy in ("guard0", "valid"):
                            pos, neg = [], []
                            for c, zz in zip(cc, z):
                                full = np.array(c["metrics"][policy]["curve"])
                                pos.append(zz[int(full.argmin())])
                                neg.append(zz[int(np.argsort(full, kind="stable")[15])])
                            rg[policy+"_at_own_min_and_median_lag"] = {"positive": stats(pos, sp), "background": stats(neg, sp), "gap": stats(np.array(neg)-pos, sp)}
                        region_summary[key]["regions"][reg] = rg
    closure = max(c[ver].get("closure", {}).get(pol, 0.) for c in cells.values() for ver in ("old", "joint") for pol in ("guard0", "valid"))
    write(OUT / "supports.json", supports)
    write(OUT / "pair_shapley.json", pairs)
    write(OUT / "region_summary.json", region_summary)
    write(OUT / "intervention_summary.json", interventions)
    write(OUT / "summary.json", {"native": panels, "shapley": sh_summary,
          "max_curve_closure_error": closure, "max_shapley_closure_error": max(p["closure"] for p in pairs)})
    print("analysis complete", {k: len(v["common_ids"]) for k, v in supports.items()}, closure)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("audit", "freeze", "score", "analyze"))
    stage = p.parse_args().stage
    globals()[stage]()
