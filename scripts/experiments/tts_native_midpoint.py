"""Frozen CPU interventions on native AV midpoint dynamics and paired residuals."""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments.tts_native_boundary_audit import array, read, write, sha, stats

PARENT = ROOT / "runs/tts_native_boundary_audit_20260926"
OUT = ROOT / "runs/tts_native_midpoint_20260926"
GROUPS = {"dynamic_cloud": ("dynamic", ("N", "T")),
          "static_cloud": ("static", ("N", "C")),
          "static_local": ("static", ("N", "Q1", "Q2"))}
LAGS = np.arange(-15, 16)


def curve_score(m, k0):
    z = np.asarray(m, dtype=np.float64).mean(0)
    j = int(z.argmin())
    b, d = float(np.median(z)), float(z[j])
    return {"C": b-d, "B": b, "D": d, "D_anchor": float(z[k0+15]),
            "best_lag": j-15, "offset": 15-j, "curve": z.tolist()}


def distances(v, a):
    """Fixed guard20 rows and original audio-index lags, official float32 eps."""
    import torch
    n = len(v)
    assert len(a) == n and n > 40
    vr, ar = torch.from_numpy(np.asarray(v, dtype=np.float32)), torch.from_numpy(np.asarray(a, dtype=np.float32))
    ix = np.arange(20, n-20)
    with torch.inference_mode():
        return np.stack([torch.nn.functional.pairwise_distance(vr[ix], ar[ix+s], eps=1e-6).numpy().astype(np.float64) for s in LAGS], axis=1)


def inputs(row, geometry):
    n = row["joint_L"]
    v, a = array(row["visual"])[:n], array(row["audio"])[:n]
    if geometry == "unit":
        # Unit geometry begins from a fixed float32 feature representation.
        v = (v.astype(np.float64)/np.linalg.norm(v.astype(np.float64), axis=1)[:, None]).astype(np.float32)
        a = (a.astype(np.float64)/np.linalg.norm(a.astype(np.float64), axis=1)[:, None]).astype(np.float32)
    assert np.isfinite(v).all() and np.isfinite(a).all()
    return v, a


def coordinates(v, a, k0):
    n = len(v)
    t = np.arange(max(0, -k0), min(n, n-k0))
    i = np.arange(20, n-20)
    assert abs(k0) <= 5 and len(i)
    accessed = np.concatenate((i, (i[:, None]+LAGS-k0).ravel()))
    assert accessed.min() >= t[0] and accessed.max() <= t[-1]
    vd, ad = v.astype(np.float64), a.astype(np.float64)
    m, r = np.full_like(vd, np.nan), np.full_like(vd, np.nan)
    m[t] = (ad[t+k0]+vd[t])/2
    r[t] = (ad[t+k0]-vd[t])/2
    mu = m[i].mean(0)
    sigma = float(np.sqrt(np.mean(np.sum((m[i]-mu)**2, axis=1))))
    rho = float(np.sqrt(np.mean(np.sum(r[i]**2, axis=1))))
    assert sigma > 0 and rho > 0
    return dict(v=vd, a=ad, M=m, R=r, mu=mu, sigma=sigma, rho=rho, t=t, i=i, k=k0)


def transform(c, alpha, beta, constant=None):
    t, k = c["t"], c["k"]
    mp = c["mu"]+alpha*(c["M"][t]-c["mu"])
    if constant is not None:
        mp = mp+constant
    rp = beta*c["R"][t]
    v, a = c["v"].copy(), c["a"].copy()
    v[t], a[t+k] = mp-rp, mp+rp
    return v, a


def freeze():
    import torch
    torch.set_num_threads(2)
    if (OUT / "protocol.json").exists():
        raise ValueError("already frozen; do not overwrite")
    records = read(PARENT / "audit.json")["records"]
    cal = sorted(set(r["id"] for r in records if r["panel"] == "static" and r["split"] == "calibration"))
    groups = {}
    for name, (panel, arms) in GROUPS.items():
        rr = [r for r in records if r["panel"] == panel and r["channel"] == "source" and r["arm"] in arms and (panel != "static" or r["image"] == "3")]
        ids = sorted(set(r["id"] for r in rr))
        eligible = [sid for sid in ids if all(r["joint_L"] > 40 for r in rr if r["id"] == sid)]
        calrows = [r for r in rr if r["id"] in cal]
        calibration = []
        for r in calrows:
            if r["joint_L"] <= 40:
                calibration.append({"id": r["id"], "arm": r["arm"], "L": r["joint_L"], "excluded": "guard20 requires L>40"})
                continue
            v, a = inputs(r, "raw")
            mm = distances(v, a)
            calibration.append({"id": r["id"], "arm": r["arm"], "L": r["joint_L"], "scores": curve_score(mm, 0)})
        pooled = [c["scores"]["best_lag"] for c in calibration if "scores" in c]
        medians = {a: float(np.median([c["scores"]["best_lag"] for c in calibration if c["arm"] == a and "scores" in c])) for a in arms}
        k0 = int(np.median(pooled))  # Python int truncates a half-integer toward zero.
        gate = abs(k0) <= 5 and max(medians.values())-min(medians.values()) <= 1
        eval_ids = [sid for sid in eligible if sid not in cal]
        groups[name] = {"panel": panel, "arms": arms, "records": rr,
                        "input_ids": ids, "eligible_ids": eligible,
                        "calibration": calibration, "calibration_arm_medians": medians,
                        "calibration_counts": dict(Counter(c["arm"] for c in calibration if "scores" in c)),
                        "k0": k0, "calibration_gate": gate,
                        "eval_ids": eval_ids, "eval_speakers": sorted(set(r["speaker"] for r in rr if r["id"] in eval_ids)),
                        "excluded": [{"id": sid, "lengths": {r["arm"]: r["joint_L"] for r in rr if r["id"] == sid}} for sid in ids if sid not in eligible]}
    protocol = {
        "status": "frozen_before_intervention_effects", "groups": groups, "old_cal_ids": cal,
        "main": "Dynamic48 held-out-cal-ID native T-N; static original74eval with group-common guard20 support",
        "bridge": "Dynamic74 including26cal exploratory only; static common4 previous70 native support bridge",
        "calibration": "All available valid raw guard20 calibration cells pooled; median integer toward0; max arm median difference<=1 and abs(k)<=5 gate. Unit uses same k0.",
        "definition": "M_t=(V_t+A_(t+k0))/2; R_t=(A_(t+k0)-V_t)/2; mu/sigma/rho estimated on I=20:L-20. Valid coordinate J requires both t and t+k0 in0:L.",
        "transform": "M'=mu+alpha(M-mu), R'=betaR; V'=M'-R', Anew_(t+k0)=M'+R'. Restore original audio indexing, fixed I and original31lags.",
        "matching": "alpha=sigmaN/sigmaT, beta=rhoN/rhoT; T2N four cells identity/M/R/both; N2T reciprocals. Qj individually then within-record average.",
        "geometry": "raw primary; unit sensitivity independently L2-normalizes original rows using float64 norm then casts float32, refits all M/R scales, same k0. No renormalization after intervention.",
        "precision": "M/R,means,scales,transforms float64; scoring reconstructed vectors cast float32, torch pairwise_distance eps1e-6; lag means float64. Full raw baseline compared to archived guard20.",
        "doses": [.75, 1., 1.25], "dose_beta": 1.,
        "controls": "Identity; inverse alpha/beta; M+linspace(-.25,.25,1024) constant; M-only anchor difference invariant. Double algebra and float32 rounding separately checked.",
        "endpoints": ["C", "B", "D", "D_anchor", "best_lag", "best_lag_changed"],
        "statistics": "Paired within-record, within-speaker means then equal speakers; PCG64(20260926),20000 draws,99CI; no percent explained, residual CI across0 not complete explanation",
        "limits": "Midpoint is not shared physiology; residual not true lip error; pairwise representation normalization not deployment or physical video improvement. Old speakers not independent new sources.",
        "parent_audit_sha256": sha(PARENT / "audit.json"),
        "source_hash_manifest_sha256": sha(PARENT / "source_hashes.json"),
        "code_sha256_at_freeze": sha(Path(__file__)),
    }
    write(OUT / "protocol.json", protocol)
    print({g: {"k0": x["k0"], "arm_medians": x["calibration_arm_medians"], "gate": x["calibration_gate"], "cal_counts": x["calibration_counts"], "eval_n": len(x["eval_ids"]), "eval_spk": len(x["eval_speakers"])} for g,x in groups.items()})


def score():
    import torch
    torch.set_num_threads(2)
    protocol = read(OUT / "protocol.json")
    assert sha(PARENT / "audit.json") == protocol["parent_audit_sha256"]
    assert sha(PARENT / "source_hashes.json") == protocol["source_hash_manifest_sha256"]
    for p,h in read(PARENT / "source_hashes.json").items():
        assert sha(p) == h, p
    for group, g in protocol["groups"].items():
        if abs(g["k0"]) > 5:
            write(OUT / group / "blocked.json", {"reason": "lag beyond support gate", "k0": g["k0"]})
            continue
        rows = {(r["id"], r["arm"]): r for r in g["records"]}
        ids = g["eligible_ids"] if group == "dynamic_cloud" else g["eval_ids"]
        for geometry in ("raw", "unit"):
            for sid in ids:
                for tarm in g["arms"][1:]:
                    dest = OUT / "pairs" / group / geometry / sid / f"{tarm}.json"
                    if dest.exists():
                        continue
                    k0 = g["k0"]
                    coords, original = {}, {}
                    for arm in ("N", tarm):
                        original[arm] = inputs(rows[sid, arm], geometry)
                        coords[arm] = coordinates(*original[arm], k0)
                    n, t = coords["N"], coords[tarm]
                    alpha, beta = n["sigma"]/t["sigma"], n["rho"]/t["rho"]
                    conditions = {"N_base": ("N", 1., 1.), "T_base": (tarm, 1., 1.),
                                  "T_M": (tarm, alpha, 1.), "T_R": (tarm, 1., beta), "T_both": (tarm, alpha, beta),
                                  "N_M": ("N", 1/alpha, 1.), "N_R": ("N", 1., 1/beta), "N_both": ("N", 1/alpha, 1/beta),
                                  "N_dose075": ("N", .75, 1.), "N_dose125": ("N", 1.25, 1.),
                                  "T_dose075": (tarm, .75, 1.), "T_dose125": (tarm, 1.25, 1.)}
                    matrices, scores, manipulation = {}, {}, {}
                    errors = {"identity_vector64": 0., "anchor_M_vector64": 0., "anchor_M_vector32": 0.,
                              "inverse_vector64": 0., "constant_distance32": 0., "raw_archived_matrix32": 0.}
                    for label,(arm, aa, bb) in conditions.items():
                        c = coords[arm]
                        vv, av = transform(c, aa, bb)
                        matrices[label] = distances(vv, av)
                        scores[label] = curve_score(matrices[label], k0)
                        after = coordinates(vv, av, k0)
                        manipulation[label] = {"sigma_before": c["sigma"], "sigma_after": after["sigma"],
                                               "rho_before": c["rho"], "rho_after": after["rho"],
                                               "alpha": aa, "beta": bb}
                        close_sigma = abs(after["sigma"]-aa*c["sigma"])
                        close_rho = abs(after["rho"]-bb*c["rho"])
                        assert max(close_sigma, close_rho) < 1e-10
                        invv, inva = transform(after, 1/aa, 1/bb)
                        errors["inverse_vector64"] = max(errors["inverse_vector64"], float(np.max(abs(invv-c["v"]))), float(np.max(abs(inva-c["a"]))))
                        if bb == 1:
                            ix = c["i"]
                            d0 = c["a"][ix+k0]-c["v"][ix]
                            d1 = av[ix+k0]-vv[ix]
                            errors["anchor_M_vector64"] = max(errors["anchor_M_vector64"], float(np.max(abs(d1-d0))))
                            df32 = av.astype(np.float32)[ix+k0]-vv.astype(np.float32)[ix]
                            errors["anchor_M_vector32"] = max(errors["anchor_M_vector32"], float(np.max(abs(df32-d0))))
                        if label.endswith("base"):
                            errors["identity_vector64"] = max(errors["identity_vector64"], float(np.max(abs(vv-c["v"]))), float(np.max(abs(av-c["a"]))))
                            cv, ca = transform(c, 1., 1., np.linspace(-.25, .25, 1024))
                            cm = distances(cv, ca)
                            errors["constant_distance32"] = max(errors["constant_distance32"], float(np.max(abs(cm-matrices[label]))))
                            direct = distances(*original[arm])
                            assert np.array_equal(direct, matrices[label]), (group, sid, label, "identity float32 changed")
                            if geometry == "raw":
                                rp = PARENT / "cells" / (rows[sid, arm]["key"] + ".npz")
                                with np.load(rp) as old:
                                    error = float(np.max(abs(old["joint"][20:-20]-matrices[label])))
                                errors["raw_archived_matrix32"] = max(errors["raw_archived_matrix32"], error)
                    assert errors["identity_vector64"] < 1e-10
                    assert errors["anchor_M_vector64"] < 1e-10
                    assert errors["inverse_vector64"] < 1e-10
                    assert errors["raw_archived_matrix32"] < 1e-5
                    assert errors["constant_distance32"] < 1e-5
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(dest.with_suffix(".npz"), **matrices)
                    write(dest, {"group": group, "geometry": geometry, "id": sid, "speaker": rows[sid,"N"]["speaker"],
                                 "T_arm": tarm, "k0": k0, "alpha_T2N": alpha, "beta_T2N": beta,
                                 "rows": {"N": len(n["i"]), "T": len(t["i"])},
                                 "coordinate_domain": {"N": [int(n["t"][0]), int(n["t"][-1])], "T": [int(t["t"][0]), int(t["t"][-1])]},
                                 "scores": scores, "manipulation": manipulation, "controls": errors,
                                 "matrix_sha256": sha(dest.with_suffix(".npz"))})
            print("scored", group, geometry, len(ids), flush=True)


def analyze():
    p = read(OUT / "protocol.json")
    summaries, coefficient_rows, controls = {}, [], []
    for group, g in p["groups"].items():
        for geometry in ("raw", "unit"):
            for support in (("eval", "full74_bridge") if group == "dynamic_cloud" else ("eval", "previous70_bridge")):
                ids = g["eligible_ids"] if support == "full74_bridge" else g["eval_ids"]
                if support == "previous70_bridge":
                    old = read(PARENT / "supports.json")["static"]["common_ids"]
                    ids = [sid for sid in ids if sid in old]
                rr = {sid: [read(OUT / "pairs" / group / geometry / sid / f"{a}.json") for a in g["arms"][1:]] for sid in ids}
                speakers = [rr[sid][0]["speaker"] for sid in ids]
                def val(r, label, field):
                    return r["scores"][label][field]
                def aggregate(fun):
                    return stats([np.mean([fun(r) for r in rr[sid]]) for sid in ids], speakers)
                result = {"ids": ids, "scores": {}, "effects": {}, "dose": {}, "coefficients": {}}
                labels = rr[ids[0]][0]["scores"].keys()
                for label in labels:
                    result["scores"][label] = {f: aggregate(lambda r,l=label,k=f: val(r,l,k)) for f in ("C", "B", "D", "D_anchor", "best_lag")}
                for direction in ("T2N", "N2T"):
                    for condition in ("base", "M", "R", "both"):
                        target = ("T_" if direction == "T2N" else "N_")+condition
                        base = "T_base" if direction == "T2N" else "N_base"
                        row = {}
                        for field in ("C", "B", "D", "D_anchor"):
                            row[field] = {
                                "treatment": aggregate(lambda r, tar=target, b=base, k=field: val(r,tar,k)-val(r,b,k)),
                                "residual_TminusN": aggregate(lambda r, tar=target, di=direction, k=field: val(r,tar,k)-val(r,"N_base",k) if di == "T2N" else val(r,"T_base",k)-val(r,tar,k)),
                            }
                        row["lag_changed_fraction"] = aggregate(lambda r, tar=target,b=base: float(val(r,tar,"best_lag") != val(r,b,"best_lag")))
                        row["lag_change"] = aggregate(lambda r,tar=target,b=base: val(r,tar,"best_lag")-val(r,b,"best_lag"))
                        result["effects"][direction+"/"+condition] = row
                for arm in ("N", "T"):
                    for dose in ("dose075", "base", "dose125"):
                        result["dose"][arm+"/"+dose] = {f: aggregate(lambda r,a=arm,d=dose,k=f: val(r,a+"_"+d,k)-val(r,a+"_base",k)) for f in ("C", "B", "D", "D_anchor")}
                for field in ("alpha_T2N", "beta_T2N"):
                    raw = [r[field] for sid in ids for r in rr[sid]]
                    result["coefficients"][field] = {"statistics": aggregate(lambda r,f=field: r[f]),
                        "pair_quantiles_0_5_25_50_75_95_100": np.quantile(raw, [0,.05,.25,.5,.75,.95,1]).tolist()}
                for target in ("N_base", "T_base", "N_M", "T_M", "N_R", "T_R", "N_both", "T_both"):
                    result.setdefault("manipulation", {})[target] = {f: aggregate(lambda r,l=target,k=f: r["manipulation"][l][k]) for f in ("sigma_before", "sigma_after", "rho_before", "rho_after")}
                summaries[f"{group}/{geometry}/{support}"] = result
                if support in ("full74_bridge", "eval"):
                    for sid in ids:
                        for r in rr[sid]:
                            controls.append(r["controls"])
                            coefficient_rows.append({"group":group,"geometry":geometry,"id":sid,"T_arm":r["T_arm"],"alpha":r["alpha_T2N"],"beta":r["beta_T2N"]})
    write(OUT / "summary.json", summaries)
    write(OUT / "coefficients.json", coefficient_rows)
    write(OUT / "controls.json", {k:max(r[k] for r in controls) for k in controls[0]})
    print("analysis complete", len(summaries))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("freeze", "score", "analyze"))
    args = parser.parse_args()
    globals()[args.stage]()
