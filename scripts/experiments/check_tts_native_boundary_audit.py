"""Independent NumPy distance, curve, region, pairing and bootstrap audit."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "runs/tts_native_boundary_audit_20260926"


def read(p):
    return json.loads(Path(p).read_text())


def digest(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1048576), b""):
            h.update(b)
    return h.hexdigest()


def load(d):
    if "key" in d:
        with np.load(d["path"]) as f:
            a = f[d["key"]]
    else:
        a = np.load(d["path"])
    return a[d.get("start", 0):]


def bdc(z):
    z = np.asarray(z)
    j = np.argmin(z)
    d = z[j]
    b = sorted(z)[15]
    return dict(B=float(b), D=float(d), C=float(b-d), best_lag=int(j)-15,
                offset=15-int(j), curve=z)


maxima = {"distance": 0., "curve_metric": 0., "regions": 0., "shapley": 0., "statistics": 0.}


def close(a, b, name, atol=1e-10):
    err = float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
    maxima[name] = max(maxima[name], err)
    assert err <= atol, (name, err)


def independent(m):
    n = m.shape[0]
    rows = {k: [] for k in ("I", "E", "P", "H", "T")}
    count = {k: [] for k in rows}
    valid_z = []
    for col in range(31):
        s = col-15
        real = [i for i in range(n) if 0 <= i+s < n]
        valid_z.append(np.mean(m[real, col]))
        if n > 40:
            idx = {
                "I": list(range(20, n-20)),
                "E": [i for i in real if i < 20 or i >= n-20],
                "P": [i for i in range(n) if not 0 <= i+s < n],
                "H": [i for i in real if i < 20],
                "T": [i for i in real if i >= n-20],
            }
            for k, ii in idx.items():
                count[k].append(len(ii))
                rows[k].append(float(np.mean(m[ii, col])) if ii else 0.)
    metrics = {"guard0": bdc(m.mean(axis=0)), "valid": bdc(valid_z)}
    for guard in (15, 20):
        if n > guard*2:
            metrics[f"guard{guard}"] = bdc(m[guard:n-guard].mean(axis=0))
    if n > 40:
        metrics["equal_valid_regions"] = bdc(np.mean([rows[k] for k in ("H", "I", "T")], axis=0))
    return dict(metrics=metrics, rows=rows, counts=count)


def bootstrap(x, speakers, expected):
    names = sorted(set(speakers))
    v = np.array([np.mean([y for y, s in zip(x, speakers) if s == name], axis=0) for name in names])
    close(v.mean(0), expected["mean"], "statistics")
    for name, vv in zip(names, v):
        close(vv, expected["per_speaker"][name], "statistics")
    assert len(names) == expected["speakers"] and len(x) == expected["n"]
    if len(names) == 1:
        assert expected["ci99"] is None
        return
    ix = np.random.Generator(np.random.PCG64(20260926)).integers(0, len(names), (20000, len(names)))
    # Independent weighted-sum formulation of the same frozen resamples.
    weights = np.stack([(ix == j).sum(1) for j in range(len(names))], axis=1)/len(names)
    boot = weights @ v
    close(np.percentile(boot, [.5, 99.5], axis=0), expected["ci99"], "statistics")
    close(np.percentile(boot, [2.5, 97.5], axis=0), expected["ci95"], "statistics")


def main():
    audit = read(OUT / "audit.json")
    records = [r for r in audit["records"] if r["panel"] != "static" or (r["split"] == "evaluation" and r["image"] == "3")]
    evaluated, matrix_cells = {}, 0
    for index, r in enumerate(records):
        p = OUT / "cells" / (r["key"]+".json")
        saved = read(p)
        assert digest(p.with_suffix(".npz")) == saved["matrix_sha256"]
        v, a = load(r["visual"]), load(r["audio"])
        with np.load(p.with_suffix(".npz")) as matrices:
            for ver in ("old", "joint"):
                m = matrices[ver]
                n = r[ver+"_L"]
                assert len(m) == n
                if ver == "joint":
                    assert n == min(r["frames"], r["sample_count"]//640-r["track_start"])-5
                for col, lag in enumerate(range(-15, 16)):
                    ix = np.arange(n)+lag
                    rhs = np.zeros((n, 1024), dtype=np.float32)
                    mask = (ix >= 0) & (ix < n)
                    rhs[mask] = a[ix[mask]]
                    diff = v[:n].astype(np.float32)-rhs+np.float32(1e-6)
                    d = np.sqrt(np.sum(diff*diff, axis=1, dtype=np.float32))
                    close(d, m[:, col], "distance", atol=1e-5)
                matrix_cells += int(m.size)
                independent_result = independent(m)
                evaluated[r["key"], ver] = independent_result
                target = saved["versions"][ver]
                for policy, mm in independent_result["metrics"].items():
                    for field, value in mm.items():
                        close(value, target["metrics"][policy][field], "curve_metric")
                if n > 40:
                    for reg, z in independent_result["rows"].items():
                        close(z, target["regions"][reg]["curve"], "regions")
                        assert independent_result["counts"][reg] == target["regions"][reg]["counts"]
        if (index+1) % 200 == 0:
            print("independently audited", index+1, flush=True)
    summary = read(OUT / "summary.json")
    support = read(OUT / "supports.json")
    lookup = {(r["panel"], r["id"], r["arm"], r["channel"]): r for r in records}
    for panel, ss in support.items():
        common = sorted({r["id"] for r in records if r["panel"] == panel and min(r["old_L"], r["joint_L"]) > 40})
        failed = {r["id"] for r in records if r["panel"] == panel and min(r["old_L"], r["joint_L"]) <= 40}
        assert [s for s in common if s not in failed] == ss["common_ids"]
    def native_value(panel, sid, channel, arm, ver, policy, field):
        return evaluated[lookup[panel, sid, arm, channel]["key"], ver]["metrics"][policy][field]
    def arms(comp):
        return (["Q1", "Q2"] if comp.startswith("Q") else [comp[0]]), (["C"] if comp.endswith("C") else ["N"])
    for panel, comparisons in summary["native"].items():
        for key, result in comparisons.items():
            channel, comp, ver, policy, sup = key.split("/")
            ids = result["ids"]
            ta, na = arms(comp)
            speakers = [lookup[panel, sid, na[0], channel]["speaker"] for sid in ids]
            for field in ("C", "B", "D", "best_lag"):
                nv = [np.mean([native_value(panel, sid, channel, a, ver, policy, field) for a in na]) for sid in ids]
                tv = [np.mean([native_value(panel, sid, channel, a, ver, policy, field) for a in ta]) for sid in ids]
                for label, values in (("N", nv), ("T", tv), ("delta", np.array(tv)-nv)):
                    bootstrap(values, speakers, result[field][label])
    pair_rows = read(OUT / "pair_shapley.json")
    for p in pair_rows:
        n, t = [evaluated[lookup[p["panel"], p["id"], arm, p["channel"]]["key"], p["version"]] for arm in (p["N_arm"], p["T_arm"])]
        zs = [np.array([x["rows"][reg] for reg in ("I", "E", "P")]) for x in (n, t)]
        counts = [np.array([x["counts"][reg] for reg in ("I", "E", "P")], dtype=float) for x in (n, t)]
        if p["policy"] == "valid":
            for c in counts:
                c[2] = 0
        weights = [c/c.sum(axis=0) for c in counts]
        f = {(a, b): bdc((zs[a]*weights[b]).sum(0)) for a in (0, 1) for b in (0, 1)}
        for a, b in f:
            for k in ("B", "D", "C"):
                close(f[a,b][k], p["cells"][f"{a}{b}"][k], "shapley")
        for k in ("B", "D", "C"):
            curve = ((f[1,0][k]-f[0,0][k])+(f[1,1][k]-f[0,1][k]))/2
            weight = ((f[0,1][k]-f[0,0][k])+(f[1,1][k]-f[1,0][k]))/2
            close(curve, p[k]["curve"], "shapley")
            close(weight, p[k]["weight"], "shapley")
    for key, result in summary["shapley"].items():
        group = key.split("/")
        pp = [p for p in pair_rows if [p[k] for k in ("panel", "channel", "comparison", "version", "policy")] == group]
        ids = sorted(set(p["id"] for p in pp))
        byid = {sid: [p for p in pp if p["id"] == sid] for sid in ids}
        speakers = [byid[sid][0]["speaker"] for sid in ids]
        for field in ("B", "D", "C"):
            for effect in ("total", "curve", "weight"):
                x = [np.mean([p[field][effect] for p in byid[sid]]) for sid in ids]
                bootstrap(x, speakers, result[field][effect])
    for key, result in read(OUT / "intervention_summary.json").items():
        panel, channel, comp, ver, intervention = key.split("/")
        ids = support[panel]["common_ids"]
        ta, na = arms(comp)
        speakers = [lookup[panel, sid, na[0], channel]["speaker"] for sid in ids]
        for field in ("B", "D", "C"):
            def effect(sid, arm):
                if ver == "joint-old":
                    return native_value(panel,sid,channel,arm,"joint",intervention,field)-native_value(panel,sid,channel,arm,"old",intervention,field)
                target, base = intervention.split("-")
                return native_value(panel,sid,channel,arm,ver,target,field)-native_value(panel,sid,channel,arm,ver,base,field)
            values = [np.mean([effect(sid,a) for a in ta])-np.mean([effect(sid,a) for a in na]) for sid in ids]
            bootstrap(values, speakers, result[field])
    hashes = read(OUT / "source_hashes.json")
    for p, h in hashes.items():
        assert digest(p) == h, p
    result = {"status": "PASS", "cells": len(records), "distance_cells": matrix_cells,
              "max_errors": maxima, "source_hashes_verified": len(hashes),
              "method": "Independent NumPy per-lag distances, explicit valid index lists, median order statistic, independently paired within-speaker averages and bootstrap multiplicities; no primary module imported"}
    (OUT / "independent_validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
