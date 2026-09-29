"""Independent numerical and support verification for the negative-pool study."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "runs/tts_negative_pool_20260926"
SOURCE = ROOT / "runs/tts_prepost_mel_20260926"
PARENT = ROOT / "runs/tts_pcm_residual_20260926"


def read(path):
    return json.loads(Path(path).read_text())


def distance(x, y):
    return np.linalg.norm(np.asarray(x,dtype=np.float64)-np.asarray(y,dtype=np.float64)+0.000001,axis=-1)


def run(base):
    p, support, cells = read(base / "protocol.json"), read(base / "support.json"), read(base / "scores.json")
    k = p["fixed_audio_minus_visual_index"]
    supports = {r["id"]:r for r in support["rows"]}
    hashes = 0
    for path, expected in support["assets"].items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == expected
        hashes += 1
    errors = []
    scores = 0
    for c in cells:
        sid, image = c["id"], c["image"]
        s = supports[sid]
        if not c["eligible"]:
            continue
        audio = np.load(PARENT / "features" / sid / "features.npz")
        visual = np.load(SOURCE / "scores" / image / (sid+"_visual.npz"))
        for arm in ("N","T"):
            aa, vv = audio["a_"+arm+"_source"], visual[arm]
            for q, observed in zip(s["queries"],c["arms"][arm]["queries"],strict=True):
                node = s["nodes"][q["node"]]
                j, i = node["j"][arm], node["j"][arm]-k
                assert observed["node"] == q["node"]
                assert 20 <= i < s["lengths"][arm]-20
                assert all(abs(node["errors"][a])<=0.02000001 for a in ("N","T"))
                expected_donors=[]
                for dn, donor in enumerate(s["nodes"]):
                    if dn != q["node"] and all(3 <= abs(donor["j"][a]-node["j"][a])<=15 for a in ("N","T")):
                        expected_donors.append(dn)
                assert expected_donors == q["donors"]
                for pool in ("native","matched"):
                    indices = [j+d for d in range(-15,16) if abs(d)>=3] if pool=="native" else [s["nodes"][dn]["j"][arm] for dn in q["donors"]]
                    for geometry in ("raw","unit"):
                        a, v = aa, vv
                        if geometry == "unit":
                            a=aa/np.linalg.norm(aa,axis=-1)[:,None]
                            v=vv/np.linalg.norm(vv,axis=-1)[:,None]
                        positive = float(distance(v[i],a[j]))
                        negatives = distance(v[i],a[indices])
                        expected={"positive":positive,"negative":float(np.mean(negatives)),"margin":float(sum(negatives-positive)/len(indices)),
                                  "rank":float(sum(1 if z>positive else (0.5 if z==positive else 0) for z in negatives)/len(indices)),
                                  "AA":float(np.mean(distance(a[j],a[indices]))),"VV":float(np.mean(distance(v[i],v[np.array(indices)-k])))}
                        for metric, value in expected.items():
                            errors.append(abs(value-observed["pools"][pool]["metrics"][geometry][metric]))
                        scores += 1
                # Positive pairs stay exactly fixed between reference pools.
                for geometry in ("raw","unit"):
                    assert observed["pools"]["native"]["metrics"][geometry]["positive"] == observed["pools"]["matched"]["metrics"][geometry]["positive"]
    assert max(errors)<1e-9
    effects, analysis = read(base / "utterance_effects.json"), read(base / "analysis.json")
    stats_error = 0.
    for name, expected in analysis.items():
        byspeaker={}
        for r in effects:
            if name in r["effects"]:
                byspeaker.setdefault(r["speaker"],[]).append(r["effects"][name])
        values=np.array([sum(byspeaker[s])/len(byspeaker[s]) for s in sorted(byspeaker)])
        rng=np.random.default_rng(20260926)
        boot=[]
        for _ in range(20):
            sample=rng.integers(0,len(values),size=(1000,len(values)))
            boot.extend(np.sum(values[sample],axis=1)/len(values))
        checks={"speaker_mean":float(sum(values)/len(values)),"speaker_ci95":np.percentile(boot,[2.5,97.5]),"speaker_ci99":np.percentile(boot,[0.5,99.5])}
        for key,val in checks.items():
            stats_error=max(stats_error,float(np.max(np.abs(np.asarray(val)-expected[key]))))
    assert stats_error<1e-12
    controls=read(SOURCE / "controls/analysis.json")
    assert controls["pass"] and len(controls["rows"])==6
    for r in controls["rows"]:
        matrix=np.load(SOURCE / "controls" / r["image"] / r["id"] / "scores.npz")["delay5"]
        offset=15-int(np.argmin(np.mean(matrix[20:-20],axis=0)))
        assert offset-r["N"]["20"]["offset"]==r["offset_delta"]
        assert abs(r["offset_delta"]-5)<=1
    result={"status":"PASS","source_hashes":hashes,"recomputed_query_geometry_pools":scores,"max_metric_error":max(errors),"max_bootstrap_error":stats_error,"historical_delay_controls":6,
            "score_sha256":hashlib.sha256((base/"scores.json").read_bytes()).hexdigest(),"analysis_sha256":hashlib.sha256((base/"analysis.json").read_bytes()).hexdigest()}
    (base/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(base.name,result)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--all",action="store_true")
    args=parser.parse_args()
    run(BASE)
    if args.all:
        for d in (-1,1):
            run(BASE/f"lag_{d:+d}")
