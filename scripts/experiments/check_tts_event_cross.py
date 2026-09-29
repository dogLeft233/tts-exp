"""Independent four-cell source recomputation and grouped contrasts."""
import json
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"runs/tts_event_cross_20260926"
CAL=ROOT/"runs/tts_static_lag_calibration_20260926"
SOURCE=ROOT/"runs/tts_prepost_mel_20260926"
PARENT=ROOT/"runs/tts_pcm_residual_20260926"


def read(path):
    return json.loads(Path(path).read_text())


def main():
    for folder in ("evaluation","evaluation_lag_-1","evaluation_lag_+1"):
        dest=BASE/folder
        support={r["id"]:r for r in read(CAL/folder/"support.json")["rows"]}
        k=read(dest/"receipt.json")["k"]
        rows=read(dest/"scores.json")
        error=0.; count=0; derived={}
        for r in rows:
            s=support[r["id"]]
            audio=np.load(PARENT/"features"/r["id"]/"features.npz")
            visual=np.load(SOURCE/"scores"/r["image"]/(r["id"]+"_visual.npz"))
            vals={}
            for geom in ("raw","unit"):
                a={z:audio["a_"+z+"_source"] for z in "NT"}
                v={z:visual[z] for z in "NT"}
                if geom=="unit":
                    a={z:x/np.linalg.norm(x,axis=1)[:,None] for z,x in a.items()}
                    v={z:x/np.linalg.norm(x,axis=1)[:,None] for z,x in v.items()}
                for qi,q in enumerate(s["queries"]):
                    node=s["nodes"][q["node"]]
                    for va in "NT":
                        for aa in "NT":
                            x=v[va][node["j"][va]-k].astype(np.float64)
                            inds=[node["j"][aa]]+[s["nodes"][d]["j"][aa] for d in q["donors"]]
                            dif=x-a[aa][inds].astype(np.float64)+1e-6
                            distances=np.linalg.norm(dif,axis=1)
                            pos,neg=distances[0],distances[1:]
                            result={"positive":float(pos),"negative":float(sum(neg)/len(neg)),"margin":float(sum(neg-pos)/len(neg)),"rank":sum(1 if d>pos else .5 if d==pos else 0 for d in neg)/len(neg)}
                            old=r["queries"][qi]["geometry"][geom][va+aa]
                            error=max(error,max(abs(result[m]-old[m]) for m in result))
                            count+=1
                            for m,z in result.items():
                                vals.setdefault((geom,m,va+aa),[]).append(z)
                for m in ("positive","negative","margin","rank"):
                    nn,nt,tn,tt=[float(np.mean(vals[(geom,m,c)])) for c in ("NN","NT","TN","TT")]
                    effects={"visual_at_N_audio":tn-nn,"audio_at_N_visual":nt-nn,"interaction":(tt-tn)-(nt-nn),"visual_at_T_audio":tt-nt,"audio_at_T_visual":tt-tn,"diagonal":tt-nn}
                    for name,z in effects.items():
                        derived.setdefault((r["id"],r["speaker"]),{}).setdefault(f"{geom}_{m}_{name}",[]).append(z)
        assert error<1e-9
        analysis=read(dest/"analysis.json")
        statserror=0.
        for name in next(iter(derived.values())):
            speaker={}
            for (_,sp),values in derived.items():
                speaker.setdefault(sp,[]).append(float(np.mean(values[name])))
            means=np.array([sum(speaker[s])/len(speaker[s]) for s in sorted(speaker)])
            rng=np.random.default_rng(20260926)
            draws=means[rng.integers(len(means),size=(20000,len(means)))].mean(1)
            expected={"speaker_mean":float(np.mean(means)),"speaker_ci95":np.percentile(draws,[2.5,97.5]),"speaker_ci99":np.percentile(draws,[.5,99.5])}
            for key,z in expected.items():
                statserror=max(statserror,float(np.max(abs(np.asarray(z)-analysis[name][key]))))
        assert statserror<1e-9
        result={"status":"PASS","four_cell_query_geometries":count,"max_source_metric_error":error,"max_grouped_statistic_error":statserror}
        (dest/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
        print(folder,result)


if __name__=="__main__":
    main()
