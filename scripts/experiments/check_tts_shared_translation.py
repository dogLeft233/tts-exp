"""Independent weighted-fit, four-cell and bootstrap reconstruction."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/"runs/tts_shared_translation_20260926"
CAL=ROOT/"runs/tts_static_lag_calibration_20260926"
SOURCE=ROOT/"runs/tts_prepost_mel_20260926"
PARENT=ROOT/"runs/tts_pcm_residual_20260926"
CROSS=ROOT/"runs/tts_event_cross_20260926"


def read(path):return json.loads(Path(path).read_text())


def main():
    cal=read(BASE/"calibration_support.json")
    good=[r for r in cal["rows"] if r["eligible"]]
    speaker_counts=Counter(r["speaker"] for r in good)
    images=read(CAL/"protocol.json")["images"]
    da=np.zeros(1024);dv=np.zeros(1024)
    for path,expected in cal["assets"].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected
    for r in good:
        js={a:[r["nodes"][q["node"]]["j"][a] for q in r["queries"]] for a in "NT"}
        weight=1/(len(speaker_counts)*speaker_counts[r["speaker"]]*len(r["queries"]))
        arrays=np.load(PARENT/"features"/r["id"]/"features.npz")
        da+=np.sum(arrays["a_T_source"][js["T"]].astype(float)-arrays["a_N_source"][js["N"]].astype(float),axis=0)*weight
        for im in images:
            v=np.load(CAL/"scores"/im["id"]/(r["id"]+"_visual.npz"))
            dv+=np.sum(v["T"][np.array(js["T"])-3].astype(float)-v["N"][np.array(js["N"])-3].astype(float),axis=0)*weight/3
    fitted=np.load(BASE/"translation.npz")
    fit_error=max(np.max(abs(da-fitted["delta_A"])),np.max(abs(dv-fitted["delta_V"])),np.max(abs((da+dv)/2-fitted["delta"])))
    assert fit_error<1e-12
    random=np.random.Generator(np.random.PCG64(20260926)).standard_normal((16,1024))
    random*=np.linalg.norm(fitted["delta"])/np.linalg.norm(random,axis=1)[:,None]
    assert np.max(abs(random-fitted["shifts"][3:]))<1e-12
    support=read(CAL/"evaluation/support.json")
    byid={r["id"]:r for r in support["rows"]}
    original={(r["id"],r["image"]):r for r in read(CROSS/"evaluation/scores.json")}
    errors=[];original_error=0.;count=0;derived={}
    conditions=read(BASE/"protocol.json")["conditions"]
    for r in read(BASE/"manifest.json"):
        assert hashlib.sha256(Path(r["file"]).read_bytes()).hexdigest()==r["sha256"]
        s=byid[r["id"]]
        audio=np.load(PARENT/"features"/r["id"]/"features.npz")
        visual=np.load(SOURCE/"scores"/r["image"]/(r["id"]+"_visual.npz"))
        observed=np.load(r["file"])["values"]
        for qi,q in enumerate(s["queries"]):
            node=s["nodes"][q["node"]]
            for ci,cell in enumerate(("NN","NT","TN","TT")):
                va,aa=cell
                inds=[node["j"][aa]]+[s["nodes"][d]["j"][aa] for d in q["donors"]]
                difference=visual[va][node["j"][va]-3].astype(float)-audio["a_"+aa+"_source"][inds].astype(float)+1e-6
                sign=(1 if aa=="T" else 0)-(1 if va=="T" else 0)
                for ti,shift in enumerate(fitted["shifts"]):
                    adjusted=difference+sign*shift
                    dist=np.sqrt(np.einsum('ij,ij->i',adjusted,adjusted))
                    pos=dist[0];neg=dist[1:]
                    result=np.array([pos,sum(neg)/len(neg),sum(neg-pos)/len(neg),sum(1 if d>pos else .5 if d==pos else 0 for d in neg)/len(neg)])
                    errors.append(float(np.max(abs(result-observed[ti,qi,ci]))))
                    count+=1
                old=original[(r["id"],r["image"])]["queries"][qi]["geometry"]["raw"][cell]
                original_error=max(original_error,max(abs(observed[0,qi,ci,mi]-old[m]) for mi,m in enumerate(("positive","negative","margin","rank"))))
        means=np.mean(observed,axis=1)
        dest=derived.setdefault(r["id"],{})
        for mi,metric in enumerate(("positive","negative","margin","rank")):
            original_effects=None
            for ti,condition in enumerate(conditions):
                nn,nt,tn,tt=means[ti,:,mi]
                contrasts={"visual_at_N_audio":tn-nn,"audio_at_N_visual":nt-nn,"interaction":(tt-tn)-(nt-nn),"visual_at_T_audio":tt-nt,"audio_at_T_visual":tt-tn,"diagonal":tt-nn}
                if ti==0:original_effects=contrasts
                for name,val in contrasts.items():
                    dest.setdefault(f"{condition}/{metric}/{name}",[]).append(val)
                    dest.setdefault(f"{condition}/{metric}/{name}_change",[]).append(val-original_effects[name])
                for cell,val in zip(("NN","NT","TN","TT"),(nn,nt,tn,tt)):
                    dest.setdefault(f"{condition}/{metric}/level_{cell}",[]).append(val)
    assert max(errors)<1e-10 and original_error<1e-10
    analysis=read(BASE/"analysis.json");effects=read(BASE/"utterance_effects.json")
    grouping_error=max(abs(np.mean(values)-r["effects"][name]) for r in effects for name,values in derived[r["id"]].items())
    assert grouping_error<1e-10
    statserror=0.
    for name,expected in analysis.items():
        byspeaker={}
        for r in effects:byspeaker.setdefault(r["speaker"],[]).append(r["effects"][name])
        values=np.array([sum(byspeaker[s])/len(byspeaker[s]) for s in sorted(byspeaker)])
        rng=np.random.default_rng(20260926)
        indices=rng.integers(len(values),size=(20000,len(values)))
        boot=np.sum(values[indices],1)/len(values)
        numbers={"speaker_mean":float(np.mean(values)),"speaker_ci95":np.percentile(boot,[2.5,97.5]),"speaker_ci99":np.percentile(boot,[.5,99.5])}
        for key,val in numbers.items():statserror=max(statserror,float(np.max(abs(np.asarray(val)-expected[key]))))
    assert statserror<1e-10
    invariance=read(BASE/"invariance.json")
    assert max(invariance[k] for k in ("max_native_full_distance_error","max_native_query_metric_error","max_guard20_C_error","max_AA_VV_distance_error"))<1e-10
    result={"status":"PASS","fit_max_error":float(fit_error),"independent_query_cell_condition_count":count,"max_source_metric_error":max(errors),"original_prior_replay_max_error":original_error,"grouping_max_error":float(grouping_error),"bootstrap_max_error":statserror,"source_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (BASE/"validation.json").write_text(json.dumps(result,indent=2)+"\n")
    print(result)


if __name__=="__main__":main()
