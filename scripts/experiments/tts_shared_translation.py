"""Calibrated common A/V translation with exactly invariant native geometry."""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np

from scripts.experiments.tts_event_cross import BASE as CROSS, CONTRASTS
from scripts.experiments.tts_independent_visual import PARENT, ROOT, read, sha, write
from scripts.experiments.tts_negative_pool import make_support
from scripts.experiments.tts_phone_video_transfer import phones
from scripts.experiments.tts_prepost_mel import BASE as SOURCE
from scripts.experiments.tts_static_lag_calibration import BASE as CAL
from scripts.experiments.tts_pcm_residual import cluster

BASE=ROOT/"runs/tts_shared_translation_20260926"
CELLS=("NN","NT","TN","TT")
METRICS=("positive","negative","margin","rank")
CONDITIONS=("original","subtract_delta","add_delta",*[f"random_{i:02d}" for i in range(16)])


def freeze():
    calibration=read(CAL/"protocol.json")
    summary_path=ROOT/"runs/aishell1_qwen_mfa_linear_n100_20260816/02_mfa_mandarin341_ready/mfa_summary.json"
    summary=read(summary_path)["records"]
    rows,excluded,assets=[],[],{}
    for r in calibration["rows"]:
        sid=r["id"]
        grids={}
        for arm,label in (("N","natural"),("T","tts")):
            entry=summary[sid][label]
            assert entry["audio_sha256"]==r["audio"][arm]["sha256"]
            assert sha(entry["textgrid"])==entry["textgrid_sha256"]
            grids[arm]={"path":entry["textgrid"],"sha256":entry["textgrid_sha256"]}
            assets[entry["textgrid"]]=entry["textgrid_sha256"]
        if [z[0] for z in phones(grids["N"]["path"])]!=[z[0] for z in phones(grids["T"]["path"])]:
            excluded.append({"id":sid,"speaker":r["speaker"],"reason":"phone sequence mismatch"})
            continue
        lengths=[]
        for im in calibration["images"]:
            f=CAL/"scores"/im["id"]/(sid+".json")
            receipt=read(f)
            vf=f.with_name(sid+"_visual.npz");af=PARENT/"features"/sid/"features.npz"
            for file,expected in ((vf,receipt["visual_sha256"]),(af,receipt["audio_features_sha256"]),(f.with_suffix(".npz"),receipt["matrix_sha256"])):
                assert sha(file)==expected
                assets[str(file)]=expected
            m=np.load(f.with_suffix(".npz"))
            lengths.append({a:len(m[a]) for a in "NT"})
        assert lengths[0]==lengths[1]==lengths[2]
        s=make_support(r,grids,lengths[0],3)
        s["textgrids"]=grids
        rows.append(s)
        if not s["eligible"]:
            excluded.append({"id":sid,"speaker":r["speaker"],"reason":"original event support gate","queries":len(s["queries"]),"occurrences":s["n_occurrences"]})
    good=[r for r in rows if r["eligible"]]
    assert len(good)==20 and len({r["speaker"] for r in good})==12
    assert read(CAL/"selection.json")["k"]==3
    support={"rows":rows,"excluded":excluded,"assets":assets,"summary_sha256":sha(summary_path)}
    p={"protocol":"shared_domain_translation_v1","calibration_protocol_sha256":sha(CAL/"protocol.json"),"lag_selection_sha256":sha(CAL/"selection.json"),
       "eval_support_sha256":sha(CAL/"evaluation/support.json"),"cross_protocol_sha256":sha(CROSS/"protocol.json"),"k":3,
       "calibration_support":"20/26 utterances,12 speakers,890query; strict existing MFA/index gate; six exclusions retained",
       "delta":"per query mean within utterance; audio once; video per image then mean3; utterance equal within speaker; speaker equal separately for A and V; delta=(deltaA+deltaV)/2",
       "conditions":list(CONDITIONS),"random":"PCG64 seed20260926,standard_normal(16,1024),each direction normalized times norm(delta); subtract from both T modalities",
       "fidelity":"float64 common translation; NN/TT distances,q,AA,VV and guard20 C invariant; guard0 including moved zero sentinel only coordinate-identity diagnostic",
       "primary":"raw margin I condition-original; cross simple effects TN-NN and NT-NN; raw rank and positive/negative I secondary",
       "statistics":"images mean within utterance,utterance mean within speaker,speaker equal,20000 bootstrap seed20260926,95/99CI",
       "interpretation":"random directions descriptive,not randomization p; no native gain attribution percent; no unit renormalization; no eval fitting",
       "source_code_sha256":sha(__file__)}
    if (BASE/"protocol.json").exists():assert read(BASE/"protocol.json")==p
    write(BASE/"protocol.json",p);write(BASE/"calibration_support.json",support)


def fit():
    support=read(BASE/"calibration_support.json")
    images=read(CAL/"protocol.json")["images"]
    utterances=[];delta_rows=[]
    for r in support["rows"]:
        if not r["eligible"]:continue
        sid=r["id"]
        idx={a:np.array([r["nodes"][q["node"]]["j"][a] for q in r["queries"]]) for a in "NT"}
        af=PARENT/"features"/sid/"features.npz"
        assert sha(af)==support["assets"][str(af)]
        audio=np.load(af)
        da=(audio["a_T_source"][idx["T"]].astype(np.float64)-audio["a_N_source"][idx["N"]].astype(np.float64)).mean(0)
        visual_means=[]
        for im in images:
            vf=CAL/"scores"/im["id"]/(sid+"_visual.npz")
            assert sha(vf)==support["assets"][str(vf)]
            v=np.load(vf)
            visual_means.append((v["T"][idx["T"]-3].astype(np.float64)-v["N"][idx["N"]-3].astype(np.float64)).mean(0))
        dv=np.mean(visual_means,0)
        delta_rows.append(np.stack([da,dv]));utterances.append({"id":sid,"speaker":r["speaker"],"queries":len(r["queries"])})
    delta_rows=np.stack(delta_rows)
    speakers=sorted({r["speaker"] for r in utterances})
    speaker_arrays=np.stack([delta_rows[[i for i,r in enumerate(utterances) if r["speaker"]==s]].mean(0) for s in speakers])
    da,dv=speaker_arrays.mean(0)
    delta=(da+dv)/2
    rng=np.random.Generator(np.random.PCG64(20260926))
    random=rng.standard_normal((16,1024))
    random=random/np.linalg.norm(random,axis=1)[:,None]*np.linalg.norm(delta)
    shifts=np.concatenate([np.zeros((1,1024)),delta[None],-delta[None],random])
    np.savez_compressed(BASE/"translation.npz",delta_A=da,delta_V=dv,delta=delta,shifts=shifts,utterance_means=delta_rows,speaker_means=speaker_arrays)
    cosine=float(np.dot(da,dv)/(np.linalg.norm(da)*np.linalg.norm(dv)))
    write(BASE/"fit.json",{"utterances":utterances,"speakers":speakers,"norm_A":float(np.linalg.norm(da)),"norm_V":float(np.linalg.norm(dv)),"norm_delta":float(np.linalg.norm(delta)),"cosine_A_V":cosine,"angle_degrees":float(np.degrees(np.arccos(np.clip(cosine,-1,1)))),"translation_sha256":sha(BASE/"translation.npz"),"calibration_support_sha256":sha(BASE/"calibration_support.json")})
    print("fit",len(utterances),len(speakers),"norms",np.linalg.norm(da),np.linalg.norm(dv),np.linalg.norm(delta),"angle",np.degrees(np.arccos(cosine)))


def metric(x,y):
    d=np.linalg.norm(x-y+1e-6,axis=1)
    p,n=d[0],d[1:]
    return np.array([p,n.mean(),n.mean()-p,np.mean((n>p)+.5*(n==p))])


def evaluate():
    p=read(BASE/"protocol.json")
    assert sha(CAL/"evaluation/support.json")==p["eval_support_sha256"]
    support=read(CAL/"evaluation/support.json")
    shifts=np.load(BASE/"translation.npz")["shifts"]
    assert sha(BASE/"translation.npz")==read(BASE/"fit.json")["translation_sha256"]
    images=read(SOURCE/"protocol.json")["images"]
    manifest=[];max_native=0.;max_q=0.;max_official=0.;max_self=0.;official_native=[]
    for r in support["rows"]:
        if not r["eligible"]:continue
        sid=r["id"]
        af=PARENT/"features"/sid/"features.npz"
        assert sha(af)==support["assets"][str(af)]
        audio=np.load(af)
        for im in images:
            vf=SOURCE/"scores"/im["id"]/(sid+"_visual.npz")
            assert sha(vf)==support["assets"][str(vf)]
            visual=np.load(vf)
            a={z:audio["a_"+z+"_source"].astype(np.float64) for z in "NT"}
            v={z:visual[z].astype(np.float64) for z in "NT"}
            values=np.zeros((len(shifts),len(r["queries"]),4,4),np.float64)
            for qi,q in enumerate(r["queries"]):
                node=r["nodes"][q["node"]]
                js={z:[node["j"][z]]+[r["nodes"][dn]["j"][z] for dn in q["donors"]] for z in "NT"}
                for ci,cell in enumerate(CELLS):
                    va,aa=cell
                    x=v[va][node["j"][va]-3]
                    y=a[aa][js[aa]]
                    for ti,s in enumerate(shifts):
                        xs=x-s if va=="T" else x
                        ys=y-s if aa=="T" else y
                        values[ti,qi,ci]=metric(xs,ys)
                        if va==aa:
                            max_q=max(max_q,float(np.max(abs(values[ti,qi,ci]-values[0,qi,ci]))))
                    # T modality self-distance invariance on all observed donor pairs.
                for modality,features,offset in (("A",a["T"],0),("V",v["T"],3)):
                    x=features[node["j"]["T"]-offset]
                    y=features[np.array(js["T"])-offset]
                    baseline=np.linalg.norm(x-y,axis=1)
                    for s in shifts[1:]:
                        max_self=max(max_self,float(np.max(abs(np.linalg.norm((x-s)-(y-s),axis=1)-baseline))))
            # Full native matrix coordinate check, moving virtual padding as well.
            # Guard20 has no virtual padding and is the actual fidelity endpoint.
            n=r["lengths"]["T"]
            v0=v["T"][:n];a0=np.pad(a["T"][:n],((15,15),(0,0)))
            original=np.stack([np.linalg.norm(v0-a0[d:d+n]+1e-6,axis=1) for d in range(31)],axis=1)
            base_curve=original[20:-20].mean(0)
            base_c=float(np.median(base_curve)-min(base_curve))
            for s in shifts:
                translated=np.stack([np.linalg.norm((v0-s)-(a0[d:d+n]-s)+1e-6,axis=1) for d in range(31)],axis=1)
                max_native=max(max_native,float(np.max(abs(translated-original))))
                curve=translated[20:-20].mean(0)
                max_official=max(max_official,abs(float(np.median(curve)-min(curve))-base_c))
            official_native.append({"id":sid,"image":im["id"],"T_guard20_C":base_c})
            dest=BASE/"scores"/im["id"]/(sid+".npz");dest.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest,values=values)
            manifest.append({"id":sid,"speaker":r["speaker"],"image":im["id"],"file":str(dest),"sha256":sha(dest),"queries":len(r["queries"])})
        print("scored",sid,flush=True)
    assert max(max_native,max_q,max_official,max_self)<1e-10
    write(BASE/"manifest.json",manifest)
    write(BASE/"invariance.json",{"max_native_full_distance_error":max_native,"max_native_query_metric_error":max_q,"max_guard20_C_error":max_official,"max_AA_VV_distance_error":max_self,"padding":"translated virtual zero sentinel for full-matrix identity only; guard20 has no sentinel","official_native":official_native})


def summarize():
    rows=read(BASE/"manifest.json");byid={}
    for r in rows:
        assert sha(r["file"])==r["sha256"]
        values=np.load(r["file"])["values"].mean(1)
        out=byid.setdefault((r["id"],r["speaker"]),{})
        for ti,condition in enumerate(CONDITIONS):
            for mi,metric_name in enumerate(METRICS):
                cells=dict(zip(CELLS,values[ti,:,mi]))
                original=dict(zip(CELLS,values[0,:,mi]))
                for name,weights in CONTRASTS.items():
                    val=sum(cells[c]*w for c,w in weights.items())
                    orig=sum(original[c]*w for c,w in weights.items())
                    out.setdefault(f"{condition}/{metric_name}/{name}",[]).append(val)
                    out.setdefault(f"{condition}/{metric_name}/{name}_change",[]).append(val-orig)
                for cell,value in cells.items():out.setdefault(f"{condition}/{metric_name}/level_{cell}",[]).append(float(value))
    analysis={}
    for name in next(iter(byid.values())):
        analysis[name]=cluster([np.mean(v[name]) for v in byid.values()],[k[1] for k in byid])
    write(BASE/"analysis.json",analysis)
    write(BASE/"utterance_effects.json",[{"id":key[0],"speaker":key[1],"effects":{name:float(np.mean(v)) for name,v in values.items()}} for key,values in byid.items()])
    for condition in CONDITIONS[:3]:
        for name in ("interaction","interaction_change","visual_at_N_audio","audio_at_N_visual"):
            r=analysis[f"{condition}/margin/{name}"]
            print(condition,name,r["speaker_mean"],r["speaker_ci99"])


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("freeze","fit","evaluate","summarize"))
    globals()[parser.parse_args().stage]()
