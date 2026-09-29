"""Four-cell representation pairing on fixed common event/phase support."""
from __future__ import annotations

import argparse

import numpy as np

from scripts.experiments.tts_independent_visual import PARENT, ROOT, read, sha, write
from scripts.experiments.tts_pcm_residual import cluster
from scripts.experiments.tts_prepost_mel import BASE as SOURCE
from scripts.experiments.tts_static_lag_calibration import BASE as CAL

BASE=ROOT/"runs/tts_event_cross_20260926"
CONTRASTS={"visual_at_N_audio":{"TN":1,"NN":-1},"audio_at_N_visual":{"NT":1,"NN":-1},
           "interaction":{"TT":1,"NN":1,"TN":-1,"NT":-1},
           "visual_at_T_audio":{"TT":1,"NT":-1},"audio_at_T_visual":{"TT":1,"TN":-1},
           "diagonal":{"TT":1,"NN":-1}}


def freeze():
    selection=read(CAL/"selection.json")
    p={"static_calibration_sha256":sha(CAL/"selection.json"),"k":selection["k"],"gate":selection["pass"],
       "source_code_sha256":sha(__file__),"conditions":"main static-calibration k, plus both k-1/k+1; no condition chosen by eval",
       "support":"exactly each repaired negative-pool common query/donor occurrence+phase support; no new support selection",
       "cells":"q(V_N,A_N),q(V_N,A_T),q(V_T,A_N),q(V_T,A_T); indices chosen on each arm own native grid for same common event IDs",
       "metric":"query mean donor distance minus positive distance and conditional rank; raw and unit Euclidean eps1e-6",
       "contrasts":CONTRASTS,"primary":"raw margin visual_at_N_audio, audio_at_N_visual, interaction; rank and unit geometry diagnostics; 99%CI conservatively read across three contrasts",
       "limits":"exploratory representation pairing in common annotation coordinates, not physically replacing raw natural audio, not official confidence decomposition",
       "statistics":"three images within utterance, speaker equal,20000 bootstrap seed20260926,95/99CI"}
    if (BASE/"protocol.json").exists():
        assert read(BASE/"protocol.json")==p
    write(BASE/"protocol.json",p)


def distance(x,y):
    return np.sqrt(np.sum((np.asarray(x,np.float64)-np.asarray(y,np.float64)+1e-6)**2,axis=-1))


def run():
    p=read(BASE/"protocol.json")
    assert sha(CAL/"selection.json")==p["static_calibration_sha256"]
    source=read(SOURCE/"protocol.json")
    for folder in ("evaluation","evaluation_lag_-1","evaluation_lag_+1"):
        prior=CAL/folder
        protocol=read(prior/"protocol.json")
        support=read(prior/"support.json")
        k=protocol["fixed_audio_minus_visual_index"]
        diagonal={(r["id"],r["image"]):r for r in read(prior/"scores.json")}
        rows=[]; max_diag=0.
        for s in support["rows"]:
            if not s["eligible"]: continue
            af=PARENT/"features"/s["id"]/"features.npz"
            assert sha(af)==support["assets"][str(af)]
            audio=np.load(af)
            for image in source["images"]:
                vf=SOURCE/"scores"/image["id"]/(s["id"]+"_visual.npz")
                assert sha(vf)==support["assets"][str(vf)]
                visual=np.load(vf)
                arrays={}
                for geometry in ("raw","unit"):
                    arrays[geometry]={"A":{},"V":{}}
                    for a in "NT":
                        for modality,arr in [("A",audio["a_"+a+"_source"]),("V",visual[a])]:
                            arrays[geometry][modality][a]=arr if geometry=="raw" else arr/np.linalg.norm(arr,axis=1)[:,None]
                records=[]
                for qn,q in enumerate(s["queries"]):
                    node=s["nodes"][q["node"]]
                    result={"node":q["node"],"geometry":{}}
                    for geometry,feat in arrays.items():
                        cells={}
                        for va in "NT":
                            for aa in "NT":
                                v=feat["V"][va][node["j"][va]-k]
                                a=feat["A"][aa]
                                pos=float(distance(v,a[node["j"][aa]]))
                                neg=distance(v,a[[s["nodes"][dn]["j"][aa] for dn in q["donors"]]])
                                cells[va+aa]={"positive":pos,"negative":float(neg.mean()),"margin":float(neg.mean()-pos),"rank":float(np.mean((neg>pos)+.5*(neg==pos)))}
                                if va==aa:
                                    old=diagonal[(s["id"],image["id"])]["arms"][va]["queries"][qn]["pools"]["matched"]["metrics"][geometry]
                                    max_diag=max(max_diag,max(abs(cells[va+aa][m]-old[m]) for m in cells[va+aa]))
                        result["geometry"][geometry]=cells
                    records.append(result)
                rows.append({"id":s["id"],"speaker":s["speaker"],"image":image["id"],"queries":records})
        assert max_diag<1e-9
        dest=BASE/folder
        write(dest/"scores.json",rows)
        write(dest/"receipt.json",{"support_sha256":sha(prior/"support.json"),"protocol_sha256":sha(prior/"protocol.json"),"k":k,"max_diagonal_error":max_diag})
        utterances={}
        for r in rows:
            scalars=utterances.setdefault((r["id"],r["speaker"]),{})
            for geom in ("raw","unit"):
                for metric in ("positive","negative","margin","rank"):
                    cell={c:float(np.mean([q["geometry"][geom][c][metric] for q in r["queries"]])) for c in ("NN","NT","TN","TT")}
                    for c,value in cell.items():
                        scalars.setdefault(f"{geom}_{metric}_level_{c}",[]).append(value)
                    for name,weights in CONTRASTS.items():
                        value=sum(cell[c]*w for c,w in weights.items())
                        scalars.setdefault(f"{geom}_{metric}_{name}",[]).append(value)
        names=next(iter(utterances.values())).keys()
        result={name:cluster([np.mean(v[name]) for v in utterances.values()],[k[1] for k in utterances]) for name in names}
        write(dest/"analysis.json",result)
        write(dest/"utterance_effects.json",[{"id":key[0],"speaker":key[1],"effects":{n:float(np.mean(v)) for n,v in vals.items()}} for key,vals in utterances.items()])
        print(folder,"k",k,flush=True)
        for name in CONTRASTS:
            r=result["raw_margin_"+name]
            print(name,r["speaker_mean"],r["speaker_ci99"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("freeze","run"))
    globals()[parser.parse_args().stage]()
