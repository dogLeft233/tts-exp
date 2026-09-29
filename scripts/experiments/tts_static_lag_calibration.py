"""Independent calibration clips rendered under the existing static protocol."""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

from scripts.experiments.tts_independent_visual import OUT, PARENT, ROOT, read, sha, write
from scripts.experiments.tts_pcm_residual import metrics
from scripts.experiments.tts_prepost_mel import BASE as EVAL, MODEL
from scripts.experiments.tts_raw_video_transfer import render

BASE=ROOT/"runs/tts_static_lag_calibration_20260926"


def freeze():
    old={r["id"]:r for r in read(PARENT/"protocol.json")["rows"]}
    ep=read(EVAL/"protocol.json")
    rows=[]
    for r in read(OUT/"protocol.json")["rows"]:
        if r["split"]!="calibration":
            continue
        rows.append({"id":r["id"],"speaker":r["speaker"],"audio":{a:{"path":old[r["id"]]["cells"][a]["audio"],"sha256":old[r["id"]]["cells"][a]["audio_sha256"]} for a in "NT"}})
    assert len(rows)==26 and not {r["id"] for r in rows}&{r["id"] for r in ep["rows"]}
    p={"rows":rows,"images":ep["images"],"eval_protocol_sha256":sha(EVAL/"protocol.json"),"split_sha256":sha(OUT/"protocol.json"),"model_sha256":sha(MODEL),"frontend_sha256":sha(ROOT/"third_party/Wav2Lip/audio.py"),
       "selection":"all26 predetermined calibration IDs, no eval-dependent selection",
       "render":"exact parent static N/T native mel path; FFV1 score crop; SyncNet same MJPEG extraction; exact source PCM audio embedding",
       "lag_rule":"k=trunc(median(all valid26*3*2 guard20 best audio-minus-visual index)); failed cells retained as missing; no model or outcome retries",
       "gate":"each arm/image calibration k-median range <=1 frame; otherwise common k retained but transfer gate fails",
       "evaluation":"reuse frozen negative-pool protocol, common k and k+-1, independently rebuild annotation/index support; exploratory repair after eval outcomes already observed",
       "source_code_sha256":sha(__file__)}
    if (BASE/"protocol.json").exists():
        assert read(BASE/"protocol.json")==p
    write(BASE/"protocol.json",p)


def mels():
    sys.path.insert(0,str(ROOT/"third_party/Wav2Lip"))
    import audio
    p=read(BASE/"protocol.json")
    assert sha(audio.__file__)==p["frontend_sha256"]
    for r in p["rows"]:
        arrays={}
        for a,info in r["audio"].items():
            assert sha(info["path"])==info["sha256"]
            arrays[a]=audio.melspectrogram(audio.load_wav(info["path"],16000)).astype(np.float32)
        dest=BASE/"mels"/(r["id"]+".npz")
        dest.parent.mkdir(exist_ok=True,parents=True)
        np.savez_compressed(dest,**arrays)
        write(dest.with_suffix(".json"),{"sha256":sha(dest)})


def run():
    import cv2
    import torch
    from scripts.experiments.masked_tts_tfg_probe.direct_mel import _load_model, chunk_mels
    from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded
    from scripts.experiments.tts_native_gain_attribution.common import gpu_compute_pids,gpu_lease
    from scripts.experiments.tts_native_gain_attribution.syncnet import SyncNetEngine
    p=read(BASE/"protocol.json")
    assert sha(MODEL)==p["model_sha256"]
    cv2.setNumThreads(1)
    torch.set_num_threads(2)
    torch.manual_seed(20260926)
    np.random.seed(20260926)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False
    with gpu_lease(gpu_peak_bytes=8<<30,disk_temp_bytes=1<<30,disk_persistent_bytes=3<<30) as gate:
        write(BASE/"compute_gate.json",gate)
        model=_load_model(MODEL,"cuda")
        engine=SyncNetEngine(batch_size=32,device="cuda")
        try:
            for image in p["images"]:
                assert sha(image["path"])==image["sha256"]
                frame=cv2.imread(image["path"])
                x1,y1,x2,y2=image["generation_box"]
                face=cv2.resize(frame[y1:y2,x1:x2],(96,96))
                masked=face.copy(); masked[48:]=0
                channels=np.concatenate([masked,face],2).transpose(2,0,1).astype(np.float32)/255
                it=torch.from_numpy(channels).unsqueeze(0).to("cuda")
                for r in p["rows"]:
                    sid=r["id"]
                    dest=BASE/"scores"/image["id"]/(sid+".json")
                    if dest.exists():
                        continue
                    assert not set(gpu_compute_pids())-{os.getpid()}
                    mf=BASE/"mels"/(sid+".npz")
                    assert sha(mf)==read(mf.with_suffix(".json"))["sha256"]
                    mel=np.load(mf)
                    af=PARENT/"features"/sid/"features.npz"
                    assert sha(af)==read(PARENT/"scores"/(sid+".json"))["features_sha256"]
                    audio=np.load(af)
                    matrices,visuals,videos={},{},{}
                    for a in "NT":
                        chunks=chunk_mels(mel[a],25)
                        frames=[]
                        for start in range(0,len(chunks),32):
                            block=np.asarray(chunks[start:start+32],dtype=np.float32)
                            mt=torch.from_numpy(block[:,None]).to("cuda")
                            with torch.inference_mode():
                                pred=model(mt,it.expand(len(block),-1,-1,-1)).cpu().numpy().transpose(0,2,3,1)*255
                            for fo in pred:
                                full=frame.copy()
                                full[y1:y2,x1:x2]=cv2.resize(fo.astype(np.uint8),(x2-x1,y2-y1))
                                frames.append(crop_zero_padded(full,image["score_box"]["box"]))
                        frames=np.stack(frames)
                        vp=BASE/"videos"/image["id"]/sid/(a+".avi")
                        render(frames,np.arange(len(frames))/25,vp)
                        v,meta=engine.extract_visual(vp)
                        visuals[a]=v
                        n=min(len(v),len(audio["a_"+a+"_source"]))
                        matrices[a]=engine.distance_matrix(v[:n],audio["a_"+a+"_source"][:n])
                        videos[a]={"path":str(vp),"sha256":sha(vp),"frames":len(frames),"visual_metadata":meta}
                    dest.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(dest.with_suffix(".npz"),**matrices)
                    vf=dest.with_name(sid+"_visual.npz")
                    np.savez_compressed(vf,**visuals)
                    write(dest,{"id":sid,"image":image["id"],"speaker":r["speaker"],"videos":videos,"cells":{a:metrics(m) for a,m in matrices.items()},"visual_sha256":sha(vf),"matrix_sha256":sha(dest.with_suffix(".npz")),"audio_features_sha256":sha(af),"mel_sha256":sha(mf)})
                    print("calibrated",image["id"],sid,flush=True)
        finally:
            engine.close()


def select():
    p=read(BASE/"protocol.json")
    values,groups,receipts=[],{},{}
    for im in p["images"]:
        for a in "NT":
            local=[]
            for r in p["rows"]:
                f=BASE/"scores"/im["id"]/(r["id"]+".json")
                receipt=read(f)
                receipts[str(f)]=sha(f)
                local.append(-receipt["cells"][a]["20"]["offset"])
            groups[im["id"]+"/"+a]={"median":float(np.median(local)),"k":local}
            values.extend(local)
    k=int(np.median(values))
    med=[g["median"] for g in groups.values()]
    result={"k":k,"median":float(np.median(values)),"groups":groups,"pass":max(med)-min(med)<=1,"cells":len(values),"receipts":receipts,"protocol_sha256":sha(BASE/"protocol.json")}
    write(BASE/"selection.json",result)
    print("locked static calibration k",k,"pass",result["pass"],flush=True)


def evaluate():
    from scripts.experiments import tts_negative_pool as pool
    selection=read(BASE/"selection.json")
    old=pool.BASE
    for delta in (0,-1,1):
        pool.BASE=BASE/("evaluation" if delta==0 else f"evaluation_lag_{delta:+d}")
        p=read(old/"protocol.json")
        p["fixed_audio_minus_visual_index"]=selection["k"]+delta
        p["static_calibration_selection_sha256"]=sha(BASE/"selection.json")
        p["static_calibration_protocol_sha256"]=sha(BASE/"protocol.json")
        p["exploratory_repair"]=True
        p["sensitivity_delta"]=delta
        write(pool.BASE/"protocol.json",p)
        pool.support(); pool.evaluate(); pool.summarize()
    pool.BASE=old


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("freeze","mels","run","select","evaluate"))
    globals()[parser.parse_args().stage]()
