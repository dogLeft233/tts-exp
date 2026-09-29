"""Audit calibration lag selection and cached CPU matrix reconstruction."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.experiments.check_tts_negative_pool import run as check_pool
from scripts.experiments.tts_static_lag_calibration import BASE, EVAL, MODEL, PARENT, ROOT, read, sha, write


def main():
    p=read(BASE/"protocol.json")
    selection=read(BASE/"selection.json")
    assert sha(MODEL)==p["model_sha256"]
    assert sha(EVAL/"protocol.json")==p["eval_protocol_sha256"]
    ep=read(EVAL/"protocol.json")
    assert p["images"]==ep["images"]
    assert not {r["id"] for r in p["rows"]}&{r["id"] for r in ep["rows"]}
    errors=[]; lag=[]; groups={}; count=0; frame_count=0
    import cv2
    from scripts.experiments.static_image_bridge.score_worker import crop_zero_padded
    cv2.setNumThreads(1)
    for im in p["images"]:
        assert sha(im["path"])==im["sha256"]
        image=cv2.imread(im["path"])
        ref=crop_zero_padded(image,im["score_box"]["box"])
        mask=np.zeros_like(image)
        x1,y1,x2,y2=im["generation_box"]
        mask[y1:y2,x1:x2]=255
        affected=crop_zero_padded(mask,im["score_box"]["box"]).max(2)>0
        outside=cv2.dilate(affected.astype(np.uint8),np.ones((3,3),np.uint8))==0
        for row in p["rows"]:
            sid=row["id"]
            f=BASE/"scores"/im["id"]/(sid+".json")
            r=read(f)
            af=PARENT/"features"/sid/"features.npz"
            vf=f.with_name(sid+"_visual.npz")
            assert sha(af)==r["audio_features_sha256"]
            assert sha(vf)==r["visual_sha256"]
            assert sha(f.with_suffix(".npz"))==r["matrix_sha256"]
            a,v,ms=np.load(af),np.load(vf),np.load(f.with_suffix(".npz"))
            for arm in "NT":
                audio=a["a_"+arm+"_source"]
                n=len(ms[arm]); video=v[arm][:n].astype(np.float64)
                padded=np.pad(audio[:n].astype(np.float64),((15,15),(0,0)))
                matrix=np.stack([np.linalg.norm(video-padded[d:d+n]+1e-6,axis=1) for d in range(31)],1)
                errors.append(float(np.max(abs(matrix-ms[arm]))))
                curve=matrix[20:-20].mean(0)
                k=int(np.argmin(curve))-15
                assert k==-r["cells"][arm]["20"]["offset"]
                lag.append(k); groups.setdefault(im["id"]+"/"+arm,[]).append(k)
                info=r["videos"][arm]
                assert sha(info["path"])==info["sha256"]
                cap=cv2.VideoCapture(info["path"])
                frames=0
                while True:
                    ok,pixels=cap.read()
                    if not ok: break
                    assert np.array_equal(pixels[outside],ref[outside])
                    frames+=1
                cap.release()
                assert frames==info["frames"]
                frame_count+=frames; count+=1
    assert max(errors)<1e-5
    independently_selected=int(np.median(lag))
    assert selection["k"]==independently_selected
    median=[float(np.median(z)) for z in groups.values()]
    assert selection["pass"]==(max(median)-min(median)<=1)
    result={"status":"PASS","matrix_cells":count,"max_error":max(errors),"decoded_frames_outside_generation_identity":frame_count,"k":independently_selected,"calibration_median_range":max(median)-min(median),"source_code_sha256":sha(__file__)}
    write(BASE/"validation.json",result)
    for folder in ("evaluation","evaluation_lag_-1","evaluation_lag_+1"):
        check_pool(BASE/folder)
    print(result)


if __name__=="__main__":
    main()
