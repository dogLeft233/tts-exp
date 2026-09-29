"""Read-only inventory of existing independently generated TTS waveforms."""
import collections
import hashlib
import os
from pathlib import Path

import numpy as np
import soundfile as sf

from scripts.experiments.tts_independent_visual import ROOT, read, sha, write

BASE=ROOT/"runs/tts_instance_inventory_20260926"
META_PATHS=[
    "runs/aishell1_mfa_linear_n25_resample_poly_20260814/qwen_cloud_tts_n25_20260814/tts_meta.json",
    "runs/aishell1_mfa_linear_n25_resample_poly_20260815/phone_trajectory_prosody_phase2_20260815/qwen_cloud_listen3_20260815/tts_meta.json",
    "runs/aishell1_qwen_mfa_linear_n100_20260816/01_tts_retry/tts_meta.json",
    "runs/lrs3_data_supplement_20260903/02_tts_cloud/tts_meta.json",
    "runs/lrs3_masked_tts_new_confirmation_20260902/01_tts/tts_meta.json",
    "runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts_meta.json",
]


def cloud():
    entries=[];counts=[];assets={}
    for relative in META_PATHS:
        path=ROOT/relative;data=read(path);rows=data["results"]
        rows=list(rows.values()) if isinstance(rows,dict) else rows
        present=0;assets[str(path)]=sha(path)
        for r in rows:
            reference=Path(r.get("reference_audio",""));audio=Path(r.get("canonical_16k_audio",""))
            if not reference.is_file() or not audio.is_file():continue
            samples,rate=sf.read(reference,dtype="float64")
            ref_pcm=hashlib.sha256(samples.tobytes()).hexdigest();present+=1
            text="".join(r.get("transcript",r.get("text","")).split())
            key=(ref_pcm,rate,text,data.get("provider"),data.get("model"))
            entry={"manifest":str(path),"id":r.get("sample_id"),"speaker":r.get("speaker_id",r.get("source_group")),"reference":str(reference),"reference_pcm_sha256":ref_pcm,"sample_rate":rate,"transcript":text,"audio":str(audio),"audio_sha256":sha(audio),"reference_file_sha256":sha(reference),"voice":r.get("voice_id")}
            entries.append((key,entry))
        counts.append({"manifest":str(path),"rows":len(rows),"both_media_present":present})
    grouped=collections.defaultdict(list)
    for key,row in entries:grouped[key].append(row)
    repeated=[]
    for rows in grouped.values():
        if len({r["audio_sha256"] for r in rows})<=1:continue
        voice=rows[0]["voice"]
        for r in rows:
            r["same_voice_as_first"]=r.pop("voice")==voice
            assets[r["reference"]]=r["reference_file_sha256"];assets[r["audio"]]=r["audio_sha256"]
        repeated.append(rows)
    write(BASE/"cloud_inventory.json",{"manifests":counts,"repeated_groups":repeated,"checked_exact_reference_pcm_text_model":len(grouped),"assets":assets})
    print("cloud repeats",len(repeated),"utterances;",[(r[0]["id"],len(r),[z["same_voice_as_first"] for z in r]) for r in repeated])


def local_and_natural():
    path=ROOT/"runs/tts_time_instance_20260917_v1/B/tts_generation.json"
    old=read(path);rows=old["rows"]
    byid=collections.defaultdict(list)
    for r in rows:byid[r["sample_id"]].append(r)
    valid=[]
    for sid,items in byid.items():
        assert len(items)==2
        assert len({(r["text_sha256"],r["reference_pcm_sha256"],r["model_id"],r["language"]) for r in items})==1
        files=[Path(r["audio"]) for r in items]
        different=len({sha(p) for p in files})==2
        valid.append({"id":sid,"source_group":items[0]["source_group"],"language":items[0]["language"],"same_input_contract":True,"different_audio":different,"audio":[str(p) for p in files],"audio_sha256":[sha(p) for p in files]})
    natural=[]
    for relative,key in [("runs/aishell1_qwen_mfa_linear_n100_20260816/00_pairs/cohort.json","records"),("runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json","records"),("data/mdc_tts/manifest.json","samples")]:
        data=read(ROOT/relative)[key];groups=collections.defaultdict(list)
        for r in data:
            speaker=r.get("speaker_id") or r.get("source_group") or r.get("source_author_id")
            text="".join(r.get("transcript",r.get("text","")).split()).upper()
            if speaker and text:groups[(str(speaker),text)].append(r.get("sample_id"))
        natural.append({"manifest":relative,"records":len(data),"same_speaker_text_groups":[v for v in groups.values() if len(v)>1]})
    m=read(ROOT/"data/multilingual_tts/manifest.json")
    chinese=[r for r in m["samples"] if r["language_code"]=="zh"]
    for r in chinese:r["exists"]=Path(r["local_path"]).is_file()
    seeds=[]
    for seed in (42,43,44):
        d=read(ROOT/f"data/english_wav2sem_analysis/seed_{seed}/pair_manifest.json")
        seeds.append({"seed":seed,"source_hashes":[r["tts"]["source_qc"]["sha256"] for r in d["records"]]})
    result={"local_english_12pairs":valid,"old_generation_sha256":sha(path),"natural_manifest_checks":natural,"new_chinese_css10":chinese,"english_seed42_43_44_same_tts":seeds[0]["source_hashes"]==seeds[1]["source_hashes"]==seeds[2]["source_hashes"],"cloud_key_env_present":bool(os.environ.get("DASHSCOPE_API_KEY"))}
    write(BASE/"local_natural_inventory.json",result)
    print("local English pairs",len(valid),"different",sum(v["different_audio"] for v in valid),"natural repeated groups",[len(x["same_speaker_text_groups"]) for x in natural])


if __name__=="__main__":cloud();local_and_natural()
