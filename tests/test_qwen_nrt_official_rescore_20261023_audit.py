import importlib.util
import json
import pickle
import wave
from pathlib import Path

import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "scripts/experiments/qwen_nrt_official_rescore_20261023_audit.py"
SPEC = importlib.util.spec_from_file_location("nrt_official_audit", SOURCE)
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_raw_distance_audit_detects_a_forged_printed_score(tmp_path):
    run, source = tmp_path / "run", tmp_path / "source"
    (run / "scores").mkdir(parents=True)
    source.mkdir()
    def save(path, obj):
        path.write_text(json.dumps(obj), encoding="utf8")
    wav, video = tmp_path / "canonical.wav", tmp_path / "source.mp4"
    pcm = np.arange(100, dtype="<i2").tobytes()
    with wave.open(str(wav), "wb") as h:
        h.setparams((1,2,16000,100,"NONE","not compressed"))
        h.writeframes(pcm)
    video.write_bytes(b"hash-only mock media; actual runner independently decodes real video")
    curve = np.abs(np.arange(31)-14).astype(np.float32) + 2
    confidence = float(np.median(curve)-2)
    arrays, tracks = run / "activesd.pckl", run / "tracks.pckl"
    arrays.write_bytes(pickle.dumps([np.tile(curve, (8, 1))]))
    tracks.write_bytes(pickle.dumps([{"track": {"frame": np.arange(13)}}]))
    roles, vids, audios = ("N","R","T1","T2","T3"), [], []
    for role in roles:
        vids.append({"sample_id":"one", "role":role, "path":str(video), "sha256":MOD.digest(video)})
        audios.append({"sample_id":"one", "role":role, "path":str(wav), "sha256":MOD.digest(wav),
                       "pcm_sha256":MOD.hashlib.sha256(pcm).hexdigest()})
        log = run / f"{role}.log"
        log.write_text(f"AV offset: 1\nMin dist: 2.000\nConfidence: {confidence:.3f}\n")
        save(run / "scores" / f"one_{role}.json", {"sample_id":"one", "role":role,
             "reference":role, "syncnet_log":str(log), "sync_c":confidence, "sync_d":2,
             "av_offset":1, "tracks":[{}], "activesd_path":str(arrays), "tracks_pickle_path":str(tracks)})
    save(source / "video_manifest.json", {"records":vids})
    save(source / "audio_manifest.json", {"records":audios})
    save(source / "analysis.json", {"records":[{"sample_id":"one", "draw_reliability":{r:{"qualified":True} for r in roles[2:]},
         "policies":{p:{r:{"C":1} for r in roles} for p in ["common","native_full","equal_count_spread"]}}]})
    zeros = {k:0 for k in ["T-N","T-R","R-N"]}
    stat = {"mean":0,"ci95":[0,0],"ci97_5":[0,0]}
    block = {"metrics":{m:{"contrasts":{k:stat for k in zeros}} for m in ["C","D","offset"]},
             "qualified_supplement":{m:{k:stat for k in zeros} for m in ["C","D","offset"]},
             "protocol_changes_C":{p:{k:stat for k in zeros} for p in ["common","native_full","equal_count_spread"]},
             "per_seed_C":{r:{k:stat for k in ["T-N","T-R"]} for r in roles[2:]}}
    save(run / "analysis.json", {"records":[{"sample_id":"one", "cohort":"new30", "official_contrasts":{m:zeros for m in ["C","D","offset"]}}],
                                    "cohorts":{"new30":block,"all40":block,"pilot10":{}}})
    save(run / "analysis_protocol.json", {"bootstrap":{"seed":20261023,"replicates":100}})
    assert MOD.audit(run, source)["status"] == "PASS"
    (run / "T3.log").write_text("AV offset: 1\nMin dist: 2.000\nConfidence: 99.000\n")
    failure = MOD.audit(run, source)
    assert failure["status"] == "FAIL"
    assert any("First-track receipt mismatch" in e for e in failure["errors"])
    assert any("Raw distance/printed/support mismatch" in e for e in failure["errors"])
