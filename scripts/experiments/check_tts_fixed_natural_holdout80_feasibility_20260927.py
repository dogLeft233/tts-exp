"""Independent original PCM16 scalar/identity/resource feasibility check."""
from pathlib import Path
import json,hashlib,wave,collections
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_fixed_natural_holdout80_feasibility_20260927'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
rows=read(OUT/'rows.json');p=read(OUT/'feasibility_protocol.json');target=p['target_RMS'];maximum=0.;peak=0.;speakers=collections.Counter()
for f,h in read(OUT/'input_seal.json')['hashes'].items():assert sha(f)==h
for r in rows:
 assert sha(r['audio']['path'])==r['audio']['sha256'];speakers[r['speaker']]+=1
 with wave.open(r['audio']['path'],'rb') as w:
  assert w.getframerate()==16000 and w.getnchannels()==1 and w.getsampwidth()==2;pcm=np.frombuffer(w.readframes(w.getnframes()),dtype='<i2').astype(np.int64)
 rms=np.sqrt(float(np.sum(pcm*pcm,dtype=np.int64))/len(pcm))/32768;gain=target/rms;y=(pcm.astype(float)/32768)*gain;maximum=max(maximum,abs(rms-r['original_RMS']),abs(gain-r['gain']));peak=max(peak,float(abs(y).max()),float(abs(y.astype(np.float32)).max()));assert abs(y).max()<=.98 and r['L']>40
 assert sha(r['old_feature'])==r['old_feature_sha256'] and sha(r['old_video'])==r['old_video_sha256']
assert len(rows)==80 and len(speakers)==40 and set(speakers.values())=={2} and maximum<1e-12
s=read(OUT/'summary.json');b=s['storage_upper_estimate'];assert b['with_10percent_bytes']==int(sum(b[k] for k in ['new_waveforms','new_frontend','AV_uncompressed_plus_NPZ_overhead','first2_FIXED_videos','largest_temporary_video','uncompressed_fourcell_raw_unit_matrices','metadata_controls_reports_reserve'])*1.1);assert b['budget_pass'] and s['reserve_after_estimate_GiB']>5
result={'status':'PASS','rows':80,'speakers':40,'max_scalar_difference':maximum,'max_fixed_peak':peak,'cache_GPU_replay_still_required':True,'new_GPU_scores':0,'checker_sha256':sha(__file__)};(OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
