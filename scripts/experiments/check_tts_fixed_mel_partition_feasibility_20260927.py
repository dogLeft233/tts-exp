"""Independent saved-cell endpoint partition and exact budget component audit."""
from pathlib import Path
import hashlib,json
import numpy as np
import soundfile as sf
import librosa
from scipy import signal
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_fixed_mel_partition_feasibility_20260927';OLD=ROOT/'runs/tts_fixed_level_generation_cross_20260927'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
if __name__=='__main__':
 p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip()
 bindings={**p['dependencies'],**read(OUT/'input_hashes.json'),**read(OUT/'natural80_binding_and_budget_hashes.json'),**read(OUT/'budget74_hashes.json')}
 for f,h in bindings.items():assert sha(f)==h
 got={(r['id'],r['arm']):r for r in read(OUT/'calibration_rows.json')};n=0;exact=0;frontends=0
 bank=librosa.filters.mel(sr=16000,n_fft=800,n_mels=80,fmin=55,fmax=7600)
 for r in read(OLD/'support.json'):
  if r['split']!='calibration':continue
  for arm in ['N','T']:
   a=np.load(r['arms'][arm]['raw']['frontend'])['mel'];b=np.load(r['arms'][arm]['FIXED']['frontend'])['mel'];u=~((a==-4)|(a==4)|(b==-4)|(b==4));up=a.copy();sat=a.copy();up[u]=b[u];sat[~u]=b[~u];row=got[r['id'],arm]
   for label,stored in [('raw',a),('FIXED',b)]:
    x=sf.read(r['arms'][arm][label]['waveform'],dtype='float32')[0]
    pre=signal.lfilter([1,-.97],[1],x)
    spec=librosa.stft(y=pre,n_fft=800,hop_length=200,win_length=800,window='hann',center=True,pad_mode='constant')
    mag=bank@abs(spec);db=20*np.log10(np.maximum(np.exp(-100/20*np.log(10)),mag))-20
    rebuilt=np.clip(8*((db+100)/100)-4,-4,4).astype(np.float32)
    assert np.array_equal(rebuilt,stored);frontends+=1
   for label,x in [('M0',a),('M1',b),('U_mask',u),('U_only',up),('S_only',sat)]:assert hashlib.sha256(x.tobytes()).hexdigest()==row['array_hashes'][label]
   # Explicit finite precision closure with float64 representation of float32 endpoints.
   d=b.astype(float)-a.astype(float);d1=up.astype(float)-a.astype(float);d2=sat.astype(float)-a.astype(float)
   assert np.array_equal(d1+d2,d) and np.isfinite(up).all() and np.isfinite(sat).all() and np.all(abs(up)<=4) and np.all(abs(sat)<=4)
   assert int(u.sum())==row['U_count'] and int((~u).sum())==row['S_count']
   for k,v in [('delta_abs_sum_U',abs(d1).sum()),('delta_abs_sum_S',abs(d2).sum()),('delta_sq_sum_U',(d1*d1).sum()),('delta_sq_sum_S',(d2*d2).sum())]:assert abs(v-row[k])<1e-10
   exact+=d.size;n+=1
 b=read(OUT/'budget74.json');rr=read(OUT/'budget74_rows.json');assert len(rr)==148
 for a,c in [('new_V_bytes','V_npy_bytes'),('mask_bytes','mask_packbits_npy_bytes'),('distance_bytes','all4_raw_unit_distance_npy_bytes')]:assert sum(r[a] for r in rr)==b['components'][c]
 assert b['peak_estimate_bytes']==sum(b['components'][k] for k in ['V_npy_bytes','mask_packbits_npy_bytes','all4_raw_unit_distance_npy_bytes'])+b['cal8_visual_npy_bytes']+b['metadata_curve_seals_margin_bytes']+b['temporary_video_allowance_bytes']
 result={'status':'PASS','cal_arms':n,'exact_partition_cells':exact,'independent_frontend_rebuilds_exact':frontends,'hashes':len(bindings),'budget_components_exact':True,'no_new_scores_or_GPU':True,'checker_sha256':sha(__file__)};(OUT/'independent_check.json').write_text(json.dumps(result,indent=2)+'\n');print(result)
