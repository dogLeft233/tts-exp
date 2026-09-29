"""Parent-requested resource-only extension: all74 eval arms, no effect access."""
from pathlib import Path
import hashlib,json,shutil,time
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/tts_fixed_mel_partition_feasibility_20260927';OLD=ROOT/'runs/tts_fixed_level_generation_cross_20260927'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2)+'\n')
if __name__=='__main__':
 support=read(OLD/'support.json');rows=[r for r in support if r['split']=='evaluation'];cal=sorted([r for r in support if r['split']=='calibration'],key=lambda r:r['id'])[0];assert len(rows)==74
 write(OUT/'budget74_amendment.json',{'time':time.time(),'scope':'root specifically requests all74eval twoarms plus firstcalN/T engineering metadata budget; no eval mel-distribution quantification or scores','code_sha256':sha(__file__),'support_sha256':sha(OLD/'support.json')});shutil.copyfile(__file__,OUT/Path(__file__).name)
 records=[];hashes={};tot=dict(V_npy_bytes=0,mask_packbits_npy_bytes=0,all4_raw_unit_distance_npy_bytes=0,new_mel_if_saved_npy_bytes=0,all_new_video_source_measured_bytes=0);maxtemp=0;maxframes=0
 for r in rows:
  for arm in ['N','T']:
   z=r['arms'][arm];front=np.load(z['FIXED']['frontend']);shape=front['mel'].shape;fp=OLD/'features'/r['id']/arm/'FIXED.npz';mp=fp.with_suffix('.json');meta=read(mp);assert sha(fp)==meta['sha256'];hashes[str(fp)]=meta['sha256'];hashes[str(mp)]=sha(mp)
   for name in ['raw','FIXED']:
    assert sha(z[name]['frontend'])==z[name]['frontend_sha256'];hashes[z[name]['frontend']]=z[name]['frontend_sha256']
   a=np.load(fp);vbytes=a['visual'].nbytes;frames=meta['video']['frames'];maxframes=max(maxframes,frames);v=2*(vbytes+128);mask=(shape[0]*shape[1]+7)//8+128;dbytes=4*2*(z['joint_L']*31*8+128);mbytes=2*(shape[0]*shape[1]*4+128)
   # Deleted FIXED file sizes are not inferred; creation metadata doesn't record byte size.
   rawmeta=read(OLD/'features'/r['id']/arm/'raw.json');rawpath=Path(rawmeta['video']['path']);assert rawpath.exists() and sha(rawpath)==rawmeta['video']['sha256'];actualsize=rawpath.stat().st_size;maxtemp=max(maxtemp,actualsize)
   for k,value in [('V_npy_bytes',v),('mask_packbits_npy_bytes',mask),('all4_raw_unit_distance_npy_bytes',dbytes),('new_mel_if_saved_npy_bytes',mbytes),('all_new_video_source_measured_bytes',2*actualsize)]:tot[k]+=value
   records.append({'id':r['id'],'arm':arm,'L':z['joint_L'],'frames':frames,'mel_shape':list(shape),'new_V_bytes':v,'mask_bytes':mask,'distance_bytes':dbytes,'raw_video_actual_bytes':actualsize})
 control=0
 for arm in ['N','T']:
  p=OLD/'features'/cal['id']/arm/'FIXED.npz';z=np.load(p);control+=4*(z['visual'].nbytes+128) # RAW/FIXED replay and U/S, each arm
 rawpixels=maxframes*224*224*3;temp_allowance=max(rawpixels+2**20,maxtemp*2);metadata=16*2**20
 peak=tot['V_npy_bytes']+tot['mask_packbits_npy_bytes']+tot['all4_raw_unit_distance_npy_bytes']+control+temp_allowance+metadata
 cap=256*2**20;free=shutil.disk_usage(OUT).free
 result={'status':'CPU_budget_only','eval_pairs':74,'eval_arms':148,'main_common_pairs':71,'new_eval_videos':296,'cal_id':cal['id'],'new_cal_videos':8,'all_new_video_retention':'zero permanent; firstcalRAW/FIXED endpoints already retained in parent; repeat and new U/S temporarily render/extract/commit hashes then remove ONLY newruntemporary','components':tot,'cal8_visual_npy_bytes':control,'metadata_curve_seals_margin_bytes':metadata,'max_frames':maxframes,'max_old_RAW_video_bytes':maxtemp,'temporary_video_allowance_bytes':temp_allowance,'peak_estimate_bytes':peak,'peak_estimate_GiB':peak/2**30,'proposed_hard_newrun_cap_bytes':cap,'cap_GiB':cap/2**30,'fits_proposed_cap':peak<=cap,'free_bytes':free,'free_GiB':free/2**30,'free_after_cap_GiB':(free-cap)/2**30,'reserve5GiB_after_cap':free-cap>=5*2**30,'input_storage':'do not duplicate existing waveforms/A or full mel endpoints; packbit U mask plus endpoint hashes and U/S mel/chunk hashes sufficient exact reconstruction; write visual.npy uncompressed so bytes known; .npy128-byte headers budgeted','caveat':'pixel raw size plus1MiB is a planning allowance, not universal FFV1 encoding upper bound; enforce absolute256MiB/5GiB gates per batch; no compression savings assumed for V/matrices; stop rather than drop inputs if limit hit','scope':'shape/source metadata only; no new render/features/scores and no eval distribution quantification'}
 write(OUT/'budget74_rows.json',records);write(OUT/'budget74_hashes.json',hashes);write(OUT/'budget74.json',result);print(json.dumps(result,indent=2))
