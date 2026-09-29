"""Final provenance/resource closure, after external result receipt PASS."""
import fcntl,json,shutil,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from scripts.experiments import tts_fixed_generator_shift_cross_20260927 as core
RUN=core.OUT;AUDIT=ROOT/'runs/tts_fixed_generator_shift_independent_audit_20260927'
def main():
 assert not (RUN/'final.json').exists();protocol,_=core.locked();receipts={}
 for name in ['static_receipt.json','calibration_live_receipt.json','result_receipt.json','dose_receipt.json']:
  path=AUDIT/name;assert core.read(path)['status']=='PASS';receipts[str(path)]=core.sha(path)
 core.write(RUN/'independent_validation.json',{'status':'PASS','created_epoch':time.time(),'receipts':receipts,'external_live_scope':'8 first-cal AVI only; eval296 creation-time producer FFmpeg metadata, independently checked saved arrays'})
 subprocess.run([sys.executable,str(ROOT/'scripts/experiments/tts_fixed_generator_shift_report_20260927.py')],check=True)
 compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True).strip();assert not compute
 with open('/tmp/tts-exp-gpu.lock','a') as lease:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(lease,fcntl.LOCK_UN)
 assert not [p for p in Path('/dev/shm/tts_fixed_generator_shift_cross_20260927').rglob('*') if p.is_file()]
 allocated=core.parent.allocated(RUN);audit=core.parent.allocated(AUDIT);reserve=2**20
 assert allocated+reserve<=core.CAP and audit<=core.AUDIT and allocated+audit+reserve<=160*2**20
 resource=core.limits(reserve);composition=ROOT/'runs/tts_fixed_label_composition_20260927'
 other=0 if (composition/'final.json').exists() else max(0,8*2**20-core.parent.allocated(composition))
 assert resource['free']-resource['remaining_commitment']-other>=core.FLOOR
 core.write(RUN/'resource_closure.json',{'passed':True,'created_epoch':time.time(),'main_allocated_before_final':allocated,'audit_allocated':audit,'final_metadata_reserve':reserve,'main_cap':core.CAP,'audit_cap':core.AUDIT,'total_cap':160*2**20,'floor':core.FLOOR,'resources':resource,'composition_remaining_reserved':other,'GPU_compute_empty':True,'lease_available':True,'temporary_files':0,'GPU_minutes_including_cal_live_reference':22.16,'GPU_cap_minutes':60,'no_remaining_scientific_work':True})
 hashes=dict(protocol['dependencies']);hashes.update(core.read(RUN/'parent_binding.json')['files']);hashes.update(receipts)
 for path in RUN.rglob('*'):
  if path.is_file():hashes[str(path)]=core.sha(path)
 for name in ['tts_fixed_generator_shift_report_20260927.py','tts_fixed_generator_shift_finalize_20260927.py','tts_generator_prototype_doses_20260927.py']:
  path=ROOT/'scripts/experiments'/name;hashes[str(path)]=core.sha(path)
 feasibility=ROOT/'runs/tts_generator_shift_holdout80_feasibility_20260927/feasibility.json'
 assert core.read(feasibility)['created_epoch']<core.read(RUN/'score_lock.json')['created_epoch']
 hashes[str(feasibility)]=core.sha(feasibility)
 for path,h in hashes.items():assert core.sha(path)==h,path
 core.write(RUN/'artifact_hashes.json',{'created_epoch':time.time(),'files':hashes,'checked':len(hashes)})
 summary=core.read(RUN/'summary.json.gz');triggers={kind:all(summary['raw/guard20/common71/'+metric+'/'+kind+'/source_S/N']['ci99'][0]>0 for metric in ['C','C_anchor']) for kind in core.KINDS}
 core.write(RUN/'final.json',{'status':'concluded','created_epoch':time.time(),'protocol_sha256':core.sha(RUN/'protocol.json'),'report_sha256':core.sha(RUN/'report.md'),'summary_sha256':core.sha(RUN/'summary.json.gz'),'independent_validation_sha256':core.sha(RUN/'independent_validation.json'),'resource_closure_sha256':core.sha(RUN/'resource_closure.json'),'artifact_hashes_sha256':core.sha(RUN/'artifact_hashes.json'),'primary_natural_phone_source_shift_gain':core.read(RUN/'analysis.json')['natural_phone_source_shift_gain'],'prospective_holdout80_trigger':triggers,'holdout80_still_requires_engineering_protocol':True,'new_eval_cells':296,'primary_clips':71,'primary_speakers':15,'legacy_queries':1411})
 print('CONCLUDED',core.sha(RUN/'final.json'))
if __name__=='__main__':main()
