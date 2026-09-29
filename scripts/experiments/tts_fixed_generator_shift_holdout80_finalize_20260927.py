"""Close audited natural80 outputs; no scoring, transformations or model calls."""
from pathlib import Path
import math,time
from scripts.experiments import tts_fixed_generator_shift_holdout80_20260927 as c
O=c.OUT
p,rows=c.locked();a=c.read(O/'analysis.json');b=c.read(O/'independent_binding.json');r=c.read(b['receipt']);assert r['status']=='PASS' and c.sha(b['receipt'])==b['receipt_sha256'];assert a['global_direction_transfer_supported'] and all(a['four_primary_tests'].values())
exit=c.read(O/'gpu_runtime/supervised_exit_evaluation.json');assert exit['returncode']==0 and exit['compute_empty'] and exit['lease_available'];assert c.allocated(c.TMP)==c.allocated(c.MFATMP)==0
resources=c.limits();c.write(O/'resource_closure.json',{'status':'CLOSED','created_epoch':time.time(),'main_remaining_commitment_after_final':0,'GPU_cumulative_wall_seconds':exit['cumulative_wall_seconds'],'GPU_released':True,'temporary_bytes':0,'resource_snapshot_before_manifest':resources,'boundaries':'all old floor4GiB/cap192MiB preserved through final; future stages have separately authorized budgets; supervisor maxrss is not worker peak'})
files=dict(p['dependencies'])
for q in O.rglob('*'):
 if q.is_file() and q.name not in ['manifest.json','final.json']:files[str(q.resolve())]=c.sha(q)
for q in [Path(__file__).resolve(),c.ROOT/'scripts/experiments/tts_fixed_generator_shift_holdout80_report_20260927.py',Path(b['receipt'])]:files[str(q)]=c.sha(q)
c.write(O/'manifest.json',{'status':'HASH_SEALED','created_epoch':time.time(),'files':files,'self_exclusions':['manifest.json','final.json'],'scope':'all main output files plus frozen original dependencies/report/finalizer/independent result receipt; deleted temporary AVI only creation metadata'})
res=c.limits();c.write(O/'final.json',{'status':'concluded','created_epoch':time.time(),'protocol_sha256':c.sha(O/'protocol.json'),'report_sha256':c.sha(O/'report.md'),'summary_sha256':c.sha(O/'summary.json.gz'),'analysis_sha256':c.sha(O/'analysis.json'),'manifest_sha256':c.sha(O/'manifest.json'),'resource_closure_sha256':c.sha(O/'resource_closure.json'),'independent_receipt':b,'four_primary_tests':a['four_primary_tests'],'global_direction_transfer_supported':True,'calibration_and_all80_retained':True,'final_resource_before_final_file':res,'main_unused_commitment_released':True,'no_future_GPU_implicitly_started':True})
assert c.limits()['allocated']<=c.CAP
print('FINAL',c.sha(O/'final.json'),'REPORT',c.sha(O/'report.md'),'ALLOCATED',c.allocated(O))
