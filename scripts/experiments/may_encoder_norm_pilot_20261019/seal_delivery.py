"""Fresh local full-member readback; requires manual visual and memory closure."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil,subprocess
from protocol import OUT,sha,write
def main():
 assert json.loads((OUT/'analysis/qa.json').read_text())['passed']
 assert json.loads((OUT/'analysis/visual_review.json').read_text())['passed']
 assert json.loads((OUT/'media_clock_review.json').read_text())['passed']
 publication=json.loads((OUT/'final_memory_publication_receipt.json').read_text());assert publication['passed'] and publication['task_done']
 source=Path(__file__).parent;capsule=OUT/'execution_sources_final';capsule.mkdir(exist_ok=False)
 for path in source.glob('*.py'):shutil.copyfile(path,capsule/path.name)
 expected=json.loads((OUT/'evaluation_admission.json').read_text())['source_sha256']['evaluate.py']
 for name in ['hubert_base_on','hubert_base_off','wav2vec2_base_on','wav2vec2_base_off','scores']:
  execution=json.loads((OUT/'jobs'/('evaluate_'+name)/'execution.json').read_text());assert execution['status']=='completed' and execution['rc']==0 and execution['source_sha256']['evaluate.py']==expected
 ssh=['ssh','-b','192.168.50.1','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','wjj@192.168.50.2']
 script='''import subprocess,json
text=subprocess.check_output(['ps','-eo','pid,ppid,args'],text=True)
rows=[r for r in text.splitlines() if '/dev/shm/may-encoder-norm-pilot-20261019/' in r and ('/train.py' in r or '/evaluate.py' in r or 'evaluation_wrapper_v1.py' in r or 'job_wrapper.py' in r)]
print(json.dumps(dict(active_trial_processes=rows)))
'''
 state=json.loads(subprocess.run(ssh+['python3','-'],input=script,text=True,capture_output=True,check=True).stdout);assert not state['active_trial_processes']
 old=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_repeat_v1/user_pause_v1/paused.json')
 assert sha(old)=='545c965703b38a42058fc5c3a65a4e662722470a306a37ca7ea9f97e1a8ec090'
 write(OUT/'postrun_process_review.json',dict(passed=True,**state,old_patrick_pause_sha256=sha(old)))
 members=[]
 for path in sorted(OUT.rglob('*')):
  if not path.is_file() or 'matplotlib-cache' in path.parts or path.name in ['delivery_manifest.json','final_delivery_receipt.json']:continue
  members.append(dict(path=path.relative_to(OUT).as_posix(),bytes=path.stat().st_size,sha256=sha(path)))
 write(OUT/'delivery_manifest.json',dict(members=members,total_bytes=sum(r['bytes'] for r in members),local_F_only=True,cloud_backup_not_claimed=True))
 for row in members:
  path=OUT/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],path
 write(OUT/'final_delivery_receipt.json',dict(passed=True,completed_utc=datetime.now(timezone.utc).isoformat(),members=len(members),total_bytes=sum(r['bytes'] for r in members),manifest_sha256=sha(OUT/'delivery_manifest.json'),report_sha256=sha(OUT/'analysis/report.md'),visual_review_sha256=sha(OUT/'analysis/visual_review.json'),memory_publication_sha256=sha(OUT/'final_memory_publication_receipt.json'),training_updates=48000,training_models=4,test_cells=24,score_matrices=48,source_same_encoder_module_only=True,single_seed_pilot=True,perceptual_claim=False,old_patrick_still_paused=True,no_active_scientific_jobs=True))
 print(json.dumps(json.loads((OUT/'final_delivery_receipt.json').read_text())))
if __name__=='__main__':main()
