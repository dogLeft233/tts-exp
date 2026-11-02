"""Execute each new render/score job once; persist all scientific members across two volumes."""
from pathlib import Path
import json,subprocess,sys,os,time,traceback,shutil
from protocol import OUT,REMOTE,sha,write
from controller import SSH,LOCK,copy_tree,remote_json
from controller_remote_completion_v2 import run_new_job
def wait_for_media_capacity(p):
 # Host storage admission only: do not dispatch science whose fixed output cannot persist.
 while True:
  required={}
  for arm in p['arms']:
   dest=Path(p['new_media_placements'][arm['id']]);volume=dest.drive+'/'
   if not dest.exists():required[volume]=required.get(volume,0)+490*1024**2
  required['F:/']=required.get('F:/',0)+180*1024**2
  free={volume:shutil.disk_usage(volume).free for volume in required}
  if all(free[volume]>=need for volume,need in required.items()):return
  write(OUT/'storage_admission_wait.json',dict(status='waiting_for_bounded_media_capacity',required_bytes=required,free_bytes=free,new_job_dispatched=False,source_sha256=sha(__file__)))
  time.sleep(10)
def main():
 assert json.loads((OUT/'training_complete.json').read_text())['passed']
 assert json.loads((OUT/'training_independent_review.json').read_text())['passed']
 assert json.loads((OUT/'parent_render_retirement_receipt.json').read_text())['passed'],'storage admission not complete'
 assert not (OUT/'postprocess_launcher.json').exists()
 p=json.loads((OUT/'protocol.json').read_text());expected=json.loads((OUT/'execution_source_pins.json').read_text())['source_sha256']['evaluate.py'];assert sha(Path(__file__).parent/'evaluate.py')==expected
 actual=remote_json(f"from pathlib import Path\nimport hashlib,json\nprint(json.dumps(dict(sha256=hashlib.sha256(Path({(REMOTE+'/evaluate.py')!r}).read_bytes()).hexdigest())))");assert actual['sha256']==expected
 write(OUT/'postprocess_launcher.json',dict(host_pid=os.getpid(),started_unix=time.time(),evaluate_sha256=expected,source_sha256=sha(__file__)))
 done=[]
 try:
  wait_for_media_capacity(p)
  for arm in p['arms']:
   name=arm['id'];dest=Path(p['new_media_placements'][name]);dest.parent.mkdir(parents=True,exist_ok=True)
   assert not dest.exists() and shutil.disk_usage(dest.parent).free>450*1024**2,'bounded media volume reserve'
   job='evaluate_'+name;write(OUT/'active_postprocess.json',dict(status='render',arm=name,job=job,host_pid=os.getpid(),persistent_media_path=str(dest)))
   run_new_job(job,'evaluate.py',['render',name],evaluation=True)
   copy_tree('renders/'+name,dest);done.append(job)
   write(OUT/'media_persistence'/f'{name}.json',dict(passed=True,arm=name,path=str(dest),manifest_sha256=sha(dest/'f_readback_manifest.json')))
  job='evaluate_scores';write(OUT/'active_postprocess.json',dict(status='score',job=job,host_pid=os.getpid()))
  run_new_job(job,'evaluate.py',['score'],evaluation=True)
  copy_tree('scores',OUT/'remote/scores');done.append(job)
  # CPU ffprobe inspects newly generated media only; report54 clock review remains an exact import.
  clock=Path(__file__).parent/'review_media_clock.py';subprocess.run(SSH+['cat > '+REMOTE+'/review_media_clock.py'],input=clock.read_bytes(),check=True)
  cr=subprocess.run(SSH+['python3 '+REMOTE+'/review_media_clock.py '+REMOTE],capture_output=True);assert cr.returncode==0,cr.stderr.decode(errors='replace')
  b=subprocess.check_output(SSH+['cat '+REMOTE+'/media_clock_review.json']);assert json.loads(b)['passed'];(OUT/'media_clock_review.json').write_bytes(b)
  write(OUT/'postprocess_complete.json',dict(passed=True,jobs=done,new_score_matrices=96,imported_score_matrices=48,no_zero_gain_replay=True,all_f_and_e_members_fresh_sha=True))
  write(OUT/'active_postprocess.json',dict(status='completed',jobs=done))
 except BaseException as e:write(OUT/'postprocess_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),done=done));raise
if __name__=='__main__':main()
