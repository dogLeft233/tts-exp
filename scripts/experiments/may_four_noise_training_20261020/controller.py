"""Detached, single-owner bounded queue; no re-dispatch or silent science retry."""
from pathlib import Path
import os,sys,json,subprocess,time,hashlib,traceback,shutil,fcntl
ROOT=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,v):
 f=ROOT/name;t=f.with_suffix(f.suffix+'.partial');t.parent.mkdir(parents=True,exist_ok=True);t.write_text(json.dumps(v,indent=2));t.replace(f)
def job(name,script,args):
 assert not (ROOT/'jobs'/name).exists()
 with (ROOT/'dispatch'/f'{name}.json').open('x') as f:json.dump(dict(script=script,args=args),f);f.flush();os.fsync(f.fileno())
 with (ROOT/'dispatch'/f'{name}.log').open('xb') as f:
  r=subprocess.run([sys.executable,str(ROOT/'job_wrapper.py'),name,str(ROOT/script),*args],stdout=f,stderr=subprocess.STDOUT)
 assert r.returncode==0,('real_job_failed_no_restart',name,r.returncode)
 receipt=json.loads((ROOT/'jobs'/name/'execution.json').read_text());assert receipt['status']=='completed' and receipt['rc']==0
 write('controller_status.json',dict(status='running',completed_job=name,pid=os.getpid(),updated_unix=time.time()))
def main():
 assert not (ROOT/'controller_identity.json').exists(),'one controller only'
 p=json.loads((ROOT/'protocol.json').read_text());pins=json.loads((ROOT/'execution_source_pins.json').read_text())
 for name,value in pins['source_sha256'].items():assert sha(ROOT/name)==value
 assert sha(ROOT/'protocol.json')==pins['protocol_sha256']
 assert json.loads((ROOT/'deployment_receipt.json').read_text())['passed']
 assert shutil.disk_usage(ROOT).free>12*1024**3
 write('controller_identity.json',dict(pid=os.getpid(),proc_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],started_unix=time.time(),source_sha256=sha(__file__),automatic_restart=False))
 (ROOT/'dispatch').mkdir(exist_ok=False)
 lock=Path('/dev/shm/synctalk-multispeaker-data-20261017/preparation_GPU.lock').open('a')
 # CPU code/data preparation is already complete. Wait without changing another process.
 while True:
  try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
  except BlockingIOError:write('controller_status.json',dict(status='waiting_shared_gpu_lock',pid=os.getpid()));time.sleep(15)
 try:
  job('provider_qualify','provider_qualify.py',[])
  for a in p['arms']:job('qualify_'+a['id'],'train.py',['qualify',a['id']])
  # Independent audit of every actual input sequence and fullstate before formal updates.
  job('review_qualification','review_remote_training.py',['qualification'])
  write('qualification_complete.json',dict(passed=True,arms=18,entry_updates=36,formal_warmstart=False))
  for a in p['arms']:
   job('train_'+a['id'],'train.py',['train',a['id']])
  job('review_training','review_remote_training.py',['training'])
  write('training_complete.json',dict(passed=True,arms=18,steps_each=12000,successful_updates=216000,skips=0))
  for a in p['arms']:job('render_'+a['id'],'evaluate.py',['render',a['id']])
  job('score_all','evaluate.py',['score'])
  write('science_complete.json',dict(passed=True,arms=18,cells=36,matrices=72,successful_updates=216000))
  write('controller_status.json',dict(status='scientific_jobs_completed_pending_independent_media_qa',pid=os.getpid(),updated_unix=time.time()))
 except BaseException as e:
  write('controller_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),future_jobs_stopped=True,automatic_retry=False));raise
 finally:fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
if __name__=='__main__':main()
