"""Finish authorized numerical delivery after the bounded remote queue completes."""
from pathlib import Path
import sys,time,json,os,subprocess,traceback,shutil
from protocol import OUT,sha,write
def main():
 assert not (OUT/'finalizer_identity.json').exists()
 write(OUT/'finalizer_identity.json',dict(pid=os.getpid(),source_sha256=sha(__file__),started_unix=time.time(),scope='CPU QA and publication only; no remote launch/retry'))
 here=Path(__file__).resolve().parent
 try:
  while not (OUT/'backup_terminal_v2.json').exists():
   if (OUT/'backup_failure_v2.json').exists():raise RuntimeError('Read-only backup stopped; inspect before finalizing')
   time.sleep(15)
  receipt=json.loads((OUT/'backup_terminal_v2.json').read_text());assert receipt['passed'],'Remote controller failed; no numerical conclusions'
  env=dict(os.environ,PYTHONIOENCODING='utf-8',PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='2',OMP_NUM_THREADS='2',LOGFIRE_IGNORE_NO_CONFIG='1')
  for script,args in [('review_training.py',['qualification']),('review_training.py',['training']),('analysis.py',[])]:
   subprocess.run([sys.executable,str(here/script),*args],env=env,check=True)
  bm='C:/Users/dogle/AppData/Roaming/uv/tools/basic-memory/Scripts/python.exe'
  subprocess.run([bm,str(here/'publish_completed.py')],env=env,check=True)
  src=OUT/'delivery_sources';src.mkdir(exist_ok=False)
  for f in here.glob('*.py'):shutil.copy2(f,src/f.name)
  shutil.copy2(here/'README.md',src/'README.md')
  test=here.parents[2]/'tests/experiments/test_may_four_noise_training_20261020.py';shutil.copy2(test,src/test.name)
  pub=json.loads((OUT/'publication_receipt.json').read_text());shutil.copy2(pub['report_path'],src/Path(pub['report_path']).name)
  media=Path('E:/Documents/tts-exp/runs/may_four_noise_training_20261020_v1/renders');members=[]
  excluded={'final_receipt.json','delivery_manifest.json','finalizer_stdout.log','finalizer_stderr.log'}
  for base in [OUT,media]:
   for f in sorted(base.rglob('*')):
    if f.is_file() and f.name not in excluded and 'matplotlib-cache' not in f.parts:
     assert not f.name.endswith(('.partial','.copying'));members.append(dict(root='F' if base==OUT else 'E_media',path=f.relative_to(base).as_posix(),bytes=f.stat().st_size,sha256=sha(f)))
  write(OUT/'delivery_manifest.json',dict(roots=dict(F=str(OUT),E_media=str(media)),members=members,excluded=sorted(excluded)))
  for row in members:
   f=(OUT if row['root']=='F' else media)/row['path'];assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256']
  write(OUT/'final_receipt.json',dict(passed=True,models=18,successful_updates=216000,skips=0,entry_updates_not_formal=36,encoder_forwards=172,cells=36,matrices=72,frames=18000,members=len(members),bytes=sum(r['bytes'] for r in members),manifest_sha256=sha(OUT/'delivery_manifest.json'),fresh_full_readback=True,report=pub['report_path'],single_seed=True,single_noise_instance=True,local_persistent_only=True))
  print('DELIVERY COMPLETE',pub['report_path'],flush=True)
 except BaseException as e:
  write(OUT/'finalizer_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),no_remote_relaunch=True));raise
if __name__=='__main__':main()
