"""Frozen sequential evaluation with a persistent terminal receipt."""
from pathlib import Path
import os,sys,time,json,subprocess,traceback,hashlib
ROOT=Path(__file__).resolve().parent
def save(name,v):
 p=ROOT/name;t=p.with_suffix('.partial');t.write_text(json.dumps(v,indent=2));t.replace(p)
def main():
 assert not (ROOT/'job.json').exists();start=time.monotonic();v=dict(pid=os.getpid(),phase='qualification',started=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),status='running',automatic_restart_allowed=False);save('job.json',v)
 try:
  c=json.loads((ROOT/'evaluation_contract.json').read_text())
  for path,digest in c['source_sha256'].items():assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest()==digest,path
  for phase,args in [('qualification',[sys.executable,'-u',str(ROOT/'eval_qualify.py')]),('render',[sys.executable,'-u',str(ROOT/'render_test.py'),'--train-gain','0']),('score',['/home/wjj/.venvs/syncnet/bin/python','-u',str(ROOT/'score_test.py')])]:
   v['phase']=phase;save('job.json',v)
   env=dict(os.environ);env['PYTHONPATH']=str(ROOT)+':'+env.get('PYTHONPATH','')
   with (ROOT/(phase+'_stdout.log')).open('xb') as log:
    child=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env);v['child_pid']=child.pid;save('job.json',v);rc=child.wait()
   assert rc==0,(phase,rc)
   if phase=='qualification':assert json.loads((ROOT/'qualification/receipt.json').read_text())['passed']
  score=json.loads((ROOT/'scores/scores.json').read_text());assert len(score['cells'])==3 and score['repeat']['matrix_max_error']==0
  save('evaluation_complete.json',dict(passed=True,checkpoint_sha256=c['checkpoint_sha256'],cells=3,clips_per_cell=4,contract_sha256=hashlib.sha256((ROOT/'evaluation_contract.json').read_bytes()).hexdigest(),persistent_delivery_pending=True))
  v.update(status='completed',rc=0)
 except Exception:
  save('evaluation_failure.json',dict(error=traceback.format_exc(),phase=v['phase'],automatic_restart_allowed=False));v.update(status='failed',rc=1)
 finally:
  v.update(elapsed_seconds=time.monotonic()-start,ended=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()));save('job.json',v)
 return v['rc']
if __name__=='__main__':raise SystemExit(main())
