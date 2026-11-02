"""One-shot locked evaluation with F readback after each render arm and scores."""
import sys,time,subprocess,json,os,traceback
from pathlib import Path
from protocol import OUT,REMOTE,write,sha
from controller import SSH,LOCK,copy_tree,remote_json
def main():
 assert json.loads((OUT/'training_complete.json').read_text())['passed']
 envpython='C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/env/Scripts/python.exe'
 subprocess.run([envpython,'-u',str(Path(__file__).parent/'review_training.py'),'training'],check=True)
 assert not (OUT/'postprocess_launcher.json').exists()
 admission=json.loads((OUT/'evaluation_admission.json').read_text());assert sha(Path(__file__).parent/'evaluate.py')==admission['source_sha256']['evaluate.py']
 actual=remote_json(f"from pathlib import Path\nimport hashlib,json\np=Path({(REMOTE+'/evaluate.py')!r});print(json.dumps(dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest())))")
 assert actual['sha256']==admission['source_sha256']['evaluate.py']
 write(OUT/'postprocess_launcher.json',dict(host_pid=os.getpid(),source_sha256=sha(__file__),admission_sha256=sha(OUT/'evaluation_admission.json'),started_unix=time.time()))
 p=json.loads((OUT/'protocol.json').read_text());done=[]
 try:
  for mode,arm in [('render',a['id']) for a in p['arms']]+[('score',None)]:
   job='evaluate_'+(arm or 'scores');command=f'flock -n {LOCK} python3 {REMOTE}/evaluation_wrapper_v1.py {job} {REMOTE}/evaluate.py {mode}'+(' '+arm if arm else '')
   write(OUT/'active_postprocess.json',dict(status=mode,arm=arm,job=job,host_pid=os.getpid()))
   rc=subprocess.run(SSH+[command],capture_output=True);copy_tree('jobs/'+job,OUT/'jobs'/job)
   copy_tree('renders/'+arm if arm else 'scores',OUT/'remote'/('renders/'+arm if arm else 'scores'))
   assert rc.returncode==0,(job,rc.returncode,rc.stderr.decode(errors='replace'));done.append(job)
  write(OUT/'postprocess_complete.json',dict(passed=True,jobs=done))
  write(OUT/'active_postprocess.json',dict(status='completed',jobs=done))
 except BaseException as e:
  write(OUT/'postprocess_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),done=done));raise
if __name__=='__main__':main()
