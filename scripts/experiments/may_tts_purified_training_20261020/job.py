"""One dispatch only. Persist exit status even when the trainer fails."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
R=Path(__file__).resolve().parent
def save(name,value):
 p=R/name;t=p.with_suffix('.partial');t.write_text(json.dumps(value,indent=2));t.replace(p)
def main():
 assert not (R/'job.json').exists()
 assert json.loads((R/'formal_admission.json').read_text())['passed']
 command=['bash',str(R/'run_env_v2.sh'),str(R/'run_training.py')]
 start=time.monotonic()
 v=dict(pid=os.getpid(),started=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,
  command_sha256=hashlib.sha256(json.dumps(command).encode()).hexdigest(),status='starting',automatic_restart_allowed=False)
 save('job.json',v)
 with (R/'formal_stdout.log').open('xb') as log:
  child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
  v.update(child_pid=child.pid,status='running');save('job.json',v)
  rc=child.wait()
 v.update(rc=rc,elapsed_seconds=time.monotonic()-start,status='completed' if rc==0 else 'failed',
  ended=datetime.datetime.now(datetime.timezone.utc).isoformat());save('job.json',v)
 return rc
if __name__=='__main__':raise SystemExit(main())
