"""One shot remote command receipt. Never resume an existing command identity."""
import sys,subprocess,json,os,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent
job=R/'jobs'/sys.argv[1];job.mkdir(parents=True,exist_ok=False)
args=sys.argv[2:];started=time.time()
receipt=dict(pid=os.getpid(),argv=args,start_unix=started,proc_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],status='running',source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'train.py',R/'runtime.py',R/'protocol.json',R/'run_env.sh',R/'evaluate.py']})
(job/'execution.json').write_text(json.dumps(receipt,indent=2))
with (job/'stdout.log').open('xb') as output:
 child=subprocess.Popen(['bash',str(R/'run_env.sh'),*args],stdout=output,stderr=subprocess.STDOUT)
 receipt['child_pid']=child.pid;(job/'execution.json').write_text(json.dumps(receipt,indent=2));rc=child.wait()
receipt.update(status='completed' if rc==0 else 'failed',rc=rc,elapsed_seconds=time.time()-started,stdout_sha256=hashlib.sha256((job/'stdout.log').read_bytes()).hexdigest())
(job/'execution.json').write_text(json.dumps(receipt,indent=2));sys.exit(rc)
