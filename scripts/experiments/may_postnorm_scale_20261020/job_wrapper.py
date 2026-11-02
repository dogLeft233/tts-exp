"""Exclusive one-shot remote job identity, durable execution marker and full log."""
import sys,subprocess,json,os,time,hashlib
from pathlib import Path
R=Path(__file__).resolve().parent;job=R/'jobs'/sys.argv[1];job.mkdir(parents=True,exist_ok=False)
args=sys.argv[2:]
pins={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),R/'train.py',R/'runtime.py',R/'evaluate.py',R/'protocol.json',R/'run_env.sh']}
receipt=dict(pid=os.getpid(),argv=args,start_unix=time.time(),proc_start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],status='running',source_sha256=pins)
def write():
    t=job/'execution.partial';t.write_text(json.dumps(receipt,indent=2));t.replace(job/'execution.json')
write();start=time.monotonic()
with (job/'stdout.log').open('xb') as f:
    child=subprocess.Popen(['bash',str(R/'run_env.sh'),*args],stdout=f,stderr=subprocess.STDOUT);receipt['child_pid']=child.pid;write();rc=child.wait()
receipt.update(status='completed' if rc==0 else 'failed',rc=rc,elapsed_seconds=time.monotonic()-start,stdout_sha256=hashlib.sha256((job/'stdout.log').read_bytes()).hexdigest());write();sys.exit(rc)
