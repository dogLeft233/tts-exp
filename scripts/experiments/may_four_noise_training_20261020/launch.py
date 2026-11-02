"""Single dispatch; a lost SSH response must be inspected, never relaunched."""
from pathlib import Path
import subprocess,json
from protocol import OUT,REMOTE,SSH,sha,write
def main():
 assert not (OUT/'controller_dispatch.json').exists()
 assert json.loads((OUT/'deployment_receipt.json').read_text())['passed']
 write(OUT/'controller_dispatch.json',dict(dispatch_intent_recorded=True,automatic_retry=False,remote=REMOTE))
 code=f'''from pathlib import Path
import subprocess,json,os
r=Path({REMOTE!r});assert not (r/'controller_dispatch.json').exists();assert not (r/'controller_identity.json').exists()
with (r/'controller_dispatch.json').open('x') as f:json.dump(dict(automatic_retry=False),f);f.flush();os.fsync(f.fileno())
with (r/'controller_stdout.log').open('xb') as log:
 p=subprocess.Popen(['/usr/bin/python3','-u',str(r/'controller.py')],stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
print(json.dumps(dict(pid=p.pid,proc_start_ticks=Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21],dispatched_once=True)))'''
 r=subprocess.run(SSH+['python3','-'],input=code,text=True,capture_output=True,timeout=30)
 assert r.returncode==0,('inspect uncertain dispatch before any action',r.stderr)
 result=json.loads(r.stdout);write(OUT/'controller_dispatch_receipt.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
