"""Durable launcher: explicit command, PID, elapsed time, child exit status."""
import json,os,pathlib,subprocess,sys,time,hashlib,signal
out=pathlib.Path(sys.argv[1]);name=sys.argv[2];command=sys.argv[3:]
start=time.time()
snapshot=out/(name+'_source_snapshot');snapshot.mkdir()
pins={}
for f in pathlib.Path(__file__).parent.glob('*.py'):
    raw=f.read_bytes();(snapshot/f.name).write_bytes(raw);pins[f.name]=hashlib.sha256(raw).hexdigest()
p=subprocess.Popen(command,stdout=(out/(name+'.stdout')).open('x'),stderr=(out/(name+'.stderr')).open('x'),start_new_session=True)
(out/(name+'.launch.json')).write_text(json.dumps(dict(pid=p.pid,launcher_pid=os.getpid(),command=command,unix=start,source_pins=pins)))
root=out
while root.name!='qwen_prediction_mechanism_20261023_attempt02' and root!=root.parent:root=root.parent
first=root.parent/'qwen_prediction_mechanism_20261023_attempt01/A_L06/started.json'
limit=max(1,28800-(time.time()-json.loads(first.read_text())['unix'])) if first.exists() else 28800
timed=False
try:rc=p.wait(timeout=limit)
except subprocess.TimeoutExpired:
    timed=True;os.killpg(p.pid,signal.SIGTERM)
    try:rc=p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);rc=p.wait()
(out/(name+'.exit.json')).write_text(json.dumps(dict(exit_code=rc,timed_out=timed,elapsed_s=time.time()-start,finished_unix=time.time())))
sys.exit(rc)
