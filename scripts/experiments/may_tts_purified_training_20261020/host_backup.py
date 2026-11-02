"""Read-only observer: copy complete checkpoints and live evidence to F; never dispatch."""
from pathlib import Path,PurePosixPath
import datetime,hashlib,json,os,subprocess,time,traceback
R='/dev/shm/may-tts-purified-training-20261020-v1'
OUT=Path('F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1')
SSH=['ssh','-b','192.168.50.1','-o','BatchMode=yes','-o','ConnectTimeout=15','wjj@192.168.50.2']
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix('.partial');t.write_text(json.dumps(v,indent=2));t.replace(p)
def safe_relative(name):
 p=PurePosixPath(name)
 assert not p.is_absolute() and '..' not in p.parts and '\\' not in name and ':' not in name
 return p
def main():
 control=OUT/'controller';control.mkdir(exist_ok=True)
 if (control/'receipt.json').exists():raise RuntimeError('observer already launched; inspect before recovery')
 completed={};errors=0
 while True:
  try:
   code='''from pathlib import Path
import json,hashlib
r=Path(%r)
job=json.loads((r/'job.json').read_text())
files=[]
for p in sorted((r/'training').rglob('*')):
 if p.is_file() and p.suffix in ['.pt','.json'] and not p.name.endswith('.partial'):
  files.append({'path':p.relative_to(r).as_posix(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
for name in ['formal_complete.json','formal_failure.json','terminal_state.pt']:
 p=r/name
 if p.exists():files.append({'path':name,'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size})
print(json.dumps({'job':job,'files':files}))
'''%R
   ret=subprocess.run(SSH+['python3','-'],input=code,text=True,capture_output=True,timeout=120,check=True)
   snap=json.loads(ret.stdout)
   for row in snap['files']:
    name=row['path'];rel=safe_relative(name);immutable=(rel.suffix=='.pt' or rel.name.startswith('val_step') or rel.name in ['initial_control.json','training_receipt.json'])
    if immutable and name in completed:
     assert completed[name]['sha256']==row['sha256'],'immutable remote file changed';continue
    dest=OUT/'remote_artifacts'/Path(*rel.parts);dest.parent.mkdir(parents=True,exist_ok=True)
    temp=dest.with_suffix(dest.suffix+'.transfer')
    subprocess.run(['scp','-q','-o','BindAddress=192.168.50.1','wjj@192.168.50.2:'+R+'/'+name,str(temp)],check=True,timeout=180)
    # Mutable JSON may advance during transfer; retry the next snapshot without publishing a false failure.
    if sha(temp)!=row['sha256']:
     if immutable:raise RuntimeError('immutable transfer SHA mismatch '+name)
     continue
    temp.replace(dest)
    if immutable:completed[name]=row
   live=OUT/'live';live.mkdir(exist_ok=True)
   for name in ['job.json','status.json','formal_trace.jsonl','formal_stdout.log','training/gain_0/train_curve.jsonl','training/gain_0/ray_indices.sha256bin']:
    dst=live/name.replace('/','__');tmp=dst.with_suffix('.transfer')
    result=subprocess.run(['scp','-q','-o','BindAddress=192.168.50.1','wjj@192.168.50.2:'+R+'/'+name,str(tmp)],capture_output=True,timeout=120)
    if result.returncode==0:tmp.replace(dst)
   terminal=snap['job']['status'] in ['completed','failed']
   save(control/'receipt.json',dict(host_pid=os.getpid(),time=datetime.datetime.now(datetime.timezone.utc).isoformat(),status=snap['job']['status'],job=snap['job'],
    completed_member_count=len(completed),completed_members=completed,all_retained=True,remote_read_only=True,automatic_restart_allowed=False,transport_errors=errors))
   if terminal:
    assert (OUT/'live/formal_trace.jsonl').exists()
    save(control/'terminal_archive_receipt.json',dict(passed=True,job=snap['job'],members=snap['files'],trace_sha256=sha(OUT/'live/formal_trace.jsonl'),
     scientific_success=snap['job']['rc']==0,automatic_restart_allowed=False));return
  except Exception:
   errors+=1;save(control/'transport_attention.json',dict(time=datetime.datetime.now(datetime.timezone.utc).isoformat(),errors=errors,error=traceback.format_exc(),
    no_training_relaunch=True))
  time.sleep(30)
if __name__=='__main__':main()
