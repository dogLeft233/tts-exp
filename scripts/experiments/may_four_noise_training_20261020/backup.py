"""Read-only remote observer; copies only closed, SHA-bound scientific members."""
from pathlib import Path
import subprocess,json,tarfile,shutil,time,os,traceback
from protocol import OUT,REMOTE,SSH,sha,write
MEDIA=Path('E:/Documents/tts-exp/runs/may_four_noise_training_20261020_v1/renders')
def snapshot():
 code=f'''from pathlib import Path
import json,hashlib
r=Path({REMOTE!r})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
files={{}};jobs=[]
def add(f):
 if f.is_file():files[f.relative_to(r).as_posix()]=dict(path=f.relative_to(r).as_posix(),bytes=f.stat().st_size,sha256=sha(f))
for d in (r/'jobs').glob('*'):
 f=d/'execution.json'
 if not f.exists():continue
 e=json.loads(f.read_text());jobs.append(dict(name=d.name,**e))
 if e['status']!='running':
  for f in d.rglob('*'):add(f)
for mode in ['qualification','training']:
 for d in (r/mode).glob('*'):
  cp=d/'latest_checkpoint.json'
  if cp.exists():
   q=json.loads(cp.read_text());f=Path(q['path'])
   if f.exists() and sha(f)==q['sha256']:
    for full in sorted(d.glob('full_step*.pt')):
     if int(full.stem.replace('full_step',''))<=q['step']:add(full)
  for name in ['receipt.json','failure.json']:
   marker=d/name
   if marker.exists():
    for f in d.rglob('*'):
     if f.is_file() and not f.name.endswith('.partial'):add(f)
for d in (r/'renders').glob('*'):
 e=r/'jobs'/('render_'+d.name)/'execution.json'
 if e.exists() and json.loads(e.read_text())['status']=='completed':
  for f in d.rglob('*'):add(f)
if (r/'science_complete.json').exists():
 for f in (r/'scores').rglob('*'):add(f)
for name in ['deployment_receipt.json','official_source_pins.json','controller_identity.json','provider_qualification.json','qualification_complete.json','qualification_independent_review.json','training_complete.json','training_independent_review.json','science_complete.json','controller_failure.json']:
 add(r/name)
status=json.loads((r/'controller_status.json').read_text()) if (r/'controller_status.json').exists() else None
training=json.loads((r/'status.json').read_text()) if (r/'status.json').exists() else None
failure=json.loads((r/'controller_failure.json').read_text()) if (r/'controller_failure.json').exists() else None
terminal=(r/'science_complete.json').exists() or failure is not None
print(json.dumps(dict(files=list(files.values()),jobs=jobs,status=status,training=training,failure=failure,terminal=terminal)))'''
 r=subprocess.run(SSH+['python3','-'],input=code,text=True,capture_output=True,timeout=60)
 assert r.returncode==0,r.stderr;return json.loads(r.stdout)
def destination(name):
 return MEDIA/Path(name).relative_to('renders') if name.startswith('renders/') else OUT/name
def copy_batch(rows):
 if not rows:return
 process=subprocess.Popen(SSH+['tar -cf - -C '+REMOTE+' --null -T -'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 process.stdin.write(b''.join(r['path'].encode()+b'\0' for r in rows));process.stdin.close()
 expected={r['path']:r for r in rows};seen=set()
 with tarfile.open(fileobj=process.stdout,mode='r|') as tar:
  for m in tar:
   assert m.isfile() and m.name in expected,m.name
   row=expected[m.name];target=destination(m.name);target.parent.mkdir(parents=True,exist_ok=True);tmp=target.with_suffix(target.suffix+'.copying')
   with tar.extractfile(m) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst,4<<20)
   assert tmp.stat().st_size==row['bytes'] and sha(tmp)==row['sha256'];tmp.replace(target);seen.add(m.name)
 assert process.wait()==0,process.stderr.read().decode(errors='replace');assert seen==set(expected)
def main():
 assert not (OUT/'backup_identity.json').exists();write(OUT/'backup_identity.json',dict(pid=os.getpid(),started_unix=time.time(),source_sha256=sha(__file__),read_only_remote=True,media_root=str(MEDIA)))
 known={};last=None
 try:
  while True:
   s=snapshot();missing=[]
   for row in s['files']:
    name=row['path']
    if name in known:assert known[name]==row,'immutable member changed';continue
    dest=destination(name)
    if dest.exists():assert dest.stat().st_size==row['bytes'] and sha(dest)==row['sha256']
    else:missing.append(row)
   copy_batch(missing)
   for row in s['files']:known[row['path']]=row
   write(OUT/'backup_manifest.json',dict(passed=True,members=list(known.values()),updated_unix=time.time(),media_root=str(MEDIA)))
   write(OUT/'active_controller.json',dict(host_pid=os.getpid(),remote=s,backed_up_members=len(known),media_root=str(MEDIA)))
   brief=(s['status'],s['training'],len(known))
   if brief!=last:print(json.dumps(dict(status=s['status'],training=s['training'],members=len(known)),ensure_ascii=False),flush=True);last=brief
   if s['terminal']:break
   time.sleep(15)
  write(OUT/'backup_terminal.json',dict(passed=not bool(s['failure']),snapshot=s,local_members=len(known),remote_new_jobs=False))
 except BaseException as e:
  write(OUT/'backup_failure.json',dict(error=repr(e),traceback=traceback.format_exc(),does_not_stop_or_relaunch_remote=True));raise
if __name__=='__main__':main()
