"""Host one-shot controller, verified F persistence after each executed stage."""
import sys,os,json,time,subprocess,tarfile,hashlib,traceback
from pathlib import Path
from protocol import OUT,REMOTE,sha,write
SSH=['ssh','-b','192.168.50.1','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','wjj@192.168.50.2']
LOCK='/dev/shm/synctalk-multispeaker-data-20261017/preparation_GPU.lock'
def remote_json(code):
 r=subprocess.run(SSH+['python3','-'],input=code,text=True,capture_output=True,check=True);return json.loads(r.stdout)
def copy_tree(relative,dest):
 """Read manifest then copy tar, independently check each persisted member."""
 script=f'''from pathlib import Path
import hashlib,json
r=Path({REMOTE!r});root=r/{relative!r};out=[]
for p in sorted(root.rglob('*')):
 if p.is_file():out.append(dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
print(json.dumps(out))
'''
 manifest=remote_json(script);dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 proc=subprocess.Popen(SSH+[f'tar -cf - -C {REMOTE}/{relative} .'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
  for item in tar:
   if not item.isfile():continue
   target=(dest/item.name).resolve();assert target.is_relative_to(dest.resolve())
   target.parent.mkdir(parents=True,exist_ok=True)
   with tar.extractfile(item) as src,target.open('wb') as dst:
    while True:
     b=src.read(4*1024**2)
     if not b:break
     dst.write(b)
 tail=proc.stdout.read();assert not any(tail),'nonzero unexpected tar trailer'
 assert proc.wait()==0,proc.stderr.read().decode()
 for row in manifest:
  q=dest/row['path'];assert q.stat().st_size==row['bytes'] and sha(q)==row['sha256'],q
 write(dest/'f_readback_manifest.json',dict(passed=True,members=manifest));return manifest
def launch(job,trainargs,mode):
 wrapper=f'{REMOTE}/job_wrapper.py'
 cmd=SSH+[f'flock -n {LOCK} python3 {wrapper} {job} {REMOTE}/train.py '+ ' '.join(trainargs)]
 started=time.time();proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 write(OUT/'active_controller.json',dict(status=mode,host_pid=os.getpid(),ssh_pid=proc.pid,job=job,trainargs=trainargs,started_unix=started))
 current=trainargs[-1] if len(trainargs)>1 else None;seen=set()
 while proc.poll() is None:
  time.sleep(10)
  if mode=='training' and current:
   snap=remote_json(f"from pathlib import Path\nimport json\np=Path({(REMOTE+'/training/'+current+'/latest_checkpoint.json')!r});print(p.read_text() if p.exists() else 'null')")
   if snap and snap['step'] not in seen:
    # A completed checkpoint marker is written after its paired validation finishes.
    cp=Path(snap['path']).name;target=OUT/'training'/current/cp;target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('wb') as f:
     ret=subprocess.run(SSH+[f'cat {snap["path"]}'],stdout=f,stderr=subprocess.PIPE);assert ret.returncode==0,ret.stderr
    assert sha(target)==snap['sha256'];seen.add(snap['step'])
    write(OUT/'training'/current/'latest_f_checkpoint.json',dict(**snap,f_path=str(target),fresh_sha256_verified=True))
 stdout,stderr=proc.communicate();copy_tree('jobs/'+job,OUT/'jobs'/job)
 if current:copy_tree(('qualification' if mode=='qualification' else 'training')+'/'+current,OUT/('qualification' if mode=='qualification' else 'training')/current)
 assert proc.returncode==0,(job,proc.returncode,stderr.decode(errors='replace'))
 return dict(job=job,rc=proc.returncode,elapsed_seconds=time.time()-started)
def main():
 mode=sys.argv[1];attempt=sys.argv[2] if len(sys.argv)>2 else '';assert mode in ['qualification','training'];flag=OUT/(mode+'_launcher'+attempt+'.json');assert not flag.exists(),'never repeat an executed controller'
 write(flag,dict(host_pid=os.getpid(),start_unix=time.time(),source_sha256=sha(__file__),mode=mode))
 p=json.loads((OUT/'protocol.json').read_text());done=[]
 try:
  if mode=='qualification':
   assert sha(OUT/'initialization.pt')==p['parent_initial_sha256']
   for a in p['arms']:done.append(launch('qualify_'+a['id']+attempt,['qualify',a['id']],mode))
  else:
   assert json.loads((OUT/'qualification_complete.json').read_text())['passed']
   for a in p['arms']:done.append(launch('train_'+a['id'],['train',a['id']],mode))
  write(OUT/(mode+'_complete.json'),dict(passed=True,mode=mode,jobs=done,completed_unix=time.time()))
  write(OUT/'active_controller.json',dict(status=mode+'_complete',host_pid=os.getpid(),jobs=done))
 except BaseException as e:
  write(OUT/(mode+'_failure'+attempt+'.json'),dict(error=repr(e),traceback=traceback.format_exc(),completed_jobs=done,host_pid=os.getpid()))
  write(OUT/'active_controller.json',dict(status='stopped_after_failure',mode=mode,host_pid=os.getpid()));raise
if __name__=='__main__':main()
