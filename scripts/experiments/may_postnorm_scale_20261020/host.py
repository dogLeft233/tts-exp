"""Observe detached remote identity, not SSH lifetime; no automatic science replay."""
from pathlib import Path
import os,json,subprocess,time,tarfile,shutil,traceback
from protocol import OUT,REMOTE,sha,write
from deploy import SSH
LOCK='/dev/shm/synctalk-multispeaker-data-20261017/preparation_GPU.lock'
def remote_json(code):
    last=None
    for attempt in range(3):
        r=subprocess.run(SSH+['python3','-'],input=code,text=True,capture_output=True,timeout=45)
        if r.returncode==0:return json.loads(r.stdout)
        last=(r.returncode,r.stderr);time.sleep(2)
    raise RuntimeError(('read_only_transport_failed_no_job_relaunch',last))
def copy_tree(relative,dest):
    manifest=remote_json(f'''from pathlib import Path
import json,hashlib
r=Path({(REMOTE+'/'+relative)!r});print(json.dumps([dict(path=p.relative_to(r).as_posix(),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sorted(r.rglob('*')) if p.is_file()]))''')
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    proc=subprocess.Popen(SSH+[f'tar -cf - -C {REMOTE}/{relative} .'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout,mode='r|') as tar:
        for item in tar:
            if not item.isfile():continue
            target=(dest/item.name).resolve();assert target.is_relative_to(dest.resolve());target.parent.mkdir(parents=True,exist_ok=True)
            temp=target.with_suffix(target.suffix+'.copying')
            with tar.extractfile(item) as src,temp.open('wb') as dst:shutil.copyfileobj(src,dst,4*1024**2)
            temp.replace(target)
    assert not any(proc.stdout.read()) and proc.wait()==0,proc.stderr.read().decode(errors='replace')
    for row in manifest:
        q=dest/row['path'];assert q.stat().st_size==row['bytes'] and sha(q)==row['sha256']
    write(dest/'f_readback_manifest.json',dict(passed=True,members=manifest));return manifest
def launch(job,script,args):
    # A unique remote dispatch marker exists before Popen; an uncertain SSH outcome is never retried.
    code=f'''from pathlib import Path
import subprocess,json,sys,os,shutil
r=Path({REMOTE!r});d=r/'dispatch';d.mkdir(exist_ok=True)
assert not (r/'jobs'/{job!r}).exists()
assert shutil.disk_usage(r).free>300*1024**2,'SHM reserve before new job'
with (d/{(job+'.json')!r}).open('x') as f:
 f.write(json.dumps(dict(job={job!r},script={script!r},args={args!r})));f.flush();os.fsync(f.fileno())
with (d/{(job+'.log')!r}).open('xb') as f:
 p=subprocess.Popen(['flock','-n',{LOCK!r},sys.executable,str(r/'job_wrapper.py'),{job!r},str(r/{script!r}),*{args!r}],stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True)
print(json.dumps(dict(dispatched_once=True,flock_pid=p.pid)))
'''
    r=subprocess.run(SSH+['python3','-'],input=code,text=True,capture_output=True,timeout=30)
    assert r.returncode==0,('dispatch outcome, no automatic retry',r.stderr)
    write(OUT/'dispatch'/(job+'.json'),dict(json.loads(r.stdout),job=job,script=script,args=args,host_pid=os.getpid()))
def observe(job,mode=None,arm=None):
    identity=None;seen=set();started=time.monotonic();lost=0;pins=json.loads((OUT/'execution_source_pins.json').read_text())
    while True:
        data=remote_json(f'''from pathlib import Path
import json,shutil
r=Path({REMOTE!r});f=r/'jobs'/{job!r}/'execution.json'
if not f.exists():print('null');raise SystemExit
e=json.loads(f.read_text());q=Path('/proc')/str(e['pid']);live=q.exists() and (q/'stat').read_text().split()[21]==e['proc_start_ticks']
child=Path('/proc')/str(e.get('child_pid',0));cp=r/'training'/{str(arm)!r}/'latest_checkpoint.json'
print(json.dumps(dict(execution=e,live=live,child_live=child.exists(),checkpoint=json.loads(cp.read_text()) if cp.exists() else None,status=json.loads((r/'status.json').read_text()) if (r/'status.json').exists() else None,free_shm_bytes=shutil.disk_usage(r).free)))''')
        if data is None:
            assert time.monotonic()-started<45,'launch marker absent; preserve dispatch, never relaunch';time.sleep(1);continue
        e=data['execution']
        if e['status']=='running' and 'child_pid' not in e:
            assert time.monotonic()-started<45,'child launch marker absent';time.sleep(1);continue
        now=(e['pid'],e['proc_start_ticks'],e.get('child_pid'))
        if identity is None:identity=now
        else:assert tuple(identity)==now,'remote identity changed'
        for name in ['train.py','runtime.py','evaluate.py','job_wrapper.py']:assert e['source_sha256'][name]==pins['source_sha256'][name]
        assert e['source_sha256']['protocol.json']==pins['protocol_sha256'] and e['source_sha256']['run_env.sh']==pins['run_env_sha256']
        write(OUT/'active_controller.json',dict(status=mode or 'evaluation',job=job,arm=arm,host_pid=os.getpid(),snapshot=data,updated_unix=time.time()))
        cp=data['checkpoint'] if mode=='training' else None
        if cp and cp['step'] not in seen:
            dest=OUT/'training'/arm/Path(cp['path']).name;dest.parent.mkdir(parents=True,exist_ok=True)
            if not dest.exists():
                temp=dest.with_suffix('.partial')
                with temp.open('wb') as f:r=subprocess.run(SSH+['cat '+cp['path']],stdout=f,stderr=subprocess.PIPE)
                assert r.returncode==0 and sha(temp)==cp['sha256'];temp.replace(dest)
            assert sha(dest)==cp['sha256'];seen.add(cp['step']);write(dest.parent/'latest_f_checkpoint.json',dict(cp,f_path=str(dest),fresh_sha256_verified=True))
        if e['status']!='running':break
        lost=0 if data['live'] and data['child_live'] else lost+1;assert lost<3,'actual job identity lost; no future jobs'
        time.sleep(10)
    copy_tree('jobs/'+job,OUT/'jobs'/job)
    if mode and arm:copy_tree(mode+'/'+arm,OUT/mode/arm)
    assert e['status']=='completed' and e['rc']==0,('real_job_failure',e)
    return dict(job=job,rc=0,execution=e,identity=identity)
def run_job(job,script,args,mode=None,arm=None):
    launch(job,script,args);return observe(job,mode,arm)
