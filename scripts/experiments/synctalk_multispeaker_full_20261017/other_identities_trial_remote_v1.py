"""Independent FP16 trial lifecycle and archive-qualified supervision cache retirement."""
import argparse,fcntl,hashlib,json,os,resource,signal,sys
from pathlib import Path
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017')
VERSION='fp16_other_identities_trial_v1'
RUN=BASE/'trials'/VERSION
sys.path.insert(0,str(BASE))
from cohort_binding import identity_root,sha
from remote_epoch_job import start_job,status,atomic_json

def verify_sources():
    contract=json.loads((RUN/'contract.json').read_text())
    for name,digest in contract['sources'].items():
        assert sha(RUN/'sources'/name)==digest, name
    return contract

def selected_images(identity):
    root=identity_root(identity,BASE,require_admission=True)
    proof=json.loads((root/'snapshots'/identity/'F_archive_verification.json').read_text())
    manifest=root/'snapshots'/identity/'manifest.json'
    assert proof['passed'] and proof['all_archived_member_bytes_verified'] and sha(manifest)==proof['manifest_sha256']
    rows=[]
    for row in json.loads(manifest.read_text())['files']:
        p=Path(row['path']);parts=p.parts
        if len(parts)==5 and parts[:2]==('prepared',identity) and parts[2] in ['train','val','test'] and parts[3] in ['gt_imgs','torso_imgs']:
            count={'train':6000,'val':750,'test':1500}[parts[2]]
            row=dict(row,keep=int(p.stem) in {*range(4),*range(count//2,count//2+4),*range(count-4,count)})
            rows.append(row)
    assert len(rows)==16500
    return root,proof,rows

def retire(identity,authorization):
    """Only archive-pinned image files; retain the 72 qualification images and all evidence."""
    verify_sources()
    root,proof,rows=selected_images(identity)
    fresh=json.loads(Path(authorization).read_text())
    assert fresh['passed'] and fresh['identity']==identity and fresh['all_archived_member_bytes_verified']
    assert fresh['archive_sha256']==proof['archive_sha256'] and fresh['manifest_sha256']==proof['manifest_sha256']
    report=RUN/'storage'/f'{identity}_cache_retirement.json'
    if report.exists():
        value=json.loads(report.read_text());assert value['passed'];return value
    owned=(root/'prepared'/identity).resolve()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:args=(proc/'cmdline').read_bytes()
        except (FileNotFoundError,PermissionError):continue
        assert not (b'train_other_identity_epoch_v1.py' in args or b'train_epoch.py' in args or b'qualify_fp16_guard_real_model_v1.py' in args),'training/diagnostic reader still running'
    def broken(*args):raise RuntimeError('exclusive file lease interrupted')
    signal.signal(signal.SIGIO,broken)
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    if soft<len(rows)+100:resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,len(rows)+100),hard))
    held=[];retired=[];before=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
    report.parent.mkdir(parents=True,exist_ok=True)
    try:
        for row in rows:
            path=root/row['path']
            assert path.resolve().is_relative_to(owned) and not path.is_symlink()
            if not path.exists():
                assert not row['keep'];continue
            assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256']
            if row['keep']:continue
            fd=os.open(path,os.O_RDWR)
            try:fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK)
            except BaseException:os.close(fd);raise
            held.append((fd,path,row))
        atomic_json(report.with_name(report.stem+'_intent.json'),dict(identity=identity,archive=proof['archive'],archive_sha256=proof['archive_sha256'],manifest_sha256=proof['manifest_sha256'],files=[r for _,_,r in held],restore='stream exact derived archive with original receiver before further loads'))
        for fd,path,row in held:
            assert fcntl.fcntl(fd,fcntl.F_GETLEASE)==fcntl.F_WRLCK
            assert path.stat().st_ino==os.fstat(fd).st_ino
            path.unlink();retired.append(row)
    finally:
        for fd,_,_ in held:
            fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK);os.close(fd)
    after=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
    value=dict(passed=True,identity=identity,files_retired=len(retired),logical_bytes=sum(r['bytes'] for r in retired),free_before=before,free_after=after,retained_control_files=72,archive_sha256=proof['archive_sha256'],manifest_sha256=proof['manifest_sha256'],exclusive_leases=True,requires_explicit_restore=True)
    atomic_json(report,value);return value

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['launch','status','retire','inventory'])
    p.add_argument('--identity',required=True);p.add_argument('--gain',type=int);p.add_argument('--epoch',type=int)
    p.add_argument('--checkpoint');p.add_argument('--resume');p.add_argument('--reference-trace');p.add_argument('--authorization')
    a=p.parse_args()
    if a.action=='retire':value=retire(a.identity,a.authorization)
    elif a.action=='inventory':
        root,proof,rows=selected_images(a.identity)
        value=dict(identity=a.identity,bytes_required=sum(((r['bytes']+4095)//4096)*4096 for r in rows if not (root/r['path']).exists()),existing=sum((root/r['path']).exists() for r in rows),free=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize)
    else:
        contract=verify_sources();assert a.identity in contract['identities'] and a.gain in [-6,0,6] and a.epoch in range(1,11)
        job=RUN/'jobs'/a.identity/'seed42'/f'gain_{a.gain:+d}'/'coarse'/f'ep{a.epoch:04d}'/'attempt01'
        if a.action=='status':value=status(job)
        else:
            command=['flock','-n',str(BASE/'preparation_GPU.lock'),'bash',str(BASE/'runtime_exec.sh'),str(RUN/'sources/train_other_identity_epoch_v1.py'),'--identity',a.identity,'--seed','42','--gain',str(a.gain),'--phase','coarse','--epoch',str(a.epoch),'--attempt','1']
            for key in ['checkpoint','resume','reference_trace']:
                if getattr(a,key):command+=['--'+key.replace('_','-'),getattr(a,key)]
            value=start_job(job,command)
    print(json.dumps(value))
if __name__=='__main__':main()
