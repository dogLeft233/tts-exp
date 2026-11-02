"""Archive-qualified retirement of completed identity train/val pixels; retain every test pixel."""
import fcntl,json,os,resource,signal,sys
from pathlib import Path
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017')
RUN=BASE/'trials/fp16_other_identities_trial_v1'
OUT=RUN/'early_score_preview_v2'
sys.path[:0]=[str(BASE),str(RUN/'sources')]
from other_identities_trial_remote_v1 import selected_images
from flow_stream import sha
from remote_epoch_job import atomic_json

def run():
    identity='PatrickLeahy';root,proof,rows=selected_images(identity)
    authorization=json.loads((OUT/'fresh_archive_verification.json').read_text())
    assert authorization['passed'] and authorization['all_archived_member_bytes_verified']
    assert authorization['identity']==identity and authorization['archive_sha256']==proof['archive_sha256'] and authorization['manifest_sha256']==proof['manifest_sha256']
    ready=json.loads((OUT/'controller_preview_ready.json').read_text());assert ready['identity']==identity
    report=OUT/'trainval_cache_retirement.json';assert not report.exists()
    for gain in [-6,0,6]:
        ep=RUN/'training'/identity/'seed42'/f'gain_{gain:+d}'/'coarse/ep0010/attempt01'
        value=json.loads((ep/'epoch_receipt.json').read_text());assert value['passed'] and value['global_step']==60000
    owned=(root/'prepared'/identity).resolve()
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:args=(proc/'cmdline').read_bytes().split(b'\0')
        except (FileNotFoundError,PermissionError):continue
        assert not any(Path(os.fsdecode(a)).name in ['train_other_identity_epoch_v1.py','train_epoch.py','coarse_preview_v1.py','coarse_preview_v2.py'] for a in args),'active data reader'
    def broken(*args):raise RuntimeError('file lease interrupted')
    signal.signal(signal.SIGIO,broken)
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE);resource.setrlimit(resource.RLIMIT_NOFILE,(min(hard,max(soft,len(rows)+100)),hard))
    held=[];before=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize
    try:
        for row in rows:
            p=root/row['path'];assert p.resolve().is_relative_to(owned) and not p.is_symlink()
            assert p.exists() and p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
            if Path(row['path']).parts[2]=='test' or row['keep']:continue
            fd=os.open(p,os.O_RDWR)
            try:fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK)
            except BaseException:os.close(fd);raise
            held.append((fd,p,row))
        assert len(held)==13452
        atomic_json(OUT/'trainval_cache_retirement_intent.json',dict(identity=identity,archive_sha256=proof['archive_sha256'],manifest_sha256=proof['manifest_sha256'],files=[x[2] for x in held],restore='original verified full archive before any train/val loading'))
        for fd,p,row in held:
            assert fcntl.fcntl(fd,fcntl.F_GETLEASE)==fcntl.F_WRLCK and p.stat().st_ino==os.fstat(fd).st_ino
            p.unlink()
    finally:
        for fd,_,_ in held:fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK);os.close(fd)
    result=dict(passed=True,identity=identity,retired_trainval_files=len(held),all_3000_test_files_retained=True,retained_trainval_controls=48,
        all16500_fresh_sha_verified_before_retirement=True,exclusive_leases=True,archive_sha256=proof['archive_sha256'],requires_restore_before_training=True,
        free_before=before,free_after=os.statvfs(root).f_bavail*os.statvfs(root).f_frsize)
    atomic_json(report,result);print(json.dumps(result))

if __name__=='__main__':
    with (BASE/'preparation_GPU.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);run()
