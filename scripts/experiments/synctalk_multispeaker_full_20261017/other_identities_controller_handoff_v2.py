"""Controller-only first-epoch handoff: POSIX checkpoint paths on Windows, no GPU replay."""
import hashlib,importlib.util,json,os,shlex,subprocess,sys,traceback
from pathlib import Path,PurePosixPath
OUT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/fp16_other_identities_trial_v1')
HERE=Path('E:/Documents/tts-exp/scripts/experiments/synctalk_multispeaker_full_20261017')
sys.path.insert(0,str(HERE))
spec=importlib.util.spec_from_file_location('frozen_trial_v1',OUT/'sources/other_identities_trial_v1.py')
trial=importlib.util.module_from_spec(spec);spec.loader.exec_module(trial)

def checkpoint_locations(previous,current):
    old=PurePosixPath(previous['checkpoint_path']);new=PurePosixPath(current['checkpoint_path']);root=PurePosixPath(trial.RUN)
    assert old!=new and previous['epoch']<10
    old_relative=old.relative_to(root).as_posix();new_relative=new.relative_to(root).as_posix()
    return old,new,trial.OUT/old_relative,trial.OUT/new_relative,old.parent.parent.parent.relative_to(root).as_posix()

def retire_previous(previous,current):
    if previous is None:return
    old,new,f_old,f_new,relative=checkpoint_locations(previous,current)
    assert f_old.resolve().is_relative_to(trial.OUT.resolve()) and trial.sha(f_old)==previous['checkpoint_sha256'] and trial.sha(f_new)==current['checkpoint_sha256']
    assert json.loads((trial.OUT/relative/'verification.json').read_text())['passed']
    trial.atomic_json(trial.OUT/relative/'checkpoint_retirement_intent.json',dict(old_sha256=previous['checkpoint_sha256'],superseded_by=current['checkpoint_sha256'],controller_revision=2))
    code='''import hashlib,fcntl,os
from pathlib import Path
old=Path(%r);new=Path(%r);root=Path(%r)
assert old.resolve().is_relative_to(root/'training') and new.resolve().is_relative_to(root/'training')
assert hashlib.sha256(new.read_bytes()).hexdigest()==%r and hashlib.sha256(old.read_bytes()).hexdigest()==%r
fd=os.open(old,os.O_RDWR)
try:
 fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK);old.unlink()
finally:fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK);os.close(fd)
'''%(old.as_posix(),new.as_posix(),trial.RUN,current['checkpoint_sha256'],previous['checkpoint_sha256'])
    trial.remote_python(code);f_old.unlink()
    trial.atomic_json(trial.OUT/relative/'checkpoint_cache_retirement.json',dict(passed=True,old_sha256=previous['checkpoint_sha256'],superseded_by=current['checkpoint_sha256'],controller_revision=2))

def main():
    handoff=json.loads((OUT/'controller_revision02/handoff.json').read_text())
    assert handoff['reason']=='windows_path_flavor_before_first_epoch_persistence' and handoff['no_optimizer_interruption']
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==handoff['source_sha256']
    assert not (OUT/'receipt.json').exists()
    original_retire=trial.retire;original_restore=trial.restore
    def retire(identity):
        if identity=='SherrodBrown':
            local=json.loads((OUT/'storage/SherrodBrown_cache_retirement.json').read_text())
            remote=json.loads(trial.remote_python('from pathlib import Path;print(Path(%r).read_text())'%(trial.RUN+'/storage/SherrodBrown_cache_retirement.json')))
            assert local==remote and local['passed'];return
        return original_retire(identity)
    def restore(identity):
        if identity!='PatrickLeahy':return original_restore(identity)
        proof=json.loads((OUT/'storage/PatrickLeahy_full_restore.json').read_text());assert proof['passed'] and proof['verified_files']==16500
        code='''import sys,json
from pathlib import Path
sys.path.insert(0,%r)
from other_identities_trial_remote_v1 import selected_images,sha
root,proof,rows=selected_images('PatrickLeahy')
assert all((root/r['path']).stat().st_size==r['bytes'] and sha(root/r['path'])==r['sha256'] for r in rows)
print(json.dumps(dict(passed=True,verified_files=len(rows),manifest_sha256=proof['manifest_sha256'])))
'''%(trial.RUN+'/sources')
        verified=json.loads(trial.remote_python(code));assert verified['passed'] and verified['manifest_sha256']==proof['manifest_sha256']
        trial.atomic_json(OUT/'controller_revision02/Patrick_live_restore_reverification.json',verified)
    trial.retire,trial.restore,trial.retire_previous=retire,restore,retire_previous
    try:trial.run()
    except BaseException:
        trial.atomic_json(OUT/'controller_revision02/controller_failure.json',dict(pid=os.getpid(),traceback=traceback.format_exc(),state='stopped_no_automatic_replay'));raise
if __name__=='__main__':main()
