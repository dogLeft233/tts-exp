"""Bounded first-identity preview delivery; never changes training state or releases its hold."""
import hashlib,json,os,shlex,shutil,subprocess,sys,tarfile,time,traceback
from pathlib import Path
HERE=Path('E:/Documents/tts-exp/scripts/experiments/synctalk_multispeaker_full_20261017')
sys.path.insert(0,str(HERE))
import other_identities_trial_v1 as trial
trial.OUT=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_fp16_fine_v2')
trial.RUN=trial.REMOTE+'/trials/patrick_fp16_fine_v2'
from execution_journal import atomic_json,exclusive_lock
from cloud_archive_bounded import run as cloud_backup
OUT=trial.OUT/'fine_score_27_v1'
REMOTE=trial.RUN+'/fine_score_27_v1'

def order():return [(g,0) for g in [0,-6,6]]+[(g,d) for g in [0,-6,6] for d in [-6,6]]

def job(action,gain,driver):
    command=['flock','-n',trial.REMOTE+'/preparation_GPU.lock','bash',trial.REMOTE+'/runtime_exec.sh',REMOTE+'/sources/patrick_fine_score_v2.py',action,'--gain',str(gain),'--driver',str(driver)]
    directory=REMOTE+f'/jobs/{action}_{gain:+d}_{driver:+d}/attempt01'
    code='import sys,json;from pathlib import Path;sys.path.insert(0,%r);from remote_epoch_job import start_job;print(json.dumps(start_job(Path(%r),%r)))'%(trial.REMOTE,directory,command)
    state=json.loads(trial.remote_python(code))
    while state['state'] in ['starting','running']:
        time.sleep(20)
        code='import sys,json;from pathlib import Path;sys.path.insert(0,%r);from remote_epoch_job import status;print(json.dumps(status(Path(%r))))'%(trial.REMOTE,directory)
        state=json.loads(trial.remote_python(code))
    assert state['state']=='completed' and state['passed'],state
    relative=f'fine_score_27_v1/jobs/{action}_{gain:+d}_{driver:+d}/attempt01'
    trial.persist(relative)
    return state

def extract_small(folder,include_movie=False):
    manifest={r['path']:r for r in json.loads((folder/'manifest.json').read_text(encoding='utf-8'))}
    with tarfile.open(folder/'artifacts.tar.gz','r|gz') as archive:
        for item in archive:
            assert item.isfile() and item.name in manifest
            if item.name=='original_pcm.mkv' and not include_movie:continue
            raw=archive.extractfile(item).read();assert hashlib.sha256(raw).hexdigest()==manifest[item.name]['sha256']
            target=folder/'local'/item.name
            assert target.resolve().is_relative_to((folder/'local').resolve())
            target.parent.mkdir(parents=True,exist_ok=True);assert not target.exists();target.write_bytes(raw)

def retire_remote_media(gain,driver,proof,cloud):
    folder=OUT/'renders'/f'train_{gain:+d}'/f'driver_{driver:+d}'
    assert proof['passed'] and proof['all_archived_member_bytes_verified'] and trial.sha(folder/'artifacts.tar.gz')==proof['archive_sha256']
    assert cloud['passed'] and cloud['independent_linux_cloud_readback'] and cloud['sha256']==proof['archive_sha256']
    rows=json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
    root=REMOTE+f'/renders/train_{gain:+d}/driver_{driver:+d}'
    selected=[r for r in rows if r['path']!='render_receipt.json']
    intent=dict(passed=True,archive_sha256=proof['archive_sha256'],manifest_sha256=proof['manifest_sha256'],cloud=cloud,files=selected,requires_explicit_restore=True)
    atomic_json(folder/'media_cache_retirement_intent.json',intent)
    code='''import fcntl,hashlib,json,os,signal
from pathlib import Path
root=Path(%r);rows=%r;held=[]
def broken(*args):raise RuntimeError('lease interrupted')
signal.signal(signal.SIGIO,broken)
try:
 for row in rows:
  p=root/row['path'];assert p.resolve().is_relative_to(root.resolve()) and not p.is_symlink()
  assert p.stat().st_size==row['bytes']
  h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  assert h.hexdigest()==row['sha256']
  fd=os.open(p,os.O_RDWR)
  try:fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_WRLCK)
  except BaseException:os.close(fd);raise
  held.append((fd,p))
 for fd,p in held:
  assert fcntl.fcntl(fd,fcntl.F_GETLEASE)==fcntl.F_WRLCK and p.stat().st_ino==os.fstat(fd).st_ino
  p.unlink()
finally:
 for fd,p in held:fcntl.fcntl(fd,fcntl.F_SETLEASE,fcntl.F_UNLCK);os.close(fd)
print(json.dumps(dict(passed=True,retired=len(held),exclusive_leases=True)))
'''%(root,selected)
    result=json.loads(trial.remote_python(code));assert result['passed'];atomic_json(folder/'remote_media_cache_retirement.json',result)
    archive=folder/'artifacts.tar.gz'
    assert archive.resolve().is_relative_to(OUT.resolve()) and trial.sha(archive)==cloud['sha256']
    for r in rows:
        if r['path']!='original_pcm.mkv':assert trial.sha(folder/'local'/r['path'])==r['sha256']
    archive.unlink()
    atomic_json(folder/'cloud_storage_locator.json',dict(passed=True,archive_sha256=cloud['sha256'],cloud=cloud,local_small_files_verified=True,lossless_movie_requires_explicit_cloud_restore=True))

def validate_aggregate(cells):
    assert len(cells)==27 and {(x['training_gain_db'],x['driver_gain_db'],x['evaluation_gain_db']) for x in cells}=={(a,b,c) for a in [-6,0,6] for b in [-6,0,6] for c in [-6,0,6]}
    for c in cells:
        assert [r['clip_id'] for r in c['clips']]==[f'test_{i:02d}' for i in range(6)]
        for support in ['guard20','full']:
            for metric in ['sync_c','sync_d','distance_d0','background_median']:
                assert abs(sum(r[support][metric] for r in c['clips'])/6-c['mean_'+support][metric])<1e-12
            assert c['mean_'+support]['offsets']==[r[support]['av_offset'] for r in c['clips']]

def run():
    with exclusive_lock(OUT/'preview_controller.lock'):
        assert not (OUT/'controller_receipt.json').exists(),'no blind restart; existing jobs/artifacts require verified handoff'
        pins=json.loads((OUT/'host_source_manifest.json').read_text(encoding='utf-8'))
        for name,digest in pins.items():assert trial.sha(HERE/name)==digest
        controller=json.loads((trial.OUT/'receipt.json').read_text(encoding='utf-8'));assert controller['status']=='fine_complete_pending_27_grid' and json.loads((trial.OUT/'fine_ready_for_27_grid.json').read_text(encoding='utf-8'))['passed']
        q=json.loads(subprocess.check_output(trial.SSH+['cat '+shlex.quote(REMOTE+'/qualification/qualification.json')]))
        assert q['passed'];state=dict(pid=os.getpid(),status='running',completed=[],passed=False,training_controller_held=True)
        atomic_json(OUT/'controller_receipt.json',state);cells=[]
        try:
            trial.persist('fine_score_27_v1/qualification');extract_small(OUT/'qualification',include_movie=True)
            for gain,driver in order():
                for name,digest in pins.items():assert trial.sha(HERE/name)==digest
                assert json.loads((trial.OUT/'receipt.json').read_text(encoding='utf-8'))['status']=='fine_complete_pending_27_grid'
                state.update(current=dict(train_gain=gain,driver_gain=driver),status='render');atomic_json(OUT/'controller_receipt.json',state)
                job('render',gain,driver)
                state['status']='score_and_independent_matrix_QA';atomic_json(OUT/'controller_receipt.json',state)
                job('score',gain,driver)
                sr=f'fine_score_27_v1/scores/train_{gain:+d}_driver_{driver:+d}'
                sp=trial.persist(sr);sf=trial.OUT/sr;extract_small(sf,include_movie=True)
                scores=json.loads((sf/'local/scores.json').read_text(encoding='utf-8'));assert scores['passed'] and scores['independent_matrix_qa'] and scores['checked_matrices']==18
                render_relative=f'fine_score_27_v1/renders/train_{gain:+d}/driver_{driver:+d}'
                size_code='from pathlib import Path;import json;p=Path(%r);print(json.dumps(sum(x.stat().st_size for x in p.rglob("*") if x.is_file())))'%(trial.RUN+'/'+render_relative)
                size=json.loads(trial.remote_python(size_code));assert shutil.disk_usage('F:/').free>size*1.1+100*1024**2,'F capacity before persistence'
                state['status']='persist_and_cloud_full_readback';atomic_json(OUT/'controller_receipt.json',state)
                proof=trial.persist(render_relative);folder=trial.OUT/render_relative;extract_small(folder)
                name=f'Patrick-fine17-descriptive-v1-train{gain:+d}-driver{driver:+d}'.replace('+','p').replace('-6','m6')
                cloud=cloud_backup(folder/'artifacts.tar.gz',proof['archive_sha256'],name,1)
                atomic_json(folder/'cloud_readback.json',cloud)
                retire_remote_media(gain,driver,proof,cloud)
                cells.extend(scores['cells']);state['completed'].append(dict(train_gain=gain,driver_gain=driver,score_archive_sha256=sp['archive_sha256'],render_archive_sha256=proof['archive_sha256']))
                atomic_json(OUT/'controller_receipt.json',state)
                if driver==0 and len(state['completed'])==3:
                    fixed=[c for c in cells if c['driver_gain_db']==c['evaluation_gain_db']==0]
                    assert len(fixed)==3
                    atomic_json(OUT/'fixed_driver0_scorer0_preview.json',dict(passed=True,phase='fine',epoch=17,seed=42,training_control_status='CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES',cells=fixed,descriptive_only=True,all_three_media_cloud_readback=True))
            validate_aggregate(cells)
            atomic_json(OUT/'all27_preview.json',dict(passed=True,cells=cells,phase='fine',epoch=17,seed=42,training_control_status='CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES',descriptive_only=True,independent_matrix_checks=162,independent_equal_clip_aggregation_checked=True))
            state.update(status='all27_ready_for_final_delivery_review',passed=True,training_hold_release_written=False);atomic_json(OUT/'controller_receipt.json',state)
        except BaseException as e:
            state.update(status='failed',passed=False,error=repr(e),traceback=traceback.format_exc());atomic_json(OUT/'controller_receipt.json',state);raise

if __name__=='__main__':run()
