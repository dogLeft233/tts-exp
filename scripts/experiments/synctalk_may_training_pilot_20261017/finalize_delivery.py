"""Seal persistent readback after evaluation; optionally archive to Baidu mount."""
import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

HOST='wjj@192.168.50.2'
REMOTE='/dev/shm/synctalk-may-training-pilot-20261017'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def remote(code):
    p=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',HOST,
                      'python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True)
    return json.loads(p.stdout)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--cloud',action='store_true');a=p.parse_args()
    assert not (a.root/'delivery_manifest.json').exists()
    status=json.loads((a.root/'backup_status.json').read_text())
    assert status['worker_complete'] and not status['worker_failed']
    code=f'''import pathlib,json,hashlib
r=pathlib.Path({REMOTE!r})
assert json.loads((r/'experiment_complete.json').read_text())['passed']
owner=json.loads((r/'supervisor_receipt.json').read_text())
proc=pathlib.Path('/proc')/str(owner['pid'])/'stat'
assert not proc.exists() or proc.read_text().split()[21]!=owner['proc_start_ticks'], 'supervisor still running'
files=[]
for p in sorted(r.rglob('*')):
 if not p.is_file() or p.is_symlink():continue
 rel=p.relative_to(r)
 if any(x in ['__pycache__','matplotlib-cache'] for x in rel.parts) or p.suffix in ['.partial','.lock']:continue
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 files.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=h.hexdigest()))
print(json.dumps(files))'''
    rows=remote(code)
    for row in rows:
        path=a.root/'remote'/row['path']
        assert path.resolve().is_relative_to((a.root/'remote').resolve())
        assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],row['path']
    result={'passed':True,'all_remote_files_verified':True,'files':rows,'total_bytes':sum(r['bytes'] for r in rows),
            'persistent_root':str(a.root.resolve()),'working_root_is_volatile':True,
            'protocol_sha256':sha(a.root/'remote/protocol.json'),
            'evaluation_complete_sha256':sha(a.root/'remote/experiment_complete.json'),
            'prepared_data_delivery_sha256':'900cf9108a12fba087f94771dce1a411c2530e1ca97a34d9d34803d1a9c0b6b6'}
    (a.root/'delivery_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    archive=a.root/'may_training_pilot_verified.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=1) as tar:
        for row in rows:tar.add(a.root/'remote'/row['path'],arcname='remote/'+row['path'],recursive=False)
        tar.add(a.root/'delivery_manifest.json',arcname='delivery_manifest.json')
    bundle={'file':archive.name,'bytes':archive.stat().st_size,'sha256':sha(archive),'verified_persistent_files':len(rows)}
    (a.root/'bundle_receipt.json').write_text(json.dumps(bundle,indent=2)+'\n')
    if a.cloud:
        destination=Path('B:/tts-exp-research-models/synctalk-may-training-pilot-20261017')
        destination.mkdir(exist_ok=False)
        shutil.copyfile(archive,destination/archive.name)
        shutil.copyfile(a.root/'bundle_receipt.json',destination/'bundle_receipt.json')
        # Full independent Linux mount readback, with a root-space guard.
        source='/home/wjj/百度网盘/tts-exp-research-models/synctalk-may-training-pilot-20261017/'+archive.name
        code=f'''import pathlib,json,hashlib,shutil
p=pathlib.Path({source!r}); h=hashlib.sha256(); n=0; before=shutil.disk_usage('/').free
with p.open('rb') as f:
 for b in iter(lambda:f.read(4*1024**2),b''):
  if shutil.disk_usage('/').free<=128*1024**2:raise RuntimeError('root_free_space_guard')
  n+=len(b);h.update(b)
assert n=={bundle['bytes']} and h.hexdigest()=={bundle['sha256']!r}
print(json.dumps(dict(passed=True,path=str(p),bytes=n,sha256=h.hexdigest(),independent_linux_readback=True,root_free_before=before,root_free_after=shutil.disk_usage('/').free)))'''
        for attempt in range(20):
            try:
                cloud=remote(code)
                break
            except subprocess.CalledProcessError as error:
                if 'root_free_space_guard' in (error.stderr or '') or attempt==19:raise
                print(json.dumps({'cloud_readback_pending':True,'attempt':attempt+1}),flush=True)
                time.sleep(30)
        (a.root/'cloud_readback_receipt.json').write_text(json.dumps(cloud,indent=2)+'\n')
    thin=Path(__file__).resolve().parents[3]/'runs/synctalk_may_training_pilot_20261017'
    thin.mkdir(exist_ok=True)
    shutil.copyfile(a.root/'delivery_manifest.json',thin/'delivery_manifest.json')
    shutil.copyfile(a.root/'bundle_receipt.json',thin/'bundle_receipt.json')
    shutil.copyfile(a.root/'remote/independent_results_qa.json',thin/'independent_results_qa.json')
    shutil.copytree(a.root/'remote/analysis',thin/'analysis',dirs_exist_ok=False)
    if a.cloud:shutil.copyfile(a.root/'cloud_readback_receipt.json',thin/'cloud_readback_receipt.json')
    print(json.dumps({'passed':True,'files':len(rows),'total_bytes':result['total_bytes'],'bundle':bundle,'cloud_verified':a.cloud}),flush=True)


if __name__=='__main__':main()
