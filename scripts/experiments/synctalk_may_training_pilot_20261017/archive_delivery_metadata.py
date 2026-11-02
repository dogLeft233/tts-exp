"""Archive the final report and small reproducibility receipts to the same cloud."""
import hashlib
import json
import shlex
import shutil
import subprocess
import time
import zipfile
from pathlib import Path

repo=Path(__file__).resolve().parents[3]
root=Path('F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017')
cloud=json.loads((root/'cloud_readback_receipt.json').read_text());assert cloud['passed']
report=repo/'basic-memory/docs/experiments/50-synctalk-may-gain-training-pilot.md'
task=repo/'basic-memory/Tasks/训练 May 电平交叉先导.md'
assert 'status: concluded' in report.read_text(encoding='utf-8')
assert 'status: done' in task.read_text(encoding='utf-8') and '[ ]' not in task.read_text(encoding='utf-8')
items=[(report,report.relative_to(repo).as_posix()),(task,task.relative_to(repo).as_posix()),
       (repo/'tests/experiments/test_synctalk_may_training_pilot_20261017.py','tests/experiments/test_synctalk_may_training_pilot_20261017.py')]
for p in sorted(Path(__file__).parent.iterdir()):
    if p.is_file() and p.suffix in ['.py','.json','.sh','.md']:items.append((p,p.relative_to(repo).as_posix()))
for name in ['bundle_receipt.json','delivery_manifest.json','cloud_readback_receipt.json','cloud_transport_attempts.json','transport_sources_manifest.json','transport_actual_execution_sources.json']:
    items.append((root/name,'delivery/'+name))
for p in sorted((root/'transport_sources').iterdir()):
    if p.is_file():items.append((p,'delivery/transport_sources/'+p.name))
thin=repo/'runs/synctalk_may_training_pilot_20261017'
for p in sorted((thin/'analysis').iterdir()):
    if p.is_file():items.append((p,p.relative_to(repo).as_posix()))
for name in ['independent_results_qa.json','ground_truth_reference.json']:
    items.append((thin/name,(thin/name).relative_to(repo).as_posix()))
records=[{'path':name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p,name in items]
archive=root/'may_training_pilot_delivery_metadata_v1.zip';assert not archive.exists()
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
    for p,name in items:z.write(p,name)
    z.writestr('metadata_manifest.json',json.dumps(records,ensure_ascii=False,indent=2))
digest=hashlib.sha256(archive.read_bytes()).hexdigest();size=archive.stat().st_size
destination=Path('B:/tts-exp-research-models/synctalk-may-training-pilot-20261017')/archive.name
assert not destination.exists();shutil.copyfile(archive,destination)
source='/home/wjj/百度网盘/tts-exp-research-models/synctalk-may-training-pilot-20261017/'+archive.name
code=f'''import pathlib,hashlib,json,zipfile,io,shutil
p=pathlib.Path({json.dumps(source,ensure_ascii=True)});before=shutil.disk_usage('/').free
payload=p.read_bytes();assert len(payload)=={size} and hashlib.sha256(payload).hexdigest()=={digest!r}
with zipfile.ZipFile(io.BytesIO(payload)) as z:
 records=json.loads(z.read('metadata_manifest.json'))
 for row in records:
  x=z.read(row['path']);assert len(x)==row['bytes'] and hashlib.sha256(x).hexdigest()==row['sha256']
print(json.dumps(dict(passed=True,path=str(p),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest(),members_verified=len(records),independent_linux_readback=True,root_free_before=before,root_free_after=shutil.disk_usage('/').free)))'''
for attempt in range(30):
    result=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','wjj@192.168.50.2',
                           'python3 -c '+shlex.quote(code)],capture_output=True,text=True)
    if result.returncode==0:receipt=json.loads(result.stdout);break
    if attempt==29:raise RuntimeError(result.stderr)
    time.sleep(5)
(root/'metadata_cloud_readback_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
shutil.copyfile(root/'metadata_cloud_readback_receipt.json',thin/'metadata_cloud_readback_receipt.json')
print(json.dumps(receipt),flush=True)
