"""Collect the completed attempt05 proof and export small local viewing copies."""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

root=Path('F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017')
repo=Path(__file__).resolve().parents[3]
bundle=json.loads((root/'bundle_receipt.json').read_text())
assert json.loads((root/'delivery_manifest.json').read_text())['passed']
temp=root/'cloud_receipt_attempt05.download'
subprocess.run(['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
    'wjj@192.168.50.2:/dev/shm/synctalk-may-cloud-readback-attempt05/receipt.json',str(temp)],check=True)
receipt=json.loads(temp.read_text())
assert receipt['passed'] and receipt['verification_attempt']==5 and receipt['parallel_chunks_verified']==9
assert receipt['bytes']==bundle['bytes'] and receipt['sha256']==bundle['sha256']
assert not (root/'cloud_readback_receipt.json').exists()
temp.replace(root/'cloud_readback_receipt.json')
attempts={'payload_unchanged':True,'science_unchanged':True,'archive_sha256':bundle['sha256'],
    'attempts':[
        {'id':1,'state':'visibility_wait_timeout','note':'20 attempts, original archive later appeared'},
        {'id':2,'state':'all_parts_staged','note':'9 parts uploaded; early readback helper stopped when original file appeared'},
        {'id':3,'state':'stopped','note':'slow sequential single-file read replaced'},
        {'id':4,'state':'stopped','note':'slow same-file range reads replaced by separate cloud chunk reads'},
        {'id':5,'state':'passed','note':'all chunks individually SHA verified and full archive byte-order SHA matched'}]}
(root/'cloud_transport_attempts.json').write_text(json.dumps(attempts,indent=2)+'\n')
thin=repo/'runs/synctalk_may_training_pilot_20261017';thin.mkdir(exist_ok=True)
for name in ['delivery_manifest.json','bundle_receipt.json','cloud_readback_receipt.json','cloud_transport_attempts.json']:
    shutil.copyfile(root/name,thin/name)
shutil.copyfile(root/'remote/independent_results_qa.json',thin/'independent_results_qa.json')
shutil.copytree(root/'remote/analysis',thin/'analysis',dirs_exist_ok=False)
shutil.copyfile(root/'remote/diagnostics/late_val_60000_60000_60000.png',thin/'analysis/late_validation_curves.png')
shutil.copytree(root/'remote/preview',thin/'preview',dirs_exist_ok=False)
shutil.copyfile(root/'remote/ground_truth_reference_attempt01/reference.json',thin/'ground_truth_reference.json')
sources=root/'transport_sources';sources.mkdir(exist_ok=True)
paths=[Path(__file__),Path(__file__).with_name('finalize_delivery.py'),Path(__file__).with_name('verify_cloud_delivery.py'),
       Path(__file__).with_name('parallel_cloud_readback.py'),Path(__file__).with_name('parallel_cloud_chunk_readback.py'),
       repo/'.codex/scratch/cloud_chunks_may_pilot_attempt02.py',repo/'.codex/scratch/may_parallel_cloud_input_attempt04.json',
       repo/'.codex/scratch/may_parallel_cloud_input_attempt05.json']
remote_paths=['/dev/shm/parallel_cloud_chunk_readback.py','/dev/shm/may_parallel_cloud_input_attempt05.json']
proof=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','wjj@192.168.50.2',
                       'sha256sum '+' '.join(remote_paths)],capture_output=True,text=True,check=True)
remote_hashes={line.split()[1]:line.split()[0] for line in proof.stdout.splitlines()}
for remote_path,local_path in zip(remote_paths,[Path(__file__).with_name('parallel_cloud_chunk_readback.py'),repo/'.codex/scratch/may_parallel_cloud_input_attempt05.json']):
    assert remote_hashes[remote_path]==hashlib.sha256(local_path.read_bytes()).hexdigest()
(root/'transport_actual_execution_sources.json').write_text(json.dumps(remote_hashes,indent=2)+'\n')
records=[]
for path in paths:
    shutil.copyfile(path,sources/path.name)
    records.append({'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
(root/'transport_sources_manifest.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
