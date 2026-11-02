"""Recover independent full-archive readback after asynchronous cloud upload."""
import hashlib
import json
import shlex
import shutil
import subprocess
from pathlib import Path

root=Path('F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017')
bundle=json.loads((root/'bundle_receipt.json').read_text())
assert json.loads((root/'delivery_manifest.json').read_text())['passed']
source='/home/wjj/百度网盘/tts-exp-research-models/synctalk-may-training-pilot-20261017/'+bundle['file']
code=f'''import pathlib,json,hashlib,shutil
p=pathlib.Path({json.dumps(source,ensure_ascii=True)});h=hashlib.sha256();n=0;before=shutil.disk_usage('/').free
with p.open('rb') as f:
 for b in iter(lambda:f.read(4*1024**2),b''):
  if shutil.disk_usage('/').free<=128*1024**2:raise RuntimeError('root_free_space_guard')
  h.update(b);n+=len(b)
assert n=={bundle['bytes']} and h.hexdigest()=={bundle['sha256']!r}
print(json.dumps(dict(passed=True,path=str(p),bytes=n,sha256=h.hexdigest(),independent_linux_readback=True,root_free_before=before,root_free_after=shutil.disk_usage('/').free)))'''
proc=subprocess.run(['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','wjj@192.168.50.2',
                     'python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True)
receipt=json.loads(proc.stdout);receipt['verification_attempt']=3
receipt['transport_note']='Single-file attempt01 exceeded its visibility wait window, then appeared. All nine attempt02 parts were staged; their helper was stopped after original archive appeared. Full original archive independently verified; parts are not counted as verified cloud backups.'
with (root/'cloud_readback_receipt.json').open('x') as f:json.dump(receipt,f,indent=2)
(root/'cloud_attempt01_timeout.json').write_text(json.dumps({'file':bundle['file'],'passed':False,
        'reason':'not visible in independent mount within 20 readback attempts; later upload completion verified in attempt03',
        'archive_identity_unchanged':True},indent=2)+'\n')
(root/'cloud_attempt02_staged.json').write_text(json.dumps({'staged_parts':9,'passed':False,
        'reason':'readback helper intentionally stopped after original full archive became visible',
        'archive_identity_unchanged':True,'files_preserved':True},indent=2)+'\n')
thin=Path(__file__).resolve().parents[3]/'runs/synctalk_may_training_pilot_20261017';thin.mkdir(exist_ok=True)
for name in ['delivery_manifest.json','bundle_receipt.json','cloud_readback_receipt.json','cloud_attempt01_timeout.json','cloud_attempt02_staged.json']:
    shutil.copyfile(root/name,thin/name)
shutil.copyfile(root/'remote/independent_results_qa.json',thin/'independent_results_qa.json')
shutil.copytree(root/'remote/analysis',thin/'analysis',dirs_exist_ok=False)
shutil.copyfile(root/'remote/diagnostics/late_val_60000_60000_60000.png',thin/'analysis/late_validation_curves.png')
shutil.copytree(root/'remote/preview',thin/'preview',dirs_exist_ok=False)
shutil.copyfile(root/'remote/ground_truth_reference_attempt01/reference.json',thin/'ground_truth_reference.json')
sources=root/'transport_sources';sources.mkdir(exist_ok=True)
repo=Path(__file__).resolve().parents[3]
for path in [Path(__file__),Path(__file__).with_name('finalize_delivery.py'),repo/'.codex/scratch/cloud_chunks_may_pilot_attempt02.py']:
    shutil.copyfile(path,sources/path.name)
print(json.dumps(receipt),flush=True)
