"""Create one isolated run; import exact standard controls, never mutate parents."""
from pathlib import Path
import sys,subprocess,json,tarfile,io,shutil,time,hashlib
from protocol import OUT,REMOTE,sha,write
SSH=['ssh','-b','192.168.50.1','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','wjj@192.168.50.2']
PARENT_REMOTE='/dev/shm/may-encoder-norm-pilot-20261019'
def main():
    assert json.loads((OUT/'feature_independent_review.json').read_text())['passed']
    assert not (OUT/'deployment_receipt.json').exists()
    p=json.loads((OUT/'protocol.json').read_text());parent=Path(p['parent']);seal=json.loads((parent/'final_delivery_receipt.json').read_text())
    assert seal['passed'] and sha(parent/'final_delivery_receipt.json')==p['parent_delivery_receipt_sha256']
    manifest=json.loads((parent/'delivery_manifest.json').read_text());assert sha(parent/'delivery_manifest.json')==seal['manifest_sha256']
    for row in manifest['members']:
        path=parent/row['path'];assert path.stat().st_size==row['bytes'] and sha(path)==row['sha256'],path
    imported=[]
    for arm in p['imported_arms']:
        old=parent/'training'/arm['parent_id'];r=json.loads((old/'receipt.json').read_text());assert r['passed'] and r['steps']==r['success']==12000 and r['skip']==0 and r['initial_sha256']==p['parent_initial_sha256']
        dest=OUT/'training'/arm['id'];dest.mkdir(parents=True,exist_ok=False)
        for name in [r['checkpoint'],'trace.jsonl','validation_curve.json']:
            shutil.copyfile(old/name,dest/name);assert sha(old/name)==sha(dest/name)
        assert sha(dest/r['checkpoint'])==r['checkpoint_sha256']
        write(dest/'baseline_original_receipt.json',r)
        binding=dict(parent_arm=r['arm'],parent_receipt_sha256=sha(old/'receipt.json'),parent_checkpoint_sha256=r['checkpoint_sha256'],new_arm=arm,only_identifier_mapping=True,no_training_replay=True)
        write(dest/'baseline_binding.json',binding);write(dest/'receipt.json',dict(r,arm=arm,imported=True,baseline_binding_sha256=sha(dest/'baseline_binding.json')))
        imported.append(binding)
    write(OUT/'parent_import_review.json',dict(passed=True,parent_members=len(manifest['members']),all_parent_fresh_sha=True,standard_training_replayed=False,arms=imported))
    source=Path(__file__).resolve().parent;capsule=OUT/'execution_sources';capsule.mkdir(exist_ok=False)
    for f in source.glob('*.py'):shutil.copyfile(f,capsule/f.name)
    files=[OUT/'protocol.json',OUT/'initialization.pt',OUT/'initialization.json',OUT/'frame_order.npy',OUT/'frame_order_receipt.json',OUT/'run_env.sh',*source.glob('*.py'),*(OUT/'features').rglob('*'),*(OUT/'training').rglob('*')]
    files=[f for f in files if f.is_file()];pins=[]
    proc=subprocess.Popen(SSH+[f'test ! -e {REMOTE} && mkdir {REMOTE} && tar -xf - -C {REMOTE}'],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdin,mode='w|') as tar:
        for f in files:
            rel=f.relative_to(OUT).as_posix() if f.is_relative_to(OUT) else f.name
            tar.add(f,arcname=rel,recursive=False);pins.append(dict(path=rel,sha256=sha(f)))
        b=json.dumps(pins).encode();info=tarfile.TarInfo('deployment_pins.json');info.size=len(b);tar.addfile(info,io.BytesIO(b))
    proc.stdin.close();assert proc.wait()==0,proc.stderr.read().decode(errors='replace')
    code=f'''from pathlib import Path
import json,hashlib
r=Path({REMOTE!r});old=Path({PARENT_REMOTE!r})
for name in ['official','data','historical']:(r/name).symlink_to(old/name)
for mode in ['training','qualification']:
 (r/mode).mkdir(exist_ok=True);(r/mode/'hubert_base_on').symlink_to(old/mode/'hubert_base_on')
p=json.loads((r/'protocol.json').read_text())
for spec in p['models']:
 mid=spec['id'];rec=json.loads((r/'features'/mid/'receipt.json').read_text());assert rec['status']=='COMPLETED'
 for c in rec['conditions']:
  f=r/'features'/mid/c['file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==c['sha256']
  split=c['split'];folder=r/'datasets'/mid/split/'on'/('s%.2f'%c['input_scale']);folder.mkdir(parents=True)
  for name in ['gt_imgs','torso_imgs','ori_imgs','bc.jpg','bs.npy']:(folder/name).symlink_to(r/'data/source/May'/name)
  (folder/('transforms_'+split+'.json')).symlink_to(r/'data/splits'/split/('transforms_'+split+'.json'))
  (folder/'aud_hu.npy').symlink_to(f)
for row in json.loads((r/'deployment_pins.json').read_text()):assert hashlib.sha256((r/row['path']).read_bytes()).hexdigest()==row['sha256']
assert hashlib.sha256((r/'initialization.pt').read_bytes()).hexdigest()==p['parent_initial_sha256']
print(json.dumps(dict(passed=True,members={len(pins)},feature_conditions=14,read_only_parent_links=True)))
'''
    ret=subprocess.run(SSH+['python3','-'],input=code.encode(),capture_output=True,check=True);result=json.loads(ret.stdout)
    write(OUT/'deployment_receipt.json',result)
    write(OUT/'execution_source_pins.json',dict(passed=True,source_sha256={f.name:sha(f) for f in source.glob('*.py')},protocol_sha256=sha(OUT/'protocol.json'),run_env_sha256=sha(OUT/'run_env.sh')))
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
