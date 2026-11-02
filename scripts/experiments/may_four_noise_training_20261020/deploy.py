"""New persistent Ubuntu capsule; only reads existing qualified assets."""
from pathlib import Path
import subprocess,json,tarfile,io,shutil
from protocol import OUT,DATA,REMOTE,SSH,sha,write
OLD=Path('F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017/remote')
def remote(code):
 r=subprocess.run(SSH+['python3','-'],input=code,text=True,capture_output=True,timeout=300)
 assert r.returncode==0,r.stderr;return json.loads(r.stdout)
def main():
 assert not (OUT/'deployment_receipt.json').exists()
 assert json.loads((OUT/'feature_independent_review.json').read_text())['passed']
 source=Path(__file__).resolve().parent;p=json.loads((OUT/'protocol.json').read_text());files=[]
 assert sha(OUT/'initialization.pt')==p['initial_file_sha256']
 for f in source.glob('*.py'):compile(f.read_text(),str(f),'exec')
 # Source read-only reference lives in an existing completed job, copied into this root.
 qualified='/dev/shm/may-tts-purified-training-20261020-v1'
 for name in ['protocol.json','initialization.pt','initialization.json','frame_order.npy','frame_order_receipt.json','feature_independent_review.json']:
  files.append((OUT/name,name))
 files += [(f,f.name) for f in source.glob('*.py')]
 files += [(f,'features/'+f.name) for f in (OUT/'features').iterdir() if f.is_file()]
 for name in ['manifest.json','protocol.json']:files.append((DATA/name,'data/'+name))
 for split in ['train','val','test']:
  for f in (DATA/'splits'/split).iterdir():
   if f.is_file():files.append((f,'data/splits/'+split+'/'+f.name))
 hist=[]
 for name in ['shared_gt_track.npy','gt_boxes.npy','scores.json']:
  f=OLD/'scores'/name;assert f.is_file();files.append((f,'historical/'+name));hist.append(dict(file=name,sha256=sha(f)))
 (OUT/'historical').mkdir(exist_ok=False)
 write(OUT/'historical/pins.json',hist);files.append((OUT/'historical/pins.json','historical/pins.json'))
 env=Path('F:/Datasets/tts-exp/may_postnorm_scale_20261020/run_env.sh').read_text()
 env=env.replace('/dev/shm/may-postnorm-scale-20261020',REMOTE)
 # Shared dependency/extension tree was qualified again today; prepend only this capsule.
 env=env.replace(':$T/deps:/dev/shm/synctalk-multispeaker-data-20261017/deps:/dev/shm/synctalk-multispeaker-data-20261017',':$T/deps')
 env=env.replace('export TMPDIR=', 'export MPLCONFIGDIR="$R/matplotlib-cache"\nexport TMPDIR=')
 (OUT/'run_env.sh').write_text(env,encoding='utf-8',newline='\n');files.append((OUT/'run_env.sh','run_env.sh'))
 # Remote review uses the exact same audit implementation with only the output-root import adapted.
 text=(source/'review_training.py').read_text().replace('from protocol import OUT,sha,write','from runtime import ROOT as OUT,sha,write')
 (OUT/'review_remote_training.py').write_text(text,encoding='utf-8');files.append((OUT/'review_remote_training.py','review_remote_training.py'))
 pins=dict(source_sha256={f.name:sha(f) for f in source.glob('*.py')},protocol_sha256=sha(OUT/'protocol.json'),run_env_sha256=sha(OUT/'run_env.sh'))
 pins['source_sha256']['review_remote_training.py']=sha(OUT/'review_remote_training.py');pins['source_sha256']['run_env.sh']=pins['run_env_sha256']
 write(OUT/'execution_source_pins.json',pins);files.append((OUT/'execution_source_pins.json','execution_source_pins.json'))
 deployment=[dict(path=name,bytes=f.stat().st_size,sha256=sha(f)) for f,name in files]
 result=remote(f'''from pathlib import Path
import json,shutil
r=Path({REMOTE!r});assert not r.exists();assert shutil.disk_usage(r.parent).free>16*1024**3
r.mkdir();print(json.dumps(dict(created=True,root=str(r))))''');assert result['created']
 process=subprocess.Popen(SSH+['tar -xf - -C '+REMOTE],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
 with tarfile.open(fileobj=process.stdin,mode='w|') as tar:
  for f,name in files:tar.add(f,arcname=name,recursive=False)
  raw=json.dumps(deployment).encode();info=tarfile.TarInfo('deployment_pins.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
 process.stdin.close();assert process.wait()==0,process.stderr.read().decode(errors='replace')
 code=f'''from pathlib import Path
import json,hashlib,shutil
r=Path({REMOTE!r});qualified=Path({qualified!r})
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
for pin in json.loads((r/'deployment_pins.json').read_text()):
 f=r/pin['path'];assert f.stat().st_size==pin['bytes'] and sha(f)==pin['sha256'],str(f)
shutil.copytree(qualified/'official',r/'official',ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
source=(qualified/'data/source/May').resolve();target=r/'data/source/May';target.mkdir(parents=True)
for name in ['gt_imgs','torso_imgs','ori_imgs']:
 shutil.copytree(source/name,target/name,symlinks=False)
for name in ['bc.jpg','bs.npy']:shutil.copy2(source/name,target/name)
manifest=json.loads((r/'data/manifest.json').read_text());checked=0
for row in manifest['source_entries']:
 f=r/'data'/row['path']
 if row['path'].split('/')[2] in ['gt_imgs','torso_imgs','ori_imgs','bc.jpg','bs.npy']:
  assert f.stat().st_size==row['bytes'] and sha(f)==row['sha256'],str(f);checked+=1
features=json.loads((r/'features/receipt.json').read_text())
for row in features['conditions']:
 split=row['split'];mid=row['arm_id'] if split=='train' else 'clean';folder=r/'datasets'/mid/split/'on'/('s%.2f'%row['input_scale']);folder.mkdir(parents=True)
 for name in ['gt_imgs','torso_imgs','ori_imgs','bc.jpg','bs.npy']:(folder/name).symlink_to(target/name)
 (folder/('transforms_'+split+'.json')).symlink_to(r/'data/splits'/split/('transforms_'+split+'.json'))
 (folder/'aud_hu.npy').symlink_to(r/'features'/row['file'])
official=[dict(path=p.relative_to(r/'official').as_posix(),sha256=sha(p),bytes=p.stat().st_size) for p in sorted((r/'official').rglob('*')) if p.is_file()]
(r/'official_source_pins.json').write_text(json.dumps(official,indent=2))
receipt=dict(passed=True,members=len(json.loads((r/'deployment_pins.json').read_text())),fresh_sha_all_transport=True,original_source_files_fresh_sha=checked,official_files=len(official),data_root_persistent_copy=True,source_parent_unchanged=True,feature_conditions=22)
(r/'deployment_receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))'''
 result=remote(code);assert result['passed'];write(OUT/'deployment_receipt.json',result)
 ret=subprocess.run(SSH+['cat '+REMOTE+'/official_source_pins.json'],capture_output=True,check=True);(OUT/'official_source_pins.json').write_bytes(ret.stdout)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
