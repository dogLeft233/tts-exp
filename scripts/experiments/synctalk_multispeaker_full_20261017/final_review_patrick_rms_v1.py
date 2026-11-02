import csv,datetime,hashlib,io,itertools,json,shlex,subprocess,tarfile,wave
from pathlib import Path
import numpy as np
from PIL import Image

P=Path('F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_v1/score_27_v1')
D=P/'delivery_v1';D.mkdir(exist_ok=False)
SSH=['ssh','-b','192.168.50.1','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','wjj@192.168.50.2']
REMOTE='/dev/shm/synctalk-multispeaker-data-20261017/score_reference/PatrickLeahy'
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def read(f):return json.loads(f.read_text(encoding='utf-8'))
def save(f,v):
 with f.open('x',encoding='utf-8') as stream:json.dump(v,stream,ensure_ascii=False,indent=2)
def remote(code):return subprocess.check_output(SSH+['python3 -'],input=code.encode())
names=['score_reference_receipt.json']+[f'test_{i:02d}/audio_features_{g:+d}dB.npy' for i in range(6) for g in [-6,0,6]]
code="from pathlib import Path;import hashlib,json;p=Path(%r);names=%r;print(json.dumps([dict(path=n,bytes=(p/n).stat().st_size,sha256=hashlib.sha256((p/n).read_bytes()).hexdigest()) for n in names]))"%(REMOTE,names)
inputs=json.loads(remote(code));save(D/'reference_manifest.json',inputs)
raw=subprocess.check_output(SSH+['tar -czf - -C '+shlex.quote(REMOTE)+' --null -T -'],input=('\0'.join(names)+'\0').encode())
(D/'reference_inputs.tar.gz').write_bytes(raw)
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as t:
 for m in t:
  row=next(x for x in inputs if x['path']==m.name);b=t.extractfile(m).read()
  assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']
  target=D/'reference_inputs'/m.name;assert target.resolve().is_relative_to((D/'reference_inputs').resolve());target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(b)
assert json.loads(remote(code))==inputs
assert read(D/'reference_inputs/score_reference_receipt.json')['passed']
summary=read(P/'all27_preview.json');assert read(P/'contract.json')['training_contract_sha256']==sha(P.parent/'contract.json');assert summary['passed'] and summary['descriptive_only']
keys=lambda c:(c['training_gain_db'],c['driver_gain_db'],c['evaluation_gain_db'])
indexed={keys(c):c for c in summary['cells']};assert len(summary['cells'])==len(indexed)==27 and set(indexed)==set(itertools.product([-6,0,6],repeat=3))
state=read(P/'controller_receipt.json');assert state['passed'] and len(state['completed'])==9 and state['status']=='all27_ready_for_final_delivery_review'
checked=0;maxmatrix=0.;maxendpoint=0.;proofs=[];csvrows=[];curves=[];pngs=0
for gain,driver in itertools.product([-6,0,6],repeat=2):
 s=P/'scores'/f'train_{gain:+d}_driver_{driver:+d}';r=P/'renders'/f'train_{gain:+d}'/f'driver_{driver:+d}'
 v=read(s/'verification.json');manifest=read(s/'manifest.json');byname={x['path']:x for x in manifest};seen=[]
 assert v['passed'] and v['all_archived_member_bytes_verified'] and sha(s/'manifest.json')==v['manifest_sha256'] and sha(s/'artifacts.tar.gz')==v['archive_sha256']
 with tarfile.open(s/'artifacts.tar.gz','r:gz') as t:
  for m in t:
   assert m.isfile() and m.name in byname;row=byname[m.name];b=t.extractfile(m).read()
   assert len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256']==sha(s/'local'/m.name);seen.append(m.name)
 assert sorted(seen)==sorted(byname)
 scores=read(s/'local/scores.json');assert scores['passed'] and scores['checked_matrices']==18 and scores['independent_matrix_qa']
 for c in scores['cells']:
  assert c==indexed[keys(c)] and keys(c)[:2]==(gain,driver)
  assert [x['clip_id'] for x in c['clips']]==[f'test_{i:02d}' for i in range(6)]
  for clip in c['clips']:
   f=s/'local'/clip['clip_id']/f"distances_score_{c['evaluation_gain_db']:+d}.npy";assert sha(f)==clip['distances_sha256']
   stored=np.load(f,allow_pickle=False);vf=np.load(s/'local'/clip['clip_id']/'visual_features.npy',allow_pickle=False).astype(np.float64)
   af=np.load(D/'reference_inputs'/clip['clip_id']/f"audio_features_{c['evaluation_gain_db']:+d}dB.npy",allow_pickle=False).astype(np.float64)
   assert vf.shape==af.shape==(245,1024) and stored.shape==(245,31)
   pad=np.zeros((275,1024),dtype=np.float64);pad[15:260]=af
   rebuilt=np.empty((245,31),dtype=np.float64)
   # Independent column-wise formulation, separate from production row-wise implementation.
   for j in range(31):
    delta=vf-pad[j:j+245]+1e-6;rebuilt[:,j]=np.sqrt(np.sum(delta*delta,axis=1))
   assert np.isfinite(rebuilt).all();err=float(np.max(np.abs(rebuilt-stored)));assert err<3e-5;maxmatrix=max(maxmatrix,err)
   for support,section in [('guard20',rebuilt[20:225]),('full',rebuilt)]:
    curve=section.mean(axis=0);index=int(curve.argmin());minimum=float(curve[index]);median=float(np.median(curve))
    vals=dict(sync_c=median-minimum,sync_d=minimum,distance_d0=float(curve[15]),background_median=median)
    assert clip[support]['av_offset']==15-index
    for key,value in vals.items():
     err=abs(value-clip[support][key]);assert err<3e-5;maxendpoint=max(maxendpoint,err)
   checked+=1
  for support in ['guard20','full']:
   for key in ['sync_c','sync_d','distance_d0','background_median']:
    assert abs(sum(x[support][key] for x in c['clips'])/6-c['mean_'+support][key])<1e-12
   assert c['mean_'+support]['offsets']==[x[support]['av_offset'] for x in c['clips']]
   csvrows.append(dict(training_gain_db=gain,driver_gain_db=driver,scorer_gain_db=c['evaluation_gain_db'],support=support,**{k:v for k,v in c['mean_'+support].items() if k!='offsets'},offsets=json.dumps(c['mean_'+support]['offsets'])))
   for j in range(31):curves.append(dict(training_gain_db=gain,driver_gain_db=driver,scorer_gain_db=c['evaluation_gain_db'],support=support,av_offset=15-j,mean_clip_distance=sum(x[support]['curve'][j] for x in c['clips'])/6))
 rv=read(r/'verification.json');cloud=read(r/'cloud_readback.json');locator=read(r/'cloud_storage_locator.json');rm=read(r/'manifest.json')
 assert rv['passed'] and rv['all_archived_member_bytes_verified'] and sha(r/'manifest.json')==rv['manifest_sha256']
 assert cloud['passed'] and cloud['independent_linux_cloud_readback'] and cloud['sha256']==rv['archive_sha256']==locator['archive_sha256'] and locator['cloud']==cloud
 for row in rm:
  if row['path']!='original_pcm.mkv':assert sha(r/'local'/row['path'])==row['sha256']
 rr=read(r/'local/render_receipt.json');assert rr['passed'] and rr['rgb_and_pcm_readback_exact'] and rr['ema']['passed'] and not rr['formal_fine17_model']
 from patrick_rms_score_receipts_v1 import validate_render
 validate_render(rr,read(P/'contract.json'),sha(P.parent/'contract.json'),gain,driver)
 assert len(rr['frames'])==1500 and [x['index'] for x in rr['frames']]==list(range(1500)) and [x['source_audio_id'] for x in rr['frames']]==list(range(50,1550))
 assert next(x for x in rm if x['path']=='original_pcm.mkv')['sha256']==rr['movie_sha256']
 for i in [0,750,1499]:
  rgb=np.asarray(Image.open(r/'local'/f'portrait_frame{i:04d}.png').convert('RGB'));head=np.load(r/'local'/f'raw_head_frame{i:04d}.npy',allow_pickle=False)
  assert hashlib.sha256(rgb.tobytes()).hexdigest()==rr['frames'][i]['rgb_sha256']
  assert hashlib.sha256(head.tobytes()).hexdigest()==rr['frames'][i]['head_float_sha256'];pngs+=1
 with wave.open(str(r/'local/original_test_pcm.wav'),'rb') as w:
  assert (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getnframes())==(16000,1,2,960000)
  assert hashlib.sha256(w.readframes(960000)).hexdigest()==rr['original_pcm_sha256']
 proofs.append(dict(training_gain_db=gain,driver_gain_db=driver,score_archive_sha256=v['archive_sha256'],render_archive_sha256=rv['archive_sha256'],cloud_receipt_sha256=sha(r/'cloud_readback.json')))
 print('reviewed',gain,driver,flush=True)
assert checked==162 and pngs==27
for name,rows in [('scores_27_cells.csv',csvrows),('mean_distance_curves.csv',curves)]:
 with (D/name).open('x',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
receipt=dict(passed=True,identity='PatrickLeahy',phase='coarse',epoch=10,seed=42,descriptive_only=True,training_control_status=summary['training_control_status'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),cells=27,checked_matrices=checked,matrix_max_abs_error=maxmatrix,endpoint_max_abs_error=maxendpoint,tolerance=3e-5,score_archive_members_fresh_sha=225,stored_png_and_float_head_pairs_checked=pngs,wav_pcm_checked=9,scope='fresh local score/archive and retained media SHA; independently rebuild all distances and endpoints from stored VF/canonical AF; all 27 equal-clip aggregations; retained PNG/head pixel hashes and PCM; review previously full 1500-frame FFV1/PCM and independent cloud whole archive readback receipts',not_repeated='no new GPU inference, no full lossless movie restore, no fresh cloud payload reread',all27_summary_sha256=sha(P/'all27_preview.json'),review_source_sha256=sha(Path(__file__)),reference_archive_sha256=sha(D/'reference_inputs.tar.gz'),proofs=proofs)
save(D/'independent_review.json',receipt)
print("INDEPENDENT_RMS27_REVIEW_PASS",flush=True)
