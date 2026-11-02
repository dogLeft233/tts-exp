"""Complete authorized preparation with shared training-reference geometry; no training."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
IDS=['CathyMcMorrisRodgers']
state=dict(stage='official_training_preprocessing_shared_reference_v2',pid=os.getpid(),identities=[],training_admitted=False,
    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
def save():(ROOT/'stage2_v2_Cathy_independent_worker_receipt.json').write_text(json.dumps(state,indent=2))
def read(path):
    for _ in range(10):
        try:return json.loads(path.read_text())
        except json.JSONDecodeError:time.sleep(.1)
    raise ValueError('partial JSON receipt')
def active(script,identity):
    found=[]
    for p in psutil.process_iter(['name','cmdline']):
        cmd=p.info['cmdline'] or []
        if p.info['name'].startswith('python') and any(Path(x).name==script for x in cmd) and identity in cmd:found.append(p.pid)
    return found
def execute(identity,phase,script,args,target,json_receipt=True,interpreter=sys.executable):
    state.update(current_identity=identity,current_phase=phase);save()
    if active(script.name,identity):
        deadline=time.time()+21600
        while active(script.name,identity):assert time.time()<deadline;time.sleep(20)
        assert target.exists(),f'existing {phase} failed'
    elif not target.exists():
        raw=script.read_bytes();digest=hashlib.sha256(raw).hexdigest();capture=ROOT/'executed_sources'/digest
        capture.mkdir(parents=True,exist_ok=True);saved=capture/script.name
        if saved.exists():assert saved.read_bytes()==raw
        else:saved.write_bytes(raw)
        with (ROOT/f'stage2_v2_{identity}_{phase}.log').open('x') as log:
            subprocess.run([interpreter,str(script)]+args,cwd=ROOT/'official',stdout=log,stderr=subprocess.STDOUT,check=True)
    assert target.exists(),f'{phase} missing output'
    if json_receipt:assert read(target)['passed'],f'{phase} qualification failed'
try:
    for identity in IDS:
        state.update(current_identity=identity,current_phase='wait_stage1');save()
        stage1=ROOT/'preprocessed'/identity/'stage1_receipt.json';deadline=time.time()+21600
        while not stage1.exists() or not read(stage1).get('passed',False):
            worker=read(ROOT/'stage1_worker_receipt.json');assert worker.get('passed') is not False,worker.get('error')
            assert time.time()<deadline;time.sleep(20)
        folder=ROOT/'prepared'/identity
        execute(identity,'layout',ROOT/'prepare_layout.py',['--root',str(ROOT),'--identity',identity],folder/'input_binding.json')
        execute(identity,'train_tracker',ROOT/'official/data_utils/face_tracking/face_tracker.py',
            ['--path',str(folder/'train/ori_imgs'),'--img_h','512','--img_w','512','--frame_num','6000'],folder/'train/track_params.pt',False)
        for split in ['val','test']:
            execute(identity,split+'_tracker',ROOT/'face_tracker_frozen.py',['--root',str(ROOT),'--identity',identity,'--split',split],folder/split/'track_params.pt',False)
        execute(identity,'train_flow',ROOT/'flow_stream.py',['--root',str(ROOT),'--identity',identity,'--split','train'],folder/'train/flow_receipt.json')
        execute(identity,'train_bundle',ROOT/'official/data_utils/face_tracking/bundle_adjustment.py',
            ['--path',str(folder/'train'),'--img_h','512','--img_w','512'],folder/'train/bundle_adjustment.pt',False)
        execute(identity,'geometry',ROOT/'prepare_geometry_v2.py',['--root',str(ROOT),'--identity',identity],folder/'geometry_v2/geometry_qualification.json')
        execute(identity,'supervision',ROOT/'supervision.py',['--root',str(ROOT),'--identity',identity],folder/'supervision_receipt.json')
        execute(identity,'ave',ROOT/'ave_cache.py',['--root',str(ROOT),'--identity',identity],ROOT/'features'/identity/'ave_qualification.json')
        execute(identity,'score_reference',ROOT/'prepare_score_reference.py',['--root',str(ROOT),'--identity',identity],ROOT/'score_reference'/identity/'score_reference_receipt.json',interpreter='/home/wjj/.venvs/syncnet/bin/python')
        row=dict(identity=identity,geometry_v2_qualified=True,supervision_qualified=True,ave_qualified=True,score_reference_prepared=True,training_admitted=False)
        state['identities'].append(row);save();print(json.dumps(row),flush=True)
        state['current_phase']='wait_verified_persistence_and_cache_retirement';save()
        reclamation=ROOT/'snapshots'/identity/'supervision_reclamation.json';deadline=time.time()+21600
        while not reclamation.exists():
            assert time.time()<deadline,'verified persistence timeout';time.sleep(20)
        assert read(reclamation)['passed'] and read(reclamation)['requires_restore_before_training']
    state['passed']=True;save()
except Exception as error:
    state.update(passed=False,error=repr(error));save();raise
