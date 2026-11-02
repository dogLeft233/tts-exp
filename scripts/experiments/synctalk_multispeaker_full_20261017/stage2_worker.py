"""Finish official geometry/supervision before any generator-training admission."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
import psutil
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
IDS=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']
state=dict(stage='geometry_and_supervision',pid=os.getpid(),identities=[],training_admitted=False)
def write():(ROOT/'stage2_worker_receipt.json').write_text(json.dumps(state,indent=2))
def active(script,identity):
    result=[]
    for p in psutil.process_iter(['name','cmdline']):
        cmd=p.info['cmdline'] or []
        if p.info['name'].startswith('python') and any(Path(x).name==script for x in cmd) and any(identity in x for x in cmd):result.append(p.pid)
    return result
def read(path):
    for _ in range(10):
        try:return json.loads(path.read_text())
        except json.JSONDecodeError:time.sleep(.1)
    raise ValueError('partial JSON receipt')
def capture(path):
    raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest();folder=ROOT/'executed_sources'/digest
    folder.mkdir(parents=True,exist_ok=True);out=folder/path.name
    if out.exists():assert out.read_bytes()==raw
    else:out.write_bytes(raw)
    return digest
def execute(identity,phase,script,args,target,json_receipt=True):
    state.update(current_identity=identity,current_phase=phase);write()
    running=active(script.name,identity)
    if running:
        deadline=time.time()+14400
        while active(script.name,identity):
            assert time.time()<deadline,f'{phase} existing process timeout';time.sleep(20)
        assert target.exists(),f'existing {phase} failed; no receipt'
    elif not target.exists():
        digest=capture(script)
        log=ROOT/f'stage2_{identity}_{phase}.log'
        with log.open('x') as output:
            subprocess.run([sys.executable,str(script)]+args,cwd=ROOT/'official',stdout=output,stderr=subprocess.STDOUT,check=True)
        assert target.exists(),f'{phase} missing output'
    if json_receipt:assert read(target)['passed'],f'{phase} failed qualification'

try:
    for identity in IDS:
        state['current_identity']=identity;state['current_phase']='wait_stage1';write()
        stage1=ROOT/'preprocessed'/identity/'stage1_receipt.json';deadline=time.time()+14400
        while not stage1.exists() or not read(stage1).get('passed',False):
            worker=read(ROOT/'stage1_worker_receipt.json')
            assert worker.get('passed') is not False,worker.get('error')
            assert time.time()<deadline,'stage1 timeout';time.sleep(20)
        folder=ROOT/'prepared'/identity
        execute(identity,'layout',ROOT/'prepare_layout.py',['--root',str(ROOT),'--identity',identity],folder/'input_binding.json')
        execute(identity,'train_tracker',ROOT/'official/data_utils/face_tracking/face_tracker.py',
            ['--path',str(folder/'train/ori_imgs'),'--img_h','512','--img_w','512','--frame_num','6000'],folder/'train/track_params.pt',False)
        for split in ['val','test']:
            execute(identity,split+'_tracker',ROOT/'face_tracker_frozen.py',
                ['--root',str(ROOT),'--identity',identity,'--split',split],folder/split/'track_params.pt',False)
        execute(identity,'train_flow',ROOT/'flow_stream.py',['--root',str(ROOT),'--identity',identity,'--split','train'],folder/'train/flow_receipt.json')
        execute(identity,'train_bundle',ROOT/'official/data_utils/face_tracking/bundle_adjustment.py',
            ['--path',str(folder/'train'),'--img_h','512','--img_w','512'],folder/'train/bundle_adjustment.pt',False)
        for split in ['val','test']:
            execute(identity,split+'_flow',ROOT/'flow_stream.py',['--root',str(ROOT),'--identity',identity,'--split',split],folder/split/'flow_receipt.json')
            execute(identity,split+'_bundle',ROOT/'bundle_frozen.py',['--root',str(ROOT),'--identity',identity,'--split',split],folder/split/'bundle_adjustment.pt',False)
        execute(identity,'pose_qa',ROOT/'qualify_pose.py',['--root',str(ROOT),'--identity',identity],folder/'pose_qualification.json')
        execute(identity,'supervision',ROOT/'supervision.py',['--root',str(ROOT),'--identity',identity],folder/'supervision_receipt.json')
        result=dict(identity=identity,pose_qualified=True,supervision_qualified=True,training_admitted=False)
        state['identities'].append(result);write();print(json.dumps(result),flush=True)
    state['passed']=True;write()
except Exception as error:
    state['passed']=False;state['error']=repr(error);write();raise
