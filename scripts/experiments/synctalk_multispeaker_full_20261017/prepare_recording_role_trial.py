"""One isolated, predeclared recording-role fallback; preparation only, no training."""
import hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from execution_journal import atomic_json,exclusive_lock
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017')
ROOT=BASE/'dataset_revisions/DebbieDingell_recording_roles_v2'
IDENTITY='DebbieDingell'
CONTRACT='37f0714cecd41cca0f531bb062b6029bad2e96fe5bc19bd3ad0827d61f3dd005'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def execute(state,name,script,args,target,json_output=True):
    state['phase']=name;atomic_json(ROOT/'preparation_worker_receipt.json',state)
    assert sha(ROOT/'preprocessing_contract_v2.json')==CONTRACT
    assert not target.exists(),'this attempt cannot overwrite/replay prior outputs'
    captured=ROOT/'executed_sources'/sha(script);captured.mkdir(parents=True,exist_ok=True)
    copy=captured/script.name
    if copy.exists():assert copy.read_bytes()==script.read_bytes()
    else:shutil.copyfile(script,copy)
    with (ROOT/(name+'.log')).open('x') as log:
        subprocess.run([sys.executable,str(script),*args],cwd=ROOT/'official',stdout=log,stderr=subprocess.STDOUT,check=True)
    assert target.exists()
    if json_output:assert json.loads(target.read_text())['passed']
def run():
    contract=json.loads((ROOT/'revision_contract.json').read_text())
    assert contract['revision']==ROOT.name and contract['no_TFG_updates_observed'] and contract['preserve_original_data']
    assert sha(ROOT/'preprocessing_contract_v2.json')==CONTRACT
    state=dict(pid=os.getpid(),phase='started',passed=None,training_admitted=False,worker_can_start_training=False,
        revision=ROOT.name,revision_contract_sha256=sha(ROOT/'revision_contract.json'),source_sha256=sha(Path(__file__)))
    with exclusive_lock(ROOT/'preparation.lock'):
        assert not (ROOT/'preparation_worker_receipt.json').exists(),'immutable attempt already exists'
        atomic_json(ROOT/'preparation_worker_receipt.json',state)
        try:
            # CPU preparation shares no caches or outputs with the failed pairing.
            state['phase']='wait_workspace_reserve';atomic_json(ROOT/'preparation_worker_receipt.json',state)
            deadline=time.monotonic()+43200
            while shutil.disk_usage(ROOT).free<=3.5*1024**3:
                cathy=json.loads((BASE/'stage2_v2_Cathy_independent_worker_receipt.json').read_text())
                assert cathy.get('passed') is not False,'Cathy preparation failed; inspect before further trial work'
                assert time.monotonic()<deadline,'workspace reserve timeout';time.sleep(30)
            execute(state,'material',ROOT/'prepare_materials.py',['--root',str(ROOT),'--identity',IDENTITY],ROOT/'material'/IDENTITY/'material_receipt.json',False)
            assert json.loads((ROOT/'material'/IDENTITY/'material_receipt.json').read_text())['media_preparation_passed']
            execute(state,'material_independent_qa',ROOT/'independent_material_qa.py',['--root',str(ROOT),'--identity',IDENTITY],ROOT/'material'/IDENTITY/'independent_material_qa.json')
            state['phase']='wait_Cathy_GPU_and_verified_cache_retirement';atomic_json(ROOT/'preparation_worker_receipt.json',state)
            deadline=time.monotonic()+43200
            while True:
                cathy=json.loads((BASE/'stage2_v2_Cathy_independent_worker_receipt.json').read_text())
                assert cathy.get('passed') is not False,'Cathy qualification failed; preserve both attempts and inspect'
                capacity=BASE/'capacity_qualification/CathyMcMorrisRodgers/capacity_qualification.json'
                if cathy.get('passed') and capacity.exists() and json.loads(capacity.read_text()).get('passed') and shutil.disk_usage(ROOT).free>3*1024**3:break
                assert time.monotonic()<deadline,'waiting Cathy qualification/workspace timeout';time.sleep(30)
            execute(state,'stage1',ROOT/'preprocess_images_landmarks.py',['--root',str(ROOT),'--identity',IDENTITY],ROOT/'preprocessed'/IDENTITY/'stage1_receipt.json')
            folder=ROOT/'prepared'/IDENTITY
            execute(state,'layout',ROOT/'prepare_layout.py',['--root',str(ROOT),'--identity',IDENTITY],folder/'input_binding.json')
            execute(state,'train_tracker',ROOT/'official/data_utils/face_tracking/face_tracker.py',
                ['--path',str(folder/'train/ori_imgs'),'--img_h','512','--img_w','512','--frame_num','6000'],folder/'train/track_params.pt',False)
            for split in ['val','test']:
                execute(state,split+'_tracker',ROOT/'face_tracker_frozen.py',['--root',str(ROOT),'--identity',IDENTITY,'--split',split],folder/split/'track_params.pt',False)
            execute(state,'train_flow',ROOT/'flow_stream.py',['--root',str(ROOT),'--identity',IDENTITY,'--split','train'],folder/'train/flow_receipt.json')
            execute(state,'train_bundle',ROOT/'official/data_utils/face_tracking/bundle_adjustment.py',
                ['--path',str(folder/'train'),'--img_h','512','--img_w','512'],folder/'train/bundle_adjustment.pt',False)
            execute(state,'geometry',ROOT/'prepare_geometry_v2.py',['--root',str(ROOT),'--identity',IDENTITY],folder/'geometry_v2/geometry_qualification.json')
            state.update(phase='geometry_qualified_pending_independent_persistence_and_remaining_gates',passed=True)
            atomic_json(ROOT/'preparation_worker_receipt.json',state)
        except BaseException:
            state.update(passed=False,error=traceback.format_exc());atomic_json(ROOT/'preparation_worker_receipt.json',state);raise
if __name__=='__main__':run()
