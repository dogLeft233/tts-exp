"""Resume only unstarted geometry preparation after an unrelated identity dependency failed."""
import hashlib,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from execution_journal import atomic_json,exclusive_lock
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017');ROOT=BASE/'dataset_revisions/DebbieDingell_recording_roles_v2';IDENTITY='DebbieDingell'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def run():
    receipt=ROOT/'geometry_resume_worker_receipt.json'
    with exclusive_lock(ROOT/'preparation.lock'):
        assert not receipt.exists()
        previous=json.loads((ROOT/'preparation_worker_receipt.json').read_text())
        assert previous['passed'] is False and previous['phase']=='wait_Cathy_GPU_and_verified_cache_retirement'
        assert 'Cathy qualification failed' in previous['error']
        assert json.loads((ROOT/'material'/IDENTITY/'independent_material_qa.json').read_text())['passed']
        assert not (ROOT/'preprocessed'/IDENTITY).exists() and not (ROOT/'prepared'/IDENTITY).exists()
        state=dict(pid=os.getpid(),phase='waiting_workspace',passed=None,training_admitted=False,worker_can_start_training=False,
            prior_worker_receipt_sha256=sha(ROOT/'preparation_worker_receipt.json'),unchanged_material_qa_sha256=sha(ROOT/'material'/IDENTITY/'independent_material_qa.json'),source_sha256=sha(Path(__file__)))
        atomic_json(receipt,state)
        try:
            deadline=time.monotonic()+43200
            while shutil.disk_usage(ROOT).free<=3*1024**3:assert time.monotonic()<deadline;time.sleep(30)
            def execute(name,script,args,target,json_output=True):
                state['phase']=name;atomic_json(receipt,state)
                assert sha(ROOT/'preprocessing_contract_v2.json')=='37f0714cecd41cca0f531bb062b6029bad2e96fe5bc19bd3ad0827d61f3dd005'
                assert not target.exists(),'no completed or failed numerical operation is replayed'
                capture=ROOT/'executed_sources'/sha(script);capture.mkdir(parents=True,exist_ok=True)
                saved=capture/script.name
                if saved.exists():assert saved.read_bytes()==script.read_bytes()
                else:shutil.copyfile(script,saved)
                with (ROOT/('resume_'+name+'.log')).open('x') as log:subprocess.run([sys.executable,str(script),*args],cwd=ROOT/'official',stdout=log,stderr=subprocess.STDOUT,check=True)
                assert target.exists()
                if json_output:assert json.loads(target.read_text())['passed']
            execute('stage1',ROOT/'preprocess_images_landmarks.py',['--root',str(ROOT),'--identity',IDENTITY],ROOT/'preprocessed'/IDENTITY/'stage1_receipt.json')
            folder=ROOT/'prepared'/IDENTITY
            execute('layout',ROOT/'prepare_layout.py',['--root',str(ROOT),'--identity',IDENTITY],folder/'input_binding.json')
            execute('train_tracker',ROOT/'official/data_utils/face_tracking/face_tracker.py',['--path',str(folder/'train/ori_imgs'),'--img_h','512','--img_w','512','--frame_num','6000'],folder/'train/track_params.pt',False)
            for split in ['val','test']:execute(split+'_tracker',ROOT/'face_tracker_frozen.py',['--root',str(ROOT),'--identity',IDENTITY,'--split',split],folder/split/'track_params.pt',False)
            execute('train_flow',ROOT/'flow_stream.py',['--root',str(ROOT),'--identity',IDENTITY,'--split','train'],folder/'train/flow_receipt.json')
            execute('train_bundle',ROOT/'official/data_utils/face_tracking/bundle_adjustment.py',['--path',str(folder/'train'),'--img_h','512','--img_w','512'],folder/'train/bundle_adjustment.pt',False)
            execute('geometry',ROOT/'prepare_geometry_v2.py',['--root',str(ROOT),'--identity',IDENTITY],folder/'geometry_v2/geometry_qualification.json')
            state.update(passed=True,phase='geometry_qualified_pending_remaining_gates');atomic_json(receipt,state)
        except BaseException:
            state.update(passed=False,error=traceback.format_exc());atomic_json(receipt,state);raise
if __name__=='__main__':run()
