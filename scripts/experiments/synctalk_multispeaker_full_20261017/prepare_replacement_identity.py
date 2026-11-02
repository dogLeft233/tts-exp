"""Full unchanged official preparation in one immutable replacement namespace."""
import argparse,fcntl,json,os,shutil,subprocess,sys,time,traceback
from pathlib import Path
from execution_journal import atomic_json,exclusive_lock
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017')
CONTRACT='37f0714cecd41cca0f531bb062b6029bad2e96fe5bc19bd3ad0827d61f3dd005'


def run(root,identity):
    assert root.parent==BASE/'dataset_revisions' and root.name==identity+'_metadata_order_v1' and root.resolve()==root
    import hashlib
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    contract=json.loads((root/'revision_contract.json').read_text())
    assert contract['identity']==identity and contract['revision']==root.name and contract['no_TFG_updates_observed']
    assert contract['preserve_original_data'] and sha(root/'preprocessing_contract_v2.json')==CONTRACT
    receipt=root/'preparation_worker_receipt.json'
    state=dict(pid=os.getpid(),phase='waiting_gpu_and_workspace',passed=None,identity=identity,revision=root.name,
        training_admitted=False,worker_can_start_training=False,source_sha256=sha(Path(__file__)),
        revision_contract_sha256=sha(root/'revision_contract.json'))
    def execute(name,script,args,target,field='passed'):
        state['phase']=name;atomic_json(receipt,state)
        assert sha(root/'preprocessing_contract_v2.json')==CONTRACT and not target.exists()
        saved=root/'executed_sources'/sha(script)/script.name;saved.parent.mkdir(parents=True,exist_ok=True)
        if saved.exists():assert saved.read_bytes()==script.read_bytes()
        else:shutil.copyfile(script,saved)
        with (root/(name+'.log')).open('x') as log:
            subprocess.run([sys.executable,str(script),*args],cwd=root/'official',stdout=log,stderr=subprocess.STDOUT,check=True)
        assert target.exists()
        if field:assert json.loads(target.read_text())[field]
    with exclusive_lock(root/'preparation.lock'):
        assert not receipt.exists();atomic_json(receipt,state)
        try:
            # Never hold the GPU lock while waiting for space: the previous
            # identity's independent persistence/retirement may need that lock.
            with (BASE/'preparation_GPU.lock').open('a+b') as gpu:
                deadline=time.monotonic()+43200
                while True:
                    assert time.monotonic()<deadline,'bounded GPU/workspace wait expired'
                    if shutil.disk_usage(root).free<=3.5*1024**3:time.sleep(30);continue
                    try:fcntl.flock(gpu.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:time.sleep(30);continue
                    if shutil.disk_usage(root).free>3.5*1024**3:break
                    fcntl.flock(gpu.fileno(),fcntl.LOCK_UN);time.sleep(30)
                execute('material',root/'prepare_materials.py',['--root',str(root),'--identity',identity],root/'material'/identity/'material_receipt.json','media_preparation_passed')
                execute('material_qa',root/'independent_material_qa.py',['--root',str(root),'--identity',identity],root/'material'/identity/'independent_material_qa.json')
                execute('stage1',root/'preprocess_images_landmarks.py',['--root',str(root),'--identity',identity],root/'preprocessed'/identity/'stage1_receipt.json')
                stage1=json.loads((root/'preprocessed'/identity/'stage1_receipt.json').read_text())
                if stage1['review_required']:
                    state['phase']='waiting_stage1_independent_face_review';atomic_json(receipt,state)
                    review=root/(identity+'_face_adjudication.json')
                    deadline=time.monotonic()+43200
                    while not review.exists():assert time.monotonic()<deadline;time.sleep(30)
                    result=json.loads(review.read_text());assert result['all_five_review_flags_resolved'] and result['complete_landmark_support_frames']==8250
                    assert all(x['official_primary_face_correct'] for x in result['reviewed_frames'])
                folder=root/'prepared'/identity
                execute('layout',root/'prepare_layout.py',['--root',str(root),'--identity',identity],folder/'input_binding.json')
                execute('train_tracker',root/'official/data_utils/face_tracking/face_tracker.py',
                    ['--path',str(folder/'train/ori_imgs'),'--img_h','512','--img_w','512','--frame_num','6000'],folder/'train/track_params.pt',None)
                for split in ['val','test']:
                    execute(split+'_tracker',root/'face_tracker_frozen.py',['--root',str(root),'--identity',identity,'--split',split],folder/split/'track_params.pt',None)
                execute('train_flow',root/'flow_stream.py',['--root',str(root),'--identity',identity,'--split','train'],folder/'train/flow_receipt.json')
                execute('train_bundle',root/'official/data_utils/face_tracking/bundle_adjustment.py',
                    ['--path',str(folder/'train'),'--img_h','512','--img_w','512'],folder/'train/bundle_adjustment.pt',None)
                execute('geometry',root/'prepare_geometry_v2.py',['--root',str(root),'--identity',identity],folder/'geometry_v2/geometry_qualification.json')
                state.update(phase='geometry_qualified_pending_independent_persistence_and_remaining_gates',passed=True);atomic_json(receipt,state)
        except BaseException:
            state.update(passed=False,error=traceback.format_exc());atomic_json(receipt,state);raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.identity)
