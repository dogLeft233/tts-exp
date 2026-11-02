"""One metadata-ordered test replacement, with byte-proven unchanged training preprocessing."""
import hashlib,json,os,shutil,subprocess,sys,traceback
from pathlib import Path
from execution_journal import atomic_json,exclusive_lock
BASE=Path('/dev/shm/synctalk-multispeaker-data-20261017');ROOT=BASE/'dataset_revisions/CathyMcMorrisRodgers_recording2_v2';IDENTITY='CathyMcMorrisRodgers'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def main_equivalent(original,candidate):
    a=next(r for r in original['recordings'] if r['role']=='main');b=next(r for r in candidate['recordings'] if r['role']=='main')
    for field in ['source','pcm_sha256','frame_rgb_sequence_sha256','frames','audio_samples']:assert a[field]==b[field],field
def run():
    with exclusive_lock(ROOT/'preparation.lock'):
        receipt=ROOT/'preparation_worker_receipt.json';assert not receipt.exists()
        state=dict(pid=os.getpid(),phase='material',passed=None,training_admitted=False,worker_can_start_training=False,
            revision=ROOT.name,source_sha256=sha(Path(__file__)),revision_contract_sha256=sha(ROOT/'revision_contract.json'))
        atomic_json(receipt,state)
        def execute(phase,script,args,target,passed_field='passed'):
            state['phase']=phase;atomic_json(receipt,state)
            assert sha(ROOT/'preprocessing_contract_v2.json')=='37f0714cecd41cca0f531bb062b6029bad2e96fe5bc19bd3ad0827d61f3dd005'
            assert not target.exists(),'never replay an executed numerical stage'
            capture=ROOT/'executed_sources'/sha(script);capture.mkdir(parents=True,exist_ok=True);saved=capture/script.name
            if saved.exists():assert saved.read_bytes()==script.read_bytes()
            else:shutil.copyfile(script,saved)
            with (ROOT/(phase+'.log')).open('x') as log:subprocess.run([sys.executable,str(script),*args],cwd=ROOT/'official',stdout=log,stderr=subprocess.STDOUT,check=True)
            assert target.exists()
            if passed_field:assert json.loads(target.read_text())[passed_field]
        try:
            assert shutil.disk_usage(ROOT).free>1.5*1024**3,'reserve material and newly processed test support'
            execute('material',ROOT/'prepare_materials.py',['--root',str(ROOT),'--identity',IDENTITY],ROOT/'material'/IDENTITY/'material_receipt.json','media_preparation_passed')
            execute('material_qa',ROOT/'independent_material_qa.py',['--root',str(ROOT),'--identity',IDENTITY],ROOT/'material'/IDENTITY/'independent_material_qa.json')
            original=json.loads((BASE/'material'/IDENTITY/'material_receipt.json').read_text());candidate=json.loads((ROOT/'material'/IDENTITY/'material_receipt.json').read_text());main_equivalent(original,candidate)
            state['phase']='full_main_JPEG_reuse_parity';atomic_json(receipt,state)
            reuse=ROOT/'reuse_stage1';reuse.mkdir();prior=json.loads((BASE/'preprocessed'/IDENTITY/'stage1_receipt.json').read_text());assert prior['passed']
            prior['splits']=[s for s in prior['splits'] if s['name'] in ['train','val']];assert len(prior['splits'])==2
            (reuse/'stage1_receipt.json').write_text(json.dumps(prior,indent=2));parity=[]
            for split,start,count in [('train',50,6000),('val',6150,750)]:
                old=BASE/'preprocessed'/IDENTITY/split/'ori_imgs';(reuse/split).mkdir();(reuse/split/'ori_imgs').symlink_to(old,target_is_directory=True)
                temporary=ROOT/('jpeg_parity_'+split);temporary.mkdir()
                subprocess.run(['ffmpeg','-v','error','-threads','4','-i',str(ROOT/'material'/IDENTITY/'main.mkv'),'-an','-vf',f'trim=start_frame={start}:end_frame={start+count},setpts=PTS-STARTPTS','-qmin','1','-q:v','1','-start_number','0',str(temporary/'%d.jpg')],check=True)
                assert len(list(temporary.glob('*.jpg')))==count
                for i in range(count):
                    p=temporary/f'{i}.jpg';assert sha(p)==sha(old/p.name),('JPEG differs',split,i)
                # These regenerated copies are byte-identical to F-archived stage1 inputs.
                for p in temporary.glob('*.jpg'):p.unlink()
                parity.append(dict(split=split,frames=count,all_regenerated_JPEG_bytes_exact=True))
            atomic_json(ROOT/'unchanged_main_input_qualification.json',dict(passed=True,main_source_PCM_RGB_exact=True,JPEG_controls=parity,training_admitted=False))
            execute('stage1',ROOT/'preprocess_images_landmarks.py',['--root',str(ROOT),'--identity',IDENTITY,'--reuse-image-root',str(reuse)],ROOT/'preprocessed'/IDENTITY/'stage1_receipt.json')
            # Parser was actually rerun; every unchanged train/val output must still match original bytes.
            for split,count in [('train',6000),('val',750)]:
                for i in range(count):
                    for suffix in ['.png','_face.png']:
                        assert sha(ROOT/'preprocessed'/IDENTITY/split/'parsing'/f'{i}{suffix}')==sha(BASE/'preprocessed'/IDENTITY/split/'parsing'/f'{i}{suffix}')
            folder=ROOT/'prepared'/IDENTITY
            execute('layout',ROOT/'prepare_layout.py',['--root',str(ROOT),'--identity',IDENTITY],folder/'input_binding.json')
            failed=json.loads((ROOT/'original_failed_manifest.json').read_text());expected={r['path']:r for r in failed['files']};links=[]
            for split in ['train','val']:
                old=BASE/'prepared'/IDENTITY/split
                for p in sorted(old.iterdir()):
                    if not p.is_file() or p.is_symlink():continue
                    relative=str(p.relative_to(BASE));assert relative in expected and sha(p)==expected[relative]['sha256']
                    target=folder/split/p.name;assert not target.exists();os.link(p,target)
                    links.append(dict(path=relative,sha256=expected[relative]['sha256']))
            masks=folder/'train/flow_masks';masks.mkdir()
            import cv2,numpy as np
            from geometry import alpha_flow_mask
            for i in [0,1,2,2500,4500,5999]:
                image=cv2.imread(str(folder/'train/parsing'/f'{i}_face.png'));alpha=alpha_flow_mask(image)
                assert cv2.imwrite(str(masks/f'{i}.png'),alpha) and np.array_equal(cv2.imread(str(masks/f'{i}.png')),alpha)
            atomic_json(ROOT/'unchanged_training_state_qualification.json',dict(passed=True,files=links,training_BFM_BA_flow_bytes_reused_exact=True,
                parser_train_val_bytes_exact=True,whole_main_PCM_RGB_and_all6750_JPEG_exact=True,training_admitted=False))
            execute('new_test_tracker',ROOT/'face_tracker_frozen.py',['--root',str(ROOT),'--identity',IDENTITY,'--split','test'],folder/'test/track_params.pt',None)
            execute('geometry',ROOT/'prepare_geometry_v2.py',['--root',str(ROOT),'--identity',IDENTITY],folder/'geometry_v2/geometry_qualification.json')
            assert sha(folder/'geometry_v2/train_poses.npy')==sha(BASE/'prepared'/IDENTITY/'geometry_v2/train_poses.npy')
            state.update(passed=True,phase='geometry_qualified_pending_remaining_gates',unchanged_training_state_reused=True);atomic_json(receipt,state)
        except BaseException:
            state.update(passed=False,error=traceback.format_exc());atomic_json(receipt,state);raise
if __name__=='__main__':run()
