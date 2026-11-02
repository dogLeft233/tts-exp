"""Bind exact split-local images to whole-recording AVE and qualified camera poses."""
import argparse,hashlib,json,os,sys
from pathlib import Path
import numpy as np,torch
from flow_stream import sha

def label(gain):return {-6:'gain_m6',0:'gain_0',6:'gain_p6'}[gain]
def run(root,identity,persisted_supervision=False):
    prepared=root/'prepared'/identity;geometry=prepared/'geometry_v2'
    qualification=json.loads((geometry/'geometry_qualification.json').read_text());assert qualification['passed']
    feature_root=root/'features'/identity
    features=json.loads((feature_root/'ave_qualification.json').read_text());assert features['passed']
    manifest=json.loads((root/'material'/identity/'split_manifest.json').read_text())
    sys.path.insert(0,str(root/'official/data_utils/face_tracking'))
    from util import euler2rot
    bundle=torch.load(prepared/'train/bundle_adjustment.pt',weights_only=False)
    result=dict(identity=identity,geometry_qualification_sha256=sha(geometry/'geometry_qualification.json'),arms=[],local_supervision_complete=True,training_admitted=False)
    inventory=None
    if persisted_supervision:
        manifest_path=root/'snapshots'/identity/'manifest.json'
        proof=json.loads((root/'snapshots'/identity/'F_archive_verification.json').read_text())
        assert proof['passed'] and proof['all_archived_member_bytes_verified'] and sha(manifest_path)==proof['manifest_sha256']
        inventory={x['path']:x for x in json.loads(manifest_path.read_text())['files']}
    output=root/'datasets'/identity;output.mkdir(parents=True,exist_ok=False)
    for split in manifest['splits']:
        name=split['name'];data=prepared/name;count=split['frames'];start=split['source_start_frame']
        assert count==dict(train=6000,val=750,test=1500)[name]
        for kind,suffix in [('ori_imgs','.jpg'),('gt_imgs','.jpg'),('torso_imgs','.png')]:
            assert (data/kind).is_dir(),f'{kind} missing: restore the verified supervision archive before dataset binding'
            actual=list((data/kind).glob('*'+suffix))
            if len(actual)!=count:
                assert persisted_supervision and kind!='ori_imgs','full supervision missing; logical binding requires explicit persisted-supervision mode'
                result['local_supervision_complete']=False
                expected={str((data/kind/f'{i}{suffix}').relative_to(root)) for i in range(count)}
                assert expected.issubset(inventory)
                for p in actual:
                    entry=inventory[str(p.relative_to(root))];assert p.stat().st_size==entry['bytes'] and sha(p)==entry['sha256']
        if name=='train':rot=euler2rot(bundle['euler']);trans=bundle['trans']
        else:
            saved=np.load(geometry/name/'pose_arrays.npz')
            rot=torch.tensor(saved['rotation'],dtype=torch.float32);trans=torch.tensor(saved['translation'],dtype=torch.float32)
        # Pinned process.py save_transforms uses these exact float32 Torch operations.
        inverse=rot.permute(0,2,1);inverse_t=-torch.bmm(inverse,trans.unsqueeze(2))
        poses=torch.eye(4,dtype=torch.float32)[None].repeat(count,1,1)
        poses[:,:3,:3]=inverse;poses[:,:3,3]=inverse_t[:,:,0]
        authority=np.load(geometry/'train_poses.npy' if name=='train' else geometry/name/'poses.npy')
        assert np.allclose(poses.numpy(),authority,rtol=0,atol=1e-7)
        transform=dict(focal_len=float(bundle['focal'][0]),cx=256.,cy=256.,h=512,w=512,frames=[])
        ids=np.arange(start,start+count,dtype=np.int64)
        for i in range(count):transform['frames'].append(dict(img_id=i,aud_id=int(ids[i]),transform_matrix=poses[i].tolist()))
        common=output/name;common.mkdir()
        transform_path=common/f'transforms_{name}.json';transform_path.write_text(json.dumps(transform,separators=(',',':')))
        for gain in [-6,0,6]:
            arm=common/label(gain);arm.mkdir()
            for kind in ['ori_imgs','gt_imgs','torso_imgs','parsing']:(arm/kind).symlink_to(data/kind,target_is_directory=True)
            for filename in ['bs.npy','bc.jpg']:os.link(data/filename,arm/filename)
            os.link(transform_path,arm/transform_path.name)
            wave=root/'material'/identity/f'{split["recording"]}_driver_{gain:+d}.wav'
            (arm/'aud.wav').symlink_to(wave)
            feature=next(x for x in features['rows'] if x['recording']==split['recording'] and x['gain_db']==gain)
            cache=feature_root/feature['features_path'];assert sha(cache)==feature['features_sha256']
            padded=np.load(cache,allow_pickle=False);aligned=padded[ids,None,:]
            assert aligned.shape==(count,1,512) and int(ids[-1])+4<len(padded)
            (arm/'whole_ave.npy').symlink_to(cache)
            row=dict(identity=identity,split=name,gain_db=gain,frames=count,recording=split['recording'],
                source_ids=ids.tolist(),whole_ave_sha256=sha(cache),transform_sha256=sha(transform_path),wave_sha256=sha(wave),
                aligned_audio_tensor_sha256=hashlib.sha256(aligned.tobytes()).hexdigest(),
                camera_conversion='pinned official float32 Torch transpose and bmm',whole_recording_attention_required=True,
                background_sha256=sha(data/'bc.jpg'),training_admitted=False)
            (arm/'binding.json').write_text(json.dumps(row,indent=2));result['arms'].append({k:v for k,v in row.items() if k!='source_ids'})
    result['logical_binding_only']=not result['local_supervision_complete']
    result['passed']=True;(output/'binding_receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True);p.add_argument('--persisted-supervision',action='store_true')
    a=p.parse_args();run(a.root,a.identity,a.persisted_supervision)
