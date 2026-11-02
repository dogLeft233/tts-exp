"""Post-hoc natural-video offset diagnostic; never replaces primary results."""
import json
import os
import sys
from pathlib import Path
import cv2
import numpy as np
import torch

from common import ROOT,DATA,sha,save_json
from score_test import SYNC,crops,summarize
from validate_results import independent_matrix,metric


def main():
    assert json.loads((ROOT/'independent_results_qa.json').read_text())['passed']
    output=ROOT/'ground_truth_reference_attempt01';output.mkdir(exist_ok=False)
    torch.set_num_threads(2);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    os.chdir(SYNC);sys.path.insert(0,str(SYNC))
    from SyncNetModel import S
    from SyncNetInstance import calc_pdist
    model=S(num_layers_in_fc_layers=1024).cuda().eval()
    model.load_state_dict(torch.load(SYNC/'data/syncnet_v2.model',map_location='cpu',weights_only=True),strict=True)
    primary=json.loads((ROOT/'scores/scores.json').read_text())
    assert sha(SYNC/'data/syncnet_v2.model')==primary['syncnet_weight_sha256']
    manifest=json.loads((DATA/'manifest.json').read_text())
    test=next(s for s in manifest['splits'] if s['name']=='test')
    track=np.load(ROOT/'scores/shared_gt_track.npy')
    result={'scope':'POST_HOC natural-video offset diagnostic, after primary analysis; no primary result changes',
            'source_sha256':sha(Path(__file__)),'primary_scores_sha256':sha(ROOT/'scores/scores.json'),
            'roi_and_audio_features':'identical to nine primary cells','clips':[],'max_independent_distance_error':0.}
    for clip in [c for c in manifest['eval_clips'] if c['split']=='test']:
        folder=output/clip['id'];folder.mkdir()
        images=np.stack([cv2.imread(str(DATA/f"source/May/gt_imgs/{i}.jpg")) for i in
                         range(clip['source_start_frame'],clip['source_start_frame']+250)])
        start=clip['source_start_frame']-test['supervision_start_frame']
        cropped=crops(images,track[start:start+250]);np.save(folder/'crops.npy',cropped)
        cv2.imwrite(str(folder/'middle_crop.png'),cropped[125])
        video=torch.from_numpy(cropped).permute(3,0,1,2).unsqueeze(0).float();features=[]
        with torch.no_grad():
            for i in range(0,245,20):
                batch=torch.cat([video[:,:,j:j+5] for j in range(i,min(i+20,245))])
                features.append(model.forward_lip(batch.cuda()).cpu())
        vf=torch.cat(features);audio=np.load(ROOT/'scores'/clip['id']/'original_audio_features.npy')
        matrix=torch.stack(calc_pdist(vf,torch.from_numpy(audio),vshift=15))
        np.save(folder/'visual_features.npy',vf.numpy());np.save(folder/'distances.npy',matrix.numpy())
        rebuilt=independent_matrix(vf.numpy(),audio);error=float(np.max(np.abs(rebuilt-matrix.numpy())))
        assert error<1e-5;result['max_independent_distance_error']=max(result['max_independent_distance_error'],error)
        row={'clip_id':clip['id'],'guard20':summarize(matrix,np.arange(20,225)),
             'full':summarize(matrix,np.arange(245)),'original_score_audio_sha256':clip['score_audio_sha256']}
        for guard in [True,False]:
            values=metric(rebuilt,guard);scope='guard20' if guard else 'full'
            for key,value in values.items():assert abs(row[scope][key]-value)<1e-5
        result['clips'].append(row)
    result['mean_guard20']={k:float(np.mean([r['guard20'][k] for r in result['clips']])) for k in
                            ['sync_c','sync_d','distance_d0','background_median']}
    result['passed']=True;save_json(output/'reference.json',result)
    print(json.dumps({'mean_guard20':result['mean_guard20'],'offsets':[r['guard20']['av_offset'] for r in result['clips']],
                      'independent_error':result['max_independent_distance_error']}),flush=True)


if __name__=='__main__':main()
