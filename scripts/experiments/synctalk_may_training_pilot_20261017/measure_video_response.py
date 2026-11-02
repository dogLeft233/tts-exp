"""Descriptive RGB/float-hash response; pixel change is not perceptual quality."""
import json
import hashlib
from pathlib import Path

import cv2
import numpy as np

from common import ROOT,DATA,label,save_json,sha


def main():
    assert json.loads((ROOT/'independent_results_qa.json').read_text())['passed']
    out=ROOT/'analysis/video_response.json';assert not out.exists()
    pairs=[]
    for g in [-6,0,6]:
        for h in [-6,6]:pairs.append(((g,0),(g,h),'input_gain_within_model'))
    for g in [-6,6]:pairs.append(((0,0),(g,0),'training_gain_at_fixed_test0'))
    result={'parent_analysis_sha256':sha(ROOT/'analysis/analysis.json'),
            'source_sha256':sha(Path(__file__)),'scope':'descriptive pixel response, not perceptual synchronization','pairs':[]}
    for base,changed,kind in pairs:
        folders=[ROOT/'renders'/('train_'+label(g))/('test_'+label(h)) for g,h in [base,changed]]
        records=[json.loads((p/'render_receipt.json').read_text()) for p in folders]
        caps=[cv2.VideoCapture(str(p/'original_pcm.mkv')) for p in folders]
        row={'kind':kind,'baseline_training_gain_db':base[0],'baseline_input_gain_db':base[1],
             'changed_training_gain_db':changed[0],'changed_input_gain_db':changed[1],
             'frames':0,'rgb_changed_frames':0,'float_hash_changed_frames':0}
        full=[];mouth=[]
        for i in range(1100):
            images=[]
            for cap,record in zip(caps,records):
                ok,x=cap.read();assert ok
                rgb=cv2.cvtColor(x,cv2.COLOR_BGR2RGB)
                assert hashlib.sha256(rgb.tobytes()).hexdigest()==record['frames'][i]['rgb_sha256']
                images.append(x)
            assert records[0]['frames'][i]['source_id']==records[1]['frames'][i]['source_id']
            difference=np.abs(images[0].astype(np.float64)-images[1].astype(np.float64))
            source_id=records[0]['frames'][i]['source_id']
            lms=np.loadtxt(DATA/f'source/May/ori_imgs/{source_id}.lms')
            x0,y0=np.maximum(np.floor(lms[48:68].min(0)).astype(int)-12,0)
            x1,y1=np.minimum(np.ceil(lms[48:68].max(0)).astype(int)+13,512)
            full.append(float(difference.mean()));mouth.append(float(difference[y0:y1,x0:x1].mean()))
            row['rgb_changed_frames']+=int(np.any(difference))
            row['float_hash_changed_frames']+=int(records[0]['frames'][i]['float_sha256']!=records[1]['frames'][i]['float_sha256'])
            row['frames']+=1
        for cap in caps:
            assert not cap.read()[0];cap.release()
        row['full_image_mean_abs_change_8bit']=float(np.mean(full))
        row['mouth_bbox_mean_abs_change_8bit']=float(np.mean(mouth))
        row['per_frame_mouth_bbox_abs_change_8bit']=mouth
        result['pairs'].append(row)
    save_json(out,result)
    print(json.dumps([{k:v for k,v in r.items() if not k.startswith('per_frame')} for r in result['pairs']]),flush=True)


if __name__=='__main__':main()
