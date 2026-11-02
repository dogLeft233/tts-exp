"""Artifact integration checks run only after a real deployment forward exists."""
import json
import math
from pathlib import Path

def check_receipt(name):
    root=Path('/home/wjj/av-sync-eval-20261028/smoke')
    receipt=json.loads((root/(name+'_receipt.json')).read_text())
    assert receipt['status']=='PASS' and receipt['strict_load']
    assert receipt['deployment_smoke_only'] and not receipt['scientific_validity_test']
    assert receipt['cuda_peak_allocated_bytes']>0
    assert receipt['input_video_shape'][0]==receipt['input_audio_shape'][0]==1
    if name=='mtd':
        assert receipt['face_detection_batch_size']==1
        assert receipt['input_video_shape']==[1,15,48,96]
        assert math.isfinite(receipt['raw_logit'][0])
    else:
        assert receipt['audio_container_codec']=='pcm_f32le'
        assert receipt['input_video_shape']==[1,14,16,3,224,224]
        assert len(receipt['logits'][0])==21
        assert all(math.isfinite(x) for x in receipt['logits'][0])

def test_mtd_real_weight_and_media_receipt():check_receipt('mtd')
def test_synchformer_real_weight_and_media_receipt():check_receipt('synchformer')
