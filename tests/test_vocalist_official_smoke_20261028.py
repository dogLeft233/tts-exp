"""Integration receipt from original VocaLiST real media GPU inference."""
import json
from pathlib import Path


def test_original_weight_and_official_dataset_real_receipt():
    r=json.loads(Path('/home/wjj/av-sync-eval-20261028/smoke/vocalist_receipt.json').read_text())
    assert r['status']=='PASS' and r['strict_load'] and r['author_dataset_verbatim']
    assert r['parameters']==80106561
    assert r['input_video_shape']==[1,15,48,96] and r['input_audio_shape']==[1,1,80,16]
    assert 0<=r['video_min']<=r['video_max']<=1
    assert r['deployment_smoke_only'] and not r['scientific_validity_test']
    assert r['cuda_peak_allocated_bytes']>0
