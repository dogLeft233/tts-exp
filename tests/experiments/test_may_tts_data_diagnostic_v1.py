from pathlib import Path
import json,hashlib
import numpy as np
O=Path('F:/Datasets/tts-exp/may_tts_purified_diagnostic_20261020_v1')
def test_independent_AVE_recompute_matches_original_training_artifact():
 actual=np.load(O/'recomputed_ave.npy');original=np.load('F:/Datasets/tts-exp/may_tts_purified_training_20261020_v1/qualification_readback/features/train_padded.npy')
 assert np.array_equal(actual,original) and np.isfinite(actual).all()
 q=json.loads((O/'loader_ave_receipt.json').read_text());assert q['all_rows']==3849 and q['actual_collate_rows']>=100 and q['optimizer_updates']==0
 assert hashlib.sha256((O/'recomputed_ave.npy').read_bytes()).hexdigest()==q['recomputed_sha256']
def test_actual_visual_label_freeze_with_distinct_audio_contexts():
 p=Path('F:/Datasets/tts-exp/may_tts_purified_data_20261020_v1');frames=json.loads((p/'prepared_dataset_v1/tts_purified/train/transforms_train.json').read_text())['frames'];by_audio={x['aud_id']:x['img_id'] for x in frames}
 assert {by_audio[i] for i in range(2641,2686)}=={2944}
 d=p/'purification_readback/results/chunk_008';visual=np.load(d/'after_visual.npy')[9:50];audio=np.load(d/'audio_features.npy')[9:50]
 assert len(visual)==41 and len(np.unique(visual,axis=0))==1 and len(np.unique(audio,axis=0))==41
def test_raw_metadata_offsets_are_not_final_audio_clock():
 x=json.loads((O/'cpu_data_audit.json').read_text());chunks=x['chunks'];start=0
 for c in chunks:assert c['global_start']==start;start+=c['frames']
 assert start==3914 and any(c['metadata_start_stale'] for c in chunks)
 assert x['joined_wave_exact'] and x['audio_ids_contiguous'] and x['mapping_exact']
