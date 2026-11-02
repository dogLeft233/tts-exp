"""CPU frontend/operator decomposition of W2vBERT residual gain invariance.

No checkpoint weights loaded, no encoder calls; learned affine is not estimated.
"""
from pathlib import Path
import sys,json,shutil
OLD=Path('F:/Datasets/tts-exp/encoder_survey_20261018')
OUT=Path('F:/Datasets/tts-exp/encoder_normalization_ablation_20261019')
sys.path.insert(0,str(OLD/'deps'))
import numpy as np
import torch
from transformers import AutoFeatureExtractor
from prepare import sha,digest
from run import write

def main():
    torch.set_num_threads(2)
    p=json.loads((OUT/'protocol.json').read_text(encoding='utf8'))
    root=OUT/'recovery_attempt03/models/w2vbert2'
    receipt=json.loads((root/'receipt.json').read_text(encoding='utf8'))
    cfgroot=OUT/'recovery_attempt03/weights/w2vbert2'
    cfg=json.loads((cfgroot/'config.json').read_text(encoding='utf8'))
    proc=AutoFeatureExtractor.from_pretrained(cfgroot,local_files_only=True)
    eps=cfg['layer_norm_eps'];assert cfg['feature_projection_input_dim']==160
    rows=[];arrays={}
    for row in p['rows']:
        rec=next(t for t in receipt['clips'] if t['id']==row['id'])
        feats=[]
        with np.load(row['input'],allow_pickle=False) as z:
            for ci,arm in enumerate(p['arms']):
                b=proc(z[arm],sampling_rate=16000,return_tensors='pt',do_normalize_per_mel_bins=False)
                raw=b['input_features'].numpy();mask=b['attention_mask'][0].numpy().astype(bool)
                assert digest(raw)==rec['conditions'][ci]['processor']['input_features']['sha256']
                assert digest(b['attention_mask'].numpy())==rec['conditions'][ci]['processor']['attention_mask']['sha256']
                feats.append(raw[0,mask])
        ln=[torch.nn.functional.layer_norm(torch.from_numpy(x),(160,),eps=eps).numpy() for x in feats]
        for ci,db in [(1,3),(2,-3)]:
            a=feats[0].astype(float);b=feats[ci].astype(float)
            expected=db*np.log(10)/10
            delta=b-a;residual=delta-expected
            rows.append(dict(id=row['id'],db=db,frames=len(a),
                expected_log_power_shift=float(expected),mean_shift=float(delta.mean()),
                shift_residual_rms=float(np.sqrt(np.mean(residual**2))),shift_residual_max=float(abs(residual).max()),
                frontend_relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(a)),
                unweighted_layernorm_relative_l2=float(np.linalg.norm(ln[ci].astype(float)-ln[0])/np.linalg.norm(ln[0].astype(float)))))
        arrays[row['id']+'_features']=np.stack(feats)
        arrays[row['id']+'_unweighted_layernorm']=np.stack(ln)
    dest=OUT/'w2vbert_cpu_diagnostic';dest.mkdir(exist_ok=False)
    np.savez_compressed(dest/'features_and_layernorm.npz',**arrays)
    source=OLD/'deps/transformers/models/wav2vec2_bert/modeling_wav2vec2_bert.py'
    shutil.copyfile(source,dest/'actual_modeling_wav2vec2_bert.py')
    source2=OLD/'deps/transformers/models/seamless_m4t/feature_extraction_seamless_m4t.py'
    shutil.copyfile(source2,dest/'actual_feature_extraction_seamless_m4t.py')
    shutil.copyfile(__file__,dest/Path(__file__).name)
    result=dict(status='COMPLETE_CPU_FRONTEND_AND_OPERATOR_ONLY',encoder_forwards=0,weights_loaded=False,
        frontend_calls=60,layernorm_calls=60,layernorm_eps=eps,
        raw_frontend_hash_replay_matches=60,rows=rows,
        mean_frontend_relative_l2=float(np.mean([r['frontend_relative_l2'] for r in rows])),
        mean_unweighted_layernorm_relative_l2=float(np.mean([r['unweighted_layernorm_relative_l2'] for r in rows])),
        max_shift_residual=float(max(r['shift_residual_max'] for r in rows)),
        max_layernorm_relative_l2=float(max(r['unweighted_layernorm_relative_l2'] for r in rows)),
        source_sha256=sha(__file__),arrays_sha256=sha(dest/'features_and_layernorm.npz'),
        interpretation='Actual source applies LayerNorm over all160 features before learned linear projection. Uniform log-power shift is algebraically removed, independent of learned affine gamma/beta; this calculation omits learned affine and is not a new encoder forward.')
    write(dest/'result.json',result)
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))

if __name__=='__main__':main()
