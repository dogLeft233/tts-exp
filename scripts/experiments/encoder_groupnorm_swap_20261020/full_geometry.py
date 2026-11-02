"""Full terminal trajectories: coordinate-invariant CKA and response-axis alignment."""
import os
os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['MKL_NUM_THREADS']='2'
from pathlib import Path
import json,hashlib
import numpy as np
from geometry import unit,centered_gram

ROOT=Path('F:/Datasets/tts-exp/encoder_groupnorm_swap_20261020_v1')
ARMS=('native','swapped','self_permuted')
MIDS=('hubert_base','wav2vec2_base')

def inner(a,b):return float(np.sum(a*b)/(np.linalg.norm(a)*np.linalg.norm(b)))
def main():
    calls=json.loads((ROOT/'forward_calls.json').read_text());features={};grams={};rawgrams={};diagnostics=[]
    for mid in MIDS:
        for split in ('train','test'):
            for arm in ARMS:
                for scale in (1.,.5,2.):
                    p=ROOT/mid/split/arm/f'scale{scale:g}/last_hidden_state.npy';x=np.load(p)
                    call=next(c for c in calls if c['encoder']==mid and c['split']==split and c['arm']==arm and c['scale']==scale and not c['repeat'])
                    assert hashlib.sha256(x.tobytes()).hexdigest()==call['last_sha256']
                    features[mid,split,arm,scale]=unit(x)
                    if scale==1.:
                        u=features[mid,split,arm,scale];grams[mid,split,arm]=centered_gram(u);rawgrams[mid,split,arm]=centered_gram(x.astype(float))
                        gram=grams[mid,split,arm];trace=float(np.trace(gram));frob=float(np.sum(gram**2))
                        diagnostics.append(dict(encoder=mid,split=split,arm=arm,frames=len(x),unit_temporal_dispersion=float(1-np.sum(u.mean(0)**2)),participation_rank=trace**2/frob))
    cross=[];within=[];response=[];time_control=[]
    permutation=np.random.default_rng(20261020).permutation(1000)
    for split in ('train','test'):
        for h_arm in ARMS:
            for w_arm in ARMS:
                a,b=features['hubert_base',split,h_arm,1.],features['wav2vec2_base',split,w_arm,1.]
                cosine=np.clip(np.sum(a*b,axis=1),-1,1)
                cross.append(dict(split=split,hubert_arm=h_arm,wav2vec2_arm=w_arm,frames=len(a),mean_cosine=float(cosine.mean()),
                    mean_angle_degrees=float(np.degrees(np.arccos(cosine)).mean()),
                    unit_direction_CKA=inner(grams['hubert_base',split,h_arm],grams['wav2vec2_base',split,w_arm]),
                    raw_linear_CKA=inner(rawgrams['hubert_base',split,h_arm],rawgrams['wav2vec2_base',split,w_arm])))
                if h_arm==w_arm:
                    g=grams['wav2vec2_base',split,w_arm];assert g.shape==(1000,1000)
                    time_control.append(dict(split=split,arm=h_arm,aligned_CKA=cross[-1]['unit_direction_CKA'],
                        fixed_time_permuted_CKA=inner(grams['hubert_base',split,h_arm],g[np.ix_(permutation,permutation)])))
        for mid in MIDS:
            base=features[mid,split,'native',1.]
            for arm in ('swapped','self_permuted'):
                value=features[mid,split,arm,1.];cos=np.clip(np.sum(base*value,axis=1),-1,1)
                within.append(dict(encoder=mid,split=split,arm=arm,frames=len(base),mean_cosine=float(cos.mean()),mean_angle_degrees=float(np.degrees(np.arccos(cos)).mean()),
                    unit_direction_CKA=inner(grams[mid,split,'native'],grams[mid,split,arm])))
                for scale in (.5,2.):
                    a=features[mid,split,'native',scale]-base;b=features[mid,split,arm,scale]-value
                    response.append(dict(encoder=mid,split=split,arm=arm,scale=scale,frames=len(base),change_global_cosine=inner(a,b),
                        native_change_rms=float(np.linalg.norm(a)/np.sqrt(len(a))),changed_change_rms=float(np.linalg.norm(b)/np.sqrt(len(b)))))
    result=dict(scope='all 1000 terminal frames per chunk; no sampling or fitted coordinate mapping',cross=cross,within=within,response_alignment=response,
        temporal_diagnostics=diagnostics,time_permutation_controls=time_control,training_updates=0)
    (ROOT/'terminal_full_geometry.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
