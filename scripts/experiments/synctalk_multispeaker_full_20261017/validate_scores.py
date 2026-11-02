"""Rebuild every distance and endpoint independently with float64 NumPy arithmetic."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
from cohort_binding import identity_root,verify_member_pins

def distance(visual,audio):
    assert visual.shape==audio.shape==(245,1024)
    padded=np.pad(audio.astype(np.float64),((15,15),(0,0)))
    return np.stack([np.linalg.norm(visual[i].astype(np.float64)[None]-padded[i:i+31]+1e-6,axis=1) for i in range(245)])

def endpoint(matrix,indexes):
    curve=matrix[indexes].mean(0);index=int(np.argmin(curve));minimum=float(curve[index]);median=float(np.median(curve))
    return dict(sync_c=median-minimum,sync_d=minimum,distance_d0=float(curve[15]),background_median=median,av_offset=15-index)

def run(identity,seed):
    inputs=identity_root(identity,ROOT,require_admission=True);verify_member_pins(ROOT,identity)
    folder=ROOT/'scores'/identity/f'seed{seed}';scores=json.loads((folder/'scores.json').read_text());assert scores['passed'] and len(scores['cells'])==27
    combinations={(c['training_gain_db'],c['driver_gain_db'],c['evaluation_gain_db']) for c in scores['cells']}
    assert combinations=={(a,b,c) for a in [-6,0,6] for b in [-6,0,6] for c in [-6,0,6]}
    max_matrix=0.;max_endpoint=0.;checked=0
    for cell in scores['cells']:
        a,b,c=[cell[k] for k in ['training_gain_db','driver_gain_db','evaluation_gain_db']]
        assert [row['clip_id'] for row in cell['clips']]==[f'test_{i:02d}' for i in range(6)]
        for row in cell['clips']:
            clip=row['clip_id'];base=folder/f'train_{a:+d}_driver_{b:+d}'/clip
            vf=np.load(base/'visual_features.npy',allow_pickle=False)
            af=np.load(inputs/'score_reference'/identity/clip/f'audio_features_{c:+d}dB.npy',allow_pickle=False)
            path=base/f'distances_score_{c:+d}.npy';assert hashlib.sha256(path.read_bytes()).hexdigest()==row['distances_sha256']
            stored=np.load(path,allow_pickle=False);rebuilt=distance(vf,af)
            error=float(np.max(np.abs(stored-rebuilt)));assert error<3e-5;max_matrix=max(max_matrix,error)
            for support,indexes in [('guard20',np.arange(20,225)),('full',np.arange(245))]:
                values=endpoint(rebuilt,indexes);assert values['av_offset']==row[support]['av_offset']
                for key in ['sync_c','sync_d','distance_d0','background_median']:
                    error=abs(values[key]-row[support][key]);assert error<3e-5;max_endpoint=max(max_endpoint,error)
            checked+=1
        for support in ['guard20','full']:
            for key in ['sync_c','sync_d','distance_d0','background_median']:
                assert abs(np.mean([r[support][key] for r in cell['clips']])-cell['mean_'+support][key])<1e-12
    result=dict(identity=identity,seed=seed,passed=True,cells=27,clip_matrices=checked,
        independent_implementation='NumPy float64 explicit zero-padding, Euclidean norm with original pairwise_distance eps1e-6; manual median/minimum',
        matrix_max_abs_error=max_matrix,endpoint_max_abs_error=max_endpoint,tolerance=3e-5,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (folder/'independent_score_qualification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.identity,a.seed)
