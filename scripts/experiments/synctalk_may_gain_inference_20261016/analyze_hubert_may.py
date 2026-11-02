"""Read-only recheck of full May HuBERT arrays and the matching 12-second rows."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def paired_slice(hidden,start_video_frame,video_frames):
    start=2*start_video_frame;end=start+2*video_frames
    assert hidden.ndim==2 and start>=0 and end<=len(hidden), 'Unsupported source-clock slice'
    return hidden[start:end]


def compare(x,b):
    assert x.shape==b.shape
    x=x.astype(np.float64);b=b.astype(np.float64)
    return dict(max_abs=float(np.max(np.abs(x-b))),
                relative_l2=float(np.linalg.norm(x-b)/np.linalg.norm(b)),
                cosine=float(np.sum(x*b)/(np.linalg.norm(x)*np.linalg.norm(b))))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    r=a.root/'attempt01';s=json.loads((r/'statistics.json').read_text())
    assert s['strict_load_missing']==s['strict_load_unexpected']==[]
    assert s['effective_posconv_weight_max_error']==0 and not s['model_training']
    assert max(x['max_abs_diff'] for x in s['repeats'])==0
    formula=[x for x in s['comparisons'] if x['stage']=='processor_epsilon_formula']
    assert len(formula)==3 and max(x['max_abs_diff'] for x in formula)==0
    arrays={}
    for row in s['stages']:
        x=np.load(row['path'],allow_pickle=False)
        assert list(x.shape)==row['shape'] and str(x.dtype)==row['dtype']
        assert hashlib.sha256(x.tobytes()).hexdigest()==row['sha256']
        assert np.isfinite(x).all()
        v=x.astype(np.float64)
        assert np.isclose(np.sqrt(np.mean(v*v)),row['rms'],rtol=1e-12)
        assert float(np.max(np.abs(v)))==row['peak']
        arrays[row['gain_db'],row['stage']]=x
    comparisons=[]
    for db in [-6,0,6]:
        h=arrays[db,'hidden'];b=arrays[0,'hidden']
        clip=paired_slice(h,5520,300);baseline_clip=paired_slice(b,5520,300)
        cache=arrays[db,'paired_cache']
        assert np.array_equal(clip,cache[5520:5820].reshape(600,1024))
        np.save(a.root/f'hidden12s_gain{db:+d}.npy',clip)
        comparisons.append(dict(gain_db=db,full_shape=list(h.shape),
                                processor=compare(arrays[db,'processor'],arrays[0,'processor']),
                                full_hidden=compare(h,b),may12s_hidden=compare(clip,baseline_clip)))
    result=dict(status='PASS',raw_arrays_verified=len(arrays),repeats_exact=True,
                official_processor_formula_exact=True,matched_rows=[11040,11640],
                corresponding_video_frames=[5520,5820],comparisons=comparisons,
                strict_weights_verified=True,training_runs=0,rendered_videos=0,
                source_statistics_sha256=hashlib.sha256((r/'statistics.json').read_bytes()).hexdigest())
    a.output.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
