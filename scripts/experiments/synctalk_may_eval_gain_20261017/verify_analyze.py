"""Independent float64 distance QA and fixed-video contrasts; no network calls."""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import soundfile as sf
import python_speech_features

FIELDS=['sync_c','sync_d','distance_d0','background_median']
GAINS=[-6,0,6]
LABEL={-6:'gain_m6',0:'gain_0',6:'gain_p6'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def independent_distance(visual,audio):
    padded=np.pad(audio.astype(np.float64),((15,15),(0,0)))
    out=np.empty((245,31),np.float64)
    for lag in range(31):
        delta=visual.astype(np.float64)-padded[lag:lag+245]+1e-6
        out[:,lag]=np.sqrt(np.sum(delta*delta,axis=1))
    return out


def independent_endpoints(matrix,guard):
    curve=matrix[20:-20].mean(0) if guard else matrix.mean(0)
    minimum=float(curve.min());background=float(np.median(curve))
    return dict(sync_c=background-minimum,sync_d=minimum,distance_d0=float(curve[15]),
                background_median=background,av_offset=int(15-np.argmin(curve)),curve=curve)


def main(root):
    result=json.loads((root/'results.json').read_text())
    admission=json.loads((root/'admission.json').read_text())
    assert digest(root/'protocol.json')==admission['protocol_sha256']==result['protocol_sha256']
    for name,value in admission['source_sha256'].items():assert digest(root/name)==value
    expected=set(itertools.product(GAINS,GAINS,GAINS,range(4)))
    keys=[(r['training_gain_db'],r['driver_gain_db'],r['evaluation_gain_db'],r['clip_index']) for r in result['clips']]
    assert len(keys)==108 and set(keys)==expected
    parent=json.loads((root/'inputs/scores/scores.json').read_text())
    parent_rows={(c['training_gain_db'],c['test_gain_db'],r['clip_id']):r for c in parent['cells'] for r in c['clips']}
    qa=dict(passed=False,clip_matrices=0,distance_cells=0,max_distance_error=0.,max_metric_error=0.,
            baseline_matrix_max_error=0.,baseline_metric_max_error=0.,audio_conditions=0,
            visual_inputs_unchanged=True,new_training_runs=0,new_generated_video_frames=0,new_visual_encoder_forwards=0)
    for clip in admission['clips']:
        original,sr=sf.read(root/'inputs/audio'/(clip['id']+'.wav'),dtype='int16')
        assert digest(root/'inputs/audio'/(clip['id']+'.wav'))==clip['score_audio_sha256']
        assert sr==16000 and original.shape==(160000,)
        for gain in GAINS:
            folder=root/'audio'/clip['id']/f'gain_{gain}'
            actual,rate=sf.read(folder/'evaluation.wav',dtype='int16')
            expected=np.rint(original.astype(np.float64)*10**(gain/20))
            assert expected.min()>=-32768 and expected.max()<=32767
            assert rate==sr and np.array_equal(actual,expected.astype(np.int16))
            mfcc=python_speech_features.mfcc(actual,sr).T
            assert np.array_equal(mfcc,np.load(folder/'mfcc.npy'))
            assert np.array_equal(np.load(folder/'features.npy'),np.load(folder/'repeat_features.npy'))
            qa['audio_conditions']+=1
    for row in result['clips']:
        training,driver,evaluation=row['training_gain_db'],row['driver_gain_db'],row['evaluation_gain_db']
        old=root/'inputs/scores'/row['clip_id']/f'train_{LABEL[training]}_test_{LABEL[driver]}'
        feature=old/'visual_features.npy'
        assert digest(feature)==row['visual_features_sha256']
        visual=np.load(feature)
        audio=np.load(root/'audio'/row['clip_id']/f'gain_{evaluation}'/'features.npy')
        actual=np.load(root/row['matrix_path'])
        assert digest(root/row['matrix_path'])==row['matrix_sha256']
        rebuilt=independent_distance(visual,audio)
        error=float(np.max(np.abs(rebuilt-actual)))
        assert error<1e-5
        qa['max_distance_error']=max(qa['max_distance_error'],error)
        for mode in ['guard20','full']:
            rebuilt_metrics=independent_endpoints(rebuilt,mode=='guard20')
            assert rebuilt_metrics['av_offset']==row[mode]['av_offset']
            assert row[mode]['windows']==(205 if mode=='guard20' else 245)
            for field in FIELDS:
                error=abs(rebuilt_metrics[field]-row[mode][field]);assert error<1e-5
                qa['max_metric_error']=max(qa['max_metric_error'],error)
            if evaluation==0:
                prior=parent_rows[(training,driver,row['clip_id'])]
                for field in FIELDS:
                    error=abs(prior[mode][field]-row[mode][field])
                    assert error==0
                    qa['baseline_metric_max_error']=max(qa['baseline_metric_max_error'],error)
        if evaluation==0:assert np.array_equal(actual,np.load(old/'distances.npy'))
        qa['clip_matrices']+=1;qa['distance_cells']+=actual.size
    for entry in admission['source_inputs']:
        path=Path(entry['absolute_path'])
        assert path.stat().st_size==entry['bytes'] and digest(path)==entry['sha256']
    qa['passed']=True
    assert not (root/'independent_qa.json').exists()
    dump(root/'independent_qa.json',qa)
    cells=[]
    for training,driver,evaluation in itertools.product(GAINS,GAINS,GAINS):
        rows=[r for r in result['clips'] if (r['training_gain_db'],r['driver_gain_db'],r['evaluation_gain_db'])==(training,driver,evaluation)]
        cells.append(dict(training_gain_db=training,driver_gain_db=driver,evaluation_gain_db=evaluation,
            **{mode:{field:float(np.mean([r[mode][field] for r in rows])) for field in FIELDS} for mode in ['guard20','full']},
            offsets=[r['guard20']['av_offset'] for r in rows]))
    lookup={(c['training_gain_db'],c['driver_gain_db'],c['evaluation_gain_db']):c for c in cells}
    scorer_contrasts=[];decomposition=[]
    for training,driver,gain in itertools.product(GAINS,GAINS,[-6,6]):
        changed=lookup[training,driver,gain];baseline=lookup[training,driver,0]
        per_clip=[r['guard20']['sync_c']-next(b['guard20']['sync_c'] for b in result['clips'] if
            (b['training_gain_db'],b['driver_gain_db'],b['evaluation_gain_db'],b['clip_index'])==(training,driver,0,r['clip_index']))
            for r in result['clips'] if (r['training_gain_db'],r['driver_gain_db'],r['evaluation_gain_db'])==(training,driver,gain)]
        scorer_contrasts.append(dict(training_gain_db=training,driver_gain_db=driver,evaluation_gain_db=gain,
            **{mode:{field:changed[mode][field]-baseline[mode][field] for field in FIELDS} for mode in ['guard20','full']},
            clip_delta_c=per_clip))
    for training,gain in itertools.product(GAINS,[-6,6]):
        row=dict(training_gain_db=training,matched_gain_db=gain)
        for mode in ['guard20','full']:
            values={}
            for field in FIELDS:
                base=lookup[training,0,0][mode][field]
                generation=lookup[training,gain,0][mode][field]-base
                evaluation=lookup[training,gain,gain][mode][field]-lookup[training,gain,0][mode][field]
                total=lookup[training,gain,gain][mode][field]-base
                score_at_baseline=lookup[training,0,gain][mode][field]-base
                interaction=evaluation-score_at_baseline
                assert abs(total-generation-evaluation)<1e-12
                values[field]=dict(total=total,driver_at_original_eval=generation,
                    scorer_at_changed_driver=evaluation,scorer_at_baseline_driver=score_at_baseline,
                    driver_evaluation_interaction=interaction)
            row[mode]=values
        decomposition.append(row)
    analysis=dict(cells=cells,scorer_contrasts=scorer_contrasts,matched_decomposition=decomposition,
        scorer_delta_summary={str(g):dict(mean=float(np.mean([r['guard20']['sync_c'] for r in scorer_contrasts if r['evaluation_gain_db']==g])),
            minimum=float(min(r['guard20']['sync_c'] for r in scorer_contrasts if r['evaluation_gain_db']==g)),
            maximum=float(max(r['guard20']['sync_c'] for r in scorer_contrasts if r['evaluation_gain_db']==g))) for g in [-6,6]},
        qualification='single exposed May identity, one training seed, four correlated clips; descriptive only; no population CI/p-values/causal percentages')
    dump(root/'analysis.json',analysis)
    with (root/'cells.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=['training_gain_db','driver_gain_db','evaluation_gain_db']+FIELDS)
        writer.writeheader()
        for c in cells:writer.writerow({**{k:c[k] for k in ['training_gain_db','driver_gain_db','evaluation_gain_db']},**c['guard20']})
    print(json.dumps(dict(qa=qa,scorer_delta_summary=analysis['scorer_delta_summary'],
                         matched_decomposition=[{k:r[k] for k in ['training_gain_db','matched_gain_db','guard20']} for r in decomposition])),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,required=True)
    main(parser.parse_args().root)
