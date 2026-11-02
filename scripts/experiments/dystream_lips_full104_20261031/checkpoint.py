"""Full Sync-C checkpoint using exactly the predeclared speaker aggregation."""
import json
from prepare import RUN,write
from analyze import STRATA,make_pairs,summarize

def main():
    rows=json.loads((RUN/'scoring_inputs.json').read_text())['rows']
    values={r['id']:r['C'] for r in json.loads((RUN/'syncnet/scores.json').read_text())}
    pairs,missing=make_pairs(rows,values)
    assert len(values)==209 and len(pairs)==104 and not missing
    write(RUN/'syncnet_checkpoint.json',dict(status='PASS',subsets={s:summarize(rows,pairs,s)[0] for s in STRATA},
        note='Holm9 pending; equal-weight speaker means, not104 independent people; historical pilot labels refer to PC-Talk'))
    write(RUN/'predeclared_support.json',dict(status='FROZEN_BEFORE_EIGHT_SCORERS',
        classifier_eligible_ids=[r['id'] for r in rows if min(r['samples']//640,r['actual_video_frames'])>=125],
        face_support_threshold=.95,complete_speaker='Both arms of every planned utterance pair in each stratum; no imputation'))
    print('SYNCNET_FULL_CHECKPOINT_PASS',len(values),len(pairs))

if __name__=='__main__':main()
