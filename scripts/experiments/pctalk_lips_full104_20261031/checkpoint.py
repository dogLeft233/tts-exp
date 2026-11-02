"""Summarize Sync-C by speaker, freeze full support, and verify unchanged pilot science."""
import json
from pathlib import Path
from prepare import RUN,sha,write
from analyze import STRATA,make_pairs,summarize

def main():
    inputs=json.loads((RUN/'scoring_inputs.json').read_text());rows=inputs['rows']
    values={r['id']:r['C'] for r in json.loads((RUN/'syncnet/scores.json').read_text())}
    pairs,missing=make_pairs(rows,values)
    assert len(values)==209 and len(pairs)==104 and not missing
    summary={name:summarize(rows,pairs,name)[0] for name in STRATA}
    write(RUN/'syncnet_checkpoint.json',dict(status='PASS',subsets=summary,note='Holm9 pending; speaker aggregation, not104 independent people'))
    eligible=[r['id'] for r in rows if min(r['samples']//640,r['actual_video_frames'])>=125]
    write(RUN/'predeclared_support.json',dict(status='FROZEN_BEFORE_NEW_EIGHT_SCORERS',classifier_eligible_ids=eligible,
        face_support_threshold=.95,complete_speaker='Both arms of all planned utterance pairs per speaker/stratum; no imputation'))
    receipt=json.loads((RUN/'reuse_manifest.json').read_text())
    for rec in receipt['files']:assert sha(RUN/rec['path'])==rec['sha256'],rec['path']
    write(RUN/'reuse_preservation_before_eight.json',dict(status='PASS',preserved_files=len(receipt['files']),
        reused_media=42,reused_syncnet=42,new_syncnet=167,metadata='Original copied scientific files untouched'))
    print('SYNCNET_FULL_CHECKPOINT_PASS',len(values),len(pairs))

if __name__=='__main__':main()
