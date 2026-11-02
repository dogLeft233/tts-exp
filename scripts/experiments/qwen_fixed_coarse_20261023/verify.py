"""Read stored tensors/waves again; recompute all replay and intervention claims."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen_fixed_coarse_20261023_attempt01'
sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_fixed_coarse_20261023.probe import read,save,sha


def main():
    p=read(RUN/'protocol.json');checks=[]
    for model in ['L06','L17']:
        old=p['records'][model]
        codes=np.load(old['new_codes']['path']);wave,sr=sf.read(old['path'],dtype='float32')
        assert sha(old['path'])==old['sha256'] and sha(old['new_codes']['path'])==old['new_codes']['sha256']
        d=RUN/f'{model}_control'
        for mode in ['unforced','forced']:
            got=np.load(d/f'{mode}_codes.npy');w,rate=sf.read(d/f'{mode}.wav',dtype='float32')
            assert np.array_equal(got,codes) and np.array_equal(w,wave) and rate==sr
        decoded,rate=sf.read(d/'decoded_replay.wav',dtype='float32')
        assert np.array_equal(decoded,wave) and rate==sr
        other='L17' if model=='L06' else 'L06'
        donor=p['records'][other];original=np.load(donor['new_codes']['path'])
        c=RUN/f'{model}_cross';got=np.load(c/'forced_codes.npy')
        w,rate=sf.read(c/'forced.wav',dtype='float32')
        assert np.array_equal(got[:,0],original[:,0]) and got.shape==original.shape
        assert len(w)==donor['samples'] and rate==24000 and np.isfinite(w).all()
        assert not np.array_equal(got[:,1:],original[:,1:])
        assert np.array_equal(np.load(c/'forced_sequences.npy'),np.load(donor['talker_sequences']['path']))
        checks.append(dict(completion_model=model,coarse_source=other,frames=len(got),samples=len(w),
            residual_code_changed_fraction=float(np.mean(got[:,1:]!=original[:,1:])),
            all_three_replays_exact=True,coarse_and_eos_exact=True))
    summary=read(RUN/'checks/mfa_lexical_repair02/summary.json')
    for c in summary['comparisons']:
        vals=np.abs([e['center_delta_ms'] for e in c['events']])
        if not len(vals):
            assert c['center'] is None and c['timing_gate'] is False
            continue
        for name,v in [('median_abs_ms',np.median(vals)),('p95_abs_ms',np.quantile(vals,.95)),('max_abs_ms',max(vals))]:
            assert abs(c['center'][name]-v)<1e-9
    for f in RUN.glob('*_exit.json'):
        assert read(f)['returncode']==0
    save(RUN/'independent_readback.json',dict(status='PASS',checks=checks,technical_gate=summary['status'],
        limitation='tensor/wave readback and statistic recomputation; not a second independent aligner or human listening',source_sha256=sha(__file__)))
    print(json.dumps(checks),flush=True)


if __name__=='__main__':main()
