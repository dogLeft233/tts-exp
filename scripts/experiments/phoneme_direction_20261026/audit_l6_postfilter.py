"""Read back every candidate WAV under the unchanged complete-signal gates."""
import numpy as np
from calibrate import ROOT, read, write, sha
from audit_conditioned_fit import measure
from separate_encoder_decoder import RUN, OLD


def main():
    contract = read(RUN / 'full_postfilter_contract.json')
    outputs = read(RUN / 'full_postfilter_outputs.json')['rows']
    expected = {x['id'] for x in contract['rows']}
    assert len(outputs) == 300 and {x['id'] for x in outputs} == expected
    assert len({x['speaker_id'] for x in outputs}) == 100
    assert all(x['split'] == 'fit' for x in contract['rows'])
    heads = {}
    for fold in range(5):
        h = read(RUN / f'full_head_fold{fold}_contract.json')
        assert len(h['train_speakers']) == 80 and len(h['test_speakers']) == 20
        assert not set(h['train_speakers']) & set(h['test_speakers'])
        assert h['full_contract_sha256'] == sha(RUN / 'full_postfilter_contract.json')
        heads[fold] = h
    for x in outputs:
        h = heads[x['fold']]
        assert x['speaker_id'] in h['test_speakers']
        assert sha(x['head']) == x['head_sha256'] == h['head_sha256']
    measured = [measure(x) for x in outputs]
    old = {x['id']: x for x in read(OLD / 'full_peak95_quality.json')['rows']}
    assert set(old) == expected
    failed = [x for x in measured if x['failures']]
    median = float(np.median([x['stoi'] for x in measured]))
    result = dict(status='PASS_BASIC_RECONSTRUCTION' if not failed and median >= .85 else 'HOLD_RECONSTRUCTION',
        sources=300, speakers=100, stoi_median=median, stoi_min=min(x['stoi'] for x in measured),
        failed=failed, rows=measured, samples_removed=0, all_floors_unchanged=True,
        new_failures=[x['id'] for x in failed if not old[x['id']]['failures']],
        paired_median_stoi_delta=float(np.median([x['stoi'] - old[x['id']]['stoi'] for x in measured])),
        candidate_only=True, learned_spectral_head=True, speaker_held_out=True,
        numeric_protocol_frozen=False, development_started=False, confirmation_started=False,
        contract_sha256=sha(RUN / 'full_postfilter_contract.json'), script_sha256=sha(__file__))
    write(RUN / 'full_postfilter_quality.json', result)
    print({k: result[k] for k in ['status', 'sources', 'speakers', 'stoi_median', 'stoi_min', 'new_failures']}, flush=True)
    print('FAILED', [(x['id'], x['stoi'], x['failures']) for x in failed], flush=True)
    for x in measured:
        if any(s in x['id'] for s in ['S0168W0257', 'S0356W0277', 'S0039W0124']): print(x['id'], x['stoi'], flush=True)


if __name__ == '__main__': main()
