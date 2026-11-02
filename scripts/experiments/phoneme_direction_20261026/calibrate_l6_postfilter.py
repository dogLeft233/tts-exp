"""Full 100-speaker fit cohort, five speaker-held-out spectral heads.

This is a candidate reconstruction repair, not an admitted protocol change.
"""
import argparse
import sys
import numpy as np
import soundfile as sf
from calibrate import ROOT, RUN as PARENT, read, write, sha
from separate_encoder_decoder import RUN, OLD, bands


def prepare():
    from diagnose_reconstruction import REGULAR
    rows = [x for x in read(PARENT / 'input_manifest.json')['fit'] if x['arm'] == 'N']
    assert len(rows) == 300 and len({x['speaker_id'] for x in rows}) == 100
    write(RUN / 'full_postfilter_contract.json', {
        'rows': rows, 'fold': 'sorted 100 speaker index modulo five; all 3 sentences together',
        'ridge_alpha': 100., 'folds': 5, 'feature_layer': 6,
        'processing': 'same fixed bounded L6 postfilter as 23-source diagnostic; no parameter search',
        'checkpoint_sha256': sha(REGULAR), 'parent_manifest_sha256': sha(PARENT / 'input_manifest.json'),
        'feature_script_sha256': sha(__file__),
        'postfilter_script_sha256': sha(ROOT / 'scripts/experiments/phoneme_direction_20261026/probe_l6_postfilter.py'),
        'candidate_only': True, 'all_floors_unchanged': True, 'samples_removed': 0,
        'development_allowed': False, 'confirmation_allowed': False})


def extract(shard):
    import torch
    from probe_weak_segments import conditioning, encode
    if shard not in range(3): raise ValueError('Three fixed shards required')
    rows = read(RUN / 'full_postfilter_contract.json')['rows']
    torch.set_num_threads(4); torch.manual_seed(20261026); torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(ROOT / 'scripts'))
    from wavlm_knn_vc_adapter import WavLMKNNVCAdapter
    adapter = WavLMKNNVCAdapter.load_pretrained(device='cpu', source=ROOT / 'third_party/knn-vc')
    available = {x['id']: x for x in read(RUN / 'separation_features.json')['rows']}
    outputs = []
    for x in rows[shard::3]:
        assert x['split'] == 'fit' and sha(x['canonical_audio_path']) == x['canonical_audio_sha256']
        a, sr = sf.read(x['canonical_audio_path'], dtype='float32'); assert sr == 16000
        b, gain = conditioning(a, 'peak95')
        if x['id'] in available:
            old = available[x['id']]; assert sha(old['path']) == old['sha256']
            z = np.load(old['path']); h = z['l6']; target = z['target']
        else:
            h = encode(adapter, b, 'peak95').numpy(); target = bands(b)
        p = RUN / 'full_l6_features' / (x['id'] + '.npz'); p.parent.mkdir(exist_ok=True)
        if p.exists(): raise FileExistsError(p)
        np.savez_compressed(p, l6=h, target=target)
        outputs.append(dict(id=x['id'], speaker_id=x['speaker_id'], path=str(p), sha256=sha(p),
                            peak_gain=float(gain[0]), source_sha256=x['canonical_audio_sha256']))
        print('FULL_FEATURE', x['id'], flush=True)
    write(RUN / f'full_l6_shard{shard}_complete.json', {'rows': outputs})


def infer():
    from probe_l6_postfilter import fit_head, predict_head, postfilter
    rows = []
    for shard in range(3): rows.extend(read(RUN / f'full_l6_shard{shard}_complete.json')['rows'])
    contract = read(RUN / 'full_postfilter_contract.json')
    assert len(rows) == 300 and {x['id'] for x in rows} == {x['id'] for x in contract['rows']}
    speakers = sorted({x['speaker_id'] for x in rows})
    prior = {x['id']: x for x in read(OLD / 'full_peak95_quality.json')['rows']}
    for x in rows:
        assert sha(x['path']) == x['sha256']
        x['arrays'] = dict(np.load(x['path'])); x['fold'] = speakers.index(x['speaker_id']) % 5
    outputs = []
    for fold in range(5):
        train = [x for x in rows if x['fold'] != fold]; test = [x for x in rows if x['fold'] == fold]
        assert len(test) == 60 and len(train) == 240
        assert not {x['speaker_id'] for x in train} & {x['speaker_id'] for x in test}
        model = fit_head([x['arrays']['l6'] for x in train], [x['arrays']['target'] for x in train])
        head = RUN / f'full_head_fold{fold}.npz'
        if head.exists(): raise FileExistsError(head)
        np.savez_compressed(head, **model)
        write(RUN / f'full_head_fold{fold}_contract.json', {
            'train_speakers': sorted({x['speaker_id'] for x in train}),
            'test_speakers': sorted({x['speaker_id'] for x in test}),
            'head_sha256': sha(head), 'full_contract_sha256': sha(RUN / 'full_postfilter_contract.json')})
        for x in test:
            old = prior[x['id']]
            assert sha(old['source']) == old['source_sha256'] == x['source_sha256']
            assert sha(old['output']) == old['output_sha256']
            a, sr = sf.read(old['source']); b, sb = sf.read(old['output']); assert sr == sb == 16000
            pred = predict_head(x['arrays']['l6'], model)
            c = postfilter(b, pred, x['peak_gain'])
            c *= np.sqrt(np.mean(a * a) / np.mean(c * c))
            dest = RUN / 'full_postfilter_audio' / (x['id'] + '.wav'); dest.parent.mkdir(exist_ok=True)
            if dest.exists(): raise FileExistsError(dest)
            sf.write(dest, c, sr, subtype='FLOAT')
            outputs.append(dict(id=x['id'], speaker_id=x['speaker_id'], source=old['source'],
                source_sha256=old['source_sha256'], output=str(dest), output_sha256=sha(dest),
                head=str(head), head_sha256=sha(head), fold=fold))
        print('FULL_FILTER_FOLD', fold, flush=True)
    write(RUN / 'full_postfilter_outputs.json', {'rows': outputs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'extract', 'infer'])
    parser.add_argument('--shard', type=int)
    args = parser.parse_args()
    if args.stage == 'extract': extract(args.shard)
    else: globals()[args.stage]()
