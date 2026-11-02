"""Versioned single-source replacement; original cohorts and gates stay immutable."""
import argparse
from collections import Counter
from pathlib import Path
import sys

from calibrate import ROOT, RUN as ORIGINAL, read, write, sha

RUN = ROOT / 'runs/phoneme_direction_20261026_repair04'
PREVIOUS = ROOT / 'runs/phoneme_direction_20261026_repair03'
PEAK = ROOT / 'runs/phoneme_direction_20261026_repair02'
OLD_ID = 'BAC009S0168W0257'
NEW_ID = 'BAC009S0168W0263'


def replace_rows(original, replacement):
    """Replace exactly four fit units, preserving all remaining identities."""
    removed = [x for x in original if x['sample_id'] == OLD_ID]
    expected = Counter({('N', 0): 1, ('L17', 1): 1, ('L17', 2): 1, ('L17', 3): 1})
    for group, sid in [(removed, OLD_ID), (replacement, NEW_ID)]:
        if Counter((x['arm'], int(x['draw'])) for x in group) != expected:
            raise ValueError('Incomplete replacement source')
        if any(x['sample_id'] != sid or x['split'] != 'fit' or x['speaker_id'] != 'S0168' for x in group):
            raise ValueError('Replacement must remain in the same fit speaker')
        if len({x['text'] for x in group}) != 1:
            raise ValueError('Inconsistent source transcript')
    if any(x['sample_id'] == NEW_ID for x in original):
        raise ValueError('Replacement already used')
    result = sorted([x for x in original if x['sample_id'] != OLD_ID] + replacement, key=lambda x: x['id'])
    if len(result) != 1200 or len({x['id'] for x in result}) != 1200:
        raise ValueError('Expected exactly 1200 unique fit units')
    counts = Counter(x['speaker_id'] for x in result if x['arm'] == 'N')
    if len(counts) != 100 or set(counts.values()) != {3}:
        raise ValueError('Expected 100 speakers with three natural sources each')
    return result


def manifest():
    selected = read(RUN / 'selected_replacement.json')
    assert selected['sample_id'] == NEW_ID and selected['score_seen'] is False
    replacement = []
    for arm in ['N', 'L17']:
        cell = RUN / 'replacement_media' / arm
        for x in read(cell / 'audio_manifest.json')['records']:
            path = Path(x['path'])
            if not path.is_absolute(): path = ROOT / path
            assert sha(path) == x['sha256']
            draw = 0 if arm == 'N' else int(x['role'][-1])
            replacement.append(dict(id=f'{NEW_ID}_{arm}_{draw}', sample_id=NEW_ID,
                speaker_id='S0168', arm=arm, draw=draw, text=selected['text'], split='fit',
                canonical_audio_path=str(path), canonical_audio_sha256=x['sha256'], samples=x['samples']))
    old = read(ORIGINAL / 'input_manifest.json')['fit']
    rows = replace_rows(old, replacement)
    for x in rows: assert x['split'] == 'fit' and sha(x['canonical_audio_path']) == x['canonical_audio_sha256']
    write(RUN / 'input_manifest_v2.json', dict(fit=rows, replacement=replacement,
        original_manifest_sha256=sha(ORIGINAL / 'input_manifest.json'),
        selection_sha256=sha(RUN / 'selected_replacement.json'),
        original_failure_preserved=str(PREVIOUS / 'full_postfilter_quality.json'),
        unchanged_units=1196, replaced_sources=1, fit_speakers=100, fit_sources=300,
        development_allowed=False, confirmation_allowed=False))


def reconstruct():
    import numpy as np
    import soundfile as sf
    import torch
    from diagnose_reconstruction import REGULAR
    from probe_weak_segments import conditioning, encode
    from separate_encoder_decoder import bands
    source = RUN / 'replacement_media/N/audio' / NEW_ID / 'N_16k_pcm16.wav'
    assert read(RUN / 'selected_replacement.json')['sample_id'] == NEW_ID
    torch.set_num_threads(4); torch.manual_seed(20261026); torch.use_deterministic_algorithms(True)
    sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'third_party/knn-vc'))
    from wavlm_knn_vc_adapter import WavLMKNNVCAdapter
    from hifigan.models import Generator
    from hifigan.utils import AttrDict
    write(RUN / 'replacement_reconstruction_contract.json', dict(source=str(source),
        source_sha256=sha(source), checkpoint_sha256=sha(REGULAR), script_sha256=sha(__file__),
        peak95=True, feature_layer=6, candidate_only=True, floors_unchanged=True))
    adapter = WavLMKNNVCAdapter.load_pretrained(device='cpu', source=ROOT / 'third_party/knn-vc')
    generator = Generator(AttrDict(read(ROOT / 'third_party/knn-vc/hifigan/config_v1_wavlm.json')))
    generator.load_state_dict(torch.load(REGULAR, map_location='cpu', weights_only=False)['generator'])
    generator.eval(); generator.remove_weight_norm()
    a, sr = sf.read(source, dtype='float32'); assert sr == 16000
    b, gain = conditioning(a, 'peak95'); h = encode(adapter, b, 'peak95')
    with torch.inference_mode(): raw = generator(h[None]).flatten().numpy()[:len(a)].astype('float64')
    raw /= gain; raw *= np.sqrt(np.mean(a.astype('float64') ** 2) / np.mean(raw ** 2))
    output = RUN / 'replacement_peak95.wav'; assert not output.exists()
    sf.write(output, raw, sr, subtype='FLOAT')
    features = RUN / 'replacement_l6.npz'; assert not features.exists()
    np.savez_compressed(features, l6=h.numpy(), target=bands(b))
    write(RUN / 'replacement_reconstruction.json', dict(id=NEW_ID + '_N_0', speaker_id='S0168',
        source=str(source), source_sha256=sha(source), output=str(output), output_sha256=sha(output),
        path=str(features), sha256=sha(features), peak_gain=float(gain[0]), samples=len(a)))
    print('REPLACEMENT_RECONSTRUCTED', flush=True)


def full_filter():
    import numpy as np
    import soundfile as sf
    from probe_l6_postfilter import fit_head, predict_head, postfilter
    rows = [x for x in read(RUN / 'input_manifest_v2.json')['fit'] if x['arm'] == 'N']
    new = read(RUN / 'replacement_reconstruction.json')
    previous = []
    for i in range(3): previous.extend(read(PREVIOUS / f'full_l6_shard{i}_complete.json')['rows'])
    previous = {x['id']: x for x in previous}
    peak = {x['id']: x for x in read(PEAK / 'full_peak95_quality.json')['rows']}
    speakers = sorted({x['speaker_id'] for x in rows})
    write(RUN / 'full_reconstruction_contract_v2.json', dict(
        rows=rows, parent_manifest_sha256=sha(RUN / 'input_manifest_v2.json'),
        previous_contract_sha256=sha(PREVIOUS / 'full_postfilter_contract.json'),
        method='same fixed bounded L6 correction; refit five heads without old source',
        folds='sorted 100 speaker index modulo five; all three sentences together',
        alpha=100, postfilter_script_sha256=sha(ROOT / 'scripts/experiments/phoneme_direction_20261026/probe_l6_postfilter.py'),
        floors=read(ORIGINAL / 'calibration_contract.json')['reconstruction_floors'],
        candidate_only=True, development_allowed=False, confirmation_allowed=False))
    for x in rows:
        feature = new if x['sample_id'] == NEW_ID else previous[x['id']]
        assert sha(feature['path']) == feature['sha256']
        assert feature['source_sha256'] == x['canonical_audio_sha256']
        x['feature'] = feature; x['arrays'] = dict(np.load(feature['path']))
        x['fold'] = speakers.index(x['speaker_id']) % 5
    outputs = []
    for fold in range(5):
        train = [x for x in rows if x['fold'] != fold]; test = [x for x in rows if x['fold'] == fold]
        assert len(train) == 240 and len(test) == 60
        model = fit_head([x['arrays']['l6'] for x in train], [x['arrays']['target'] for x in train])
        head = RUN / f'head_fold{fold}.npz'; assert not head.exists(); np.savez_compressed(head, **model)
        write(RUN / f'head_fold{fold}_contract.json', dict(
            train_ids=[x['id'] for x in train], test_ids=[x['id'] for x in test],
            train_speakers=sorted({x['speaker_id'] for x in train}),
            test_speakers=sorted({x['speaker_id'] for x in test}), head_sha256=sha(head),
            contract_sha256=sha(RUN / 'full_reconstruction_contract_v2.json')))
        for x in test:
            old = new if x['sample_id'] == NEW_ID else peak[x['id']]
            assert sha(old['source']) == old['source_sha256'] == x['canonical_audio_sha256']
            assert sha(old['output']) == old['output_sha256']
            a, sr = sf.read(old['source']); b, sb = sf.read(old['output']); assert sr == sb == 16000
            c = postfilter(b, predict_head(x['arrays']['l6'], model), x['feature']['peak_gain'])
            c *= np.sqrt(np.mean(a * a) / np.mean(c * c))
            dest = RUN / 'reconstruction_v2' / (x['id'] + '.wav'); dest.parent.mkdir(exist_ok=True)
            assert not dest.exists(); sf.write(dest, c, sr, subtype='FLOAT')
            outputs.append(dict(id=x['id'], speaker_id=x['speaker_id'], source=old['source'],
                source_sha256=old['source_sha256'], output=str(dest), output_sha256=sha(dest),
                head=str(head), head_sha256=sha(head), fold=fold))
        print('FULL_FILTER_FOLD', fold, flush=True)
    write(RUN / 'full_reconstruction_outputs_v2.json', dict(rows=outputs))


def quality():
    import numpy as np
    from audit_conditioned_fit import measure
    contract = read(RUN / 'full_reconstruction_contract_v2.json')
    outputs = read(RUN / 'full_reconstruction_outputs_v2.json')['rows']
    expected = {x['id'] for x in contract['rows']}
    assert len(outputs) == 300 and {x['id'] for x in outputs} == expected
    assert not any(OLD_ID in x['id'] for x in outputs)
    for fold in range(5):
        c = read(RUN / f'head_fold{fold}_contract.json')
        assert len(c['train_speakers']) == 80 and len(c['test_speakers']) == 20
        assert not set(c['train_speakers']) & set(c['test_speakers'])
        assert not any(OLD_ID in sid for sid in c['train_ids'] + c['test_ids'])
        for x in outputs:
            if x['fold'] == fold:
                assert x['id'] in c['test_ids'] and x['speaker_id'] in c['test_speakers']
                assert sha(x['head']) == x['head_sha256'] == c['head_sha256']
    rows = [measure(x) for x in outputs]
    median = float(np.median([x['stoi'] for x in rows])); failed = [x for x in rows if x['failures']]
    write(RUN / 'full_reconstruction_quality_v2.json', dict(
        status='PASS_BASIC_RECONSTRUCTION' if not failed and median >= .85 else 'HOLD_RECONSTRUCTION',
        sources=300, speakers=100, stoi_median=median, stoi_min=min(x['stoi'] for x in rows),
        rows=rows, failed=failed, floors_unchanged=True, replaced_sources=1,
        old_failure_preserved=True, numeric_protocol_frozen=False,
        development_started=False, confirmation_started=False))
    print('FULL_QUALITY', median, 'failed', [(x['id'], x['stoi'], x['failures']) for x in failed], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['manifest', 'reconstruct', 'full_filter', 'quality'])
    globals()[parser.parse_args().stage]()
