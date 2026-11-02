"""Fit-only speaker-held-out spectral probes; not a replacement decoder.

Linear recoverability can locate an information bottleneck, but failure of a
linear probe does not prove that a representation contains no information.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import soundfile as sf
from calibrate import ROOT, RUN as PARENT, read, write, sha, select_calibration

RUN = ROOT / 'runs/phoneme_direction_20261026_repair03'
OLD = ROOT / 'runs/phoneme_direction_20261026_repair02'


def bands(a):
    """15 third-octave log-power bands at the encoder's original 400/320 grid."""
    n = (len(a) + 319) // 320
    a = np.pad(a, (0, n * 320 + 80 - len(a)))
    frames = np.lib.stride_tricks.sliding_window_view(a, 400)[::320]
    power = abs(np.fft.rfft(frames * np.hanning(400), n=512)) ** 2
    freq = np.fft.rfftfreq(512, 1 / 16000)
    centers = 150 * 2 ** (np.arange(15) / 3)
    values = np.stack([power[:, (freq >= c / 2 ** (1 / 6)) &
                                  (freq < c * 2 ** (1 / 6))].sum(1) for c in centers], 1)
    return 10 * np.log10(np.maximum(values, 1e-12))


def extract():
    import torch
    from probe_weak_segments import conditioning, encode
    from diagnose_reconstruction import REGULAR
    torch.set_num_threads(4); torch.manual_seed(20261026)
    torch.use_deterministic_algorithms(True)
    fit = read(PARENT / 'input_manifest.json')['fit']
    failed = {x['id'] for x in read(ROOT / 'runs/phoneme_direction_20261026_repair01/regular_full_fit_quality.json')['failed']}
    control = {x['id'] for x in select_calibration(fit) if x['arm'] == 'N'}
    rows = [x for x in fit if x['arm'] == 'N' and x['id'] in failed | control]
    assert len(rows) == 23 and all(x['split'] == 'fit' for x in rows)
    write(RUN / 'separation_contract.json', {
        'rows': [x['id'] for x in rows], 'fold': 'sorted speaker index modulo five',
        'ridge_alpha': 100., 'feature_context_frames': 0,
        'targets': '15 original third-octave log-power bands, 400 samples / hop320',
        'weak_definition': 'original per-frame total bandpower 20dB below maximal frame; diagnostic only',
        'stages': ['conv', 'postnorm', 'l6', 'lin_pre'],
        'candidate_only': True, 'gates_changed': False, 'source_sha256': sha(__file__),
        'regular_checkpoint_sha256': sha(REGULAR), 'parent_manifest_sha256': sha(PARENT / 'input_manifest.json'),
        'development_allowed': False, 'confirmation_allowed': False})
    sys.path.insert(0, str(ROOT / 'scripts')); sys.path.insert(0, str(ROOT / 'third_party/knn-vc'))
    from wavlm_knn_vc_adapter import WavLMKNNVCAdapter
    adapter = WavLMKNNVCAdapter.load_pretrained(device='cpu', source=ROOT / 'third_party/knn-vc')
    from hifigan.models import Generator
    from hifigan.utils import AttrDict
    g = Generator(AttrDict(read(ROOT / 'third_party/knn-vc/hifigan/config_v1_wavlm.json')))
    g.load_state_dict(torch.load(REGULAR, map_location='cpu', weights_only=False)['generator'])
    g.eval()
    captured = {}
    def hook(name, transpose=False):
        def save(module, inputs, output):
            x = output.detach().cpu()
            if transpose: x = x.transpose(1, 2)
            captured[name] = x.squeeze(0).numpy().copy()
        return save
    hooks = [adapter.model.wavlm.feature_extractor.register_forward_hook(hook('conv', True)),
             adapter.model.wavlm.layer_norm.register_forward_hook(hook('postnorm'))]
    prior = {x['id']: x for x in read(OLD / 'full_peak95_quality.json')['rows']}
    outputs = []
    for x in rows:
        assert sha(x['canonical_audio_path']) == x['canonical_audio_sha256']
        a, sr = sf.read(x['canonical_audio_path'], dtype='float32'); assert sr == 16000
        b, gain = conditioning(a, 'peak95')
        h = encode(adapter, b, 'peak95')
        with torch.inference_mode(): z = g.lin_pre(h).numpy()
        old = prior[x['id']]; assert sha(old['output']) == old['output_sha256']
        decoded, _ = sf.read(old['output'])
        target = bands(b)
        features = dict(captured, l6=h.numpy(), lin_pre=z)
        assert all(v.shape[0] == len(target) for v in features.values())
        p = RUN / 'stage_features' / (x['id'] + '.npz'); p.parent.mkdir(exist_ok=True)
        if p.exists(): raise FileExistsError(p)
        np.savez_compressed(p, **features, target=target, decoded=bands(decoded * gain))
        outputs.append(dict(id=x['id'], speaker_id=x['speaker_id'], path=str(p), sha256=sha(p),
                            source_sha256=x['canonical_audio_sha256']))
        print('EXTRACTED', x['id'], flush=True)
    for h in hooks: h.remove()
    write(RUN / 'separation_features.json', {'rows': outputs})


def probe():
    from scipy.linalg import solve
    rows = read(RUN / 'separation_features.json')['rows']
    speakers = sorted({x['speaker_id'] for x in rows})
    data = []
    for x in rows:
        assert sha(x['path']) == x['sha256']
        data.append(dict(x, arrays=dict(np.load(x['path'])), fold=speakers.index(x['speaker_id']) % 5))
    outputs = []
    for fold in range(5):
        train = [x for x in data if x['fold'] != fold]
        test = [x for x in data if x['fold'] == fold]
        assert not {x['speaker_id'] for x in train} & {x['speaker_id'] for x in test}
        y = np.concatenate([x['arrays']['target'] for x in train]); ym = y.mean(0)
        preds = {x['id']: {} for x in test}
        for stage in ['conv', 'postnorm', 'l6', 'lin_pre']:
            h = np.concatenate([x['arrays'][stage] for x in train]).astype('float64')
            hm = h.mean(0); hs = np.maximum(h.std(0), 1e-6); h = (h - hm) / hs
            gram = h.T @ h; gram.flat[::len(gram) + 1] += 100.
            w = solve(gram, h.T @ (y - ym), assume_a='pos')
            for x in test: preds[x['id']][stage] = (x['arrays'][stage] - hm) / hs @ w + ym
        for x in test:
            target = x['arrays']['target']; energy = (10 ** (target / 10)).sum(1)
            weak = energy < energy.max() * .01
            measures = {}
            for stage, pred in dict(preds[x['id']], decoded=x['arrays']['decoded']).items():
                error = (pred - target) ** 2
                measures[stage] = dict(rmse_db=float(np.sqrt(error.mean())),
                    weak_rmse_db=float(np.sqrt(error[weak].mean())) if weak.any() else None,
                    other_rmse_db=float(np.sqrt(error[~weak].mean())),
                    band_correlations=[float(np.corrcoef(target[:, j], pred[:, j])[0, 1]) for j in range(15)])
            outputs.append(dict(id=x['id'], fold=fold, frames=len(target), weak_frames=int(weak.sum()), measures=measures))
        print('PROBED_FOLD', fold, flush=True)
    write(RUN / 'separation_quality.json', {'rows': outputs, 'linear_probe_only': True,
        'not_information_theoretic_loss': True, 'gates_changed': False,
        'contract_sha256': sha(RUN / 'separation_contract.json')})
    for x in outputs:
        if 'S0168' in x['id']: print(x, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['extract', 'probe'])
    globals()[p.parse_args().stage]()
