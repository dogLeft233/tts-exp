"""Fit-only diagnostic: reversible conditioning and local encoder context.

No original waveform is mixed into any decoded output. Candidate processing
changes the reconstruction contract and is not admitted by this diagnostic.
"""
import sys
import numpy as np
import soundfile as sf
import torch
from scipy.ndimage import uniform_filter1d
from calibrate import ROOT, RUN as PARENT, read, write, sha, select_calibration
from diagnose_reconstruction import REGULAR

RUN = ROOT / 'runs/phoneme_direction_20261026_repair02'
PREVIOUS = ROOT / 'runs/phoneme_direction_20261026_repair01'


def conditioning(a, kind):
    if kind == 'peak95':
        gain = np.full(len(a), .95 / max(float(np.max(abs(a))), 1e-8))
    elif kind == 'local_rms16':
        # Fixed one-second energy window, bounded +30 dB amplification.
        energy = uniform_filter1d(a.astype('float64') ** 2, 16000, mode='reflect')
        gain = np.minimum(10 ** (30 / 20), 10 ** (-16 / 20) / np.sqrt(np.maximum(energy, 1e-12)))
    elif kind == 'local_context':
        gain = np.ones(len(a))
    else:
        raise ValueError(kind)
    return (a * gain).astype('float32'), gain


def encode(adapter, b, kind):
    n = (len(b) + 319) // 320
    padded = np.pad(b, (0, n * 320 + 80 - len(b)))
    if kind != 'local_context':
        return adapter.extract(torch.from_numpy(padded))
    # Two-second central chunks, one-second context on each side. Absolute
    # sample starts are multiples of the encoder hop; no resampling/warping.
    chunks = []
    for lo in range(0, n, 100):
        hi = min(n, lo + 100)
        start, end = max(0, lo - 50), min(n, hi + 50)
        h = adapter.extract(torch.from_numpy(padded[start * 320:end * 320 + 80]))
        chunks.append(h[lo - start:hi - start])
    return torch.cat(chunks)


def main():
    torch.set_num_threads(4)
    torch.manual_seed(20261026)
    torch.use_deterministic_algorithms(True)
    fit = read(PARENT / 'input_manifest.json')['fit']
    failures = {x['id'] for x in read(PREVIOUS / 'regular_full_fit_quality.json')['failed']}
    controls = {x['id'] for x in select_calibration(fit) if x['arm'] == 'N'}
    rows = [x for x in fit if x['arm'] == 'N' and x['id'] in failures | controls]
    assert len(rows) == 23 and all(x['split'] == 'fit' for x in rows)
    kinds = ['peak95', 'local_rms16', 'local_context']
    write(RUN / 'probe_contract.json', {
        'rows': [x['id'] for x in rows], 'variants': kinds,
        'checkpoint_sha256': sha(REGULAR), 'script_sha256': sha(__file__),
        'parent_contract_sha256': sha(PARENT / 'calibration_contract.json'),
        'candidate_only': True, 'no_original_waveform_mixing': True,
        'floors_unchanged': True, 'development_allowed': False, 'confirmation_allowed': False})
    sys.path.insert(0, str(ROOT / 'scripts'))
    sys.path.insert(0, str(ROOT / 'third_party/knn-vc'))
    from wavlm_knn_vc_adapter import WavLMKNNVCAdapter
    from hifigan.models import Generator
    from hifigan.utils import AttrDict
    adapter = WavLMKNNVCAdapter.load_pretrained(device='cpu', source=ROOT / 'third_party/knn-vc')
    g = Generator(AttrDict(read(ROOT / 'third_party/knn-vc/hifigan/config_v1_wavlm.json')))
    g.load_state_dict(torch.load(REGULAR, map_location='cpu', weights_only=False)['generator'])
    g.eval(); g.remove_weight_norm()
    outputs = []
    for x in rows:
        assert sha(x['canonical_audio_path']) == x['canonical_audio_sha256']
        a, sr = sf.read(x['canonical_audio_path'], dtype='float32')
        for kind in kinds:
            b, gain = conditioning(a, kind)
            h = encode(adapter, b, kind)
            assert tuple(h.shape) == ((len(a) + 319) // 320, 1024)
            with torch.inference_mode():
                raw = g(h[None]).flatten().numpy()[:len(a)].astype('float64')
            raw /= gain
            raw *= np.sqrt(np.mean(a.astype('float64') ** 2) / np.mean(raw ** 2))
            p = RUN / 'probe_audio' / kind / f'{x["id"]}.wav'
            p.parent.mkdir(parents=True, exist_ok=True)
            if p.exists():
                raise FileExistsError(p)
            sf.write(p, raw, sr, subtype='FLOAT')
            record = {'id': x['id'], 'kind': kind, 'source': x['canonical_audio_path'],
                      'source_sha256': x['canonical_audio_sha256'], 'output': str(p),
                      'output_sha256': sha(p), 'failed_before': x['id'] in failures}
            write(p.with_suffix('.json'), record)
            outputs.append(record)
        print('PROBED', x['id'], flush=True)
    write(RUN / 'probe_outputs.json', {'rows': outputs})


if __name__ == '__main__':
    main()
