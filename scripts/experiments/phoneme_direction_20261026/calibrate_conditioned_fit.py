"""Full 300-source audit candidate with fixed peak conditioning, fit only."""
import argparse
import sys
import numpy as np
import soundfile as sf
import torch
from calibrate import ROOT, RUN as PARENT, read, write, sha
from diagnose_reconstruction import REGULAR
from probe_weak_segments import RUN, conditioning, encode


def worker(shard):
    if shard not in range(3):
        raise ValueError('Three fixed shards required')
    rows = [x for x in read(PARENT / 'input_manifest.json')['fit'] if x['arm'] == 'N']
    assert len(rows) == 300 and len({x['speaker_id'] for x in rows}) == 100
    assert all(x['split'] == 'fit' for x in rows)
    write(RUN / f'full_peak95_shard{shard}_contract.json', {
        'rows': [x['id'] for x in rows[shard::3]], 'kind': 'peak95',
        'rationale': 'Standard peak conditioning; not a per-source score-selected rule',
        'checkpoint_sha256': sha(REGULAR), 'script_sha256': sha(__file__),
        'conditioning_script_sha256': sha(ROOT / 'scripts/experiments/phoneme_direction_20261026/probe_weak_segments.py'),
        'parent_contract_sha256': sha(PARENT / 'calibration_contract.json'),
        'all_floors_unchanged': True, 'candidate_only': True,
        'development_allowed': False, 'confirmation_allowed': False})
    torch.set_num_threads(4); torch.manual_seed(20261026)
    torch.use_deterministic_algorithms(True)
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
    for x in rows[shard::3]:
        assert sha(x['canonical_audio_path']) == x['canonical_audio_sha256']
        a, sr = sf.read(x['canonical_audio_path'], dtype='float32')
        assert sr == 16000 and a.ndim == 1
        b, gain = conditioning(a, 'peak95')
        h = encode(adapter, b, 'peak95')
        assert tuple(h.shape) == ((len(a) + 319) // 320, 1024)
        with torch.inference_mode():
            raw = g(h[None]).flatten().numpy()[:len(a)].astype('float64')
        raw /= gain
        raw *= np.sqrt(np.mean(a.astype('float64') ** 2) / np.mean(raw ** 2))
        p = RUN / 'full_peak95_audio' / f'{x["id"]}.wav'
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.exists():
            raise FileExistsError(p)
        sf.write(p, raw, sr, subtype='FLOAT')
        record = {'id': x['id'], 'speaker_id': x['speaker_id'], 'source': x['canonical_audio_path'],
                  'source_sha256': x['canonical_audio_sha256'], 'output': str(p),
                  'output_sha256': sha(p), 'samples': len(a), 'frames': len(h)}
        write(p.with_suffix('.json'), record); outputs.append(record)
        print('RECONSTRUCTED', x['id'], flush=True)
    write(RUN / f'full_peak95_shard{shard}_complete.json', {'rows': outputs})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('shard', type=int)
    worker(parser.parse_args().shard)
