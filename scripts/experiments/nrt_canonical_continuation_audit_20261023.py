"""Independently replay fixed-gain PCM serialization; never edit scientific WAVs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run):
    plan = read(run / 'analysis_protocol.json')
    exception = plan['authorized_protocol_exception']
    key_exception = (exception['sample_id'], exception['role'])
    copied = run / 'source_attempt01'
    bridge = read(run / 'source_copy_manifest.json')
    original = Path(bridge['source_run'])
    preflight = read(copied / 'canonical_preflight/audit.json')['records']
    canonical = {(r['sample_id'], r['role']): r for r in read(run / 'audio_manifest.json')['records']}
    if len(canonical) != 80 or len(preflight) != 80:
        raise ValueError('All80 required')
    target = read(run / 'protocol.json')['target_rms']
    rows = []
    for source in preflight:
        key = (source['sample_id'], source['role'])
        record = canonical[key]
        float_path = copied / Path(source['float_path']).relative_to(original)
        assert sha(float_path) == source['float_sha256']
        wave, rate = sf.read(float_path, dtype='float32')
        assert rate == 16000 and len(wave) == source['samples']
        rms = np.sqrt(np.mean(wave.astype(np.float64) ** 2))
        gain = target / float(rms)
        normalized = wave * np.float32(gain)
        saturated = np.clip(normalized, np.float32(-1), np.float32(32767 / 32768))
        changed = np.flatnonzero(saturated != normalized).tolist()
        if key == key_exception:
            assert changed == exception['sample_indices_0based']
            final = saturated
        else:
            assert float(np.abs(wave).max() * gain) < 1
            assert not changed
            final = normalized
        memory = io.BytesIO()
        sf.write(memory, final, 16000, format='WAV', subtype='PCM_16')
        memory.seek(0)
        predicted, _ = sf.read(memory, dtype='int16')
        actual, rate = sf.read(record['path'], dtype='int16')
        assert rate == 16000 and np.array_equal(predicted, actual)
        assert len(actual) == source['samples']
        assert sha(record['path']) == record['sha256']
        assert hashlib.sha256(actual.astype('<i2').tobytes()).hexdigest() == record['pcm_sha256']
        old = source['existing_canonical']
        exact = None
        if old is not None:
            old_path = copied / Path(old['path']).relative_to(original)
            exact = sha(old_path) == old['sha256'] == sha(record['path'])
            assert exact
        rows.append({'sample_id': key[0], 'role': key[1], 'PASS': True,
                     'changed_sample_indices': changed, 'original_PCM_byte_exact': exact,
                     'samples': len(actual), 'gain': gain,
                     'actual_RMS': float(np.sqrt(np.mean((actual.astype(float) / 32768) ** 2)))})
    assert sum(r['original_PCM_byte_exact'] is True for r in rows) == 63
    assert sum(bool(r['changed_sample_indices']) for r in rows) == 1
    for entry in bridge['files']:
        assert sha(run / entry['copy_relative_path']) == entry['sha256']
    return {'status': 'PASS', 'records': rows, 'cells': len(rows),
            'original63_PCM_byte_exact': True, 'only_authorized3_samples_saturated': True,
            'copied_source_files_verified': len(bridge['files']),
            'source_sha256': sha(__file__), 'no_GPU_or_audio_write': True}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--run', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.run)
    with (args.run / 'independent_canonical_audit.json').open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}))


if __name__ == '__main__':
    main()
