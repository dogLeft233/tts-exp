"""Independent CPU audit of the cloud corpus; no producer imports or API."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf


def audit(out):
    p = json.loads((out / 'protocol.json').read_text())
    m = json.loads((out / 'audio_manifest.json').read_text())
    rows = m['records']
    expected = {(s['sample_id'], d) for s in p['samples'] for d in [1, 2, 3]}
    assert len(rows) == 60 and {(r['sample_id'], r['draw_index']) for r in rows} == expected
    sources = {s['sample_id']: s for s in p['samples']}
    errors = []
    rms_errors = []
    checked = set()
    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for row in rows:
        source = sources[row['sample_id']]
        assert row['target_text'] == ''.join(source['transcript'].split())
        assert row['reference_audio_sha256'] == source['input24_sha256']
        assert sha(source['input24_path']) == source['input24_sha256']
        assert sha(row['provider_audio']) == row['provider_sha256']
        assert sha(row['path']) == row['sha256']
        raw = subprocess.check_output(['ffmpeg', '-v', 'error', '-i', row['provider_audio'],
            '-ac', '1', '-ar', '16000', '-f', 'f32le', '-acodec', 'pcm_f32le', '-'])
        audio = np.frombuffer(raw, dtype='<f4')
        samples, sr = sf.read(row['path'], dtype='int16')
        assert sr == 16000 and samples.ndim == 1 and len(samples) == len(audio) == row['samples']
        gain = p['target_rms'] / np.sqrt(np.mean(audio.astype('float64') ** 2))
        predicted = np.floor(audio * np.float32(gain) * 32768).clip(-32768, 32767).astype('int32')
        error = int(np.abs(predicted - samples.astype('int32')).max())
        assert error <= 1 and abs(gain - row['gain']) < 1e-10
        rms_error = abs(np.sqrt(np.mean((samples.astype('float64') / 32768) ** 2)) - p['target_rms'])
        assert rms_error < 3e-6
        assert hashlib.sha256(samples.astype('<i2').tobytes()).hexdigest() == row['pcm_sha256']
        checked.update([row['path'], row['provider_audio'], source['input24_path']])
        errors.append(error)
        rms_errors.append(float(rms_error))
    result = dict(status='PASS', audio_count=60, sources=20,
        same_reference_sources=sum(not s['additional_reference'] for s in p['samples']),
        extended_reference_sources=sum(bool(s['additional_reference']) for s in p['samples']),
        maximum_reconstruction_PCM16_error_LSB=max(errors), maximum_RMS_target_error=max(rms_errors),
        provider_unique_bytes=len({r['provider_sha256'] for r in rows}), checked_files=len(checked),
        no_output_selection=True, no_time_editing_verified_by_sample_count=True,
        protocol_sha256=sha(out / 'protocol.json'), manifest_sha256=sha(out / 'audio_manifest.json'))
    (out / 'synthesis_independent_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    audit(parser.parse_args().out)
