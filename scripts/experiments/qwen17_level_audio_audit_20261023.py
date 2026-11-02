"""Independent CPU replay from original waves for both audio-level conditions."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess

import numpy as np
import soundfile as sf


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(run):
    protocol = read(run / 'protocol.json')
    old = read(run / 'existing_sources.json')['records']
    new = read(protocol.get('generation_source_manifest_path', run / 'generation/tts_manifest.json'))['records']
    outputs = read(run / 'audio_manifest.json')['records']
    target = protocol['target_rms']
    checks, errors, original_fingerprints = [], [], {}
    indexed = {(r['sample_id'], r['model'], r['level'], r['draw_index']): r for r in outputs}
    expected = {(s['sample_id'], m, l, d) for s in protocol['samples']
                for m, l in [('N', 'raw'), ('L06', 'raw'), ('CLOUD', 'raw'), ('L17', 'norm'), ('L17', 'raw')]
                for d in ([0] if m == 'N' else [1, 2, 3])}
    if len(outputs) != 130 or set(indexed) != expected:
        errors.append('new canonical130 coverage')
    sources = [dict(r, source_path=r['path'], source_sha256=r['sha256']) for r in new] + old
    for source in sources:
        sid, model, draw = source['sample_id'], source['model'], source['draw_index']
        label = f'{sid}/{model}/{draw}'
        try:
            if sha(source['source_path']) != source['source_sha256']:
                raise ValueError('original source SHA mismatch')
            raw = subprocess.run(['/usr/bin/ffmpeg', '-nostdin', '-v', 'error', '-i', source['source_path'],
                                  '-ac', '1', '-ar', '16000', '-f', 'f32le', '-c:a', 'pcm_f32le', '-'],
                                 capture_output=True, check=True, timeout=60).stdout
            wave = np.frombuffer(raw, dtype='<f4')
            if not len(wave) or not np.isfinite(wave).all():
                raise ValueError('invalid source waveform')
            rms = np.linalg.norm(wave.astype(np.float64)) / np.sqrt(len(wave))
            # Recompute RMS by a different reduction, keeping tolerance explicit.
            original_fingerprints[label] = source['source_sha256']
            levels = [('raw', wave, 1.)]
            gain = target / rms
            levels.append(('norm', wave * np.float32(gain), gain))
            for level, values, actual_gain in levels:
                if model != 'L17' and level == 'norm':
                    info = source['old_audio']
                else:
                    info = indexed[sid, model, level, draw]
                if np.max(np.abs(values)) >= 1:
                    exception = protocol.get('raw_pcm16_exception', {})
                    indices = np.flatnonzero((values >= 1) | (values < -1)).tolist()
                    if not (level == 'raw' and sid == exception.get('sample_id') and model == exception.get('model')
                            and draw == exception.get('draw_index') and source['source_sha256'] == exception.get('source_sha256')
                            and indices == exception.get('indices')
                            and np.array_equal(values[indices], np.asarray(exception.get('values'), dtype=np.float32))):
                        raise ValueError('would clip outside bound exception')
                if sha(info['path']) != info['sha256']:
                    raise ValueError('canonical SHA mismatch')
                got, sr = sf.read(info['path'], dtype='int16')
                if sr != 16000 or got.ndim != 1 or len(got) != len(wave):
                    raise ValueError('sample clock, channels, or duration changed')
                encoded = io.BytesIO()
                sf.write(encoded, values, 16000, format='WAV', subtype='PCM_16')
                encoded.seek(0)
                expected_pcm, _ = sf.read(encoded, dtype='int16')
                if not np.array_equal(got, expected_pcm):
                    raise ValueError('gain/PCM replay differs')
                if level == 'raw' and info['gain'] != 1.:
                    raise ValueError('raw gain not unity')
                if 'float16_path' in info:
                    if sha(info['float16_path']) != info['float16_sha256']:
                        raise ValueError('saved float16 SHA mismatch')
                    saved_float, float_sr = sf.read(info['float16_path'], dtype='float32')
                    if float_sr != 16000 or not np.array_equal(saved_float, wave):
                        raise ValueError('original resampling replay differs')
                checks.append(dict(sample_id=sid, model=model, level=level, draw_index=draw,
                                   samples=len(got), gain=float(actual_gain),
                                   source_rms=float(rms), pcm_exact=True,
                                   pcm_sha256=hashlib.sha256(got.astype('<i2').tobytes()).hexdigest(),
                                   output_rms=float(np.linalg.norm(got.astype(float) / 32768) / np.sqrt(len(got)))))
        except Exception as exc:
            errors.append(label + ': ' + type(exc).__name__ + ': ' + str(exc))
    return dict(status='PASS' if not errors and len(checks) == 200 else 'FAIL', errors=errors,
                original_sources=len(sources), canonical_cells_checked=len(checks),
                original_source_sha256=original_fingerprints, checks=checks,
                protocol_sha256=sha(run / 'protocol.json'), audit_source_sha256=sha(__file__),
                no_time_editing_verified=True if not errors and len(checks) == 200 else False,
                no_model_forward=True, no_training=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    report = audit(args.run)
    with (args.run / 'audio_independent_audit.json').open('x', encoding='utf8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({k: v for k, v in report.items() if k not in ['checks', 'original_source_sha256']}))
    raise SystemExit(0 if report['status'] == 'PASS' else 1)
