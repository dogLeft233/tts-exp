import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import soundfile as sf


def test_original_wave_replay_checks_both_levels_and_rejects_raw_gain(tmp_path, monkeypatch):
    source = Path(__file__).parents[1] / 'scripts/experiments/qwen17_level_audio_audit_20261023.py'
    spec = importlib.util.spec_from_file_location('audio_audit', source)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    target = .0376838172675746
    samples = [{'sample_id': f's{i}', 'speaker_id': f'p{i}'} for i in range(10)]
    old, new, outputs = [], [], []
    bound_exception = None
    rng = np.random.default_rng(7)
    for sample in samples:
        for model in ['N', 'L06', 'L17', 'CLOUD']:
            for draw in ([0] if model == 'N' else [1, 2, 3]):
                path = tmp_path / f"source_{sample['sample_id']}_{model}_{draw}.wav"
                wave = (rng.normal(size=128) * .05).astype(np.float32)
                if sample['sample_id'] == 's0' and model == 'L06' and draw == 3:
                    wave[:8] = np.float32(1.05)
                sf.write(path, wave, 16000, subtype='FLOAT')
                row = dict(sample, model=model, draw_index=draw, source_path=str(path), source_sha256=audit.sha(path))
                if sample['sample_id'] == 's0' and model == 'L06' and draw == 3:
                    bound_exception = dict(sample_id='s0', model='L06', draw_index=3, source_sha256=audit.sha(path),
                                           indices=list(range(8)), values=wave[:8].tolist())
                rms = np.sqrt(np.mean(wave.astype(float) ** 2))
                for level in ['raw', 'norm']:
                    gain = 1. if level == 'raw' else target / rms
                    dest = path.with_name(path.stem + '_' + level + '.wav')
                    sf.write(dest, wave * np.float32(gain), 16000, subtype='PCM_16')
                    info = dict(sample, model=model, draw_index=draw, level=level, path=str(dest), sha256=audit.sha(dest), gain=gain)
                    if model != 'L17' and level == 'norm':
                        row['old_audio'] = info
                    else:
                        outputs.append(info)
                if model == 'L17':
                    new.append(dict(row, path=str(path), sha256=audit.sha(path)))
                else:
                    old.append(row)
    (tmp_path / 'generation').mkdir()
    def save(name, value):
        (tmp_path / name).write_text(json.dumps(value), encoding='utf8')
    save('protocol.json', dict(samples=samples, target_rms=target, raw_pcm16_exception=bound_exception))
    save('existing_sources.json', {'records': old})
    save('generation/tts_manifest.json', {'records': new})
    save('audio_manifest.json', {'records': outputs})
    def decoded(command, **kwargs):
        wave, _ = sf.read(command[command.index('-i') + 1], dtype='float32')
        return SimpleNamespace(stdout=wave.astype('<f4').tobytes())
    monkeypatch.setattr(audit.subprocess, 'run', decoded)
    result = audit.audit(tmp_path)
    assert result['status'] == 'PASS', result['errors']
    assert result['canonical_cells_checked'] == 200
    save('protocol.json', dict(samples=samples, target_rms=target))
    unbound = audit.audit(tmp_path)
    assert unbound['status'] == 'FAIL'
    assert any('outside bound exception' in e for e in unbound['errors'])
    save('protocol.json', dict(samples=samples, target_rms=target, raw_pcm16_exception=bound_exception))
    outputs[0]['gain'] = 2.
    save('audio_manifest.json', {'records': outputs})
    changed = audit.audit(tmp_path)
    assert changed['status'] == 'FAIL'
    assert any('raw gain not unity' in e for e in changed['errors'])
