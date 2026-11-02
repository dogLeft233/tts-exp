import array
import hashlib
import importlib.util
import io
import math
from pathlib import Path
import sys
import wave
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'scripts'))
spec = importlib.util.spec_from_file_location('subset', Path(__file__).parents[1] / 'scripts/build_aishell3_decoder_subset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_partial_archive_file_requires_exact_independent_pin():
    row = dict(member='train/wav/SSB1/a.wav', bytes=100, sha256='abc')
    module.check_pin(row, dict(path=row['member'], size=100, lfs={'oid': 'abc'}))
    with pytest.raises(ValueError, match='mismatch'):
        module.check_pin(row, dict(path=row['member'], size=100, lfs={'oid': 'def'}))


def test_selection_has_disjoint_speakers_text_and_pcm_and_5h_training():
    rows, texts, speakers = [], {}, {}
    for i in range(44):
        speaker = f'SSB{i:04d}'
        speakers[speaker] = dict(gender='male' if i < 9 else 'female', age_group='B', accent='north')
        for j in range(30):
            name = f'{speaker}{j:04d}.wav'
            rows.append(dict(member=f'train/wav/{speaker}/{name}', speaker=speaker,
                official_split='train', sample_rate=44100, channels=1, sample_width=2,
                duration_s=15, pcm_sha256=f'{i}/{j}'))
            texts[name] = dict(text=chr(0x4e00+i*30+j), annotation='test')
    a = module.select(rows, texts, speakers)
    assert a == module.select(list(reversed(rows)), texts, speakers)
    train = [r for r in a if r['split'] == 'train']
    val = [r for r in a if r['split'] == 'validation']
    assert sum(r['duration_s'] for r in train) == 18000
    assert sum(r['duration_s'] for r in val) == 1800
    for key in ('speaker', 'text', 'pcm_sha256'):
        assert not ({r[key] for r in train} & {r[key] for r in val})


def test_real_soxr_resampling_preserves_clock_and_rejects_alias(tmp_path):
    ffmpeg = Path('F:/Apps/GPT-SoVITS-v2pro-20250604/runtime/ffmpeg.exe')
    if not ffmpeg.exists():
        pytest.skip('Pinned local FFmpeg not available')
    pcm = array.array('h', (round(6000*math.sin(2*math.pi*3000*i/44100)+
                                6000*math.sin(2*math.pi*12000*i/44100)) for i in range(44100)))
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as w:
        w.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
        w.writeframes(pcm.tobytes())
    data = stream.getvalue()
    row = dict(member='train/wav/SSB1/a.wav', speaker='SSB1', split='train',
               sha256=hashlib.sha256(data).hexdigest(), frames=44100)
    result = module.convert(data, row, tmp_path, str(ffmpeg))
    assert result['output_frames'] == 16000
    with wave.open(str(tmp_path/result['audio_path']), 'rb') as w:
        samples = array.array('h', w.readframes(w.getnframes()))
    def amplitude(hz):
        # Ignore finite-filter boundary transients.
        return abs(sum(samples[i]*complex(math.cos(2*math.pi*hz*i/16000),
                                          math.sin(2*math.pi*hz*i/16000))
                       for i in range(1000, 15000)))
    assert amplitude(4000) / amplitude(3000) < .001
