import importlib.util
import hashlib
import io
import json
from pathlib import Path
import wave
import pytest

spec = importlib.util.spec_from_file_location('verify_subset', Path(__file__).parents[1] / 'scripts/verify_aishell3_decoder_subset.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_unfinished_dataset_cannot_be_admitted(tmp_path):
    with pytest.raises(FileNotFoundError):
        module.verify(tmp_path)


def test_readback_detects_changed_pcm(tmp_path):
    selection = []
    summary = dict(admission='COMPLETE_ARCHIVE_AUDIT', source_archive_status='PASS', splits={})
    for i, split in enumerate(('train','validation')):
        speaker = f'SSB{i}'
        pcm = bytes([i+1, 0])*16000
        stream = io.BytesIO()
        with wave.open(stream,'wb') as wav:
            wav.setparams((1,2,16000,0,'NONE','not compressed'))
            wav.writeframes(pcm)
        data = stream.getvalue()
        row = dict(split=split, speaker=speaker, text=f'text{i}',
                   member=f'train/wav/{speaker}/a.wav', bytes=88244, sha256=f'source{i}',
                   frames=44100, audio_path=f'audio/{split}/{speaker}/a.wav',
                   output_frames=16000, output_sha256=hashlib.sha256(data).hexdigest(),
                   output_pcm_sha256=hashlib.sha256(pcm).hexdigest(), endpoint_samples=0)
        path=tmp_path/row['audio_path']
        path.parent.mkdir(parents=True)
        path.write_bytes(data)
        selection.append(row)
        (tmp_path/(split+'.jsonl')).write_text(json.dumps(row)+'\n',encoding='utf-8')
        summary['splits'][split]=dict(utterances=1,speakers=1,duration_s=1.,audio_bytes=len(data),endpoint_samples=0)
    (tmp_path/'summary.json').write_text(json.dumps(summary),encoding='utf-8')
    (tmp_path/'selection.json').write_text(json.dumps(dict(rows=selection)),encoding='utf-8')
    assert module.verify(tmp_path)['verified_wavs']==2
    path=tmp_path/selection[0]['audio_path']
    changed=bytearray(path.read_bytes());changed[-1]^=1;path.write_bytes(changed)
    with pytest.raises(ValueError,match='Output SHA mismatch'):
        module.verify(tmp_path)
