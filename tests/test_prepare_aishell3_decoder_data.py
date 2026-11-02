import importlib.util
import gzip
import io
import json
from pathlib import Path
import tarfile
import wave


spec = importlib.util.spec_from_file_location(
    'aishell3_prepare', Path(__file__).parents[1] / 'scripts/prepare_aishell3_decoder_data.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pcm_wav():
    stream = io.BytesIO()
    with wave.open(stream, 'wb') as out:
        out.setparams((1, 2, 44100, 0, 'NONE', 'not compressed'))
        out.writeframes(b'\x01\x00' * 44100)
    return stream.getvalue()


def make_archive(path, name='train/wav/SSB0001/a.wav'):
    data = pcm_wav()
    with tarfile.open(path, 'w:gz') as tar:
        member = tarfile.TarInfo(name)
        member.size = len(data)
        tar.addfile(member, io.BytesIO(data))


def test_complete_archive_reads_crc_and_pcm(tmp_path):
    archive = tmp_path / 'source.tgz'
    make_archive(archive)
    result = module.audit(archive, tmp_path / 'audit')
    assert result['status'] == 'PASS'
    assert result['bytes_read'] == archive.stat().st_size
    assert result['wav_duration_s'] == 1.0
    row = json.loads((tmp_path / 'audit/archive_members.jsonl').read_text())
    assert row['speaker'] == 'SSB0001' and row['sample_rate'] == 44100


def test_missing_gzip_trailer_is_not_admitted(tmp_path):
    archive = tmp_path / 'source.tgz'
    make_archive(archive)
    archive.write_bytes(archive.read_bytes()[:-8])
    result = module.audit(archive, tmp_path / 'audit')
    assert result['status'] == 'FAIL'
    assert result['errors']


def test_members_after_tar_zero_blocks_are_also_audited(tmp_path):
    first, second = io.BytesIO(), io.BytesIO()
    for stream, speaker in [(first, 'SSB0001'), (second, 'SSB0002')]:
        with tarfile.open(fileobj=stream, mode='w') as tar:
            data = pcm_wav()
            member = tarfile.TarInfo(f'train/wav/{speaker}/a.wav')
            member.size = len(data)
            tar.addfile(member, io.BytesIO(data))
    archive = tmp_path / 'source.tgz'
    archive.write_bytes(gzip.compress(first.getvalue()+second.getvalue()))
    result = module.audit(archive, tmp_path / 'audit')
    assert result['status'] == 'PASS' and result['wav_members'] == 2


def test_path_traversal_is_rejected(tmp_path):
    archive = tmp_path / 'source.tgz'
    make_archive(archive, '../escape.wav')
    result = module.audit(archive, tmp_path / 'audit')
    assert result['status'] == 'FAIL'
    assert not (tmp_path / 'escape.wav').exists()


def test_short_wav_payload_is_rejected():
    import pytest
    with pytest.raises(ValueError, match='shorter'):
        module.wav_info(pcm_wav()[:-2])
