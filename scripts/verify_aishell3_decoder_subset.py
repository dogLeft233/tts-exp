"""Read every delivered WAV and independently verify manifests and split isolation."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import wave


def verify(root):
    root = Path(root).resolve()
    summary = json.loads((root/'summary.json').read_text(encoding='utf-8'))
    selection = json.loads((root/'selection.json').read_text(encoding='utf-8'))['rows']
    source = {r['member']: r for r in selection}
    pins = {}
    for p in (root/'source_mirror_pins').glob('SSB*.json'):
        for r in json.loads(p.read_text(encoding='utf-8'))['rows']:
            pins[r['path']] = r
    observed, groups, paths, members = {}, {}, set(), set()
    for split in ('train', 'validation'):
        rows = [json.loads(l) for l in (root/(split+'.jsonl')).read_text(encoding='utf-8').splitlines()]
        groups[split] = {k: set() for k in ('speaker', 'text', 'output_pcm_sha256')}
        total_frames, total_bytes, endpoints = 0, 0, 0
        for row in rows:
            if row['split'] != split or row['member'] in members:
                raise ValueError('Duplicate member or mislabeled split')
            members.add(row['member'])
            p = (root/row['audio_path']).resolve()
            if not p.is_relative_to(root) or p in paths:
                raise ValueError('Unsafe or duplicate output path')
            paths.add(p)
            data = p.read_bytes()
            if hashlib.sha256(data).hexdigest() != row['output_sha256']:
                raise ValueError('Output SHA mismatch')
            with wave.open(io.BytesIO(data), 'rb') as wav:
                frames = wav.getnframes()
                if (wav.getframerate(),wav.getnchannels(),wav.getsampwidth(),wav.getcomptype()) != (16000,1,2,'NONE'):
                    raise ValueError('Output format mismatch')
                pcm = wav.readframes(frames)
            if len(pcm) != frames*2 or frames != row['output_frames']:
                raise ValueError('Output length mismatch')
            if hashlib.sha256(pcm).hexdigest() != row['output_pcm_sha256']:
                raise ValueError('Output PCM mismatch')
            original = source[row['member']]
            if (row['sha256'],row['speaker'],row['text']) != (original['sha256'],original['speaker'],original['text']):
                raise ValueError('Selection identity mismatch')
            if abs(frames - original['frames']*16000/44100) > 1:
                raise ValueError('Clock mismatch')
            if summary['admission'] == 'INDIVIDUAL_WAV_MIRROR_SHA256':
                pin = pins[row['member']]
                if pin['lfs']['oid'] != original['sha256'] or pin['size'] != original['bytes']:
                    raise ValueError('Independent source pin mismatch')
            for key in groups[split]:
                groups[split][key].add(row[key])
            total_frames += frames
            total_bytes += len(data)
            endpoints += row['endpoint_samples']
        actual = dict(utterances=len(rows), speakers=len(groups[split]['speaker']),
                      duration_s=total_frames/16000, audio_bytes=total_bytes, endpoint_samples=endpoints)
        expected = summary['splits'][split]
        if (any(actual[k] != expected[k] for k in actual if k != 'duration_s') or
                abs(actual['duration_s']-expected['duration_s']) > 1/16000):
            raise ValueError('Split summary mismatch')
        observed[split] = actual
    for key in groups['train']:
        if groups['train'][key] & groups['validation'][key]:
            raise ValueError('Split overlap: '+key)
    if members != set(source):
        raise ValueError('Selection and delivered sources differ')
    if {p.resolve() for p in (root/'audio').rglob('*.wav')} != paths:
        raise ValueError('Unexpected or missing audio files')
    return dict(status='PASS', verified_wavs=len(paths), splits=observed,
                source_archive_status=summary['source_archive_status'],
                source_admission=summary['admission'],
                speaker_text_pcm_overlap=0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root')
    parser.add_argument('--receipt', required=True)
    args = parser.parse_args()
    result = verify(args.root)
    receipt = Path(args.receipt)
    with receipt.open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2)
    print(json.dumps(result))
