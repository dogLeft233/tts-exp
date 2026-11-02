"""Create a fixed, speaker-disjoint reconstruction dataset from an audited archive."""
import argparse
import array
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import tarfile
import time
import urllib.request
import wave

from prepare_aishell3_decoder_data import wav_info

SEED = 'aishell3-decoder-20261025-v1'


def rank(value):
    return hashlib.sha256((SEED + ':' + value).encode()).hexdigest()


def check_pin(row, remote):
    if (remote.get('path') != row['member'] or remote.get('size') != row['bytes'] or
            remote.get('lfs', {}).get('oid') != row['sha256']):
        raise ValueError('Independent mirror SHA-256 mismatch: ' + row['member'])


def corroborate(selection, directory):
    """Admit complete files from a truncated archive only with independent LFS pins."""
    repository = 'shenyunhang/AISHELL-3'
    base = 'https://huggingface.co/api/datasets/' + repository
    def get_json(url):
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as response:
                    value = json.load(response)
                    if response.headers.get('Link'):
                        raise ValueError('Unexpected paginated speaker folder; refusing partial pins')
                    return value
            except Exception:
                if attempt == 3:
                    raise
                time.sleep(2)
    revision = get_json(base)['sha']
    folders = sorted({str(PurePosixPath(r['member']).parent) for r in selection})
    directory.mkdir(parents=True, exist_ok=False)
    def get_folder(folder):
        url = f'{base}/tree/{revision}/{folder}?limit=1000'
        rows = get_json(url)
        (directory / (PurePosixPath(folder).name+'.json')).write_text(
            json.dumps(dict(url=url, rows=rows), indent=2), encoding='utf-8')
        return rows
    pins = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for rows in pool.map(get_folder, folders):
            pins.update({r['path']: r for r in rows})
    for row in selection:
        check_pin(row, pins.get(row['member'], {}))
    result = dict(status='PASS', repository=repository, revision=revision,
                  verified_files=len(selection), verification='Every selected full WAV byte size and SHA-256 match independent mirror LFS pins')
    (directory / 'verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    return result


def read_texts(path):
    result = {}
    for line in Path(path).read_text(encoding='utf-8-sig').splitlines():
        if not line.strip():
            continue
        name, annotation = line.split(maxsplit=1)
        text = ''.join(c for c in annotation if '\u3400' <= c <= '\u9fff')
        if not text:
            continue
        if name in result and result[name]['text'] != text:
            raise ValueError('Conflicting text: ' + name)
        result[name] = dict(text=text, annotation=annotation)
    return result


def select(rows, texts, speakers, seconds_per_speaker=450, source_split='train'):
    groups = {}
    seen_pcm = set()
    for row in sorted(rows, key=lambda r: r['member']):
        if (row.get('official_split') != source_split or
            row.get('sample_rate') != 44100 or row.get('channels') != 1 or
            row.get('sample_width') != 2 or not 3 <= row.get('duration_s', 0) <= 15):
            continue
        name = PurePosixPath(row['member']).name
        if name not in texts or row['pcm_sha256'] in seen_pcm:
            continue
        seen_pcm.add(row['pcm_sha256'])
        item = dict(row, **texts[name])
        groups.setdefault(row['speaker'], []).append(item)
    for values in groups.values():
        values.sort(key=lambda r: rank(r['member']))
    eligible = {s for s, values in groups.items()
                if sum(v['duration_s'] for v in values) >= seconds_per_speaker}
    selected, used_speakers, used_texts = [], set(), set()
    # Ten percent additional validation data; preserve 5h of actual training.
    for split, quotas in [('validation', {'male': 1, 'female': 3}),
                          ('train', {'male': 8, 'female': 32})]:
        for gender, count in quotas.items():
            taken = 0
            for speaker in sorted(eligible - used_speakers, key=rank):
                if speakers.get(speaker, {}).get('gender') != gender:
                    continue
                chosen, local_texts, duration = [], set(), 0
                for row in groups[speaker]:
                    if row['text'] in used_texts or row['text'] in local_texts:
                        continue
                    chosen.append(dict(row, split=split, **speakers[speaker]))
                    local_texts.add(row['text'])
                    duration += row['duration_s']
                    if duration >= seconds_per_speaker:
                        break
                if duration < seconds_per_speaker:
                    continue
                selected.extend(chosen)
                used_texts.update(local_texts)
                used_speakers.add(speaker)
                taken += 1
                if taken == count:
                    break
            if taken != count:
                raise ValueError(f'Insufficient eligible {split}/{gender} speakers: {taken}/{count}')
    return selected


def convert(data, row, root, ffmpeg):
    if hashlib.sha256(data).hexdigest() != row['sha256']:
        raise ValueError('Source WAV changed: ' + row['member'])
    destination = root / 'audio' / row['split'] / row['speaker'] / PurePosixPath(row['member']).name
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, '-nostdin', '-hide_banner', '-loglevel', 'error', '-n',
               '-threads', '1', '-i', 'pipe:0', '-map_metadata', '-1',
               '-af', 'aresample=16000:resampler=soxr:precision=28:dither_method=none',
               '-ac', '1', '-ar', '16000', '-c:a', 'pcm_s16le', str(destination)]
    process = subprocess.run(command, input=data, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if process.returncode:
        raise RuntimeError(process.stderr.decode(errors='replace'))
    output = destination.read_bytes()
    info = wav_info(output)
    if (info['sample_rate'], info['channels'], info['sample_width']) != (16000, 1, 2):
        raise ValueError('Wrong output format')
    if abs(info['frames'] - row['frames'] * 16000 / 44100) > 1:
        raise ValueError('Resampling changed duration')
    with wave.open(io.BytesIO(output), 'rb') as wav:
        pcm = array.array('h', wav.readframes(wav.getnframes()))
    peak = max(abs(min(pcm)), abs(max(pcm)))
    if not peak:
        raise ValueError('Silent output')
    return dict(row, audio_path=destination.relative_to(root).as_posix(),
                output_bytes=len(output), output_sha256=hashlib.sha256(output).hexdigest(),
                output_pcm_sha256=info['pcm_sha256'], output_frames=info['frames'],
                output_sample_rate=16000, output_duration_s=info['duration_s'],
                peak_pcm16=peak, endpoint_samples=pcm.count(-32768)+pcm.count(32767))


def build(archive, audit_dir, output, ffmpeg, recover_verified_prefix=False, source_split='train'):
    archive, audit_dir, output = Path(archive), Path(audit_dir), Path(output)
    audit = json.loads((audit_dir / 'archive_audit.json').read_text(encoding='utf-8'))
    recover = (recover_verified_prefix and audit['status'] == 'FAIL' and
               len(audit['errors']) == 1 and audit['errors'][0]['type'] == 'EOFError')
    if (audit['status'] != 'PASS' and not recover) or archive.stat().st_size != audit['archive_bytes']:
        raise ValueError('Archive has not passed audit')
    rows = [json.loads(line) for line in (audit_dir / 'archive_members.jsonl').read_text(encoding='utf-8').splitlines()]
    texts = read_texts(audit_dir / f'metadata/{source_split}/content.txt')
    speakers = {}
    for line in (audit_dir / 'metadata/spk-info.txt').read_text(encoding='utf-8-sig').splitlines():
        fields = line.split()
        if len(fields) == 4 and fields[0].startswith('SSB'):
            speakers[fields[0]] = dict(age_group=fields[1], gender=fields[2], accent=fields[3])
    selection = select(rows, texts, speakers, source_split=source_split)
    output.mkdir(parents=True, exist_ok=False)
    source_verification = corroborate(selection, output / 'source_mirror_pins') if recover else None
    (output / 'selection.json').write_text(json.dumps(dict(seed=SEED, source_audit=audit,
        policy=f'Official {source_split} source; custom speaker split; 3-15s; 40 train + 4 validation speakers; 450s each; unique PCM/text; no gain/crop',
        rows=selection), ensure_ascii=False, indent=2), encoding='utf-8')
    (output / 'ffmpeg_version.txt').write_bytes(subprocess.check_output([ffmpeg, '-version']))
    lookup = {row['member']: row for row in selection}
    results, pending = [], []
    before = archive.stat()
    with ThreadPoolExecutor(max_workers=4) as pool:
        try:
            with tarfile.open(archive, mode='r|gz', bufsize=1024**2, ignore_zeros=True) as tar:
                for member in tar:
                    if member.name not in lookup:
                        continue
                    data = tar.extractfile(member).read()
                    pending.append(pool.submit(convert, data, lookup.pop(member.name), output, ffmpeg))
                    if len(pending) >= 8:
                        results.append(pending.pop(0).result())
                    if (len(results)+len(pending)) % 250 == 0:
                        print(json.dumps(dict(stage='resample', submitted=len(results)+len(pending),
                                              total=len(selection))), flush=True)
                    # Independent source-file pins, not the broken tar footer,
                    # establish admission in recovery mode. Do not read damage.
                    if recover and not lookup:
                        break
        except EOFError:
            if not recover or lookup:
                raise
        results.extend(f.result() for f in pending)
    if lookup:
        raise ValueError('Selected sources missing from archive')
    after = archive.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Source archive changed')
    results.sort(key=lambda r: (r['split'], r['speaker'], r['member']))
    summary = dict(status='PASS', seed=SEED, archive_sha256=audit['archive_sha256'],
                   source_archive_status=audit['status'],
                   official_source_split=source_split,
                   benchmark_limit='Official AISHELL-3 test is not an independent evaluation set after using it for reconstruction training' if source_split=='test' else None,
                   admission='INDIVIDUAL_WAV_MIRROR_SHA256' if recover else 'COMPLETE_ARCHIVE_AUDIT',
                   source_verification=source_verification,
                   output_format='16000 Hz mono PCM16 WAV; soxr precision28; no gain/crop; no dither',
                   splits={})
    for split in ('train', 'validation'):
        subset = [r for r in results if r['split'] == split]
        with (output / (split + '.jsonl')).open('x', encoding='utf-8') as out:
            for row in subset:
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
        summary['splits'][split] = dict(utterances=len(subset),
            speakers=len({r['speaker'] for r in subset}),
            duration_s=sum(r['output_duration_s'] for r in subset),
            audio_bytes=sum(r['output_bytes'] for r in subset),
            endpoint_samples=sum(r['endpoint_samples'] for r in subset))
    summary['speaker_overlap'] = len({r['speaker'] for r in results if r['split']=='train'} &
                                     {r['speaker'] for r in results if r['split']=='validation'})
    summary['text_overlap'] = len({r['text'] for r in results if r['split']=='train'} &
                                  {r['text'] for r in results if r['split']=='validation'})
    summary['pcm_overlap'] = len({r['output_pcm_sha256'] for r in results if r['split']=='train'} &
                                 {r['output_pcm_sha256'] for r in results if r['split']=='validation'})
    if summary['speaker_overlap'] or summary['text_overlap'] or summary['pcm_overlap']:
        raise ValueError('Split overlap')
    (output / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--audit-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--ffmpeg', required=True)
    parser.add_argument('--recover-verified-prefix', action='store_true',
                        help='Allow only complete WAVs externally SHA-256 corroborated after archive EOF failure')
    parser.add_argument('--source-split', choices=['train', 'test'], default='train')
    args = parser.parse_args()
    build(args.archive, args.audit_dir, args.output, args.ffmpeg,
          args.recover_verified_prefix, args.source_split)
