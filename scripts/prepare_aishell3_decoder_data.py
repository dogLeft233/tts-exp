"""Audit an existing AISHELL-3 tar.gz without unpacking it wholesale.

The archive is read through gzip EOF (including its CRC), and every regular
member is hashed. A failed archive is recorded and never admitted for selection.
"""
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile
import time
import wave


class HashReader:
    def __init__(self, source):
        self.source = source
        self.sha256 = hashlib.sha256()
        self.md5 = hashlib.md5()
        self.count = 0

    def read(self, size=-1):
        data = self.source.read(size)
        self.sha256.update(data)
        self.md5.update(data)
        self.count += len(data)
        return data


def wav_info(data):
    with wave.open(io.BytesIO(data), 'rb') as wav:
        frames = wav.getnframes()
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        pcm = wav.readframes(frames)
        if len(pcm) != frames * channels * width:
            raise ValueError('WAV PCM payload shorter than header')
        if wav.getcomptype() != 'NONE' or frames <= 0:
            raise ValueError('Empty or compressed WAV')
        return dict(frames=frames, sample_rate=rate, channels=channels,
                    sample_width=width, duration_s=frames / rate,
                    pcm_sha256=hashlib.sha256(pcm).hexdigest())


def audit(archive, output):
    archive, output = Path(archive), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    before = archive.stat()
    rows, seen, errors = [], set(), []
    status = dict(status='RUNNING', archive=str(archive.resolve()),
                  archive_bytes=before.st_size)
    hashed = None
    started = time.monotonic()
    try:
        with archive.open('rb') as raw:
            hashed = HashReader(raw)
            with gzip.GzipFile(fileobj=hashed, mode='rb') as gz:
                with tarfile.open(fileobj=gz, mode='r|', bufsize=1024**2,
                                  ignore_zeros=True) as tar:
                    for member in tar:
                        path = PurePosixPath(member.name)
                        if path.is_absolute() or '..' in path.parts or member.name in seen:
                            raise ValueError('Unsafe or duplicate member: ' + member.name)
                        seen.add(member.name)
                        if member.isdir():
                            continue
                        if not member.isfile():
                            raise ValueError('Unsupported tar member: ' + member.name)
                        data = tar.extractfile(member).read()
                        if len(data) != member.size:
                            raise ValueError('Truncated tar member: ' + member.name)
                        row = dict(member=member.name, bytes=len(data),
                                   sha256=hashlib.sha256(data).hexdigest())
                        if member.name.lower().endswith('.wav'):
                            row.update(wav_info(data))
                            row['speaker'] = path.parent.name
                            row['official_split'] = path.parts[0]
                        else:
                            target = output / 'metadata' / path
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(data)
                        rows.append(row)
                        if len(rows) % 2000 == 0:
                            print(json.dumps(dict(stage='audit', members=len(rows),
                                  compressed_bytes=hashed.count,
                                  elapsed_s=round(time.monotonic()-started))), flush=True)
                # tar stops at end-of-archive blocks; explicitly read gzip trailer.
                while gz.read(1024**2):
                    pass
            if raw.read(1):
                raise ValueError('Unread compressed bytes')
        after = archive.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('Source archive changed during audit')
        status['status'] = 'PASS'
    except Exception as exc:
        errors.append(dict(type=type(exc).__name__, message=str(exc)))
        print(json.dumps(dict(stage='archive_error', type=type(exc).__name__,
                              message=str(exc)), ensure_ascii=True), flush=True)
        status['status'] = 'FAIL'
    status.update(errors=errors, regular_members=len(rows),
                  wav_members=sum('frames' in r for r in rows),
                  wav_duration_s=sum(r.get('duration_s', 0) for r in rows),
                  bytes_read=hashed.count if hashed else 0,
                  archive_sha256=hashed.sha256.hexdigest() if hashed else None,
                  archive_md5=hashed.md5.hexdigest() if hashed else None,
                  elapsed_s=time.monotonic()-started)
    with (output / 'archive_members.jsonl').open('x', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    (output / 'archive_audit.json').write_text(
        json.dumps(status, ensure_ascii=True, indent=2), encoding='utf-8')
    print(json.dumps(status, ensure_ascii=True), flush=True)
    return status


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', required=True)
    parser.add_argument('--audit-dir', required=True)
    args = parser.parse_args()
    result = audit(args.archive, args.audit_dir)
    raise SystemExit(0 if result['status'] == 'PASS' else 1)
