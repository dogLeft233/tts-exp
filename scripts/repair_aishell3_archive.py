"""Complete a verified truncated prefix into a NEW archive; never edit source."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import time
import urllib.request

SOURCE = Path(r'E:\Documents\tts-exp\data\aishell3\data_aishell3.tgz')
DEST = Path(r'C:\Users\dogle\.codex\tts-exp-artifacts\aishell3-decoder5h-20261025\data_aishell3.repaired.tgz')
REVISION = '13ba8ea6d75d7e8cc3565f5bb2f6345f26fb2b93'
URL = f'https://huggingface.co/datasets/SMIIP-lab/AISHELL-3/resolve/{REVISION}/data_aishell3.tgz'
TOTAL = 19057141777
EXPECTED_SHA = 'be2507d431ad59419ec871e60674caedb2b585f84ffa01fe359784686db0e0cc'
PREFIX_SHA = '8a9d7647eb6bfbdbeaa812140829d768f9af77414ca8ef59996503c9f8f9ffad'


def fetch_range(start, end):
    expected_range = f'bytes {start}-{end}/{TOTAL}'
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL + f'?range={start}-{end}',
                                         headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=90) as response:
                if response.status != 206 or response.headers.get('Content-Range') != expected_range:
                    raise ValueError('Server did not honor exact range')
                data = response.read(end-start+2)
            if len(data) != end-start+1:
                raise ValueError('Range size mismatch')
            return data
        except Exception:
            if attempt == 3:
                raise
            time.sleep(2)


def main():
    before = SOURCE.stat()
    sha = hashlib.sha256()
    copied = 0
    if DEST.exists():
        if not before.st_size <= DEST.stat().st_size <= TOTAL:
            raise ValueError('Resume file has invalid size')
        prefix = hashlib.sha256()
        with DEST.open('rb') as previous:
            while copied < before.st_size:
                data = previous.read(min(8*1024**2, before.st_size-copied))
                prefix.update(data)
                sha.update(data)
                copied += len(data)
            if prefix.hexdigest() != PREFIX_SHA:
                raise ValueError('Resume prefix differs from audited source')
            for data in iter(lambda: previous.read(8*1024**2), b''):
                sha.update(data)
                copied += len(data)
        print(json.dumps(dict(stage='resume_verified_prefix', bytes=copied)), flush=True)
    else:
        with SOURCE.open('rb') as source, DEST.open('xb') as target:
            for data in iter(lambda: source.read(8*1024**2), b''):
                target.write(data)
                sha.update(data)
                copied += len(data)
            if sha.hexdigest() != PREFIX_SHA:
                raise ValueError('Local prefix differs from audited source')
    with DEST.open('ab') as target:
        chunk = 4*1024**2
        ranges = [(start, min(start+chunk-1, TOTAL-1))
                  for start in range(copied, TOTAL, chunk)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            for i in range(0, len(ranges), 24):
                batch = ranges[i:i+24]
                for data in pool.map(lambda pair: fetch_range(*pair), batch):
                    target.write(data)
                    sha.update(data)
                    copied += len(data)
                target.flush()
                print(json.dumps(dict(stage='download_tail', bytes=copied,
                                      total=TOTAL)), flush=True)
    after = SOURCE.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError('Original source changed')
    if copied != TOTAL or sha.hexdigest() != EXPECTED_SHA:
        raise ValueError('Completed archive failed pinned SHA-256')
    receipt = dict(status='PASS', original=str(SOURCE), original_bytes=before.st_size,
                   original_sha256=PREFIX_SHA, repaired=str(DEST), bytes=copied,
                   sha256=sha.hexdigest(), mirror_url=URL, revision=REVISION,
                   verification='Full archive matches Hugging Face LFS SHA-256; original unchanged')
    DEST.with_suffix('.repair_receipt.json').write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
