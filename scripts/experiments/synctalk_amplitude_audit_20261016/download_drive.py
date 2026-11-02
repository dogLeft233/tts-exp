"""Download the public official Drive assets; never accepts credentials."""
import argparse
import hashlib
import json
import re
from pathlib import Path

import requests


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--id', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.parent.mkdir(parents=True, exist_ok=True)
    url = 'https://drive.google.com/uc'
    session = requests.Session()
    r = session.get(url, params={'export': 'download', 'id': a.id}, timeout=40)
    r.raise_for_status()
    if 'text/html' in r.headers.get('content-type', ''):
        fields = dict(re.findall(r'name="([^"]+)" value="([^"]+)"', r.text))
        action = re.search(r'<form[^>]+action="([^"]+)"', r.text)
        if not action:
            raise RuntimeError('Public download unavailable: no download form')
        r = session.get(action.group(1), params=fields, stream=True, timeout=90)
        r.raise_for_status()
    if 'text/html' in r.headers.get('content-type', ''):
        raise RuntimeError('Download returned HTML, not an archive')
    h = hashlib.sha256()
    count = 0
    with a.output.with_suffix('.partial').open('wb') as f:
        for b in r.iter_content(1024 * 1024):
            count += len(b)
            if count > 2_000_000_000:
                raise RuntimeError('Asset exceeds frozen 2GB local download budget')
            f.write(b)
            h.update(b)
    a.output.with_suffix('.partial').rename(a.output)
    receipt = {'official_drive_id': a.id, 'bytes': count, 'sha256': h.hexdigest(),
               'source': f'https://drive.google.com/file/d/{a.id}/view',
               'path': str(a.output.resolve())}
    a.output.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
