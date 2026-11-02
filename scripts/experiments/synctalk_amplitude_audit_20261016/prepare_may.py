"""Extract a bounded, hash-verified subset of the downloaded official May archive."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    selected = ['May/aud.wav', 'May/aud_ave.npy', 'May/bs.npy', 'May/bc.jpg',
                'May/transforms_train.json', 'May/transforms_val.json']
    splits = {}
    with zipfile.ZipFile(a.archive) as z:
        for split in ['train', 'val']:
            data = json.loads(z.read('May/transforms_' + split + '.json'))
            splits[split] = {'frames': len(data['frames']), 'first': data['frames'][0],
                             'last': data['frames'][-1]}
            for f in data['frames'][:4]:
                i = f['img_id']
                selected.extend([f'May/gt_imgs/{i}.jpg', f'May/ori_imgs/{i}.jpg',
                                 f'May/ori_imgs/{i}.lms', f'May/torso_imgs/{i}.png'])
        files = []
        for name in sorted(set(selected)):
            raw = z.read(name)
            dest = a.output / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(raw)
            files.append({'archive_member': name, 'bytes': len(raw),
                          'sha256': hashlib.sha256(raw).hexdigest()})
    receipt = {'archive': str(a.archive.resolve()), 'subset_only': True, 'files': files,
               'splits': splits, 'selection': 'Original complete audio/cache/config; first four original frames per split'}
    (a.output / 'subset_manifest.json').write_text(json.dumps(receipt, indent=2))
    print('subset_bytes', sum(f['bytes'] for f in files))


if __name__ == '__main__':
    main()
