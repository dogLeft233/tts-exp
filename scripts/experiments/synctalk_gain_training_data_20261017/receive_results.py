"""Verify the Linux bundle and materialize only new results on the local disk."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads((a.root / 'bundle_receipt.json').read_text())
    archive = a.root / receipt['file']
    assert sha(archive) == receipt['sha256']
    comparisons, extracted = 0, 0
    newline_differences = []
    with tarfile.open(archive) as t:
        for member in t.getmembers():
            if not member.isfile() and not member.islnk():
                continue
            if member.name.startswith('dataset/'):
                relative = member.name.removeprefix('dataset/')
                if relative == 'manifest.json':
                    target = a.root / 'remote_manifest.json'
                else:
                    data = t.extractfile(member).read()
                    local = (a.root / relative).read_bytes()
                    if data != local:
                        assert relative.endswith('.json') and data == local.replace(b'\r\n', b'\n'), relative
                        assert json.loads(data) == json.loads(local)
                        newline_differences.append({'path': relative, 'linux_sha256': hashlib.sha256(data).hexdigest(),
                                                    'windows_sha256': hashlib.sha256(local).hexdigest(),
                                                    'json_semantics_equal': True})
                    comparisons += 1
                    continue
            elif member.name == 'independent_qa.json':
                target = a.root / 'remote_independent_qa.json'
            else:
                target = a.root / member.name
            assert target.resolve().is_relative_to(a.root.resolve())
            data = t.extractfile(member).read()
            if target.exists():
                assert target.read_bytes() == data, target
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            extracted += 1
    result = {'bundle_sha256': receipt['sha256'], 'derived_files_compared_to_linux': comparisons,
              'new_result_files_extracted': extracted, 'passed': True,
              'json_newline_differences': newline_differences,
              'source_assets_and_wavs_require_byte_identity': True}
    (a.root / 'local_remote_join_qa.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
