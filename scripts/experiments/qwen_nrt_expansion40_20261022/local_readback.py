"""Read back a sealed experiment archive into a fresh local backup directory."""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import tarfile


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    p.add_argument('archive_sha256')
    p.add_argument('manifest_sha256')
    args = p.parse_args()
    folder = args.archive.resolve().parent
    name = 'qwen_nrt_expansion40_20261022_attempt01'
    target = folder / name
    assert sha(args.archive) == args.archive_sha256
    assert not target.exists(), 'Never overwrite an existing backup'
    with tarfile.open(args.archive, 'r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            dest = (folder / member.name).resolve()
            assert dest == target or dest.is_relative_to(target)
            assert member.isdir() or member.isfile(), 'No symlinks or special files'
        archive.extractall(folder, members=members, filter='data')
    manifest_path = target / 'artifact_manifest.json'
    assert sha(manifest_path) == args.manifest_sha256
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    total = 0
    json_count = 0
    for row in manifest['files']:
        path = (target / row['path']).resolve()
        assert path.is_relative_to(target)
        assert path.stat().st_size == row['bytes']
        assert sha(path) == row['sha256'], str(path)
        total += row['bytes']
        if path.suffix.lower() == '.json':
            json.loads(path.read_text(encoding='utf-8'))
            json_count += 1
    assert len(manifest['files']) == manifest['file_count']
    assert total == manifest['total_bytes']
    actual = {str(f.relative_to(target)).replace('\\', '/') for f in target.rglob('*') if f.is_file()}
    assert actual == {r['path'] for r in manifest['files']} | {'artifact_manifest.json'}
    receipt = dict(status='PASS', utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   directory=str(target), file_count=manifest['file_count'], total_bytes=total,
                   parsed_JSON_count=json_count, archive_sha256=args.archive_sha256,
                   manifest_sha256=args.manifest_sha256, exact_file_inventory=True,
                   source_sha256=sha(Path(__file__).resolve()),
                   scope='All sealed bytes and JSON read back; not a model regeneration or human assessment')
    with (folder / 'local_readback_verification.json').open('x', encoding='utf-8') as f:
        json.dump(receipt, f, ensure_ascii=False, indent=2)
        f.write('\n')
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == '__main__':
    main()
