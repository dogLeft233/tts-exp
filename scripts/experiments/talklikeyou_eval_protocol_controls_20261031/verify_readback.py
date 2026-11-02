"""Verify every archived scientific member and its original input bindings."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile
import time


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--receipt', type=Path, required=True)
    ap.add_argument('--destination', type=Path, required=True)
    args = ap.parse_args()
    receipt = json.loads(args.receipt.read_text())
    assert sha(args.archive) == receipt['sha256']
    assert args.archive.stat().st_size == receipt['bytes']
    with tarfile.open(args.archive) as archive:
        members = archive.getmembers()
        roots = {Path(member.name).parts[0] for member in members}
        assert len(roots) == 1
        assert all(not Path(m.name).is_absolute() and '..' not in Path(m.name).parts
                   and not m.issym() and not m.islnk() for m in members)
        archive.extractall(args.destination, filter='data')
    root = args.destination/next(iter(roots))
    read = lambda path: json.loads(Path(path).read_text())
    manifest = read(root/'artifact_manifest.json')
    for row in manifest['files']:
        p = root/row['path']
        assert p.stat().st_size == row['bytes'] and sha(p) == row['sha256'], row['path']
    protocol = read(root/'protocol.json')
    for row in protocol['unique_rows']:
        assert sha(root/'source_inputs'/(row['id']+'.mp4')) == row['original_mp4_sha256']
        assert sha(root/'source_inputs'/(row['id']+'.wav')) == row['audio_sha256']
    # All stages existing before resume retain their exact original receipts.
    gate = read(root/'resume_gate_preflight.json')
    for name, expected in gate['stage_hashes'].items():
        assert sha(root/'stages'/name) == expected
    prefix = '/home/data2/wangjiajun/av-evaluation/scorers/syncnet/'
    for original, expected in protocol['source_pins'].items():
        if original.startswith(prefix) and original.endswith('.py'):
            assert sha(root/'upstream_sources'/original[len(prefix):]) == expected
    assert read(root/'independent_audit.json')['status'] == 'PASS'
    result = dict(status='PASS', archive_sha256=receipt['sha256'], archive_bytes=receipt['bytes'],
        verified_files=manifest['count'], verified_bytes=manifest['bytes'],
        verified_original_mp4_and_wav_pairs=len(protocol['unique_rows']),
        unchanged_pre_resume_stage_receipts=len(gate['stage_hashes']),
        verified_core_source_pins=True, ended=time.time())
    (root/'local_readback_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
