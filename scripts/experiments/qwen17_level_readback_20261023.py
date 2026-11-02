"""Read every archived member and check saved score CSV without extraction."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import tarfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, required=True)
    ap.add_argument('--run-name', required=True)
    ap.add_argument('--receipt', type=Path, required=True)
    args = ap.parse_args()
    digest = sha(args.archive)
    errors, member_count, verified_bytes = [], 0, 0
    with tarfile.open(args.archive, 'r:*') as tf:
        prefix = args.run_name + '/'
        def body(relative):
            handle = tf.extractfile(prefix + relative)
            if handle is None:
                raise ValueError('Missing member ' + relative)
            return handle.read()
        manifest_body = body('artifact_manifest.json')
        manifest = json.loads(manifest_body)
        members = {m.name: m for m in tf.getmembers() if m.isfile()}
        if len(members) != manifest['count'] + 1:
            errors.append('Unexpected file count')
        for record in manifest['files']:
            name = prefix + record['path']
            path = PurePosixPath(name)
            if path.is_absolute() or '..' in path.parts or name not in members:
                errors.append('Unsafe/missing member ' + name)
                continue
            info = members[name]
            h, size = hashlib.sha256(), 0
            with tf.extractfile(info) as f:
                for block in iter(lambda: f.read(1 << 20), b''):
                    h.update(block)
                    size += len(block)
            if size != record['size'] or h.hexdigest() != record['sha256']:
                errors.append('SHA/size mismatch ' + name)
            member_count += 1
            verified_bytes += size
        mapping = {r['original_path']: r['archive_relative_path'] for r in json.loads(body('reused_evidence_map.json'))['records']}
        protocol = json.loads(body('protocol.json'))
        remote_repo = PurePosixPath(protocol['model_dir']).parents[1]
        remote_run = remote_repo / 'runs' / args.run_name
        def locate(original):
            p = PurePosixPath(original)
            return str(p.relative_to(remote_run)) if p.is_relative_to(remote_run) else mapping[original]
        index = json.loads(body('comparison_index.json'))['records']
        rows = list(csv.DictReader(io.StringIO(body('all_draw_scores.csv').decode('utf8'))))
        csv_index = {(r['sample_id'], r['model'], r['level'], int(r['draw_index'])): r for r in rows}
        if len(rows) != len(csv_index) or len(rows) != len(index) or len(index) != 200:
            errors.append('Draw CSV/index count mismatch')
        for record in index:
            key = record['sample_id'], record['model'], record['level'], record['draw_index']
            row = csv_index[key]
            raw = body(locate(record['score_path']))
            if hashlib.sha256(raw).hexdigest() != record['score_sha256']:
                errors.append('Archived index receipt SHA mismatch ' + str(key))
            score = json.loads(raw)
            for col, field in [('Sync-C', 'sync_c'), ('Sync-D', 'sync_d'), ('offset_frames', 'av_offset')]:
                if abs(float(row[col]) - score[field]) > 1e-12:
                    errors.append('CSV score mismatch ' + str(key))
            if row['audio_sha256'] != record['audio_sha256']:
                errors.append('CSV audio binding mismatch ' + str(key))
        for audit in ['independent_audit.json', 'audio_independent_audit.json', 'generation_independent_audit.json']:
            if json.loads(body(audit))['status'] != 'PASS':
                errors.append('Archived audit not PASS ' + audit)
    if sha(args.archive) != digest:
        errors.append('Archive changed during readback')
    result = dict(status='PASS' if not errors else 'FAIL', errors=errors, archive=str(args.archive.resolve()),
                  archive_sha256=digest, archive_bytes=args.archive.stat().st_size,
                  artifact_manifest_sha256=hashlib.sha256(manifest_body).hexdigest(),
                  members_rehashed=member_count, payload_bytes_rehashed=verified_bytes,
                  draw_csv_rows_checked=len(rows), no_extraction=True,
                  model_weight_bodies_included=False, cloud_backup_claimed=False,
                  readback_source_sha256=sha(__file__))
    with args.receipt.open('x', encoding='utf8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'PASS' else 1)


if __name__ == '__main__':
    main()
