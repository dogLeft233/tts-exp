"""Collect bounded reused evidence and seal a completed experiment for archiving.

Copies evidence referenced by the 70 reused cells, not model weight bodies.
The original absolute paths in frozen receipts remain unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def read(path):
    return json.loads(Path(path).read_text(encoding='utf8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    with path.open('x', encoding='utf8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run.resolve()
    root = run.parents[1]
    for name in ['independent_audit.json', 'audio_independent_audit.json']:
        if read(run / name)['status'] != 'PASS':
            raise ValueError('Archive requires independent audit PASS')
    if (run / 'driver_exit.json').exists():
        if read(run / 'driver_exit.json')['status'] != 'COMPLETE':
            raise ValueError('Archive requires actual completed execution')
    else:
        completion = read(run / 'execution_completion_addendum.json')
        if completion['status'] != 'COMPLETE' or completion['returncode'] != 0:
            raise ValueError('Archive requires CPU index completion evidence')
        for stage in ['audio', 'render', 'official']:
            receipt = read(run / (stage + '_driver_exit.json'))
            if receipt['returncode'] != 0 or receipt['timed_out']:
                raise ValueError('Scientific stage incomplete: ' + stage)
    packet = run / 'reused_evidence'
    packet.mkdir(exist_ok=False)
    paths = set()
    index = read(run / 'comparison_index.json')
    for row in index['records']:
        if not row['reused']:
            continue
        paths.update([row['score_path'], row['audio_path'], row['source_path']])
        score = read(row['score_path'])
        paths.update(score[k] for k in ['video_path', 'canonical_audio_path', 'input_mkv',
                                       'syncnet_log', 'pipeline_log', 'tracks_pickle_path', 'activesd_path'])
        paths.update(score['checks'])
        paths.update(c['crop_file'] for c in score['crop_tracks'])
        for value in score.values():
            if isinstance(value, dict) and 'returncode' in value and 'log' in value:
                paths.add(value['log'])
        pretrack = Path(score['activesd_path']).parents[2] / 'pyavi' / score['reference'] / 'audio.wav'
        if pretrack.is_file():
            paths.add(str(pretrack))
    protocol = read(run / 'protocol.json')
    paths.add(protocol['downstream']['face'])
    for code_root in [root / 'third_party/Wav2Lip', root / 'third_party/syncnet_python']:
        paths.update(str(p) for p in code_root.rglob('*.py') if '__pycache__' not in p.parts)
    if protocol.get('predecessor_run'):
        predecessor = Path(protocol['predecessor_run'])
        paths.update(str(f) for f in predecessor.rglob('*') if f.is_file() and '__pycache__' not in f.parts)
    for sample in protocol['samples']:
        for key in ['input24_path', 'source_path', 'transcript_path']:
            if key in sample and Path(sample[key]).is_file():
                paths.add(sample[key])
    for record in protocol['weights_manifest']['files']:
        if Path(record['path']).suffix != '.safetensors':
            paths.add(str(Path(protocol['model_dir']) / record['path']))
    generation_manifests = [root / 'runs/qwen_full_tts_extension_20261022_attempt01/tts/tts_manifest.json',
                            root / 'runs/qwen_nrt_expansion40_20261022_attempt01/generation/tts_manifest.json']
    paths.update(str(p) for p in generation_manifests)
    sample_ids = {s['sample_id'] for s in protocol['samples']}
    for manifest_path in generation_manifests:
        for record in read(manifest_path)['records']:
            if record['sample_id'] in sample_ids:
                paths.update(v['path'] for v in record.values() if isinstance(v, dict) and 'path' in v)
    for path in protocol['source_pins']:
        p = Path(path)
        if p.suffix in ['.py', '.yaml', '.txt', '.json']:
            paths.add(path)
    mappings = []
    for path in sorted(paths):
        original = Path(path)
        if original.is_relative_to(run):
            continue
        if not original.is_relative_to(root):
            raise ValueError('Unexpected external evidence path ' + path)
        dest = packet / original.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, dest)
        source_hash = sha(original)
        if sha(dest) != source_hash:
            raise ValueError('Evidence copy mismatch')
        mappings.append(dict(original_path=path, archive_relative_path=str(dest.relative_to(run)),
                             sha256=source_hash, size=dest.stat().st_size))
    save(run / 'reused_evidence_map.json', dict(status='COPIED_AND_SHA_VERIFIED', records=mappings,
         model_weight_bodies_included=False,
         explanation='Frozen receipts retain original paths; this map locates copied dependencies inside the archive.'))
    # Include the actual analysis/audit/report/archiving sources and tests.
    snapshot = run / 'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in root.glob('scripts/experiments/qwen17_level_*20261023.py'):
        shutil.copy2(path, snapshot / path.name)
    for path in root.glob('tests/test_qwen17_level_*20261023.py'):
        shutil.copy2(path, snapshot / path.name)
    members = []
    for path in sorted(run.rglob('*')):
        if path.is_file() and '__pycache__' not in path.parts:
            members.append(dict(path=str(path.relative_to(run)), size=path.stat().st_size, sha256=sha(path)))
    save(run / 'artifact_manifest.json', dict(status='SEALED_AFTER_ALL_COMPUTE_AND_REPORT',
         files=members, count=len(members), bytes=sum(r['size'] for r in members),
         excludes=['artifact_manifest.json itself', 'later archive transport receipt', 'model weight bodies']))
    print(json.dumps(dict(reused_files=len(mappings), members=len(members), bytes=sum(r['size'] for r in members))))


if __name__ == '__main__':
    main()
