"""Seal completed, independently audited outputs; archive outside the run."""
from pathlib import Path
import hashlib
import json
import tarfile
import time

ROOT = Path(__file__).resolve().parents[3]
RUN = ROOT / 'runs/qwen_nrt_expansion40_20261022_attempt01'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def main():
    assert not (RUN / 'artifact_manifest.json').exists()
    assert read(RUN / 'independent_generation_readback.json')['status'] == 'PASS'
    assert read(RUN / 'independent_downstream_audit.json')['status'] == 'PASS'
    assert read(RUN / 'downstream_completion.json')['new_video_count'] == 150
    assert read(RUN / 'asr/results.json')['count'] == 150
    assert read(RUN / 'decline_summary.json')['no_primary_sample_removed']
    assert (RUN / 'report.md').is_file()
    for stage in ('audio', 'render', 'extract', 'analyze', 'asr'):
        receipt = read(RUN / (stage + '_execution.json'))
        assert receipt['returncode'] == 0 and not receipt.get('timed_out', False)
    for row in read(RUN / 'helper_source_bindings.json')['files']:
        assert sha(ROOT / row['path']) == row['source_sha256']
    analysis = read(RUN / 'analysis.json')
    save(RUN / 'final_status.json', dict(
        status='COMPLETE_WITH_INDEPENDENT_NUMERICAL_READBACK', time_unix=time.time(),
        no_training=True, no_primary_sample_removed=True, primary_cohort='new30',
        measured={k: v['measurable_speakers'] for k, v in analysis['cohorts'].items()},
        two_main_new30_bonferroni_positive=analysis['two_main_new30_bonferroni_positive'],
        generated=dict(new_R=30, new_T=90, new_video=150, new_embedding_arrays=300),
        reused=dict(pilot_speakers=10, video=50, embedding_arrays=100),
        primary_ASR_count=150, pytest_passed=15,
        limits=['Development corpus; no guaranteed unseen population or upstream-training separation.',
                'No human listening/visual accuracy validation or independent full model regeneration.',
                'T-R includes prefix decoder-context differences; not pure LM-task causal attribution.'],
        protocol_sha256=sha(RUN / 'protocol.json'),
        numerical_audits={n: sha(RUN / n) for n in ['independent_generation_readback.json', 'independent_downstream_audit.json']},
        local_backup='C:/Users/dogle/.codex/tts-exp-artifacts/qwen-nrt-expansion40-20261022/',
        local_backup_status='Must be verified by separate local_readback_verification.json after transfer'))
    rows = []
    for path in sorted(RUN.rglob('*')):
        assert not path.is_symlink()
        if path.is_file():
            rows.append(dict(path=str(path.relative_to(RUN)), bytes=path.stat().st_size, sha256=sha(path)))
    manifest = dict(file_count=len(rows), total_bytes=sum(row['bytes'] for row in rows), files=rows)
    save(RUN / 'artifact_manifest.json', manifest)
    archive = Path('/tmp') / (RUN.name + '.tar.gz')
    with tarfile.open(archive, 'x:gz', compresslevel=3) as tf:
        tf.add(RUN, arcname=RUN.name)
    receipt = dict(archive=str(archive), bytes=archive.stat().st_size, archive_sha256=sha(archive),
                   manifest_sha256=sha(RUN / 'artifact_manifest.json'),
                   file_count=manifest['file_count'], total_bytes=manifest['total_bytes'])
    save(Path('/tmp') / (RUN.name + '_archive_receipt.json'), receipt)
    print(json.dumps(receipt, indent=2), flush=True)


if __name__ == '__main__':
    main()
