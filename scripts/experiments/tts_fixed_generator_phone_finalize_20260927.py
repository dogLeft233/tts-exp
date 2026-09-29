"""Seal a completed, independently checked generator prototype trial."""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.experiments import tts_fixed_generator_phone_prototype_20260927 as core

RUN = core.OUT
AUDIT = ROOT / 'runs/tts_fixed_generator_phone_prototype_independent_audit_20260927'


def main():
    assert not (RUN / 'final.json').exists()
    protocol, rows = core.locked()
    assert core.read(RUN / 'analysis.json')['status'] == 'PENDING_INDEPENDENT'
    receipts = {}
    for name in ['static_receipt.json', 'identity_live_receipt.json', 'fit_receipt.json',
                 'calibration_live_receipt.json', 'result_receipt.json']:
        path = AUDIT / name
        assert core.read(path)['status'] == 'PASS', name
        receipts[str(path)] = core.sha(path)
    core.write(RUN / 'independent_validation.json', {
        'status': 'PASS', 'created_epoch': time.time(), 'receipts': receipts,
        'external_live_media_scope': '22 first-cal new AVI plus 2 old baseline videos; not all eval AVI',
        'evaluation_media_scope': '592 producer FFmpeg/PTS/JPEG checks at creation; independent cached-array and metadata checks',
    })
    subprocess.run([sys.executable, str(ROOT / 'scripts/experiments/tts_fixed_generator_phone_report_20260927.py')], check=True)
    compute = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    assert not compute, 'GPU must be released before conclusion'
    import fcntl
    with open('/tmp/tts-exp-gpu.lock', 'a') as lease:
        fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(lease, fcntl.LOCK_UN)
    temporary = Path('/dev/shm/tts_fixed_generator_phone_prototype_20260927')
    assert not [p for p in temporary.rglob('*') if p.is_file()]
    main_bytes = core.allocated(RUN)
    audit_bytes = core.allocated(AUDIT)
    free = shutil.disk_usage(RUN).free
    # Reserve 1 MiB for the full input/output hash manifest and final receipt.
    reserve = 2**20
    assert main_bytes + reserve <= core.CAP and audit_bytes <= core.AUDIT
    assert main_bytes + audit_bytes + reserve <= 320 * 2**20
    resources = core.limits(reserve)
    core.write(RUN / 'resource_closure.json', {
        'passed': True, 'created_epoch': time.time(), 'main_allocated_before_final': main_bytes,
        'audit_allocated': audit_bytes, 'final_metadata_reserved': reserve,
        'main_cap': core.CAP, 'audit_cap': core.AUDIT, 'total_cap': 320 * 2**20,
        'free': free, 'old_floor': core.FLOOR, 'remaining_commitment_check': resources,
        'gpu_compute_empty': True, 'gpu_lease_available': True, 'temporary_files': 0,
        'no_remaining_scientific_work': True,
    })
    hashes = dict(protocol['dependencies'])
    hashes.update(receipts)
    for path in RUN.rglob('*'):
        if path.is_file():
            hashes[str(path)] = core.sha(path)
    for name in ['tts_fixed_generator_phone_report_20260927.py', 'tts_fixed_generator_phone_finalize_20260927.py']:
        path = ROOT / 'scripts/experiments' / name
        hashes[str(path)] = core.sha(path)
    for path, digest in hashes.items():
        assert core.sha(path) == digest, path
    core.write(RUN / 'artifact_hashes.json', {'created_epoch': time.time(), 'files': hashes, 'checked': len(hashes)})
    core.write(RUN / 'final.json', {
        'status': 'concluded', 'created_epoch': time.time(), 'protocol_sha256': core.sha(RUN / 'protocol.json'),
        'report_sha256': core.sha(RUN / 'report.md'), 'summary_sha256': core.sha(RUN / 'summary.json.gz'),
        'event_summary_sha256': core.sha(RUN / 'event_summary.json.gz'),
        'independent_validation_sha256': core.sha(RUN / 'independent_validation.json'),
        'artifact_hashes_sha256': core.sha(RUN / 'artifact_hashes.json'),
        'resource_closure_sha256': core.sha(RUN / 'resource_closure.json'),
        'criteria': core.read(RUN / 'analysis.json')['criteria'], 'all74_cells': 592,
        'primary_clips': 71, 'primary_speakers': 15, 'legacy_queries': 1411,
    })
    print('CONCLUDED', core.sha(RUN / 'final.json'), flush=True)


if __name__ == '__main__':
    main()
