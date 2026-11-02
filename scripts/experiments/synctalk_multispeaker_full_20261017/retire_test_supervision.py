"""Retire only manifest-pinned heldout GT/torso after both seed deliveries are verified."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
from cohort_binding import identity_root,verify_member_pins

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def run(identity):
    assert identity in ['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers']
    inputs=identity_root(identity,ROOT,require_admission=True);verify_member_pins(ROOT,identity)
    base=inputs/'snapshots'/identity;proof=json.loads((base/'F_archive_verification.json').read_text())
    assert proof['passed'] and proof['all_archived_member_bytes_verified'] and proof['manifest_sha256']==sha(base/'manifest.json')
    for seed in [42,2026]:
        directory=ROOT/'media_manifests'/identity/f'seed{seed}'
        data=json.loads((directory/'F_delivery_verification.json').read_text());cloud=json.loads((directory/'cloud_readback_receipt.json').read_text())
        assert data['passed'] and cloud['passed'] and cloud['independent_linux_cloud_readback'] and data['archive_sha256']==cloud['sha256']
    output=base/'post_result_test_supervision_reclamation.json'
    if output.exists():assert json.loads(output.read_text())['passed'];return
    owners=[inputs/'prepared'/identity/'test'/kind for kind in ['gt_imgs','torso_imgs']]
    rows=[row for row in json.loads((base/'manifest.json').read_text())['files'] if any((inputs/row['path']).is_relative_to(owner) for owner in owners)]
    assert len(rows)==3000
    for row in rows:
        file=inputs/row['path'];assert file.resolve().is_relative_to((inputs/'prepared'/identity/'test').resolve()) and not file.is_symlink()
        assert file.stat().st_size==row['bytes'] and sha(file)==row['sha256']
    for row in rows:(inputs/row['path']).unlink()
    output.write_text(json.dumps(dict(passed=True,identity=identity,retired_files=3000,retired_bytes=sum(r['bytes'] for r in rows),
        verified_derived_archive_retained=True,both_seed_cloud_deliveries_verified=True),indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);run(p.parse_args().identity)
