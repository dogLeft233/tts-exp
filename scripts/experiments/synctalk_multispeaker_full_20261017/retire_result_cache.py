"""After independent cloud and F proofs, retire only the verified rendered-result cache."""
import argparse,hashlib,json
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()

def run(identity,seed):
    assert identity in ['SherrodBrown','PatrickLeahy','ChrisVanHollen','CathyMcMorrisRodgers'] and seed in [42,2026]
    base=ROOT/'media_manifests'/identity/f'seed{seed}'
    manifest=json.loads((base/'manifest.json').read_text());proof=json.loads((base/'F_delivery_verification.json').read_text())
    cloud=json.loads((base/'cloud_readback_receipt.json').read_text())
    assert proof['passed'] and proof['all_archived_member_bytes_verified'] and cloud['passed'] and cloud['independent_linux_cloud_readback']
    assert cloud['sha256']==proof['archive_sha256'] and proof['manifest_sha256']==sha(base/'manifest.json')
    report=base/'result_cache_retirement.json'
    if report.exists():
        prior=json.loads(report.read_text());assert prior['passed'] and prior['archive_sha256']==proof['archive_sha256'];print(json.dumps(prior));return
    owners=[(ROOT/kind/identity/f'seed{seed}').resolve() for kind in ['renders','scores','preview']]
    targets=[]
    for row in manifest['files']:
        file=ROOT/row['path']
        if file.suffix=='.json' or not any(file.resolve().is_relative_to(owner) for owner in owners):continue
        assert file.is_file() and not file.is_symlink() and file.stat().st_size==row['bytes'] and sha(file)==row['sha256']
        targets.append(row)
    for row in targets:(ROOT/row['path']).unlink()
    result=dict(passed=True,identity=identity,seed=seed,retired_files=len(targets),retired_bytes=sum(x['bytes'] for x in targets),
        archive_sha256=proof['archive_sha256'],
        formal_media_preserved_in_independently_verified_cloud_archive=True,all_score_JSON_and_training_states_unchanged=True)
    report.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();run(a.identity,a.seed)
