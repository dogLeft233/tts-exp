"""Retire only failed-pair SHM video copies after raw F and full failure evidence verification."""
import argparse,hashlib,json
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def run(root,proof,check_readers=True):
    assert root.name=='synctalk-multispeaker-data-20261017'
    identity=proof['identity'];assert identity in ['DebbieDingell','CathyMcMorrisRodgers'];folder=root/'material'/identity
    assert proof['passed'] and proof['failed_pair_evidence_independently_archived'] and proof['identity']==identity
    geometry=root/'prepared'/identity/'geometry_v2/geometry_qualification.json'
    assert sha(geometry)==proof['failed_geometry_sha256']
    result=json.loads(geometry.read_text());assert not result.get('passed') and any(r.get('passed') is False for r in result['splits'])
    original=json.loads((folder/'material_receipt.json').read_text());assert len(proof['files'])==2
    targets=[]
    for role in ['main','test']:
        record=next(r for r in original['recordings'] if r['role']==role)
        backed=next(r for r in proof['files'] if r['role']==role)
        target=folder/(role+'.mkv');assert target.is_file() and not target.is_symlink() and target.resolve().is_relative_to(folder.resolve())
        assert target.stat().st_size==record['bytes']==backed['bytes']
        assert sha(target)==record['sha256']==backed['sha256'];targets.append(target)
    if check_readers:
        import psutil
        for process in psutil.process_iter():
            try:assert not any(Path(f.path) in targets for f in process.open_files()),'active failed-pair video reader'
            except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    output=folder/'failed_pair_canonical_cache_retirement.json';assert not output.exists()
    size=sum(p.stat().st_size for p in targets)
    for p in targets:p.unlink()
    receipt=dict(passed=True,identity=identity,reclaimed_bytes=size,removed=[str(p) for p in targets],
        failure_evidence_archive_sha256=proof['failure_evidence_archive_sha256'],persistent_F_videos=proof['files'],
        failed_geometry_and_all_original_metadata_preserved=True,original_PCM_and_drivers_preserved=True,
        requires_restore_for_future_failed_pair_video_redecode=True,candidate_revision_untouched=True,training_admitted=False)
    output.write_text(json.dumps(receipt,indent=2));return receipt
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--proof',type=Path,required=True);a=p.parse_args();print(json.dumps(run(a.root,json.loads(a.proof.read_text()))))
