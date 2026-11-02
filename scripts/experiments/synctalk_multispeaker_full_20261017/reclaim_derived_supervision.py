"""Retire only regenerable GT/torso caches after full F archive byte verification."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
import psutil
from flow_stream import sha
def run(root,identity,proof,report_name='supervision_reclamation.json'):
    assert report_name in ['supervision_reclamation.json','post_training_supervision_reclamation.json']
    record=json.loads(proof.read_text())
    manifest_path=root/'snapshots'/identity/'manifest.json'
    assert record['passed'] and record['all_archived_member_bytes_verified'] and record['identity']==identity
    assert sha(manifest_path)==record['manifest_sha256']
    manifest=json.loads(manifest_path.read_text())
    targets=[root/'prepared'/identity/split/kind for split in ['train','val','test'] for kind in ['gt_imgs','torso_imgs']]
    owner=(root/'prepared'/identity).resolve()
    for target in targets:
        assert target.is_dir() and not target.is_symlink() and target.resolve().is_relative_to(owner)
    expected={r['path']:r for r in manifest['files'] if any((root/r['path']).is_relative_to(t) for t in targets)}
    current={str(p.relative_to(root)) for t in targets for p in t.rglob('*') if p.is_file()}
    assert current==expected.keys()
    for p in psutil.process_iter():
        try:
            for f in p.open_files():assert not any(Path(f.path).is_relative_to(t) for t in targets),'active cache reader'
        except (psutil.NoSuchProcess,psutil.AccessDenied):continue
    for path,row in expected.items():
        p=root/path;assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']
    report=root/'snapshots'/identity/report_name;assert not report.exists()
    before=sum(x['bytes'] for x in expected.values())
    for target in targets:shutil.rmtree(target)
    result=dict(passed=True,identity=identity,removed=[str(x) for x in targets],reclaimed_bytes=before,
        persistent_archive=record['archive'],archive_sha256=record['archive_sha256'],
        all_archived_member_bytes_verified=True,original_images_tracks_and_geometry_retained=True,
        requires_restore_before_training=True,training_admitted=False)
    report.write_text(json.dumps(result,indent=2));print(json.dumps(result))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True);p.add_argument('--proof',type=Path,required=True);p.add_argument('--report-name',default='supervision_reclamation.json')
    a=p.parse_args();run(a.root,a.identity,a.proof,a.report_name)
