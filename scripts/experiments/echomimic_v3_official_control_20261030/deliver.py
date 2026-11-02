"""Verify an immutable downloaded diagnostic archive and independently read scores."""
from pathlib import Path,PurePosixPath
import argparse
import hashlib
import json
import shutil
import tarfile
import numpy as np


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(8*1024**2):h.update(b)
    return h.hexdigest()


def read(p):return json.loads(Path(p).read_text(encoding='utf8'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output;receipt=read(out/'delivery_receipt.json');archive=out/'evidence.tar.gz'
    assert sha(archive)==receipt['sha256'] and archive.stat().st_size==receipt['bytes']
    root=(out/'evidence').resolve();root.mkdir(exist_ok=True)
    with tarfile.open(archive) as tf:
        for member in tf:
            name=PurePosixPath(member.name)
            assert member.isfile() and not name.is_absolute() and '..' not in name.parts
            dest=(root/str(name)).resolve();assert dest.is_relative_to(root)
            dest.parent.mkdir(parents=True,exist_ok=True)
            data=tf.extractfile(member).read()
            if dest.exists():assert sha(dest)==hashlib.sha256(data).hexdigest()
            else:dest.write_bytes(data)
    assert sha(root/'evidence_manifest.json')==receipt['manifest_sha256']
    manifest=read(root/'evidence_manifest.json')
    for record in manifest['files']:
        dest=root/record['path'];assert dest.stat().st_size==record['bytes'] and sha(dest)==record['sha256']
    scores=read(root/'scores.json');plan=read(root/'plan.json');result=read(root/'paired_comparison.json')
    lookup={(r['condition'],r['id']):r for r in scores};no_op=[]
    for r in plan['rows']:
        if r['expected_frames']>81:continue
        base=read(root/'prefix_pixel_audits'/('baseline__'+r['id']+'.json'))
        capped=read(root/'prefix_pixel_audits'/('cap_only__'+r['id']+'.json'))
        assert base['clipped_decoded_prefix_sha256']==capped['clipped_decoded_prefix_sha256']
        assert abs(lookup['baseline',r['id']]['C']-lookup['cap_only',r['id']]['C'])<1e-5
        no_op.append(r['id'])
    errors=[]
    for r in scores:
        matrix=np.load(root/'syncnet'/(r['condition']+'__'+r['id'])/'distances.npy').astype(float)
        curve=matrix.mean(0);c=np.median(curve)-curve.min();errors.append(abs(c-r['C']))
        assert errors[-1]<1e-5
    for arm,summary in result['conditions'].items():
        rows=[r for r in result['pairs'] if r['condition']==arm]
        fields=['N','T','T_minus_N','N_change','T_change','difference_in_differences']
        values=np.array([[r[k] for k in fields] for r in rows])
        for j,k in enumerate(fields):assert abs(values[:,j].mean()-summary[k])<1e-12
        rng=np.random.default_rng(20261030);indices=rng.integers(0,6,(10000,6))
        bootstrap=np.stack([values[index].mean(0) for index in indices])
        np.testing.assert_allclose(np.quantile(bootstrap,[.025,.975],axis=0),summary['descriptive_bootstrap_ci95'],atol=1e-12,rtol=0)
    old=out.parent/'chinese20_rms24_20261030_attempt01/evidence.tar.gz'
    assert sha(old)=='ca8cd17cb6974016e69e3470c9cd01fb9927bb2d9d0b8eefca2f51ff3f09380b'
    for name in ['paired_comparison.json','paired_comparison.csv','scores.json','report.md','comparison.png','comparison.pdf','contact_sheet.png','independent_audit.json','offset_control_summary.json','plan.json','all41_lufs_audit.json']:
        dest=out/name
        if dest.exists():assert sha(dest)==sha(root/name)
        else:shutil.copyfile(root/name,dest)
    verification=dict(status='PASS',archive_sha256=sha(archive),verified_files=len(manifest['files']),
        score_count=len(scores),max_raw_reduction_error=max(errors),bootstrap_recomputed=True,
        original_evidence_archive_unchanged=True,cap_no_op_pixel_and_score_controls=no_op)
    with (out/'local_readback.json').open('x',encoding='utf8') as f:json.dump(verification,f,indent=2)
    print(json.dumps(verification,indent=2))


if __name__=='__main__':main()
