"""Preserve a rejected isolated preparation, including all stage1/material bytes."""
import argparse,json
from pathlib import Path
from cohort_binding import REMOTE,REVISIONS,sha


def run(root,identity):
    assert root == REMOTE/'dataset_revisions'/REVISIONS[identity] and root.resolve()==root
    worker=json.loads((root/'geometry_resume_worker_receipt.json').read_text())
    assert worker['passed'] is False and not worker['training_admitted']
    geometry=json.loads((root/'prepared'/identity/'geometry_v2/geometry_qualification.json').read_text())
    assert any(s.get('passed') is False for s in geometry['splits'])
    out=root/'failure_snapshot_attempt01';out.mkdir(exist_ok=False)
    paths=[];aliases=[]
    for name in ['material','preprocessed','prepared','executed_sources','diagnostics']:
        for path in (root/name).rglob('*'):
            if not path.is_file() or path.is_symlink():continue
            relative=path.relative_to(root)
            if name=='prepared' and any(x in ['ori_imgs','parsing'] for x in relative.parts):
                reference=root/'preprocessed'/Path(*relative.parts[1:])
                assert reference.is_file() and sha(reference)==sha(path)
                aliases.append(dict(path=relative.as_posix(),same_bytes_as=reference.relative_to(root).as_posix(),sha256=sha(path)))
                continue
            assert path.resolve().is_relative_to(root)
            paths.append(path)
    paths += [p for p in root.iterdir() if p.is_file() and not p.is_symlink() and p.suffix in ['.json','.py','.log','.sh']]
    rows=[dict(path=p.relative_to(root).as_posix(),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(set(paths))]
    manifest=dict(identity=identity,revision=root.name,geometry_qualified=False,training_admitted=False,
                  files=rows,byte_exact_stage1_aliases=aliases,
                  external_training_flow='independently verified F flow_cache_persistence_receipt.json; raw cache already retired')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'files.nul').write_bytes(b''.join(r['path'].encode()+b'\0' for r in rows))
    print(json.dumps(dict(files=len(rows),raw_bytes=sum(r['bytes'] for r in rows),passed=True)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True)
    a=p.parse_args();run(a.root,a.identity)
