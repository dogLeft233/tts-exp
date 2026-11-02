"""Seal derived arrays and supervision without duplicating persisted stage1 inputs."""
import argparse,hashlib,json
from pathlib import Path
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def run(root,identity,kind='derived'):
    assert kind in ['derived','qualification']
    out=root/('snapshots' if kind=='derived' else 'qualification_snapshots')/identity;out.mkdir(parents=True,exist_ok=False)
    rows=[]
    bases=[root/'prepared'/identity,root/'features'/identity] if kind=='derived' else [root/'prepared'/identity/'geometry_v2',root/'score_reference'/identity,root/'diagnostics'/identity]
    for base in bases:
        for p in sorted(base.rglob('*')):
            rel=p.relative_to(base)
            if any(x in ['ori_imgs','parsing','flow_masks'] for x in rel.parts):continue
            if p.is_file():rows.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)))
    assert rows and ((root/'prepared'/identity/'pose_qualification.json').exists() or (root/'prepared'/identity/'geometry_v2/geometry_qualification.json').exists())
    manifest=dict(identity=identity,files=rows,training_admitted=False,
        omitted_persisted_inputs=['ori_imgs','parsing'],regenerable_omission='flow_masks = 255-G of persisted parsing/*_face.png',
        source_sha256=sha(Path(__file__)))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'files.nul').write_bytes(b''.join(x['path'].encode()+b'\0' for x in rows))
    print(json.dumps(dict(identity=identity,files=len(rows),bytes=sum(x['bytes'] for x in rows),manifest_sha256=sha(out/'manifest.json'))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True);p.add_argument('--kind',choices=['derived','qualification'],default='derived')
    a=p.parse_args();run(a.root,a.identity,a.kind)
