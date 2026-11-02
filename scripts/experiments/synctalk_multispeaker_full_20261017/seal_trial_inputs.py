"""Manifest all version-owned material or stage1 bytes without modifying originals."""
import argparse,hashlib,json
from pathlib import Path
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def run(root,identity,kind):
    assert root.name=='DebbieDingell_recording_roles_v2' and identity=='DebbieDingell'
    assert kind in ['material','stage1']
    base=root/('material' if kind=='material' else 'preprocessed')/identity
    receipt=json.loads((base/('independent_material_qa.json' if kind=='material' else 'stage1_receipt.json')).read_text());assert receipt['passed']
    if kind=='stage1':assert sum(s['frames'] for s in receipt['splits'])==8250
    paths=[p for p in sorted(base.rglob('*')) if p.is_file()]
    assert all(not p.is_symlink() and p.resolve().is_relative_to(base.resolve()) for p in paths)
    if kind=='stage1':assert len(paths)==41251
    out=root/'input_snapshots'/kind;out.mkdir(parents=True,exist_ok=False)
    rows=[dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=sha(p)) for p in paths]
    result=dict(identity=identity,kind=kind,dataset_revision=root.name,files=rows,training_admitted=False)
    (out/'manifest.json').write_text(json.dumps(result,indent=2));(out/'files.nul').write_bytes(b''.join(r['path'].encode()+b'\0' for r in rows))
    print(json.dumps(dict(passed=True,kind=kind,files=len(rows),bytes=sum(r['bytes'] for r in rows))))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--identity',required=True);p.add_argument('--kind',choices=['material','stage1'],required=True);a=p.parse_args();run(a.root,a.identity,a.kind)
