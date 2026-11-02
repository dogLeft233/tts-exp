import hashlib,json
from pathlib import Path
root=Path('/dev/shm/synctalk-multispeaker-data-20261017')
ids=['SherrodBrown','PatrickLeahy','DebbieDingell','CathyMcMorrisRodgers']
for identity in ids:assert json.loads((root/'material'/identity/'independent_material_qa.json').read_text())['passed']
files=[]
for p in sorted((root/'material').rglob('*')):
    if not p.is_file():continue
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    files.append(dict(path=str(p.relative_to(root)),bytes=p.stat().st_size,sha256=h.hexdigest()))
result=dict(passed=True,files=files,identities=ids,training_admitted=False,
    scope='raw canonical material, exact audio gain, clocks, source preservation and cross-recording duplicate checks; official geometry/face coverage still pending')
(root/'material_delivery_manifest.json').write_text(json.dumps(result,indent=2))
print(json.dumps(dict(passed=True,files=len(files),bytes=sum(x['bytes'] for x in files),training_admitted=False)),flush=True)
