"""Seal an unqualified preparation attempt without modifying its evidence."""
import argparse,hashlib,json,subprocess
from pathlib import Path
ROOT=Path('/dev/shm/synctalk-multispeaker-data-20261017')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def run(identity='DebbieDingell'):
    assert identity in ['DebbieDingell','CathyMcMorrisRodgers']
    out=ROOT/'failure_snapshots'/(identity+'_recording_pair_v1');out.mkdir(parents=True,exist_ok=False)
    geo=json.loads((ROOT/'prepared'/identity/'geometry_v2/geometry_qualification.json').read_text())
    assert not geo.get('passed') and any(s.get('passed') is False for s in geo['splits'])
    bases=[ROOT/'prepared'/identity,ROOT/'diagnostics'/identity]
    files=[]
    for base in bases:
        for p in sorted(base.rglob('*')):
            if not p.is_file() or p.is_symlink():continue
            if any(x in ['ori_imgs','parsing','flow_masks'] for x in p.relative_to(base).parts):continue
            files.append(p)
    worker='stage2_v2_worker_receipt.json' if identity=='DebbieDingell' else 'stage2_v2_Cathy_independent_worker_receipt.json'
    files += [ROOT/worker,ROOT/f'stage2_v2_{identity}_geometry.log',ROOT/'selection.json',ROOT/'preprocessing_contract_v2.json',Path(__file__)]
    files += list((ROOT/'material'/identity).glob('*.json'))
    files += list((ROOT/'executed_sources').rglob('*.py'))
    files=sorted(set(files))
    manifest=dict(identity=identity,dataset_revision='recording_pair_v1',
        archive_verified=False,geometry_qualified=False,training_admitted=False,
        files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],
        external_dependencies=[f'F material/{identity} and original material_delivery_manifest.json',
            f'F preprocessed_packages/{identity}_stage1.tar.gz and manifest',
            f'F flow_caches/{identity}/train/rigid_flow.npy.gz and persistence.json'],
        omitted_cache='ori_imgs/parsing/flow_masks: persisted stage1 inputs and exact derived masks; training dense flow already F-persisted and retired')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'files.nul').write_bytes(b''.join(str(p.relative_to(ROOT)).encode()+b'\0' for p in files))
    print(json.dumps(dict(files=len(files),bytes=sum(x['bytes'] for x in manifest['files']),geometry_qualified=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--identity',default='DebbieDingell');a=p.parse_args();run(a.identity)
