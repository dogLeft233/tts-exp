"""Archive raw evidence and all80 media without copying model weights or caches."""
from pathlib import Path
import argparse, hashlib, json, shutil, subprocess, tarfile
from prepare import B, RUN, sha, write

def main():
    assert json.loads((RUN/'independent_audit.json').read_text())['status']=='PASS'
    assert (RUN/'summary.json').exists()
    preserved=json.loads((RUN/'resume02_preserved_science.json').read_text())['files']
    assert all(sha(RUN/p)==value for p,value in preserved.items()), 'Completed science changed during resume'
    write(RUN/'resume02_preservation_audit.json',dict(status='PASS',unchanged_completed_files=len(preserved),
        scope='Environment-only wrapper repair; seven completed scorer artifacts and input binding unchanged'))
    parent=Path(json.loads((RUN/'protocol.json').read_text())['parent_run'])
    assert sha(RUN/'syncnet/scores.json')==sha(parent/'syncnet/scores.json')
    # Carry completed parent tracks/crops as immutable evidence, without new inference.
    for name in ['pywork','pycrop']:
        target=RUN/'syncnet/work'/name
        assert not target.exists()
        shutil.copytree(parent/'syncnet/work'/name,target)
    sources=list(Path(__file__).parent.glob('*.py'))+list((B/'legacy_snapshot').glob('*.py'))
    sources += [B/name for name in ['run_scorer.sh','gpu_guard.py','score_avs_u.py','score_classifier.py','score_retrieval.py','score_peavs.py','score_syncreward.py']]
    sources = list(dict.fromkeys(sources))
    sources += [B/'assets/peavs/peavs_extract_bootstrap.py']
    sources += [p for p in [B/'deployment_validation.json',B/'data_validation.json',B/'scorers/source_manifest.json',B/'scorers/peavs_weight_manifest.json'] if p.is_file()]
    sources += list((B/'assets').rglob('*.verified.json'))
    face=Path(json.loads((RUN/'protocol.json').read_text())['generation']['face_path'])
    face_dest=RUN/'input_assets/3.png';face_dest.parent.mkdir(exist_ok=True);shutil.copyfile(face,face_dest)
    assert sha(face_dest)==json.loads((RUN/'protocol.json').read_text())['generation']['face_sha256']
    bindings=[]
    for index,p in enumerate(sources):
        dest=RUN/'code_snapshot'/f'{index:03d}_{p.name}';dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
        bindings.append(dict(original_path=str(p),archive_path=str(dest.relative_to(RUN)),sha256=sha(dest)))
    write(RUN/'code_snapshot/manifest.json',bindings)
    for env in ['eval-base','cogenav','synchformer','syncreward','peavs']:
        text=subprocess.check_output([str(B/'envs'/env/'bin/python'),'-m','pip','freeze']).decode()
        (RUN/'code_snapshot'/('environment-'+env+'.freeze.txt')).write_text(text)
    records=[];links=[]
    for p in sorted(RUN.rglob('*')):
        rel=p.relative_to(RUN)
        if p.is_symlink():links.append(dict(path=str(rel),target=str(p.readlink())));continue
        if not p.is_file():continue
        if any(str(rel).startswith('syncnet/work/'+name+'/') for name in ['pyframes','pytmp','pyavi']):continue
        if rel.parts[0]=='retrieval_fixtures' and p.suffix=='.jpg':continue
        if p.name in ['evidence_manifest.json','delivery_receipt.json']:continue
        records.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=sha(p)))
    write(RUN/'evidence_manifest.json',dict(status='PASS',files=records,symlinks_not_archived=links,
        exclusions='Rebuildable SyncNet decoded frames/tmp/AVI and retrieval JPEG caches only; raw matrices, crops/track metadata, embeddings, logits, PEAVS features, audio and all80 generated/scoring videos included. Model weights remain pinned in deployed private assets.'))
    archive=RUN.parent/(RUN.name+'_evidence.tar.gz')
    assert not archive.exists(),'Never overwrite a delivered evidence archive'
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for rec in records:tf.add(RUN/rec['path'],arcname=rec['path'],recursive=False)
        tf.add(RUN/'evidence_manifest.json',arcname='evidence_manifest.json',recursive=False)
    result=dict(status='PASS',archive_path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(records)+1,
        manifest_sha256=sha(RUN/'evidence_manifest.json'))
    write(RUN/'delivery_receipt.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
