"""Package immutable science outputs; shared models and rebuildable caches stay put."""
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
from prepare import B,D,RUN,SOURCE,sha,write

def main():
    assert json.loads((RUN/'independent_audit.json').read_text())['status']=='PASS'
    assert json.loads((RUN/'summary.json').read_text())['status']=='COMPLETE_EXPLORATORY'
    assert json.loads((RUN/'independent_statistics_audit.json').read_text())['status']=='PASS'
    reuse=json.loads((RUN/'reuse_manifest.json').read_text())
    mutable={'cogenav/scores.json','cogenav/unmeasurable_faces03.json'}
    for rec in reuse['files']:
        assert sha(rec['parent_path'])==rec['sha256'],rec['parent_path']
        if rec['path'] not in mutable:assert sha(RUN/rec['path'])==rec['sha256'],rec['path']
    parent=Path(reuse['parent_run'])
    old=json.loads((parent/'cogenav/scores.json').read_text());current={r['id']:r for r in json.loads((RUN/'cogenav/scores.json').read_text())}
    assert all(current[r['id']]==r for r in old)
    write(RUN/'reuse_preservation_final.json',dict(status='PASS',unchanged_copied_scientific_files=len(reuse['files'])-len(mutable),cogenav_reused_scores_exact=len(old),parent_unchanged=True))
    pins=json.loads((RUN/'source_pins.json').read_text())
    for rel,h in pins['source'].items():assert sha(SOURCE/rel)==h
    for rel,h in pins['weights'].items():assert sha(SOURCE/'pctalk/checkpoints'/rel)==h
    for path,h in json.loads((RUN/'scorer_pins.json').read_text())['files'].items():assert sha(path)==h
    snapshot=RUN/'code_snapshot';snapshot.mkdir(exist_ok=True);bindings=[]
    sources=list(Path(__file__).parent.glob('*.py'))+list((B/'legacy_snapshot').glob('*.py'))
    sources += [B/n for n in ['run_scorer.sh','gpu_guard.py','score_avs_u.py','score_classifier.py','score_retrieval.py','score_peavs.py','score_syncreward.py']]
    sources += [p for p in [B/'scorers/source_manifest.json',B/'scorers/peavs_weight_manifest.json',B/'assets/peavs/peavs_extract_bootstrap.py'] if p.is_file()]
    sources += list((B/'assets').rglob('*.verified.json'))
    for i,p in enumerate(dict.fromkeys(sources)):
        dest=snapshot/f'{i:03d}_{p.name}';shutil.copyfile(p,dest)
        bindings.append(dict(original_path=str(p),archive_path=str(dest.relative_to(RUN)),sha256=sha(dest)))
    write(snapshot/'manifest.json',bindings)
    for name,env in [(n,B/'envs'/n) for n in ['eval-base','cogenav','synchformer','syncreward','peavs']]+[('pctalk',D/'env')]:
        freeze=subprocess.check_output([str(env/'bin/python'),'-m','pip','freeze'],text=True)
        (snapshot/f'environment-{name}.freeze.txt').write_text(freeze)
    files=[];links=[]
    for p in sorted(RUN.rglob('*')):
        rel=p.relative_to(RUN)
        if p.is_symlink():links.append(dict(path=str(rel),target=str(p.readlink())));continue
        if not p.is_file() or '__pycache__' in rel.parts:continue
        if any(str(rel).startswith('syncnet/work/'+folder+'/') for folder in ['pyframes','pytmp','pyavi']):continue
        if rel.parts[0]=='retrieval_fixtures' and p.suffix=='.jpg':continue
        if p.name in ['evidence_manifest.json','delivery_receipt.json']:continue
        files.append(dict(path=str(rel),bytes=p.stat().st_size,sha256=sha(p)))
    write(RUN/'evidence_manifest.json',dict(status='PASS',files=files,symlinks_not_archived=links,
        exclusions='Rebuildable SyncNet pyframes/pytmp/pyavi and retrieval JPEG caches, model links and Python bytecode. Raw matrices, crops/tracks, embeddings, logits, model outputs, all209 audio+videos included.'))
    archive=RUN.parent/(RUN.name+'_evidence.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for rec in files:tf.add(RUN/rec['path'],arcname=rec['path'],recursive=False)
        tf.add(RUN/'evidence_manifest.json',arcname='evidence_manifest.json',recursive=False)
    write(RUN/'delivery_receipt.json',dict(status='PASS',archive_path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),content_files=len(files)))
    print((RUN/'delivery_receipt.json').read_text())

if __name__=='__main__':main()
