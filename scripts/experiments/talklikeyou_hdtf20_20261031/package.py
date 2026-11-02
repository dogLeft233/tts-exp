"""Freeze complete selected evidence; exclude only rebuildable visual caches."""
from pathlib import Path
import json,shutil,subprocess,tarfile
from prepare import B,D,RUN,DATA,sha,write

def main():
    assert json.loads((RUN/'independent_audit.json').read_text())['status']=='PASS'
    previous=RUN/'scorer_runtime_comparison.json'
    shutil.copyfile(previous,RUN/'scorer_runtime_comparison_before_scoring.json')
    subprocess.run([str(B/'envs/eval-base/bin/python'),str(Path(__file__).with_name('verify_runtime.py'))],check=True)
    assert sha(previous)==sha(RUN/'scorer_runtime_comparison_before_scoring.json')
    shutil.copytree(DATA,RUN/'input_assets')
    sources=list(Path(__file__).parent.glob('*.py'))+list((B/'legacy_snapshot').glob('*.py'))
    sources += [B/name for name in ['run_scorer.sh','gpu_guard.py','score_avs_u.py','score_classifier.py','score_retrieval.py','score_peavs.py','score_syncreward.py']]
    sources += [D/name for name in ['source_receipt.json','source_verification.json','download_receipt.json','environment.freeze.txt']]
    sources += list((D/'TalkLikeYou/talklikeyou').rglob('*.py'))+list((D/'TalkLikeYou/src').rglob('*.py'))
    sources += [B/'assets/peavs/peavs_extract_bootstrap.py']
    sources += [p for p in [B/'deployment_validation.json',B/'data_validation.json',B/'scorers/source_manifest.json',B/'scorers/peavs_weight_manifest.json'] if p.is_file()]
    sources += list((B/'assets').rglob('*.verified.json'))
    sources += list((B/'scorers/syncnet').glob('*.py'))
    bindings=[]
    for index,p in enumerate(dict.fromkeys(sources)):
        dest=RUN/'code_snapshot'/f'{index:03d}_{p.name}';dest.parent.mkdir(exist_ok=True);shutil.copyfile(p,dest)
        bindings.append(dict(original_path=str(p),archive_path=str(dest.relative_to(RUN)),sha256=sha(dest)))
    write(RUN/'code_snapshot/manifest.json',bindings)
    for env in ['eval-base','cogenav','synchformer','syncreward','peavs']:
        (RUN/'code_snapshot'/('environment-'+env+'.freeze.txt')).write_text(subprocess.check_output([str(B/'envs'/env/'bin/python'),'-m','pip','freeze'],text=True))
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
      exclusions='Rebuildable SyncNet frames/tmp/AVI and retrieval JPEG only; all native media, PCM, raw matrices/features, logits, cloud responses/provider outputs included.'))
    archive=RUN.parent/(RUN.name+'_evidence.tar.gz');assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for rec in records:tf.add(RUN/rec['path'],arcname=rec['path'],recursive=False)
        tf.add(RUN/'evidence_manifest.json',arcname='evidence_manifest.json')
    write(RUN/'delivery_receipt.json',dict(status='PASS',archive_path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(records)+1,manifest_sha256=sha(RUN/'evidence_manifest.json')))
    print((RUN/'delivery_receipt.json').read_text())
if __name__=='__main__':main()
