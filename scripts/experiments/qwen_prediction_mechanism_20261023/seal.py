"""Preserve bounded dependencies and create a read-back SHA manifest/TAR."""
from pathlib import Path
import argparse,hashlib,json,sys,tarfile,time,shutil
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from scripts.experiments.qwen_prediction_mechanism_20261023.mechanism import read,save,sha

def main(run,archive):
    snapshot=run/'dependency_snapshot';snapshot.mkdir();mapping=[];seen={};candidates=set()
    bank=read(run/'B_protocol.json')
    for r in bank['rows']:
        candidates.add(r['codes']['path'])
        if 'record' in r:
            candidates.add(r['record']);rec=read(r['record'])
            candidates.update(rec[k]['path'] for k in ['new_codes','reference_codes','talker_sequences'])
    for s in bank['samples']:
        candidates.update(s[k] for k in ['input24_path','transcript_path','source_path'])
    prior=read(ROOT/'runs/qwen_fixed_coarse_20261023_attempt01/protocol.json')
    for directory in prior['model_dirs'].values():
        candidates.update(str(p) for p in Path(directory).glob('*.json'))
        candidates.update(str(p) for p in (Path(directory)/'speech_tokenizer').glob('*.json'))
    for r in prior['records'].values():candidates.add(r['path'])
    for model in ['L06','L17']:
        for f in read(run/f'A_{model}/pins.json')['model_source']:candidates.add(f)
    for f in [ROOT/'scripts/experiments/qwen_full_tts_extension_20261022/tts_stage.py',ROOT/'tests/test_qwen_prediction_mechanism_20261023.py']:
        candidates.add(str(f))
    if (run/'B_entropy_score_description.json').exists():candidates.update(read(run/'B_entropy_score_description.json')['pins'])
    if (run/'C_protocol.json').exists():
        p=read(run/'C_protocol.json')
        for s in p['samples']:candidates.update(s[k] for k in ['input24_path','transcript_path','source_path'])
        for r in p['rows']:
            if r['reuse_L06_baseline']:
                candidates.update([r['old_record'],r['old_score'],r['old_audio']['path'],r['old_audio']['source_path']])
                rec=read(r['old_record']);candidates.update(rec[k]['path'] for k in ['new_codes','reference_codes','talker_sequences'])
                score=read(r['old_score']);candidates.update([score[k] for k in ['video_path','input_mkv','tracks_pickle_path','activesd_path','pipeline_log','syncnet_log']]);candidates.update(score['checks'])
                candidates.update(str(f) for f in Path(score['activesd_path']).parent.glob('*.pckl'))
                candidates.update(t['metadata']['crop_file'] for t in score['tracks'])
    total=0
    for source in sorted(candidates):
        p=Path(source);assert p.is_file()
        size=p.stat().st_size;assert size<=32*1024**2,('dependency exceeds bounded small-file copy',p,size)
        digest=sha(p);total+=size;assert total<1024**3
        if digest not in seen:
            dest=snapshot/(digest[:16]+'_'+p.name);shutil.copyfile(p,dest);assert sha(dest)==digest;seen[digest]=str(dest.relative_to(run))
        mapping.append({'original_path':str(p),'snapshot_path':seen[digest],'sha256':digest,'bytes':size})
    save(run/'dependency_mapping.json',{'files':mapping,'deduplicated_files':len(seen),'input_bytes_before_deduplication':total,'weights':'not duplicated: exact original persistent model/checkpoint SHA pins retained'})
    sources=run/'final_source_snapshot';sources.mkdir()
    for f in Path(__file__).parent.glob('*.py'):shutil.copyfile(f,sources/f.name)
    entries=[]
    for p in sorted(run.rglob('*')):
        if p.is_file():entries.append({'path':str(p.relative_to(run)),'bytes':p.stat().st_size,'sha256':sha(p)})
    priorrun=run.parent/'qwen_prediction_mechanism_20261023_attempt01'
    failures=[{'path':str(p.relative_to(priorrun)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(priorrun.rglob('*')) if p.is_file()]
    save(run/'artifact_manifest.json',{'run':run.name,'files':entries,'failed_attempt':priorrun.name,'failed_attempt_files':failures,'scientific_products_frozen':True})
    archive=Path(archive);archive.parent.mkdir(parents=True,exist_ok=True)
    required=sum(e['bytes'] for e in entries)+sum(e['bytes'] for e in failures)
    assert required<20*1024**3 and shutil.disk_usage(archive.parent).free>required+2*1024**3
    with tarfile.open(archive,'x') as tar:
        tar.add(run,arcname=run.name);tar.add(priorrun,arcname=priorrun.name)
    observed={}
    with tarfile.open(archive,'r:') as tar:
        for m in tar:
            if not m.isfile():continue
            f=tar.extractfile(m);h=hashlib.sha256()
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
            observed[m.name]={'bytes':m.size,'sha256':h.hexdigest()}
    for name,items in [(run.name,entries),(priorrun.name,failures)]:
        for item in items:assert observed[name+'/'+item['path']]=={'bytes':item['bytes'],'sha256':item['sha256']}
    receipt={'status':'PASS','archive':str(archive),'sha256':sha(archive),'bytes':archive.stat().st_size,'readback_file_members':len(observed),'manifest_sha':sha(run/'artifact_manifest.json'),'all_manifest_members_read_back':True,'unix':time.time()}
    save(archive.with_suffix('.receipt.json'),receipt);print(json.dumps(receipt),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);ap.add_argument('--archive',required=True);a=ap.parse_args();main(Path(a.run),Path(a.archive))
