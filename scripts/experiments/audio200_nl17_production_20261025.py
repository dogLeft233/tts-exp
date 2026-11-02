"""Separate N/L17 data phase, preserving all200 speakers and600 sources.

Projects existing caches read-only; only missing N/L17 media are produced.
The SHA-bound original producer supplies the unmodified workers/downstream.
"""
from __future__ import annotations
import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
BACKEND=ROOT/'scripts/experiments/audio200_production_20261025.py'
BACKEND_SHA='e2693bac7d6e232108f5c6aebb2a78e598b69c69c70374fa51d7e592d183a50c'
PHASE='N_L17_source600_v1'
CAP=60*1024**3


def backend():
    import hashlib
    assert hashlib.sha256(BACKEND.read_bytes()).hexdigest()==BACKEND_SHA
    spec=importlib.util.spec_from_file_location('frozen_original_media_backend',BACKEND)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    return h


def select_four(rows):
    out=[copy.deepcopy(r) for r in rows if r['arm'] in ('N','L17')]
    assert len(out)==len({r['id'] for r in out})
    groups={}
    for r in out:groups.setdefault(r['sample_id'],[]).append(r)
    for rr in groups.values():
        assert {(r['arm'],r['draw']) for r in rr}=={('N',0),('L17',1),('L17',2),('L17',3)}
    return out


def projected_config(old,samples,cell):
    new=copy.deepcopy(old)
    assert old['arm']=='L17'
    new.update(samples=samples,expected_cells=3*len(samples),raw_manifest_path=str(cell/'raw_manifest.json'),analysis_protocol_path=str(cell/'analysis_protocol.json'))
    return new


def prepare(run):
    h=backend()
    oldrun=ROOT/'runs/audio200_production_20261025_attempt01'
    recoveryrun=ROOT/'runs/audio200_production_20261025_attempt02'
    oldpath=oldrun/'protocol.json';recoverypath=recoveryrun/'protocol.json'
    h.verify({str(oldpath):'f2d591056ec61f67ebe6bbf2b26a5260bc0b95eb24257de32e2882e34c61dc5e',
              str(recoverypath):'5b4cbeca0e971a226521cb9e7a559be00a4978488031b937be84f486f5289d29'})
    old=h.read(oldpath);original=h.read(recoverypath)
    sourcefile=Path(original['source_manifest_path']);sources={r['source_id']:r for r in h.read(sourcefile)['rows']}
    assert h.sha(sourcefile)==original['source_pins'][str(sourcefile)]
    assert len(sources)==600
    assert len({r['speaker_id'] for r in sources.values()})==200
    assert {s:sum(r['split']==s for r in sources.values()) for s in ('fit','development','confirmation')}=={'fit':300,'development':120,'confirmation':180}
    registry=ROOT/'runs/syncnet_audio_dataset40_20261024_attempt03/dataset280_consumer_manifest.json'
    cached=select_four(h.read(registry)['rows']);cache_origins=[dict(path=str(registry),sha256=h.sha(registry))]
    for b in original['batches'][:3]:
        manifest=Path(b['media_manifest_path']).parent/'features/fit/manifest.json'
        completion=manifest.parent.parent/'completion.json'
        rr=select_four(h.read(manifest)['rows']);cached.extend(rr)
        cache_origins.extend([dict(path=str(manifest),sha256=h.sha(manifest)),dict(path=str(completion),sha256=h.sha(completion))])
    assert len(cached)==248 and len({r['sample_id'] for r in cached})==62
    assert len({r['id'] for r in cached})==248
    for r in cached:
        source=sources[r['sample_id']]
        assert source['split']=='fit' and r['speaker_id']==source['speaker_id'] and h.clean(r['text'])==h.clean(source['text'])
        if 'split' in r:assert r['split']=='fit'
        else:r['split']='fit'
        for prefix in ('sequence','scoring_audio','replay','canonical_audio','video','score','activesd'):
            h.verify({r[prefix+'_path']:r[prefix+'_sha256']})
        h.verify({r['FA']['path']:r['FA']['sha256']})
    n10=[];n_group=original['batches'][3]['groups'][0];n_cell=Path(n_group['path'])
    assert n_group['arm']=='N'
    n_protocol=h.read(n_cell/'protocol.json')
    for s in n_protocol['samples']:
        p=n_cell/'official/scores'/f"{s['sample_id']}_N.json"
        obj=h.read(p);assert obj['status']=='COMPLETE'
        n10.append(dict(id=s['sample_id']+'_N_0',sample_id=s['sample_id'],speaker_id=s['speaker_id'],
            split=s['split'],arm='N',draw=0,text=h.clean(s['transcript']),score_path=str(p),score_sha256=h.sha(p),status='COMPLETE'))
    assert len(n10)==10
    origins=[]
    for stage in ('synthesize','canonical','render','official'):
        p=recoveryrun/'stage_receipts'/f'batch003_N_{stage}.json';o=h.read(p)
        assert o['status']=='COMPLETE';h.verify(o['artifacts'])
        origins.append(dict(path=str(p),sha256=h.sha(p)))
    assert not (run/'protocol.json').exists() and not run.exists()
    run.mkdir(parents=True)
    h.put(run/'reused248_manifest.json',dict(phase=PHASE,status='EXISTING_FOUR_ARM_CACHE_VERIFIED',rows=cached,models_loaded=0,new_forward=False,new_FA=False))
    h.put(run/'ready_media10_manifest.json',dict(phase=PHASE,status='READY_MEDIA_PENDING_FEATURES',rows=n10,origin_receipts=origins,new_generation=False,new_scoring=False))
    batches=[];group_diffs=[]
    base003=original['batches'][3]
    old_l17=next(g for g in base003['groups'] if g['arm']=='L17')
    old_config=h.read(Path(old_l17['path'])/'protocol.json')
    # No L17 draw existed in the paused old group; do not overlap another producer.
    assert not (Path(old_l17['path'])/'raw_manifest.json').exists()
    for name,subset in [('bridge003a',old_config['samples'][:1]),('batch003_tail',old_config['samples'][1:])]:
        cell=run/'batches/fit'/name/'L17';cell.mkdir(parents=True)
        old_analysis=Path(old_config['analysis_protocol_path'])
        with (cell/'analysis_protocol.json').open('xb') as f:f.write(old_analysis.read_bytes())
        config=projected_config(old_config,subset,cell)
        config['source_pins'][str(cell/'analysis_protocol.json')]=h.sha(cell/'analysis_protocol.json')
        h.put(cell/'protocol.json',config)
        samples=[s['sample_id'] for s in subset]
        batches.append(dict(id=name,original_batch_id='batch003',split='fit',source_ids=samples,
            groups=[dict(arm='L17',path=str(cell),protocol_sha256=h.sha(cell/'protocol.json'))],
            ready_media_ids=[s+'_N_0' for s in samples],media_manifest_path=str(cell.parent/'media_manifest.json')))
        group_diffs.append(dict(group_path=str(cell),original_group_protocol_path=str(Path(old_l17['path'])/'protocol.json'),
            original_group_protocol_sha256=old_l17['protocol_sha256'],subset_source_ids=samples,
            allowed_changes=['samples_subselection','expected_cells','raw_manifest_path','analysis_protocol_path','additional_exact_analysis_source_pin']))
    for b in original['batches'][4:]:
        bb=copy.deepcopy(b);bb['original_batch_id']=bb['id']
        bb['groups']=[g for g in bb['groups'] if g['arm'] in ('N','L17')]
        assert len(bb['groups'])==2
        bb['ready_media_ids']=[]
        folder='confirmation_sealed' if bb['split']=='confirmation' else bb['split']
        bb['media_manifest_path']=str(run/'batches'/folder/bb['id']/'media_manifest.json')
        batches.append(bb)
    assert len(batches)==55 and len({s for b in batches for s in b['source_ids']})==538
    assert not ({r['sample_id'] for r in cached}&{s for b in batches for s in b['source_ids']})
    assert {r['sample_id'] for r in cached}|{s for b in batches for s in b['source_ids']}==set(sources)
    stage_map={f"{b['id']}_{g['arm']}_{s}":dict(path=str(run/'stage_receipts'/f"{b['id']}_{g['arm']}_{s}.json"))
        for b in batches for g in b['groups'] for s in ('synthesize','canonical','render','official')}
    assert len(stage_map)==432
    consumer_map={b['id']:dict(path=str(run/'consumer_receipts'/f"{b['id']}.json")) for b in batches}
    pins={str(oldpath):h.sha(oldpath),str(recoverypath):h.sha(recoverypath),str(sourcefile):h.sha(sourcefile),str(Path(__file__)):h.sha(__file__),str(BACKEND):BACKEND_SHA}
    for p in cache_origins+origins:pins[p['path']]=p['sha256']
    protocol=dict(schema='audio200_nl17_producer_v1',phase=PHASE,status='PREPARED_NO_GPU',run=str(run),project_root=str(ROOT),
        expected_units=2400,cache_reused_units=248,ready_media_units=10,new_media_units=2142,new_feature_units=2152,new_tts_draws=1614,
        source600_manifest_path=str(sourcefile),source600_manifest_sha256=h.sha(sourcefile),
        split_freeze_path=str(ROOT/'runs/audio_data_inventory200_20261024_attempt05/source_split_freeze.json'),
        split_freeze_sha256='3ffb6d516470d5185a71e18d5bd8390b3180de7d6178640a2726c5eaef6fc749',
        original_producer_protocol_path=str(oldpath),original_producer_protocol_sha256=h.sha(oldpath),
        recovery_producer_protocol_path=str(recoverypath),recovery_producer_protocol_sha256=h.sha(recoverypath),
        reused_cache_manifest_path=str(run/'reused248_manifest.json'),reused_cache_manifest_sha256=h.sha(run/'reused248_manifest.json'),
        ready_media_manifest_path=str(run/'ready_media10_manifest.json'),ready_media_manifest_sha256=h.sha(run/'ready_media10_manifest.json'),
        work_batches=batches,reuse_cache_sources=cache_origins,ready_media_origin_receipts=origins,projected_group_diffs=group_diffs,
        stage_receipt_map=stage_map,consumer_receipt_map=consumer_map,material_roots=[str(oldrun),str(recoveryrun),str(run)],
        source_pins=pins,backend_source_path=str(BACKEND),backend_source_sha256=BACKEND_SHA,
        target_rms=original['target_rms'],seed_bases=original['seed_bases'],generation_kwargs=original['generation_kwargs'],
        non_streaming_mode=original['non_streaming_mode'],bundles={'L17':original['bundles']['L17']},
        aggregate_output_cap_bytes=CAP,minimum_free_bytes=12*1024**3,batch_budget_bytes=2*1024**3,
        no_training=True,excluded_model='L06 deferred; all original artifacts retained',confirmation_policy='mechanical preparation only; numerical outputs sealed',
        failure_policy='retain first failure and stop; no redraw/score selection or in-place retry')
    h.verify(pins);h.verify({protocol['split_freeze_path']:protocol['split_freeze_sha256']})
    h.put(run/'protocol.json',protocol)
    h.put(run/'cpu_preparation.json',dict(status='PASS_NL17_PROJECTED_PLAN_AND_CACHE_REUSE',phase=PHASE,
        protocol_sha256=h.sha(run/'protocol.json'),ready_cached_units=248,ready_media_pending_features=10,new_media=2142,
        planned_units=2400,planned_sources=600,planned_speakers=200,work_batches=55,stage_entries=432,
        old_raw_video_score_repeated=False,models_loaded=0,GPU_used=False,training=False,free_bytes=shutil.disk_usage(run).free))
    print(json.dumps(h.read(run/'cpu_preparation.json')),flush=True)


def collect(h,protocol,batch):
    ready={r['id']:r for r in h.read(protocol['ready_media_manifest_path'])['rows']}
    out=[ready[k] for k in batch['ready_media_ids']]
    for g in batch['groups']:
        cell=Path(g['path']);config=h.read(cell/'protocol.json');arm=g['arm']
        for s in config['samples']:
            for role in config['roles']:
                draw=0 if arm=='N' else int(role[-1]);score=cell/'official/scores'/f"{s['sample_id']}_{role}.json"
                obj=h.read(score);assert obj['status']=='COMPLETE'
                out.append(dict(id=f"{s['sample_id']}_{arm}_{draw}",sample_id=s['sample_id'],speaker_id=s['speaker_id'],split=s['split'],
                    arm=arm,draw=draw,text=h.clean(s['transcript']),score_path=str(score),score_sha256=h.sha(score),status='COMPLETE'))
    assert len(out)==4*len(batch['source_ids'])
    select_four(out)
    return out


def execute(run,release_path,contract):
    h=backend();p=h.read(run/'protocol.json');r=h.read(release_path)
    assert r['phase']==p['phase']==PHASE and r['allow_media_generation'] is True and r['allow_training'] is False
    assert r['producer_protocol_sha256']==h.sha(run/'protocol.json') and r['producer_source_sha256']==h.sha(__file__)
    h.verify(p['source_pins'])
    h.verify({p['reused_cache_manifest_path']:p['reused_cache_manifest_sha256'],p['ready_media_manifest_path']:p['ready_media_manifest_sha256']})
    allowed=set(r['allowed_source_ids']);batches=[b for b in p['work_batches'] if set(b['source_ids'])<=allowed]
    assert allowed and {s for b in batches for s in b['source_ids']}==allowed
    for name in ('.audio_data_supplement_gpu.lock','.syncnet_audio_dataset_cache_gpu.lock','.audio200_preparation_gpu.lock'):
        if (ROOT/'runs'/name).exists():raise RuntimeError('other pipeline lock present; WAIT outside launch '+name)
    h.LOCK.mkdir();h.put(h.LOCK/'owner.json',dict(pid=os.getpid(),run=str(run),utc=h.utc(),release_path=str(release_path),release_sha256=h.sha(release_path)))
    deadline=time.monotonic()+int(r.get('hard_timeout_seconds',10800));error=None
    attempt_dir=run/'executions';attempt_dir.mkdir(exist_ok=True);attempt=attempt_dir/f'{len(list(attempt_dir.iterdir()))+1:03d}';attempt.mkdir()
    h.put(attempt/'launch.json',dict(pid=os.getpid(),command=sys.argv,release_sha256=h.sha(release_path),phase=PHASE,utc=h.utc()))
    try:
        for b in batches:
            if sum(f.stat().st_size for root in p['material_roots'] for f in Path(root).rglob('*') if f.is_file())>CAP:raise RuntimeError('combined material cap')
            if shutil.disk_usage(run).free<p['minimum_free_bytes']+p['batch_budget_bytes']:raise RuntimeError('batch disk admission floor')
            h.event(run,'NL17_BATCH_START',batch=b['id'],sources=b['source_ids'])
            for g in b['groups']:
                arm=g['arm'];cell=Path(g['path'])
                assert arm in ('N','L17')
                command=[h.SYNC if arm=='N' else h.INTERPRETERS['L17'],str(BACKEND),'worker-synthesize','--run',str(cell),'--protocol-sha256',g['protocol_sha256']]
                h.run_stage(run,g,'synthesize',command,cell/'raw_manifest.json',deadline,gpu=arm!='N')
                for stage in ('canonical','render','official'):
                    output={'canonical':cell/'audio_manifest.json','render':cell/'video_manifest.json','official':cell/'official/summary.json'}[stage]
                    cmd=[h.SYNC,str(h.DOWNSTREAM),'--stage',stage,'--run',str(cell),'--protocol',str(cell/'protocol.json'),'--protocol-sha256',g['protocol_sha256']]
                    h.run_stage(run,g,stage,cmd,output,deadline,gpu=stage!='canonical')
            media=Path(b['media_manifest_path']);rows=collect(h,p,b)
            if not media.exists():h.put(media,dict(phase=PHASE,status='COMPLETE',rows=rows,source_ids=b['source_ids'],no_training=True))
            else:assert h.read(media)['rows']==rows
            consumer=Path(r['consumer_source_path'])
            assert h.sha(consumer)==r['consumer_source_sha256'] and h.sha(contract)==r['consumer_contract_sha256']
            receipt=run/'consumer_receipts'/f"{b['id']}.json"
            if receipt.exists():
                saved=h.read(receipt);assert saved['status']=='COMPLETE';h.verify(saved['artifacts'])
            else:
                h.wait_gpu(run,deadline);log=run/'logs'/f"{b['id']}_features.log";log.parent.mkdir(exist_ok=True)
                command=[h.SYNC,str(consumer),'consume','--manifest',str(media),'--output-root',str(media.parent/'features'),
                    '--contract',str(contract),'--release',str(release_path),'--parent-lock-owner',str(os.getpid())]
                with log.open('x') as f:result=subprocess.run(command,cwd=ROOT,env={**os.environ,**h.ENV},stdout=f,stderr=subprocess.STDOUT,timeout=10800)
                h.put(receipt,dict(status='COMPLETE' if result.returncode==0 else 'FAILED',returncode=result.returncode,command=command,
                    artifacts=h.file_seal(media.parent/'features') if result.returncode==0 else {},log=str(log)))
                if result.returncode:raise RuntimeError('NL17 consumer failed')
            h.event(run,'NL17_BATCH_COMPLETE',batch=b['id'],units=len(rows))
        h.put(attempt/'completion.json',dict(status='RELEASED_NL17_SCOPE_COMPLETE',source_ids=sorted(allowed),units=len(allowed)*4,no_training=True))
    except BaseException as e:
        error=repr(e);h.put(attempt/'failure.json',dict(status='FAILED_RETAINED_NO_RETRY',error=error,no_training=True));raise
    finally:
        (h.LOCK/'owner.json').unlink();h.LOCK.rmdir();h.event(run,'NL17_EXECUTION_EXIT',error=error,no_training=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','execute']);parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--release',type=Path);parser.add_argument('--consumer-contract',type=Path)
    a=parser.parse_args()
    if a.stage=='prepare':prepare(a.run)
    else:
        if not a.release or not a.consumer_contract:parser.error('exact --release and --consumer-contract required')
        execute(a.run,a.release,a.consumer_contract)
