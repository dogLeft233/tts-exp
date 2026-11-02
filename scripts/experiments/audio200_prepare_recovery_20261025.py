"""CPU-only reviewed repair: route one guard-blocked official stage to a new run.

Preserves every scientific input, output, original failure and old receipt.
No models, GPU work, resampling, retries or media regeneration.
"""
from __future__ import annotations
import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[2]
OLD_SHA='f2d591056ec61f67ebe6bbf2b26a5260bc0b95eb24257de32e2882e34c61dc5e'
PRODUCER_SHA='e2693bac7d6e232108f5c6aebb2a78e598b69c69c70374fa51d7e592d183a50c'
BAD_BATCH='batch003'
ARMS=('N','L06','L17')
STAGES=('synthesize','canonical','render','official')


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):
            h.update(b)
    return h.hexdigest()


def put(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
        f.write('\n')


def verify(pins):
    for p,h in pins.items():
        if sha(p)!=h:
            raise ValueError('bound input/output drift '+str(p))


def exact_copy(old,new):
    old,new=Path(old),Path(new)
    new.parent.mkdir(parents=True,exist_ok=True)
    with new.open('xb') as f:
        f.write(old.read_bytes())
    assert sha(old)==sha(new)
    return sha(new)


def guard_only_failure(summary,terminal,receipt):
    assert summary['status']=='INCOMPLETE'
    assert (summary['expected'],summary['completed'],summary['failed'],summary['pending'])==(10,0,0,10)
    assert receipt['status']=='FAILED' and receipt['returncode']==1
    assert terminal['type']=='RuntimeError'
    assert terminal['message'].startswith('other GPU process present:')
    assert 'self.assert_gpu_idle()' in terminal['traceback']
    assert 'runner.score()' in terminal['traceback']
    assert summary['terminal_error']==terminal


def check_protocol_diff(old,new,recovery_path,recovery_sha):
    expected=copy.deepcopy(old)
    newroot=Path(new['run'])
    b=expected['batches'][3]
    assert b['id']==BAD_BATCH and b['split']=='fit'
    oldgroup=b['groups'][0]
    assert oldgroup['arm']=='N'
    oldgroup['path']=str(newroot/'batches/fit/batch003/N')
    b['media_manifest_path']=str(newroot/'batches/fit/batch003/media_manifest.json')
    expected.update(run=str(newroot),original_protocol_path=str(Path(old['run'])/'protocol.json'),
        original_protocol_sha256=OLD_SHA,material_roots=[old['run'],str(newroot)],
        reviewed_recovery_path=str(recovery_path),reviewed_recovery_sha256=recovery_sha)
    assert new==expected, 'recovery changes more than exact reviewed routes and provenance'


def prepare(oldrun,newrun,oldcontroller,newcontroller):
    oldrun,newrun,oldcontroller,newcontroller=(Path(p) for p in (oldrun,newrun,oldcontroller,newcontroller))
    assert sha(oldrun/'protocol.json')==OLD_SHA
    assert sha(ROOT/'scripts/experiments/audio200_production_20261025.py')==PRODUCER_SHA
    assert oldrun!=newrun and oldcontroller!=newcontroller
    assert not newrun.exists() and not newcontroller.exists()
    old=read(oldrun/'protocol.json')
    assert old['run']==str(oldrun) and len(old['batches'])==57
    oldgroup=oldrun/'batches/fit/batch003/N'
    failed_receipt=oldrun/'stage_receipts/batch003_N_official.json'
    summary=oldgroup/'official/summary.json'
    terminal=oldgroup/'official/terminal_failure.json'
    guard_only_failure(read(summary),read(terminal),read(failed_receipt))
    assert not list((oldgroup/'official/scores').glob('*.json'))
    # No earlier-scored unit exists; only the frozen inputs were prepared.
    input_manifest=oldgroup/'official/input_manifest.json'
    assert sha(input_manifest)==read(summary)['input_manifest_sha256']
    original_pins={str(oldrun/'protocol.json'):OLD_SHA,str(ROOT/'scripts/experiments/audio200_production_20261025.py'):PRODUCER_SHA}
    failure_files=[failed_receipt,oldrun/'stage_receipts/batch003_N_official_launch.json',summary,terminal,input_manifest,
        oldgroup/'official/protocol.json',Path(read(failed_receipt)['log']),
        oldcontroller/'batches/batch003/producer_exit.json',oldcontroller/'batches/batch003/producer_launch.json']
    failure_pins={str(p):sha(p) for p in failure_files}
    # Verify all completed outputs before creating a new attempt.
    verified_pins={}
    copied_stages=[]
    for b in old['batches']:
        for arm in ARMS:
            for stage in STAGES:
                key=f"{b['id']}_{arm}_{stage}"
                if int(b['id'][5:])<=2 or (b['id']==BAD_BATCH and arm=='N' and stage!='official'):
                    oldreceipt=oldrun/'stage_receipts'/f'{key}.json'
                    obj=read(oldreceipt)
                    assert obj['status']=='COMPLETE'
                    verify(obj['artifacts'])
                    verified_pins.update(obj['artifacts'])
                    copied_stages.append((key,oldreceipt,sha(oldreceipt)))
    consumer_pins=[]
    for b in old['batches'][:3]:
        p=oldrun/'consumer_receipts'/f"{b['id']}.json"
        obj=read(p);assert obj['status']=='COMPLETE'
        verify(obj['artifacts']);verified_pins.update(obj['artifacts'])
        consumer_pins.append((b['id'],p,sha(p)))
    completed_controller=[]
    for b in old['batches'][1:3]:
        p=oldcontroller/'batches'/b['id']/'completion.json'
        obj=read(p)
        assert obj['status']=='PASS_BATCH_AND_PARENT_AUDIT' and obj['batch']==b['id']
        verify(obj['artifacts']);verified_pins.update(obj['artifacts'])
        completed_controller.append((b['id'],p,sha(p)))
    verify(original_pins);verify(failure_pins)
    newrun.mkdir(parents=True);newcontroller.mkdir(parents=True)
    newgroup=newrun/'batches/fit/batch003/N'
    copy_pins={}
    for name in ('protocol.json','raw_manifest.json','audio_manifest.json','video_manifest.json'):
        copy_pins[str(newgroup/name)]=exact_copy(oldgroup/name,newgroup/name)
    stages={f"{b['id']}_{a}_{s}":dict(path=str(newrun/'stage_receipts'/f"{b['id']}_{a}_{s}.json"))
            for b in old['batches'] for a in ARMS for s in STAGES}
    for key,origin,h in copied_stages:
        dest=Path(stages[key]['path']);assert exact_copy(origin,dest)==h
        stages[key].update(origin_path=str(origin),origin_sha256=h)
    consumers={b['id']:dict(path=str(newrun/'consumer_receipts'/f"{b['id']}.json")) for b in old['batches']}
    for key,origin,h in consumer_pins:
        dest=Path(consumers[key]['path']);assert exact_copy(origin,dest)==h
        consumers[key].update(origin_path=str(origin),origin_sha256=h)
    controller_reuse={}
    for key,origin,h in completed_controller:
        dest=newcontroller/'batches'/key/'completion.json'
        assert exact_copy(origin,dest)==h
        controller_reuse[key]=dict(path=str(dest),origin_path=str(origin),origin_sha256=h)
    # This legacy reference file contains historical rows and must stay byte exact.
    reuse280_sha=exact_copy(oldrun/'reuse280.json',newrun/'reuse280.json')
    recovery=dict(schema='audio200_reviewed_gpu_guard_recovery_v1',status='PREPARED_CPU_REUSE_VERIFIED_PENDING_PARENT_RELEASE',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_run=str(oldrun),new_run=str(newrun),
        old_controller_run=str(oldcontroller),new_controller_run=str(newcontroller),
        failed_batch=BAD_BATCH,failed_arm='N',failure_pins=failure_pins,
        original_protocol_path=str(oldrun/'protocol.json'),original_protocol_sha256=OLD_SHA,
        producer_source_path=str(ROOT/'scripts/experiments/audio200_production_20261025.py'),producer_source_sha256=PRODUCER_SHA,
        group_routes=[dict(batch_id=BAD_BATCH,arm='N',old_group_path=str(oldgroup),new_group_path=str(newgroup),
            old_media_manifest_path=old['batches'][3]['media_manifest_path'],
            new_media_manifest_path=str(newgroup.parent/'media_manifest.json'),protocol_sha256=sha(newgroup/'protocol.json'))],
        input_copy_pins=copy_pins,stage_receipt_map=stages,consumer_receipt_map=consumers,
        controller_completion_reuse=controller_reuse,reuse280_path=str(newrun/'reuse280.json'),reuse280_sha256=reuse280_sha,
        successful_stage_receipts_copied=len(copied_stages),consumer_receipts_copied=3,
        completed_media_sources=22,completed_media_units=154,legacy_reused_units=280,
        current_batch_N_raw_canonical_render_units=10,raw_or_video_generation_repeated=False,
        failed_official_stage_copied=False,models_loaded=0,GPU_used=False,training=False,
        material_roots=[str(oldrun),str(newrun)],aggregate_output_cap_bytes=60*1024**3,
        verified_existing_artifact_count=len(verified_pins),reviewed_failure_reason='other session appeared at exact GPU idle guard; no scoring started')
    recovery_path=newrun/'reviewed_recovery.json';put(recovery_path,recovery)
    new=copy.deepcopy(old)
    new.update(run=str(newrun),original_protocol_path=str(oldrun/'protocol.json'),original_protocol_sha256=OLD_SHA,
        material_roots=[str(oldrun),str(newrun)],reviewed_recovery_path=str(recovery_path),reviewed_recovery_sha256=sha(recovery_path))
    new['batches'][3]['groups'][0]['path']=str(newgroup)
    new['batches'][3]['media_manifest_path']=str(newgroup.parent/'media_manifest.json')
    check_protocol_diff(old,new,recovery_path,sha(recovery_path))
    put(newrun/'protocol.json',new)
    verify(original_pins);verify(failure_pins);verify(copy_pins)
    put(newrun/'cpu_recovery_validation.json',dict(status='PASS_CPU_EXACT_REUSE_AND_ROUTE_DIFF',
        original_protocol_sha256=OLD_SHA,recovery_protocol_sha256=sha(newrun/'protocol.json'),reviewed_recovery_sha256=sha(recovery_path),
        allowed_semantic_changes=['run','batch003.N.path','batch003.media_manifest_path','recovery_provenance_only'],
        all_other_protocol_values_exact=True,all_existing_media_generation_repeated=False,
        old_failure_unchanged=True,successful_stage_receipts_copied=len(copied_stages),consumer_receipts_copied=3,
        previous_controller_completions_copied=2,input_copy_files=4,verified_existing_artifact_count=len(verified_pins),
        free_bytes=shutil.disk_usage(newrun).free,models_loaded=0,GPU_used=False,training=False,
        preparation_source_sha256=sha(__file__)))
    print(json.dumps(read(newrun/'cpu_recovery_validation.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('old-run','new-run','old-controller','new-controller'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();prepare(a.old_run,a.new_run,a.old_controller,a.new_controller)
