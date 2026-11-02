"""Immutable CPU preparation and serial, resumable 200-speaker media production.

No training. GPU stages require an exact parent release and an idle accelerator.
Existing 280 units are referenced, never regenerated or relabelled.
"""
from __future__ import annotations
import argparse
import csv
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import wave

ROOT = Path(__file__).resolve().parents[2]
SYNC = '/home/wjj/.venvs/syncnet/bin/python'
INTERPRETERS = {'L06': '/home/wjj/.venvs/qwen3/bin/python', 'L17': '/home/wjj/.venvs/qwen3-tts-1p7b/bin/python'}
TARGET = .0376838172675746
BASES = [2026102201, 2026102202, 2026102203]
KWARGS = dict(do_sample=True, temperature=.9, top_k=50, top_p=1., max_new_tokens=512)
DOWNSTREAM = ROOT/'scripts/experiments/qwen17_level_comparison10_20261023/downstream.py'
HELPER = ROOT/'scripts/experiments/qwen_full_tts_extension_20261022/tts_stage.py'
ENV = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2')
FREE = 12*1024**3
LOCK = ROOT/'runs/.audio200_preparation_gpu.lock'


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(1<<20), b''):
            h.update(block)
    return h.hexdigest()


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def put(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')


def verify(pins):
    for p, expected in pins.items():
        if sha(p) != expected:
            raise ValueError('bound file changed: '+str(p))


def clean(s):
    return ''.join(s.split())


def utc():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def pcm_check(row):
    p = Path(row['remote_path'])
    if sha(p) != row['sha256']:
        raise ValueError('natural file SHA mismatch '+row['source_id'])
    with wave.open(str(p)) as f:
        if (f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()) != (1, 2, 16000, row['frames']):
            raise ValueError('natural PCM header drift')
        raw = f.readframes(f.getnframes())
    if hashlib.sha256(raw).hexdigest() != row['pcm_sha256']:
        raise ValueError('natural PCM SHA drift')


def generation_seed(speaker, draw):
    if draw not in (1, 2, 3):
        raise ValueError('only three frozen draws')
    return BASES[draw-1]+100*int(speaker[1:])


def make_roster(source, old, original40):
    rows = source['rows']
    by = {r['source_id']: r for r in rows}
    assert len(rows) == len(by) == 600
    assert len({r['speaker_id'] for r in rows}) == 200
    assert {s: sum(r['split']==s for r in rows) for s in ('fit','development','confirmation')} == {'fit':300,'development':120,'confirmation':180}
    old_sources = {r['sample_id'] for r in old['rows']}
    assert len(old['rows']) == 280 and len(old_sources) == 40 and old_sources <= by.keys()
    expected = {('N',0),*((a,d) for a in ('L06','L17') for d in (1,2,3))}
    original = {r['sample_id']: r for r in original40['samples']}
    for sid in old_sources:
        r = by[sid]
        assert r['split']=='fit' and r['ref_source_id']==sid
        assert r['sha256']==original[sid]['source_sha256']
        assert clean(r['text'])==clean(original[sid]['transcript'])
        group = [x for x in old['rows'] if x['sample_id']==sid]
        assert {(x['arm'],x['draw']) for x in group}==expected
        assert all(x['speaker_id']==r['speaker_id'] and clean(x['text'])==clean(r['text']) for x in group)
    order = {'fit':0,'development':1,'confirmation':2}
    new = sorted((r for r in rows if r['source_id'] not in old_sources), key=lambda r:(order[r['split']],r['source_id']))
    assert len(new)==560
    return new


def source_pins_from_runtime(executable, old_runtime):
    code = "import importlib.metadata as m,json;print(json.dumps({n:m.version(n) for n in ['torch','transformers','qwen-tts','numpy','soundfile']}))"
    packages = json.loads(subprocess.check_output([executable,'-c',code],text=True))
    expected = old_runtime.get('packages')
    if expected is not None:
        for k,v in expected.items():
            assert packages[k]==v, (executable,k,packages[k],v)
    freeze = subprocess.check_output([executable,'-m','pip','freeze'],timeout=60)
    pins = old_runtime.get('source_files',old_runtime.get('source_pins',{}))
    verify(pins)
    return dict(executable=executable,packages=packages,pip_freeze_sha256=hashlib.sha256(freeze).hexdigest(),source_pins=pins)


def prepare(run, batch_size):
    if os.name!='posix':
        raise RuntimeError('CPU preparation must verify actual Ubuntu paths')
    source_file = ROOT/'runs/audio_data_inventory200_20261024_attempt05/selected600_natural_manifest.json'
    split_file = source_file.parent/'source_split_freeze.json'
    old_file = ROOT/'runs/syncnet_audio_dataset40_20261024_attempt03/dataset280_consumer_manifest.json'
    p06_file = ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01/protocol.json'
    p17_file = ROOT/'runs/qwen17_level_comparison10_20261023_attempt01/protocol.json'
    verify({str(source_file):'3ef28c8076aba23e64e2bc1a22963e24dba1e9a89ed9e2aeaf7fdaae6821e608',
            str(split_file):'3ffb6d516470d5185a71e18d5bd8390b3180de7d6178640a2726c5eaef6fc749',
            str(old_file):'046b4cbb1a53fc11b2455f90eb943a7cfbac57c80e6ec34039a4f00b0b865acf'})
    source,old,p06,p17 = (read(p) for p in (source_file,old_file,p06_file,p17_file))
    new = make_roster(source,old,p06)
    run.mkdir(parents=True,exist_ok=False)
    for r in source['rows']:
        pcm_check(r)
    # Pin each historical implementation bundle separately. No GPU/model import.
    l06_runtime = read(ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01/generation/runtime.json')
    l17_runtime = read(ROOT/'runs/audio_data_supplement_20261024_attempt04/runtime_pins.json')
    l17_historical = {str(p):h for p,h in l17_runtime['source_pins'].items()}
    verify(l17_historical)
    bundles = {
        'L06':source_pins_from_runtime(INTERPRETERS['L06'], l06_runtime),
        'L17':source_pins_from_runtime(INTERPRETERS['L17'], {'source_pins':l17_historical}),
    }
    for arm, p in [('L06',p06),('L17',p17)]:
        model = Path(p['model_dir'])
        files = {str(x):sha(x) for x in sorted(model.rglob('*')) if x.is_file() and '.cache' not in x.parts and x.suffix in ('.json','.safetensors','.txt','.model')}
        if arm=='L06':
            assert files[str(model/'model.safetensors')]==p['model_weights_sha256']
            assert files[str(model/'speech_tokenizer/model.safetensors')]==p['codec_weights_sha256']
        else:
            for f in p['weights_manifest']['files']:
                assert sha(model/f['path'])==f['sha256']
        bundles[arm].update(model_dir=str(model),model_pins=files)
    shared_pins = {str(p):sha(p) for p in (source_file,split_file,old_file,p06_file,p17_file,Path(__file__),HELPER,DOWNSTREAM)}
    for p,h in p17['source_pins'].items():
        # Keep only dependencies actually consumed by the unchanged downstream.
        if '/third_party/' in p or p==p17['downstream']['face']:
            assert sha(p)==h
            shared_pins[p]=h
    downstream_pins = {p:h for p,h in shared_pins.items() if '/third_party/' in p or p in (str(DOWNSTREAM),str(HELPER),str(Path(__file__)),p17['downstream']['face'])}
    prepared = []
    for r in new:
        dest = run/'input24'/r['split']/f"{r['source_id']}.wav"
        dest.parent.mkdir(parents=True,exist_ok=True)
        command = ['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-i',r['remote_path'],'-map','0:a:0','-vn','-ac','1','-ar','24000','-c:a','pcm_f32le',str(dest)]
        subprocess.run(command,check=True,capture_output=True,timeout=60)
        prepared.append(dict(sample_id=r['source_id'],speaker_id=r['speaker_id'],split=r['split'],
            source_path=r['remote_path'],source_sha256=r['sha256'],transcript=r['text'],
            duration_s=r['duration_s'],input24_path=str(dest),input24_sha256=sha(dest),conversion_command=command))
    # First two fixed new fit sources form a bounded short/long pilot.
    assert [r['sample_id'] for r in prepared[:2]]==['BAC009S0002W0322','BAC009S0002W0363']
    chunks = [prepared[:2]]
    for split in ('fit','development','confirmation'):
        subset=[r for r in prepared[2:] if r['split']==split]
        chunks.extend(subset[i:i+batch_size] for i in range(0,len(subset),batch_size))
    batches=[]
    for index, samples in enumerate(chunks):
        name=f'batch{index:03d}'
        split=samples[0]['split']
        assert all(s['split']==split for s in samples)
        batch=run/'batches'/('confirmation_sealed' if split=='confirmation' else split)/name
        group_records=[]
        for arm in ('N','L06','L17'):
            cell=batch/arm
            roles=['N'] if arm=='N' else [arm+'_norm'+str(d) for d in (1,2,3)]
            analysis=cell/'analysis_protocol.json'
            put(analysis,dict(status='DATA_PREPARATION_ONLY',no_training=True,
                scientific_analysis=False,confirmation_policy='mechanical extraction only; no effect estimates/rankings'))
            cellpins={**downstream_pins,str(analysis):sha(analysis)}
            config=dict(samples=samples,roles=roles,expected_cells=len(samples)*len(roles),
                cohort='audio200_source600_frozen',target_rms=TARGET,raw_manifest_path=str(cell/'raw_manifest.json'),
                analysis_protocol_path=str(analysis),source_pins=cellpins,downstream=p17['downstream'],
                seed_bases=BASES,generation_kwargs=KWARGS,non_streaming_mode=False,
                model_dir=bundles[arm]['model_dir'] if arm!='N' else None,arm=arm,
                bundle=bundles.get(arm),no_training=True,execution=dict(stage_timeout_s=10800))
            put(cell/'protocol.json',config)
            group_records.append(dict(arm=arm,path=str(cell),protocol_sha256=sha(cell/'protocol.json')))
        batches.append(dict(id=name,split=split,source_ids=[s['sample_id'] for s in samples],groups=group_records,
                            media_manifest_path=str(batch/'media_manifest.json')))
    protocol=dict(schema='audio200_media_production_v1',status='PREPARED_NO_GPU',no_training=True,
        run=str(run),project_root=str(ROOT),created_utc=utc(),source_pins=shared_pins,bundles=bundles,
        source_manifest_path=str(source_file),reuse_manifest_path=str(old_file),expected_units=4200,
        reused_units=280,new_units=3920,new_N=560,new_TTS_per_model=1680,batches=batches,
        ordered_sources=prepared,minimum_free_bytes=FREE,preflight_budget_bytes=57210000000,
        output_cap_bytes=60*1024**3,lock_path=str(LOCK),seed_bases=BASES,generation_kwargs=KWARGS,
        target_rms=TARGET,non_streaming_mode=False,
        failure_policy='first real stage failure retained, stop; no redraw/retry/limiter/selection; reviewed new attempt only',
        generation_policy='full native ICL and ratio cut; N uses same input24 roundtrip; all predetermined draws retained',
        GPU_policy='wait for non-system GPU work; never kill/share; exact gnome-remote-desktop-daemon allowance only')
    put(run/'protocol.json',protocol)
    put(run/'reuse280.json',dict(status='REFERENCE_ONLY',rows=old['rows'],source_manifest_sha256=sha(old_file)))
    put(run/'cpu_preparation.json',dict(status='PREPARED_NO_GPU_OR_MODEL_LOAD',sources600_verified=True,
        new_input24=560,models_loaded=0,all40_anchors_match=True,new_units=3920,batch_count=len(batches),
        free_bytes=shutil.disk_usage(run).free,protocol_sha256=sha(run/'protocol.json'),no_training=True))
    print(json.dumps(read(run/'cpu_preparation.json')),flush=True)


def gpu_processes():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader'],text=True,timeout=15)
    unexpected=[]
    for row in csv.reader(output.splitlines()):
        if not row:
            continue
        pid,name= int(row[0].strip()),row[1].strip()
        try:
            actual=os.readlink('/proc/'+str(pid)+'/exe')
        except FileNotFoundError:
            continue
        if name!='/usr/libexec/gnome-remote-desktop-daemon' or actual!=name:
            unexpected.append(dict(pid=pid,name=name,actual=actual,memory=row[2].strip()))
    return unexpected


def event(run,name,**kw):
    item=dict(event=name,pid=os.getpid(),utc=utc(),**kw)
    with (run/'events.jsonl').open('a',encoding='utf-8') as f:
        f.write(json.dumps(item,ensure_ascii=False,allow_nan=False)+'\n')
    print(json.dumps(item,ensure_ascii=False),flush=True)


def wait_gpu(run, deadline):
    last=None
    while True:
        if shutil.disk_usage(run).free < FREE+256*1024**2:
            raise RuntimeError('insufficient disk headroom')
        other=gpu_processes()
        if not other:
            return
        if other!=last:
            event(run,'WAIT_GPU',unexpected_processes=other)
            last=other
        if time.monotonic()>deadline:
            raise TimeoutError('GPU wait deadline exceeded; no other process killed')
        time.sleep(30)


def file_seal(directory):
    return {str(p):sha(p) for p in sorted(directory.rglob('*')) if p.is_file() and not p.name.endswith('.log')}


def run_stage(run,group,stage,cmd,output,deadline,gpu):
    cell=Path(group['path'])
    receipt=run/'stage_receipts'/f"{cell.parent.name}_{group['arm']}_{stage}.json"
    if receipt.exists():
        old=read(receipt)
        if old['status']!='COMPLETE':
            raise RuntimeError('previous failure remains; reviewed new attempt required')
        verify(old['artifacts'])
        return
    if any((run/'stage_receipts').glob(receipt.stem+'_launch*.json')):
        raise RuntimeError('unresolved previous launch; do not automatically retry')
    verify({str(cell/'protocol.json'):group['protocol_sha256']})
    for s in read(cell/'protocol.json')['samples']:
        verify({s['source_path']:s['source_sha256'],s['input24_path']:s['input24_sha256']})
    if gpu:
        wait_gpu(run,deadline)
    log=run/'logs'/f'{receipt.stem}.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    put(receipt.with_name(receipt.stem+'_launch.json'),dict(command=cmd,pid=os.getpid(),utc=utc(),no_training=True))
    start=time.monotonic()
    proc=None
    error=None
    try:
        with log.open('x',encoding='utf-8') as f:
            proc=subprocess.Popen(cmd,cwd=ROOT,env={**os.environ,**ENV},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
                if time.monotonic()-start>10800 or time.monotonic()>deadline:
                    os.killpg(proc.pid,signal.SIGTERM)
                    proc.wait(timeout=15)
                    raise TimeoutError('own child hard deadline')
                if shutil.disk_usage(run).free<FREE:
                    os.killpg(proc.pid,signal.SIGTERM)
                    proc.wait(timeout=15)
                    raise RuntimeError('disk floor breached')
        if proc.returncode:
            raise RuntimeError('stage child failed; see '+str(log))
        if not output.is_file():
            raise RuntimeError('successful exit omitted manifest')
        if stage=='official' and read(output).get('status')!='COMPLETE':
            raise RuntimeError('official output not complete')
    except BaseException as exc:
        error=repr(exc)
    artifact_pins=file_seal(cell) if error is None else {}
    put(receipt,dict(status='FAILED' if error else 'COMPLETE',stage=stage,command=cmd,
        returncode=proc.returncode if proc else None,error=error,elapsed_s=time.monotonic()-start,
        output=str(output),artifacts=artifact_pins,log=str(log),log_sha256=sha(log),no_training=True))
    if error:
        raise RuntimeError(error)


def collect_batch(batch):
    records=[]
    for group in batch['groups']:
        arm=group['arm'];cell=Path(group['path']);p=read(cell/'protocol.json')
        samples={s['sample_id']:s for s in p['samples']}
        for role in p['roles']:
            draw=0 if arm=='N' else int(role[-1])
            for sid,s in samples.items():
                score=cell/'official/scores'/f'{sid}_{role}.json'
                obj=read(score)
                assert obj['status']=='COMPLETE' and obj['sample_id']==sid and obj['speaker_id']==s['speaker_id']
                records.append(dict(id=f'{sid}_{arm}_{draw}',sample_id=sid,speaker_id=s['speaker_id'],split=s['split'],
                    arm=arm,draw=draw,text=clean(s['transcript']),score_path=str(score),score_sha256=sha(score),status='COMPLETE'))
    assert len(records)==7*len(batch['source_ids'])
    return records


def execute(run,release_file,consumer_contract):
    p=read(run/'protocol.json');release=read(release_file)
    assert release['allow_media_generation'] is True and release['allow_training'] is False
    assert release['producer_protocol_sha256']==sha(run/'protocol.json')
    assert release['producer_source_sha256']==sha(__file__)
    allowed=set(release['allowed_source_ids'])
    planned={x for b in p['batches'] for x in b['source_ids']}
    assert allowed and allowed<=planned
    batches=[b for b in p['batches'] if set(b['source_ids'])<=allowed]
    assert {x for b in batches for x in b['source_ids']}==allowed, 'release must use exact frozen whole batches'
    verify(p['source_pins'])
    assert shutil.disk_usage(run).free >= FREE + int(release.get('additional_disk_budget_bytes',0))
    for prior in ('runs/.audio_data_supplement_gpu.lock','runs/.syncnet_audio_dataset_cache_gpu.lock'):
        if (ROOT/prior).exists():
            raise RuntimeError('another pipeline lock exists: '+prior)
    if LOCK.exists():
        raise RuntimeError('GPU lock exists; parent review required')
    LOCK.mkdir()
    put(LOCK/'owner.json',dict(pid=os.getpid(),run=str(run),utc=utc(),release_path=str(release_file),release_sha256=sha(release_file)))
    deadline=time.monotonic()+int(release.get('hard_timeout_seconds',7*86400))
    attempt_dir=run/'executions'
    attempt_dir.mkdir(exist_ok=True)
    attempt=attempt_dir/f'{len(list(attempt_dir.iterdir()))+1:03d}'
    attempt.mkdir()
    put(attempt/'launch.json',dict(pid=os.getpid(),command=sys.argv,utc=utc(),release_sha256=sha(release_file)))
    error=None
    try:
        for b in batches:
            if sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>p['output_cap_bytes']:
                raise RuntimeError('new run output cap exceeded')
            event(run,'BATCH_START',batch=b['id'],sources=b['source_ids'])
            for group in b['groups']:
                cell=Path(group['path']);arm=group['arm']
                synth_cmd=[SYNC if arm=='N' else INTERPRETERS[arm],str(Path(__file__)),'worker-synthesize','--run',str(cell),'--protocol-sha256',group['protocol_sha256']]
                run_stage(run,group,'synthesize',synth_cmd,cell/'raw_manifest.json',deadline,gpu=arm!='N')
                for stage in ('canonical','render','official'):
                    output={'canonical':cell/'audio_manifest.json','render':cell/'video_manifest.json','official':cell/'official/summary.json'}[stage]
                    command=[SYNC,str(DOWNSTREAM),'--stage',stage,'--run',str(cell),'--protocol',str(cell/'protocol.json'),'--protocol-sha256',group['protocol_sha256']]
                    run_stage(run,group,stage,command,output,deadline,gpu=stage!='canonical')
            media=Path(b['media_manifest_path'])
            if not media.exists():
                put(media,dict(schema='audio200_batch_media_v1',status='COMPLETE',source_ids=b['source_ids'],rows=collect_batch(b),no_training=True,
                    producer_protocol_sha256=sha(run/'protocol.json')))
            else:
                assert read(media)['rows']==collect_batch(b)
            event(run,'BATCH_MEDIA_COMPLETE',batch=b['id'],media_manifest=str(media),media_manifest_sha256=sha(media))
            if consumer_contract:
                consumer=ROOT/'scripts/experiments/audio200_feature_consumer_20261025.py'
                assert release['consumer_contract_sha256']==sha(consumer_contract)
                assert release['consumer_source_sha256']==sha(consumer)
                wait_gpu(run,deadline)
                out=media.parent/'features'
                command=[SYNC,str(consumer),'consume','--manifest',str(media),'--output-root',str(out),
                    '--contract',str(consumer_contract),'--release',str(release_file),'--parent-lock-owner',str(os.getpid())]
                receipt=run/'consumer_receipts'/f"{b['id']}.json"
                if receipt.exists():
                    assert read(receipt)['status']=='COMPLETE'
                    verify(read(receipt)['artifacts'])
                else:
                    log=run/'logs'/f"{b['id']}_features.log"
                    with log.open('x',encoding='utf-8') as f:
                        result=subprocess.run(command,cwd=ROOT,env={**os.environ,**ENV},stdout=f,stderr=subprocess.STDOUT,timeout=10800)
                    if result.returncode:
                        put(receipt,dict(status='FAILED',returncode=result.returncode,log=str(log)))
                        raise RuntimeError('feature consumer failed')
                    put(receipt,dict(status='COMPLETE',artifacts=file_seal(out),command=command))
            event(run,'BATCH_COMPLETE',batch=b['id'],features_requested=bool(consumer_contract))
        if allowed==planned:
            final=read(run/'reuse280.json')['rows'][:]
            for b in p['batches']:
                final.extend(read(b['media_manifest_path'])['rows'])
            assert len(final)==len({r['id'] for r in final})==4200
            if not (run/'consumer_manifest.json').exists():
                put(run/'consumer_manifest.json',dict(status='ALL4200_MEDIA_COMPLETE',rows=final,scope='media only; consumer receipts determine features/FA readiness',no_training=True))
        put(attempt/'completion.json',dict(status='RELEASED_SCOPE_COMPLETE',sources=len(allowed),media=len(allowed)*7,utc=utc(),no_training=True))
    except BaseException as exc:
        error=repr(exc)
        put(attempt/'failure.json',dict(status='FAILED_RETAINED_NO_RETRY',error=error,utc=utc(),no_training=True))
        raise
    finally:
        (LOCK/'owner.json').unlink()
        LOCK.rmdir()
        event(run,'EXECUTION_EXIT',error=error,no_training=True)


def synthesize(run,digest):
    if sha(run/'protocol.json')!=digest:
        raise ValueError('cell protocol SHA')
    p=read(run/'protocol.json');verify(p['source_pins'])
    if p['arm']=='N':
        rows=[dict(sample_id=s['sample_id'],speaker_id=s['speaker_id'],role='N',path=s['input24_path'],sha256=s['input24_sha256']) for s in p['samples']]
        put(run/'raw_manifest.json',dict(records=rows,count=len(rows),no_training=True))
        return
    os.environ.update(ENV)
    import gc
    import inspect
    import random
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel,Qwen3TTSTokenizer
    bundle=p['bundle'];verify(bundle['source_pins']);verify(bundle['model_pins'])
    assert sys.executable==bundle['executable']
    freeze=subprocess.check_output([sys.executable,'-m','pip','freeze'],timeout=60)
    assert hashlib.sha256(freeze).hexdigest()==bundle['pip_freeze_sha256']
    spec=importlib.util.spec_from_file_location('frozen_full_icl_helpers',HELPER)
    h=importlib.util.module_from_spec(spec);spec.loader.exec_module(h)
    assert sha(HELPER)=='38a48e335633c5fd2c5f0317dcf09c52fdacf756d7e5f9d09aef6c35f0f3c9aa'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)
    def seed(v):
        random.seed(v);np.random.seed(v%2**32);torch.manual_seed(v);torch.cuda.manual_seed_all(v)
    seed(20261022)
    modeldir=Path(p['model_dir']);out=run/'generation';out.mkdir(exist_ok=False)
    w=Qwen3TTSModel.from_pretrained(str(modeldir),local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
    w.model.eval().requires_grad_(False);old=w.model.speech_tokenizer;w.model.speech_tokenizer=None
    del old;gc.collect();torch.cuda.empty_cache()
    codec=Qwen3TTSTokenizer.from_pretrained(str(modeldir/'speech_tokenizer'),local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
    codec.model.eval().requires_grad_(False);w.model.speech_tokenizer=codec
    assert w.model.dtype==codec.model.dtype==torch.float32
    configs={name:dict(dtype=str(m.dtype),attention=getattr(m.config,'_attn_implementation',None)) for name,m in [('LM',w.model),('codec',codec.model),('codec_encoder',codec.model.encoder),('codec_decoder',codec.model.decoder)]}
    runtime=dict(executable=sys.executable,torch=torch.__version__,numpy=np.__version__,GPU=torch.cuda.get_device_name(),configs=configs,
        source_pins={inspect.getfile(type(m)):sha(inspect.getfile(type(m))) for m in [w,w.model,w.model.talker,w.model.talker.code_predictor,codec.model]},no_training=True)
    put(out/'runtime.json',runtime)
    rows=[]
    for s in p['samples']:
        directory=out/s['sample_id'];directory.mkdir();text=h.strip_whitespace(s['transcript'])
        x,sr=sf.read(s['input24_path'],dtype='float32');x=h.waveform(x,sr)
        with torch.inference_mode():
            prompt=w.create_voice_clone_prompt(ref_audio=(x,24000),ref_text=text,x_vector_only_mode=False)
        assert len(prompt)==1 and prompt[0].icl_mode and not prompt[0].x_vector_only_mode
        ref=h.codes(prompt[0].ref_code);reference=h.save_array(directory/'reference_codes.npy',ref)
        h.save_array(directory/'reference_speaker_embedding.npy',h.array(prompt[0].ref_spk_embedding))
        for draw in (1,2,3):
            value=generation_seed(s['speaker_id'],draw);seed(value);start=time.monotonic()
            with torch.inference_mode(),h.capture_official(w) as cap:
                waves,rate=w.generate_voice_clone(text=text,language='Chinese',voice_clone_prompt=prompt,non_streaming_mode=False,**KWARGS)
            assert all(cap[k]==1 for k in ('core_generate_calls','decoder_calls','talker_calls'))
            new=cap['new_codes'][0];joined=cap['decode_codes'][0];full=cap['full_waves'][0]
            assert np.array_equal(joined,np.concatenate([ref,new]))
            cut=h.official_cut(len(ref),len(joined),len(full));waveform=h.waveform(waves[0],rate)
            assert np.array_equal(full[cut:],waveform)
            if p['arm']=='L06':
                with torch.inference_mode():
                    repeat,repeat_rate=codec.decode([{'audio_codes':torch.as_tensor(joined,device=codec.device)}])
                assert np.array_equal(full,h.waveform(repeat[0],repeat_rate))
            eos=cap['talker_kwargs'].get('eos_token_id');role=p['arm']+'_norm'+str(draw)
            record=dict(sample_id=s['sample_id'],speaker_id=s['speaker_id'],model=p['arm'],role=role,draw_index=draw,seed=value,
                **h.save_wave(directory/f'draw{draw}_raw_float.wav',waveform),reference_codes=reference,
                new_codes=h.save_array(directory/f'draw{draw}_new_codes.npy',new),
                decode_concat_codes=h.save_array(directory/f'draw{draw}_concat_codes.npy',joined),
                full_decoded=h.save_wave(directory/f'draw{draw}_full_decoded.wav',full),
                talker_sequences=h.save_array(directory/f'draw{draw}_talker_sequences.npy',cap['talker_sequences']),
                eos_in_talker_sequences=bool(np.any(cap['talker_sequences']==eos)) if isinstance(eos,int) else None,
                target_text=text,reference_text=text,reference_audio_sha256=s['input24_sha256'],official_ratio_cut_samples=cut,
                talker_effective_kwargs=cap['talker_kwargs'],core_explicit_kwargs=cap['core_kwargs'],
                duration_ratio=len(waveform)/rate/s['duration_s'],elapsed_s=time.monotonic()-start)
            put(directory/f'draw{draw}.json',record);rows.append(record)
            print(json.dumps(dict(event='GENERATION_COMPLETE',sample_id=s['sample_id'],arm=p['arm'],draw=draw,elapsed_s=record['elapsed_s'])),flush=True)
    put(run/'raw_manifest.json',dict(records=rows,count=len(rows),protocol_sha256=digest,no_training=True))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['prepare','execute','worker-synthesize'])
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--batch-size',type=int,default=10)
    parser.add_argument('--release',type=Path)
    parser.add_argument('--consumer-contract',type=Path)
    parser.add_argument('--protocol-sha256')
    a=parser.parse_args()
    if a.stage=='prepare':
        if not 1<=a.batch_size<=10:
            parser.error('batch-size must be1..10')
        prepare(a.run,a.batch_size)
    elif a.stage=='execute':
        if not a.release:
            parser.error('--release required')
        execute(a.run,a.release,a.consumer_contract)
    else:
        synthesize(a.run,a.protocol_sha256)


if __name__=='__main__':
    main()
