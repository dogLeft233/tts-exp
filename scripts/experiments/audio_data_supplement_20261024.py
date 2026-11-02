"""Staged, append-only supplement preparation. No execution without parent release.

Only the missing 20 sources x 3 L17 draws are implemented. Larger cohorts and
feature extraction remain HOLD until their independent contracts are frozen.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
REMOTE = '/home/wjj/tts-audio/tts-exp'
NAME = 'audio_data_supplement_20261024_attempt01'
TARGET = .0376838172675746
BASES = [2026102201, 2026102202, 2026102203]
KWARGS = dict(do_sample=True, temperature=.9, top_k=50, top_p=1., max_new_tokens=512)
DOWNSTREAM = 'scripts/experiments/qwen17_level_comparison10_20261023/downstream.py'
HELPER = 'scripts/experiments/qwen_full_tts_extension_20261022/tts_stage.py'
ENV = dict(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2')
DISPLAY_EXECUTABLES = frozenset(['/usr/libexec/gnome-remote-desktop-daemon'])


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''): h.update(block)
    return h.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf8'))


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False); f.write('\n')


def frozen_rows(protocol40, manifest20):
    """Selection uses identities only, never outcomes/qualification/score."""
    samples = protocol40['samples']
    if len(samples) != 40 or len({s['speaker_id'] for s in samples}) != 40:
        raise ValueError('require the existing forty independent speakers')
    existing = manifest20['rows']
    covered = {s['sample_id'] for s in existing}
    if len(covered) != 20 or not covered <= {s['sample_id'] for s in samples}:
        raise ValueError('old twenty must be a subset of frozen forty')
    for sid in covered:
        cells = [(r['arm'], r['draw']) for r in existing if r['sample_id'] == sid]
        if len(cells) != 7 or set(cells) != {('N', 0), *[(a,d) for a in ['L06','L17'] for d in (1,2,3)]}:
            raise ValueError('incomplete existing three-arm roster')
    missing = [dict(s, split='fit') for s in samples if s['sample_id'] not in covered]
    if len(missing) != 20: raise ValueError('expected exactly sixty missing L17 cells')
    return sorted(missing, key=lambda s: s['sample_id'])


def validate_settings(p):
    if p['seed_bases'] != BASES or p['generation_kwargs'] != KWARGS or p['non_streaming_mode'] is not False:
        raise ValueError('original seed/sampling/streaming contract changed')
    if p['target_rms'] != TARGET: raise ValueError('fixed RMS contract changed')


def prepare(run, execution_root):
    old = ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01/protocol.json'
    small = ROOT/'runs/syncnet_preference_probe_20261024_attempt01/input_manifest.json'
    original = ROOT/'runs/qwen17_level_comparison10_20261023_attempt01/protocol.json'
    p40, p17 = read(old), read(original)
    validate_settings(p40); validate_settings(p17)
    rows = frozen_rows(p40, read(small))
    remote_run = execution_root+'/runs/'+run.name
    remote_script = execution_root+'/scripts/experiments/'+Path(__file__).name
    sync = '/home/wjj/.venvs/syncnet/bin/python'
    tts = '/home/wjj/.venvs/qwen3-tts-1p7b/bin/python'
    # Preserve historical pins; do not rewrite frozen historical manifests.
    pins = dict(p17['source_pins'])
    for rel in [DOWNSTREAM, HELPER, 'scripts/experiments/'+Path(__file__).name]:
        pins[execution_root+'/'+rel] = sha(ROOT/rel)
    for item in p17['weights_manifest']['files']:
        pins[p17['model_dir']+'/'+item['path']] = item['sha256']
    for path in [old, small, original]: pins[execution_root+'/'+path.relative_to(ROOT).as_posix()] = sha(path)
    if (run/'protocol.json').exists(): raise FileExistsError('prepared protocol already exists')
    run.mkdir(parents=True, exist_ok=True)
    runtime = read(run/'runtime_pins.json') if (run/'runtime_pins.json').exists() else None
    if runtime:
        pins.update(runtime['source_pins'])
        pins[remote_run+'/runtime_pins.json'] = sha(run/'runtime_pins.json')
    budget = read(run/'budget.json') if (run/'budget.json').exists() else None
    if budget: pins[remote_run+'/budget.json'] = sha(run/'budget.json')
    tasks = []
    for sample in rows:
        sid = sample['sample_id']; folder = run/'cells'/sid
        cell_remote = remote_run+'/cells/'+sid
        write(folder/'analysis_protocol.json',dict(status='DATA_PREPARATION_ONLY',no_training=True,
            intention_denominator=3,draws=[1,2,3],quality='retain every predetermined draw; absent EOS retained',
            inference='no effect estimation, model selection, refitting or confirmation analysis'))
        cell_pins={**pins,cell_remote+'/analysis_protocol.json':sha(folder/'analysis_protocol.json')}
        config = dict(samples=[sample], roles=['L17_norm1','L17_norm2','L17_norm3'], expected_cells=3,
                      cohort='missing20_existing40_fit_only', target_rms=TARGET,
                      raw_manifest_path=cell_remote+'/raw_manifest.json', source_pins=cell_pins,
                      analysis_protocol_path=cell_remote+'/analysis_protocol.json',
                      downstream=p17['downstream'], model_dir=p17['model_dir'], seed_bases=BASES,
                      generation_kwargs=KWARGS, non_streaming_mode=False,
                      execution=dict(stage_timeout_s=10800), no_training=True)
        write(folder/'protocol.json', config)
        digest = sha(folder/'protocol.json')
        previous = None
        for stage in ['synthesize','canonical','render','official','features']:
            if stage == 'synthesize':
                command = [tts,remote_script,'worker-synthesize','--run',cell_remote,'--protocol-sha256',digest]
                output = 'raw_manifest.json'; timeout = 7200
            elif stage == 'features':
                command = None; output = None; timeout = None
            else:
                command = [sync,execution_root+'/'+DOWNSTREAM,'--stage',stage,'--run',cell_remote,
                           '--protocol',cell_remote+'/protocol.json','--protocol-sha256',digest]
                output = {'canonical':'audio_manifest.json','render':'video_manifest.json','official':'official/summary.json'}[stage]
                timeout = 10800 if stage != 'canonical' else 600
            key = sid+'_'+stage
            tasks.append(dict(id=key, sample_id=sid, stage=stage, depends_on=previous,
                              command=command, cwd=execution_root, timeout_s=timeout,
                              gpu=stage in ['synthesize','render','official','features'],
                              cell_dir=cell_remote, output_manifest=cell_remote+'/'+output if output else None,
                              protocol_sha256=digest, status='HOLD_INTERFACE' if command is None else 'PENDING'))
            previous = key
    protocol = dict(schema_version=1, status='HOLD_PARENT_GATES', execution_root=execution_root,
                    run=remote_run, scope='missing60_L17_only', no_training=True,
                    samples=rows, expected_new_audio=60, expected_new_video=60,
                    target_rms=TARGET, source_pins=pins, tasks=tasks,
                    minimum_free_bytes=12*1024**3, next_unit_temporary_peak_bytes=122779452,
                    output_cap_bytes=20*1024**3,
                    reuse_candidates=220, larger_cohort_status='HOLD_INVENTORY_SELECTION_STORAGE',
                    feature_status='HOLD_ACCEPTED_EXTRACTOR_INTERFACE', runtime_pins=runtime, budget=budget,
                    contracts=dict(text='whitespace-stripped frozen transcript equals target and self reference; no ASR',
                        timing='complete waveform; no silence trim, time stretch, padding or best-lag correction',
                        failures='EOS absent retained; no score-based redraw, no retry in same cell; new attempt after review',
                        implementation='L06 and L17 are separately pinned model/environment bundles; same kwargs do not establish identical implementation',
                        old_exception='historical raw PCM clipping exception is not granted to new outputs',
                        quality='retain raw FLOAT/code/EOS, canonical RMS/PCM and every official track, clock and intermediate'))
    write(run/'protocol.json', protocol)
    write(run/'dryrun.json', dict(status='HOLD', prepared=True, media_generated=0, models_loaded=0,
        sources=20, new_L17_draws=60, tasks=len(tasks), protocol_sha256=sha(run/'protocol.json'),
        pending=['parent release binding two PASS gates', 'current input/source/weight verification on execution host',
                 'storage headroom for full batch', 'accepted feature extractor interface'],
        execute_command=[sync,remote_script,'execute','--run',remote_run,'--release',remote_run+'/parent_release.json']))


def verify_pins(pins):
    for path, expected in pins.items():
        if sha(path) != expected: raise ValueError('changed bound input: '+path)


def require_release(path, protocol_path):
    release = read(path)
    if release.get('status') != 'RELEASED' or release.get('scope') != 'missing60_L17_only':
        raise ValueError('parent release missing or wrong scope')
    if release.get('protocol_sha256') != sha(protocol_path): raise ValueError('release protocol mismatch')
    gates = release.get('gates', {})
    if set(gates) != {'replay20','inventory200'}: raise ValueError('two parent-reviewed gates required')
    for gate in gates.values():
        if sha(gate['path']) != gate['sha256'] or read(gate['path']).get('status') != 'PASS':
            raise ValueError('gate not PASS or changed')
    return release


def gpu_idle():
    command = ['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory','--format=csv,noheader']
    output = subprocess.check_output(command, text=True, timeout=15)
    allowed=[]
    for row in csv.reader(output.splitlines()):
        if not row: continue
        if len(row)!=3: raise ValueError('invalid GPU process inventory')
        pid=int(row[0].strip());reported=row[1].strip();memory=row[2].strip()
        actual=os.readlink('/proc/'+str(pid)+'/exe')
        # Exact system executable and reported absolute path; no substring match.
        if reported not in DISPLAY_EXECUTABLES or actual!=reported:
            raise RuntimeError('GPU occupied; HOLD without killing other work: '+','.join(row))
        allowed.append(dict(pid=pid,executable=actual,reported_name=reported,used_gpu_memory=memory))
    return dict(command=command, output=output, allowed_display_processes=allowed)


def artifacts(manifest_path):
    """Bind all produced files in this isolated cell, including model traces."""
    path = Path(manifest_path)
    if not path.is_file(): raise ValueError('missing successful output manifest')
    data = read(path)
    if path.name == 'summary.json' and data.get('status') != 'COMPLETE': raise ValueError('official incomplete')
    return {str(p):sha(p) for p in path.parent.rglob('*') if p.is_file()}


def kill_own(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try: process.wait(timeout=10)
        except subprocess.TimeoutExpired: os.killpg(process.pid,signal.SIGKILL); process.wait()


def execute(run, release):
    if os.name != 'posix': raise RuntimeError('execution requires the frozen Linux environments')
    p = read(run/'protocol.json'); require_release(release,run/'protocol.json')
    if not p.get('runtime_pins'): raise RuntimeError('HOLD: frozen CPU runtime/source probe missing')
    for executable, expected in p['runtime_pins']['environment_freeze'].items():
        actual=subprocess.check_output([executable,'-m','pip','freeze'],timeout=60)
        if hashlib.sha256(actual).hexdigest()!=expected: raise ValueError('frozen environment changed: '+executable)
    if str(run) != p['run']: raise ValueError('execution run path differs from frozen plan')
    verify_pins(p['source_pins'])
    if p.get('budget'):
        needed=p['minimum_free_bytes']+p['budget']['incremental_peak_estimate_bytes']
        if shutil.disk_usage(run).free<needed: raise RuntimeError('HOLD: batch estimate plus frozen free floor exceeds available disk')
    # Shared advisory lock plus real GPU process check. Stale locks require review.
    lock = Path(p['execution_root'])/'runs/.audio_data_supplement_gpu.lock'
    lock.mkdir(exist_ok=False)
    write(lock/'owner.json',dict(pid=os.getpid(),run=str(run),started_epoch=time.time()))
    completed = set()
    try:
        for task in p['tasks']:
            if task['stage'] == 'features': continue  # Explicit HOLD until accepted API is bound in a new protocol.
            receipt = run/'receipts'/(task['id']+'.json')
            if receipt.exists():
                old = read(receipt)
                if old['status'] != 'COMPLETE': raise RuntimeError('prior failure retained; use reviewed new attempt')
                verify_pins(old['artifacts']); completed.add(task['id']); continue
            if task['depends_on'] and task['depends_on'] not in completed: raise ValueError('dependency incomplete')
            if shutil.disk_usage(run).free < p['minimum_free_bytes']+p['next_unit_temporary_peak_bytes']:
                raise RuntimeError('HOLD: free disk below original floor plus next-unit temporary peak')
            if sum(f.stat().st_size for f in run.rglob('*') if f.is_file()) > p['output_cap_bytes']:
                raise RuntimeError('run disk cap exceeded')
            verify_pins(p['source_pins'])
            cell = Path(task['cell_dir'])
            if sha(cell/'protocol.json') != task['protocol_sha256']: raise ValueError('cell protocol drift')
            s = read(cell/'protocol.json')['samples'][0]
            verify_pins({s['input24_path']:s['input24_sha256'],s['source_path']:s['source_sha256'],s['transcript_path']:s['transcript_sha256']})
            gpu = gpu_idle() if task['gpu'] else None
            launch = run/'receipts'/(task['id']+'_launch.json')
            write(launch,dict(task=task,pid=os.getpid(),GPU_preflight=gpu,environment=ENV,free_bytes=shutil.disk_usage(run).free))
            log = run/'logs'/(task['id']+'.log'); log.parent.mkdir(exist_ok=True)
            start = time.monotonic(); proc = None; timed = False; error = None; produced = {}
            try:
                with log.open('x') as f:
                    proc = subprocess.Popen(task['command'],cwd=task['cwd'],env={**os.environ,**ENV},stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
                    try: proc.wait(timeout=task['timeout_s'])
                    except subprocess.TimeoutExpired: timed=True; kill_own(proc)
                if proc.returncode or timed: raise RuntimeError('child failure or timeout')
                produced = artifacts(task['output_manifest'])
            except BaseException as exc:
                error = repr(exc)
                if proc: kill_own(proc)
            write(receipt,dict(status='FAILED' if error else 'COMPLETE',command=task['command'],pid=proc.pid if proc else None,
                returncode=proc.returncode if proc else None,timed_out=timed,elapsed_s=time.monotonic()-start,
                log=str(log),log_sha256=sha(log) if log.exists() else None,error=error,artifacts=produced,no_training=True))
            if error: raise RuntimeError(error)
            completed.add(task['id'])
        completion=run/'media_completion.json'
        if not completion.exists():
            write(completion,dict(status='MEDIA_COMPLETE_FEATURE_HOLD',completed_tasks=len(completed),
                expected_new_media=60,feature_status=p['feature_status'],no_training=True))
    finally:
        (lock/'owner.json').unlink(); lock.rmdir()


def synthesize(run, digest):
    """Frozen L17 official full-ICL bundle, three native draws, no retries."""
    if sha(run/'protocol.json') != digest: raise ValueError('cell protocol mismatch')
    p = read(run/'protocol.json'); validate_settings(p); verify_pins(p['source_pins'])
    os.environ.update(ENV)
    import gc
    import inspect
    import random
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel, Qwen3TTSTokenizer
    sys.path.insert(0,str(ROOT))
    from scripts.experiments.qwen_full_tts_extension_20261022 import tts_stage as h
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False)
    def seed(value):
        random.seed(value); np.random.seed(value % 2**32); torch.manual_seed(value); torch.cuda.manual_seed_all(value)
    seed(20261022)
    modeldir=Path(p['model_dir']); out=run/'generation'; out.mkdir(exist_ok=False)
    w=Qwen3TTSModel.from_pretrained(str(modeldir),local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
    w.model.eval().requires_grad_(False); old=w.model.speech_tokenizer; w.model.speech_tokenizer=None
    del old; gc.collect(); torch.cuda.empty_cache()
    codec=Qwen3TTSTokenizer.from_pretrained(str(modeldir/'speech_tokenizer'),local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
    codec.model.eval().requires_grad_(False); w.model.speech_tokenizer=codec
    if w.model.dtype != torch.float32 or codec.model.dtype != torch.float32: raise ValueError('FP32 required')
    write(out/'runtime.json',dict(executable=sys.executable,python=sys.version,torch=torch.__version__,numpy=np.__version__,
        GPU=torch.cuda.get_device_name(),source_pins={inspect.getfile(type(m)):sha(inspect.getfile(type(m))) for m in [w,w.model,w.model.talker,w.model.talker.code_predictor,codec.model]},
        configs={name:dict(dtype=str(m.dtype),attention=getattr(m.config,'_attn_implementation',None)) for name,m in [('LM',w.model),('codec',codec.model)]},
        no_training=True))
    s=p['samples'][0]; text=h.strip_whitespace(s['transcript'])
    x,sr=sf.read(s['input24_path'],dtype='float32'); x=h.waveform(x,sr)
    with torch.inference_mode(): prompt=w.create_voice_clone_prompt(ref_audio=(x,24000),ref_text=text,x_vector_only_mode=False)
    if len(prompt)!=1 or not prompt[0].icl_mode or prompt[0].x_vector_only_mode: raise ValueError('full ICL required')
    ref=h.codes(prompt[0].ref_code); reference=h.save_array(out/'reference_codes.npy',ref); rows=[]
    for draw,base in enumerate(BASES,1):
        value=base+100*int(s['speaker_id'][1:]); seed(value); start=time.monotonic()
        with torch.inference_mode(),h.capture_official(w) as cap:
            waves,rate=w.generate_voice_clone(text=text,language='Chinese',voice_clone_prompt=prompt,non_streaming_mode=False,**KWARGS)
        if any(cap[k]!=1 for k in ['core_generate_calls','decoder_calls','talker_calls']): raise ValueError('capture count')
        new=cap['new_codes'][0]; joined=cap['decode_codes'][0]; full=cap['full_waves'][0]
        if not np.array_equal(joined,np.concatenate([ref,new])): raise ValueError('reference prefix concat')
        cut=h.official_cut(len(ref),len(joined),len(full)); wave=h.waveform(waves[0],rate)
        if not np.array_equal(full[cut:],wave): raise ValueError('official ratio cut')
        eos=cap['talker_kwargs'].get('eos_token_id'); role='L17_norm'+str(draw)
        record=dict(sample_id=s['sample_id'],speaker_id=s['speaker_id'],model='L17',role=role,draw_index=draw,seed=value,
            **h.save_wave(out/f'draw{draw}_raw_float.wav',wave),reference_codes=reference,
            new_codes=h.save_array(out/f'draw{draw}_new_codes.npy',new),
            talker_sequences=h.save_array(out/f'draw{draw}_talker_sequences.npy',cap['talker_sequences']),
            eos_in_talker_sequences=bool(np.any(cap['talker_sequences']==eos)) if isinstance(eos,int) else None,
            target_text=text,reference_text=text,reference_audio_sha256=s['input24_sha256'],
            official_ratio_cut_samples=cut,talker_effective_kwargs=cap['talker_kwargs'],core_explicit_kwargs=cap['core_kwargs'],
            duration_ratio=len(wave)/rate/s['duration_s'],elapsed_s=time.monotonic()-start)
        write(out/f'draw{draw}.json',record); rows.append(record)
    write(run/'raw_manifest.json',dict(records=rows,count=3,protocol_sha256=digest))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('stage',choices=['prepare','dryrun','execute','worker-synthesize'])
    ap.add_argument('--run',type=Path,default=ROOT/'runs'/NAME); ap.add_argument('--execution-root',default=REMOTE)
    ap.add_argument('--release',type=Path); ap.add_argument('--protocol-sha256'); args=ap.parse_args()
    if args.stage=='prepare': prepare(args.run,args.execution_root)
    elif args.stage=='dryrun': print(json.dumps(read(args.run/'dryrun.json'),ensure_ascii=False,indent=2))
    elif args.stage=='execute':
        if not args.release: ap.error('--release required')
        sequence=1
        while (args.run/'executor_attempts'/f'{sequence:03}').exists(): sequence+=1
        attempt=args.run/'executor_attempts'/f'{sequence:03}';attempt.mkdir(parents=True,exist_ok=False)
        launch=attempt/'launch.json';exit_path=attempt/'exit.json'
        write(launch,dict(pid=os.getpid(),command=[sys.executable,*sys.argv],source_sha256=sha(__file__),
            protocol_sha256=sha(args.run/'protocol.json'),release_sha256=sha(args.release),started_epoch=time.time(),no_training=True))
        start=time.monotonic();error=None
        try: execute(args.run,args.release)
        except BaseException as exc: error=repr(exc);raise
        finally:
            write(exit_path,dict(pid=os.getpid(),status='FAILED_OR_HOLD' if error else 'MEDIA_COMPLETE_FEATURE_HOLD',
                returncode=1 if error else 0,error=error,elapsed_s=time.monotonic()-start,no_training=True))
    else:
        if not args.protocol_sha256: ap.error('--protocol-sha256 required')
        synthesize(args.run,args.protocol_sha256)


if __name__=='__main__': main()
