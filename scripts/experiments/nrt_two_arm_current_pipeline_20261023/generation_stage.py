"""Historical16: current frozen full-ICL pipeline, all16 codec round trips and48 draws.

Imports only the SHA-bound pilot's pure validation/observational hooks. One GPU
process, no training, no retries, no selection or changes to existing outputs.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import importlib.util
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
import numpy as np
import soundfile as sf

PILOT_SHA='38a48e335633c5fd2c5f0317dcf09c52fdacf756d7e5f9d09aef6c35f0f3c9aa'
BASES=[2026102201,2026102202,2026102203]
KWARGS={'do_sample':True,'temperature':.9,'top_k':50,'top_p':1.,'max_new_tokens':512}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def validate_manifest(m):
    rows=m['samples']
    ids=[1,8,15,22,29,35,41,47,53,59,65,71,77,83,89,95]
    if len(rows)!=16 or [int(r['original_sample_id']) for r in rows]!=ids:
        raise ValueError('frozen first original ID per actual speaker required')
    if len({r['sample_id'] for r in rows})!=16 or len({r['speaker_id'] for r in rows})!=16:
        raise ValueError('16 unique samples and true speakers required')
    if [r['speaker_id'] for r in rows]!=sorted(r['speaker_id'] for r in rows):raise ValueError('sorted true speakers required')
    if m['seed_bases']!=BASES or m['generation_kwargs']!=KWARGS or m.get('non_streaming_mode',False) is not False:
        raise ValueError('generation settings differ from frozen current pipeline')
    jobs=[]
    for row in rows:
        text=''.join(str(row['transcript']).split())
        if not text:raise ValueError('empty transcript')
        for index,base in enumerate(BASES,1):
            jobs.append(dict(sample_id=row['sample_id'],speaker_id=row['speaker_id'],seed_index=index,
                 seed=base+100*int(row['speaker_id'][1:]),target_text=text,reference_text=text))
    if len(jobs)!=48 or len({r['seed'] for r in jobs})!=48:raise ValueError('frozen48 draws required')
    return jobs


def require_exact(a,b,message):
    if not np.array_equal(a,b):raise RuntimeError(message)


class Budget:
    """Preflight payload sizes and check actual writes; no raw waves are cropped."""
    def __init__(self,directory,cap=1_000_000_000):self.directory=Path(directory);self.cap=cap
    def used(self):return sum(p.stat().st_size for p in self.directory.rglob('*') if p.is_file())
    def reserve(self,size):
        if self.used()+int(size)>self.cap:raise RuntimeError('generation disk cap would be exceeded')
    def check(self):self.reserve(0)
    def json(self,path,obj):
        payload=(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode('utf-8')
        self.reserve(len(payload))
        with Path(path).open('xb') as f:f.write(payload)
        self.check()
    def wave(self,helper,path,value):
        self.reserve(np.asarray(value).nbytes+4096)
        result=helper.save_wave(path,value);self.check();return result
    def array(self,helper,path,value):
        self.reserve(np.asarray(value).nbytes+4096)
        result=helper.save_array(path,value);self.check();return result


def load_helpers():
    path=Path(__file__).resolve().parents[1]/'qwen_full_tts_extension_20261022'/'tts_stage.py'
    if sha(path)!=PILOT_SHA:raise ValueError('pilot helper source SHA mismatch')
    spec=importlib.util.spec_from_file_location('frozen_full_tts_pilot',path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    return helper,path


def run(args):
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2')
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    budget=Budget(output);start=time.monotonic();events=output/'events.jsonl'
    def event(kind,**fields):
        row={'event':kind,'elapsed_s':time.monotonic()-start,'utc_unix':time.time(),'pid':os.getpid(),**fields}
        line=json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n';budget.reserve(len(line.encode()))
        with events.open('a') as f:f.write(line)
        print(line.strip(),flush=True)
    def alarm(signum,frame):raise TimeoutError(f'hard timeout {args.timeout_seconds}s')
    if hasattr(signal,'SIGALRM'):signal.signal(signal.SIGALRM,alarm);signal.alarm(args.timeout_seconds)
    event('started',command=sys.argv,executable=sys.executable)
    try:
        manifest_path=Path(args.manifest).resolve()
        if sha(manifest_path)!=args.manifest_sha256:raise ValueError('protocol SHA mismatch')
        m=json.loads(manifest_path.read_text());jobs=validate_manifest(m);h,helper_path=load_helpers()
        for path,expected in m['source_pins'].items():
            if sha(path)!=expected:raise ValueError('frozen source changed: '+path)
        if str(Path(__file__).resolve()) not in m['source_pins']:raise ValueError('generation source must be protocol-pinned')
        model_dir=Path(m['model_dir']).resolve();codec_dir=model_dir/'speech_tokenizer';parent=Path(m['parent_run'])
        weights={'model.safetensors':m['model_weights_sha256'],'speech_tokenizer/model.safetensors':m['codec_weights_sha256']}
        for name,value in weights.items():
            if sha(model_dir/name)!=value:raise ValueError('weights mismatch: '+name)
        inputs={}
        for row in m['samples']:
            for field,key in [('input24_path','input24_sha256'),('source_path','source_sha256')]:
                if sha(row[field])!=row[key]:raise ValueError('source/input binding mismatch')
            x,sr=sf.read(row['input24_path'],dtype='float32');x=h.waveform(x,sr)
            if len(x)!=row['input24_samples']:raise ValueError('frozen full input length mismatch')
            inputs[row['sample_id']]=x
        parent_input=json.loads((parent/'input_manifest.json').read_text())
        if sha(parent/'input_manifest.json')!=m['parent_manifest_sha256']:raise ValueError('parent manifest binding')
        bindings=json.loads((parent/'execution_bindings.json').read_text());cr=parent/bindings['codec_directory']
        parent_results={r['sample_id']:r for r in json.loads((cr/'results.json').read_text())['rows']}
        bridge_sample=parent_input['samples'][0];old_r=parent_results[bridge_sample['sample_id']]
        if sha(bridge_sample['input24_path'])!=bridge_sample['input24_sha256']:raise ValueError('bridge input changed')
        bx,bsr=sf.read(bridge_sample['input24_path'],dtype='float32');inputs[bridge_sample['sample_id']]=h.waveform(bx,bsr)
        if sha(old_r['codes']['path'])!=old_r['codes']['sha256'] or sha(old_r['output']['path'])!=old_r['output']['file_sha256']:
            raise ValueError('bridge assets mismatch')
        original_codes=h.codes(np.load(old_r['codes']['path'],allow_pickle=False))
        original_r,sr=sf.read(old_r['output']['path'],dtype='float32');original_r=h.waveform(original_r,sr)
        import torch
        from qwen_tts import Qwen3TTSModel,Qwen3TTSTokenizer
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(False);torch.cuda.reset_peak_memory_stats()
        def seed_all(seed):
            random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        seed_all(20261022);event('loading_model',weights=weights)
        wrapper=Qwen3TTSModel.from_pretrained(str(model_dir),local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
        wrapper.model.eval().requires_grad_(False)
        old_codec=wrapper.model.speech_tokenizer;wrapper.model.speech_tokenizer=None;del old_codec;gc.collect();torch.cuda.empty_cache()
        tokenizer=Qwen3TTSTokenizer.from_pretrained(str(codec_dir),local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
        tokenizer.model.eval().requires_grad_(False);wrapper.model.speech_tokenizer=tokenizer
        configs={}
        for name,module in [('LM',wrapper.model),('codec',tokenizer.model),('codec_encoder',getattr(tokenizer.model,'encoder',None)),('codec_decoder',getattr(tokenizer.model,'decoder',None))]:
            configs[name]={'type':type(module).__name__,'attention':getattr(getattr(module,'config',None),'_attn_implementation',None),'dtype':str(getattr(module,'dtype',None))}
        if any(c['dtype']!='torch.float32' for c in configs.values()):raise RuntimeError('non FP32 model')
        source_files={str(Path(inspect.getfile(cls)).resolve()):sha(inspect.getfile(cls)) for cls in [Qwen3TTSModel,Qwen3TTSTokenizer,type(wrapper.model),type(tokenizer.model)]}
        source_files.update({str(Path(__file__).resolve()):sha(__file__),str(helper_path):sha(helper_path)})
        budget.json(output/'runtime.json',{'protocol_sha256':sha(manifest_path),'weights':weights,'source_files':source_files,'configs':configs,
                    'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','qwen-tts','numpy','soundfile']},
                    'generation_kwargs':m['generation_kwargs'],'non_streaming_mode':False,
                    'core_generate_defaults':{k:h.simple(v.default) for k,v in inspect.signature(wrapper.model.generate).parameters.items() if v.default is not inspect.Parameter.empty},
                    'no_training':True,'device':'cuda:0','GPU':torch.cuda.get_device_name(0),'strict_deterministic_algorithms':False})
        def make_prompt(row):
            text=h.strip_whitespace(row['transcript'])
            with torch.inference_mode():p=wrapper.create_voice_clone_prompt(ref_audio=(inputs[row['sample_id']],24000),ref_text=text,x_vector_only_mode=False)
            if len(p)!=1 or not p[0].icl_mode or p[0].x_vector_only_mode:raise RuntimeError('full ICL mode mismatch')
            return p,h.codes(p[0].ref_code)
        def decode(code):
            with torch.inference_mode():waves,sr=tokenizer.decode([{'audio_codes':torch.as_tensor(code,device=tokenizer.device)}])
            if len(waves)!=1:raise RuntimeError('decoder batch size')
            return h.waveform(waves[0],sr)
        event('version_bridge_started',sample_id=bridge_sample['sample_id'])
        bridge_prompt,bridge_codes=make_prompt(bridge_sample);require_exact(bridge_codes,original_codes,'bridge encoding differs')
        bridge_wave=decode(bridge_codes);require_exact(bridge_wave,original_r,'bridge reconstruction differs')
        budget.json(output/'reference_bridge.json',{'status':'PASS','sample_id':bridge_sample['sample_id'],'encode_codes_exact':True,
                    'raw_R_float_exact':True,'input24_sha256':bridge_sample['input24_sha256'],'parent_codes':old_r['codes'],'parent_output':old_r['output'],
                    'codes':budget.array(h,output/'bridge_codes.npy',bridge_codes),'output':budget.wave(h,output/'bridge_R_raw_24k_float.wav',bridge_wave)})
        del bridge_prompt;event('version_bridge_pass')
        r_records=[];t_records=[]
        for row in m['samples']:
            sid=row['sample_id'];directory=output/sid;directory.mkdir();text=h.strip_whitespace(row['transcript'])
            event('reference_prompt_started',sample_id=sid,speaker_id=row['speaker_id'])
            prompt,ref=make_prompt(row);ref_saved=budget.array(h,directory/'reference_codes.npy',ref)
            embedding=budget.array(h,directory/'reference_speaker_embedding.npy',h.array(prompt[0].ref_spk_embedding))
            r_wave=decode(ref);r_primary=budget.wave(h,directory/'R_raw_24k_float.wav',r_wave)
            r_result={'sample_id':sid,'speaker_id':row['speaker_id'],'role':'R',**r_primary,'codes':ref_saved,'input24_path':row['input24_path'],
                      'input24_sha256':row['input24_sha256'],'speaker_embedding':embedding,'input_samples':len(inputs[sid]),
                      'output_minus_input_samples':len(r_wave)-len(inputs[sid]),'policy':'unaltered full standalone decode, no normalization/trim/pad/warp'}
            budget.json(directory/'reference.json',r_result);r_records.append(r_result)
            event('reference_codec_complete',sample_id=sid,samples=len(r_wave),code_steps=len(ref))
            for job in [j for j in jobs if j['sample_id']==sid]:
                index=job['seed_index'];seed=job['seed'];seed_dir=directory/f'seed{index}';seed_dir.mkdir()
                budget.reserve(16_000_000);seed_all(seed);event('generation_started',**{k:job[k] for k in ('sample_id','speaker_id','seed_index','seed')});t0=time.monotonic()
                with torch.inference_mode(),h.capture_official(wrapper) as captured:
                    returned,sr=wrapper.generate_voice_clone(text=text,language='Chinese',voice_clone_prompt=prompt,non_streaming_mode=False,**m['generation_kwargs'])
                if any(captured[k]!=1 for k in ('core_generate_calls','decoder_calls','talker_calls')):raise RuntimeError('official call count mismatch')
                if len(returned)!=1 or len(captured['new_codes'])!=1:raise RuntimeError('official batch mismatch')
                new=captured['new_codes'][0];captured_ref=captured['ref_codes'][0];joined=captured['decode_codes'][0]
                require_exact(ref,captured_ref,'captured reference mismatch');require_exact(joined,np.concatenate([ref,new]),'concat mismatch')
                full=captured['full_waves'][0];cut=h.official_cut(len(ref),len(joined),len(full));official=h.waveform(returned[0],sr)
                require_exact(full[cut:],official,'official waveform/cut mismatch');again=decode(joined)
                require_exact(full,again,'same-concat repeat not exact')
                if h.official_cut(len(ref),len(joined),len(again))!=cut:raise RuntimeError('repeat cut mismatch')
                eos=captured['talker_kwargs'].get('eos_token_id');seq=captured['talker_sequences']
                has_eos=bool(np.any(seq==eos)) if isinstance(eos,int) else None
                result={**job,'role':f'T{index}',**budget.wave(h,seed_dir/'official_T_raw_24k_float.wav',official),
                        'primary_condition':'official full self-ICL TTS, prefix-context decode followed by official ratio cut',
                        'new_codes':budget.array(h,seed_dir/'new_codes.npy',new),'reference_codes':ref_saved,
                        'decode_concat_codes':budget.array(h,seed_dir/'decode_concat_codes.npy',joined),
                        'talker_sequences':budget.array(h,seed_dir/'talker_sequences.npy',seq),
                        'full_decoded':budget.wave(h,seed_dir/'official_full_reference_plus_new_24k_float.wav',full),
                        'reference_steps':len(ref),'new_steps':len(new),'total_decode_steps':len(joined),'official_ratio_cut_samples':cut,
                        'nominal_ref_steps_times_1920':len(ref)*1920,'cut_rounding_difference_samples':cut-len(ref)*1920,
                        'same_concat_repeat_exact':True,'talker_effective_kwargs':captured['talker_kwargs'],'core_explicit_kwargs':captured['core_kwargs'],
                        'eos_in_talker_sequences':has_eos,'possible_max_token_truncation':has_eos is False,'elapsed_s':time.monotonic()-t0}
                budget.json(seed_dir/'record.json',result);t_records.append(result)
                event('generation_complete',sample_id=sid,seed_index=index,seed=seed,samples=len(official),new_steps=len(new),eos_present=has_eos,elapsed_generation_s=result['elapsed_s'])
        if len(t_records)!=48 or len(r_records)!=16:raise RuntimeError('incomplete frozen cohort')
        budget.json(output/'codec_manifest.json',{'status':'COMPLETE','count':16,'protocol_sha256':sha(manifest_path),'records':r_records})
        budget.json(output/'tts_manifest.json',{'status':'COMPLETE','count':48,'protocol_sha256':sha(manifest_path),'records':t_records,
                     'selection_policy':'all three predetermined draws retained, no score selection or retry'})
        raw_records=[dict(sample_id=r['sample_id'],speaker_id=r['speaker_id'],role='N',path=r['input24_path'],sha256=r['input24_sha256']) for r in m['samples']]
        raw_records += [dict(sample_id=r['sample_id'],speaker_id=r['speaker_id'],role=r['role'],path=r['path'],sha256=r['sha256']) for r in r_records+t_records]
        budget.json(output/'raw_manifest.json',dict(status='COMPLETE',count=80,protocol_sha256=sha(manifest_path),records=raw_records))
        del prompt,wrapper,tokenizer,module;gc.collect();torch.cuda.synchronize();peak=int(torch.cuda.max_memory_allocated());torch.cuda.empty_cache()
        budget.json(output/'completion.json',{'status':'COMPLETE','R_count':16,'T_count':48,'elapsed_s':time.monotonic()-start,
                     'new_R_count':16,'new_TTS_count':48,'pid':os.getpid(),'no_training':True,
                     'tts_manifest_path':str(output/'tts_manifest.json'),'codec_manifest_path':str(output/'codec_manifest.json'),
                     'peak_cuda_allocated_bytes':peak,'tts_manifest_sha256':sha(output/'tts_manifest.json'),'codec_manifest_sha256':sha(output/'codec_manifest.json'),
                     'gpu_model_released':True,'disk_bytes_before_completion':budget.used()})
        event('complete',R_count=16,T_count=48,gpu_model_released=True)
    except BaseException as exc:
        # Failure metadata is small; preserve original scientific output and never retry.
        payload={'status':'FAILED','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()}
        with (output/'failure.json').open('x') as f:json.dump(payload,f,indent=2)
        print(json.dumps(payload),flush=True);raise
    finally:
        if hasattr(signal,'SIGALRM'):signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--timeout-seconds',type=int,default=7200)
    args=parser.parse_args()
    if args.timeout_seconds!=7200:parser.error('frozen hard timeout must be 7200 seconds')
    run(args)
