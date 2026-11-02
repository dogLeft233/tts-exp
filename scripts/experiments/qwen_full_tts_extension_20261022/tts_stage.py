"""Frozen full self-ICL TTS extension. Observational hooks retain official decoding.

No optimizer/training; only new run outputs. The primary waveform is the exact
official ICL return (reference-prefix context decoded, then official ratio cut).
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import gc
import hashlib
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


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()


def write(path,obj):
    with Path(path).open('x',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def strip_whitespace(text):
    return ''.join(c for c in str(text) if not c.isspace())


def array(value):
    if hasattr(value,'detach'):return value.detach().cpu().numpy().copy()
    return np.asarray(value).copy()


def codes(value):
    a=array(value)
    if a.ndim!=2 or a.shape[1]!=16 or a.shape[0]==0 or not np.issubdtype(a.dtype,np.integer):
        raise ValueError(f'expected nonempty integer [T,16] codes: {a.shape} {a.dtype}')
    if a.min()<0 or a[:,0].max()>=4096 or a[:,1:].max()>=2048:
        raise ValueError('codes outside semantic/acoustic vocabularies')
    return a


def waveform(values,sr):
    x=np.asarray(values,dtype=np.float32)
    if sr!=24000 or x.ndim!=1 or x.size==0 or not np.isfinite(x).all():
        raise ValueError('invalid raw mono24k float waveform')
    return x


def save_wave(path,x,sr=24000):
    x=waveform(x,sr);sf.write(path,x,sr,subtype='FLOAT')
    y,sy=sf.read(path,dtype='float32')
    if sy!=sr or not np.array_equal(x,y):raise RuntimeError('FLOAT readback mismatch')
    return {'path':str(Path(path).resolve()),'sha256':sha(path),'sample_rate_hz':sr,'samples':len(x),
            'float_pcm_sha256':hashlib.sha256(x.astype('<f4').tobytes()).hexdigest(),
            'duration_s':len(x)/sr,'rms':float(np.sqrt(np.mean(x.astype(float)**2))),
            'peak':float(np.max(abs(x))),'samples_abs_ge_1':int(np.sum(abs(x)>=1))}


def save_array(path,value):
    a=np.asarray(value);np.save(path,a,allow_pickle=False)
    return {'path':str(Path(path).resolve()),'sha256':sha(path),'shape':list(a.shape),'dtype':str(a.dtype)}


def official_cut(ref_steps,total_steps,full_samples):
    if not (0<ref_steps<total_steps) or full_samples<=0:
        raise ValueError('invalid reference/new decode partition')
    # Intentionally the official floating-ratio formula, including its rounding.
    return int(ref_steps/max(total_steps,1)*full_samples)


def simple(value):
    if value is None or isinstance(value,(bool,str,int,float)):return value
    if isinstance(value,(tuple,list)) and all(v is None or isinstance(v,(bool,str,int,float)) for v in value):return list(value)
    return {'type':type(value).__name__,'shape':list(value.shape) if hasattr(value,'shape') else None}


@contextmanager
def capture_official(wrapper):
    """Observe returns/inputs without modifying tensors or returned objects."""
    core=wrapper.model;tokenizer=core.speech_tokenizer;talker=core.talker
    original_generate=core.generate;original_decode=tokenizer.decode;original_talker=talker.generate
    captured={'core_generate_calls':0,'decoder_calls':0,'talker_calls':0}

    def observe_core(*args,**kwargs):
        result=original_generate(*args,**kwargs)
        captured['core_generate_calls']+=1
        captured['new_codes']=[codes(c) for c in result[0]]
        prompt=kwargs['voice_clone_prompt']
        captured['ref_codes']=[codes(c) for c in prompt['ref_code']]
        captured['core_kwargs']={k:simple(v) for k,v in kwargs.items() if k not in ('input_ids','ref_ids','voice_clone_prompt')}
        return result

    def observe_talker(*args,**kwargs):
        result=original_talker(*args,**kwargs)
        captured['talker_calls']+=1
        if not hasattr(result,'sequences'):raise RuntimeError('talker result lacks token sequences')
        captured['talker_sequences']=array(result.sequences)
        captured['talker_kwargs']={k:simple(v) for k,v in kwargs.items() if k not in ('inputs_embeds','attention_mask','trailing_text_hidden','tts_pad_embed')}
        return result

    def observe_decode(encoded):
        result=original_decode(encoded)
        captured['decoder_calls']+=1
        captured['decode_codes']=[codes(item['audio_codes']) for item in encoded]
        captured['full_waves']=[waveform(a,result[1]).copy() for a in result[0]]
        captured['sample_rate_hz']=int(result[1])
        return result

    core.generate=observe_core;talker.generate=observe_talker;tokenizer.decode=observe_decode
    try:yield captured
    finally:
        core.generate=original_generate;talker.generate=original_talker;tokenizer.decode=original_decode


def compare_waveforms(a,b):
    common=min(len(a),len(b))
    return {'sample_count_a':len(a),'sample_count_b':len(b),'length_delta_samples':len(a)-len(b),
            'array_exact':bool(np.array_equal(a,b)),
            'overlap_mae':float(np.mean(abs(a[:common].astype(float)-b[:common].astype(float)))) if common else None,
            'overlap_max_abs':float(np.max(abs(a[:common].astype(float)-b[:common].astype(float)))) if common else None,
            'interpretation':'sample-domain diagnostic, not perceptual or phoneme-timing equivalence'}


def validate_manifest(m):
    samples=m['samples'];bases=m['seed_bases']
    if len(samples)!=10 or len({r['sample_id'] for r in samples})!=10 or len({r['speaker_id'] for r in samples})!=10:
        raise ValueError('frozen 10-source/10-speaker cohort required')
    if bases!=[2026102201,2026102202,2026102203]:raise ValueError('unexpected seed bases')
    expected={'do_sample':True,'temperature':.9,'top_k':50,'top_p':1.,'max_new_tokens':512}
    if m['generation_kwargs']!=expected:raise ValueError('unexpected frozen generation settings')
    if m.get('non_streaming_mode',False) is not False:raise ValueError('non_streaming_mode changed')
    rows=[]
    for s in samples:
        text=strip_whitespace(s['transcript'])
        if not text:raise ValueError('empty transcript')
        for index,base in enumerate(bases,1):
            rows.append({'sample_id':s['sample_id'],'speaker_id':s['speaker_id'],'seed_index':index,
                         'seed':base+100*int(s['speaker_id'].removeprefix('S')),'target_text':text,'reference_text':text})
    return rows


def run(args):
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8';os.environ['OMP_NUM_THREADS']='2'
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    event_path=output/'events.jsonl'
    def event(kind,**fields):
        row={'event':kind,'elapsed_s':time.monotonic()-start,'utc_unix':time.time(),'pid':os.getpid(),**fields}
        with event_path.open('a') as f:f.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
        print(json.dumps(row,ensure_ascii=False,allow_nan=False),flush=True)
    def alarm(signum,frame):raise TimeoutError(f'TTS hard timeout after {args.timeout_seconds}s')
    if hasattr(signal,'SIGALRM'):signal.signal(signal.SIGALRM,alarm);signal.alarm(args.timeout_seconds)
    event('started',executable=sys.executable,command=sys.argv)
    try:
        manifest_path=Path(args.manifest).resolve()
        if sha(manifest_path)!=args.manifest_sha256:raise ValueError('manifest SHA not frozen expected value')
        manifest=json.loads(manifest_path.read_text());jobs=validate_manifest(manifest)
        model_dir=Path(manifest['model_dir']).resolve();codec_dir=model_dir/'speech_tokenizer'
        parent=Path(manifest['parent_run']).resolve()
        if sha(parent/'input_manifest.json')!=manifest['parent_manifest_sha256']:raise ValueError('parent manifest SHA mismatch')
        parent_inputs={r['sample_id']:r for r in json.loads((parent/'input_manifest.json').read_text())['samples']}
        parent_binding=json.loads((parent/'execution_bindings.json').read_text());parent_codec=parent/parent_binding['codec_directory']
        parent_results={r['sample_id']:r for r in json.loads((parent_codec/'results.json').read_text())['rows']}
        weights={'model.safetensors':manifest['model_weights_sha256'],'speech_tokenizer/model.safetensors':manifest['codec_weights_sha256']}
        for name,expected in weights.items():
            if sha(model_dir/name)!=expected:raise ValueError(f'weight SHA mismatch {name}')
        prepared={}
        for s in manifest['samples']:
            sid=s['sample_id'];original=parent_inputs[sid]
            if s['input24_path']!=original['input24_path'] or s['input24_sha256']!=original['input24_sha256']:raise ValueError('parent sample/reference binding mismatch')
            if sha(s['input24_path'])!=s['input24_sha256'] or sha(s['source_path'])!=s['source_sha256']:raise ValueError('input SHA mismatch')
            x,sr=sf.read(s['input24_path'],dtype='float32');x=waveform(x,sr)
            rec=parent_results[sid]
            if sha(rec['codes']['path'])!=rec['codes']['sha256'] or sha(rec['output']['path'])!=rec['output']['file_sha256']:raise ValueError('parent R payload mismatch')
            c=codes(np.load(rec['codes']['path'],allow_pickle=False));r,rs=sf.read(rec['output']['path'],dtype='float32');r=waveform(r,rs)
            prepared[sid]=(s,x,c,r,rec)
        import torch
        from qwen_tts import Qwen3TTSModel,Qwen3TTSTokenizer
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(2);torch.set_num_interop_threads(1)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(False);torch.cuda.reset_peak_memory_stats()
        random.seed(20261022);np.random.seed(20261022);torch.manual_seed(20261022);torch.cuda.manual_seed_all(20261022)
        event('loading_model',weights=weights)
        wrapper=Qwen3TTSModel.from_pretrained(str(model_dir),local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
        wrapper.model.eval().requires_grad_(False)
        # Official full-model loader consumes attn_implementation before loading
        # the tokenizer. Explicitly bind the identical parent tokenizer path.
        old_codec=wrapper.model.speech_tokenizer
        wrapper.model.speech_tokenizer=None
        del old_codec;gc.collect();torch.cuda.empty_cache()
        tokenizer=Qwen3TTSTokenizer.from_pretrained(str(codec_dir),local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
        tokenizer.model.eval().requires_grad_(False);wrapper.model.speech_tokenizer=tokenizer
        if tokenizer.model.dtype!=torch.float32:raise RuntimeError('codec dtype differs from parent')
        source_files={str(Path(inspect.getfile(cls)).resolve()):sha(inspect.getfile(cls)) for cls in [Qwen3TTSModel,Qwen3TTSTokenizer,type(wrapper.model),type(tokenizer.model)]}
        source_files[str(Path(__file__).resolve())]=sha(__file__)
        configs={}
        for name,module in [('LM',wrapper.model),('codec',tokenizer.model),('codec_encoder',getattr(tokenizer.model,'encoder',None)),('codec_decoder',getattr(tokenizer.model,'decoder',None))]:
            conf=getattr(module,'config',None);configs[name]={'type':type(module).__name__,'attention':getattr(conf,'_attn_implementation',None),'dtype':str(getattr(module,'dtype',None))}
        del module
        effective_defaults={k:simple(v.default) for k,v in inspect.signature(wrapper.model.generate).parameters.items() if v.default is not inspect.Parameter.empty}
        write(output/'runtime.json',{'manifest_sha256':sha(manifest_path),'weights':weights,'source_files':source_files,
              'packages':{n:importlib.metadata.version(n) for n in ['torch','transformers','qwen-tts','numpy','soundfile']},
              'device':'cuda:0','GPU':torch.cuda.get_device_name(0),'seed_policy':'base+100*numeric speaker ID, reset before each planned generation',
              'configs':configs,'core_generate_defaults':effective_defaults,'generation_kwargs':manifest['generation_kwargs'],
              'official_context_policy':'ref+new codes decoded jointly; official ratio cut removes reference waveform; standalone newcodes is separate diagnostic',
              'no_training':True,'strict_deterministic_algorithms':False})
        outputs=[];context_controls=[]
        for sample in manifest['samples']:
            sid=sample['sample_id'];s,x,parent_codes,parent_r,parent_rec=prepared[sid]
            text=strip_whitespace(s['transcript']);sample_dir=output/sid;sample_dir.mkdir()
            event('reference_prompt_started',sample_id=sid)
            with torch.inference_mode():prompt=wrapper.create_voice_clone_prompt(ref_audio=(x,24000),ref_text=text,x_vector_only_mode=False)
            if len(prompt)!=1 or not prompt[0].icl_mode or prompt[0].x_vector_only_mode:raise RuntimeError('full ICL reference mode mismatch')
            ref_codes=codes(prompt[0].ref_code)
            if not np.array_equal(ref_codes,parent_codes):raise RuntimeError('reference encoding differs from parent R codes')
            ref_code_saved=save_array(sample_dir/'reference_codes.npy',ref_codes)
            embedding=save_array(sample_dir/'reference_speaker_embedding.npy',array(prompt[0].ref_spk_embedding))
            write(sample_dir/'reference.json',{'sample_id':sid,'input24_path':s['input24_path'],'input24_sha256':s['input24_sha256'],
                  'ref_text':text,'target_text':text,'ref_codes':ref_code_saved,'speaker_embedding':embedding,'parent_reference_codes_exact':True})
            for job in [j for j in jobs if j['sample_id']==sid]:
                seed=job['seed'];index=job['seed_index'];directory=sample_dir/f'seed{index}';directory.mkdir()
                random.seed(seed);np.random.seed(seed%(2**32));torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
                event('generation_started',**{k:job[k] for k in ('sample_id','speaker_id','seed_index','seed')})
                t0=time.monotonic()
                with torch.inference_mode(),capture_official(wrapper) as captured:
                    returned,sr=wrapper.generate_voice_clone(text=text,language='Chinese',voice_clone_prompt=prompt,
                                           non_streaming_mode=False,**manifest['generation_kwargs'])
                if any(captured[k]!=1 for k in ['core_generate_calls','decoder_calls','talker_calls']):raise RuntimeError('unexpected number of official calls')
                if len(returned)!=1 or len(captured['new_codes'])!=1:raise RuntimeError('batch count mismatch')
                new=captured['new_codes'][0];ref=captured['ref_codes'][0];joined=captured['decode_codes'][0]
                if not np.array_equal(ref,ref_codes) or not np.array_equal(joined,np.concatenate([ref,new],axis=0)):raise RuntimeError('official reference/new decode concatenation mismatch')
                full=captured['full_waves'][0];cut=official_cut(len(ref),len(joined),len(full));official=waveform(returned[0],sr)
                if not np.array_equal(full[cut:],official):raise RuntimeError('official output does not match captured fullwave/cut')
                # Validate same-context decode, not the invalid newcodes-alone equality assumption.
                with torch.inference_mode():again,again_sr=tokenizer.decode([{'audio_codes':torch.as_tensor(joined,device=tokenizer.device)}])
                repeated=waveform(again[0],again_sr);repeat_cut=official_cut(len(ref),len(joined),len(repeated))
                if not np.array_equal(full,repeated) or repeat_cut!=cut or not np.array_equal(repeated[repeat_cut:],official):raise RuntimeError('same-concat exact repeat decode failed')
                primary=save_wave(directory/'official_T_raw_24k_float.wav',official)
                full_saved=save_wave(directory/'official_full_reference_plus_new_24k_float.wav',full)
                new_saved=save_array(directory/'new_codes.npy',new);concat_saved=save_array(directory/'decode_concat_codes.npy',joined)
                sequence_saved=save_array(directory/'talker_sequences.npy',captured['talker_sequences'])
                eos=captured['talker_kwargs'].get('eos_token_id');seq=captured['talker_sequences']
                has_eos=bool(np.any(seq==eos)) if isinstance(eos,int) else None
                # Diagnostic only: no reference decoder context, never primary T.
                with torch.inference_mode():alone_waves,alone_sr=tokenizer.decode([{'audio_codes':torch.as_tensor(new,device=tokenizer.device)}])
                alone=waveform(alone_waves[0],alone_sr);alone_saved=save_wave(directory/'diagnostic_T_newcodes_only_24k_float.wav',alone)
                diagnostic={'newcodes_only':alone_saved,'official_vs_newcodes_only':compare_waveforms(official,alone)}
                if index==1:
                    reference_join=np.concatenate([ref,parent_codes],axis=0)
                    with torch.inference_mode():context_waves,context_sr=tokenizer.decode([{'audio_codes':torch.as_tensor(reference_join,device=tokenizer.device)}])
                    context_full=waveform(context_waves[0],context_sr);context_cut=official_cut(len(ref),len(reference_join),len(context_full));context=context_full[context_cut:]
                    context_saved=save_wave(sample_dir/'diagnostic_R_with_reference_context_24k_float.wav',context)
                    context_full_saved=save_wave(sample_dir/'diagnostic_R_reference_plus_R_full_24k_float.wav',context_full)
                    context_codes_saved=save_array(sample_dir/'diagnostic_R_reference_plus_R_codes.npy',reference_join)
                    control={'sample_id':sid,'reference_steps':len(ref),'parent_R_steps':len(parent_codes),'official_ratio_cut_samples':context_cut,
                             'output':context_saved,'full_output':context_full_saved,'codes':context_codes_saved,
                             'parent_R_path':parent_rec['output']['path'],'parent_R_sha256':parent_rec['output']['file_sha256'],
                             'context_vs_parent_R':compare_waveforms(context,parent_r)}
                    write(sample_dir/'diagnostic_R_context_control.json',control);context_controls.append(control)
                result={**job,**primary,'primary_condition':'official full self-ICL TTS, prefix-context decode followed by official ratio cut',
                        'new_codes':new_saved,'reference_codes':ref_code_saved,'decode_concat_codes':concat_saved,'talker_sequences':sequence_saved,
                        'full_decoded':full_saved,'reference_steps':len(ref),'new_steps':len(new),'total_decode_steps':len(joined),
                        'official_ratio_cut_samples':cut,'nominal_ref_steps_times_1920':len(ref)*1920,
                        'cut_rounding_difference_samples':cut-len(ref)*1920,'same_concat_repeat_exact':True,
                        'talker_effective_kwargs':captured['talker_kwargs'],'core_explicit_kwargs':captured['core_kwargs'],
                        'eos_in_talker_sequences':has_eos,'possible_max_token_truncation':has_eos is False,
                        'diagnostics':diagnostic,'elapsed_s':time.monotonic()-t0}
                write(directory/'record.json',result);outputs.append(result)
                event('generation_complete',sample_id=sid,seed_index=index,seed=seed,samples=len(official),new_steps=len(new),eos_present=has_eos,elapsed_generation_s=result['elapsed_s'])
        if len(outputs)!=30 or len(context_controls)!=10:raise RuntimeError('incomplete frozen cohort')
        write(output/'tts_manifest.json',{'status':'COMPLETE','count':len(outputs),'manifest_sha256':sha(manifest_path),'records':outputs,
              'context_diagnostic_count':len(context_controls),'context_controls':context_controls,'selection_policy':'all three predetermined seeds retained, no score selection'})
        del prompt,wrapper,tokenizer;gc.collect();torch.cuda.synchronize();peak=int(torch.cuda.max_memory_allocated());torch.cuda.empty_cache()
        write(output/'completion.json',{'status':'COMPLETE','count':len(outputs),'elapsed_s':time.monotonic()-start,'peak_cuda_allocated_bytes':peak,
              'tts_manifest_sha256':sha(output/'tts_manifest.json'),'gpu_model_released':True})
        event('complete',count=len(outputs),gpu_model_released=True)
    except BaseException as exc:
        write(output/'failure.json',{'status':'FAILED','type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
        event('failed',error_type=type(exc).__name__,message=str(exc));raise
    finally:
        if hasattr(signal,'SIGALRM'):signal.alarm(0)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--manifest',required=True);parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--timeout-seconds',type=int,default=1800)
    run(parser.parse_args())
