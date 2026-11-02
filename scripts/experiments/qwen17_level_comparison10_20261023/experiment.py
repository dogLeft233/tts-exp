"""Frozen 10-person Qwen-size and whole-wave RMS comparison; no training."""
from pathlib import Path
import argparse, datetime, gc, hashlib, importlib.util, inspect, json, os, random, signal, subprocess, sys, time

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen17_level_comparison10_20261023_attempt01'
CLOUD=ROOT/'runs/qwen_audio31_cloud_subset10_20261023_attempt01'
OLD=ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01'
sys.path.insert(0,str(ROOT))

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(p):return json.loads(Path(p).read_text('utf8'))
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf8') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')

def role(model,level,draw):return f'{model}_{level}'+(str(draw) if draw else '')
def planned_roles():return ['N_raw']+[role(m,l,d) for m,l in [('L06','raw'),('CLOUD','raw'),('L17','norm'),('L17','raw')] for d in (1,2,3)]

def prepare():
    RUN.mkdir(exist_ok=False)
    cp=read(CLOUD/'protocol.json');op=read(OLD/'protocol.json')
    aa={(r['sample_id'],r['role']):r for r in read(OLD/'audio_manifest.json')['records']}
    ca={(r['sample_id'],r['role']):r for r in read(CLOUD/'audio_manifest.json')['records']}
    ss={(r['sample_id'],r['role']):r for r in cp['reused_official_scores']}
    sources=[];reuse=[]
    for s in cp['samples']:
        sid=s['sample_id']
        for model,draw in [('N',0)]+[(m,d) for m in ['L06','CLOUD'] for d in (1,2,3)]:
            oldrole='N' if model=='N' else ('T' if model=='L06' else 'C')+str(draw)
            a=(ca if model=='CLOUD' else aa)[sid,oldrole]
            src=a['provider_audio'] if model=='CLOUD' else a['source_path']
            h=a['provider_sha256'] if model=='CLOUD' else a['source_sha256']
            score=CLOUD/'official/scores'/f'{sid}_{oldrole}.json' if model=='CLOUD' else Path(ss[sid,oldrole]['path'])
            if model!='CLOUD' and sha(score)!=ss[sid,oldrole]['sha256']:raise ValueError('old score changed')
            if sha(src)!=h or sha(a['path'])!=a['sha256']:raise ValueError('old audio changed')
            r=read(score)
            if r['status']!='COMPLETE' or r['canonical_audio_sha256']!=a['sha256']:raise ValueError('old audio/score mismatch')
            identity=dict(sample_id=sid,speaker_id=s['speaker_id'],model=model,draw_index=draw)
            sources.append(dict(**identity,source_path=src,source_sha256=h,old_audio=a))
            reuse.append(dict(**identity,level='norm',reused=True,audio_path=a['path'],audio_sha256=a['sha256'],
                source_path=src,source_sha256=h,score_path=str(score),score_sha256=sha(score)))
    pins={str(p):sha(p) for p in [CLOUD/'protocol.json',CLOUD/'audio_manifest.json',OLD/'protocol.json',OLD/'audio_manifest.json',
       Path(__file__),Path(__file__).with_name('downstream.py'),ROOT/'scripts/experiments/qwen_full_tts_extension_20261022/tts_stage.py']}
    pins.update(cp['source_pins'])
    # Cloud's source pins cover the previous runner; retain rather than alter them.
    model=ROOT/'models/Qwen3-TTS-12Hz-1.7B-Base'
    deploy=read(ROOT/'runs/qwen3_tts_1p7b_deployment_20261023_attempt01/weights_manifest.json')
    for r in deploy['files']:
        f=model/r['path']
        if sha(f)!=r['sha256']:raise ValueError('deployed weight changed')
    d=dict(cp['downstream'],inference_sha256='42cfc8d3060921e6714c8e08243e5508d2e637d29cd0be78760492c5d26380a9')
    protocol=dict(run_id=RUN.name,created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),samples=cp['samples'],cohort='cloud10_identical',
       expected_cells=130,roles=planned_roles(),target_rms=cp['target_rms'],downstream=d,source_pins=pins,
       analysis_protocol_path=str(RUN/'analysis_protocol.json'),model_dir=str(model),weights_manifest=deploy,
       seed_bases=op['seed_bases'],generation_kwargs=op['generation_kwargs'],non_streaming_mode=False,
       execution=dict(stage_timeout_s=10800,generation_timeout_s=7200),no_training=True,
       raw_policy='same source waveform and 16k mono PCM16 path, unit gain, no trimming/padding/time warp; stop on clipping',
       norm_policy='whole complete waveform fixed RMS; no retry, output selection, time edit, limiter or target adjustment',
       normalized_existing_scores=70,new_syntheses=30,new_videos_and_scores=130,planned_total_comparisons=200,
       interpretation='provider and model bundle comparison, SyncNet score only; cloud weights not identified with open model')
    save(RUN/'analysis_protocol.json',dict(primary=['norm:L17-N','norm:L17-L06','norm:L17-CLOUD'],
       unit='10 speakers; three draws averaged within person',bootstrap_replicates=10000,seed=20261023,
       primary_interval=1-.05/3,descriptive_interval=.95,
       secondary='raw pairwise gaps; norm-minus-raw within model; differences of T-N gaps; no additional confirmatory claims',
       missing='retain every predetermined draw and report incomplete support, no best draw; all draws main'))
    save(RUN/'protocol.json',protocol);save(RUN/'existing_sources.json',dict(records=sources));save(RUN/'reused_scores.json',dict(records=reuse))
    snap=RUN/'source_snapshot';snap.mkdir()
    for p in [Path(__file__),Path(__file__).with_name('downstream.py')]:
        (snap/p.name).write_bytes(p.read_bytes())
    print('PREPARED',sha(RUN/'protocol.json'),flush=True)

def validate():
    p=read(RUN/'protocol.json')
    for f,h in p['source_pins'].items():
        if sha(f)!=h:raise ValueError('source pin changed '+f)
    return p

def generate():
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',CUBLAS_WORKSPACE_CONFIG=':4096:8',OMP_NUM_THREADS='2')
    p=validate();out=RUN/'generation';out.mkdir(exist_ok=False)
    import numpy as np
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel,Qwen3TTSTokenizer
    from scripts.experiments.qwen_full_tts_extension_20261022 import tts_stage as h
    start=time.monotonic()
    def event(name,**kw):
        r=dict(event=name,pid=os.getpid(),elapsed_s=time.monotonic()-start,**kw)
        with (out/'events.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
        print(json.dumps(r),flush=True)
    def alarm(*a):raise TimeoutError('generation7200s hardtimeout')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(7200)
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(False);torch.cuda.reset_peak_memory_stats()
    def seed(n):random.seed(n);np.random.seed(n%(2**32));torch.manual_seed(n);torch.cuda.manual_seed_all(n)
    seed(20261022);event('load_started')
    modeldir=Path(p['model_dir'])
    w=Qwen3TTSModel.from_pretrained(str(modeldir),local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
    w.model.eval().requires_grad_(False);old=w.model.speech_tokenizer;w.model.speech_tokenizer=None;del old;gc.collect();torch.cuda.empty_cache()
    codec=Qwen3TTSTokenizer.from_pretrained(str(modeldir/'speech_tokenizer'),local_files_only=True,device_map='cuda:0',dtype=torch.float32,attn_implementation='eager')
    codec.model.eval().requires_grad_(False);w.model.speech_tokenizer=codec
    if w.model.dtype!=torch.float32 or codec.model.dtype!=torch.float32:raise ValueError('FP32required')
    save(out/'runtime.json',dict(executable=sys.executable,torch=torch.__version__,gpu=torch.cuda.get_device_name(),
       configs={n:dict(dtype=str(m.dtype),attention=getattr(m.config,'_attn_implementation',None)) for n,m in [('LM',w.model),('codec',codec.model)]},
       no_training=True,protocol_sha256=sha(RUN/'protocol.json')))
    records=[]
    for s in p['samples']:
        sid=s['sample_id'];text=h.strip_whitespace(s['transcript']);directory=out/sid;directory.mkdir()
        if sha(s['input24_path'])!=s['input24_sha256']:raise ValueError('reference changed')
        x,sr=sf.read(s['input24_path'],dtype='float32');x=h.waveform(x,sr)
        with torch.inference_mode():prompt=w.create_voice_clone_prompt(ref_audio=(x,24000),ref_text=text,x_vector_only_mode=False)
        if len(prompt)!=1 or not prompt[0].icl_mode or prompt[0].x_vector_only_mode:raise ValueError('fullICLrequired')
        ref=h.codes(prompt[0].ref_code);refrec=h.save_array(directory/'reference_codes.npy',ref)
        for index,base in enumerate(p['seed_bases'],1):
            n=base+100*int(s['speaker_id'][1:]);seed(n);event('generation_started',sample_id=sid,draw_index=index,seed=n)
            t=time.monotonic()
            with torch.inference_mode(),h.capture_official(w) as cap:
                waves,rate=w.generate_voice_clone(text=text,language='Chinese',voice_clone_prompt=prompt,non_streaming_mode=False,**p['generation_kwargs'])
            if any(cap[k]!=1 for k in ['core_generate_calls','decoder_calls','talker_calls']):raise ValueError('capturecount')
            new=cap['new_codes'][0];joined=cap['decode_codes'][0];full=cap['full_waves'][0]
            if not np.array_equal(joined,np.concatenate([ref,new])):raise ValueError('prefixconcat')
            cut=h.official_cut(len(ref),len(joined),len(full));wave=h.waveform(waves[0],rate)
            if not np.array_equal(full[cut:],wave):raise ValueError('officialcut')
            eos=cap['talker_kwargs'].get('eos_token_id');eospresent=bool(np.any(cap['talker_sequences']==eos)) if isinstance(eos,int) else None
            r=dict(sample_id=sid,speaker_id=s['speaker_id'],model='L17',draw_index=index,seed=n,
               **h.save_wave(directory/f'draw{index}_raw_float.wav',wave),reference_codes=refrec,
               new_codes=h.save_array(directory/f'draw{index}_new_codes.npy',new),
               talker_sequences=h.save_array(directory/f'draw{index}_talker_sequences.npy',cap['talker_sequences']),
               reference_audio_sha256=s['input24_sha256'],target_text=text,reference_text=text,
               eos_in_talker_sequences=eospresent,duration_ratio=len(wave)/rate/s['duration_s'],
               talker_effective_kwargs=cap['talker_kwargs'],core_explicit_kwargs=cap['core_kwargs'],
               official_ratio_cut_samples=cut,elapsed_s=time.monotonic()-t)
            save(directory/f'draw{index}.json',r);records.append(r);event('generation_complete',sample_id=sid,draw_index=index,count=len(records),EOS=eospresent,duration=r['duration_s'])
    save(out/'tts_manifest.json',dict(records=records,count=len(records),protocol_sha256=sha(RUN/'protocol.json')))
    save(out/'completion.json',dict(status='COMPLETE',count=len(records),pid=os.getpid(),elapsed_s=time.monotonic()-start,
         peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),no_training=True))
    signal.alarm(0);event('finished',count=len(records))

def audio():
    import numpy as np
    import soundfile as sf
    p=validate();records=[];bridge=[]
    sources=read(RUN/'existing_sources.json')['records']
    sources += [dict(sample_id=r['sample_id'],speaker_id=r['speaker_id'],model='L17',draw_index=r['draw_index'],source_path=r['path'],source_sha256=r['sha256']) for r in read(RUN/'generation/tts_manifest.json')['records']]
    for r in sources:
        sid=r['sample_id'];model=r['model'];draw=r['draw_index']
        if sha(r['source_path'])!=r['source_sha256']:raise ValueError('rawsourcechanged')
        f=RUN/'float16'/sid/(role(model,'source',draw)+'.wav');f.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['/usr/bin/ffmpeg','-nostdin','-v','error','-n','-i',r['source_path'],'-ac','1','-ar','16000','-c:a','pcm_f32le',str(f)],check=True,timeout=60)
        x,sr=sf.read(f,dtype='float32');rms=float(np.sqrt(np.mean(x.astype(float)**2)))
        if sr!=16000 or x.ndim!=1 or not len(x) or not np.isfinite(x).all() or rms<=0:raise ValueError('invalid wave')
        # Reconstruct the existing normalized PCM independently, before any new video.
        norm=x*np.float32(p['target_rms']/rms)
        if max(abs(x))>=1 or max(abs(norm))>=1:raise ValueError('raw/norm clipping; no limiter or retry')
        if model!='L17':
            import io
            b=io.BytesIO();sf.write(b,norm,16000,format='WAV',subtype='PCM_16');b.seek(0)
            got,_=sf.read(b,dtype='int16');old,_=sf.read(r['old_audio']['path'],dtype='int16')
            if not np.array_equal(got,old):raise ValueError('oldnormalizedPCMbridgefailed')
            bridge.append(dict(sample_id=sid,model=model,draw_index=draw,pcm_exact=True,normalized_sha256=r['old_audio']['sha256']))
        for level,y,gain in [('raw',x,1.)]+([('norm',norm,p['target_rms']/rms)] if model=='L17' else []):
            name=role(model,level,draw);dest=RUN/'audio'/sid/(name+'_16k_pcm16.wav');dest.parent.mkdir(parents=True,exist_ok=True)
            if dest.exists():raise FileExistsError('nooverwrite')
            sf.write(dest,y,16000,subtype='PCM_16');pcm,_=sf.read(dest,dtype='int16')
            if len(pcm)!=len(x):raise ValueError('samplecountchanged')
            records.append(dict(sample_id=sid,speaker_id=r['speaker_id'],model=model,level=level,draw_index=draw,role=name,cohort=p['cohort'],
               path=str(dest),sha256=sha(dest),samples=len(pcm),sr=16000,pcm_sha256=hashlib.sha256(pcm.astype('<i2').tobytes()).hexdigest(),
               source_path=r['source_path'],source_sha256=r['source_sha256'],float16_path=str(f),float16_sha256=sha(f),
               gain=gain,rms_before=rms,rms_after=float(np.sqrt(np.mean((pcm.astype(float)/32768)**2))),no_time_editing=True))
    if len(records)!=130 or len(bridge)!=70:raise ValueError('coverage')
    save(RUN/'audio_manifest.json',dict(records=records,count=130,protocol_sha256=sha(RUN/'protocol.json')))
    save(RUN/'normalized_pcm_bridge.json',dict(status='PASS',count=70,records=bridge))
    print('AUDIO130_BRIDGE70_PASS',flush=True)

def index():
    rows=read(RUN/'reused_scores.json')['records']
    for r in read(RUN/'audio_manifest.json')['records']:
        score=RUN/'official/scores'/f"{r['sample_id']}_{r['role']}.json"
        rows.append(dict(sample_id=r['sample_id'],speaker_id=r['speaker_id'],model=r['model'],level=r['level'],draw_index=r['draw_index'],
          reused=False,audio_path=r['path'],audio_sha256=r['sha256'],source_path=r['source_path'],source_sha256=r['source_sha256'],
          score_path=str(score),score_sha256=sha(score) if score.exists() else None))
    if len(rows)!=200:raise ValueError('indexcoverage')
    save(RUN/'comparison_index.json',dict(records=rows,expected_cells=200,protocol_sha256=sha(RUN/'protocol.json')))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','generate','audio','index']);args=ap.parse_args()
    globals()[args.stage]()
