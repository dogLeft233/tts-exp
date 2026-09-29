"""Pre-mechanism quality audit; never replaces or trims TTS outputs."""
import argparse
import importlib.metadata
import unicodedata
import numpy as np
import soundfile as sf
from scripts.experiments.tts_chinese_instance import BASE,ROOT,read,write,sha

MODEL=ROOT/'runs/mfa_linear_content_pause_prosody_mapping_20260925/asr_model_small'

def edit(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        new=[i]
        for j,y in enumerate(b,1):new.append(min(new[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=new
    return row[-1]

def asr():
    from faster_whisper import WhisperModel
    from opencc import OpenCC
    cc=OpenCC('t2s')
    def norm(s):return ''.join(c for c in cc.convert(unicodedata.normalize('NFKC',s).lower()) if not c.isspace() and not unicodedata.category(c).startswith('P'))
    p=read(BASE/'protocol.json')
    proto={'scope':'quality audit before MFA/TFG; preserve full waveforms; no sample selection by SyncNet','model':str(MODEL),'hashes':{str(f):sha(f) for f in MODEL.iterdir() if f.is_file()},'runtime':{'faster_whisper':importlib.metadata.version('faster-whisper'),'ctranslate2':importlib.metadata.version('ctranslate2')},'decoding':{'device':'cpu','compute_type':'int8','cpu_threads':4,'language':'zh','beam_size':5,'temperature':0,'vad_filter':False,'condition_on_previous_text':False,'initial_prompt':None},'normalization':'NFKC lowercase OpenCC t2s; strip whitespace and Unicode punctuation','statistics':'character edit distance / reference character length; report all arms, paired CER and duration; not human truth or forced-alignment content guarantee'}
    out=BASE/'quality';write(out/'protocol.json',proto)
    model=WhisperModel(str(MODEL),device='cpu',compute_type='int8',cpu_threads=4)
    for r in p['rows']:
        for a in ('N','C','Q1','Q2'):
            dest=out/'asr'/r['id']/(a+'.json')
            if dest.exists():continue
            if a in r['audio']:info=r['audio'][a]
            else:
                f=BASE/'tts'/r['id']/(a+'.json')
                if not f.exists() or read(f)['status']!='COMPLETE':continue
                info=read(f)
            assert sha(info['path'])==info['sha256']
            previous=ROOT/'runs/tts_chinese_instance_20260926/quality/asr'/r['id']/(a+'.json')
            if BASE.name!='tts_chinese_instance_20260926' and previous.exists() and read(previous)['audio_sha256']==info['sha256']:
                z=read(previous);z['reused_receipt_sha256']=sha(previous);z['reused_receipt']=str(previous);write(dest,z);continue
            segments,metadata=model.transcribe(info['path'],language='zh',beam_size=5,temperature=0,vad_filter=False,condition_on_previous_text=False)
            ss=[{'start':z.start,'end':z.end,'text':z.text,'avg_logprob':z.avg_logprob,'no_speech_prob':z.no_speech_prob} for z in segments]
            raw=''.join(z['text'] for z in ss);ref=norm(r['text']);hyp=norm(raw);d=edit(ref,hyp)
            write(dest,{'id':r['id'],'speaker':r['speaker'],'arm':a,'audio_sha256':info['sha256'],'raw':raw,'normalized':hyp,'reference':ref,'edit_distance':d,'cer':d/max(1,len(ref)),'segments':ss,'audio_duration':metadata.duration})
        print('quality_asr',r['id'],flush=True)

def summary():
    from scripts.experiments.tts_pcm_residual import cluster
    p=read(BASE/'protocol.json');rows=[]
    for r in p['rows']:
        d={'id':r['id'],'speaker':r['speaker'],'natural_seconds':r['natural_seconds'],'arms':{}}
        for a in ('N','C','Q1','Q2'):
            info=r['audio'].get(a)
            if info is None:
                f=BASE/'tts'/r['id']/(a+'.json')
                if not f.exists():continue
                info=read(f)
                if info['status']!='COMPLETE':continue
            duration=sf.info(info['path']).duration
            z={'duration':duration,'ratio':duration/r['natural_seconds'],'sha256':info['sha256']}
            if a.startswith('Q'):z['termination']=info.get('termination',{})
            ar=BASE/'quality/asr'/r['id']/(a+'.json')
            if ar.exists():z.update({k:read(ar)[k] for k in ('cer','edit_distance','normalized','reference')})
            d['arms'][a]=z
        rows.append(d)
    distributions={}
    for a in ('N','C','Q1','Q2'):
        vals=[r['arms'][a] for r in rows if a in r['arms']]
        z={'n':len(vals)}
        for key in ('duration','ratio','cer'):
            x=np.array([v[key] for v in vals if key in v]);z[key]={'n':len(x),'quantiles':dict(zip(('min','p25','median','p75','p90','p95','max'),np.quantile(x,[0,.25,.5,.75,.9,.95,1]).tolist())),'mean':float(x.mean())} if len(x) else None
        z['duration_gt30']=sum(v['duration']>30 for v in vals);z['ratio_outside_half_to_1p5']=sum(not .5<=v['ratio']<=1.5 for v in vals);z['ratio_gt3']=sum(v['ratio']>3 for v in vals)
        distributions[a]=z
    paired={}
    for a in ('C','Q1','Q2'):
        rr=[r for r in rows if a in r['arms'] and 'cer' in r['arms'][a] and 'cer' in r['arms']['N']]
        if rr:paired[a+'-N']=cluster([r['arms'][a]['cer']-r['arms']['N']['cer'] for r in rr],[r['speaker'] for r in rr])
    sid=p['rows'][0]['id'];f=BASE/'tts'/sid/'Q1_replay.wav';repeat=None
    if f.exists():
        original=BASE/'tts'/sid/'Q1.wav';x,sr=sf.read(original,dtype='int16');y,_=sf.read(f,dtype='int16');n=min(len(x),len(y))
        repeat={'id':sid,'sha_equal':sha(f)==sha(original),'lengths':[len(x),len(y)],'common_rmse':float(np.sqrt(np.mean(((x[:n].astype(float)-y[:n])/32768)**2)))}
    write(BASE/'quality/summary.json',{'records':rows,'distributions':distributions,'paired_cer':paired,'same_seed_replay':repeat,'abnormal_gt30':[{'id':r['id'],'arm':a,**z} for r in rows for a,z in r['arms'].items() if z['duration']>30],'token_limit':'Use recorded termination metadata when present; distinguish EOS, max_new_tokens, and context_capacity. Original uninstrumented outputs have unknown stops; waveform length alone is insufficient.'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('asr','summary'));globals()[p.parse_args().stage]()
