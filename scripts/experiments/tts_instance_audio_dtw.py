"""Audio-only path construction. No SyncNet or visual arrays are read here."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'runs/tts_chinese_instance_primed_20260926'
OUT=ROOT/'runs/tts_instance_audio_dtw_20260926'
ORACLE=ROOT/'runs/tts_instance_local_oracle_20260926'

def read(p):return json.loads(Path(p).read_text())
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def phones(p):
    text=Path(p).read_text().split('name = "phones"',1)[1].split('item [',1)[0]
    return [(t,float(a),float(b)) for a,b,t in re.findall(r'intervals \[\d+\]:\s*xmin = ([\d.e+-]+)\s*xmax = ([\d.e+-]+)\s*text = "([^"]*)"',text) if t.strip() and t not in ('sil','sp','spn')]

def protect():
    """Block score arrays (and oracle outcomes) even if accidentally accessed."""
    def audit(event,args):
        if event=='open' and isinstance(args[0],(str,bytes)):
            p=str(Path(args[0]).resolve())
            if any(p.startswith(str(SOURCE/x)) for x in ('scores','audio_features','control_scores')) or p.startswith(str(ORACLE)):
                raise RuntimeError('audio-only stage attempted forbidden access: '+p)
    sys.addaudithook(audit)

def cmvn(x):return (x-x.mean(axis=0))/np.maximum(x.std(axis=0),1e-8)

def dtw(x,y):
    if not len(x) or not len(y):return [],None
    local=((x[:,None,:]-y[None,:,:])**2).sum(axis=2)
    cost=np.full((len(x)+1,len(y)+1),np.inf);cost[0,0]=0.;back=np.zeros((len(x),len(y)),dtype=np.uint8)
    for i in range(len(x)):
        for j in range(len(y)):
            options=(cost[i,j],cost[i,j+1],cost[i+1,j]);step=int(np.argmin(options));cost[i+1,j+1]=local[i,j]+options[step];back[i,j]=step
    i,j=len(x)-1,len(y)-1;path=[]
    while True:
        path.append([i,j])
        if i==0 and j==0:break
        step=back[i,j]
        if step in (0,1):i-=1
        if step in (0,2):j-=1
    return path[::-1],float(cost[-1,-1])

def map_phones(x,y,ph1,ph2):
    if [p[0] for p in ph1]!=[p[0] for p in ph2]:return {'status':'phone_sequence_mismatch','phones':[]}
    result=[]
    for n,(a,b) in enumerate(zip(ph1,ph2)):
        ix=np.where((np.arange(len(x))/80>=a[1])&(np.arange(len(x))/80<a[2]))[0]
        iy=np.where((np.arange(len(y))/80>=b[1])&(np.arange(len(y))/80<b[2]))[0]
        path,cost=dtw(x[ix],y[iy]);global_path=[[int(ix[i]),int(iy[j])] for i,j in path]
        result.append({'event':f'phone_{n}','phone':a[0],'spans':[list(a[1:]),list(b[1:])],'path':global_path,'cost':cost})
    return {'status':'complete','phones':result}

def mapped_time(phone,direction,t):
    path=np.asarray(phone['path']);src=0 if direction=='Q1Q2' else 1;dst=1-src
    if not len(path):return None,False
    xx=np.unique(path[:,src]);yy=np.array([path[path[:,src]==i,dst].mean() for i in xx]);outside=t<xx[0]/80 or t>xx[-1]/80
    return float(np.interp(t,xx/80,yy/80)),bool(outside)

def frontend():
    sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'));import audio
    assert audio.hp.sample_rate==16000 and audio.get_hop_size()==200 and audio.hp.num_mels==80 and not audio.hp.use_lws
    return audio

def freeze():
    protect();m=read(SOURCE/'manifest.json');assets={};rows=[]
    for r in m['rows']:
        item={'id':r['id'],'speaker':r['speaker'],'audio':{},'grids':{}}
        for a in ('Q1','Q2'):
            for kind in ('audio','grids'):
                z=r[kind][a];assert sha(z['path'])==z['sha256'];item[kind][a]=z;assets[z['path']]=z['sha256']
        rows.append(item)
    p={'rows':rows,'audio_assets':assets,'source_manifest_sha256':sha(SOURCE/'manifest.json'),
       'features':'exact Wav2Lip melspectrogram frontend output (normalized logmel), per-audio full-time per-band CMVN std=max(std,1e-8); frame time m/80 centered STFT',
       'dtw':'each identical ordered phone occurrence, no crossing; endpoints required; cumulative sum 80-D squared cost; steps diag,(1,0),(0,1); exact ties in that order',
       'mapping':'one common path, per-source-frame mean target index, both directions; linear interpolation at query audio center j/25+.1075; endpoint clamp',
       'indices':'closest target time among original MFA j+-2, same phone center, 0<=j<original arm valid length and 20<=j-k<length-20; tie abs shift then index; fallback original on missing path/empty pool',
       'support':'original primary20/mfa_only48, k3 primary,k2/4 sensitivity own frozen query sets; fixed visuals/self; no success selection',
       'control':'a1_001/Q1 calibration; identity; per phone >=80ms waveform warp first half into75%, second half25%, unchanged boundaries/duration; linear waveform sampling modifies local spectrum; compare MFA and DTW errors both directions, >=160ms separately',
       'scoring':'only after audio path AND query index seals; raw/unit native baselines; cross-self and DTW-MFA, speaker equal,20000 PCG64 seed20260926,95/99CI',
       'audio_frontend_sha256':sha(ROOT/'third_party/Wav2Lip/audio.py'),'hparams_sha256':sha(ROOT/'third_party/Wav2Lip/hparams.py'),'code_sha256':sha(__file__)}
    assert not (OUT/'protocol.json').exists();write(OUT/'protocol.json',p)

def audio_stage():
    protect();p=read(OUT/'protocol.json');audio=frontend()
    for r in p['rows']:
        target=OUT/'maps'/(r['id']+'.json')
        if target.exists():continue
        xx={};ph={}
        for a in ('Q1','Q2'):
            for kind in ('audio','grids'):assert sha(r[kind][a]['path'])==r[kind][a]['sha256']
            raw=audio.melspectrogram(audio.load_wav(r['audio'][a]['path'],16000)).T.astype(float);xx[a]=cmvn(raw);ph[a]=phones(r['grids'][a]['path'])
            f=OUT/'mel'/r['id']/(a+'.npz');f.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(f,raw=raw,cmvn=xx[a])
        result=map_phones(xx['Q1'],xx['Q2'],ph['Q1'],ph['Q2']);result.update(id=r['id'],speaker=r['speaker']);write(target,result)
    controls(audio,p)
    paths=[OUT/'protocol.json',OUT/'controls.json',OUT/'control_warp.wav',*[f for sub in ('maps','mel') for f in (OUT/sub).rglob('*') if f.is_file()]]
    write(OUT/'audio_mapping_seal.json',{'assets':{str(f):sha(f) for f in paths},'all_rows':len(p['rows']),'forbidden_reads_guard':'SOURCE scores/audio_features/control_scores and all oracle artifacts blocked by Python audit hook','stage':'audio mappings and controls complete before score access'})

def controls(audio,p):
    import soundfile as sf
    r=next(r for r in p['rows'] if r['id']=='a1_001');path=r['audio']['Q1']['path'];x=audio.load_wav(path,16000);ph=phones(r['grids']['Q1']['path']);warped=x.copy();bounds=[]
    for _,lo,hi in ph:
        if hi-lo<.08:continue
        begin,end=int(round(lo*16000)),min(len(x),int(round(hi*16000)));ii=np.arange(begin,end);u=(ii-begin)/(end-begin)
        original=np.where(u<=.75,u/1.5,.5+(u-.75)*2);warped[ii]=np.interp(begin+original*(end-begin),np.arange(len(x)),x);bounds.append((lo,hi,begin/16000,end/16000))
    sf.write(OUT/'control_warp.wav',warped,16000,subtype='FLOAT')
    z=cmvn(audio.melspectrogram(x).T.astype(float));w=cmvn(audio.melspectrogram(warped).T.astype(float));ident=map_phones(z,z,ph,ph);modified=map_phones(z,w,ph,ph);records=[]
    for n,(label,lo,hi) in enumerate(ph):
        if hi-lo<.08:continue
        _,_,b,e=next(v for v in bounds if v[0]==lo and v[1]==hi)
        for direction in ('Q1Q2','Q2Q1'):
            for phase in (.25,.5,.75):
                t=lo+phase*(hi-lo);u=(t-b)/(e-b)
                truth=b+(1.5*u if u<=.5 else .75+.5*(u-.5))*(e-b) if direction=='Q1Q2' else b+(u/1.5 if u<=.75 else .5+2*(u-.75))*(e-b)
                it,ic=mapped_time(ident['phones'][n],direction,t);dt,dc=mapped_time(modified['phones'][n],direction,t)
                records.append({'event':n,'duration':hi-lo,'direction':direction,'phase':phase,'source_time':t,'truth_time':truth,'dtw_time':dt,'identity_time':it,'endpoint_clamped':dc,'identity_clamped':ic,'mfa_error_ms':abs(t-truth)*1000,'dtw_error_ms':abs(dt-truth)*1000,'identity_error_ms':abs(it-t)*1000,'mfa_index_error_frames':abs(int(np.floor((t-.1075)*25+.5))-int(np.floor((truth-.1075)*25+.5))),'dtw_index_error_frames':abs(int(np.floor((dt-.1075)*25+.5))-int(np.floor((truth-.1075)*25+.5)))})
    summary={}
    for name,cutoff in [('all',.08),('phone_ge160ms',.16)]:
        rr=[r for r in records if r['duration']>=cutoff-1e-9];summary[name]={'n':len(rr),**{key:{'mean':float(np.mean([r[key] for r in rr])),'median':float(np.median([r[key] for r in rr])),'max':float(np.max([r[key] for r in rr]))} for key in ('mfa_error_ms','dtw_error_ms','identity_error_ms','mfa_index_error_frames','dtw_index_error_frames')}}
    write(OUT/'controls.json',{'id':'a1_001','records':records,'summary':summary,'identity_paths':ident,'warp_paths':modified,'not_timbre_preserving':True})

def indices():
    protect();seal=read(OUT/'audio_mapping_seal.json')
    for f,h in seal['assets'].items():assert sha(f)==h
    outputs={};supports={}
    for label in ('primary','mfa_only'):
        for k in (3,2,4):
            f=SOURCE/'mechanism'/f'{label}_k{k}'/'support.json';supports[str(f)]=sha(f);rows=[]
            for s in read(f):
                if not s['eligible']:continue
                mm={v['event']:v for v in read(OUT/'maps'/(s['id']+'.json'))['phones']}
                for qi,q in enumerate(s['queries']):
                    node=s['nodes'][q['node']];dirs={}
                    for a,b in (('Q1','Q2'),('Q2','Q1')):
                        origin=node['j'][b];source=node['j'][a]/25+.1075;target,endpoint=mapped_time(mm[node['event']],a+b,source) if node['event'] in mm else (None,False)
                        lo,hi=node['span'][b];js=[j for j in range(origin-2,origin+3) if 0<=j<s['lengths'][b] and 20<=j-k<s['lengths'][b]-20 and lo<=j/25+.1075<hi]
                        fallback=not js or target is None
                        chosen=origin if fallback else min(js,key=lambda j:(abs(j/25+.1075-target),abs(j-origin),j))
                        dirs[a+b]={'source_time':source,'target_time':target,'original_j':origin,'chosen_j':chosen,'visual_i':node['j'][a]-k,'self_j':node['j'][a],'fallback':fallback,'endpoint_clamped':endpoint,'candidate_indices':js,'target_outside_candidate_range':False if fallback else target<js[0]/25+.1075 or target>js[-1]/25+.1075,'shift_ms':(chosen-origin)*40}
                    rows.append({'id':s['id'],'speaker':s['speaker'],'query':qi,'event':node['event'],'phase':node['phase'],'directions':dirs})
            f=OUT/'indices'/f'{label}_k{k}.json';write(f,rows);outputs[str(f)]=sha(f)
    write(OUT/'query_mapping_seal.json',{'audio_mapping_seal_sha256':sha(OUT/'audio_mapping_seal.json'),'support_assets':supports,'mapping_assets':outputs,'stage':'all discrete mappings sealed before scoring'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','audio_stage','indices'));globals()[p.parse_args().stage]()
