"""Objective instance variation and repeat-noise diagnostics, CPU only."""
import sys
import numpy as np
import soundfile as sf
import python_speech_features
from scripts.experiments.tts_chinese_instance import BASE,ROOT,read,write,sha
from scripts.experiments.tts_phone_video_transfer import phones
from scripts.experiments.tts_chinese_instance_analysis import stats

def main():
    p=read(BASE/'protocol.json');manifest=read(BASE/'manifest.json');k=read(BASE/'calibration.json')['k']
    supports={s['id']:s for s in read(BASE/'mechanism'/f'primary_k{k}'/'support.json')}
    sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'));import audio
    rows=[];effects=[]
    for r in manifest['rows']:
        sid=r['id'];data={}
        for a in ('Q1','Q2'):
            if a not in r['audio']:continue
            x,sr=sf.read(r['audio'][a]['path'],dtype='int16');info=r['audio'][a]
            data[a]={'sha256':sha(info['path']),'duration':len(x)/sr,'rms':float(np.sqrt(np.mean((x.astype(float)/32768)**2))),'clipped_fraction':float(np.mean(abs(x.astype(float))>=32767)),'duration_ratio_to_N':len(x)/sr/r['natural_seconds']}
        item={'id':sid,'speaker':r['speaker'],'arms':data}
        if len(data)==2:item['distinct_pcm']=data['Q1']['sha256']!=data['Q2']['sha256']
        rows.append(item)
        s=supports.get(sid)
        if not s or not s['eligible']:continue
        aa=np.load(BASE/'audio_features'/(sid+'.npz'));mf={};mel={}
        for a in ('Q1','Q2'):
            x,sr=sf.read(r['audio'][a]['path'],dtype='int16');mf[a]=python_speech_features.mfcc(x,sr)
            mel[a]=audio.melspectrogram(audio.load_wav(r['audio'][a]['path'],16000)).astype(float)
        vals=[]
        for query in s['queries']:
            node=s['nodes'][query['node']];j1,j2=node['j']['Q1'],node['j']['Q2']
            vectors={'audio_embedding':(aa['Q1'][j1],aa['Q2'][j2]),'MFCC_window':(mf['Q1'][4*j1:4*j1+20].ravel(),mf['Q2'][4*j2:4*j2+20].ravel())}
            m1=int(np.floor((j1/25+.1075)*80+.5));m2=int(np.floor((j2/25+.1075)*80+.5))
            vectors['mel_center']=(mel['Q1'][:,min(m1,mel['Q1'].shape[1]-1)],mel['Q2'][:,min(m2,mel['Q2'].shape[1]-1)])
            z={}
            for name,(v1,v2) in vectors.items():
                d=float(np.linalg.norm(v1.astype(float)-v2.astype(float)));den=(np.linalg.norm(v1)+np.linalg.norm(v2))/2
                z[name+'_distance']=d;z[name+'_relative']=float(d/den)
            durations={a:node['span'][a][1]-node['span'][a][0] for a in ('N','Q1','Q2')}
            z['query_phone_duration_abs_difference']=abs(durations['Q1']-durations['Q2'])
            z['NQ_phone_duration_abs_difference']=.5*(abs(durations['N']-durations['Q1'])+abs(durations['N']-durations['Q2']))
            z['NQ_minus_QQ_duration_difference']=z['NQ_phone_duration_abs_difference']-z['query_phone_duration_abs_difference'];vals.append(z)
        effects.append({'id':sid,'speaker':r['speaker'],'effects':{key:float(np.mean([v[key] for v in vals])) for key in vals[0]}})
    sid=p['rows'][0]['id'];paths=[BASE/'tts'/sid/(a+'.wav') for a in ('Q1','Q1_replay','Q2')]
    pcm=[sf.read(f,dtype='int16')[0].astype(float) for f in paths]
    repeat={'same_file_sha':sha(paths[0])==sha(paths[1]),'lengths':[len(x) for x in pcm]}
    for label,index in [('repeat',1),('independent',2)]:
        n=min(len(pcm[0]),len(pcm[index]));repeat[label+'_common_sample_rmse']=float(np.sqrt(np.mean((pcm[0][:n]-pcm[index][:n])**2))/32768)
    write(BASE/'manipulation.json',{'records':rows,'event_effects':effects,'event_analysis':stats(effects),'same_seed_replay':repeat,'distinct_count':sum(r.get('distinct_pcm',False) for r in rows),'not_reading_accuracy':'forced alignment and signal diagnostics do not establish transcription correctness'})

if __name__=='__main__':main()
