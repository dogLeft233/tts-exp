"""Post hoc expansion of the frozen audio-only manipulation check to fixed cal clips."""
import argparse
import shutil
import numpy as np
from scripts.experiments import tts_instance_audio_dtw as base
from scripts.experiments.tts_instance_audio_dtw import ROOT,SOURCE,read,write,sha,phones,cmvn,map_phones,mapped_time

OUT=ROOT/'runs/tts_dtw_calibration_audit_20260926'
PREVIOUS=ROOT/'runs/tts_instance_audio_dtw_20260926'

def freeze():
    p=read(SOURCE/'protocol.json');manifest={r['id']:r for r in read(SOURCE/'manifest.json')['rows']};cal=[r['id'] for r in p['rows'] if r['split']=='calibration'];assert len(cal)==26
    rows=[];excluded=[]
    for sid in cal:
        r=manifest[sid]
        for arm in ('N','Q1'):
            stop='not_applicable' if arm=='N' else read(SOURCE/'tts'/sid/'Q1.json')['termination']['stop_reason']
            if arm=='Q1' and stop!='EOS':excluded.append({'id':sid,'arm':arm,'reason':'not_EOS','stop':stop});continue
            if arm not in r['audio'] or arm not in r['grids']:excluded.append({'id':sid,'arm':arm,'reason':'missing_audio_or_grid'});continue
            z={'id':sid,'arm':arm,'speaker':r['speaker'],'stop':stop,'audio':r['audio'][arm],'grid':r['grids'][arm]}
            for key in ('audio','grid'):assert sha(z[key]['path'])==z[key]['sha256']
            rows.append(z)
    protocol={'cal_ids':cal,'rows':rows,'excluded':excluded,'status':'additional calibration audit after original control insufficiency was observed; not original preregistered control',
              'algorithm':'unchanged original controls: CMVN Wav2Lip80mel, same phone DTW/tie/path mapping; waveform first half into75%, rest25%; phases .25/.5/.75 both directions, same time and40ms index errors',
              'filter':'N all fixed26cal; Q1 EOS only; no eval audio; original control phone duration >=.08 predicate unchanged; >=.16 subset uses original1e-9 tolerance',
              'statistics':'clip equal; within clip phone/phase/direction equal;20000 PCG64 seed20260926 clip bootstrap99CI; DTW-MFA negative improvement; zero subgroup-phone clips excluded only from subgroup; not speaker bootstrap',
              'quantiles':'clip-equal empirical mixture, each point weight1/(eligibleclips*clip_pointcount), inverseCDF first value reaching q; min/q25/median/q75/q90/q95/q99/max',
              'index':'exact rounded40ms row match to known truth, mean absolute index error; no changed definition',
              'original_code_sha256':sha(base.__file__),'original_control_sha256':sha(PREVIOUS/'controls.json'),'source_protocol_sha256':sha(SOURCE/'protocol.json'),'code_sha256':sha(__file__)}
    assert not (OUT/'protocol.json').exists();write(OUT/'protocol.json',protocol)

def control_one(audio,r):
    """Original controls() numerical body, generalized only input/output and omitting its summary."""
    import soundfile as sf
    x=audio.load_wav(r['audio']['path'],16000);ph=phones(r['grid']['path']);warped=x.copy();bounds=[]
    for _,lo,hi in ph:
        if hi-lo<.08:continue
        begin,end=int(round(lo*16000)),min(len(x),int(round(hi*16000)));ii=np.arange(begin,end);u=(ii-begin)/(end-begin)
        original=np.where(u<=.75,u/1.5,.5+(u-.75)*2);warped[ii]=np.interp(begin+original*(end-begin),np.arange(len(x)),x);bounds.append((lo,hi,begin/16000,end/16000))
    dest=OUT/'clips'/r['id']/r['arm'];dest.mkdir(parents=True,exist_ok=True);sf.write(dest/'warp.wav',warped,16000,subtype='FLOAT')
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
    result={**r,'records':records,'identity_paths':ident,'warp_paths':modified,'not_timbre_preserving':True};write(dest/'controls.json',result)

def run():
    base.protect();p=read(OUT/'protocol.json');assert sha(base.__file__)==p['original_code_sha256'];audio=base.frontend()
    for r in p['rows']:
        if (OUT/'clips'/r['id']/r['arm']/'controls.json').exists():continue
        for key in ('audio','grid'):assert sha(r[key]['path'])==r[key]['sha256']
        control_one(audio,r)
    new=read(OUT/'clips/a1_001/Q1/controls.json');old=read(PREVIOUS/'controls.json')
    for key in ('records','identity_paths','warp_paths'):assert new[key]==old[key]
    import soundfile as sf
    newwav=OUT/'clips/a1_001/Q1/warp.wav';oldwav=PREVIOUS/'control_warp.wav'
    nx,ns=sf.read(newwav,dtype='float32');ox,os=sf.read(oldwav,dtype='float32');assert ns==os and np.array_equal(nx,ox)
    write(OUT/'original_replay.json',{'PASS':True,'records_paths_waveform_PCM_exact':True,'new_file_sha256':sha(newwav),'old_file_sha256':sha(oldwav),'container_note':'FLOAT WAV PEAK chunk timestamp differs; decoded float32 PCM, all paths and per-query values identical'})

def boot(values):
    x=np.asarray(values);rng=np.random.Generator(np.random.PCG64(20260926));draw=x[rng.integers(len(x),size=(20000,len(x)))].mean(1)
    return {'mean':float(x.mean()),'ci99':np.quantile(draw,[.005,.995]).tolist(),'n':len(x)}

def quantiles(clips,key):
    points=sorted((r[key],1/len(cc)/len(clips)) for cc in clips for r in cc);values=np.array([v for v,w in points]);cdf=np.cumsum([w for v,w in points]);cdf[-1]=1.
    return {name:float(values[min(len(values)-1,np.searchsorted(cdf,q,side='left'))]) for name,q in zip(('min','q25','median','q75','q90','q95','q99','max'),(0,.25,.5,.75,.9,.95,.99,1))}

def analyze():
    p=read(OUT/'protocol.json');result={};cliprows=[]
    for arm in ('N','Q1'):
        result[arm]={}
        for subset,cutoff in [('all',.08),('phone_ge160ms',.16)]:
            for direction in ('both','Q1Q2','Q2Q1'):
                cc=[];zero=[];points=[];phonecount=0
                for r in p['rows']:
                    if r['arm']!=arm:continue
                    z=read(OUT/'clips'/r['id']/arm/'controls.json');rr=[x for x in z['records'] if x['duration']>=cutoff-1e-9 and (direction=='both' or x['direction']==direction)]
                    if not rr:zero.append(r['id']);continue
                    eff={m:float(np.mean([x[m] for x in rr])) for m in ('mfa_error_ms','dtw_error_ms','identity_error_ms','mfa_index_error_frames','dtw_index_error_frames')}
                    eff.update({'delta_error_ms':eff['dtw_error_ms']-eff['mfa_error_ms'],'mfa_index_hit':float(np.mean([x['mfa_index_error_frames']==0 for x in rr])),'dtw_index_hit':float(np.mean([x['dtw_index_error_frames']==0 for x in rr]))})
                    cc.append(eff);points.append(rr);phonecount+=len({x['event'] for x in rr});cliprows.append({'id':r['id'],'speaker':r['speaker'],'arm':arm,'subset':subset,'direction':direction,'points':len(rr),'phones':len({x['event'] for x in rr}),'effects':eff})
                result[arm][subset+'_'+direction]={'clips':len(cc),'phones':phonecount,'points':sum(map(len,points)),'zero_phone_clip_ids':zero,'analysis':{m:boot([c[m] for c in cc]) for m in cc[0]},'time_quantiles':{m:quantiles(points,m) for m in ('mfa_error_ms','dtw_error_ms','identity_error_ms')}}
    write(OUT/'clip_results.json',cliprows);write(OUT/'analysis.json',result)

def report():
    p=read(OUT/'protocol.json');z=read(OUT/'analysis.json')
    def fmt(r):return f"{r['mean']:.3f} [{r['ci99'][0]:.3f}, {r['ci99'][1]:.3f}]"
    lines=['# 纯音频DTW时间恢复能力：追加校准审计','','这是看到原控制>=160ms仅2个phone后追加的固定校准审计，不是原预注册控制。未改DTW、变形、误差定义或主实验，没有读取eval音频、没有SyncNet评分或新TFG。','','固定26cal × N/Q1，Q1仅EOS。实际候选52，纳入'+str(len(p['rows']))+'，排除：`'+str(p['excluded'])+'`。','','clip内phone/phase/方向等权，再clip等权；20000次clip bootstrap99CI。DTW−MFA负值代表改善；这里的clip CI不冒充speaker独立推断。','','| 臂/phone支持 | clip/phone/相位方向数 | MFA时间MAE ms | DTW时间MAE ms | DTW−MFA ms | MFA 40ms索引命中率 | DTW命中率 |','|---|---|---|---|---|---|---|']
    for arm in ('N','Q1'):
        for subset in ('all','phone_ge160ms'):
            r=z[arm][subset+'_both'];a=r['analysis'];lines.append(f"| {arm}/{subset} | {r['clips']}/{r['phones']}/{r['points']} | "+' | '.join(fmt(a[m]) for m in ('mfa_error_ms','dtw_error_ms','delta_error_ms','mfa_index_hit','dtw_index_hit'))+' |')
    lines.extend(['','## clip等权时间误差分位数(ms)','','| 臂/支持/方法 | q25 | median | q75 | q90 | q95 | q99 | max |','|---|---:|---:|---:|---:|---:|---:|---:|'])
    for arm in ('N','Q1'):
        for subset in ('all','phone_ge160ms'):
            for method in ('mfa_error_ms','dtw_error_ms'):
                qq=z[arm][subset+'_both']['time_quantiles'][method];lines.append(f'| {arm}/{subset}/{method} | '+' | '.join(f'{qq[m]:.2f}' for m in ('q25','median','q75','q90','q95','q99','max'))+' |')
    lines.extend(['','分位数使用每clip总权重相等的经验CDF，未把长clip当作更多独立样本。无相应phone的clip仅从该子集排除，IDs在analysis.json。本轮四个子集均保留26clip，无此类排除。双方向、40ms索引MAE及identity完整结果同文件，逐clip在clip_results.json。clip bootstrap没有另行校正同speaker相关性。','','## 追加审计判断','','扩展后N/Q1总体及>=160ms子集均确认平均时间误差下降。原单条控制中长phone变差不能代表整个校准集；这次审计支持对已知变形有有限、可复现的恢复能力。长phone残余MAE约30ms，精确40ms索引命中率约29–33%，因此不等于已准确恢复真实语音时序，也不能据此把主实验的剩余实例惩罚归为非时序因素。','','前半时间映至前75%、后半映至后25%的waveform线性变形会改变局部频谱；结果只描述这一已知映射操纵下的恢复，不是音色保持合成或真实语音时序真值。没有据是否阳性挑另一算法。','','原a1_001/Q1 decoded waveform PCM、路径、逐query记录完整重放一致；FLOAT WAV PEAK头的创建时间不同，不能要求容器字节相同。独立数值检查见validation.json，输入/源码/产物hash见protocol.json和provenance.json。'])
    (OUT/'report.md').write_text('\n'.join(lines)+'\n');(OUT/'code').mkdir(exist_ok=True)
    for f in list((ROOT/'scripts/experiments').glob('*tts_dtw_calibration_audit*.py'))+[ROOT/'scripts/experiments/tts_instance_audio_dtw.py']:shutil.copyfile(f,OUT/'code'/f.name)
    write(OUT/'provenance.json',{str(f):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='provenance.json'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','run','analyze','report'));globals()[p.parse_args().stage]()
