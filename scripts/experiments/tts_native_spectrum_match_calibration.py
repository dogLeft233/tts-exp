"""CPU-only old26cal native long-term spectrum midpoint matching, frozen once."""
from pathlib import Path
import argparse,hashlib,json,time,types
import numpy as np
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'runs/tts_native_spectrum_match_calibration_20260926'
V1=ROOT/'runs/tts_acoustic_modulation_calibration_20260926'
V2=ROOT/'runs/tts_acoustic_modulation_calibration_20260926_v2'
STATIC=ROOT/'runs/tts_chinese_instance_primed_20260926/protocol.json'
CONDS=('raw','identity','EQ','ENVeq')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def frozen_module(folder,name):
    path=folder/'code_snapshot'/name;m=types.ModuleType('frozen_'+name[:-3]);m.__file__=str(ROOT/'scripts/experiments'/name)
    exec(compile(path.read_text(),str(path),'exec'),m.__dict__);return m
v1=frozen_module(V1,'tts_acoustic_modulation_calibration.py')
v2=frozen_module(V2,'tts_acoustic_modulation_env_control.py')
def spectral_vectors(x):
    z=v1.stft(x);a=abs(z);maximum=float(a.max());log=20*np.log10(np.maximum(a,maximum*1e-4)) if maximum else np.zeros_like(a)
    mean=log.mean(1);kernel=np.hanning(9);kernel/=kernel.sum();smooth=np.convolve(np.pad(mean,(4,4),mode='reflect'),kernel,mode='valid');shape=smooth-smooth[1:].mean()
    ar=a.mean(1);peak=float(ar.max());arlog=20*np.log10(np.maximum(ar,peak*1e-4)) if peak else np.zeros_like(ar)
    return {'mean_log_db':mean,'mean_log_centered_db':mean-mean[1:].mean(),'smooth_mean_log_shape_db':shape,'arithmetic_db':arlog,'arithmetic_centered_db':arlog-arlog[1:].mean()},z

def eq(x,gain_db):
    z=v1.stft(x);modified=z*10**(gain_db[:,None]/20);y=v1.istft(modified,len(x));g=v1.rms(x)/v1.rms(y) if v1.rms(y) else 1.;y*=g
    nz=abs(z)>0;ratio=modified[nz]/z[nz]
    return y,{'rms_restore_gain':g,'design_zero_bins_preserved':bool(np.all(modified[~nz]==0)),'design_zero_bin_fraction':float(np.mean(~nz)),'design_phase_error_max':float(np.max(abs(np.angle(ratio)))) if ratio.size else 0.}

def freeze():
    assert not (OUT/'protocol.json').exists();parent=read(V1/'protocol.json');static=read(STATIC);cal=[r for r in static['rows'] if r['split']=='calibration'];assert sorted(r['id'] for r in cal)==parent['cal_ids'];sources=[]
    for r in cal:
        for arm,key in [('N','N'),('T','C')]:
            c=r['audio'][key];old=next(x for x in parent['sources'] if x['id']==r['id'] and x['arm']==arm);assert c['sha256']==old['sha256']==sha(c['path']);sources.append({'id':r['id'],'arm':arm,'speaker':r['speaker'],**c})
    files=[Path(__file__),ROOT/'scripts/experiments/check_tts_native_spectrum_match_calibration.py',V1/'code_snapshot/tts_acoustic_modulation_calibration.py',V2/'code_snapshot/tts_acoustic_modulation_env_control.py']
    p={'status':'frozen_before_calibration','created_epoch':time.time(),'scope':'CPU only old26cal/52 native waveforms; no evaluation waveform, GPU, new TTS, new GxE scores or adaptive tuning','cal_ids':parent['cal_ids'],'sources':sources,'conditions':CONDS,'stft':{'sample_rate':16000,'window':'periodic Hann512','hop':128,'padding':'v1 reflect center256 plus whole-hop tail','log':'20log10(max(abs(X),max(abs(X))*1e-4)); all-zero uses zero log'},'spectrum':{'smooth':'mean_t log; reflectpad4 and normalized np.hanning(9) convolution','center':'subtract bins[1:] mean after smoothing','target':'m=(s_N+s_T)/2','gain_db':'clip(m-s_arm,-6,+6), frequency constant over time','eq':'single multiply original complex X; no floor energy fill; original length ISTFT; restore own global RMS; no iteration','distance':'RMS(s_N[1:]-s_T[1:]); raw nearzero<1e-8dB excluded from ratio only; retained in all output'},'ENVeq':'frozen v2 envelope_match(original arm, float64 EQ target), fixed20 steps; no early stopping','headroom':'one min(1,.98/maxpeak) for both arms x all4 conditions; save FLOAT32 pre_headroom and final waveforms','gates':{'finite_length_no_clip':True,'rms_relative_error_max':1e-5,'identity_float64_max_abs':1e-8,'pair_spectrum_reduction_median_min':.5,'pair_fraction_reduction_atleast25_min':.9,'ENVeq_EQ_hann_envelope_median_db_max':.35,'ENVeq_EQ_fraction_error_atmost075_min':.9,'EQ_and_ENVeq_R_absolute_relative_change_median_max':.02},'statistics':'descriptive cal only; all26 preserved; no SyncNet scoring or hypothesis inference','failure':'preserve failure and stop; no parameter/iteration/sample changes','parent_hashes':{str(f):sha(f) for f in [STATIC,V1/'protocol.json',V2/'protocol.json']},'code_hashes':{str(f):sha(f) for f in files}}
    write(OUT/'protocol.json',p);(OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n');(OUT/'code_snapshot').mkdir()
    for f in files:(OUT/'code_snapshot'/f.name).write_bytes(f.read_bytes())
    print('frozen cal26, no eval waveform or score read',flush=True)

def run():
    p=read(OUT/'protocol.json');assert sha(OUT/'protocol.json')==(OUT/'protocol.sha256').read_text().strip();assert not (OUT/'summary.json').exists()
    for f,h in {**p['parent_hashes'],**p['code_hashes']}.items():assert sha(f)==h,f
    metrics=[];pairs=[];manifest=[];traces={};gains={};identity=[]
    for sid in p['cal_ids']:
        sources={s['arm']:s for s in p['sources'] if s['id']==sid};inputs={}
        for arm,c in sources.items():
            assert sha(c['path'])==c['sha256'];x,sr=sf.read(c['path'],dtype='float64');assert sr==16000 and x.ndim==1;inputs[arm]=x
        vec={a:spectral_vectors(x)[0] for a,x in inputs.items()};target=(vec['N']['smooth_mean_log_shape_db']+vec['T']['smooth_mean_log_shape_db'])/2;before={};meta={}
        for arm,x in inputs.items():
            requested=target-vec[arm]['smooth_mean_log_shape_db'];gain=np.clip(requested,-6,6);yi,im=eq(x,np.zeros(257));ye,em=eq(x,gain);env,eg,trace=v2.envelope_match(x,ye)
            before[arm]={'raw':x,'identity':yi,'EQ':ye,'ENVeq':env};meta[arm]={'gain_clipping_fraction':float(np.mean(abs(requested)>6)),'gain_db_min':float(gain.min()),'gain_db_max':float(gain.max()),**em};identity.append(float(abs(yi-x).max()));traces[f'{sid}/{arm}']=trace
            write(OUT/'gains'/sid/(arm+'.json'),meta[arm]);npz=OUT/'gains'/sid/(arm+'.npz');np.savez_compressed(npz,requested_db=requested,gain_db=gain,target=target,original_shape=vec[arm]['smooth_mean_log_shape_db'],ENVeq_gain=eg)
        g=v2.shared_headroom(before);gains[sid]=g;pair_vectors={}
        for arm,x in inputs.items():
            raw_values=None;arm_rows={};pair_vectors[arm]={}
            for cond in CONDS:
                item={'id':sid,'arm':arm,'condition':cond}
                for stage,factor in [('pre_headroom',1.),('waveforms',g)]:
                    path=OUT/stage/sid/arm/(cond+'.wav');path.parent.mkdir(parents=True,exist_ok=True);sf.write(path,before[arm][cond]*factor,16000,subtype='FLOAT');item[stage]={'path':str(path),'sha256':sha(path)}
                y,_=sf.read(item['waveforms']['path'],dtype='float64');m,values=v1.analyze(y);vectors,_=spectral_vectors(y);values.update(vectors);values['sample_hann_envelope']=v2.envelope(y);pair_vectors[arm][cond]=vectors
                if cond=='raw':raw_values=values
                m.update(v1.couplings(values,raw_values));m.update({k+'_rmse_vs_raw':v1.rms((val-raw_values[k])[1:]) for k,val in vectors.items()})
                source_rms=v1.rms(x);expected=source_rms*g;row={'id':sid,'arm':arm,'speaker':sources[arm]['speaker'],'condition':cond,**m,'exact_length':len(y)==len(x),'restored_rms_relative_error':abs(v1.rms(before[arm][cond])-source_rms)/source_rms if source_rms else 0.,'saved_rms_relative_error':abs(v1.rms(y)-expected)/expected if expected else 0.,'common_headroom_gain':g,'zero_sample_fraction':float(np.mean(y==0)),'original_zero_sample_fraction':float(np.mean(x==0)),'pre_headroom_peak':float(abs(before[arm][cond]).max())}
                if cond in ('EQ','ENVeq'):row['R_RMS_relative_change_from_identity']=row['residual_rms_db']/arm_rows['identity']['residual_rms_db']-1 if arm_rows['identity']['residual_rms_db'] else 0.
                if cond=='ENVeq':
                    target_wave,_=sf.read(OUT/'waveforms'/sid/arm/'EQ.wav',dtype='float64');row['target_envelope_db_rmse']=v1.rms(20*np.log10(v2.envelope(y)/v2.envelope(target_wave)))
                arm_rows[cond]=row;metrics.append(row);path=OUT/'acoustics'/sid/arm/(cond+'.npz');path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,**values);item['acoustics']={'path':str(path),'sha256':sha(path)};manifest.append(item)
        pair={'id':sid,'speaker':sources['N']['speaker'],'distances':{}}
        for cond in CONDS:pair['distances'][cond]={k:v1.rms((pair_vectors['N'][cond][k]-pair_vectors['T'][cond][k])[1:]) for k in pair_vectors['N'][cond]}
        d=pair['distances']['raw']['smooth_mean_log_shape_db'];pair['near_zero_original']=d<1e-8;pair['EQ_reduction']=None if d<1e-8 else 1-pair['distances']['EQ']['smooth_mean_log_shape_db']/d;pairs.append(pair);print('calibrated',sid,flush=True)
    reduction=[r['EQ_reduction'] for r in pairs if r['EQ_reduction'] is not None];env=[r['target_envelope_db_rmse'] for r in metrics if r['condition']=='ENVeq'];changes={c:[abs(r['R_RMS_relative_change_from_identity']) for r in metrics if r['condition']==c] for c in ('EQ','ENVeq')}
    gates={'finite':all(r['all_finite'] for r in metrics),'exact_length':all(r['exact_length'] for r in metrics),'rms_restored':all(r['restored_rms_relative_error']<=1e-5 for r in metrics),'rms_saved':all(r['saved_rms_relative_error']<=1e-5 for r in metrics),'no_clip':all(r['sample_clipping_fraction']==0 and r['peak']<=.9800001 for r in metrics),'identity_float64':max(identity)<=1e-8,'spectrum_reduction_median':bool(len(reduction) and np.median(reduction)>=.5),'spectrum_reduction_90pct':bool(len(reduction) and np.mean(np.array(reduction)>=.25)>=.9),'ENVeq_envelope_median':bool(np.median(env)<=.35),'ENVeq_envelope_90pct':bool(np.mean(np.array(env)<=.75)>=.9),**{c+'_R_median':bool(np.median(v)<=.02) for c,v in changes.items()}}
    summary={'status':'calibration_passed' if all(gates.values()) else 'calibration_failed','gates':gates,'pair_spectrum':{c:{k:v1.summary_stats([r['distances'][c][k] for r in pairs]) for k in pairs[0]['distances'][c]} for c in CONDS},'spectrum_reduction':v1.summary_stats(reduction),'fraction_reduction_atleast25':float(np.mean(np.array(reduction)>=.25)) if reduction else None,'near_zero_original_ids':[r['id'] for r in pairs if r['near_zero_original']],'ENVeq_envelope_db_rmse':v1.summary_stats(env),'ENVeq_fraction_atmost075':float(np.mean(np.array(env)<=.75)),'R_absolute_relative_change':{c:v1.summary_stats(v) for c,v in changes.items()},'identity_float64_max_abs':max(identity),'headroom_gains':gains,'metrics_by_condition':{c:{key:v1.summary_stats([r.get(key) for r in metrics if r['condition']==c]) for key in ['envelope_db_rmse_vs_raw','envelope_linear_nrmse_vs_raw','f0_common_voiced_median_abs_cents','f0_voicing_disagreement_fraction','voiced_fraction','residual_rms_db','arithmetic_db_rmse_vs_raw','arithmetic_centered_db_rmse_vs_raw','mean_log_db_rmse_vs_raw','mean_log_centered_db_rmse_vs_raw','zero_sample_fraction','peak']} for c in CONDS},'gain_clipping_fraction':v1.summary_stats([read(OUT/'gains'/sid/(a+'.json'))['gain_clipping_fraction'] for sid in p['cal_ids'] for a in ('N','T')]),'scope':'old26cal only; no scoring or subsequent expansion'}
    for name,data in [('summary',summary),('metrics',metrics),('pairs',pairs),('waveform_manifest',manifest),('envelope_iterations',traces)]:write(OUT/(name+'.json'),data)
    print(json.dumps(summary['gates'],indent=2));print(summary['status'],flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=('freeze','run'));a=ap.parse_args();globals()[a.stage]()
