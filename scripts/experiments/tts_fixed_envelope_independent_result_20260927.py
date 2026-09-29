"""Independent scalar, cached-feature, curve and bootstrap audit; no model forward."""
import hashlib
import json
import math
import sys
import time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/tts_fixed_envelope_generation_cross_20260927'
OLD=ROOT/'runs/tts_fixed_level_generation_cross_20260927'
OUT=ROOT/'runs/tts_fixed_envelope_independent_audit_20260927'
read=lambda p:json.loads(Path(p).read_text())
ah=lambda x:hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def sha(p):
    d=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):d.update(block)
    return d.hexdigest()

def distance(V,A):
    # Reconstruct each query's complete lag neighborhood, independently of
    # producer's lag-column pairwise_distance loop.
    padded=np.pad(A,((15,15),(0,0)))
    out=[]
    for i in range(len(V)):
        delta=torch.from_numpy(V[i:i+1])-torch.from_numpy(padded[i:i+31])
        out.append(torch.linalg.vector_norm(delta+1e-6,ord=2,dim=1).numpy())
    return np.array(out,dtype='float64')

def endpoints(matrix,policy):
    L=len(matrix)
    if policy=='guard20':
        if L<=40:return None
        c=matrix[20:L-20].mean(axis=0)
    elif policy=='guard0':c=matrix.mean(axis=0)
    else:c=np.array([matrix[max(0,-k):min(L,L-k),k+15].mean() for k in range(-15,16)])
    j=int(c.argmin());B=float(np.median(c));D=float(c[j]);anchor=float(c[18])
    return {'C':B-D,'B':B,'D':D,'C_anchor':B-anchor,'D_anchor':anchor,'best_lag':j-15,'offset':15-j,'curve':c.tolist()}

def main():
    torch.set_num_threads(2);p=read(RUN/'protocol.json');summary=read(RUN/'summary.json')
    assert sha(RUN/'protocol.json')==read(RUN/'seal.json')['protocol_sha256']
    for path,h in p['dependencies'].items():assert sha(path)==h,path
    assert read(RUN/'input_gate.json')['passed'] and read(RUN/'cal_controls.json')['passed']
    for path,h in read(RUN/'input_seal.json')['operations'].items():assert sha(path)==h,path
    for split in ('calibration','evaluation'):
        seal=RUN/('feature_seal_'+split+'.json');lock=read(RUN/('score_lock_'+split+'.json'))
        assert sha(seal)==lock['feature_seal_sha256'] and sha(RUN/'input_seal.json')==lock['input_seal_sha256']
        for path,h in read(seal).items():assert sha(path)==h,path
    loaded=[read(RUN/'runtime'/('loaded_'+s+'.json')) for s in ('calibration','evaluation')]
    assert loaded[0]==loaded[1] and loaded[0]['runtime']==p['runtime']
    sys.path.insert(0,str(ROOT/'third_party/Wav2Lip'));import audio
    import python_speech_features as psf
    R=p['target_RMS'];W=np.hanning(513);W=W/W.sum();all_scores={};wave_count=0;curve_count=0
    max_curve=0.;max_old_baseline=0.;max_scalar=0.;matrix_values=0;manip={a:[] for a in 'NT'}
    for row in p['rows']:
        actual=read(RUN/'scores'/(row['id']+'.json'));cal=row['split']=='calibration';all_scores[row['id']]={}
        old_score=read(OLD/'scores'/(row['id']+'.json'))
        for arm in 'NT':
            z=row['arms'][arm];op=read(RUN/'operations'/row['id']/(arm+'.json'))
            assert sha(z['FIXED']['waveform'])==z['FIXED']['sha256'];assert sha(z['source']['path'])==z['source']['sha256']
            x,sr=sf.read(z['FIXED']['waveform'],dtype='float64');assert sr==16000
            e=np.sqrt(np.maximum(np.convolve(np.pad(x*x,256,mode='reflect'),W,mode='valid'),0))
            g=np.maximum(e/R,.1)**(-.5);y=x*g;a=R/np.sqrt(np.mean(y*y));g=g*a;y=(y*a).astype('float32')
            assert ah(y)==op['waveform_float32_sha256'] and ah(g)==op['gain_float64_sha256']
            assert np.isfinite(y).all() and len(y)==len(x) and np.all(y[x==0]==0) and np.all(g>0)
            assert max(abs(y))<=.98 and abs(np.sqrt(np.mean(y.astype(float)**2))/R-1)<=1e-6
            assert max(abs(y.astype(float)/g-x))<=1e-7
            ey=np.sqrt(np.maximum(np.convolve(np.pad(y.astype(float)**2,256,mode='reflect'),W,mode='valid'),0));S=e>=.1*R
            spread=np.diff(np.percentile(20*np.log10(e[S]),[10,90]))[0];after=np.diff(np.percentile(20*np.log10(ey[S]),[10,90]))[0]
            assert abs(spread-op['spread_input_dB'])<1e-12 and abs(after-op['spread_output_dB'])<1e-12
            if cal:manip[arm].append(float(after/spread))
            mel=audio.melspectrogram(y).astype('float32');mfcc=np.asarray(psf.mfcc(y.astype('float64')*32768,16000),dtype='float64')
            n=min(len(y)//640-5,(len(mfcc)-20)//4+1);windows=np.array([mfcc[i*4:i*4+20].T for i in range(n)],dtype='float32')
            assert ah(mel)==op['mel_sha256'] and ah(mfcc)==op['MFCC_float64_sha256'] and ah(windows)==op['windows_sha256']
            starts=op['mel_chunk_starts'];assert starts==[min(int(i*(80./25)),mel.shape[1]-16) for i in range(op['frames'])]
            assert op['MFCC_window_starts']==list(range(0,4*n,4))
            assert ah(np.array([mel[:,i:i+16] for i in starts],dtype='float32'))==op['chunks_float32_sha256']
            wave_count+=1
            assert sha(z['baseline_features'])==z['baseline_features_sha256']
            with np.load(z['baseline_features']) as f:fixed=(f['visual'],f['audio'])
            dest=RUN/'features'/row['id']/arm;meta=read(dest/'metadata.json')
            V=np.load(dest/'V.npy');A=np.load(dest/'A.npy');L=z['joint_L']
            for part,arr in [('visual',V),('audio',A)]:assert sha(meta[part]['path'])==meta[part]['sha256'] and ah(arr)==meta[part]['raw_sha256']
            assert meta['decode']['pixel_sha256']==meta['video_at_creation']['pixel_sha256']
            assert meta['decode']['frames']==meta['video_at_creation']['frames']==op['frames']
            assert L==min(op['frames'],len(y)//640)-5 and V.shape==(op['frames']-4,1024) and A.shape==(n,1024)
            all_scores[row['id']][arm]={}
            for geom in ('raw','unit'):
                arrays={}
                for name,pair in [('FIXED',fixed),('ENV',(V,A))]:
                    arrays[name]=tuple((q[:L].astype(float)/np.linalg.norm(q[:L].astype(float),axis=1)[:,None]).astype('float32') if geom=='unit' else q[:L] for q in pair)
                all_scores[row['id']][arm][geom]={}
                for cell,(vc,ac) in p['cells'].items():
                    if cal and cell not in ('q00','q11'):continue
                    mat=distance(arrays[vc][0],arrays[ac][1]);matrix_values+=mat.size
                    policies={}
                    for policy in ('guard20','valid','guard0'):
                        got=endpoints(mat,policy);expect=actual['cells'][arm][geom][cell][policy];policies[policy]=got
                        if got is None:assert expect is None;continue
                        for key,value in got.items():max_curve=max(max_curve,float(np.max(abs(np.asarray(value)-np.asarray(expect[key])))))
                        if cell=='q00':
                            legacy=old_score['cells'][arm][geom]['FIXED__FIXED']['policies'][policy]
                            for key,value in got.items():max_old_baseline=max(max_old_baseline,float(np.max(abs(np.asarray(value)-np.asarray(legacy[key])))))
                        curve_count+=1
                    all_scores[row['id']][arm][geom][cell]=policies
        print('independent',row['id'],flush=True)
    assert max_curve<1e-10 and max_old_baseline<1e-10,(max_curve,max_old_baseline)
    assert all(sum(x<.9 for x in manip[a])>=23 and np.median(manip[a])<.8 for a in 'NT')
    calrows=[r for r in p['rows'] if r['split']=='calibration'];controls=read(RUN/'cal_controls.json')
    old_controls=read(OLD/'cal_controls.json')
    for cond,cell in [('FIXED','q00'),('ENV','q11')]:
        lags={a:float(np.median([all_scores[r['id']][a]['raw'][cell]['guard20']['best_lag'] for r in calrows])) for a in 'NT'}
        assert lags==controls['condition_arm_medianlags'][cond]
        assert all(abs(lags[a]-3)<=1 and abs(lags[a]-old_controls['condition_arm_medianlags']['FIXED'][a])<=1 for a in 'NT')
        assert abs(lags['N']-lags['T'])<=1
    first=next(r for r in calrows if r['id']==p['controls']['first_cal'])
    for arm in 'NT':
        z=first['arms'][arm];L=z['joint_L']
        with np.load(z['RAW_features']) as f:rawV=f['visual'];rawA=f['audio']
        delayed=np.load(RUN/'controls'/arm/'delay5_A.npy')
        basecurve=distance(rawV[:L],rawA[:L])[25:L-25].mean(0)
        delaycurve=distance(rawV[:L],delayed[:L])[25:L-25].mean(0)
        shift=int(delaycurve.argmin()-basecurve.argmin());overlap=float(abs(delaycurve[5:]-basecurve[:-5]).max())
        assert abs(shift-5)<=1 and overlap<=.15
        assert abs(overlap-controls['delay'][arm]['overlap_error'])<1e-10
        for cond in ('RAW','FIXED','ENV'):
            c=read(RUN/'controls'/arm/(cond+'.json'))
            A=np.load(c['A']['path']);V=np.load(c['V']['path'])
            assert sha(c['A']['path'])==c['A']['sha256'] and sha(c['V']['path'])==c['V']['sha256']
            if cond=='RAW':refA,refV=rawA,rawV
            elif cond=='FIXED':
                with np.load(z['baseline_features']) as f:refA=f['audio'];refV=f['visual']
            else:
                refA=np.load(RUN/'features'/first['id']/arm/'A.npy');refV=np.load(RUN/'features'/first['id']/arm/'V.npy')
            assert np.array_equal(A,refA) and np.array_equal(V,refV)
            assert c['tests']['passed'] and c['tests']['pixel_equal'] and c['tests']['stream_full_exact']
    live=read(RUN/'first_cal_live.json');binding=read(RUN/'first_cal_media_audit.json');release=read(RUN/'first_cal_release.json')
    assert sha(binding['receipt'])==binding['receipt_sha256'];external=read(binding['receipt'])
    assert external['status']=='PASS' and external['protocol_sha256']==sha(RUN/'protocol.json')
    assert external['media_hashes']==live['files']==release['released'] and len(live['files'])==12
    assert live['time']<release['time']<read(RUN/'score_lock_calibration.json')['time']<read(RUN/'score_lock_evaluation.json')['time']
    count=0
    def stat(x,groups):
        names=sorted(set(groups));means=np.array([np.mean([v for v,g in zip(x,groups) if g==s]) for s in names])
        draws=np.random.default_rng(20260926).integers(len(names),size=(20000,len(names)))
        return {'mean':float(means.mean()),'ci99':np.quantile(means[draws].mean(1),[.005,.995]).tolist(),'n':len(x),'speakers':len(names),'group_means':dict(zip(names,means.tolist()))}
    def check_stat(got,expected):
        nonlocal max_scalar,count
        for k in got:
            if k=='group_means':err=max(abs(got[k][s]-expected[k][s]) for s in got[k])
            else:err=float(np.max(abs(np.asarray(got[k])-np.asarray(expected[k]))))
            max_scalar=max(max_scalar,err)
        count+=1
    for view,v in summary['views'].items():
        geom,policy,support=view.split('/');use=[r for r in p['rows'] if r['split']=='evaluation' and (support=='all74' or r['id'] in p['main_ids'])];groups=[r['speaker'] for r in use]
        assert len(use)==(74 if support=='all74' else 71)
        for metric,expected in v.items():
            q={a:np.array([[all_scores[r['id']][a][geom][c][policy][metric] for c in ['q00','q10','q01','q11']] for r in use]) for a in 'NT'}
            check_stat(stat(q['T'][:,0]-q['N'][:,0],groups),expected['baseline_T_minus_N'])
            check_stat(stat(q['T'][:,3]-q['N'][:,3],groups),expected['processed_T_minus_N'])
            effect={a:{'G':q[a][:,1]-q[a][:,0],'E':q[a][:,2]-q[a][:,0],'I':q[a][:,3]-q[a][:,1]-q[a][:,2]+q[a][:,0],'total':q[a][:,3]-q[a][:,0]} for a in 'NT'}
            for name in ['G','E','I','total']:
                for arm,x in [('N',effect['N'][name]),('T',effect['T'][name]),('T_minus_N',effect['T'][name]-effect['N'][name])]:check_stat(stat(x,groups),expected['effects'][name][arm])
    assert max_scalar<1e-10,max_scalar
    effect_rows=read(RUN/'effects.json');effect_error=0.
    for rec in effect_rows:
        geom,policy,support=rec['view'].split('/');v=all_scores[rec['id']][rec['arm']][geom]
        q=[v[c][policy][rec['metric']] for c in ('q00','q10','q01','q11')]
        expected={'q00':q[0],'q10':q[1],'q01':q[2],'q11':q[3],'G':q[1]-q[0],'E':q[2]-q[0],'I':q[3]-q[1]-q[2]+q[0],'total':q[3]-q[0]}
        effect_error=max(effect_error,max(abs(rec[k]-a) for k,a in expected.items()))
    assert effect_error<1e-10
    ci=summary['primary_total_T_minus_N']['ci99'];decision='GAP_SHRINK_CONFIRMED' if ci[1]<0 else 'GAP_EXPANSION_CONFIRMED' if ci[0]>0 else 'GAP_CHANGE_NOT_CONFIRMED';assert summary['status']==decision
    receipt={'status':'PASS','time':time.time(),'protocol_sha256':sha(RUN/'protocol.json'),'audit_code_sha256':sha(__file__),
      'all200_waveforms_gains_frontends_exact':wave_count,'distance_entries_recomputed':matrix_values,'policy_curves':curve_count,
      'curve_endpoint_max_error':max_curve,'old_FIXED_baseline_max_error':max_old_baseline,'statistic_endpoints':count,'statistic_max_error':max_scalar,
      'cal_migration_delay_and_exact_replays_independently_pass':True,'first_live_external_receipt_and_chronology_pass':True,
      'per_clip_effect_rows':len(effect_rows),'per_clip_effect_max_error':effect_error,
      'no_GPU_forward_or_media_read':True,'no_support_or_parameter_change':True,'summary_sha256':sha(RUN/'summary.json'),
      'features_seals':{s:sha(RUN/('feature_seal_'+s+'.json')) for s in ('calibration','evaluation')},'primary_status':decision}
    target=OUT/'result_receipt.json';assert not target.exists();target.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'receipt':str(target),'sha256':sha(target)}))

if __name__=='__main__':main()
