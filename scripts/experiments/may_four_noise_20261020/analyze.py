"""Independent read-back of four-family frozen noise scan and descriptive plots."""
from pathlib import Path
import sys,json,argparse,hashlib,collections
sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/plotdeps')
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def unit(a):
    a=a.astype(float);return a/np.sqrt(np.sum(a*a,1))[:,None]
def independent_moments(x,y):
    x=x.astype(float);y=y.astype(float);d=y-x;u=unit(y)-unit(x)
    return np.column_stack((np.einsum('ij,ij->i',x,x),np.einsum('ij,ij->i',y,y),np.einsum('ij,ij->i',d,d),np.einsum('ij,ij->i',u,u),np.abs(d).max(1),(d!=0).sum(1)))
def normalized(x,center=True):
    x=x.astype(float).copy()
    if center:x-=x.mean()
    return (x/np.sqrt(np.mean(x*x))).astype(np.float32)

def verify_noise_sources(out,p):
    import soundfile as sf
    from scipy.signal import welch
    for pin in p['input_pins']:
        assert sha(pin['source_audio'])==pin['source_sha256']
        raw,sr=sf.read(pin['source_audio'],dtype='float64');assert sr==16000
        x=raw.astype(np.float32);native=(x-x.mean())/np.sqrt(x.var()+1e-7)
        assert digest(native)==pin['native_sha256'];np.testing.assert_array_equal(native[:320080],np.load(out/(pin['split']+'_native.npy')))
    sources={}
    for r in p['interference_source_pins']:
        assert sha(r['path'])==r['sha256'];x,sr=sf.read(r['path'],dtype='float64');assert sr==16000 and x.ndim==1;sources[r['speaker']]=x
    powers=[]
    for x in sources.values():
        _,v=welch(x[:960000],fs=16000,window='hann',nperseg=1024,noverlap=512,detrend='constant',scaling='density');powers.append(v/v.sum())
    psd=np.mean(powers,0);psd=np.maximum(psd,psd.max()*1e-8);psd[0]=0
    assert sha(out/'speech_shape_psd.npy')==p['speech_psd_sha256'];np.testing.assert_array_equal(psd,np.load(out/'speech_shape_psd.npy'))
    for r in p['noise_pins']:
        assert sha(out/r['path'])==r['sha256'];n=np.load(out/r['path']);m=r['construction'];length=len(n)
        if r['kind'] in ['single_speech','babble4']:
            parts=[]
            for c in m['components']:
                x=sources[c['speaker']][c['start_sample']:c['start_sample']+length];assert len(x)==length;parts.append(x)
            expected=normalized(parts[0]) if r['kind']=='single_speech' else normalized(sum(normalized(x).astype(float) for x in parts)/2)
        elif r['kind']=='modulated_ssn':
            rng=np.random.default_rng(m['generator_seed']);z=rng.standard_normal(length);f=np.fft.rfftfreq(length,1/16000)
            shape=np.sqrt(np.interp(f,np.linspace(0,8000,len(psd)),psd));shape[0]=0;z=np.fft.irfft(np.fft.rfft(z)*shape,n=length)
            t=np.arange(length)/16000;f1,f2=m['envelope_frequencies_hz'];p1,p2=m['envelope_phases']
            envelope=.15+.85*(.5+.5*np.sin(2*np.pi*f1*t+p1))*(.5+.5*np.sin(2*np.pi*f2*t+p2));expected=normalized(z*envelope)
        else:
            assert r['kind']=='clicks';z=np.zeros(length)
            for event in m['events']:
                start=event['start'];width=event['width'];t=np.arange(width)/16000
                click=np.hanning(width)*np.exp(-3*np.arange(width)/width)*np.sin(2*np.pi*event['carrier_hz']*t+event['phase'])
                click[1:-1]-=click.sum()/(width-2);end=min(start+width,length);z[start:end]+=event['amplitude']*click[:end-start]
            expected=normalized(z,center=False)
        expected*=np.float32(r['native_rms']*10**(-30/20));np.testing.assert_array_equal(expected,n)
        snr=20*np.log10(r['native_rms']/np.sqrt(np.mean(n.astype(float)**2)));assert abs(snr-30)<1e-5
    return len(p['noise_pins'])

def main(out):
    p=read(out/'protocol.json');done=read(out/'completion.json');assert done['status']=='COMPLETED' and done['forwards']==746
    assert sha(out/'protocol.json')==done['protocol_sha256'] and sha(Path(__file__).with_name('run.py'))==p['script_sha256']
    previous=Path(p['previous']);assert sha(previous/'delivery_manifest.json')==p['previous_manifest_sha256']
    old_manifest={r['path']:r for r in read(previous/'delivery_manifest.json')['members']}
    noise_count=verify_noise_sources(out,p)
    rows=read(out/'rows.json');calls=read(out/'forward_calls.json');controls=read(out/'controls.json');waves=read(out/'waveform_audit.json')
    assert len(calls)==746 and len(set(c['path'] for c in calls))==746
    assert len(rows)==4844 and len(waves)==344
    recovered=(out/'recovery_protocol.json').exists()
    assert collections.Counter(c['kind'] for c in controls)==({'previous_all_stage_anchor_bridge':4,'recovery_all_stage_anchor_bridge':38,'deterministic_repeat':2,'epsilon_compensation':48} if recovered else {'previous_all_stage_anchor_bridge':8,'deterministic_repeat':2,'epsilon_compensation':48})
    bykey={(r['precision'],r['split'],r['kind'],r['seed'],r['snr_db'],r['arm'],r['stage']):r for r in rows};assert len(bykey)==len(rows)
    floors={(r['split'],r['arm'],r['stage']):r['unit_delta'] for r in rows if r['precision']=='clean32vs64'}
    ratios=[]
    for a in rows:
        if a['precision']=='clean32vs64' or a['arm']!='A':continue
        key=(a['precision'],a['split'],a['kind'],a['seed'],a['snr_db'],'B',a['stage']);b=bykey[key]
        resolved=a['precision']=='fp64' or (a['unit_delta']>10*floors[a['split'],'A',a['stage']] and b['unit_delta']>10*floors[a['split'],'B',a['stage']])
        ratios.append(dict(precision=a['precision'],split=a['split'],kind=a['kind'],seed=a['seed'],snr_db=a['snr_db'],stage=a['stage'],A=a['unit_delta'],B=b['unit_delta'],ratio=b['unit_delta']/a['unit_delta'] if a['unit_delta'] else None,relative_l2_ratio=b['relative_l2']/a['relative_l2'] if a['relative_l2'] else None,resolved=bool(resolved)))
    write(out/'ratios.json',ratios)
    native={s:np.load(out/(s+'_native.npy')) for s in ['train','test']};checks=0;sample_pairs=0;last_frames=0;moment_max_error=0.;bridge_count=0
    noise_prefix={(r['split'],r['kind'],r['seed']):np.load(out/r['path'])[:320080].copy() for r in p['noise_pins']}
    for c in calls:
        parts=c['path'].split('/');mode,split=parts[:2];dtype=np.float32 if mode=='fp32' else np.float64;s=native[split].astype(dtype);arm=parts[-1]
        if len(parts)==3:y=s*(.5 if arm=='anchorB' else 1)
        else:
            kind,seed=parts[2].rsplit('_',1);q=int(parts[3][3:]);n=noise_prefix[split,kind,int(seed)].astype(dtype)*dtype(10**((30-q)/20));y=(s+n)*(.5 if arm in ['B','B_compensated'] else 1)
        assert digest(y)==c['input_sha256'];path=out/c['path'];last=np.load(path/'last.npy');assert digest(last)==c['last_sha256'];checks+=1
        samples=np.load(path/'samples.npz')
        if len(parts)==3:
            prior_label='anchorA' if arm=='repeatA' else arm;old_path=previous/mode/split/prior_label
            for stage in p['stages']:
                old_file=old_path/(stage+'.npy');assert sha(old_file)==old_manifest[old_file.relative_to(previous).as_posix()]['sha256']
                a=np.load(old_file,mmap_mode='r');np.testing.assert_array_equal(samples[stage],a[samples[stage+'__indices']])
                if stage=='last':np.testing.assert_array_equal(last,a)
                bridge_count+=1
        elif arm in ['A','B']:
            reference=out/mode/split/('anchor'+arm);ref_samples=np.load(reference/'samples.npz');frames=np.load(path/'frames.npz')
            for stage in p['stages']:
                idx=samples[stage+'__indices'];assert np.array_equal(idx,ref_samples[stage+'__indices'])
                rebuilt=independent_moments(ref_samples[stage],samples[stage]);np.testing.assert_allclose(rebuilt,frames[stage][idx],rtol=3e-14,atol=1e-28)
                moment_max_error=max(moment_max_error,float(np.max(np.abs(rebuilt-frames[stage][idx]))));sample_pairs+=len(idx)
                r=bykey[mode,split,kind,int(seed),q,arm,stage]
                assert r['unit_delta']==float(np.sqrt(frames[stage][:,3].mean())) and r['relative_l2']==float(np.sqrt(frames[stage][:,2].sum()/frames[stage][:,0].sum()))
            np.testing.assert_allclose(independent_moments(np.load(reference/'last.npy'),last),frames['last'],rtol=3e-14,atol=1e-28);last_frames+=len(last)
        else:
            assert arm=='B_compensated';a_path=path.parent/'A';a=np.load(a_path/'samples.npz');np.testing.assert_array_equal(last,np.load(a_path/'last.npy'))
            assert all(np.array_equal(a[k],samples[k]) for k in p['stages'])
        if checks%100==0:print(json.dumps(dict(verified_inputs=checks)),flush=True)
    # Cross-precision floor metrics are independently checked at saved raw sample rows.
    for split in native:
        for arm in ['A','B']:
            a=np.load(out/'fp32'/split/('anchor'+arm)/'samples.npz');b=np.load(out/'fp64'/split/('anchor'+arm)/'samples.npz');frames=np.load(out/'fp64'/split/('precision_anchor'+arm)/'frames.npz')
            for stage in p['stages']:
                rebuilt=independent_moments(a[stage],b[stage]);np.testing.assert_allclose(rebuilt,frames[stage][a[stage+'__indices']],rtol=3e-14,atol=1e-28)
                assert floors[split,arm,stage]==float(np.sqrt(frames[stage][:,3].mean()))
    for w in waves:
        s=native[w['split']].astype(np.float32 if w['precision']=='fp32' else np.float64)
        n=noise_prefix[w['split'],w['kind'],w['seed']].astype(s.dtype)*s.dtype.type(10**((30-w['snr_db'])/20))
        d=s+n-s;e=(s+n)*.5-s*.5
        assert np.array_equal(d,2*e) and float(np.mean(d!=0))==w['input_changed_fraction']==w['half_changed_fraction']
        active=n!=0;assert active.mean()==w['active_fraction'];assert abs(float(np.max(np.abs(n)))-w['noise_peak'])<1e-12
    extra_checks=0
    if recovered:
        assert done['recovery_protocol_sha256']==sha(out/'recovery_protocol.json')
        refpins=read(out/'reference_pins.json')
        for c in read(out/'recovery_bridge_calls.json'):
            label=c['path'].split('/')[-1];arm=label[-1];job_index=int(c['path'].split('/')[1][3:]);job=read(out/'jobs.json')[job_index];mode=job['mode'];split=job['split']
            y=native[split].astype(np.float32 if mode=='fp32' else np.float64)*(.5 if arm=='B' else 1);assert digest(y)==c['input_sha256']
            folder=out/c['path'];last=np.load(folder/'last.npy');assert digest(last)==c['last_sha256'];samples=np.load(folder/'samples.npz')
            for stage in p['stages']:
                pin=next(x for x in refpins if x['mode']==mode and x['split']==split and x['arm']==arm and x['stage']==stage);assert c['stage_sha256'][stage]==pin['array_sha256']
                a=np.load(pin['path'],mmap_mode='r');np.testing.assert_array_equal(samples[stage],a[samples[stage+'__indices']])
                if stage=='last':np.testing.assert_array_equal(last,a)
            extra_checks+=1
        assert extra_checks==34
    summaries=[]
    for mode,grid in [('fp32',p['snr_grid_db']),('fp64',p['fp64_grid_db'])]:
        for kind in p['kinds']:
            for split in ['train','test','both']:
                for q in grid:
                    r=[r for r in ratios if r['precision']==mode and r['kind']==kind and r['stage']=='last' and r['snr_db']==q and (split=='both' or r['split']==split)];v=[r['ratio'] for r in r if r['ratio'] is not None]
                    summaries.append(dict(precision=mode,kind=kind,split=split,snr_db=q,cases=len(r),resolved=sum(x['resolved'] for x in r),ratio_min=min(v) if v else None,ratio_median=float(np.median(v)) if v else None,ratio_max=max(v) if v else None,
                        all_smaller=bool(len(v)==len(r) and all(x['resolved'] and x['ratio']<1 for x in r)),all_at_least20pct_smaller=bool(len(v)==len(r) and all(x['resolved'] and x['ratio']<=.8 for x in r)),
                        A_median=float(np.median([x['A'] for x in r])),B_median=float(np.median([x['B'] for x in r]))))
    write(out/'summary.json',summaries)
    precision=[]
    for mode in ['fp32','fp64']:
        for split in native:
            for kind in p['kinds']:
                for q in p['fp64_grid_db']:
                    for arm in ['A','B']:
                        if mode!='fp32':continue
                        delta=[]
                        for dtype in ['fp32','fp64']:
                            folder=out/dtype/split;anchor=np.load(folder/('anchor'+arm)/'last.npy');val=np.load(folder/f'{kind}_{p["seeds"][0]}'/f'snr{q:03d}'/arm/'last.npy');delta.append(unit(val)-unit(anchor))
                        precision.append(dict(split=split,kind=kind,snr_db=q,arm=arm,fp32_response_relative_error_to_fp64=float(np.linalg.norm(delta[0]-delta[1])/np.linalg.norm(delta[1]))))
    write(out/'precision_diagnostics.json',precision)
    findings=[]
    for kind in p['kinds']:
        finding={'kind':kind}
        for split in ['train','test','both']:
            ss=[r for r in summaries if r['precision']=='fp32' and r['kind']==kind and r['split']==split]
            finding[split]=dict(all_smaller_levels=[r['snr_db'] for r in ss if r['all_smaller']],all_at_least20pct_smaller_levels=[r['snr_db'] for r in ss if r['all_at_least20pct_smaller']])
        findings.append(finding)
    write(out/'findings.json',findings)
    write(out/'independent_qa.json',dict(passed=True,noise_arrays_regenerated_exact=noise_count,source_audio_fresh_sha=6,may_full_split_standardization_exact=2,input_hashes=checks,extra_recovery_bridge_hashes=extra_checks,raw_stage_frame_samples=sample_pairs,full_last_frame_pairs=last_frames,anchor_stage_bridges=bridge_count,epsilon_pairs=48,waveform_pairs=len(waves),max_moment_sample_error=moment_max_error,forwards=746,admitted_forwards=746+extra_checks,training_updates=0))
    plot(out,p,ratios,summaries)
    print(json.dumps(dict(passed=True,findings=findings),ensure_ascii=False),flush=True)

def plot(out,p,ratios,summaries):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    names={'single_speech':'单录音背景讲话','babble4':'四人 babble','modulated_ssn':'带起伏的语音谱形噪声','clicks':'稀疏合成点击声'}
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained');fig.suptitle('May 四类噪声：整体音量×0.5后，编码影响减小还是增大？\n阴影：三实例范围；×：部分FP32条件进入精度诊断区；比值<1为减敏',fontsize=14)
    for ax,kind in zip(axs.flat,p['kinds']):
        for split,color in [('train','#176cb2'),('test','#c05b2a')]:
            rr=[r for r in summaries if r['precision']=='fp32' and r['kind']==kind and r['split']==split];x=[r['snr_db'] for r in rr];y=[r['ratio_median'] for r in rr]
            ax.plot(x,y,'o-',color=color,label=split+' · 三实例中位数');ax.fill_between(x,[r['ratio_min'] for r in rr],[r['ratio_max'] for r in rr],color=color,alpha=.15)
            unresolved=[r for r in rr if r['resolved']<r['cases']]
            ax.scatter([r['snr_db'] for r in unresolved],[r['ratio_median'] for r in unresolved],marker='x',s=90,color='#555',zorder=5)
        ax.axhline(1,color='#444',lw=1);ax.axhline(.8,color='#777',ls=':',lw=.8);ax.set(title=names[kind],xlabel='名义整 split 添加噪声 SNR（dB，越大越弱）',ylabel='低音量 / 原音量的末层噪声方向扰动');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.savefig(out/'four_noise_response_ratios.png',dpi=155);fig.savefig(out/'four_noise_response_ratios.svg');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained');fig.suptitle('同一首实例的 FP32 / FP64 核对（不是重新训练）',fontsize=17)
    for ax,kind in zip(axs.flat,p['kinds']):
        for split,color in [('train','#176cb2'),('test','#c05b2a')]:
            for mode,ls,marker in [('fp32','-','o'),('fp64','--','x')]:
                rr=[next(r for r in ratios if r['precision']==mode and r['split']==split and r['kind']==kind and r['seed']==p['seeds'][0] and r['snr_db']==q and r['stage']=='last') for q in p['fp64_grid_db']]
                ax.plot(p['fp64_grid_db'],[r['ratio'] for r in rr],ls,marker=marker,color=color,label=split+' '+mode.upper())
        ax.axhline(1,color='#444',lw=.8);ax.set(title=names[kind],xlabel='名义整 split SNR（dB）',ylabel='低音量 / 原音量噪声方向扰动');ax.grid(alpha=.2);ax.legend(fontsize=8);ax.ticklabel_format(axis='y',style='plain',useOffset=False)
    fig.savefig(out/'four_noise_precision.png',dpi=155);fig.savefig(out/'four_noise_precision.svg');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained');fig.suptitle('噪声变弱时的绝对编码扰动（各自匹配音量参照）',fontsize=17)
    for ax,kind in zip(axs.flat,p['kinds']):
        for split,color in [('train','#176cb2'),('test','#c05b2a')]:
            for arm,ls in [('A','-'),('B','--')]:
                ss=[r for r in summaries if r['precision']=='fp32' and r['kind']==kind and r['split']==split]
                ax.semilogy([r['snr_db'] for r in ss],[r[arm+'_median'] for r in ss],ls,color=color,label=split+' '+('原音量' if arm=='A' else '音量×0.5'))
        ax.set(title=names[kind],xlabel='名义整 split SNR（dB）',ylabel='末层单位方向扰动 RMS');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.savefig(out/'four_noise_absolute_response.png',dpi=155);fig.savefig(out/'four_noise_absolute_response.svg');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);main(p.parse_args().output)
