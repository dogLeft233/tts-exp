"""Read-back verification, dose contrasts and publication plots; no inference."""
from pathlib import Path
import sys,json,argparse,hashlib,importlib.util,collections
sys.path.insert(0,'F:/Datasets/tts-exp/encoder_survey_20261018/plotdeps')
import numpy as np

def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4<<20),b''):h.update(b)
    return h.hexdigest()
def unit(x):
    x=x.astype(np.float64);return x/np.sqrt(np.sum(x*x,1))[:,None]

def independent_metrics(x,y):
    x=x.astype(np.float64);y=y.astype(np.float64);d=y-x;u=unit(y)-unit(x)
    return np.column_stack((np.einsum('ij,ij->i',x,x),np.einsum('ij,ij->i',y,y),np.einsum('ij,ij->i',d,d),np.einsum('ij,ij->i',u,u),np.abs(d).max(1),(d!=0).sum(1)))

def main(out):
    design=read(out/'protocol.json');complete=read(out/'completion.json')
    assert complete['status']=='COMPLETED' and complete['forwards']==538 and sha(out/'protocol.json')==complete['protocol_sha256']
    assert sha(Path(__file__).with_name('run.py'))==design['script_sha256']
    rows=read(out/'rows.json');calls=read(out/'forward_calls.json');controls=read(out/'controls.json');waves=read(out/'waveform_audit.json')
    follow=out/'transition_fp64';follow_complete=read(follow/'completion.json')
    assert follow_complete['status']=='COMPLETED' and follow_complete['forwards']==28
    assert sha(follow/'protocol.json')==follow_complete['protocol_sha256']
    assert sha(Path(__file__).with_name('transition_fp64.py'))==read(follow/'protocol.json')['script_sha256']
    rows+=read(follow/'rows.json');waves+=read(follow/'waveform_audit.json')
    def folder(mode,split):return follow/split if mode=='fp64_transition' else out/mode/split
    assert len(calls)==538 and len({c['path'] for c in calls})==538
    assert len([c for c in controls if c['kind']=='clean_repeat'])==2
    assert len([c for c in controls if c['kind']=='prior_30dB_bridge'])==24
    assert len([c for c in controls if c['kind']=='epsilon_compensation'])==48
    assert all(c.get('all_stages_exact',c.get('exact',all(c.get('stages_exact',{}).values()))) for c in controls)
    bykey={(r['precision'],r['split'],r['kind'],r['seed'],r['snr_db'],r['arm'],r['stage']):r for r in rows}
    assert len(bykey)==len(rows)==3556
    floors={(r['split'],r['arm'],r['stage']):r['unit_delta'] for r in rows if r['precision']=='clean32vs64'}
    ratios=[]
    for r in rows:
        if r['precision']=='clean32vs64' or r['arm']!='A':continue
        k=(r['precision'],r['split'],r['kind'],r['seed'],r['snr_db'],'B',r['stage']);b=bykey[k]
        ra=b['unit_delta']/r['unit_delta'] if r['unit_delta'] else None
        rel=b['relative_l2']/r['relative_l2'] if r['relative_l2'] else None
        resolved=r['precision'].startswith('fp64') or all(bykey[(r['precision'],r['split'],r['kind'],r['seed'],r['snr_db'],a,r['stage'])]['unit_delta']>10*floors[r['split'],a,r['stage']] for a in ['A','B'])
        ratios.append(dict(precision=r['precision'],split=r['split'],kind=r['kind'],seed=r['seed'],snr_db=r['snr_db'],stage=r['stage'],A=r['unit_delta'],B=b['unit_delta'],ratio=ra,relative_l2_ratio=rel,resolved=bool(resolved),exact_frames_A=r['exact_frame_fraction'],exact_frames_B=b['exact_frame_fraction']))
    write(out/'ratios.json',ratios)
    # Full read-back against saved input arrays and every stage's independently calculated raw sample moments.
    native={s:np.load(out/(s+'_native.npy')) for s in ['train','test']};checks=0;maxerr=0.;sample_pairs=0;last_pairs=0
    for c in calls:
        parts=c['path'].split('/');precision,split=parts[:2];dtype=np.float32 if precision=='fp32' else np.float64
        s=native[split].astype(dtype);label=parts[-1]
        if len(parts)==3:y=s*(.5 if label=='anchorB' else 1)
        else:
            kind,seed=parts[2].rsplit('_',1);q=int(parts[3][3:]);n=np.load(out/f'noise/{split}_{kind}_{seed}.npy').astype(dtype)*dtype(10**((30-q)/20))
            y=(s+n)*(.5 if label in ['B','B_compensated'] else 1)
        assert hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()==c['input_sha256']
        p=out/c['path'];last=np.load(p/'last.npy');assert hashlib.sha256(last.tobytes()).hexdigest()==c['last_sha256'];checks+=1
        if label in ['A','B']:
            anchor_path=out/precision/split/('anchor'+label);sample=np.load(p/'samples.npz');anchor_samples=np.load(anchor_path/'samples.npz');fm=np.load(p/'frames.npz')
            for stage in design['stages']:
                idx=sample[stage+'__indices'];assert np.array_equal(idx,anchor_samples[stage+'__indices'])
                rebuilt=independent_metrics(anchor_samples[stage],sample[stage]);stored=fm[stage][idx]
                np.testing.assert_allclose(rebuilt,stored,rtol=3e-14,atol=1e-28);maxerr=max(maxerr,float(np.max(np.abs(rebuilt-stored))));sample_pairs+=len(idx)
                key=(precision,split,kind,int(seed),q,label,stage);r=bykey[key]
                assert r['unit_delta']==float(np.sqrt(fm[stage][:,3].mean()))
                assert r['relative_l2']==float(np.sqrt(fm[stage][:,2].sum()/fm[stage][:,0].sum()))
            full=independent_metrics(np.load(anchor_path/'last.npy'),last)
            np.testing.assert_allclose(full,fm['last'],rtol=3e-14,atol=1e-28);last_pairs+=len(last)
        del last
    follow_checks=0
    for c in read(follow/'forward_calls.json'):
        parts=c['path'].split('/');split=parts[0];label=parts[-1];s=native[split].astype(np.float64)
        if len(parts)==2:y=s*(.5 if label=='anchorB' else 1)
        else:
            kind,seed=parts[1].rsplit('_',1);q=int(parts[2][3:]);n=np.load(out/f'noise/{split}_{kind}_{seed}.npy').astype(np.float64)*10**((30-q)/20)
            y=(s+n)*(.5 if label=='B' else 1)
        assert hashlib.sha256(y.tobytes()).hexdigest()==c['input_sha256']
        p=follow/c['path'];last=np.load(p/'last.npy');assert hashlib.sha256(last.tobytes()).hexdigest()==c['last_sha256'];follow_checks+=1
        sample=np.load(p/'samples.npz')
        if len(parts)==2:
            old=out/'fp64'/split/label;np.testing.assert_array_equal(last,np.load(old/'last.npy'));ss=np.load(old/'samples.npz')
            assert all(np.array_equal(sample[k],ss[k]) for k in design['stages'])
        else:
            anchor=out/'fp64'/split/('anchor'+label);anchor_sample=np.load(anchor/'samples.npz');fm=np.load(p/'frames.npz')
            for stage in design['stages']:
                idx=sample[stage+'__indices'];rebuilt=independent_metrics(anchor_sample[stage],sample[stage]);stored=fm[stage][idx]
                np.testing.assert_allclose(rebuilt,stored,rtol=3e-14,atol=1e-28);maxerr=max(maxerr,float(np.max(np.abs(rebuilt-stored))));sample_pairs+=len(idx)
                r=bykey['fp64_transition',split,kind,int(seed),q,label,stage];assert r['unit_delta']==float(np.sqrt(fm[stage][:,3].mean()))
            np.testing.assert_allclose(independent_metrics(np.load(anchor/'last.npy'),last),fm['last'],rtol=3e-14,atol=1e-28);last_pairs+=len(last)
    assert follow_checks==28
    for w in waves:
        s=native[w['split']].astype(np.float32 if w['precision']=='fp32' else np.float64)
        n=np.load(out/f"noise/{w['split']}_{w['kind']}_{w['seed']}.npy").astype(s.dtype)*s.dtype.type(10**((30-w['snr_db'])/20))
        a=s+n;b=a*.5;d=a-s;e=b-s*.5
        assert np.array_equal(d,2*e) and float(np.mean(d!=0))==w['input_changed_fraction']==w['half_changed_fraction']
    prior=Path(design['prior']);manifest={r['path']:r for r in read(prior/'delivery_manifest.json')['members']};bridges=0
    for split in native:
        for arm in ['A','B']:
            old=prior/split/'chunk00'/('anchor1' if arm=='A' else 'anchorhalf')/'last.npy'
            assert sha(old)==manifest[old.relative_to(prior).as_posix()]['sha256']
            np.testing.assert_array_equal(np.load(old),np.load(out/'fp32'/split/('anchor'+arm)/'last.npy'));bridges+=1
        for kind in ['white','pink']:
            for seed in design['seeds']:
                for arm in ['A','B']:
                    old=prior/split/'chunk00'/f'{kind}_{seed}'/arm/'last.npy'
                    assert sha(old)==manifest[old.relative_to(prior).as_posix()]['sha256']
                    np.testing.assert_array_equal(np.load(old),np.load(out/'fp32'/split/f'{kind}_{seed}'/'snr030'/arm/'last.npy'));bridges+=1
    # Verify compensation output directly, not just producer booleans.
    for c in controls:
        if c['kind']!='epsilon_compensation':continue
        p=out/'fp32'/c['split']/f"{c['noise']}_{c['seed']}"/f"snr{c['snr_db']:03d}"
        np.testing.assert_array_equal(np.load(p/'A/last.npy'),np.load(p/'B_compensated/last.npy'))
        a=np.load(p/'A/samples.npz');b=np.load(p/'B_compensated/samples.npz')
        assert all(np.array_equal(a[k],b[k]) for k in design['stages'])
    precision=[]
    for split in native:
        for kind in ['white','pink']:
            seed=design['seeds'][0]
            for q in [80,90,100]+design['fp64_grid_db']:
                for arm in ['A','B']:
                    ds=[]
                    for mode in ['fp32','fp64_transition' if q<=100 else 'fp64']:
                        p=folder(mode,split);ref=np.load(p/('anchor'+arm)/'last.npy');val=np.load(p/f'{kind}_{seed}'/f'snr{q:03d}'/arm/'last.npy')
                        ds.append(unit(val)-unit(ref))
                    ratio=float(np.linalg.norm(ds[0]-ds[1])/np.linalg.norm(ds[1]))
                    precision.append(dict(split=split,kind=kind,snr_db=q,arm=arm,fp32_response_error_relative_to_fp64=ratio))
    write(out/'precision_diagnostics.json',precision)
    summary=[]
    for mode,grid in [('fp32',design['snr_grid_db']),('fp64',design['fp64_grid_db']),('fp64_transition',[80,90,100])]:
        for q in grid:
            rr=[r for r in ratios if r['precision']==mode and r['snr_db']==q and r['stage']=='last'];values=[r['ratio'] for r in rr if r['ratio'] is not None]
            ww=[w for w in waves if w['precision']==mode and w['snr_db']==q]
            summary.append(dict(precision=mode,snr_db=q,cases=len(rr),resolved=sum(r['resolved'] for r in rr),ratio_min=min(values) if values else None,ratio_median=float(np.median(values)) if values else None,ratio_max=max(values) if values else None,
                A_median=float(np.median([r['A'] for r in rr])),B_median=float(np.median([r['B'] for r in rr])),input_changed_fraction_min=min(w['input_changed_fraction'] for w in ww),input_changed_fraction_max=max(w['input_changed_fraction'] for w in ww)))
    write(out/'summary.json',summary)
    stage_summary=[]
    for mode,grid in [('fp32',design['snr_grid_db']),('fp64',design['fp64_grid_db']),('fp64_transition',[80,90,100])]:
        for stage in design['stages']:
            for q in grid:
                rr=[r for r in ratios if r['precision']==mode and r['stage']==stage and r['snr_db']==q]
                values=[r['ratio'] for r in rr if r['ratio'] is not None]
                stage_summary.append(dict(precision=mode,stage=stage,snr_db=q,cases=len(rr),resolved=sum(r['resolved'] for r in rr),ratio_min=min(values) if values else None,ratio_median=float(np.median(values)) if values else None,ratio_max=max(values) if values else None))
    write(out/'stage_summary.json',stage_summary)
    zero_frames=[]
    for r in ratios:
        p=folder(r['precision'],r['split'])/f"{r['kind']}_{r['seed']}"/f"snr{r['snr_db']:03d}"
        if r['stage']!='last':continue
        a=np.load(p/'A/frames.npz');b=np.load(p/'B/frames.npz')
        for stage in design['stages']:
            aa=a[stage][:,5]==0;bb=b[stage][:,5]==0
            zero_frames.append(dict(precision=r['precision'],split=r['split'],kind=r['kind'],seed=r['seed'],snr_db=r['snr_db'],stage=stage,frames=len(aa),both_exact=int(np.sum(aa&bb)),only_B_exact=int(np.sum(~aa&bb)),only_A_exact=int(np.sum(aa&~bb))))
    write(out/'exact_frame_pairs.json',zero_frames)
    candidates=[]
    for kind in ['white','pink']:
        g=design['snr_grid_db']
        for i in range(len(g)-2):
            levels=g[i:i+3];rr=[r for r in ratios if r['precision']=='fp32' and r['stage']=='last' and r['kind']==kind and r['snr_db'] in levels]
            if len(rr)==18 and all(r['resolved'] and r['ratio'] is not None and r['ratio']<=.8 for r in rr):candidates.append(dict(kind=kind,snr_db=levels))
    write(out/'independent_qa.json',dict(passed=True,input_hashes=checks+follow_checks,sample_stage_frame_pairs=sample_pairs,full_last_frame_pairs=last_pairs,prior_bridges=bridges,metric_moment_max_absolute_error=maxerr,waveform_pairs=len(waves),epsilon_compensated_pairs=48,candidates=candidates,forward_count=566,main_forwards=538,exploratory_forwards=28,training_updates=0))
    plot(out,ratios,summary,waves,floors)
    print(json.dumps(dict(passed=True,summary=summary,candidates=candidates),ensure_ascii=False),flush=True)

def plot(out,ratios,summary,waves,floors):
    import os
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.ticker import ScalarFormatter
    font=FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':font.get_name(),'axes.unicode_minus':False,'font.size':10})
    fig,axs=plt.subplots(2,2,figsize=(14,9),layout='constrained');fig.suptitle('May 实际音频：噪声强度扫描与数值下限（冻结 wav2vec2，无训练）',fontsize=17)
    colors={'white':'#176cb2','pink':'#c05b2a'}
    for kind,name in [('white','白噪声'),('pink','粉红噪声')]:
        x=list(range(0,101,10))
        rr=[r for r in ratios if r['precision']=='fp32' and r['stage']=='last' and r['kind']==kind]
        for arm,ls in [('A','-'),('B','--')]:
            y=[np.median([r[arm] for r in rr if r['snr_db']==q]) for q in x]
            axs[0,0].semilogy(x,y,ls,color=colors[kind],label=f'{name} · '+('原音量' if arm=='A' else '音量×0.5'))
        for split,ls in [('train','-'),('test','--')]:
            vals=[[r['ratio'] for r in rr if r['snr_db']==q and r['split']==split] for q in x]
            axs[0,1].plot(x,[np.median(v) for v in vals],ls,color=colors[kind],label=name+' · '+split)
            axs[0,1].fill_between(x,[min(v) for v in vals],[max(v) for v in vals],color=colors[kind],alpha=.12)
    axs[0,0].set(title='噪声变弱时，两种音量的编码扰动一起减小',ylabel='末层单位方向扰动 RMS',xlabel='名义整段 SNR（dB，越大噪声越弱）');axs[0,0].legend(fontsize=9)
    axs[0,1].axhline(1,color='#444',lw=.8);axs[0,1].ticklabel_format(axis='y',style='plain',useOffset=False)
    axs[0,1].set(title='同一噪声的扰动比：不同片段并非一致下降',ylabel='低音量扰动 / 原音量扰动',xlabel='名义整段 SNR（dB）');axs[0,1].legend(fontsize=8);axs[0,1].text(.03,.05,'阴影：三个种子范围',transform=axs[0,1].transAxes,fontsize=9)
    for mode,color,label in [('fp32','#176cb2','FP32（配对 4 条件）'),('fp64','#9d3677','FP64（配对 4 条件诊断）')]:
        ss=sorted([s for s in summary if s['precision']==mode or (mode=='fp64' and s['precision']=='fp64_transition')],key=lambda s:s['snr_db']);x=[s['snr_db'] for s in ss]
        if mode=='fp32':
            medians=[float(np.median([r['A'] for r in ratios if r['precision']=='fp32' and r['stage']=='last' and r['seed']==20261020 and r['snr_db']==q])) for q in x]
            y=[v if v else np.nan for v in medians]
        else:y=[s['A_median'] if s['A_median'] else np.nan for s in ss]
        axs[1,0].semilogy(x,y,'o-' if mode=='fp32' else 's--',color=color,label=label)
    discrepancy=np.median([v for (s,a,k),v in floors.items() if k=='last' and a=='A'])
    axs[1,0].axhline(discrepancy,color='#777',ls=':',label='无添加噪声 FP32/64 方向差（诊断参照）')
    axs[1,0].set(title='极弱噪声：FP32 出现数值平台，FP64 继续下降',xlabel='名义整段 SNR（dB）',ylabel='原音量末层方向扰动 RMS');axs[1,0].legend(fontsize=8)
    axs[1,0].annotate('260 dB：FP32 双方响应为 0',xy=(260,1.3e-6),xytext=(150,4e-5),fontsize=8,arrowprops={'arrowstyle':'->','color':'#555'})
    for mode,color,marker in [('fp32','#176cb2','o'),('fp64','#9d3677','s')]:
        ww=[w for w in waves if w['precision']==mode or (mode=='fp64' and w['precision']=='fp64_transition')]
        xx=sorted(set(w['snr_db'] for w in ww));vv=[[w['input_changed_fraction'] for w in ww if w['snr_db']==q] for q in xx]
        axs[1,1].plot(xx,np.array([np.median(v) for v in vv])*100,marker=marker,color=color,label=mode.upper())
    axs[1,1].set(title='输入端：添加后能改变浮点采样值的比例',xlabel='名义整段 SNR（dB）',ylabel='被已知噪声改变的采样值（%）',ylim=(-3,103))
    axs[1,1].text(.03,.06,'原音量与音量×0.5的变化掩码完全一致',transform=axs[1,1].transAxes,fontsize=9);axs[1,1].legend()
    for ax in axs.flat:ax.grid(alpha=.2)
    fig.savefig(out/'noise_strength_scan.png',dpi=160);fig.savefig(out/'noise_strength_scan.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,4.5),layout='constrained')
    names=['first_gn','cnn7','projection','transformer4','transformer8','transformer9','last'];labels=['首层 GN','CNN7','投影','T4','T8','T9','末层']
    for split,color in [('train','#176cb2'),('test','#c05b2a')]:
        for mode,ls,marker in [('fp32','-','o'),('fp64_transition','--','x')]:
            y=[next(r['ratio'] for r in ratios if r['precision']==mode and r['split']==split and r['kind']=='white' and r['seed']==20261020 and r['snr_db']==90 and r['stage']==k) for k in names]
            ax.plot(range(7),y,ls,marker=marker,color=color,label=split+' · '+('FP32' if mode=='fp32' else 'FP64'))
    ax.axhline(1,color='#777',lw=.8);ax.set_xticks(range(7),labels);ax.set(ylabel='音量×0.5 / 原音量的噪声方向扰动',title='90dB 白噪声首种子：局部弱化在后层展开，输入噪声仍存在')
    ax.grid(alpha=.2);ax.legend(ncol=2);fig.savefig(out/'noise_layer_transition.png',dpi=160);fig.savefig(out/'noise_layer_transition.svg');plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
