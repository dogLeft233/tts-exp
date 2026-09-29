"""Evaluator-selected local timing upper bound on fixed instance-pair queries; CPU."""
import argparse
import shutil
import numpy as np
from scripts.experiments.tts_independent_visual import ROOT,read,write,sha
from scripts.experiments.tts_pcm_residual import cluster

SOURCE=ROOT/'runs/tts_chinese_instance_primed_20260926'
OUT=ROOT/'runs/tts_instance_local_oracle_20260926'
MODES={'r0':(0,True),'r40':(1,True),'r80':(2,True),'r160':(4,True),'phone_all':(None,True),'free160':(4,False)}

def candidates(node,arm,length,k,radius,same_phone):
    j=node['j'][arm];lo,hi=node['span'][arm]
    js=np.arange(k+20,length+k-20)
    js=js[(js>=0)&(js<length)]
    if radius is not None:js=js[abs(js-j)<=radius]
    if same_phone:js=js[(js/25+.1075>=lo)&(js/25+.1075<hi)]
    assert j in js
    return js

def freeze():
    assets={};supports={}
    for label in ('primary','mfa_only'):
        for k in (3,2,4):
            f=SOURCE/'mechanism'/f'{label}_k{k}'/'support.json';ss=[s for s in read(f) if s['eligible']]
            assets[str(f)]=sha(f);supports[f'{label}_k{k}']=[s['id'] for s in ss]
            for s in ss:
                for path in [SOURCE/'audio_features'/(s['id']+'.npz'),*[SOURCE/'scores/3'/s['id']/(a+'.npz') for a in ('Q1','Q2')]]:assets[str(path)]=sha(path)
            c=f.with_name('cells.json');assets[str(c)]=sha(c)
    p={'source':str(SOURCE),'assets':assets,'support_ids':supports,'modes':MODES,'main':'primary,k3,raw,r80',
       'query':'unchanged original support per label/k; both visual indices j_arm-k and self distances fixed; no oracle-success filtering',
       'candidate':'real audio index 0<=j<original matrix arm length AND 20<=j-k<length-20; same occurrence iff center j/25+.1075 within original interval; r in frame offsets0/1/2/4; phone_all no radius; free160 no phone restriction',
       'distance':'Euclidean(v-a+1e-6), raw or separately L2 normalized; CPU cached embeddings only; original lag matrices cannot encode arbitrary cross-arm event coordinates',
       'selection':'minimum cross distance; tie smallest abs shift then ascending audio index',
       'endpoint':'cross-positive minus fixed self-positive: each direction and their mean. Query mean within utterance then speaker equal, 20000 PCG64 seed20260926,95/99CI',
       'limits':'optimistic oracle favoring cross pairing, not deployable alignment/sync improvement; positive component only, not q/nativeC causal decomposition',
       'code_sha256':sha(__file__)}
    assert not (OUT/'protocol.json').exists()
    write(OUT/'protocol.json',p)

def run():
    p=read(OUT/'protocol.json')
    for f,h in p['assets'].items():assert sha(f)==h
    for label in ('primary','mfa_only'):
        for k in (3,2,4):
            folder=SOURCE/'mechanism'/f'{label}_k{k}';ss=[s for s in read(folder/'support.json') if s['eligible']]
            records=[];cells=[]
            for s in ss:
                audio=np.load(SOURCE/'audio_features'/(s['id']+'.npz'))
                visual={a:np.load(SOURCE/'scores/3'/s['id']/(a+'.npz'))['visual'].astype(float) for a in ('Q1','Q2')}
                for geo in ('raw','unit'):
                    aa={a:audio[a].astype(float) for a in visual};vv={a:v.copy() for a,v in visual.items()}
                    if geo=='unit':
                        aa={a:x/np.linalg.norm(x,axis=1)[:,None] for a,x in aa.items()};vv={a:x/np.linalg.norm(x,axis=1)[:,None] for a,x in vv.items()}
                    values={mode:[] for mode in MODES}
                    for qi,q in enumerate(s['queries']):
                        node=s['nodes'][q['node']];selfd={a:float(np.linalg.norm(vv[a][node['j'][a]-k]-aa[a][node['j'][a]]+1e-6)) for a in aa}
                        for mode,(radius,same) in MODES.items():
                            dirs={};eff={}
                            for a,b in (('Q1','Q2'),('Q2','Q1')):
                                js=candidates(node,b,s['lengths'][b],k,radius,same)
                                ds=np.linalg.norm(vv[a][node['j'][a]-k]-aa[b][js]+1e-6,axis=1)
                                selected=int(np.lexsort((js,abs(js-node['j'][b]),ds))[0]);cross=float(ds[selected]);shift=int(js[selected]-node['j'][b]);name=a+b
                                dirs[name]={'self':selfd[a],'cross':cross,'candidates':js.tolist(),'chosen':int(js[selected]),'shift_frames':shift}
                                eff.update({name+'_gap':cross-selfd[a],name+'_self':selfd[a],name+'_cross':cross,name+'_win':float(cross<=selfd[a]),name+'_candidate_count':len(js),name+'_abs_shift_ms':abs(shift)*40,name+'_shift_ms':shift*40})
                            eff.update({'gap':.5*(eff['Q1Q2_gap']+eff['Q2Q1_gap']),'self':np.mean(list(selfd.values())),'cross':.5*(eff['Q1Q2_cross']+eff['Q2Q1_cross'])})
                            eff['win']=float(eff['gap']<=0)
                            values[mode].append(eff);records.append({'id':s['id'],'speaker':s['speaker'],'geometry':geo,'mode':mode,'query':qi,'event':node['event'],'phase':node['phase'],'directions':dirs,'effects':eff})
                    for mode,vals in values.items():cells.append({'id':s['id'],'speaker':s['speaker'],'geometry':geo,'mode':mode,'queries':len(vals),'effects':{key:float(np.mean([v[key] for v in vals])) for key in vals[0]}})
            results={}
            for geo in ('raw','unit'):
                results[geo]={}
                for mode in MODES:
                    subset=[c for c in cells if c['geometry']==geo and c['mode']==mode]
                    results[geo][mode]={key:cluster([c['effects'][key] for c in subset],[c['speaker'] for c in subset]) for key in subset[0]['effects']}
            dest=OUT/f'{label}_k{k}';write(dest/'queries.json',records);write(dest/'cells.json',cells);write(dest/'analysis.json',results)
            print(label,k,len(ss),flush=True)

def report():
    def fmt(z):return f"{z['speaker_mean']:+.3f} [{z['speaker_ci99'][0]:+.3f}, {z['speaker_ci99'][1]:+.3f}]"
    lines=['# 实例错配的局部时间纠错乐观上界','','固定query、visual、self距离；只用评价器挑交叉audio最佳时刻。不是可部署对齐，不是同步改善；只判断positive距离成分。以下speaker等权均值与99%CI；95%CI在JSON。','','## 主支持 primary20，k3','','| 候选 | raw Pcross−Pself | unit Pcross−Pself | raw达到self比例 | 平均候选/方向 | 平均绝对位移ms |','|---|---|---|---:|---:|---:|']
    z=read(OUT/'primary_k3/analysis.json')
    for mode in MODES:
        r=z['raw'][mode];lines.append(f"| {mode} | {fmt(r['gap'])} | {fmt(z['unit'][mode]['gap'])} | {r['win']['speaker_mean']:.3f} | {np.mean([r[a+'_candidate_count']['speaker_mean'] for a in ('Q1Q2','Q2Q1')]):.2f} | {np.mean([r[a+'_abs_shift_ms']['speaker_mean'] for a in ('Q1Q2','Q2Q1')]):.1f} |")
    main=z['raw']['r80']['gap']
    lines.extend(['',f"主±80ms结果 {fmt(main)}。"+('仍确认正残余，有限局部错时不足以解释全部正距离惩罚。' if main['speaker_ci99'][0]>0 else '没有确认正残余；当前MFA不能排除局部时序解释。CI跨0不等于证明零效应或等效，更不证明时序是全部原因。'),'',
                  '同实例基线没有同样的oracle优化，交叉配对得到刻意优待；最佳时间由同一个评价器选择。phone中心归属也不保证整个约200ms窗口只含该phone。'])
    lines.extend(['','## r80两个方向','','| 支持/k/geometry | VQ1→AQ2 | VQ2→AQ1 | 合并 |','|---|---|---|---|'])
    for label in ('primary','mfa_only'):
        for k in (3,2,4):
            z=read(OUT/f'{label}_k{k}/analysis.json')
            for geo in ('raw','unit'):
                r=z[geo]['r80'];lines.append(f'| {label}/{k}/{geo} | '+' | '.join(fmt(r[a]) for a in ('Q1Q2_gap','Q2Q1_gap','gap'))+' |')
    lines.extend(['','各模式两方向、候选数量、signed/absolute实际位移及达到self比例均有speaker CI，见analysis.json；全部候选与选择在queries.json。各k使用原支持，不筛oracle成功点。r0复算、全部距离独立标量核验、候选穷举及bootstrap复核见validation.json。','','有限±80ms同phone上界后仍正则有限局部错时不足以解释全部实例惩罚；若消除，只能说MFA不能排除时序解释。不作贡献比例、不归因嘴型真值、不中途扩大搜索。'])
    (OUT/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    z=read(OUT/'primary_k3/analysis.json')['raw'];names=list(MODES);means=np.array([z[m]['gap']['speaker_mean'] for m in names]);ci=np.array([z[m]['gap']['speaker_ci99'] for m in names]).T
    fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained');ax.errorbar(range(len(names)),means,yerr=np.vstack((means-ci[0],ci[1]-means)),fmt='o',capsize=4);ax.axhline(0,color='gray',lw=1);ax.set_xticks(range(len(names)),names);ax.set_ylabel('Cross minus fixed self distance, 99% CI');ax.set_title('Optimistic local timing oracle: primary 20, k=3');fig.savefig(OUT/'summary.svg');fig.savefig(OUT/'summary.png',dpi=180);plt.close(fig)
    (OUT/'code').mkdir(exist_ok=True)
    for f in (ROOT/'scripts/experiments').glob('*tts_instance_local_oracle*.py'):shutil.copyfile(f,OUT/'code'/f.name)
    write(OUT/'provenance.json',{str(f):sha(f) for f in OUT.rglob('*') if f.is_file() and f.name!='provenance.json'})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','run','report'));globals()[p.parse_args().stage]()
