"""Compare frozen results without rerunning TFG/scorers or inventing legacy p-values."""
from pathlib import Path
import csv
import hashlib
import itertools
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/cross_tfg/nine_scorer_comparison_20261031'
FOLDERS={'TalkLikeYou':'talklikeyou','FlashHead':'soulx_flashhead_pro','Echo':'echomimic_v3_flash'}
METRICS=['SyncNet','CoGenAV','PEAVS','SyncReward','AVS_u','MTDVocaLiST','VocaLiST','Synchformer','SparseSync']
PERCENT={'MTDVocaLiST','VocaLiST','Synchformer','SparseSync'}

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def factor(metric):return 100 if metric in PERCENT else 1

def holm(values):
    order=np.argsort(values,kind='stable');out=np.empty(len(values))
    out[order]=np.minimum(1,np.maximum.accumulate(np.array(values)[order]*np.arange(len(values),0,-1)))
    return out.tolist()

def contrast(d):
    d=np.asarray(d);rng=np.random.default_rng(20261030)
    ci=np.quantile(d[rng.integers(0,len(d),(10000,len(d)))].mean(1),[.025,.975])
    k=0
    for _ in range(50):
        signs=2*rng.integers(0,2,(2000,len(d)))-1
        k+=np.count_nonzero(abs((signs*d).mean(1))>=abs(d.mean())-1e-15)
    return {'n':len(d),'delta':float(d.mean()),'ci95':ci.tolist(),'p':(int(k)+1)/100001}

# Values below are explicitly transcribed from canonical historical reports.
# Legacy ZH has no unified nine-endpoint Holm family in those reports.
ZH_ALL={
 'SyncNet':(.745,20,[.378809,1.095653]),
 'CoGenAV':(-.000836,20,[-.016507,.014700]),
 'PEAVS':(-.408085,20,None), # published default64 audit, NOT historical batch1 -.403537
 'SyncReward':(-.019522,20,[-.042717,.006511]), # FP32 native, NOT BF16 controls
 'AVS_u':(-.0075285793,20,[-.0162387060,.0016378580]),
 'MTDVocaLiST':(-.00565642,20,[-.01320332,.00326934]), # official K5, not superseded mean logit
 'VocaLiST':(-.00937,20,[-.01915,-.00021]), # report precision is three decimal pp
 'Synchformer':(.012937,6,[.001997,.032509]),
 'SparseSync':(-.0000931124,6,[-.0002314945,.0000018742]),
}
ZH_STRICT={
 'SyncNet':(.740,12,[.399160,1.068167]),'CoGenAV':(.001689,12,[-.015830,.019705]),
 'PEAVS':(-.474832,12,None),'SyncReward':(-.031487,12,[-.046649,-.016069]),
 'AVS_u':(-.0066387291,12,None),'MTDVocaLiST':(-.00716549,12,[-.01419698,-.00048511]),
 'VocaLiST':(-.00777,12,[-.01977,.00368]),
 'Synchformer':ZH_ALL['Synchformer'],'SparseSync':ZH_ALL['SparseSync'],
}

def cell(row):
    m=row['metric'];spec='.3f' if m=='SyncNet' or (row['model']=='Wav2Lip-ZH' and m=='VocaLiST') else '.6f'
    delta=format(row['display_delta'],'+'+spec)
    q='NA' if row['holm9_p'] is None else f"{row['holm9_p']:.5f}"
    text=f"{delta}; n={row['n_sources']}; q={q}"
    return '**'+text+'**' if row['holm9_p'] is not None and row['holm9_p']<.05 else text

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    sources=[];rows=[];pairs={}
    for model,folder in FOLDERS.items():
        run=ROOT/'results'/folder/'chinese20_rms24_20261030_attempt01'
        summary=read(run/'summary.json');pairs[model]=read(run/'paired_scores.json')
        sources += [dict(path=str(p.relative_to(ROOT)),sha256=sha(p)) for p in [run/'summary.json',run/'paired_scores.json']]
        for subset in ['same_reference12','all20']:
            for metric,v in summary['subsets'][subset].items():
                rows.append(dict(model=model,cohort='Chinese20',subset=subset,metric=metric,n_sources=v['n_sources'],
                    delta=v.get('delta'),display_delta=v.get('delta',0)*factor(metric),
                    ci95=v.get('ci95'),holm9_p=v['holm9_p'],significance='SIGNIFICANT' if v['holm9_p']<.05 else 'NOT_CONFIRMED',
                    unit='percentage_points' if metric in PERCENT else 'native',provenance='frozen_summary'))
    for subset,values in [('same_reference12',ZH_STRICT),('all20',ZH_ALL)]:
        for metric,(delta,n,ci) in values.items():
            rows.append(dict(model='Wav2Lip-ZH',cohort='Chinese20-legacy',subset=subset,metric=metric,n_sources=n,delta=delta,
                display_delta=delta*factor(metric),ci95=ci,holm9_p=None,significance='NOT_COMPUTED_NOT_NONSIGNIFICANT',
                unit='percentage_points' if metric in PERCENT else 'native',provenance='canonical_reports_rounded_legacy',
                caveat='Classifier rows include legacy one-frame video/audio mismatch at S0031; no rescore or revised5-source statistic available' if metric in ['Synchformer','SparseSync'] else None))
    # English EARS40 is an external-cohort Wav2Lip reference, never a fourth ZH arm.
    report=ROOT/'basic-memory/docs/experiments/75-ears40-english-wav2lip-nine-scorers.md'
    source_text=report.read_text();table=source_text[source_text.index('## 九个冻结主端点'):source_text.index('## 真实音频时移控制')]
    for line in table.splitlines():
        parts=[p.strip() for p in line.split('|')]
        if len(parts)<11 or parts[1] not in METRICS:continue
        metric=parts[1];display_delta=float(parts[4]);q=float(parts[8])
        ci=[float(x.strip())/factor(metric) for x in parts[5].strip('[]').split(',')]
        rows.append(dict(model='Wav2Lip-EN',cohort='EARS40-English',subset='all40',metric=metric,n_sources=40,
            delta=display_delta/factor(metric),display_delta=display_delta,ci95=ci,holm9_p=q,
            significance='SIGNIFICANT' if q<.05 else 'NOT_CONFIRMED',unit='percentage_points' if metric in PERCENT else 'native',provenance='report75_frozen100k_table'))
    assert sum(r['model']=='Wav2Lip-EN' for r in rows)==9
    # New retrospective family: all three same-input new-model contrasts,27 endpoints per stratum.
    contrasts=[]
    for subset in ['same_reference12','all20']:
        family=[]
        for a,b in itertools.combinations(FOLDERS,2):
            for metric in METRICS:
                aa={r['source']:r for r in pairs[a][metric]};bb={r['source']:r for r in pairs[b][metric]}
                ids=sorted(aa.keys()&bb.keys());ids=[s for s in ids if subset=='all20' or aa[s]['strict_reference']]
                value=contrast([aa[s]['delta']-bb[s]['delta'] for s in ids])
                family.append(dict(model_a=a,model_b=b,metric=metric,subset=subset,sources=ids,**value))
        for r,q in zip(family,holm([v['p'] for v in family])):r['holm27_p']=q
        contrasts+=family
    docpaths=[
      'basic-memory/docs/experiments/68-cloud-mtd-synchformer-native-preference.md',
      'basic-memory/Experiments/云端Qwen Audio3.1Plus的20来源原生四指标分歧 2026-10-27.md',
      'basic-memory/Experiments/PEAVS与SyncReward官方推理审计及批量掩码问题 2026-10-27.md',
      'basic-memory/Tasks/AVS_u中文域同步控制与自然TTS原生比较 2026-10-27.md',
      *[str(p.relative_to(ROOT)) for p in (ROOT/'basic-memory/docs/experiments').glob('7[01256789]-*.md')],
      'basic-memory/docs/adr/052-integer-frame-native-window-preselection.md']
    sources += [dict(path=p,sha256=sha(ROOT/p)) for p in docpaths if (ROOT/p).exists()]
    write(OUT/'comparison.json',dict(status='ANALYZED',script_sha256=sha(Path(__file__)),rows=rows,pairwise_contrasts=contrasts,source_files=sources,
        legacy_q_missing='No unified Holm9 published in legacy ZH reports; source-level originals unavailable in current local workspace; read-only old-host SSH closed. Missing q is not nonsignificance.',
        new_contrast_family='Retrospective CPU-only27-cell Holm family per stratum; no change to original frozen Holm9/Holm18 results'))
    columns=['model','cohort','subset','metric','n_sources','delta','display_delta','unit','holm9_p','significance','provenance']
    with (OUT/'comparison.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    lines=[]
    for subset in ['same_reference12','all20']:
        lines += [f'## {subset}','','每格为 Δ；独立来源n；Holm9 q。K5与P0四行单位为百分点。NA＝未计算，不能判不显著。','',
          '|评分器|TalkLikeYou|FlashHead Pro|EchoMimicV3-Flash|Wav2Lip中文旧批次|','|---|---|---|---|---|']
        for metric in METRICS:
            group=[next(r for r in rows if r['subset']==subset and r['metric']==metric and r['model']==model) for model in [*FOLDERS,'Wav2Lip-ZH']]
            lines.append('|'+metric+(' (pp)' if metric in PERCENT else '')+'|'+'|'.join(cell(r) for r in group)+'|')
        lines.append('')
    lines += ['## Wav2Lip英语EARS40：跨语料参考','','|评分器|来源n|Δ|Holm9 q|','|---|---:|---:|---:|']
    for r in rows:
        if r['model']=='Wav2Lip-EN':lines.append(f"|{r['metric']}{' (pp)' if r['metric'] in PERCENT else ''}|40|{format(r['display_delta'],'+.3f' if r['metric']=='SyncNet' else '+.6f')}|{r['holm9_p']:.6f}|")
    lines+=['','## 同输入三模型效应差：本次事后Holm27','','每格差为(A的T−N)−(B的T−N)。不包含Wav2Lip，不替代旧报告Holm18。','',
      '|分层|A−B|指标|共同n|效应差|边际95%CI|Holm27 q|','|---|---|---|---:|---:|---|---:|']
    for r in contrasts:
        if r['holm27_p']>=.05:continue
        scale=factor(r['metric']);d=r['delta']*scale;lo,hi=np.array(r['ci95'])*scale
        spec='+.3f' if r['metric']=='SyncNet' else '+.6f'
        lines.append(f"|{r['subset']}|{r['model_a']}−{r['model_b']}|{r['metric']}|{r['n']}|{format(d,spec)}|[{format(lo,spec)},{format(hi,spec)}]|{r['holm27_p']:.6f}|")
    (OUT/'tables.md').write_text('\n'.join(lines)+'\n')
    plot(rows)
    print((OUT/'tables.md').read_text())

def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    models=[*FOLDERS,'Wav2Lip-ZH'];colors=['#156a9b','#d28d23','#577a56','#7a647e']
    fig,axes=plt.subplots(3,3,figsize=(14,11),layout='constrained')
    for ax,metric in zip(axes.flat,METRICS):
        ax.axvline(0,color='gray',ls='--',lw=.8)
        for i,(model,color) in enumerate(zip(models,colors)):
            r=next(r for r in rows if r['model']==model and r['subset']=='all20' and r['metric']==metric)
            d=r['display_delta'];y=3-i
            if r['ci95'] is not None:
                lo,hi=np.array(r['ci95'])*factor(metric);ax.plot([lo,hi],[y,y],color=color,lw=2)
            ax.scatter(d,y,s=50,facecolors='none' if model=='Wav2Lip-ZH' else color,edgecolors=color,zorder=3)
            if r['holm9_p'] is not None and r['holm9_p']<.05:ax.annotate('*',(d,y),xytext=(4,4),textcoords='offset points',fontsize=16)
        ax.set_yticks(range(4),[f'{model} (n={next(r["n_sources"] for r in rows if r["model"]==model and r["subset"]=="all20" and r["metric"]==metric)})' for model in reversed(models)],fontsize=8)
        ax.set_ylim(-.5,3.65);ax.set_title(metric,loc='left',weight='bold');ax.set_xlabel('TTS - natural'+(' (percentage points)' if metric in PERCENT else ''))
        ax.spines[['right','top','left']].set_visible(False);ax.grid(axis='x',alpha=.15)
    fig.suptitle('Chinese cohort: source-paired effects by TFG model\nMarginal95% CI; * frozen Holm9 q<.05 | hollow: legacy Wav2Lip, q unavailable / protocol differs',fontsize=13)
    fig.savefig(OUT/'paired_effects.png',dpi=160);fig.savefig(OUT/'paired_effects.pdf');plt.close(fig)

if __name__=='__main__':main()
