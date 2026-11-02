"""Independent source-level paired analysis; reads raw scorer artifacts only."""
from pathlib import Path
import argparse, csv, hashlib, json
import numpy as np

METRICS=['SyncNet','CoGenAV','PEAVS','SyncReward','AVS_u','MTDVocaLiST','VocaLiST','Synchformer','SparseSync']
FOLDERS=dict(zip(METRICS,['syncnet','cogenav','peavs','syncreward','avs_u','mtd','vocalist','synchformer','sparsesync']))

def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,d):Path(p).write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')

def statistics(n,t,seed=20261030):
    n=np.asarray(n,dtype=float);t=np.asarray(t,dtype=float)
    assert n.shape==t.shape and n.ndim==1 and len(n) and np.isfinite(n).all() and np.isfinite(t).all()
    d=t-n;rng=np.random.default_rng(seed)
    ci=np.quantile(d[rng.integers(0,len(d),size=(10000,len(d)))].mean(1),[.025,.975])
    exceed=0
    for _ in range(50):
        signs=rng.integers(0,2,size=(2000,len(d)))*2-1
        exceed+=int(np.sum(abs((signs*d).mean(1))>=abs(d.mean())-1e-15))
    sd=float(d.std(ddof=1)) if len(d)>1 else 0
    return dict(n_sources=len(d),N_mean=float(n.mean()),T_mean=float(t.mean()),delta=float(d.mean()),
        ci95=ci.tolist(),paired_dz=float(d.mean()/sd) if sd>0 else None,
        T_higher=int(np.sum(d>0)),N_higher=int(np.sum(d<0)),ties=int(np.sum(d==0)),signflip_p=(exceed+1)/100001)

def holm(ps):
    a=np.asarray(ps,dtype=float);order=np.argsort(a,kind='stable');out=np.empty_like(a)
    out[order]=np.minimum(1,np.maximum.accumulate(a[order]*np.arange(len(a),0,-1)))
    return out.tolist()

def load_values(root,metric):
    folder=FOLDERS[metric]
    if metric in ['SyncNet','CoGenAV']:
        return {r['id']:float(r['C' if metric=='SyncNet' else 'eq1']) for r in read(root/folder/'scores.json') if r.get('status','PASS')=='PASS'}
    if metric=='PEAVS':
        return {Path(r['filename']).stem:float(r['score']) for r in read(root/folder/'scores.json')['scores']}
    if metric=='SyncReward':
        return {Path(r['video_path']).stem:float(r['reward']) for r in map(json.loads,(root/folder/'scores.jsonl').read_text().splitlines())}
    result={}
    for row in read(root/folder/'batch_scores.json'):
        if row['status']!='PASS':continue
        # Resolve from the delivery root rather than trusting original machine paths.
        r=read(root/folder/row['id']/'deployment_validation.json')['results']['0']
        result[row['id']]=float(r['frame_mean'] if metric=='AVS_u' else r['author_metrics']['K5'] if metric in ['MTDVocaLiST','VocaLiST'] else r['p_zero'])
    return result

def make_pairs(rows,values):
    pairs=[];missing=[]
    for source in sorted({r['source'] for r in rows}):
        group=[r for r in rows if r['source']==source];natural=[r for r in group if r['arm']=='N'];tts=[r for r in group if r['arm']!='N']
        assert len(natural)==1 and len(tts)==3
        absent=[r['id'] for r in group if r['id'] not in values]
        if absent:missing.append(dict(source=source,unmeasurable_ids=absent));continue
        n=values[natural[0]['id']];draws=[values[r['id']] for r in tts];t=float(np.mean(draws))
        pairs.append(dict(source=source,speaker=natural[0]['speaker'],strict_reference=natural[0]['strict_reference'],N=n,T=t,T_draws=draws,delta=t-n))
    return pairs,missing

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args();root=a.run
    assert read(root/'status.json')['status']=='SCORING_COMPLETE'
    inputs=read(root/'scoring_inputs.json');rows=inputs['rows'];aliases=inputs['aliases']
    summary=dict(status='COMPLETE_EXPLORATORY',planned_sources=20,planned_videos=80,
        normalization=read(root/'protocol.json')['audio'],unit='source; mean of three existing TTS draws paired against one natural',
        audio_duplicates=read(root/'protocol.json')['duplicate_audio_groups'],
        limitations=['Chinese cloning references differ: primary12 same utterance, secondary8 extended references.',
        'Some TTS draws are identical, not independent synthesis replicates.',
        'PC-Talk native mel windows omit approximately0.2s of final visual support; full audio unchanged. Official PC-Talk mel/encoder/motion processing retained; no time shifts, external padding or posthoc offset correction.',
        'SyncNet scores reused byte-for-byte from completed parent checkpoint; official internal MP3/69ms first-pair diagnostic retained, not modified.',
        'Native face support below unchanged95percent is UNMEASURABLE for CoGenAV/AVS_u; complete source requires all four media.',
        'Native short inputs are unmeasurable for the two fixed5s classifiers; complete pairs vary by metric.',
        'SyncReward uses unchanged official window_segments including internal short-clip tail repetition/audio padding; no input media padding. Native4.75s sensitivity is separate.',
        'RMS -24dBFS is a new gain condition; do not splice historical lower-gain N scores.',
        'Scores are evaluator evidence, not perceptual or visual ground truth; no combined total or vote.',
        'Sign-flip sensitivity assumes symmetry/exchangeability, not randomized TTS allocation. CI are marginal; Holm9 p is separate.'],
        subsets={},coverage={},controls={})
    pairsets={};flat=[]
    by_item={r['id']:{k:r[k] for k in ['id','source','speaker','arm','draw','duration_s','strict_reference','audio_sha256','video_sha256']} for r in rows}
    for metric in METRICS:
        values=load_values(root,metric)
        if metric!='PEAVS':
            for alias,original in aliases.items():
                if original in values:values[alias]=values[original]
        assert set(values).issubset({r['id'] for r in rows}), (metric,'unknown IDs',set(values)-{r['id'] for r in rows})
        assert all(np.isfinite(v) for v in values.values())
        for row in rows:by_item[row['id']][metric]=values.get(row['id'])
        pairs,missing=make_pairs(rows,values);pairsets[metric]=pairs
        summary['coverage'][metric]=dict(measured_media=len(values),planned_media=80,complete_sources=len(pairs),excluded_pairs=missing)
        for pair in pairs:flat.append(dict(metric=metric,**{k:pair[k] for k in ['source','speaker','strict_reference','N','T','delta']}))
        if metric in ['MTDVocaLiST','VocaLiST','Synchformer','SparseSync','AVS_u']:
            records=[read(root/FOLDERS[metric]/r['id']/'deployment_validation.json') for r in read(root/FOLDERS[metric]/'batch_scores.json') if r['status']=='PASS']
            tested=decreased=0
            for rec in records:
                rr=rec['results'];key='frame_mean' if metric=='AVS_u' else 'p_zero' if metric in ['Synchformer','SparseSync'] else None
                native=rr['0'][key] if key else rr['0']['author_metrics']['K5']
                for shift in ['-6400','6400']:
                    v=rr[shift][key] if key else rr[shift]['author_metrics']['K5'];tested+=1;decreased+=v<native
            summary['controls'][metric]=dict(tested_unique_clip_shifts=tested,primary_decreased=int(decreased),
                status='PASS' if tested and decreased/tested>=.8 else 'CONTROL_FAILED',rule='At least80% real +/-400ms shifts lower the native primary endpoint; diagnostic only')
    co=read(root/'cogenav/scores.json');controls=[c for r in co for c in r['raw_shift_controls']]
    moved=sum(c['pass_within_one_frame'] for c in controls)
    positive=float(np.median([r['peak_excess'] for r in co]))>0
    summary['controls']['CoGenAV']=dict(tested_unique_clip_shifts=len(controls),peak_movement_within_one_frame=moved,positive_median_peak_excess=positive,
        status='PASS' if bool(controls) and moved/len(controls)>=.8 and positive else 'CONTROL_FAILED')
    summary['controls']['SyncNet']=dict(status='NOT_RETESTED_THIS_COHORT',note='Single-fixture deployment is not cohort qualification')
    for metric in ['PEAVS','SyncReward']:summary['controls'][metric]=dict(status='NOT_RETESTED_THIS_COHORT',note='Prior Chinese controls failed; do not infer synchronization truth from native-score preference')
    for subset in ['same_reference12','extended_reference8','all20']:
        group={}
        for metric in METRICS:
            pairs=[r for r in pairsets[metric] if subset=='all20' or r['strict_reference']==(subset=='same_reference12')]
            group[metric]=statistics([r['N'] for r in pairs],[r['T'] for r in pairs]) if pairs else dict(n_sources=0,status='UNMEASURABLE')
        ps=[group[m].get('signflip_p',1.) for m in METRICS]
        for m,v in zip(METRICS,holm(ps)):group[m]['holm9_p']=v
        summary['subsets'][subset]=group
    native_ids={r['source'] for r in rows if all(s['samples']>=76000 and s['actual_video_frames']>=119 for s in rows if s['source']==r['source'])}
    native_pairs=[r for r in pairsets['SyncReward'] if r['source'] in native_ids]
    summary['syncreward_full_native_span_sensitivity']=dict(
        interpretation='Only complete source pairs whose four native clips each cover4.75s; descriptive supplementary, different cohort from primary',
        sources=sorted(native_ids),statistics=statistics([r['N'] for r in native_pairs],[r['T'] for r in native_pairs]) if native_pairs else None)
    write(root/'paired_scores.json',pairsets);write(root/'summary.json',summary)
    write(root/'per_video_scores.json',list(by_item.values()))
    with (root/'per_video_scores.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(next(iter(by_item.values())).keys()));writer.writeheader();writer.writerows(by_item.values())
    with (root/'paired_scores.csv').open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=['metric','source','speaker','strict_reference','N','T','delta']);writer.writeheader();writer.writerows(flat)
    lines=['# PC-Talk：中文20人自然/TTS九评分器配对比较','',
        '状态：完整运行结束；以下是评价器分数比较，不是感知同步或嘴型准确度结论。',
        '固定图片3.png、原生25fps、person192、lip1、无EMC、face-index1、seed42。全部输入单声道16k PCM16、RMS −24dBFS；评分ALAC音轨逐样点等于生成驱动音频。',
        '每人3次既有TTS先取均值，再与该人自然语音配对；完全相同音频/视频的重复项保留明确复用凭据。英文未纳入。',
        '12人同句克隆参考为主要分层，8人扩展参考单列；短输入不补静音，各指标实际配对人数在表中。差值为TTS−自然，正值只表示该评分器给TTS更高分。','']
    for subset,group in summary['subsets'].items():
        lines += [f'## {subset}','','|评分器|配对人数|自然均值|TTS均值|差值|95% CI|TTS较高|Holm9 p|','|---|---:|---:|---:|---:|---|---:|---:|']
        for m,r in group.items():
            if not r['n_sources']:lines.append(f'|{m}|0|—|—|—|不可测|—|—|');continue
            spec='.3f' if m=='SyncNet' else '.6g' if m in ['Synchformer','SparseSync'] else '.6f'
            fmt=lambda x,signed=False:format(x,('+' if signed else '')+spec)
            lines.append(f"|{m}|{r['n_sources']}|{fmt(r['N_mean'])}|{fmt(r['T_mean'])}|{fmt(r['delta'],True)}|[{fmt(r['ci95'][0],True)}, {fmt(r['ci95'][1],True)}]|{r['T_higher']}/{r['n_sources']}|{r['holm9_p']:.5f}|")
        lines.append('')
    lines+=['## 适用边界','',*['- '+x for x in summary['limitations']],'','## 实际测量与控制','']
    for m,c in summary['coverage'].items():lines.append(f"- {m}：{c['measured_media']}/80媒体，{c['complete_sources']}/20完整配对；控制状态 {summary['controls'].get(m,{}).get('status','未指定')}。")
    (root/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
    print(json.dumps(summary['subsets'],ensure_ascii=False,indent=2))

if __name__=='__main__':main()
