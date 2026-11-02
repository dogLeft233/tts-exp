"""Full-cohort speaker-level inference, with all repeated sentences kept clustered."""
from pathlib import Path
import argparse,csv,json
import numpy as np
from score_math import METRICS,FOLDERS,load_values,statistics,holm,read,write

STRATA=['strict_quality','all_pairs','new_only_strict','new_only_all']

def selected(row,stratum):
    return row['paired_available'] and ('strict' not in stratum or row['strict_quality']) and ('new_only' not in stratum or not row['pilot_seen'])

def make_pairs(rows,values):
    result=[];missing=[]
    for sid in sorted({r['source'] for r in rows if r['paired_available']}):
        group=[r for r in rows if r['source']==sid]
        assert len(group)==2 and {r['arm'] for r in group}=={'N','T'}
        n=next(r for r in group if r['arm']=='N');t=next(r for r in group if r['arm']=='T')
        absent=[r['id'] for r in group if r['id'] not in values]
        if absent:missing.append(dict(source=sid,speaker=n['speaker'],unmeasurable_ids=absent));continue
        result.append(dict(source=sid,speaker=n['speaker'],strict_quality=n['strict_quality'],pilot_seen=n['pilot_seen'],
            N=values[n['id']],T=values[t['id']],delta=values[t['id']]-values[n['id']]))
    return result,missing

def speaker_pairs(rows,pairs,stratum):
    intended={}
    for row in rows:
        if row['arm']=='N' and selected(row,stratum):intended.setdefault(row['speaker'],set()).add(row['source'])
    available={r['source']:r for r in pairs};result=[];excluded=[]
    for speaker,ids in sorted(intended.items()):
        missing=sorted(ids-set(available))
        if missing:excluded.append(dict(speaker=speaker,missing_sources=missing));continue
        sources=sorted(ids);n=float(np.mean([available[s]['N'] for s in sources]));t=float(np.mean([available[s]['T'] for s in sources]))
        result.append(dict(speaker=speaker,sources=sources,n_pairs=len(sources),N=n,T=t,delta=t-n))
    return result,excluded,intended

def summarize(rows,pairs,stratum):
    sp,excluded,intended=speaker_pairs(rows,pairs,stratum)
    if sp:
        result=statistics([r['N'] for r in sp],[r['T'] for r in sp],seed=20261031)
        result['n_speakers']=result.pop('n_sources')
    else:result=dict(status='UNMEASURABLE',n_speakers=0)
    result.update(n_pairs=sum(r['n_pairs'] for r in sp),planned_pairs=sum(map(len,intended.values())),
        planned_speakers=len(intended),excluded_speakers=excluded,
        unit='Equal-weight speaker means; each speaker must have both arms of every planned pair in this stratum')
    return result,sp

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);root=p.parse_args().run
    inputs=read(root/'scoring_inputs.json');rows=inputs['rows'];assert len(rows)==209
    pairsets={};speaker_sets={k:{} for k in STRATA};coverage={};byvideo={r['id']:dict(r) for r in rows};controls={}
    paired_ids={r['id'] for r in rows if r['paired_available']}
    for metric in METRICS:
        values=load_values(root,metric)
        for alias,original in inputs['aliases'].items():
            if metric!='PEAVS' and original in values:values[alias]=values[original]
        assert set(values).issubset(byvideo) and all(np.isfinite(v) for v in values.values())
        for row in rows:byvideo[row['id']][metric]=values.get(row['id'])
        pairs,missing=make_pairs(rows,values);pairsets[metric]=pairs
        coverage[metric]=dict(measured_media=len(values),planned_media=209,complete_pairs=len(pairs),planned_pairs=104,
            excluded_pairs=missing,unpaired_measured_ids=[r['id'] for r in rows if not r['paired_available'] and r['id'] in values])
        if metric in ['AVS_u','MTDVocaLiST','VocaLiST','Synchformer','SparseSync']:
            all_tested=all_down=paired_tested=paired_down=0
            for r in read(root/FOLDERS[metric]/'batch_scores.json'):
                if r['status']!='PASS':continue
                rr=read(root/FOLDERS[metric]/r['id']/'deployment_validation.json')['results']
                def endpoint(z):return z['frame_mean'] if metric=='AVS_u' else z['author_metrics']['K5'] if metric in ['MTDVocaLiST','VocaLiST'] else z['p_zero']
                for shift in ['-6400','6400']:
                    down=endpoint(rr[shift])<endpoint(rr['0']);all_tested+=1;all_down+=down
                    if r['id'] in paired_ids:paired_tested+=1;paired_down+=down
            controls[metric]=dict(status='PASS' if paired_tested and paired_down/paired_tested>=.8 else 'CONTROL_FAILED',
                paired_tested=paired_tested,paired_decreased=int(paired_down),all_tested=all_tested,all_decreased=int(all_down),
                rule='At least80% real +/-400ms decreases on paired media; unpaired N diagnostic counted separately')
    co=read(root/'cogenav/scores.json');pcs=[c for r in co if r['id'] in paired_ids for c in r['raw_shift_controls']]
    acs=[c for r in co for c in r['raw_shift_controls']];moved=sum(c['pass_within_one_frame'] for c in pcs)
    positive=bool(co) and float(np.median([r['peak_excess'] for r in co if r['id'] in paired_ids]))>0
    controls['CoGenAV']=dict(status='PASS' if pcs and moved/len(pcs)>=.8 and positive else 'CONTROL_FAILED',paired_tested=len(pcs),
        paired_peak_moved_within1frame=moved,positive_paired_median_peak_excess=positive,all_tested=len(acs))
    for m in ['SyncNet','PEAVS','SyncReward']:controls[m]=dict(status='NOT_RETESTED_THIS_COHORT',note='Retain prior control and PEAVS batch-dependence limits')
    subsets={}
    for stratum in STRATA:
        group={}
        for m in METRICS:group[m],speaker_sets[stratum][m]=summarize(rows,pairsets[m],stratum)
        for m,q in zip(METRICS,holm([group[m].get('signflip_p',1.) for m in METRICS])):group[m]['holm9_p']=q
        subsets[stratum]=group
    descriptive={}
    for m in METRICS:
        values={r['id']:r[m] for r in byvideo.values() if r[m] is not None}
        descriptive[m]={}
        for arm in ['N','T']:
            z=[values[r['id']] for r in rows if r['arm']==arm and r['id'] in values]
            descriptive[m][arm]=dict(n_media=len(z),mean=float(np.mean(z)) if z else None)
    summary=dict(status='COMPLETE_EXPLORATORY',planned_media=209,planned_pairs=104,planned_speakers=21,
        primary='strict_quality',supplementary=['all_pairs','new_only_strict','new_only_all'],subsets=subsets,coverage=coverage,controls=controls,
        statistics='Speaker means then10k speaker bootstrap/100k signflip seed20261031,Holm9 separately per stratum; marginal95%CI',
        exploratory='pilot_seen refers to historical PC-Talk pilot sentences, not DyStream prior score inspection. Same speakers, not independent participants.',
        descriptive_unbalanced_media_means=descriptive,protocol=read(root/'protocol.json'))
    write(root/'summary.json',summary);write(root/'paired_scores.json',pairsets);write(root/'speaker_scores.json',speaker_sets)
    write(root/'per_video_scores.json',list(byvideo.values()))
    flat=[dict(metric=m,**r) for m,pp in pairsets.items() for r in pp]
    with (root/'paired_scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['# DyStream Chinese-LiPS全量：105自然+104TTS，21说话人','',
        '每人最多5句先取说话人内配对均值，再按说话人做统计；104句不是104独立人。',
        '主要严格质量84对/17人；补充全104对/21人；另报去掉PC-Talk曾试跑的21句后的同队列敏感性。已知缺TTS的一条自然仅描述。',
        'PEAVS208配对媒体batch64+64+64+16重新计算；单侧自然另batch1，不进入配对组。',
        '评价器证据，非感知真值/完整无泄露证明。95%CI为边际，显著性按每分层Holm9。','']
    for name,group in subsets.items():
        lines += [f'## {name}','','|端点|完整说话人|完整句对|自然|TTS|T−N|95%CI|Holm9 p|','|---|---:|---:|---:|---:|---:|---|---:|']
        for m,r in group.items():
            if not r['n_speakers']:lines.append(f'|{m}|0|0|—|—|—|—|—|');continue
            fmt=lambda x:format(x,'.3f' if m=='SyncNet' else '.6f')
            lines.append(f"|{m}|{r['n_speakers']}|{r['n_pairs']}|{fmt(r['N_mean'])}|{fmt(r['T_mean'])}|{fmt(r['delta'])}|[{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]|{r['holm9_p']:.5f}|")
        lines.append('')
    lines += ['## 支持与控制','',*[f'- {m}: {c["measured_media"]}/209媒体，{c["complete_pairs"]}/104完整句对；控制{controls[m]["status"]}。' for m,c in coverage.items()],
        '', '## 协议与边界','',*['- '+x for x in summary['protocol']['limitations']],
        '- 原21对及八类逐片分数明确SHA复用，PEAVS因batch依赖重新评；旧试跑未改变。',
        '- SyncNet内部MP3转换和原生offset保留，没有强制内部PCM或偏移补偿消融。',
        '- 完整说话人要求预定句对全部N/T可测；缺测不补值、不用坏分数删句或调门槛。',
        '- 每项原生K5/P0在本表为0–1，不能与Sync-C混作总分；未显著不等价。']
    (root/'report.md').write_text('\n'.join(lines)+'\n');print(json.dumps(subsets,indent=2))

if __name__=='__main__':main()
