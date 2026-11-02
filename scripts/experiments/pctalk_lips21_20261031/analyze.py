"""Source-paired nine-endpoint pilot; no model calls or score-based exclusions."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from score_math import METRICS,FOLDERS,load_values,statistics,holm,read,write

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);root=p.parse_args().run
    inputs=read(root/'scoring_inputs.json');rows=inputs['rows'];assert len(rows)==42
    pairsets={};pervideo={r['id']:dict(r) for r in rows};coverage={};controls={}
    for metric in METRICS:
        values=load_values(root,metric);assert set(values).issubset(pervideo)
        for alias,original in inputs['aliases'].items():
            if metric!='PEAVS' and original in values:values[alias]=values[original]
        assert all(np.isfinite(v) for v in values.values())
        for row in rows:pervideo[row['id']][metric]=values.get(row['id'])
        pairs=[];missing=[]
        for source in sorted({r['source'] for r in rows}):
            n,t=[next(r for r in rows if r['source']==source and r['arm']==arm) for arm in ['N','T']]
            absent=[r['id'] for r in [n,t] if r['id'] not in values]
            if absent:missing.append(dict(source=source,ids=absent));continue
            pairs.append(dict(source=source,speaker=n['speaker'],strict_quality=n['strict_quality'],
                N=values[n['id']],T=values[t['id']],delta=values[t['id']]-values[n['id']]))
        pairsets[metric]=pairs
        coverage[metric]=dict(measured_media=len(values),planned_media=42,complete_sources=len(pairs),unmeasurable_pairs=missing)
        if metric in ['MTDVocaLiST','VocaLiST','Synchformer','SparseSync','AVS_u']:
            tested=decreased=0
            for r in read(root/FOLDERS[metric]/'batch_scores.json'):
                if r['status']!='PASS':continue
                rr=read(root/FOLDERS[metric]/r['id']/'deployment_validation.json')['results']
                def endpoint(z):
                    return z['frame_mean'] if metric=='AVS_u' else z['author_metrics']['K5'] if metric in ['MTDVocaLiST','VocaLiST'] else z['p_zero']
                for shift in ['-6400','6400']:
                    tested+=1;decreased+=endpoint(rr[shift])<endpoint(rr['0'])
            controls[metric]=dict(tested=tested,decreased=int(decreased),status='PASS' if tested and decreased/tested>=.8 else 'CONTROL_FAILED',rule='>=80% real +/-400ms shifts decrease native endpoint')
    co=read(root/'cogenav/scores.json');cs=[c for r in co for c in r['raw_shift_controls']]
    moved=sum(c['pass_within_one_frame'] for c in cs)
    positive=bool(co) and float(np.median([r['peak_excess'] for r in co]))>0
    controls['CoGenAV']=dict(tested=len(cs),peak_moved_within1frame=moved,positive_median_peak_excess=positive,
        status='PASS' if cs and moved/len(cs)>=.8 and positive else 'CONTROL_FAILED')
    for metric in ['SyncNet','PEAVS','SyncReward']:
        controls[metric]=dict(status='NOT_RETESTED_THIS_COHORT',note='Prior control limitations retained; no claim of perceptual truth')
    subsets={}
    for subset in ['strict_quality17','all21']:
        group={}
        for metric in METRICS:
            pairs=[r for r in pairsets[metric] if subset=='all21' or r['strict_quality']]
            group[metric]=statistics([r['N'] for r in pairs],[r['T'] for r in pairs],seed=20261031) if pairs else dict(n_sources=0,status='UNMEASURABLE')
        for metric,q in zip(METRICS,holm([group[m].get('signflip_p',1.) for m in METRICS])):group[metric]['holm9_p']=q
        subsets[subset]=group
    summary=dict(status='COMPLETE_EXPLORATORY',planned_sources=21,planned_videos=42,unit='one N and one existing T per distinct audio speaker',
        primary='strict_quality17',supplementary='all21',subsets=subsets,coverage=coverage,controls=controls,
        statistics='10k paired bootstrap,100k two-sided signflip seed20261031; Holm9 separately per subset; marginal95% CI',
        protocol=read(root/'protocol.json'))
    write(root/'summary.json',summary);write(root/'paired_scores.json',pairsets);write(root/'per_video_scores.json',list(pervideo.values()))
    flat=[dict(metric=m,**r) for m,pairs in pairsets.items() for r in pairs]
    with (root/'paired_scores.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(flat[0]));w.writeheader();w.writerows(flat)
    lines=['# PC-Talk Chinese-LiPS21 配对九评分器试跑','',
        '每人按成功ID字典序选首句，21位说话人各一对自然/TTS；17对无既有饱和/触顶标记为主要分析，全部21对补充。',
        '音频来自Chinese-LiPS官方test；每位音频说话人固定一张性别匹配HDTF异人人像。单TTS draw、RMS−24、person192/lip1/无EMC/seed42；按原生脸检最大框在left-right排序中的索引选主脸。',
        '生成与评分同一完整PCM，不变速、不补帧或移音。分数是评价器证据，不等同人类同步质量。','']
    for subset,group in subsets.items():
        lines += [f'## {subset}','','|评分器|配对人数|自然|TTS|T−N|95%CI|T较高|Holm9 p|','|---|---:|---:|---:|---:|---|---:|---:|']
        for metric,r in group.items():
            if not r['n_sources']:lines.append(f'|{metric}|0|—|—|—|—|—|—|');continue
            fmt=lambda v:format(v,'.3f' if metric=='SyncNet' else '.6f')
            lines.append(f"|{metric}|{r['n_sources']}|{fmt(r['N_mean'])}|{fmt(r['T_mean'])}|{fmt(r['delta'])}|[{fmt(r['ci95'][0])}, {fmt(r['ci95'][1])}]|{r['T_higher']}/{r['n_sources']}|{r['holm9_p']:.5f}|")
        lines.append('')
    lines += ['## 支持与控制','',*[f'- {m}: {coverage[m]["measured_media"]}/42媒体，{coverage[m]["complete_sources"]}/21完整配对；{controls[m]["status"]}。' for m in METRICS],
        '', '## 边界','',*['- '+x for x in summary['protocol']['limitations']],
        '- 旧中文20来源实验为固定3.png、三TTS逻辑draw与不同音频/参考；本批21人单draw、多HDTF人像，不能把绝对分变化归因于某一个条件。',
        '- 官方SyncNet内部仍包含MPEG4/MP3转换；本轮没有移除内部音频延迟的消融。',
        '- PEAVS按42条字典序组成一个原生batch（上限64），与旧64+16不同；不把跨批绝对值视为完全可比。',
        '- 原始自然音频饱和或TTS触顶不会被后续标量归一化修复；全部保留并单列严格质量支持。']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(subsets,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
