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
        return {r['id']:float(r['C' if metric=='SyncNet' else 'eq1']) for r in read(root/folder/'scores.json')}
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

def pair_clips(rows, values):
    pairs=[];missing=[]
    for source in sorted({r['source'] for r in rows}):
        group=[r for r in rows if r['source']==source]
        assert len(group)==2 and {r['arm'] for r in group}=={'N','T'}
        n=next(r for r in group if r['arm']=='N');t=next(r for r in group if r['arm']=='T')
        if n['id'] not in values or t['id'] not in values:
            missing.append(source);continue
        pairs.append(dict(source=source,speaker=n['speaker'],N=values[n['id']],T=values[t['id']],delta=values[t['id']]-values[n['id']],qc_flag=n['tts_qc_flag']))
    return pairs,missing

def speaker_pairs(rows, clips):
    result=[];missing=[]
    for speaker in sorted({r['speaker'] for r in rows}):
        group=[r for r in clips if r['speaker']==speaker]
        if len(group)!=2:
            missing.append(speaker);continue
        n=float(np.mean([r['N'] for r in group]));t=float(np.mean([r['T'] for r in group]))
        result.append(dict(source=speaker,speaker=speaker,N=n,T=t,delta=t-n,clips=[r['source'] for r in group],qc_flag=any(r['qc_flag'] for r in group)))
    return result,missing

def dump_table(root,name,rows):
    write(root/(name+'.json'),rows)
    with (root/(name+'.csv')).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def controls(root):
    result={}
    for metric in ['AVS_u','MTDVocaLiST','VocaLiST','Synchformer','SparseSync']:
        folder=FOLDERS[metric];tested=decreased=0
        for rec in read(root/folder/'batch_scores.json'):
            if rec['status']!='PASS':continue
            rr=read(root/folder/rec['id']/'deployment_validation.json')['results']
            def primary(r):
                return r['frame_mean'] if metric=='AVS_u' else r['p_zero'] if metric in ['Synchformer','SparseSync'] else r['author_metrics']['K5']
            for shift in ['-6400','6400']:
                tested+=1;decreased+=primary(rr[shift])<primary(rr['0'])
        result[metric]=dict(tested_unique_clip_shifts=tested,primary_decreased=int(decreased),status='PASS' if tested and decreased/tested>=.8 else 'CONTROL_FAILED')
    co=read(root/'cogenav/scores.json');cc=[c for r in co for c in r['raw_shift_controls']]
    moved=sum(c['pass_within_one_frame'] for c in cc)
    positive=float(np.median([r['peak_excess'] for r in co]))>0
    result['CoGenAV']=dict(tested_unique_clip_shifts=len(cc),peak_movement_within_one_frame=moved,positive_median_peak_excess=positive,status='PASS' if cc and moved/len(cc)>=.8 and positive else 'CONTROL_FAILED')
    for metric in ['SyncNet','PEAVS','SyncReward']:
        result[metric]=dict(status='NOT_RETESTED_THIS_COHORT',note='Native preference is not perceptual synchronization truth; retain prior diagnostic limitations')
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);root=p.parse_args().run
    assert read(root/'status.json')['status']=='SCORING_COMPLETE'
    inputs=read(root/'scoring_inputs.json');rows=inputs['rows'];assert len(rows)==80
    summary=dict(status='COMPLETE_EXPLORATORY',planned_speakers=20,planned_clips=40,planned_videos=80,
        unit='twenty speakers: mean of two segment scores per arm; one cloud TTS draw per segment',
        normalization=read(root/'protocol.json')['audio'],coverage={},controls=controls(root),subsets={},
        limitations=['HDTF was a TalkLikeYou training source; not unseen and not the published official test split.',
          'Fixed 10s windows can truncate words/sentences. ASR transcripts and ASR-based TTS QC are not human ground truth.',
          'Per-clip portrait changes across clips, held identical within N/T; old Chinese cohort used fixed3.png and three TTS draws.',
          'Native complete-mel windows omit ~0.2s of visual tail; full scoring PCM unchanged, no padding/retiming/offset correction.',
          'Nine scorer endpoints are not interchangeable or perceptual truth. Six shift diagnostics are reported separately.',
          'Bootstrap CIs are marginal; sign-flip sensitivity assumes speaker-difference symmetry/exchangeability; Holm9 is separate.'])
    by_item={r['id']:{k:r[k] for k in ['id','source','speaker','arm','draw','duration_s','audio_sha256','video_sha256']} for r in rows}
    all_pairs={};clip_rows=[];speaker_rows=[]
    for metric in METRICS:
        values=load_values(root,metric)
        for alias,original in inputs['aliases'].items():
            if original in values:values[alias]=values[original]
        assert set(values)<=set(by_item) and all(np.isfinite(v) for v in values.values())
        for sid in by_item:by_item[sid][metric]=values.get(sid)
        clips,missing=pair_clips(rows,values);speakers,missing_speakers=speaker_pairs(rows,clips)
        all_pairs[metric]=speakers
        summary['coverage'][metric]=dict(measured_media=len(values),complete_clips=len(clips),complete_speakers=len(speakers),excluded_clips=missing,excluded_speakers=missing_speakers)
        clip_rows += [dict(metric=metric,**r) for r in clips]
        speaker_rows += [dict(metric=metric,**r) for r in speakers]
    for subset in ['all20','qc_clean_speakers_sensitivity']:
        group={}
        for metric,pairs in all_pairs.items():
            selected=pairs if subset=='all20' else [r for r in pairs if not r['qc_flag']]
            group[metric]=statistics([r['N'] for r in selected],[r['T'] for r in selected]) if selected else dict(n_sources=0,status='UNMEASURABLE')
        for metric,q in zip(METRICS,holm([group[m].get('signflip_p',1.) for m in METRICS])):group[metric]['holm9_p']=q
        summary['subsets'][subset]=group
    real=read(root/'syncnet/real_scores.json');ng=read(root/'syncnet/scores.json');lookup={r['id']:r for r in ng}
    assert len(real)==40
    gt_pairs=[]
    for r in real:
        source=r['id'][:-2];n=lookup[source+'_N'];inputrow=next(x for x in rows if x['id']==source+'_N')
        gt_pairs.append(dict(source=source,speaker=inputrow['speaker'],N=n['C'],T=r['C'],delta=r['C']-n['C'],qc_flag=inputrow['tts_qc_flag']))
    gt,_=speaker_pairs(rows,gt_pairs)
    summary['real_video_syncnet_diagnostic']=dict(statistics([r['N'] for r in gt],[r['T'] for r in gt]),
        interpretation='T field here means real-video GT prefix, N means generated natural; prespecified separate diagnostic, no Holm9 inference',
        N_D_mean=float(np.mean([lookup[r['id'][:-2]+'_N']['D'] for r in real])),GT_D_mean=float(np.mean([r['D'] for r in real])))
    summary['syncnet_components']={}
    for arm in ['N','T']:
        rr=[r for r in ng if r['id'].endswith('_'+arm)]
        summary['syncnet_components'][arm]=dict(C=float(np.mean([r['C'] for r in rr])),D=float(np.mean([r['D'] for r in rr])),B=float(np.mean([r['B'] for r in rr])),offset_counts={str(o):sum(r['offset']==o for r in rr) for o in sorted({r['offset'] for r in rr})})
    dump_table(root,'per_video_scores',list(by_item.values()));dump_table(root,'clip_pairs',clip_rows);dump_table(root,'paired_scores',speaker_rows)
    write(root/'paired_scores_by_metric.json',all_pairs);write(root/'summary.json',summary)
    lines=['# TalkLikeYou HDTF20：自然/TTS九评分器比较','',
      '20说话人×2片段，40个自然/TTS配对；每段一次指定Plus云克隆。每人先平均两个片段，再以20人为统计单位。输入统一RMS−27dBFS，每对同图。',
      '固定10秒窗可能截断词句；ASR文本与回读QC不是人工真值。HDTF训练重叠、非论文官方测试划分。','',
      '|评分器|配对人数|自然均值|TTS均值|TTS−自然|边际95%CI|Holm9 p|','|---|---:|---:|---:|---:|---|---:|']
    for m in METRICS:
        r=summary['subsets']['all20'][m]
        if not r['n_sources']:lines.append(f'|{m}|0|—|—|—|不可测|—|');continue
        precision=3 if m=='SyncNet' else 6;ci=r['ci95']
        lines.append(f"|{m}|{r['n_sources']}|{r['N_mean']:.{precision}f}|{r['T_mean']:.{precision}f}|{r['delta']:+.{precision}f}|[{ci[0]:+.{precision}f}, {ci[1]:+.{precision}f}]|{r['holm9_p']:.5f}|")
    r=summary['real_video_syncnet_diagnostic']
    lines += ['',f"真实视频同支持参考：生成自然Sync-C {r['N_mean']:.3f}，真实视频{r['T_mean']:.3f}，GT−生成{r['delta']:+.3f}；真实视频与N生成视频均使用同一完整RMS27音轨及相同画面帧数。独立诊断，不混入九项Holm家族。",'','## 控制与边界','']
    for m,c in summary['controls'].items():lines.append(f"- {m}: {c['status']}; {json.dumps(c,ensure_ascii=False)}")
    lines += ['','## 协议差异','']+[ '- '+x for x in summary['limitations']]
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(summary['subsets']['all20'],indent=2))
if __name__=='__main__':main()
