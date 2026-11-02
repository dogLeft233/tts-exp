"""Frozen four-source phrase localization: preflight before local effects."""
import argparse, hashlib, json, pickle, subprocess, unicodedata, wave, os, time, shutil, sys
from pathlib import Path
import numpy as np
from scipy.signal import correlate, correlation_lags

ROOT=Path('/home/wjj/tts-audio/tts-exp')
OUT=ROOT/'runs/qwen_phrase_local_20261024_attempt02'
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x): Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def norm(s):
    indices=[i for i,c in enumerate(s) if not c.isspace() and not unicodedata.category(c).startswith('P')]
    return {'original':s,'normalized':''.join(s[i].lower() for i in indices),'original_indices':indices}
def pcm(p):
    z=subprocess.run(['ffmpeg','-v','error','-i',str(p),'-f','s16le','-ac','1','-ar','16000','-'],capture_output=True,check=True)
    return np.frombuffer(z.stdout,dtype='<i2').astype(float)
def binding(y,x,nominal):
    cc=correlate(y,x,mode='full',method='fft'); ll=correlation_lags(len(y),len(x))
    ok=abs(ll+nominal*16000)<4801
    lag=int(ll[ok][np.argmax(cc[ok])]); a=max(0,lag); b=min(len(y),len(x)+lag)
    xx=x[a-lag:b-lag]; yy=y[a:b]
    return {'crop_zero_canonical_s':-lag/16000,'lag_samples_y_minus_x':lag,'correlation':float(np.corrcoef(xx,yy)[0,1]),'overlap_samples':len(xx),'nominal_zero_s':nominal,'delay_s':nominal+lag/16000}
def probe(p):
    return json.loads(subprocess.run(['ffprobe','-v','error','-select_streams','v:0','-show_frames','-show_entries','frame=best_effort_timestamp_time','-of','json',str(p)],capture_output=True,check=True).stdout)
def align():
    rows=read(OUT/'input_manifest.json')['audio']
    smoke=ROOT/'runs/funasr_deployment_20261024_attempt01/smoke_attempt01.json'
    existing={r['sha256']:r for r in read(smoke)['audio']}; results=[]; todo=[]
    for r in rows:
        if r['sha256'] in existing:
            z=dict(existing[r['sha256']]);z.update(id=r['id'],sample_id=r['sample_id'],model=r['model'],draw=r['draw'],reuse_origin=str(smoke),reuse_sha256=sha(smoke));results.append(z)
        else:todo.append(r)
    save(OUT/'alignment_manifest.json',{'audio':rows,'reused':len(results),'new':len(todo)})
    runner=ROOT/'runs/funasr_deployment_20261024_attempt01/funasr_deployment_smoke_validated.py'
    shutil.copyfile(runner,OUT/'alignment_runner.py')
    receipts=[]
    for b in range(0,len(todo),8):
        man=OUT/f'align_batch_{b//8+1}.json';dest=OUT/f'align_result_{b//8+1}.json';log=OUT/f'align_batch_{b//8+1}.log';save(man,{'audio':todo[b:b+8]})
        cmd=['/home/wjj/.venvs/funasr-cpu-20261024/bin/python',str(runner),'--manifest',str(man),'--model-root','/home/wjj/tts-audio/model_cache/funasr_20261024','--output',str(dest)]
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='-1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HUB_OFFLINE='1')
        t=time.time()
        with log.open('w') as stream:
            p=subprocess.Popen(cmd,stdout=stream,stderr=subprocess.STDOUT,env=env)
            samples=[]
            while p.poll() is None:
                ps=subprocess.run(['ps','-p',str(p.pid),'-o','pid,etime,pcpu,pmem,rss,args'],capture_output=True,text=True).stdout;samples.append({'time':time.time(),'ps':ps});time.sleep(1)
        rec={'command':cmd,'pid':p.pid,'returncode':p.returncode,'elapsed_s':time.time()-t,'environment':{k:env[k] for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','HF_HUB_OFFLINE']},'process_samples':samples,'log':str(log)};receipts.append(rec);save(OUT/'alignment_receipts.json',receipts)
        assert p.returncode==0
        for z in read(dest)['audio']:
            r=next(r for r in rows if r['sha256']==z['sha256']);z.update(sample_id=r['sample_id']);results.append(z)
    assert len(results)==28
    # Smoke fa validated structurally by the same corrected validator; retain native ASR range failures.
    save(OUT/'alignment.json',{'audio':results,'smoke_source':str(smoke),'smoke_sha256':sha(smoke)})
    save(OUT/'input_sha_after_alignment.json',{r['source']:sha(r['source']) for r in rows})
    print(json.dumps({'alignment_count':len(results),'new':len(todo),'reused':len(existing),'fa_issues':[(r['id'],r['fa_issues']) for r in results if r['fa_issues']],'ASR_mismatch':[(r['id'],r['asr_text']) for r in results if not r['transcript_exact_match']]},ensure_ascii=False),flush=True)

def supplement():
    rows=read(OUT/'input_manifest.json')['audio']; rr=[]
    for r in rows:
        x=pcm(r['source']);ts=[]
        for t in r['tracks']:
            y=pcm(t['crop_path']);base=t['audio_binding']['crop_zero_canonical_s'];shift=round(-base*16000)
            segments=[]
            for u in [.2,.5,.8]:
                a=int(len(x)*u)-2000;b=a+4000
                if a<0 or b>len(x):continue
                # Match each canonical 250ms waveform in crop within +/-20ms of measured global shift.
                template=x[a:b];lo=max(0,a+shift-320);hi=min(len(y),b+shift+320)
                if hi-lo<len(template):continue
                cc=correlate(y[lo:hi],template,mode='valid',method='fft');best=int(np.argmax(cc));start=lo+best
                segments.append({'canonical_start_sample':a,'length':b-a,'delay_samples':start-a,'correlation':float(np.corrcoef(template,y[start:start+b-a])[0,1]),'search_radius_samples':320})
            n=t['array_shape'][0];f=t['frames'];z=base
            valid=[]
            for i in range(n):
                js=list(range(i-15,i+16))
                if min(js)<0 or max(js)>=n:continue
                if not all(t['crop_source_frames_valid'][i:i+5]):continue
                if z+.04*min(js)-1/16000<0 or z+.04*max(js)+.215>len(x)/16000:continue
                valid.append(i)
            ts.append({'track_index':t['track_index'],'frame_start':f[0],'frame_end':f[-1],'correlation_chunks':segments,'all_31lag_actual_support_rows':valid,'whole_track_C_error':t['reconstructed_C']-t['official_C'],'whole_track_D_error':t['reconstructed_D']-t['official_D']})
        rr.append({'id':r['id'],'tracks':ts})
    save(OUT/'time_binding_supplement.json',{'audio':rr,'binding_limit':'MP3 processing changes waveform; high whole/250ms block correlation plus consistent sample lag is empirical time-binding evidence, not waveform identity or guaranteed phonetic boundary accuracy. No local effects consulted.','RF_detail':'audio RF j>0 begins one preceding preemphasis sample: base+.04*j-.0000625; ends base+.04*j+.215. Video RF [frame[i]/25,(frame[i+4]+1)/25).','sensitivity':[[0,0],[-.04,-.04],[.04,.04],[.04,-.04],[-.04,.04]]})

VARIANTS={'nominal':(0,0),'left':(-.04,-.04),'right':(.04,.04),'contract':(.04,-.04),'expand':(-.04,.04)}
def support(i,t,r):
    n=t['array_shape'][0];js=list(range(i-15,i+16));base=t['audio_binding']['crop_zero_canonical_s']
    reasons=[]
    if min(js)<0 or max(js)>=n:reasons.append('lag_feature_padding')
    if not all(t['crop_source_frames_valid'][i:i+5]):reasons.append('video_duplicate_tail')
    # Each MFCC frame operates on the preemphasized complete crop signal.
    if min(js)*640-1<0 or max(js)*640+3440>t['crop_audio_samples']:reasons.append('crop_audio_or_mfcc_padding')
    if base+min(js)*.04-1/16000<0 or base+max(js)*.04+.215>r['canonical_audio_samples']/16000:reasons.append('outside_canonical_audio')
    j=i+t['fixed_best_column']-15
    return {'row':i,'matched_audio_feature_row':j,'video_RF':[t['frames'][i]/25,(t['frames'][i+4]+1)/25],'matched_audio_RF':[base+j*.04-(1/16000 if j>0 else 0),base+j*.04+.215],'all_31lag_actual_support':not reasons,'invalid_reasons':reasons}
def inside(w,a,b):return min(w['video_RF'][0],w['matched_audio_RF'][0])>=a and max(w['video_RF'][1],w['matched_audio_RF'][1])<=b
def direction(vals):
    if any(v is None for v in vals):return 'INCOMPLETE'
    if all(v>1e-6 for v in vals):return 'positive'
    if all(v< -1e-6 for v in vals):return 'negative'
    return 'mixed_or_near_zero'
def effects():
    assert not (OUT/'local_metrics.json').exists()
    rows=read(OUT/'input_manifest.json')['audio'];aligns={r['id']:r for r in read(OUT/'alignment.json')['audio']};phrases=read(OUT/'protocol.json')['phrases']
    metrics=[];winmaps=[]
    for r in rows:
        a=aligns[r['id']];timestamps=a['alignment'][0]['timestamp'];assert not a['fa_issues']
        arrays=pickle.loads(Path(r['activesd_path']).read_bytes())
        for t,ar in zip(r['tracks'],arrays):
            ar=np.asarray(ar,float);windows=[support(i,t,r) for i in range(len(ar))];winmaps.append({'id':r['id'],'track_index':t['track_index'],'windows':windows})
            np.savez_compressed(OUT/f"distance_{r['id']}_track{t['track_index']}.npz",distance=ar)
            pos=0
            for pi,phrase in enumerate(phrases[r['sample_id']]):
                end=pos+len(phrase);start_s=timestamps[pos][0]/1000;end_s=timestamps[end-1][1]/1000
                changed=[op for op in a['transcript_diff_opcodes'] if op[0]!='equal' and (max(pos,op[1])<min(end,op[2]) or op[1]==op[2] and pos<=op[1]<end)]
                content='content_uncertain_ASR_mismatch' if changed else 'ASR_exact_for_phrase'
                for name,(ds,de) in VARIANTS.items():
                    lo=start_s+ds;hi=end_s+de;accepted=[w['row'] for w in windows if w['all_31lag_actual_support'] and inside(w,lo,hi)]
                    crosses=[w['row'] for w in windows if w['all_31lag_actual_support'] and not inside(w,lo,hi) and max(w['video_RF'][0],w['matched_audio_RF'][0])<hi and min(w['video_RF'][1],w['matched_audio_RF'][1])>lo]
                    z={'id':r['id'],'sample_id':r['sample_id'],'model':r['model'],'draw':r['draw'],'track_index':t['track_index'],'phrase_index':pi,'phrase':phrase,'character_span':[pos,end],'nominal_bounds':[start_s,end_s],'bounds':[lo,hi],'variant':name,'fixed_best_column':t['fixed_best_column'],'valid_rows':accepted,'cross_boundary_rows':crosses,'window_count':len(accepted),'content_status':content,'ASR_diff_overlapping_phrase':changed,'status':'OK' if len(accepted)>=5 else 'INCOMPLETE_UNDER_5_WINDOWS','contrast':None,'matched_distance':None,'median_background':None}
                    if len(accepted)>=5:
                        mu=ar[accepted].mean(0);matched=float(mu[t['fixed_best_column']]);median=float(np.median(mu));z.update(contrast=median-matched,matched_distance=matched,median_background=median,lag_means=mu.tolist())
                    metrics.append(z)
                pos=end
    save(OUT/'window_time_maps.json',winmaps);save(OUT/'local_metrics.json',metrics)
    lookup={(x['sample_id'],x['model'],x['draw'],x['phrase_index'],x['variant']):x for x in metrics};comparisons=[]
    for sid,pp in phrases.items():
        for pi,ph in enumerate(pp):
            for ma,mb in [('L17','L06'),('L06','N'),('L17','N')]:
                variants={};content=[]
                for name in VARIANTS:
                    draw=[]
                    for d in [1,2,3]:
                        x=lookup[(sid,ma,d,pi,name)];y=lookup[(sid,mb,0 if mb=='N' else d,pi,name)]
                        content.extend([x['content_status'],y['content_status']])
                        draw.append({'draw':d,'A_windows':x['window_count'],'B_windows':y['window_count'],**{key:(x[key]-y[key] if x[key] is not None and y[key] is not None else None) for key in ['contrast','matched_distance','median_background']}})
                    variants[name]={'draws':draw,'direction':direction([z['contrast'] for z in draw])}
                nominal=variants['nominal']['direction'];robust=nominal in ['positive','negative'] and all(z['direction']==nominal for z in variants.values())
                comparisons.append({'sample_id':sid,'phrase_index':pi,'phrase':ph,'comparison':ma+'-'+mb,'nominal_direction':nominal,'boundary_robust':robust,'content_status':'content_uncertain_ASR_mismatch' if 'content_uncertain_ASR_mismatch' in content else 'ASR_exact_for_phrase','variants':variants})
    save(OUT/'comparisons.json',comparisons)
    lines=['|source|phrase|comparison|draw1|draw2|draw3|nominal direction|5-boundary robust|content|','|---|---|---|---:|---:|---:|---|---|---|']
    for c in comparisons:
        v=[('NA' if d['contrast'] is None else f"{d['contrast']:+.3f}")+f" ({d['A_windows']}/{d['B_windows']})" for d in c['variants']['nominal']['draws']]
        lines.append('|'+ '|'.join([c['sample_id'],c['phrase'],c['comparison'],*v,c['nominal_direction'],str(c['boundary_robust']),c['content_status']])+'|')
    (OUT/'comparison_table.md').write_text('\n'.join(lines),encoding='utf-8');print('\n'.join(lines),flush=True)

def present():
    import html
    rows=read(OUT/'input_manifest.json')['audio'];metrics=read(OUT/'local_metrics.json');maps={x['id']:x['windows'] for x in read(OUT/'window_time_maps.json')};clips=OUT/'clips';clips.mkdir(exist_ok=False)
    receipts=[];page=['<!doctype html><meta charset="utf-8"><title>Four-source phrase evidence</title><style>body{font:16px sans-serif;max-width:1200px;margin:auto}article{padding:10px;border-bottom:1px solid #ccc}video{width:280px}audio{width:310px}svg{max-width:100%}</style><h1>9 frozen phrases · 28 original utterances</h1><p>Automatic FA boundaries; 0.3s context. Local contrast is diagnostic, not official segment Sync-C or perceptual quality. All 63 phrase/utterance clips are included. ASR uncertainty and insufficient support remain visible.</p>']
    for r in rows:
        canonical=pcm(r['source']).astype('<i2');duration=len(canonical)/16000
        rr=[z for z in metrics if z['id']==r['id'] and z['variant']=='nominal']
        ar=np.asarray(pickle.loads(Path(r['activesd_path']).read_bytes())[0],float);t=r['tracks'][0]
        matched=ar[:,t['fixed_best_column']]; background=np.median(ar,axis=1);w=900;h=220; ymin=min(float(ar.min()),0);ymax=float(ar.max());sx=lambda i:40+i/(len(ar)-1)*(w-60);sy=lambda y:h-25-(y-ymin)/(ymax-ymin)*(h-50)
        svg=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}"><rect width="100%" height="100%" fill="white"/><text x="40" y="17" font-size="13">{r["id"]}: blue fixed-lag distance; gray rowwise median (display only)</text>']
        for z in rr:
            a,b=z['nominal_bounds'];svg.append(f'<rect x="{sx(a/.04):.2f}" y="25" width="{(b-a)/.04/(len(ar)-1)*(w-60):.2f}" height="{h-50}" fill="#e6f4ff" opacity=".4"/>')
        for data,color in [(background,'#999'),(matched,'#1467b3')]:
            points=' '.join(f'{sx(i):.2f},{sy(y):.2f}' for i,y in enumerate(data));svg.append(f'<polyline fill="none" stroke="{color}" stroke-width="1.6" points="{points}"/>')
        svg.extend([f'<text x="40" y="{h-4}" font-size="12">video-track time: 0 to {(len(ar)-1)*.04:.2f}s · shaded nominal FA phrase bounds</text>','</svg>']);(OUT/f'curve_{r["id"]}.svg').write_text(''.join(svg),encoding='utf-8')
        page.append(f'<h2>{r["id"]}</h2><img src="curve_{r["id"]}.svg">')
        for z in rr:
            lo=max(0,z['nominal_bounds'][0]-.3);hi=min(duration,z['nominal_bounds'][1]+.3);stem=f'{r["id"]}_p{z["phrase_index"]+1}';wav=clips/(stem+'.wav');mp4=clips/(stem+'.mp4')
            with wave.open(str(wav),'wb') as ww:ww.setparams((1,2,16000,0,'NONE','not compressed'));ww.writeframes(canonical[round(lo*16000):round(hi*16000)].tobytes())
            cmd=['ffmpeg','-nostdin','-v','error','-n','-ss',str(lo),'-i',r['video_path'],'-ss',str(lo),'-i',r['source'],'-t',str(hi-lo),'-map','0:v:0','-map','1:a:0','-c:v','libx264','-preset','veryfast','-crf','20','-c:a','aac','-b:a','96k','-shortest',str(mp4)]
            tt=time.time();p=subprocess.run(cmd,capture_output=True,text=True);receipts.append({'id':stem,'bounds':[lo,hi],'command':cmd,'returncode':p.returncode,'elapsed_s':time.time()-tt,'stderr':p.stderr,'wav_sha256':sha(wav),'mp4_sha256':sha(mp4) if p.returncode==0 else None});assert p.returncode==0
            value='NA' if z['contrast'] is None else f'{z["contrast"]:.3f}'
            page.append(f'<article><b>{html.escape(z["phrase"])}</b> · local contrast {value} · {z["window_count"]} windows · {z["content_status"]} · {z["status"]}<br><audio controls src="clips/{stem}.wav"></audio><video controls src="clips/{stem}.mp4"></video></article>')
        print('presentation '+r['id'],flush=True)
    (OUT/'index.html').write_text('\n'.join(page),encoding='utf-8');save(OUT/'presentation_receipts.json',receipts)
    save(OUT/'final_input_sha.json',{'audio':[{ 'source':r['source'],'before':r['sha256'],'after':sha(r['source']),'unchanged':sha(r['source'])==r['sha256']} for r in rows]})
    shutil.copyfile(Path(__file__),OUT/'analysis_source.py')

def finalize():
    comparisons=read(OUT/'comparisons.json');metrics=read(OUT/'local_metrics.json');inputs=read(OUT/'input_manifest.json')['audio'];alignment=read(OUT/'alignment.json')['audio']
    test=ROOT/'tests/experiments/test_qwen_phrase_local_20261024.py';shutil.copyfile(test,OUT/'test_analysis.py');p=subprocess.run([sys.executable,str(test)],capture_output=True,text=True);save(OUT/'tests_receipt.json',{'command':[sys.executable,str(test)],'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'pytest_limit':'local default Python has no numpy; remote analysis venv has no pytest. Same three test functions executed directly without package/environment changes.'});assert p.returncode==0
    incomplete=[{k:z[k] for k in ['id','phrase','variant','window_count','status']} for z in metrics if z['status']!='OK']
    content=[{'id':z['id'],'ASR_text':z['asr_text']['normalized'],'reference':z['given_text']['normalized'],'diff_opcodes':z['transcript_diff_opcodes']} for z in alignment if not z['transcript_exact_match']]
    save(OUT/'limitations.json',{'incomplete_audio_phrase_configurations':incomplete,'ASR_content_uncertainty':content,'interpretation':'ASR mismatch is not confirmed real speech error. Given-text FA diagnostics retained. No human content or boundary verification performed.'})
    summary={'audio':28,'phrases':9,'phrase_utterances':63,'audio_phrase_configurations':len(metrics),'comparisons':len(comparisons),'draw_contrasts':len(comparisons)*5*3,'ASR_uncertain_audio':len(content),'incomplete_audio_phrase_configurations':len(incomplete),'nominal_INCOMPLETE_comparisons':sum(z['nominal_direction']=='INCOMPLETE' for z in comparisons),'nominal_same_sign_comparisons':sum(z['nominal_direction'] in ['positive','negative'] for z in comparisons),'boundary_robust_same_sign_ASR_no_mismatch':sum(z['boundary_robust'] and z['content_status']=='ASR_exact_for_phrase' for z in comparisons),'WAV_clips':len(list((OUT/'clips').glob('*.wav'))),'MP4_clips':len(list((OUT/'clips').glob('*.mp4'))),'source_curves':len(list(OUT.glob('curve_*.svg'))),'final_input_SHA_unchanged':all(z['unchanged'] for z in read(OUT/'final_input_sha.json')['audio'])}
    save(OUT/'summary.json',summary)
    report='''# 四句话的对应短语局部 SyncNet 诊断

本次只分析冻结的4来源、28已有canonical音频和原生视频，共9个预先按文字划分的短语。自然语音每来源1条，Qwen0.6B/1.7B各3draw。没有训练、新TTS/TFG生成、SyncNet特征推理、拉伸音频或重新归一化。全部63个短语发声均提供0.3秒上下文试听与视频，入口 index.html；没有按局部分数选择展示。

## 观察结果

L17−L06在3draw与5种边界配置均同方向的短语：S0002“抑制作用最大的限购”为正（+1.743/+1.864/+0.332）；S0014“折合楼面价”为负（−2.441/−2.052/−0.392），“三点八万元”为负（−0.622/−0.732/−1.497）。其余可估计短语的模型间对比有混合方向。

S0014这两个短语虽然1.7B低于0.6B，两模型相对自然音频仍在3draw及5边界配置均为正。因此模型间下降不能写成TTS低于自然。S0002第二短语0.6B相对自然稳定为负，而1.7B相对自然方向混合。整句正样本也可包含下降的短语。各条的匹配距离和背景距离同时保留；局部对比度上升本身不等于原时刻匹配距离改善。

完整27行、每行三draw与支持数量见 comparison_table.md；comparisons.json保留全部5边界配置与三个端点。315个音频×短语×边界配置中11项不足5窗，全部为S0014“每平方米”：L17draw1五配置均不足（名义2窗），其余L17draw2/3及L06draw3在右移/收缩配置各4窗。因而含L17的两个名义比较不完整；L06−N虽然名义三draw均正，边界稳健性不完整。没有改短语、lag集合或最低窗数来挽救。

## 内容与边界限制

新补齐20条ASR/fa，复用已有8条；28条fa的字符数量、返回文字、单调及音频范围结构检查通过。ASR有3条与给定文字不同：S0002自然音频识别为“而对面楼市成交…”；S0007自然识别“长袖上午”、L06draw3识别“长袖上舞”。这可能是ASR识别错误，不能确认原语音说错。对应第一短语及所有相关比较标为content_uncertain_ASR_mismatch，给定文字FA数值完整保留为诊断；自然音频的不确定性传播至两个含N的比较。其余短语只能称自动检查未发现内容不一致。未进行人工听辨或人工边界验证。

## 指标与时间映射

每原生视频先固定原官方整track最优lag；短语内不重新找最优。对完全相同的有效短语窗先逐lag平均，再取31lag均值的中位数，减固定lag的平均匹配距离。它不是官方短片段Sync-C，也不是整句得分的可加贡献。另报matched_distance与median_background，距离越低表示该匹配距离较小。负lag可以取短语外上下文，但每个lag仍要求实际音频支持。

原activesd为逐视频窗×31lag，列k对应音频feature j=i+k−15，官方offset=15−k。读取真实track.frame；视频RF含5帧（200ms），音频输入20个MFCC帧（25ms窗口、10ms步长）对应215ms，另需前一采样点用于预加重。要求matched两侧RF全部在对应短语内，且所有31lag均有真实feature/crop/canonical采样支持，排除零padding、MFCC尾padding及原生视频结束后重复帧。跨短语窗完整单列。5配置为名义、整体左/右40ms、两端收缩/扩张40ms；每配置最少5窗，接近零阈值1e−6。

28音轨均通过经验性波形相关绑定到canonical：crop零点对应canonical −0.0690625秒；全局相关最低0.9853，84个固定250ms片块全部峰值1105样本延迟。S0002_L17draw1尾部片块相关0.797，其余83块0.957–0.9998。MP3处理改变波形，因此这支持时间映射而非波形身份或人工音素时间准确性。详细证据在input_manifest.json与time_binding_supplement.json，视频frame/PTS、所有输入SHA均保留。

## 验证与执行

28个原canonical WAV、原视频、score、activesd的SHA核对通过；原矩阵整句C/D重建最大舍入误差0.000496056（阈值0.00051），28原音频运行后SHA不变。父代理独立读取原矩阵及FA边界，315配置支持逐行一致，912局部标量及1155配对标量最大误差0，见parent_raw_array_audit.json（由父侧生成）。三项RF/lag/padding/边界与缺失保留测试通过。

20新ASR/fa分8/8/4三批，CPU、两线程、offline，各进程exit0，约15.35/15.37/13.34秒；进程采样与命令在alignment_receipts.json。63个展示MP4均ffmpeg exit0；WAV直接切取原PCM样本，展示MP4仅用已有视频和canonical音轨，不参与评分。所有矩阵分析为CPU无推理，最后没有遗留本任务进程。

attempt01在读取C_protocol.rows内不存在的transcript字段时失败，仅写冻结protocol，未计算局部结果；改从samples读取，attempt02完成。没有按结果重试、策略寻优或变更冻结设计。测试的本机默认Python缺numpy、远端venv缺pytest，未改环境，直接调用同一测试文件三项测试函数通过。低支持、ASR不确定和所有反向数值均保留。

本结果是事后选择的四个案例的评价器局部定位，只能称观察到3draw方向一致。自然每来源只有一次，不当作三个独立样本；无显著性、泛化、人工同步质量或上游因果结论。
'''
    (OUT/'report.md').write_text(report,encoding='utf-8');shutil.copyfile(Path(__file__),OUT/'analysis_source.py')
def preflight():
    OUT.mkdir(exist_ok=False)
    design=ROOT/'runs/qwen_phrase_local_design_20261024_attempt01/protocol.json'
    assert sha(design)=='601f1fabcc48a184b6d64976ebd639cc1674160fb5b0096d5f6179121e8a1646'
    protocol=read(design);save(OUT/'protocol.json',protocol)
    old=ROOT/'runs/qwen17_level_comparison10_20261023_attempt02'
    new=ROOT/'runs/qwen_prediction_mechanism_20261023_attempt02'
    transcripts={r['sample_id']:r['transcript'] for r in read(old/'protocol.json')['samples']}
    for row in read(new/'C_protocol.json')['samples']:
        if row['sample_id'] in protocol['phrases']:
            transcripts[row['sample_id']]=row['transcript']
    text={sid:norm(transcripts[sid]) for sid in protocol['phrases']}
    for sid,v in text.items(): assert v['normalized']==''.join(protocol['phrases'][sid]),(sid,v)
    save(OUT/'frozen_text.json',text)
    candidates=[]
    for x in read(old/'comparison_index.json')['records']:
        if x['sample_id'] in protocol['scope']['source_ids'][:2] and x['level']=='norm' and x['model'] in ['N','L06','L17']:
            candidates.append((x['sample_id'],x['model'],x['draw_index'],x['score_path'],x['score_sha256']))
    plan=read(new/'C_downstream_plan.json'); scores={x['id']:(x['score_path'],x['score_sha']) for x in plan['reuse']}
    for job in plan['jobs']:
        if job['policy']!='baseline': continue
        j=Path(job['directory'])
        for x in read(j/'unit_mapping.json'):
            p=j/'official/scores'/f'{x["sample_id"]}_{x["role"]}.json';scores[x['id']]=(str(p),sha(p))
    for sid in protocol['scope']['source_ids'][2:]:
        p=ROOT/'runs/qwen_nrt_official_rescore_20261023_attempt01/scores'/f'{sid}_N.json'
        candidates.append((sid,'N',0,str(p),sha(p)))
        for m in ['L06','L17']:
            for d in [1,2,3]:
                p,h=scores[f'{sid}_{m}_baseline_{d}'];candidates.append((sid,m,d,p,h))
    assert len(candidates)==28
    rows=[]
    for sid,m,d,p,h in candidates:
        assert sha(p)==h
        s=read(p); a=Path(s['canonical_audio_path']);v=Path(s['video_path']);act=Path(s['activesd_path'])
        assert sha(a)==s['canonical_audio_sha256'];assert sha(v)==s['source_video_sha256'];assert sha(act)==s['activesd_sha256']
        tp=act.parent/'tracks.pckl'; tracks=pickle.loads(tp.read_bytes()); arrays=pickle.loads(act.read_bytes()); assert len(tracks)==len(arrays)==len(s['tracks'])
        canonical=pcm(a); pre=Path(s['tracks'][0]['metadata']['crop_file']).parents[2]/'pyavi'/s['reference']/'audio.wav'
        prebind=binding(pcm(pre),canonical,0.)
        pts=[float(x['best_effort_timestamp_time']) for x in probe(v)['frames']]
        assert all(abs(t-i/25)<1e-6 for i,t in enumerate(pts))
        ts=[]
        for k,(track,ar,official) in enumerate(zip(tracks,arrays,s['tracks'])):
            f=np.asarray(track['track']['frame'],int); ar=np.asarray(ar);assert ar.ndim==2 and ar.shape[1]==31;assert np.all(np.diff(f)==1)
            mu=ar.astype(float).mean(0);idx=int(mu.argmin()); c=float(np.median(mu)-mu[idx]); dd=float(mu[idx]);assert max(abs(c-official['sync_c']),abs(dd-official['sync_d']))<=.00051
            crop=Path(official['metadata']['crop_file']); y=pcm(crop); bind=binding(y,canonical,f[0]/25)
            assert bind['correlation']>.95
            assert ar.shape[0]==min(len(f),len(y)//640)-5
            ts.append({'track_index':k,'frames':f.tolist(),'array_shape':list(ar.shape),'fixed_best_column':idx,'official_offset':int(15-idx),'official_C':official['sync_c'],'official_D':official['sync_d'],'reconstructed_C':c,'reconstructed_D':dd,'crop_path':str(crop),'crop_sha256':sha(crop),'crop_audio_samples':len(y),'audio_binding':bind,'crop_source_frames_valid':[bool(z<len(pts)) for z in f]})
        row={'id':f'{sid}_{m}_{d}','sample_id':sid,'model':m,'draw':d,'source':str(a),'sha256':sha(a),'transcript':transcripts[sid],'score_path':p,'score_sha256':h,'video_path':str(v),'video_sha256':sha(v),'canonical_audio_samples':len(canonical),'video_frame_count':len(pts),'video_pts':pts,'activesd_path':str(act),'activesd_sha256':sha(act),'tracks_path':str(tp),'tracks_sha256':sha(tp),'pretrack_audio_path':str(pre),'pretrack_audio_sha256':sha(pre),'pretrack_audio_binding':prebind,'tracks':ts}
        rows.append(row); print(json.dumps({'id':row['id'],'shapes':[t['array_shape'] for t in ts],'delays':[t['audio_binding']['delay_s'] for t in ts]},ensure_ascii=False),flush=True)
    save(OUT/'input_manifest.json',{'audio':rows,'scope':protocol['scope']})
    source=ROOT/'third_party/syncnet_python'
    save(OUT/'preflight.json',{'status':'READY_FOR_PARENT_TIME_MAPPING_REVIEW','audio_count':len(rows),'source_pins':{str(source/f):sha(source/f) for f in ['SyncNetInstance.py','run_pipeline.py','run_syncnet.py']},'formula':{'lag':'j=i+k-15; offset=15-k','video_RF':'track.frame[i] through track.frame[i+4], half-open [frame[i]/25,(frame[i+4]+1)/25)','audio_RF':'crop_zero_canonical + [0.04*j, 0.04*j+0.215); MFCC 20 frames, 25ms window/10ms step, includes preemphasis preceding sample','padding':'require 0<=i-15 and i+15<feature_rows; also full canonical sample support and native video frames; matched audio and video RF contained in phrase'},'missing_syncnet_inference':False})
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('phase',choices=['preflight','align','supplement','effects','present','finalize']);a=ap.parse_args();start=time.time();rc=0
    try: globals()[a.phase]()
    except BaseException:rc=1;raise
    finally:
        if OUT.exists():save(OUT/f'{a.phase}_process_receipt.json',{'phase':a.phase,'pid':os.getpid(),'returncode':rc,'start_unix':start,'end_unix':time.time(),'elapsed_s':time.time()-start,'inference':'none' if a.phase!='align' else 'CPU only; subprocess receipts include monitoring'})
        if a.phase=='finalize' and rc==0:save(OUT/'final_manifest.json',{'files':[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='final_manifest.json'],'self_excluded':True})
