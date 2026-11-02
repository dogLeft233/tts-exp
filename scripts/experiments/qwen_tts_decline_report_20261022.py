"""Final posthoc diagnostic report and inspectable audio packet, no primary edits."""
from pathlib import Path
import json
import csv
import hashlib
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
RUN=ROOT/'runs/qwen_tts_decline_audit_20261022_attempt01'


def read(p):return json.loads(p.read_text())
def fmt(x):return f'{x:+.3f}'
def ci(x):return '['+', '.join(fmt(y) for y in x)+']'


def main():
    influence=read(RUN/'influence.json');audio=read(RUN/'audio_diagnostics.json')['records'];ASR=read(RUN/'asr_attempt02/results.json')['records']
    vad=read(RUN/'cap_vad_sensitivity/results.json')['records'];cap=read(RUN/'posthoc_cap_intervals.json')
    assert len(audio)==len(ASR)==50 and len(vad)==5
    ab={(r['sample_id'],r['role']):r for r in audio};ar={(r['sample_id'],r['role']):r for r in ASR}
    out=RUN/'figures';out.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),constrained_layout=True)
    for ax,policy,label in zip(axes,['common','native_full'],['Common valid support','Full native valid support']):
        rr=influence[policy]['rows'];x=np.arange(len(rr));y=np.array([r['T_minus_N'] for r in rr]);z=np.array([r['T_minus_R'] for r in rr])
        ax.bar(x-.18,y,.36,label='T - N',color='#2563eb');ax.bar(x+.18,z,.36,label='T - R',color='#ea580c')
        ax.axhline(0,color='black',lw=.7);ax.set_xticks(x,[r['speaker_id'][1:] for r in rr],rotation=45);ax.set_title(label);ax.set_xlabel('Speaker ID');ax.set_ylabel('Paired Sync-C difference');ax.legend(fontsize=8)
    fig.suptitle('Posthoc influence audit: all 10 speakers retained')
    fig.savefig(out/'all_speakers.png',dpi=160);plt.close(fig)
    sid='BAC009S0011W0121';j=ab[sid,'T3'];x,sr=sf.read(j['path']);frames=np.lib.stride_tricks.sliding_window_view(x,320)[::160];en=np.sqrt(np.mean(frames*frames,axis=1))
    fig,ax=plt.subplots(figsize=(11,3),layout='constrained');ax.plot(np.arange(len(en))*.01,en,lw=.6,color='#64748b')
    v=next(r for r in vad if r['role']=='T3')
    for span in v['spans_s']:ax.axvspan(span['start'],span['end'],color='#22c55e',alpha=.25)
    ax.set(xlabel='Original generated-audio time (s)',ylabel='20 ms RMS',title='S0011 T3: 40.88 s total; VAD detects 3.56 s of speech (green, proxy)')
    fig.savefig(out/'capped_audio_energy.png',dpi=160);plt.close(fig)
    packet=RUN/'listening_packet';packet.mkdir(exist_ok=False);clips=[]
    for sid,role,start,end,name in [('BAC009S0011W0121','N',0,None,'S0011_N.wav'),('BAC009S0011W0121','T3',0,6,'S0011_T3_first6s.wav'),
                                    ('BAC009S0011W0121','T3',12,18,'S0011_T3_tail12_18s.wav'),('BAC009S0010W0121','N',0,None,'S0010_N.wav'),('BAC009S0010W0121','T1',0,None,'S0010_T1.wav')]:
        source=ab[sid,role];pcm,fs=sf.read(source['path'],dtype='int16');segment=pcm[int(start*fs):None if end is None else int(end*fs)]
        dest=packet/name;sf.write(dest,segment,fs,subtype='PCM_16');check,cs=sf.read(dest,dtype='int16');assert cs==fs and np.array_equal(check,segment)
        clips.append(dict(sample_id=sid,role=role,start_s=start,end_s=end,path=str(dest),sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),source_sha256=source['sha256'],no_new_gain_or_score=True))
    (packet/'manifest.json').write_text(json.dumps(dict(status='NOT_HUMAN_LISTENED_OR_ANNOTATED',selection='posthoc decline and generation anomaly inspection; not blind',records=clips),indent=2)+'\n')
    with (RUN/'all50_audio_and_ASR.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','role','transcript','ASR_hypothesis','CER_raw_format_sensitive','duration_s','duration_ratio','quiet_fraction_minus40','EOS'])
        for j in audio:
            a=ar[j['sample_id'],j['role']];w.writerow([j['sample_id'],j['role'],j['transcript'],a['normalized'],a['CER'],j['duration_s'],j['duration_ratio_to_natural'],j['quiet_fraction_below_minus40dBFS'],j.get('EOS')])
    lines=['## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent',
           '- Origin Mode: validate with bounded independent local ASR diagnostics',
           '- Origin Date: 2026-10-22',
           '- Verification Status: ANALYZED_WITH_EXECUTED_ASR_PROXY_AND_NUMERICAL_READBACK; not human or perceptual validation',
           '- Version Label: decline_audit_v1','',
           '# 完整TTS下降样本、生成质量与少数样本影响诊断','',
           '**结论：少数下降确实拉低平均值，特别是S0011无EOS长输出影响整段评价；但不能把其余下降都归因于0.6B模型把文字读坏。去掉该异常也没有建立稳定TTS优势。**','',
           '这是看过上一批分数后由用户提出的事后诊断；原主要结果和全部30次生成保持封存。下面的删人/删异常计算仅量化影响，不是新主要分析、不能依据它选择保留的样本。没有新TTS/TFG或训练；独立Whisper-small仅做CPU转写。','',
           '## 谁下降最多','',
           '主要共同窗口、每人先平均3种子：','',
           '| 来源 | 目标句子 | T−N | T−R | N原分数 | T平均分数 |','|---|---|---:|---:|---:|---:|']
    for row in sorted(influence['common']['rows'],key=lambda r:r['T_minus_N']):
        text=''.join(ab[row['sample_id'],'N']['transcript'].split())
        lines.append(f"| {row['sample_id']} | {text} | {fmt(row['T_minus_N'])} | {fmt(row['T_minus_R'])} | {row['N']:.3f} | {row['T']:.3f} |")
    lines+=['','最负的单次共同窗口输出是S0009/T2：T−N −1.381、T−R −1.024；该人的另两次方向不同，说明不能只看说话人均值。S0010与S0011三个种子在共同窗口下都比N低。','',
            'S0010/S0011/S0006是按T−N人均降幅最大的三人，也恰好是N共同窗口分数最高的三人。这是本队列的描述性共同点；差值里本身含−N，且来源已按结果排序，不能从这种关系证明天花板、均值回归或小模型质量原因。','',
            '![全部说话人变化](figures/all_speakers.png)','',
            '## 少数下降是否影响整体','',
            '| 支持 | 保留全部T−N / T−R | 只在诊断中略去S0011/T3后 | 该诊断的95%区间，T−N / T−R |',
            '|---|---|---|---|']
    for policy,label in [('common','共同窗口'),('native_full','完整有效时段'),('equal_count_spread','等数均匀覆盖')]:
        z=influence[policy]['summaries'];c=cap[policy]
        lines.append(f"| {label} | {fmt(z['T_minus_N']['mean'])} / {fmt(z['T_minus_R']['mean'])} | {fmt(c['T_minus_N']['mean'])} / {fmt(c['T_minus_R']['mean'])} | {ci(c['T_minus_N']['CI95'])} / {ci(c['T_minus_R']['CI95'])} |")
    lines+=['','以上保持原支持不变，只将S0011的T均值由3个draw改为2个，其余人不动；10人仍等权。全部诊断区间跨零，不能据此确认优势。共同窗口均值变化只有约+0.005；长异常解释的是整段均值的负向贡献，不能解释全部主要分析不确定性。','',
            '对每一个人都做过逐个移除：','',
            '| 支持/比较 | 原均值 | 移除任意一人后的均值范围 |','|---|---:|---|']
    for policy in ['common','native_full','equal_count_spread']:
        for key in ['T_minus_N','T_minus_R']:
            s=influence[policy]['summaries'][key];vals=[r['remaining_mean'] for r in s['leave_one_out']]
            lines.append(f"| {policy} / {key} | {fmt(s['mean'])} | {fmt(min(vals))} 至 {fmt(max(vals))} |")
    lines+=['','共同窗口下，移除任意单人都不会改变两项均值的正方向；最大正贡献者S0007也有影响，不能只审视负贡献。完整时段移除S0011后T−N/T−R约为+0.024/+0.062；影响表的所有10人结果保存在influence.json。按分数挑选移除S0010和S0011等组合只能作事后描述，不能升级为筛样后的确认性证据。','',
            '## 是不是本地小TTS生成不好','',
            '50条N/R/T1/T2/T3均用同一独立Whisper-small、CPU int8、beam5、temperature0转写，未给目标文字提示；文本经NFKC、简繁转换及去标点/空格。CER只作内容代理，不是自然度、音色或视觉嘴型真值；数字书写差异仍计为错误。','',
            '- S0010：最差的T1和T2均完整识别为“高周转就是牺牲利润”，CER=0；T3识别“高周软就是牺牲利润”，差1字。分低不能直接等同于整句读坏。',
            '- S0011：正常T1/T2在无VAD转写中均全句正确，仍在共同窗口下降；原N/R反而将“九龙仓”识别为“九龙桑”。这不支持把该人的所有下降都归给文字错误。',
            '- S0006：N/R及T2/T3同样识别为“同比增长22.6%”；原始CER=0.667主要混入数字写法。T1首字“同”被识别为“筒”，但它的共同窗口得分反而较高。',
            '- S0009/T2：识别为“产业标杆全显升级”，与R相同（CER=0.125），但T−R约−1.024；T1全句正确。内容可识别性与Sync-C变化并不一一对应。',
            '- S0005：T全部识别“10年后就是15万亿元”，N/R写成“十年后就是十五万亿元”；原始CER=0.400不能算作已经证实的口误。','',
            '时长/停顿也没有统一方向：S0010三次T为3.36/2.56/2.88秒、N为3.04秒；S0006 T较短，但不少上升者也更短。S0010弱能量帧比例整体偏多，S0006部分种子更少。所有50条均无饱和削波、共同RMS已配平。不能从当前10人建立单一声学成因。','',
            '### S0011/T3：更像没有正常结束','',
            '原N约5.02秒；T3为40.88秒/511新编码步，无EOS。无VAD ASR先正确读出目标句，随后反复识别无关句，原始CER=11.5；这不能直接写成音频真的反复说了那些话。','',
            f"同一源PCM补做明确标记的事后Silero-VAD敏感性（同S0011的N/R/T1/T2/T3共5条），T3只检测{v['VAD_speech_s']:.2f}秒语音，6秒后检测到的语音为{v['VAD_after_6s_speech_s']:.2f}秒；筛选语音段后转写为“{v['raw']}”，CER={v['CER']:.3f}。这支持‘开头读句、后面拖出较长未检出语音的尾部’的工程诊断，不能据ASR幻觉指认真实重复讲话。未人工试听，非语音尾部是否为噪声/异常声学纹理尚待听辨。",'',
            '![异常长音频能量与VAD](figures/capped_audio_energy.png)','',
            '它没有正常结束这一点有token元数据与时长支持，足以作为生成可靠性问题；但没有更大TTS或其它解码配置的匹配对照，不能证明根因就是0.6B参数规模。声音自然度/韵律是否差，需要独立人工试听或合格质量测量，不由本次Sync-C/CER代替。','',
            '## 转写与试听入口','',
            '完整50条结果见all50_audio_and_ASR.csv；无VAD原始分段见asr_attempt02/asr/，S0011五条VAD敏感性见cap_vad_sensitivity/。下面试听包选择于观察结果之后，非盲评、未有人类标注；片段没有重新归一化，也未用于重评分。','']
    for item in clips:
        rel='listening_packet/'+Path(item['path']).name
        lines.append(f"- [{Path(item['path']).name}]({rel})：来源{item['sample_id']}/{item['role']}，原时钟{item['start_s']}–{item['end_s'] if item['end_s'] is not None else '结尾'}秒。")
    lines+=['','## 验证与边界','',
            '6项CER/文本规范化测试通过；统计影响量与原封存分数逐人逐种子核对。原封存两run、全部源PCM哈希与ASR输入绑定另作读回复核；不冒充重新生成或人工内容认证。独立ASR没有使用SyncNet或TTS权重。','',
            '11/11统计误读复查：报告全部支持与全部人/种子（Simpson）；单位仍为10人而非50音频（生态）；历史开发与按降幅挑选诊断对象（Berkson）；不按结果改主要样本/支持（collider）；非筛查基率任务（base rate不适用）；高N与差值关系存在数学耦合，未认定均值回归原因；无删除主异常/幸存者筛选；所有探索与组合为事后而非确认性发现（look-elsewhere/forking）；ASR/声学代理不作TTS规模因果或感知证据；处理在生成/评分之后作诊断，未据此倒推学习因果（reverse causality）。','',
            'ASR首次工程尝试因PyAV metadata_errors参数不兼容，在任何转写输出之前失败；日志/源码保留。attempt02用标准wave读取完全相同的16k PCM，float32/32768数组直接输入，50条完成；不是换模型、改解码参数或按分数重采样。ASR-only隔离环境依赖安装失败记录保留，原TTS/Wav2Lip/SyncNet环境未更改。','',
            '本轮仅诊断与报告，没有训练、重新TTS合成或新增TFG评分；原结论保持。']
    (RUN/'report.md').write_text('\n'.join(lines)+'\n')
    print('REPORT_COMPLETE',len(ASR),'ASR rows',flush=True)


if __name__=='__main__':main()
