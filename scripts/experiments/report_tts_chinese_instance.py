"""Compact scientific report and standalone plots for the frozen experiment."""
import json
import shutil
from pathlib import Path
import numpy as np
from scripts.experiments.tts_chinese_instance import BASE,ROOT,read,write,sha

def fmt(x):return f"{x['speaker_mean']:+.3f} [{x['speaker_ci99'][0]:+.3f}, {x['speaker_ci99'][1]:+.3f}]"

def main():
    p=read(BASE/'protocol.json');native=read(BASE/'native.json');cal=read(BASE/'calibration.json');k=cal['k'];quality=read(BASE/'quality/summary.json')
    mainfolder=BASE/'mechanism'/f'primary_k{k}';analysis=read(mainfolder/'analysis.json');support=read(mainfolder/'support.json');valid=[s for s in support if s['eligible']]
    lines=['# 中文同后端实例匹配与同协议原生同步','',
           '全部100条可核验输入，旧26cal / 74eval（15speaker），不是新speaker确认。主脸image3；固定15条每speaker首eval做image6/9。N自然，C历史云Qwen，Q1/Q2严格本地Qwen同文本同参考seed42/43。',
           '', '## 生成状态与质量', '',
           '原批首次lazy CUDA graph初始化在seed作用域内额外消耗RNG。固定前缀逐PCM复现；正常暖即时重复一致。恢复批先对固定首参考显式预热，再进入每调用seed作用域，全量新生成。不是持续KV污染结论；不因cold输出恰好更好而选择模式。', '',
           'a1_005/Q1在统一暖语义仍156.8秒重复，1960 codec steps+88 prefill达到2048上下文容量而非EOS。原样保留，不换seed、不缩短上限、不裁尾；native主分母不按ASR筛选。', '',
           '| 臂 | 完成数 | 时长中位数(s) | CER均值 | CER中位数 |', '|---|---:|---:|---:|---:|']
    for a,z in quality['distributions'].items():
        lines.append(f"| {a} | {z['n']} | {z['duration']['quantiles']['median']:.3f} | {z['cer']['mean']:.3f} | {z['cer']['quantiles']['median']:.3f} |")
    terms={a:{} for a in ('Q1','Q2')}
    for r in p['rows']:
        for a in terms:
            z=read(BASE/'tts'/r['id']/(a+'.json'));stop=z.get('termination',{}).get('stop_reason','missing');terms[a][stop]=terms[a].get(stop,0)+1
    lines.extend(['',f"停止原因：`{json.dumps(terms,ensure_ascii=False)}`。首条同seed重放：`{json.dumps(quality['same_seed_replay'],ensure_ascii=False)}`。",'',
                  'ASR固定Whisper-small CPU int8、中文、beam5、无prompt/VAD；NFKC、小写、OpenCC繁简转换、去空白标点。主机制要求N/Q1/Q2 CER≤0.10及Q1/Q2 EOS，再过原音素/event门；ASR只是文本忠实度代理。C停止元数据未知，只用原成功receipt/可解码PCM、CER和event门。', '',
                  '## 独立校准', '',f"共同k={k}；N/Q1/Q2 arm中位数={{{', '.join(a+': '+str(cal['medians'][a]) for a in ('N','Q1','Q2'))}}}；臂间≤1帧门：{cal['transfer_gate']}。C中位数={cal['medians'].get('C')}，未用于选k。额外脸沿用主k，有跨脸校准限制。", '',
                  '## 同协议 native', '', '以下均为speaker等权均值与99%CI；完整95%CI在JSON。', '', '| guard | GQ | GC | GQ−GC |', '|---|---|---|---|'])
    for g in ('20','0','15'):
        z=native['3']['analysis'];lines.append('| '+g+' | '+' | '.join(fmt(z[g+'_'+m]) for m in ('GQ','GC','GQ-GC'))+' |')
    short=[]
    for r in p['rows']:
        if r['split']!='evaluation':continue
        for a in ('N','C','Q1','Q2'):
            f=BASE/'scores/3'/r['id']/(a+'.json')
            if '20' not in read(f)['metrics']:short.append(r['id']+'/'+a)
    lines.extend(['',f"74条eval中，上表所有guard使用同一guard20可评分四臂共同支持n={native['3']['analysis']['20_GQ']['n']}。各自最大guard20可评分支持：GQ n={native['3']['maximal_pair_analysis']['20_GQ']['n']}，{fmt(native['3']['maximal_pair_analysis']['20_GQ'])}；GC n={native['3']['maximal_pair_analysis']['20_GC']['n']}，{fmt(native['3']['maximal_pair_analysis']['20_GC'])}。缺少guard20所需>40行的臂：{', '.join(short)}。不按ASR删native；guard0/15主表保持guard20支持，避免短句分母同时变化。"])
    lines.extend(['','GQ先在条内平均Q1/Q2，再减N；GC=C−N。C与Q含provider/模型/注册/日期等差异，不能解释为单一模型架构效应。','',
                  '## 同实例与跨条件 event 匹配', '',
                  'q为相同音素实例/相位和相同负donor IDs上的平均负距离−正距离，区别于官方Sync-C。STT=.5(q11+q22−q12−q21)，SNT=.25[(qNN+q11−qN1−q1N)+(qNN+q22−qN2−q2N)]。','',
                  '| k | 合格条/说话人 | STT raw margin | SNT raw margin | SNT−STT |','|---|---|---|---|---|'])
    for kk in (k,k-1,k+1):
        f=BASE/'mechanism'/f'primary_k{kk}';ss=read(f/'support.json');vv=[s for s in ss if s['eligible']];z=read(f/'analysis.json')['3']['raw']
        lines.append(f"| {kk} | {len(vv)}/{len({s['speaker'] for s in vv})} | "+' | '.join(fmt(z['margin_'+m]) if 'margin_'+m in z else '不可判' for m in ('STT','SNT','SNT-STT'))+' |')
    lines.extend(['','| 主k敏感性 | STT | SNT | SNT−STT |','|---|---|---|---|'])
    for geo,metric in [('raw','rank'),('unit','margin'),('unit','rank'),('raw','positive'),('raw','negative')]:
        z=analysis['3'][geo];lines.append('| '+geo+' '+metric+' | '+' | '.join(fmt(z[metric+'_'+m]) if metric+'_'+m in z else '不可判' for m in ('STT','SNT','SNT-STT'))+' |')
    lines.extend(['','### 此阶段新增解释对比','','Tdiag=(q11+q22)/2，Toffdiag=(q12+q21)/2；两者相减恒等于STT。此项在评分启动后、读取数值结果与正式分析前由理论端追加，不冒充原预注册主端点。','','| k / 几何 / 指标 | Tdiag−NN | Toffdiag−NN |','|---|---|---|'])
    for kk in (k,k-1,k+1):
        zz=read(BASE/'mechanism'/f'primary_k{kk}'/'analysis.json')['3']
        for geo,metric in [('raw','margin'),('raw','rank'),('unit','margin'),('unit','rank')]:
            z=zz[geo];lines.append(f'| {kk} / {geo} / {metric} | '+' | '.join(fmt(z[metric+'_'+m]) if metric+'_'+m in z else '不可判' for m in ('Tdiag-NN','Toffdiag-NN'))+' |')
    lines.extend(['','## 内容与event支持损失及同支持native','','| 支持规则 | eligible条/speaker | event-only条 | GQ同支持 | GC同支持 |','|---|---|---:|---|---|'])
    for label in ('primary','mfa_only','strict','cloud4'):
        f=BASE/'mechanism'/f'{label}_k{k}';ss=read(f/'support.json');vv=[s for s in ss if s['eligible']];z=read(f/'native_on_support.json')
        lines.append(f"| {label} | {len(vv)}/{len({s['speaker'] for s in vv})} | {sum(s.get('mfa_event_eligible',False) for s in ss)} | "+' | '.join(fmt(z['20_'+m]) if '20_'+m in z else '不可判' for m in ('GQ','GC'))+' |')
    audit=read(BASE/'support_audit.json');qd=audit['quality_detail']
    manip=read(BASE/'manipulation.json')
    lines.extend(['',f"操纵检查：100/100 Q1/Q2 PCM不同，同seed重放PCM完全相同。主支持上Q1/Q2 SyncNet音频embedding平均距离{manip['event_analysis']['audio_embedding_distance']['speaker_mean']:.3f}，相对距离{manip['event_analysis']['audio_embedding_relative']['speaker_mean']:.3f}；MFCC/mel相对差见manipulation.json。Q1/Q2音素时长平均绝对差{1000*manip['event_analysis']['query_phone_duration_abs_difference']['speaker_mean']:.1f}ms，N/Q为{1000*manip['event_analysis']['NQ_phone_duration_abs_difference']['speaker_mean']:.1f}ms，未据此调整结果。",'',
                  f"k±1敏感性共同保留{len(audit['lag_common']['ids'])}条，三k共同query数{sum(audit['lag_common']['query_counts'].values())}；各k自己的支持与query仍完整列出，没有挑选敏感性赢家。"])
    lines.extend(['',f"74eval各臂ASR CER门失败数：{ {a:z['asr_failure_count'] for a,z in qd['by_arm'].items()} }。N/Q1/Q2失败交集组合：`{json.dumps(qd['asr_failure_combinations'])}`。共同CER/EOS门保留{len(audit['primary']['independent_quality_pass_ids'])}条/{audit['primary']['independent_quality_pass_speakers']}speaker。",'',
                  'ASR门失败只说明固定识别器与参考转写不一致，不能全称真实读错；a1_005/Q1的上下文容量停止是独立可核验的生成失败。逐臂CER分位数、每speaker支持和失败交集均在support_audit.json。', '',
                  '| 支持敏感性 raw margin | STT | SNT | SNT−STT |','|---|---|---|---|'])
    for label in ('mfa_only','strict','cloud4'):
        z=read(BASE/'mechanism'/f'{label}_k{k}'/'analysis.json')['3']['raw'];small=label=='strict' and (audit[label]['speakers']<8 or audit[label]['eligible']<16)
        lines.append('| '+label+('（仅描述）' if small else '')+' | '+' | '.join(fmt(z['margin_'+m]) if 'margin_'+m in z else '不可判' for m in ('STT','SNT','SNT-STT'))+' |')
    lines.extend(['','### 固定额外源图敏感性','','| image | native共同条数 | GQ | GC | event条/speaker | STT | SNT |','|---|---:|---|---|---|---|---|'])
    for im in ('6','9'):
        z=native[im]['analysis'];e=analysis[im]['raw'];n=e.get('margin_STT',{})
        lines.append(f"| {im} | {z['20_GQ']['n']} | {fmt(z['20_GQ'])} | {fmt(z['20_GC'])} | {n.get('n',0)}/{n.get('speakers',0)} | "+' | '.join(fmt(e['margin_'+m]) if 'margin_'+m in e else '不可判' for m in ('STT','SNT'))+' |')
    lines.extend(['','各条排除原因、query与occurrence数量在各support.json。主门不足不放宽；CER=0仅在≥8speaker且≥16条时作推断，其他情况只描述。源图6/9为预定每speaker首条，未按质量或event支持换样本，只有5条/5speaker进入event，不能声称充分源图泛化。完整结果见native.json与mechanism各analysis.json。', '',
                  '## 解释范围', '',
                  '- 正STT可削弱“只有自然/TTS类别不兼容”的解释；它不证明实例匹配导致native高分，也不证明TTS特有（没有自然重复发音）。',
                  '- SNT−STT只描述跨类别与同类别错配惩罚之差，不能称纯域贡献；N↔Q和Q1↔Q2的时长/韵律差异分布未匹配。既有MFA时长差描述见manipulation.json。',
                  '- MFA音素事件身份不是物理发音真值；phone不等于viseme，约200ms窗口可跨多个phone；最近格点的误差与支持已记录。',
                  '- 单位范数改变几何，只比较方向与稳健性，不给原生优势贡献百分比。',
                  '- 未训练模型、未根据eval挑lag或换seed。内容门和恢复amendment在任何本轮Sync结果前冻结，但恢复为发现生成问题后的修订实验。', '',
                  '## 复核与产物', '',
                  f"独立数值复核：`{json.dumps(read(BASE/'validation.json'),ensure_ascii=False)}`。", '',
                  f"TFG/延迟控制PASS={read(BASE/'controls_validation.json')['PASS']}（4臂重复visual/matrix误差0；8个±5帧实际PCM延迟均恢复预期offset）。生成、ASR独立校验分别见generation_validation.json与quality/validation.json；6项必要测试见tests.txt。", '',
                  'protocol.json、quality_rule_protocol.json、tts_runtime.json、calibration.json、native.json、quality/summary.json、mechanism/*/{support,cells,analysis,native_on_support}.json、control_scores与provenance.json。复现使用TTS_CHINESE_INSTANCE_RUN环境变量指向本run，按synthesize→quality→align→score→controls→calibrate/native/mechanism→独立check执行。'])
    (BASE/'report.md').write_text('\n'.join(lines)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    z=native['3']['analysis'];names=('GQ','GC','GQ-GC');values=[z['20_'+m] for m in names]
    means=np.array([v['speaker_mean'] for v in values]);ci=np.array([v['speaker_ci99'] for v in values]).T
    axes[0].errorbar(np.arange(3),means,yerr=np.vstack((means-ci[0],ci[1]-means)),fmt='o',capsize=5);axes[0].axhline(0,color='gray',lw=1);axes[0].set_xticks(range(3),names);axes[0].set_ylabel('Native Sync-C difference, 99% CI')
    arms=('N','Q1','Q2');z=analysis['3']['raw'];mat=np.array([[z['margin_q_'+a+b]['speaker_mean'] for b in arms] for a in arms])
    im=axes[1].imshow(mat,cmap='viridis');axes[1].set_xticks(range(3),arms);axes[1].set_yticks(range(3),arms);axes[1].set_xlabel('Audio instance');axes[1].set_ylabel('Video instance');axes[1].set_title('Same-event raw margin')
    for i in range(3):
        for j in range(3):axes[1].text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',color='white')
    fig.colorbar(im,ax=axes[1]);fig.savefig(BASE/'summary.png',dpi=180);fig.savefig(BASE/'summary.svg');plt.close(fig)
    code=BASE/'code';code.mkdir(exist_ok=True)
    for f in (ROOT/'scripts/experiments').glob('*tts_chinese_instance*.py'):shutil.copyfile(f,code/f.name)
    paths=[f for f in BASE.rglob('*') if f.is_file() and f.name!='provenance.json' and f.suffix in ('.json','.npz','.wav','.avi','.py','.md','.svg','.png')]
    write(BASE/'provenance.json',{str(f):sha(f) for f in paths})

if __name__=='__main__':main()
