"""Final expanded-cohort report, after independent numerical audits only."""
from pathlib import Path
import csv
import json
import numpy as np

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen_nrt_expansion40_20261022_attempt01'


def read(p):return json.loads(p.read_text())
def signed(x):return f'{x:+.3f}'
def interval(x):return '['+', '.join(signed(z) for z in x)+']'


def main():
    a=read(RUN/'analysis.json');p=read(RUN/'protocol.json');raw=read(RUN/'independent_generation_readback.json');audit=read(RUN/'independent_downstream_audit.json')
    assert raw['status']=='PASS' and audit['status'].startswith('PASS')
    asr=read(RUN/'asr/results.json');assert asr['count']==150
    assets=RUN/'report_assets';assets.mkdir(exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(9,4.3),layout='constrained')
    for offset,key,color in [(-.12,'T-N','#2563eb'),(.12,'T-R','#ea580c')]:
        for pos,cohort in enumerate(['new30','all40','pilot10']):
            c=a['cohorts'][cohort]['summaries'].get('common',{}).get('C',{}).get('contrasts',{}).get(key)
            if not c:continue
            lo,hi=c['CI97_5'];m=c['mean'];ax.errorbar(m,pos+offset,xerr=[[m-lo],[hi-m]],fmt='o',capsize=4,color=color,label=key if pos==0 else None)
    ax.axvline(0,color='black',lw=.8,ls='--');ax.set_yticks(range(3),['Added 30 (primary)','All 40 (secondary)','Pilot 10 (reused)']);ax.invert_yaxis()
    ax.set(xlabel='Paired Sync-C difference (97.5% bootstrap interval)',title='Frozen N/R/T expansion: main comparison uses the added 30 speakers');ax.legend()
    fig.savefig(assets/'cohort_contrasts.png',dpi=180);plt.close(fig)
    fig,axs=plt.subplots(2,1,figsize=(15,7),layout='constrained')
    for ax,policy in zip(axs,['common','native_full']):
        rr=a['records'];x=np.arange(len(rr));tn=[];tr=[]
        for r in rr:
            c=r['policies'][policy];t=np.mean([c[k]['C'] for k in ['T1','T2','T3']]);tn.append(t-c['N']['C']);tr.append(t-c['R']['C'])
        ax.bar(x-.18,tn,.36,label='T-N',color='#2563eb');ax.bar(x+.18,tr,.36,label='T-R',color='#ea580c');ax.axhline(0,color='black',lw=.7)
        ax.axvspan(-.5,9.5,color='gray',alpha=.12);ax.set_xticks(x,[r['speaker_id'][1:] for r in rr],rotation=60,fontsize=8)
        ax.set(ylabel='Paired Sync-C difference',title=policy+'; gray background = reused pilot speakers');ax.legend(fontsize=8)
    fig.savefig(assets/'all40_individual_contrasts.png',dpi=150);plt.close(fig)
    with (assets/'all_endpoints.csv').open('x',newline='') as f:
        w=csv.writer(f);w.writerow(['sample_id','speaker_id','cohort','support','condition','C','B','Dmin','D0','offset','query_count'])
        for r in a['records']:
            for policy,cells in r['policies'].items():
                for role,v in cells.items():w.writerow([r['sample_id'],r['speaker_id'],r['cohort'],policy,role,*[v[k] for k in ['C','B','Dmin','D0','offset','query_count']]])
    reliability=a['all120_draws_reliability'];samples={r['sample_id']:r for r in p['samples']}
    bad=[r for r in reliability if not r['qualified']];no_eos=[r for r in reliability if r['EOS'] is not True]
    valid=[r for r in a['records'] if r['cohort']=='new30'];main=a['cohorts']['new30']['summaries'].get('common',{}).get('C',{})
    if len(valid)<30:headline='部分新增来源缺少共同真实评分支持，主要队列未完整可测；下面仅列覆盖情况和可测来源描述，不宣布总体优势。'
    elif all(main['contrasts'][key]['CI97_5'][0]>0 for key in ['T-N','T-R']):headline='新增30人中，完整TTS相对N与R的共同窗口Sync-C均为正，且两项预定97.5%区间下界大于零；稳定性仍须结合完整时段、条件生成诊断及单模型/开发数据边界。'
    else:headline='新增30人没有同时确认完整TTS相对N与R的稳定正向Sync-C；逐项结果如下，不能将区间跨零视为等效，也不按结果继续加样本或换种子。'
    lines=['## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent',
           '- Origin Mode: run with numerical validation',
           '- Origin Date: 2026-10-22',
           '- Verification Status: EXECUTED_AND_NUMERICALLY_AUDITED; no human perception/accuracy validation or independent full regeneration',
           '- Version Label: qwen_nrt_expansion40_v1','',
           '# Qwen N/R/T 扩样：40位说话人、每人3个固定TTS种子','',headline,'',
           '原10人的N/R/T音频、50个视频和100份音视特征原样复用；新增30人、30份R重建和90条完整TTS、150个N/R/T视频、300份新音视特征；另完成150条新增源PCM的独立CPU ASR。没有训练。','',
           '## 新增30人主要结果与合并结果','',
           '| 队列 | 可测/预定人数 | N / R / T 平均C | T−N，97.5%区间 | T−R，97.5%区间 | 提高人数 T−N / T−R |',
           '|---|---:|---|---|---|---|']
    for cohort,target in [('new30',30),('all40',40),('pilot10',10)]:
        z=a['cohorts'][cohort];c=z['summaries'].get('common',{}).get('C')
        if not c:continue
        tn,tr=c['contrasts']['T-N'],c['contrasts']['T-R']
        lines.append(f"| {cohort} | {z['measurable_speakers']}/{target} | {c['means']['N']:.3f} / {c['means']['R']:.3f} / {c['T_mean']:.3f} | {signed(tn['mean'])} {interval(tn['CI97_5'])} | {signed(tr['mean'])} {interval(tr['CI97_5'])} | {tn['positive']} / {tr['positive']} |")
    lines+=['','每人先平均3个TTS种子，再按说话人重抽样10000次，seed20261022。新增30人是两项预设主比较；97.5%分位区间按Bonferroni两比较口径，普通95%也保存。40人合并和旧10人是次要/桥接结果；音频条数、重复生成和帧数不当独立人数。','',
            '样本、超参、支持集和生成资格规则在新增TTS前冻结，但这是看过旧10人的扩样；来源语料历史已暴露，无上游训练重合排除，不将它称作未见盲验证或严格人群确认试验。', '',
            '![不同队列配对区间](report_assets/cohort_contrasts.png)','',
            '## 所有预定支持与种子','',
            '| 队列/支持 | T−N，普通95%区间 | T−R，普通95%区间 |','|---|---|---|']
    for cohort in ['new30','all40','pilot10']:
        for policy in ['common','native_full','equal_count_spread']:
            z=a['cohorts'][cohort]['summaries'].get(policy,{}).get('C')
            if not z:continue
            tn,tr=z['contrasts']['T-N'],z['contrasts']['T-R']
            lines.append(f"| {cohort}/{policy} | {signed(tn['mean'])} {interval(tn['CI95'])} | {signed(tr['mean'])} {interval(tr['CI95'])} |")
    lines+=['','共同支持为五条件同绝对q窗口；native_full使用各自全部真实全lag窗口；equal_count_spread在各自原生时间轴均匀取相同数量、互不重复的q。均无边缘补零、时间拉伸或SyncNet offset重对齐。T和N/R说话时刻不同，未假装按共同q比较的是相同音素。','',
            '| 新30支持/种子 | T−N，95%区间 | T−R，95%区间 |','|---|---|---|']
    for policy in ['common','native_full','equal_count_spread']:
        for role in ['T1','T2','T3']:
            z=a['cohorts']['new30']['summaries'].get(policy,{}).get('C')
            if not z:continue
            tn,tr=z['per_seed'][role]['T-N'],z['per_seed'][role]['T-R']
            lines.append(f"| {policy}/{role} | {signed(tn['mean'])} {interval(tn['CI95'])} | {signed(tr['mean'])} {interval(tr['CI95'])} |")
    lines+=['','以上支持/种子及下面辅助端点是描述性敏感性，不额外宣称多重显著发现。','',
            '## 生成可靠性与条件分析','',
            f"全部120条固定draw中，无EOS {len(no_eos)}条；预先冻结的条件规则未通过{len(bad)}条。新90条中的无EOS数为{sum(samples[r['sample_id']]['cohort']=='new30' for r in no_eos)}。原始输出均保留、未挑最佳种子或重采样。",'',
            '条件规则：EOS=True且生成时长/自然时长介于0.5和2.0之间（含边界）；仅在补充分析中对每人剩余合格draw取平均，原五条件支持不变，所有draw仍进入主要分析。规则根据pilot问题制定、在新30生成前冻结；不代表人工内容/自然度合格，也不是可随意删样本的依据。','',
            '| 队列/支持 | 条件分析覆盖人数 | T−N，95%区间 | T−R，95%区间 |','|---|---:|---|---|']
    for cohort in ['new30','all40','pilot10']:
        z=a['cohorts'][cohort]
        for policy in ['common','native_full','equal_count_spread']:
            v=z['conditional_generation'].get(policy,{}).get('C',{}).get('contrasts')
            if not v:continue
            tn,tr=v['T-N'],v['T-R'];lines.append(f"| {cohort}/{policy} | {z['conditional_coverage']} | {signed(tn['mean'])} {interval(tn['CI95'])} | {signed(tr['mean'])} {interval(tr['CI95'])} |")
    lines+=['','未通过draw明细：','', '| 来源/种子 | EOS | 时长比 |','|---|---|---:|']
    for row in bad:lines.append(f"| {row['sample_id']}/{row['role']} | {row['EOS']} | {row['duration_ratio']:.3f} |")
    lines+=['','## 辅助端点、个体差异与内容代理','',
            '| 新30共同窗口端点 | T−N，95%区间 | T−R，95%区间 |','|---|---|---|']
    for metric in ['C','B','Dmin','D0','offset']:
        z=a['cohorts']['new30']['summaries'].get('common',{}).get(metric)
        if not z:continue
        tn,tr=z['contrasts']['T-N'],z['contrasts']['T-R'];lines.append(f"| {metric} | {signed(tn['mean'])} {interval(tn['CI95'])} | {signed(tr['mean'])} {interval(tr['CI95'])} |")
    lines+=['','C=B−Dmin；D0是零lag距离；offset=15−argmin、单位帧。C高不等于真人嘴型更准确，也不能据自配分差断言SyncNet偏爱TTS。','',
            '![全部40人的变化](report_assets/all40_individual_contrasts.png)','',
            '| 说话人/来源 | 队列 | 共同窗口T−N | 共同窗口T−R | 共同q数 |','|---|---|---:|---:|---:|']
    for r in a['records']:
        c=r['policies']['common'];t=np.mean([c[k]['C'] for k in ['T1','T2','T3']]);lines.append(f"| {r['sample_id']} | {r['cohort']} | {signed(t-c['N']['C'])} | {signed(t-c['R']['C'])} | {r['common_query_count']} |")
    warnings=[r['sample_id'] for r in a['records'] if r['low_common_query_warning']]
    lines+=['',f'共同q少于25的来源：{warnings}；不可测来源：{a["unmeasurable"]}。没有按结果替换来源。逐个移除每一说话人的对称影响量保存在analysis.json各C summary内，不能只删负贡献者。','',
            '全部150条新增N/R/T使用本地同一Whisper-small、CPU int8/beam5/temperature0、无目标提示、无VAD转写。源PCM逐样本绑定，输入float32/32768，无PyAV转码。CER简繁/标点/空白归一后计算，数字写法仍有假差异，不能当自然度或错读真值。', '',
            'ASR原始/分段与CER见asr/results.json；由非EOS或异常时长触发的整组五条件VAD敏感性见asr/vad_results.json，属于探索性代理，不决定评分或条件生成资格；无人工试听或视觉真值。','',
            '## 执行、核验与边界','',
            f"pilot10全部支持/端点数值桥最大误差{a['pilot_bridge_max_error']}。单独S0002编码/原R decode exact桥通过；30新R的全16码本、90新T的new/ref/concat编码、完整FLOAT与官方比例裁切、seed/EOS独立CPU读回通过；每条T同concat重复decode exact。",'',
            'T依然是官方full self-ICL，同句自然参考和固定语料转写，FP32、同权重、同eager请求（nested SDPA配置如实记录）、温度0.9/top_k50/top_p1/max_new_tokens512。N/R/T所有下游同RMS 0.0376838172675746、同Wav2Lip GAN/face/inference源码、同SyncNet V2和源PCM。原R是standalone codec，T包含参考prefix解码上下文，故T−R不能单独归因于文字LM学习任务。', '',
            '新GPU生成阶段明确进程退出后才启动下游；CPU ASR与后续GPU视频阶段可并行。独立审计细节见independent_downstream_audit.json和independent_generation_readback.json；只核验已执行产物，不冒充独立重新生成全部TTS/视频。原codec10与full-TTS10封存文件未改。', '',
            '11/11误读复核：全部人/种子/支持保留、异质性不当一致（Simpson）；单位是说话人（生态）；历史语料和预选3..8秒限制（Berkson）；条件生成是输出条件化诊断不作主因果（collider）；非准确率筛查基率任务；不按极端分数挑人而不宣称消除均值回归；无按EOS/分数删主异常（survivorship）；两项主比较97.5%口径其余描述（look-elsewhere）；扩样前协议冻结但不是从未看数据（forking）；自配C/ASR不作质量/训练目标因果；因果链未闭合、无学习收益反推（reverse causality）。', '',
            '完整run在Ubuntu /home/wjj/tts-audio/tts-exp/runs/qwen_nrt_expansion40_20261022_attempt01/；本地完整备份预定C:/Users/dogle/.codex/tts-exp-artifacts/qwen-nrt-expansion40-20261022/qwen_nrt_expansion40_20261022_attempt01/，最终逐文件SHA读回以artifact_manifest和local_readback_verification为准。小报告/分析镜像在E盘同run目录。', '',
            '本轮到扩样、独立复核和报告为止，不启动训练、不扩大解码参数搜索、不因中间正负结果追加到显著。']
    (RUN/'report.md').write_text('\n'.join(lines)+'\n')
    print(headline,flush=True)


if __name__=='__main__':main()
