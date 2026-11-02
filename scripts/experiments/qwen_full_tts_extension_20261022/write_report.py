"""Write the final report from executed, independently audited artifacts only."""
from pathlib import Path
import json

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/qwen_full_tts_extension_20261022_attempt01'


def read(p):return json.loads(p.read_text())
def signed(v):return f'{v:+.3f}'
def interval(v):return '['+', '.join(signed(x) for x in v)+']'


def main():
    a=read(RUN/'analysis.json');tts=read(RUN/'tts/tts_manifest.json')
    diag=read(RUN/'report_assets/diagnostic_summary.json')
    own=read(RUN/'independent_tts_readback.json')
    audit=read(RUN/'independent_downstream_audit.json')
    if own['status']!='PASS' or not audit['status'].startswith('PASS'):raise RuntimeError('independent verification not passed')
    m=a['summary']['common']['C'];contrasts=m['contrasts']
    tn=contrasts['T-N'];tr=contrasts['T-R']
    positive=[tn['CI97_5'][0]>0,tr['CI97_5'][0]>0]
    if positive==[True,True]:
        headline='本开发队列支持完整TTS的自配Sync-C高于自然N与单独codec R；分差出现在完整生成流程中，尚未隔离文字模型的学习任务或解码上下文贡献。'
    elif positive==[True,False]:
        headline='本开发队列支持完整TTS相对自然N的自配Sync-C正差异；相对单独codec R尚未确认，不能明确归因到文字生成环节。'
    elif positive==[False,True]:
        headline='完整TTS相对codec R的自配Sync-C为正，但本开发队列尚未确认相对自然N的优势，不能据此解释既有自然/TTS优势。'
    else:
        headline='两项预设比较均未通过正向97.5%区间门；本开发队列尚未确认完整TTS评分优势，也不能把这一结果视为没有差异或等效。'
    lines=['## Material Passport','',
           '- Origin Skill: academic-research-suite / experiment-agent',
           '- Origin Mode: run, with numerical validation',
           '- Origin Date: 2026-10-22',
           '- Verification Status: EXECUTED_AND_NUMERICALLY_AUDITED; no independent full TTS/TFG regeneration or human validation',
           '- Version Label: qwen_full_tts_extension_v1','',
           '# 完整 Qwen TTS 补组：复用 N/R，比较 T−N 与 T−R','',headline,'',
           '执行完成：10位历史开发说话人，每人3个预定TTS种子；新增30条T音频、30条Wav2Lip视频、60份SyncNet特征，复用原N/R音频、视频与特征。没有训练、没有筛除来源/种子。统计单位为10人。','',
           '## 主要结果','',
           '| 条件 | 共同有效窗口平均 Sync-C |','|---|---:|',
           f"| N 自然语音 | {m['means']['N']:.3f} |",f"| R 原录音的Qwen自身codec重建 | {m['means']['R']:.3f} |",
           f"| T 完整自克隆TTS，先在每人内平均3种子 | {m['T_mean']:.3f} |",'',
           '| 预定比较 | 配对平均差 | 描述性95%区间 | 两项主比较用97.5%区间 | 正/负说话人数 |',
           '|---|---:|---|---|---:|']
    for key in ['T-N','T-R']:
        s=contrasts[key];lines.append(f"| {key} | {signed(s['mean'])} | {interval(s['CI95'])} | {interval(s['CI97_5'])} | {s['positive']}/{s['negative']} |")
    lines+=['', '预先冻结10000次说话人bootstrap、seed20261022；先平均同一说话人的T1/T2/T3，再配对。97.5%分位区间用于两项预定主比较的Bonferroni口径；10人的开发队列小样本区间不是确认性人群结论。未计算p值；区间跨零不证明等效。','',
            '![配对Sync-C与主比较区间](report_assets/paired_sync_c.png)','',
            '## 取窗敏感性与全部种子','',
            '| 支持规则 | N / R / T 平均C | T−N，95%区间 | T−R，95%区间 |',
            '|---|---|---|---|']
    labels={'common':'五条件相同绝对q窗口（主要）','native_full':'各自全部真实有效窗口','equal_count_spread':'等窗口数，均匀覆盖各自时间轴'}
    for policy,label in labels.items():
        q=a['summary'][policy]['C'];cc=q['contrasts']
        lines.append(f"| {label} | {q['means']['N']:.3f} / {q['means']['R']:.3f} / {q['T_mean']:.3f} | {signed(cc['T-N']['mean'])} {interval(cc['T-N']['CI95'])} | {signed(cc['T-R']['mean'])} {interval(cc['T-R']['CI95'])} |")
    lines+=['','| 种子索引 | 平均C | T−N，95%区间 | T−R，95%区间 |','|---|---:|---|---|']
    for k in ['T1','T2','T3']:
        q=m['per_seed'][k]
        lines.append(f"| {k} | {m['means'][k]:.3f} | {signed(q['T-N']['mean'])} {interval(q['T-N']['CI95'])} | {signed(q['T-R']['mean'])} {interval(q['T-R']['CI95'])} |")
    lines+=['','| 支持规则 × 种子 | T−N均值 | T−R均值 |','|---|---:|---:|']
    for policy in ['native_full','equal_count_spread']:
        for k in ['T1','T2','T3']:
            q=a['summary'][policy]['C']['per_seed'][k]
            lines.append(f"| {policy} × {k} | {signed(q['T-N']['mean'])} | {signed(q['T-R']['mean'])} |")
    if tn['mean']>0 and tr['mean']>0 and all(a['summary']['native_full']['C']['contrasts'][k]['mean']<0 for k in ['T-N','T-R']):
        lines+=['','两项点估计在共同窗口与覆盖完整有效时段之间均改变方向，当前差异对支持规则敏感。无token上限异常的T1/T2在完整有效时段下也均为负点估计，不能把方向变化全部直接归给异常T3；同样不能把负点估计写成TTS普遍有害。应先稳定复现完整TTS优势，再做模块归因或方法训练。']
    lines+=['', '以上敏感性、分种子及辅助端点均为描述性诊断，不追加显著性发现。全部人/种子保留。','',
            '## 评分端点拆分','',
            'C=B−Dmin；D0为零lag距离；offset=15−argmin（25fps，单位帧）。C升高不能直接称为同步更准确，尤其需要看B和D0。','',
            '| 端点 | T−N，95%区间 | T−R，95%区间 |','|---|---|---|']
    for metric in ['C','B','Dmin','D0','offset']:
        cc=a['summary']['common'][metric]['contrasts']
        lines.append(f"| {metric} | {signed(cc['T-N']['mean'])} {interval(cc['T-N']['CI95'])} | {signed(cc['T-R']['mean'])} {interval(cc['T-R']['CI95'])} |")
    lines+=['','## 逐说话人结果','',
            '| 来源 | N | R | T1 | T2 | T3 | T均值 | T−N | T−R | 共同q数 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in a['records']:
        vals=[r['policies']['common'][k]['C'] for k in ['N','R','T1','T2','T3']];tm=sum(vals[2:])/3
        lines.append('| '+r['sample_id']+' | '+' | '.join(f'{v:.3f}' for v in vals)+f' | {tm:.3f} | {signed(tm-vals[0])} | {signed(tm-vals[1])} | {r["common_query_count"]} |')
    lines+=['','## 生成结构、控制与异常','',
            'T使用与原R相同的Qwen3-TTS-12Hz-0.6B-Base snapshot和codec权重。目标/参考文字均为固定语料转写去空白，参考语音仍为同一句自然录音，full self-ICL，non_streaming_mode=False；do_sample=True、temperature=0.9、top_k=50、top_p=1、max_new_tokens=512。三个预定seed base为2026102201/02/03，实际seed加100×说话人数字ID。所有默认subtalker设置与真实生成参数留在各record。','',
            '官方路径是：文字/参考条件→新语音编码→拼接参考编码和新编码一起decode→按官方浮点比例公式截取新句。T主组保留此原样输出；没有把new-codes-only诊断替代主组。','',
            f"- 10/10参考编码逐元素等于原R编码；另用原R编码直接decode，10/10原始FLOAT逐元素复现旧R。请求eager，但实际codec nested encoder/decoder config为sdpa；该桥接确认本次与旧R数值一致，不能写成所有内部层都eager。",
            '- 30/30官方拼接编码重复decode均逐元素一致；独立CPU读回检查全部保存编码拼接、原始完整波形、官方裁切、种子和EOS通过。两条官方浮点比例裁切比reference_steps×1920少1样本，已原样保留。',
            f"- R加入参考解码上下文后，{10-diag['context_R_exact_count']}/10波形不再与原R完全一致，重叠样本MAE范围{diag['context_R_mae_range'][0]:.6f}–{diag['context_R_mae_range'][1]:.6f}。官方T与new-codes-only也有{30-diag['official_vs_new_only_exact_count']}/30不完全相同。因此T−R包含生成编码与decoder上下文的共同变化，不能隔离文字模型损失/学习任务。该音频差异还没有单独的TFG分数效应证明。",
            '- S0011（BAC009S0011W0121）seed3达到max_new_tokens上限：511个新codec步，40.88秒，未出现EOS，possible_max_token_truncation=True；其余29条有EOS。全部保留，不据分数删除或重采样。输出没有经过人工内容认证，因此不能保证所有T都正确读完同一句文字。',
            '- 执行编排偏离：codec桥接进程在主TTS最后一条生成期间短暂并行；桥接event窗口约10.020秒，精确进程退出时刻未捕获。两进程均exit0、无OOM/failure；原参数/种子未变，数值控制通过。完整记录见execution_deviation.json，不能声称严格串行运行，也未补跑替换主结果。','',
            '下游与原N/R统一24k→16k FLOAT→固定RMS 0.0376838172675746→PCM16，没有削波、时间拉伸或按offset重对齐。新30视频使用相同Wav2Lip GAN权重、image3与生成源码/参数；SyncNet使用对应源PCM而非MP4压缩音轨，固定相同脸框。','',
            f"五条件共同q数范围{min(diag['common_query_counts'])}–{max(diag['common_query_counts'])}；低于25的来源{diag['low_common_query_samples']}。q均保证±15全部来自真实embedding，不纳入边缘补零。旧N/R scalar在原q上精确复现，最大误差{a['parent_score_bridge_max_error']}; 主要分析改为五条件共同q，故新表的N/R均值可以与原报告不同，不能拿原均值直接减T。",'',
            '## 数值核验与解释边界','',
            '12项相关pytest通过；独立TTS原始产物CPU读回PASS；独立下游从原始embedding复算50组距离、三种支持的全部端点/配对bootstrap，并核验30canonical音频与视频绑定、60新特征、旧parent完整seal。详情以独立JSON审计为准。没有独立重新合成30条TTS或重新渲染视频，因此这里只确认已执行产物的数值可追溯性。','',
            '统计置信等级：CAUTION。历史已暴露AISHELL开发来源、单静态脸/单TFG、10人、同句自ICL参考、一个达到长度上限的输出、一次GPU编排偏离；没有上游训练重合排除、人工内容/音素标注或合格视觉同步盲评。自配C差异不能证明评分器错误、真实嘴型更准或某训练方法有效。','',
            '11/11统计误读检查：','',
            '| 检查项 | 处理与限制 |','|---|---|',
            '| Simpson反转 | 报全部说话人及3种子；没有把聚合方向当所有人一致 |',
            '| 生态谬误 | 独立单位10人；帧/30次生成不是独立样本 |',
            '| Berkson选择 | 历史开发来源、元数据长度筛选；无独立人群推广 |',
            '| Collider偏倚 | 不按生成时长或分数剔除；最短共同支持由输出长度决定，另报两种预定敏感性 |',
            '| 基率忽视 | 非诊断准确率/筛查研究，不适用 |',
            '| 均值回归 | 不按极端分数选来源或最好seed |',
            '| 幸存者偏倚 | 全10人×3seed保留，达到上限一条明确标注 |',
            '| 多重搜索 | 两项预定主比较97.5%区间；其它端点/支持/种子描述性列全 |',
            '| 分析岔路 | protocol在T波形前冻结；无事后换主要支持或删异常 |',
            '| 因果越界 | T−R包含生成编码和decoder上下文，不能隔离LM训练任务或感知质量 |',
            '| 反向因果 | 音频→冻结TFG→冻结评分顺序明确；并不闭合学习机制解释 |','',
            '## 产物与复现入口','',
            '- 协议：protocol.json；T源manifest：tts/tts_manifest.json；主结果：analysis.json；逐端点CSV与图：report_assets/。',
            '- 原N/R不重写：runs/qwen_codec_roundtrip_20261022_attempt01，212个复用资产在协议中绑定；原2740文件seal独立核验。',
            '- 新run：/home/wjj/tts-audio/tts-exp/runs/qwen_full_tts_extension_20261022_attempt01/。',
            '- 本地完整副本：C:/Users/dogle/.codex/tts-exp-artifacts/qwen-full-tts-extension-20261022/qwen_full_tts_extension_20261022_attempt01/；对应parent完整副本在qwen-codec-roundtrip-20261022目录。最终SHA与读回校验另见artifact_manifest/local_readback_verification。',
            '- 执行命令及timeout/PID/exitcode在tts_console.log、tts/events.jsonl和各*_execution.json；源快照在source_snapshot/，权重/环境在protocol和runtime/environment JSON。',
            '- 本次不训练、不扩大TTS参数搜索、不恢复历史暂停实验；无commit/push/云归档。','']
    (RUN/'report.md').write_text('\n'.join(lines))
    print(headline)


if __name__=='__main__':main()
