"""Final run report; never creates a progress log or modifies frozen design."""
from pathlib import Path
import argparse,json,time

def main(run):
    read=lambda p:json.loads(p.read_text())
    gate=read(run/'B_gate_and_calibration.json');audit=read(run/'A_CPU_independent_audit.json');cpath=run/'C_analysis.json'
    lines=['# Qwen双规模条件预测与采样机制：执行结果','',
        '## Material Passport','',
        'Origin Skill: academic-research-suite / experiment-agent；Mode: run；2026-10-23。执行代理：GPT-6.1-sol / medium；接口提供priority，无独立fast开关。无训练。',
        '',f'运行：`{run}`。冻结设计保持原文：`frozen_design.md`。',
        '', '## A：完整历史技术资格','',
        '两模型各一次成功原生重放与一次完整16码teacher-forced重放：历史codes与FLOAT波形均逐值一致，RNG一致；原始/有效logits及概率差均为0，argmax一致。每帧粗码cb0→细码cb1..15，最后EOS仅cb0，顺序计数检查通过。',
        '', '未来码改变未影响此前预测；改变frame1的cb7会改变cb8及下一主干粗码预测；三个短前缀输出均与指定完整码逐值一致。异常退出hook恢复测试通过。',
        '', '| 模型 | NumPy重建政策logits最大差 | NumPy softmax最大差 | 有效掩码 |',
        '|---|---:|---:|---|']
    for r in audit['rows']:lines.append(f'| {r["model"]} | {r["max_abs_policy_logits_CPU_rebuild"]:.3g} | {r["max_abs_probability_CPU_softmax"]:.3g} | 逐值一致 |')
    lines.extend(['','首次attempt01在原生forward后因processor元数据中的合法`-inf`无法写入严格JSON而失败；原native.npz及全栈保留。同seed工程attempt02以字符串编码非有限配置值，数值数组保持NPZ原样，完成重放。两次原生trace整个NPZ的SHA逐字一致；失败attempt的wave/codes未持久化。首失败原生forward确已发生，未换seed；首次裸后台进程的实际退出码未捕获，不把推测exit1当凭据。后续进程均保留PID、命令、stdout/stderr与实际子进程退出receipt。','',
        '## B：共同完整历史诊断','',
        '10来源、每来源N1/T06三draw/T17三draw，共70完整历史×2读取模型=140 passes。独立单位是来源。原有S0011/L06/draw3和S0016/L06/draw3各511帧完整保留；按前已冻结EOS/时长规则不进入主诊断，但进入全部可计算历史补充。没有截尾或评分选样。',
        '', '来源等权；来源内N/T06/T17等权，T先平均draw；内容帧等权。粗码单列，15细码逐码本报告再等权。粗码共享内容ID0..2047，EOS2150单列；原始与原生政策后概率均保留。自然是同句参考ICL历史，不能称独立语言建模泛化。',
        '', '| 类别 | L17−L06熵差 nat | 97.5%来源bootstrap区间 | 同方向来源 | 推进门 |',
        '|---|---:|---|---:|---|'])
    for name,r in gate['gates'].items():lines.append(f'| {name} | {r["mean_L17_minus_L06_nat"]:+.6f} | [{r["CI97_5"][0]:+.6f}, {r["CI97_5"][1]:+.6f}] | {r["same_direction_sources"]}/10 | {"通过" if r["pass"] else "未通过"} |')
    lines.extend(['','Bootstrap=20000，seed2026102307。推进需至少一类差异幅度≥0.10nat、≥8/10同向且97.5%区间不跨0；所有通过类均须在两模型方向内有界匹配，否则停止C。以上是冻结探索/工程门，不是已验证的科学最小效应或感知阈值。','',
        '| 模型 | 类别 | 校准状态 | temperature | 目标熵 nat | 实现熵 nat |','|---|---|---|---:|---:|---:|'])
    for model,categories in gate['temperatures'].items():
        for name,r in categories.items():
            temp=r.get('temperature');achieved=r.get('achieved')
            lines.append(f'| {model} | {name} | {r["status"]} | {temp if temp is not None else "未匹配"} | {r["target"]:.6f} | {achieved if achieved is not None else "未匹配/不推进"} |')
    lines.extend(['','只按B预测logits校准，未用Sync-C挑温度；[0.4,1.6]内确定性求解，目标误差≤0.02nat，top-k/top-p/repetition/EOS合法政策保持。完整逐历史/逐码本熵、JSD、top1分歧、真实后续码NLL/名次及EOS见`B_history_diagnostics.json`。零政策概率明确给出+inf NLL和零支持率，不伪造有限NLL。','',
        '旧分差关联仅两个来源内中心化候选，来源cluster区间及逐一留出敏感性见`B_entropy_score_description.json`；它们不改变推进门。','', '## C：确认阶段',''])
    if gate['status']=='STOP_C':lines.extend(['**按预设门停止C。** 未生成确认TTS、未生成新TFG、未运行新的SyncNet评分。未过熵差门表示本次筛查没有达到冻结推进标准，不等于两模型分布相同或所有解码机制均无效；通过差异门但无法有界匹配则属于本政策未可操纵。C代码仅预实现/编译及合成fixture检查，不能把实现当已执行结果。'])
    elif cpath.exists():
        c=read(cpath);lines.extend([f'已执行C。主比较完整支持来源：{len(c["complete_primary_source_ids"])}/10。名单、缺失、资格和逐政策失败见C协议/资格/音频/ASR/下游产物。','', '| 比较 | 点估计Sync-C | 97.5%区间 |','|---|---:|---|'])
        for name,r in c['primary_and_gap_contrasts'].items():
            if r['mean'] is not None:lines.append(f'| {name} | {r["mean"]:+.3f} | [{r["CI97_5"][0]:+.3f}, {r["CI97_5"][1]:+.3f}] |')
        lines.extend(['','上述仅为数值统计；实质支持还要求内容/EOS资格及实际自由轨迹操纵达成。实际熵、原生分差是否确认、反向模型损坏、完整支持和严格敏感性均须同时读，不以L17降分独自称救援。'])
    else:lines.extend(['B达到推进条件，但C尚无完整分析结果；不得写成已完成或科学阴性。实际阻塞/失败见对应状态与日志。'])
    lines.extend(['','## 解释边界与交付','',
        'H1仅检验条件预测分布及temperature解码政策是否参与当前两checkpoint的原生评分差；H2仅诊断。共享codec不等于相同隐藏状态，未进行跨维隐藏交换或训练。低熵、低NLL或可分性不能直接证明质量机制。即使温度政策成功，也不识别熵唯一中介、纯参数量或人类嘴型质量。TTS原生驱动与自身评分音轨同步改变，未分离TFG与SyncNet贡献。',
        '', '必要控制、原始失败attempt、源码、输入小依赖、SHA manifest与完整证据TAR均保留；归档读回receipt单列。E盘仅小镜像，完整证据远端与C盘。没有修改模型、安装包、旧run或无关dirty，没有训练、云调用、扩人/seed或为阳性扩搜索。',''])
    with (run/'report.md').open('x',encoding='utf8') as f:f.write('\n'.join(lines))

if __name__=='__main__':ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);a=ap.parse_args();main(Path(a.run))
