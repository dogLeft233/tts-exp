---
title: repeated-take-crossover-20261015
type: note
status: done
phase: concluded_pilot
current_step: 4
priority: medium
project: tts-exp
date: '2026-10-15'
head_research: paused
tags:
- tts
- tfg
- syncnet
- repeated-takes
- causal-identification
permalink: tts-exp/codex/tts-exp/repeated-take-crossover-20261015
---

# repeated-take-crossover-20261015

## Objective

用户2026-10-15明确授权执行同说话人、同文本的N1/N2独立自然录制与T1/T2独立Qwen TTS发声实验，检验原N/T交叉降分是否超出普通异实例失配。该授权只恢复这项小实验，跨TFG增强头、旧训练、中心和已关闭审计保持暂停。

## Current State

**已完成**：正式结论 [[44-repeated-take-crossover]]；32视频/128评分格/64嵌入/128距离数组与全部原件已本地读回。ROOT `runs/repeated_take_crossover_20261015_attempt01/root_actual_qa.json` 实际PASS（SHA `2e0721cbe90f1642d0a4a507cd90671f1946133af9d609d709048fd28e64c438`），独立重建距离max误差3.815e−6、全部曲线/来源bootstrap误差0、样本时钟误差0。自然/TTS类内失配损失3.650/4.275、跨类3.963/4.708、额外0.314/0.433的描述区间均跨零。普通实例失配已复现原大幅非对角下降模式，不把原N/T强交互当独立来源偏好证据。

32段Qwen ASR成功/0重试，32原MFA TextGrid和48角色对/1967音素行均已本地读回；数字/专名等识别差异不删样，局部时间未统一。TTS成功GPU约5.23min，视频+评分5.44min；全部相关执行已退出，无后续实验自动启动。完整配置/版本修订、原失败和有限外推边界见报告44。增强头继续暂停。

以下保留本任务的来源与执行方式，终局以报告44和实际receipt为准：

ROOT维护理论、冻结小协议、BM和一次关键实际QA；代码/运行与有界数据查找使用GPT-6 Luna/max代理。实际8个中文来源的自然重复实例/独立参考已准备，manifest_v3与波形相关检查合格。云尝试1注册400，修复后尝试2注册200但Jan22合成404，无TTS音频；一次只读models GET200。模型可见列表不是完整语音服务资格证明。原件保留，秘密无落盘。

已在0个TTS波形/视频/分数时冻结本地Qwen0.6B provider修订，沿现有LAN环境和权重，固定独立参考+两采样seed，无安装/下载。FP16首个T1实际数值失败（PID2643278/rc1），原件保留；最新protocol_v3只改FP32，同seed/文本/参考/采样，独立attempt02。16段自然16k/RMS处理实际完成；矩阵与MFA源码就绪。原目标矩阵、RMS、时间、来源统计及上限不变，不把本地结果标旧云模型。MFA等完整quartet一次对齐，避免重复算自然边界。

FP32实际16/16输出成功、8对PCM均不同，同一speaker参考/目标相同；pilot44.499s和剩余269.161s均OS rc0，原24k FLOAT波形已本地读回。full8配置 `matrix_inputs_full8_v3_localqwen_fp32.json` SHA `1ccc73a749d3027bb39d81b09100f389512a0300aac927812044c6a88435449d`，ROOT实际32源hash通过；CPU标准化32段完成，audio_manifest SHA `7aab8a8edd0b369ee1452c9f18f598b60a9f92e0ab17a662645b346e8ab5638b`。已放行串行pilot4 render/score、按工程成本决定剩余4；MFA同canonical32新目录一次执行，内容QC仅诊断、不按分数挑样。

执行根 `runs/repeated_take_crossover_20261015_attempt01/`，专用源码 `scripts/experiments/repeated_take_crossover_20261015/`。不把旧训练身份或SHM缓存当本次执行来源。

## Design Boundary

每个speaker/text有真实独立自然N1/N2，以及同一独立参考句/同一voice配置的T1/T2。不同请求ID不等于不同音频；两TTS若PCM相同，记录确定性/重复，不能靠加噪、变速或编辑伪造第二次发声。自然复制/变速/重合成不能充当N2。优先同情绪/同录制设置，异情绪对不伪称受控重复。LibriSpeech排除。

先一个既有TFG（Wav2Lip），4来源×4驱动=16视频，控制与产物正常才补到8来源/32视频上限。每个quartet保存完整4×4评分矩阵，重用固定视频视觉特征，直接源PCM评分，固定RMS与显式时钟。使用全31lags共同有效支持，记录speech覆盖/静音/长度，保留C/B/Dmin/D0/offset和完整曲线。

主对比为同一driver的自配→类内异实例损失和自配→跨类实例损失；跨类相对类内差是描述性参照，不能自动归因为来源偏好或当普遍失配常数。内容相同不等于局部音素时间相同；外部对齐/已知lag与必要处理sham独立于SyncNet优化，未满足时间控制则保留相应解释边界。独立mouth-motion测量只作辅助，不称真实发音准确性。

## Acceptance and Cost

先确认speaker/text/take/实际PCM不同、voice参考独立、原/标准化PCM与视频身份、共同有效支持及全矩阵完整性；只做一次必要复核。不追加实现tests/pytest/compileall、全环境审计、模型安装或无界网盘扫描。API key只在runtime/无回显stdin传入，不落盘/argv/BM/log；使用与费用按实际receipt记，失败不无界重试。

若现有中文没有合格重复自然录制，可选公开有明确重复编号的英文小pilot，事先声明其是失配机制实验，不作为原中文TTS效应的独立确认，也不自动认证无预训练污染。

## Requirements

### Requirement: 产出具备类内重复发声参照的完整交叉矩阵与来源统计

SHALL 保留合格来源的四段独立实际音频、固定单TFG的4×4评分与原数组、外部音频时间诊断及可复现报告。

#### Scenario: 同文本不同发声实例实验完成

- **WHEN** 自然来源和Qwen不同PCM资格通过且冻结矩阵实际终局
- **THEN** 全部纳入来源都有4×4原数组和C/B/D0/Dmin/offset，同来源统计可从原件重算，类内/跨类及时间混杂边界进入报告。

## Observations

- [status] done
- [authorization] 用户要求实际执行重复发声对照，并允许使用此前提供的Qwen TTS API key；秘密本身不入记忆。
- [decision] 保持增强头暂停，只执行小型重复发声交叉实验；先单TFG、16视频，最多32。
- [result] 8来源自然发声实例的最大lag/RMS归一化波形相关0.0839–0.1551；未发现近同录音，语料不提供正式重录/session标签。
- [decision] 云Jan22合成404时没有TTS波形；执行前切换既有本地Qwen0.6B并保留版本边界，避免无限API调试。
- [result] 全8来源32视频128格已完成并通过一次ROOT实际QA；ASR/MFA与原数组均本地读回，正式报告44归档，未恢复head。

## Relations

- relates_to [[论文实验数据整理与写作主线]]
- completed_by [[44-repeated-take-crossover]]
- project [[独立自然发声与Qwen重复合成的交叉失配实验 Spec 2026-10-15]]
- relates_to [[研究交接总览]]
- relates_to [[独立40人同长交叉与评分边界探索 2026-09-29]]
- relates_to [[统一无损音轨后的跨模型剩余优势与自然音频替换审计 2026-09-26]]
