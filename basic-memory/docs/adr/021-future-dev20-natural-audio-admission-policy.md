---
title: 021-future-dev20-natural-audio-admission-policy
type: decision
permalink: tts-exp/docs/adr/021-future-dev20-natural-audio-admission-policy
status: open
project: tts-exp
decided: 2026-10-08 19:28:57.661274+00:00
tags:
- audio-enhancement
- prospective
- dev-policy
---

# 021-future-dev20-natural-audio-admission-policy

## Decision
新独立Dev20采用预先固定的完整540行三臂、三模型、三seed政策，以原自然PCM评分与原音换轨；在未来效果读取前冻结Cbest、D0、Canchor、GUARD20、ASR一致性与视觉跟踪可用性门槛。完整机器可读政策与分析代码仍由Luna/max子代理实现；本次写入不代表未来方法已晋级。

## Context and rationale
用户目标是可使用的自然音频增强头。仅提高最佳lag的Sync-C会混合零延迟同步、频谱幅值和局部时序效应，因此同时要求D0及固定锚点不恶化。旧Dev部分短片无法覆盖GUARD20，保持旧分母后另外固定S0102–S0111每人两段新Dev；不能以较长新样本替换旧失败行或称开发集为确认集。

完整规范位于openspec/changes/develop-cross-tfg-audio-head-20261003/future-dev20-admission-policy-attempt01.md，raw SHA-256 d0b7c983f3ef4b70943e714d009e79128226d182eb1327132b5d21b75052c866（7693字节）。样本及源PCM来自future_dev20_three_method_protocol_attempt06.json与已实际核验的source_stage_qc_attempt01.json。权重仍须实际Fit40终局资格确认并在Dev前冻结。

## Consequences
FULL中，候选比RMS和identity的每模型Cbest均须正，等模型比RMS至少+0.050。GUARD20中两个比较每模型及等模型Cbest均须正；D0和Canchor在两支持/两参考/各模型及等模型上均须非负。各固定sample/model/face/seed内按三臂公共查询键裁两端20并保留至少41，不新加FULL全31lag真实上下文条件。每个端点保留全部10说话人、20片段、3seed与540计划行。

固定48端点开发统计族，用PCG64 seed20261009共享100000×10说话人重采样、原始固定SE、中心化最大绝对统计量及higher95分位；区间为描述性，不据此宣称确认显著性。零SE仅接受10值完全相同；缺失端点使整个固定族不完整。

ASR使用固定qwen3-asr-flash-2026-02-10、zh、ITN=false，无真值/提示/上下文。自然1/自然2/RMS各20次共60基线，之后候选20次；独立调用不cache、不批量自动重试。每片excess=d(nat1,cand)-max(d(nat1,nat2),d(nat1,RMS))，说话人等权均值≤0.020且每片≤0.100；全部80非空规范化转写可用。此门槛是转写一致性的工程公差，非人工真值CER或人类可懂度。

全部540视频均需来源、原音、解码帧和时钟一致，tracking_valid_fraction≥0.950。嘴部几何/速度等仅描述，不据幅度变大判更佳同步。最终声称限于原音评分提升；人类感知另需独立政策。模型不可用、缺失或失败保留完整分母，不能换片救分析。Confirm80/40sp和LatentSync继续封闭。

## Alternatives
只用Cbest筛选会允许零延迟距离恶化。事后只分析GUARD20可测子集会改变预定样本。ASR直接当真值CER会把模型重复波动误判为内容损失。复用确认统计消费者会与其仅允许Confirm族的源码契约冲突，故使用明确的开发分析。

## Observations
- [decision] 新Dev数值准入政策在其效果读取前冻结；旧Dev与最终Confirm政策不变。
- [rationale] 零延迟/固定锚点共同约束最佳lag收益，完整原音与分母保持用户使用目标。
- [consequence] 通过开发门槛后固定方法再确认；未通过继续Fit迭代并保留所有失败。
- [verification] ROOT读完未来query/content/visual源码可行性报告；没有未来Dev效果或新ASR调用。

## Relations
- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]