---
title: LRS3 bridge TTS quality comparison
type: experiment
permalink: tts-exp/experiments/lrs3-bridge-tts-quality-comparison
status: in_progress
date: '2026-09-13'
change_id: compare-lrs3-bridge-tts-quality
tags:
- lrs3
- bridge
- tts-quality
- replacement
- openspec
---

# LRS3 bridge TTS quality comparison

用户请求继续设计bridge实验，检验本地0.6B Qwen与云端Qwen的质量差异是否关联replacement增益。OpenSpec与下游流水线代码已实现；正式模型推理和人工听评尚未执行。

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| Initial planned experiment specification | September 13, 2026 | user request |

完整交接在 `openspec/changes/compare-lrs3-bridge-tts-quality/README.md`；算法、质量评估、统计与终态见同目录design.md，验收见specs/lrs3-bridge-tts-quality/spec.md，输入绑定见input-bindings.json，下游按tasks.md实现。

已核对历史LRS3 bridge使用的是dashscope_vc/qwen3-tts-vc-2026-01-22云端目标，而不是本地0.6B。22条原始云端TTS→MFA-linear目标→确认队列的metadata链和88个自然/云端/目标/face媒体file hash通过。本地同队列产物尚需审计并按规则补齐，不能直接用中文样本替代。

实验固定确认轮22条/22 source groups，双方共用文本、自然reference、MFA-linear/WavLM/HiFi-GAN流程和alpha=0.75自然相位bridge。四臂N/B0/B_LOCAL/B_CLOUD各两次独立渲染，共176视频、308科学评分cell；replacement都配原始自然N的共同支持前缀。生成端沿用历史full-frame box，评分端共享预先冻结的SyncNet裁剪与时间支持，所以不是旧数值的直接重放。

主端点是逐条cloud replacement gain减local gain，先平均repeat再22组配对bootstrap。各来源绝对正增益另作两项次要检验。原始TTS与M目标分别做独立匿名人工听评，至少3名共同评审者；无人工评分时自动比较仍可交付，质量明确NOT_ASSESSED。质量与provider混杂，不能作音质因果或泛化确认。

## Observations

- [status] planned
- [decision] 只写spec，本轮未进行模型推理、TTS合成或人工听评。
- [requirement] 不将云端质量更高视为已验证事实；独立测量raw和target听感质量。
- [requirement] 来源优势、绝对replacement增益和质量关联分别报告，不能相互替代。
- [requirement] 保留22条denominator，任一输入/对齐缺失如实阻断，禁止按成功或得分筛选子集。
- [limitation] 复用已见fit样本，单次TTS输出；两次渲染不能估计TTS随机生成方差。
- [validation] OpenSpec strict校验、输入file hash/metadata链、文档链接和矩阵计数检查通过；实现测试均待下游完成。
- [report] openspec/changes/compare-lrs3-bridge-tts-quality/README.md

## Relations

- relates_to [[LRS3 natural-to-TTS bridge confirmation result]]
- relates_to [[AISHELL-1 n25 Qwen cloud MFA-linear 2026-08-14]]


## Implementation update (2026-09-13)

- [implementation] Added `scripts/experiments/lrs3_bridge_tts_quality/` with staged audit, TTS provenance, MFA/WavLM targets, bridge, quality pack, serial Wav2Lip render, exact mux/SyncNet score matrix, paired statistics, versioned analysis, and independent validation.
- [validation] Stage 00 audit completed for the frozen 22-record/22-source-group cohort. Relevant tests: 19 passed; Ruff, bytecode compilation, and `openspec validate compare-lrs3-bridge-tts-quality --strict` passed.
- [resource] GPU lease rejects active compute processes/nonzero utilization and serializes every GPU stage. Formal TTS/WavLM/Wav2Lip/SyncNet execution was intentionally not started because the filesystem had about 154 MiB free; no new cloud TTS call was made.
- [boundary] No new LOCAL/CLOUD Sync-C, Sync-D, quality, or terminal result exists yet. Historical +0.07/+0.03 observations are not used as results for this comparison.