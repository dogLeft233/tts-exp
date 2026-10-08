---
title: Wav2Lip spectral structure replacement 2026-09-08
type: experiment
permalink: tts-exp/experiments/wav2-lip-spectral-structure-replacement-2026-09-08
status: concluded
date: '2026-09-08'
hypothesis: 保留natural相位/时间网格的逐时刻MFA-linear谱迁移，是否比重建处理及整句平均log谱修正带来更大replacement收益
report: openspec/changes/test-wav2lip-spectral-structure-replacement/
tags:
- wav2lip
- replacement
- spectral-structure
- concluded
- no-useful-gain
---

# Wav2Lip spectral structure replacement 2026-09-08

用户要求承接最初 natural 向 MFA-linear 靠近时的小幅 replacement 收益，设计简洁且可正确实现的下游 OpenSpec。首轮 MAG_075 ΔSync-C=+0.078，18/23改善，未校正95% CI=[+0.016,+0.136]；确认轮固定0.75为+0.032，CI=[−0.034,+0.105]。尚无确认收益。时间控制分支的失败不直接否定保留natural时间网格的声学迁移假设，但不能继续反复修旧门禁。

新设计冻结22条已观察过的seen-fit记录、α=0.75、原face/ROI。科学臂N、RT（α=0重建处理）、MAG（逐时刻log magnitude迁移）、ENV（将mean_t(log|M|−log|N|)广播加到natural时变log谱）；另设独立N_REPEAT生成。三变换统一采用原STFT、RMS/峰值缩放和PCM规则。ENV仍来自配对TTS，未匹配扰动范数；不能宣称纯内容特异性或任意EQ有效。

采用整数平台已冻结的目标时间U窗口，全部候选与未经修改的N PCM评分。负控制仅把已有P音频mux到新V_N，检查同一视频错配敏感性，不生成P驱动视频，不修复历史generated-own门禁。这是新的局部窗口endpoint，不是历史整条视频收益的原协议复现。

Stage A先生成N/N_REPEAT/RT：66视频、88cells，检查双侧±0.05重复/RT等效性与局部错配敏感性；科学通过且独立验收valid才运行B的MAG/ENV，增加44视频/44cells，总上限110视频/132cells。统一source-group bootstrap，控制95%CI，候选两条正面路线使用97.5%CI。gain必须同时胜过N和RT（C CI下界>0且均值>0.05，D安全与offset要求通过）。MAG优于ENV才支持逐时刻构造增量；ENV有效且两臂C/D差的CI均在±0.05内才支持范围内平均谱足够。不显著不等于等效。

## Observations
- [status] concluded
- [execution_status] stage_b_complete_validated
- [decision] 科学终态为 NO_USEFUL_GAIN_ESTABLISHED：MAG和ENV都未相对N/RT形成预设gain；停止本构造，不追加科学重复、不训练、不修复旧门禁、不宣称泛化。
- [protocol] 冻结22 records/22 source groups，N/RT/MAG/ENV加N_REPEAT，最多110 GPU视频与132 CPU评分cells；前置失败则B为0。
- [validation] OpenSpec strict validation、Ruff、5个focused tests与独立validator均通过；输入join、音频重建、视频身份、矩阵/统计验收valid。
- [result] Stage A 22/22记录、88 cells；Stage B 22/22记录、44 cells；Stage A控制通过。MAG-N ΔC=-0.8895，97.5% CI=[-1.1764,-0.6028]；ΔD=-0.8429，CI=[-1.1508,-0.5592]。ENV-N ΔC=-0.0422，CI=[-0.0944,+0.0093]；ΔD=-0.0365，CI=[-0.0890,+0.0179]。MAG-ENV ΔC=-0.8473，CI=[-1.1200,-0.5696]；ΔD=-0.8064，CI=[-1.1062,-0.5356]；U offset agreement均22/22。
- [boundary] 本轮为seen-fit机制探索，U局部endpoint与历史file-level分数不同；ENV/MAG差异也包含扰动量与重建影响，不能证明音素机制。ENV接近N但未达到gain阈值，不等价于已证明平均谱足够。
- [report] openspec/changes/test-wav2lip-spectral-structure-replacement/；实现为scripts/experiments/wav2lip_spectral_structure_replacement/。
- [artifact] runs/wav2lip_spectral_structure_replacement_20260908_cpu_prepare_v3/{final.json,analysis.json,validation.json,result.md}；共110 videos/132 cells。

## Relations
- follows [[LRS3 natural-to-TTS bridge confirmation result]]
- relates_to [[Wav2Lip integer plateau control 2026-09-07]]

## Changelog
| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 按用户要求设计声学对照OpenSpec，严格校验与输入join审计通过；仅planned，尚未执行 | September 8, 2026 | user |
| 按用户要求开始执行Stage A；BM状态更新为running，等待CPU/输入门禁与GPU实验 | September 8, 2026 | user |
| Stage A/B完成且独立验收valid；科学终态NO_USEFUL_GAIN_ESTABLISHED，记录效应量与停止边界 | September 8, 2026 | user |
