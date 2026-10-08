---
title: Wav2Lip natural content residual probe 2026-09-08
type: experiment
permalink: tts-exp/experiments/wav2-lip-natural-content-residual-probe-2026-09-08
status: concluded
date: '2026-09-08'
report: openspec/changes/probe-wav2lip-natural-content-residual/
tags:
- wav2lip
- replacement
- content-conditioning
- residual
- control-failed
---

# Wav2Lip natural content residual probe 2026-09-08

历史shift复核没有确认replacement，也没有确认评分程序错误。历史同J的FULL_FRAME_V4自由ΔC=+0.341、正向ΔD=+0.179，但固定natural anchor收益=-0.099且CI跨零；两评分入口的配对差异均未确认。偏移分支维持CLOSE_SHIFT_DIAGNOSTIC。

下一步做一次小型、无需训练的自然基底辅助增量检验。历史masked任务的PAIRED_TTS胜NAT_ONLY只能说明缺失音频时的辅助价值；旧phone-aligned TTS-mel residual也已有负结果。本轮复用自然重建模型，计算同模型correct-minus-zero条件增量，加到未遮挡natural mel上，检验是否存在绝对收益。它不是把TTS mel直接搬入natural，也不是新训练的全上下文生成头。

## Observations

- [status] concluded
- [question] 已有自然重建模型的内容条件增量能否在未遮挡natural基底上改善固定Wav2Lip的replacement指标？
- [protocol] 固定父new-confirmation的16条已观察记录、8个源组每组2条，三个固定seed在driver层先均值融合。四臂N/CORRECT/WRONG/SHUFFLE；WRONG是同音素错实例，SHUFFLE是K内时间列排列。固定a上限0.25及单点扰动上限0.5，范数控制经clip后再次审计。
- [scope] 缓存natural mel为前61440 samples/3.84s的80×308特征；这是完整未遮挡的短支持基线，不是整段长录音实验。评分mux仍保留整条原始natural PCM。旧masked模型内部仍用masked context，本轮只增加natural bypass。
- [budget] 最多66个93帧静态头像TFG视频、84个CPU评分cell；无训练、无新TTS、无vocoder。先mel/scorer parity、N重复与同视频错时敏感性控制，独立验收后再运行48个候选。
- [endpoint] 全帧224×224 frozen SyncNetScorer，88×31矩阵；U=range(30,58)，所有臂同支持和基线k_N。C/D/固定anchor收益均定义正值更好，8组等权bootstrap；自由offset分数上升不能替代anchor改善。
- [gate] CORRECT/N须C、D、anchor三项95%CI下界全>0，至少7/8组三项同正且mean ΔC>0.05。只有再胜WRONG和SHUFFLE且范数控制有效，才可称CONTENT_RESIDUAL_SIGNAL；只胜错配却未胜N为NO_INCREMENT_ESTABLISHED。
- [boundary] 所有结果保持replacement_confirmed=false、waveform_head_authorized=false、generalization_established=false、historical_shift_gate_repaired=false。这里内容指语音条件，不等同句义语义；错实例含声道等混杂，打乱含不连续效应。
- [preflight] 设计期仅CPU资产/公式预检：144个缓存driver hash通过，16条mask位置契约一致，每seed三条件checkpoint一致；规范公式下postclip范数比范围0.996–1.008，16/16通过预设范数范围。尚未新生成或新评分，不是科学收益结果。
- [implementation] 下游待实现scripts/experiments/wav2lip_natural_content_residual/及独立validator；按tasks执行并阶段自审，更新同一BM笔记而非新建重复记录。
- [report] openspec/changes/probe-wav2lip-natural-content-residual/{proposal.md,design.md,tasks.md,specs/wav2lip-natural-content-residual/spec.md}

- [run] 实际run为 `runs/wav2lip_natural_content_residual_20260908_v2`；Stage A生成18个视频并完成36个评分cell，独立validator PASS，Stage B未执行。
- [result] parity 2/2通过；N重复2/2通过；固定natural anchor delay损伤 bootstrap mean=+1.162568，95% CI [+0.844647,+1.482466]，8/8组为正。
- [result] delay offset差异满足预设 `-5±1` 的记录为13/16，低于14/16门槛，因此Stage A为 `CONTROL_FAILED`，没有运行CORRECT/WRONG/SHUFFLE候选。
- [conclusion] 本轮没有测试出内容残差相对natural的收益，不能判定内容路线无效；结果也没有确认replacement、修复historical shift gate或评分流程问题。工程与媒体/评分链路通过，失败是预注册控制门禁。
- [report] 终态与数字见 `runs/wav2lip_natural_content_residual_20260908_v2/{control_analysis.json,control_validation.json,final.json,result.md}`。

## Relations

- follows [[Wav2Lip historical shift rescore 2026-09-08]]
- relates_to [[LRS3 masked TTS trajectory-specificity diagnosis]]
- contrasts_with [[LRS3 Wav2Lip phone-aligned mel causal NO-GO]]
- relates_to [[LRS3 masked TTS TFG confirmation]]

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 用户要求解释shift结论并设计下一阶段；创建小型OpenSpec，完成缓存/公式只读预检，实验待实施 | September 8, 2026 | user |
| 按OpenSpec执行Stage A；18视频/36评分完成，控制门禁失败，Stage B按预注册规则停止；工程实现与独立validator通过 | September 8, 2026 | user |
