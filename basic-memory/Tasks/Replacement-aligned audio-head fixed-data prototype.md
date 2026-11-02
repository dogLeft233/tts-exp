---
title: Replacement-aligned audio-head fixed-data prototype
type: task
permalink: tts-exp/tasks/replacement-aligned-audio-head-fixed-data-prototype
assigned_to: downstream-agent
current_step: 6
description: Implement and run the lean fixed-data replacement-aligned CrossAttentionResidual
  prototype specified on 2026-08-30.
started: '2026-08-30'
status: done
steps:
- Implement target-margin loss and trust hinge
- Implement prototype runner and frozen-chain seam
- Connect A/B/U/F official evaluator
- Run P0 and one-record P1
- Optionally run exact-four-record P2 after P1 pass
- Run verification and handoff
tags:
- task
- lrs3
- wav2lip
- syncnet
- fixed-data
- prototype
---

# Replacement-aligned audio-head fixed-data prototype

## Observations
- [description] 按 `openspec/changes/prototype-replacement-aligned-audio-head/` 实现并运行精简算法原型：保留 natural-query/TTS-key-value CrossAttentionResidual 与 frozen WavLM/HiFi-GAN/Wav2Lip/SyncNet，使用数学充分的 full-offset target-margin loss，先单记录再可选四记录验证。
- [status] done
- [close_date] 2026-08-30
- [result] P1 = FIXED_DATA_PROTOTYPE_ONE_RECORD_PASS（单记录 20-step adapter，official U/F 两 repeat 均 favorable：d_gain +0.128/+0.194，c_gain +0.012/+0.030）。P2 = FIXED_DATA_PROTOTYPE_SHARED_NOT_CONFIRMED（共享 100-step adapter，0/4 official wins：3× PROXY_OFFICIAL_TRANSFER_FAILURE，1× OFFSET_POLICY_MISMATCH fail-closed）。代理 gate 偏保守（P1 proxy 报 NO_PROXY_GAIN 但 official pass）。完整记录见 openspec/changes/prototype-replacement-aligned-audio-head/report.md。
- [assigned_to] downstream-agent
- [current_step] 6
- [decision] H/D/V/T/S、H-only authorization和v3 optimizer/no-training gates不再是当前原型前置条件。
- [natural_direct_comparison] 追加测试（2026-08-30，run6/run7）：自然驱动对照 A（natural PCM 驱动 Wav2Lip 渲染 + 同 natural PCM 评分）两次独立渲染 D=6.7953/6.7812、C=5.6924/5.6097。训练后 replacement F D=6.7253、C=5.7190（4 次渲染字节一致，完全确定）→ 在过拟合记录上 F 优于自然：D 低 0.056–0.070（超过 A 渲染波动 ~0.014），C 高 0.027–0.109（A 的 C 波动 ~0.083，幅度部分在噪声内）。未训练 pristine U（D=6.8534–6.9195）差于自然。仅此固定记录、无泛化含义。
- [claim_boundary] 训练与评分可以使用相同固定记录；只证明 constructability，不声称泛化。

## Steps
1. [ ] 实现并测试 `replacement_target_margin_loss_components` 与 log-mel trust hinge。
2. [ ] 实现独立 prototype runner、checkpoint gate和frozen-chain gradient seam。
3. [ ] 接通 A/B/U/F official Wav2Lip→natural PCM replacement→SyncNet full-chain evaluator。
4. [x] 运行 P0 tests和单记录20-step P1；输出唯一 terminal status（FIXED_DATA_PROTOTYPE_ONE_RECORD_PASS）。
5. [x] 仅在P1通过后选择是否运行fresh exact-four-record 100-step P2（已运行；FIXED_DATA_PROTOTYPE_SHARED_NOT_CONFIRMED）。
6. [x] 运行 regression/Ruff/compile/OpenSpec checks并交付 curves、artifacts和decision。

## Context
- Source of truth: `openspec/changes/prototype-replacement-aligned-audio-head/`。
- 核心 loss 对 target coordinate施加绝对 D goal，并对全部30个错误offset施加 C goal gap；loss为零时可证明target是唯一argmin、hard D低于step0且hard C高于step0。
- primary endpoint始终是candidate驱动Wav2Lip后的视频，与untouched natural PCM配对并由official SyncNet V2保存完整31-point curve。
- A=natural direct，B=wrong utterance，U=pristine resynthesis，F=trained adapter；primary paired comparison仅F vs U。
- 旧 H-only工作保留为历史基础设施，见 [[H-only historical endpoint sensitivity assay]]。

## Relations
- replaces [[H-only historical endpoint sensitivity assay]]
- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]
- relates_to [[Direct waveform upper bound]]

## Local readiness audit — 2026-08-30
- [data] `tmp/lrs3_policy_a1_200_20260828/policy_cohort` has 280/280 complete records across 29 source groups; every record has natural video/audio, paired 16 kHz mono TTS, crop, verified 96-frame/61,440-sample alignment, and >=96 Wav2Lip boxes.
- [models] Local WavLM, HiFi-GAN, Wav2Lip, SyncNet, and knn-vc paths all exist.
- [decision] P1 is fixed to `lrs3_6ORDQFh0Byw_00004` with B donor `lrs3_6StqiaPoS2U_00004`; P2 adds `lrs3_6VWPHKABRQA_00003` and `lrs3_6VnKV1sr5VQ_00002` for four distinct source groups.
- [clarification] The four newer raw records lack paired TTS and policy boxes, but this is not a data blocker because the complete 280-record local policy cohort is reusable now that H/D/V/T/S restrictions are abandoned.
