---
title: handoff_lrs3_natural_to_tts_bridge
type: note
permalink: tts-exp/imported-legacy/repo-root/tmp/handoff-lrs3-natural-to-tts-bridge
---

# Handoff：LRS3 natural-to-TTS bridge 后续策略

## 当前任务焦点

判断 natural-to-TTS bridge confirmation 实验完成后，是否继续最初计划中的下游训练/音频头开发。当前结论是：先不要进入训练或部署，先处理 control calibration 问题。

## 已完成工作

已按照以下 OpenSpec 完成并验证实验：

- `openspec/changes/confirm-lrs3-natural-to-tts-bridge/`
- run root：`runs/lrs3_natural_to_tts_bridge_confirmation_20260904/`
- terminal artifact：`runs/lrs3_natural_to_tts_bridge_confirmation_20260904/04_final/final.json`

实验固定使用 22 条 fit-only records、22 个 source groups、4 个 audio arms、88 个视频和 132 个 SyncNet cells。Stage 00–04 均已完成，独立 validator、focused tests、discovery regression tests、Ruff、Python compilation 和 OpenSpec validation 均通过。

## 核心结果

- `N_REPEAT` repeatability control：通过，offset agreement `22/22`。
- `BRIDGE_075` movement：通过，`22/22` 达到阈值；mean progress `0.5118`，95% CI `[0.4539, 0.5685]`。
- `BRIDGE_075` 换回 natural audio 后的 replacement compatibility：描述性结果通过，未观察到明显 degradation；但这不是显著 gain 证明。
- primary SyncC gain：未通过；mean `+0.0315`，95% CI `[-0.0342, 0.1047]`，区间跨过 0。
- `LOCAL_SWAP` own-audio validity 和 sensitivity control：均失败。它配回自身音频时相较 natural baseline 已明显变差，也没有呈现 spec 预期的 replacement damage 模式。

最终决策：

```text
engineering_decision = GO
scientific_decision = CONTROL_FAILED
reference_conditioned_audio_head_spec_eligible = false
```

## 对原计划的判断

不要直接按照原计划进入 reference-conditioned audio-head training、identity transfer、deployment 或 held-out evaluation。原计划需要在 control 通过、bridge 得到科学确认后才进入下游；当前 eligibility flag 为 false。

这不是判定 bridge 完全失败，而是当前 endpoint calibration 未成立，且本次没有统计证据证明 bridge 比 natural baseline 有 SyncC 提升。

## 下一步建议

1. 先审计 `LOCAL_SWAP` control 的实现与解释，不重写当前 run：
   - 检查 `V_LOCAL_SWAP/A_LOCAL_SWAP` 与 `V_LOCAL_SWAP/A_N` 的 cell wiring；
   - 检查 strict mux 后 decoded PCM 是否保持预期 byte identity；
   - 检查两个视频是否来自同一 record、同一 face geometry 和同一 Wav2Lip binding；
   - 检查 SyncNet reference 参数是否与 audio cell 语义一致；
   - 分析 local swap 是否过强，导致 own-audio validity 本身失效。
2. 如果发现工程错误，只修复工程错误并使用新的 run root 重跑，不修改当前结果。
3. 如果实现正确但 local swap 过强或不适配 endpoint，另写新的 OpenSpec，设计更温和但不能由单一 global offset 修复的 local timing control；不能覆盖当前 run。
4. 新 control 通过后，再重新评估两个不同目标：
   - replacement compatibility：换回 natural audio 后不明显变差；
   - useful SyncC gain：相较 natural baseline 获得统计确认的提升。

## 关键约束

- 不访问 sealed validation/test media。
- 不根据 score、render 或 codec 结果筛选 records。
- 不修改 parent/discovery artifacts。
- 当前阶段不训练、不 fine-tune、不部署、不做 held-out evaluation。
- 不重新运行 MFA/DTW，不生成新的 TTS，除非新的 OpenSpec 明确授权。
- 不 commit、push 或创建 PR，除非用户明确要求。

## 参考位置

实验实现：`scripts/experiments/lrs3_natural_to_tts_bridge_confirmation/`

实验测试：`tests/experiments/lrs3_natural_to_tts_bridge_confirmation/`

最终 artifact hash：`df0ca9767e70ccc384c86c1da23243c53fa609be12abd1dda20f9075b6732c6e`

## Suggested skills

- `openspec`：设计或审计下一份 control calibration 实验规范。
- `basic-memory`：读取并更新项目级实验记忆。
- `codegraph`：追踪 control cell wiring、mux、render 和 scoring 的调用关系。
- `pytest` / 项目测试流程：验证修复后的协议与 control 实现。