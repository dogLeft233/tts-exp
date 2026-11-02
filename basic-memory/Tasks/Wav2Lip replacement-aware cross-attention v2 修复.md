---
title: Wav2Lip replacement-aware cross-attention v2 修复
type: task
permalink: tts-exp/tasks/wav2-lip-replacement-aware-cross-attention-v2-修复
assigned_to: claude
context: A1-200 audit and endpoint-v1 fixed-sample failure are complete in worktree
  worktree-lrs3-wavlm-resynthesis-50. Endpoint-v2 redesign is preregistered at openspec/changes/redesign-a1-endpoint-objective-v2/
  and delegated to BM child A1 positive-margin full-curve objective v2. Keep scientific
  untouched-natural-audio promotion blocked; child completion cannot remove the blocker
  without a separate strict cohort audit.
current_step: 9
description: Rebuild and repair replacement-aware TTS cross-attention training through
  frozen Wav2Lip and SyncNet, with full scorer parity and strict natural-audio replacement
  gates.
priority: high
started: '2026-08-27'
status: blocked
steps:
- Recover and version the historical cross-attention protocol and red reproducer
- Freeze source-group-disjoint cohorts and official zero-residual baselines
- Rebuild full-frame differentiable Wav2Lip and pass curve-level parity
- Implement and test full-offset scorer-aligned replacement objective
- Pass gradient, freezing, exact-length, and numerical smoke gates
- Run A0/A1/A2 one-change-at-a-time pilot ladder
- Run mel trust-region A3 only if the preregistered shortcut trigger fires
- Run strict official replacement four/five-cell audit for each surviving variant
- Apply stop/continue decision gate without scaling failed variants
- Scale one frozen winner and run a single sealed source-group audit
tags:
- task
- wav2lip
- syncnet
- cross-attention
- replacement
- lrs3
blockers:
- H1 baseline-anchored full-curve objective fixed-sample gate failed; redesign is
  required before any new pilot.
- Scientific untouched-natural-audio replacement gain remains unproven; parent blocker
  must remain.
---

# Wav2Lip replacement-aware cross-attention v2 修复

## Context

本任务研究用 frozen WavLM layer 6 + HiFi-GAN 生成 candidate audio，再经 frozen official Wav2Lip 生成 candidate video，并在 candidate video 音轨替换为 untouched natural PCM 后，能否相对 natural video + natural audio 获得真实 SyncNet V2 replacement gain。唯一 scientific endpoint 是 untouched-natural-audio replacement；candidate video + candidate audio 只能作为 self-consistency diagnostic。

当前工作在 isolated worktree `worktree-lrs3-wavlm-resynthesis-50`。历史 A1-200 已完成完整 objective-transfer 审计，H1-only repair OpenSpec 已执行到 fixed-sample gate，但 gate 失败。因此 parent task 仍 blocked。

## Confirmed diagnosis

- **H1 — CONFIRMED_PRIMARY:** A1 只优化 locked target entry `d(s*)`，而 checkpoint selector 也只最小化 target distance；target proxy 变好不代表完整 Sync-C/Sync-D endpoint 变好。
- **H2 — CONFIRMED_CONTRIBUTOR:** residual 幅度增大后出现 nonlinear waveform collapse；alpha 约 `0.25+` 时 C/D 明显退化。
- **H3 — CONFIRMED_CONTRIBUTOR:** adapter 对 paired、cross-group shuffled 和 reversed TTS context 功能性不敏感。
- **H4 — CONFIRMED_CONTRIBUTOR, limited scope:** patch→uint8 compositor surrogate 在初始方向审计中出现 2/3 disagreement。
- **H5 — REFUTED_AS_PRIMARY:** 96-frame segment 已失败，full-file 只放大失败。
- **H6 — REFUTED:** offset、policy span、source/crop identity、scorer 和 PCM replacement round-trip 均通过。

历史 A1-200 strict endpoint 为 scientific NO-GO：replacement joint wins `0/40`，Sync-C 和 Sync-D benefit 均显著朝错误方向变化。该结果不是数据不足或 codec/offset 错误的证据。

## Completed implementation and checks

- canonical render policy、original-source/static-fallback Wav2Lip policy、official-forward/surrogate-backward path、31-point curve 和 exact PCM replacement 已实现并通过既有 parity/audit tests。
- 新增 full-curve selector：只有 `Sync-C_delta > 0` 且 `Sync-D_benefit > 0` 的 checkpoint 才 eligible；无 eligible 时回退 step 0；target distance 只能作为 tie-breaker。
- selector 已在历史 A1-200 cached history 上拒绝 step 2000，并记录 `NO_ELIGIBLE_ENDPOINT_CHECKPOINT`；历史 red loop 仍保持 `OBJECTIVE_TRANSFER_FAILURE`。
- 新增 isolated `endpoint` objective，使用 natural-video + untouched-natural-audio reference curve：
  ```text
  L = relu(Sync-D - D_ref) + relu(C_ref - Sync-C)
  ```
- active hinge 的 non-finite/zero gradient fail closed；两个 hinge 均满足时记录 `ENDPOINT_CONSTRAINTS_SATISFIED`。
- 既有 target-entry objective 保持默认，未修改 frozen WavLM、HiFi-GAN、Wav2Lip、SyncNet、STE、render policy、residual scale 或 TTS context。
- 实验测试最终为 `63 passed`；changed files targeted Ruff 通过；`openspec validate repair-a1-objective-transfer --strict` 通过。

## H1 fixed-sample result

有效 run artifact：

- Training: `tmp/lrs3_policy_a1_endpoint_20_20260828_retry2/train/summary.json`
- Official step-0 audit: `tmp/lrs3_policy_a1_endpoint_20_20260828_retry2/official_validation_one/summary.json`
- Official step-20 audit: `tmp/lrs3_policy_a1_endpoint_20_20260828_retry2/official_validation_one_last/summary.json`
- Gate report: `tmp/lrs3_policy_a1_endpoint_20_20260828_retry2/fixed_sample_gate.json`

proxy validation aggregate：

```text
step 0:  Sync-C 7.69709, Sync-D 7.46527
step 20: Sync-C 7.67147, Sync-D 7.48446
C delta: -0.02562
D benefit: -0.01919
selector: step 0 fallback
```

step-20 official strict audit（1 validation record, five cells）：

```text
replacement Sync-C delta: -0.343
replacement Sync-D benefit: -0.458
replacement joint wins: 0/1
PCM exactness: 5/5
AV offset: -2
```

## Current decision

H1 selector 和 objective seam 工程实现正确，但 baseline-anchored full-curve objective 没有 transfer 到 replacement endpoint。fixed-sample gate 正式决策为：

```text
REDESIGN_OBJECTIVE
```

这不是 `SCIENTIFIC_NO_GO` 对整个 WavLM→HiFi-GAN→Wav2Lip 方向的最终否定，而是当前 objective candidate 未通过 transfer gate。不得因为 target distance、self-pair 或 proxy 局部指标改善而 promotion。

## Decision gate

- [x] 历史 A1-200 objective-transfer red loop 完成并保持可复现。
- [x] H1 cached selector 完成，step 2000 被拒绝并 fallback step 0。
- [x] H1 full-curve endpoint objective seam 完成并通过 finite/gradient tests。
- [x] fixed-sample 20-step proxy 与官方 one-record replacement evidence 完成。
- [x] fixed-sample decision = `REDESIGN_OBJECTIVE`。
- [ ] 新 objective redesign：尚未开始，必须先明确 bounded/transfer-valid 数学目标。
- [ ] 新 A1 pilot：blocked。
- [ ] A2/A3、增加 steps、扩大 cohort：blocked。
- [ ] decoder/frozen-module unfreezing：blocked。
- [ ] sealed-test hyperparameter selection：blocked。
- [ ] scientific untouched-natural-audio promotion：blocked。

## Next action

设计新的 objective candidate，但不得直接继续训练。下一版必须先解决当前候选的可优化性与 endpoint transfer 问题，并通过新的 objective seam tests：

1. 保留 selector-only gate 和 strict evaluator 不变。
2. 重新定义 full-curve objective，避免 target-only、无界 `-Sync-C` 和仅以自然 baseline hinge 造成大量 zero-gradient 的问题。
3. 对固定 synthetic curves 验证 C/D 各方向的 gradient，明确 active/inactive hinge 行为。
4. 在 fixed validation sample 上重新执行最多 20 steps；proxy 与 official replacement endpoint 必须同向改善，否则继续 `REDESIGN_OBJECTIVE`。
5. 只有 fixed-sample gate 通过后，才可创建独立 pilot spec；不得自动运行 A2/A3 或 scale-up。

H2 residual、H3 TTS conditioning、H4 compositor STE 仍应串行处理，不能混入下一次 H1 objective redesign。parent blocker 只有在 untouched-natural-audio strict promotion 全部通过后才能移除。

## Observations

- [decision] 当前 H1 candidate 不是全套链路的最终否定，而是明确的 `REDESIGN_OBJECTIVE`；不能用额外 steps 或 cohort 规模掩盖失败。
- [insight] selector repair 成功阻止了历史 step-2000 proxy checkpoint 被接受，但 selector 修复本身不产生模型增益。
- [insight] baseline-anchored objective 的 implementation seam 正常，然而 20-step proxy C/D 与官方 replacement C/D 均未改善。
- [constraint] untouched natural PCM replacement 是唯一 scientific promotion endpoint；self-pair 不能替代它。
- [constraint] 历史 artifacts、失败 retry artifacts 和当前 parent blocker 均保留；不覆盖历史 run、不上传本地实验数据、不恢复无关 Mandarin YAML 删除。
- [status] blocked

## Relations

- implements [[repair-a1-objective-transfer]]
- informed_by [[A1-200 objective-transfer 漏洞审计]]
- constrained_by [[tts-exp|Wav2Lip replacement is primary objective]]
- blocked_by [[对齐 Wav2Lip–SyncNet proxy 与官方评估 policy]]

## Active child task

Objective redesign has been preregistered in repository OpenSpec `openspec/changes/redesign-a1-endpoint-objective-v2/` and BM child [[A1 positive-margin full-curve objective v2]]. The child may proceed through synthetic and fixed-sample gates, but this parent remains `blocked`; completing the spec or one-record diagnostic does not remove the scientific untouched-natural-audio blocker.

- delegates_to [[A1 positive-margin full-curve objective v2]]
