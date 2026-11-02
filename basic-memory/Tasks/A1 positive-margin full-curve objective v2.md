---
title: A1 positive-margin full-curve objective v2
type: task
permalink: tts-exp/tasks/a1-positive-margin-full-curve-objective-v2
assigned_to: downstream-agent
context: Implement the new OpenSpec at openspec/changes/redesign-a1-endpoint-objective-v2/
  in isolated A1 mode. The prior zero-margin endpoint objective failed fixed-sample
  proxy and official transfer and remains immutable. Objective-v2 uses frozen step-0/natural
  anchors, 1% positive hard goals, official-forward distributed-backward statistics,
  and bounded normalized hinges. Run synthetic and unchanged regressions before one
  max-20-step fixed-sample gate; no H2/H3/H4, A2/A3, scale-up, unfreezing, sealed-test
  tuning, or parent unblock.
current_step: 8
description: Implement and gate a bounded positive-margin official-forward full-curve
  objective after endpoint-v1 failed transfer.
parent_task: Wav2Lip replacement-aware cross-attention v2 修复
priority: high
started: '2026-08-28'
status: done
steps:
- Implement distinct objective-v2 identity and immutable dual anchors
- Implement preregistered official-forward distributed-backward bounded equations
- Pass deterministic synthetic hard-direction and gradient matrix
- Pass unchanged selector, red-loop, policy, coordinate, PCM, and H1-H6 regressions
- Run one fixed sample for at most 20 total steps in a unique artifact directory
- Run unchanged official step-0/step-20 five-cell replacement audit
- Apply conjunctive transfer decision and preserve parent blocker
tags:
- task
- wav2lip
- syncnet
- objective-redesign
- full-curve
- replacement
- lrs3
completed: '2026-08-28'
---

# A1 positive-margin full-curve objective v2

## Context

这是 blocked parent task [[Wav2Lip replacement-aware cross-attention v2 修复]] 的 H1 objective-redesign 子任务，对应 repository OpenSpec：

```text
openspec/changes/redesign-a1-endpoint-objective-v2/
```

前一 change `repair-a1-objective-transfer` 的 selector 修复有效，但 zero-margin objective

```text
relu(D-D_ref) + relu(C_ref-C)
```

在唯一允许的 fixed-sample 20-step gate 失败：proxy `C_delta=-0.02562`、`D_benefit=-0.01919`；official untouched-natural-audio replacement `C_delta=-0.343`、`D_benefit=-0.458`、joint wins `0/1`，PCM five-cell exact。结论保持 `REDESIGN_OBJECTIVE`；失败 artifact `tmp/lrs3_policy_a1_endpoint_20_20260828_retry2/fixed_sample_gate.json` 不得覆盖。

## Frozen redesign

新 objective 必须使用新的 identifier（provisional `endpoint_margin_st`），不能改写历史 `target_entry` 或 `endpoint` 语义。每个 exact slice 在优化前冻结：

- `c0`：pristine zero-output adapter 产生的 31-point candidate proxy curve；
- `r`：policy-aligned natural video + untouched natural audio 的 31-point curve。

Hard anchors/goals：

```text
D_anchor = min(D(c0), D(r))
C_anchor = max(C(c0), C(r))
rho_D = rho_C = 0.01
D_goal = 0.99 * D_anchor
C_goal = 1.01 * C_anchor
S_D = max(D_anchor, 1e-3)
S_C = max(C_anchor, 1e-3)
```

Forward 必须严格等于 official hard `D=min(curve)` 与 `C=median(curve)-min(curve)`；backward 使用 preregistered mean-normalized softmin 与 central-five sorted median surrogate，通过 straight-through correction 分散梯度。`tau=0.05*S_D`，不得 real-data tuning。

Bounded normalized loss：

```text
u_D = relu((D_st-D_goal)/S_D)
u_C = relu((C_goal-C_st)/S_C)
h(u) = u/(1+u)
L = h(u_D) + h(u_C)
```

因此每项位于 `[0,1)`，总损失位于 `[0,2)`；正 violation 保持正导数，两个 hard positive goals 都满足后才允许 zero gradient。禁止 target-entry、ranking、mel、residual、self-pair 或无界 `-Sync-C` 项。

## Steps
1. [x] 在 `syncnet_loss.py` 添加 distinct objective-v2 helper；保留 endpoint-v1 原样。
2. [x] 在 `replacement_train.py` 添加 isolated A1 dispatch、immutable dual-anchor provenance 与 fail-closed resume checks。
3. [x] 完成 OpenSpec 指定的 float64 synthetic matrix：official-forward identity、D/C one-active、both-active、both-satisfied、worse、permutation、scale、ties、invalid 与 hard finite-difference direction。
4. [x] 完成 candidate-waveform/adapter component/total gradient tests；active zero/non-finite/cancellation 必须失败。
5. [x] 运行 unchanged selector、historical red loop、policy/coordinate/PCM 与 H1–H6 regression suites。
6. [ ] 仅在全部 seam gates 通过后，对既有 fixed sample 运行一个新的 unique run，最多 20 total steps，报告 0/1/5/10/15/20。
7. [ ] 对 step 0/20 执行 unchanged official five-cell untouched-natural-PCM replacement audit。
8. [ ] 只有 proxy 与 official hard C 上升、hard D 下降、offset 不恶化且 selector 接受 post-step-0 checkpoint，才记录 `ALLOW_NEW_A1_PILOT_SPEC`；否则继续 `REDESIGN_OBJECTIVE`。

当前证据：
- objective seam + provenance tests：59 passed。
- LRS3 regression suite：79 passed。
- targeted Ruff：passed。
- `openspec validate redesign-a1-endpoint-objective-v2 --strict`：valid。
- full repository pytest：235 passed, 3 skipped, 12 failed；失败均来自已存在且明确保留的 `data/wav2sem_analysis/mandarin_viseme_map.yaml` 删除，不属于本 change。
- 尚未运行 fixed sample、official audit；parent scientific blocker 保持 blocked。
## Locked variables and blockers

- WavLM、HiFi-GAN、Wav2Lip、SyncNet 全部 frozen。
- render policy、boxes、crop、sample/split、optimizer、learning rate、segment length、residual scale、TTS context、STE 与 strict evaluator unchanged。
- H2/H3/H4 不能混入；A2/A3、额外 steps、换 fixed sample、扩大 cohort、sealed-test tuning、decoder unfreezing均禁止。
- candidate self-pair 只可 diagnostic；唯一 scientific endpoint 仍是 candidate-driven video + untouched natural PCM + official SyncNet V2。
- fixed-sample 通过只允许另写 pilot spec，不能自动训练或解除 parent blocker。
- 本地实验数据不得上传第三方；所有 artifact 使用 current-worktree-local unique directory；不得恢复或修改无关 Mandarin YAML deletion。

## Evidence contract

每一 gate 记录 command、source/policy/evaluator/checkpoint hashes、changed-variable declaration、dual-anchor provenance、full hard/surrogate curves、goals/hinges/gradients、selector、official scores、failure reason 和 final decision。所有 constants 已在 real-data run 前由 OpenSpec 固定；失败后不得调参重跑。

## Observations
- [description] 实现并验证 bounded positive-margin official-forward full-curve objective-v2；只在 synthetic、integrated 与 fixed-sample transfer gates 全部通过后允许撰写新 pilot spec。
- [status] done
- [assigned_to] downstream-agent
- [current_step] 8
- [completed] 2026-08-28
- [decision] 使用 step-0 与 natural reference 的 per-metric better anchor，并要求固定 1% improvement；不再接受 merely-not-worse constraint。
- [decision] forward 只认 unchanged official hard C/D，surrogate 仅承担 backward；soft improvement 不能替代 hard gate。
- [decision] 使用 bounded `u/(1+u)` positive hinge，避免无界 confidence reward 与 clipped active zero-gradient。
- [evidence] objective seam/provenance tests：59 passed；LRS3 regression suite：79 passed；targeted Ruff passed。
- [evidence] objective-v2 training artifact：`tmp/lrs3_policy_a1_endpoint_margin_st_20_20260828/train/summary.json`，validation steps `[0,1,5,10,15,20]`，dual-anchor entries `60`，20 optimizer steps complete。
- [evidence] proxy validation hard Sync-C `7.697092 → 7.693303`（失败，下降），hard Sync-D `7.465267 → 7.471979`（失败，上升），offset error `3.625 → 3.650`（失败）；selector `selected_step=0`，`eligible_candidates=[]`。
- [evidence] official step-0 artifact：`tmp/lrs3_policy_a1_endpoint_margin_st_20_20260828/official_step0/summary.json`；step-20 artifact：`tmp/lrs3_policy_a1_endpoint_margin_st_20_20260828/official_step20/summary.json`；两者均 five-cell complete、PCM 5/5 exact、AV offset `-2`。
- [evidence] official replacement raw Sync-C `7.015 → 7.045`、raw Sync-D `7.468 → 7.438`，方向相对 step-0 通过，但 replacement 仍未超过 natural baseline，且 proxy/selector conjunctive gate 失败。
- [decision] fixed-sample gate final decision `REDESIGN_OBJECTIVE`，reason `SURROGATE_DIRECTION_MISMATCH`；未创建 new pilot，未运行 A2/A3，未增加 steps，parent blocker 保持 blocked。
- [evidence] gate report：`tmp/lrs3_policy_a1_endpoint_margin_st_20_20260828/fixed_sample_gate.json`。
- [constraint] full repository pytest 的 12 failures 均为既有无关 `data/wav2sem_analysis/mandarin_viseme_map.yaml` deletion，未恢复或修改该文件。
- [constraint] parent scientific task remains blocked until a separate strict cohort audit proves untouched-natural-audio replacement promotion。
## Relations

- parent_task [[Wav2Lip replacement-aware cross-attention v2 修复]]
- implements [[redesign-a1-endpoint-objective-v2]]
- supersedes_candidate [[repair-a1-objective-transfer]]
- informed_by [[A1-200 objective-transfer 漏洞审计]]
- constrained_by [[tts-exp|Wav2Lip replacement is primary objective]]
