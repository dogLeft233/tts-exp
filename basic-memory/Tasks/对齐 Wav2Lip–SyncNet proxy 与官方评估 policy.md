---
title: 对齐 Wav2Lip–SyncNet proxy 与官方评估 policy
type: task
permalink: tts-exp/tasks/对齐-wav2-lip-sync-net-proxy-与官方评估-policy
assigned_to: downstream-agent
context: 'Execute repository OpenSpec change openspec/changes/align-wav2lip-syncnet-proxy-policy
  in worktree worktree-lrs3-wavlm-resynthesis-50. Engineering parity and fixed-sample
  objective-transfer now pass: decision ALLOW_FIXED_SAMPLE_DIAGNOSTIC and transfer
  decision ALLOW_NEW_A1_PILOT. A new small A1 pilot may be designed, but scientific
  replacement promotion remains unproven. Keep A2/A3, scale-up, decoder unfreezing,
  sealed-test selection, and extension of the existing A1 blocked. Do not overwrite
  historical runs, upload local data, or touch the unrelated Mandarin YAML deletion.'
current_step: 9
description: Implement and validate the OpenSpec change that aligns the differentiable
  Wav2Lip–SyncNet training proxy with the frozen official strict replacement evaluation
  policy.
parent_task: Wav2Lip replacement-aware cross-attention v2 修复
priority: high
started: '2026-08-27'
status: open
steps:
- Lock baseline and red-capable no-training differential fixture
- Implement canonical render-policy manifest and fail-closed validation
- Align original source frame mapping and official Wav2Lip boxes
- Implement official-forward differentiable-backward compositor
- Apply post-compositor frozen official SyncNet tracking/crop geometry
- Canonicalize target curve index and offset conversion
- Pass engineering parity and focused validation gates
- Conditionally run fixed-sample 20-step objective transfer
- Issue decision and synchronize OpenSpec/BM tasks
tags:
- task
- openspec
- wav2lip
- syncnet
- proxy-parity
- replacement
- lrs3
---

# 对齐 Wav2Lip–SyncNet proxy 与官方评估 policy

## Observations
- [description] 按 OpenSpec change `align-wav2lip-syncnet-proxy-policy` 完成 training proxy 与 frozen official Wav2Lip/SyncNet strict endpoint 的 source/frame/box/compositor/tracking/offset policy 对齐；工程 parity 与 fixed-sample objective-transfer 已通过，但 scientific replacement promotion 仍未证明。
- [status] open
- [assigned_to] downstream-agent
- [current_step] 9
- [decision] `ALLOW_FIXED_SAMPLE_DIAGNOSTIC`；fixed-sample transfer 为 `ALLOW_NEW_A1_PILOT`，只允许设计新的小规模 A1 pilot。
- [parent_task] [[Wav2Lip replacement-aware cross-attention v2 修复]]
## Source of truth
下游 implementation 已按该 OpenSpec change 完成主要工程 gates；本节仍是 normative source of truth，不得以 BM 摘要扩大实验范围：

`openspec/changes/align-wav2lip-syncnet-proxy-policy/`

包含：

- `proposal.md`：问题、范围、新 capability 与 impact；
- `specs/wav2lip-syncnet-proxy-policy/spec.md`：normative SHALL/MUST requirements 与 testable scenarios；
- `design.md`：canonical manifest、official-forward/surrogate-backward compositor、post-compositor SyncNet crop、offset convention 和分层 gate 的技术决策；
- `tasks.md`：逐项执行清单，已按实际完成状态同步，未完成证据项保持未勾选；
- `.openspec.yaml`：change metadata。

OpenSpec planning artifacts 已通过 strict validation。当前实现的 engineering decision 为 `ALLOW_FIXED_SAMPLE_DIAGNOSTIC`，fixed-sample objective-transfer decision 为 `ALLOW_NEW_A1_PILOT`。这只解锁新小规模 A1 pilot 的设计，不解锁 scientific promotion、A2/A3 或 scale-up。
## Repository state and locality

工作仓库/worktree：

`/home/wjj/tts-audio/tts-exp/.claude/worktrees/lrs3-wavlm-resynthesis-50`

branch：

`worktree-lrs3-wavlm-resynthesis-50`

开始时必须重新确认 git status。最近状态包含：

```text
D  data/wav2sem_analysis/mandarin_viseme_map.yaml
?? scripts/experiments/lrs3_syncnet_finetune/
?? tests/experiments/lrs3_syncnet_finetune/
?? openspec/
```

约束：

- 不恢复、覆盖或提交无关 Mandarin YAML 删除；
- `scripts/experiments/lrs3_syncnet_finetune/` 与 tests 可能仍未跟踪，不得假设 main 已包含；
- CodeGraph 索引此前来自原 checkout；若提示 worktree mismatch，以 worktree-local source 为准；
- 所有新 diagnostic 写入 worktree-local 新唯一 `tmp/` 目录；不覆盖历史 run；
- 不上传本地音频、视频或数据到第三方服务。

## Why this task exists

固定 box/tensor 条件下已经确认：

```text
Wav2Lip mel parity                PASS
mel chunk starts/count           PASS
6-channel face input             PASS
raw generated patch              PASS
uint8 compositor                 PASS
SyncNet MFCC/embeddings           PASS
31-point distance curve          PASS
source/crop/box policy parity     FAIL
locked target offset alignment   FAIL
```

当前训练路径：

```text
crop_video_path
→ 224×224 crop frames
→ static full-frame boxes [0,224,0,224]
→ differentiable Wav2Lip
→ direct SyncNet forward_lip
```

strict official endpoint：

```text
original face_path
→ official Wav2Lip detector
→ sample detector failure
→ static-first-frame fallback box [2,161,61,174]
→ official uint8 compositor
→ codec
→ untouched natural PCM mux
→ official SyncNet detector/tracker/crop
→ official SyncNet score
```

已测 source/geometry 差异不是微小数值噪声：original-fixed 与 crop-fixed official output MAE 约 `11.57/255`；original static-fallback 与 original full-frame forced-box MAE 约 `20.60/255`。fixed-sample 20-step objective 在旧 proxy 与 original forced-box path 上方向同向，但 baseline argmin 分别为 `+1` 与 `+3`，因此不能 scientific promotion。

主要 decision artifact：

`tmp/lrs3_a1_proxy_parity_decision_20260827/decision_report.json`

当前 decision 为 `FIX_PROXY`。

## Execution steps
以 OpenSpec `tasks.md` 为完整 checklist；本次执行已完成并验证以下阶段：

1. [x] 锁定 baseline、hash 和 worktree-local red-capable no-training differential fixture；确认旧 `crop_video_path + static_full_frame` 与 canonical original-source/static-fallback policy 确实不同。
2. [x] 实现 canonical render-policy manifest 及 fail-closed provenance validation；固定 frame/audio identity、box trace、chunk starts、SyncNet geometry、31-point target curve 和 failures。
3. [x] 对齐 original source frame mapping 和 official detector/static-first-frame fallback boxes；static fallback 使用 whole-input first frame repeated semantics。
4. [x] 实现 official-forward / differentiable-backward compositor；mel、face input、raw patch、pre-encode compositor parity 通过，且保持有限非零梯度。
5. [x] 在 compositor 后重放 frozen official SyncNet reference crop geometry；candidate tracker 与 natural reference 的 overlap geometry 已审计，保留 tracker span 差异 caveat。
6. [x] 统一 `curve index ↔ internal offset ↔ official reported offset`，重算 natural untouched-audio target curve；proxy 与 instrumented pre-encode curve parity 通过。
7. [x] 完成 no-training four-layer engineering parity 与 decision：`ALLOW_FIXED_SAMPLE_DIAGNOSTIC`，`failures=[]`，`first_divergence=null`。
8. [x] 在 engineering gate 通过后完成同一 fixed sample、query、TTS context、checkpoint、optimizer、LR、20-step objective transfer；结果为 `ALLOW_NEW_A1_PILOT`。
9. [ ] 完成最终 validation 命令、OpenSpec checkbox 与 BM 父/子任务同步；当前仍需运行 strict validation、ruff、compileall 并保留 1.3、3.3、4.3、5.4 的未完成证据项。

## Current decision

- Engineering parity：`ALLOW_FIXED_SAMPLE_DIAGNOSTIC`。
- Objective transfer：proxy 与 official target distance 均下降，Sync-C 均上升，target offset 未恶化；`ALLOW_NEW_A1_PILOT`。
- Scientific promotion：未通过或未证明。不得把 fixed-sample result 当作 cohort-level untouched-natural-audio replacement gain。
- 下一步最多设计新的小规模 policy-aligned A1 pilot；不得直接运行 A2/A3、延长现有 A1、扩大 cohort、解冻模块或使用 sealed test 选超参。

## Hard gates

### Engineering gate

必须同时满足：

- source frame indices/hashes、box trace、chunk starts/count 完全对齐；
- mel max absolute error `<=1e-4`；
- pre-encode official-forward compositor 在预注册容差内一致；
- candidate waveform 与 adapter gradients 非空、有限、非零；
- post-compositor SyncNet crop 重放通过；
- direct 31-point curve max absolute error `<=1e-4`；
- proxy 与 instrumented pre-encode path 的 argmin 和 locked target curve entry 完全一致；
- summary 显式包含 `status=complete`、`failures=[]`、hashes 和 `first_divergence`。

任一失败：decision 保持 `FIX_PROXY`，不得运行 fixed-sample overfit 或新训练。

### Conditional objective-transfer gate

只有 engineering gate 通过后，固定 sample/query/TTS context/checkpoint/optimizer/LR/20 steps；要求 step 0→20：

- proxy target distance 下降；
- 相同 official path target distance 下降；
- proxy Sync-C 上升；
- 相同 official path Sync-C 上升；
- target offset 不恶化。

失败则 `FIX_PROXY` 或 `REDESIGN_OBJECTIVE`；通过最多允许 `ALLOW_NEW_A1_PILOT`，不自动允许 A2/A3 或 scale-up。

### Frozen scientific endpoint

唯一主 endpoint 不变：

```text
candidate audio
→ frozen official Wav2Lip
→ candidate-driven video
→ replace with untouched natural PCM
→ frozen official SyncNet
```

相对 natural-video + natural-audio，最终 promotion 必须同时满足：Sync-C mean/median 上升，Sync-D mean/median 下降，joint wins > 50%，target offset 不恶化，source-group cluster-bootstrap 95% CI 支持正确方向。candidate self-pair 只能作为 co-adaptation diagnostic。

## Explicit prohibitions

- 不直接运行 A2 或 A3；
- 不延长当前 A1；
- 不扩大 cohort；
- 不解冻 WavLM、HiFi-GAN、Wav2Lip 或 SyncNet；
- 不修改 loss、LR 或 adapter architecture 来掩盖 policy mismatch；
- 不移动 strict evaluator 去匹配 crop/full-frame proxy；
- 不以 candidate self-pair 代替 untouched-natural-audio replacement；
- 不补零、重复帧、伪造 provenance 或放宽 coverage；
- 不覆盖历史 run，不上传本地数据。

## Required final report

下游最终报告至少包含：

1. reproduction command 和 red/green output；
2. first divergence；
3. Wav2Lip input/patch/compositor parity table；
4. SyncNet tracking/crop/31-point curve table；
5. gradient evidence；
6. fixed-sample result（仅在 gate 允许时）；
7. confirmed/refuted/unresolved hypotheses；
8. `FIX_PROXY`、`REDESIGN_OBJECTIVE` 或 `ALLOW_NEW_A1_PILOT` decision；
9. artifact paths、branch/commit、测试命令与准确结果；
10. OpenSpec checkbox 与 BM 父/子任务更新位置。

## Final engineering evidence

- Worktree-local decision artifact：`tmp/lrs3_proxy_policy_alignment_20260827/engineering_decision.json`。
- No-training fixture：`fixture_summary.json`，`status=complete`、`failures=[]`、`first_divergence=null`；canonical source 为原始 `face_path`，96 requested frames，static fallback box `[2,161,61,174]`，effective Wav2Lip frame count 93。
- Wav2Lip boundary parity：mel MAE `0`、face-input MAE `0`、raw patch max error `3.8743e-06`、pre-encode compositor max error `1` uint8 level。
- Direct SyncNet parity：step-0 curve max error `2.0981e-05`；objective-transfer 五个 evaluation points 的最大 curve error `4.3392e-05`；proxy/official argmin 与 locked target entry 一致。
- Gradient：adapter grad norm `36.4733`；candidate waveform gradient finite 且 nonzero。
- Tracker caveat：natural span `[0,101]`、candidate span `[0,94]`；95-frame overlap IoU mean `0.8888`、median `0.8842`、minimum `0.7838`。这不是 exact tracker identity，已作为 unresolved cohort-scale caveat 保留。
- Fixed-sample transfer：20 steps、LR `1e-5`、WavLM/HiFi-GAN/Wav2Lip/SyncNet frozen；target distance delta proxy `-0.31864`、official `-0.31862`；Sync-C delta proxy `+0.08207`、official `+0.08207`；target offset 未恶化。
- Decision：`ALLOW_NEW_A1_PILOT` 仅表示允许设计新的小规模 A1 pilot；不代表 scientific replacement promotion，不允许 A2/A3、scale-up、延长旧 A1、decoder unfreeze 或 sealed-test selection。
- Validation status：OpenSpec strict validation、targeted changed-file ruff 和 compileall 已通过；full pytest 为 `186 passed, 3 skipped, 12 failed`，12 个失败全部来自保留删除的 `data/wav2sem_analysis/mandarin_viseme_map.yaml`，未恢复或修改该文件。仍未单独保存 1.3 red output、3.3 source-only/box-only、4.3 codec 分层和 5.4 zero-residual/step-level track artifacts。

## Relations

- parent_task [[Wav2Lip replacement-aware cross-attention v2 修复]]
- follows [[A1 Wav2Lip–SyncNet proxy parity 与 TTS 条件归因]]
- constrained_by [[tts-exp|Wav2Lip replacement is primary objective]]
- repairs [[Wav2Lip proxy-to-official parity gate]]
