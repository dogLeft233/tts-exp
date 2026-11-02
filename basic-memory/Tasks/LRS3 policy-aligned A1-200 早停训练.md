---
title: LRS3 policy-aligned A1-200 早停训练
type: task
permalink: tts-exp/tasks/lrs3-policy-aligned-a1-200-早停训练
assigned_to: claude
context: 'Policy-aligned preparation, A1 training, and strict audit are complete.
  The worktree-local artifacts are under tmp/lrs3_policy_a1_200_20260828/. Policy
  cohort: 280 records with train/validation/test=200/40/40, 29 source groups, failures=[];
  full-input detector provenance version 3. Training completed 2000/2000 steps with
  validation target distance 15.92308→11.29079, best step 2000, early stopping not
  triggered. Strict test audit: 40 records, 4 source groups, 5 cells, 200 scores and
  200 muxes, PCM exact for all. Primary replacement endpoint is scientific NO-GO:
  Sync-C mean/median delta -4.289075/-4.358500; Sync-D benefit mean/median -2.678700/-2.675500;
  replacement wins 0/40 joint. Do not promote, scale up, run A2/A3, or unfreeze decoder;
  next work is objective/rendering linkage analysis.'
current_step: 7
parent_task: Wav2Lip replacement-aware cross-attention v2 修复
priority: high
started: '2026-08-28'
status: done
steps:
- 'Audit about 200 policy-valid records and source groups (complete: 475 eligible;
  200/40/40 candidate split)'
- Implement group-disjoint validation and early stopping
- Generate batch canonical policy artifacts with fail-closed backup selection
- Test and validate implementation
- Run A1-200 training
- Strict replacement audit best checkpoint
- Synchronize BM decisions
tags:
- task
- lrs3
- wav2lip
- syncnet
- early-stopping
completed: '2026-08-28'
---

# LRS3 policy-aligned A1-200 早停训练

## Observations
- [description] 使用约 200 条 LRS3 training records 训练 replacement-aware cross-attention adapter，加入 source-group-disjoint validation 与早停，并对 best checkpoint 做 untouched-natural-audio replacement audit。
- [status] done
- [assigned_to] claude
- [current_step] 7
- [constraint] 只训练 cross-attention residual adapter；WavLM、HiFi-GAN、Wav2Lip、SyncNet 全部冻结。
- [constraint] primary endpoint 仍为 candidate-driven video + untouched natural PCM；candidate self-pair 仅作诊断。
- [decision] A1-200 对 untouched-natural-audio replacement gain 为 scientific NO-GO；不进入 A2/A3、继续 scale-up 或解冻 decoder。
- [result] 40-record strict test audit 完整，replacement Sync-C mean/median delta = -4.289075/-4.358500，Sync-D benefit mean/median = -2.678700/-2.675500；两个 95% source-group cluster-bootstrap CI 均完全落在错误方向，joint wins = 0/40。
- [result] training validation target distance 从 15.92308 降至 11.29079，但该 proxy 改善没有转移到 strict replacement endpoint。
- [result] policy cohort 为 280 条记录，train/validation/test = 200/40/40，29 个 source groups，failures = []；full-input detector provenance 为 version 3。
- [result] 早停配置 patience=3、min_delta=0.001、validation_every=200、max_steps=2000；训练完成 2000 步，best step=2000，未触发早停。
## Steps
1. [x] 核对约 200 条 policy/provenance 完整记录及 source-group 分布。— 475/500 input-valid；冻结 200 train / 40 validation / 40 test。
2. [x] 实现固定 group-disjoint train/validation split 与 early stopping。— fixed protocol_split preserved; best/last checkpoints; patience=3 validation checks, min_delta=0.001。
3. [x] 批量生成/验证 official SyncNet crop、natural 31-point curve 与 official Wav2Lip box trace。— 280 records complete, 25 strict exclusions logged; final full-input detector trace version 3; 200/40/40, 29 groups, failures=[]。
4. [x] 添加 focused tests，运行 ruff/compile/pytest。— 21 focused tests passed; targeted ruff and compileall passed。
5. [x] 运行约 200 条训练记录的 A1，保存 best/latest checkpoint 和指标。— 280 records, 2000/2000 steps, validation target 15.92308→11.29079, no early stop, best step 2000。
6. [x] 对 best checkpoint 做 strict untouched-natural-audio replacement audit。— test split 40 records/4 source groups/5 cells；200 scores、200 muxes、failures=0、PCM exact=true；replacement Sync-C mean/median delta -4.289075/-4.358500，Sync-D benefit mean/median -2.678700/-2.675500，CI 均为错误方向，joint wins 0/40。
7. [x] 同步本任务与父任务决策。— 本任务 done；父任务保持 blocked，blocker 为 scientific untouched-natural-audio replacement gain not demonstrated；不运行 A2/A3、不继续 scale-up、不解冻 decoder。
## Context
Implementation is in `scripts/experiments/lrs3_syncnet_finetune/` in worktree `worktree-lrs3-wavlm-resynthesis-50`. Canonical policy parity and fixed-sample objective transfer passed. Existing 34-record strict A1 audit was a scientific NO-GO, so this A1-200 run was a scale diagnostic requested by the user, not automatic promotion.

Worktree-local artifacts:
- Input audit: `tmp/lrs3_policy_a1_200_20260828/input_audit.json` (475/500 input-valid across 43 groups).
- Policy cohort: `tmp/lrs3_policy_a1_200_20260828/policy_cohort/policy_manifest.json` and manifests; 280 complete records, train/validation/test=200/40/40, 29 groups, 25 strict exclusions, failures=[]; detector provenance version 3 detects on complete original videos before the policy segment is truncated.
- Training: `tmp/lrs3_policy_a1_200_20260828/train/summary.json`, `checkpoint.pt`, and `checkpoint_last.pt`; A1, 2000/2000 steps, best step 2000, validation target 15.92308→11.29079, early stopping not triggered.
- Strict audit: `tmp/lrs3_policy_a1_200_20260828/audit_test/summary.json`; 40 test records from 4 source groups, five cells, 200 score checkpoints and 200 muxes, no failures, exact PCM replacement for every mux.

The primary scientific endpoint is candidate-driven video with untouched natural audio versus natural video with untouched natural audio. The trained adapter failed this endpoint: replacement Sync-C delta was -4.289075 mean / -4.358500 median with 95% CI [-4.507529, -3.813261], and Sync-D benefit was -2.678700 mean / -2.675500 median with 95% CI [-3.111233, -2.195563]; replacement wins were C=0/40, D=0/40, joint=0/40. Candidate self-pair is diagnostic only and also regressed. The result is scientific NO-GO despite validation-proxy improvement. Keep the parent replacement task blocked; analyze objective/rendering/training linkage before any new scale, A2/A3, or decoder unfreezing.

An earlier relative-path smoke wrote under shared `third_party/syncnet_python/tmp`; that directory was inspected, removed, and verified absent.
## Relations
- parent_task [[Wav2Lip replacement-aware cross-attention v2 修复]]
- follows [[对齐 Wav2Lip–SyncNet proxy 与官方评估 policy]]
- constrained_by [[tts-exp|Wav2Lip replacement is primary objective]]