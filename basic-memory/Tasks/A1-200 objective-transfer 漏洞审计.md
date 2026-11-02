---
title: A1-200 objective-transfer 漏洞审计
type: task
permalink: tts-exp/tasks/a1-200-objective-transfer-漏洞审计
assigned_to: downstream-agent
context: Audit the A1-200 objective-transfer failure in worktree worktree-lrs3-wavlm-resynthesis-50.
  Start with a deterministic cached red feedback loop, then test H1 objective/selector
  mismatch, H2 unbounded residual collapse, H3 ignored TTS conditioning, H4 STE gradient
  mismatch, H5 96-frame/full-file mismatch, and H6 offset/policy correctness one variable
  at a time. Existing artifacts are under tmp/lrs3_policy_a1_200_20260828. Keep A2/A3,
  scale-up, decoder unfreezing, and sealed-test selection blocked; do not overwrite
  runs or upload local data.
current_step: 9
description: 对 policy-aligned A1-200 的 objective-transfer failure 建立红灯反馈环并做单变量机制审计，确认
  replacement 退化的 primary root cause 后再交接独立 repair spec。
parent_task: Wav2Lip replacement-aware cross-attention v2 修复
priority: high
started: '2026-08-28'
status: done
steps:
- 建立并运行 cached objective-transfer red feedback loop
- 从 validation curves 最小化 fixed-record reproducer
- 完成 H1 curve/selector 审计与 residual alpha sweep
- 完成 paired/shuffled/reversed TTS conditioning 审计
- 完成分层 STE official finite-difference direction 审计
- 完成 96-frame segment 与 full-file边界审计
- 复核 offset、natural span、policy identity 与 strict scorer round-trip
- 添加 regression tests并形成 root-cause ranking
- 清理并更新父任务，决定是否另建 repair spec
tags:
- task
- wav2lip
- syncnet
- cross-attention
- replacement
- lrs3
- debugging
- objective-transfer
completed: '2026-08-28'
---

# A1-200 objective-transfer 漏洞审计

## Spec identity

- Spec ID: `LRS3-REPL-AUDIT-OT-01`
- Version: `1.0`
- Date: `2026-08-28`
- Owner: downstream agent
- Parent: [[Wav2Lip replacement-aware cross-attention v2 修复]]
- Predecessor evidence: [[LRS3 policy-aligned A1-200 早停训练]]

## Observations

- [description] 对 policy-aligned A1-200 的 objective-transfer failure 做可复现、单变量、机制级漏洞审计，确定 replacement 退化来自目标函数、residual 失控、TTS conditioning 退化、STE backward mismatch、segment/full-file mismatch 或其他实现漏洞中的哪一项；本任务不直接训练新 variant。
- [status] in-progress
- [assigned_to] downstream-agent
- [current_step] 3
- [priority] high
- [decision] 先建立能对当前真实症状变红、对未来修复结果变绿的缓存级反馈环；在反馈环存在前禁止凭代码阅读直接修复。
- [constraint] primary endpoint 锁定为 candidate-driven video + untouched natural PCM 相对 natural video + untouched natural PCM；candidate self-pair 仅作诊断。
- [constraint] WavLM、HiFi-GAN、Wav2Lip、SyncNet 全部冻结；审计阶段不得运行 A2/A3、扩大数据、增加训练步数或解冻 decoder。
- [known-failure] validation target distance `15.92308→11.29079`，但 validation Sync-D `7.46527→9.03675`、Sync-C `7.69709→4.27038`；strict replacement C/D/joint wins 均为 `0/40`。
- [known-failure] step-2000 adapter feature residual RMS 约为 natural query RMS 的 `110.65%`；跨 source-group shuffled TTS 只改变约 `2.42%` 的 feature residual、约 `0.131%` 的 adapter waveform change；反转 TTS feature 时间顺序的差异约 `7.8e-8`。

## Problem statement

A1-200 的 checkpoint selection metric 持续改善且 early stopping 未触发，但同一 source-group-disjoint validation 上完整 official 31-point curve 的 Sync-C 与 Sync-D 已明显退化，strict untouched-natural-audio replacement 又进一步确认强负效应。这说明至少存在一个 load-bearing loophole：训练、选择或 conditioning 允许模型优化一个与科学 endpoint 不一致的捷径。

下游 agent 必须回答：

1. 当前 implementation 是否存在能改变 forward 数值或 offset 语义的 correctness bug？
2. 若 forward 无 correctness bug，哪个 objective shortcut 允许 `d(s*)` 降低而 Sync-C/Sync-D 同时恶化？
3. residual 从“微调”增长到大于 natural feature RMS 是否是退化的必要或充分条件？
4. paired TTS 是否对 candidate 有可测、可归因的贡献，还是 adapter 主要依赖 natural query/近似固定 context aggregate？
5. straight-through surrogate gradient 是否与 official finite-difference 改善方向一致？
6. 96-frame policy training 到 full-file inference 是否只是放大因素，还是能独立复现主要退化？

## Locked evidence and paths

工作目录必须是当前 isolated worktree，禁止写主 checkout：

`/home/wjj/tts-audio/tts-exp/.claude/worktrees/lrs3-wavlm-resynthesis-50`

现有只读证据：

- Train summary: `tmp/lrs3_policy_a1_200_20260828/train/summary.json`
- Best checkpoint: `tmp/lrs3_policy_a1_200_20260828/train/checkpoint.pt`
- Last checkpoint: `tmp/lrs3_policy_a1_200_20260828/train/checkpoint_last.pt`
- Policy manifest: `tmp/lrs3_policy_a1_200_20260828/policy_cohort/policy_manifest.json`
- Source manifest: `tmp/lrs3_policy_a1_200_20260828/policy_cohort/source_manifest.json`
- Crop manifest: `tmp/lrs3_policy_a1_200_20260828/policy_cohort/crop_manifest.json`
- Strict audit: `tmp/lrs3_policy_a1_200_20260828/audit_test/summary.json`
- Relevant code: `scripts/experiments/lrs3_syncnet_finetune/{replacement_train.py,replacement_model.py,syncnet_loss.py,wav2lip_proxy.py,replacement_evaluate.py}`
- Focused tests: `tests/experiments/lrs3_syncnet_finetune/test_replacement_v2.py`

不得覆盖上述 artifact。所有新 artifact 写入唯一 worktree-local 目录，例如：

`tmp/lrs3_policy_a1_200_loophole_audit_20260828/`

## Scope

### In scope

- cached summary/checkpoint differential analysis；
- fixed-record、fixed-checkpoint、single-variable inference diagnostics；
- paired/shuffled/reversed/null-like TTS context ablation；
- residual scale sweep；
- full 31-point curve decomposition；
- STE gradient 与 official finite-difference direction check；
- 96-frame 与 full-file boundary comparison；
- correctness regression tests；
- root-cause ranking和最小修复建议。

### Out of scope

- 训练新的 A1/A2/A3 cohort；
- 使用 strict test 或 sealed test 选择超参数；
- 修改 official evaluator 来匹配训练 proxy；
- 解冻 WavLM、HiFi-GAN、Wav2Lip、SyncNet 或 decoder；
- 以 candidate self-pair 代替 replacement endpoint；
- 在审计尚未隔离根因时同时修改 loss、architecture 和 render policy。

## Phase 1 — Build the red-capable feedback loop

先实现一个缓存级、确定性、agent-runnable 的 objective-transfer checker。推荐入口：

`python scripts/experiments/lrs3_syncnet_finetune/audit_objective_transfer.py --train-summary <summary> --strict-summary <summary> --expect healthy`

要求：

- 不加载 GPU 模型；目标运行时间小于 5 秒；
- 读取 validation step 0/final 的 `target_distance`、Sync-C、Sync-D；
- 读取 strict replacement mean/median、CI、wins 与 integrity；
- 在当前 A1-200 artifact 上非零退出并打印结构化 `OBJECTIVE_TRANSFER_FAILURE`；
- 对合成 healthy fixture 绿色退出；
- 支持未来候选 run 复用，不能把当前路径或当前数值硬编码成唯一失败条件；
- 输出 JSON，至少包含 `proxy_improved`、`validation_endpoint_improved`、`strict_endpoint_improved`、`audit_integrity`、`verdict`。

初始 red 判据至少捕获：

```text
proxy target distance 显著下降
AND validation Sync-C 下降
AND validation Sync-D 上升
AND strict replacement C/D endpoint 不支持正确方向
```

Phase 1 完成条件：下游 agent 必须实际运行一次命令，记录 invocation、exit code 和结构化 red output。没有这个反馈环不得进入修复。

## Phase 2 — Reproduce and minimise

从 train summary 的 validation rows 中找出最小 fixed-record reproducer：

1. 比较 step 0 与 step 200/400/600/800/2000 的 per-record curve；
2. 找到最早同时满足 `target_distance↓`、`Sync-C↓`、`Sync-D↑` 的 checkpoint interval；
3. 找到单条或最小两条样本即可稳定复现的 subset；
4. 固定 sample ID、source group、policy segment、offset index、checkpoint 和 seed；
5. 证明去掉任一 load-bearing element 后 repro 变绿或症状消失。

如果中间 checkpoint 未保存，不得伪造；使用 summary 中保存的 curve 做 cached minimisation，并明确记录“不能重放 intermediate model state”的证据缺口。只有在后续获准时才可做短程 deterministic replay。

## Phase 3 — Ranked falsifiable hypotheses

开始 probe 前必须公开以下排序；可以根据新证据调整，但不得只保留单一假设。

### H1 — A1 single-entry objective/selector mismatch（当前最高优先级）

预测：`d(s*)` 可持续下降，而 `min(curve)` 上升、`median-min` 下降；若按 validation joint Sync-C/Sync-D 选择 checkpoint，step 600+ 会被拒绝，step 0–400 胜出。

反证：在相同 curve 上 target 改善总是伴随 Sync-D 和 Sync-C 同方向改善。

### H2 — Unbounded residual causes representation/waveform collapse

预测：对同一 checkpoint 做 inference-only residual scale `alpha∈{0,.05,.1,.25,.5,1}`，C/D 退化随 alpha 增大出现；小 alpha 保留 zero-residual 行为，而 alpha=1 复现崩坏。记录 feature residual/query RMS 与 waveform difference-to-signal ratio。

反证：alpha sweep 对 C/D 无系统效应，或 alpha 很小时已完整复现退化。

### H3 — Paired TTS conditioning is functionally ignored

预测：paired TTS 与 cross-group shuffled TTS 的 feature/waveform/official-score 差异远小于 paired 与 zero-residual 的差异；reversed TTS feature 顺序在无位置编码结构下数值不变。

反证：paired context 在多 source groups 上稳定优于 shuffled，并对 strict replacement C/D 产生 preregistered、显著的正确方向差异。

### H4 — STE backward direction is misaligned with official forward

预测：对固定样本，在 surrogate gradient 方向的 `±epsilon` official finite-difference sweep 中，至少一段常用 epsilon 范围出现负 directional agreement，或 surrogate 预测改善但 official target/C/D 变坏。epsilon 必须覆盖量化阈值，不能只测试小于 PCM/uint8 分辨率的扰动。

反证：多个固定样本、多个非饱和 epsilon 上 surrogate 与 official direction 稳定一致。

### H5 — 96-frame training to full-file inference mismatch amplifies failure

预测：同一 checkpoint 在 policy-aligned 96-frame strict segment 上明显优于 full-file；若 96-frame 也同量级退化，则它不是主因。

反证：segment/full-file 的 paired deltas 接近，或 segment 已经复现核心退化。

### H6 — Offset/curve coordinate or baseline locking bug

预测：internal offset、official reported offset、curve index 或 natural span 中至少一处 round-trip 不一致；修正后 locked target 会与 natural curve argmin 对齐。

反证：所有 31 个 index round-trip、natural argmin、policy span 和 strict scorer offset 在 fixed fixtures 上完全一致。

## Phase 4 — Single-variable probe matrix

每次只改变一项，其余 sample、checkpoint、source frames、boxes、audio span、codec 和 scorer 固定。

1. **Cached curve probe**：step × target/min/median/C/argmin；输出 per-record JSON 和 aggregate。
2. **Residual alpha probe**：`alpha={0,.05,.1,.25,.5,1}`；先 proxy，再对最小 subset 做 official strict score。
3. **TTS context probe**：paired、cross-group shuffled、within-sample reversed；必要时增加固定 aggregate/null-like context，但必须单独标注其分布外性质。
4. **STE probe**：分别检查 waveform→mel、patch→uint8 compositor、frame→SyncNet crop 三个边界；不得一次合并全部边界后归因。
5. **Length probe**：同一记录的 policy 96-frame segment 与 full-file；不得更换 source group。
6. **Coordinate probe**：31-index round-trip、natural baseline argmin、candidate argmin、reported offset。

每个 probe 输出：输入 hashes、唯一改变变量、预期、观测、effect size、verdict、是否需要下一 probe。禁止只给截图或口头判断。

## Phase 5 — Correct-seam regression tests

先写测试再修复；测试必须覆盖真实调用 seam，而不是只验证孤立数学函数。

至少需要：

1. cached objective-transfer checker 在 current A1-200 fixture 上能红、healthy fixture 能绿；
2. checkpoint residual scale alpha=0 精确等价于 zero-residual；
3. trained checkpoint paired/shuffled context sensitivity被结构化报告，不能只断言“非零”；
4. TTS key/value reversal invariance被显式记录为当前无位置编码架构性质；
5. validation selector fixture：target 改善但 C/D 恶化时不得选为 best；
6. STE boundary tests：official forward exact，surrogate backward finite/nonzero；如 directional fidelity 无合适 unit seam，必须用 integration artifact 而非伪造浅层测试。

不要把当前 `test_paired_and_shuffled_contexts_have_observable_effect` 当成 trained-checkpoint 证据；它只证明随机合成 adapter 理论上能响应 context。

## Phase 6 — Root-cause decision and repair handoff

审计结束后按以下等级报告：

- `CONFIRMED_PRIMARY`：最小 repro、单变量 probe、机制和 endpoint effect 全部成立；
- `CONFIRMED_CONTRIBUTOR`：能放大失败但不能独立解释 train/validation 退化；
- `REFUTED`：预测未出现；
- `UNRESOLVED`：缺 checkpoint、测试 seam 或算力证据，禁止猜测。

只有在至少一个 primary root cause 被确认后，才能另外创建 repair spec。repair spec 必须一次只改一个 load-bearing variable，并保留当前 strict evaluator。不得在本审计任务中顺手训练新 variant。

## Acceptance gates

本任务完成必须同时满足：

- [ ] 存在一个小于 5 秒、确定性、当前 run 上变红的 cached feedback-loop command；
- [ ] 找到最小 fixed-record reproducer，或清楚记录为何只能做到 cached minimisation；
- [ ] 至少测试 H1–H5，每项有 falsifiable prediction 和结构化 artifact；
- [ ] residual alpha、TTS context、STE boundary、segment/full-file 均保持单变量；
- [ ] 明确区分 implementation correctness、surrogate-gradient fidelity 和 scientific-objective validity；
- [ ] 不用 test/sealed split 选修复参数；
- [ ] 给出 primary/contributor/refuted/unresolved 排名；
- [ ] 添加正确 seam 的 regression tests，并运行 focused tests；
- [ ] 清理所有 `[DEBUG-*]` instrumentation 和临时 prototype；
- [ ] 更新本 BM task 与父任务 blocker，但不虚报 scientific promotion。

## Stop conditions

遇到以下任一情况立即停止当前分支并记录：

- official forward parity、PCM exactness、source/box/audio policy identity 任一失败；
- probe 同时改变两个以上变量，无法归因；
- 需要使用 sealed test 选择 epsilon、alpha、margin 或 checkpoint；
- intermediate checkpoint 不存在却试图从 summary 反推模型状态；
- candidate self-pair 改善但 untouched-natural replacement 未改善；
- codec error 掩盖 pre-encode mismatch；
- 需要解冻任何 frozen model 才能让 diagnostic 成立。

## Safety and repository constraints

- 不上传本地音频、视频、manifest 或 checkpoint 到第三方服务；
- 不写主 checkout；只在当前 worktree 与唯一新 artifact 目录工作；
- 不覆盖或重命名历史 run；
- 不恢复无关 Mandarin YAML 删除；
- 不通过补零、重复帧、默认 full-frame box 或伪造 provenance 扩大样本；
- GPU 被占用时先确认进程属于本实验，再终止；不得杀无关进程；
- 不修改 strict evaluator 去迁就 training proxy；
- 所有数值结论必须带 sample count、source-group count 和输入 hash/provenance。

## Required deliverables

1. `audit_objective_transfer.py` 或等价的 red-capable checker；
2. 最小 reproducer config/fixture；
3. `probe_matrix.json`：所有单变量 probe 的输入、预测、结果和 verdict；
4. residual-alpha、TTS-context、STE、segment/full-file 的独立子 artifact；
5. focused regression tests 与运行日志；
6. `root_cause_report.md/json`：confirmed/refuted/unresolved 排名；
7. 更新后的本 BM task 与父任务；
8. 若且仅若 primary root cause 确认，另起 repair spec，不在本任务中直接扩训。

## Steps
1. [x] 建立并运行 cached objective-transfer red feedback loop：新增 `scripts/experiments/lrs3_syncnet_finetune/audit_objective_transfer.py`，对真实 A1-200 输出 `OBJECTIVE_TRANSFER_FAILURE`、非零退出，且 `audit_integrity=true`。
2. [x] 从 validation curves 最小化 fixed-record reproducer：新增 `audit_validation_curves.py`，从逐 record 31-point curve 找到 step 600 的单 record reproducer，并确认该 record 持续到 step 2000。
3. [x] 完成 H1 curve/selector 审计与 residual alpha sweep：cached H1 在 step 600 首次成立；完整 C↑/D↓ selector 没有合格的后续 checkpoint，会回退到 step 0，而 proxy selector 选中 step 2000。新增 `audit_residual_alpha.py`，在 9 条、5 个 source groups 的 validation diagnostic 上完成 alpha `{0,.05,.1,.25,.5,1}` 扫描；alpha=0 精确等价 zero-residual，alpha=1 与 step-2000 summary seam 最大差异约 `1e-4`。alpha≥0.25 出现明显 C/D 退化，H2 判为 `CONFIRMED_CONTRIBUTOR`，但尚不能单独声明 primary。
4. [x] 完成 paired/shuffled/reversed TTS conditioning 审计：新增 `audit_tts_context.py`，在同一 9 条、5 个 source groups 的 validation diagnostic 上完成 zero-residual、paired、cross-group shuffled、reversed 四种 context 的 official 31-point curve。shuffled 相对 paired 的 feature effect mean 为 `0.0933`、waveform effect mean 为 `0.000334`；reversed feature effect mean 为 `7.42e-8`；paired 相对 zero-residual 的 Sync-C delta 为 `-3.7083`、Sync-D benefit 为 `-1.8242`。H3 判为 `CONFIRMED_CONTRIBUTOR`，显示 adapter 对 paired TTS identity/顺序功能性不敏感。
5. [x] 完成分层 STE official finite-difference direction 审计：新增 `audit_ste_direction.py`，在 H1 fixed validation record `lrs3_6W2dsnhC18Q_00005` 上分层测试 waveform→mel、patch→uint8 compositor、frame→SyncNet crop。waveform→mel 为 4/4 agreement，frame→crop 为 3/3 agreement；patch→compositor 为 1/3 agreement、2/3 disagreement。H4 判为 `CONFIRMED_CONTRIBUTOR`，但证据范围仅为 1 条 record、每层 1 个 deterministic direction，不能单独声明 primary。
6. [x] 完成 96-frame segment 与 full-file 单变量边界审计：新增 `audit_length_boundary.py`，使用同一 fixed validation reproducer 的 official strict endpoint。96-frame segment 的 replacement Sync-C delta / Sync-D benefit 为 `-2.207/-1.243`，full-file 为 `-3.004/-1.991`；两者均为 core failure，且 4/4 mux 均保持 `pcm_exact_match=true`。H5 判为 `REFUTED_AS_PRIMARY`，full-file 仅放大已有退化。
7. [x] 复核 offset、natural span、policy identity 与 strict scorer round-trip：新增 `audit_coordinate_roundtrips.py`，40/40 validation natural argmins 对齐，31-index conversion 全部通过，strict 200/200 scores 与 muxes 完整，H5 length scores 完整，H6 判为 `REFUTED`；3 条 crop track 尾部差异记录为 non-blocking warning。
8. [x] 添加 regression tests，重跑原始 red loop，形成 root-cause ranking：实验目录测试 `58 passed`，targeted Ruff 通过；red loop 仍输出 `OBJECTIVE_TRANSFER_FAILURE`；新增 `probe_matrix.json` 与 `root_cause_report.json`，H1 判为 `CONFIRMED_PRIMARY`，H2/H3/H4 为 contributors，H5/H6 非 primary。
9. [x] 清理 instrumentation，写 root-cause report，更新父任务并决定是否另建 repair spec：required audit artifacts 已完成，父任务 blocker 已保持，H1-only repair spec 已创建于 `openspec/changes/repair-a1-objective-transfer/`。
## Downstream-agent handoff prompt

在 worktree `worktree-lrs3-wavlm-resynthesis-50` 执行 BM task `tts-exp/tasks/a1-200-objective-transfer-漏洞审计`。严格从 Phase 1 的缓存级红灯反馈环开始；在该命令实际对 A1-200 失败 artifact 变红前，不得修改 loss 或训练新模型。随后按 H1–H6 做单变量 probe，优先使用现有 summary/checkpoint 和最小 fixed-record subset。primary endpoint 始终是 finetuned candidate-driven video + untouched natural PCM；保持四个基础模型冻结，不运行 A2/A3，不用 sealed test 选参数，不覆盖历史 artifact。每个结论必须给出 falsifiable prediction、输入 hashes、effect size 和 confirmed/refuted/unresolved verdict。完成后更新本任务及父任务；只有确认 primary root cause 后才另建 repair spec。

## Context for resumption
The A1-200 objective-transfer audit is complete. Phase 1 red feedback loop remains deterministic and red: historical A1-200 has `proxy_improved=true`, `validation_endpoint_improved=false`, `strict_endpoint_improved=false`, `audit_integrity=true`, and exit code 1 under `--expect healthy`. H1 is confirmed primary: cached target-entry objective and proxy selector choose step 2000 despite full validation C/D regression; endpoint-aware selection falls back to step 0.

H2 residual alpha is a confirmed contributor on 9 validation records/5 source groups: alpha 0 is exact zero-residual, alpha>=0.25 causes clear C/D collapse, and alpha 1 reproduces the checkpoint seam. H3 TTS conditioning is a confirmed contributor: cross-group shuffled context changes waveform by only 0.033% relative to paired change and reversed context is near invariant. H4 is a limited confirmed compositor STE contributor: 2/3 official finite-difference directions disagree for one fixed validation sample, while mel and crop boundaries agree. H5 is refuted as primary because the official 96-frame segment already fails (C/D `-2.207/-1.243`) before full-file amplification (`-3.004/-1.991`). H6 is refuted: 40/40 policy natural argmins and coordinate conversions pass; strict 200/200 scores and muxes are complete; three crop full-track tail discrepancies are nonblocking warnings.

Required artifacts:
- `tmp/lrs3_policy_a1_200_loophole_audit_20260828/probe_matrix.json`
- `tmp/lrs3_policy_a1_200_loophole_audit_20260828/root_cause_report.json`
- `tmp/lrs3_policy_a1_200_loophole_audit_20260828/validation_curve_audit.json`
- `tmp/lrs3_policy_a1_200_alpha_probe_20260828/summary.json`
- `tmp/lrs3_policy_a1_200_tts_context_probe_20260828/summary.json`
- `tmp/lrs3_policy_a1_200_ste_probe_20260828/summary.json`
- `tmp/lrs3_policy_a1_200_length_probe_ffv1_20260828/summary.json`
- `tmp/lrs3_policy_a1_200_coordinate_audit_20260828/summary.json`

Regression tests and targeted Ruff pass; experiment directory tests pass `58`. Parent task remains blocked for scientific promotion. H1-only repair spec is created at `openspec/changes/repair-a1-objective-transfer/`, validated with `openspec validate repair-a1-objective-transfer --strict`. It requires selector-only cached repair before objective-only fixed-sample work and prohibits H2/H3/H4/H5/H6 changes, A2/A3, scale-up, decoder unfreezing, and sealed-test parameter selection.
## Relations

- parent_task [[Wav2Lip replacement-aware cross-attention v2 修复]]
- follows [[LRS3 policy-aligned A1-200 早停训练]]
- constrained_by [[tts-exp|Wav2Lip replacement is primary objective]]
- investigates [[对齐 Wav2Lip–SyncNet proxy 与官方评估 policy]]
