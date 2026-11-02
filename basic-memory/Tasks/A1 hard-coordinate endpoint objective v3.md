---
title: A1 hard-coordinate endpoint objective v3
type: task
permalink: tts-exp/tasks/a1-hard-coordinate-endpoint-objective-v3
assigned_to: claude/downstream-agent
current_step: 4
description: Correct objective-v2 attribution and implement a coherent hard endpoint-coordinate-shape
  objective with outcome-clean evidence and exact AdamW transaction gates.
parent_task: Wav2Lip replacement-aware cross-attention v2 修复
started: '2026-08-29'
status: in-progress
steps:
- Create the immutable v2 erratum and historical evidence registry
- Freeze outcome-clean disjoint D/V/T/S manifests
- Implement endpoint_hard_coord_v3 with tie and shape contracts
- Add exact AdamW transaction and real frozen-chain seam tests
- Freeze deterministic official controls and fixed repeat gates
- Run regressions and request separate transfer authorization only if all gates pass
tags:
- task
- lrs3
- wav2lip
- syncnet
- objective-v3
blockers:
- Current available 280-record cohort is historical H; no outcome-clean D/V/T/S pool
  is available
- Real frozen-chain D shadow probes and any V/T transfer remain blocked until clean
  partitions exist
- Scientific promotion, pilot, A2/A3, extra steps and cohort scale-up remain unauthorized
---

# A1 hard-coordinate endpoint objective v3

## Observations
- [description] 纠正 objective-v2 的失败归因，并实现/验证 coherent hard endpoint/coordinate/shape objective：pristine step 0 是唯一 D/C/shape anchor，natural curve 只固定 replacement offset；取消 straight-through surrogate，在真实 frozen chain + exact AdamW proposal 上先证明逐约束 hard descent。
- [status] in-progress
- [assigned_to] claude/downstream-agent
- [current_step] 1
- [parent_task] [[Wav2Lip replacement-aware cross-attention v2 修复]]
- [decision] v2 保持 `REDESIGN_OBJECTIVE` 且不 promotion；primary correction 是 `NO_VALIDATION_TRANSFER`，proxy/official、control-media 与 scientific failure 作为 secondary predicates，不能再硬编码成 `SURROGATE_DIRECTION_MISMATCH`。
- [constraint] 当前只授权 spec、实现和 no-training seam tests；不得自动运行 20-step transfer、pilot、A2/A3、增加 steps、扩大 cohort、改变 architecture 或解冻 frozen modules。

## Source of truth

```text
openspec/changes/redesign-a1-hard-coordinate-objective-v3/
```

该 OpenSpec 在 2026-08-29 经独立 adversarial review 修订后再次通过 strict validation（15 parsed deltas）。历史 v2 OpenSpec 与 artifact 保持不可变。

## Evidence correcting v2

- 40-record historical validation step 0→20：hard D `+0.006712`、hard C `-0.003789`；surrogate D `+0.008708`、surrogate C `-0.008945`，hard/soft 同向变差。
- repeated training diagnostic：hard D `-0.077984`、hard C `+0.049160`；surrogate D `-0.045035`、surrogate C `+0.049688`，hard/soft 同向改善。
- one-record official replacement：C `+0.030`、D `-0.030` apparent step-0-relative movement，但仍低于 natural、joint wins `0/1`；five cells 是一个 record/source-group 的 recombination，不是 n=5。
- v2 gate writer 将 decision/reason 写成常量；natural-control video hash 在 step-0/20 audit 间不同，所以 official movement 尚未通过 deterministic control contract。
- v2 directional tests 只优化 free 31-point curve；没有覆盖 coupled model/AdamW Jacobian。zero-initialized output projection 还导致 step-0 upstream attention/norm gradients 为零。

## Frozen evidence partitions

任何 v3 endpoint 计算前必须注册 source-group-disjoint manifests：

```text
H = 所有曾被观察 curve/metric/render/official outcome 的 historical records
D = development-only real-Jacobian probes
V = 新 validation selector cohort
T = 新 one-record transfer gate
S = later independent scientific cohort
```

旧 200/40/40 cohort、repeated diagnostic 和 fixed samples 全部进入 H，仅作 red regression。跨 manifest 或 prior-outcome leakage 直接 `EVIDENCE_MANIFEST_LEAKAGE`。

## Objective-v3

```text
L_D = bounded exact hard positive-margin D violation vs c0
L_C = bounded exact hard positive-margin C violation vs c0
L_O = strict-margin all-offset order violations at s_ref=argmin(natural curve)
L_M = C-margin-tied median-inflation violation (at most 10% of required C margin)
L_S = one-sided per-offset inflation violations vs c0
L_v3 = L_D + L_C + L_O + L_M + L_S
```

- forward/backward 都使用 pinned scalar `torch.min`/`torch.median`；ties、near-ties、ReLU boundary 明确定义并测试。
- D/C one-percent margins 必须超过十倍 numerical tolerance；zero/degenerate anchors fail closed。
- `rho_O=0.001`；median inflation cap `m_M=0.10*(0.01*C0)` 且 per-offset `m_S=m_M`；proxy margin 不等于 official guarantee，official post-mux offset/gap 另行 gate。
- 每个 proposal 单独检查 raw `V_D/V_C`、30 个 `V_O,s`、30 个 `V_S,s`；mean/total 改善不能掩盖任一 offset 变差。

- `c0` 必须来自 frozen seed 下实际将训练的 fresh zero-output adapter；`init_adapter_checkpoint`、separate random pristine adapter 或 anchor 前 mutation 均禁止。

## Real parameter transaction

- 完整冻结 AdamW、clipping、dtype、parameter order、runtime 与 deterministic settings。
- snapshot/restore 必须覆盖 parameters/buffers、optimizer groups/state/steps、pre/post-clip grads、Python/NumPy/CPU/all-CUDA RNG、modes/backend flags，并断言 AMP/scheduler absent。
- clean D manifest 上的 real frozen-chain shadow step 必须逐约束 non-worse、无新 active violation、raw total 超过 summed tolerance 严格下降，并 byte/state-equivalent restore；否则 `PARAMETER_SEAM_DIRECTION_FAILURE`。

## Official control

- T gate 复用 byte-identical natural video/decoded-frame/untouched PCM control。
- 每 checkpoint exact 3 renders；每 canonical five-cell medium exact 3 scores；三个 primary replacement renders 各 score 3 次。
- exact-media score 或 decoded-frame hash 不一致即 evaluator/render nondeterminism。
- gate cell 只有 `finetuned_video_natural_audio`；C/D improvement 必须各超过 `2*q`，official best/second gap 也须超过 `2*q`。
- five cells 的 effective scientific n 仍为 1。

## Steps
1. [x] 建立 v2 evidence erratum、derived classifier 和 H historical correction；旧 gate artifact 未覆盖。新 artifact：`tmp/lrs3_endpoint_hard_coord_v3_20260829/v2_historical_erratum.json`，derived primary 为 `NO_VALIDATION_TRANSFER`。
2. [ ] 冻结 outcome-clean、source-group-disjoint D/V/T/S manifests；对现有 280 条 cohort 的 preflight 正确失败为 `insufficient clean source groups for D`，不能从历史 cohort 伪造新 evidence。
3. [x] 禁止 warm start；以实际 fresh zero-output trainable adapter 接入 `endpoint_hard_coord_v3`，实现 hard min/median、coordinate、median/shape、near-tie 和 provenance helpers。
4. [x] 实现 target-device AdamW execution witness：版本化 seam manifest、真实 route observer、三次三步 default route repeats、explicit route replay、clipping/state parity、canonical witness hash、immutable artifact、witness-backed optimizer contract 和 transaction loop；新增/更新 v3 witness/rollback tests。真实 D manifest probe 仍被第 2 步阻塞。
5. [x] 保留 official repeat/determinism、score-resolution、absolute-offset/gap pure gates 和 failure taxonomy；实际 T official audit 未执行。
6. [x] 运行 compile、Ruff、LRS3 experiment suite（104 passed）、OpenSpec strict validation；未运行任何新训练、pilot、A2/A3 或 promotion。
## Blocker
现有可用 policy cohort 的 280 个 source groups 都有 prior curve/metric/render/outcome，全部必须归入 H；当前没有合法 D/V/T/S pool。AdamW raw request 与 effective route 已按新合同解决：历史默认仍保留 `foreach/fused=None`，在目标 CUDA runtime 观测到实际 `foreach` route，再显式 replay 为 `foreach=True, fused=False`，三次三步 trajectory、clipping、state/schema、参数 tolerance、observer-on/off parity 和 witness hash 均通过。当前仍需在 clean D pool 可用后运行真实 frozen-chain D shadow probes；不得把 prospective witness 反推为 historical kernel 证明。Parent scientific task 继续 blocked，training/pilot/A2/A3/transfer/promotion 均未授权。

## Relations

- parent_task [[Wav2Lip replacement-aware cross-attention v2 修复]]
