---
title: H-only historical endpoint sensitivity assay
type: task
permalink: tts-exp/tasks/h-only-historical-endpoint-sensitivity-assay
assigned_to: downstream-agent
blockers:
- Current worktree TheNHz resources contain no eligible H records; local metadata
  freeze correctly fails closed
- Real P0-P3 require a separate external immutable authorization artifact
- Candidate input and frozen pristine/historical adapter pair may be unavailable
- Clean D/V/T/S closure and real D shadow remain unresolved
current_step: 5
description: Historical H-only assay retained as completed engineering infrastructure
  but abandoned as the active algorithm-validation path on 2026-08-30.
parent_task: A1 hard-coordinate endpoint objective v3
started: '2026-08-29'
status: done
steps:
- Complete original H-only assay design and basic pure seams
- Audit and correct overstated completion claims
- Run synthetic contract hardening child task
- Design metadata inventory and artifact freeze separately
- Implement or run real P0/P1 only with separate authorization
- Consider P2/P3 only after exact gates
tags:
- task
- lrs3
- wav2lip
- syncnet
- historical-evidence
- endpoint-sensitivity
---

# H-only historical endpoint sensitivity assay

## Observations
- [description] H-only no-training assay 的 synthetic contracts 与 metadata freeze 已完成并保留为历史机制审计工具，但 2026-08-30 用户决定放弃 H/D/V/T/S 作为当前算法原型前置条件。
- [status] done
- [assigned_to] downstream-agent
- [current_step] 5
- [decision] 不再实现 real P0–P3、external authorization 或 H eligibility；旧 artifacts/specs 不删除、不改写，但不再控制新训练路径。
- [replacement] 当前执行主线由 `openspec/changes/prototype-replacement-aligned-audio-head/` 接管：保留 CrossAttentionResidual 与 untouched-natural replacement endpoint，先做数学 loss seam、单记录20-step和可选四记录共享原型。
- [claim_boundary] 新原型只允许 fixed-data constructability 结论，不声称 validation、transfer、generalization 或 scientific gain。

## Source of truth

```text
openspec/changes/assay-h-only-historical-endpoint-sensitivity/
```

- `proposal.md`：动机、scope 与三层授权；
- `design.md`：canonical inventory、fixed selection/timebase、A/B/C、P0–P3、external authorization、phase artifacts 与 non-claims；
- `specs/historical-endpoint-sensitivity/spec.md`：15 个 parsed SHALL/MUST requirement deltas；
- `tasks.md`：下游实现/验证/运行任务卡；
- `AGENTS.md`：OpenCode/Claude Code 工作合同。

Validation：

```bash
openspec validate assay-h-only-historical-endpoint-sensitivity --strict
```

当前 strict validation PASS。

## Canonical endpoint

P0、P1、P2、P3 的每条 authoritative curve 均必须来自：

```text
driver/candidate PCM
→ frozen official Wav2Lip with frozen target frames/boxes
→ candidate decoded video
→ mux target untouched natural PCM exact segment
→ demux target PCM identity verification
→ official SyncNet V2, vshift=15, full 31-point curve
```

proxy/self-pair 只能在 `diagnostics/candidate_self_pair/`，不能填入 replacement facts。

## Evidence inventory and fixed records

preflight 必须冻结 machine-checkable `evidence_inventory.json`：覆盖所有 local known records/source groups、H prior-outcome provenance、prospective exclusions、authoritative manifest hashes、current namespace fingerprint 和 completeness flags。missing/stale/unknown/uncovered lineage fail closed。

selection：

```text
selection_key = sha256(
  "historical-endpoint-sensitivity-v1" + "\0" + source_group + "\0" + sample_id
)
```

按 key 选择 4 条、4 个不同 H source groups。每条必须 exact 25/1 fps、`frame_start`、96 frames、`sample_start=frame_start*640`、61,440 signed PCM16-LE samples 和 versioned frame/sample mapping。

## Mandatory controls

1. `A_NATURAL_DIRECT`：目标 exact natural segment驱动Wav2Lip；
2. `B_WRONG_UTTERANCE`：frozen cyclic donor 的 exact mapped segment驱动Wav2Lip；
3. `C_NATURAL_PCM_ROUNDTRIP`：sample-identical PCM roundtrip驱动Wav2Lip；
4. 三者最终都mux回目标 untouched natural segment；
5. 每condition exact3 repeats。

## Phase and threshold contract

```text
P0 exact determinism PASS
→ P1 endpoint sensitive (>=3/4)
→ P2 candidate path observed (>=3/4 valid records)
→ P3 frozen adapter influence
```

所有 SKIP/MIXED/INVALID/contradictory 状态 terminal。`P2_SKIPPED_NO_EXISTING_INPUT` 不能进入P3。

固定：

```text
q = 0.001
τ_curve = 10q = 0.010
D = min(curve)
M = median(curve)
C = M-D
```

offset使用v3 global near-tie rule；best-second gap `<=2q` 为 inconclusive，不能支持offset witness或P3 joint win。

P2 canonical PCM为exact 61,440 PCM16-LE samples、exact3 repeats、frozen per-frame mouth ROI；total classifier precedence覆盖invalid、causal contradiction、audio inactive、Wav2Lip inactive、visual-only neutral与path observed。

P3只要求common upstream/seed/policy identity；adapter-dependent candidate PCM是causal output，预期可不同。precedence固定：missing pair → invalid → replacement witness → self-pair-only → no witness。

## Three authorization layers

1. **Design + pure/synthetic contracts**：已完成并验证；
2. **Authoritative metadata freeze**：implementation已完成并验证；只可扫描/哈希/写create-only metadata，不运行媒体；当前工作树现有TheNHz资源因无clean H eligibility而正确blocked。
3. **Real P0–P3**：需再次明确要求，并由runner外部创建immutable authorization artifact，绑定assay/inventory/request hashes与allowed phases。

runner不得自行创建authorization。synthetic CLI保持non-authoritative；只有完整metadata preflight可产生 `PREFLIGHT_FROZEN`，且仍不能替代external authorization。

## No-training enforcement

assay必须通过static AST/import guard拒绝 training entrypoint、optimizer、backward和step；runtime必须eval、requires_grad false、inference_mode、zero optimizer并验证before/after model state hash。任何违反为 `TRAINING_FORBIDDEN_IN_H_ASSAY`。

## Immutable outputs

每个phase独立写入：

```text
phases/p0/...
phases/p1/...
phases/p2/...
phases/p3/...
```

existing matching run directory可继续；每个target file create-only。bootstrap hash order固定为：先对排除self-hash的canonical inventory payload求 `inventory_payload_sha256`；assay manifest引用该hash，再对排除自身self-hash的manifest求 `assay_manifest_sha256`。inventory是唯一不携带manifest hash的pre-manifest artifact，从而没有hash cycle。

每个phase hash-link previous decision、authorization、inventory与assay manifest。`final/`只由explicit finalize create-once。

## Steps
1. [x] 完成原 H-only assay design、synthetic hardening 与 metadata freeze。
2. [x] 验证现有资源因没有 eligible H 而 fail closed。
3. [x] 2026-08-30 根据用户决策停止 H/D/V/T/S、real P0–P3 与 no-training authorization 主线。
4. [x] 将后续算法验证移交 [[Replacement-aligned audio-head fixed-data prototype]]。

## Blockers

- Current worktree TheNHz resources contain no eligible H records；metadata freeze 正确返回 `BLOCKED_INSUFFICIENT_H_ELIGIBILITY`，不得补帧、换 cohort 或把 prospective 记录改成 H。
- real P0/P1仍要求独立 external immutable authorization；metadata preflight不能替代授权，real runner仍未实现。
- candidate input和frozen pristine/historical pair可能缺失；缺失必须terminal SKIP。
- clean D/V/T/S closure与real D shadow仍未完成，本assay不能解决。

## Relations

- parent_task [[A1 hard-coordinate endpoint objective v3]]
- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]
- relates_to [[Raw TTS strict replacement NO-GO]]
- relates_to [[LRS3 MFA-linear TFG native/replacement 2026-09-13|LRS3 MFA-linear replacement NO-GO]]
- relates_to [[External fit Stage01 denominator blocked]]
