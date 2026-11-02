---
title: H-only metadata artifact freeze
type: task
permalink: tts-exp/tasks/h-only-metadata-artifact-freeze
assigned_to: downstream-agent
blockers:
- Real P0/P1 requires a separate external immutable authorization artifact
- Clean D/V/T/S closure remains unresolved and is outside this H-only change
current_step: 5
description: Freeze authoritative H-only metadata inventory, deterministic exact-four
  selection, and create-only preflight artifacts without media/model/training execution.
parent_task: H-only historical endpoint sensitivity assay
started: '2026-08-30'
status: done
steps:
- Write and validate metadata freeze OpenSpec
- Implement discovery and source closure
- Implement selection and create-only hash chain
- Run full verification
- Stop before real P0/P1
tags:
- task
- lrs3
- wav2lip
- syncnet
- historical-evidence
- metadata-freeze
completed: '2026-08-30'
---

# H-only metadata artifact freeze

## Observations
- [description] 在不解码媒体、不加载模型、不训练的前提下，闭合 H-only evidence namespace，冻结 exact-four deterministic selection，并 create-only 生成 hash-linked `PREFLIGHT_FROZEN` artifacts。
- [status] done
- [assigned_to] downstream-agent
- [current_step] 5
- [openspec] `freeze-h-only-assay-metadata-artifacts`
- [decision] 当前 loss、synthetic contract hardening 与 metadata-only freeze 已完成；真实资源 dry run 仍按 H-only 规则 fail-closed，不运行 real P0/P1。
- [authority] 成功 `PREFLIGHT_FROZEN` 只表示 metadata/artifacts frozen；P0 仍要求 externally created matching authorization。
- [non_goals] 不训练、不 decode/render/mux/score、不下载数据、不扩大 cohort、不改变 D/V/T/S、不做 scientific promotion。

## Steps
1. [x] 编写 closed-world discovery、content-addressed inventory、exact-four selection、create-only tree、hash chain与authorization handoff OpenSpec。
2. [x] 实现 checked-in discovery policy、namespace coverage、two-pass source rehash与partition closure。
3. [x] 实现 eligible-universe/selection hashes和decision-last create-only preflight writer/verifier。
4. [x] 运行LRS3 experiment regression、Ruff、compileall、strict validation与diff checks；metadata focused 32 passed、LRS3 experiment suite 163 passed、目录级 Ruff 全绿、compileall 与两个 OpenSpec strict validation 通过。
5. [x] 达到 metadata stop condition 后停止；现有 TheNHz 资源的 metadata-only dry run 返回 `BLOCKED_INSUFFICIENT_H_ELIGIBILITY`，未生成 decision marker，real P0/P1 仍需 external authorization。
## Context

Source of truth：

```text
openspec/changes/freeze-h-only-assay-metadata-artifacts/
```

该 change 接续 [[H-only synthetic contract hardening]]。`historical_endpoint_sensitivity.py` 现已完成 independent namespace discovery、closed-world root policy、eligible-universe/selection hash、two-pass source rehash、decision-last create-only transaction和standalone verifier。成功 fixture 可生成并验证完整 `PREFLIGHT_FROZEN` tree；当前工作树的真实可见资源只有prospective TheNHz records，因此真实 metadata run正确返回 `BLOCKED_INSUFFICIENT_H_ELIGIBILITY`，不生成decision marker。

固定 artifact tree：

```text
<unique-worktree-local-root>/preflight/
  freeze_request.json
  source_registry.json
  evidence_inventory.json
  assay_manifest.json
  status.json
  report.md
  decision.json   # written last; sole commit marker
```

Mandatory terminal states包括 `EVIDENCE_INVENTORY_INCOMPLETE`、`PROSPECTIVE_EVIDENCE_CONTAMINATION_RISK`、`BLOCKED_INSUFFICIENT_H_ELIGIBILITY`、`SOURCE_CHANGED_DURING_FREEZE`、`ARTIFACT_ALREADY_EXISTS` 与 `ARTIFACT_WRITE_FAILED`。所有failure不得生成authoritative decision。

## Relations
- parent_task [[H-only historical endpoint sensitivity assay]]
- follows [[H-only synthetic contract hardening]]
- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]
