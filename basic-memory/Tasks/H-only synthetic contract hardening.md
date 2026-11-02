---
title: H-only synthetic contract hardening
type: task
permalink: tts-exp/tasks/h-only-synthetic-contract-hardening
assigned_to: downstream-agent
blockers:
- Real media execution remains separately unauthorized
- Authoritative inventory discovery and artifact freeze are explicitly deferred
- Clean D/V/T/S closure remains unresolved
current_step: 6
description: Harden H-only assay synthetic authority, fixed-denominator classifiers,
  decision/hash gates, and no-training tests without loading real media or models.
parent_task: H-only historical endpoint sensitivity assay
started: '2026-08-30'
status: done
steps:
- Make synthetic execution non-authoritative
- Implement fixed four-record P1 derivation
- Require complete P2/P3 synthetic facts
- Bind phase and authorization decisions
- Complete no-training mutation tests
- Run verification and stop
tags:
- task
- lrs3
- wav2lip
- syncnet
- historical-evidence
- synthetic-contract
completed: '2026-08-30'
---

# H-only synthetic contract hardening

## Observations
- [description] 只完成 H-only data-free synthetic contract hardening；不运行真实媒体、模型或训练。
- [status] done
- [assigned_to] downstream-agent
- [current_step] 6
- [openspec] `harden-h-only-assay-synthetic-contracts`
- [parent_task] [[H-only historical endpoint sensitivity assay]]
- [decision] synthetic 结果永久 non-authoritative；P1 固定 exact 4-record；P2/P3 从完整 facts 派生；phase gate 绑定 decision/hash；real authorization 外部提供。
- [verification] focused 27 passed、LRS3 experiment regression 129 passed、Ruff、compile、OpenSpec strict 均通过。
- [stop] synthetic hardening 已达到停止条件；metadata/artifact freeze、real P0-P3、training、D/V/T/S 和 cohort expansion 仍未授权。
## Steps
1. [x] 将默认状态改为 non-authoritative `SYNTHETIC_CONTRACT_PASS`，并让 synthetic decision无法解锁P0。
2. [x] 实现exact 4-record P1 authority classifier，从A/B/C repeat facts派生near-tie offset与witness。
3. [x] 收紧P2/P3 synthetic fact completeness，移除caller boolean/aggregate scalar捷径。
4. [x] 将runtime phase gate绑定canonical previous decision/hash，并在real stub前校验external authorization。
5. [x] 补齐static/runtime no-training escape-path、parameter/buffer mutation与zero media/model call tests。
6. [x] 运行focused/full regression、Ruff、compile与OpenSpec strict后STOP。
## Context
Source of truth:

```text
openspec/changes/harden-h-only-assay-synthetic-contracts/
```

Implementation is limited to `historical_endpoint_sensitivity.py` and its focused tests. The synthetic CLI now emits only `SYNTHETIC_CONTRACT_PASS` with `execution_mode=synthetic`, `authoritative=false`, `phase_unlock_allowed=false`, and `real_artifacts_frozen=false`; it does not load media/model/training dependencies.

P1 uses one exact four-record authority path. Each record requires A/B/C exact three-repeat facts, derives 31-point near-tie offsets, gaps, D/C and B witness, and ignores caller `b_witness`, offsets and gaps. P2 derives activity from candidate/A repeats, PCM identity/MAE, decoded RGB, ROI MAD, mux identity, curves and provenance. P3 requires complete pristine/historical branches, derived PCM, mux identity, repeat determinism, full curves and common-input identity; aggregate scalars cannot witness.

The phase gate validates canonical decision self-hash, expected previous phase/status, real authority fields and inventory/assay identity. The real CLI validates an externally supplied exact-schema authorization before returning `REAL_MEDIA_RUNNER_NOT_IMPLEMENTED`; it never creates or modifies authorization. Static and runtime no-training guards cover optimizer imports/construction, training entrypoints, backward/step, parameters, buffers, modes and checkpoint state.

focused 27 tests, LRS3 experiment regression 131 tests, Ruff, compileall, `openspec validate harden-h-only-assay-synthetic-contracts --strict`, and `git diff --check` all passed. The child task is complete for this scope, then STOP. Real inventory/artifact freeze, official Wav2Lip/SyncNet execution, real P0-P3, training, D/V/T/S and cohort expansion remain separate and unauthorized.
## Relations
- parent_task [[H-only historical endpoint sensitivity assay]]
- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]
- relates_to [[Raw TTS strict replacement NO-GO]]
- relates_to [[LRS3 MFA-linear TFG native/replacement 2026-09-13|LRS3 MFA-linear replacement NO-GO]]
