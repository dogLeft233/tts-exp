---
title: Natural-conditioned local Wav2Lip control direction
type: experiment
permalink: tts-exp/experiments/natural-conditioned-local-wav2-lip-control-direction
status: no-go
experiment_id: natural-conditioned-local-wav2lip-control
route: natural-only
capture_scope: stage-00-protocol-lock
tags:
- lrs3
- wav2lip
- natural-only
- replacement
- protocol
- planned
---

# Natural-conditioned local Wav2Lip control direction

## Context

在 strict raw-TTS replacement 2×2 与 phone-aligned TTS-mel causal intervention 均为完整 NO-GO 后，开启一条不同的 natural-only 路线。目标保持不变：由冻结 Wav2Lip 生成 candidate video，再把音轨替换为 untouched natural audio `N` 后，正式 Target 必须同时改善 official Sync-C 与 Sync-D，相对 `Wav2Lip(face,N)+N` baseline。旧 TTS residual、MFA/TTS alignment、TTS teacher cache 与 candidate diagonal 不进入新路线。

## Hypothesis

在 natural audio 的 Wav2Lip mel `M_N` 附近，可能存在一个 bounded、smooth、跨 source group 可泛化的 local control direction。先以 per-clip natural-derived oracle 验证存在性，再蒸馏成只读取 natural input 的 mel policy 和 exact-length waveform head；oracle 本身不是 deployable result。

## Protocol

- LRS3 frozen source-group protocol：24 fit groups、6 internal-dev groups、6 validation groups、7 sealed test groups。
- fit 在任何训练前再固定为 18 fit-train + 6 fit-selection；validation/test 不参与 search 或 model selection。
- 新路线只允许 natural Wav2Lip mel 作为 deployable policy/head 输入；不读取 TTS、MFA、face、visual、source-group ID 或 SyncNet score。
- official file-level replacement evaluator 是唯一权威；candidate diagonal 只能作 diagnostic。
- 每个阶段要求独立 code review、hash-bound `code_review.json`、运行后 `artifact_review.json`，并将两者与 scientific `decision.json` 分离。

## Planned stages

1. natural-only protocol lock 与 test lock 验证；
2. differentiable proxy / official mel seam parity；
3. fit-only per-clip local mel oracle 与 held-out temporal-anchor/official fit confirmation；
4. fit oracle label expansion；
5. shared natural-only mel policy 与 internal-dev zero-shot；
6. sample-exact waveform reachability；
7. natural-only waveform head；
8. frozen internal-dev 与 validation official replacement；
9. 第二个 TFG family zero-shot 后才允许讨论 plug-and-play；双 family validation GO 前 test 继续 sealed。

## Observations

- [status] planned
- [hypothesis] natural-conditioned local control may be more compatible with final natural audio than copying TTS mel residual #natural-only
- [constraint] prior TTS-motion and phone-aligned TTS-mel transfer routes are closed after complete scientific NO-GO results #negative
- [requirement] every stage needs source/test hash-bound code review and independent artifact review #reproducibility
- [requirement] candidate diagonal, validation/test scores, and TTS-derived inputs cannot select or prove success #replacement

## Relations

- follows [[LRS3 Wav2Lip phone-aligned mel causal NO-GO]]
- follows [[Raw TTS strict replacement NO-GO]]
- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]
- depends_on [[tts-exp|Natural reference audio available at inference]]
- [status] Stage 00 protocol lock is BLOCKED, not scientific NO-GO: the frozen parent contains fit group `6W2dsnhC18Q` with 3 records, while the pre-registered expansion rule requires the first 4 records per fit group.
- [evidence] On 2026-08-23, all 24 fit-group counts were inspected without score-based replacement or validation/test access; 23 groups meet the minimum and `6W2dsnhC18Q` is the only short group.
- [artifact] `runs/lrs3_natural_control_20260823/00_protocol_lock_retry1/` contains failure manifest, summary, and decision artifacts with `decision=BLOCKED`, `next_allowed_stage=null`, and no test lock.
- [decision] Preserve the fixed four-per-group rule and fail closed; do not silently reduce the count, substitute another sample, or continue to Stage 01 until the protocol policy is explicitly revised and re-reviewed.
- [review] Independent code review and artifact review both PASS after binding current source/test hashes and the three BLOCKED artifacts; this validates implementation integrity, not scientific GO.
- [status] On 2026-08-23, Stage 00 retry2 reached `GO` after an explicit expansion-policy revision; the prior retry1 BLOCKED state is historical and remains preserved.
- [decision] Adopt `uniform_common_minimum_v1`: retain all 24 fit groups, select exactly 3 records per group by ascending `sample_id`, and prohibit score-based substitution.
- [artifact] `runs/lrs3_natural_control_20260823/00_protocol_lock_retry2/` contains `GO` decision, 24 pilot records, 72 expansion records, a 54/18 fit-train/fit-selection expansion split, and a sealed unvisited test lock.
- [review] Retry2 code and artifact reviews both PASS under `_reviews/00_protocol_lock_retry2/`; current source and all four retry2 artifacts pass hash-bound validation.
- [gate] Stage 01 `01_proxy_parity` is now the only allowed next stage; no GPU or test media was used for the protocol lock.
- [final-status] Stage02B held-out internal-dev policy generalization completed with scientific NO_GO; mean ΔC=-0.0100, mean ΔD=-0.0107, strict joint win 1/6. This natural-only representation is closed and must not be resumed from the stale Stage01 next-stage text.
- [superseded] Future work follows the separately preregistered TTS visual-teacher chain rather than tuning the natural-only policy.

## Updated Relations
- superseded_by [[TTS 视觉教师到 replacement-safe 音频控制链路]]
- evidence [[Stage02B policy generalization NO-GO]]