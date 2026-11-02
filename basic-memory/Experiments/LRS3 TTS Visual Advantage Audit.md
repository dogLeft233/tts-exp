---
title: LRS3 TTS Visual Advantage Audit
type: experiment
permalink: tts-exp/experiments/lrs3-tts-visual-advantage-audit
status: blocked
date: August 24, 2026
primary_endpoint: natural-clock exact-time canonical mouth landmark benefit
secondary_endpoint: visual-only constrained DTW shape benefit
test_status: sealed_unvisited
tags:
- lrs3
- tts
- wav2lip
- visual-motion
- landmarks
- replacement
- preregistered
---

# LRS3 TTS Visual Advantage Audit

本实验检验 TTS 驱动的冻结 Wav2Lip 视频是否在视觉嘴部运动上比 natural 驱动视频更接近同一条 LRS3 real paired video。它不是新的 candidate-audio replacement 训练，而是判断是否存在值得蒸馏的 TTS visual teacher signal。

固定三臂：`G` 为真实 paired LRS3 视频，`V_N` 为 original natural audio 驱动的 official Wav2Lip 视频，`V_T` 为 reference-conditioned raw TTS audio 驱动的视频。primary endpoint 是不允许 shift/stretch/warp 的 natural-clock exact-time canonical mouth-landmark distance；secondary endpoint 是 visual-only constrained DTW shape distance。SyncNet、audio/phone/MFA alignment 和 candidate scores 不参与视觉判定。

已有 `02_landmark_teacher_audit_retry2` 是已见负证据：60 records 中 45 accepted，median teacher benefit `-0.0584`，positive `1/60`，但旧源码和 durable review 已缺失。新实验先复用旧 fit cache 做 comparator engineering parity，再在 24 fit groups 每组一条、与旧 cache disjoint 的 fresh confirmation cohort 上检验。internal-dev、validation 和 84 个 sealed test IDs 均不访问；若 fresh fit confirmation 不通过则停止。

## Observations
- [status] blocked
- [question] TTS-driven Wav2Lip video 是否在 natural clock 或 content-warped mouth shape 上比 natural-driven video 更接近真实 LRS3 paired motion？
- [protocol] Pilot 使用旧 cache 中 24 fit groups 的 48 records；confirmation 从每个 fit group 固定选择第一条未进入旧 cache 的 record，共 24 records/24 groups。
- [primary-endpoint] Exact-time canonical mouth-landmark distance，禁止任何时移、伸缩或 DTW。
- [secondary-endpoint] Visual-only constrained DTW mouth-shape distance，只能解释 shape-only advantage。
- [stop-rule] 两个 endpoint 在 fresh fit confirmation 均 NO_GO 时停止，不访问 internal-dev/validation/test，不做 motion distillation。
- [prior] 旧 landmark audit 为 NO_GO 且来源审查不耐久，因此作为 pilot prior 而非独立 confirmatory result。

## Relations
- followed_by [[TTS 视觉教师到 replacement-safe 音频控制链路]]
- superseded_for_future_work_by [[执行 TTS 视觉教师到 replacement-safe 音频头链路]]

- follows [[Natural-conditioned local Wav2Lip control direction]]
- relates_to [[LRS3 Wav2Lip phone-aligned mel causal NO-GO]]
- relates_to [[Encoder-only SyncNet 微调与固定视频迁移性]]
- part_of [[tts-exp]]
- [status] Stage 01 retry7 completed on 2026-08-24 with engineering gate BLOCKED; scientific decision remains unavailable.
- [artifact] Retry7 produced 48/48 successful extractions, 48 records, 144 feature archives, and run-local snapshots with matching hashes.
- [coverage] Only 28/48 records met the complete eligibility gate (minimum 44); only 20/24 fit groups had an eligible record. The main failure was natural-clock exact-time coverage, not decode or metric non-finiteness.
- [direction] Among 28 records overlapping the old landmark audit, exact-time benefit sign agreement was 24/28 (0.8571428571428571); this is engineering context only and not a scientific decision.
- [sealed] Internal-dev, validation, and test media/features/scores remained unopened; the Stage 00 test lock remains sealed_unvisited.
- [stop] Fresh fit confirmation is blocked and must not run unless a separately reviewed retry restores the preregistered comparator coverage gate without changing thresholds.
- [investigation] Coverage investigation retry1 completed at `runs/lrs3_tts_visual_advantage_20260824/01_metric_parity_investigation_retry1/coverage_investigation.json`; artifact SHA-256 is `d1b165dcff3d4985cf18f7a519edb0aa30882fc71d721587ab5063fb39df9658`.
- [decomposition] All 20 ineligible pilot records failed exact-time coverage; 11 failed coverage only and 9 also failed the three-arm valid-fraction gate. All metric fields remained finite, with no snapshot hash, frame-count, or extraction failure.
- [timing] Recomputed TTS clock-pair coverage had mean 0.8798801993965659 and median 0.8881304512703829, versus natural mean 0.987490820046637 and median 0.9893617021276596; TTS temporal-tail loss is the primary engineering blocker.
- [provenance] Frozen TTS metadata permits duration ratios from 0.5 to 1.5; 16/48 pilot ratios fall outside a narrower 0.85–1.18 interval used only for diagnosis, showing that raw TTS timing variation is incompatible with the current natural-clock comparator on many fixed records.
- [interpretation] The investigation does not establish or reject TTS visual advantage and does not justify changing the preregistered thresholds. A duration-compatible or duration-preserving audit would require a separately preregistered protocol and must not be presented as the current raw-TTS result.
- [review] Investigation code review is NOT_APPLICABLE because no source changed; artifact review PASS at `_reviews/lrs3_tts_visual_advantage/01_metric_parity_investigation_retry1/artifact_review.json`.
- [prior-positive] Basic Memory 的 AISHELL-1 clean read-speech n=15 结果显示，高质量 paired TTS 经 MFA-linear natural-clock 重合成后相对 natural 的 generation ΔSync-C 为 `+0.964`（95% CI `[+0.650,+1.266]`，13/15），因此不能把当前 raw-TTS audit BLOCKED 解读为 MFA-linear/TTS generation advantage 不存在。
- [domain-boundary] 该 MFA-linear 正效应依赖规整朗读域和 TTS 质量；RAMC/AliMeeting 自发语音试点中 MFA-linear 分别为 `−1.223` 与 `−0.899` Sync-C，不能默认跨域推广。
- [replacement-prior] 既有 AISHELL-1 n25 audio-track replacement 记录显示 MFA-linear original-track Sync-C `6.5691` 降至 natural replacement `5.5034`（Δ `−1.0657`），Sync-D `7.1016` 升至 `8.1166`（Δ `+1.0150`）；因此 MFA-linear 可以有 generation advantage，但该 advantage 在 replacement 后消失。
- [interpretation] 当前 audit 的 raw-TTS duration blocker 与历史 MFA-linear positive generation evidence 并不矛盾：MFA-linear 可能解决 natural-clock 时长问题，但不能单独解决 candidate-driven video 与 original natural audio 之间的 acoustic/representation compatibility。
- [prior-positive] Basic Memory 的 AISHELL-1 clean read-speech n=15 结果显示，高质量 paired TTS 经 MFA-linear natural-clock 重合成后相对 natural 的 generation ΔSync-C 为 `+0.964`（95% CI `[+0.650,+1.266]`，13/15），因此不能把当前 raw-TTS audit BLOCKED 解读为 MFA-linear/TTS generation advantage 不存在。
- [domain-boundary] 该 MFA-linear 正效应依赖规整朗读域和 TTS 质量；RAMC/AliMeeting 自发语音试点中 MFA-linear 分别为 `−1.223` 与 `−0.899` Sync-C，不能默认跨域推广。
- [replacement-prior] 既有 AISHELL-1 n25 audio-track replacement 记录显示 MFA-linear original-track Sync-C `6.5691` 降至 natural replacement `5.5034`（Δ `−1.0657`），Sync-D `7.1016` 升至 `8.1166`（Δ `+1.0150`）；因此 MFA-linear 可以有 generation advantage，但该 advantage 在 replacement 后消失。
- [interpretation] 当前 audit 的 raw-TTS duration blocker 与历史 MFA-linear positive generation evidence 并不矛盾：MFA-linear 可能解决 natural-clock 时长问题，但不能单独解决 candidate-driven video 与 original natural audio 之间的 acoustic/representation compatibility。