---
title: 执行 TTS 视觉教师到 replacement-safe 音频头链路
type: task
permalink: tts-exp/tasks/执行-tts-视觉教师到-replacement-safe-音频头链路
assigned_to: claude
current_step: 2
priority: high
status: blocked
steps:
- Stage00 duration-compatible protocol/test lock
- Stage01 exact-time TTS visual teacher audit
- Stage02 identity-chain and TTS attribution
- Stage03 TTS-informed Wav2Lip seam oracle
- Stage04 sample-exact waveform reachability
- Stage05 shared waveform head held-out evaluation
- Stage06 second-TFG zero-shot
- Stage07 sealed test and closeout
tags:
- task
- tts
- visual-teacher
- replacement
- audio-head
- wav2lip
---

# 执行 TTS 视觉教师到 replacement-safe 音频头链路

## Context

项目最终目标是生成 candidate driver `C = head(N, optional TTS conditioning)`，用 `C` 驱动冻结 TFG，但最终交付与 evaluator audio 始终是 untouched original natural audio `N`。唯一权威比较是 `mux(TFG(face,C),N)` 相对 `mux(TFG(face,N),N)`：Sync-C 必须上升、Sync-D 必须下降。candidate diagonal、fixed-video reward、feature/proxy improvement 和 oracle fit 都不能替代该 endpoint。

本任务不再调或重跑已经完整 NO_GO 的 raw TTS strict 2×2、phone-aligned fixed-alpha mel residual、133-record MFA3/MFA-linear strict replacement、natural-only mel policy、fixed-video CEM 和旧 feature/waveform heads。新链路首先做 duration-compatible TTS visual teacher audit；只有 visual teacher 经真实 paired video 验证后，才依次允许 identity-chain attribution、一个 TTS-informed natural-anchored Wav2Lip seam oracle、sample-exact waveform reachability、shared waveform head 和第二 TFG zero-shot。

完整批准计划位于 `/home/wjj/.claude/plans/foamy-knitting-mccarthy.md`。首次实施范围只到 Stage01 decision；Stage02 及以后不得提前实现。

## Observations
- [description] 分阶段验证 TTS visual teacher、replacement-compatible control existence、waveform reachability、shared-head generalization 和跨 TFG 可迁移性。
- [status] blocked
- [assigned_to] claude
- [current_step] 2
- [requirement] official file-level natural-audio replacement 是唯一权威成功标准。
- [constraint] 23:00–08:00 禁止 GPU；使用前检查 utilization、memory 和 compute processes。
- [constraint] 不使用 Ditto，不覆盖旧 run，不按结果替换样本，不 commit/push 除非用户明确要求。
- [boundary] Stage01 scientific NO_GO 将停止整条 TTS visual-teacher control 路线；不能以 DTW、native score 或 partial positives挽救。

## Steps
1. [x] Stage00：冻结 duration-compatible fit-only cohort、parents/assets/source hashes；父 split 保持 24=18+6，但因 immutable candidate/render parent 结构性缺少 `6W2dsnhC18Q`，本协议固定采用 pre-score common-support 23，有效 split 为 17 fit-train + 6 fit-selection，并保持 sealed internal-dev/validation/test lock。
2. [x] Stage01：复用 exact-length MFA3 candidate 和 frozen renders，对 real/N/M 做 exact-time visual teacher audit；完成 code/artifact review，结果为工程 BLOCKED（115/133 eligible，21/23 groups，minimum 122），scientific decision not_available，Stage02 不授权。
3. [ ] Stage02：仅 visual GO 后做 natural autoencode/decode identity-chain 和 M-vs-A TTS-specific visual attribution。
4. [ ] Stage03：仅 attribution GO 后做唯一 rank-4 TTS-residual PCA、natural-anchored、visual-only searched Wav2Lip seam oracle；official replacement只做冻结后确认。
5. [ ] Stage04：仅 seam GO 后做 direct sample-exact waveform reachability和同算法 identity control。
6. [ ] Stage05：仅 waveform GO 后训练 N+TTS shared waveform head，并依次做 fit-selection、internal-dev、validation。
7. [ ] Stage06：仅 Wav2Lip held-out GO 后冻结 head，对 LeapTalk 做 zero-shot；失败只能称 Wav2Lip-specific adapter。
8. [ ] Stage07：仅双 family validation GO 后一次解封 sealed test并最终收口。
## Context for resumption
- Visual split/test lock: `runs/lrs3_tts_visual_advantage_20260824/00_protocol_lock_retry2/`.
- Raw visual audit blocker: `runs/lrs3_tts_visual_advantage_20260824/01_metric_parity_retry7/`，因 raw TTS clock coverage不足而 BLOCKED，不能降低 0.90 valid/0.85 exact coverage gates。
- Exact-length candidate: `runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/02_candidate_audio_retry4/`.
- Frozen 133-record strict render/evaluation parent: `runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/03_strict_replacement_face_ready_retry7/`.
- Completed MFA-linear NO_GO: `runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/04_statistics_exploratory_retry2/`.
- New namespace: `scripts/experiments/lrs3_tts_visual_control/`, tests under `tests/experiments/lrs3_tts_visual_control/`, reviews under `_reviews/lrs3_tts_visual_control/`, runs under `runs/lrs3_tts_visual_control_20260825/`.

## Relations
- implements [[TTS 视觉教师到 replacement-safe 音频控制链路]]
- follows [[LRS3 TTS Visual Advantage Audit]]
- follows [[LRS3 Wav2Lip phone-aligned mel causal NO-GO]]
- follows [[Raw TTS strict replacement NO-GO]]
- relates_to [[tts-exp|Natural reference audio available at inference]]
- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]
- [priority] high

## Project Relation
- project [[TTS 视觉教师到 replacement-safe 音频控制链路]]
- [protocol] Stage00 implementation now uses pre-score common-support policy `pre_score_common_support_23_v1`: parent split remains 24=18+6, effective cohort is 23 groups with 17 fit-train + 6 fit-selection; exclusion is fixed to `6W2dsnhC18Q` before any visual score.
- [artifact] Stage00 source: `scripts/experiments/lrs3_tts_visual_control/protocol.py`; tests: `tests/experiments/lrs3_tts_visual_control/test_protocol.py`; CPU test result is 11 passed in wav2lip venv.
- [constraint] System Python lacks pytest; only the project venv test result is treated as the CPU test execution.
- [artifact] Stage00 protocol lock completed `GO` at `runs/lrs3_tts_visual_control_20260825/00_protocol_lock_retry2/`: 133 records, 23 effective groups, 17 fit-train + 6 fit-selection, minimum eligible 122, scientific decision not available.
- [review] Stage00 code review and independent artifact review both `PASS`; final source/test CPU suite is 38 passed across the new namespace.
- [context] Stage01 visual teacher audit was started from the frozen Stage00 manifest; no internal-dev, validation, or test media was opened and no Stage02 implementation was authorized.
- [optimization] Stage01 retry1 was stopped before completion because each video initialized a separate MediaPipe landmarker; retry2 reuses one frozen landmarker handle across all 399 videos without changing the visual metric contract.
- [review] Stage01 source review now binds runner, shared video feature primitive, and both test files; final Stage01/shared visual CPU regression is 24 passed.
- [optimization] Stage01 retry3 was stopped after confirming single-handle reuse was still too slow for the 10-minute job ceiling; retry4 uses four spawned CPU workers, each with one frozen landmarker, and merges contiguous chunks in source order without changing metrics or gates.
- [context] Stage01 retry4 was the active media run; earlier retry1-3 remain preserved partial/preflight failures.
- [result] Stage01 retry4 artifact review PASS: 133/133 records extracted, 399 features independently recomputed, 115 eligible, 21/23 effective groups covered.
- [decision] Stage01 engineering decision is BLOCKED and scientific decision is not_available because fixed minimum eligible count is 122 and missing groups are `6ZiN9ZJT294` and `79tRTivyMSM`.
- [gate] Stage02 is not authorized; do not lower valid/coverage thresholds, alter cohort membership, or reinterpret denominator failure as scientific NO_GO.
- [artifact] Stage01 artifacts: `runs/lrs3_tts_visual_control_20260825/01_visual_teacher_audit_retry4/`; artifact review: `_reviews/lrs3_tts_visual_control/01_visual_teacher_audit/artifact_review_retry4.json`.
