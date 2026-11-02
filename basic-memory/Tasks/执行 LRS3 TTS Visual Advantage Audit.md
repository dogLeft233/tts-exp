---
title: 执行 LRS3 TTS Visual Advantage Audit
type: task
permalink: tts-exp/tasks/执行-lrs3-tts-visual-advantage-audit
assigned_to: claude
current_step: 3
started: August 24, 2026
status: blocked
steps:
- 创建 Git branch、Stage 00 protocol/test lock、tests 与 durable reviews
- 实现视觉特征、exact-time/visual-DTW metrics 与 controls
- 运行旧 fit cache pilot engineering parity
- gate PASS 后运行 fresh fit render/audit
- 完成统计、decision、Git 与记忆收口
tags:
- task
- lrs3
- tts
- visual-audit
---

# 执行 LRS3 TTS Visual Advantage Audit

实现并执行 `[[LRS3 TTS Visual Advantage Audit]]`：重建可审查的 exact-time/visual-DTW comparator，先在已见 fit teacher cache 上做 engineering parity，再在每个 fit group 一条的 fresh cohort 上确认 TTS visual advantage。任何开发阶段都不得访问 internal-dev、validation 或 sealed test media。

## Observations
- [description] 实现 fit-only TTS visual advantage audit，完成 protocol lock、视觉指标、pilot parity、fresh confirmation、stage review、Git 与 Basic Memory 收口。
- [status] blocked
- [assigned_to] claude
- [current_step] 1
- [constraint] 不覆盖旧 run，不使用 score-based substitution，不访问 internal-dev/validation/test；GPU 前检查时间窗口与占用。

## Steps
1. [ ] 创建 Git branch、Stage 00 protocol/test lock、tests 与 durable code/artifact review
2. [ ] 实现 sequential decode、persistent landmarks、canonical mouth features、exact-time/visual-DTW metrics 与 controls
3. [ ] 运行旧 fit cache pilot engineering parity；不重新渲染 teacher videos
4. [ ] 若 gate PASS，检查 GPU 后渲染 24 条 fresh fit records 的 natural/TTS arms
5. [ ] 完成 cluster statistics、artifact review 与 GO/NO_GO/BLOCKED decision
6. [ ] 更新 experiment note、Task status、CONTEXT（仅结论改变主线时）和 Git tracking；不 push

## Context
- 计划文件：`/home/wjj/.claude/plans/foamy-knitting-mccarthy.md`
- 仓库：`/home/wjj/tts-audio/tts-exp`
- parent protocol：`runs/lrs3_motion_teacher_20260822/00_protocol_audit_retry1/`
- old teacher cache：`runs/lrs3_motion_teacher_20260822/01_wav2lip_teacher_pilot_retry3/`
- old landmark prior：`runs/lrs3_motion_teacher_20260822/02_landmark_teacher_audit_retry2/`，NO_GO，45/60 accepted，positive 1/60
- old strict replacement：`runs/lrs3_motion_teacher_20260822/11_raw_tts_strict_2x2_retry1/`，joint 0/60
- source manifest：`runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json`
- sealed test：7 groups、84 IDs；当前任务不得打开
- Git 只跟踪 source/tests/small review provenance；runs/media/checkpoints 不提交，且不 push。

## Relations
- project [[LRS3 TTS Visual Advantage Audit]]
- relates_to [[Natural-conditioned local Wav2Lip control direction]]

- [blocker] Stage 00 blocked on August 24, 2026 because `checkpoints/mediapipe/face_landmarker.task` is absent from the repository; no media, features, or scores were opened/created.
- [implementation] Pure visual feature/metric modules and 19 CPU tests exist, but they are not scientifically activated until the approved landmarker asset is available and reviewed.
- [progress] 2026-08-24: 修复视觉模块 review findings：exact-time 改为单调一对一配对并保守计算 coverage；DTW 保留原始帧位置并正确实现 max_run；解码提前终止和 malformed frame fail-closed；invalid derived features 与 persistence schema 加强校验。audit suite 24 tests、相关 frame-quality/prepare-frames 16 tests通过；Stage 00 仍因缺少 checkpoints/mediapipe/face_landmarker.task 阻塞，未打开媒体。
- [review] 2026-08-24: Stage 01 visual feature/metric code review PASS；最终 audit suite 26 tests、相关 frame-quality/prepare-frames 合计 42 tests通过；malformed frame、unknown frame count、strict persistence schema 均 fail-closed。Stage 01 artifact review 仍为 BLOCKED，因为 Stage 00 缺少 checkpoints/mediapipe/face_landmarker.task；未打开媒体、未生成科学 scores。
- [progress] 2026-08-24: Stage 00 protocol lock retry2 GO；Stage 01 metric parity retry7 complete。48/48 extraction succeeded, but 28/48 records passed eligibility versus required 44, and only 20/24 fit groups had an eligible record.
- [status] blocked
- [current_step] 3
- [blocker] Fresh confirmation is not authorized; do not use GPU or open internal-dev, validation, or test media. Retry7 artifact review is at `_reviews/lrs3_tts_visual_advantage/01_metric_parity/artifact_review.json`.
- [review] 2026-08-24: Stage 01 parity code review PASS; artifact review BLOCKED solely on preregistered coverage gate. All snapshots, feature archives, hashes, and finite metric fields verified; sealed test lock remains unvisited.
- [investigation] 2026-08-24: coverage investigation retry1 completed at `runs/lrs3_tts_visual_advantage_20260824/01_metric_parity_investigation_retry1/coverage_investigation.json`, hash `d1b165dcff3d4985cf18f7a519edb0aa30882fc71d721587ab5063fb39df9658`.
- [finding] All 20 ineligible pilot records failed exact-time coverage; 11 were coverage-only failures and 9 also failed the valid-fraction gate. No extraction, snapshot hash, frame-count, or finite-metric failure was found.
- [finding] TTS clock-pair coverage mean/median was 0.8798801993965659/0.8881304512703829 versus natural 0.987490820046637/0.9893617021276596. The frozen TTS metadata duration gate allows 0.5–1.5 duration ratios, creating substantial raw-TTS timing mismatch with the natural-clock comparator.
- [review] Investigation artifact review PASS at `_reviews/lrs3_tts_visual_advantage/01_metric_parity_investigation_retry1/artifact_review.json`; no source or protocol thresholds changed.
- [decision] Keep Stage 01 BLOCKED and do not run fresh confirmation. Any duration-compatible or duration-preserving follow-up requires a separately preregistered protocol version.
- [investigation] 2026-08-24: coverage investigation retry1 completed at `runs/lrs3_tts_visual_advantage_20260824/01_metric_parity_investigation_retry1/coverage_investigation.json`, hash `d1b165dcff3d4985cf18f7a519edb0aa30882fc71d721587ab5063fb39df9658`.
- [finding] All 20 ineligible pilot records failed exact-time coverage; 11 were coverage-only failures and 9 also failed the valid-fraction gate. No extraction, snapshot hash, frame-count, or finite-metric failure was found.
- [finding] TTS clock-pair coverage mean/median was 0.8798801993965659/0.8881304512703829 versus natural 0.987490820046637/0.9893617021276596. The frozen TTS metadata duration gate allows 0.5–1.5 duration ratios, creating substantial raw-TTS timing mismatch with the natural-clock comparator.
- [review] Investigation artifact review PASS at `_reviews/lrs3_tts_visual_advantage/01_metric_parity_investigation_retry1/artifact_review.json`; no source or protocol thresholds changed.
- [decision] Keep Stage 01 BLOCKED and do not run fresh confirmation. Any duration-compatible or duration-preserving follow-up requires a separately preregistered protocol version.