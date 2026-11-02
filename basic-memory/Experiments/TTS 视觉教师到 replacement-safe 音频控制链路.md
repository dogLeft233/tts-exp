---
title: TTS 视觉教师到 replacement-safe 音频控制链路
type: experiment
permalink: tts-exp/experiments/tts-视觉教师到-replacement-safe-音频控制链路
capture_scope: visual-teacher-to-replacement-safe-control
current_stage: stage01
experiment_id: tts-visual-replacement-control
status: blocked
tags:
- tts
- visual-teacher
- replacement
- wav2lip
- audio-head
- protocol
---

# TTS 视觉教师到 replacement-safe 音频控制链路

## Context

本协议把研究拆成五个可判定瓶颈：真实 TTS visual teacher 是否存在；该 teacher 能否映射为 natural-compatible Wav2Lip seam control；control 是否能由 sample-exact waveform 实现；shared head 是否能在 held-out groups 泛化；同一 waveform head 是否能跨 TFG。每层只在上一层科学 GO 后启动，避免再次把 teacher、alignment、renderer、proxy 和 generalization 混为一个失败。

正式 estimand 始终是：

```text
Baseline = mux(frozen_TFG(face,N), N)
Target   = mux(frozen_TFG(face,C), N)
benefit_C = SyncC(Target) - SyncC(Baseline)
benefit_D = SyncD(Baseline) - SyncD(Target)
```

两个 benefit 同时为正且通过预注册 held-out group statistics 才算成功。

## Stage map

### Stage00 — protocol/test lock
冻结现有 exact-natural-clock MFA3 candidate和133条frozen render cohort；验证全部 fit-only，并绑定 frozen parent 的24-group split与 pre-score common-support 23 policy（有效17 fit-train + 6 fit-selection），与6 internal-dev/6 validation/7 test groups隔离。Stage00不打开媒体。

### Stage01 — visual teacher audit
对 real paired video `G`、natural-driven `V_N`、MFA-linear/TTS-driven `V_M` 使用 exact-time canonical mouth distance：`D(G,V_N)-D(G,V_M)>0` 为正。SyncNet/native score不参与选择或决策；constrained visual DTW仅为secondary。旧0.90 valid fraction与0.85 exact coverage不降低，denominator不足为BLOCKED。

### Stage02 — identity-chain attribution
构造 `A=N` 经M同一frozen encoder/decoder/vocoder链重建；用M-vs-A visual difference隔离TTS信息与codec/interface effect。A不用于搜索natural enhancement。identity失败只禁用该renderer；mel seam仍可单独测试。

### Stage03 — one seam oracle
唯一action family为fit-train上从`M_mel-N_mel`计算的rank-4 PCA residual basis，加四个平滑temporal knots；candidate以N mel为anchor。per-record CEM只看real-video exact-time visual loss，official replacement score在candidate全部冻结后才对fit-selection做一次确认。失败不换rank/mask/alpha或上RL。

### Stage04 — waveform reachability
对winning seam target做bounded direct waveform optimization，同时对N mel做同算法identity target。区分target不可达、renderer失败和waveform生成后replacement benefit丢失。direct waveform是oracle，不是deployable result。

### Stage05–07 — shared head, cross-TFG, sealed test
shared head推理只读N+TTS并输出bounded exact-length waveform，不读video/SyncNet/CEM/group ID。17 effective fit-train/6 fit-selection后依次一次性评估internal-dev和validation；Wav2Lip GO后冻结head，对LeapTalk做zero-shot（不使用Ditto）。双family validation GO前test始终sealed；最终只解封一次。

## Reviews and decisions

每stage必须有hash-bound preregistration、pre-run `code_review.json`、post-run independent `artifact_review.json`、`summary.json`、`decision.json`。Review PASS只证明工程可信。工程不完整=`BLOCKED/scientific not_available`；工程完整但方向gate失败=`NO_GO`。所有retry用新目录，旧artifact不可覆盖。

## Observations
- [goal] 从TTS视觉优势中提取能在换回untouched natural audio后仍保留的TFG控制增益。
- [decision] TTS visual verification是第一科学stage；visual NO_GO即停止后续control/head。
- [decision] natural autoencode/decode只做identity和TTS归因控制，不重复natural-only enhancement search。
- [decision] 新seam oracle使用TTS-derived low-rank subspace和独立real-video visual objective，区别于旧fixed-alpha phone residual与natural-only SyncNet CEM。
- [requirement] official file-level replacement是Stage03以后唯一权威科学终点。
- [requirement] 每个stage独立code/artifact review、immutable retry、split/test lock和Basic Memory更新。
- [constraint] 23:00–08:00禁用GPU；不使用Ditto；不commit/push除非明确授权。

## Relations
- executed_by [[执行 TTS 视觉教师到 replacement-safe 音频头链路]]
- follows [[LRS3 MFA-linear TFG native/replacement 2026-09-13|LRS3 MFA-linear replacement NO-GO]]
- follows [[LRS3 TTS Visual Advantage Audit]]
- follows [[Natural-conditioned local Wav2Lip control direction]]
- constrained_by [[tts-exp|Wav2Lip replacement is primary objective]]
- depends_on [[tts-exp|Natural reference audio available at inference]]
- [result] Stage00 retry2 artifact review PASS: 133 records, 23 effective groups, 17 fit-train + 6 fit-selection, minimum eligible 122, no media opened.
- [result] Stage01 retry4 artifact review PASS at `runs/lrs3_tts_visual_control_20260825/01_visual_teacher_audit_retry4/`; all 133 records extracted and 399 feature artifacts independently recomputed.
- [result] Stage01 engineering decision is BLOCKED, scientific decision is not_available: 115/133 records eligible versus fixed minimum 122, with only 21/23 effective groups covered; missing groups are `6ZiN9ZJT294` and `79tRTivyMSM`.
- [boundary] Do not lower valid/coverage thresholds, substitute cohort records, or enter Stage02; this is denominator incompleteness, not a scientific NO_GO.
- [artifact] Stage01 code review and artifact review are PASS; retry1-3 remain immutable partial/preflight failures.
