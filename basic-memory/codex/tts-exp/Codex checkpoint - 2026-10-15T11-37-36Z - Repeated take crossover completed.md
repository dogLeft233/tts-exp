---
title: Codex checkpoint - 2026-10-15T11-37-36Z - Repeated take crossover completed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-15-t11-37-36-z-repeated-take-crossover-completed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5a8d006892aae7ce937c4b1dc48fc8e64ec48c24
started: '2026-10-15T11:37:36+00:00'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-15T11-37-36Z - Repeated take crossover completed

## Summary

The authorized eight-source repeated-take experiment is complete and reproduces large Sync-C losses inside both natural and TTS classes, changing the interpretation of the earlier natural/TTS crossover.

## Story

The original project investigates TTS talking-face score gains and a transferable natural-audio head. The human paused the head, questioned whether crossed natural/TTS scoring simply penalized different speech realizations, and explicitly authorized this small control. ROOT designed/froze the estimands and maintained BM, while the original GPT-6 Luna/max agents prepared assets, code and execution. Four-speaker16-video pilot extended to eight/32 only on engineering/runtime completion, without consulting effect signs. No head, old training, centers, Dev/Confirm or closed audit was restarted.

Eight same-speaker/same-text natural pairs and two independently seeded full-ICL Qwen takes per source yielded32videos and128score cells. Sentence RMS was fixed; each recording retained its actual clock; source PCM, fixed face/ROI and common real31lag support were used. Every source had both natural and TTS within-class loss positive. Natural within/cross losses were3.650/3.963 and TTS4.275/4.708; extra losses0.314[-0.579,1.656] and0.433[-0.101,1.006], ordinary95% speaker bootstrap. The substantive finding is ordinary instance mismatch reproduces the dramatic off-diagonal decline. Do not interpret the old N/T interaction as independent source preference or a generator-quality share. The intervals are not an equivalence test or proof of zero source effect. Native self-paired TTS-minus-natural C remained+0.568[0.096,1.100], a compound score contrast.

Cloud Jan22 cloning enrollment authenticated but synthesis returned404 before any waveform. Existing local Qwen0.6B FP16 failed numerically; a separate, pre-output frozen FP32 protocol succeeded with identical texts/references/seeds/sampling. All failures were preserved, no installs/downloads or score-driven regeneration. The key authenticated32 Qwen Audio3.0 ASR requests with0retry, entered only through a no-echo private PTY and never stored. Current TTS is the local provider, not historical cloud data. AISHELL has distinct utterances but no formal retake/session labels or guaranteed unseen pretraining status. LibriSpeech remains excluded from independent inference.

## Working State

Authoritative final report: `tts-exp/docs/experiments/44-repeated-take-crossover`, file `basic-memory/docs/experiments/44-repeated-take-crossover.md`. Task `tts-exp/codex/tts-exp/repeated-take-crossover-20261015` is done; spec concluded. Research map and paper-writing topic reflect the correction; short HANDOFF overrides the old head snapshot. CONTEXT is99lines. There is no pending authorized experiment; all scientific processes exited and the LAN remains running, head paused.

Run root `runs/repeated_take_crossover_20261015_attempt01/`, mirrored on `/home/wjj/tts-audio/tts-exp/` at wjj@192.168.50.2:22. Actual source32/audio32/video32/features64/distances128, score/launch/completion receipts, cloud failures, original TTS and32MFA TextGrids are local and LAN machine-local copies, not a certified cloud backup. Latest protocolv3 SHAf86fa38175cd8adecadc2c638b6f90f5862824d57d21c545a9fd24e9b1f3369d; fullconfig1ccc73a749d3027bb39d81b09100f389512a0300aac927812044c6a88435449d; matrix66615d8910e6d3455902370d087981b4782de72338aa8944fc9ce4c35ab084ae. Do not repeat launches from older checkpoints.

Independent ROOT actual QA `root_actual_qa.json` SHA2e0721cbe90f1642d0a4a507cd90671f1946133af9d609d709048fd28e64c438 passed original arrays,64embeddings,128distance reconstructions and all source statistics; max NumPy-versus-Torch float32 distance error3.815e-6, reduction/bootstrap error0, clock error0, max PCM16 RMS quantization error1.100e-6. Source equal units8,2000bootstrap seed20261015; query counts79–121;10boundary cells, none on diagonal. No implementation tests were added or run.

Audio timing diagnostic in `timing_diagnostics_canonical32/`:32originalTextGrids,48role-pairs/1967matchedphone rows, final summary SHAcdd41398f41efb520017c17513ae497883ee5b7728fa4004ba1d8a839ae7e1e6. Onset-adjusted center MAE natural0.497s/TTS0.414s/cross0.692s; these are MFA estimates and local time remains different. All32ASR hypotheses are retained at `cloud_asr_qc_attempt01/`, with numeric/proper-name/technical-word ambiguities and no human phonetic adjudication; no samples were excluded. Thus same input transcript is not a guarantee of exactly identical spoken phones.

Successful TTS GPU wall5.23min,32video+score5.44min; MFA CPU4m14s and ASR243s. Figures/tables live in `analysis_delivery_attempt01/`. ROOT viewed the heatmap and corrected paired-loss preview; use `paired_losses_readable_v2.svg/.png`, SVG SHA658571cd73433e7c5cb3bb7ba898f7b2dad38a0fe75e49a85efadaee0a5175b1. Original posthoc analyzer a46beabb execution bytes were not retained; current plot-fixed1201184 is committed, and `plot_v2_receipt.json` explicitly records the narrow historical source gap. Do not pretend its bytes match the first analyzer. Original scores/raw arrays and independent ROOT recalculation are preserved and current tools can recompute values.

## Changed Files

Local Git commit `5a8d006892aae7ce937c4b1dc48fc8e64ec48c24` adds only7dedicated scripts under `scripts/experiments/repeated_take_crossover_20261015/` (cloud TTS/client diagnostic, local Qwen, matrix runner, ASR QC, timing, delivery). It excludes9inherited tracked changes and1004untracked entries. BM report44, concluded spec/done Task, evidence map, experiment index, paper topic, CONTEXT and HANDOFF were updated outside that commit. Runs/BM/scratch remain local artifacts.

## Verification

Actual TTS16, video32, matrix128, ASR32 and MFA32 completed with their OS/HTTP receipts; all requested local transfers and hashes verified. ROOT read original numerical arrays once and checked both figure previews. BM reindex/readback confirmed report concluded and Task done, exactly one status observation and head paused; CONTEXT99. Public GitHub REST confirms canonical repository and0exact-head PR; new commit lookup422 means local/unverified remotely. Same-chat search385notes/no more pages and direct-read exact agent/session/started selected10-50-21Z as predecessor. Original prior advice checkpoint is superseded by this completed report/task, not by new head authorization.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- Local/unverified remotely commit `5a8d006892aae7ce937c4b1dc48fc8e64ec48c24`.
- Report44, `root_actual_qa.json`, frozen fullconfig and original phase receipts are the primary resume pointers.

## Observations

- [result] Same-class independent realizations produce natural3.650/TTS4.275 Sync-C loss; dramatic crossed loss is not source-specific.
- [decision] Reinterpret prior N/T strong interaction as instance pairing dependence; pure source claims require additional local-clock/sham controls.
- [decision] Keep head and old experiments paused; this bounded pilot is complete, not an instruction to extend automatically.
- [next_step] Await the user's next publication or controlled-confirmation scope, using report44 and the updated evidence map; do not launch GPU work from stale checkpoints.

## Relations

- continues [[Codex checkpoint - 2026-10-15T10-50-21Z - Repeated take crossover and local Qwen dispatch]]
- relates_to [[44-repeated-take-crossover]]
- relates_to [[repeated-take-crossover-20261015]]
- relates_to [[研究主线与证据地图]]
- relates_to [[论文实验数据整理与写作主线]]