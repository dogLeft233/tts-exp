---
title: Codex checkpoint - 2026-10-15T04-41-45Z - Predictive audit proposal interpretation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-15-t04-41-45-z-predictive-audit-proposal-interpretation
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-15T04:41:45.255508+00:00'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a132d7-ec88-7b91-97b7-31d2625bf033
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-15T04-41-45Z - Predictive audit proposal interpretation

## Summary
Reviewed completed audit43 and interpreted its findings as proposal motivation for separating TTS condition markers, SyncNet score mechanisms and actual audiovisual synchrony.

## Story
The user wanted an autonomous two-layer audit following Smeu et al. CVPR2025, modeled after the existing experiment session. Earlier in this chat we inspected that workflow and supplied a root-led, delegated target prompt, then explicitly added the paper and its evidence mapping. No task or automation was created or messaged here; no audit experiment was executed by this chat. The latest user reports completion and asks what it contributes to the proposal. We read the actual completed report, current corrections and delivery manifest instead of inferring results from the prior plan.

The result supports a focused preliminary observation: TTS-like audio is easy to identify, but those differences do not reliably predict per-utterance gains in the primary cohort. Processing-contract dependence and evaluator-distance diagnostics motivate investigating score validity before selecting an enhancement target. This is not proof of a universal shortcut explanation and does not establish a successful enhancement model.

## Working State
Canonical evidence: `basic-memory/docs/experiments/43-processing-chain-predictive-audit.md` and `runs/processing_chain_predictive_audit_20261015_attempt01/delivery_manifest_final_v2.json`. Main Wav2Lip cohort has 74 pairs/13 actual speakers. Grouped held-out acoustic classification AUC0.972; seven-family gain regression R2 -0.140, MAE0.850 versus mean-baseline0.743. No family passed corrected positive-increment criteria. AUC is ranking discrimination, not accuracy; negative prediction R2 is not negative physical causality.

Same-pair legacy contract has limited aggregate prediction R2+0.095 (relative-silence+0.175), wide intervals and no stable single family. Contracts cannot be pooled or their prediction difference called causal codec attribution. Source PCM Delta C+0.935 with zero-lag distance interval crossing zero; legacy Delta C+1.032 with zero-lag distance worsening+1.016. Confidence gain is not direct mouth or perceptual improvement.

LibriSpeech is excluded from proposal support following user-reported leakage; exact upstream contamination was not reverified. Cross-TFG prediction and same-layer cross-domain holdout were not measured. Codec/sample-rate constants and missing timing fields are non-identifiable/unmeasured, not evidence of no causal role. Enhancement-head research remains paused; discussing future work does not authorize restarting it.

The proposal can use these results as preliminary evidence for controlled separation of fixed-video evaluator sensitivity, generator response evaluated against unchanged natural audio, and independent visual/human synchrony. A real-only alignment model is a possible later approach, not a requirement proved by this audit. Small new-source causal controls would test the hypotheses; they were not executed here.

Earlier recovered high-gain historical media remain under `recovered_media/wav2lip_a1_007/` (Delta C+2.162) and `recovered_media/wav2lip_s0912_202/` (+1.402), with score JSON and READMEs. Their native score examples are not visual-quality proof.

## Changed Files
Only this immutable checkpoint was added. No code, report, map, HANDOFF, task or automation was changed. Git status had1013 pre-existing dirty/untracked entries, not attributed to this request.

## Verification
Read Startup Router, research map, experiment instructions, report43, manifest and selected existing QA. Recomputed SHA256/byte sizes of primary OOF QA, legacy L2 QA and auxiliary summary QA: all3 match manifest. This is artifact-integrity verification, not an independent model rerun. No scientific forward, fitting or tests ran here.

Read Git branch/root/SHA/status. GitHub API confirmed repository and returned no PR for this branch. HEAD lookup returnedHTTP422, so SHA remains local/unverified remotely. gh is unavailable. Metadata search found2 same-chat checkpoints with no more pages and no legacy matches; directly read latest previous frontmatter and verified exact agent/session identity. Initial checkpoint CLI attempt failed on Windows stdin encoding before creation; retry explicitly sets UTF8. CLI source confirms omitted overwrite flag passes overwrite=False and output_format=json explicitly.

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- Local HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907` (not verified on remote).
- `basic-memory/docs/experiments/43-processing-chain-predictive-audit.md`, sections7-11 and eligibility correction.
- `runs/processing_chain_predictive_audit_20261015_attempt01/delivery_manifest_final_v2.json`.

## Observations
- [result] Strong condition discrimination does not yield reliable primary held-out gain prediction with the frozen seven-family model.
- [decision] Separate observational prediction, historical causal effects and visual/perceptual evidence; do not convert a TTS classifier direction into an enhancement target.
- [blocker] Cross-model prediction, clean cross-domain validation and some factor-specific identification remain absent.
- [next_step] Draft proposal preliminary-results and research-question paragraphs around measurement validity and generator-versus-evaluator mechanisms, using bounded future causal tests and independent synchrony endpoints.

## Relations
- continues [[Codex checkpoint - 2026-10-14T04-48-13Z - Remote Wav2Lip and B-drive inventory]]
- relates_to [[43-processing-chain-predictive-audit]]
- relates_to [[研究主线与证据地图]]