---
title: Codex checkpoint - 2026-10-15T02-36-18Z - Primary predictive audit complete
  and extensions bounded
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-15-t02-36-18-z-primary-predictive-audit-complete-and-extensions-bounded
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-15 02:36:18+00:00
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

# Codex checkpoint - 2026-10-15T02-36-18Z - Primary predictive audit complete and extensions bounded

## Summary

The existing-data primary audit separates strong TTS/natural classification from unsuccessful paired Sync-C gain prediction; full OOF statistics and independent arithmetic QA have passed, while two bounded secondary strata remain before the Chinese report.

## Story

The original project sought mechanisms and a transferable natural-audio enhancement head. The human paused that research. The active, separate goal now audits seven processing/acoustic/timeline feature families using existing assets, inspired by the CVPR2025 leading-silence shortcut paper. Read the complete attachment objective before any goal continuation. No new TTS, ASR, TFG or SyncNet forwards, GPU work, enhancement training or causal interventions are authorized by this audit.

ROOT owns interpretation, compact protocol, data admission, BM and report. Three original GPT-6 Luna/max agents own assets/features, fixed logistic/ridge statistics, and independent QA. We reuse qualified cached outputs, keep source layers and score contracts separate, and stop asset recovery after bounded manifest-led lookups. Historical cohorts are not prospective confirmation.

The primary 74-pair/13-speaker generator-layer cohort is complete. Group-held-out acoustic L1 AUC is 0.971692, joint AUC 0.971363. Joint seven-family L2 R2 is -0.139984 and MAE 0.849871, versus training-mean baseline R2 -0.006278 and MAE 0.742632. Metadata-only R2 is slightly positive, 0.012500, but MAE is worse than baseline. No family passed the frozen adjusted positive-increment plus relative-threshold rule. Thus easy condition discrimination has not become reliable individual gain prediction in this cohort and fixed model class.

Existing curve decomposition adds a useful boundary: source-PCM mean delta C +0.935 has uncertain fixed-zero-lag delta D0 (-0.072, 95% CI -0.243 to +0.099); legacy higher delta C +1.032 accompanies worse D0 +1.016. These are scorer responses, not independent visual-quality evidence. Do not turn R2 or B/D arithmetic into a causal explanation percentage.

## Working State

Authoritative task is [[processing-chain-predictive-audit-20261015]]. Run root is `runs/processing_chain_predictive_audit_20261015_attempt01/`. Its living Task/map still describe the earlier smoke/full-feature phase and need updating to the completed primary results. Other research HANDOFF, automation and head Tasks must not be overwritten. The old enhancement line remains paused, including finite-six source-only work.

Primary `full_execution_attempt01/readback_attempt01/model_full_attempt01/analysis.json` SHA f5a1903a88b428ac70cf0d6045d158f8a08091f2d39a97a005eff06a4a6ae1d5 is authoritative. The full worker exited; PASS receipt records 21.835 seconds. Direct OS wait returncode was not captured and remains null; do not infer rc0 or rerun completed analysis for bookkeeping. `qa_C/final_primary_oof_independent_qa_attempt01.json` SHA f4ad67b83f10b37752fba394daaa0c7df6768abda8a41aa87241c93567c559bc closes all 21 checks: joins, folds, metrics, all primary/ablation intervals, and null arithmetic. Main feature source/cache and primary QA are closed.

English original-layer extension has 26 source WAVs, 13 pairs/13 LibriSpeech speakers, qualified copies, and completed 5.861-second extraction. No English fit has run. A metadata-only repair corrects source-native sample count paired with 1/source-rate timebase for 24k TTS; original decoded16k count is retained in nonpredictor provenance. Corrected CSV `feature_english_original_metadata_attempt02/features.csv` SHA a51b29f2b5987d408071faa4d5207ef4a4f0c80153b21ae50203e53d66d08565, source-binding sidecar `inventory_A/secondary_assets_attempt02/english_timeline_metadata_source_binding_attempt01.json` SHA 5ba186de93826e0a048caeb31b9d1bc8e5f9b984fb343a682c10d2d01d1a2e31 await ROOT actual inspection and full input binding. Preserve old CSV/cache/receipts; no re-decode. English L2 is unavailable because source/video/score contract join is unqualified.

Legacy extension attempt04 has 148 bodies/74 paired scores/13 groups, both asset layer and score feature layer corrected to scorer, separate `legacy_mux_audio_guard20` contract. No legacy extraction or fit has run. `legacy_scorer_layer_binding_receipt_attempt04.json` SHA 05ac8b80306802c7939a617f2d4a1123aa893f60e0dbf2e20968be852ef162b2 names exact assets/scores. The plan attempt03 contains obsolete nested invocation fields; only the new `invocation_fields` dictionary points to attempt04 and scorer. ROOT must publish one unambiguous child binding before A extracts the 148 waves using already-reviewed fixed feature wrapper/core. Do not reuse old03 mux-layer inputs.

Multiset source-group membership has been corrected offline to 139 groups across five datasets, but all 250 expected paired TTS WAVs are absent in the bounded route. Report15 has 100 Ditto pairs/15 speakers but only10/400 exact required paths and no score-body binding. Codec47 speaker mapping remains UNKNOWN, not confirmed single-speaker. These are qualified coverage gaps; no further broad scans or restoration.

Machine-local assets include ignored BM/run/scratch and remote LAN copies. Git has 9 tracked dirty and 1004 untracked entries, largely inherited; no new commit or PR. LAN CPU uses the already-qualified shared Python environment, offline, CUDA hidden, max2 threads. Secrets never enter files, argv, BM or logs.

## Changed Files

- `scripts/experiments/processing_chain_predictive_audit_20261015/`: reusable fixed decoder/features, simple grouped modeling CLI and data-only plotting.
- Dedicated run: protocol/full bindings, source inventory, features/cache, paired scores, folds/OOF/statistics, independent QA and literature/history evidence.
- `basic-memory/codex/tts-exp/processing-chain-predictive-audit-20261015.md` and `basic-memory/Research/研究主线与证据地图.md`: task/navigation owners; update only the audit section.

## Verification

ROOT and C qualified primary 148 decoded WAVs with smoke-cache reuse, actual UID/score/source joins, units and leakage exclusions. C independently recomputed primary 21 L1 and 23 L2 models' OOF metrics, 2000-source-bootstrap intervals, seven-family paired increments and 199 grouped nulls from actual outputs; numerical differences are at floating-point rounding only. ROOT independently verified existing curve endpoint formulas and group summaries. No whole-repo tests, model forwards, training or repeated primary audit were run. Plot smoke failed at None-CI finalization; saved-result rendering repaired it without refitting. Earlier attempts remain intact.

## References

- [Repository](https://github.com/dogLeft233/tts-exp); exact-head PR query returned none. Current SHA `5688784dd5c6c1272074f0c4b6c5ff79f117c907` is local/unpushed (GitHub commit lookup422), not a verified pushed commit link.
- Complete objective: `C:/Users/dogle/.codex/attachments/d8c5d710-fd31-4ca1-82de-dd63d5abe328/goal-objective.md`.
- `literature/evidence_chain.md`, `history/evidence_inventory.md`: original-paper evidence and historical causal boundaries.
- `protocol_full_stage_v1.json`, `analysis_input_binding_full_primary_v1.json`: frozen primary execution definitions.

## Observations

- [result] Primary acoustic TTS/natural discrimination is strong, but the fixed seven-family model does not improve held-out paired Sync-C gain prediction over the training mean.
- [decision] Keep source/generator/scorer layers and native/PCM/replacement contracts separate; B/D/offset and query counts never become C predictors.
- [decision] The enhancement-head research stays paused. Complete only the independent existing-data audit, with negative or unavailable outcomes accepted.
- [blocker] English score/source binding is unqualified; codec47 source grouping and multiset/report15 paired bodies are insufficient in the bounded lookup, so those predictions remain unavailable.
- [next_step] Admit the corrected legacy scorer-layer feature batch and English metadata-only table, run their already-frozen secondary statistics once, then independent new-stratum QA and the Chinese final report.

## Relations

- continues [[Codex checkpoint - 2026-10-15T01-36-02Z - Observational audit bound and feature smoke passed]]
- relates_to [[processing-chain-predictive-audit-20261015]]
- relates_to [[研究主线与证据地图]]