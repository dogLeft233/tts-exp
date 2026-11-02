---
title: Codex checkpoint - 2026-10-08T04-25-46Z - Sealed Dev seed and exact missing-source
  recovery after CPU collection failure
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t04-25-46-z-sealed-dev-seed-and-exact-missing-source-recovery-after-cpu-collection-failure
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
username: dogle
branch: codex/cross-tfg-audio-head-20261003
agent: codex
cwd: E:/Documents/tts-exp
trigger: compact
model: gpt-6.1-sol
git_sha: 6126c7d1aeba5498f1f61042a49afe8f1e9ff59f
hostname: LAPTOP-E133GSLE
capture: deliberate
project: tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
started: '2026-10-08T04:25:46.9617050+00:00'
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T04-25-46Z - Sealed Dev seed and exact missing-source recovery after CPU collection failure

## Summary

The installed three-model Dev execution chain has a verified runtime seed; its first CPU collection stopped on three missing shared source files, now isolated by a complete inventory and ready for exact create-only restoration.

## Story

The user wants a usable natural-audio enhancement head across TFG models, with enhanced audio driving generation and the complete original natural PCM restored for delivery. The active goal remains the frozen Fit40/20-speaker step120 candidate, then Dev20/10 speakers with three seeds and Wav2Lip, MuseTalk, Ditto: 360 identity/RMS controls followed by 180 candidate rows. Acceptance requires all three models positive and an equal-model advantage of at least +0.050 over RMS, alongside frozen anchor/distance/content/visual/GUARD20 checks and the full failure denominator. Confirm80 and LatentSync remain sealed. Code and experiments stay with GPT-6 Luna/max agents; root maintains protocol, theory, private SSH, Git, BM and independent QC.

The source graph is now committed locally at `6126c7d1aeba5498f1f61042a49afe8f1e9ff59f`. Root verified all nineteen installed script byte identities against the Git index. An exact source-snapshot copy is checked in under `openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/dev360_samehost_source_snapshot_attempt01.json`; narrow `.gitattributes` overrides preserve the core06 CRLF bytes and snapshot bytes. Existing unrelated edits were preserved. This commit is unpublished.

CPU phase1 actually installed/verified the source graph and returned template07. An initial source-only seed draft discarded qualified generator hooks; full nested comparison caught it before execution. Corrected seed02 preserves the complete template backend semantics. Phase2 q00-q02 then succeeded and seal-seed produced a real 70,742-byte READY_FOR_COLLECTION seed. The following collector failed because `stagea_syncnet_evaluator.py` was absent. This is an execution dependency failure, not an enhancement result; no Dev GPU rows have launched.

Root and the independent agent reviewed a bounded readonly inventory worker before dispatch. The actual inventory covers 43 distinct paths: 40 match exact content pins, only the evaluator, PCM16 lineage helper and `scripts/audio_enhancement/heads.py` are missing, and no source differs. Root downloaded the original 29,902-byte inventory, verified its raw and canonical hashes and inspected all 43 entries. A create-only installer02 now stages those three files, rehashes six related existing helpers, accepts absent inventory byte metadata only with exact SHA pins, and never overwrites differing files. Root has read its full source and index; independent review also passes. Actual install and the fresh recovery request remain pending.

The lesson is to resume from the successfully sealed actual seed rather than replay the failed phase2 request sequence. Fresh recovery outputs binding03/plan03/preflight03 must be collected and independently read back before the detached 360-control GPU launch. Scientific source pins and frozen selection remain unchanged.

## Working State

- Durable project state: ADR017 retains measured pilot capacity with explicit scenario/slack and live gates; GPU floor 18 GiB, RAM floor 16 GiB. Fit40 mechanism evidence supports a spectral/static contribution and an additional gate-timing contribution in D0. Sync-C gain remains a scorer endpoint pending the independent visual/content guards.
- Current server: `root@connect.westd.seetacloud.com:18625`, root `/root/tts-research-system-20261001`, GPU UUID `GPU-d09622d0-3535-e662-bb45-3df2b049fcea`, boot `bf68e47e-a58c-4393-b5e3-ff12f81889da`. Credentials stay only in private runtime environment/SecureString; never put them in files. Prior server addresses are not the active endpoint.
- Actual sealed seed: remote `cross_tfg_audio_head_20261003/job_control/dev360_phase2_bind_plan_preflight_attempt01/output/runtime_binding_seed_sealed_attempt02.json`, 70,742 B, raw SHA `016429a4dc6c74c8809bd834d07447931cec217332739b6443f9d92f68d476f2`, self SHA `7f0babaab9f8a2214f5362eda8f9c70bfbaeeeffe98621d2f0195d2d3cead716`.
- Actual inventory: remote parent `job_control/dev360_seed_project_root_inventory_attempt03/output/` deliberately contains filename `project_root_file_inventory_attempt02.json`; local original readback `.codex/scratch/dev360_seed_project_root_inventory_attempt03/downloads/project_root_file_inventory_attempt03.json`, raw SHA `d183c19cf1d0d2b40f506620eaed2892e42f128ae1feeb5594eee32b2116a221`, self SHA `497af0f2061dc335c218a1c9084678c1314637e23dd4957f201807ab652b804d`.
- Fragile machine state: private PTY57588/bridge07 port12249; SHM Dev PCM and the empty controls360 leaf. Inspect the PTY prompt before writing; run exact reviewed fresh request indices only once. Ignored scratch packets and their saved responses are recovery evidence, not published code.
- Task99/Experiment98 still describe the prior installed/template phase and need updating after actual recovery. Compact durability03 and postcandidate ASR/visual packets are sealed source-only; terminal result bindings and remote execution are pending. Full recoverable media archive is still required after Dev.

## Verification

Actual phase2 q00-q02 PASS; q03 failed with missing evaluator, saved response SHA `34097dd41e0a88ec33275d86867cca611c6dae196b72193f7b7f952640b7d704`. Actual readonly inventory03 dispatch/download PASS; diagnostic status INCOMPLETE_OR_MISMATCHED means precisely three absent sources, zero mismatches. Root full inventory/raw/self checks PASS. Installer02 source/index/full allowlist and independent review PASS; execution is pending. No generic test suite, GPU Dev experiment, ASR API call or Confirm access was performed in this logical turn.

## Changed Files

The local commit contains the pinned Dev chain scripts, capacity gates, exact source snapshot, narrow `.gitattributes` preservation and 93-line CONTEXT update. `HANDOFF.md` and BM task/experiment snapshots lag the latest inventory. Fresh ignored installer packet: `.codex/scratch/dev360_common_source_restore_attempt02/request_index_attempt02.json`, 6,477 B, raw SHA `ecafc28da04dffea6280bf35948fafd80c59d8743b2881ce48fe450a4151e8c8`. Worker `install_exact_missing_sources_attempt02.py`, 10,611 B, SHA `f1ff44ac2be7b8ccdde446867e7f4d1faaf3ca8d964f463178c49807aef7c814`. Recovery02 is source-only pending installer receipt and root review.

## References

[Repository](https://github.com/dogLeft233/tts-exp), [verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97). Readonly GitHub branch PR search returned no matching PR; current local HEAD is unpublished.

## Observations

- [result] The sealed current-host seed is real and verified; 40/43 required paths match with only three missing and zero content conflicts.
- [decision] Restore only those exact pinned missing sources, retain earlier failures unchanged, and recover into fresh binding/plan/preflight outputs without replaying seal-seed.
- [decision] Keep the candidate, source graph, cohort, controls, full natural-PCM delivery, PCM16 scoring ROI/source-clock policy and Confirm holdout frozen.
- [blocker] Runtime collection/preflight and Dev GPU execution await the actual create-only installer receipt and recovery readback.
- [next_step] Dispatch reviewed installer02, verify its original receipt, then review and execute fresh recovery02 to obtain actual binding03/plan03/preflight03 and launch the 360 controls.

## Relations

- continues [[Codex checkpoint - 2026-10-08T03-50-05Z - Installed Dev chain and recovered CPU staging awaiting runtime collection]]
- relates_to [[开发跨 TFG 的自然音频增强头]]