---
title: Codex checkpoint - 2026-10-04T15-41-48Z - Qualified frontend measurements and
  continued shared-head execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t15-41-48-z-qualified-frontend-measurements-and-continued-shared-head-execution
username: dogle
branch: codex/cross-tfg-audio-head-20261003
agent: codex
project: tts-exp
capture: deliberate
started: '2026-10-04T15:41:48.1072847+00:00'
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
model: gpt-6.1-sol
trigger: compact
repository: dogLeft233/tts-exp
hostname: LAPTOP-E133GSLE
cwd: E:/Documents/tts-exp
git_sha: 10bbea2f3808c31d0fd36bd30d00f81268ca9221
repo_root: E:/Documents/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T15-41-48Z - Qualified frontend measurements and continued shared-head execution

## Summary
Measured frontend input responses now distinguish Wav2Lip floor-induced spectral changes from Whisper common offsets; old output recovery is reopening capacity for the pending shared enhancement-head generation.

## Story
The original question was why TTS improves Chinese talking-face synchronization. The current user goal is a usable natural-audio enhancement head that transfers across Wav2Lip, MuseTalk and Ditto: drive generation with enhanced audio, restore the original natural PCM and clock, and evaluate primary SyncNet Cbest. The user authorized continued experiments, Qwen ASR/TTS and local or network data. Root maintains theory, the evaluation protocol, independent evidence checks and Basic Memory; experimental and model code belongs to GPT-6 Luna at max effort. Negative candidates lead to fit-only redesign rather than ending the work.

The development protocol still has 20 clips from 10 speakers and three seeds. The 80-clip confirmation cohort remains sealed. Expanded historical fit40 has independently verified source members and PCM identities. V4 improved over identity but lost to fixed RMS on all three frontends; fixed RMS remains a useful calibration baseline. Three bounded shared spectral profiles have fit-only evidence and a frozen 540-row development plan. Their actual generation is still pending sufficient admitted shared-memory space. The staged 900-row join will compare those candidates with the frozen identity and RMS baselines. The latest qualified V4 ASR check is 20 clips, 297 characters and zero errors; the user's latest API-key message reiterates runtime-only Qwen ASR authorization and does not replace that evidence.

The actual CPU fit8 probe ran successfully after measuring the deployed Transformers 4.39.2 runtime and pinning its Whisper source and callable signature. Its first verifier failed because it asked summary rows for an absent source_manifest_row_id field; the unchanged scientific producer emits sample_id. A fresh verifier joined those exact IDs to the frozen manifest and passed all output seals, 384 curves and 96 CSV rows. Wav2Lip's fixed lower mel floor changes local spectral directions under gain; Whisper gain principally shifts logmel common mode. Ditto's measured fixed affine input preserves gain response. These are frontend-input results, not claims of encoder or generated-video invariance. Next native encoder measurements must retain that distinction.

Recovery is concrete: failed-fit output and its duplicate remote tar have both been retired after full C/F archive recovery, strict restoration and independent preservation checks. Fresh post-QC verified all 32 remaining protected paths. Old Ditto retry04 has a completed sealed 1.615GB archive with all 61,562 files and its 239-pass/1-failed history preserved. A four-session private SFTP transfer to C is running; F mirroring and strict restore have been reviewed but not executed. Exact output-run and duplicate-tar retirement must follow recovery and a fresh activity/reference audit. Both releases are needed to reach the unchanged 5GiB generation admission gate.

## Working State
Durable repository state is branch codex/cross-tfg-audio-head-20261003, local unpushed HEAD `10bbea2f3808c31d0fd36bd30d00f81268ca9221`. Existing Task45 and Experiment44 record failed-fit retirement and old Ditto archive completion; this checkpoint adds actual CPU probe and current transfer state. Historical dirty core/test files are untouched. No unit tests were added or run.

Machine-local and fragile: private PTY 50274 retains SecureStrings; bridge06 on port10887. Server endpoint is `root@connect.weste.seetacloud.com:12349`; credentials stay in runtime memory and must never enter files or notes. The active C transfer process is PID350048. Actual C root is `C:/Users/dogle/.codex/tts-exp-artifacts/recoverable-output-archive-20261004-attempt05-ditto-parallel4`; the independently fresh F destination is `F:/tts-exp-archives/recoverable-output-archive-20261004-attempt06-ditto-parallel4`. Mirror07 fixes the previously wrong C06 constant while keeping F06. Last observed transfer completed 1.388GB of 1.615GB. Shared namespace is `/dev/shm/cross_tfg_audio_head_20261003`; system namespace is `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`. Last measured shared free space is 2,819,063,808 bytes with old Ditto run and tar still present.

## Changed Files
- `CONTEXT.md`, `openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md`, and the committed deployed Ditto input-constant inspector anchor current qualified protocol evidence.
- `scripts/experiments/cross_tfg_audio_head_20261003/frontend_gain_feature_probe_attempt03.py` and its preflight/runtime helper are untracked experimental source, actually executed and not yet committed.
- `runs/cross_tfg_audio_head_20261003/frontend_gain_feature_probe_attempt03_readback/`: actual downloaded summary, curves, CSV and receipt; ignored/local evidence.
- `.codex/scratch/cross_tfg_audio_head_20261003/frontend_gain_feature_probe_attempt01/request_index_attempt08.json`: fresh verifier-only repair, executed; prior failed verifier preserved.
- `.codex/scratch/recoverable_output_ditto_transfer_attempt07/transfer_and_restore_index_attempt07.json`: reviewed mirror07 and strict-restore05 pointers; not yet executed.
- `.codex/scratch/ditto_retry04_readonly_audit_attempt02/`: full-reviewed read-only audit executed and returned PASS for every sealed tree member, 239-pass/1-failed bindings and zero active or queued references; external C/F recovery remains separate.
- `.codex/scratch/cross_tfg_audio_head_20261003/shared_profile_content_guard_attempt05/secure_launch/secure_launch_attempt08/`: private ASR template now matches the qualified eight-field bridge completion schema. Static preparation only; no API call.

## Verification
Actual frontend summary is 191,356 bytes, SHA `6af5b7a66a73f8a40138fb3cac2d98adcb33751327cce4bc5c75b8d77eaad888`; curves NPZ is 459,001 bytes, SHA `143daf5014c15ff4f59157744b809592f5dcfbbe4f70b8fef2bcb6c902232665`. Corrected verifier actual exit0, untruncated complete stream and OUTPUT_SEALS_PASS; it retains 8 exact IDs, 384 curves, input-only scope and measured runtime pins. Root read actual rows and receipt and independently checked the fresh request pin/bash syntax. Numerical array recomputation is the next evidence check rather than an already claimed pass.

Failed-fit tar retirement receipt SHA `f06758ea5fac065d4ad45899155b86917fe321d079b862651b090a4285ebfca4`; independent post-QC response SHA `531c5b237ea46c4093d37a7f9d6eb571bab2c17542903057ad3382fe8ff5ca1c` verifies exact target absence and all32 identities. Old Ditto archive SHA `d0b48b51c4a0c0ddf9a173fd14b0a4139069dfe48adac7a3a2726055813ae6ed`; member manifest SHA `15236ac0611c08f1a7f2abd665c0c0bdebac7b8e3b98cc1429109b188f58ee1c`. Root FULL-read the active SFTP client, mirror07, strict-restore05 and read-only audit02, checking actual source paths, byte/hash seals and command pins before action. C/F archive recovery is not yet complete.

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Last verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- `openspec/changes/develop-cross-tfg-audio-head-20261003/`: authoritative data and transfer protocol.
- `runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/v5_candidate_design_v3.md`: fit40 next-slate proposal; coefficients unfrozen and confirmation closed.

## Observations
- [result] Actual fit8 frontend features and corrected output-seal verification passed; the verifier repair did not rerun or change scientific outputs.
- [result] Failed-fit output and duplicate tar are retired with independent protected-path post-QC; historical evidence remains recoverable on C/F.
- [decision] Preserve input, learned encoder, generated motion and original-audio scoring as separate stages of the mechanism chain.
- [decision] Release only exact historical output paths after full local recovery and fresh activity/reference checks; keep plans, code, inputs, model weights and audit controls.
- [decision] Unsuccessful shared development candidates trigger redesign on historical fit; exposed development remains selection evidence and confirmation stays sealed until freeze.
- [blocker] Shared generation awaits old Ditto C/F recovery and exact retirement to reach the existing storage admission gate.
- [next_step] Complete the active Ditto transfer, F mirror and strict restore, qualify the read-only audit and retire its exact output run and duplicate tar, then launch the frozen shared540 generation.

## Relations
- continues [[Codex checkpoint - 2026-10-04T14-54-31Z - Recovered output space and measured frontend continuation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]