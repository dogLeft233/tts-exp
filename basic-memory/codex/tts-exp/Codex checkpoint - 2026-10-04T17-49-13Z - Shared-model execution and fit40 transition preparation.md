---
title: Codex checkpoint - 2026-10-04T17-49-13Z - Shared-model execution and fit40
  transition preparation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t17-49-13-z-shared-model-execution-and-fit40-transition-preparation
status: open
repository: dogLeft233/tts-exp
hostname: LAPTOP-E133GSLE
username: dogle
trigger: compact
model: gpt-6.1-sol
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
cwd: E:/Documents/tts-exp
project: tts-exp
capture: deliberate
repo_root: E:/Documents/tts-exp
started: '2026-10-04T17:49:13.2198518+00:00'
branch: codex/cross-tfg-audio-head-20261003
git_sha: 41332dae20743b1d5f2554094c6a0c8d6a7575d2
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T17-49-13Z - Shared-model execution and fit40 transition preparation

## Summary
The shared-head development run is executing once on the GPU, while the next local-transition and native-encoder measurements have been prepared without opening confirmation data.

## Story
The research began by recovering the TTS advantage over natural Mandarin speech from local code, Git and experiment artifacts. The user then asked for a usable, broadly portable natural-audio enhancement head: transform the input for native TFG generation, restore the original natural PCM and clock, and improve SyncNet Cbest across models. Root maintains the theory, protocol, provenance and independent numerical review; GPT-6 Luna/max agents implement experimental code. Qwen ASR/TTS, the server, online data and historical mounted-drive data are authorized. The latest instruction reiterates Qwen ASR for inadequate transcription.

Report 25 already supports the Mandarin acoustic-to-encoder-to-motion mechanism and rejects donor phonetic transplantation as a general head. V4 improved over identity but lost to RMS normalization across all three models (model-equal delta over RMS -0.044299629). RMS remains the practical fallback while stronger heads are developed. The frozen three shared-frequency profiles have fit-only gains over RMS; their 540 native candidate generations are now running on the development cohort, with 360 frozen identity/RMS baselines reserved for the eventual 900-row comparison. No partial progress is a candidate effect estimate.

The latest actual resource03 snapshot at 2026-10-04 17:34:32 UTC shows 49/540 PASS, no failed/incomplete rows, 148 GPU stages admitted and 147 finished. The queue is serialized and requires measured 18 GiB free GPU headroom and known PID ownership at every native child. Do not relaunch it or change its frozen plan. After actual terminal completion, root will bind the sealed producer response, execute the prepared CPU join/visual guards and Qwen content check, and independently review the downloaded numerical bundle before selecting a method. Native-encoder forwards remain queued until the shared GPU run is terminal.

The next mechanism-focused iteration uses the exact canonical 40 fit clips from 20 speakers, with five fixed speaker folds. Root actually recomputed the 320 input diagnostic arrays from decoded source PCM and obtained PASS with maximum discrepancy 1.9073486328125e-6. Positive dB-magnitude flux, independently derived negative flux, and physical 40 ms power are different estimands; V5 relative-linear flux is yet another declared quantity. Agent QC03 corrects a summary unit label in QC02 without changing its correct arrays. A source-only 40-WAV staging package and a signed-transition scaffold V2 were delivered after the last checkpoint but remain awaiting root source review; no V5 transform or candidate selection has run.

## Working State
Durable repository identity is dogLeft233/tts-exp, branch codex/cross-tfg-audio-head-20261003. Current HEAD is local/unpublished `41332dae20743b1d5f2554094c6a0c8d6a7575d2`; the verified pushed commit is c4a199890bd529928ec34b9be6da86a10a737c97. There is no PR for this branch. The local CONTEXT.md modification describes the actual launch; HANDOFF.md is ignored and machine-local. Historical dirty source/tests and all previous attempts are preserved.

Fragile active execution: namespace `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/shared_profile_shortlist_dev_attempt03_execution_attempt01`; output `/dev/shm/cross_tfg_audio_head_20261003/shared_profile_shortlist_dev_attempt03_execution_attempt01/run`; launcher PID 549953/startticks 770442756, runner PID 550013/startticks 770442788, status PID 550016. Plan SHA256 `43e5f45a987153219bc423afbb86512f441ff12db20c3cb56dfb776e14e567df`; rows SHA256 `64a41fb0c76943ac535d38afa9347d9ca0e91ad075e54bb514be262945a63abb`. Last snapshot measured SHM free 5925351424 B, root free 2174087168 B, GPU free 24910 MiB and no unknown GPU processes. Its outer response SHA256 is `f9221dc7560ad028955e671ca09964be46537b6167def2136fcf83ab4b79820b`. Snapshot each existing response before a later poll overwrites the canonical response.

The server is root@connect.weste.seetacloud.com:12349 with verified host pin `SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0`. Credentials remain runtime-only in the private PTY 50274 SecureStrings; bridge06 port 10887/PID252640 has complete EOF drains. Never save secrets, echo them, or place them in argv. The credential-bearing conversation remains the authorized recovery source if the private runtime is lost.

Postrun attempt04 sources are actually staged through packet03 requests 00–02. Worker03 SHA256 `0fe46e3bcd749d4a8d26b8c82f9bb1ad0ea87b3b75dc9cd43de7cbda3f90f53c`; visual guard SHA256 `e154a46aa70e3c434ecc3ae9618e82e072d06fe36126eca19571404f6ca6968f`. No join/visual analysis has run. Terminal discovery04 is root-qualified and unrun, index SHA256 `eef8dc007b218bcb3028f770f57cef43bde7ab7d785e1bc1acb689e7fa8db470`, request SHA256 `b6f15811e1a2c5433a287d52f60187ae48e811036d1ed717704871866f0dbb20`; terminal discovery03 has a preserved duplicate-key emitter error and must not be used. The terminal-to-launch builder01 has only been read through its first 240 lines; its repository parent calculation is under agent review and it must not be executed until full source review and actual terminal evidence.

Native probe168 is actually staged in namespace `native_encoder_probe_attempt04`, request index04 SHA256 `657eff15a7f26e4feda809522abcc4e9fb18b4d6edbcc9f8262fbdac1b540299`. Wav2Lip/MuseTalk/Ditto forward requests 04–06 are unrun. The streaming packer is actually uploaded but packing is unrun. Its durable exact source copies are under `scripts/experiments/cross_tfg_audio_head_20261003/`: native runner02 `4ded47e2e163bbb21734ec26c79f910ecf8ae2c2b44c188cb2eeaf6c7336d34c`, packer01 `1a00218b59e6a6e676f0bff0128c3918f10e48149f17e294edfac4a7655c9754`, verifier01 `7d71a8638a72f9411cc8c7e81a9b509573e970c4dc6eccd1ec485ed7d120bbe6`. Native contract V2 `9329b5fd60a7a8047a89a4bd1ff3855812d211a5e427ce51ecb81984aaa51205` was delivered and is awaiting root full review; it corrects the runtime-note pin and Ditto five-valid-output chunk wording.

Fit40 artifacts live in ignored `runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/`. Source manifest SHA256 `0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020`; 40 source WAVs total 5658606 B. Root actual QC receipt `fit40_root_numeric_qc_attempt01/fit40_independent_input_numeric_qc_receipt.json` SHA256 `3cc09536b92f65a9bc0802206516f8a1233ffca650722fec4c3ec43c1bb3c0ce`, array NPZ SHA256 `544199f39058505711779112779c369b487f0bff62584c8c8afc4ca11ae957ed`. QC02 linear mean-square summary was mislabeled dBFS; arrays are correct. Agent QC03 explicitly separates mean-square/dBFS and retains identical NPZ. Its source and source-stage bundle02 remain unreviewed by root. The r090 source/config are PREPARED_NOT_FROZEN.

V5 scaffold V2 final agent pin is `cf86b95b65b2c4a423082e0f717713983915cdc0674ee7ddb4fba2534e9906e0`; contract V2 `82525f8e7d77c9707a68cba5ad82cef0c6e03ec933dc584c204d275a9d518769`. Both await root full review. The prior checkpoint's V1 b344 digest has not been recovered; observed V1 is a7c2286ac3fdab82a75b09ca5b0809073ea33ff19fc88d703d5ea3f11f677020. This is a preparation provenance discrepancy, with no evaluated transform or scientific result attached to those bytes. Fresh V2 preserves the discrepancy and starts from observed V1. No silent replacement is permitted.

## Changed Files
CONTEXT.md and ignored HANDOFF.md were updated to the actual shared-head launch and staged continuations. The three native runner/packer/verifier exact-byte copies are untracked under scripts. Agent-delivered native contract V2, fit40 materializer/diagnostic/QC03/package sources and V5 scaffold V2 are not yet committed. Selectively commit only root-qualified sources and docs; do not stage historical dirty files or credential-bearing runtime material.

## Verification
Root performed actual bridge dispatch/readback of the shared launch, resource03 and postrun staging03. All examined outer envelopes are complete, untruncated, exit code zero with no stream error. Root fully read and qualified native runner02/packer/verifier and their pins; AST and shell syntax checks passed, without running tests or forwards. Root actually ran independent fit40 PCM/FFT/input numerical QC: all 40 rows and 320 comparisons passed. Source lengths remain unchanged; three center-at-N measurements repeat only a tail sample for a diagnostic query. No unit tests were added or run.

The resource03 index was edited by an agent after dispatch, but its request/actual response bytes were unchanged. Root reconstructed and verified the exact original dispatched index (1378 B, SHA256 `b208620a8c520766d063534371746e0c4715460ccd7122d1e107c1449dc3b367`) at `.codex/scratch/shared540_actual_status_readbacks/resource03_actual_dispatched_index_b208.json`. Current modified metadata has a different digest. Future READY/dispatched indices must remain immutable. Old recovered run/tar retirement and strict C/F restoration are complete with independent post-QC; no need to repeat archive recovery.

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Active task and experiment snapshots: Task47 and Experiment46; `.codex/scratch/update_cross_tfg_snapshots_attempt47.ps1`.
- Full prior runtime/source/evidence recovery pointers are in the verified predecessor checkpoint below.

## Observations
- [result] Shared540 actual status is RUNNING with 49 PASS and no failed/incomplete rows; no development effect conclusion is available until the sealed 900-row comparison.
- [result] Root independently matched 320 diagnostic arrays on the exact 40-clip fit expansion, maximum error 1.9073486328125e-6.
- [decision] Continue stronger heads against RMS, preserve source clock/original playback PCM, model-equal score and worst-model/content/visual guards. Confirmation80/40 unseen speakers and reserved LatentSync stay sealed until final freeze.
- [decision] V5 remains limited to six candidate/ablation slots plus identity/RMS, maximum960 rows. Signed gates use declared common κ and retain physical-power diagnostics; fitted values are not frozen before shared540 readout.
- [decision] All preparation and ready request revisions use fresh paths; retain prior byte provenance, failures and complete cohort accounting.
- [blocker] Shared540 has not reached terminal; postrun analysis and native-encoder forwards are waiting for that actual state rather than being treated as failed hypotheses.
- [next_step] Continue spaced read-only shared540 monitoring while fully reviewing the new V2/fit40 sources, then bind the actual terminal producer and execute the qualified postrun analysis.

## Relations
- continues [[Codex checkpoint - 2026-10-04T17-08-06Z - Running shared-model enhancement and queued encoder measurements]]