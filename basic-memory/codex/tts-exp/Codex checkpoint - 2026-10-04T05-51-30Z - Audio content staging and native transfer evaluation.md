---
title: Codex checkpoint - 2026-10-04T05-51-30Z - Audio content staging and native
  transfer evaluation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t05-51-30-z-audio-content-staging-and-native-transfer-evaluation
branch: codex/cross-tfg-audio-head-20261003
started: '2026-10-04T05:51:30.1002138+00:00'
hostname: LAPTOP-E133GSLE
repository: dogLeft233/tts-exp
username: dogle
cwd: E:/Documents/tts-exp
status: open
project: tts-exp
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
repo_root: E:/Documents/tts-exp
capture: deliberate
agent: codex
trigger: compact
git_sha: 1679f680278a291d5e2a9091d2a2ceac2a41a70c
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T05-51-30Z - Audio content staging and native transfer evaluation

## Summary
The frozen native spectral head is undergoing three-model development validation; complete shared-profile fit matrices and two native LatentSync arms are ready for independent evaluation.

## Story
The project began by asking why TTS improves talking-face generation. The user now wants a portable, usable enhancement head: transform natural audio only for generation, then restore the original waveform and evaluate against that original source clock. Experiments and implementation remain delegated to GPT-6 Luna with max reasoning; the root maintains the scientific protocol, artifact review, Basic Memory and independent numerical checks. Negative candidates remain part of the record and drive the next iteration.

The current strategy combines a learned frequency-and-time mask trained against the native SyncNet score field with a simpler shared frequency-profile search. This separates benefits beyond fixed RMS from model-specific conditioning artifacts. The learned v4 checkpoint completed 96 training steps and exact production CLI parity; its frozen 180-row development run has actually produced 20 PASS rows across Wav2Lip, MuseTalk and Ditto, with no failed rows at the last read. These are engineering results, not a development efficacy conclusion. The shared-profile fit producer has all 360 rows and its repaired consumer has independently verified 2,530 source references, 24 groups and zero issues. Its terminal NO_EVALUATION_ENDPOINTS is expected for the fit-only plan; the consumer exit code was not captured and must not be inferred.

Ditto's complete raw-matrix readback and root reconstruction show that v3 does not improve over RMS; this motivates v4 rather than ending the research. Native LatentSync identity and RMS generation both now pass. The RMS video was downloaded and locally decoded: all 62,639 original audio samples and the video elementary streams match their frozen identities. A matched 13-frame contact sheet was actually viewed; both arms have changing mouths. Paired SyncNet scoring has not yet run. SadTalker remains at a narrow scalar/array compatibility failure, with failed attempts preserved.

ADR 010 records audio-only ASR checking before video completion. CPU staging must execute the production transform exactly; each candidate receives its own Qwen ASR call, with fsync journaling and all 20 clips retained. Natural ASR is reusable only after exact F32 and PCM16 input identities match. Final video rows must join those staged PCM hashes before these checks support the complete method. The latest CPU staging packet03, shared exporter/selector packet02 and LatentSync scorer03 are prepared, not dispatched. The prior shared packet had a 63-digit ZIP hash and CRLF shell worker; both were caught before execution and corrected in a fresh packet.

## Working State
Durable protocol: openspec/changes/develop-cross-tfg-audio-head-20261003/spec.md and transfer-iteration.md; development uses 20 clips from 10 male speakers, three paired seeds and fixed common support. Confirmation remains sealed. At least three positive model families, useful improvement beyond RMS, independent audio/mouth guards, heldout transfer, CLI/config/weights/latency and report 26 remain required.

Verified local Git HEAD is 1679f680278a291d5e2a9091d2a2ceac2a41a70c on codex/cross-tfg-audio-head-20261003, unpublished with no PR. Existing dirty/untracked historical work is preserved. Machine-local, ignored transport packets, remote processes and /dev/shm outputs are fragile; never confuse a prepared packet with an executed receipt. Remote access is root@connect.weste.seetacloud.com:12349 with the pinned public host key; passwords and API keys remain runtime-only and are absent from memory and files. The private PTY 50274 owns the bridge and SecureStrings.

Current evidence pointers: .codex/scratch/v4_fit_root_readback_qc_attempt01.json and v4_fit_root_numeric_qc_attempt01.json; runs/cross_tfg_audio_head_20261003/ditto_recovery_attempt05_raw_matrix_readback01/root_independent_numeric_qc_attempt01.json; .codex/scratch/latentsync_fit_rms_readback_attempt01/root_local_readback_qc_attempt01.json; .codex/scratch/latentsync_fit_identity_rms_visual_readback_attempt01/root_visual_readback_receipt_attempt01.json. Task revision 34 and experiment revision 33 are stale relative to this checkpoint and need reconciliation after the next actual selection milestone.

## Changed Files
- scripts/audio_enhancement_eval/stage_v4_dev_audio_only_attempt02.py implements exact CPU production staging with separate source identity validation and peak-mask diagnostics.
- scripts/audio_enhancement_eval/run_v4_audio_only_asr_guard_v2.py adds per-response fsync journaling and atomic finalization; it has not called the API.
- .codex/scratch/root_shared_fit_numeric_qc_attempt01.py is prepared root-independent QC of the future archive and selector report; it has not run.
- Fresh transport packets: .codex/scratch/cross_tfg_audio_head_20261003/v4_audio_stage_attempt01/request_index_attempt03.json; shared_frequency_fit_raw_archive_attempt04/packet_attempt02/request_index_attempt02.json; .codex/scratch/latentsync_fit_pair_scoring_attempt03/packet_attempt01/packet_index_attempt01.json.

## Verification
Actual root numerical QC passed for the 96-step fit and 241 Ditto matrices/1,687 arrays, including query clocks, support masks, paired groups and endpoints. LatentSync RMS readback hashes, exact natural PCM decoding and three matching video elementary streams passed locally. Producer/consumer terminal shared-fit receipts were read directly. Root reviewed complete CPU staging/ASR source and prior shared exporter, adapter and selector source; fresh corrections still need binding review before dispatch. No pytest/unittest suite was added or run.

## References
- [Repository](https://github.com/dogLeft233/tts-exp).
- [Verified pushed reference commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97).
- Current local unpublished HEAD: `1679f680278a291d5e2a9091d2a2ceac2a41a70c`.

## Observations
- [result] Shared fit has 360 PASS rows and zero consumer issues; fit-only terminal status does not supply evaluation endpoints or a captured exit code.
- [result] Native LatentSync identity and RMS inference pass exact original-audio and local video integrity checks; their paired score comparison remains unexecuted.
- [decision] Preserve all failed attempts, frozen source clocks, natural-audio scoring and fixed short-clip guard denominators; do not select candidates from development or confirmation.
- [decision] Candidate ASR may run on exact audio-only staging before full video completion, with later PCM identity joins required by ADR 010.
- [blocker] The main development run is incomplete; shared raw readback/selector QC, candidate ASR and LatentSync paired scoring are pending. Shared-memory capacity is low, so the next 540-row run needs verified output placement.
- [next_step] Review and dispatch CPU staging packet03, verify its actual 20-clip manifest, then launch independently journaled candidate Qwen ASR calls while development rendering continues.

## Relations
- continues [[Codex checkpoint - 2026-10-04T05-12-47Z - Running native v4 validation and verified Ditto matrices]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]