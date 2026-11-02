---
title: Codex checkpoint - 2026-10-07T15-06-57Z - Verified migrated context and launched
  forty sample cache reconstruction
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t15-06-57-z-verified-migrated-context-and-launched-forty-sample-cache-reconstruction
git_sha: 957a9c80ec460e44278463a6111411d0fc8b5446
model: gpt-6.1-sol
repo_root: E:/Documents/tts-exp
trigger: compact
agent: codex
project: tts-exp
cwd: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
username: dogle
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
started: '2026-10-07T15:06:57.6003196+00:00'
branch: codex/cross-tfg-audio-head-20261003
capture: deliberate
hostname: LAPTOP-E133GSLE
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T15-06-57Z - Verified migrated context and launched forty sample cache reconstruction

## Summary
The migrated GPU now has an independently verified deterministic audio context and an actual running forty-sample natural/RMS cache producer, enabling the frozen enhancement-head training route to resume.

## Story
The original work asked why Chinese TTS improves TFG scores; the user's current goal is a useful natural-audio enhancement head across models, with original natural PCM restored for final mux and evaluation. Root maintains the theory, frozen protocol, independent evidence and Basic Memory; GPT-6 Luna/max agents implement and run prepared experiments through root's private SSH dispatch. Simple shared acoustic profiles already fell below the required extra gain over RMS, so the current route learns a bounded transition head on Fit40 and evaluates three models before opening unseen confirmation data.

After the server migration, old shared-memory assets disappeared. The retained source and input evidence allowed a new deterministic GPU context instead of silently mixing scores from two machines. This turn actually downloaded the 15,622,636-byte context through three bounded chunks, reassembled its exact SHA, and independently checked all five arrays, WAV/PCM bytes, crop geometry, static face and frame clock. Producer05 was then staged and preflighted against all forty inputs and launched on the qualified RTX 4080 SUPER. The last actual status contains 29/40 passing rows and remains RUNNING; this checkpoint does not assert a forty-row terminal result or any optimizer update.

Training09 revision04 is sealed but still pending root source review and actual CPU closure preflight. Its last request only binds the real completed producer outputs and does not start optimization. Ditto restoration attempt02 actually failed on an official Hugging Face connection timeout/refusal before accepting any weight. A new fixed-revision mirror03 packet is prepared, unexecuted, and preserves all twelve original byte/SHA pins. MuseTalk qualified source assets and original alias workspace also require restoration. New 360 control and 180 candidate execution sources are being completed with independently decodable raw-media companions; score JSON alone cannot satisfy the frozen visual and original-PCM guards.

## Working State
Durable local commit `957a9c80ec460e44278463a6111411d0fc8b5446` is unpublished on `codex/cross-tfg-audio-head-20261003`. It contains CONTEXT.md and the context04/producer05/runner06 sources. Task attempt82 and experiment attempt81 were written and directly read back; their phase is new_context_independent_pass_fit40_producer_running. ADR015 requires fresh same-host controls and retains historical 540 evidence only for provenance.

Machine-local server access is root@connect.westd.seetacloud.com:18625, fingerprint SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0. Secrets remain private runtime state. Private bridge PTY57588/port12249/PID10892 must be checked before dispatch. Actual producer PID16438/start tick323961118 writes under /dev/shm/cross_tfg_audio_head_20261003/fit40_native_rms_geometry_attempt05/run; its persistent control is under /root/tts-research-system-20261001/cross_tfg_audio_head_20261003/job_control/fit40_native_rms_geometry_attempt05. Shared memory remains fragile and terminal artifacts need immediate bounded backup. Local scratch responses, ignored HANDOFF.md and unrelated older dirty files are not durable scientific evidence by themselves.

## Changed Files
The committed files are CONTEXT.md, scripts/experiments/cross_tfg_audio_head_20261003/export_engineering_context_attempt04.py, build_fit40_natural_cache_rms_geometry_attempt05.py and run_fit40_natural_cache_rms_geometry_attempt06.py. Consult git show for exact paths. New orchestration and independent receipts remain in .codex/scratch/ and runs/cross_tfg_audio_head_20261003/.

## Verification
Root actually executed root_context04_independent_readback_attempt01.py: PASS_INDEPENDENT_CONTEXT_BYTES_ARRAYS_PCM_AND_CLOCK, receipt SHA e0912245c72dd7aba2c9aa7c638039db62c2c7add05a46548571e8c23508a216. Full context NPZ SHA is 6678dce21f71693b2c3da7cc2eaf4f405d0983b293ff95165b4e43257dfbbc2a. Independent backend presence verified 46 entries: 24 exact existing files, 22 absent assets and 436,317,664 hashed bytes. The historical archive read-only inventory passed; old absent visual/chain files stayed recorded as absent.

All twenty-two producer staging requests actually passed, with 36 ZIP members and eight reassembled historical evidence files checked. Actual producer preflight checked source pins, forty input identities, GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea, strict QC06 runtime and two empty GPU observations. Latest actual status response SHA ee56b29b159bff2b79847a4bfe444a4fbbe8228d500eff125c248a9bc1198937a records 29/40 rows and 311.40 seconds. Ditto02 failure response SHA 5422e033cc4cfe7072357480b7cda269733b35b0e18fd870b5b181c29d9621149 records curl exit7, supervisor dead and no accepted final asset. No generic tests were run; independent scientific QC and source/packet checks are the actual verification here.

## References
[Repository](https://github.com/dogLeft233/tts-exp) and [last verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97) were resolved by fresh read-only GitHub queries. Current HEAD is local/unpublished. The current branch pull-request query matched no PR.

## Observations
- [result] Migrated deterministic context bytes, arrays, original PCM mapping and clock independently passed; this qualifies engineering context, not enhancement efficacy.
- [decision] Preserve fixed Fit40/20 speakers, Adam120, Dev20/10 speakers across three seeds and Wav2Lip/MuseTalk/Ditto; acceptance remains three positive models and equal-model additional +0.050 over RMS with frozen guards. Confirm80 and LatentSync remain sealed.
- [decision] Replace lost assets using identical fixed-version byte/SHA pins, and generate fresh same-host controls; retain old results and failed engineering attempts with their original meaning.
- [blocker] Producer terminal40, zero-head40, optimizer120 and terminal120 independent rescore have not executed yet; the Ditto/MuseTalk missing assets still prevent complete three-model execution.
- [next_step] Read fresh producer05 status, independently qualify and back up its actual forty-row terminal outputs, then finish training09 source/CPU closure review and bind the real outputs for zero-head40 followed by fixed120 training while restoring the two other backends.

## Relations
- continues [[Codex checkpoint - 2026-10-07T14-15-13Z - Exported same GPU context and restoring full method execution]]