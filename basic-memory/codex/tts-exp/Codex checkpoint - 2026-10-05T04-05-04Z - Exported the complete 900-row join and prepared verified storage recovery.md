---
title: Codex checkpoint - 2026-10-05T04-05-04Z - Exported the complete 900-row join
  and prepared verified storage recovery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-05-t04-05-04-z-exported-the-complete-900-row-join-and-prepared-verified-storage-recovery
status: open
cwd: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
model: gpt-6.1-sol
project: tts-exp
started: '2026-10-05T04:05:04.0512852+00:00'
username: dogle
branch: codex/cross-tfg-audio-head-20261003
trigger: compact
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
git_sha: 16adacdf2a12d8e3c40d313c7a2d3c1cc8e57f6b
agent: codex
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-05T04-05-04Z - Exported the complete 900-row join and prepared verified storage recovery

## Summary
The complete 900-row evidence join is now exported, the ASR branch is verified down to all 60 PCM16 payloads, and storage recovery is prepared to unlock the all40 training cache.

## Story
The original project asked why TTS improves talking-face generation. The user's expanded objective is a reusable natural-audio enhancement head: h(x) drives several TFG models, then the exact original natural PCM and clock replace the delivered audio track and remain the evaluation audio. The user explicitly wants continued implementation after failed candidates, delegates experiments and code to GPT-6 Luna/max subagents, and authorizes Qwen ASR/TTS, downloads, and Baidu storage. The root maintains the scientific protocol, independent evidence checks, memory, and private dispatch.

We preserved the frozen development and confirmation separation. Dev20 contains 10 male speakers, three seeds, and Wav2Lip/MuseTalk/Ditto; unseen Confirm80 and LatentSync remain sealed until a candidate freeze. Shared-profile exploration and the separate nine-parameter, 120-update training route have distinct budgets. A positive method must outperform RMS across at least three models, meet the equal-model +0.050 criterion, preserve speech content and visual guards, and include all failed planned units. We have not selected a method or opened confirmation data.

The previous join failure was a frame-map schema incompatibility, not a failed acoustic candidate. A read-only full-asset census verified all 900 rows: 360 baselines and 540 candidates, with N logical crops, N+2 decoded frame-map entries, and two nominal decoder tails. The minimal join07 repair then actually ran to completion: 900 PASS rows, 60 contexts, 2,774 assets, FULL support eligibility 900, and guard20 eligibility 855. It exported a 47,938,337-byte sealed archive with ten bounded readback ranges. Root verified the complete census independently; the joint archive has not yet been downloaded or its effect numerics independently recomputed, so no gain claim follows from completion alone.

The content branch now has actual local copies of all 60 ASR PCM16 WAVs, not only text receipts. Every transfer field, container hash, RIFF format, sample count, and PCM payload hash passed; independent text QC found zero CER and zero disagreement against the natural-ASR transcripts. No extra cloud calls are needed at this point. The native encoder probe remains an engineering repair: Wav2Lip CPU preflight passed, while MuseTalk reported the absent/nonregular frozen face-parser path and Ditto reported an environment.yaml raw-SHA mismatch. Exact lstat, resolution, and byte diagnostics are prepared before any repair. No 168 native forwards have run.

The all40 producer source and frozen inputs are already staged, but its 4 GiB shared-memory admission gate is still unmet. The read-only storage census found exact local-backed SyncNet output and 6,000 stageb leaves, plus duplicate portable archive parts; it also found additional unbacked files that must be preserved. V4 is terminal with 180 PASS results but needs a complete raw archive before retirement. Separate packets are ready for read-only duplicate-part verification and precise SyncNet retirement; a memfd archive workflow is being prepared for V4 to avoid another shared-memory archive copy. No retirement has executed at this checkpoint. Once the fresh space gate passes, the all40 producer must establish same-GPU zero-head/RMS parity before the fixed 120 training updates.

## Working State
The branch remains local and unpublished at `16adacdf2a12d8e3c40d313c7a2d3c1cc8e57f6b`; no PR exists. `CONTEXT.md` is tracked and modified, while the current 48-line `HANDOFF.md`, scratch packets, downloaded evidence, and active private bridge are machine-local. Task61 and Experiment60 were written and read back but predate the terminal join07 and native06 results; refresh their current state after archive QC. Historical dirty source files remain untouched. Credentials are runtime-only and must never enter notes, requests, or source.

The private PowerShell dispatch session is 50274 and bridge06 uses port 10887 with private environment authentication. If that process disappears, re-establish credential handling privately rather than copying a token into a file. The SSH endpoint is root@connect.weste.seetacloud.com:12349 with the pinned host fingerprint documented in the project reference; passwords and API values are intentionally omitted.

## Changed Files
- `runs/cross_tfg_audio_head_20261003/shared_profile_asr05_actual_readback_attempt01/qc_attempt10/pcm16_full_readback_verification_attempt04.json`: complete 60-WAV verification; SHA `dfa9390fe60911447df81f354656445e266281a454d5f1b419899d616a5c4577`.
- `runs/cross_tfg_audio_head_20261003/shared_profile_900row_schema_census_attempt03_actual_readback_attempt01/census.json`: actual full census, 323,604 bytes; SHA `a5f1d8aa03cd2df4db8307f306c44d99dca2f442b7e1241d9dbd2264e93a7372`.
- `.codex/scratch/cross_tfg_audio_head_20261003/shared_profile_900row_join_attempt07/`: minimal repaired producer and actual launch/status receipts; terminal response SHA `b271a65848a5f2a7b6c255efeb4d8f2e1984147ec7febad4f6f59f67f7df3f50`.
- `runs/cross_tfg_audio_head_20261003/shared_profile_900row_join_attempt07_readback_attempt01/`: manifest-first bounded-range readback packet; archive SHA `ad01790e24138a1d0c062f23c753b26856d66448f3dc0f715f5e86b4615c68d0`.
- `.codex/scratch/cross_tfg_audio_head_20261003/storage_recovery_cleanup_attempt01/`: prepared duplicate-parts verification and exact backed-file retirement packets. These are source-only until private dispatch and independent receipt checks.
- `CONTEXT.md` and ignored `HANDOFF.md`: refreshed project entry points, currently needing the newest completed join state.

## Verification
Root actually verified all 60 downloaded ASR WAVs and all 900 census rows, including self-seal, uniqueness, frame-map cardinality, model/method counts, and 31 lag entries. Join07 actually exited zero with no stderr and exported the sealed complete archive; its numerical metrics are still awaiting local full readback and recomputation. Native06 actually passed Wav2Lip CPU preflight and failed MuseTalk and Ditto for the exact dependency/source reasons above. Storage audit actually hashed current files and checked terminal status and process references; it did not delete data. No pytest or unit tests ran. No training optimizer steps, trained weights, ablation, unseen confirmation, or report26 exist yet.

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Last verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current local, unpushed commit: `16adacdf2a12d8e3c40d313c7a2d3c1cc8e57f6b`.

## Observations
- [result] Full raw-asset census and join07 completed with 900 PASS rows; content checks include all 60 actual PCM16 WAV payloads.
- [decision] Preserve failed execution receipts and repair only dependency/schema behavior; keep frozen scientific endpoints, split, budgets, and original evaluation PCM unchanged.
- [decision] Retire only exact local-restored, freshly reverified files; preserve stageb's 2,400 extra files and joint-consumer extras until their own backup closure.
- [blocker] The all40 producer is waiting for its actual 4 GiB shared-memory gate, and MuseTalk/Ditto native CPU preflight need source/asset repair.
- [next_step] Download and independently recompute the complete join07 archive while completing verified storage recovery to admit the all40 producer.

## Relations
- continues [[Codex checkpoint - 2026-10-05T03-28-20Z - Verified complete ASR and staged all40 production while preserving frame-map failure]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]