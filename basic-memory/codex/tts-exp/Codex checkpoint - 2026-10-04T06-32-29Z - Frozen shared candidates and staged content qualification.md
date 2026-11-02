---
title: Codex checkpoint - 2026-10-04T06-32-29Z - Frozen shared candidates and staged
  content qualification
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t06-32-29-z-frozen-shared-candidates-and-staged-content-qualification
hostname: LAPTOP-E133GSLE
started: '2026-10-04T06:32:29.2014213+00:00'
model: gpt-6.1-sol
repo_root: E:/Documents/tts-exp
project: tts-exp
cwd: E:/Documents/tts-exp
trigger: compact
git_sha: 1679f680278a291d5e2a9091d2a2ceac2a41a70c
username: dogle
branch: codex/cross-tfg-audio-head-20261003
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
status: open
agent: codex
repository: dogLeft233/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T06-32-29Z - Frozen shared candidates and staged content qualification

## Summary

Shared acoustic fitting now has independently verified raw arrays and three frozen candidates; the learned v4 candidate has verified audio staging and an ongoing three-model dev run, while Qwen content checks and two further model qualifications are being brought into execution.

## Story

The original question was why self-cloned TTS can improve Chinese talking-face generation. The user has extended that into a practical method: enhance natural audio only as the generator input, restore untouched natural PCM on the original source clock, and obtain a useful improvement across several TFG families. The root owns theory, protocol, Basic Memory, transport, and independent artifact review; experiment code and execution are assigned to gpt-6-luna/max subagents. The latest user explicitly permits Qwen ASR when transcription quality is poor. Credentials remain runtime-only and must never be copied into notes or source files.

The authoritative mechanism report 25 remains CONCLUDED. Its recipient-compatibility result motivates recipient-local acoustic contrast rather than transplanting donor TTS spectra. Current shared DSP candidates preserve phase, sample count, source clock, fixed RMS, and bounded local masks. Twelve profiles were fitted on eight historical clips from four speakers. A fixed utility combining native best-lag confidence, anchored confidence, distance, and far-lag contrast ranked three profiles without consulting dev outcomes. All twelve fit outcomes are retained, including failures. This is now a qualified fitting result, not a generalization claim.

The parallel learned v4 head uses native full-context Wav2Lip prediction/paste and a best-confidence-plus-far-contrast reward with distance/anchor penalties. Its frozen checkpoint is under dev validation on 20 new clips from ten speakers and three paired seeds across Wav2Lip, MuseTalk, and Ditto. CPU-only audio staging gives independent content checks a head start while videos finish. The staged waveform must later join exactly to every generated-video ledger PCM; staging alone is not completed video evidence.

## Working State

Durable protocol is in `openspec/changes/develop-cross-tfg-audio-head-20261003/`. Basic Memory task `开发跨 TFG 的自然音频增强头` remains active and the experiment `跨 TFG 自然音频增强头的训练与原音换轨验证` remains implemented. Their latest stored snapshots still precede these new artifacts and need reconciliation. No report 26 conclusion or confirmation result exists. Forty male speakers/eighty confirmation clips remain sealed; confirmation seeds/family must be frozen before access.

The shared fit archive is `runs/cross_tfg_audio_head_20261003/shared_frequency_fit_raw_readback_attempt01/fit_raw_matrices.tar.gz` (5,275,898 bytes; SHA256 `bf35f9377db4a2ce222369c0bc976dba8cc40a4c46e3d9b16e3ddc3326bb6223`). Its selector SHA is `53053394155deff7215b7ecdeb6bcd5f7196250c5feab8f8a2d0849345fb243b`; root numeric QC receipt SHA is `42938a487c2c9442825727d018bb4b3957c9cc4501fbf096642b7b524b9de0ec`. Shortlist: `local_r090_accent015_tilt_p025`, `local_r090_flatten015_tilt_m025`, `local_r090_accent_0.30_tilt0`. The first profile's descriptive fit ΔCbest versus RMS is +0.010/+0.051/+0.096 for Wav2Lip/MuseTalk/Ditto, equal-model +0.052; this uses only eight fitting clips, and its Wav2Lip D0 slightly regresses. These numbers cannot be promoted to dev efficacy.

The subagent actually built a metadata-only 540-row plan at `runs/cross_tfg_audio_head_20261003/shared_profile_shortlist_dev_attempt03/plan.json`, raw SHA `43e5f45a987153219bc423afbb86512f441ff12db20c3cb56dfb776e14e567df`. The root reviewed the full builder02 and exact two-line builder03 diff (correct control name `rms_localcompress`, fresh run ID). Root independent plan metadata QC is next; no audio/model inference launched, baseline joins are pending qualification.

Machine-local/fragile v4 output is `/dev/shm/cross_tfg_audio_head_20261003/stageb_v4_dev_attempt02/run`; latest actual readback is 79/180 PASS, 27 Wav2Lip/26 MuseTalk/26 Ditto, zero failed, empty stderr, stable launcher/worker PID ticks. This is an observed counter, not an assumed current count. Source closure18 and dev plan seals are qualified. Frozen v4 checkpoint SHA is `3b200be269089a1ca8e7afe7d3d5481ad9c45b45c18eb4c94e5d2e4dab7941f4`.

Actual CPU staging completed exit0 for 20 candidates and all40 natural/candidate WAVs were remotely decoded and hash-checked. Local manifest is `runs/cross_tfg_audio_head_20261003/v4_audio_stage_attempt01_readback/audio_only_candidate_manifest.json`, raw SHA `85be3063e9d324e6d2b519a99ec37eb781f0ba2314db47f0e014a4d7e83b9cf8`. Root staged-manifest QC validates all180 planned contexts, exact transform replay, no active peak caps, and maximum fixed-RMS error 3.24451530919756e-9.

Qwen ASR scientific worker v2, `scripts/audio_enhancement_eval/run_v4_audio_only_asr_guard_v2.py` (SHA `c8df1680300bf3a2096416720befb197fd18e4a8889c8a139fcab0ae8eeb3ac4`), and namespace were actually uploaded. Actual preflight04 PASS binds all20 staged candidates, gold transcriptions, natural-ASR cache exact PCM, and FFmpeg. No API calls yet. Root caught a stale transcript expected digest before wrapper01 upload, then a remaining extra final character in wrapper02 before upload; both launch attempts remain unexecuted. Agent is making fresh wrapper03/launcher03/private client04 from complete parsed tokens, with a required64-character assertion. Scientific worker and study conditions do not change. Private client host-pin/stdin/environment-only credential design was fully reviewed.

LatentSync native identity03 and RMS05 videos are already actual PASS with original PCM and identical elementary video streams across mux variants; root viewed paired 13-frame contact sheet. Pair scorer04 now has actual source/import/checkpoint preflight05 PASS. Actual launch05 stopped with exit75 because a peer GPU compute PID was active, before any scorer worker. A CPU-only60-minute queue wrapper is prepared for review and must retain two idle snapshots five seconds apart and >=18,432MiB before dispatch. SadTalker05 remains at a NumPy scalar-array construction error; a narrow t[0]/t[1] scalarization repair is pending, not a scientific failure.

Targeted storage inventory06 actually completed exit0 for precisely three namespaces. SHM project outputs occupy17.756GB; active v4 occupies724MB. Root instructed recovery agent to archive at least5GB of sealed finished Ditto/shared-fit/MuseTalk outputs, preserving all authoritative controls. Local B: is mounted CloudFS with1.26TB free. Archive plus local/B readback must qualify before scoped deletion. Active inputs, weights, v4, LatentSync and SadTalker are excluded. No new deletion or archive occurred in this interval.

## Changed Files

Experiment code remains dirty/untracked and machine-local until a scoped commit. Relevant new files include audio staging attempt02, ASR worker v2 and key-free launch wrappers, shared shortlist builders01–03, and LatentSync pair queue sources. Root created artifact-only numeric/staging QC scripts under `.codex/scratch/`; those are local analysis receipts, not model implementation. Ignored `HANDOFF.md` was updated but has an older v4 counter and is not an authoritative run ledger. Historical unrelated dirty files are preserved. Current `CONTEXT.md` is93 lines and also needs reconciliation after the execution milestone.

## Verification

Shared export/selection worker actually completed exit0 with empty stderr. Three archive/selector/chunk-manifest downloads passed raw hashes. Root independently recomputed360 NPZ/2520 arrays over24 paired contexts, all source clocks/common support/F32 curves and selector utility; max curve error5.7220458984375e-6, max metric error6.67572021484375e-6, utility discrepancy0. Audio staging actual exit0 and root manifest QC PASS are described above. LatentSync scorer preflight05 PASS is separate from its GPU-blocked launch. Qwen preflight PASS is separate from zero API calls. No pytest/unittest were added or run.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current HEAD `1679f680278a291d5e2a9091d2a2ceac2a41a70c` is local/unpublished; no PR exists for this branch.
- Formal mechanism report: `tts-exp/docs/experiments/25-tts-tfg-mechanism`.
- Root QC: `runs/cross_tfg_audio_head_20261003/shared_frequency_fit_raw_readback_attempt01/root_independent_numeric_qc_attempt01.json` and `v4_audio_stage_attempt01_readback/root_staged_manifest_qc_attempt01.json`.
- Storage receipt: `.codex/scratch/latentsync_space_inventory_attempt04/03_targeted_onelevel_usage_attempt06_request.json.response.json`.

## Observations

- [result] Shared fit360 arrays and fit-only shortlist are independently qualified; dev540 metadata is built without inference or confirmation access.
- [decision] Keep original natural PCM/source clock primary, separate fit/dev/confirmation, retain failed attempts and short-clip GUARD incompleteness, and require exact staged-to-rendered PCM joins before final content evidence.
- [blocker] Available storage needs recoverable archival before540-row rendering; ASR launch wrappers need exact complete-token correction; LatentSync scoring awaits safe GPU admission and SadTalker needs a narrow scalar repair.
- [next_step] Launch the independently checked20-candidate Qwen ASR guard with corrected key-free wrappers and private stdin credentials, then qualify the540-row plan and storage for dev execution while v4 completes.

## Relations

- continues [[Codex checkpoint - 2026-10-04T05-51-30Z - Audio content staging and native transfer evaluation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[009-shared-profile-fit-and-lag-contrast-selection]]
- relates_to [[010-audio-content-check-before-render-completion]]