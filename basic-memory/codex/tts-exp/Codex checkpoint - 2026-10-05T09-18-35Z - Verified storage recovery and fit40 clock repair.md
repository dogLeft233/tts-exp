---
title: Codex checkpoint - 2026-10-05T09-18-35Z - Verified storage recovery and fit40
  clock repair
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-05-t09-18-35-z-verified-storage-recovery-and-fit40-clock-repair
repo_root: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
project: tts-exp
trigger: compact
started: '2026-10-05T09:18:35.9665651+00:00'
capture: deliberate
username: dogle
agent: codex
repository: dogLeft233/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
model: gpt-6.1-sol
status: open
git_sha: 16270c228a1dcd3419e67186d28a0335bf9c8dc3
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-05T09-18-35Z - Verified storage recovery and fit40 clock repair

## Summary

The natural-audio enhancement project now has verified storage recovery and encoder evidence, but the first fit40 production row stopped at an overly strict video-frame mapping check; training and confirmation have not run.

## Story

The original question was why cloned TTS improves Chinese TFG synchrony. The user expanded it into a practical, reusable enhancement head: generate video with transformed natural audio, restore the exact original PCM and timing, and improve Sync-C across several TFG models. Root maintains theory, frozen evaluation, private dispatch and independent QC; GPT-6 Luna/max agents implement and execute. The user also authorized Qwen ASR/TTS and downloads. The existing 60-item ASR audit has zero CER/disagreement, so the repeated ASR authorization does not itself warrant re-transcription.

The retained RMS baseline improves equal-model Dev Sync-C by 0.154 across Wav2Lip, MuseTalk and Ditto. Three shared local spectral profiles failed the stronger multi-model criterion. Encoder audits explain why a uniform spectral recipe is difficult: MuseTalk responds much more to the local profiles than to global gain, while Wav2Lip shows the reverse pattern. These are measured encoder responses on eight fit clips, not a paired causal video result. The next method is the frozen nine-parameter static/rising/falling band head, trained on an independent 40-row fit set for exactly 120 Adam updates and evaluated with original natural score audio.

Storage recovery is now closed with root-verified backups, restore inventories and retirement receipts. V4 retired 45,814 files (1,347,004,407 logical bytes); native Wav2Lip/MuseTalk retired 116 files (58,899,632 bytes). Scientific receipts and protected directories remain. The fresh producer02 preflight passed source/GPU/storage checks, and the producer was launched. It stopped after 20.322 seconds with exit 2 on S0052_j00_BAC009S0052W0121 because decoded AVI and JPEG counts differed. It completed zero of 40 rows; 39 rows were not run. This is an engineering preparation failure, and no head weights or enhancement result exist yet. Source audit shows the downstream mapper already handles monotone duplicate JPEG observations, but the actual first-row clocks/counts must be inspected before changing the guard.

## Working State

Durable repository anchor: branch `codex/cross-tfg-audio-head-20261003`, local/unpushed HEAD `16270c228a1dcd3419e67186d28a0335bf9c8dc3`. This commit changes only CONTEXT.md to document encoder evidence, completed storage retirement and the pending fit40 clock repair. The project task snapshot68 and experiment snapshot67 were overwritten and read back: [[开发跨 TFG 的自然音频增强头]] and [[跨 TFG 自然音频增强头的训练与原音换轨验证]]. Their typed Git field predates the new entry-only commit; their scientific state remains accurate. HANDOFF.md is an ignored machine-local copy of snapshot68. Historical five dirty source/test files and many untracked implementation files are preserved; no broad staging was performed.

Frozen science: Dev20/10 male speakers × three seeds × Wav2Lip/MuseTalk/Ditto; Confirm80/40 unseen speakers plus LatentSync stays sealed until method freeze. Require at least three positive models and equal-model gain above 0.050 over RMS, with D0/Canchor/content/visual guards and the complete failure denominator. GUARD20 remains null. No report26, trained checkpoint or unseen confirmation exists.

Machine-local execution: private PTY50274 idle, bridge06 port10887; server `root@connect.weste.seetacloud.com:12349`, pinned host fingerprint `SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0`. Credentials remain only in private runtime/environment, never notes/files. Remote namespace `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`; shared-memory namespace `/dev/shm/cross_tfg_audio_head_20261003`. Producer02/controller are terminal; there is no active GPU experiment. The failed namespace is preserved with 114 partial files (~6.17 MB), not reusable as successful fit geometry.

Root actual evidence pointers:
- `.codex/scratch/fit40_native_rms_geometry_packet_attempt02/failure_readback_attempt01/`: full producer execution receipt, execution state, worker state and stderr downloaded with exact SHA verification. Receipt 9,204 B SHA `cd168bdfd0f5bc58d097419576b126b8f6ccc56b776f32d331da96cb392b6746`.
- `.codex/scratch/fit40_native_rms_geometry_packet_attempt02/compact_status_attempt02/`: corrected reader02 verifies all four staged source pins, terminal 0/40 and no GPU process. Actual stage digest is `9c908bf1ccdaac82a9e2ee5d5ef81957155f5a657b3a93895e4de33758122831` (some older narrative pointers had a typo).
- `.codex/scratch/cross_tfg_audio_head_20261003/v4_output_retirement_launch_attempt09/root_all_scan_semantic_qc_attempt10.json`: authoritative retirement QC; three complete scans cover four unique configured roots with zero active/unresolved references. Prior QC09 field guesses are preserved but superseded.
- `.codex/scratch/cross_tfg_audio_head_20261003/v4_output_retirement_attempt06_postreadback_attempt03/retirement_receipt_attempt06.remote.json`: actual 136,557 B receipt, SHA `354df7e37193ee86268958fa69e6f9ceeef968f336f992e9bc6c1c7154ec778a`; target absent, protected dirs intact. C:/v4-a03 contains the qualified whole archive and strict restored tree.
- `runs/cross_tfg_audio_head_20261003/native_output_retirement_attempt03/readback_attempt01/root_independent_readback_qc_attempt01.json`: exact 116-path byte/SHA join and terminal/reference checks; native whole archive remains qualified.
- `.codex/scratch/fit120_final_rescore_packet_attempt05/root_stage_attempt06/root_index_attempt06.json`: actual source-only stage PASS, 26 extracted members; remote receipt 12,381 B SHA `05aa3f03a946addce0017275a931b4446b803e1293dbfe91a15159f1350a4843`. Late binding/launch are closed until actual fit terminal receipts.
- `runs/cross_tfg_audio_head_20261003/native_encoder_document_corrections_attempt02/`: additive source-equivalence and condition-pair reports root-read in full; original evidence remains unchanged.
- `runs/cross_tfg_audio_head_20261003/native_encoder_ditto_retry_attempt02/requests_attempt02/request_index_attempt02.json`: source-only Ditto56 retry package prepared, not staged/run; root has only partially reviewed it.

Last measured SHM available 4,317,995,008 B after producer failure; cgroup available ~31.4 GB. The trainer's future 4 GiB reserve must be measured again after the complete producer. Local C free ~6.60 GB, F ~9.40 GB (filesystem qualification pending), E ~1.30 GB; B CloudDrive ~1.259 TB requires a new whole-file SHA reread before recovery use. Preserve RMS WAVs, cache, track maps and rendered RMS videos through the zero-head gate. V4 backup memfd holder629596/start774776231 may still be alive until ~10:53 UTC; no new release is authorized without actual consumer/reference proof.

## Changed Files

CONTEXT.md was committed as `16270c2`. Task68/Experiment67 and ignored HANDOFF.md record the actual failure state. Source-only learned Dev adapter and candidate-only 180-row builder were added by mechanism_audit; they await root review and actual fixed-step freeze binding. Their selection_record is explicitly the fixed terminal optimizer step120 protocol freeze, with Dev/Confirm selection false.

## Verification

Root read the entire failed producer receipt/stderr, independently checked source-stage/raw-file pins, and confirmed zero completed rows. Root downloaded the V4 retirement receipt and verified its full raw SHA; native retirement progress joined all 116 unique paths to manifest bytes/SHA. Root rechecked canonical GitHub repository, the last pushed c4a1998 commit and branch PR filter via fresh HTTP200 pages. No PR matches the branch. No pytest or general test suite was run. Training120, final fit rescore, learned Dev180 and Confirm were not run.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Last verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local/unpushed current commit: `16270c228a1dcd3419e67186d28a0335bf9c8dc3`.

## Observations

- [result] RMS is a usable measured baseline; the three fixed shared local profiles did not meet the stronger across-model criterion.
- [result] Qualified local-profile encoder response differs substantially between MuseTalk and Wav2Lip; this motivates learning transition-conditioned bands rather than selecting one static spectral profile.
- [decision] Preserve exact original PCM/clock for mux-back and scoring; retain frozen cohorts, complete failures and sealed Confirm.
- [decision] Fresh attempt03 must rerun all 40 producer rows after the smallest evidence-supported operational mapping repair; do not reuse partial failed geometry or relax scientific gates.
- [blocker] Producer02's first-row AVI/JPEG count guard prevents fit geometry completion; trained method remains unexecuted.
- [next_step] Dispatch the reviewed read-only first-row artifact probe, verify actual frame/PTS/monotone mapping, then stage and run the fresh repaired 40-row producer before rebinding training120 and final rescore.

## Relations

- continues [[Codex checkpoint - 2026-10-05T08-48-34Z - Qualified local encoder response and exact backed output retirement]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]