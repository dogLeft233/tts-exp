---
title: Codex checkpoint - 2026-10-09T01-42-51Z - Verified D0 restart and ASR consistency
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t01-42-51-z-verified-d0-restart-and-asr-consistency
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T01:42:51Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6-luna
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-09T01-42-51Z - Verified D0 restart and ASR consistency

## Summary

DirectD0 now has a unique, verified STARTED_DETACHED launch after correcting and hashing all 40 training WAV paths; a fresh 20-clip Qwen ASR calibration completed 60/60 calls with identical normalized transcripts for natural repeats and RMS inputs, while post-launch training state and the 540-row media/score join remain outstanding.

## Story

The user’s continuing objective is to explain why TTS audio can improve TFG lip-sync, then build a portable audio-enhancement head that can drive video generation across multiple TFG models before restoring the exact original natural PCM and sample clock. Work should continue through failed engineering attempts and end with a usable, well-qualified method and clear mechanism conclusion. Root maintains the experimental logic and evidence; delegated agents prepared the experiment code and packets. The active server is the migrated `root@connect.westd.seetacloud.com:18625`; credentials remain outside files and notes.

Historical Controls360 results still motivate separating the evaluator’s locally best lag from original-clock synchronization: RMS versus identity had Cbest mean difference +0.152045 (95% interval +0.002389 to +0.301701), while Dmin was +0.003304 across zero and D0 was +0.052387 across zero. This suggests RMS can improve the locally selected SyncNet score without a clear improvement at the original clock. It is a working explanation, not the final causal finding. DirectD0 is intended to test original-clock improvement with a non-worsening Cbest hinge.

Candidate180’s actual ordered ledger completed 180/180 PASS across three models (60/model) and nine model/seed groups (20/group). All seven terminal files were downloaded and checked against byte counts and hashes by the root validator; independent reviews also passed. The run’s repeat ledger is empty (`NOT_RUN_SUBRUN`), so its candidate is not repeat-qualified and these metadata checks are not a scientific admission result. The terminal inventory contains one preserved reader-schema mismatch (`cross_tfg` versus the producer’s `cross-tfg`); the producer’s actual schema was independently verified.

The previous DirectD0 attempt failed before training because the source root was joined with `source_wav` twice. A fresh read-only inventory verified all 40 files at the parent `fit40_source_inputs` root. The corrected manifest changed the source root only, and a unique launch request returned `STARTED_DETACHED`: supervisor PID 573383, start ticks 334637264. This establishes process launch only; no post-launch snapshot has yet established optimizer progress or passage through the required audio-equivalence and gradient gates.

For a new 20-clip cohort, phase A produced and fully checked natural PCM16 and RMS PCM16 inputs. The root client completed 60 fresh Qwen ASR requests (natural twice and RMS once per clip), all HTTP 200, without transcript prompts or retries. All 20 clips had identical normalized text across the three conditions. This supports transcript consistency for this cohort but does not establish human transcription accuracy or explain SyncNet effects.

The candidate/control joint media and NPZ readback packet is sealed but unrun. It targets 180 candidate rows plus 360 controls and will join archived videos, natural audio, and score arrays on identity and clock without rendering or training. The independent readback worker and packet sources still need root review before dispatch. A separate prospective Dev20 full evaluation remains downstream of candidate qualification.

## Working State

Durable state: the active Basic Memory task `开发跨 TFG 的自然音频增强头` remains active; its linked experiment `跨 TFG 自然音频增强头的训练与原音换轨验证` is marked implemented at phase `candidate180_terminal_d0_control_isolation`. The exact latest task and experiment notes were read back. The same-chat predecessor checkpoint is `Codex checkpoint - 2026-10-08T20-20-34Z - Verified candidate and corrected source-root recovery`.

Machine-local/fragile state: `HANDOFF.md` was refreshed from actual evidence and read back (6,420 bytes, SHA256 `2B1CE6A4FA9C947E7453283773EBA449180436D914B39AB605C3FA9060E4F713`). Scratch packets, the private PowerShell PTY 57588 and its in-memory credentials/bridge state are local. The DirectD0 status readback packet is `.codex/scratch/fit40_d0_gatedc_status04_readback_attempt01/request_index_attempt01.json`; the joined-media worker packet is `.codex/scratch/dev540_joint_media_readback_attempt02/`. Do not assume either packet has run or that a process is still alive merely because its launch receipt exists.

Git is on `codex/cross-tfg-audio-head-20261003`, HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`. A read-only `git ls-remote` found no branch ref, and the GitHub API showed no PR for this branch. Five existing tracked modifications and hundreds of untracked local experiment artifacts remain; preserve them. `HANDOFF.md` is ignored/local. No tests were run in this turn and no push or PR was created.

## Changed Files

- `HANDOFF.md`: refreshed with the corrected Fit40 source root, actual DirectD0 launch identity, ASR calibration result, and still-pending status/join work.
- Basic Memory task and experiment notes: refreshed to record the current candidate180 / DirectD0 isolation phase and ASR consistency result.
- Local scratch: status04 and detached watcher03 packets are available for read-only launch-state observation; the dev540 joined-media packet is available for later dispatch after root audit.

## Verification

Candidate180’s seven terminal files passed full byte/hash and metadata validation, with the reader’s one schema failure preserved. Fit40’s 40 file hashes and lengths matched the frozen source manifest; the corrected DirectD0 request returned a launch receipt binding PID 573383 and start ticks 334637264. The ASR result and independent summary cover 60 unique successful calls and 20/20 transcript-equality checks. `HANDOFF.md`, active BM note status, local branch, HEAD, GitHub repository URL, and absence of a branch PR were read back. No model or repository tests were run in this turn.

A read-only DirectD0 status dispatch was attempted through private PTY 57588, but the host rejected the non-empty `write_stdin` call with `approval required by policy, but AskForApproval is set to Never`. The command did not run and no remote request was sent. Therefore the last verified remote state remains the `STARTED_DETACHED` receipt. The 540-row media readback packet was not dispatched.

## References

- [Repository](https://github.com/dogLeft233/tts-exp) (current HEAD is local and unpushed).
- [Current local commit](https://github.com/dogLeft233/tts-exp/commit/5688784dd5c6c1272074f0c4b6c5ff79f117c907) is intentionally not linked because the commit is not present on GitHub.
- `HANDOFF.md` for the current concise machine-local snapshot.
- `.codex/scratch/fit40_d0_gatedc_training_packet_attempt02/launcher_status_receipt_fix_attempt04/requests_attempt01/06_launch_fit40_d0_gatedc_once_attempt04.json.response.json` for the actual launch response.
- `.codex/scratch/dev20_asr_repeat_baseline_calibration_attempt05/` for the 60-call run and independent transcript-equality summary.
- `.codex/scratch/dev540_joint_media_readback_attempt02/` for the sealed but undispatched 540-row readback packet.

## Observations

- [result] Candidate180 terminal metadata is complete and independently checked, but its empty repeat ledger leaves repeat qualification outstanding.
- [result] DirectD0 has a unique actual detached-start receipt after a corrected 40/40 source-file verification; optimizer progress is unknown until status is retrieved.
- [result] Natural-repeat and RMS ASR outputs matched exactly after normalization on all 20 new clips.
- [decision] Keep locally best SyncNet lag (Cbest) distinct from original-clock D0; RMS gains in Cbest alone do not settle the native-clock mechanism.
- [decision] Preserve all existing failed receipts and the producer/reader schema mismatch; use new immutable packet attempts for corrections.
- [blocker] The current private PTY write path rejects commands under the host’s approval policy, preventing read-only status and media packet dispatch.
- [next_step] Restore a permitted local dispatch path and run the already-reviewed read-only DirectD0 status packet, then continue with the root audit and dispatch of the sealed 540-row candidate/control media join.

## Relations

- continues [[Codex checkpoint - 2026-10-08T20-20-34Z - Verified candidate and corrected source-root recovery]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]