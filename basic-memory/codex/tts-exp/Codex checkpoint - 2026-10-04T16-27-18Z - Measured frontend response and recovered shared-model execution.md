---
title: Codex checkpoint - 2026-10-04T16-27-18Z - Measured frontend response and recovered
  shared-model execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t16-27-18-z-measured-frontend-response-and-recovered-shared-model-execution
repo_root: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
repository: dogLeft233/tts-exp
cwd: E:/Documents/tts-exp
trigger: compact
model: gpt-6.1-sol
agent: codex
capture: deliberate
started: '2026-10-04T16:27:18.1107471+00:00'
username: dogle
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
project: tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 41332dae20743b1d5f2554094c6a0c8d6a7575d2
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T16-27-18Z - Measured frontend response and recovered shared-model execution

## Summary

Measured and independently checked frontend gain responses, preserved the old Ditto outputs in two verified local copies, and retired the original run while preparing the shared three-model enhancement experiment.

## Story

The user began by asking why TTS improves talking-face generation and then made the practical objective explicit: deliver a natural-audio enhancement head h(x), generate video with h(x), restore the original natural PCM and source clock, and obtain transferable SyncNet improvement. Continue through failed candidates to a complete usable method. Root owns theory, protocol, memory, private dispatch and independent QC; experiment/model code belongs to GPT-6 Luna/max subagents. Qwen ASR/TTS is authorized when useful; secrets remain private runtime data.

The current scientific lesson is that waveform gain becomes different model inputs. Actual fit8 frontend measurements distinguish an almost common Whisper log-mel offset from the Wav2Lip fixed-floor response that also changes centered spectral shape. This motivates measuring native encoder directions and local transitions, then tying those changes to actual generated motion and restored-audio scores. The input probe alone does not establish a video effect. V4 already improved over identity but lost to fixed RMS across all three models, so it remains a rejected candidate for the stronger method target; the work continues with shared local spectral profiles and an expanded historical fit cohort.

The immediate operational obstacle was SHM capacity. The historical Ditto retry04 archive is now complete and whole-hash verified on C and F, with a strict F restore and independent canonical member comparison. The original run was actually removed through a narrowly guarded worker, and an independent postQC found its target absent and all 32 protected outside identities unchanged. The remaining remote tar is an exact duplicate; only that pinned tar is pending retirement before fresh resource admission and shared540 launch. Do not retransfer or repeat the restore.

## Working State

Durable task: [[开发跨 TFG 的自然音频增强头]] and experiment [[跨 TFG 自然音频增强头的训练与原音换轨验证]]. Authoritative protocol: `openspec/changes/develop-cross-tfg-audio-head-20261003/`. Development remains 20 clips/10 male speakers × three seeds and Wav2Lip/MuseTalk/Ditto. Confirmation 80 clips/40 unseen speakers and reserved LatentSync remain sealed. Full cohort, failed rows, original PCM/clock, fixed-RMS comparison and independent content/visual checks are preserved. Fit40 under ADR013 is qualified but coefficients are not fitted/frozen.

Local unpublished HEAD is `41332dae20743b1d5f2554094c6a0c8d6a7575d2` on `codex/cross-tfg-audio-head-20261003`; there is no PR. Historical dirty source/tests remain untouched. `HANDOFF.md`, `.codex/scratch/`, and most `runs/` artifacts are machine-local/fragile. Task46/Experiment45 reflect frontend QC and C/F completion but predate actual run retirement; reconcile their state after tar retirement and launch.

Server endpoint is `root@connect.weste.seetacloud.com:12349`, host key `SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0`. Private PTY 50274 and bridge06 port10887/PID252640 were active; runtime SecureStrings are fragile and must be requalified after loss. Namespace `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`; output SHM `/dev/shm/cross_tfg_audio_head_20261003`. No secrets belong in files or memory.

Shared540 packet07 staging requests00–03 actually passed. CPU04 and GPU launch05 have not run. Plan SHA `43e5f45a987153219bc423afbb86512f441ff12db20c3cb56dfb776e14e567df`; fresh admission index `runs/cross_tfg_audio_head_20261003/shared_profile_shortlist_dev_attempt03_execution_attempt01/admission_snapshot_attempt04/requests_attempt04/request_index_attempt04.json`, SHA `f7bd978334ed677dcc77002cb3eb6b05a6e37b54d58cb007cedd06d53a8a12ea`, uses singular request_path/request_raw_sha256 fields. Keep ≥5GiB SHM and ≥18GiB GPU/two samples/no unknown GPU applications gates. Packet07 CPU04 request SHA `0d290e1645ed6c08efb3250aff44b69b897095b95cdcdfecc64a6a04fb96ba23`; launch05 SHA `1fb049c22ba797727145ea504f306421152cc8a75dd6454c612c58ad29c00d80`; producer status06 SHA `2b90a8401a9406ca3150376424f003327a632bba6148a6bd0baf3f7456da0662`.

Native encoder probe168 units is prepared but not run. New packet02 index SHA `f738251674e32c1b8edacc4c3982a2c45f0194e0952d320c9c21d6df872762da`, runner SHA `8887f36787a6507987d629ebf943ad5c67b492885a5f5c6b9252ca12ba2c478a`, standalone plan SHA `ab34453272b391d00d99d6aa8713c13a31ca460c97b0b5854d9e0889a6807132`. Root reviewed an earlier source only partially; fully read the newer source and actual request arguments before dispatch. It must use the exact frozen fit360 plan/schema. Avoid concurrent GPU work with shared540. Seven conditions are natural, ±6dB, fixed RMS and three profiles on eight identical historical WAVs; no dev/confirm/video scoring.

900-row score join04 input bundle is staged but not run. ASR private client08 is statically reviewed; no new API call. README09/map09 now point to actual packet07 status06. Future terminal binding must come from actual complete bridge06 response bytes and preserve all540 rows. Cross agent is preparing corrected postrun worker/discovery: manifest PASS requires both FAILED=0 and INCOMPLETE=0. Candidate visual production/join is not run.

## Changed Files

Current commit contains `scripts/experiments/cross_tfg_audio_head_20261003/frontend_gain_feature_probe_attempt03.py`, its preflight03 and runtime contract02, `CONTEXT.md`, and the measured frontend section of `openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md`. Immutable earlier attempts and existing dirty files are preserved. Scratch native probe, postrun packets, and retirement packets are untracked local preparations.

## Verification

Actual frontend probe03 produced summary191356B SHA `6af5b7a66a73f8a40138fb3cac2d98adcb33751327cce4bc5c75b8d77eaad888`, NPZ459001B SHA `143daf5014c15ff4f59157744b809592f5dcfbbe4f70b8fef2bcb6c902232665`, and 96-row CSV. Fresh verifier04 fixed only the roster join after a failed verifier, leaving scientific outputs unchanged. Root independently recomputed original archived PCM→Ditto affine/frame RMS and NPZ/CSV summaries: 640 numeric checks PASS, maximum difference2.657e−8; receipt `.codex/scratch/root_frontend_probe_numeric_qc_attempt01.json`, SHA `ecbc5e0d37a84fe2012dde7d301a7fb5d486b7dff8ecba047226b188cfd61a96`. Whisper ±6dB means −.150000013/+.149999997 and centered absolute RMS≈5.6e−7; W2L means−.359316/+.412290, centered absolute RMS .098804/.083135. Frozen relative_rms labels are absolute differences, not dimensionless ratios. Actual Transformers4.39.2 source/runtime was qualified.

Old Ditto C/F archive1615391171B SHA `d0b48b51c4a0c0ddf9a173fd14b0a4139069dfe48adac7a3a2726055813ae6ed`: C404 ranges COMPLETE/transfer processexit0; F61562 files/4557 directories/1732134403 logical bytes strict restore PASS. Root66118-member digest `2f66a0bf53606e02d0eda35b88c99d32d4f15e48da04cff33a09fdfde26dbe2f` matches fresh remote audit02. C root `C:/Users/dogle/.codex/tts-exp-artifacts/recoverable-output-archive-20261004-attempt05-ditto-parallel4/`; F root `F:/tts-exp-archives/recoverable-output-archive-20261004-attempt06-ditto-parallel4/`.

Actual run retirement02 requests00–09 PASS; receipt `.codex/scratch/ditto_retry04_output_retirement_exec_attempt02/retirement_receipt_attempt02_remote.json`,8146B SHA `504fb355aeb91a19fb718196e02b3b0f912d51f1c9cd4dd9afb04782a8f83cb2`. Independent postQC01 exit0 checked target absence, exact receipt/intent and all32 protected identities; outer response SHA `903cb3d91c075421b7cc12d04c5fd74e1ef71767254a4ed31ea503b1ed7b5036`. Pending tar-only target is `SHM/recoverable_output_archive_ditto_retry04_attempt01/stageb_ditto_attempt03_retry04_output_archive_attempt01.tar.gz`, inode782950/dev118/mtime1791126019. Run target is already absent: do not invoke deleted-tree validation. Reuse pinned executed external C/F validator and audit02 activity/queue checks, exact single unlink, then independent postQC.

`git diff --cached --check` passed for current commit; no pytest/unit tests were added or run. Native probe and new postrun packets have not been executed. Shared540 GPU work is still pending.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Last verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local unpublished current commit `41332dae20743b1d5f2554094c6a0c8d6a7575d2`.
- `.codex/scratch/recoverable_output_ditto_transfer_attempt07/root_ditto_recovery_bindings_qc_attempt01.json`,2123B SHA `4e3c1c8eedfa9ce40d741172201bb08ecb2d75a8f149e546f9cf33cc118be751`.

## Observations

- [result] Whisper and Wav2Lip convert the same gain perturbation into different frontend trajectories; local shape and encoder direction need model-specific measurement.
- [result] Historical Ditto outputs are fully recoverable on C/F and the original run is retired with independent protected-path postQC.
- [decision] Preserve sealed confirmation, fixed-RMS baseline, original PCM/source clock and separate input/encoder/video/evaluator estimands; failed candidates trigger bounded historical-fit redesign.
- [decision] Run one native GPU experiment at a time and retire only exact outputs backed by verified recoverable copies.
- [blocker] Shared540 awaits removal of the one pinned remote duplicate tar and fresh resource admission; native encoder/postrun/ASR/visual output remain unexecuted.
- [next_step] Review and execute the tar-only retirement packet, independently verify it, then admit and launch the frozen shared540 experiment.

## Relations

- continues [[Codex checkpoint - 2026-10-04T15-41-48Z - Qualified frontend measurements and continued shared-head execution]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]