---
title: Codex checkpoint - 2026-10-05T07-23-09Z - Qualified RMS and binary V4 restore
  while continuing learned head
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-05-t07-23-09-z-qualified-rms-and-binary-v4-restore-while-continuing-learned-head
project: tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
branch: codex/cross-tfg-audio-head-20261003
cwd: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
status: open
username: dogle
started: '2026-10-05T07:23:09.2454468+00:00'
git_sha: af6e1caaa532f8e46b2e1401cf9e813c3767eac5
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-05T07-23-09Z - Qualified RMS and binary V4 restore while continuing learned head

## Summary
Reusable RMS baseline is committed and independently qualified; 112 native encoder forwards completed, and the V4 archive now has a complete independently verified local restore.

## Story
The original question was why TTS improves talking-face generation. The user expanded it into a usable natural-audio enhancement head for many TFG models: enhanced audio drives generation, then exact original natural PCM and clock are restored for scoring and delivery. Root maintains theory, frozen protocols, independent qualification, BM and private dispatch; GPT-6 Luna/max agents implement code and experiments. Qwen ASR/TTS and data/storage recovery remain authorized. Credentials stay in private runtime only.

The qualified shared900 development evidence supports RMS as a reusable baseline: FULL Cbest gains versus identity are +0.234 Wav2Lip, +0.053 MuseTalk, +0.173 Ditto, equal-model +0.154. Three manual local profiles add -0.007584, +0.005065, -0.026333 over RMS; none meets the frozen three-positive-model/equal-model +0.050 criterion. GUARD retains 45 ineligible rows and the null full-ten-speaker result. All60 ASR calls had zero CER/disagreement; the user's latest Qwen ASR authorization does not require redundant ASR. Visual540 structure is qualified, not a perceptual-quality claim. Confirm80/40 unseen speakers and LatentSync remain sealed. Learned weights, method freeze, confirmation and report26 do not exist yet.

Root independently decoded the three CLI qualification WAVs from RIFF chunks and original PCM16, verified exact float32 RMS-only transformation with zero sample error, no clipping and preserved clock, and checked all13 config/evidence pins. The reusable config/Chinese guide was committed at local e7761aefa5e12663f23bb3a3d08e888bf6724b2d; CONTEXT now reflects that state and remains93 lines, committed at local af6e1caaa532f8e46b2e1401cf9e813c3767eac5. The learned route remains a three-band static/rising/falling nine-parameter head with exactly120 updates, preceded by separate historical fit40 cache/geometry and all40 same-GPU zero/RMS qualification within1e-5. Producer stages00-04 and the full21-member optimizer source package are qualified; producer05/06 and optimizer have not run. Final step120 checkpoint needs separate all40 final rescoring because ledger row120 is pre-update, not a common final model.

The native168 run finished: Wav2Lip56 and MuseTalk56 complete, Ditto0/56 failed before encoding because its data_root resolved one directory too high. Preserve the full112/168 incomplete denominator and full stderr. Qualified attempt18 used the nested ditto_pytorch directory. A pinned read-only asset probe and Ditto-only56 operational-path retry are being prepared; scientific plan bytes must remain unchanged. Root read the first370 lines of native archive source and found likely duplicate-summary and receipt-file set bugs before dispatch; agent is preparing fresh corrections. The112 completed features will support analysis of encoder sensitivity beyond frontend input spectra.

Storage recovery now closes the V4 backup gate. The1435576320-byte sealed archive and50436197-byte inventory passed root full typed-member/all-file hash qualification. Windows restore04 failed because text-mode os.open converted LF to CRLF; its failed tree and full stderr remain. Fresh binary restore05 actually passed49,297 members/45,814 files/1,347,004,407 logical bytes. Root then independently rewalked the entire restored tree and checked every path/type/full-file SHA: COMPLETE_EXACT_PATH_TYPE_BYTE_SHA256_PASS, receipt610 bytes SHA034f95180a2b982299793a96eec13d359e1e447e8f3d048ce305961ac16fc790. POSIX metadata is not claimed on Windows. ADR014 now prefers this fully qualified C backup; skip redundant compressed V4 overlay writes. Exact V4 source remains intact until fresh zero process/queue references and corrected retirement worker pass. Retirement04 namespace/worker/config uploads and readonly preflight have actually passed; its deletion has not run. Local agent generated the actual recovery binding; root's additional tree QC gate is now complete.

After V4 retirement, native112 outputs may leave SHM about15MB short of the fixed4GiB training admission. Archive them to persistent overlay with1GiB reserve, download/strictly qualify locally, then retire only their backed output files while preserving controls, failed Ditto evidence and source assets. Do not relax admission or create a SHM archive that frees no net space. Continue producer40, actual late-bound training120, final rescore, frozen development evaluation, method selection and unseen confirmation. A fit-only equal-three-model SPSA route and model-conditioned head fallback are specifications only; they cannot replace completed evidence.

## Working State
Current HEAD af6e1caaa532f8e46b2e1401cf9e813c3767eac5 is local/unpushed on codex/cross-tfg-audio-head-20261003, no PR. Task66 and Experiment65 were updated/read back before the final binary restore closure. Preserve all historical dirty files and untracked experiments. Ignored HANDOFF.md, scratch packets, C/E backups and processes are fragile machine-local state.

Private PTY50274 is idle at a prompt; bridge06 port10887 holds private runtime credentials. Original direct-stream PID453768 ended with exit1 and saved full stderr; do not restart transfer because C archive is fully qualified. Local restore50349 and independent QC24433 completed successfully. V4 holder629596/start774776231/fd3 expires around10:53 UTC. Endpoint root@connect.weste.seetacloud.com:12349 has pinned host identity. Server namespace is /root/tts-research-system-20261001/cross_tfg_audio_head_20261003, SHM /dev/shm/cross_tfg_audio_head_20261003. GPU is idle in the actual native terminal snapshot. Native output archive/probe packet04 has not been dispatched.

## Changed Files
- scripts/audio_enhancement/configs/rms_baseline_v1.json and rms_baseline_v1.zh-CN.md: committed reusable RMS baseline.
- CONTEXT.md: committed current actual project state,93 lines.
- .codex/scratch/root_qualify_rms_cli_attempt01.py: actual independent PCM/float32 qualification.
- C:/v4-a03/root_c_archive_full_hash_qc_attempt01.json:723-byte full archive QC.
- C:/v4-a03/root_restored_tree_independent_qc_attempt01.json:610-byte complete restored-tree QC.
- .codex/scratch/cross_tfg_audio_head_20261003/v4_memfd_restore_attempt05/: binary helper, preflight and actual1722-byte strict receipt.
- .codex/scratch/cross_tfg_audio_head_20261003/v4_output_retirement_attempt04/: corrected frozen worker/config/binder and actual stage responses; no deletion.
- runs/cross_tfg_audio_head_20261003/native_encoder_probe_execution_attempt02/: actual112/168 terminal status and Ditto failure diagnosis; archive and retry sources pending review.
- BM docs/adr/014-sealed-overlay-v4-recovery: amended decision, fully read back.

## Verification
Root RMS CLI full sample checks and13 pins pass. Native actual W56/M56 pass; D56 dependency failures retained. C archive all49k typed members/full45k file hashes pass. Binary strict restore and independent exact restored-tree rewalk pass. Retirement04 source pins and allfour actual staging operations pass, no source deletion. Fit120 full21-member source-stage receipt and literal64-character pins pass; production/training/final rescoring remain unrun. No pytest/general test suite ran.

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- openspec/changes/develop-cross-tfg-audio-head-20261003/: frozen development/confirmation protocol.
- .codex/scratch/cross_tfg_snapshot66_body.txt: current typed task/experiment body before latest restore closure.

## Observations
- [result] RMS is a qualified reusable baseline; all manual local profiles fail the incremental frozen criterion.
- [result] Native encoder run completed112/168 forwards; Ditto failures require a bound operational data_root correction.
- [result] Complete C archive and binary restored-tree independent qualification close the V4 backup gate.
- [decision] Preserve original PCM/clock, full failure denominator, sealed confirmation and all scientific gates; continue learned local-feature training rather than stopping at RMS.
- [decision] Retire exact V4 output only after fresh zero references and all actual recovery evidence; archive native outputs to persistent overlay before freeing their SHM.
- [blocker] Training still needs exact qualified V4/native output retirement and fixed4GiB space admission; no trained local head exists.
- [next_step] Bind the actual C/strict/tree QC evidence, review and dispatch exact V4 retirement04, then close native archive/space recovery and launch fit40 production.

## Relations
- continues [[Codex checkpoint - 2026-10-05T06-41-13Z - Launched native encoder investigation and staged learned head sources]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]