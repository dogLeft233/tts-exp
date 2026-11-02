---
title: Codex checkpoint - 2026-10-04T22-02-43Z - Qualified zero-initialization gradients
  and preserved cross-model evaluation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t22-02-43-z-qualified-zero-initialization-gradients-and-preserved-cross-model-evaluation
repo_root: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
status: open
capture: deliberate
repository: dogLeft233/tts-exp
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
cwd: E:/Documents/tts-exp
started: '2026-10-04T22:02:43.8303220+00:00'
username: dogle
branch: codex/cross-tfg-audio-head-20261003
project: tts-exp
trigger: compact
agent: codex
git_sha: 338327bc93299bd2e2818a4c2b97fe17968586f8
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T22-02-43Z - Qualified zero-initialization gradients and preserved cross-model evaluation

## Summary

The shared nine-parameter recipient transition head now has an actual zero-initialization CPU result: it preserves canonical RMS PCM exactly while all nine parameters retain nonzero gradients; independent artifact recovery and numerical replication are the next steps.

## Story

The project began by explaining why Chinese TTS can improve talking-face generation. The user's current target is a portable enhancement head that drives video generation with transformed natural speech, restores the exact original natural PCM and clock afterward, and improves synchronization across several generators. Continue until the method, inference interface, weights, latency and independent confirmation are complete. Root owns scientific design and independent qualification; GPT-6 Luna/max agents implement source and experiments. Alibaba Qwen ASR/TTS and historical Baidu data are authorized, but secrets remain runtime-only.

The current approach shapes recipient-local positive and negative spectral transitions through a bounded nine-parameter mask, retains utterance length and phase in the construction, and projects the final output onto the canonical RMS baseline. Global RMS preservation does not preserve local 40 ms power; positive-minus-negative flux contains amplitude changes as well as spectral movement. These distinctions remain part of the mechanism analysis. The held-out confirmation cohort and reserved LatentSync stay sealed.

Root actually deployed and ran the synthetic zero-theta supplement once, then independently inspected its terminal artifacts. The worker returned zero; all ten engineering checks passed, zero theta produced exactly the baseline PCM, and nine coordinate finite differences plus the directional comparison agreed with autograd. This is CPU engineering evidence only. The prior 40-row independent NumPy audit remains COMPLETE_WITH_NUMERIC_MISMATCHES: 16 rows passed every check and 24 failed only the frozen flux-identity tolerance. A fresh arithmetic diagnostic includes final-residual rounding and promotes original float32 magnitudes before subtraction; it has now run once and produced COMPLETE_DIAGNOSTIC for all 40 rows. Independent enumeration of its residual/bound arrays remains next.

Shared three-model evaluation continued to 302/540 complete PASS rows at the last actual read. Source inspection also resolved Ditto's actual wrapper versus neural-module types and its offline audio-queue sentinel. Persistent frontend qualification is still source-only; resident rows must point explicitly to the corresponding cold-reference row rather than rewrite receipt identities or introduce extra calibration renders. The existing W2L v4 objective already rewards confidence and far-lag separation with anchored-distance guards; do not revert to the earlier D0-only objective. Its factory explicitly requires CUDA, and the nine-parameter head needs a new adapter. Hard uint8/OpenCV rendering uses a straight-through derivative, so full-score hard-forward finite differences cannot be represented as validation of that derivative.

## Working State

Durable design is tracked in the existing enhancement Task, Experiment and ADRs; the latest report25 remains concluded, while report26 and the method deliverable are unfinished. Branch is codex/cross-tfg-audio-head-20261003; HEAD 338327bc93299bd2e2818a4c2b97fe17968586f8 is local and unpublished, with no PR. Existing dirty files and the numerous scratch/run artifacts are machine-local; git status is captured in .codex/scratch/checkpoint_DQ_git_status.txt. No new commit was made in this interval.

Machine-local authoritative pointers: .codex/scratch/root_zero_theta_cpu_source_stage_receipt_attempt01.json (SHA-256 17a8aedcc712f3bc19dc8e73af44a3c0ed835e45e402c71024d014f6174d1b27); .codex/scratch/v3_zero_theta_gradient_cpu_packet_attempt01/13_run_zero_theta_cpu_attempt01_request.json.response.json (419eea14301e905238ff8d3b4af9d21ee4b291d8062d8c88fc23e6c0613e5fdf); and .codex/scratch/v3_zero_theta_gradient_result_inspection_attempt01/03_inspect_zero_theta_result_attempt01_request.json.response.json (df3049efc0ab999dce906898ad3aa509914519dc92e50acc56b3a255c887eee7). Terminal output pins are result.json 5648 bytes/68459ce9271c1400ce6f56c5ca056534352bb98edcb1c65bc7b6bb2d28747633, result_manifest.json 2335 bytes/c3b7dc842ca52b4f696b17b2c170c069c71969086a12b86928526930ef77f26b, and zero_theta_diagnostic_arrays.npz 12537 bytes/ebd7457feb6547383dadf100def9bf957f4cbcb2839f2f41de13be0c0b6741f1. The three-file download packet is unexecuted and its source-stage pointer must be corrected before adoption.

The independent zero-theta NumPy audit attempt04 source/config is qualified but unrun. Float32 cancellation diagnostic attempt02 actually completed; result_manifest.json is 761 bytes/SHA-256 9082ff6bddbe8ffe77bb9e5fc0d7dcd9b4d2fc5c98fccbf2354fc30be4a3c7a9, summary.json is 107697 bytes/1966b870e09307c85fcb3294f89384af29e8baa46092727b37fb4dbf9c6c05e5. Its result is pending root array-level interpretation. Stream02 is rejected before execution because its terminal-schema assumptions and seven-field versus four-field inventory comparison are incompatible with actual producers; fresh Stream03 is being prepared. Persistent Muse07/Ditto08 and plan05 are being prepared with explicit cold-reference mappings. Do not execute GPU qualification, postrun scoring or ASR until the active 540-row run is terminal and exact discovery binds its seals. Remote server reference is root@connect.weste.seetacloud.com:12349; private password/API key are not in this note. Runtime-only credentials and bridge state may need reconstruction from authorized human context after process loss.

## Changed Files

Actual deployed source: scripts/experiments/cross_tfg_audio_head_20261003/qualify_shared_recipient_transition_head_v3_zero_theta_gradient_attempt01.py. Qualified independent consumer: independent_numpy_zero_theta_audit_attempt04.py in the same directory. Prepared diagnostic: diagnose_v3_flux_float32_cancellation_attempt02.py and its contract. Native frontend and recovery stream revisions remain untracked, source-only work and must pass root review before execution. CONTEXT.md, HANDOFF.md and the mutable Task/Experiment still describe an earlier snapshot and need reconciliation after the pending numeric readbacks.

## Verification

Root checked raw source/config pins, parsed Python source and embedded request Python, ran bash syntax checks, inspected exact dispatcher response envelopes and source seals, and actually verified all three remote zero-theta artifacts. Recorded runtime was Python3.10.12/NumPy1.26.4/Torch2.5.1+cu124 with one thread and CUDA uninitialized. All28 arrays were finite; nine gradients were nonzero; directional absolute error was 1.34911713e-6 against the frozen 0.0004087027544 allowance. No dataset audio, model forward, training, video, API or persisted PCM was involved. No pytest or unit suite was run. Shared540 last read: 302 PASS, no failed/incomplete rows, same process identities; remaining SHM 3833921536 bytes and root disk 2164178944 bytes. Latest measured resource response SHA-256 41eb408e7992f3d72b886b545384d7301f2b0d87db7938a1087f9112c187197a.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Latest verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)

## Observations

- [result] Zero-theta CPU engineering qualification passed exact-baseline identity and all nine coordinate plus directional gradient comparisons; transfer efficacy is still pending.
- [decision] Preserve the original 24 flux-identity failures and explain their arithmetic using a separate sealed diagnostic instead of changing the frozen tolerance.
- [decision] Preserve explicit cold-reference identities and the 48-row persistent qualification roster; do not add implicit calibration renders.
- [blocker] Shared540 still occupies the GPU; postrun, native parity and held-out confirmation remain pending.
- [next_step] Correct and root-review the three-file zero-theta download provenance, recover the actual files, then run the already qualified independent NumPy04 replication.

## Relations

- continues [[Codex checkpoint - 2026-10-04T21-33-02Z - Independently audited fit40 signal path and prepared zero-initialization gradient check]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]