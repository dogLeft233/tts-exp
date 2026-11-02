---
title: Codex checkpoint - 2026-10-04T22-35-01Z - Independently verified transition
  gradients and explained numeric discrepancies
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t22-35-01-z-independently-verified-transition-gradients-and-explained-numeric-discrepancies
cwd: E:/Documents/tts-exp
project: tts-exp
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
capture: deliberate
repository: dogLeft233/tts-exp
status: open
started: '2026-10-04T22:35:01.9288132+00:00'
branch: codex/cross-tfg-audio-head-20261003
agent: codex
hostname: LAPTOP-E133GSLE
git_sha: 338327bc93299bd2e2818a4c2b97fe17968586f8
username: dogle
repo_root: E:/Documents/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T22-35-01Z - Independently verified transition gradients and explained numeric discrepancies

## Summary

The nine-parameter transition head now has independent NumPy gradient and baseline verification, while all 40 flux-identity discrepancies fit a measured float32 rounding bound and cross-model evaluation continues.

## Story

The original research asks why Chinese TTS improves talking-face generation. The current user target is a usable natural-audio enhancement head across several generators: transformed audio drives video, then the exact original natural PCM and clock are restored. Root maintains theory, frozen protocols, independent artifact qualification and Basic Memory; GPT-6 Luna/max agents implement experiments and source. Qwen ASR/TTS and historical Baidu data are authorized. Continue through method, interface, weights, latency and independent confirmation rather than stop after exclusions or failed candidates.

The shared transition head uses nine bounded parameters to shape recipient-local spectral transitions, preserves length and construction phase, and projects the result onto canonical RMS PCM. Root actually recovered all three zero-initialization artifacts and ran the independent NumPy04 audit once. All nine coordinate finite differences and the directional derivative agree within the original tolerances, and zero parameters reproduce the saved canonical RMS baseline exactly. Three supplemental per-cell activity bits disagree between Torch and NumPy STFT arithmetic; actual head gating uses frame-total activity. Preserve the audit's COMPLETE_WITH_MEASURED_COMPARISONS status. A fresh agent-generated arithmetic diagnostic derives magnitudes from saved Torch complex components and reports exact reproduction of all worker activity bits; root must review that source and measured result before adopting the explanation.

Root independently enumerated the flux02 diagnostic's 40 rows, 22,078 valid transitions and three support masks. Every float32 residual lies below its derived rounding bound, and both identities become exactly zero when original float32 magnitudes are promoted before subtraction. This explains arithmetic differences without changing the original 24 rows that failed the frozen 2e-6 tolerance. None of these engineering checks establishes transfer efficacy.

Shared540 reached 349 complete PASS rows, no failed or incomplete rows, at the latest actual read. The GPU remains occupied. Root found a unit error in the source-only native adapter02: V3 exports a dB mask, whereas V4 mask-TV expects natural-log gain. A fresh adapter03 converts by ln(10)/20 and retains the literal frozen objective weights. Its bounded one-row CPU runtime and waveform Jacobian check are being prepared; no native model runtime has executed. Persistent Muse07/Ditto08, plan05, score-replay02 and consumer03 remain source-only. The frozen A/B input WAV headers must be read before accepting canonical float32 bindings.

## Working State

Durable design is tracked in the existing enhancement Task/Experiment and ADRs. Report25 is concluded; report26, the portable method deliverable and held-out confirmation remain unfinished. Confirm80/40 unseen speakers and reserved LatentSync remain sealed. Branch codex/cross-tfg-audio-head-20261003, HEAD 338327bc93299bd2e2818a4c2b97fe17968586f8 is local/unpublished, with no PR. Existing dirty files remain untouched. CONTEXT.md is 93 lines and HANDOFF.md is 52 lines; both now point to the actual numeric work, although their 331-row resource count predates the latest read. Task53/Experiment52 similarly need reconciliation. HANDOFF, runs, BM and scratch files are ignored or machine-local; no new commit was made in this interval.

Exact measured pointers: runs/cross_tfg_audio_head_20261003/v3_flux_float32_cancellation_attempt02/output/summary.json, SHA-256 1966b870e09307c85fcb3294f89384af29e8baa46092727b37fb4dbf9c6c05e5; root receipt .codex/scratch/root_v3_flux_arithmetic_independent_receipt_attempt02.json, 539 bytes/0b29f560eeecae3ed707a57782753086389814844c6dc15e87adb642416db338. Independent zero-theta output is runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_numpy_audit_attempt04/output/independent_result.json, 18354 bytes/5035cd6cc9920eabb49033853629ef088f3f2664234db9bd6ee7e43315577fc4. Actual downloaded zero-theta inputs live in shared_recipient_transition_head_v3_zero_theta_gradient_attempt01_readback_attempt03/files/ under the same run root. Standalone QC05 readback receipt is .codex/scratch/v3_zero_theta_gradient_readback_qc_attempt05/independent_readback_qc_attempt05.json, 832 bytes/4c1b4adf7e745433765cb0a73f538f90c77514c284b5af33017791cefc7b6830; earlier rejected download/QC revisions remain unchanged.

Shared540 resource06 latest response is 29811 bytes/SHA-256 b60590c55aeba109c8847af8b33e32e0753453d4a5d75ba9aa7fcf9402532836, actual time 2026-10-04T22:22:07Z, SHM free 3452878848 bytes and root free 2164154368 bytes. Same launcher/runner/wrapper identities remain active. Next resource read must wait until at least 22:37:07Z and back up the previous raw response. Remote reference root@connect.weste.seetacloud.com:12349, host key SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0. Password/API key stay runtime-only. Private dispatch PS session50274 and bridge port10887 are fragile; reconstruct credentials only from authorized human context if lost.

## Changed Files

Root updated CONTEXT.md and ignored HANDOFF.md; git diff --check passed. Qualified workers are diagnose_v3_flux_float32_cancellation_attempt02.py and independent_numpy_zero_theta_audit_attempt04.py under scripts/experiments/cross_tfg_audio_head_20261003/. Native adapter03 and bounded CPU worker are being prepared. Recovery stream03 is source-qualified but unrun; strict terminal discovery, postrun, ASR and native-consumer seals must precede any streaming or retirement. Agents are preparing a raw-byte reproducibility ZIP under openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/ so authoritative CRLF configuration pins survive Git normalization. Do not stage all untracked attempts or unrelated historical changes.

## Verification

Root checked the three actual downloads' raw bytes/SHA-256, all 28 finite CPU arrays, and NumPy04's manifest, 27 finite arrays, gradients, baseline and saved-source comparisons. NumPy coordinate-FD versus worker-FD maximum error is about 1.90e-7; versus autograd about 5.68e-5 within original atol .0002 + rtol .12. Directional error is 1.467788e-6 within .000408702754. Only activity cells [213,6], [247,4], [256,8] disagree, with NumPy/worker sums 64502/64503. Separate analytic synthesis is descriptive and differs at 8370 samples by maximum 5.96046e-8; saved-source baseline identity passes exactly. Root flux02 array QC found zero bound exceedances, maximum residual 4.76837158203125e-6 and maximum error/bound .06435823535893856. Python AST, full source differences and exact envelopes were reviewed; no pytest/unit suite was run. Persistent score-replay02 was fully read against actual StageA helper contracts; header request and consumer03 still await root review/dispatch.

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Latest verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)

## Observations

- [result] Independent NumPy verifies zero-parameter RMS identity and all nine coordinate plus directional gradient comparisons at the frozen tolerance.
- [result] All 40 flux rows fit the derived float32 cancellation bound; original numeric-mismatch statuses remain preserved.
- [decision] Convert V3 dB masks to natural-log gain before the existing V4 TV penalty; do not retune objective weights while correcting units.
- [decision] Keep confirmation sealed and preserve complete cohorts and failures; engineering qualification does not establish transfer efficacy.
- [blocker] Shared540 remains active, so postrun/ASR/native GPU qualification and subsequent held-out confirmation are pending.
- [next_step] Read and root-qualify the exact A/B WAV-header request, dispatch it read-only, and use the measured formats to finish persistent frontend binding while the bounded CPU adapter is prepared.

## Relations

- continues [[Codex checkpoint - 2026-10-04T22-02-43Z - Qualified zero-initialization gradients and preserved cross-model evaluation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]