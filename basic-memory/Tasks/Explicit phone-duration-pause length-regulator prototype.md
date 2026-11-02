---
title: Explicit phone-duration/pause length-regulator prototype
type: task
permalink: tts-exp/tasks/explicit-phone-duration-pause-length-regulator-prototype
assigned_to: claude
current_step: 5
priority: high
started: '2026-08-15'
status: done
completed: '2026-08-15'
steps:
- Implement pure duration/pause allocation and direct tokenwise feature length regulator.
- Add synthetic known-warp training runner with source hashes and exact frame/owner
  coverage.
- Add unit tests for allocation, pause separation, MFA mapping, dynamic shapes and
  split contract.
- Run tests, CPU dry-run and non-cloud smoke output.
- Run 20/5 feature-only gate; stop before audio/SyncNet or new Qwen data if validation
  does not beat baseline by 5%.
tags:
- task
- length-regulator
- duration
- pause
- mfa-linear
---

# Explicit phone-duration/pause length-regulator prototype

## Observations
- [gate] Full CUDA 20/5 run completed 2026-08-15. Four S0765 samples completed 36 synthetic conditions and achieved 0.7511 weighted feature-loss improvement over the global baseline, with exact owner/pause coverage; sample 3 was fail-closed because natural speech token `w` had no equal-label TTS token (TTS began with `ʔ`). The complete 5-sample validation gate therefore failed, so no audio rendering, SyncNet, or new cloud TTS was run.
- [decision] Keep unmatched speech fail-closed rather than silently treating `w`/`ʔ` as equivalent; any replacement policy requires a separate explicitly defined experiment.

- [implementation] Added `scripts/phone_duration_length_regulator.py` with exact stable largest-remainder allocation, separate phone/pause scales, ordered MFA plan construction, unmatched-pause placeholders, short-span nearest-frame fallback, and direct fixed-anchor attention output.
- [implementation] Added `scripts/train_phone_duration_length_regulator.py` with local-TTS-only WavLM-L6 preparation, deterministic 3x3 phone/pause synthetic conditions, source/token hashes, 20/5 split validation, and fail-closed preparation reporting.
- [verification] Added 13 unit tests covering `spn`, exact allocation, placeholders, speech rejection, owner coverage, dynamic output, gradients, split contract, and synthetic conditions. CPU dry-run and one-train/one-S0765 non-cloud smoke completed; smoke was structural only and did not justify downstream audio.

- [description] Implement a new direct feature-level length regulator after the failed nearest-resample residual adapter phase. The prototype separates phone-internal duration from boundary pause duration and consumes the full source phone trajectory/context.
- [status] done
- [assigned_to] claude
- [current_step] 5
- [started] 2026-08-15
- [decision] No Qwen cloud calls, no SyncNet selection, and no S0770. Use 20 local-TTS train utterances and five S0765 validation utterances.
- [constraint] Existing failed adapter outputs and Qwen AISHELL/RAMC test cohorts remain untouched.

## Steps
1. [x] Implement pure duration/pause allocation and direct tokenwise feature length regulator.
2. [x] Add synthetic known-warp training runner with source hashes and exact frame/owner coverage.
3. [x] Add unit tests for allocation, pause separation, MFA mapping, dynamic shapes and split contract.
4. [x] Run tests, CPU dry-run and non-cloud smoke output.
5. [x] Run 20/5 feature-only gate; stop before audio/SyncNet or new Qwen data because the complete split gate failed on one unmatched speech token.
## Context
The previous phase-1 residual adapter trained on nearest-resampled local-TTS features but worsened S0765 validation. This prototype must not extend that model. It should use `load_inputs()` from `scripts/run_aishell1_vocoder_domain_split.py`, token semantics/matching from `scripts/knn_vc_retrieval.py`, and the frozen WavLM-L6/1024-D interface from `scripts/wavlm_knn_vc_adapter.py`.

The desired target is still not a ground-truth hybrid waveform. This phase only tests an explicit, deterministic phone-duration/pause renderer against TTS-only analytic known-warp targets. Natural full WavLM features must not be training targets.

Related research: [[Explicit duration/pause-conditioned acoustic rendering]], [[AISHELL-1 duration/time-warp temporal adapter phase 1]], [[TTS Feature Alignment and Duration Model Research]]
