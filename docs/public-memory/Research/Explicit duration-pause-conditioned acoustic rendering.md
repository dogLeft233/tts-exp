---
title: Explicit duration/pause-conditioned acoustic rendering
type: research
permalink: tts-exp/research/explicit-duration-pause-conditioned-acoustic-rendering
date: '2026-08-15'
status: complete
topic: duration-pause-conditioned-acoustic-rendering
project_context: tts-exp
sources:
- https://arxiv.org/abs/2006.04558
- https://proceedings.mlr.press/v97/qian19c.html
- https://arxiv.org/html/2604.21164v2
- https://arxiv.org/html/2506.13295v2
tags:
- research
- duration
- pause
- prosody
- voice-conversion
- mfa-linear
- supervision
---

# Explicit duration/pause-conditioned acoustic rendering

## Context

This research note connects a small literature review on duration-controlled speech generation and voice-conversion disentanglement with the current AISHELL-1/Qwen MFA-linear experiments. The practical question is how to preserve TTS content/speaker characteristics while using natural phone duration, pauses and prosody when no ideal hybrid waveform target exists.

## Literature findings

### FastSpeech 2

FastSpeech 2 explicitly extracts duration, pitch and energy from training speech and uses them as acoustic conditions. Its length regulator makes target frame occupancy an explicit model input rather than an emergent side effect. This is the closest established pattern to replacing the current fixed phone-local interpolation.

### MAGIC-TTS

MAGIC-TTS separates token content duration from boundary pause duration and provides explicit local timing controls. This distinction is directly relevant to RAMC: phone-internal duration, pause allocation, `spn`, and silence should not be collapsed into one duration ratio. The work also emphasizes high-confidence local timing supervision and robustness when controls are missing.

### Speech editing test-time training

Recent speech-editing work uses direct acoustic supervision in unedited regions and auxiliary constraints in edited regions. The edited region has no exact ground-truth acoustic target, so duration constraints, phoneme prediction and boundary continuity provide indirect supervision. This is a useful template for our missing hybrid-target problem.

### AutoVC

AutoVC shows that a carefully designed bottleneck and self-reconstruction objective can support content/style transfer without parallel conversion targets. However, it depends on representation bottlenecks and does not by itself identify the desired combination of TTS timbre and natural timing.

## Relation to project evidence

The existing BM records show that:

- The native-TTS corruption denoising adapter reduced training loss but did not transfer to MFA-linear trajectories.
- The first duration/time-warp adapter phase trained on 734 local-TTS phone segments but failed speaker-disjoint S0765 validation: feature loss `0.012119 → 0.012701`, cosine distance `0.005946 → 0.006309`.
- The prematched/regular vocoder split shows a real vocoder-domain confound, but changing the vocoder does not remove the trajectory mismatch.
- Feature or reconstruction loss improvements do not reliably predict downstream SyncNet improvement.

Therefore, increasing the current residual adapter capacity is not justified. The failed phase-1 model mainly saw a coarse retimed trajectory and learned a residual; it was not an explicit duration-conditioned acoustic renderer.

## Current design conclusion

The target should be factorized rather than represented by one unavailable hybrid waveform:

```text
TTS:      phone content, continuous trajectory, target speaker/timbre
natural:  phone duration, pause allocation, relative F0/energy/prosody
model:    explicit length regulator + conditional acoustic renderer
```

Training should combine:

- TTS self-reconstruction for content/speaker anchoring;
- explicit MFA duration and pause supervision;
- natural low-dimensional prosody supervision rather than full natural WavLM matching;
- direct supervision in unchanged regions;
- phoneme, duration and boundary-consistency losses in retimed regions;
- speaker/content invariance checks and a frozen-vocoder renderability gate.

The current Qwen AISHELL n25 and Qwen RAMC n25 remain test/diagnostic sets. They should not be used to tune a new renderer. A provider-specific renderer would require an independent Qwen training pool with multiple utterances per speaker and speaker-disjoint validation.

## Decision and next step

Do not continue the current residual temporal adapter, do not run SyncNet for its checkpoint, and do not make additional cloud TTS calls yet. The next implementation should be a small feature-only prototype of an explicit phone duration/pause-conditioned renderer that receives the full source phone trajectory and local context, uses a length regulator, and is first evaluated with synthetic known-warp plus speaker-disjoint audio/content gates.

Only after that prototype passes should we consider generating an independent Qwen training pool and adding natural prosody weak supervision.

## Observations

- [insight] Established duration-controlled speech systems supervise explicit duration/pitch/energy variables rather than relying on a residual correction over an already warped trajectory. #duration #prosody
- [insight] Phone content duration and boundary pause duration should be separate controls in spontaneous speech. #pause #ramc
- [problem] The desired hybrid waveform is unavailable as a direct target; full natural WavLM matching would erase the TTS acoustic anchor. #supervision
- [decision] Replace the current residual adapter proposal with an explicit length-regulated conditional acoustic renderer; keep Qwen test cohorts untouched. #architecture
- [result] Literature review completed on 2026-08-15. #research

## Sources

- https://arxiv.org/abs/2006.04558
- https://proceedings.mlr.press/v97/qian19c.html
- https://arxiv.org/html/2604.21164v2
- https://arxiv.org/html/2506.13295v2

## Relations

- relates_to [[第一阶段：duration/time-warp temporal adapter 机制验证]]
- relates_to [[AISHELL-1 duration/time-warp temporal adapter phase 1]]
- relates_to [[Train learned phone trajectory adapter]]
- relates_to [[VC-inspired trainable speaker-robust rhythm transfer plan]]
- relates_to [[TTS Feature Alignment and Duration Model Research]]
- relates_to [[AISHELL-1 n25 Qwen cloud MFA-linear 2026-08-14]]
- relates_to [[RAMC n25 Qwen cloud MFA-linear 2026-08-15]]
