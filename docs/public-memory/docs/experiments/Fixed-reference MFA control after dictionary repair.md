---
title: Fixed-reference MFA control after dictionary repair
type: report
permalink: tts-exp/docs/experiments/fixed-reference-mfa-control-after-dictionary-repair
alignment_gate: passed_with_local_spn
date: '2026-08-14'
status: active
sample_count: 15
tags:
- mfa
- fixed-reference
- mfa-linear
- provenance
- syncnet
---

# Fixed-reference MFA control after dictionary repair

## Result

2026-08-14: Other agent repaired local MFA setup. Root cause was MFA 2.2.17 reading an incompatible MFA 3.x `mandarin_mfa.dict`, plus unsegmented Chinese `.lab` text. Repaired workflow uses the compatible 2.x dictionary and character-segmented transcripts.

Fresh same-environment alignment was run for 15 samples (`S0765/S0901/S0912`, five each), separately for natural and fixed-reference TTS. Both sides used the same MFA binary, dictionary, acoustic model, `--single_speaker --clean --num_jobs 4`, and character segmentation.

Tokens: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/fixed_reference_mfa_sameenv_20260814/tokens.json`.

Results:

- natural and fixed TTS: 15/15 each aligned; no command failures;
- no whole-utterance `spn` collapse;
- local `spn` remains on some words and is recorded, not converted to valid phones;
- same-environment token matching restored high coverage.

Fixed-reference MFA-linear generated from these same-environment tokens:

`runs/aishell1_mfa_linear_n25_resample_poly_20260814/fixed_reference_mfa_linear_sameenv_20260814/`

- 15/15 outputs;
- 0 failures;
- exact natural sample length for all outputs;
- mean coverage: S0765 `0.9969`, S0901 `0.9385`, S0912 `0.9883`;
- mean fallback frames: S0765 `0.8`, S0901 `21.8`, S0912 `3.2`;
- peak maxima: S0765 `0.0866`, S0901 `0.0809`, S0912 `0.0787`.

The earlier fixed-reference MFA-linear output using frozen historical natural tokens is invalid as a control because historical tokens used a different MFA label/configuration. It is preserved but not used.

## Decision

Fixed-reference control now passes the audio/provenance gate sufficiently for exploratory downstream Wav2Lip + SyncNet. Downstream results must remain single fixed-face evidence, not speaker-matched generalization or causal proof.
