---
title: mvp-vs-natural-forensics-51-100
type: note
permalink: tts-exp/docs/experiments/mvp-vs-natural-forensics-51-100
---

# MVP vs natural audio forensics (samples 51–100)

## Status

- corpus: complete
- natural_alignment: complete
- mvp_alignment: unavailable
- phone_pause_prosody: unavailable
- score_ledger: unavailable
- correlations: unavailable
- overall: partial

## Paired score result

- SyncNet score ledger was unavailable; score deltas and score correlations were not computed.

## Interpretation

The analysis is observational. A feature difference or correlation with SyncNet does not establish causality and cannot by itself separate waveform effects from Ditto, preprocessing, tracking, or SyncNet interactions.

MVP alignment-derived timing and pause features were unavailable; no uniform or source-TTS fallback was used.

## Highest FDR-ranked associations

- No non-constant feature association was estimable.

## Evidence boundaries

The historical paired score pattern rules out a simple claim that MVP failed because of a larger global AV offset: its mean offset improved while Sync-C declined. Timing, pause structure, local prosody, phonetic clarity, and coupled acoustic transformations remain hypotheses until independent MVP MFA and a complete per-sample score ledger are available.