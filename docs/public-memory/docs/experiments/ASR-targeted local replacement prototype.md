---
title: ASR-targeted local replacement prototype
type: report
permalink: tts-exp/docs/experiments/asr-targeted-local-replacement-prototype
tags:
- asr
- replacement
- syncnet
- prototype
- proxy
---

# ASR-targeted local replacement prototype

## Context

This was a deliberately small, fit-only prototype testing whether replacing natural-ASR error regions with aligned TTS waveform segments changes frozen Wav2Lip output under strict untouched-natural-audio SyncNet scoring. It reuses the discovery cohort and is not fresh confirmation or a population-level claim. A separate candidate-driven proxy was subsequently evaluated to quantify candidate-audio/video self-consistency without changing the strict endpoint.

## Frozen design

- Use only the ordered 24 fit-only samples and 24 distinct source groups from the locked ASR experiment.
- Rebuild target/control selection from an explicit score-free projection of `03_asr`: identifiers, transcripts, edit operations, forced-reference timing, audio identity, and floor/ceil sample bounds.
- Target blocks require natural substitution/deletion, contiguous reference words, finite natural/TTS forced spans, no mixed insertion, and a complete TTS operation interval containing only equal operations.
- Use two distinct deterministic within-sample correct-region control assignments. Their contrasts are descriptive comparators, not randomized causal estimates; the natural-baseline gains are the conditional intervention estimands.
- Patch with one SciPy 1.18.0 `scipy.signal.resample`, exact natural destination length, fixed in-segment raised-cosine ramp, and no normalization/global timing transform.
- Render exactly `natural`, `asr_targeted`, `target_control_0`, and `target_control_1`; the strict endpoint scores every render after muxing untouched canonical natural PCM.
- The candidate-driven proxy muxes the same rendered videos with their candidate driver audio and scores those media separately; it is never substituted for the strict endpoint.
- Require identical selected SyncNet track metadata/support across paired conditions; otherwise the sample cannot contribute to paired inference.
- Bootstrap distinct source-group rows with 10,000 `PCG64(20260901)` draws and NumPy linear quantiles; require at least 12 complete source groups.

## Final result

Run directory: `runs/lrs3_asr_targeted_local_replacement_prototype_20260901/`.

Metadata-only lock passed for 24 samples/source groups with sealed split access false. Strict reconstruction found 71 natural error blocks, 52 reference-mappable blocks, and 21 TTS-clean target blocks in 13 samples/source groups. The lower count versus the initial 24-block/15-sample readiness scan is intentional: mixed insertion blocks and TTS donor intervals containing any insertion are rejected.

All 52 strict PCM cells passed exact-length, finite, mono 16 kHz PCM16 validation, including byte-identical natural identity after decode. All 52 Wav2Lip renders, strict replacement muxes, and strict SyncNet full-matrix cells passed. All four selected tracks matched within every sample, SyncNet parity passed, plots were complete, and engineering is `GO`.

The strict untouched-natural-audio replacement decision is `NO_PROTOTYPE_SUPPORT` with 13 complete source groups. The paired results were:

- `baseline_C_gain`: median `-0.0341`, bootstrap 95% CI `[-0.2092, -0.0054]`;
- `baseline_D_gain`: median `-0.0501`, bootstrap 95% CI `[-0.2184, -0.0016]`;
- descriptive `control_C_adv`: median `-0.0218`, bootstrap 95% CI `[-0.0826, 0.0337]`;
- descriptive `control_D_adv`: median `-0.0378`, bootstrap 95% CI `[-0.0915, 0.0308]`.

The natural-baseline gains are positive for 3/13 samples for Sync-C and 3/13 for Sync-D; 4/13 improve on at least one of the two, and 2/13 improve on both. The natural-baseline intervals are entirely below zero in this frozen prototype, while the control contrasts cross zero. The effect is small in absolute score units but is not equivalent to an unchanged result under the predeclared paired bootstrap; it is a mild negative result rather than a large collapse.

## Candidate-driven proxy result

The separate proxy run is stored under `runs/lrs3_asr_targeted_local_replacement_prototype_20260901/06_candidate_proxy/`. It completed 52/52 proxy media contracts, 52/52 SyncNet parity checks, 52/52 distance hash checks, and 13/13 paired track checks, with sealed split access false.

- `proxy_C_gain`: median `-0.0210`, bootstrap 95% CI `[-0.0885, 0.0631]`, positive in 5/13 source groups;
- `proxy_D_gain`: median `-0.0072`, bootstrap 95% CI `[-0.0463, 0.0440]`, positive in 6/13 source groups;
- `proxy_control_C_adv`: median `-0.0012`, bootstrap 95% CI `[-0.0823, 0.0963]`;
- `proxy_control_D_adv`: median `0.0053`, bootstrap 95% CI `[-0.0119, 0.0283]`.

The proxy therefore shows no reliable positive candidate-audio/video gain either. Its result is `COMPLETE_DESCRIPTIVE_ONLY` and `NOT_A_REPLACEMENT_DECISION`; the strict science decision remains `NO_PROTOTYPE_SUPPORT`. A positive proxy result in a future experiment would still indicate only candidate-pair co-adaptation unless it survives untouched-natural-audio replacement scoring.

## Observations

- [decision] Preserve strict untouched-natural-audio replacement as the only replacement-safe endpoint; candidate-driver self-consistency is a separate descriptive proxy.
- [decision] Make source_group the sole inferential unit and enforce one retained sample per source group.
- [decision] Treat ASR-targeted versus ASR-correct controls as descriptive rather than causal targeting evidence.
- [decision] Do not promote this exact waveform intervention to learned prosody/audio-head work based on this run alone; any follow-up needs a new, explicitly bounded spec and fresh fit-only confirmation.
- [insight] Complete TTS operation-interval cleanliness is required; checking mapped equal words alone can admit inserted donor errors.
- [insight] The local waveform patch produced neither a reliable candidate-pair proxy gain nor a strict replacement gain in this prototype.
- [insight] The strict paired effect is mildly negative and statistically separated from zero under the frozen bootstrap, whereas the proxy effects are near zero with intervals crossing zero.
- [pattern] Track mismatch must block paired primary inference because SyncNet crops scorer input by selected track.

## Relations

- implements [[ASR sync error correlation experiment]]
- relates_to [[Wav2Lip replacement primary objective]]
