---
title: RAMC n25 Qwen cloud MFA-linear 2026-08-15
type: note
permalink: tts-exp/experiments/ramc-n25-qwen-cloud-mfa-linear-2026-08-15
cohort: ramc_n25
date: '2026-08-15'
mfa_version: 3.4.1
provider: qwen3-tts-vc-2026-01-22
sample_count: 25
speaker_count: 25
status: complete
tags:
- ramc
- qwen
- mfa-linear
- n25
- syncnet
---

# RAMC n25 Qwen cloud MFA-linear 2026-08-15

## Context

This was a controlled rerun of the existing MagicData-RAMC telephone-conversation pilot. The question was whether the large Sync-C failure of the earlier local-TTS MFA-linear run was mainly caused by weak TTS acoustic trajectories, or whether spontaneous telephone speech and pause structure remain a blocker even with a strong cloud TTS model.

## Frozen protocol

- Cohort: the already prepared RAMC n25 set at `data/ramc_alimeeting_pilot50/ramc25_utterances/`.
- Exactly 25 distinct speakers, one Mandarin turn per conversation; no new download and no sample reselection.
- Qwen DashScope model: `qwen3-tts-vc-2026-01-22`; each natural recording was used as that sample's reference.
- MFA: clean MFA 3.4.1, `mandarin_china_mfa` dictionary, `mandarin_mfa` acoustic model.
- MFA-linear: frozen WavLM-Large layer 6 and pinned prematched HiFi-GAN, 16 kHz, exact natural output length, no loudness normalization.
- `spn` is retained as unknown speech and never converted to silence. Six paired unknown-speech tokens were accepted only through an explicit override and remain separately reported.
- Wav2Lip/SyncNet used the fixed existing S0765 face. The scores are exploratory fixed-face associations, not speaker-matched generalization or causal mouth-motion evidence.

## Artifacts and completion

- Qwen TTS metadata: `runs/ramc_alimeeting_pilot50/ramc25_qwen_cloud_tts_20260815/tts_meta.json`
- Clean MFA and tokens: `runs/ramc_alimeeting_pilot50/ramc25_qwen_mfa3_20260815/`
- MFA-linear audio/QC: `runs/ramc_alimeeting_pilot50/ramc25_qwen_mfa_linear_20260815/`
- Wav2Lip/SyncNet: `runs/ramc_alimeeting_pilot50/ramc25_qwen_wav2lip_syncnet_20260815/`
- TTS generation: 25/25, all 25 speakers, no failures.
- MFA TextGrids: 50/50.
- MFA-linear: 25/25, finite mono 16 kHz, exact natural lengths, no clipping at 0.999.
- The Wav2Lip/SyncNet matrix: 75/75 cells complete.

## Alignment QC

- Mean coverage: 0.917594; minimum: 0.783410.
- Total fallback frames: 883; maximum per sample: 94.
- Mean matched speech-phone duration MAE: 27.705 ms.
- Mean relative speech-phone duration gap: 0.413541.
- Natural MFA silence mean: 1.66716 s; Qwen MFA silence mean: 1.27200 s.
- Six unknown-speech tokens occur in paired natural/Qwen regions for samples 06, 12, and 15. They are not silence and should not be interpreted as ordinary clean phone alignment.

The RAMC alignment is substantially less clean than AISHELL-1 Qwen n25 (coverage 0.9738 versus 0.9176; fallback 174 versus 883). Coverage alone is not treated as a causal explanation for the SyncNet result, but the pause and unknown-speech mismatch is a concrete reason to prioritize silence-aware/spontaneous-speech-aware mapping before scaling the cohort.

## Wav2Lip + SyncNet results

Directions: Sync-C higher is better; Sync-D lower is better.

| arm | mean Sync-C | mean Sync-D |
|---|---:|---:|
| natural raw | 6.27576 | 7.35104 |
| Qwen raw TTS | 7.02596 | 7.27744 |
| Qwen MFA-linear | 6.67664 | 7.37988 |

### Qwen raw TTS versus natural

- ΔSync-C: +0.75020; 23/25 improved; paired p=5.42e-06; bootstrap 95% CI [+0.50752, +0.99844].
- ΔSync-D: −0.07360; 15/25 improved; paired p=0.4211; bootstrap 95% CI [−0.24440, +0.09836].
- Joint improvement on both metrics: 15/25.

### Qwen MFA-linear versus natural

- ΔSync-C: +0.40088; 17/25 improved; paired p=0.00884; bootstrap 95% CI [+0.13492, +0.66768].
- ΔSync-D: +0.02884, meaning slightly worse because lower is better; 12/25 improved; paired p=0.7986; bootstrap 95% CI [−0.18052, +0.24720].
- Joint improvement on both metrics: 11/25.
- It therefore does not pass the strict two-metric RAMC gate, despite passing the Sync-C direction.

### Qwen MFA-linear versus Qwen raw TTS

- ΔSync-C: −0.34932; only 7/25 improved; paired p=0.00243.
- ΔSync-D: +0.10244, also worse; 11/25 improved; paired p=0.2092.
- Joint improvement: 7/25.

The raw Qwen waveform is the strongest of these three RAMC arms on both mean metrics. MFA-linear preserves a positive Sync-C advantage over natural, but the natural-clock projection loses substantial Sync-C relative to the Qwen raw trajectory and does not improve Sync-D.

## Comparison with the old local-TTS RAMC matrix

The old local result is `runs/ramc_alimeeting_pilot50/ramc25_wav2lip_3arm/analysis.json`.

- Local MFA-linear: mean Sync-C 5.03912, Sync-D 7.28972.
- Qwen MFA-linear minus local MFA-linear: +1.63752 Sync-C and +0.09016 Sync-D. Qwen Sync-C is higher on 21/25; Sync-D is lower on 11/25.
- Old local MFA-linear versus natural was ΔC −1.2228 and ΔD −0.08856. Qwen changes ΔC to +0.40088, a +1.62368 swing, but changes ΔD to +0.02884, a +0.11740 worsening.
- Qwen raw TTS also exceeds local raw TTS by +0.63864 Sync-C, while Sync-D is nearly unchanged (+0.00132).

Thus the stronger Qwen acoustic trajectory clearly repairs the local-TTS Sync-C collapse in the telephone domain. It does not establish that MFA-linear itself solves the domain: the raw Qwen arm remains better, and the Sync-D/timing side remains unresolved.

## Decision and next step

Do not blindly expand RAMC and do not start multilingual comparison yet. The controlled experiment answered the immediate question: TTS quality explains a large part of the previous RAMC Sync-C failure, but spontaneous telephone pause/unknown-speech alignment and natural-clock projection remain active failure modes.

The next experiment should be audio-only and non-destructive: inspect Qwen raw versus MFA-linear around natural/Qwen silence mismatches, `spn` spans, phone boundaries, fallback frames, and any repeated or transient waveform artifacts. Then test a silence-aware/spontaneous-speech-aware mapping variant on the frozen n25 cohort, with a new output directory and the same three-arm evaluation. Scaling data or comparing languages should wait until this mechanism is isolated, because those changes would confound corpus domain, aligner resources, TTS quality, and face evaluation.

## Observations

- [decision] Qwen RAMC n25 is complete; retain all old local results and do not overwrite them. #ramc #mfa-linear
- [insight] Stronger TTS substantially rescues Sync-C, but MFA-linear is still worse than raw Qwen on both mean SyncNet metrics. #qwen #tts
- [problem] The strict two-metric natural-reference gate remains unmet because MFA-linear Sync-D is slightly worse than natural. #syncnet
- [problem] RAMC has poorer phone/pause coverage and six paired `spn` regions, making it unsuitable for a blind scale-up before mapping diagnostics. #alignment

## Relations

- relates_to [[AISHELL-1 n25 Qwen cloud MFA-linear 2026-08-14]]
- relates_to [[MagicData-RAMC]]
- relates_to [[MFA-linear]]
- relates_to [[Qwen TTS]]
