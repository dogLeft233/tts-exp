---
title: AISHELL-1 n25 Qwen cloud MFA-linear 2026-08-14
type: note
permalink: tts-exp/experiments/aishell-1-n25-qwen-cloud-mfa-linear-2026-08-14
status: complete
date: '2026-08-14'
cohort: aishell1_n25
provider: qwen3-tts-vc-2026-01-22
mfa_version: 3.4.1
sample_count: 25
speakers:
- S0765
- S0901
- S0906
- S0912
- S0913
s0770_excluded: true
tags:
- aishell1
- qwen
- mfa-linear
- n25
---

# AISHELL-1 n25 Qwen cloud MFA-linear

## Context

This thread evaluated the first fixed n=25 cloud-TTS input to the MFA-linear pipeline. The purpose was feasibility validation: retain the Qwen TTS acoustic trajectory while placing it on the natural utterance clock, without training a new renderer.

## Frozen protocol

- Cohort: 25 paired AISHELL-1 utterances, five each from `S0765`, `S0901`, `S0906`, `S0912`, and `S0913`.
- `S0770` was excluded from selection, tuning, and evaluation.
- Qwen DashScope model: `qwen3-tts-vc-2026-01-22`, with the paired natural recording as the reference for each sample.
- MFA: 3.4.1, `mandarin_china_mfa` dictionary, `mandarin_mfa` acoustic model; 50/50 natural and Qwen TextGrids completed.
- Token preparation preserves `raw_token` and removes Mandarin IPA tone marks only for matching. Empty, `sil`, `sp`, and `<sil>` are silence; `spn` is unknown speech and is never silently converted to silence. This cohort had zero unknown speech tokens.
- MFA-linear uses frozen WavLM-Large layer 6 from pinned `bshall/knn-vc` revision `c616845c4e309e24d5927f15adbdf277a3d65358`, followed by the frozen prematched HiFi-GAN, with no loudness normalization.

## Outcome

All 25 Qwen TTS files and all 25 MFA-linear files were generated successfully. The MFA-linear outputs are finite mono 16 kHz files with exact natural sample lengths. No sample reached absolute amplitude 0.999; the maximum peak was 0.6165. The targeted MFA and retrieval regression tests passed: 39 tests.

The basic audio gate is therefore successful, but the alignment gate is not yet clean enough for the downstream matrix:

- Mean frame coverage: 0.9738; minimum: 0.8562.
- Total fallback frames: 174.
- All unmatched spans inspected were silence intervals; speech phone labels matched in order.
- The fallback frames arise from natural/Qwen pause-count differences. The current implementation uses a global-relative TTS position when a natural token is unmatched, which is potentially unsafe for an unmatched natural pause because it can inject non-silence TTS features into that interval.
- Mean matched speech-phone duration MAE: 21.85 ms; mean relative absolute gap: 0.2086.
- MFA token silence duration averaged approximately 1.20 s per natural utterance versus 0.43 s per Qwen utterance, so Qwen voice cloning still does not preserve natural pauses. Speaker-level speech-phone MAE means were 14.7 ms (S0765), 23.5 ms (S0901), 19.9 ms (S0906), 23.4 ms (S0912), and 27.8 ms (S0913); these are descriptive only and do not explain speaker-level downstream outcomes by themselves.

## Artifacts

- Qwen TTS metadata: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/qwen_cloud_tts_n25_20260814/tts_meta.json`
- Clean MFA output and tokens: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/qwen_cloud_mfa3_n25_20260814/`
- MFA-linear audio and summary: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/qwen_cloud_mfa_linear_n25_20260814/`
- Audio QC: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/qwen_cloud_mfa_linear_n25_20260814/qc.json`
- Reproducible token parser: `scripts/prepare_aishell1_qwen_cloud_mfa_tokens.py`

## Downstream Wav2Lip + SyncNet

The full three-arm matrix completed with 75/75 finite scores using one fixed `S0765` face (`natural_raw/1.mp4`). The run used a dedicated Qwen provenance manifest and separate per-arm, per-speaker, per-sample Wav2Lip work directories and SyncNet data directories.

- Output: `runs/aishell1_mfa_linear_n25_resample_poly_20260815/qwen_cloud_mfa_linear_n25_wav2lip_syncnet_qwenprov_20260815/`
- Natural raw: mean Sync-C 5.9976, mean Sync-D 7.2361
- Qwen raw TTS: mean Sync-C 6.5819, mean Sync-D 7.2422
- Qwen MFA-linear: mean Sync-C 6.5691, mean Sync-D 7.1016
- MFA-linear versus natural: mean ΔC +0.5714, mean ΔD −0.1345; C improved on 17/25 and D improved on 16/25, with joint improvement on 13/25.
- MFA-linear versus Qwen raw: mean ΔC −0.0128, mean ΔD −0.1406; C improved on 10/25 and D improved on 15/25, with joint improvement on 9/25. The paired tests were not significant for this comparison (C p=0.928, D p=0.128).
- Speaker pattern versus Qwen raw: MFA-linear was positive on both directions for S0765 (ΔC +0.5342, ΔD −0.3722) and S0906 (+0.4280, −0.3434), roughly neutral-to-mixed for S0913 (+0.0908, −0.1330), and negative for S0901 (−0.4852, −0.0856) and S0912 (−0.6320, +0.2312).

These are fixed-face exploratory SyncNet associations, not speaker-matched generalization or causal mouth-motion evidence. The result does not establish Qwen MFA-linear as superior to Qwen raw TTS on Sync-C; its main measurable advantage in this run is lower average Sync-D, while the earlier silence-mapping caveat remains unresolved.

## Comparison with the local TTS experiment

The comparable local n25 resample-poly matrix is `runs/rhythm_timing/20260813_syncnet_aishell1_n25_resample_poly_face1/summary.json`; it uses the same 25 paired utterances, the same fixed face hash, and the same Wav2Lip/SyncNet checkpoint hashes.

- Local TTS MFA-linear: mean Sync-C 5.9036 and Sync-D 7.3408; versus its natural arm, ΔC −0.0709 and ΔD +0.0694. It fails the aggregate rule of being better than natural on both metrics.
- Qwen MFA-linear: mean Sync-C 6.5691 and Sync-D 7.1016; versus its natural arm, ΔC +0.5714 and ΔD −0.1345. It passes the aggregate rule on both metrics.
- The Qwen MFA-linear minus local MFA-linear difference is +0.6655 Sync-C and −0.2392 Sync-D; Qwen is higher on C for 21/25 samples and lower on D for 15/25.
- Speaker-level qualification is not universal: Qwen passes both directions for S0765, S0906, and S0913, but not for S0901 or S0912. The local experiment similarly had persistent failures for S0901 and S0912.

Thus Qwen is a clear aggregate improvement over the local TTS experiment and turns the first n25 feasibility gate positive, but it does not solve the speaker-dependent failure or prove superiority to Qwen raw TTS on Sync-C. The fixed-face limitation and unresolved silence mapping remain.

## Next data-domain experiment
The next controlled domain test used the existing MagicData-RAMC telephone-conversation n25 rather than downloading more data. Qwen cloud TTS was generated with each paired natural recording as its reference, followed by clean MFA-3.4.1 and the same frozen MFA-linear pipeline.

The RAMC Qwen run completed 25/25 TTS, 50/50 TextGrids, 25/25 MFA-linear files, and 75/75 Wav2Lip/SyncNet cells. Qwen raw TTS scored Sync-C 7.02596 and Sync-D 7.27744; Qwen MFA-linear scored 6.67664 and 7.37988; natural scored 6.27576 and 7.35104. Qwen MFA-linear versus natural was ΔC +0.40088 (p=0.00884) but ΔD +0.02884 (p=0.7986), so it did not pass the strict two-metric RAMC gate. It was also worse than Qwen raw by ΔC −0.34932 and ΔD +0.10244. Compared with the old local RAMC MFA-linear result, Qwen improved Sync-C by +1.63752 but worsened Sync-D by +0.09016.

The result answers the diagnostic question: stronger TTS repairs much of the local-TTS Sync-C collapse, but spontaneous telephone pause structure and MFA-linear projection remain unresolved. RAMC alignment was poorer than AISHELL-1 (coverage 0.9176 versus 0.9738; fallback 883 versus 174), and six paired `spn` regions were explicitly retained as unknown speech. These findings are recorded in [[RAMC n25 Qwen cloud MFA-linear 2026-08-15]].

Do not blindly expand RAMC or start multilingual comparison yet. First inspect silence mismatches, `spn` spans, phone boundaries, fallback behavior, and waveform artifacts, then test a silence-aware/spontaneous-speech-aware mapping variant on the frozen n25 cohort.
## Decisions
- The AISHELL-1 n25 Qwen cloud matrix is complete and passes the aggregate two-metric natural-reference gate, but remains a fixed-face exploratory result and does not establish speaker-independent generalization.
- The existing RAMC telephone n25 Qwen rerun is complete and is documented in [[RAMC n25 Qwen cloud MFA-linear 2026-08-15]]. It substantially improves Sync-C over the old local-TTS MFA-linear run but fails the strict two-metric gate and remains worse than Qwen raw TTS.
- Do not download more spontaneous Chinese data or start multilingual comparison yet. First isolate silence-aware, unknown-speech-aware, and boundary/projection failure modes on the frozen RAMC cohort.
- All follow-up variants must use new output directories and retain source, token, model, video, and score hashes; existing results must not be overwritten.
- Do not interpret fixed-face SyncNet changes as causal mouth-motion evidence.
## Observations

- [decision] The first Qwen cloud n25 MFA-linear feasibility run is complete, but downstream evaluation is held behind a silence-aware alignment check. #mfa-linear
- [insight] Speech-phone label matching is substantially better than the aggregate coverage number suggests; the observed unmatched spans are pauses, not pronunciation labels. #alignment
- [problem] Qwen TTS has materially shorter silence durations than natural speech, and global-relative fallback is not a safe default for unmatched natural silence. #prosody
- [learning] Finite/length/clipping checks can pass while the internal pause mapping remains acoustically ambiguous. #audio-qc

## Relations

- relates_to [[AISHELL-1]]
- relates_to [[MFA-linear]]
- relates_to [[Qwen TTS]]
