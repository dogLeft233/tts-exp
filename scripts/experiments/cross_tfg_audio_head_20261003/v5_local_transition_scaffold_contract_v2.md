# V5 local-transition scaffold V2 contract

Status: preparation only. The V2 source is not a frozen candidate, was not run on audio, and has no selected coefficients. It does not read development or confirmation audio, call an encoder, render video, or score synchronization. The fit40 family and its coefficients remain unfrozen pending the shared-profile development readout.

## V2 source and provenance

- V2 source: `v5_local_transition_transform_scaffold_v2.py` in this directory, SHA-256 `cf86b95b65b2c4a423082e0f717713983915cdc0674ee7ddb4fba2534e9906e0`.
- V2 was derived from the V1 source bytes present in this workspace: `v5_local_transition_transform_scaffold_v1.py`, SHA-256 `a7c2286ac3fdab82a75b09ca5b0809073ea33ff19fc88d703d5ea3f11f677020`.
- A prior checkpoint records V1 SHA-256 `b34450397c5fa18b8cd4f820ae095b356499ecabb96b3c6a814b676642e63cbd`. Those bytes were not recovered here. This is a preparation-provenance discrepancy only; the `a7c...` V1 source is the observed parent of V2, and no V1 effects were evaluated in this preparation.
- V1 is preserved byte-for-byte. Do not substitute the unrecovered `b344...` identity for the observed source.

## Transform contract

`CandidateSpecV2` has no parameter defaults. A future caller must provide candidate ID, gate kind, contrast polarity and strength, flux quantiles (explicit nulls for ungated), mask bound, smoothing scales, RMS/cap choices, and the 16 kHz / 512-point / 128-sample clock. The helper checks the pinned `audio_io.py`, `baselines.py`, and shared-profile smoothing source before importing those operations.

The declared gate kinds are `ungated`, `positive_flux`, and `negative_flux`. Positive and negative flux are normalized increases and decreases in linear STFT magnitude. Their soft quantile gates are acoustic transition descriptors; they are not VAD, phoneme labels, or speech-onset annotations. Diagnostics report gate coverage and mask dose so different activity is visible.

For the positive/negative × accent/flattening four-cell bank, `common_kappa_for_four_transition_cells_v2` requires all four cells, the same original waveform and source clock, and matching transform settings. It computes one utterance-level κ from the maximum absolute raw mask across the bank, and records both the bank hash and each cell mask hash. The caller must pass `additional_prepared` explicitly, even when it is empty. To compare an ungated reference under the same κ, include that named reference there; it then contributes to the same maximum. Applying κ to a mask that is not registered in the bank fails closed. No cell can silently choose its own κ.

V2 begins with the original mono float32 PCM, constructs its source STFT at fixed source-clock indices, smooths log magnitude, forms per-frame frequency-centered contrast, and multiplies that contrast by the declared polarity, strength, and gate. Applying a mask multiplies the complex source-STFT coefficients by positive real magnitude gains. The source phase is retained at that coefficient-construction step. ISTFT reconstruction and later reanalysis may change phase; the contract does not claim waveform phase identity or phase preservation after reconstruction. The V2 mask arms do not apply the r090 compressor before STFT.

This phase wording supersedes the looser phase-preservation wording in the earlier V1 slate: the guarantee is only at complex-STFT coefficient construction. The output keeps the original 16 kHz sample count and source timing. Optional fixed RMS and peak handling are separate explicit stages. Diagnostics retain raw log magnitude and contrast, positive and negative flux, gate values, raw and projected masks, STFT frame centers, κ and its bank/cell hashes, RMS/peak values, and 40 ms physical input/output frame powers with the excluded tail count. Arithmetic zero-mean dB masks and equal RMS do not imply equal physical frame power.

## Proposed six-slot budget (not frozen)

The current proposal has six candidate/ablation slots plus identity and fixed-RMS controls: the independent r090 local-envelope control; one ungated compressor-off accent; and four positive/negative flux × accent/flattening cells. This remains a proposal, not a frozen roster or set of coefficients. `localcompress_r090_control_attempt01.py` is separately owned and pinned at SHA-256 `4c24c5d15d280186da2be4c07931a72625cf6d3edc24bb3de4211206aa81fa09`; its prepared config SHA-256 is `2c9f63d74bc204ff46c766b5cad18dc9655cf79a228f93b76cc0a84533913e62`. That control uses fixed RMS `0.0376838172675746` → centered 32 ms envelope compression (ratio 0.90, gain bounds −6/+6 dB, floor `1e-5`) → the same fixed RMS. It has no peak cap and does not resample or alter length. The compressor changes the working waveform before its own spectral processing, so it is separate from the compressor-off, original-STFT mask arms.

## Pinned implementation inputs

| Source | SHA-256 |
|---|---|
| `scripts/audio_enhancement/audio_io.py` | `7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004` |
| `scripts/audio_enhancement/baselines.py` | `7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2` |
| `scripts/audio_enhancement/shared_frequency_profile_v1.py` | `fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253` |
| `scripts/audio_enhancement/localcompress_r090_control_attempt01.py` | `4c24c5d15d280186da2be4c07931a72625cf6d3edc24bb3de4211206aa81fa09` |

The separate historical fit subset is defined by ADR013 and its independently checked 40-member raw/audio receipt. A future plan must retain its original 320 canonical slots, use only the declared endpoint subset where qualified, and keep all missing, duplicate, failed, and non-estimable slots visible. This contract does not authorize a fit run or coefficient selection.
