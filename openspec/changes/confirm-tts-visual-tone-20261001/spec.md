# Confirm tone-aware visual phone endpoint

## Scope and freeze boundary

This change versions the independent MediaPipe/MFA visual phone analysis for Mandarin tone-marked phone labels. It uses the already qualified natural CAL TextGrids and CAL baseline tracks only. It does not read EVAL TextGrids or EVAL endpoint values while freezing classification, latency, thresholds, or diagnostics. The 2026-09-30 visual source snapshot and qualification bundle remain preserved and unchanged.

The new code is a wrapper around the frozen visual-analysis core. Only visual phone-class assignment changes. WORLD/physical phone identity and MFA phone labels retain tone, aspiration, length, and other phone distinctions.

## Visual tone mapping

For visual classification only, remove exactly the five Unicode modifier tone bars found in the CAL natural-phone inventory before applying the frozen normalizer and class allowlists:

| Character | Code point | Unicode name |
|---|---|---|
| ˥ | U+02E5 | MODIFIER LETTER EXTRA-HIGH TONE BAR |
| ˦ | U+02E6 | MODIFIER LETTER HIGH TONE BAR |
| ˧ | U+02E7 | MODIFIER LETTER MID TONE BAR |
| ˨ | U+02E8 | MODIFIER LETTER LOW TONE BAR |
| ˩ | U+02E9 | MODIFIER LETTER EXTRA-LOW TONE BAR |

The CAL mapping table preserves every raw phone label and its occurrence count, removed tone marks, visual base label, and visual class. It is sealed against the 20 natural CAL TextGrid hashes and the old qualification/alignment artifacts. No other Unicode diacritic is removed by this adapter; existing rules in the frozen visual core remain unchanged.

## CAL-only eligibility and coverage

Freeze the global latency from baseline CAL tracks and qualified natural CAL phone intervals using the existing rule: median signed extrema-event minus MFA midpoint offset. Retain the original rematched event controls, including both known shifts, static-repeat control, and shuffled-track control. These controls describe event-timing calibration only; they do not gate the fixed-latency phone-class endpoint. The event-timing metric remains uncalibrated unless the original event controls pass.

The fixed detected-extrema-index timestamp transform is a separate implementation QC: shift timestamps by exactly +5 and -5 frames while keeping detected indices and aperture samples unchanged. It must report unchanged event indices, unchanged aperture multiset, and the expected timestamp deltas. This QC does not calibrate the rematching event metric.

The phone-class calibration gate requires the fixed CAL latency support, 20 natural CAL rows across 10 speakers with two clips each, valid baseline tracks, and the frozen minimum token count for each class separately. The artifact must report (a) per-class clip coverage, (b) clips with both phone classes jointly available, and (c) speakers with both clips jointly available. Per-class 80% coverage must never be described as 80% joint endpoint coverage. Joint coverage is reported descriptively and does not silently remove clips from the EVAL primary denominator.

CAL diagnostics also report class separation at the frozen latency, at zero latency, on a static repeated-aperture track, and under 100 deterministic within-clip shuffles of valid aperture samples. These are descriptive diagnostics with no new positivity threshold. They retain all intended CAL clips and speakers in their denominator fields.

## EVAL endpoint and support

The EVAL primary remains the per-clip mean open-vowel aperture minus bilabial aperture at the CAL-frozen latency, using full recipient clocks and common valid frame support across all eight visual arms. Each clip contributes equal weight within its speaker; speakers contribute equal weight. Preserve the full intended 80-clip/40-speaker denominator. Missing class support or missing arm tracks remain missingness in that denominator; a phone-qualified complete paired-speaker subset may be reported only as a predeclared secondary support analysis, selected from natural TextGrid phone availability and feature-track availability without inspecting aperture magnitudes or endpoint effects. Also report the same phone-separation contrasts at zero latency as a non-gating sensitivity analysis; it cannot replace the CAL-frozen-latency result.

Add a secondary token-ranking diagnostic. For each clip, average aperture within each supported natural phone token, then compute Mann–Whitney AUC for open-vowel token means versus bilabial token means; ties count 0.5. Use the same natural phone tokens that have valid common-clock observations in all eight arms. Apply the CAL-frozen per-class minimum token count. Keep missing/low-token clips in the intended denominators and expose their support. Average random1/random2/random3 AUC values within clip, then average the two clip values equally within speaker and speakers equally overall. Report `orth_plus - baseline` and `orth_plus - mean(random)` as secondary contrasts with the ordinary speaker-level interval procedure. AUC is invariant to positive affine rescaling of all token means, but it remains a class-ranking measure rather than a synchrony or perceived-quality measure.

Continue reporting class means/separation, overall aperture, absolute aperture velocity, mouth width, and pose proxies separately so changes in distribution, movement magnitude, and pose remain visible. Event timing remains exploratory; no phone endpoint result implies a timing or perceptual-quality improvement.

## Provenance requirements

Freeze one JSON record binding this spec hash, the new adapter source hash, frozen visual-core source hash, old MFA snapshot/qualification/alignment hashes, CAL feature and TextGrid hashes, raw-to-visual mapping hash, new CAL calibration output hash, and the protocol identifier. The calibration must record `evaluation_textgrids_read=false`. Preserve the previous tone-only diagnostic output and original frozen 2026-09-30 source and qualification artifacts unchanged.
