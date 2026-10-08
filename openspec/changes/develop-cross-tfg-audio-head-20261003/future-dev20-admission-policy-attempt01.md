# Future Dev20 admission policy — attempt 01

Frozen by ROOT before reading any future Dev20 effects. This policy applies only to the new S0102–S0111 development cohort; it does not alter the original Dev or any confirmation policy. The method/checkpoint remains unbound until the actual Fit40 DirectD0 terminal qualifies. No future Dev render or score has run.

## Design and denominator

Use the exact source roster and source pins in `future_dev20_three_method_protocol_attempt06.json`: ten speakers, two clips each, three seeds 20261003/20261004/20261005, three native models, and identity/RMS/candidate. Generate all 540 rows anew. Fix one Fit40-only candidate before launch. Retain every planned row, every failure and all twenty clips; no replacement, best-seed choice or complete-subset rescue. A failed or missing mandatory row makes the method ineligible for admission. Later iterations on this cohort are explicitly development, not independent confirmation.

The sole primary scorer track is `native_roi_original_pcm16` with the complete original natural waveform. Strip the renderer audio for final delivery and restore the full original natural PCM and clock. Preserve exact sample count, mono 16 kHz source semantics and actual rational video timestamps. No best-offset shifting, audio time stretching, truncation or dropping leading/trailing silence.

## Support and aggregation

For each fixed sample/model/face/seed, intersect the valid query-row keys across all three conditions. Use the producer/consumer source-clock key semantics unchanged. FULL requires at least one common query. GUARD20 trims twenty keys at each end of that same ordered common sequence and requires at least forty-one remaining (FULL common count at least81). Measure actual masks and contiguous source-clock mapping; duration or nominal frame count does not establish eligibility. Report lag-padding provenance. This policy does not add a per-lag validity mask or impose an all31-real-lags gate on FULL.

Every mandatory FULL and GUARD20 endpoint must cover all ten speakers, their two clips and all three seeds. Average seeds equally within clip, clips equally within speaker, speakers equally within model, then models equally. Do not pool queries or clips by duration.

## Point admission rules

For each contrast candidate versus RMS and candidate versus identity, define positive-oriented improvements as Cbest(candidate)−Cbest(reference), D0(reference)−D0(candidate), and Canchor(candidate)−Canchor(reference). Canchor uses the predeclared fixed anchor; do not replace it with a favorable lag.

- On FULL, each model's Cbest improvement must be strictly positive for both references; the equal-model Cbest improvement versus RMS must be at least+0.050.
- On GUARD20, each model and the equal-model Cbest improvement must be strictly positive for both references. The +0.050 practical-effect threshold applies to FULL only.
- For D0 and Canchor, every model and the equal-model improvement must be nonnegative for both references on both supports. Zero point slack is allowed; no epsilon may hide a negative effect. Compute with float64 aggregates from the pinned native score arrays.
- All twenty clips must satisfy GUARD20 and every native-artifact, original-PCM, timestamp, content-consistency and visual-tracking availability check below. A partial subset cannot pass.

These are development engineering gates. Passing them licenses a fixed-method confirmation run; it does not establish human perceptual synchronization.

## Joint uncertainty report

Report exactly48 endpoints: two contrasts × three improvement metrics × four model estimands (three individual models and their equal mean) × FULL/GUARD20. These development intervals are descriptive alongside the point gates; do not call them confirmatory significance or reuse the old RMS99% family.

For each endpoint form ten speaker-level improvements after the fixed seed/clip means. Let μ be their equal mean and SE=sample-standard-deviation(ddof1)/sqrt(10). Generate one shared100000×10 integer resampling matrix using NumPy Generator(PCG64(20261009)), `integers(0,10,size=(100000,10),dtype=int64)`, in the declared speaker order S0102…S0111. Record the little-endian contiguous index bytes' SHA-256. For each draw take the mean of its ten resampled speakers. For SE>0 compute the absolute centered statistic abs((μ*−μ)/SE). Take each draw's maximum across48 endpoints, then the 0.95 quantile with `method='higher'` as c. Report μ±c×SE for all endpoints. This uses the original fixed SE, not per-draw SE.

For exactly zero SE, require all ten observed speaker values to be identical, assign statistic0 and a point interval atμ. Any nonfinite value or inconsistent zero-SE state fails the family. Missing/ineligible endpoints yield an explicitly incomplete family; never reduce its membership. Persist float64 speaker arrays, draw-index identity, endpoint order and all failures. Freeze a separate final confirmation family before opening Confirm.

## Reference-free ASR consistency

Use `qwen3-asr-flash-2026-02-10`, language zh, `enable_itn=false`, no text prompt, reference transcript or context. Bind actual canonical mono16k PCM16 WAV payload hashes and N before requests. Use the existing NFKC/casefold normalization with Unicode whitespace, separator and punctuation removal, then character Levenshtein divided by the normalized first hypothesis length.

For each of20 clips make separate natural1 and natural2 calls, plus one RMS call:60 baseline calls. Each call is independent and bypasses hypothesis caches; no automatic bulk retry. After the candidate is frozen and its actual PCM is available make20 candidate calls, once per unique clip, not once per TFG/seed. Preserve all planned calls and any failures. All80 normalized hypotheses must be available and nonempty.

Let e_i=d(natural1,candidate)−max(d(natural1,natural2),d(natural1,RMS)). Average two clips equally per speaker and ten speakers equally. Require the equal-speaker mean e≤0.020 and each clip e_i≤0.100. Negative excess is retained. These tolerances are predeclared engineering limits on excess transcription disagreement, not calibrated human intelligibility or ground-truth CER. Do not select or replace clips using baseline ASR results. Model availability failure is an explicit dependency outcome; changing the snapshot requires a fresh prospective policy and complete fresh baseline calls.

## Visual availability and quality boundary

Use the pinned independent visual-motion audit and landmarker asset on all540 actual renders. Require artifact/face/clock identity, decoded frame count and per-frame rational timestamps to match; each planned video must have tracking_valid_fraction≥0.950. Retain invalid frames and report their counts. Do not treat that threshold as a lip-motion quality metric.

Report eye-normalized mouth aperture, velocity, acceleration/jerk and their paired changes descriptively. A larger aperture is not itself a pass or better motion. This development method may be described as improving original-audio SyncNet scores after the above gates; claims of better perceived lip synchronization require a separately frozen human or independent visual criterion.

## Execution boundary

This policy is prospective. Actual checkpoint, future540 results, masks, ASR hypotheses, visual measurements and numerical analysis remain unbound. No policy field can be backfilled from observed future effects. Source-byte isolation, canonical PCM loading and runtime qualification remain mandatory. Keep Confirm80/40speakers and LatentSync closed until a usable fixed method passes development admission.
