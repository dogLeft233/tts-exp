## ADDED Requirements

### Requirement: Measurement-gated TTS increment

The second-stage experiment SHALL test a single additional waveform M on the same frozen12 groups: self-cloned TTS mapped to N's phone time coordinates, decoded with the same frozen WavLM/HiFi-GAN path used by C. Its purpose is to separate M−N absolute gain from M−C gain beyond direct decoder processing. The latest phone-core/reference/base-interaction nulls do not establish that this cross-generator TTS path is useful; it is a new bounded test of that question.

Entry requires both generators' smoke, media/PCM validation, A natural measurement controls and B real-only metric calibration to pass; it does NOT require positive A/C candidate results. Require cached local Qwen0.6B and MFA3 assets and enough remaining time under the2-hour total GPU ceiling. If these fail, write DEFERRED_MEASUREMENT, BLOCKED_TTS_ASSET or DEFERRED_BUDGET as applicable and report the already completed A/B/C results. Do not silently switch to a cloud provider, download a larger model, or select the more favorable generator.

#### Scenario: Direct reconstruction has no gain

- **WHEN** A is scientifically negative but its measurement controls pass
- **THEN** D may run on both models if the other entry requirements pass; A positivity is not a selection gate

### Requirement: One frozen self-clone and alignment recipe

P SHALL use each complete6–10second official transcript as text and ref_text, its own N as ref_audio, language=English, seed42, Qwen/Qwen3-TTS-12Hz-0.6B-Base ICL through `scripts/tts/faster_qwen3.py`. No ASR or paid API is required. Set offline model-loading mode and require already cached complete weights. Freeze actual selected backend class, installed defaults and model revision at smoke; the provider has a fallback, so detect and reject backend changes across items. Generate once per group, with at most the shared engineering retry allowance. Save raw output, resample once to16k mono using one pinned resampler, without trimming internal pauses or selecting the best take.

Use the existing MFA3 `english_mfa` dictionary and acoustic model, and helpers from `scripts/experiments/lrs3_mfa_linear_replacement_mfa3_exploratory/protocol.py`: normalization, build_mfa3_command/run_mfa3_alignment and parse_alignment_pair. Snapshot executable/model/dictionary hashes. Natural and TTS align to the same normalized full transcript. Read helper signatures before calling; use a small adapter for input records. Do not invoke old cohort-locked runner or the older24-record MFA wrapper.

Extract N/TTS features using the same frozen WavLM interface as direct_v1. Use `build_frame_mapping` in `lrs3_mfa_linear_replacement/mfa_alignment.py` and `interpolate_conditioning` in its candidate_audio.py unchanged: match speech-phone labels/order, linear within matched phones, natural feature centers define the output clock; existing silence fallback is allowed and logged. Any unmatched natural speech phone, nonfinite mapping or unknown phone rejects the M cell. No alternate dictionary, time-warp search or fallback to global stretch.

Decode mapped features with the identical vocoder used for C. Save conditioning, mapping trace and raw waveform. Apply direct_v1's identical right-tail-only length adjustment and PCM_16 serialization to length L, with adjustment≤640 samples and no loudness normalization. Do not import a different PCM scaling convention from an old runner. Save N/TTS phone TextGrids and hashes; require mapped speech frames>0 and report silence fallback counts.

M failure after frozen cohort does not replace the group. Preserve all12 in the failure table; D cannot establish a signal without all12 complete. A/B-direct/C may still finish. TTS construction has14 possible inputs (12 formal+2smoke); do not generate for excluded groups.

#### Scenario: Alignment fails on a frozen sample

- **WHEN** Natural speech has no matching TTS phone
- **THEN** Preserve its failure and the12-group denominator; no new TTS take, new sample or global stretching is used to rescue it

### Requirement: Paired comparison to natural and decoder control

A SHALL first produce M42 smoke for each model on the first smoke group (2 videos). After both pass, add M42/M43 for every formal group/model (48 videos/48fresh embedding cells). All scientific audio remains N. Use the exact A support, previously frozen k0, seeds, weighting, formulas,99%CI and validation; do not pick a new candidate anchor.

Report M−N, M−C and C−N separately for C/D/fixed-anchor metrics. For fixed-anchor contrast X−Y use z_Y[k0]−z_X[k0], always k0 from original N42. Apply A's per-model SIGNAL rule to M−N. Additionally call TTS_INCREMENT_OBSERVED only when M−N passes, M−C C and fixed-anchor gains have99%CI lower bounds>0, and at least10/12 groups jointly improve both quantities for M−C. If M beats C but not N, label REPAIR_RELATIVE_TO_DECODER_ONLY. If M beats N without the incremental conjunction, label GAIN_TTS_INCREMENT_UNRESOLVED. Otherwise NO_TTS_REPLACEMENT_GAIN_ESTABLISHED.

Test generator interaction using the paired difference of M−N effects, as in A; report M−C interaction descriptively too. All12 group rows and controls are visible. Comparisons share a cohort and are correlated; this remains exploratory and has no claim of familywise confirmation over all experiments.

#### Scenario: TTS only repairs the decoder's damage

- **WHEN** M−C is positive while M−N fails its gate
- **THEN** State decoder-relative repair without natural-input improvement; do not call it replacement success

### Requirement: Visual follow-through and limited mechanism claims

B SHALL apply the same fixed visual rules to M vs N, using common R/N42/M42/N43/M43 support, and separately report M−C descriptively. If D runs, add24 M/N blind pairs (one seed42 pair per group/model) to the24 C/N pairs, retaining the same4QC trials:52 trials per reviewer. If D does not run, deliver the28-trial base package. Decide the package contents from D's execution status before any human ratings; do not choose the candidate from its score.

Report automated and human support for C and M separately. If M passes D and B for the same generator, it merits a separate new-source confirmation. M−C includes TTS synthesis, mapping and conditioning changes and does not by itself prove a semantic-content mechanism. No classifier training, learned audio head, larger parameter sweep or old teacher gate reopening follows automatically.

#### Scenario: Candidate selection would bias the blind review

- **WHEN** Both C and M were executed
- **THEN** Both candidate sets enter the reviewer packages regardless of SyncNet ranking
