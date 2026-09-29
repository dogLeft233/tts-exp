## ADDED Requirements

### Requirement: New-source integer-delay intervention

The experiment SHALL use all12 frozen groups and both generators from design.md, with a single waveform intervention: DELAY[n]=N[n−3200], out-of-range zero, original length L. This is +200ms/+5 video frames, with no interpolation, time-stretching, gain, or phoneme editing. Save PCM and its exact source-index map. A renders one DELAY video per group/model at seed42, using the same reference/parameters as N42:24 new videos/24 fresh embedding cells.

Motivation: `runs/lrs3_wav2lip_timing_transfer_20260905_v8/result.md` had real-domain sensitivity22/22 and generated-domain sensitivity21/22, but generated response0/22 and self local alignment0/22; the outcome was GENERATED_RESPONSE_UNRESOLVED, not proof of absent audio control. The new probe removes interpolation from this particular control and adds Ditto and independent mouth trajectories. Global-delay success does not establish local phoneme editing or replacement benefit.

#### Scenario: A scorer-only shift is supplied

- **WHEN** A DELAY cell lacks a fresh generator forward driven by its frozen DELAY PCM
- **THEN** Validation fails; reindexing audio embeddings alone cannot answer generator response

### Requirement: Distinguish delay signs and preserve support

The branch SHALL reuse N42 embeddings and A's fresh DELAY embeddings, both evaluated against untouched N. Use U=range(25,115). Natural lag domain is−15..15; DELAY-video domain is−20..10, with actual q=r+lag in both. All windows must exist, no padding. Compute z, C, D and offset=−best_lag with the same distance/mean formulas as A.

For an ideal delayed video, expected best_lag_DELAY=best_lag_N−5, so offset_DELAY−offset_N=+5. This is opposite to A's scoring-audio-delay control. Also compare video embedding V_DELAY[r+5] against A_N[r+lag_N] on the same U. Let d_unc be mean distance V_DELAY[r] to A_N[r+lag_N], d_comp the distance using V_DELAY[r+5], and rescue=d_unc−d_comp. Freeze lag_N from N42; no new lag selection in rescue.

A model's scorer-transfer gate requires valid A natural measurement controls, offset change within±1frame of+5 in at least10/12 groups, rescue>0 in at least10/12, and lower99%CI(mean rescue)>0. Report corrected distance minus N's anchor distance as residual error, even if rescue passes; recovery from damage does not imply improvement beyond N.

#### Scenario: Wrong direction would accidentally pass

- **WHEN** A synthetic ideal delayed video is tested
- **THEN** Its reported offset increases5, the scorer-audio delay from A decreases5, and compensated error is lower; tests cover both directions

### Requirement: Independent mouth-response corroboration

The experiment SHALL extract/reuse canonical mouth features for N42 and DELAY at each model, using B's pinned extractor; it writes any new features under run/C. Use common valid t and t+5 within U for both paths, with ≥81 retained rows. Center compared sequences separately on that same support.

e_unc=mean((center(N[t])−center(DELAY[t]))²); e_comp=mean((center(N[t])−center(DELAY[t+5]))²). Visual transfer score v=(e_unc−e_comp)/(e_unc+e_comp+1e-12). Stationary trajectories give0 and remain in the denominator. Calibrate this formula with synthetic R_DELAY[t]=R[t−5] on valid interior indices; compensated error≤1e-12 and uncompensated error>1e-6 in at least10/12 groups. No search over shifts.

Visual-transfer gate: synthetic calibration passes, all12 groups observed, mean v>0.02, lower99%CI>0, at least10/12 groups v>0. Missing generated features produce VISUAL_RESPONSE_UNRESOLVED, not a zero score or dropped denominator. Bootstrap uses the shared PCG64(20260910),20000 group draws and linear quantiles. This compares timing response between generated videos; it does not measure closeness to real speech dynamics as B does.

#### Scenario: Both generated videos are nearly static

- **WHEN** Compensated and uncompensated errors are both zero
- **THEN** v=0 and visual transfer is not established, even if an offset estimate appears favorable

### Requirement: Actionable bounded diagnosis

The branch SHALL give each model one outcome: TIMING_TRANSFER_OBSERVED when both scorer and visual gates pass; SCORER_VISUAL_DISAGREEMENT when only one passes; RESPONSE_NOT_ESTABLISHED when neither passes with valid calibration; CONTROL_FAILED/BLOCKED when measurement/engineering gates fail. Report all actual values. A failed transfer gate is a scope limitation, not proof that the model ignores audio.

Compute paired model difference v_Ditto−v_Wav2Lip with99%CI. A model-specific transfer difference requires this CI exclude0, not merely one pass and one fail. Results guide the next hypothesis: transfer observed but A/D no gain → this fixed candidate did not exploit measurable timing control; measurement insensitive → improve measurement before training; differing models → independently confirm the model interaction. No result authorizes training or changes old gate status.

#### Scenario: Transfer exists without replacement gain

- **WHEN** A model passes C but fails A and D
- **THEN** The conclusion states measurable timing response and no established benefit for the tested candidates; it does not call the delay a successful replacement

### Requirement: Independent validation

The validator SHALL independently reconstruct DELAY PCM, support, sign, distances, compensation, visual scores, CIs and outcomes from manifests/arrays. The branch consumes A's immutable completed video rows without mutating A or B. It can complete while human ratings are pending.

#### Scenario: Implementation optimizes a compensation shift

- **WHEN** Any compensation other than the predeclared+5 frames is used
- **THEN** Validation rejects the run even if its score is better
