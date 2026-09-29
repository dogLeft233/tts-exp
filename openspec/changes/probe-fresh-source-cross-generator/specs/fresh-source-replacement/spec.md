## ADDED Requirements

### Requirement: Frozen new-source paired media

The experiment SHALL follow design.md for 12 new source groups, one fixed direct_v1 candidate, common reference images, two generators, and unchanged natural scoring PCM. Only this capability's calculations below add to that shared contract.

#### Scenario: A different clip is from an old video

- **WHEN** Its source group appears in historical usage or a sealed split
- **THEN** P excludes it before opening its media; a new sample ID does not establish independence

#### Scenario: Ditto cannot run its smoke

- **WHEN** The pinned installed backend fails smoke after one explained engineering retry
- **THEN** A reports BLOCKED_DITTO and any available Wav2Lip results as single-model descriptive results; it does not claim a cross-generator test completed

### Requirement: Fixed support and natural anchor

The scorer SHALL save fresh visual/audio embeddings for all 100 formal cells using SyncNet V2. Reuse the forward interface in `scripts/experiments/wav2lip_roi_peak_recheck/worker.py`, but construct only genuinely supported distances, not its padded edge rows. Weight SHA is `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`; CPU batch20/threads2.

Let frame-window starts U=`range(25,115)`, q=`r+j-15`, j=0..30. For every cell verify visual r and audio q windows exist in the140-frame video and frozen full-length N. Distances are float32 `sqrt(sum((V[r]-A[q]+1e-6)^2))`; mean over U in float64 gives z[j]. C=median(z)−min(z), D=min(z), offset=15−argmin(z); ties take first column.

For each group/model, freeze k0=argmin(z_N42) from natural controls before examining C. Fixed-anchor gain for each seed s is A_s=z_Ns[k0]−z_Cs[k0]. C gain=C_Cs−C_Ns; D gain=D_Ns−D_Cs. Average each gain over seeds42/43, then equally over12 groups. Seeds are repetitions, not extra independent samples. Sync-C is secondary to fixed-anchor gain; report all three and the offsets. Do not realign video, audio, or the evaluation support after scoring.

#### Scenario: Candidate only improves a searched peak

- **WHEN** ΔSync-C is positive but fixed-anchor gain does not pass the signal rule
- **THEN** The result is NO_REPLACEMENT_SIGNAL_ESTABLISHED, with both numbers reported

### Requirement: Timing sensitivity and repeats

The experiment SHALL calibrate using N42 before final candidate interpretation. A synthetic +5-frame audio delay means A_delay[q]=A_N[q−5], no padding. Use the same U, and compare its lag curve to N in the matched lag domain −10..20 (q=r+lag−5). These curves must agree within1e-4; best lag increases by5 and reported offset decreases by5. Separately compute uncompensated damage at natural k0 in the original −15..15 domain. This checks scoring sensitivity, not generator equivariance.

Per model, require natural k0 interior (1..29), shifted damage>0 in at least10/12 groups and its group-bootstrap95%CI lower bound>0. Every same-seed N42_repeat must match original C/D/fixed-anchor within0.01 and offset within1frame. Report pixel differences even when scores match. Failure is CONTROL_FAILED for that model; candidate results may be reported descriptively but cannot pass signal gates.

Report across-seed N43−N42 and C43−C42 distributions. For each model and endpoint define noise floor as mean over groups of the absolute N43−N42 difference (for A use z at k0). This is a conservative empirical reference from two seeds, not an estimated universal variance.

#### Scenario: Shift is absorbed by offset search

- **WHEN** Matched-domain C is unchanged but uncorrected fixed-anchor distance worsens
- **THEN** The control can pass; unchanged free-offset C alone is not a failure

#### Scenario: Repeated generation is unstable

- **WHEN** Same-seed repeat exceeds the frozen tolerance
- **THEN** Report CONTROL_FAILED and the instability, without rerolling seeds or enlarging tolerances

### Requirement: Exploratory signal and model interaction

The analysis SHALL use source-group bootstrap, PCG64(20260910), 20000 draws, NumPy linear quantiles and the same indices for all contrasts. Controls use95%CI; scientific contrasts use99%CI. These are exploratory intervals, not a correction for the entire historical search.

A model has SIGNAL only if controls pass, all12 groups have every required cell, mean ΔSync-C>0.050, C and fixed-anchor gains each have99%CI lower bound>0 and mean above their noise floor, D gain has99%CI lower bound>−0.100, and at least10/12 groups jointly have positive C and fixed-anchor gains. Otherwise use NO_REPLACEMENT_SIGNAL_ESTABLISHED (or the applicable control/engineering status). No complete-case dropping of failed groups.

Compute groupwise interaction I_A=gain_A(Ditto)−gain_A(Wav2Lip), and likewise I_C; report means/99%CIs. One model passing and the other failing does not itself prove a difference. Only Ditto SIGNAL plus lower99%CI(I_A)>0 and lower99%CI(I_C)>0 allows `DITTO_DEPENDENT_SIGNAL`; both SIGNAL allows `TWO_GENERATOR_EXPLORATORY_SIGNAL`. A sole signal without supported interaction is `SINGLE_GENERATOR_SIGNAL_INTERACTION_UNRESOLVED`. All are bounded exploratory labels with replacement_confirmed=false.

#### Scenario: Significance differs between generators

- **WHEN** Ditto passes its own gate but the interaction CI crosses zero
- **THEN** The report labels the interaction unresolved and does not conclude Wav2Lip uniquely explains historical failure

### Requirement: Independent completion

The validator SHALL independently reconstruct all distances, paired gains, bootstrap intervals, controls, cell counts and terminal labels from embeddings and manifests. It SHALL verify all mux PCM bindings and all expected IDs, then write validation before final.

#### Scenario: Signed output contains wrong statistics

- **WHEN** analysis.json is edited and its hash updated without changing raw arrays
- **THEN** The independent recomputation rejects it and final cannot report a valid scientific conclusion
