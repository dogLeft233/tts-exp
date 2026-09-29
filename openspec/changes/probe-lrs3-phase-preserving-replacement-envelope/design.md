## Context

See `proposal.md` for motivation and `specs/lrs3-phase-preserving-replacement-envelope/spec.md` for the behavioral contract.

The completed MFA-linear/DTW comparison already provides natural audio, exact-natural-length MFA-linear candidates, face videos, and frozen evaluator assets for 133 records across 23 source groups. Its replacement results are negative, but it gives a useful failed endpoint. Selecting the first ordered record from each source group yields 23 independent fit-only examples without outcome-based selection:

```text
parent protocol SHA-256: e8c50a459c8197c86ec11cf06a01c000e3f62def981d16d8ed9275a38a473784
parent replacement SHA-256: 157cd678f5f03be6eae9986a82babfdfb1f1a44e611bb0b86476c6e19b7eb272
selected-ID SHA-256: c56e420d9ade7e04f5558f37fbf68ee68463d3817060baab79e8e6e0bf7e8fbd
records/source groups: 23/23
```

The experiment is deliberately a boundary probe, not another model comparison. It asks whether retaining natural phase while moving magnitude toward the aligned TTS candidate creates a nontrivial compatible point between the known natural and failed-TTS endpoints.

## Goals / Non-Goals

**Goals:**

- Establish that the pipeline can observe both a non-identical compatible control and a deliberately timing-incompatible control.
- Test one interpretable continuum from natural audio toward aligned TTS acoustics.
- Identify the largest pre-registered blend strength that meets the fixed-cohort compatibility gates.
- Produce reusable candidate and scoring artifacts for a later, separately specified identity test.

**Non-Goals:**

- Mapping every possible replacement-preserving audio transformation.
- Proving that a blend sounds like the target speaker.
- Producing a usable TTS or voice-conversion system.
- Re-running alignment, generating TTS, tuning transform strengths, or accessing held-out media.
- Treating failure on this one construction as proof that replacement-compatible identity transfer is impossible.

## Decisions

### 1. Use one record per source group

Stage 00 scans the bound parent cohort in its existing order and keeps the first occurrence of each source group. It verifies exactly 23 records and the registered ordered-ID hash before decoding media.

**Why:** This preserves source diversity, removes within-group dependence, cuts rendering cost from 133 to 23 records, and does not use replacement outcomes.

**Alternative considered:** All 133 records with cluster bootstrap. Rejected for this proof-of-concept because it multiplies GPU and SyncNet work without adding new source groups.

### 2. Test one phase-preserving spectral path

All waveforms use exact 16-kHz mono PCM16 input. The `MAG` transform runs deterministically on CPU in float64 with:

```text
n_fft = 1024
win_length = 1024
hop_length = 256
window = periodic Hann
center = true
pad_mode = reflect
magnitude_floor = 1e-7
```

For natural STFT `S_N`, MFA-linear STFT `S_M`, and strength `alpha`:

```text
log_mag_alpha = (1 - alpha) * log(max(abs(S_N), eps))
                + alpha * log(max(abs(S_M), eps))
phase_N = S_N / max(abs(S_N), eps)
S_alpha = exp(log_mag_alpha) * phase_N
x_alpha = ISTFT(S_alpha, length=natural_sample_count)
```

The output is RMS-matched once to natural using one global linear factor. If its absolute peak is at least `0.999`, one additional global linear attenuation sets the peak to `0.999`; no compressor, limiter, denoiser, time shift, or iterative reconstruction is allowed. The exact scale factors are recorded. The waveform is then canonicalized once to PCM16.

**Why:** The transform changes spectral magnitude in a known TTS direction while fixing phase to the natural source and preserving exact length. It requires no learned model or new dependency.

**Alternative considered:** Waveform interpolation. Rejected because summing separately phased speech can create echo and comb-filter artifacts that obscure the spectral/timing question.

**Alternative considered:** A trained voice-conversion model. Deferred until this experiment shows that a nontrivial compatible region exists.

### 3. Use four fixed strengths and two controls

The complete arm order is:

```text
N, INV, MAG_025, MAG_050, MAG_075, MAG_100, SHIFT_200
```

`INV` is constructed by exact sign inversion before PCM16 canonicalization. Any `-32768` edge case that prevents deterministic inversion blocks that record rather than changing the rule. `SHIFT_200` prepends 3,200 zeros and drops the final 3,200 samples.

**Why:** Four blend points are enough to expose a coarse boundary without an adaptive search. Polarity inversion checks Wav2Lip-mel equivalence with different PCM bytes; a 200-ms delay checks that the replacement evaluator responds to a known timing mismatch.

**Alternative considered:** Pitch, formant, codec, noise, and multiple delay sweeps. Rejected to keep the first experiment causal and small. They require separate specs only if this path is uninformative.

### 4. Measure actual Wav2Lip-domain movement

Before reading replacement outcomes, compute official Wav2Lip mel tensors for `N`, the bound MFA-linear candidate `M`, and each generated arm. For flattened tensors, define:

```text
d = mel(M) - mel(N)
u = mel(X) - mel(N)
progress(X) = dot(u, d) / max(dot(d, d), 1e-12)
orthogonal_ratio(X) = norm(u - progress(X) * d) / max(norm(d), 1e-12)
```

The report also includes mel MAE to natural and MFA-linear, RMS, peak, clipping, DC offset, and applied scale factors. Mel shape mismatch or a degenerate `d` blocks the record.

**Why:** Construction strength is not evidence that the rendered candidate actually moved in the frontend domain. Directional progress supplies a simple nontriviality check without importing a speaker model.

### 5. Render a narrow two-column matrix

Every arm is freshly rendered using the same frozen Wav2Lip checkpoint and one hash-bound face geometry per record. For each arm `X`, score:

```text
V_X / A_N   replacement cell
V_X / A_X   own-audio cell
```

`V_N/A_N` is stored once, so the run contains 161 videos and 299 unique cells. Cross-arm combinations are omitted. Strict muxing copies the video elementary stream, writes PCM s16le audio, and verifies decoded audio bytes before official SyncNet V2.

**Why:** Replacement against natural and own-audio preference answer the registered question. A complete 7-by-7 matrix would add 828 irrelevant cells.

### 6. Use fixed-cohort non-inferiority rather than superiority

For each arm:

```text
gap_C(X) = SyncC(V_X, A_N) - SyncC(V_N, A_N)
gap_D(X) = SyncD(V_N, A_N) - SyncD(V_X, A_N)
```

The analysis uses 10,000 deterministic bootstrap draws over the 23 records with seed `20260904`. The compatibility margin is `-0.10` for both metrics, matching the earlier codec identity tolerance. Offset agreement means the scored offset differs from `V_N/A_N` by no more than one frame.

The analysis evaluates the four levels in their frozen order and reports every level. `max_compatible_alpha` is the largest level that passes all compatibility and nontriviality gates; no interpolation or post-hoc level is tested.

**Why:** The desired behavior is preservation of natural-audio compatibility, not improvement over natural. The ordered fixed levels make the output easy to interpret while avoiding adaptive search.

### 7. Keep the conclusion narrow

The terminal outcomes are:

```text
BLOCKED
CONTROL_FAILED
PHASE_PRESERVING_REPLACEMENT_ENVELOPE_FOUND
ONLY_TRIVIAL_REPLACEMENT_EQUIVALENCE_FOUND
```

Only `PHASE_PRESERVING_REPLACEMENT_ENVELOPE_FOUND` sets `identity_characterization_experiment_eligible=true`. This flag authorizes writing a separate spec; it does not establish speaker transfer or execute new modeling.

## Stage Layout

```text
runs/lrs3_phase_preserving_replacement_envelope_20260904/
  00_protocol/
  01_candidates/
  02_audio_diagnostics/
  03_videos/
  04_scores/
  05_final/
```

Each stage validates upstream hashes. Candidate, video, mux, and score cells use temporary files followed by atomic rename. Resume accepts a cell only when its full input/configuration/output identity matches; score-dependent retries are forbidden.

## Risks / Trade-offs

- **[The transform may change timbre without producing a coherent target speaker]** → Call the result TTS-direction movement, not identity transfer; require a separate identity experiment.
- **[Natural phase plus foreign magnitude may be inconsistent under ISTFT]** → Measure the realized waveform and official Wav2Lip mel rather than assuming the requested blend was preserved.
- **[The fixed cohort is small]** → Scope conclusions to these 23 fit-only source groups and require a new confirmation spec before broader claims.
- **[SyncNet may tolerate the delayed control through offset search]** → Gate sensitivity on both offset displacement and replacement-score degradation; emit `CONTROL_FAILED` if neither responds.
- **[A coarse alpha grid can miss a narrow boundary]** → Report only registered points. A denser follow-up is justified only after this run, not added during execution.
- **[RMS and safety scaling alter absolute magnitude]** → Freeze the linear rule, record every factor, and measure the actual Wav2Lip mel after scaling.

## Migration Plan

1. Implement the isolated experiment package by reusing existing PCM, hashing, rendering, strict-mux, SyncNet, and bootstrap helpers.
2. Run focused synthetic tests for cohort selection, STFT blending, controls, cell enumeration, endpoint signs, and terminal decisions.
3. Lock Stage 00 and validate zero outcome reads and zero sealed-data access.
4. Generate all candidate arms and freeze audio diagnostics before scoring.
5. Render and score the complete narrow matrix with hash-validated resume.
6. Finalize the registered decision once and independently validate the run.
7. Rollback removes only incomplete temporary files from the new run root; prior experiment artifacts are never modified.
