## Purpose

Defines a fail-closed prototype for testing whether a small TTS-only waveform model can improve exact-length MFA-linear audio synchronization to paired real-video coordinates under a frozen, official-aligned SyncNet V2 objective.

## ADDED Requirements

### Requirement: The prototype accepts only MFA-linear TTS as model input
The prototype SHALL give the trainable model exactly one signal input: a finite mono 16 kHz MFA-linear TTS waveform whose sample count is exactly `640 * video_frame_count`. The model SHALL NOT receive natural waveform, natural acoustic features, transcript features, video features, cross-attention keys/values, or a sample identity embedding. Natural audio MAY be used before training to create the MFA alignment and one detached coordinate-calibration artifact, but it MUST NOT enter the trainable model, candidate-audio branch, preservation loss, or optimizer graph.

#### Scenario: Valid TTS-only record
- **WHEN** a record provides a hash-locked real-video track and an exact-length MFA-linear TTS waveform
- **THEN** the prototype admits the record and records that the trainable input modalities are exactly `mfa_linear_tts_waveform`

#### Scenario: Natural or attention conditioning is connected
- **WHEN** a model forward call, batch, or gradient graph contains natural audio/features, video conditioning, or a second audio sequence used for cross-attention
- **THEN** preflight fails with an input-isolation error before an optimizer step

#### Scenario: Input timebase is invalid
- **WHEN** the input is not mono 16 kHz, is non-finite, or does not contain exactly 640 samples per frozen 25 fps video frame
- **THEN** the record is rejected rather than resampled, padded, trimmed, or substituted during training

### Requirement: Real-video supervision uses one frozen coordinate system
The prototype SHALL derive supervision from a hash-locked real-video face track at exactly 25 fps. It SHALL freeze the selected track, frame interval, crop geometry, frame hashes, frame count, corresponding sample interval, and SyncNet model hash before training. Every candidate for a record SHALL reuse this byte- and coordinate-identical visual input.

#### Scenario: Frozen real-video track is reused
- **WHEN** step 0, a trained checkpoint, and official verification are evaluated for one record
- **THEN** all conditions use the same real-video frame interval and the artifacts report matching visual-track and geometry hashes

#### Scenario: Visual coordinates drift
- **WHEN** frame count, start frame, crop geometry, decoded frame hashes, sample interval, or SyncNet checkpoint differs between conditions
- **THEN** the record receives an engineering `NO_GO` and no sync comparison is reported as valid

### Requirement: The differentiable SyncNet frontend matches the official frontend
The prototype SHALL use the frozen SyncNet V2 checkpoint in evaluation mode. Its differentiable audio path SHALL have forward values equivalent to signed PCM16 serialization followed by the official `python_speech_features.mfcc` settings: 16 kHz, 25 ms frame length, 10 ms step, 13 coefficients, 26 filters, 512-point FFT, pre-emphasis `0.97`, appended energy, and rectangular analysis window. Normalized waveform SHALL be converted with a forward-exact saturating and rounding PCM16 straight-through operator before MFCC extraction.

For a record with `F` video frames and `N` audio samples, the prototype SHALL form exactly `W = min(F, floor(N/640)) - 5` windows. Window `i` SHALL contain real-video frames `[i,i+5)` and candidate MFCC frames `[4i,4i+20)`. The loss SHALL use raw Euclidean distance between the frozen 1024-dimensional SyncNet embeddings, not cosine distance or a learned classifier.

#### Scenario: MFCC parity fixture
- **WHEN** deterministic random, speech-like, silence, near-clipping, and minimum-length waveforms are processed by both frontends
- **THEN** all MFCC values are finite and agree within the recorded numerical tolerance, with the tolerance no larger than `2e-5` absolute and relative before SyncNet embedding

#### Scenario: Window-coordinate fixture
- **WHEN** synthetic indexed MFCC frames and video frames are windowed
- **THEN** the prototype produces exactly `W` windows and every audio start index is exactly four times its paired video start index

#### Scenario: File-level parity fixture
- **WHEN** a candidate is serialized, muxed with the frozen real-video track, demuxed, and scored by the official SyncNet path
- **THEN** demuxed PCM is sample-identical to the differentiable frontend's forward PCM and the proxy and official 31-point distance curves differ by at most `q = 0.001` at every offset

### Requirement: The target offset has one explicit sign convention
For each admitted record, the prototype SHALL represent a signed candidate-audio offset `s` in video frames by pairing visual coordinate `i` with audio coordinate `i+s`. The 31-point curve SHALL contain offsets `s ∈ {-15,…,+15}` in ascending order, so curve index is `s+15`; the official reported AV offset has the opposite sign. The target `s_ref` SHALL be the unique minimum of a detached natural-reference curve computed on the same frozen real-video coordinates, or an existing hash-validated artifact with exactly that meaning.

Natural-reference audio SHALL be discarded from the training loader after `s_ref` and its provenance are frozen. A record whose natural curve is non-finite, whose stored offset violates the sign contract, or whose best-versus-second-best distance gap is at most `2q` SHALL be ineligible.

#### Scenario: Offset sign fixture
- **WHEN** a synthetic audio embedding is shifted by known offsets `-15`, `0`, and `+15`
- **THEN** curve indices `0`, `15`, and `30` respectively are selected and the reported official AV offsets are `+15`, `0`, and `-15`

#### Scenario: Ambiguous natural coordinate
- **WHEN** the natural-reference curve has a best-versus-second-best gap less than or equal to `0.002`
- **THEN** the record is rejected rather than choosing zero, the first tie, or the step-0 candidate offset

### Requirement: The primary loss is sufficient for the claimed Sync-D and Sync-C improvement
Before training each record, the prototype SHALL freeze the step-0 candidate curve `c0`, where the zero-initialized model output is sample-identical to the MFA-linear input. Define:

```text
D0 = min_s c0[s]
C0 = median_s c0[s] - D0
ΔD = max(3q, 0.01 D0)
ΔC = max(3q, 0.01 C0)
D_goal = D0 - ΔD
C_goal = C0 + ΔC
S_D = max(D0, 1e-3)
S_C = max(C0, 1e-3)
```

The record SHALL be rejected if `D0 <= ΔD` or `C0 <= 0`. For candidate curve `c`, the primary loss SHALL be:

```text
L_D = ReLU((c[s_ref] - D_goal) / S_D)
L_R = mean_{s != s_ref} ReLU((C_goal - (c[s] - c[s_ref])) / S_C)
L_sync = L_D + L_R
```

All 31 candidate distances SHALL remain differentiable. The real-video embeddings, `s_ref`, step-0 curve, goals, and scales SHALL be detached. No candidate-audio self-pair, another utterance, cosine loss, BCE label, soft offset label, or detached wrong-offset distance may replace this objective.

#### Scenario: Zero-loss implication
- **WHEN** a finite synthetic candidate curve has `L_sync = 0`
- **THEN** `s_ref` is its unique global minimum, candidate Sync-D is at least `ΔD` lower than step 0, and candidate Sync-C is at least `ΔC` higher than step 0

#### Scenario: Individual constraint violation
- **WHEN** only the target-distance constraint or any one wrong-offset separation constraint is violated
- **THEN** `L_sync` is positive and backward produces a finite nonzero gradient on every candidate curve element involved in that violation

#### Scenario: Incorrect positive or negative construction
- **WHEN** the target index is derived from the candidate curve, wrong offsets are detached, or audio is compared with itself
- **THEN** the loss-contract test fails before real training

### Requirement: The model is identity-initialized and preservation-bounded
At step 0 the model SHALL return the MFA-linear waveform sample-for-sample. It SHALL preserve shape for every forward call and bound its additive waveform residual. Training SHALL add a candidate-to-MFA Wav2Lip-compatible normalized log-mel trust hinge:

```text
E_mel = mean(abs(mel(candidate) - mel(mfa_linear)))
L_trust = ReLU((E_mel - 0.10) / 0.10)
L_total = L_sync + L_trust
```

The model output, residual, PCM16 forward value, MFCC, embeddings, every loss component, and every gradient SHALL be finite. A checkpoint SHALL be invalid if its normalized log-mel distance exceeds `0.10`, if output length changes, or if more than `0.01%` of samples require PCM16 saturation.

#### Scenario: Fresh model identity
- **WHEN** a freshly initialized model processes an admitted MFA-linear waveform
- **THEN** its output has exact sample equality with the input, its step-0 curve equals the direct MFA-linear curve within `q`, and its residual is identically zero

#### Scenario: Candidate leaves the trust region
- **WHEN** a checkpoint lowers `L_sync` but violates the log-mel or saturation limit
- **THEN** the checkpoint is rejected and cannot pass a prototype stage

### Requirement: Frozen modules retain an input-gradient path
All SyncNet parameters and buffers SHALL stay frozen and byte-identical, and SyncNet SHALL stay in evaluation mode. The fixed real-video embedding path SHALL run without gradients and may be cached. The candidate waveform → PCM16 straight-through → MFCC → SyncNet audio branch path SHALL run without `no_grad`, inference mode, or detach, so gradients reach the waveform model. Only waveform-model parameters SHALL be passed to the optimizer.

#### Scenario: End-to-end gradient smoke
- **WHEN** one valid record performs backward from `L_total` at step 0
- **THEN** at least one output-layer parameter has a finite nonzero gradient, the aggregate trainable-model gradient norm is finite and positive, and the candidate waveform is connected to the loss

#### Scenario: Gradient after first update
- **WHEN** one optimizer step is applied and backward is run again
- **THEN** at least one upstream trainable parameter receives a finite nonzero gradient and only declared waveform-model parameters have changed

#### Scenario: Frozen state mutation
- **WHEN** any SyncNet parameter or buffer changes, the module enters training mode, or candidate embeddings are detached
- **THEN** the stage terminates with engineering `NO_GO`

### Requirement: Execution is staged and fail-closed
The prototype SHALL expose three ordered stages: `P0_SEAM`, `P1_ONE_RECORD`, and optional `P2_SHARED_FOUR`. P0 SHALL complete all synthetic mathematics, frontend parity, offset sign, identity, gradient, freeze, and artifact-schema checks without real optimization. P1 SHALL use one frozen fit-only historical record, exactly 20 AdamW steps, no checkpoint search, and evaluate only step 0 and step 20. P2 SHALL remain `NOT_REQUESTED` unless P1 passes and the run explicitly enables P2; if enabled, it SHALL start from a fresh initialization, use exactly four frozen fit-only historical records from distinct source groups, one shared model, exactly 100 AdamW steps, and evaluate only steps 0 and 100.

#### Scenario: P0 failure
- **WHEN** any mandatory P0 test fails
- **THEN** P1 and P2 are not run and the terminal decision is engineering `NO_GO`

#### Scenario: P1 failure
- **WHEN** P1 does not satisfy every P1 acceptance gate
- **THEN** P2 is not run and the result remains a completed negative fixed-data prototype rather than triggering retries or changed constants

#### Scenario: Optional P2 is not enabled
- **WHEN** P1 passes but the invocation does not explicitly request P2
- **THEN** the artifact records `P2_NOT_REQUESTED` and P1 remains the terminal prototype result

### Requirement: Official verification determines stage acceptance
P1 SHALL pass only when all integrity gates hold, proxy `L_sync` decreases from step 0 to step 20, the official file-level step-20 curve matches the proxy curve within `q`, official Sync-D decreases by more than `3q`, official Sync-C increases by more than `3q`, the official best signed shift equals `s_ref` with best-versus-second gap greater than `2q`, and preservation gates pass. Otherwise P1 SHALL return one explicit failure status.

If P2 is run, it SHALL pass only when all four records are valid, median official Sync-D gain is greater than `3q`, median official Sync-C gain is greater than `3q`, and at least three of four records improve both metrics while matching `s_ref` and passing preservation. P2 SHALL NOT change P1's loss, model, optimizer, step count, records, or thresholds after observing outcomes.

#### Scenario: Proxy improves but official score does not
- **WHEN** training loss decreases but either official metric fails its margin or proxy/official curves exceed parity tolerance
- **THEN** the stage fails with a proxy-to-official transfer status and no success claim is emitted

#### Scenario: P1 satisfies every gate
- **WHEN** the one-record run satisfies all mathematical, gradient, integrity, preservation, parity, offset, Sync-D, and Sync-C predicates
- **THEN** the terminal P1 status is `FIXED_DATA_TTS_ONLY_SYNC_CONSTRUCTED`

#### Scenario: P2 satisfies every gate
- **WHEN** an explicitly enabled four-record shared run satisfies every P2 predicate
- **THEN** the terminal P2 status is `FIXED_DATA_SHARED_TTS_ONLY_SYNC_CONSTRUCTED`

### Requirement: Artifacts preserve provenance and claim boundaries
Each run SHALL write a machine-readable lock manifest, config, model-state hashes, frozen-module hashes before and after training, P0 evidence, per-step losses and gradient norms, step-0 and final proxy curves, exact candidate PCM hashes, official full curves, QC, and one terminal decision. Existing run files SHALL not be silently overwritten; resume SHALL reuse only hash-valid completed cells.

A passing result SHALL be described only as fixed-data constructability on records used for optimization. It SHALL NOT be described as validation, transfer, generalization, perceptual improvement, content/speaker preservation, Wav2Lip or TFG improvement, or replacement-safe audio.

#### Scenario: Complete positive run
- **WHEN** a stage passes
- **THEN** its artifact contains enough hashes, curves, formulas, thresholds, and per-predicate outcomes to recompute the decision without rerunning training

#### Scenario: Unsupported scientific claim
- **WHEN** a report attempts to infer held-out or downstream benefit from P1 or P2
- **THEN** validation fails the claim-boundary contract even if all fixed-data gates passed
