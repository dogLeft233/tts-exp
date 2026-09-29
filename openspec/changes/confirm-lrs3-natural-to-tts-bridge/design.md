## Context

See `proposal.md` for motivation and `specs/lrs3-natural-to-tts-bridge-confirmation/spec.md` for the normative contract.

The discovery run established a useful direction but ended `CONTROL_FAILED`: `MAG_075` passed its movement and replacement sub-gates, while the global `SHIFT_200` control did not produce the expected SyncNet response. This confirmation therefore changes only two things: it uses records not present in the discovery cohort, and it replaces the globally shifted control with a local sequence corruption.

Frozen provenance:

```text
parent protocol SHA-256: e8c50a459c8197c86ec11cf06a01c000e3f62def981d16d8ed9275a38a473784
parent replacement SHA-256: 157cd678f5f03be6eae9986a82babfdfb1f1a44e611bb0b86476c6e19b7eb272
discovery final file SHA-256: 9bab2a853edda0cdb16e79e2b9a8c6d32ce3e36d721e4a1c1140306d9ffac8c1
confirmation ordered-ID SHA-256: 5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72
confirmation records/source groups: 22/22
```

The confirmation cohort is selected before media decoding or confirmation scoring by scanning the ordered 133-record parent cohort and retaining the second occurrence of each source group. Exactly 22 groups have a second record; the remaining group is omitted by rule. This cohort is disjoint from the 23 first-occurrence records used for discovery.

## Goals / Non-Goals

**Goals:**

- Verify that independent rerendering of the same natural input is stable under the frozen endpoint.
- Verify that a local timing corruption is good with its own audio but bad after natural-audio replacement.
- Confirm on unused fit records that one natural-based bridge both moves toward TTS and retains useful replacement behavior.
- Produce one terminal decision with enough provenance for a downstream agent to reproduce and validate it.

**Non-Goals:**

- Searching additional blend strengths or transformation families.
- Proving speaker identity, perceptual audio quality, general LRS3 performance, or deployment readiness.
- Training a reference-conditioned model.
- Reinterpreting or modifying the discovery run.
- Accessing sealed validation/test media.

## Decisions

### 1. Use an unused, deterministic 22-group cohort

Stage 00 reads only the two immutable parent manifests and the bound discovery final artifact. It scans the parent cohort in order, selects the second record in each source group, and verifies exactly 22 records, 22 groups, disjointness from the discovery cohort, and ordered sample-ID hash:

```text
5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72
```

No historical or confirmation score may influence selection.

**Why:** This gives a small confirmation set that was not used to choose `BRIDGE_075`, without opening held-out validation/test data.

### 2. Keep only four driver arms

The fixed arm order is:

```text
N, N_REPEAT, LOCAL_SWAP, BRIDGE_075
```

- `N` is untouched natural PCM.
- `N_REPEAT` has decoded PCM bytes exactly equal to `N` but is rendered independently in a separate work directory. It measures render/scorer repeatability.
- `LOCAL_SWAP` deterministically swaps the middle two quarters of natural PCM.
- `BRIDGE_075` is the single discovery-selected natural-to-TTS bridge.

**Why:** One positive control, one negative control, and one candidate answer the question. Repeating all four discovery strengths would turn confirmation back into a search.

### 3. Use a local corruption that one global offset cannot repair

For natural PCM `x` with `L` samples, define:

```text
b1 = floor(L / 4)
b2 = floor(L / 2)
b3 = floor(3L / 4)
LOCAL_SWAP = x[0:b1] || x[b2:b3] || x[b1:b2] || x[b3:L]
```

The operation reorders existing PCM samples without resampling, scaling, padding, or changing length. Its first and last quarters remain in place while the middle quarters exchange order. A single global SyncNet offset cannot align all four sections.

**Why:** The previous global delay could be absorbed by offset search. This control creates an internal sequence error instead.

### 4. Freeze the bridge construction at 75%

The bound MFA-linear waveform is used only as an acoustic target; the generated bridge is based on natural audio and remains on the natural time grid. No new TTS or alignment is run.

Use CPU float64 with:

```text
sample rate = 16000
n_fft = 1024
win_length = 1024
hop_length = 256
window = periodic Hann
center = true
pad_mode = reflect
magnitude floor = 1e-7
alpha = 0.75
```

For natural STFT `S_N` and bound MFA-linear target STFT `S_M`:

```text
log_mag = 0.25 * log(max(abs(S_N), eps))
          + 0.75 * log(max(abs(S_M), eps))
phase_N = S_N / max(abs(S_N), eps)
S_bridge = exp(log_mag) * phase_N
bridge = ISTFT(S_bridge, length=L)
```

Apply the discovery contract unchanged: one global RMS match to natural, then only if peak is at least `0.999`, one global attenuation to `0.999`, followed by one PCM16 canonicalization. Record all factors. No iterative repair, filter, denoiser, compressor, limiter, or time adjustment is allowed.

### 5. Score a six-cell matrix

Freshly render all four arms for all 22 records through the frozen Wav2Lip contract, yielding 88 videos. Score exactly these cells per record:

```text
V_N/A_N
V_N_REPEAT/A_N
V_LOCAL_SWAP/A_N
V_LOCAL_SWAP/A_LOCAL_SWAP
V_BRIDGE_075/A_N
V_BRIDGE_075/A_BRIDGE_075
```

This yields exactly 132 official SyncNet V2 cells. Every mux copies the video stream, uses 16-kHz mono PCM s16le audio, and verifies decoded PCM bytes and video elementary-stream identity.

**Why:** The control's own-audio cell proves it remains a valid driver; its natural-audio cell tests sensitivity. The bridge's own-audio cell is diagnostic, while its natural-audio cell is authoritative.

### 6. Freeze controls before interpreting the bridge

Use 10,000 deterministic bootstrap draws over the 22 source groups with seed `20260904`. All confidence bounds are two-sided percentile 95% intervals and strict inequalities.

For `N_REPEAT`, relative to `V_N/A_N`:

```text
repeat_gap_C = SyncC(V_N_REPEAT,A_N) - SyncC(V_N,A_N)
repeat_gap_D = SyncD(V_N,A_N) - SyncD(V_N_REPEAT,A_N)
```

The repeatability control passes when both lower confidence bounds are greater than `-0.10` and at least 20/22 offsets differ by no more than one frame.

For `LOCAL_SWAP`, first require its own-audio pair to be non-inferior to `V_N/A_N` by the same two `-0.10` bounds and 20/22 offset rule. Then define:

```text
damage_C = SyncC(V_LOCAL_SWAP,A_LOCAL_SWAP) - SyncC(V_LOCAL_SWAP,A_N)
damage_D = SyncD(V_LOCAL_SWAP,A_N) - SyncD(V_LOCAL_SWAP,A_LOCAL_SWAP)
```

The sensitivity control passes when both lower confidence bounds are greater than `0.10` and at least 18/22 records have both `damage_C > 0` and `damage_D > 0`.

If either control fails, the terminal decision is `CONTROL_FAILED`; bridge outcomes are reported descriptively but not interpreted.

### 7. Require movement, compatibility, and one primary gain

Before any confirmation SyncNet score is read, compute official Wav2Lip mel movement from natural toward the bound MFA-linear target:

```text
d = mel(M) - mel(N)
u = mel(BRIDGE_075) - mel(N)
progress = dot(u, d) / max(dot(d, d), 1e-12)
```

Movement passes when at least 20/22 records have `progress >= 0.15` and the lower 95% confidence bound of mean progress is greater than `0.15`.

For bridge replacement:

```text
gap_C = SyncC(V_BRIDGE_075,A_N) - SyncC(V_N,A_N)
gap_D = SyncD(V_N,A_N) - SyncD(V_BRIDGE_075,A_N)
```

Compatibility passes when the lower confidence bounds of both gaps are greater than `-0.10` and at least 20/22 offsets are within one frame of baseline. The discovery-selected primary useful-effect endpoint passes when the lower confidence bound of `gap_C` is greater than `0`. `gap_D` remains the non-inferiority safety endpoint.

The bridge is confirmed only when controls, movement, compatibility, and the primary useful-effect endpoint all pass.

## Terminal Decisions

```text
BLOCKED
CONTROL_FAILED
NATURAL_TO_TTS_BRIDGE_CONFIRMED
NATURAL_TO_TTS_BRIDGE_NOT_CONFIRMED
```

Only `NATURAL_TO_TTS_BRIDGE_CONFIRMED` sets `reference_conditioned_audio_head_spec_eligible=true`. This authorizes writing a separate OpenSpec only; it does not authorize training or deployment.

## Stage Layout

```text
runs/lrs3_natural_to_tts_bridge_confirmation_20260904/
  00_protocol/
  01_audio/
  02_videos/
  03_scores/
  04_final/
```

Stage 01 writes candidate audio and freezes Wav2Lip-mel diagnostics before Stage 03 may read any confirmation score. Every stage validates parent and upstream hashes. Resume accepts only complete cells with exact input/config/output identity. Score-based retry is forbidden.

## Risks / Trade-offs

- **Local swapping may be too severe.** That is intentional: it calibrates endpoint sensitivity, not realistic audio quality.
- **Twenty-two groups are a small confirmation cohort.** Conclusions remain fit-only and limited to this construction.
- **The primary SyncC gain was selected from discovery evidence.** It is frozen here before unused-cohort scoring and is not generalized to other endpoints.
- **A bridge may pass sync metrics without proving target-speaker identity.** Identity and perceptual quality remain outside this experiment.

## Execution Order

1. Implement protocol bindings, deterministic transforms, matrix enumeration, statistics, and independent validation with synthetic tests.
2. Run focused tests, relevant render/mux regressions, Ruff, Python compilation, and strict OpenSpec validation before media execution.
3. Lock Stage 00 and verify zero score reads, zero media decodes during selection, and no sealed access.
4. Generate all audio and freeze movement diagnostics.
5. Render 88 videos and score the complete 132-cell matrix without outcome-based retry.
6. Evaluate controls first, then the bridge, and write one self-hashed terminal artifact.
7. Independently validate Stages 00-04 and report the registered decision without broader claims.
