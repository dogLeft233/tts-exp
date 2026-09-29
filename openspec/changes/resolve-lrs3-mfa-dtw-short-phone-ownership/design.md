## Context

The original protocol used one exclusive owner per WavLM frame based on whether the frame center fell inside an MFA phone interval. A 20 ms feature grid cannot guarantee a center inside every short phone, even when the phone is temporally covered by a neighboring frame's nominal support. The resulting 11 failures were all matched speech phones with natural frames and no TTS center-owned frame.

## Goals / Non-Goals

**Goals:**

- Make short-phone handling explicit, deterministic, and auditable before downstream scoring.
- Keep the phone-instance constraint and the TTS-only emitted conditioning contract unchanged.
- Preserve the original 133-record cohort and all downstream promotion/replacement gates.
- Make feature-tail behavior explicit rather than silently extrapolating.

**Non-Goals:**

- No cross-phone DTW path, label substitution, nearest-frame fallback, linear fallback, band relaxation, model training, or score-based record selection.
- No change to historical MFA-linear artifacts or the first blocked run.

## Decisions

### 1. Use nominal half-stride support intervals

For frame index `i`, sample rate 16 kHz, and stride 320 samples, define its nominal support as:

```text
[(i + 0.5) * stride / sample_rate - stride / (2 * sample_rate),
 (i + 0.5) * stride / sample_rate + stride / (2 * sample_rate))
```

A TTS frame belongs to a phone-local source set when the support interval and that exact MFA3 phone interval have positive temporal overlap. This is a representation rule for selecting source indices, not a change to the WavLM extractor or decoder.

### 2. Permit shared boundary support, not cross-phone ownership

The source sets are intentionally not an exclusive partition. If one nominal support interval overlaps two adjacent phone intervals, that frame index appears in both local sets. No frame is added to a phone unless its nominal support overlaps that phone's interval, and each DTW path still runs only over one matched TTS token instance. Shared indices are recorded in diagnostics.

This resolves a short phone at a discretization boundary without borrowing a frame from a non-adjacent or differently labeled instance. It also avoids making the result depend on an arbitrary earlier/later tie choice.

### 3. Keep natural frame rows and silence policy unchanged

Natural frame rows continue to use the existing exclusive center-time `frame_owners` contract so each output natural frame is represented exactly once. Silence output rows continue to be copied from MFA-linear. Only the TTS source-frame set used by matched speech-phone DTW changes.

### 4. Do not extrapolate feature tails

A TTS phone with no positive-overlap nominal support remains an engineering failure. Frames beyond the final nominal support interval are unowned and are never assigned to the final phone by padding, clamping, nearest-frame selection, or extrapolation. The complete 133-record candidate matrix is still required before Wav2Lip or SyncNet.

### 5. Run the revision in a new protocol and run root

The revision uses protocol ID `lrs3_mfa_dtw_short_phone_support_20260904` and run root `runs/lrs3_mfa_dtw_replacement_short_phone_20260904`. Stage 00 is rebuilt from the same hash-bound parents and ordered cohort. The previous `lrs3_mfa_dtw_replacement_20260904` run is not modified or resumed under the new rule.

## Validation

Focused tests cover shared boundary support, a matched short TTS phone producing a valid hard-DTW path, and rejection of unknown ownership policies. The rerun must report the policy, support definition, tail policy, shared-frame count, and any failures in machine-readable artifacts.
