## Purpose

Defines the protocol revision for constructing LRS3 same-phone hard-DTW candidates when a valid short MFA3 phone contains no WavLM frame center.

## ADDED Requirements

### Requirement: Short-phone TTS support is deterministic and phone-local
The revised experiment SHALL represent WavLM frame `i` by the nominal half-stride support interval centered at `(i + 0.5) * 320 / 16000` seconds. A TTS frame SHALL be included in a matched MFA3 phone's source set only when its nominal support has positive temporal overlap with that exact phone interval. Source sets SHALL be allowed to share a frame only between adjacent overlapping phone intervals; no source frame may be imported from another phone instance, label, or non-overlapping interval.

#### Scenario: Short phone overlaps a boundary frame
- **WHEN** a matched speech phone has natural frames but no TTS frame center inside its interval, and a nominal TTS frame support overlaps the phone
- **THEN** the frame is included in that phone's local TTS source set and same-phone hard-DTW proceeds without nearest-frame or linear fallback

#### Scenario: Boundary support is shared
- **WHEN** one nominal frame support overlaps two adjacent phone intervals
- **THEN** the frame index may occur in both local source sets, both paths remain independently endpoint-constrained to their respective matched instances, and the shared-frame count is recorded

#### Scenario: No nominal support exists
- **WHEN** a matched speech phone has no positive-overlap TTS nominal support, including a phone wholly beyond the final nominal feature support
- **THEN** the record is rejected as engineering `BLOCKED` and no fallback, extrapolation, or downstream score is created

### Requirement: The ownership revision does not change scientific comparison gates
The revised run SHALL use the same ordered 133-record cohort, parent hashes, WavLM/HiFi-GAN/Wav2Lip/SyncNet assets, hard-DTW band and tie order, silence policy, candidate PCM contract, source-group bootstrap, promotion gate, and strict replacement authorization as the original comparison. It SHALL use a new protocol ID and run root and SHALL NOT mutate the original blocked run.

#### Scenario: Revised Stage 00 binds the same cohort
- **WHEN** revised Stage 00 is built
- **THEN** it records the original parent bindings and ordered cohort hash/count plus the new support and feature-tail policies, without reading validation/test media or using downstream outcomes for membership

#### Scenario: Complete candidates authorize downstream work
- **WHEN** all 133 revised candidates pass their audio, provenance, and mapping checks
- **THEN** Stage 01 is `GO` and only then may the original diagonal Wav2Lip/SyncNet stage run

#### Scenario: Incomplete revised candidates remain sealed
- **WHEN** any revised cohort record fails candidate construction or validation
- **THEN** Stage 01 is engineering `BLOCKED`, the scientific decision is unavailable, and no diagonal or replacement score is authorized
