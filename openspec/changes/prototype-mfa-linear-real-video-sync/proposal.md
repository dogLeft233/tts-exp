## Why

The prior cross-attention experiment coupled natural-audio conditioning, TTS conditioning, resynthesis, and SyncNet supervision, so a failure could not distinguish an unnecessary conditioning/alignment mechanism from a bad synchronization loss. A smaller prototype is needed to test the direct question: can an identity-initialized waveform model improve already exact-length MFA-linear TTS audio against the absolute time coordinates of its paired real video, with no natural-audio model input and no cross-attention?

## What Changes

- Add a narrow LRS3 prototype whose only trainable-model input is mono 16 kHz, exact-length MFA-linear TTS waveform; natural waveform/features and cross-attention are absent from the model and training graph.
- Reuse the existing identity-initialized waveform `ResidualTCN` rather than adding an encoder, vocoder, attention block, or general training framework.
- Supervise the candidate waveform with frozen SyncNet V2 real-video embeddings and a differentiable PCM16/MFCC/audio-window path that is numerically checked against the official scorer.
- Optimize one full 31-offset target-margin objective, anchored to the same record's step-0 MFA-linear curve, so zero primary loss implies a strictly better target distance and target-offset separation than step 0. A detached natural-reference offset may calibrate SyncNet's coordinate convention but natural audio may not condition the adapter or enter the candidate audio branch.
- Add a small log-mel trust-region hinge and waveform integrity gates to limit trivial/adversarial SyncNet solutions.
- Gate execution in three small stages: mathematical/frontend/gradient tests, one fixed-record 20-step overfit, and an optional exact-four-record shared 100-step prototype only after the one-record stage passes.
- Verify the trained candidate with the official file-level SyncNet V2 path on the same frozen real-video track and preserve complete distance curves, hashes, losses, gradients, and terminal decisions.
- Keep claims intentionally narrow: a pass establishes fixed-data constructability of the TTS-only real-video objective, not held-out generalization, perceptual quality, Wav2Lip/TFG gain, or replacement safety.

## Capabilities

### New Capabilities

- `mfa-linear-real-video-sync-prototype`: Train and verify a minimal TTS-only waveform residual model using a scorer-aligned frozen-SyncNet loss against paired real video, with explicit coordinate, frontend-parity, gradient, preservation, and staged prototype gates.

### Modified Capabilities

None.

## Impact

- Adds a compact experiment package under `scripts/experiments/mfa_linear_real_video_sync/` and focused tests under `tests/experiments/mfa_linear_real_video_sync/`.
- Reuses `scripts/pnp_audio_enhancer.py::ResidualTCN`, the existing LRS3 MFA-linear renderer/assets, the frozen SyncNet V2 checkpoint, and proven differentiable MFCC/window/curve logic from the historical `lrs3_syncnet_finetune` worktree where available; required seams are copied or promoted explicitly rather than imported from a disposable worktree.
- Reads hash-locked fit-only LRS3 policy/cohort assets and existing real-video track geometry; it does not modify prior runs, checkpoints, source media, or third-party code.
- Writes a new immutable prototype run with input locks, frontend parity evidence, detached visual/reference curves, step-0 anchors, per-step loss/gradient records, official curves, QC, and separate engineering/prototype decisions.
- Adds no production API and changes no existing experiment result. The optional four-record stage is not a validation split and cannot support a generalization claim.
