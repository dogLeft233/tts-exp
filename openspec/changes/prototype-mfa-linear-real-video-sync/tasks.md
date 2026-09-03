## 1. Package and Input Locks

- [x] 1.1 Create `scripts/experiments/mfa_linear_real_video_sync/` and its test package with the frozen constants, P1/P2 record IDs, allowed CLI fields, and a one-waveform training-record schema; verify a schema test rejects natural, video-conditioning, identity, transcript-feature, and second-audio fields.
- [x] 1.2 Implement hash-validated loading of the existing MFA-linear summary/audio and policy metadata, including exact `protocol_split=train`, source-group, 25 fps, 96-frame, 61,440-sample, mono 16 kHz, finite, and exact-length checks; verify protocol tests reject every mismatched field without padding, trimming, resampling, or substitution.
- [ ] 1.3 Materialize one lossless frozen real-video tracked segment per selected record and persist source/geometry/frame/segment hashes; verify step-0 and candidate conditions decode to the same 96 BGR frame hashes.
- [ ] 1.4 Build the one-time natural-reference coordinate artifact, freeze the signed `s_ref`, full 31-point curve and provenance, then remove the natural path/tensor from the training record; verify sign fixtures at `-15/0/+15`, stored-offset consistency, ambiguous-gap rejection, and a loader assertion that no natural signal reaches training.

## 2. SyncNet Frontend and Loss

- [x] 2.1 Promote the minimal PCM16 straight-through and differentiable SyncNet V2 MFCC logic into the new package without runtime imports from the historical worktree; verify deterministic random, speech-like, silence, near-clipping, and minimum-length fixtures match `python_speech_features` within `atol/rtol <= 2e-5`.
- [x] 2.2 Implement exact SyncNet audio/video windows and the official zero-padded 31-offset raw-Euclidean curve; verify indexed fixtures produce 91 windows for 96 frames/61,440 samples, use audio start `4i`, and map internal shift `s` to curve index `s+15` and official offset `-s`.
- [x] 2.3 Implement the frozen step-0 goals and full-curve `L_D + L_R` target-margin loss exactly as specified; verify D-only, one-negative-only, combined, satisfied, tied, and boundary-offset fixtures prove zero-loss D/C implications and finite nonzero gradients on all involved live curve values.
- [x] 2.4 Implement candidate-to-MFA normalized log-mel trust loss and waveform/PCM QC; verify tests cover the `0.10` hinge boundary, exact length, finite values, residual bound, and `1e-4` saturation-fraction rejection.

## 3. Model and Gradient Contract

- [x] 3.1 Integrate the existing `ResidualTCN` with channels 32, dilations `(1,2,4,8,16,32,64,128)`, residual scale `0.05`, and one `[B,1,N]` input; verify fresh output and direct MFA-linear PCM/curve are exact at step 0 and no attention, encoder, vocoder, video, or second-audio module is instantiated.
- [x] 3.2 Implement frozen SyncNet loading and detached cached real-video embeddings while leaving the candidate `forward_aud` input path differentiable; verify SyncNet stays in eval, every parameter has `requires_grad=False`, no SyncNet optimizer parameter exists, and parameter/buffer hashes remain unchanged.
- [x] 3.3 Add the disposable two-backward gradient smoke: verify step-0 output-projection gradient is finite/nonzero, an upstream model gradient becomes finite/nonzero after one throwaway update, candidate waveform/embedding are not detached, and the real P1 model starts again from the frozen initial-state hash.

## 4. Training, Official Verification, and Decisions

- [x] 4.1 Implement deterministic AdamW execution with seed `20260903`, CUDA float32, explicit default betas/epsilon, learning rate `1e-4`, weight decay `0.01`, global clip `1.0`, no AMP/scheduler, P1 exactly 20 steps, and P2 exactly 100 all-four-record mean-loss steps; verify config serialization and tests reject retries, checkpoint search, warm starts, random sampling/crops, or changed constants.
- [x] 4.2 Implement lossless FFV1-video/PCM16-audio condition muxing and official SyncNet V2 full-curve scoring without rerunning face tracking; verify demuxed PCM identity, decoded-frame identity, model hash, window count, and proxy-versus-official per-offset error `<= q`.
- [x] 4.3 Implement P0/P1/P2 predicate evaluation and explicit failure precedence, including proxy descent, D/C margins, target offset/gap, trust/QC, frozen-state, and three-of-four P2 aggregation; verify decision tests recompute each terminal status from synthetic artifacts and never promote a proxy-only improvement.
- [x] 4.4 Implement create-once run artifacts and hash-valid resume for locks, P0 evidence, step histories, checkpoints, PCM, proxy/official curves, QC, validation, and one terminal decision; verify corrupted, partial, stale, or conflicting cells fail closed rather than being overwritten or reused.

## 5. Verification and Fixed-Data Execution

- [x] 5.1 Run the focused new test suite plus existing `pnp_audio_enhancer` and LRS3 MFA-linear regression tests; verify all selected tests pass and record the exact commands and versions in `01_p0_seam/test_results.json`.
- [x] 5.2 Run P0 frontend/file parity, identity, mathematics, freeze, and disposable-gradient checks; verify every mandatory predicate passes before creating any P1 optimizer history, otherwise emit engineering `NO_GO` and stop.
- [x] 5.3 If P0 passes, run P1 once on `lrs3_6ORDQFh0Byw_00004` for exactly 20 steps and official-score only steps 0 and 20; verify artifacts yield either `FIXED_DATA_TTS_ONLY_SYNC_CONSTRUCTED` or one explicit negative status with no retry or threshold change.
- [x] 5.4 Implement P2 behind explicit opt-in; if and only if P1 passed and the invocation explicitly requests P2, run the fresh exact-four-record 100-step shared prototype and verify its fixed predicates, otherwise record `P2_NOT_REQUESTED` without running it.
- [x] 5.5 Produce a concise run report that labels all results as fixed-data constructability only, then run `openspec validate prototype-mfa-linear-real-video-sync --strict` and verify both the OpenSpec change and machine-readable run-artifact validator pass.
