## Context

See `proposal.md` for motivation and `specs/mfa-linear-real-video-sync-prototype/spec.md` for the behavior contract.

The usable historical implementation is split across two places. The current tree contains an identity-initialized exact-length waveform model in `scripts/pnp_audio_enhancer.py::ResidualTCN`, the LRS3 MFA-linear renderer and tests, SyncNet V2 third-party code, and fit-only MFA-linear artifacts. The historical worktree `.claude/worktrees/lrs3-wavlm-resynthesis-50/` contains a well-tested differentiable SyncNet V2 MFCC/window/curve seam and the prior target-margin mathematics. The new package must promote the small required seams into the current tree; runtime imports from another worktree are forbidden.

The previous cross-attention path used natural WavLM features as query, TTS WavLM features as key/value, a frozen HiFi-GAN decoder, a differentiable Wav2Lip proxy, and SyncNet after natural-audio replacement. That path entangled conditioning, resynthesis, generated-video transfer, and the loss. This prototype removes all four complications: an MFA-linear waveform goes directly through an identity residual model, and its audio embedding is compared directly with a fixed real-video embedding. The only remaining proxy/official gap should be differentiable-versus-file PCM/MFCC execution.

## Goals / Non-Goals

**Goals:**

- Make the TTS-only hypothesis identifiable: the trainable forward signature has one waveform tensor and no sample-specific side channel.
- Make the SyncNet objective mathematically strong enough that satisfying it entails improvement in both raw official-style Sync-D and Sync-C relative to the exact step-0 MFA-linear baseline.
- Reuse fixed 96-frame LRS3 tracked-video segments and existing MFA-linear waveforms so P0 and P1 can be implemented and run quickly.
- Detect the common silent failures: wrong offset sign, 4:1 MFCC/video misindexing, cosine substitution, candidate self-pairing, detached audio embeddings or wrong offsets, BatchNorm mutation, float/PCM scale mismatch, and proxy/file disagreement.
- Produce a one-record constructability answer after exactly 20 updates; retain an explicitly optional four-record shared check without presenting it as validation.

**Non-Goals:**

- Natural-audio conditioning, natural/TTS cross-attention, WavLM, HuBERT, HiFi-GAN, Wav2Lip, TFG, a learned aligner, or a waveform time-warp operator.
- Held-out validation, population inference, architecture or hyperparameter search, or comparison across TTS systems.
- ASR/PER, speaker verification, MOS, perceptual quality, or a claim that the candidate preserves content or identity.
- Replacement evaluation. Fixed real video plus candidate audio answers an audio-to-real-video alignment question, not whether a candidate drives a generated face whose gain survives original-audio replacement.

## Decisions

### 1. Use the existing waveform residual model directly

Instantiate the current `ResidualTCN` with a frozen prototype configuration:

```text
input/output                 [B,1,N] normalized waveform
channels                     32
dilations                    (1,2,4,8,16,32,64,128)
kernel                       3, two convolutions per block
normalization/activation     existing GroupNorm/GELU implementation
residual_scale               0.05
output projection            exact zero initialization
trainable input modalities   mfa_linear_tts_waveform only
```

The model computes `candidate = input + 0.05*tanh(residual_logits)` and therefore has exact identity at step 0. Its non-causal receptive field is sufficient for a small local acoustic/timing test while remaining far smaller and easier to audit than the old cross-attention/resynthesis stack. The prototype does not add frame features, video conditioning, positional embeddings, speaker IDs, or a second sequence.

Alternative considered: feed MFA-linear TTS through frozen WavLM, replace cross-attention with a feature TCN, and decode with HiFi-GAN. Rejected because step 0 would be codec resynthesis rather than the actual MFA-linear baseline, and decoder/front-end gradients would reintroduce a second failure surface. Alternative considered: reuse cross-attention with TTS as both query and key/value. Rejected because it does not perform the requested ablation and preserves unnecessary attention collapse modes.

### 2. Freeze one known P1 record and one deterministic optional P2 set

The parent asset root is configurable only as a path, but its content is hash-locked. The default roots are:

```text
policy records:
.claude/worktrees/lrs3-wavlm-resynthesis-50/
  tmp/lrs3_policy_a1_200_20260828/policy_cohort/records/

MFA-linear summary/audio:
runs/lrs3_cem_fixed_video_20260820/12_policy_train_mfa_linear/
```

P1 is fixed to:

```text
lrs3_6ORDQFh0Byw_00004
source_group = 6ORDQFh0Byw
protocol_split = train
original video frames = [0,96)
audio samples = [0,61440)
```

Its current MFA-linear parent is `lrs3_6ORDQFh0Byw_00004.wav`, hash `191b3bfeedb63c18da94b5ef80d460442141873e77f73c81b5b085b8b2b0b245`; preflight must verify rather than trust these values.

If P2 is explicitly enabled, use exactly these four records, all required to retain `protocol_split=train`, verified 96-frame alignment, distinct source groups, exact parent MFA-linear length, and an unambiguous natural target curve:

```text
lrs3_6ORDQFh0Byw_00004  / 6ORDQFh0Byw
lrs3_70VZ1SOzSnc_00003  / 70VZ1SOzSnc
lrs3_6xtmm0MnaS0_00006  / 6xtmm0MnaS0
lrs3_73jPh0eRPSY_00002  / 73jPh0eRPSY
```

The latter three are the first eligible new source groups under ascending:

```text
sha256("mfa-linear-real-video-sync-v1\0" + source_group + "\0" + sample_id)
```

with the P1 group forced first. Persist the computed keys. Do not replace a missing or newly ineligible record. These records are historical fit data and are optimized and scored in place; P2 is a shared-fit check, not a held-out test.

Alternative considered: choose four records after viewing step-0 scores. Rejected because it would make even the fixed-data result outcome-selected. Alternative considered: build a new train/validation cohort. Rejected as beyond the rapid constructability question.

### 3. Materialize a lossless, fixed real-video segment once

For each selected record, use its already verified policy frame mapping to decode exactly 96 frames from the real tracked face crop. Materialize one canonical 25 fps, lossless FFV1 AVI segment and record every decoded BGR frame hash. Pair it with exactly 61,440 samples. Candidate conditions replace only the audio stream; the canonical video stream and decoded frame hashes must remain identical.

The differentiable visual path reads these same decoded BGR uint8 frames, preserves the official `[0,255]` float scale and channel order, forms `[i,i+5)` windows, and runs frozen `SyncNetModel.forward_lip()` under `no_grad`. Cache only the resulting detached real-video embeddings. Do not run a detector or derive a new crop per checkpoint.

The official verifier creates a condition AVI by stream-copying the canonical FFV1 video and muxing signed PCM16 candidate audio, verifies demuxed PCM and decoded frame hashes, then invokes the unmodified official SyncNet V2 scoring path on that fixed crop. It does not rerun face detection/tracking; otherwise the visual coordinate would no longer be fixed.

Alternative considered: run `run_pipeline.py` independently for every candidate. Rejected because detector/tracker and re-encode differences could masquerade as audio-model movement. Alternative considered: supervise from the full uncropped source video. Rejected because SyncNet expects its tracked face crop and the loss must match the official visual input.

### 4. Use natural audio only to calibrate the frozen coordinate label

The model and optimizer never load natural audio. During lock construction only, take the exact natural PCM segment corresponding to the same 96 frames and compute a detached 31-point natural/real-video curve. Define internal signed shift by:

```text
visual coordinate i ↔ audio coordinate i+s
curve index = s+15
official reported AV offset = -s
```

Set `s_ref` to the unique natural-curve minimum. Require a best-versus-second gap greater than `2q`, where `q=0.001`. Persist the curve, `s_ref`, opposite-sign official value, hashes, and gap. Then construct the training record without a natural-audio path or tensor. A loader-level assertion rejects extra natural fields.

The existing policy record can be used as provenance and a cross-check, but the target curve must be recomputed for the exact 96-frame segment rather than silently reusing a full-track curve. If the recomputed target offset differs from stored metadata, fail closed and investigate the segment/sign mapping.

This scalar calibration is necessary because SyncNet's convention need not place the natural pair at internal shift zero. Hard-coding zero would be a positive-label bug, while deriving the target from the candidate curve would let the model move its own label. Per-step or per-checkpoint natural curves are forbidden.

### 5. Make the PCM/MFCC forward path file-exact

Promote the historical differentiable MFCC and window logic into the new package, with focused parity tests. The normalized candidate is converted to a straight-through PCM tensor as:

```text
scaled = clamp(candidate * 32768, -32768, 32767)
quantized = round(scaled)
pcm_st = scaled + stop_gradient(quantized - scaled)
```

The forward value is therefore exactly the signed PCM16 integer written to disk, while the backward derivative through unsaturated values is one. MFCC uses float64 internally to match `python_speech_features`, then returns the requested training dtype. It must retain the official rectangular frame window; substituting Hann is an error.

For `F=96`, `N=61440`, both modalities produce `W=91` synchronized windows. Audio window `i` starts at MFCC frame `4i`, and the video window starts at frame `i`. Both embeddings have shape `[91,1024]`. The curve uses the official zero-padded audio convention over internal shifts `[-15,+15]` and raw `pairwise_distance`; do not normalize embeddings and do not use cosine similarity.

P0 parity has three levels:

1. MFCC arrays versus `python_speech_features` on deterministic fixtures, `atol/rtol <= 2e-5`;
2. proxy embeddings/31-point curve versus the same PCM and visual frames through an independent non-differentiable local path;
3. proxy curve versus mux/demux plus official file scorer, max absolute error `<=q`.

The third check catches serialization, scale, stream, and frame extraction errors that a tensor-only test cannot.

### 6. Optimize a hard target-margin objective, not cosine or positive-pair loss alone

Let `c0` be the detached step-0 curve from the exact identity output and let `s_ref` come only from Decision 4. Freeze the formulas in the capability spec:

```text
D0 = min(c0)
C0 = median(c0) - D0
ΔD = max(3q, 0.01D0)
ΔC = max(3q, 0.01C0)
D_goal = D0 - ΔD
C_goal = C0 + ΔC

L_D = relu((c[s_ref] - D_goal) / max(D0,1e-3))
L_R = mean_{s != s_ref} relu(
        (C_goal - (c[s] - c[s_ref])) / max(C0,1e-3))
L_sync = L_D + L_R
```

All 31 elements of `c` remain live in autograd. Unlike the old local ranking helper, wrong-offset distances are not detached. This matters because the claim includes target uniqueness and Sync-C, which depends on the rest of the curve. Unit tests construct D-only, one-negative-only, combined, tie, boundary-offset, and already-satisfied curves and check values plus gradient signs.

If `L_sync=0`, the target distance is below the pristine best by `ΔD`, all 30 alternatives exceed the target by at least `C_goal`, the target is the unique argmin, and the median gap exceeds pristine `C0` by `ΔC`. This is the exact claim the loss is allowed to make. Ordinary `1-cos`, BCE on a positive pair, local offsets only, or minimizing `c[s_ref]` alone cannot establish it and are prohibited.

### 7. Use one small preservation hinge and explicit waveform QC

Compute the existing Wav2Lip-compatible normalized log-mel for the candidate and MFA-linear input only:

```text
E_mel = mean(abs(mel(candidate)-mel(input)))
L_trust = relu((E_mel-0.10)/0.10)
L_total = L_sync + L_trust
```

Natural mel is not used. The hinge is inactive near identity and prevents unrestricted pursuit of SyncNet distance. The model's `residual_scale=0.05` supplies a second hard pointwise bound. Persist residual RMS/peak, input/candidate RMS/peak, mel distance, PCM saturation count/fraction, and finite checks every step. Any final mel distance above `0.10`, changed length, non-finite value, or saturation fraction above `1e-4` invalidates the checkpoint even if sync improves.

Alternative considered: no preservation term. Rejected because the full-curve loss can reward adversarial audio. Alternative considered: natural-mel reconstruction. Rejected because it would add natural-audio supervision contrary to this TTS-only experiment and would confound sync with acoustic imitation.

### 8. Freeze SyncNet while preserving gradients to the model

The fixed visual embedding may be computed with `inference_mode` and detached. The candidate path must be:

```text
ResidualTCN candidate
→ PCM16 straight-through
→ differentiable MFCC
→ 91 audio windows
→ frozen SyncNet forward_aud (eval, grad enabled for input)
→ live 31-point curve
→ L_total
```

Set every SyncNet parameter `requires_grad=False`, keep all SyncNet BatchNorm layers in eval, and exclude SyncNet from the optimizer. Do not wrap `forward_aud` in `no_grad` or detach its result. Hash SyncNet parameters and buffers before and after every stage.

Because the residual output projection starts at zero, the first backward is expected to update that projection while upstream residual-body gradients can be zero. P0 therefore checks a disposable one-step copy: step-0 output-projection gradient is finite/nonzero; after one update and a second backward, at least one upstream parameter gradient is finite/nonzero. The copy is discarded. P1 then starts from a fresh state under the same seed, so P0 does not consume an experimental update.

### 9. Freeze optimizer and execution constants

Use one configuration without search:

```text
seed                 20260903
device                CUDA, required and recorded
dtype                 float32; MFCC internal float64
AMP/scheduler         absent
PyTorch determinism   deterministic algorithms on; cuDNN benchmark off
optimizer             AdamW
learning_rate         1e-4
betas/epsilon         PyTorch defaults, serialized explicitly
weight_decay          0.01
gradient clipping     global norm 1.0
P1                     one record, one full segment per step, 20 steps
P2                     four records every step, mean record loss, 100 steps
checkpoint selection  step 0 and final only
```

For P2, process records in the frozen listed order, divide each record loss by four, accumulate gradients, then take one optimizer step. This avoids a large batched SyncNet graph while making every step cover all records. No early stop, retry seed, warm start, per-record adapter, random crop, sampler, loss reweighting, or post-result constant change is allowed.

The CUDA requirement is an execution choice for a rapid full-chain gradient prototype. If unavailable, implementation/tests can complete but the real stage records `BLOCKED_CUDA_UNAVAILABLE`; it does not silently switch numerical regimes.

### 10. Separate engineering, P1, and optional P2 decisions

P0 contains pure mathematics, indexed-window fixtures, official frontend/file parity, model identity, disposable gradient/freeze checks, and artifact validation. Any failure is terminal engineering `NO_GO`.

P1 trains and scores `lrs3_6ORDQFh0Byw_00004` on the same fixed 96-frame segment. This intentional fit-and-score loop asks only whether the objective is constructible. Acceptance requires all predicates in the capability spec, including official raw curve margins and parity. Preserve both failed and successful outputs.

P2 is disabled by default and requires both a P1 pass and an explicit `--run-p2`-style opt-in. It starts fresh and trains one shared adapter on the fixed four records. Its three-of-four joint-win and median gates test whether the same mechanism is shared across a tiny fixed set, not whether it transfers.

Use explicit failure statuses rather than a generic fail, at minimum:

```text
INPUT_LOCK_FAILURE
NATURAL_INPUT_LEAKAGE
VISUAL_COORDINATE_MISMATCH
TARGET_OFFSET_AMBIGUOUS
OFFSET_SIGN_MISMATCH
MFCC_PARITY_FAILURE
FILE_PROXY_PARITY_FAILURE
IDENTITY_FAILURE
FROZEN_STATE_MUTATION
NO_CANDIDATE_GRADIENT
NONFINITE_TRAINING
TRUST_REGION_FAILURE
NO_PROXY_LOSS_DESCENT
OFFICIAL_D_MARGIN_FAILURE
OFFICIAL_C_MARGIN_FAILURE
OFFICIAL_OFFSET_FAILURE
PROXY_OFFICIAL_TRANSFER_FAILURE
FIXED_DATA_TTS_ONLY_SYNC_CONSTRUCTED
FIXED_DATA_SHARED_TTS_ONLY_SYNC_CONSTRUCTED
P2_NOT_REQUESTED
```

### 11. Keep code and artifacts compact

Proposed package and run layout:

```text
scripts/experiments/mfa_linear_real_video_sync/
├── config.py        # frozen constants, records, paths, hashes
├── protocol.py      # locks, fixed real-video segment, input isolation
├── syncnet_loss.py  # PCM16-ST, MFCC/windows/curve, target-margin loss
├── train.py         # P1/P2 fixed-step optimizer
├── evaluate.py      # lossless mux, official curve, predicates
└── run.py           # P0/P1 and explicit optional P2 orchestration

tests/experiments/mfa_linear_real_video_sync/
├── test_protocol.py
├── test_syncnet_frontend.py
├── test_target_margin_loss.py
├── test_gradient_freeze.py
└── test_decision.py

runs/lrs3_mfa_linear_real_video_sync_20260903/
├── 00_lock/
├── 01_p0_seam/
├── 02_p1_one_record/{step0,step20}/
├── 03_p2_shared_four/          # absent unless explicitly requested
├── logs/
├── validation.json
└── decision.json
```

Reuse current JSON/hash/audio helpers where they reduce code, but do not add a generic experiment framework, registry, database, dashboard, service, or third-party fork. Scientific constants are serialized in config and are not exposed as convenient tuning flags. Allowed CLI variation is limited to parent asset root, output path, stage/resume, device assertion, and explicit P2 opt-in.

## Risks / Trade-offs

- **[The model can exploit SyncNet without improving human-perceived sync]** → Keep residual and log-mel bounds, preserve full audio/QC, and limit the claim to frozen-scorer constructability.
- **[Natural-reference offset introduces natural-derived information]** → Use it once only as a detached coordinate label; prohibit natural tensors from the model, preservation loss, and training loader; report this distinction explicitly.
- **[A waveform TCN is not an explicit monotonic time warp]** → Treat a negative result as evidence about this small residual model and loss, not proof that MFA-linear cannot be improved.
- **[Official zero padding can make edge windows affect the curve]** → Match the official curve exactly and use the same 96-frame support for every condition; do not compare curves with different lengths.
- **[Full-curve ranking can increase wrong-offset distances rather than reduce the positive distance]** → Include a separate absolute `L_D` goal and require official D improvement, not ranking alone.
- **[P1 and P2 reuse optimization records]** → Label both fixed-data constructability; do not bootstrap or claim validation/generalization.
- **[Historical assets live partly in another worktree]** → Hash-lock inputs, promote code seams into the current tree, import no runtime code from that worktree, and fail if assets disappear.
- **[CUDA nondeterminism or file container behavior can obscure tiny margins]** → Use deterministic settings, PCM/frame identity checks, complete curves, and margins larger than three output quanta.

## Migration Plan

This is additive experimental code; no production migration is required.

1. Promote the minimal frontend/loss seams and add pure P0 tests.
2. Implement input locking, fixed-video materialization, and independent official parity.
3. Connect the existing identity waveform model and prove freeze/gradient behavior on a disposable copy.
4. Implement fixed-step training and machine-recomputable decision logic.
5. Run P0, then P1 exactly once if P0 passes.
6. Leave P2 unrun unless explicitly requested after a P1 pass.
7. Run focused tests and `openspec validate prototype-mfa-linear-real-video-sync --strict`.
8. On failure, preserve artifacts and stop; rollback consists only of removing the additive package/change in a separate authorized action, never rewriting prior runs.
