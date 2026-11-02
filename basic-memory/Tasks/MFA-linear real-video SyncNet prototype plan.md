---
title: MFA-linear real-video SyncNet prototype plan
type: task
permalink: tts-exp/tasks/mfa-linear-real-video-sync-net-prototype-plan
status: in-progress
source_of_truth:
- openspec/changes/prototype-mfa-linear-real-video-sync/
- openspec/changes/evaluate-mfa-linear-sync-transfer-and-replacement/
validated_on: '2026-09-03'
tags:
- openspec
- prototype
- mfa-linear
- syncnet
- real-video
- tts-only
- p1
- p2
- transfer
- replacement
- wav2lip
---

# MFA-linear real-video SyncNet prototype and transfer plan

## Current state

The first change, `openspec/changes/prototype-mfa-linear-real-video-sync/`, is implemented and strictly valid. Its trainable path is TTS-only: exact-length mono 16 kHz MFA-linear waveform → identity `ResidualTCN` → PCM16 straight-through → differentiable official-equivalent MFCC → frozen SyncNet audio branch → full 31-offset Euclidean curve against fixed real-video embeddings. Natural audio is used only upstream for MFA construction and one detached exact-segment target-offset calibration; it is absent from the training record, model, trust loss, and optimizer graph.

The corrected immutable P1 run is:

```text
runs/lrs3_mfa_linear_real_video_sync_20260903_v4/
```

It passed `FIXED_DATA_TTS_ONLY_SYNC_CONSTRUCTED` for `lrs3_6ORDQFh0Byw_00004` after exactly 20 AdamW steps. Focused tests and regressions passed (`89 passed`), official/proxy per-offset error stayed below `q=0.001`, SyncNet remained frozen, candidate gradients were live, and waveform QC passed.

```text
Sync-D: 5.9715294838 -> 5.6018080711  (gain 0.3697214127)
Sync-C: 5.4751944542 -> 5.8830161095  (gain 0.4078216553)
proxy L_sync: 0.1351340562 -> 0.1008756384
```

P2 remains `P2_NOT_REQUESTED`; the sealed P1 v4 run must never be appended to.

## Next-stage specification

A separate planning-only change now defines the post-P1/P2 evaluation:

```text
openspec/changes/evaluate-mfa-linear-sync-transfer-and-replacement/
```

All four OpenSpec artifacts are complete and strict validation passed on 2026-09-03. The change deliberately separates:

1. P1 one-record constructability (already passed).
2. P2 fresh shared-four fixed-data fitting (existing implementation, still must be run explicitly in a new immutable root).
3. Adapter-heldout real-video transfer on exactly eight records from eight source groups disjoint from P2.
4. Strict downstream replacement through one frozen Wav2Lip/SyncNet stack.

The follow-up cannot start scientific evaluation unless a fresh P2 run validates as `FIXED_DATA_SHARED_TTS_ONLY_SYNC_CONSTRUCTED`. The evaluation cohort is selected by a fixed SHA256 rule before candidate inference, is source-group-disjoint from P2, and is called adapter-heldout rather than dataset-heldout.

## Real-video transfer gate

The frozen P2 adapter is evaluated once with no optimizer, retry, checkpoint search, or side channel. Its only model input remains the MFA-linear waveform. Baseline and candidate are scored on identical 96-frame real-video coordinates using complete official 31-point curves.

A record succeeds only if both official Sync-D and Sync-C gains are at least `3q=0.003`, the candidate unique minimum matches the detached natural target offset, and all QC/provenance checks pass. Promotion requires at least 6/8 record successes and median D/C gains each at least `0.003`. This is descriptive empirical transfer, not population inference.

## Strict replacement gate

Replacement runs only if real-video transfer passes. On the same eight records, one frozen Wav2Lip checkpoint renders three driver arms:

- `G_N`: natural driver;
- `G_B`: MFA-linear baseline driver;
- `G_C`: adapted candidate driver.

Each generated visual row is canonicalized once to lossless FFV1. Three evaluation audios (`E_N`, `E_B`, `E_C`) form a complete 3×3 matrix, 9 cells per record and 72 cells total. The only authoritative contrast holds natural evaluation audio fixed:

```text
D gain = D(G_B_E_N) - D(G_C_E_N)
C gain = C(G_C_E_N) - C(G_B_E_N)
```

`G_N_E_N` defines the downstream oracle offset. Native diagonal and cross-audio cells are diagnostics only; they cannot promote the experiment. Replacement promotion again requires at least 6/8 successes and median D/C gains each at least `0.003`.

## Claim boundary

The P1 pass establishes only fixed-data one-record constructability. A future P3 pass would establish empirical transfer only to the preregistered eight adapter-heldout fit-pool groups. A future P4 pass would add evidence only for one hash-locked Wav2Lip/SyncNet fixed-crop protocol. None imply sealed-test or population generalization, multi-TFG replacement safety, perceptual lip sync, intelligibility, content/speaker preservation, acoustic quality, or deployability.

## Immutable execution history

The original `runs/lrs3_mfa_linear_real_video_sync_20260903/` remains an immutable P0 failure from pytest duplicate-module collection. `v2` and `v3` remain immutable pre-optimizer engineering failures from the CUDA-device comparison and missing `CUBLAS_WORKSPACE_CONFIG`. They were not overwritten or reused.

## Observations
- [status] in-progress
- [progress] Composite state `p1-passed-follow-up-specified-p2-pending` is normalized to `in-progress`; P1 passed and P2 remains pending/not requested.

- [decision] Do not expand the sealed prototype change; post-prototype transfer and replacement live in a separate OpenSpec change #openspec
- [decision] P2 is a hard upstream gate and must run fresh rather than warm-starting from or appending to P1 #p2 #fail-closed
- [decision] Hold natural evaluation audio fixed when comparing MFA-driven and candidate-driven generated videos #replacement
- [decision] Use a complete 3×3 matrix so diagonal co-adaptation remains diagnostic and cannot establish replacement gain #wav2lip #syncnet
- [decision] Use eight source-group-disjoint records with a fixed pre-score selection rule and a descriptive 6/8 gate #transfer
- [result] P1 passed with official D gain `0.3697214127` and C gain `0.4078216553` #p1

## Relations

- relates_to [[Wav2Lip replacement-aware cross-attention supervision]]
- relates_to [[LRS3 MFA-linear TFG native/replacement 2026-09-13|LRS3 MFA-linear replacement NO-GO]]
- relates_to [[Direct waveform upper bound]]
- relates_to [[tts-exp|Wav2Lip replacement is primary objective]]
