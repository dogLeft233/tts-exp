## Context

The 16-record frozen-Wav2Lip confirmation reported `CONFIRMED_MODALITY_ONLY`:

```text
PAIRED_TTS over NAT_ONLY
Sync-C: 8/8 positive, median 1.50075, 95% CI [0.79750, 1.65100]
Sync-D: 8/8 positive, median 0.99800, 95% CI [0.46700, 1.32250]

PAIRED_TTS over PHONE_CENTROID
Sync-C: 5/8 positive, median 0.17950, 95% CI [-0.27250, 0.38400]
Sync-D: 5/8 positive, median 0.06950, 95% CI [-0.40950, 0.22800]
```

`NAT_ONLY` is a masked reconstruction with zero TTS features; it is not unchanged natural mel. `PHONE_CENTROID` preserves phone identity and target timing but removes occurrence-specific variation. The result therefore shows useful TTS-side conditioning, not audible TTS retention or exact trajectory use.

All 24 parent evaluation records have now appeared in downstream analysis. This change is a mechanism and engineering diagnostic. It must not be described as a new independent confirmation.

## Goals and non-goals

**Goals**

- Determine whether the paired reconstruction is shown to beat the unchanged natural-mel engineering reference.
- Determine whether the frozen reconstructor and downstream endpoint prefer the exact paired trajectory over two matched trajectory controls.
- If that sensitivity is absent, test one fixed hard-negative training change without changing the architecture or searching hyperparameters.
- Produce an explicit stop or next-step decision before any waveform work.

**Non-goals**

- Claim audible feature retention, waveform reachability, population generalization, or deployability.
- Tune donors, reversal rules, loss weights, margins, seeds, checkpoints, masks, or cohorts after observing outcomes.
- Add a vocoder, waveform decoder, new feature encoder, attention module, or end-to-end Wav2Lip loss.
- Reinterpret the completed confirmation status.

## Part A: natural-reference audit

Read the complete existing confirmation score matrix and use its unchanged `NATURAL_MEL`, `PAIRED_TTS`, `PHONE_CENTROID`, and `NAT_ONLY` rows. Do not render or score anything in this part.

For each record, take the median of the three seeded rows for each reconstructed condition. Compare paired and centroid conditions separately with the record's single natural-mel score:

```text
paired_reference_C = C(PAIRED_TTS) - C(NATURAL_MEL)
paired_reference_D = D(NATURAL_MEL) - D(PAIRED_TTS)

centroid_reference_C = C(PHONE_CENTROID) - C(NATURAL_MEL)
centroid_reference_D = D(NATURAL_MEL) - D(PHONE_CENTROID)
```

Also report `PHONE_CENTROID` over `NAT_ONLY` as a descriptive decomposition of the confirmed modality gain. Call this a practical centroid pass only when both interval lower bounds exceed zero and at least 7/8 groups are positive for both metrics. Use the same record-to-source-group aggregation and the same 10,000 whole-group bootstrap configuration as the completed confirmation.

Assign one engineering-audit label:

- `PAIRED_BEATS_NATURAL_REFERENCE` when both paired-reference interval lower bounds exceed zero and at least 7/8 groups are positive for both metrics.
- `PAIRED_NOT_SHOWN_TO_BEAT_NATURAL_REFERENCE` otherwise.
- `NOT_EVALUATED` if the existing matrix or provenance is incomplete.

The second label does not mean natural mel is proven superior. Part A is retrospective and cannot promote a scientific claim.

## Part B: frozen trajectory controls

Use exactly the 16 records and all evaluation masks frozen by the completed confirmation. Reuse the three original `full_correct` checkpoints. Keep existing `PAIRED_TTS` and `PHONE_CENTROID` predictions and scores unchanged, and add two conditions.

### Same-phone wrong instance

For every target evaluation mask, choose one donor from the frozen training masks that:

- has the same normalized phone label;
- has a different sample ID and source group;
- satisfies the existing WavLM support checks.

Sort eligible donors by source group, sample ID, canonical index, and mask hash. Select the donor using the integer represented by the first 16 hexadecimal characters of the target mask hash modulo the candidate count. This selection uses identities only and is fixed before prediction.

Build the condition through the existing `build_example` donor-mask path and `phone_phase_linear` interpolation. It must preserve the target natural context, target mask, core position, and destination length; only the source phone trajectory changes. Call it `SAME_PHONE_WRONG_INSTANCE`.

### Within-phone reversed trajectory

Build the normal paired aligned core, reverse its frame order inside the target core, make it contiguous, and keep all feature values outside the core equal to zero. Call it `WITHIN_PHONE_REVERSED`. Its feature hash must differ from the paired feature hash for every evaluated mask.

If any target has no valid donor, any control is identical to its paired input, or any required cell is missing, Part B is `NOT_EVALUATED`; do not silently drop masks or records.

### Frozen evaluation

For both controls and all three original checkpoints:

1. predict every evaluation mask;
2. compute the existing patch, velocity, and total reconstruction losses;
3. build record-level mel drivers with the existing overlap, denormalization, clamp, and outside-core equality rules;
4. render with the same frozen Wav2Lip checkpoint;
5. replace audio with the corresponding untouched natural audio;
6. score with the same frozen official SyncNet V2.

This adds 96 downstream cells: 16 records, two controls, and three seeds. Existing paired and centroid cells are reused rather than rerendered.

For each control, define positive-is-better contrasts:

```text
reconstruction gain = total loss(control) - total loss(PAIRED_TTS)
Sync-C gain          = C(PAIRED_TTS) - C(control)
Sync-D gain          = D(control) - D(PAIRED_TTS)
```

For reconstruction, take the mask median within each seed and record. Then take the seed median per record and the two-record median per source group. For SyncNet, start with the seed median per record and use the same final group aggregation. Report eight group values, positive-group counts, medians, and 10,000 whole-group bootstrap 95% intervals using `PCG64(20260902)` and NumPy linear quantiles.

A contrast passes only when its interval lower bound exceeds zero and at least 7/8 source groups are positive.

Assign one Part B status:

- `TFG_TRAJECTORY_SIGNAL` when reconstruction, Sync-C, and Sync-D pass for both controls.
- `RECONSTRUCTION_TRAJECTORY_SIGNAL_ONLY` when reconstruction passes for both controls but the full downstream rule does not.
- `NO_TRAJECTORY_SIGNAL` when the complete matrix is valid but reconstruction does not pass for both controls.
- `NOT_EVALUATED` when construction, provenance, replacement, or scoring is incomplete.

These labels describe sensitivity to the exact paired trajectory under this frozen experiment. They are not evidence of audible retention.

## Part C: one conditional training modification

Part C is a feasibility arm, not a mandatory continuation:

- Skip it as `SKIPPED_ALREADY_SENSITIVE` if Part B is `TFG_TRAJECTORY_SIGNAL`.
- Skip it as `SKIPPED_ENDPOINT_LIMITED` if Part B is `RECONSTRUCTION_TRAJECTORY_SIGNAL_ONLY`.
- Skip it as `SKIPPED_INVALID_DIAGNOSIS` if Part B is `NOT_EVALUATED`.
- Run it only when Part B is `NO_TRAJECTORY_SIGNAL`.

Reuse the existing model architecture, frozen train/evaluation split, examples, three seeds, initial-state construction, schedules, 600 steps, batch size 16, optimizer, learning rate, weight decay, gradient clipping, patch loss, and velocity loss. Do not change the masks.

For each scheduled training example, construct the same deterministic same-phone wrong-instance condition defined in Part B. Run the model once with the paired condition and once with the wrong-instance condition. Use one fixed objective:

```text
ranking penalty = max(0, 0.01 + paired total loss - wrong-instance total loss)
training loss   = paired total loss + 0.10 × ranking penalty
```

Do not tune the margin or weight. The checkpoint after exactly 600 steps is the evaluated checkpoint; do not select an intermediate checkpoint.

First evaluate reconstruction on the frozen evaluation masks under `PAIRED_TTS`, `SAME_PHONE_WRONG_INSTANCE`, `WITHIN_PHONE_REVERSED`, and `NAT_ONLY`. The reconstruction gate passes when:

- paired over each hard control passes the same eight-group reconstruction rule used in Part B; and
- the new model's eight-group median paired total loss is no more than 5% above the corresponding original-model median.

If the reconstruction gate fails, stop without downstream rendering and assign `TRAINING_NO_GO`.

If it passes, render only those four conditions for 16 records and three seeds, giving 192 cells. Reuse the frozen rendering, strict replacement, and scoring path. Assign:

- `TRAINING_TFG_GO` when paired over both hard controls passes for Sync-C and Sync-D, and paired over `NAT_ONLY` still passes the existing modality rule.
- `TRAINING_NO_GO` otherwise.
- `NOT_EVALUATED` if the required training or evaluation matrix is incomplete.

`TRAINING_TFG_GO` permits a new independent-record confirmation of the modified mel model. It does not directly permit waveform decoding.

## Final decision

Write one final recommendation by applying this precedence order so that exactly one result is selected:

1. `STOP_INVALID_EXPERIMENT` if any required executed stage is `NOT_EVALUATED`.
2. `CONFIRM_TRAJECTORY_MODEL_ON_NEW_RECORDS` when Part B is `TFG_TRAJECTORY_SIGNAL` or Part C is `TRAINING_TFG_GO`.
3. `USE_SIMPLER_PHONE_CONTROL_PATH` when no trajectory-sensitive downstream result exists and the practical centroid-over-`NAT_ONLY` rule passes.
4. `RETAIN_PAIRED_MODALITY_PATH` when no trajectory-sensitive downstream result or practical centroid pass exists, but Part A is `PAIRED_BEATS_NATURAL_REFERENCE`.
5. `USE_NATURAL_REFERENCE_PATH` otherwise.

A skipped Part C is not an invalid stage. The final report must state that the audit uses previously inspected records. No result from this change alone opens the waveform-decoder gate.

## Minimal outputs

```text
runs/lrs3_masked_tts_trajectory_specificity_20260902/
├── 00_binding/manifest.json
├── 01_reference_audit/{analysis.json,report.md}
├── 02_controls/{manifest.json,features.npz}
├── 03_frozen_diagnosis/{reconstruction.json,syncnet.json,analysis.json,report.md}
├── 04_training/                         # present only when Part C runs
├── decision.json
└── summary.json
```

Keep per-cell predictions, videos, replacement media, and logs under their relevant stage directories, but do not duplicate parent artifacts.
