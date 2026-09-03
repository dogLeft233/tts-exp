## Context

The completed independent run at `runs/lrs3_masked_tts_new_confirmation_20260902/` used 16 records from eight source groups, 222 evaluation masks, three fixed hard-negative checkpoints, frozen direct-mel Wav2Lip, strict untouched-natural-audio replacement, and frozen official SyncNet V2. Its main result was:

```text
paired over SAME_PHONE_WRONG_INSTANCE downstream: not passed
paired over WITHIN_PHONE_REVERSED downstream: passed
paired over NAT_ONLY downstream: passed
final status: NOT_CONFIRMED_ON_NEW_RECORDS
```

This means the TTS path affects the endpoint, but it does not yet show that the endpoint uses the correct occurrence-specific TTS trajectory.

A post-run audit found one analysis defect. Reconstruction paired rows were indexed by `(sample_id, seed)` although each pair has several masks. The final paired mask overwrote the other paired rows. Each control row already contains its correct same-mask `paired_loss` and `reconstruction_gain`, so this is repairable without recomputation of model outputs. Correct pairing retains both reconstruction passes:

```text
SAME_PHONE_WRONG_INSTANCE: median 0.14533, 95% CI [0.06750, 0.17535], 8/8 positive
WITHIN_PHONE_REVERSED:     median 0.04496, 95% CI [0.03184, 0.06514], 8/8 positive
```

The missing experimental condition is natural audio encoded as WavLM-L6 and placed in the model's existing TTS-feature slot. `NAT_ONLY` is not this condition: it keeps masked natural-mel context and zeros the TTS slot. Existing `NATURAL_MEL` evaluations also are not this condition: they bypass the reconstructor and feed complete natural mel directly to Wav2Lip.

## Goals and non-goals

**Goals**

- Correct the mask-pairing defect and preserve a reproducible correction record.
- Use existing artifacts to show how much paired-versus-control change survives each stage of the pipeline.
- Test whether the frozen model can exploit occurrence-specific natural-reference WavLM features through its existing TTS input slot.
- Test a cleaner same-speaker/channel wrong-instance control when sufficient matched occurrences exist.
- Produce one bounded recommendation before any further training or waveform work.

**Non-goals**

- Retrain a model, tune a loss, search checkpoints, regenerate Qwen TTS, or select new records.
- Treat natural-slot input as a text-only TTS system or as proof of waveform reconstruction.
- Infer equivalence from a failed superiority test.
- Select masks, donors, thresholds, or conditions using reconstruction, Wav2Lip, or SyncNet outcomes.
- Modify or delete parent run artifacts.

## Frozen inputs

Bind hashes before computing results for:

```text
runs/lrs3_masked_tts_new_confirmation_20260902/
  00_cohort/manifest.json
  01_tts/tts_meta.json
  02_alignment/alignment.json
  03_data/{feature_manifest.json,mask_manifest.json}
  04_reconstruction/reconstruction.json
  05_drivers/drivers.json
  06_renders/{box_manifest.json,render_manifest.json,box_repairs.json}
  05_syncnet/summary.json

runs/lrs3_masked_tts_trajectory_specificity_20260902/
  04_training/{20260901,20260902,20260903}/hard_negative/checkpoint.pt

scripts/wavlm_knn_vc_adapter.py
Torch Hub checkout: bshall/knn-vc@c616845c4e309e24d5927f15adbdf277a3d65358
Torch Hub checkpoint: checkpoints/WavLM-Large.pt
```

The binding manifest SHALL record and verify the adapter source hash, resolved local checkout path and git revision, `WavLM-Large.pt` SHA-256, and this interface:

```text
sample rate: 16000 Hz mono float32
selected layer: 6
feature dimension: 1024
frame stride: 320 samples / 20 ms
vad_trigger_level: 0
loudness normalization: disabled
waveform policy: complete bound natural waveform; no crop, pad, resample, or amplitude change
phone-to-feature span: slice [round(start_s × 16000 / 320), round(end_s × 16000 / 320)), with both slice endpoints clipped to [0, feature_count]
```

A missing or changed checkout revision, adapter hash, checkpoint hash, or interface value makes natural-slot extraction `NOT_EVALUATED`; remote fallback or implicit model download is prohibited.

The binding SHALL verify 16 unique records, eight source groups with two records each, 222 masks, three seeds, four existing conditions, and 192 complete downstream score cells. Parent artifacts are immutable inputs. All new outputs go under:

```text
runs/lrs3_masked_tts_natural_slot_signal_path_20260903/
```

## Part 0: reconstruction correction

Fix future analysis by pairing reconstruction rows with one of these equivalent methods:

```text
preferred: use the stored per-row reconstruction_gain
allowed:   map paired rows by (sample_id, seed, mask_sha256)
```

Pairing only by `(sample_id, seed)` is prohibited. Add a regression fixture with two masks for the same sample and seed whose paired losses differ; the test must fail under the old implementation.

Reaggregate in this order:

```text
mask median within record and seed
→ seed median within record
→ two-record median within source group
→ median and whole-source-group bootstrap across eight groups
```

Use 10,000 draws, `PCG64(20260902)`, and NumPy linear quantiles. Write corrected analysis in the new run; do not overwrite the parent run. The two corrected summaries shown in Context are reproducibility checks. A mismatch greater than `1e-5` in a reported median or interval endpoint makes Part 0 `NOT_EVALUATED`.

## Part A: existing signal-path audit

This part performs no model inference, rendering, replacement, or SyncNet scoring. Reconstruct control inputs deterministically from the bound manifests and read existing predictions, drivers, videos, boxes, and scores.

Audit these contrasts:

```text
PAIRED_TTS versus SAME_PHONE_WRONG_INSTANCE
PAIRED_TTS versus WITHIN_PHONE_REVERSED
```

For every compatible cell, report the following positive magnitudes:

1. **Input core difference:** RMS feature difference and first-difference RMS between the two aligned 1024-D WavLM cores.
2. **Predicted core difference:** RMS mel difference and first-difference RMS over the target core.
3. **Driver difference:** RMS mel difference only over frames patched by the bound masks; unchanged context must not dilute the metric.
4. **Rendered response:** mean absolute RGB difference in `[0,1]` over the lower half of the shared frozen face box for paired decoded frames. Frame count, frame size, and box identity must match.
5. **Endpoint response:** existing paired-control Sync-C and Sync-D gains.

Use the same mask, seed, record, and source-group hierarchy where a level exists. At each stage report both controls side by side and report the ratio:

```text
wrong-instance magnitude / reversed magnitude
```

when the denominator is nonzero. This audit is descriptive: it SHALL NOT invent a pass threshold or replace the existing downstream decision. Independently encoded RGB differences are response magnitudes that include rendering and codec effects; they SHALL NOT by themselves identify Wav2Lip as a causal bottleneck. The table may show where observed contrast magnitude changes, while causal localization requires a later controlled test.

## Part B: natural WavLM in the TTS slot

### Construction

Add exactly one condition named:

```text
NATURAL_WAVLM_IN_TTS_SLOT
```

For each of the 16 records:

1. read the bound untouched natural audio and verify its hash;
2. extract 1024-D WavLM-L6 using the exact frozen extractor, sample rate, dtype, and frame convention used for the Qwen TTS features;
3. derive each target phone's natural-audio WavLM span from the bound natural MFA alignment and the existing 20 ms frame convention;
4. use the existing `phone_phase_linear` path to map that span to the same destination core used by `PAIRED_TTS`;
5. keep all values outside the target core exactly zero;
6. preserve the natural-mel context, mask, target, core location, and destination length.

Write source audio, alignment, extracted-feature, and per-mask aligned-feature hashes. Every aligned natural-slot array must be finite, have the same shape as the paired TTS input, and differ in hash from the paired input. A missing or invalid cell makes Part B `NOT_EVALUATED`; do not drop masks or records.

This is an oracle/reference condition because it exposes the target natural phone to the conditioning slot. It is allowed as a diagnostic because natural reference audio is available in the intended setting, but it is not a TTS-only result.

### Frozen evaluation

Use the existing three hard-negative checkpoints without modification. For all 222 masks:

- predict `NATURAL_WAVLM_IN_TTS_SLOT`;
- compute the existing patch, velocity, and total reconstruction losses;
- build one record-level driver per record and seed with the existing overlap, denormalization, clamp, and outside-core equality rules.

Render exactly 48 new cells:

```text
16 records × 3 seeds × 1 new condition
```

Use the bound frozen Wav2Lip checkpoint and shared face boxes, strictly replace each output with its corresponding untouched natural audio, and score with the bound frozen SyncNet. Reuse existing `PAIRED_TTS` and `NAT_ONLY` cells rather than rerendering them.

### Contrasts and rules

Define positive-is-better contrasts:

```text
natural slot over zero input
  reconstruction = loss(NAT_ONLY) - loss(NATURAL_WAVLM_IN_TTS_SLOT)
  Sync-C          = C(NATURAL_WAVLM_IN_TTS_SLOT) - C(NAT_ONLY)
  Sync-D          = D(NAT_ONLY) - D(NATURAL_WAVLM_IN_TTS_SLOT)

natural slot over paired TTS
  reconstruction = loss(PAIRED_TTS) - loss(NATURAL_WAVLM_IN_TTS_SLOT)
  Sync-C          = C(NATURAL_WAVLM_IN_TTS_SLOT) - C(PAIRED_TTS)
  Sync-D          = D(PAIRED_TTS) - D(NATURAL_WAVLM_IN_TTS_SLOT)
```

Aggregate reconstruction by mask median, seed median, record, and source group. Aggregate downstream by seed median, record, and source group. A metric passes only when its 95% whole-group bootstrap lower bound is greater than zero and at least 7/8 group values are positive.

Report reconstruction and frozen-TFG evidence as independent axes:

- `NATURAL_SLOT_OVER_ZERO_TFG_PASS` when Sync-C and Sync-D pass against `NAT_ONLY`.
- `NATURAL_SLOT_OVER_TTS_TFG_PASS` when Sync-C and Sync-D pass against `PAIRED_TTS`.
- For each contrast, independently report `RECONSTRUCTION_PASS` or `RECONSTRUCTION_NOT_SHOWN`.
- Qualify a TFG pass as `RECONSTRUCTION_SUPPORTED` when its reconstruction axis also passes, otherwise as `DOWNSTREAM_ONLY`.
- Use `TFG_NOT_SHOWN` when either downstream metric fails; this is not an equivalence claim.

This separation prevents a real frozen-Wav2Lip/SyncNet improvement from being discarded merely because mel reconstruction loss did not improve. Conversely, reconstruction-only evidence does not establish a TFG advantage.

## Part C: matched wrong-instance control

### Score-free preflight

Build candidates only from phone occurrences in the bound 16-record cohort. A donor must:

- have the same normalized phone label;
- be a different phone occurrence, where occurrence identity is `(sample_id, tts_phone_index)`; multiple masks that reference one occurrence do not count as distinct donors;
- have a valid bound Qwen TTS WavLM span;
- come first from the same record, otherwise from the other record in the same source group.

Rank candidates without model outputs or scores by this fixed tuple:

```text
source tier: same record before same source group
number of left/right normalized-phone mismatches
absolute phone-duration difference
sample ID
canonical index
mask hash
```

Freeze the first three distinct donor occurrences for each target. A target is eligible only when all three exist. Freeze one common eligible-mask subset before inference. Here, "complete" means that every evaluated condition contains every cell defined by this frozen subset; it does not mean that all original 222 masks must remain. The downstream stage may run only if every record has at least two eligible masks and all eight source groups remain represented. Otherwise Part C reports `INSUFFICIENT_MATCHED_DONOR_COVERAGE` and stops after the coverage report.

### Frozen evaluation

For eligible masks, construct all three donor ranks with the existing donor-mask and `phone_phase_linear` path. Each condition must preserve the target natural context, target core, and destination length, and must differ from paired by hash. Run reconstruction for all three ranks so donor sensitivity can be measured cheaply.

Reuse existing paired per-mask predictions for reconstruction. For downstream comparison, rebuild a `PAIRED_COMMON_MASKS` driver using exactly the frozen eligible-mask subset; do not compare a subset wrong-instance driver with the existing all-mask paired driver. Render only the predeclared first-ranked donor, which is fixed by metadata before inference; donor ranks two and three are reconstruction robustness checks and cannot replace rank one after scores are observed.

Render, replace, and score at most 96 new cells:

```text
PAIRED_COMMON_MASKS: 16 records × 3 seeds = 48
MATCHED_WRONG rank 1: 16 records × 3 seeds = 48
```

For reconstruction, compute paired-over-wrong gain for each target, donor rank, and seed, then aggregate in this order:

```text
donor-rank median → mask median → seed median → record → source group
```

Also report each donor rank separately as a robustness table. For downstream, compute rank-one paired-over-wrong Sync gains and aggregate:

```text
seed median → record → source group
```

Use the same positive directions and eight-group pass rule as the prior trajectory experiment. Treat reconstruction and frozen-TFG evidence as independent axes:

- Report `MATCHED_INSTANCE_TFG_SIGNAL` when both Sync-C and Sync-D pass, and qualify it as `RECONSTRUCTION_SUPPORTED` or `DOWNSTREAM_ONLY` according to the reconstruction axis.
- Report `MATCHED_INSTANCE_RECONSTRUCTION_ONLY` when reconstruction passes but the complete downstream rule does not.
- Report `NO_MATCHED_INSTANCE_SIGNAL` when neither reconstruction nor the complete downstream rule passes.

A downstream-only result is useful TFG evidence but not evidence that the mel reconstruction objective itself prefers the correct occurrence.

This stage remains exploratory because donor design and records were chosen after inspecting the earlier experiment. A positive result permits a future preregistered independent confirmation; it does not itself confirm generalization.

## Final decision

For decision purposes, "Part C lacks a downstream signal" includes `MATCHED_INSTANCE_RECONSTRUCTION_ONLY`, `NO_MATCHED_INSTANCE_SIGNAL`, and the valid skip `INSUFFICIENT_MATCHED_DONOR_COVERAGE`. `NOT_EVALUATED` remains invalid.

Apply this precedence order and write exactly one recommendation:

1. `STOP_INVALID_EXPERIMENT` if any required executed part is `NOT_EVALUATED` or provenance is incomplete.
2. `CONFIRM_MATCHED_INSTANCE_CONTROL_ON_FRESH_RECORDS` if Part C reports `MATCHED_INSTANCE_TFG_SIGNAL`, whether reconstruction-supported or downstream-only.
3. `PIVOT_TO_NATURAL_REFERENCE_CONDITIONING` if Part C lacks a downstream signal and Part B reports both `NATURAL_SLOT_OVER_ZERO_TFG_PASS` and `NATURAL_SLOT_OVER_TTS_TFG_PASS`.
4. `RETAIN_MODALITY_ONLY_CLAIM` if Part B reports `NATURAL_SLOT_OVER_ZERO_TFG_PASS` but natural slot is not shown to beat paired TTS and Part C lacks a downstream signal.
5. `REDESIGN_CONDITIONING_OBJECTIVE_BEFORE_MORE_DATA` otherwise.

Part B reconstruction labels qualify but do not veto a frozen-TFG decision. `INSUFFICIENT_MATCHED_DONOR_COVERAGE` is a valid Part C skip, not an invalid experiment. Part A is descriptive and cannot by itself select a recommendation.

The final report must state:

- the corrected reconstruction summaries;
- that natural-slot input is an oracle/reference condition;
- that failed superiority is not equivalence;
- that the cohort has already been inspected;
- that no waveform, audible-retention, deployability, or population-generalization claim is permitted;
- that the waveform-decoder gate remains `CLOSED`.

## Minimal implementation

Prefer one new runner under the existing trajectory-specificity package and reuse current feature extraction, `build_example`, phone interpolation, prediction, patching, rendering, strict replacement, scoring, and aggregation functions. Do not copy checkpoints or parent media.

Suggested stages:

```text
bind
correct-analysis
trace
natural-slot
matched-preflight
matched-run
analyze
all
```

The `all` stage SHALL honor the matched-preflight skip and SHALL NOT rerun complete stages whose manifests and source hashes still match.

## Minimal outputs

```text
runs/lrs3_masked_tts_natural_slot_signal_path_20260903/
├── 00_binding/manifest.json
├── 01_correction/{analysis.json,report.md}
├── 02_trace/{analysis.json,report.md}
├── 03_natural_features/{manifest.json,*.npy}
├── 04_natural_slot/{reconstruction.json,drivers.json,render_manifest.json,summary.json,analysis.json}
├── 05_matched_control/{preflight.json,reconstruction.json,drivers.json,render_manifest.json,summary.json,analysis.json}
├── decision.json
└── summary.json
```

Per-cell predictions, videos, replacements, and logs may remain under their stage directories. Manifests must contain enough hashes to resume safely, but no duplicate copy of an immutable parent artifact is required.
