## Context

See `proposal.md` for motivation and `specs/masked-tts-reconstruction-prototype/spec.md` for the behavior contract.

The prior cross-attention line did not fail because TTS and natural features were indistinguishable. Exp31 measured natural/TTS WavLM-L6 mean CKA `0.3828`, held-out frame classification accuracy `0.9411`, and AUC `0.9871`; however, the old adapter's mean attention-row entropy was `0.9978`, its argmax positions had no diagonal or phone-block structure, and paired TTS did not outperform natural or shuffled KV. The repository also retains an unfinished Exp31 Week2 path with cached features, MFA3 phone timings, and three 200-step phone-pooled training arms, but no durable validation/official test decision. Its training diagnostics do not show clear TTS dominance and cannot establish feasibility.

The fundamental missing supervision is an ideal waveform containing “TTS phonetic clarity with every natural acoustic attribute.” No such target exists. This prototype therefore does not attempt the final enhancement task. It creates an identifiable pretext task with a real target: hide a natural phone from the model, align the corresponding TTS phone to the natural clock, and test whether training and inference with the complete paired TTS feature modality improves prediction of the clean natural log-mel phone on unseen source groups. A pass establishes incremental predictive utility of paired aligned TTS features under this task. It does not identify phonetic content, semantic clarity, speaker, prosody, or another TTS factor as the cause; nor does it mean TTS-specific clarity survived, a waveform is reachable, or a TFG benefits.

## Goals / Non-Goals

**Goals:**

- Establish whether deterministic phone-aligned paired-TTS features add held-out reconstruction information beyond natural context alone.
- Make natural-content leakage, TTS-branch collapse, wrong-TTS dependence, source-group leakage, and denominator failure directly testable.
- Reuse local fit-only LRS3, cached WavLM-L6, and MFA3 timings so implementation and execution remain fast and network-free.
- Produce a compact implementation structure that a downstream agent can build without rediscovering cohorts, controls, losses, or decisions.

**Non-Goals:**

- Prove that the reconstructed patch is clearer than natural or preserves a useful TTS acoustic characteristic.
- Generate waveform, render Wav2Lip, run SyncNet, estimate replacement effect, or claim visual gain.
- Learn free cross-attention, soft-DTW, optimal transport, a vocoder, a TFG, or an ASR model.
- Optimize mask rules, phone alignment, architecture, steps, loss weights, seed count, or decision thresholds after held-out results exist.
- Access any parent record whose `protocol_split` is not exactly `train`, including locally cached derived media/features for such records.

## Decisions

### 1. Reuse exactly 30 fit-only records and create a fresh group split

The metadata roots are frozen to the existing Exp31/policy-cohort assets under:

```text
.claude/worktrees/lrs3-wavlm-resynthesis-50/
├── data/splits/exp_a_split.json
├── tmp/lrs3_policy_a1_200_20260828/policy_cohort/
│   ├── source_manifest.json
│   └── records/<sample_id>.json
└── tmp/runs/diagnostic_week1/
    ├── features/<sample_id>_{natural,tts}_wavlm_l6.npy
    └── week2_mfa/phone_alignments/<sample_id>.json
```

Preflight may read split/cohort metadata to identify records, but it shall open record media, cached arrays, and phone JSON only after selecting IDs whose source-manifest `protocol_split` is exactly `train`. The current metadata projection yields 30 records and ten source groups. The frozen split is:

```text
train groups (19 records):
6WeS1bXRBOk, 6XNrzmh0EVs, 6XS8TA4RBog,
6YKYo00mFAg, 6ZiN9ZJT294, 6qTfX4U6CS0

evaluation groups (11 records):
6tSlMoMNSlY, 6tpsSu5O0Ws, 6wk4dkYSrV0, 6xtmm0MnaS0
```

All IDs, parent paths, code/config values, audio hashes, feature hashes, alignment hashes, source groups, and protocol labels are written to `00_lock`. A mismatch, missing asset, duplicate ID, cross-split group, or non-`train` parent makes engineering `NO_GO`; records are not substituted. Existing Exp31 model checkpoints and downstream metrics are not inputs.

Alternative considered: reuse the old Exp31 train/val/test labels directly. Rejected because those local arms contain parent records labeled `validation` or `test`; this prototype uses only the parent fit/train media.

### 2. Build deterministic matched lexical-phone masks

For every selected record, phone labels are normalized only by `unicodedata.normalize("NFC", str(label).strip())`; no case folding, pronunciation remapping, IPA substitution, or lexical aliasing is allowed. Labels whose normalized case-folded value is one of `{"", "sil", "sp", "spn", "<eps>", "<sil>", "silence"}` are administrative and excluded. Run deterministic edit-distance alignment over the remaining ordered natural and TTS labels, with equal cost for every non-equal operation and deterministic tie order `equal/substitute`, `delete`, then `insert`. An eligible pair must be an `equal` operation with byte-identical normalized lexical labels; silence/administrative labels, insertion/deletion/substitution, non-finite/non-monotonic timing, boundary collapse, and out-of-span phones are excluded with reason codes.

Canonical ordering is frozen before schedule generation: groups use the explicit order listed in Decision 1; records within group sort by UTF-8 `sample_id`; masks within record sort by `(natural_phone_operation_index, natural_core_start_frame, natural_core_end_frame, tts_phone_operation_index)`. The ordered group/record/mask lists and their canonical JSON hash are persisted. All eligible phones are used for training through indices into these lists and evaluated once per seed.

Natural audio support is the first `61,440` samples (3.84 s) at 16 kHz, matching the cached natural WavLM feature extraction. A target phone must produce 4–40 Wav2Lip-compatible natural mel frames and at least two TTS WavLM frames. Every example uses a fixed 96-mel-frame natural window. With core bounds `[s,e)`, define `center=floor((s+e-1)/2)` and `window_start=clip(center-48, 0, total_mel_frames-96)`; the window is `[window_start,window_start+96)`. The masked input support is the target core expanded by exactly four mel frames on each side and clipped to this window. Reconstruction loss is computed only on the unexpanded target core. The aligned TTS tensor is also length 96, contains mapped features only at core-frame positions, and is exactly zero elsewhere; separate length-96 binary channels identify the core and masked support. All eligible phones are used for training with the frozen source-group-balanced sampler; all eligible phones are evaluated once per seed.

Readiness requires at least 12 training records across at least five frozen train groups and 80 training masks, plus at least eight evaluation records, all four frozen evaluation groups, 40 evaluation masks, and five masks per evaluation group. Otherwise science is `INSUFFICIENT`; phone rules and groups are not relaxed.

### 3. Use fixed phone-phase feature alignment, not waveform MFA-linear

MFA defines corresponding phone blocks only. Natural mel is extracted by the hash-locked `third_party/Wav2Lip/audio.py::melspectrogram` and `hparams.py` contract: 16 kHz, pre-emphasis `0.97`, `n_fft=800`, Hann `win_size=800`, `hop_size=200`, 80 bins, `fmin=55`, `fmax=7600`, librosa STFT `center=True`/constant padding, dB floor `-100`, reference `20`, symmetric clipped normalization to `[-4,4]`. Natural mel frame `m` has analysis center `m*200/16000` seconds; a phone core contains exactly frame centers in `[phone_start,phone_end)`. The implementation and hparams files are code-hash locked.

Cached WavLM-L6 uses 16 kHz, 1024 dimensions, and 320-sample stride. Reuse the Exp31 boundary convention exactly: `b(t)=int(round(t*16000/320))` using Python rounding, select the clamped half-open rows `[b(start),b(end))`, and reject rather than extend a selection with fewer than two rows. For a TTS selection of `n` rows and a natural core of `L` mel frames, destination phase is `u_i=(i+0.5)/L`; source coordinate is `x_i=clip(u_i*n-0.5,0,n-1)`, with `left=floor(x_i)`, `right=min(left+1,n-1)`, and `alpha=x_i-left`. The mapped vector is `(1-alpha)*Z[left]+alpha*Z[right]`. Every source row, coordinate, boundary clip, and weight is recorded. The resulting core trajectory is inserted into the length-96 aligned-TTS tensor at core-relative positions; all other rows are zero. This deterministic operator is named `phone_phase_linear_v1`.

This is not claimed to solve sub-phone alignment. It differs from old MFA-linear because it transports cached features only; it does not resample, splice, decode, or generate waveform. A complete result supports or rejects this particular coarse aligner plus reconstruction task, not every monotonic alignment method.

Natural per-band mean/std and TTS L6 mean/std are estimated from training groups only, with finite positive std floors, locked, and reused unchanged for evaluation. Each fixed natural window is standardized; target-plus-guard rows are then set to zero in normalized space. The model receives standardized masked natural mel, a binary masked-support channel, a binary target-core channel, and the length-96 aligned TTS tensor. No natural WavLM feature, transcript text, phone ID, downstream score, or clean target-core mel enters the trainable input path.

### 4. Freeze one tiny reconstruction architecture and two training arms

The implementation exposes one `MaskedNaturalReconstructor` architecture:

```text
[96,80] masked normalized natural mel
+ [96,1] masked-support channel
+ [96,1] target-core channel
    → 1-D context projection / four residual TCN blocks
[96,1024] aligned TTS-L6 tensor (zero outside core)
    → linear projection / two 1-D residual blocks
framewise concatenate → fusion projection → [96,80] predicted normalized mel
```

Hidden width is `128`; context dilations are `(1,2,4,8)`; TTS-branch dilations are `(1,2)`; kernels are width `3`; normalization and activation are fixed in config; trainable parameters must remain below 1.5 million. No attention, recurrence, pretrained trainable module, or waveform decoder is allowed. The implementation shall prove that no tensor derived from clean natural target-core mel enters either input branch after masking.

For fixed seeds `20260901`, `20260902`, and `20260903`, instantiate the architecture once per seed, save its initial state hash, and clone that exact state into two arms:

```text
FULL_CORRECT: aligned paired-TTS tensor
NAT_ONLY:    all-zero TTS tensor
```

Before either arm trains, precompute one shared 600-step schedule per seed with `np.random.Generator(np.random.PCG64(seed))`. For each of the `600*16` batch positions, sample one frozen train group uniformly with replacement, then one record from that group uniformly with replacement, then one eligible mask from that record uniformly with replacement. FULL and NAT_ONLY consume the byte-identical schedule in the same order. Use CPU only with deterministic PyTorch algorithms, one recorded thread configuration, AdamW, learning rate `1e-3`, weight decay `1e-4`, batch size `16`, gradient clip `1.0`, and exactly `600` optimizer steps. There is no validation-based checkpoint selection, early stopping, retry by seed, device switch, or hyperparameter search; only step 600 is evaluated.

The train loss is reduced over the target core `[s,e)` in window-relative indices:

```text
L_patch = mean_{t=s..e-1,b=1..80} |pred[t,b] - target[t,b]|

L_velocity = mean_{t=s+1..e-1,b=1..80}
  |(pred[t,b]-pred[t-1,b]) - (target[t,b]-target[t-1,b])|

L_total = L_patch + 0.25 * L_velocity
```

Only adjacent frame pairs with both endpoints inside the unexpanded core enter `L_velocity`; no core/guard or window-boundary pair is included. The minimum four-frame core makes this reduction non-empty.

The shuffled and null conditions are not included in FULL training. This prevents a ranking term from succeeding by deliberately damaging controls. A separate NAT_ONLY arm provides a trained context-only baseline.

### 5. Evaluate one primary paired contrast and two sensitivity diagnostics

For every evaluation mask and seed, compute:

```text
FULL_CORRECT: FULL checkpoint + paired aligned TTS
FULL_ZERO:    same FULL checkpoint + all-zero TTS tensor
FULL_SHUFFLED:same FULL checkpoint + deterministic wrong-phone TTS donor
              drawn only from frozen training groups and mapped to target length
NAT_ONLY:     NAT_ONLY checkpoint + all-zero TTS tensor
```

All four conditions share the target, 96-frame natural context, masks, support, normalization, initial-state seed, and frozen batch unit. There is no modality-present flag. `FULL_ZERO` and `FULL_SHUFFLED` are sensitivity diagnostics because the FULL model is trained only with paired nonzero TTS; their losses cannot promote or rescue feasibility and are not treated as clean in-distribution causal controls.

A shuffled donor, when available, must come from a frozen training-group mask, therefore never depends on another evaluation unit; have a different lexical phone label; have source duration within `[0.5,2.0]` of the evaluation target duration; and be selected by ascending duration mismatch then stable SHA-256 rank under seed `20260901`. It is resampled in feature space with `phone_phase_linear_v1` and inserted into the same target core. Shuffled-donor assignment occurs only after the independent primary mask manifest freezes. If no donor exists, the primary target remains and records `FULL_SHUFFLED.status="diagnostic_missing"`; diagnostic coverage is reported but is not an engineering or scientific gate. Required FULL_CORRECT, FULL_ZERO, and NAT_ONLY outputs/losses must be finite. Record target-core RMS prediction differences between FULL_CORRECT and FULL_ZERO and, where available, FULL_SHUFFLED, but interpret them only as branch/input sensitivity.

The primary algorithm comparison is `FULL_CORRECT` versus the separately trained, identically initialized `NAT_ONLY`: both saw the same natural inputs, targets, sampler units, steps, optimizer constants, and initial model state; only the paired aligned-TTS tensor differed during training and inference.

### 6. Aggregate masks conservatively and decide feasibility once

For each evaluation mask and seed, the sole positive-is-better primary gain is:

```text
nat_gain = L_total(NAT_ONLY) - L_total(FULL_CORRECT)
```

Also report, as non-promoting sensitivity diagnostics:

```text
zero_gap = L_total(FULL_ZERO) - L_total(FULL_CORRECT)
shuf_gap = L_total(FULL_SHUFFLED) - L_total(FULL_CORRECT)
```

Aggregate in this order without treating masks or seeds as independent units:

1. median over masks within each record and seed;
2. median over records within each source group and seed;
3. median over the three seeds within each source group.

Report all mask/record/group/seed values. Compute a deterministic 10,000-draw whole-source-group percentile bootstrap over the four final evaluation-group `nat_gain` rows using `PCG64(20260901)` and NumPy `method="linear"`. Compute observed relative loss reduction against NAT_ONLY from median group-level NAT_ONLY and FULL_CORRECT losses. Do not bootstrap or gate on diagnostic zero/shuffled gaps.

Decision logic:

- any lock, leakage, denominator, tensor, training, checkpoint, or evaluation contract failure: engineering `NO_GO`, science `NOT_EVALUATED`;
- readiness denominator below the frozen minimum: engineering `GO`, science `INSUFFICIENT` without model promotion;
- otherwise science is `ALGORITHM_FEASIBLE` only if all of the following hold:
  1. the 95% bootstrap lower bound is greater than zero for `nat_gain`;
  2. every one of the four evaluation source groups has positive final `nat_gain`;
  3. observed median relative `L_total` reduction is at least `5%` against NAT_ONLY;
  4. each of the three seeds has positive overall median `nat_gain` before seed aggregation;
- every other complete outcome is `NO_ALGORITHM_FEASIBILITY_SUPPORT`.

The all-group and three-seed gates intentionally trade power for a reliable small-prototype claim. `ALGORITHM_FEASIBLE` means only that training and inference with paired aligned TTS improves masked natural acoustic reconstruction relative to an identically initialized and scheduled natural-context-only model under this frozen task. FULL_ZERO/FULL_SHUFFLED can reveal branch sensitivity or gross wrong-input behavior but cannot establish correct-content causality. A feasibility pass authorizes a later factorized objective in which natural supervises clock/style and TTS/reference supervises content. It does not show retained TTS clarity, waveform reachability, visual gain, SyncNet gain, or replacement safety.

### 7. Keep implementation and artifacts compact

Proposed package and run tree:

```text
scripts/experiments/masked_tts_reconstruction/
├── config.py       # all frozen constants and paths
├── protocol.py     # fit-only lock, split, phones, masks, shuffle donors
├── features.py     # mel, normalization, phone-phase interpolation
├── model.py        # one tiny reconstructor and losses
├── train.py        # fixed-seed FULL/NAT_ONLY training
├── evaluate.py     # required paired evaluation + optional sensitivity
├── analyze.py      # hierarchical aggregation/bootstrap/decision
└── run.py          # preflight/train/evaluate/analyze/resume

tests/experiments/masked_tts_reconstruction/

runs/lrs3_masked_tts_reconstruction_prototype_20260901/
├── 00_lock/
├── 01_masks/
├── 02_features/
├── 03_train/<seed>/{full_correct,nat_only}/
├── 04_eval/<seed>/
├── 05_analysis/
├── logs/
├── validation.json
├── summary.json
└── decision.json
```

Reuse existing atomic JSON/hash/NPZ helpers where appropriate. Do not create a base experiment framework, registry, dashboard, service, database, or vendor-code fork. The scientific prototype is CPU-only. CLI options are limited to asset root, run path, stage, and resume; scientific constants are not CLI flags.

## Risks / Trade-offs

- **[Natural target can wash out TTS-specific clarity]** Exact natural reconstruction teaches alignment, domain conversion, and information use, not enhancement. → Limit the claim to injectability; any later head needs separate natural-anchor and TTS-content objectives.
- **[Coarse within-phone alignment]** Normalized phase may miss closure/release or formant landmarks. → Name and freeze the tested aligner; a negative result does not reject phone-constrained soft-DTW or monotonic transport.
- **[Residual natural-context predictability]** Neighboring natural frames and coarticulation may predict part of a masked phone even after the four-frame STFT guard. → Define the task as “paired TTS beyond residual natural-context predictability,” use a trained NAT_ONLY primary baseline, mask before the trainable context encoder, prohibit natural contextual embeddings/labels, and report rather than overclaim zero/wrong-input sensitivity.
- **[Wrong-phone shuffled arm is out of distribution]** FULL is not trained on shuffled inputs. → Draw donors only from training groups to avoid evaluation-unit dependence and treat shuffled strictly as a non-promoting sensitivity diagnostic; feasibility depends only on FULL_CORRECT versus identically initialized/scheduled NAT_ONLY.
- **[Small four-group evaluation]** Intervals are coarse. → Require all-group sign consistency, three-seed stability, and a 5% practical effect; make no population claim.
- **[Historical/discovery asset reuse]** The records and features were used in prior diagnostics. → Call this a feasibility prototype only; a positive result requires a fresh fit-only follow-up before a learned replacement claim.
- **[Parent worktree locality]** Cached assets live in an existing worktree. → Hash-lock every consumed file, accept an explicit asset-root path, never modify the parent, and fail clearly if assets disappear.
- **[Sealed-data adjacency]** Existing Exp31 caches contain records with parent validation/test labels. → Select IDs from source-manifest metadata first and never open non-`train` record JSON, arrays, phone files, audio, videos, or downstream results.

## Migration Plan

This is an additive prototype with no migration.

1. Add the compact package and synthetic tests.
2. Run metadata-only fit/train lock and reproduce exactly 30 records, ten groups, and the frozen 19/11 group split.
3. Build mask/shuffle manifests and stop if readiness denominators fail.
4. Run feature/alignment leakage tests and a two-record overfit smoke; the smoke is engineering-only and cannot change constants.
5. Train six fixed final models (FULL and NAT_ONLY for three seeds) for 600 steps each.
6. Evaluate all four conditions once on frozen evaluation masks, validate artifacts, and compute the decision.
7. Run focused tests and `openspec validate prototype-lrs3-masked-tts-reconstruction --strict`.
8. Preserve any failed or interrupted run; resume only hash-valid cells and never overwrite old experiments or access sealed media.
