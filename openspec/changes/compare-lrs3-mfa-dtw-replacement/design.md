## Context

See `proposal.md` for motivation and `specs/lrs3-mfa-dtw-comparison/spec.md` for the behavioral contract.

The completed parent experiment already provides the exact comparison population and all MFA-linear cells:

- strict protocol manifest: `runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/03_strict_replacement_face_ready_retry7/protocol_manifest.json`;
- complete four-cell result: `.../03_strict_replacement_face_ready_retry7/replacement_manifest.json`, SHA-256 `157cd678f5f03be6eae9986a82babfdfb1f1a44e611bb0b86476c6e19b7eb272`;
- paired MFA3 alignment rows: `.../01_mfa3_screen_retry1/alignment_manifest.json`, SHA-256 `111c270ed6d5c37f2318bc0b0e1becc327370c94740c9eada2228dade5d01c98`;
- historical statistics: `.../04_statistics_exploratory_retry2/analysis.json`, SHA-256 `880563c361948f77a62e6a3b9f507393ac6d6c253cdb2c9ced4cc8c671c05690`.

The strict protocol's ordered cohort has 133 records and ordered sample-id hash `61f8c982041cdfdded8daf8850d31e127943386ba2cb7035aa742e94cea9a973`. The paired alignment parent contains 146 clean records; the implementation must perform an identity-preserving join from the frozen 133 records rather than consume all 146.

Historical MFA-linear candidate construction extracts paired natural and TTS WavLM-L6 features, constructs a natural-frame-to-TTS-frame map from MFA3 phone boundaries, samples only TTS features, decodes with the pinned HiFi-GAN, and canonicalizes to the natural sample count. DTW changes only construction of the matched-speech portion of that map.

## Goals / Non-Goals

**Goals:**

- Make the DTW-versus-linear comparison a one-variable, same-record experiment.
- Reuse immutable historical MFA-linear/natural cells instead of re-rendering them.
- Preserve enough path and provenance information to reconstruct every source coordinate.
- Enforce a hard promotion boundary so replacement data cannot influence DTW selection.
- Produce terminal artifacts that answer separately: “does DTW improve candidate-driven TFG?”, “does it improve over MFA-linear after replacement?”, and “is it positive versus the natural baseline after replacement?”.

**Non-Goals:**

- Training an aligner, vocoder, TFG, or scorer.
- Testing soft-DTW, unconstrained DTW, cross-phone paths, relaxed phone boundaries, or multiple path bands.
- Claiming unseen-source, speaker, or LRS3-wide generalization.
- Improving acoustic quality, changing silence policy, or decoding a mel-only candidate.
- Re-running the completed MFA-linear or natural baseline cells.

## Decisions

### 1. Add an isolated staged experiment package

Create `scripts/experiments/lrs3_mfa_dtw_replacement/` with a single resumable CLI and small pure modules for bindings, DTW mapping, statistics, media operations, and validation. Use a new run root such as `runs/lrs3_mfa_dtw_replacement_20260903/` with append-only stage directories:

```text
00_protocol/
01_candidates/
02_diagonal/
03_diagonal_analysis/
04_replacement/       # absent unless promoted
05_final/
```

Each stage reads and validates the immediately preceding self-hashed decision plus all relevant immutable parent bindings. A stage writes to a temporary per-cell path and atomically renames completed JSON/media outputs. Resume accepts only already-complete cells whose sidecar hashes still match.

**Why:** A new package and run root avoid changing the historical protocol while allowing reuse of its mature audio, mux, render, and SyncNet helpers.

**Alternative considered:** Modify the MFA-linear exploratory package in place. Rejected because its protocol ID, artifact hashes, and completed negative result are immutable.

### 2. Bind the 133 records through a three-parent join

Stage 00 loads the strict protocol/replacement manifest, MFA3 alignment manifest, and historical statistics. It first validates each whole-file hash, then joins by `sample_id` while preserving strict-protocol order. For each record it verifies source group, natural audio, face video, MFA-linear candidate, paired TextGrid hashes, candidate/render/mux/score identities, and expected four historical cells. The output manifest records the source paths and hashes needed to regenerate DTW candidates but never derives membership from score values.

**Why:** The alignment parent has 13 additional clean records, so joining from it directly would silently change the population.

**Alternative considered:** Use the first 133 alignment records. Rejected because that does not reproduce the face-ready cohort.

### 3. Reuse the audited hard-DTW geometry and convert each path to interpolation coordinates

For every matched non-silence phone instance:

1. identify natural and TTS WavLM frame indices owned by the matched token instances using the existing 320-sample center-time convention;
2. L2-normalize frame vectors and compute `1 - cosine_similarity` in float64 on CPU;
3. run endpoint-constrained hard-DTW with the existing normalized-coordinate `band_ratio=0.5` and the audited predecessor/tie order;
4. group path source indices by natural local frame;
5. use the arithmetic mean source index as that natural frame's continuous TTS coordinate;
6. convert the coordinate to the existing `left`, `right`, and interpolation-alpha representation.

Singleton phone spans naturally produce a trivial endpoint path. Silence rows are copied from the MFA-linear mapping unchanged. The complete map must be globally nondecreasing inside each phone, but no continuity constraint is imposed across MFA phone boundaries because both methods already reset their local phase there.

The trace stores token indices/labels, global and local frame indices, full path, path cost, horizontal/vertical/diagonal counts, continuous source coordinate, linear baseline coordinate, and displacement. Pure validation reconstructs all invariants from the trace.

**Why:** Averaging visited source indices preserves the prior hard-DTW path semantics while feeding the exact interpolation interface used by MFA-linear. It avoids adding a separate feature-averaging or vocoder path.

**Alternatives considered:**

- Average all visited TTS vectors directly, as in the earlier feature-target audit. Rejected because this changes both alignment and feature aggregation.
- Warp waveform samples from the DTW path. Rejected because it changes the decoder/interface and reintroduces phone-splice artifacts.
- Use soft-DTW. Rejected because temperature and regularization become additional tunable variables.

### 4. Keep natural reference information outside the emitted conditioning

Natural WavLM frames are read only while constructing cosine costs. Conditioning is generated by interpolating the paired TTS feature sequence at DTW-derived source coordinates. Candidate metadata binds natural and TTS feature array hashes, the path trace, conditioning hash, decoder output hash, PCM hash, and a `natural_values_in_conditioning=false` contract check. A test uses disjoint synthetic value ranges to prove emitted values come from TTS interpolation only.

**Why:** The project permits natural reference audio at inference, but replacement claims require that natural acoustics are not simply copied into the candidate.

### 5. Reuse the historical audio and media contracts exactly

Candidate generation calls the existing pinned `WavLMKNNVCAdapter`, `candidate_from_features`, canonical PCM16 helper, and waveform QC. Stage 00 rejects any disagreement between current assets and the historical candidate/strict-protocol bindings. Execution fixes Python/NumPy/PyTorch seeds, deterministic algorithms where supported, and the prior 08:00–23:00 GPU-use window. Before a GPU stage, the runner records `nvidia-smi` state and refuses a materially occupied device rather than competing with another job.

Diagonal rendering uses the same frozen Wav2Lip checkpoint, invocation flags, face video, ffmpeg/ffprobe binaries, and official SyncNet V2 model/pipeline as the parent. It creates a strict `DTW video + DTW audio` mux and verifies decoded candidate PCM equality before scoring.

**Why:** Keeping all downstream interfaces fixed makes alignment-map policy the intended independent variable.

### 6. Treat historical cells as immutable controls, not caches to rewrite

The new run stores lightweight copied score rows with pointers and hashes to the historical replacement manifest. It never copies or rewrites parent media. The diagonal comparison reads only historical `G_M_E_M`; replacement reporting, if promoted, reads historical `G_M_E_N` and `G_N_E_N`. Each row is joined by sample ID and source group and checked against the bound cohort before statistics.

**Why:** Re-rendering controls adds cost and creates avoidable renderer-version variance. The frozen hashes and complete engineering review make the existing cells valid controls.

### 7. Use a strict source-group cluster-bootstrap promotion gate

Stage 03 computes per-record:

```text
delta_C = SyncC(DTW video, DTW audio) - SyncC(MFA-linear video, MFA-linear audio)
delta_D = SyncD(MFA-linear video, MFA-linear audio) - SyncD(DTW video, DTW audio)
```

It uses 10,000 source-group cluster-bootstrap draws with fixed seed `20260903`, sampling source groups with replacement and retaining all rows of each sampled group. It reports means, percentile two-sided 95% intervals, per-record wins, and group counts. Promotion requires both lower confidence bounds to be `> 0`; no secondary metric can rescue either co-primary endpoint. The decision artifact includes its own canonical payload hash, all parent hashes, and `replacement_authorized`.

**Why:** The user asked to run replacement only if DTW has a TFG advantage. Requiring simultaneous positive lower bounds gives “advantage” an unambiguous pre-observation meaning and respects correlated records from the same source video.

**Alternatives considered:** Positive sample means alone or a majority-win rule. Rejected because either can promote a noisy, non-replicated difference.

### 8. Seal replacement structurally until promotion

The replacement command imports no DTW path-selection options and begins by validating Stage 03's self-hash and exact `replacement_authorized=true`. Before authorization it must not create `04_replacement/`. Once authorized, it reuses each Stage 02 DTW elementary video, muxes untouched natural audio with video stream copy and PCM s16le, decodes the muxed PCM, and requires byte identity with the source. It then runs official SyncNet once per record.

**Why:** A separate command and absent output directory make the gate observable and prevent accidental downstream execution.

### 9. Report three replacement contrasts and two distinct conclusions

If Stage 04 exists, Stage 05 reports:

1. DTW replacement minus natural baseline;
2. MFA-linear replacement minus natural baseline;
3. DTW replacement minus MFA-linear replacement.

For Sync-D, signs are reversed so positive always means better. All contrasts use the same source-group bootstrap and record-level wins. The terminal artifact has separate fields for `dtw_over_linear_replacement` and `replacement_safe_vs_natural`. A positive first field never upgrades a negative/uncertain second field.

If Stage 03 does not promote, Stage 05 instead records `NO_DTW_TFG_ADVANTAGE`, `replacement_status=sealed_not_run`, and zero replacement artifacts.

## Risks / Trade-offs

- **[Natural-conditioned path can overfit local WavLM similarity]** → Keep natural values out of conditioning, freeze all DTW choices before scoring, and require same-record paired downstream evidence.
- **[Hard-DTW ties can vary across implementations]** → Use float64 CPU costs, a documented predecessor order, exact synthetic tie tests, and stored full paths.
- **[Short phones provide little or no extra freedom over linear mapping]** → Report singleton/trivial-path rates and coordinate displacement; treat a failed advantage gate as the intended result rather than relaxing constraints.
- **[Historical renderer cells could be incompatible with the new run]** → Bind every model/tool/flag/input hash and refuse the comparison on any mismatch.
- **[Cluster bootstrap is sensitive to few source groups]** → Report source-group count, group-level summaries, intervals, and record wins; make no population-wide generalization claim.
- **[Candidate generation and 133 Wav2Lip renders are costly]** → Load each frozen model once per stage, support hash-validated resume, and avoid all control re-renders and any replacement work before promotion.
- **[A better diagonal score may still be co-adaptation]** → Keep diagonal and replacement conclusions separate and require untouched-natural comparison for replacement safety.

## Migration Plan

1. Add the isolated experiment package and synthetic/provenance/statistics tests.
2. Validate Stage 00 against immutable parent artifacts without opening new media or using the GPU.
3. Generate the 133 DTW candidates and run independent artifact validation.
4. Render and score the 133 diagonal cells, then finalize the pre-registered gate.
5. Run Stage 04 only if the gate explicitly authorizes it; otherwise create only the terminal sealed decision.
6. Preserve all historical runs unchanged. Rollback consists of stopping the new runner and removing only incomplete temporary files in the new run root; completed stage artifacts remain as evidence and are never overwritten.
