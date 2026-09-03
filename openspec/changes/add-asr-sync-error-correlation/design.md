## Context

See `proposal.md` for motivation and `specs/asr-sync-error-correlation-experiment/spec.md` for the behavior contract.

The repository already contains the required paired assets and most of the evaluation mechanics:

- The canonical n=500 LRS3 source manifest is `runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json`; it provides source video, natural 16 kHz audio, official transcript path, transcript, timing-QC metadata, and hashes.
- The matching raw TTS metadata is `runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts_meta.json`; it provides canonical 16 kHz TTS audio and hashes.
- `runs/lrs3_mfa_linear_replacement_20260824/00_protocol_lock_retry1/manifest.json` freezes 24 fit-only `fresh_confirmation` records, one per source group, and carries sealed-split provenance.
- `scripts/experiments/lrs3_mfa_linear_replacement/strict_mux.py` and the equivalent tested contract in `scripts/experiments/lrs3_direct_audio/run_wavlm_hifigan_lrs3.py` establish PCM-preserving video/audio replacement.
- `third_party/syncnet_python/SyncNetInstance.py` computes five-video-frame/twenty-MFCC-frame embeddings, a `T x (2*vshift+1)` distance matrix, whole-track offset/Sync-D/Sync-C, and a median-filtered framewise confidence. Whole-clip `Confidence` is not the confidence of an independently scored five-frame clip.
- `third_party/syncnet_python/run_pipeline.py` stores face-track frame ranges in `tracks.pckl`; cropped track audio begins at `track.frame[0]/25`, which must be added back when mapping local scores to the original audio clock.
- The main `.venv` currently contains PyTorch/torchaudio 2.5.1, Transformers 4.51.3, Hugging Face Hub 0.36.2, SciPy, NumPy, and SoundFile. `torchaudio.functional.forced_align` is available. Matplotlib is absent and is the only expected plotting dependency to add. SyncNet remains isolated in `~/.venvs/syncnet` because its dependency set differs.

The LRS3 official word timings describe natural speech only. They cannot be applied directly to raw TTS audio. The design therefore uses the same uncorrected greedy ASR output for error identity in both arms, but a separate per-arm CTC forced alignment solely to place reference words on each arm's clock.

## Goals / Non-Goals

**Goals:**

- Make one copy-paste entry point run the experiment in ordered, inspectable stages.
- Keep every scientific choice in one frozen config and every stage bound to content hashes.
- Expose exact ASR words/timestamps, reference timing, edit operations, SyncNet distances, local scores, masks, and metrics so a result can be audited without rerunning a model.
- Keep vendor code unchanged; wrap the existing SyncNet implementation and verify numerical parity.
- Make interruption recovery cell-local and deterministic without building a workflow framework.

**Non-Goals:**

- Training, fine-tuning, calibrating, or comparing ASR models.
- Using Whisper, cloud ASR, a language model, a beam decoder, or a lexicon.
- Treating source groups as verified speaker identities.
- Opening sealed internal-dev, validation, or test media.
- Correcting TTS duration, globally warping time, or producing new lip-synced video.
- Claiming causality between audio quality, ASR errors, and lip synchronization.
- Generalizing this package into a reusable experiment SDK, service, dashboard, database, or distributed job system.

## Decisions

### 1. Package shape and one orchestration entry point

Add the narrow package below:

```text
scripts/experiments/asr_sync_error_correlation/
├── __init__.py
├── config.py              # frozen constants and CLI config validation
├── io.py                  # strict JSON, hashes, atomic writes, success markers
├── preflight.py           # environment/model/cache checks
├── protocol.py            # parent joins and 24x2 manifest
├── asr_ctc.py             # emissions, greedy decode, forced alignment
├── word_errors.py         # normalization, edit DP, error spans
├── strict_inputs.py       # thin reuse of strict mux contract
├── syncnet_adapter.py     # runs in the SyncNet environment; exports distances
├── local_sync.py          # parity, track/time mapping, local confidence
├── analyze.py             # masks, metrics, bootstrap, decisions
├── visualize.py           # deterministic paired timeline PNGs
└── run.py                 # stage runner and --resume

tests/experiments/asr_sync_error_correlation/
├── test_protocol.py
├── test_asr_ctc.py
├── test_word_errors.py
├── test_local_sync.py
├── test_analyze.py
└── test_resume.py
```

`run.py` is the only operator-facing entry point. Modules remain importable for tests, but no plugin registry, base stage class, DAG engine, or configuration framework is introduced.

Alternatives considered:

- Five unrelated numbered scripts: easy initially, but encourages hand-edited paths and skipped stages.
- A general pipeline framework: unnecessary for seven linear stages and conflicts with the no-overengineering goal.

### 2. Operator commands and model acquisition boundary

The intended commands are:

```bash
source /home/wjj/tts-audio/tts-exp/.venv/bin/activate

# One-time network-enabled acquisition plus environment smoke tests.
python -m scripts.experiments.asr_sync_error_correlation.run \
  --stage preflight \
  --run-dir runs/lrs3_asr_sync_error_correlation_20260831 \
  --acquire-asr-model \
  --device auto

# Full run. This path must load the locked snapshot offline.
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m scripts.experiments.asr_sync_error_correlation.run \
  --run-dir runs/lrs3_asr_sync_error_correlation_20260831 \
  --device auto \
  --resume
```

For debugging, `--stage prepare|mux|asr|sync|align|analyze|plot` runs that stage only after validating all required predecessor markers. `--resume` never weakens validation.

The implementation adds a small experiment requirements/constraints file recording the tested main-environment versions and Matplotlib. It does not recreate or upgrade the SyncNet environment. Preflight invokes both Python executables and prints an exact remediation command for a missing dependency; it does not install packages implicitly.

Model acquisition uses `huggingface_hub.snapshot_download` with model ID `facebook/wav2vec2-large-960h-lv60-self`. Network-enabled acquisition resolves `main` to the returned immutable commit SHA, hashes the snapshot files used by Transformers, and writes a lock. Data stages accept only the immutable lock and call model/processor loading with `local_files_only=True`. The repository-local cache root defaults to `checkpoints/huggingface`; both cache path and lock are configurable but recorded.

Alternatives considered:

- A persistent inference server: adds lifecycle and protocol failure modes with no benefit for 48 short files.
- Loading `main` for every run: simpler but not reproducible.
- Downloading during ASR inference: makes resume and offline reruns unreliable.

### 3. Frozen configuration

Before media processing, write `00_preflight/config.json` with at least:

```json
{
  "schema_version": 1,
  "seed": 20260831,
  "arms": ["natural", "tts"],
  "asr": {
    "model_id": "facebook/wav2vec2-large-960h-lv60-self",
    "decoder": "ctc_argmax_greedy",
    "sample_rate_hz": 16000,
    "batch_size": 1,
    "dtype": "float32"
  },
  "syncnet": {
    "fps": 25,
    "vshift": 15,
    "min_track": 50,
    "embedding_video_frames": 5,
    "embedding_audio_mfcc_frames": 20,
    "median_filter_width": 9,
    "checkpoint_sha256": "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
  },
  "analysis": {
    "low_score_k": 1.0,
    "low_score_comparison": "strict_less_than",
    "std_ddof": 0,
    "bootstrap_draws": 10000,
    "bootstrap_seed": 20260831
  }
}
```

CLI overrides are limited to operational paths, device, stage, resume, and explicit model acquisition. Scientific constants are not exposed as casual CLI flags. Changing a scientific constant requires a new config/code hash and therefore a new run directory.

### 4. Run directory and stage contracts

Use this fixed layout:

```text
runs/lrs3_asr_sync_error_correlation_20260831/
├── 00_preflight/
│   ├── config.json
│   ├── environment.json
│   ├── asr_model_lock.json
│   ├── assets.json
│   └── decision.json
├── 01_manifest/
│   ├── manifest.json
│   ├── test_lock.json
│   └── decision.json
├── 02_inputs/
│   ├── natural/<sample_id>.mkv
│   ├── tts/<sample_id>.mkv
│   ├── cells/<sample_id>.<arm>.json
│   └── failures.json
├── 03_asr/
│   ├── emissions/<arm>/<sample_id>.npz
│   ├── records/<arm>/<sample_id>.json
│   ├── cells/<sample_id>.<arm>.success.json
│   └── failures.json
├── 04_sync/
│   ├── work/<arm>/<sample_id>/...
│   ├── distances/<arm>/<sample_id>.npz
│   ├── records/<arm>/<sample_id>.json
│   ├── cells/<sample_id>.<arm>.success.json
│   └── failures.json
├── 05_alignment/
│   ├── records/<arm>/<sample_id>.json
│   ├── grids/<arm>/<sample_id>.npz
│   ├── cells/<sample_id>.<arm>.success.json
│   └── failures.json
├── 06_analysis/
│   ├── per_record.json
│   ├── summary.json
│   └── decision.json
├── 07_plots/
│   ├── <sample_id>.png
│   └── index.json
├── logs/<stage>/<arm>/<sample_id>.log
├── summary.json
└── decision.json
```

A stage writes a temporary sibling file, validates it, then atomically renames it. A cell success marker contains hashes of its direct inputs, config, relevant source files, model lock/checkpoint, outputs, and runtime identity. Resume trusts no bare output file.

NPZ files contain only non-object NumPy arrays and are loaded with `allow_pickle=False`. JSON uses `allow_nan=False`; undefined statistics use `null` plus a reason code.

### 5. Manifest construction and strict input preparation

`protocol.py` reads metadata before media:

1. Validate the stage-00 parent manifest, decision, and test lock.
2. Copy the ordered `fresh_confirmation.records` array exactly.
3. Join each sample to the canonical source manifest by `sample_id`.
4. Join each sample to TTS metadata by `sample_id`.
5. Verify source group, transcript, video hash, natural audio hash, TTS audio hash, and TTS reference-audio hash.
6. Parse the official `.txt` metadata and require the normalized `Text:` line and ordered timing-table words to match the canonical transcript.
7. Expand each sample into adjacent `natural` then `tts` arm-records.

The manifest stores repo-relative paths where possible and hashes all parent files. It copies the inherited sealed-split lock and records that no sealed media was accessed.

`strict_inputs.py` calls the existing strict mux behavior rather than reimplementing FFmpeg policy. The natural and TTS arm both receive a new MKV so container/codec handling is symmetric. TTS audio longer than the source video is not truncated; later analysis explicitly uses only score points supported by the selected video track. Audio outside that overlap remains visible in ASR/error diagnostics and is counted as excluded from the correlation grid.

### 6. ASR emissions, greedy decoding, and timestamps

For each standalone canonical arm audio:

1. Verify mono, 16 kHz, finite PCM and the manifest hash.
2. Load processor/tokenizer/model from the locked local snapshot.
3. Run one waveform at a time under inference mode in float32.
4. Save float32 log-probabilities (or logits plus an explicit representation field), frame argmax IDs, input sample count, output frame count, and model/tokenizer bindings in compressed NPZ/JSON.
5. Collapse argmax IDs without any reference information.

Greedy timing uses the model convolution stride:

```text
frame_stride_s = model.config.inputs_to_logits_ratio / 16000
span = [first_frame * frame_stride_s,
        (last_frame + 1) * frame_stride_s)
```

Spans are clipped only to the decoded audio duration. A collapsed token retains all contributing frame indices. Word spans run from the first non-delimiter token start through the last non-delimiter token end. Diagnostic word confidence is the geometric mean of contributing argmax posterior probabilities, computed in log space. Confidence never changes a word or an error label.

The versioned normalizer is deliberately small:

1. Unicode NFKC.
2. Convert typographic apostrophes to ASCII `'`.
3. Uppercase.
4. Preserve `A-Z` and apostrophe; replace every other character with a space.
5. Collapse whitespace and strip.

Preflight verifies that every character and delimiter required by the frozen 24 transcripts maps to non-unknown tokenizer IDs.

### 7. Per-arm reference timing by CTC forced alignment

Use `torchaudio.functional.forced_align` from the audited main environment against the saved arm emissions and tokenizer IDs. This is not a second recognizer and does not alter greedy output.

- Encode the normalized full reference with the model tokenizer, excluding blank IDs and retaining the word delimiter.
- Run forced alignment on log probabilities.
- Merge aligned token frames into reference word spans with the same frame stride.
- Require every reference word to have a non-empty, monotonic span inside arm audio.
- Store token scores and word-level minimum/mean alignment scores as diagnostics, but do not filter or relocate words by score.

For the natural arm, parse official LRS3 word spans and join by normalized word index. Report start error, end error, midpoint error, median absolute errors, and 95th percentiles. Because forced alignment is used for both arms for symmetry, official timings are a QA diagnostic, not a replacement. Any token-count disagreement among manifest transcript, `Text:` line, timing table, and normalized reference fails the record.

Alternative considered: official timings for natural plus interpolated timings for TTS. This is invalid because raw TTS duration and prosody differ. MFA would add a separate model/dictionary/OOV pipeline already known to be brittle in this repository; the available CTC forced aligner is smaller and uses the emissions already produced.

### 8. Deterministic word edit alignment and error intervals

Implement a conventional `(N+1) x (M+1)` dynamic-programming table over reference and greedy words with insertion, deletion, and substitution cost 1. During backtrace:

1. Take an exact-match diagonal when valid.
2. Otherwise prefer substitution on a cost tie.
3. Then prefer deletion.
4. Then prefer insertion.

Emit atomic operations with stable IDs. Compute WER as `(substitutions + deletions + insertions) / reference_word_count`.

Timing assignment is arm-local:

- equal: no error span;
- deletion: forced reference word span;
- insertion: greedy word span;
- substitution: min start and max end across its forced-reference and greedy spans.

Group adjacent non-equal operations in alignment order. A group's interval is the union of every participating reference/prediction span. Merge groups whose half-open intervals overlap or touch within `1e-9` seconds. Store both atomic and merged forms so tests can verify no interval lost provenance.

### 9. SyncNet adapter, track selection, and exact local score

Do not edit `third_party/syncnet_python`. The main runner invokes:

1. Existing `run_pipeline.py` in `~/.venvs/syncnet` with strict mux, unique reference, `--min_track 50`, and `--overwrite` only inside a new cell work directory.
2. `syncnet_adapter.py` with the same interpreter. It loads `tracks.pckl`, selects the longest track with lowest-index tie-break, loads `syncnet_v2.model`, calls the existing `SyncNetInstance.evaluate` on the corresponding crop AVI, and writes the returned distance matrix plus track metadata.

Selection happens before looking at any distance value. The selected natural/TTS tracks for a sample are expected to match because video is copied unchanged; mismatch is recorded and fails engineering completeness rather than choosing the better arm.

For returned `dists` with shape `[T, 31]`:

```text
mdist[j]            = mean_t(dists[t, j])
j_star               = argmin_j(mdist[j])
av_offset_frames     = 15 - j_star
sync_d               = mdist[j_star]
sync_c               = median(mdist) - sync_d
local_c_raw[t]       = median(mdist) - dists[t, j_star]
local_c_filtered     = scipy.signal.medfilt(local_c_raw, kernel_size=9)
```

Parity checks:

- offset returned by existing `evaluate` equals recomputed offset exactly;
- returned confidence and recomputed Sync-C differ by at most `1e-6` before text formatting;
- values parsed from the upstream three-decimal log differ from recomputed offset/Sync-D/Sync-C by at most `5e-4` plus floating-point epsilon;
- all distances and scores are finite.

Time mapping follows the code's actual padding convention. Let `s` be selected track's first source frame and let local distance row be `t`. Column `j_star` corresponds to unpadded audio-feature index:

```text
a = t + j_star - vshift = t - av_offset_frames
audio_center_s = (s + a + 2) / 25
```

The `+2` is the center of the five-frame embedding window. Exclude rows when `a<0`, `a>=T`, the mapped center is outside the arm audio/track support, or the row is among the first/last four local rows whose 9-value median filter would use zero padding. Record exclusion counts by reason. This prevents SyncNet's shift padding or median-filter padding from becoming artificial low-sync events.

### 10. Grid, masks, and record metrics

The valid grid is the ordered set of retained `audio_center_s` values; no interpolation is needed because SyncNet already emits one point per 25 Hz step. At every center:

```text
error_mask[t] = any(error_start <= center < error_end)
threshold     = mean(local_c_filtered) - population_std(local_c_filtered)
low_mask[t]   = local_c_filtered[t] < threshold
badness[t]    = -local_c_filtered[t]
```

Compute counts first, then metrics from counts. Do not insert epsilons into denominators. The JSON null rules are those in the spec. Spearman uses the binary error mask and continuous badness, not `-low_mask`. `scipy.stats.spearmanr` warnings for constant inputs are converted into the specified null reason.

Record badness contrast is:

```text
mean(badness[error_mask]) - mean(badness[~error_mask])
```

Positive Spearman and positive contrast support the hypothesis. The low mask supports the requested overlap metrics but does not replace the continuous association endpoint.

### 11. Aggregation, uncertainty, and decisions

Aggregate in three views:

- `natural`: all analyzable natural records;
- `tts`: all analyzable TTS records;
- `paired`: samples with both required metrics defined in both arms.

Count-based IoU/precision/recall are micro aggregates from summed intersections/denominators. Macro reports include count, mean, median, quartiles, min, and max of defined per-record values. WER is reported per record and by arm.

For each arm, bootstrap the median Spearman and median badness contrast by resampling whole sample records with replacement 10,000 times using NumPy `Generator(PCG64(20260831))`. Use the 2.5 and 97.5 percentiles. For cross-arm deltas, resample paired sample rows and recompute the median of `tts-natural`. No frame is treated as an independent resampling unit.

Decision logic is implemented once in `analyze.py` exactly as specified:

1. Any required engineering failure -> engineering `NO_GO`, science `NOT_EVALUATED`.
2. For each arm, insufficient defined records/error cells/groups -> `INSUFFICIENT`.
3. Otherwise both lower confidence limits greater than zero -> `SUPPORT`; else `NO_SUPPORT`.
4. Preserve natural and TTS statuses separately. A top-level label may summarize the tuple (`SUPPORTED_BOTH_ARMS`, `SUPPORTED_TTS_ONLY`, `SUPPORTED_NATURAL_ONLY`, `NO_SUPPORT`, or `INSUFFICIENT`) but must not hide the component statuses.

### 12. Minimal plots

Use Matplotlib's non-interactive `Agg` backend. One PNG per sample has two stacked arm panels and a shared x-axis up to the larger arm duration:

- dark line: filtered local confidence;
- dashed line: arm threshold;
- translucent red spans: ASR error intervals;
- light gray spans: regions excluded from the SyncNet grid;
- title: sample ID, arm, WER, Sync-C, offset, Spearman;
- footer: config hash and record-status identifiers.

Use fixed figure size, DPI, colors, and fonts. No interactive HTML or dashboard is added. `index.json` maps sample IDs to plot paths and arm statuses.

### 13. Failure and resume behavior

A cell failure appends one structured row:

```json
{
  "stage": "04_sync",
  "sample_id": "lrs3_...",
  "arm": "tts",
  "exception_type": "RuntimeError",
  "message": "short secret-free message",
  "log_path": "logs/04_sync/tts/lrs3_....log",
  "retryable": true
}
```

The runner finishes other independent cells so diagnostics are complete, then exits nonzero and writes an incomplete stage decision. It does not aggregate science until all 48 required cells pass.

On resume, each cell is handled independently:

- valid marker + matching bindings + valid output hashes -> reuse;
- missing marker, mismatched binding, corrupt output, or previous failure -> recompute into temporary paths;
- unrelated successful cells -> leave untouched.

No code deletes the run root. Operator cleanup is manual and therefore visible.

### 14. Test strategy

Tests use tiny synthetic arrays and fixtures; they do not download Wav2Vec2 or run S3FD in the normal unit suite.

Required coverage:

- parent manifest joins, frozen order, 24x2 expansion, and sealed-split rejection;
- normalizer and tokenizer representability;
- CTC collapse including blanks, repeated letters separated by blanks, delimiter timing, and confidence;
- forced-alignment word grouping with a tiny fabricated emission;
- edit-DP tie order and exact spans for equal/insert/delete/substitute/many-to-many blocks;
- SyncNet formulas against a hand-built distance matrix;
- track tie-break, nonzero track-start mapping, positive/negative offsets, shift-padding exclusion, and median-filter-edge exclusion;
- masks at half-open boundaries and strict threshold equality;
- null metric behavior and strict finite JSON;
- bootstrap determinism and paired resampling;
- decision thresholds immediately below/at/above each gate;
- resume reuse, stale-binding recomputation, and non-empty-directory refusal.

One opt-in integration smoke test runs a short local fixture through the locked Wav2Vec2 snapshot. A second opt-in test runs the existing SyncNet evaluator and adapter on one already available fit sample and checks parity. Full n=48 execution is an experiment run, not a unit test.

## Risks / Trade-offs

- **[Circular timing concern]** Forced alignment uses the same acoustic emissions as greedy ASR, so error location is not independent of the recognizer. → Keep error identity strictly greedy, use forced alignment only for missing/reference timing, retain natural official-timing diagnostics, and state this limitation in the final summary.
- **[Local confidence is derived, not a published independent Sync-C]** The upstream code defines whole-track Sync-C and logs framewise confidence but does not name a per-window Sync-C metric. → Call the series `local_c`, preserve the exact formula, export raw distances, and require whole-track parity; never label each point as standalone Sync-C.
- **[Global offset can absorb constant delay]** Scoring at the selected global offset measures local deviations after a whole-track shift, not absolute zero-offset synchronization. → Report global offset beside local association and limit the claim accordingly.
- **[Autocorrelated 25 Hz cells]** Naive cell-level significance would be anti-conservative. → Bootstrap whole samples/source groups and prohibit frame-level p-values.
- **[Sparse natural-arm errors]** A strong ASR may produce too few natural error cells. → Predeclare `INSUFFICIENT`; do not lower the gate or substitute a weaker model after observing results.
- **[TTS extends beyond video]** Some TTS files are longer than their source video. → Preserve raw audio, report excluded tails, and analyze only score-supported overlap.
- **[Face tracking variability]** Separate arm pipeline runs can theoretically produce different tracks despite copied video. → Use score-independent selection and require paired track metadata parity for engineering GO.
- **[CPU/GPU numerical differences]** Marginal logits could alter greedy words across devices. → Use one device and float32 for the entire run, record provenance, and do not mix device outputs in one decision.
- **[Current environment lacks Matplotlib]** Plotting would fail late if not addressed. → Make Matplotlib an explicit preflight dependency and smoke-test `Agg` rendering before media access.
- **[Model download mutability or interruption]** Loading an unpinned `main` can change behavior. → Resolve once, lock immutable revision and files, and require offline local loading thereafter.

## Migration Plan

This is an additive experiment; no existing run or API is migrated.

1. Add the package, tests, and a minimal pinned experiment dependency file.
2. Run unit tests without network/model downloads.
3. Acquire and lock Wav2Vec2 once; run preflight and both opt-in one-sample smoke tests.
4. Run the 24x2 experiment into a new empty run directory.
5. Validate all artifacts and decisions with the package's validator and `openspec validate --strict` for the change.
6. If implementation or smoke tests fail, remove only the newly created experiment package/run directory after inspecting them; existing manifests, checkpoints, SyncNet code, and sealed locks remain untouched.
