# Ditto-native feature-head feasibility validation

## Scope

This note records the first downstream feasibility check for the natural-to-TTS feature-head idea. It is deliberately not a generalization result: the adapter was trained on a small training-only subset, and the Ditto/SyncNet check used one training-style control sample. No test split was used for fitting, prototype construction, or model selection.

## Interface audit

The current representation-only adapter uses `facebook/hubert-base-ls960` features with dimension 768. Ditto v0.4 HuBERT does not consume those tensors. Its native path is:

```text
16-kHz waveform
  -> hubert_streaming_fix_kv.onnx
  -> two 20-ms streaming outputs, 1024 dimensions
  -> mean over pairs of outputs
  -> 25-fps [T, 1024] feature sequence
  -> audio2motion (audio_feat_dim = 1024 + 35)
```

The Ditto PyTorch configuration points to `aux_models/hubert_streaming_fix_kv.onnx`; the frontend reports `feat_dim = 1024`, and its aggregation is `reshape(chunksize[1], 2, 1024).mean(1)`. The downstream model expects 1103 dimensions after concatenating the 1024-D audio feature with 8-D emotion, 2-D eye-open, 6-D eye-ball, and 63-D speaker-condition features.

Therefore the earlier 768-D checkpoint was not injected into Ditto. A separate 1024-D, zero-initialized residual adapter was trained in Ditto's native feature space.

## Implementation

Added `scripts/validate_ditto_native_feature_head.py`. It supports:

- native streaming HuBERT feature extraction;
- phone-conditioned train-only natural/TTS centroids;
- 1024-D residual adapter training;
- an ordinary Ditto baseline, an explicit identity wrapper control, and adapted feature injection through a runtime facade;
- PyTorch Ditto inference without modifying Ditto source or TensorRT engines.

The adapter checkpoint stores the native interface, frontend/aggregation description, prototype support, training pair IDs, and the explicit `training-set feasibility only` classification.

## Feasibility run

A small eight-pair subset from the remote wav2sem/AISHELL-style data was used solely to test the mechanics. The exact remote pair manifest, audio files, adapter checkpoint, rendered videos, and SyncNet logs remain on the authorized remote workspace and are not committed here; this report is therefore an archived result summary, not a standalone reproduction bundle. The native adapter trained successfully for two epochs:

- native dimension: 1024;
- frontend output rate: 25 fps;
- paired training items: 8;
- phone labels with both-arm support: 105;
- training loss: 0.00994 -> 0.00940;
- no NaN/Inf or interface shape failures.

The remote outputs were kept under the run directory and were not committed as repository artifacts.

## Matched downstream validation (new)

A second remote validation was run after the initial historical feasibility check. **The downstream SyncNet comparison used exactly one natural sample (`sample_id=1`)**, not eight samples. The eight pairs above were used to train the existing adapter; they were not all rendered and scored in this rerun. This is therefore still a single-sample, training-style feasibility result and is not a generalization evaluation.

All conditions used the same natural waveform, source image, Ditto PyTorch configuration, SyncNet model, and seed (`42`). The matrix was:

- ordinary: unwrapped Ditto baseline;
- identity: explicit passthrough wrapper;
- adapted: existing native adapter checkpoint;
- random: deterministic random residual with amplitude scale `0.1`, matched to the adapter's configured residual scale;
- raw TTS: same source image/seed with the previously generated TTS waveform.

The old checkpoint was only repackaged into a safe-load-compatible payload because the remote PyTorch version rejected legacy NumPy metadata under `weights_only=True`. Its state dictionary was preserved without retraining or parameter updates.

| Condition | Sync-C (Confidence) | Sync-D (Min dist) | AV offset |
|---|---:|---:|---:|
| natural + ordinary | 6.695 | 7.517 | 0 |
| natural + identity wrapper | 6.695 | 7.517 | 0 |
| natural + adapted | 6.623 | 7.550 | 0 |
| natural + random residual | 6.317 | 7.887 | 1 |
| raw TTS | 8.606 | 6.044 | 1 |

Relative to the explicit identity wrapper:

| Condition | ΔSync-C | ΔSync-D |
|---|---:|---:|
| ordinary | 0.000 | 0.000 |
| adapted | -0.072 | +0.033 |
| random residual | -0.378 | +0.370 |
| raw TTS | +1.911 | -1.473 |

For this one sample, the adapted condition did not improve either SyncNet metric: Sync-C decreased slightly and Sync-D increased slightly. The identity wrapper exactly matched ordinary Ditto, validating that the wrapper itself was transparent for this run. Random residual was worse, while raw TTS remained substantially better than all natural-feature conditions.

The matched output summary is retained on the authorized remote workspace at:

```text
runs/feature_head_feasibility_matched/strict_one/matched_results.json
```

The remote provenance included hashes for the natural/TTS audio, source image, repackaged adapter, SyncNet model, and Ditto inference entrypoint. No credentials or remote output files are committed here.

## Multi-sample matched downstream validation (new)

A follow-up matched validation expanded the downstream comparison to **four training-style samples (`sample_id=1,2,3,4`)**. The existing adapter was reused without retraining. `sample_id=1` is the previously recorded strict matched run; samples 2--4 were newly rendered and scored with the same control matrix. No held-out split was used, and samples 5--8 were not included in this completed run. The result is therefore still training-style feasibility evidence, not a generalization evaluation.

For every sample, ordinary and identity produced identical SyncNet values, so the explicit passthrough wrapper remained transparent. The per-sample results were:

| Sample | Condition | Sync-C | Sync-D | AV offset |
|---:|---|---:|---:|---:|
| 1 | natural + ordinary / identity | 6.695 | 7.517 | 0 |
| 1 | natural + adapted | 6.623 | 7.550 | 0 |
| 1 | natural + random residual | 6.317 | 7.887 | 1 |
| 1 | raw TTS | 8.606 | 6.044 | 1 |
| 2 | natural + ordinary / identity | 4.957 | 9.979 | 0 |
| 2 | natural + adapted | 4.844 | 10.124 | 0 |
| 2 | natural + random residual | 4.837 | 9.903 | 0 |
| 2 | raw TTS | 5.548 | 8.215 | 1 |
| 3 | natural + ordinary / identity | 4.737 | 9.725 | 0 |
| 3 | natural + adapted | 4.776 | 9.461 | 0 |
| 3 | natural + random residual | 4.852 | 9.341 | 0 |
| 3 | raw TTS | 5.936 | 7.308 | 1 |
| 4 | natural + ordinary / identity | 5.375 | 9.389 | 0 |
| 4 | natural + adapted | 5.825 | 8.696 | 0 |
| 4 | natural + random residual | 5.027 | 9.584 | 0 |
| 4 | raw TTS | 8.908 | 6.348 | 1 |

Relative to identity, the adapted deltas were:

| Sample | ΔSync-C | ΔSync-D | Both metrics improve? |
|---:|---:|---:|:---:|
| 1 | -0.072 | +0.033 | no |
| 2 | -0.113 | +0.145 | no |
| 3 | +0.039 | -0.264 | yes |
| 4 | +0.450 | -0.693 | yes |
| **mean** | **+0.076** | **-0.195** | **2/4 samples** |

The current adapter therefore shows a small favorable mean direction over four samples, but not consistent per-sample improvement: it improves both metrics on 2/4 samples and worsens both on 2/4 samples. This is materially weaker than the raw-TTS control, which improves Sync-C and lowers Sync-D relative to identity on all four samples. The amplitude-matched random residual is also not a reliable explanation: its effects change direction across samples and do not reproduce the raw-TTS advantage.

The remote summary is retained at:

```text
runs/feature_head_feasibility_matched/multi_4/matched_results.json
```

The remote output contains generated videos, SyncNet preprocessing directories, and logs; these artifacts were not added to Git. The adapter checkpoint was reused as-is (the earlier safe-load repackaging only), with seed `42` and random residual scale `0.1`.


The following values are retained as historical context only. They came from separate old runs and were not a same-run adapter-vs-identity control:

| Condition | Sync-C (Confidence) | Sync-D (Min dist) | AV offset |
|---|---:|---:|---:|
| natural waveform + ordinary Ditto baseline (historical run) | 4.678 | 8.535 | 1 |
| natural waveform + trained native adapter (historical run) | 4.997 | 8.266 | 1 |
| raw TTS waveform baseline | 8.606 | 6.044 | 1 |

The old 4.678/8.535 row was ordinary unwrapped Ditto, not an explicit identity wrapper. Its difference from the old adapted row must not be interpreted as a clean adapter effect. The matched validation above supersedes it for the identity-control question.

The result supports the following narrow conclusion:

> The feature-head-to-Ditto experimental route is technically viable when the adapter is retrained for Ditto's native 1024-D feature interface.

It does **not** support:

- reusing the old 768-D checkpoint;
- claiming that representation loss improvement transfers to SyncNet;
- claiming generalization or test-set performance;
- claiming that phone-centroid supervision is a verified TFG ground truth;
- training a waveform decoder before the native-feature route is properly evaluated.

The next valid step is a larger, pre-registered training/validation split run with at least identity, adapted, random-residual, natural, and raw-TTS controls, followed by held-out evaluation. If the adapted arm does not consistently improve Sync-C/Sync-D over identity across samples, the present weak target should be considered insufficient rather than tuned indefinitely.


## Held-out downstream validation

A preregistered held-out run was subsequently executed using four pairs that were not used to train the frozen adapter: `remote_wav2sem/10`--`remote_wav2sem/13`. The adapter checkpoint was loaded without retraining, target recomputation, or held-out tuning. The exact control matrix was `ordinary`, `identity`, `adapted`, `random`, and `raw_tts`; the seed was `42`, and the random-residual scale was `0.1`. The held-out preflight recorded the source-manifest SHA-256, adapter SHA-256, natural/TTS audio hashes, source-image hashes, and token counts. The run was classified as `heldout_downstream_evaluation`, not as another training-style feasibility run.

The SyncNet pipeline initially required two environment-specific corrections: the SyncNet subprocess needed the available ffmpeg directory on `PATH`, and these short clips required `--min_track 50` rather than the default 100-frame track threshold. These execution parameters were applied uniformly to all 20 videos. All 20 condition/sample combinations produced finite Sync-C and Sync-D values; no failed metric was converted to zero.

Per-sample identity-relative results are shown below. Positive ΔSync-C and negative ΔSync-D are improvements.

| Sample | Adapted ΔSync-C | Adapted ΔSync-D | Adapted both improve? | Random ΔSync-C | Random ΔSync-D | Raw TTS ΔSync-C | Raw TTS ΔSync-D |
|---:|---:|---:|:---:|---:|---:|---:|---:|
| 10 | +0.123 | -0.232 | yes | -0.572 | +0.588 | +2.426 | +0.122 |
| 11 | +0.334 | -0.459 | yes | +0.279 | +0.292 | +5.243 | -1.521 |
| 12 | -0.063 | -0.086 | no | -0.039 | +0.227 | +2.911 | -1.295 |
| 13 | +0.148 | -0.286 | yes | -0.494 | +0.744 | +0.918 | -0.431 |
| **mean** | **+0.136** | **-0.266** | **3/4** | **-0.206** | **+0.463** | **+2.875** | **-0.781** |
| **sample SD** | **0.162** | **0.154** | — | **0.400** | **0.245** | **1.793** | **0.764** |

Ordinary and identity were exactly equal on all four held-out samples (ΔSync-C = 0 and ΔSync-D = 0), providing a second transparency check for the wrapper. The adapted condition improved both metrics on three of four samples and had a favorable mean direction. An exact percentile bootstrap over the four paired samples gives a 95% interval of approximately `[-0.010, +0.281]` for mean ΔSync-C and `[-0.402, -0.136]` for mean ΔSync-D. With only four held-out samples, this is uncertainty information rather than a powered significance claim. The random residual control moved in the opposite mean direction and failed to improve both metrics on any sample, which argues against bounded perturbation alone explaining the adapted effect.

Raw TTS remained substantially stronger in Sync-C on every held-out sample and improved both metrics on three of four samples; its ΔSync-D on sample 10 was slightly positive. Thus the frozen phone-centroid adapter shows a reproducible but small held-out downstream effect relative to identity, while it does not close the gap to directly supplying TTS audio. This supports continuing with better supervision and larger validation, not claiming that the current weak target is solved.

The complete remote result and preflight records remain outside Git under the authorized run directory:

```text
runs/feature_head_heldout_20260809/heldout_results.json
runs/feature_head_heldout_20260809/preflight.json
```

## Provenance and reproducibility

- Ditto source: remote `third_party/ditto-talkinghead` checkout.
- Ditto configuration: v0.4 HuBERT PyTorch configuration.
- Feature frontend: `hubert_streaming_fix_kv.onnx`.
- Adapter run classification: training-set feasibility only.
- SyncNet: remote `third_party/syncnet_python`, `syncnet_v2.model`.
- No remote credentials are included in this document, source tree, checkpoint metadata, or Git history.
