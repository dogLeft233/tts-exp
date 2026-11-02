# V3 enhancement CLI and configuration contract — attempt 03

**Status: source implemented, runtime unqualified.** The versioned entrypoint is `scripts/audio_enhancement/enhance_v3_attempt01.py`; the matching zero-parameter example is `scripts/audio_enhancement/configs/v3_zero_engineering_attempt01.json`. Neither has been run or runtime-qualified. The V3 head source still declares `PREPARATION_ONLY = True`; no trained V3 parameter checkpoint is available. The engineering-zero profile is not an evaluated method and does not represent identity: at zero theta the CLI requires this head to return its canonical RMS baseline exactly.

## Command

```text
python scripts/audio_enhancement/enhance_v3_attempt01.py \
  --input original.wav \
  --output enhanced.float32.wav \
  --config profile.json \
  --metadata enhanced.meta.json
```

All four paths are required. Existing output or metadata paths are rejected. The command accepts one RIFF/WAVE input and emits canonical mono 16 kHz IEEE-float32 WAV plus a JSON provenance sidecar. The enhanced WAV is only a TFG generator-drive signal. Delivery and primary scoring must use the exact untouched original input file as mux audio; the CLI never replaces or rewrites that input.

## Configuration schema

The config schema is `cross_tfg_v3_enhancement_cli_config_v1`. It uses exactly these top-level fields: `schema`, `profile_role`, `head_schema`, `implementation`, `constructor`, `parameter`, `source_pins`, `runtime`, `input`, and `output`. `candidate_provenance` is required for `METHOD_CANDIDATE` and forbidden for `ENGINEERING_ONLY`.

`implementation` and every `source_pins` entry contain exactly `{path, bytes, sha256}`. Paths are repository-relative POSIX paths. The source set must include the CLI entrypoint, V3 head, package initializer, audio decoder, RMS baseline, package pipeline, and overdub module. The loader verifies every declared file before imports and verifies imported module paths and hashes afterward. It rejects path escapes and symlinks. The implementation pin must match the V3 head and its identical `source_pins` entry.

`constructor` must contain every keyword field of the pinned `SharedRecipientTransitionHeadV3.__init__`, with no missing or extra fields. The supplied engineering profile has the complete frozen CPU02 constructor, including the 16 kHz / 512-sample Hann STFT, 128-sample hop, four band edges, reference band, flux/gate floors and quantiles, smoothing rules, mask bound, RMS policy, peak policy, and 25 fps clock. The loader does not silently use constructor defaults.

`parameter` is exactly:

```json
{
  "name": "band_contrasts_db",
  "shape": [3, 3],
  "dtype": "float32",
  "temporal_components": ["static", "positive_flux", "negative_flux"],
  "non_reference_band_order": [0, 1, 2],
  "values": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
}
```

The three rows correspond to the static, positive-flux, and negative-flux components. Columns follow ascending band order after omitting the configured reference band. The reference band's weight is fixed at 0 dB. All nine values must be finite float32-representable numbers.

`runtime` must bind `python_version` as three integers, `numpy_version`, `torch_version`, explicit `device` (`cpu` or `cuda[:index]`), and positive `torch_intraop_threads` and `torch_interop_threads`. There is no automatic device fallback. `ENGINEERING_ONLY` is CPU-only and rejects CUDA initialization.

The `input` policy is fixed to uncompressed RIFF/WAVE PCM or IEEE-float WAV, arithmetic-mean channel downmix, deterministic `numpy_kaiser_sinc_beta8.6_radius12_over_cutoff` resampling, mono float32, and 16 kHz. The `output` policy is RIFF/WAVE IEEE-float32 mono at 16 kHz with exact canonical input sample count, no overwrite, and `peak_violation: reject_without_clipping_or_post_gain`.

### Profile roles and candidate provenance

- `ENGINEERING_ONLY` requires all nine theta values to be exactly zero and forbids candidate provenance. It can be used for deterministic I/O, clock, and metadata qualification only. It must not be relabeled as a method candidate or used to claim enhancement benefit.
- `METHOD_CANDIDATE` requires pinned `checkpoint`, `training_record`, and `selection_record` files. The parameter checkpoint is JSON schema `cross_tfg_v3_parameter_checkpoint_v1` and binds the V3 head, implementation pin, canonical constructor digest, and exact parameter matrix. The training record schema `cross_tfg_v3_candidate_training_record_v1` must have status `COMPLETE`, positive `completed_optimizer_steps`, and matching checkpoint and head hashes. The selection record schema `cross_tfg_v3_candidate_selection_record_v1` must have status `FROZEN` and bind the checkpoint and training-record hashes. The entrypoint rejects a candidate if any file is absent, unpinned, structurally inconsistent, or has a mismatched parameter matrix. The engineering config contains nine exact zeros and no candidate-provenance field; no checkpoint or training/selection record is fabricated to fill that profile.

No candidate artifact or training/selection record currently exists. The code defines a fail-closed future loading path; it does not manufacture or imply learned weights.

## Waveform, validation, and clock behavior

1. The decoder reads supported RIFF/WAVE PCM and IEEE-float formats, downmixes channels by arithmetic mean, and deterministically resamples to 16 kHz. It records source file SHA-256, original encoding, sample rate, channel count, and source frame count, alongside the canonical float32 PCM hash and count.
2. Empty, malformed, non-finite, and unsupported input is rejected. With the centered 512-sample reflect-padded STFT, canonical inputs of 256 samples or fewer are rejected.
3. The RMS baseline is produced from the exact canonical source using the constructor's positive finite `target_rms` and `rms_epsilon`. Effectively silent input is left unchanged by the baseline. A configured `peak_ceiling` is positive finite or null. The baseline cap is uniform attenuation, never hard clipping.
4. The head must return finite mono float32 output with the exact canonical source count. The implementation reports input, baseline, and final output RMS and peak. Final V3 output RMS may differ from `target_rms`; the CLI records its measured value instead of promising exact target RMS.
5. If `peak_ceiling` is finite and final output exceeds it, the CLI fails. It never hard-clips or applies a second gain after the head. The float WAV writer receives `peak_ceiling=None`, then the decoded output is checked for bit-identical PCM readback.
6. The source input remains untouched. The sidecar maps canonical samples to the original source clock using the source and canonical rates, sample-0 origin, exact counts, and `retiming: false`. For delivery or scoring, a mux consumer must recheck the original file SHA and attach the original file, not the enhanced WAV.

The result sidecar schema is `cross_tfg_v3_enhancement_cli_result_v1`. It binds config and implementation/source pins, role, constructor, theta values, runtime/device, original source-file identity, source encoding/rate/channels/count, canonical PCM hash/rate/count/resampler, enhanced file and decoded-PCM hashes, exact readback status, RMS/peak measurements, and one-pass timing. The timing explicitly excludes metadata serialization/write and is labeled a single cold invocation, not a latency benchmark.

## Latency profiling design

After runtime qualification, collect cold-start and steady-state timings separately for input read/decode/canonicalization, RMS baseline, head construction, head forward, float-WAV write/readback, and metadata output. For a later trained candidate, include checkpoint validation/load as a separate interval. Use at least three steady-state invocations for each declared duration bucket, and report median and p90, real-time factor, sample count/duration, device, thread settings, source/runtime pins, peak process memory, and CUDA peak allocation when applicable. Do not include TFG generation or SyncNet scoring in audio-transform latency. Do not present zero-theta timing as trained-method performance.

## Evidence correction

The independent NumPy fit audit completed with status `COMPLETE_WITH_NUMERIC_MISMATCHES`: 16 of 40 rows met every check; the remaining 24 rows differed only on the predeclared exact flux-identity diagnostic. This is a numerical diagnostic, not a candidate-effect or selection result. The zero-theta NumPy audit completed with status `COMPLETE_WITH_MEASURED_COMPARISONS`; its `activity_cells_u8` mask differs at exactly `(213, 6)`, `(247, 4)`, and `(256, 8)`, with 64,503 saved Torch active cells and 64,502 NumPy active cells. The source-only CPU qualification remains engineering evidence. The CLI source has not yet run, and no development/confirmation, model, scoring, training, or API run is introduced by this contract.
