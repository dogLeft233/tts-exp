# V3 enhancement CLI and configuration contract

**Status: proposed interface; not implemented.** The V3 source currently declares `PREPARATION_ONLY = True`. No trained V3 parameter artifact or production CLI is available. The zero-theta configuration described here is an engineering-only path for checking serialization, waveform length, and metadata behavior; it is not an evaluated enhancement method.

## Command interface

Proposed invocation:

```text
python -m scripts.audio_enhancement.enhance_v3 \
  --input original.wav \
  --output enhanced.float32.wav \
  --config profile.json \
  --metadata enhanced.meta.json
```

The command reads one uncompressed RIFF/WAVE input and writes one canonical mono IEEE-float32 WAV plus a JSON metadata sidecar. It must refuse to overwrite an existing output unless a future, explicitly designed `--overwrite` option is selected. This command is a contract example only; it must not be treated as runnable until the CLI implementation and source pins exist.

The enhanced WAV is an intermediate generator-drive signal. Final delivery and primary scoring must mux the untouched original input audio. The CLI must not replace, rewrite, normalize, or re-encode the original file.

## Configuration format

The JSON document uses schema `cross_tfg_v3_enhancement_cli_config_v1`. It binds the complete constructor, nine finite parameters, source/runtime pins, and an explicit use role. A loader rejects an unknown schema, missing constructor field, changed implementation pin, wrong parameter shape, non-finite parameter, or unsupported role. It never fills missing values from code defaults.

Example of the zero-theta engineering configuration shape (values below are the frozen CPU02 diagnostic constructor, not a fitted profile):

```json
{
  "schema": "cross_tfg_v3_enhancement_cli_config_v1",
  "profile_role": "ENGINEERING_ONLY",
  "head_schema": "cross_tfg_shared_recipient_transition_head_v3",
  "implementation": {
    "path": "scripts/audio_enhancement/shared_recipient_transition_head_v3.py",
    "sha256": "77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87"
  },
  "constructor": {
    "sample_rate_hz": 16000,
    "n_fft": 512,
    "hop_length": 128,
    "win_length": 512,
    "window_kind": "hann",
    "window_periodic": true,
    "stft_center": true,
    "stft_pad_mode": "reflect",
    "stft_normalized": false,
    "stft_onesided": true,
    "stft_return_complex": true,
    "band_edges_hz": [0.0, 500.0, 1500.0, 3500.0, 8000.0],
    "reference_band_index": 3,
    "flux_denominator_floor": 1e-8,
    "activity_floor": 1e-6,
    "flux_normalization_scale": 1.0,
    "positive_flux_quantile": 0.6,
    "negative_flux_quantile": 0.6,
    "gate_temperature": 0.15,
    "gate_smoothing_frames": 3,
    "gate_smoothing_pad_mode": "reflect",
    "mask_smoothing_frequency_bins": 3,
    "mask_smoothing_time_frames": 3,
    "mask_smoothing_pad_mode": "reflect",
    "frequency_common_mode_policy": "remove_per_frame",
    "max_abs_mask_db": 3.0,
    "target_rms": 0.0376838172675746,
    "rms_epsilon": 1e-12,
    "peak_ceiling": null,
    "baseline_match_atol": 2e-5,
    "video_fps_num": 25,
    "video_fps_den": 1,
    "frame_power_floor": 1e-12
  },
  "parameter": {
    "name": "band_contrasts_db",
    "shape": [3, 3],
    "dtype": "float32",
    "temporal_components": ["static", "positive_flux", "negative_flux"],
    "non_reference_band_order": [0, 1, 2],
    "values": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
  },
  "source_pins": {
    "canonical_audio_decoder": {
      "path": "scripts/audio_enhancement/audio_io.py",
      "bytes": 14294,
      "sha256": "7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004"
    },
    "rms_baseline": {
      "path": "scripts/audio_enhancement/baselines.py",
      "bytes": 15123,
      "sha256": "7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2"
    }
  },
  "runtime": {
    "numpy": "1.26.4",
    "torch": "2.5.1+cu124",
    "device": "cpu"
  }
}
```

`band_contrasts_db` is a 3-by-3 float32 matrix: rows are static, positive-flux, and negative-flux components; columns are the three non-reference bands. The fourth reference band is fixed at 0 dB. The matrix is the complete learned parameter set for this source revision: nine values, with no implicit optimizer state or other trainable tensors. The constructor and all runtime-relevant source pins are serialized beside it. The loader verifies SHA-256 and byte count for every declared source pin before constructing the module, then copies the validated matrix into `band_contrasts_db`.

An actual method candidate must use a distinct role such as `METHOD_CANDIDATE` and must also bind the frozen fit/selection record, checkpoint bytes/hash, and training report. A candidate profile cannot be created by relabeling the zero-theta engineering configuration. The zero configuration must retain `profile_role: ENGINEERING_ONLY`, all nine values exactly zero, and may be used only for engineering checks. Zero theta produces the canonical RMS baseline in this implementation; it is not an identity transform.

## Waveform and validation behavior

- Accept only PCM or IEEE-float RIFF/WAVE inputs supported by the pinned decoder. Decode and downmix multichannel audio by arithmetic channel mean. Convert to mono float32 at 16 kHz with the pinned deterministic resampler when the source rate differs.
- Reject empty, malformed, or non-finite decoded/canonical PCM. For the centered 512-sample reflect-padded STFT, reject canonical inputs of 256 samples or fewer. Record original file hash, source encoding, sample rate, channels, frame count, and the canonical PCM float32 hash and sample count.
- The enhanced output must be finite mono float32 with exactly the canonical input sample count. No time stretch, padding, trimming, frame insertion/deletion, or sample-index retiming is allowed. Record the input and output counts and hashes separately.
- The RMS baseline uses the constructor's positive finite `target_rms` and `rms_epsilon`; the frozen implementation leaves effectively silent inputs unchanged. The optional `peak_ceiling` must be positive finite or `null`. Baseline peak limiting is uniform attenuation, never hard clipping. Record source, baseline, and final-output RMS/peak measurements and the applied gains. The final V3 residual can move the output away from `target_rms`; the CLI must report that measured value rather than claim exact target equality.
- If a finite `peak_ceiling` is configured, reject an enhanced output above it. Do not silently clip or apply a second writer-side gain, because that would alter the head output. With `peak_ceiling: null`, retain the generated float32 samples unchanged. Write IEEE-float32 WAV without a second peak limiter and verify exact PCM readback.
- Keep the original file immutable. The sidecar binds its file SHA and source format to the canonical sample count and output hash. For downstream muxing, reference the original audio path/hash and its source clock; never substitute the enhanced WAV for natural audio in delivery or scoring.

## Metadata sidecar

The sidecar schema is `cross_tfg_v3_enhancement_cli_result_v1`. It records the config file hash, all source pins, profile role, parameter values, decoder/resampler revision, input file hash and source format, canonical sample rate/count/hash, enhanced WAV file hash and decoded PCM hash, measured RMS/peak values, runtime/device, and per-stage timing. Its clock section declares source origin `sample_0`, canonical rate 16000 Hz, the rational source-rate relationship, identical input/output canonical sample counts, and `retiming: false`. A mux consumer must verify the metadata's original-file SHA before attaching that file as the audio stream.

## Latency measurement plan

Latency is measured only after an implementation exists. Separate file read/decode, canonicalization, baseline generation, head construction/load, warm-up, forward pass, WAV write, and readback validation. Report cold-start and steady-state timings separately; collect at least three steady-state runs per declared duration bucket and report median and p90. Record exact waveform sample count/duration, device, thread settings, runtime/source pins, peak process memory, and CUDA peak allocation when applicable. Report real-time factor as transform wall time divided by canonical audio duration. Do not include TFG generation or SyncNet scoring in the audio-transform latency number. The zero-theta engineering profile must not be presented as a trained-method speed result.

## Current evidence boundary

The CPU02 qualification and zero-theta CPU run establish numerical/serialization engineering behavior only. The independent fit NumPy audit completed with status `COMPLETE_WITH_NUMERIC_MISMATCHES`: 16 checks passed and 24 predeclared flux-identity checks did not meet their exact identity criterion. The zero-theta NumPy consumer completed with status `COMPLETE_WITH_MEASURED_COMPARISONS`; three `activity_cells_u8` entries differ at the frozen `1e-6` floor. These outcomes remain attached to their original artifacts. This contract introduces no method effect, trained checkpoint, user-facing executable, or new model/data/API run.
