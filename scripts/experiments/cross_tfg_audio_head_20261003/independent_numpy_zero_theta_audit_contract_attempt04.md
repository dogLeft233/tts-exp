# Independent NumPy audit for the zero-theta supplement — attempt 04

## Purpose and scope

This is a source-only, unexecuted consumer for the synthetic zero-parameter gradient supplement. It synthesizes the declared waveform and objective in NumPy, reloads the pinned independent NumPy signal consumer, and compares the sealed CPU02 and zero-theta outputs: baseline hashes, valid/active/activity support, fluxes, gates, masks, frame power, sample clocks, objective values, and coordinate/directional finite differences. It imports neither the V3 producer/head nor Torch.

The consumer preserves measurements and comparisons. It accepts the supplement worker's three declared terminal statuses only when the result and manifest agree, and validates both against the supplement's pinned config/source observations. It reads no fit WAV, development audio, or confirmation audio; it does not persist waveforms or run a frontend, generator, video, scorer, API, optimizer, or candidate selection.

## Sealed source pins

| Role | Path | Bytes | SHA-256 |
|---|---|---:|---|
| Zero-theta NumPy consumer04 | `scripts/experiments/cross_tfg_audio_head_20261003/independent_numpy_zero_theta_audit_attempt04.py` | 44,759 | `27f472cb6e6ba15a8a6a6dadf7b541856a565d8c97817bcc059e92d720fd60bb` |
| Consumer config04 | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_numpy_audit_attempt04/audit_config.json` | 6,632 | `2939d4ff3fd259da4db9d989548a8e928a6be4771af2c1f40dfa1190e81dcfd3` |
| Independent signal consumer03 | `scripts/experiments/cross_tfg_audio_head_20261003/independent_numpy_audit_shared_recipient_transition_head_v3_attempt03.py` | 98,390 | `ed540178ae9564a567f5939dd1fdb8560d422bf1a301d8d3f01c06da972fe40b` |
| Signal consumer config05 | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_independent_numpy_audit_attempt05/audit_config.json` | 5,382 | `7d2696cb49f2a459d0d065a3cfe14a83a2a90811f71879594cfc6ddbbcdf92c5` |
| CPU qualification worker02 | `scripts/experiments/cross_tfg_audio_head_20261003/qualify_shared_recipient_transition_head_v3_cpu_attempt02.py` | 55,636 | `4f5a1d272eb57fa331241a2063bbb25e8c91b28b61d826487ec80a24d1cf9eb3` |
| CPU qualification config02 | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02/qualification_config.json` | 10,721 | `0abfbb9797a84254e6d0a6e72266862a079e5066f2fec394e0796b0c2664d2f5` |
| Zero-theta worker01 | `scripts/experiments/cross_tfg_audio_head_20261003/qualify_shared_recipient_transition_head_v3_zero_theta_gradient_attempt01.py` | 27,243 | `6193181ccb890fe1f9b49d97e7ab2960499d511816693d3de48669eca4a3c536` |
| Zero-theta config01 | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_gradient_attempt01/qualification_config.json` | 5,396 | `594f5e4ff86e373d99a69815596148b72e7bdf58f9b9d6c2d65c37e4ebdbd1dd` |
| V3 head provenance | `scripts/audio_enhancement/shared_recipient_transition_head_v3.py` | 32,233 | `77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87` |
| NumPy RMS baseline | `scripts/audio_enhancement/baselines.py` | 15,123 | `7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2` |

## Attempt04 corrections

- Resolve the config as a regular file under the repository root; artifact input/output paths remain directories.
- Refer to the CPU02 config using the actual frozen config-pin key `qualification_config02`; the supplement manifest retains its own declared `qualification_config` record.
- Form each sinusoid's angular-frequency scalar as float32 of the full Python-double product `2π × frequency`, matching the pinned PyTorch probe's scalar conversion point; waveform and objective analytic cross-checks remain descriptive, with no new synthesis tolerance.
- Derive the activity-cell mask locally from `_forward_numpy`'s returned `magnitude` and the pinned activity floor, then compare it with the supplement's saved `activity_cells_u8`. The pinned helper does not return an `activity_cells_u8` dictionary entry.

## Input binding and runtime

Pinned local runtime: `C:\Users\dogle\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`, CPython 3.12.14, NumPy 2.3.5. `PYTHONNOUSERSITE=1`; OMP, MKL, OpenBLAS, and NumExpr thread counts are one. Torch must remain absent from `sys.modules`.

The locally restored CPU02 output is `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02_readback_attempt01/files/`. Its 1,076-byte result manifest has raw SHA-256 `d6e3641557d65819295146623d47502ff2edc0097afa23e1b3fc9bd5336f986d`. The zero-theta supplement output has not been restored locally. When available, pass its actual local output directory; attempt04 verifies the manifest and every member's declared hash and size, then records the input manifest hashes. No input readback has been substituted or guessed.

Invocation after ROOT review and zero-supplement restoration:

```powershell
& 'C:\Users\dogle\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B `
  'scripts/experiments/cross_tfg_audio_head_20261003/independent_numpy_zero_theta_audit_attempt04.py' `
  --repo-root 'E:\Documents\tts-exp' `
  --config 'runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_numpy_audit_attempt04/audit_config.json' `
  --cpu02-output-dir 'runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02_readback_attempt01/files' `
  --zero-supplement-output-dir '<actual restored zero-theta output directory>' `
  --output-dir 'runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_numpy_audit_attempt04/output'
```

No command was run. The output is bounded by the config and contains only derived diagnostic arrays, result JSON, and a manifest; it contains no waveform samples.
