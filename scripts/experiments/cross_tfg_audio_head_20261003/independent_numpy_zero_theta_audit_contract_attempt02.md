# Independent NumPy audit for the zero-theta supplement — attempt 02

## Purpose and boundary

This is a source-only, unexecuted audit consumer for the synthetic zero-parameter gradient supplement. It independently reconstructs the probe and V3 NumPy signal path, then compares the sealed CPU02 and zero-theta supplement arrays, baseline hash, support/gate/mask/power fields, objective values, and coordinate/directional finite differences. It imports only the pinned independent NumPy consumer; it does not import the V3 producer/head or Torch.

This is an engineering consistency check. It reads no fit waveform files beyond the source already embedded in the sealed CPU02 matrix archive, writes no waveform samples, and makes no training, frontend, generator, video, SyncNet, development, confirmation, or efficacy claim. Any numerical differences remain in the output. The consumer accepts the supplement's three declared terminal statuses and checks that its result and manifest agree; a non-PASS status is preserved as evidence rather than relabeled.

## Sealed local sources

| Role | Path | Bytes | SHA-256 |
|---|---|---:|---|
| Consumer | `scripts/experiments/cross_tfg_audio_head_20261003/independent_numpy_zero_theta_audit_attempt02.py` | 43,946 | `602894a6db479eaf125fddf35040c48090849c623ef2a299ff6a49dfdebf42eb` |
| Config | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_numpy_audit_attempt02/audit_config.json` | 6,632 | `2dfe229aa2bfb1ec21781502afdf9bbec6968643e26b7a304950c39884aa13e3` |
| Independent signal consumer03 | `scripts/experiments/cross_tfg_audio_head_20261003/independent_numpy_audit_shared_recipient_transition_head_v3_attempt03.py` | 98,390 | `ed540178ae9564a567f5939dd1fdb8560d422bf1a301d8d3f01c06da972fe40b` |
| Signal consumer config05 | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_independent_numpy_audit_attempt05/audit_config.json` | 5,382 | `7d2696cb49f2a459d0d065a3cfe14a83a2a90811f71879594cfc6ddbbcdf92c5` |
| CPU qualification worker02 | `scripts/experiments/cross_tfg_audio_head_20261003/qualify_shared_recipient_transition_head_v3_cpu_attempt02.py` | 55,636 | `4f5a1d272eb57fa331241a2063bbb25e8c91b28b61d826487ec80a24d1cf9eb3` |
| CPU qualification config02 | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02/qualification_config.json` | 10,721 | `0abfbb9797a84254e6d0a6e72266862a079e5066f2fec394e0796b0c2664d2f5` |
| Zero-theta worker | `scripts/experiments/cross_tfg_audio_head_20261003/qualify_shared_recipient_transition_head_v3_zero_theta_gradient_attempt01.py` | 27,243 | `6193181ccb890fe1f9b49d97e7ab2960499d511816693d3de48669eca4a3c536` |
| Zero-theta config | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_gradient_attempt01/qualification_config.json` | 5,396 | `594f5e4ff86e373d99a69815596148b72e7bdf58f9b9d6c2d65c37e4ebdbd1dd` |
| V3 head provenance | `scripts/audio_enhancement/shared_recipient_transition_head_v3.py` | 32,233 | `77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87` |
| NumPy RMS baseline | `scripts/audio_enhancement/baselines.py` | 15,123 | `7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2` |

The attempt02 consumer and config pin one another by the consumer entry in `source_pins`. The CPU02 readback is locally present at `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02_readback_attempt01/files/`; its result manifest is 1,076 bytes, SHA-256 `d6e3641557d65819295146623d47502ff2edc0097afa23e1b3fc9bd5336f986d`. The zero-theta supplement output has not been locally restored in this workspace yet, so its final CLI input path and raw manifest hash remain pending. The consumer requires the actual restored output and verifies the full manifest/member seals and pinned config/source bindings before analysis.

## Runtime and invocation shape

Pinned local runtime: `C:\Users\dogle\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`, CPython 3.12.14, NumPy 2.3.5. `PYTHONNOUSERSITE=1`; OMP/MKL/OpenBLAS/NumExpr thread counts are each one. Torch must remain absent from `sys.modules`.

After ROOT has reviewed this contract and the missing zero-supplement readback is restored, the consumer entry point is:

```powershell
& 'C:\Users\dogle\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B `
  'scripts/experiments/cross_tfg_audio_head_20261003/independent_numpy_zero_theta_audit_attempt02.py' `
  --repo-root 'E:\Documents\tts-exp' `
  --config 'runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_numpy_audit_attempt02/audit_config.json' `
  --cpu02-output-dir '<actual restored CPU02 output directory>' `
  --zero-supplement-output-dir '<actual restored zero-theta output directory>' `
  --output-dir 'runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_zero_theta_numpy_audit_attempt02/output'
```

No invocation has been made from this contract. Output is bounded to the configured NPZ/JSON/manifest files; the NPZ contains derived signals and diagnostics but no source or output waveform arrays.
