# V3 float32 flux-identity rounding diagnostic — attempt 02

## Purpose and scope

Attempt 02 is a fresh source-only arithmetic explanation of the preserved 40-row `fit_flux_balance_identity` boolean mismatch. It reads only the pinned CPU02 and independent NumPy03 artifacts. It does not read WAV/PCM, import Torch or the head, run a model/generator/scorer, change an earlier tolerance, or make a quality/transfer claim. Attempt 01 and its prior mismatch remain unchanged.

The output retains the full 40-row denominator, validity masks, transition frame indices and source-sample centers, the float32 positive/negative/total-delta paths, saved NumPy03 and CPU02 flux comparisons, and per-transition residual/bound arrays. It also records each row’s maximum residual and its frame, plus current/previous frame-total ratios and cancellation ratios.

## Arithmetic and checks

Inputs are the exact `fit_source_magnitude_f32` values from the sealed NumPy03 archive. The float64 reference promotes each original float32 magnitude operand before subtraction; it does not promote the already-rounded float32 bin difference. In float64 it separately calculates positive and negative bin-change totals, the direct bin-change total, and independently summed current-minus-previous frame totals. These are independent reductions over the same saved float32 inputs, not a source-waveform reanalysis or a claim of exact-rational arithmetic.

The binary32 bound uses `u=2^-24`, smallest subnormal `eta=2^-149`, and `gamma_(n−1) sum(abs(terms)) + n eta` for each 257-term reduction. It propagates operation bounds `u*abs(exact_result)+eta/2` across binwise subtraction, positive/negative reductions, separately rounded frame totals and their subtraction, denominator and scale divisions, the left/right subtraction, and the final rounded `L-R` residual subtraction. Separate bounds are saved for direct `sum(diff)` and the rounded frame-total delta. Every actual binary32 intermediate is checked for finiteness; any nonfinite value prevents a successful diagnostic because the declared no-overflow assumptions would not hold. The derived bound is descriptive and does not reclassify the preserved NumPy03 status.

The 40 rows and every valid transition remain in the denominator. No transitions are dropped, repaired, thresholded after inspection, or used for method selection. In particular, a residual outside the propagated bound is reported as such; it does not change the upstream record.

## Frozen source and input pins

| Role | Path | Bytes | SHA-256 |
|---|---|---:|---|
| Diagnostic source02 | `scripts/experiments/cross_tfg_audio_head_20261003/diagnose_v3_flux_float32_cancellation_attempt02.py` | 39,397 | `c0dc646a73b3c37440e96ab092ee7e699865fc6a48e8cfde79d3cf5da14659d1` |
| Diagnostic config02 | `runs/cross_tfg_audio_head_20261003/v3_flux_float32_cancellation_attempt02/config.json` | 7,209 | `d2d49c72890d6cb20255bb04070eff15095702fbe360c791f2e2d58a2b6f7192` |
| CPU02 output manifest | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02_readback_attempt01/files/result_manifest.json` | 1,076 | `d6e3641557d65819295146623d47502ff2edc0097afa23e1b3fc9bd5336f986d` |
| CPU02 matrices | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02_readback_attempt01/files/qualification_matrices.npz` | 3,019,026 | `ac84bd7327f68d9b4e1199bb6a401abc188500a34acf2de69a8469a6a6933818` |
| CPU02 row ledger | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02_readback_attempt01/files/results.jsonl` | 129,124 | `93de50973394f3eb5e3e1c0b58019a0155284779b3227f786fd87959e697ed70` |
| CPU02 summary | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02_readback_attempt01/files/summary.json` | 12,330 | `48559809f4473c84054f61c36ae6c724b6d2ad189172c3087a61fa47fd1d3012` |
| NumPy03 result manifest | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_independent_numpy_audit_execution_attempt01/result_manifest.json` | 1,190 | `d4b4d8c726df6ec3514fda7298be196783c1eef340a6e3de05db8fe59ce71b83` |
| NumPy03 matrices | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_independent_numpy_audit_execution_attempt01/independent_numpy_matrices.npz` | 49,896,372 | `142d83083f00c6fc09b53c516a4a8dbe6bbc1e97dba5fd39547bb56efc178c81` |
| NumPy03 row audit | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_independent_numpy_audit_execution_attempt01/audit_results.jsonl` | 611,903 | `19e5c603fc5e9f974831ec726259e9a18c70205ec232db4cc24208431d046748` |
| NumPy03 summary | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_independent_numpy_audit_execution_attempt01/summary.json` | 402,832 | `8f31921648da0f91918e29d92fc12cb57e1d4dc2e6eda8a1f8690c453218418e` |
| CPU02 config | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02/qualification_config.json` | 10,721 | `0abfbb9797a84254e6d0a6e72266862a079e5066f2fec394e0796b0c2664d2f5` |
| NumPy03 config | `runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_independent_numpy_audit_attempt05/audit_config.json` | 5,382 | `7d2696cb49f2a459d0d065a3cfe14a83a2a90811f71879594cfc6ddbbcdf92c5` |
| Frozen V3 head | `scripts/audio_enhancement/shared_recipient_transition_head_v3.py` | 32,233 | `77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87` |
| Independent NumPy consumer03 | `scripts/experiments/cross_tfg_audio_head_20261003/independent_numpy_audit_shared_recipient_transition_head_v3_attempt03.py` | 98,390 | `ed540178ae9564a567f5939dd1fdb8560d422bf1a301d8d3f01c06da972fe40b` |
| Fit40 source manifest | `runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/fit40_transfer_bundle_attempt01/fit40_source_transfer_manifest.json` | 61,469 | `0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020` |
| RMS baseline helper | `scripts/audio_enhancement/baselines.py` | 15,123 | `7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2` |

The config independently pins both input manifests and all declared members. It also checks row order across the two NPZ matrices and JSONL ledgers, CPU02-to-NumPy03 manifest lineage, and fit40 source-manifest identity before calculations.

## Runtime and invocation

The read-only runtime is CPython 3.12.14 with NumPy 2.3.5, no Torch, `PYTHONNOUSERSITE=1`, and one thread for OMP/MKL/OpenBLAS/NumExpr. The matrix source is only `fit_source_magnitude_f32`; source WAVs are not opened. Outputs are a per-transition NPZ, 40-row JSONL, summary, and result manifest. They contain no PCM, reconstructed magnitudes, embeddings, or model outputs.

No run has been made. After source review, the command is:

```powershell
& 'C:\Users\dogle\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B `
  'scripts/experiments/cross_tfg_audio_head_20261003/diagnose_v3_flux_float32_cancellation_attempt02.py' `
  --repo-root 'E:\Documents\tts-exp' `
  --config 'runs/cross_tfg_audio_head_20261003/v3_flux_float32_cancellation_attempt02/config.json' `
  --expected-config-sha256 'd2d49c72890d6cb20255bb04070eff15095702fbe360c791f2e2d58a2b6f7192' `
  --expected-config-bytes 7209 `
  --cpu-output-dir 'runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02_readback_attempt01/files' `
  --audit-output-dir 'runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_independent_numpy_audit_execution_attempt01' `
  --output-dir 'runs/cross_tfg_audio_head_20261003/v3_flux_float32_cancellation_attempt02/output'
```
