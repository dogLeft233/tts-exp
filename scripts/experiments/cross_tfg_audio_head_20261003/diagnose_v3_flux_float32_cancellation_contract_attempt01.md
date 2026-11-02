# V3 float32 flux-identity rounding diagnostic — attempt 01

## Purpose and scope

This source-only diagnostic explains the preserved 40-row `fit_flux_balance_identity` boolean mismatches from the CPU02 engineering qualification and independent NumPy03 readback. It uses the sealed float32 STFT-magnitude arrays and already-saved flux arrays. It does not reload source audio, import the V3 head/producer or Torch, run a model/generator/scorer, alter the earlier tolerance, or make an efficacy claim. The result is arithmetic provenance for the existing `COMPLETE_WITH_NUMERIC_MISMATCHES` artifact.

For each supported adjacent frame, it records the float32 per-bin difference, positive/negative reductions, direct difference reduction, separately rounded current-minus-previous frame totals, each division stage, and the resulting left/right relative-flux values. It also compares the independently saved NumPy03 and CPU02 flux values, retains the CPU and NumPy validity masks, and writes each valid transition's source-frame total ratio and cancellation ratio. It does not replace or drop any of the 40 fit rows.

The error bound uses the binary32 unit roundoff `u=2^-24`, smallest subnormal `η=2^-149`, the standard `γ_(n−1)` sum bound for up to `n−1` rounded additions, and a per-rounded-operation absolute allowance `u·|z|+η/2`. These bounds are propagated separately through the two real operation paths. Assumptions are finite inputs, no overflow, round-to-nearest, gradual underflow, and at most 256 additions in each 257-bin reduction. The bound is descriptive; it does not reclassify the previously recorded audit status.

## Frozen source and input pins

| Role | Path | Bytes | SHA-256 |
|---|---|---:|---|
| Diagnostic source01 | `scripts/experiments/cross_tfg_audio_head_20261003/diagnose_v3_flux_float32_cancellation_attempt01.py` | 33,809 | `89d4b39bb65348429360f221c2b19fceb82d56b021f2771109818a8406a6234d` |
| Diagnostic config01 | `runs/cross_tfg_audio_head_20261003/v3_flux_float32_cancellation_attempt01/config.json` | 6,551 | `619c544f4d37e79483818b1ee6ee8a45094b285eaab739b254e09533788ff126` |
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
| Frozen V3 head source | `scripts/audio_enhancement/shared_recipient_transition_head_v3.py` | 32,233 | `77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87` |
| Frozen independent NumPy consumer03 | `scripts/experiments/cross_tfg_audio_head_20261003/independent_numpy_audit_shared_recipient_transition_head_v3_attempt03.py` | 98,390 | `ed540178ae9564a567f5939dd1fdb8560d422bf1a301d8d3f01c06da972fe40b` |
| Fit40 source manifest | `runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/fit40_transfer_bundle_attempt01/fit40_source_transfer_manifest.json` | 61,469 | `0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020` |
| RMS baseline helper | `scripts/audio_enhancement/baselines.py` | 15,123 | `7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2` |

The config independently pins each input output-manifest raw hash and every manifest member's byte count/hash. It also verifies row order across both NPZ matrices and both JSONL ledgers, the CPU02→NumPy03 manifest lineage, and the original fit40 source-manifest identity before calculating anything.

## Runtime and invocation

The declared read-only runtime is the local bundled CPython 3.12.14 with NumPy 2.3.5, no Torch, `PYTHONNOUSERSITE=1`, and one thread for OMP/MKL/OpenBLAS/NumExpr. The diagnostic's only matrix source is `fit_source_magnitude_f32` from the pinned NumPy03 archive; the original WAV files are not opened. All 40 row IDs, transition support masks, per-transition measurements, per-row residual maxima/frame indices, total-magnitude ratios, and derived error bounds are retained in output.

No run was performed. After source review, an invocation is:

```powershell
& 'C:\Users\dogle\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' -B `
  'scripts/experiments/cross_tfg_audio_head_20261003/diagnose_v3_flux_float32_cancellation_attempt01.py' `
  --repo-root 'E:\Documents\tts-exp' `
  --config 'runs/cross_tfg_audio_head_20261003/v3_flux_float32_cancellation_attempt01/config.json' `
  --expected-config-sha256 '619c544f4d37e79483818b1ee6ee8a45094b285eaab739b254e09533788ff126' `
  --expected-config-bytes 6551 `
  --cpu-output-dir 'runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_cpu_qualification_attempt02_readback_attempt01/files' `
  --audit-output-dir 'runs/cross_tfg_audio_head_20261003/shared_recipient_transition_head_v3_independent_numpy_audit_execution_attempt01' `
  --output-dir 'runs/cross_tfg_audio_head_20261003/v3_flux_float32_cancellation_attempt01/output'
```

The bounded output contains a per-transition compressed NPZ, a 40-row JSONL, a summary, and a file manifest. It contains no PCM samples, reconstructed magnitudes, frontend embeddings, or model outputs.
