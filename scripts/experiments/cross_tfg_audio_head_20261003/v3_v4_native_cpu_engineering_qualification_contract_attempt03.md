# V3 through V4 CPU engineering qualification — attempt 03

**State:** prepared, not dispatched. This is a single-row engineering check on an already exposed historical fit utterance. It performs no optimizer step and reads no development or confirmation data. CPU support remains unqualified until an actual run completes.

## Inputs and frozen pins

- V4 fit plan raw SHA-256: `b66b0addc25ed6c2b7f7831f8643ba4f35bc227a35ce9e4dd5c4faa0b540801b` (v4 overlay); the exact V2 parent plan is `e491d0bb210fe78647f4845543a88c629d37cecc09a0823c287560fbb1e71a06`, payload `7200d93801ca0749b472937645d1e955e5b39c730f6a2d1a5e24d3f380f6dce9`, and the merged effective payload is `d8c383419df73181bd0be342e3554055e4e04107cb8294b4e19dd4bd5bc0514c`.
- Selected row: `S0052_BAC009S0052W0121__full_natural_context_v2`; source sample `S0052_BAC009S0052W0121`; speaker `S0052`; 62,639 samples at 16 kHz.
- Source WAV raw SHA-256: `2d45f2b65dec29f77bcd56f05f883b16207a99e7848df3a06a203819a915b261`; canonical float32 PCM SHA-256: `f4e611e9f816665ad7c4dce4980f461c4d9af4fe6fe9678c8f756bd03fa6b36c`.
- Context NPZ and metadata SHA-256: `6616e649412cdf5c5f78ce51fa7d8403364c56528f93777a2e35eb82fc941ec1` / `46bab1ff647cbaabc3eea247dc315431d6ad976ca142a9c5e331ce4e81094450`.
- Full-audio embedding cache NPZ and metadata SHA-256: `e59ebe53cbe6b0486c7dbca2a811ba593857979a82e691265ca9c1ff1a739dd5` / `dbd214132a3394924f4ff935e5604a8c1603847e51f7531638a78a83a2dae6a4`.
- V4 RMS cache NPZ and metadata SHA-256: `cbd1de399e37e5433fcad9dadb8e29ddf7a55fbde1b89aa0610fdad3fc9bf16e` / `7987936df048720c310c349b15cdb05745718a973eca7fede0c9c95361efb452`.
- Native geometry manifest SHA-256: `1fd5d97d977bb541ef9ebfdd467d066c9f2e678ae5e0fc5f5d96c8b555aecd43`.
- V3 head SHA-256: `77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87`; V4 runtime SHA-256: `dd751132bd3d8dec5abce288c2f765df78d9b7ef9150cdfb0ccff52ba72ee458`; V2 full-context scorer SHA-256: `d086c61cdb993bffa5f9ff7a55bf4b66b1b9127028f8b80eb3b5f1be74464d7b`.
- V4/V3 adapter attempt04 SHA-256: `4f4f6f496bd15e073268694673b8df5678404e672aafa625ff9a6abc135248aa`.
- Worker: `scripts/experiments/cross_tfg_audio_head_20261003/qualify_v3_v4_native_cpu_attempt06.py`, raw SHA-256 `195299402d15e07275309398d2d60b7a292248d57b44695114354b727a3c1c1f`.
- Config: `runs/cross_tfg_audio_head_20261003/v3_v4_native_cpu_qualification_attempt04/qualification_config.json`, raw SHA-256 `80b7a74dfbd962e5e7dc84c5a16e8ef5ce02ed00c5551a8b31e02aa6f710a507`.
- New code is staged only under `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/v3_v4_native_cpu_qualification_attempt04/code/payload/src`; the prior sealed fit source tree is not modified. Original plan, WAV, context, cache, and geometry remain pinned read-only paths outside this staging namespace.

The config hashes every member of the Python source closure, including V4 `heads.py`, `training.py`, audio I/O, RMS baseline, native crop, Wav2Lip loader/frontend, `fit_cli_v4_attempt01.py` (the authoritative parent-overlay loader), and package modules. Runtime identity pins are cross-checked against the same closure.

## Sequence and gates

1. First perform a standalone, read-only readiness check. Verify exact raw hashes, sizes, sample count and regular-file status for every frozen input and source, plus interpreter/runtime paths and available RAM/disk. Do not import the model stack or run a forward during readiness. If any retired tmpfs input is absent or differs, stop and preserve the failure; do not substitute a path or regenerate cached inputs.
2. The readiness request loads the exact b66b overlay and e491 V2 parent, applies the pinned V4 loader’s recursive merge and `__overlay_delete_keys__` behavior, validates both raw/payload pins and the effective payload `d8c383419df73181bd0be342e3554055e4e04107cb8294b4e19dd4bd5bc0514c`, and checks the merged eight-row fit roster. It recomputes the PSF source-tree digest using the V2 helper rule relative to the actual `python_speech_features` package root and compares it with the frozen plan. It records/checks the actual `librosa.__version__` and distribution version against the frozen literal, and reports MemAvailable plus cgroup current/max/free memory. The engineering run uses literal CPython 3.10.12, NumPy 1.26.4, Torch 2.5.1+cu124 and the pinned `python_speech_features` overlay. It requires empty `CUDA_VISIBLE_DEVICES`, `PYTHONNOUSERSITE=1`, and OMP/MKL/OpenBLAS/NumExpr/Torch thread counts of one. The worker checks Torch's lazy CUDA-init flag without invoking CUDA availability routines and fails if CUDA initializes.
3. Read the exact effective historical fit row, original WAV, fixed context, original-speech embedding cache, per-row native geometry, and V4 RMS cache. The worker uses the pinned V4 overlay loader to merge the b66b parent overlay with its verified e491 V2 parent; it does not treat the overlay wrapper as a flat plan. Do not run any cache builder or write to an original cache path.
4. Run the V3 head at zero parameters and require byte-identical waveform equality with V4's canonical NumPy RMS waveform. Score it through the unchanged V4 full-resolution paste/tracked-crop path. Compare all 31 lag distances and 27 visual embeddings against the sealed V4 RMS cache at absolute tolerances `1e-5`.
5. At the predeclared nonzero 3×3 diagnostic parameter matrix from the pinned V3 CPU config, compute the unchanged frozen V4 objective and full-loss gradients with respect to the nine parameters and output waveform.
6. Separately differentiate the **pure V4 SyncNet-derived reward** (excluding waveform, visual-norm and mask-TV regularizers) with respect to both the nine parameters and the candidate output waveform. Both gradients must exist, be finite, and have strictly nonzero L2 norm. This is an engineering connectivity gate; the number of nonzero entries is descriptive only. A full-loss gradient does not substitute for this gate.
7. Check coordinate and directional finite differences only for the separate smooth waveform-only probe. These checks do not finite-difference the native SyncNet score or validate the hard straight-through renderer path.
8. At zero parameters, compute the waveform Jacobian using nine forward JVPs, one per canonical basis direction. Stack the nine length-62,639 directional derivatives transiently, form the 9×9 Gram matrix and its eigenspectrum/SVD, then discard the sample-by-parameter matrix. This avoids one reverse-mode pass per waveform output. Report absolute and relative cutoffs (`1e-12` and `1e-6`), rank and retained-subspace condition number. A COMPLETE_PASS also requires all nine singular directions to exceed the already frozen combined absolute/relative rank threshold; a missing direction is a control failure requiring a separate identifiability decision, not a reason to relax the cutoff. This local waveform Jacobian is not a SyncNet efficacy metric.

Any parity, pure-reward-gradient, FD, or resource failure is retained as a control failure. Do not relax thresholds based on observed results.

## Frozen V4 objective and units

The adapter's reward remains the frozen comparison to the same-row RMS baseline:

`ΔCbest + 0.25·Δfar − 2·relu(ΔD0)^2 − relu(−ΔCanchor)^2`

The full loss keeps existing regularizers `0.1·visual-log-norm-ratio² + 0.02·relative-waveform-MSE + 0.002·mask-TV`, with exactly zero slack on D0 and Canchor guards. All seven weights are hard-checked against the frozen V4 plan.

V3 emits its projected magnitude mask in dB. V4's `_mask_tv` consumes natural-log gain, as defined by the frozen head (`ln(gain) = dB·ln(10)/20`); the pinned adapter converts units before applying the unchanged reduction. The final peak ceiling is mapped from the V3 CPU config's `null` to the V4 RMS baseline's `.99`. The candidate uses the unchanged original audio/query clock.

## Runtime and retained outputs

After the lower-level model loader returns source records, the worker explicitly checks the imported `librosa` module version and recomputes the `python_speech_features` package tree SHA with the pinned V2 helper, comparing both to the effective frozen-plan execution values before any Wav2Lip or SyncNet forward. The declared limits remain 1,800 seconds, 8 GiB RSS high-water, 16 MiB diagnostic NPZ, frame batch 1, and SyncNet batch 1. The NPZ stores scores, embeddings, full-loss gradients, pure-reward gradients, masks/gates, frame-power arrays, and only the 9×9 Jacobian Gram/eigen/singular values; it does not store input/output waveforms or the N×9 Jacobian. Summary and receipt bind all inputs, source hashes, runtime, gates and output hashes. No optimizer, generator training, API call, dev read, confirmation read, or selection occurs.

Even a complete pass would qualify only this selected fit row on this CPU implementation. It would not establish CPU suitability at broader scale or candidate benefit across speakers/frontends.