# V3 through V4 CPU engineering qualification — attempt 01

**State:** prepared, not dispatched. This packet qualifies one already exposed historical fit input only. It does not train or select a method, and it does not read development or confirmation audio.

## Frozen inputs and source pins

- V4 fit plan: raw SHA-256 `b66b0addc25ed6c2b7f7831f8643ba4f35bc227a35ce9e4dd5c4faa0b540801b`.
- Selected fit row: `S0052_BAC009S0052W0121__full_natural_context_v2`; canonical source has 62,639 samples at 16 kHz. WAV raw SHA-256 `2d45f2b65dec29f77bcd56f05f883b16207a99e7848df3a06a203819a915b261`; canonical float32 PCM SHA-256 `f4e611e9f816665ad7c4dce4980f461c4d9af4fe6fe9678c8f756bd03fa6b36c`.
- V4 native geometry manifest: SHA-256 `1fd5d97d977bb541ef9ebfdd467d066c9f2e678ae5e0fc5f5d96c8b555aecd43`.
- Fixed full-audio embedding cache: NPZ SHA-256 `e59ebe53cbe6b0486c7dbca2a811ba593857979a82e691265ca9c1ff1a739dd5`; metadata SHA-256 `dbd214132a3394924f4ff935e5604a8c1603847e51f7531638a78a83a2dae6a4`.
- V4 RMS cache: NPZ SHA-256 `cbd1de399e37e5433fcad9dadb8e29ddf7a55fbde1b89aa0610fdad3fc9bf16e`; metadata SHA-256 `7987936df048720c310c349b15cdb05745718a973eca7fede0c9c95361efb452`.
- V3 head: `scripts/audio_enhancement/shared_recipient_transition_head_v3.py`, SHA-256 `77acdff03cca22261852f48364cbc75263db58a14cb565c08aa0d268afca6e87`.
- V4 runtime: `scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v4_attempt01.py`, SHA-256 `dd751132bd3d8dec5abce288c2f765df78d9b7ef9150cdfb0ccff52ba72ee458`.
- V2 full-context scorer: SHA-256 `d086c61cdb993bffa5f9ff7a55bf4b66b1b9127028f8b80eb3b5f1be74464d7b`.
- V4/V3 bridge adapter: `scripts/audio_enhancement/v3_v4_native_runtime_adapter_attempt03.py`, 6,892 bytes, SHA-256 `4f4f6f496bd15e073268694673b8df5678404e672aafa625ff9a6abc135248aa`.
- Qualification worker: `scripts/experiments/cross_tfg_audio_head_20261003/qualify_v3_v4_native_cpu_attempt04.py`, 47,698 bytes, SHA-256 `ce89f2c94abca9164322e209b8c6ae3f5ed9877d1ac81d0602dbfe58fef69472`.
- Configuration: `runs/cross_tfg_audio_head_20261003/v3_v4_native_cpu_qualification_attempt02/qualification_config.json`, 8,907 bytes, SHA-256 `45d44cc26f7d9d066cb269bc235a94a24a596854e7451f71e4491a531bef3966`.

The configuration source closure also pins the exact V4 `heads.py`, `training.py`, audio I/O, RMS baseline, native crop, Wav2Lip loader/frontend, and package modules. At runtime the worker rehashes all declared files and requires each named source identity to agree with its closure entry.

## Qualification sequence

1. Verify the literal CPython 3.10.12 Wav2Lip interpreter, NumPy 1.26.4, Torch 2.5.1+cu124, pinned python_speech_features overlay, and one thread per CPU backend. `CUDA_VISIBLE_DEVICES` is empty; the worker reads Torch's lazy-init flag without calling CUDA availability routines and fails if CUDA initializes.
2. Load the exact historical fit row, original WAV, fixed context arrays, full-audio original-speech embedding cache, per-row native crop geometry, and cached V4 RMS output. No cache builder runs and no original cache path is written.
3. Evaluate the V3 head at zero parameters and require byte-identical output to V4's canonical NumPy RMS waveform. Re-render and score through V4's 512-pixel native paste/tracked-crop path; compare all 31 lag distances and all 27 visual embeddings with the frozen V4 RMS cache at absolute tolerances `1e-5`.
4. Evaluate one fixed, nonzero 3×3 diagnostic parameter matrix from the already pinned V3 CPU qualification config. Record the actual V4 loss, all nine parameter gradients, candidate waveform gradient, raw/projected masks, flux gates, clock centers, and frame-power diagnostics. This is a gradient/path check, not an optimization step or efficacy result.
5. Separately check coordinate and directional finite differences for the smooth waveform-only objective. They do not finite-difference the native SyncNet score or validate the renderer's hard straight-through estimator.
6. Compute the zero-parameter waveform Jacobian Gram matrix and singular values after the V3 RMS anchor; report absolute and relative cutoffs, rank and condition number. The Jacobian's nine directions are diagnostic and do not establish useful audio/video effects.

## Frozen V4 loss and units

The adapter evaluates the V4 reward relative to the same-row RMS baseline:

`ΔCbest + 0.25·Δfar − 2·relu(ΔD0)^2 − relu(−ΔCanchor)^2`

with the existing regularizers `0.1·visual-log-norm-ratio² + 0.02·relative-waveform-MSE + 0.002·mask-TV`. D0 and Canchor guard slack remain exactly zero. The worker hard-checks these seven coefficients against literals as well as the frozen plan.

V3 exports its projected mask in dB. V4's `_mask_tv` receives natural-log magnitude gain, as produced by `heads.py` (`ln(gain) = dB·ln(10)/20`). Adapter attempt03 converts before applying the unchanged V4 TV reduction and records both natural-log TV and its dB-equivalent for unit clarity. The head's peak ceiling is explicitly mapped from the V3 CPU config's `null` to the V4 RMS baseline's `.99`, so the zero waveform and rendered output use the V4 policy.

## Runtime and output boundaries

The declared cap is 1,800 seconds, 8 GiB RSS high-water, a 16 MiB diagnostics NPZ, frame batch 1, and SyncNet batch 1. Exceeding a parity, gradient, FD, or resource check produces a retained control-failure record; it does not relax the tolerances or discard the row. The worker writes an NPZ of scores, embeddings, masks, gates, gradients, Jacobian Gram/eigenspectrum, and frame-power arrays plus summary and receipt JSON. It never writes the source or candidate PCM waveform, and it performs no optimizer step, generator training, API call, development read, or confirmation read.

Even a complete CPU pass would establish only that this selected row can be evaluated through the CPU implementation under these checks. It would not establish CPU suitability for broader work, or any candidate benefit across speakers/frontends. The output is an engineering gate for parent review before any separate training decision.
