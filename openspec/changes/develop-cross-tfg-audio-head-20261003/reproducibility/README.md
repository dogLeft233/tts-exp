# V3 transition-head source and configuration bundle

This source-only bundle preserves the frozen CPU qualification and independent audit sources as raw bytes. `v3-transition-head-source-only-attempt02.zip` contains the pinned workers, consumers, contracts, and configurations, plus the 40-row fit source manifest. The adjacent `member_index_attempt02.json` records raw byte counts and SHA-256 hashes for every ZIP member. Configuration files are copied byte-for-byte; CRLF line endings are retained.

## Recorded execution state

- CPU qualification attempt 02 completed with status `COMPLETE`, 40 fit rows, and its declared engineering checks passing.
- The independent NumPy fit consumer attempt 03 and config attempt 05 remain `PREPARED_NOT_DISPATCHED`. They are included as frozen source and configuration, not as a completed independent fit audit.
- Zero-theta CPU attempt 01 completed with status `COMPLETE_ENGINEERING_CHECKS_PASS`.
- Zero-theta NumPy audit attempt 04 completed with status `COMPLETE_WITH_MEASURED_COMPARISONS`. Its exact `activity_cells_u8` comparison differs at three cells: `(frequency_bin, frame) = (213, 6), (247, 4), (256, 8)`. The saved Torch mask contains 64,503 active cells; the NumPy reconstruction contains 64,502.
- Flux diagnostic attempt 02 completed with status `COMPLETE_DIAGNOSTIC`.
- The local zero-theta artifact readback QC attempt 05 passed. Its small source and receipt are included. The archive also includes the NumPy-only activity-cell diagnostic source, but not its measurements or generated result arrays.

An independent NumPy-only arithmetic diagnosis, kept outside the ZIP because it reads the already-produced arrays, found that the CPU02 archive contains the Torch source-STFT real and imaginary arrays. Float32 `hypot` on those components reproduces all 64,503 zero-worker activity bits. For the three mismatches, the component-derived magnitudes are approximately `1.053208734e-6`, `1.173743840e-6`, and `0.905245543e-6` against the unchanged `1e-6` floor. NumPy's centered STFT on the same saved source gives `0.989036494e-6`, `0.973771535e-6`, and `1.156981170e-6`. This isolates the binary differences to Torch-versus-NumPy STFT arithmetic on the saved waveform. Analytic NumPy source synthesis differs from the saved waveform by at most `5.96e-8` and changes two NumPy classifications, but it does not account for the same-source Torch-versus-NumPy comparison.

## Scope

The ZIP excludes WAV/PCM payloads, NPZ or other result arrays, model weights, generated media, remote logs, credentials, and dispatcher artifacts. The fit manifest records source identities and hashes only; its WAV payloads are not included. The preserved configs keep their original status text even where later output manifests establish that a run completed. The execution state above comes from those actual manifests and QC receipts.

The activity-cell diagnosis source is also preserved in the ZIP; its JSON measurements remain outside the bundle at `.codex/scratch/v3_zero_theta_activity_cell_diagnostic_attempt03/activity_cell_diagnostic_attempt03.json`.

## Actual controls360 attempt17 failed-terminal media archive (2026-10-08)

- The archived run retained the full 360-row denominator: 120 PASS, 240 FAILED; run status COMPLETE_WITH_FAILURES, result manifest INCOMPLETE.
- The excluded-media complement contains 26,328 media files (682,851,546 bytes) in a 26,331-member TAR (653,472,590 bytes, SHA-256 26381df471e4863fe85d558d72fe7a7b8f152095d5617570632f60bad4bac81b).
- The TAR was copied to B:\tts-exp-archives\controls360_attempt17_failedterminal_media_complement_attempt04 on the mounted CloudFS/CloudDrive volume and reread byte-for-byte. Cloud synchronization is not confirmed; compact-snapshot blobs remain separately required for full reconstruction.
- Small exact-byte receipts and QC evidence are in [controls360_attempt17_failedterminal_media_complement_attempt04/](controls360_attempt17_failedterminal_media_complement_attempt04/), with source/destination pins in its proof_copy_manifest.json. The TAR and large compact source index are not duplicated into this repository.
