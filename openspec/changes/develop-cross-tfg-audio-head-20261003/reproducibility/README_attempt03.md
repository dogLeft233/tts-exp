# V3 transition-head source and configuration bundle — attempt 03

This source-only archive preserves the exact pinned CPU qualification and independent audit sources as raw bytes. `v3-transition-head-source-only-attempt03.zip` contains the pinned workers, consumers, contracts, configurations, the 40-row fit source manifest, and this README as its root `README.md`. The adjacent `member_index_attempt03.json` records raw byte counts and SHA-256 hashes for every ZIP member. JSON configuration bytes are preserved exactly, including their original CRLF line endings.

## Recorded execution state

- CPU qualification attempt 02 completed with status `COMPLETE` for the 40 historical fit rows; declared engineering checks passed.
- The independent NumPy fit audit using consumer attempt 03 and config attempt 05 completed with status `COMPLETE_WITH_NUMERIC_MISMATCHES`: 16 checks passed and 24 predeclared flux-identity checks did not meet their exact identity criterion. The frozen config still says `PREPARED_NOT_DISPATCHED`; the actual result manifest is authoritative for execution status. This remains a numerical diagnostic, not a candidate-effect or selection result.
- Zero-theta CPU attempt 01 completed with status `COMPLETE_ENGINEERING_CHECKS_PASS`.
- Zero-theta NumPy audit attempt 04 completed with status `COMPLETE_WITH_MEASURED_COMPARISONS`. Its exact `activity_cells_u8` comparison differs at `(frequency_bin, frame)` cells `(213, 6)`, `(247, 4)`, and `(256, 8)`: the saved Torch mask has 64,503 active cells and the NumPy reconstruction has 64,502. No threshold or source artifact was changed to make the arrays agree.
- Flux diagnostic attempt 02 completed with status `COMPLETE_DIAGNOSTIC`.
- Local zero-theta artifact readback QC attempt 05 passed.

An independent NumPy-only arithmetic diagnosis found the CPU02 qualification arrays retain the Torch source-STFT real and imaginary components. Float32 `hypot` of those components reproduces all 64,503 zero-worker activity bits. At the three mismatches, component-derived magnitudes are about `1.053208734e-6`, `1.173743840e-6`, and `0.905245543e-6` against the unchanged `1e-6` floor. NumPy centered-STFT magnitudes on the same saved waveform are about `0.989036494e-6`, `0.973771535e-6`, and `1.156981170e-6`. This localizes the binary differences to Torch-versus-NumPy STFT arithmetic near the floor. Analytic NumPy source synthesis differs from the saved waveform by at most `5.96e-8`; it is not the explanation for the same-waveform Torch/NumPy mismatch.

## Scope and boundary

The archive includes source, configuration, contract, the fit40 source-identity manifest, the compact zero-theta readback receipt, and the source for the arithmetic diagnosis. It excludes WAV/PCM payloads, result arrays, model weights, generated media, remote logs, credentials, and dispatcher artifacts. The fit manifest records source identities and hashes only. The V3 CLI contract is proposed documentation only; the CLI is not implemented, and no trained V3 parameter artifact is available. Zero theta is marked `ENGINEERING_ONLY` and is not an identity transform: this implementation returns its canonical RMS baseline at zero parameters.

The separate `v3-cli-contract.md` defines the proposed command/configuration behavior, exact-length finite float32 output, original-audio mux provenance, and latency measurement plan. It reports no new inference, training, API call, or model-effect run.
