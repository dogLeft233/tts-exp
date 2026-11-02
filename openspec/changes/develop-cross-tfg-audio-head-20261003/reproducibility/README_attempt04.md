# V3 transition head: source status and numerical evidence — attempt 04

This note records the sealed source-only qualification bundle and the later CLI source work. It does not replace or change the raw-byte archive, index, or README from attempt 03.

## Recorded execution state

- CPU qualification attempt 02 completed with status `COMPLETE` for the 40 historical fit rows; its declared engineering checks passed.
- The independent NumPy fit audit using consumer attempt 03 and config attempt 05 completed with status `COMPLETE_WITH_NUMERIC_MISMATCHES`: **16 of 40 rows met every check. The remaining 24 rows differed only on the predeclared exact flux-identity diagnostic.** This is a numerical diagnostic, not a candidate-effect or selection result.
- Zero-theta CPU attempt 01 completed with status `COMPLETE_ENGINEERING_CHECKS_PASS`.
- Zero-theta NumPy audit attempt 04 completed with status `COMPLETE_WITH_MEASURED_COMPARISONS`. Its `activity_cells_u8` comparison differs at exactly `(frequency_bin, frame) = (213, 6), (247, 4), (256, 8)`: the saved Torch mask has 64,503 active cells and the NumPy reconstruction has 64,502. No threshold or source artifact was changed to force agreement.
- Flux diagnostic attempt 02 completed with status `COMPLETE_DIAGNOSTIC`.
- Local zero-theta artifact readback QC attempt 05 passed. An independent arithmetic diagnosis reproduced all 64,503 Torch activity bits from the saved Torch STFT components; the three Torch-versus-NumPy differences are near the unchanged `1e-6` activity floor.

## CLI source status

The source entrypoint is [enhance_v3_attempt01.py](../../../../scripts/audio_enhancement/enhance_v3_attempt01.py). The explicitly zero-valued engineering example is [v3_zero_engineering_attempt01.json](../../../../scripts/audio_enhancement/configs/v3_zero_engineering_attempt01.json), and the current contract is [v3-cli-contract-attempt03.md](../v3-cli-contract-attempt03.md). These are source artifacts only: the CLI has not been run or runtime-qualified. No trained V3 checkpoint, training record, or selection record is available. `ENGINEERING_ONLY` is not an identity transform; at zero theta the CLI requires output to equal the configured canonical RMS baseline exactly.

The CLI decodes the original RIFF/WAVE through the pinned decoder, performs the frozen mono float32 16 kHz canonicalization and RMS baseline, and emits a new float32 WAV plus a provenance sidecar. It rejects final peak-ceiling violations without clipping or post-gain. The enhanced WAV is for generator input only. Delivery and scoring continue to use the exact original audio file, with its sample rate, channel count, sample count, encoding, and file hash recorded in metadata.

## Archive and boundary

`v3-transition-head-source-only-attempt03.zip`, `member_index_attempt03.json`, and the attempt 03 README remain immutable historical artifacts. The later CLI/config/contract sources are outside that archive. This note records the corrected audit units and the newer implementation status; it does not imply a model, development/confirmation, scoring, training, or API run.
