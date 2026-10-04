# Phase-clock diagnostics and native-feature-to-fit-row join contract

Revision note: v3 corrects the native clock-map field name and records the verified local restoration of the fit360 ledger/manifest. v1 and v2 are retained unchanged.

Status: preparation contract only. No transform, encoder, video, score, or confirmation data were run or opened for this contract. The active shared 540-row run and its inputs are unchanged.

## Frozen source bindings inspected

| Source | SHA-256 |
|---|---|
| `scripts/audio_enhancement/phase_clock_diagnostics_v1.py` | `25ed6f204dea7e5c149f7c897dfccfebe972ad0ef30eda1a2b1dbdc59ae216c5` |
| `scripts/audio_enhancement/v5_local_transition_transform_scaffold_v2.py` | `cf86b95b65b2c4a423082e0f717713983915cdc0674ee7ddb4fba2534e9906e0` |
| `scripts/experiments/cross_tfg_audio_head_20261003/native_encoder_probe_attempt02.py` | `4ded47e2e163bbb21734ec26c79f910ecf8ae2c2b44c188cb2eeaf6c7336d34c` |
| `scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt03.py` | `d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94` |
| `scripts/audio_enhancement/shared_frequency_profile_v1.py` | `fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253` |
| Frozen fit plan `runs/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03/frozen_plan/plan.json` | `1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7` |

The frozen fit selector binds the terminal result ledger SHA `e67c753a2c29048757f629d18ac259b9f146f980cfaf94959d17048874fb0e0d` and result-manifest SHA `85273a6712f7080e04e7a54ab424d7ccfcd2c86df9f0009db318ab555e953512`. The terminal remote files were `/dev/shm/cross_tfg_audio_head_20261003/shared_frequency_fit_recovery_attempt04/run/results.jsonl` and `result_manifest.json`. The verified local restoration is `C:\Users\dogle\.codex\tts-exp-artifacts\recoverable-output-restore-20261004-attempt01\dev\shm\cross_tfg_audio_head_20261003\shared_frequency_fit_recovery_attempt04\run\results.jsonl` and sibling `result_manifest.json`; their observed hashes match the two frozen values above. Restore metadata is `.codex/scratch/recoverable_output_archive_attempt02/full_restore_receipt_attempt01.json` (`RESTORED_ALL_MEMBERS_SHA256_VERIFIED`; archive SHA `cd567ec15fade9d8f1d1ce6c0d1ae043e0c1d8042aa7178caf546c69b0f69039`). Before any later join, recheck the restored files against these hashes and verify the manifest reports 360 results and the same plan hash; do not substitute a partial ledger.

## Phase, sample clock, and masked-array serialization

`phase_clock_reanalysis_diagnostics_v1` requires equal-length mono float32 source and output PCM. It recomputes source and output STFTs with the same pinned function and accepts the constructed output STFT from before ISTFT. Source/output STFT frame centers and windows must match exactly. It compares three distinct stages on the intersection of nonzero source, constructed, and reanalysis coefficients:

1. `construction_vs_source`: `angle(constructed * conj(source))` before ISTFT.
2. `reanalysis_vs_constructed`: `angle(STFT(output PCM) * conj(constructed))` after ISTFT/reanalysis.
3. `reanalysis_vs_source`: `angle(STFT(output PCM) * conj(source))`.

These comparisons do not establish waveform-phase identity. The constructed coefficients may carry the source STFT phase while ISTFT followed by analysis changes phase. Record the source PCM sample clock, exact sample count, STFT centers/window, and candidate/query map separately. No audio shift, resampling, frame remap, or lag correction is permitted by this diagnostic.

The helper deliberately emits NaN at undefined measurements. The existing V2 writer (`write_diagnostic_arrays_npz_v2`, lines 684–706) rejects any float/complex NaN or infinity. A future writer must therefore use the following versioned encoding rather than passing the helper arrays directly to that writer:

- Finite arrays retain their original key, dtype, shape, and values.
- A float/complex array with NaNs is serialized as `<key>__finite_values` plus `<key>__valid` (`uint8`, same shape). Valid entries retain their original values and dtype; invalid entries use a finite zero placeholder **only for storage**. The JSON encoding manifest records the original key, original dtype and shape, value/mask key names, invalid count, the exact source validity field, and that invalid values reconstruct as NaN. On decode, restore NaNs wherever the mask is zero. Reject infinities and any unrecognized nonfinite field; do not silently coerce them.
- `physical_frame_power_ratio_db_40ms` uses the existing `physical_frame_power_ratio_valid_both_nonzero_40ms`. Source/output centroids use `source_frame_nonzero_energy` / `output_frame_nonzero_energy`; centroid difference is valid only on their intersection. `local_xcorr_curve` uses `local_xcorr_valid`. The row-level xcorr argmax is valid iff that row has at least one valid lag; its `INT64_MIN` argmax sentinel and NaN correlation are missing-data markers, not measured values. `*_phase_delta_rad` zeros outside support are invalid/placeholders as identified by the corresponding `*_phase_supported` array.
- Summary JSON represents undefined scalar statistics as `null` (Python `None`), with valid and planned denominators. Never emit JSON NaN/Infinity.

The 40 ms squared-energy centroid and normalized local waveform xcorr/lag are acoustic descriptions on the unchanged sample clock. They are not phoneme, syllable, articulation, or mouth-motion times. A 40 ms window being numerically equal to a 25-fps frame period does not make it an encoder receptive field.

## Native 168 feature rows joined to the completed fit-360 video/score rows

The native encoder probe has 8 frozen historical fit clips × 7 audio conditions × 3 frontends = 168 planned rows. It measures encoder features only: `generator_model_forward=false`, `video_created_or_read=false`, `syncnet_forward_or_scores_read=false`; its `feature_array` is not SyncNet's audio embedding. The fit plan is 8 clips × 15 methods × 3 frontends × the frozen seed `20261003` = 360 rows.

Build a join only from PASS rows in the terminal sealed fit ledger, keyed first by exact clip/source/audio/model identity and then by verified canonical float32 PCM hash. `condition` and `method_id` are hints to find candidate rows; they do not establish equivalence.

| Join layer | Native probe fields | Fit plan/result fields and required equality |
|---|---|---|
| Source utterance | `sample_id`, `speaker_id`, `source_pcm.pcm_float32_sha256`, `source_pcm.wav_sha256`, `source_pcm.sample_count`, `source_pcm.sample_rate_hz`, `source_pcm.encoding` | Plan row `sample_id`, `speaker_id`, `natural_pcm.pcm_sha256`, `natural_pcm.file_sha256`, `natural_pcm.sample_count`, `natural_pcm.sample_rate_hz`, `natural_pcm.channels`, `natural_pcm.subtype`; result `natural_pcm_sha256` and `score.score_audio_pcm_sha256`. Re-decode and rehash actual files before matching. |
| Generator-condition waveform | `condition_waveform.pcm_float32_sha256`, `sample_count`, `sample_rate_hz`; `condition`; `feature_array.sha256`, `shape`, `dtype`, `path` | Plan row `method_id`, `method_config_sha256`, `seed`, `model_id`; PASS result `generator_pcm.pcm_sha256`, `generator_pcm.sample_count`, top-level `generator_pcm_sha256`, and `render.generator_pcm_sha256`. Require exact canonical PCM SHA and sample-count/rate equality. Check actual condition audio file and digest. |
| Frontend/model identity | Probe summary `frontend`, `model_id`, `runtime`, `runtime_source_details`; row `native_frontend_detail` | Match result identity fields `model_id`, `audio_frontend`, `seed` to the plan row, then bind the generator to `plan.models[model_id]`: `model_sha256`, `model_revision` where present, `generator.checkpoint_sha256`, `source_sha256`, `preprocessing_sha256`, and `generator_audio_preprocessing_sha256`; also check model-specific runtime/qualification/adapter pins. Do not use result `checkpoint_sha256` alone: it is the row method checkpoint field and can be null for a model whose generator weights are pinned in the model registry. Require exact pinned frontend/checkpoint/preprocessing identity, not just `wav2lip`/`muse`/`ditto` labels. |
| Native feature clock | `clock_map.fps_num/fps_den`, **`source_sample_rate_hz`**, `sample_count`, `query_count`, `queries[].query_index`, rational `query_timestamp_num/den`, `source_clock_frame_index`, `source_sample_start`, `source_sample_stop_exclusive`; NPZ `feature_array.sha256` | Fit result `score.raw_matrix_npz.path/file_sha256` and array hashes for `query_frame_index`, `query_timestamp_num`, `query_timestamp_den`, `valid_query_mask`, `distance_by_query_lag`, `lag_offsets_frames`; `render.selected_track_crop.frame_map_path/frame_map_sha256`; `score.primary_syncnet_input` and `score.primary_pcm16_window`. Join query observations on exact rational source-clock timestamps and valid masks; do not assume equal query counts or same array indices. |
| Generated video | None in the native probe | Fit PASS result `row_id`, `split`, `sample_id`, `speaker_id`, `model_id`, `audio_frontend`, `seed`, `method_id`, `method_config_sha256`; `render.generated_video_path/generated_video_sha256`, `render.silent_video_path/silent_video_sha256`, `render.timestamps_path/timestamps_sha256`, `render.selected_track_crop.{path,file_sha256,frame_map_path,frame_map_sha256,tracks_pickle_path,tracks_pickle_sha256}`. Keep these bound to the same PASS result row and score NPZ. |

For direct overlap in this plan, the candidate condition-to-method lookup is: `natural`→`identity`, `fixed_rms_0p0376838173`→`rms_fixed`, and each of the three profile conditions→the identically named profile method. `global_minus_6db` and `global_plus_6db` have no matching fit-360 methods and therefore cannot be joined to fit-360 videos/scores. Even for overlapping names, accept a pair only after matching the source PCM digest, condition PCM digest, model/frontend pins, and source-clock support.

The fit runner scores video against the original natural audio, not the transformed generator-condition waveform: `score.score_audio_pcm_sha256 == natural_pcm_sha256`, while `render.generator_pcm_sha256` is the transformed condition used to generate the video. Keep these roles separate. To relate a native encoder feature response to a video score, bind the feature row to the generator condition by PCM hash and to the score row by the exact same fit `row_id`, crop/frame-map seals, and SyncNet NPZ. Do not substitute the native encoder feature vectors for SyncNet audio/video embeddings or interpret input-feature changes alone as mouth-motion or synchronization effects.

## Before any future join

Require terminal run status and matching result-manifest/result-ledger raw hashes; 360 planned rows, all PASS rows and failures retained in the denominator; unique result `row_id`; complete exact audio-file and feature/NPZ digest validation; exact model/frontend/checkpoint/source pins; and exact rational clock intersection. A hash mismatch, absent artifact, unsupported condition, or clock mismatch is a non-join retained as a missing planned pairing, not a method-name fallback. This contract does not authorize a run and freezes no settings.
