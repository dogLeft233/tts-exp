# Native audio-encoder probe contract

Status: reusable source and protocol contract only. The native encoder-forward probe is not recorded as executed by this note. The staging/readback packer operation is separate from the 168-unit encoder probe. No development or confirmation audio belongs to this probe.

## Frozen inputs and scope

- The CLI plan is the original fit-only 360-row plan at `runs/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03/frozen_plan/plan.json`, SHA-256 `1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7`. It contains eight historical utterances from S0052–S0055 across Wav2Lip, MuseTalk, and Ditto, with the frozen identity/RMS/local-compression/profile roster. The probe uses it to bind original PCM and method configs; it does not score or render the 360 plan rows.
- The standalone probe sidecar is `runs/cross_tfg_audio_head_20261003/native_encoder_probe_attempt01/native_encoder_probe_plan_attempt05.json`, SHA-256 `f205263a89cc135dad52a7db5b11efca927aab457ee640c694bd050863af811f`. Its full roster is 8 clips × 7 conditions × 3 frontends = 168 feature units. Conditions are natural PCM; fixed global −6 dB and +6 dB; fixed-RMS target `0.0376838172675746` without a peak ceiling; and three predeclared profile arms.
- The runner is `scripts/experiments/cross_tfg_audio_head_20261003/native_encoder_probe_attempt02.py`, SHA-256 `4ded47e2e163bbb21734ec26c79f910ecf8ae2c2b44c188cb2eeaf6c7336d34c`. Its original run-folder source has the same hash. It checks fit split, original PCM agreement across frontends, model roster, profile/config hashes, fixed-RMS control, sample count, and canonical PCM digest before extracting features.

## Native frontend timing and interpretation

The clock authority is each pinned frontend’s own query count, tail/chunk/padding rules, and timestamp `i/25` seconds. Conditions for the same sample and frontend must have identical native query maps. Do not force query counts to match across frontends, shift audio, or infer waveform receptive field from a selected feature-index span.

| Frontend | Input/runtime contract | Clock handling |
|---|---|---|
| Wav2Lip | Frozen GAN frontend and mel implementation. The measured mel path has 16 kHz input, 80 mel bands, 800-sample FFT/window, 200-sample hop, pre-emphasis 0.97, −100 dB floor, and symmetric ±4 feature clipping. | Preserve the official 16-column mel windows and 3–4-column starts at 25 fps, plus the runner’s emitted query indices/timestamps and valid tail. |
| MuseTalk / Whisper | Qualified deployed Transformers 4.39.2 feature extractor; the pinned Muse call omits `do_normalize`. The processor forms a padded Whisper segment; the encoder sees the segment, then Muse selects hidden-state positions for video frames. | Preserve the exact selected hidden-state indices, mask/tail, and `i/25` output query map. The 10-index span is not a waveform receptive-field claim because encoder self-attention sees the full segment. |
| Ditto / HuBERT | Frozen ONNX raw-input affine is `(x−2.586069967946969e−05)/0.20523720979690552`; constants are pinned to the qualified graph audit. This describes the graph input only, not encoder invariance. | Use the pinned native 25 Hz chunking, padding, and five-context-feature mapping from the runner; preserve its exact query index and timestamp arrays. |

Runtime and gain-path evidence is documented in `runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/frontend_gain_and_clock_notes_v4.md`, SHA-256 `819f740ceed94bc7a78cb4c1aa995f116292f99e2d919234cf4c83b40efaf47`. Its linked input-only probe shows frontend tensor responses, including Wav2Lip floor occupancy changes, near-additive Whisper level movement on waveform-supported frames, and the fixed Ditto input affine. These measurements stop before the learned encoders. They cannot establish encoder invariance, mouth motion, or synchronization benefit.

The encoder metrics are deliberately separate: per-query feature norm and direction; query-time-centered feature trajectory; and adjacent-query delta direction and magnitude. “Centered” here means subtraction of the per-dimension mean over the native query-time axis, not frequency centering. These are encoder representation descriptors. They are not lip-motion labels or a replacement for paired native video, SyncNet, or an independent mouth-motion association against original-audio timing.

## Readback tools

- Exact-byte durable runner copy: `scripts/experiments/cross_tfg_audio_head_20261003/native_encoder_probe_attempt02.py`, SHA-256 `4ded47e2e163bbb21734ec26c79f910ecf8ae2c2b44c188cb2eeaf6c7336d34c`.
- Exact-byte archive packer copy: `scripts/experiments/cross_tfg_audio_head_20261003/pack_native_encoder_probe_attempt01.py`, SHA-256 `1a00218b59e6a6e676f0bff0128c3918f10e48149f17e294edfac4a7655c9754`.
- Exact-byte safe reassembly/verifier copy: `scripts/experiments/cross_tfg_audio_head_20261003/verify_native_encoder_probe_bundle_attempt01.py`, SHA-256 `7d71a8638a72f9411cc8c7e81a9b509573e970c4dc6eccd1ec485ed7d120bbe6`.
- The packer binds both plans and runner, checks per-frontend 56-row output rosters and sealed NPZ members, then writes bounded archive chunks. The verifier reassembles from those chunks and validates archive members, source rows, conditions, and scope before safe extraction. Neither tool invokes a model.

## Execution boundary

The latest probe request index is `runs/cross_tfg_audio_head_20261003/native_encoder_probe_attempt01/request_index_attempt04.json`, SHA-256 `657eff15a7f26e4feda809522abcc4e9fb18b4d6edbcc9f8262fbdac1b540299`. It identifies a 168-unit fit-only probe and requires root dispatch after the competing shared540 workload clears. Treat its scope as planned until an actual per-frontend receipt and a verified archive exist. A prior packer upload/staging receipt does not count as encoder execution.
