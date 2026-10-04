# Native audio-encoder probe contract V2

Status: source and protocol contract only. The 168-unit native encoder-forward probe remains unrun. Packer staging is separate from encoder execution. This work reads only the eight frozen historical fit utterances; it does not render or score video and does not open development or confirmation audio.

## Frozen inputs

- Fit plan: `runs/cross_tfg_audio_head_20261003/shared_frequency_profiles_fit_attempt03/frozen_plan/plan.json`, SHA-256 `1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7`. It binds eight historical clips from speakers S0052–S0055 across Wav2Lip, MuseTalk, and Ditto.
- Native probe sidecar: `runs/cross_tfg_audio_head_20261003/native_encoder_probe_attempt01/native_encoder_probe_plan_attempt05.json`, SHA-256 `f205263a89cc135dad52a7db5b11efca927aab457ee640c694bd050863af811f`.
- Its roster is 8 clips × 7 conditions × 3 frontends = 168 feature units: natural PCM, fixed −6 dB, fixed +6 dB, fixed RMS target `0.0376838172675746` without a peak ceiling, and the three fit-shortlisted local profiles. Profile selection is fit-only descriptive provenance, not dev-tuned parameters.
- Exact-byte runner: `native_encoder_probe_attempt02.py`, SHA-256 `4ded47e2e163bbb21734ec26c79f910ecf8ae2c2b44c188cb2eeaf6c7336d34c`.
- Exact-byte packer: `pack_native_encoder_probe_attempt01.py`, SHA-256 `1a00218b59e6a6e676f0bff0128c3918f10e48149f17e294edfac4a7655c9754`.
- Exact-byte verifier: `verify_native_encoder_probe_bundle_attempt01.py`, SHA-256 `7d71a8638a72f9411cc8c7e81a9b509573e970c4dc6eccd1ec485ed7d120bbe6`.
- Request index: `runs/cross_tfg_audio_head_20261003/native_encoder_probe_attempt01/request_index_attempt04.json`, SHA-256 `657eff15a7f26e4feda809522abcc4e9fb18b4d6edbcc9f8262fbdac1b540299`.

The runner verifies exact original PCM agreement across frontends, profile/config and source pins, fixed-RMS settings, sample count, and canonical float32 digest. It retains every frontend’s native query map; conditions for one sample/frontend must have the same map. No waveform is shifted or retimed to force cross-frontend query counts to match.

## Native input and query clocks

| Frontend | Exact input path | Query and timing contract |
|---|---|---|
| Wav2Lip | Frozen 16 kHz audio path, 80-band mel, 800-sample FFT/window, 200-sample hop, pre-emphasis 0.97, −100 dB floor, and symmetric ±4 feature clipping. | Keep the official 16-column mel windows and emitted starts, which advance by 3 or 4 mel columns at 25 fps. The runner records sample starts, query indices/timestamps, and tail support. |
| MuseTalk / Whisper | Qualified deployed Transformers **4.39.2** runtime. The pinned processor call omits `do_normalize`. Whisper log-mel is computed for a padded segment; the encoder sees that segment before Muse selects per-query hidden states. | At query `q`, the runner records the selected 10-token span beginning at `floor(2q)` on the 50 Hz encoder-token axis, along with native masks and `q/25` timestamp. This is a feature-index selection span, not a waveform receptive-field claim: Whisper self-attention sees the full segment. |
| Ditto / HuBERT | Pinned ONNX graph input affine is `(x − 2.586069967946969e−05) / 0.20523720979690552`; the constants were recovered from the graph’s Constant nodes and independently re-encoded. This describes the graph input only, not encoder invariance. | The runner returns `ceil(N/640)` query vectors of width 1024. With `chunksize=(3,5,2)`, each five-output chunk starts every 3200 samples (200 ms), has a 6480-sample window, uses 2000 samples of left speech padding and 6480 samples of right padding, and emits five query vectors. `source_clock_map` records each query’s chunk and unclipped source interval. It is not a five-feature-per-query map. |

The frontend implementation/runtime evidence is pinned in `runs/cross_tfg_audio_head_20261003/v5_fit_expansion_preparation_attempt02/frontend_gain_and_clock_notes_v4.md`, SHA-256 `819f740ceed94bc7a78cb4c1aa995f116292f99e2d919234cf4c83b40efafa47`.

## Interpretation boundary

The outputs separate per-query feature norm and direction, query-time-centered trajectory, and adjacent-query delta direction and magnitude. “Centered” means subtracting each feature dimension’s mean over that frontend’s native query-time axis; it does not mean frequency centering. These are learned-encoder representations only. They do not measure mouth motion, phoneme accuracy, or SyncNet synchronization, and cannot replace paired native video endpoints or an independently timed motion analysis against original audio.

## Source pins

- `scripts/audio_enhancement/shared_frequency_profile_v1.py`: `fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253`.
- `scripts/audio_enhancement/baselines.py`: `7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2`.
- `scripts/audio_enhancement/audio_io.py`: `7b951990343cf5d68f3f6046e77cd5c40ee736777d196de3723418bca5965004`.
- Runtime note above: `819f740ceed94bc7a78cb4c1aa995f116292f99e2d919234cf4c83b40efafa47`.

No native encoder result or effect is claimed by this contract.
