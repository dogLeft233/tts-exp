## Why

The current natural-audio reconstruction path (`WavLM-Large layer 6 -> prematched HiFi-GAN`) is not a high-fidelity codec: prior LRS3 and AISHELL-1 identity controls found that a reconstructed-audio-driven Wav2Lip video prefers the reconstructed audio over the untouched natural audio. This codec-domain co-adaptation can independently cause strict replacement failure, so it must be isolated before another alignment or TTS-transfer model is trained.

A native-16-kHz acoustic neural codec provides a narrow first test. If it preserves the natural waveform and Wav2Lip mel domain substantially better than the semantic WavLM reconstruction path, its generated video should be more substitutable with the untouched natural PCM. If this identity test fails, adding MFA warping, Soft-DTW, residual prediction, or a larger TTS-conditioned model would not resolve the codec interface uncertainty.

## What Changes

- Add a no-training, natural-only LRS3 codec identity experiment using the official Descript Audio Codec (DAC) 16-kHz, 8-kbps checkpoint with every available quantizer.
- Reuse the exact ordered 50-record, 38-source-group cohort from the completed LRS3 WavLM-HiFi-GAN direct-resynthesis experiment.
- Compare three driver-audio arms in one fresh frozen-Wav2Lip run: untouched natural PCM (`N`), historical WavLM-L6/HiFi-GAN direct reconstruction (`W`), and DAC-16-kHz reconstruction (`D`).
- Render one video per driver arm and score the complete 3-by-3 video/audio matrix with strict video-stream-copy and decoded-PCM verification.
- Measure waveform and Wav2Lip-input-mel fidelity without time-shifting, loudness matching, or otherwise correcting codec output after decoding.
- Pre-register separate decisions for (a) improvement over the current WavLM/HiFi-GAN replacement penalty and (b) practical identity compatibility with the untouched-natural baseline.
- Keep every TTS, MFA, DTW, residual, training, and held-out evaluation path sealed; a positive result only authorizes a separately specified TTS/alignment experiment.

## Capabilities

### New Capabilities

- `lrs3-dac16k-codec-identity`: Defines a provenance-locked natural-only neural-codec fidelity and strict replacement-substitutability experiment against the current WavLM/HiFi-GAN path.

### Modified Capabilities

None.

## Impact

- Adds a future experiment package under `scripts/experiments/lrs3_dac16k_codec_identity/` and focused tests under `tests/experiments/lrs3_dac16k_codec_identity/`.
- Reads immutable cohort and WavLM-reconstruction artifacts from `runs/lrs3_wavlm_hifigan_direct_20260826/` and writes to a new run root `runs/lrs3_dac16k_codec_identity_20260904/`.
- Adds a pinned experimental dependency on Descript Audio Codec source tag `0.0.5` (commit `408235a9dcd2983684c87615a1bc2a8954f6eb47`) and its official native-16-kHz, 8-kbps checkpoint. The checkpoint SHA-256 must be locked before any cohort media is processed.
- Requires no model training. GPU work is limited to frozen codec inference and frozen Wav2Lip rendering; SyncNet scoring and statistics remain frozen evaluation operations.
