# Teacher-free CoGenAV feature decoding

The fitted Adapter + HiFi-GAN accept **raw**, unnormalized audio-only CoGenAV
Base features `[T, 768]` at 25 Hz. The checkpoint contains train-only affine
statistics, adapter parameters and generator parameters. WavLM, reference audio,
text and the training dataset are not needed for decoding.

```bash
/home/wjj/.venvs/cogenav-sync-20261024/bin/python \
  scripts/experiments/cogenav_decoder_pilot_20261025/decode_features.py \
  --features features.npy \
  --checkpoint runs/cogenav_decoder_pilot_20261025_attempt01/checkpoints/best.pt \
  --knn-root third_party/knn-vc \
  --num-samples 160000 \
  --output reconstruction.wav \
  --device cuda
```

`--features` also accepts the frozen cache `.pt` dictionary's `cogen` tensor;
other cached fields are ignored. For a teacher-free delivery use `.npy` directly.
`--num-samples` is optional and defaults to `T * 640`; it must be an integer in
`(0, T * 640]`. It trims the output clock rather than stretching the waveform.
The CLI rejects empty, malformed or nonfinite feature arrays, and refuses to
overwrite an existing output WAV.

Output is mono 16 kHz FLOAT WAV without loudness normalization, gain changes or
clipping. Protocol revision03 repeats 16 raw vectors at each edge before the
adapter/generator and removes the corresponding 10,240 waveform samples on each
side; it then trims to the requested exact length. The receipt next to the WAV
records feature/checkpoint/output SHA-256, duration and trained protocol hash.

Inference code needs Python 3.10, PyTorch 2.5.1, NumPy, SoundFile and the pinned
`third_party/knn-vc` source at `c616845c4e309e24d5927f15adbdf277a3d65358`
(`hifigan/models.py`, `hifigan/utils.py`, `hifigan/config_v1_wavlm.json`).
Matplotlib is only needed by optional plotting utilities. The full measured runtime
is recorded in the run's `environment_freeze.txt`; the existing CoGenAV runtime
already supplies these packages. The CLI does not load encoder weights.

The five-hour source was repurposed AISHELL-3 official test audio, with a new
speaker-disjoint train/validation split. Do not describe the original official
test partition as an independent benchmark after this training. This pilot has
waveform reconstruction fine-tuning, without adversarial fine-tuning; poor
quality would not prove that CoGenAV features are irreversibly insufficient.
