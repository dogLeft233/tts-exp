# Clock-preserving audio enhancement

This package is the model-agnostic waveform boundary for the cross-TFG audio
head. It accepts natural audio at any ordinary PCM/float RIFF-WAV rate, mixes
to mono, resamples deterministically to 16 kHz, and emits finite mono float32
audio with an unchanged canonical sample count. Float WAV output is read back
and compared bit-for-bit before the write call returns.

## Components

- `audio_io.py`: WAV parser, deterministic windowed-sinc resampler, canonical
  float32 reader and atomic float WAV writer.
- `baselines.py`: the historical fixed RMS positive control
  (`0.0376838172675746`), bounded flat gain, a bounded 32 ms smoothed-RMS
  compressor, and a low-dimensional magnitude-profile fallback. The compressor
  preserves the sample clock and profile search is role-gated to
  `split_role="fit"`; the combined spectral profile has a separate total-mask
  dB bound.
- `heads.py`: optional PyTorch spectral residual head. It predicts a bounded
  real magnitude mask, so each input STFT phase is retained. Its final layer
  starts at zero and the initial waveform mapping is exact identity. Global
  RMS, local envelope compression, spectral residual, post-head input/target
  RMS matching and the final uniform peak limiter are independently
  configurable. The mask network uses utterance-level normalized
  log-magnitude input and a roughly 120 ms local receptive field. Optional
  frequency-centering and per-STFT-frame energy normalization are explicitly
  separate ablations; frame normalization fails closed if it would exceed the
  declared final mask bound.
- `training.py`: callable-adapter boundary for differentiable frozen Wav2Lip
  rendering plus frozen SyncNet scoring against untouched natural audio. It
  checks that the score gradient reaches enhanced audio and the head receives
  trainable gradients. A small total-variation penalty keeps the predicted
  log-magnitude mask smooth in time and frequency.
- `pipeline.py`: stable `transform(audio_16k_mono_f32, config) -> waveform`
  callable for identity, RMS, bounded profile and learned-head methods.
- `overdub.py`: optional MKV/MOV stream-copy mux that drops any generator audio,
  carries the original PCM stream, and compares decoded audio and elementary
  video-stream hashes before publishing the output.

The waveform package does not choose speaker splits, fit profile parameters,
or select a candidate using a confirm set. `fit_cli.py` provides the actual
fit-plan runner; `wav2lip_fit_runtime.py` provides its frozen differentiable
generator/scorer adapter. The fit plan binds input inventories, model/frontend
hashes, video context, and loss definition. Applying a saved waveform head does
not require a TFG model's hidden features.

Peak policy is explicit. The in-memory identity transform and float32 WAV
writer preserve exact samples by default. `rms_stage_peak_ceiling` can limit
the gain of the pre-head RMS stage; `peak_ceiling` uniformly attenuates the
final stack output and applies to every stack method. The file writer can
also request uniform attenuation and records its applied gain. No path
hard-clips samples, and silence is left at exact zero. The pinned Wav2Lip
inference frontend does not peak-normalize audio before computing mels.

## Minimal API

```python
import torch
from scripts.audio_enhancement import read_canonical_wav, write_float32_wav
from scripts.audio_enhancement.baselines import apply_global_rms
from scripts.audio_enhancement.heads import EnhancementStack

x, source = read_canonical_wav("natural.wav")
positive_control, rms_info = apply_global_rms(x)
stack = EnhancementStack()
y, mask, diagnostics = stack(torch.from_numpy(x), use_global_rms=False,
                             use_spectral_head=True, post_rms_match="input",
                             return_diagnostics=True)
write_float32_wav("enhanced.wav", y.detach().cpu().numpy())
```

The model/container runner should retain `natural.wav` as the authoritative
SyncNet scoring audio. The enhanced WAV is the generator-control input; any
container mux is a separate, explicitly verified step.

`python -m scripts.audio_enhancement.cli input.wav enhanced.wav --config run.json`
also writes a canonical 16 kHz natural reference WAV next to the output (or
to `--natural-reference-wav` when specified). Existing outputs are protected;
`--overwrite` is required to replace them.

The fit runner's `shared_audio_head.pt` is a self-describing checkpoint: it
stores the head architecture, complete waveform transform settings (including
RMS target, post-head matching, peak bounds and 16 kHz clock), fit-plan/source
seals, and the state dictionary. Apply that checkpoint without a second,
hand-maintained config file:

```bash
python -m scripts.audio_enhancement.cli natural.wav enhanced.wav \
  --checkpoint runs/<fit-run>/shared_audio_head.pt --device cuda
```

The CLI reconstructs the transform settings from the checkpoint and records
them with checkpoint/config/source hashes, input/output RMS and sample-clock
checks in its JSON result. If an explicit `--config` is also supplied, its
method, transform settings and head architecture must agree with the
checkpoint. Bare state-dict files still require an explicit config. Keep the
canonical natural reference as the scoring audio; the enhanced WAV only drives
the generator. Optional muxing uses the original input WAV by default, which
preserves its original sample rate and channel count. Use `--mux-original-wav`
only when a different pristine PCM WAV should be muxed. The generated video
audio is discarded, and the mux report records the original-WAV and canonical
16 kHz reference as separate inputs.

## Current evidence

The first historical-fit candidate completed 24 optimizer steps with nonzero
gradients and parameter updates. Its four-method development run completed all
240 planned generation/mux/scoring rows. The independent five-method audit now
includes 300 rows over 10 new male speakers, two clips each and three paired
seeds. FULL original-audio ΔSync-C versus identity is +0.234 for fixed RMS,
+0.267 for local compression, +0.256 for fit24, and +0.217 for fit96. GUARD20
remains incomplete because one clip has 34 queries below its frozen minimum 41;
its planned denominator is retained. These are exploratory development means.
A second candidate completed 96 steps using full historical audio context,
but was 0.017 below fixed RMS on this development cohort. Both checkpoints
remain experimental candidates. Cross-model
usefulness will be selected from actual generated videos scored against the
original natural audio, followed by untouched-speaker/model confirmation.

The fixed-zero-lag v3 candidate has completed 96 historical-fit steps, with
zero-initialization parity and nonzero gradients and updates independently
verified. Its reward combines relative zero-lag distance reduction and cosine
gain against a bounded RMS reference, with visual norm, waveform and mask
smoothness penalties. Its new development effect is pending. All 100 audio
items from the five-method run match official AISHELL transcripts under Qwen
ASR (CER 0); v3 content and cross-model checks remain pending.

Actual identity CLI I/O validation used a 48 kHz stereo PCM16 source. It emitted
an exact-identity 16 kHz mono reference and preserved the original 48 kHz stereo
PCM samples in an MKV mux, including the unchanged video stream. This checks
the input/output boundary and original-soundtrack path. The fit24 checkpoint
also passed actual CPU CLI application on that input: its finite 8,000-sample
enhanced waveform was saved, and the original stereo PCM and video stream/frame
timestamps were preserved exactly. These engineering checks do not establish
the learned candidate's synchronization benefit.
