---
title: waveform_reachability_source_design_attempt01
type: note
permalink: tts-exp/experiments/assets/receiver-local-response-coordinate/codec-attempt01/design/waveform-reachability-source-design-attempt01
---

# MuseTalk waveform reachability: source-only design

**Status: SOURCE_ONLY; not execution approval.** No encoder, optimizer, candidate waveform, renderer, scorer, or test was run. Next experiment remains undecided pending codec results. Pins are in `waveform_reachability_source_pins_attempt01.json`.

## Pinned path and gradient boundary

`AudioProcessor.get_audio_feature` loads mono audio at 16 kHz, splits it into 30-second segments, then calls Transformers `WhisperFeatureExtractor`. The pinned extractor uses 80 Slaney mel bands, `n_fft=400`, `hop_length=160`, zero-pads each segment to 480,000 samples / 3,000 mel frames, computes centered Hann-STFT power, projects to mel, drops the last STFT frame, clamps at `1e-10`, takes `log10`, floors at `segment_max-8`, and returns `(log+4)/4`. Its Torch branch converts input with `torch.from_numpy` and returns `log_spec.numpy()`, which detaches autograd. Audio decode/resample through `librosa.load` is also outside the graph.

A source-faithful differentiable mirror would start from the already-decoded fixed PCM tensor, implement that feature path in Torch without NumPy conversion, and invoke the same frozen/eval Whisper encoder with autograd enabled (no `no_grad`/inference mode). Frozen weights still allow gradients to the input. Encoder conv/GELU, position addition, four self-attention layers, final layer norm, hidden-state collection, and `get_whisper_chunk` stack/slice/rearrange are differentiable. Fixed segment/frame indices and pad/crop masks are selections, not learnable variables. The `max-8` floor is piecewise differentiable and segment-global. Self-attention spans all 1,500 encoder positions, so local waveform edits can change hidden states at distant times. STFT synthesis overlap also broadens sample support around edited frames.

## One preferred parameterization

Let `x` be the accepted fixed-RMS natural driver, preserving its content, sample count, and clock. The target RMS is that driver's already established fixed RMS. Final video mux and scoring use untouched original natural PCM; its RMS is separate and must not replace the driver's target RMS.

Use a bounded smooth log-power mask on the driver's waveform STFT before Whisper:

`X_theta(f,t) = X(f,t) * exp(0.5 * G_theta(f,t))`

`G_theta` uses a handful of fixed broad log-frequency bands, gated by each existing MFA phone core with short smooth edge ramps. Preserve phase per STFT bin during construction, then iSTFT by overlap-add to exact length and normalize once: `y = y0 * RMS(x) / RMS(y0)`. OLA/reanalysis does not guarantee output phase preservation and can create local-power collateral; reject clipping instead of clipping. No shifts, resampling, time stretch, or independent prompt-row rotations. Perturb the source phone core and evaluate the actual frontend's frozen output-owner row `I+4`; this is only the frozen ownership mapping, not a measured delay or permission to shift PCM.

Fit the accepted alias-consistent native `+` delta on unique underlying Whisper primitives, counting each primitive once across overlapping prompt copies. Regularize mask size and waveform/spectral deviation. This is a reachability probe, not a trained reusable head, and should not optimize SyncNet directly.

## Conditional budget and stop gates

If separately authorized: first CPU zero-parameter parity against accepted frontend/native arrays; any mismatch beyond a tolerance frozen in advance stops the probe. Then at most 20 first-order steps per clip, batch one, frozen Whisper, one signed target, full 30-second-padded / 1,500-position / four-layer context. One encoder forward is roughly 18B MAC (~35 GFLOP at 2 FLOPs/MAC), plus backward/frontend; memory depends on attention implementation. Do not crop context.

Before any run, freeze mask amplitude, objective weights, numerical parity tolerance, and waveform-deviation cap; all four are currently **unfrozen**, and this memo is not executable settings. Stop without relaxing bounds if parity fails, target movement is unreachable within the cap, length/RMS/clock/finite/no-clipping constraints fail, or collateral dominates. A candidate still needs original-clock visual endpoints and full SyncNet evaluation. Existing native `+` lowered full Sync-C in both clips; native-margin attainment alone is not success. If full C worsens, report the conflict and stop—no dose escalation or training from this result.

This proposes a source-faithful reachability test, not proof of reachability or a physical-latency correction.