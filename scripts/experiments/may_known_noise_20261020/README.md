# Known-noise May encoder pilot

Frozen wav2vec2 inference; no model training, TFG generation or SyncNet scoring.
The May source is not assumed clean. This experiment controls **added** noise
exactly, using synthesized white/pink noise and three paired seeds per type.
The noise is not an estimate of May's existing background noise.

Use the exact original whole-split float32 z-score, then mix without another
normalization, limiting, integer encoding or resampling. Whole-split RMS-based
added-noise SNR is30dB; local quiet sections can have much lower SNR.

| Arm | Encoder input | Added-noise SNR relative to May content |
|---|---|---|
| A | s+n | 30dB |
| B | 0.5(s+n) | 30dB |
| C | s+0.5n | 36.0206dB |
| D | 0.5s+n | 23.9794dB |

Anchors are E(s) and E(0.5s). Noise-induced feature change is always measured
against the anchor with the **matching May-content scale**, never against a
different-scale anchor. The primary endpoint is RMS Euclidean distance between
per-frame unit-normalized noisy/anchor vectors, not raw feature amplitude.
Relative L2 change is secondary. These are encoder perturbation metrics,
not noise separation accuracy, speech quality, or downstream performance.

The same known noise waveform is used in all four arms. Different splits use
seed offsets to avoid repeating a noise signal. White and pink share a seeded
Gaussian draw; pink applies a frequency-domain1/sqrt(f) amplitude filter with
zero DC, before mean/RMS normalization. No noise dataset or denoiser is used.

Four arms + one epsilon-compensated B control per type/seed/chunk,14 historical
chunks,28 clean-anchor forwards and one repeatA:449 frozen forwards. The
compensated B uses first GroupNorm epsilon2.5e-6 and is compared to A using
default epsilon1e-5. It is a mechanism diagnostic, not a deployment proposal.

```powershell
$python='C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/env/Scripts/python.exe'
$result='F:/Datasets/tts-exp/may_known_noise_20261020_v1'
# Use a fresh $result for any new replay. Never overwrite a sealed run.
& $python -X utf8 scripts/experiments/may_known_noise_20261020/run.py --output $result
& $python -X utf8 scripts/experiments/may_known_noise_20261020/analyze.py --output $result
& $python -X utf8 scripts/experiments/may_known_noise_20261020/plot.py --output $result
& $python -X utf8 -m unittest discover -s tests/experiments -p test_may_known_noise_20261020.py -v
```

The original data, weights and dependency locations are constants near the
producer's start. SHA-qualified paths must be preserved if porting machines.
`protocol.json` is frozen before any forward. `completion.json` confirms actual
completion. `independent_qa.json` reconstructs inputs/noise, all449 input hashes,
full moments and raw sample vectors. Model weights remain frozen and their
state hash unchanged. There is no optimizer.

Three seed ranges are descriptive for this May identity and synthetic profiles;
they are not confidence intervals or independent speaker repetitions. The
complete encoder output is retained, as in report59, including the single tail
frame per split omitted by the historical two-frame pairing cache.
