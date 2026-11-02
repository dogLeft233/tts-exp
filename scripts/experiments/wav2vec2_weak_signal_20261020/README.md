# May wav2vec2 weak-response layer audit

Frozen inference only. Replays the actual report56 May train/val/test audio,
whole-split z-score, 320000-sample chunk hop and 320080-sample maximum chunk
length. Uses the identical facebook/wav2vec2-base weights and FP32 eager path.
There are 14 chunks and 44 forwards: baseline/half/epsilon-compensated half
on every chunk, plus baseline repeats on the first train/test chunks.

The three meanings of weak response are kept separate:

1. Low waveform RMS at a frame's receptive-field center, using complete-split
   baseline q20/q80 thresholds and a 50ms RMS window.
2. Low first-convolution temporal variance per channel in a baseline chunk.
   This is the relevant GroupNorm variance, measured before learned affine.
3. Low temporal output variance per channel in each baseline layer/chunk.

None is a verified speech/noise label. Transformer output has whole-chunk
context, so its frame energy label is descriptive, not a local receptive field.

The first GroupNorm is also run with epsilon=2.5e-6 while input is multiplied
by0.5, as a diagnosis. This is not a proposed deployment change. All original
weights and other normalization settings remain unchanged; no optimizer exists.

On this Windows host, dependencies and weights are reused read-only:

```powershell
$python='C:/Users/dogle/.codex/tts-exp-artifacts/gain-common-direction-20261018/env/Scripts/python.exe'
$result='F:/Datasets/tts-exp/wav2vec2_weak_signal_20261020_v1'
# Set $result to a fresh path for a new replay; never overwrite the sealed v1.
& $python -X utf8 scripts/experiments/wav2vec2_weak_signal_20261020/run.py --output $result
& $python -X utf8 scripts/experiments/wav2vec2_weak_signal_20261020/analyze.py --output $result
& $python -X utf8 scripts/experiments/wav2vec2_weak_signal_20261020/plot.py --output $result
& $python -X utf8 -m unittest discover -s tests/experiments -p test_wav2vec2_weak_signal_20261020.py -v
```

`run.py` refuses an existing output. Protocol and source are frozen before
forward execution. The fixed historical paths are constants near the top;
porting to another machine requires preserving the file SHA identities.

Every layer's full time and channel moments are stored, alongside 32 paired
raw sample vectors per layer/chunk and complete last hidden states. The CPU
consumer independently checks moments, strata, raw samples, GroupNorm formulas,
and exact report56 endpoint cache correspondence. It does not import the GPU
producer. The historical pairing cache drops one final encoder frame per
split; `cache_aligned_endpoint` uses exactly the retained cache support, while
the layer plots use complete encoder outputs including that tail.

Outputs: `analysis.json`, three summary CSVs, `independent_qa.json`, the layer
plot, full moments/sample evidence and the final member-SHA manifest.
Formal report: `basic-memory/docs/experiments/59-wav2vec2-weak-signal-layer-audit.md`.
