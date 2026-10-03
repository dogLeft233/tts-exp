# Shared-head transfer and iteration

## Why these controls are separate

The application endpoint is `C(V_m(h(x)), x)`: the generator receives the
enhanced waveform and the scorer receives the original waveform. A useful
transform must therefore change the generated mouth trajectory while retaining
its correspondence to the original source clock and speech content.

The current Wav2Lip adapter uses an unnormalized input waveform to form its
log-mel condition. The recovered Ditto and MuseTalk source paths instead use
HuBERT and Whisper features. Internal feature normalization can attenuate a
global gain change; its presence and effective gain response must be checked
from actual model execution. No gain invariance is assumed from a frontend name.
Frequency ratios and short-time envelopes can change after global level
normalization, so they remain plausible shared controls even when gain has a
small effect. This is a mechanism hypothesis, not a transfer result.

## Current candidates

| Candidate | Role | Actual state |
|---|---|---|
| Identity | Paired direct-drive reference | 60 development rows and original-PCM independent audit completed |
| Fixed RMS | Historical positive control | FULL development ΔCbest +0.234 versus identity |
| RMS + local compression | Isolate bounded envelope changes | FULL ΔCbest +0.267 versus identity, +0.033 versus RMS |
| Spectral head fit24 | First differentiable candidate | FULL ΔCbest +0.256 versus identity, +0.022 versus RMS |
| Fullcontext head fit96 | Correct short-window training support | FULL ΔCbest +0.217 versus identity, −0.017 versus RMS; five-method independent audit completed |
| Fixed-zero-lag head v3 | Reward D0 and unit-direction alignment | FULL ΔCbest +0.129 versus identity, −0.105 versus RMS; D0 −0.264 versus RMS |

These are exploratory Wav2Lip development means over ten new male speakers,
two clips per speaker and three paired seeds. All nine five-method FULL
endpoints are complete. GUARD20 retains an incomplete endpoint status because
one clip has 34 common queries in each seed, below the frozen minimum 41; its
speaker has only one of two clips estimable on that support. No sample or
endpoint is removed to change that status. The original four-method audit remains
in `runs/cross_tfg_audio_head_20261003/stageb_consumer04_actual_evidence_attempt01/`;
the full five-method ledger and independent audit are in
`runs/cross_tfg_audio_head_20261003/stageb_joint_attempt02_readback/`.

Independent FULL score decomposition places almost all of fit24's gain over
RMS in B (+0.021), with D0 effectively unchanged. Local compression additionally
reduces D0 by 0.018 versus RMS. These evaluator diagnostics motivate fixed-lag
and unit-normalized-feature objectives for the next candidate; visible-mouth
and content measurements are evaluated separately. They do not establish transfer
to another frontend or an untouched confirmation effect.

Fit96 versus RMS changes B by +0.003, Dmin by +0.019, and D0 by +0.057;
its Canchor falls by 0.054. More fullcontext training on the old median-minus-
softmin objective did not improve development performance. The next candidate
uses an RMS-rendered historical-fit reference and directly rewards fixed-zero-
lag distance and unit-norm cosine, with embedding-norm and normalized waveform
penalties. No development or confirmation rows enter that reference cache.

The fixed-zero-lag v3 candidate completed all 96 historical-fit steps and all
60 development renders. The eight zero-initialization reference comparisons,
96 reward/loss decompositions and checkpoint seals pass. The six-method
360-row common-support analysis is complete: all 14 FULL endpoints are complete,
and all 14 GUARD20 endpoints retain the planned short-clip incompleteness.
The local authoritative archive and independent readback are in
`runs/cross_tfg_audio_head_20261003/joint_v3_analysis_attempt03/`.

Compared with RMS, v3 changes D0 by −0.264 and Canchor by
+0.153; both directions hold for all ten speakers. Its best lag is zero
in 38/60 contexts, versus 9/60 for RMS. However, B falls by 0.111,
Dmin falls by only 0.005, and Cbest falls by 0.105.
Local compression remains the current highest-Cbest candidate. These are
exploratory development means; no confirmation interval is inferred from them.

The fit proxy's RMS best lag is +2/+3 on the eight historical clips, whereas
native development best lags are mostly 0/+1. A center-27-query comparison and
RMS control do not account for that difference. Same-fit input diagnostics
compare mel chunks, six-channel face tensors, predictions, pasted pixels and
original-audio SyncNet embeddings through the official codec path before the
next learned objective is fitted. The original v3 run remains immutable.

MuseTalk's corrected common crop/paste geometry, 144-frame 25 fps portrait and
original-PCM mux have passed engineering readback. Four shared waveform
methods are being prepared for its fixed development cohort. Ditto's full
StreamSDK import now passes with isolated Cython/setuptools and Python headers;
native portrait qualification follows. These engineering results permit actual
method evaluation; they do not provide cross-model efficacy evidence.

All 300 independent visual/speech-motion guards are complete. Local compression
raises the equal-speaker zero-lag aperture correlation with original-audio
40 ms log RMS by 0.014 versus fixed RMS. Tracking is complete for all five
methods; larger aperture or velocity is not treated as a quality improvement.
Qwen ASR completed 100 unique audio requests: every normalized hypothesis
matches the official AISHELL transcript, giving CER 0 for each method's 20
clips. These guards cover the five-method development run, not v3 or heldout
confirmation.

The fit96 head uses the same waveform parameterization and historical eight
training clips as fit24. It changes both audio context and training duration,
so a fit96 versus fit24 difference is a candidate comparison, not an isolated
causal estimate of either change. If that distinction matters after development
results, fit a matching-step/fullcontext control before attributing the cause.

## Decisions driven by development results

1. Independently reconstruct the completed four-method raw distance matrices,
   including common source-clock windows, failures and planned denominators.
2. Apply fit96 in a new candidate-only run. A new joint analysis derives
   five-method support summaries from original immutable matrices; unchanged
   generator controls are reused only with exact source/model/config/seed and
   artifact bindings. Preserve the original four-method analysis separately.
3. Evaluate the same waveform candidates in actual Ditto and MuseTalk
   generation. The shared mode uses one configuration/weight set. Calibrated
   backend presets are separate methods with explicit calibration costs.
4. When a head raises C by changing background distance or embedding norm,
   inspect B, Dmin, D0, offset, unit-norm cosine diagnostics, and independent
   mouth motion. Do not silently substitute a convenient score definition.
5. When a learned head fails transfer, evaluate a small fit-only family of
   bounded frequency profiles and local compression parameters with the same
   RMS/peak policy. Select a shared controller using model-equal and per-model
   fit objectives, then verify it on the fixed development cohort. Candidate
   families and comparisons are frozen before their new renders.
6. If the small historical fit set is unstable, expand within the already
   designated historical fitting speakers. Keep development selection and
   untouched confirmation separate. A failed candidate remains in the method
   table and informs the next training or control change.

## Delivery boundary

A runnable checkpoint and exact original-audio mux are necessary engineering
evidence. Useful-method evidence additionally requires transfer across the
declared frontends, a comparison with RMS, content/visual checks, and the frozen
untouched-speaker/model evaluation. LatentSync remains excluded from candidate
selection until the shared method is frozen.
