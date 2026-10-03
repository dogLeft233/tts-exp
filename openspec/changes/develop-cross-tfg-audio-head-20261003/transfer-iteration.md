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
| Identity | Paired direct-drive reference | 60 development rows completed; independent reconstruction pending |
| Fixed RMS | Historical positive control | Same completed development experiment |
| RMS + local compression | Isolate bounded envelope changes | Same completed development experiment |
| Spectral head fit24 | First differentiable candidate | 24 actual updates; 60 development rows completed |
| Fullcontext head fit96 | Correct short-window training support | 96 actual steps completed; new development generation next |

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
