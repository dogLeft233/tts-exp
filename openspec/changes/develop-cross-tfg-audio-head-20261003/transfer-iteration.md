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

The same-fit RMS diagnostic confirms almost identical official and proxy
mel/face inputs and model predictions (maximum differences about 1e-6).
Both center-27-query curves have best lag +2. Re-encoding both branches through
the same DIVX and official scoring path reduces mean absolute matrix difference
from 0.419 to 0.185; codec processing contributes substantially, but the remaining
resize/rounding and generated-frame crop differences require a full-512-portrait
comparison on that same fit clip. A fit/dev cohort difference is not itself
evidence of a proxy timing bug. See the actual diagnostic receipts in
`runs/cross_tfg_audio_head_20261003/proxy_native_diagnostics_readback_attempt01/`.

MuseTalk's corrected common crop/paste geometry, 144-frame 25 fps portrait and
original-PCM mux have passed engineering readback. Its four-method, 240-row
development run has started with the fixed cohort. An initial launcher incorrectly
required a Git checkout for the qualified codeload source; the new control entry
verifies the existing official-source qualification and file hashes, retaining
the failed attempt and the identical frozen plan.

Ditto's native portrait now passes the full renderer wrapper: 145 frames at
25 fps, Torch CUDA rendering and four measured CPU-only ONNX sessions. The
first completed forward was rejected by a postrun Cython-extension path guard;
the fresh run pins the actual extension and preserves the failed record.
Its native AAC mux is retained for provenance. A separate local PCM_F32LE mux
has now been decoded and matched exactly to the original 92,225 samples, with
the H.264 elementary stream unchanged and every presentation timestamp equal
to frame_index/25. The sampled portrait frames have been inspected. These
engineering results permit shared method
evaluation; cross-model efficacy awaits the complete paired development rows.

All 300 independent visual/speech-motion guards are complete. Local compression
raises the equal-speaker zero-lag aperture correlation with original-audio
40 ms log RMS by 0.014 versus fixed RMS. Tracking is complete for all five
methods; larger aperture or velocity is not treated as a quality improvement.
Qwen ASR completed 100 unique audio requests: every normalized hypothesis
matches the official AISHELL transcript, giving CER 0 for each method's 20
clips. Twenty additional v3 candidate ASR calls are complete: CER and
disagreement against the reused natural-audio hypotheses are both zero for all
20 clips and 10 development speakers. No reference transcript was sent to the
provider. All 40 candidate/reference PCM16 WAVs in the new ASR archive have
also been independently bound to the exact ASR inputs and source lengths.
The 60 v3 visual trajectories have now been paired against the same Wav2Lip
identity contexts at exact common rational timestamps. All 1,440 geometry
statistics independently recompute exactly, with complete 20-clip, 10-speaker,
three-seed support. Equal-speaker mean aperture changes by +0.002777 eye-distance
units (eight of ten speakers positive), area by +0.002383 squared units, and
width by -0.001995 units. These are descriptive shape changes. Signed mean
velocity is near zero and is not a measure of temporal motion strength.

The full 300-row official SyncNet embedding replay has been restored locally
and all 309 archive members verified. An observed norm/direction decomposition
uses saved official diagonal matrices and float32 counterfactual cross cells.
For fit24 versus RMS, equal-speaker FULL best-lag gain is 0.021843, comprising
0.006375 from the norm term and 0.015469 from the direction term. For fit96 v2,
the total is -0.016684, with norm -0.003693 and direction -0.012991. Root's
independent 120-context curve/Shapley reconstruction and seed-to-clip-to-speaker
aggregation match all 3,510 checked Shapley fields exactly. Canonical float32
confidence subtraction matters: replacing it with float64 subtraction changes
some fields by up to 4.768e-7. Saved embeddings reconstructed in NumPy also have
small reported distance residuals; original official scores are preserved.
This is descriptive geometry, with no causal mediation claim or development
confidence interval. GUARD20 retains the planned short clip and incomplete
speaker rather than promoting the complete subset to the full cohort.

The fit96 head uses the same waveform parameterization and historical eight
training clips as fit24. It changes both audio context and training duration,
so a fit96 versus fit24 difference is a candidate comparison, not an isolated
causal estimate of either change. If that distinction matters after development
results, fit a matching-step/fullcontext control before attributing the cause.

## Decisions driven by development results

### Shared candidates frozen after complete fit qualification

The completed Wav2Lip/MuseTalk/Ditto v3 development contrasts against RMS
are −0.105/−0.013/−0.170 in FULL Cbest. A zero-lag improvement did not make
v3 a useful shared best-confidence head. Ditto's complete240-row raw-array
audit is qualified; GUARD20 retains its predeclared short-clip incompleteness.

The twelve recipient-local spectral profiles and three controls completed360
fit rows on eight historical clips, three paired seeds, and three frontends.
Independent reconstruction validates2520 arrays, exact common clocks/support,
float32 curves and selector utility. The frozen fit-only shortlist is:

1. `local_r090_accent015_tilt_p025`
2. `local_r090_flatten015_tilt_m025`
3. `local_r090_accent_0.30_tilt0`

The first profile's fit Cbest gains against RMS are +0.010/+0.051/+0.096;
its Wav2Lip zero-lag distance slightly worsens. These eight-clip fitting
descriptions do not establish development or confirmation efficacy. All twelve
profile results remain available. The utility, tie rule, equal-model weighting,
worst-model penalty, controls and three-profile limit remain those in ADR009;
the shortlist is fixed before new development renders.

The candidate-only540-row dev plan uses the same20 clips, ten speakers,
three paired seeds and three native models. Root metadata QC validates every
row against the base natural/score/face/source-clock bindings and the exact
fit method configurations/model records. Controls will be reused only after
exact completed-result qualification; they are not declared complete by the
metadata plan. Plan SHA is
`43e5f45a987153219bc423afbb86512f441ff12db20c3cb56dfb776e14e567df`.
Its artifacts and independent receipt are under
`runs/cross_tfg_audio_head_20261003/shared_profile_shortlist_dev_attempt03/`.

Native v4 uses the frozen96-step checkpoint and best-confidence/far-lag
reward with distance/anchor regression penalties. Its180-row dev run is
ongoing. CPU-only staging qualified20 unique candidate waveforms with exact
production-transform replay, bounded masks, no active peak cap and fixed-RMS
error below3.25e−9. Qwen ASR then completed20 new candidate calls: all match
the official text and natural-audio hypotheses. The final content guard must
still bind those ASR input hashes to each generated-video ledger waveform;
audio staging or an earlier candidate's CER cannot replace that join.

LatentSync's native identity/RMS video and original-audio score paths now pass
single historical-fit engineering qualification; this has not opened heldout
candidate selection. SadTalker is undergoing a narrowly pinned scalar shape
compatibility repair. Confirmation audio/effects and its seed family remain
unopened until the candidate/configuration and evaluation family are frozen.

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
