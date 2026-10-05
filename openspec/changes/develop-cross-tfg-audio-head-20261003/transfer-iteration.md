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
fit rows on eight historical clips, one shared seed (20261003), and three frontends.
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
reward with distance/anchor regression penalties. Its180-row native dev run
completed180/180 PASS. CPU-only staging qualified20 unique candidate waveforms with exact
production-transform replay, bounded masks, no active peak cap and fixed-RMS
error below3.25e−9. Qwen ASR then completed20 new candidate calls: all match
the official text and natural-audio hypotheses. Independent checks qualified
all80 WAVs and20 real API requests, with0 errors across297 annotated characters.
The actual180-row render join then matched the enhanced generator PCM to
those ASR input hashes and preserved natural score PCM/source clocks. Content
has20 independent waveform units. Independent visual extraction completed
240 MuseTalk/Ditto baseline rows and180 V4 rows; the V4 archive's189 members,
180 trajectory NPZs and2520 arrays passed root verification. The completed
joint evidence archive binds540 rows,1700 assets and1721 regular members.
Root independently reconstructed42171 numeric fields and every nine-way query
support, with maximum float32 reduction difference5.72e−6 under declared1e−5
tolerance. All10 speakers per model are complete on FULL support:

| Model | V4 minus identity Cbest | V4 minus RMS Cbest |
|---|---:|---:|
| Wav2Lip | +0.194 | −0.040 |
| MuseTalk | +0.024 | −0.029 |
| Ditto | +0.110 | −0.064 |
| Model-equal | +0.109 | −0.044 |

These are development means. V4 remains below RMS on every model and does not
meet the useful-method target. Relative to RMS, model-equal B rises0.016 while
Dmin rises0.060 and D0 rises0.036; Canchor falls0.021. GUARD20 keeps513 eligible
and27 ineligible planned rows. S0022 has only one eligible clip, so its complete
ten-speaker endpoint is null. The separate common nine-complete-speaker subset
has model-equal V4-minus-RMS Cbest−0.023; it does not replace FULL.

The common visual summary completes540 rows and180 contexts; root recomputed
8640 fields with zero difference. Mean normalized aperture is0.072927 for V4
and0.071474 for RMS; velocity RMS is0.474874 and0.474698 respectively. Original
PCM40ms log-RMS association has zero-lag aperture Pearson r0.348833 versus
0.349786, and onset-velocity r0.022574 versus0.020315. Root independently
recomputed34264 fields over all31 sensitivity lags (maximum difference7.78e−16).
Neither a larger mouth nor these descriptive correlations establish quality.
The frozen-weight audio-only diagnostic below separates mean log-mask μ(f)
from its temporal residual δ(t,f), with dev20 and historical-fit8 kept separate.

Authoritative local score and independent reduction evidence are in
`runs/cross_tfg_audio_head_20261003/v4_cross_frontend_score_evidence_readback_attempt01/`
and `v4_join_root_independent_readback_attempt01/`. The native candidate,
baseline matrices and failed/unexecuted repairs remain immutable.

LatentSync's native identity/RMS video and original-audio score paths now pass
single historical-fit engineering qualification; its complete92×31 paired
matrices, original PCM16 and source clocks have passed independent readback.
This has not opened heldout candidate selection. SadTalker's narrowly pinned
scalar shape compatibility repair completed both engineering arms; attempt06
has now passed strict local restoration and root raw-matrix/clock reduction.
Its single historical-fit pair has RMS-minus-identity Cbest −0.044 and serves
only as engineering qualification. Confirmation audio/effects and its seed family remain
unopened until the candidate/configuration and evaluation family are frozen.

Shared540 generation awaits5 GiB free SHM. The first old fit output is fully
archived on C and F with identical SHA and strict restoration of all47,105
files. A later full hash disagreed for the B mount, so ADR012 uses the verified
C/F mirrors plus restoration and a fresh exact activity/input-reference audit.
The first exact run output has now been retired after that audit; its task
code/control/ditto_control, frozen plan/rows, inputs/weights and first remote
archive remain. The other three outputs use sequential archives in the existing
manifest format and restoration on F. MuseTalk's source-preserving packer has
completed a sealed 517,027,505-byte archive. C/F whole-file hashes and strict F
restoration of all9,644 files passed; a fresh full-tree/dependency audit and
guarded exact-run retirement released683,323,392 allocated bytes. The actual
2,819-byte retirement receipt is sealed as SHA256
`087b7278b43caafe81c0ac164d1dac4b7bc64cdabf2ce513f1181b1dc64923fe`.
The temporary remote Muse tar passed a separate fourteen-protected-path and
live-reference gate, then exact-file removal; its1,790-byte receipt is sealed
as SHA256 `dafc818359b4cc281d441d3b4d704840bd53a52e18aca84903bd14390ca62cc8`.
The pack controls and C/F archives remain. The failed-fit archive's second
attempt stopped at its1GiB reserve and preserved its source; fresh attempt03
uses allocated bytes plus1GiB as its conservative admission budget and has
actually completed a1,291,222,198-byte source-preserving archive (SHA256
`234006a3cc5efef762f1a943b79529fe4443751fad7f56a76cf718579e888638`).
Its C/F whole-file transfer and strict restoration of all46,089 files now pass;
root independently rehashed all323 C ranges and the F archive and result files.
The strict restoration receipt is52551 B, SHA256
`e4ea0cef5056e92b5f1e1455d0183ce66a530e8af107228ac951c5f34cd65302`.
The source run and temporary remote tar remain pending separate exact retirement.
The first read-only audit failed on a presumed `TASK_ROOT/code` directory;
actual inventory locates code in `TASK_ROOT/scripts` and root source files,
with controls under `job_control/shared_frequency_profiles_fit_attempt03`.
The fresh audit protects those actual paths and preserves the failed receipt.
Its actual21,769-byte receipt is SHA256
`4fc41b5b2878461a040a45b9c6696b275abe9405da501a87d06943cddaf86f7a`.
All nine gates pass, with zero active or unresolved queued references. Root
independently read the F archive's sealed member manifest and reproduced the
50,635-record canonical digest, matching the freshly rehashed source tree.
Old Ditto's actual target is `stageb_ditto_attempt03_retry04/run`, independently
bound to its existing plan and controls. A combined new format would require more than the available
root space for only15% deduplication. The qualified score archive now removes
the active analysis dependency on the MuseTalk/Ditto source trees; each still
requires verified archives and restoration before its exact run is retired.

1. Complete recoverable space release and run the frozen540 native development
   renders for the shared spectral/local profiles. Reuse controls only with
   exact source/model/config/seed and artifact bindings.
2. Use the completed frozen V4 audio-only μ/δ/a/r diagnostic to define a small
   local spectral-gate family. Any subsequent video ablation uses the common
   scale in ADR011 and a new frozen plan.
3. Inspect B, Dmin, D0, offsets and independent mouth motion for every shared
   candidate. Preserve FULL/GUARD20 failures and planned denominators.
4. If the small historical fit set is unstable, expand within the already
   designated historical fitting speakers. Keep development selection and
   untouched confirmation separate. A failed candidate remains in the method
   table and informs the next training or control change.
5. Freeze the selected shared configuration/weights and complete untouched
   speaker and reserved-model confirmation, component ablations and delivery.
   Calibrated backend presets remain separately named modes with costs.

## Local spectral structure qualified from actual waveforms

### Completed shared-profile development

All 540 new candidate renders and 360 qualified reused control rows now have
a complete, locally restored 900-row evidence archive. Independent CPU replay
uses the producer's Torch 2.5.1 float32 reductions and exactly reproduces all
1,755 eligible curves and metrics across 1,800 FULL/GUARD20 scopes. The earlier
NumPy tolerance failure and Torch digest failure are retained as engineering
failures. Root separately reconstructs seed-to-clip-to-speaker means and equal
model weights; all 384 stored report comparisons agree within 5.55e-17.

| Profile | Wav2Lip minus RMS Cbest | MuseTalk minus RMS Cbest | Ditto minus RMS Cbest | Model-equal |
|---|---:|---:|---:|---:|
| accent0.15, tilt+0.25 | -0.016 | +0.002 | -0.008 | -0.008 |
| accent0.30, tilt0 | +0.011 | -0.009 | +0.013 | +0.005 |
| flatten0.15, tilt-0.25 | +0.031 | -0.063 | -0.047 | -0.026 |

No profile meets three positive models and model-equal +0.050 over RMS.
Accent0.30 gains +0.159 over identity, so the RMS comparator absorbs most of
that apparent benefit. FULL retains all 900 rows. GUARD20 keeps 855 eligible
and 45 planned short-support rows for S0022/W0123; its full ten-speaker endpoint
remains null and its nine-speaker sensitivity does not replace FULL.

The three manual profiles are concluded below the shared-selection criterion.
The independent historical fit40, nine-parameter local spectral-change route
continues with fixed RMS, same-GPU zero/RMS qualification and 120 fixed updates.
No learned weights, configuration selection or confirmation effect is claimed.
The authoritative full-precision root contrasts are in
`runs/cross_tfg_audio_head_20261003/shared_profile_900row_join_attempt07_readback_attempt01/root_whole_archive_attempt01/root_full_paired_metrics_attempt01.json`.

The actual CPU diagnostic covers20 development clips from10 speakers and8
historical-fit clips from4 speakers. All20 development transforms reproduce
the production V4 waveform exactly. Root recovered the complete46,753,644-byte
NPZ (SHA256 `69925e7659e8a3e2defd5acb55a7aab5e6b92660fcbd8f9fc377aa9c1e1982be`),
verified all196 arrays, and independently reconstructed180 decomposition
fields with maximum error2.13e−14. An alternative NumPy STFT and manual Pearson
reduction from the28 hash-bound original PCM WAVs verified340 association
fields with maximum error1.33e−15. These are descriptive frozen-weight
measurements; no new render, ASR call, training or confirmation read occurred.

For each log-mask, remove its time mean μ(f), then write the temporal residual
as δ(f,t)=a(t)+r(f,t), where a is the arithmetic frequency mean and r has zero
arithmetic frequency mean. The table uses clip means followed by speaker
means; each speaker has equal weight within its own cohort.

| Measurement | Development20 /10 speakers | Historical fit8 /4 speakers |
|---|---:|---:|
| Centered static μ RMS, dB | 0.386 | 0.378 |
| Dynamic δ RMS, dB | 0.771 | 0.765 |
| Common-frequency a share of δ squared log-mask norm | 29.658% | 30.135% |
| Residual-frequency r share of δ squared log-mask norm | 70.342% | 69.865% |
| a RMS / r RMS, dB | 0.420 /0.644 | 0.422 /0.637 |
| Pearson a vs original local logRMS | −0.082 | −0.087 |
| Pearson a vs positive local logmag flux | +0.434 | +0.487 |

The centered static profile suppresses0–300Hz by about1.200dB and300–1000Hz
by0.508dB on development clips, while3–6kHz rises0.212dB; fit profiles are
similar. The dynamic residual also has opposing low/high band trajectories.
The zero-frequency-mean definition itself induces some opposition, so that
pattern alone does not establish a speech-articulation mechanism. Likewise,
the activity proxy marks98.793% of dev frames active and only0.702% quiet;
18/20 clips have no quiet frame. It is not a phoneme or VAD annotation and
does not support a quiet/silence rule.

These quantities are unweighted log-mask squared norms, not waveform power.
A zero-mean dB residual can still change physical frame power because the
source spectrum is uneven and gain is exponential. ADR011 therefore requires
the shared κ and explicit frame-power controls for causal μ/δ/a/r ablations.
The global DC gain cancels after exact RMS matching when the peak cap is
inactive. The local common gain's weak RMS association and stronger flux
association make spectral-change gates a concrete next candidate; they do
not convert V4's negative comparison with RMS into a useful-method result.

Authoritative analysis is
`runs/cross_tfg_audio_head_20261003/v4_logmask_delta_common_mode_attempt04/`.
Its final figure passed root visual review; rows and cohort statistics are
unchanged from the prior data-qualified version. Raw-array and original-PCM
independent receipts are retained alongside their corresponding readbacks.

## Qualified expanded fit inputs and deployed input normalization

ADR013's canonical expanded subset is now backed by actual audio, rather than
only roster metadata. Root independently rehashed20 official AISHELL archives
(728,484,140 B), inspected6,944 WAV headers, reconstructed all320 original
floor-mapped slots, and verified the40 unique j=0/15 endpoint PCM16 and
canonical F32 payloads. The20 speakers remain disjoint from dev and sealed
confirmation, with no replacement clips. The scanner's non-endpoint payload
status label is ambiguous; the qualified result is the canonical mapping and
40 endpoint payloads. The member report is183,939 B, SHA256
`1b4fd8472842c9c30c4611aa760b3c2b2d9f01193d3aaf7f487f5f8096b0a140`.
The maximum six new candidates, speaker folds and960 native-row budget in
ADR013 are unchanged. This is data qualification, not a new fit or effect.

The deployed Ditto HuBERT ONNX has now passed exact raw-input constant
inspection. Full-model SHA256 was independently recomputed before reading the
two scalar TensorProto payloads. Constant nodes410 and412 feed Sub411 and
Div413, giving the fixed affine preprocessing
`(x - 2.586069967946969e-05) / 0.20523720979690552`.
These are fixed float32 constants, not statistics recomputed for each input
clip. Root downloaded the original audit and receipt and independently
re-encoded both scalar values, reproducing their payload hashes. The audit
is4,647 B, SHA256
`873d525f1b1b4669594b9fbfb50fa3421f5c537da70cf967e5b3ee88ac74d449`.
Collector attempt04 failed before graph parsing because it assumed the wrong
report field; attempt05 binds the actual `source_and_weights.weights` table
and exact source commit/path/size/hash. No audio or inference was involved.
Later convolution, LayerNorm and encoder responses require the bounded
frontend probe and native generation; this affine path alone does not establish
encoder gain invariance or explain the complete transfer effect.

The900-row analysis bundle is actually staged after four bridge parts of at
most5 MiB, whole-ZIP hash verification and extraction of its eight payloads.
The analysis still awaits the540 candidate terminal rows. Execution packet07
step04 is a CPU preflight; step05 produces candidate WAVs. The60-unit ASR
guard must bind those actual generator files, retain unavailable units and
check every distinct available model/seed PCM reference before API calls.

## Measured frontend response on historical fit8

The CPU probe actually ran against the deployed Transformers 4.39.2 Whisper
feature extractor, the official Wav2Lip mel path and the qualified Ditto fixed
input affine. Eight frozen fit clips, four gain conditions and three frontends
produced 384 time curves and 96 summary cells. Original source PCM and clock
were preserved; no encoder forward, generated video or confirmation input was
used. Root independently recomputed 640 numeric checks, including Ditto frame
RMS directly from the archived PCM. Maximum summary-versus-curve error was
2.657e-8. The first output verifier used an absent summary field; its fresh
roster-only repair passed without changing or rerunning scientific outputs.

Whisper's log-power normalization predicts a common feature offset of gain
in dB divided by 40. The measured equal-clip offsets for minus/plus6 dB were
-0.150000013/+0.149999997, with centered shape RMS changes of approximately
5.6e-7. Wav2Lip's unclamped normalization predicts minus/plus0.480; its fixed
lower floor instead yields mean offsets -0.359316/+0.412290 and centered shape
changes 0.098804/0.083135. Thus even a global waveform gain can alter local
frequency directions through a fixed floor. Whisper's relative floor moves
with the input, preserving centered shape in these measured inputs.

Ditto's fixed input affine carries the gain into the first convolution. Its
frame-RMS temporal correlation remains near one under gain, and values above
absolute one are not clipped by that affine. These findings distinguish
frontend input behavior; they do not establish learned-encoder invariance.
The next mechanism probe compares identity, RMS and the three shared profiles
using actual encoded trajectories. Local spectral/transition controls remain
the working candidate family, with actual original-audio video scoring and
the RMS comparison determining whether they transfer.

Evidence: `runs/cross_tfg_audio_head_20261003/frontend_gain_feature_probe_attempt03_readback/`
contains summary SHA256 `6af5b7a66a73f8a40138fb3cac2d98adcb33751327cce4bc5c75b8d77eaad888`
and curves SHA256 `143daf5014c15ff4f59157744b809592f5dcfbbe4f70b8fef2bcb6c902232665`.
Independent receipt: `.codex/scratch/root_frontend_probe_numeric_qc_attempt01.json`,
SHA256 `ecbc5e0d37a84fe2012dde7d301a7fb5d486b7dff8ecba047226b188cfd61a96`.
These run/scratch artifacts are local evidence. Fields named
`centered_relative_rms_change` and `frame_rms_relative_change` in the frozen
probe are absolute RMS differences, not dimensionless relative ratios.

## Delivery boundary

A runnable checkpoint and exact original-audio mux are necessary engineering
evidence. Useful-method evidence additionally requires transfer across the
declared frontends, a comparison with RMS, content/visual checks, and the frozen
untouched-speaker/model evaluation. LatentSync remains excluded from candidate
selection until the shared method is frozen.

## Implementation resource appendix

The timings below are engineering-only single-input measurements used for
capacity planning. They are not throughput guarantees: each measurement covers
one historical-fit utterance and one fixed portrait, and includes a different
amount of model startup, admission wait, tracking, scoring and file handling.
No development or confirmation effect is inferred from these measurements.

| Frontend | Qualified implementation | Measured engineering input and time | Model assets / staging size |
|---|---|---|---|
| SadTalker | Official `v0.0.2-rc` source commit `28b8dfc2ebed9a00238090e5d46329cb77643207`; 256-pixel crop, 25 fps, enhancer off; Python 3.10.12, Torch 2.5.1+cu124, NumPy 1.26.4. | One 22,400-sample 16 kHz mono F32 crop, 35 output frames per arm. Native inference took 12.25 s (identity) and 11.94 s (RMS); the two-arm wrapper run took 186.98 s overall, including setup/admission and downstream work. | Four official assets total 1,183,756,176 B. The paired engineering archive is 3,085,913 B (260 regular files); this small result archive is not a full-run storage estimate. Strict readback: `runs/cross_tfg_audio_head_20261003/sadtalker_fit_smoke_attempt06_readback_attempt03/independent_local_qc_attempt03.json`. |
| LatentSync | Pinned native inference profile, 20 steps, guidance 1.5, DeepCache off; CUDA UNet and CPU face-detection sessions. | One 62,639-sample 16 kHz input (3.915 s), 128-frame 25 fps static source, 99 generated frames per arm. UNet inference took 222.725 s (identity) and 215.480 s (RMS); complete arm runs took 255.969 s and 272.349 s. | Official UNet 5,072,222,488 B; derived FP16 runtime UNet 2,536,318,892 B; VAE 334,707,217 B; Whisper 75,572,083 B; face assets 16,923,827 B and 5,030,888 B. Keeping both the original UNet and runtime derivative requires at least 7,608,541,380 B for those two files alone, before source, environment, cache or outputs. Paired engineering archive: `runs/cross_tfg_audio_head_20261003/latentsync_fit_pair_raw92_attempt07_readback/latentsync_fit_pair_raw92_attempt07.tar.gz`. |

For scale only, multiplying the LatentSync one-input inference time by 240
render rows gives roughly 14.6 GPU-hours of model inference; multiplying the
observed complete-arm time gives about 17.6 serialized render hours. This is a
rough planning scenario for similarly short inputs, not a frozen runtime SLA.
Longer audio, startup, GPU admission, tracking, scoring, retries and storage
can change wall time. SadTalker's one short pair is too small a sample for a
stable full-cohort wall-time estimate. The shared-run launch policy separately
requires at least 18 GiB free GPU memory at admission; that is a scheduler
headroom rule, not a measured minimum model allocation.
