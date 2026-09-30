# WORLD source, spectral shape and local energy transfer

Status: draft for independent review. Choices use calibration acoustics only;
no fresh source-axis evaluation effect has been inspected by the principal.
Freeze the reviewed text and producer input plan before physical evaluation
forwards. Keep the preceding source-axis run and its protocol immutable.

## Motivation and scope

In the fresh 20-pair calibration set, TTS has a larger WORLD voiced fraction
and lower full-track aperiodicity, and its normalized spectral envelope changes
less between adjacent frames. These descriptions are stable to global gain.
Full-track AP also includes unvoiced frames, so the difference does not by
itself establish less noise within voiced speech. Report active/voiced/phone
conditioned descriptors before interpreting that distinction.

Test whether the donor's WORLD excitation coordinates or spectral shape can
transfer a generator response in both directions, at the recipient's phonetic
clock, with unchanged scoring audio. This identifies manipulations in a WORLD
analysis/synthesis system. CheapTrick SP and D4C AP are operational coordinates;
do not claim that they are independent measurements of vocal tract and glottis.

## Inputs, units and frozen domains

- Reuse the existing 20 calibration / 80 evaluation pairs and both N/T PCM
  identities. Do not select additional utterances using any model endpoint.
- Main physical evaluation uses the already fixed primary image. The secondary
  image is a separately labeled replication if run, never a substituted primary.
- WORLD: pyworld 0.3.5, DIO floor 65 Hz / ceiling 500 Hz / 5 ms frames,
  StoneMask, CheapTrick and D4C. Bind package, extension and producer hashes.
- Analyze the same FIXED float waveforms as the representation experiment:
  RMS 0.0376838172675746. Preserve source PCM, length, sample rate and clock.
- Save F0, time, SP and AP arrays for every N/T input; save pre/post synthesis
  waveforms, gains, donor maps, attempted rows and every qualification failure.
- Synthesis has its own phase. Compare both repeats and spectral/envelope
  differences; a negative unaligned waveform SNR is not alone a fidelity gate.
  Report it without shifting the synthesized waveform to maximize agreement.
- Synthesis normally adds a final partial frame. Right-trim to the exact
  recipient sample count; record the discarded tail and its length. If shorter,
  mark an engineering failure rather than padding or silently stretching it.

## Phone map

Use the frozen MFA 3.4.1 Mandarin acoustic model/dictionary and publisher text,
not SyncNet or video landmarks, to obtain N/T TextGrids. Bind their hashes and
qualified input clocks. Preserve silence as the recipient's own parameters.

The physical phone normalizer is `world-phone-identity-v1`: Unicode NFC and
outer whitespace removal only. Preserve case, aspiration, length, palatalization,
stress and tone marks. It is separate from the broader visual phone-class
normalizer. Freeze an explicit raw-to-normalized table from the model inventory
and calibration labels plus the normalizer source hash. Apply this same rule to
new evaluation labels; unknown/OOV labels have no donor exposure. Silence/unknown
sets use the existing MFA adapter's frozen labels, with its source hash bound.
Text normalization uses that adapter's publisher-text/character segmentation;
save exact lab text and raw/NFC word tokens. Choose exactly one `words` and one
`phones` tier (or a uniquely identified speaker-prefixed suffix); ambiguity is
a qualification failure. Save selected tier names. Clamp a qualified boundary
within its existing 40 ms audio tolerance to [0,duration] for the donor map,
while retaining the original boundary and clamp amount in the audit.

Pair word tokens by transcript position, requiring identical nonempty word
labels. Within each paired word, pair speech phones by an order-preserving
longest common subsequence of the frozen normalized labels. Resolve equal
solutions lexicographically by the vector of recipient indices, then donor
indices. A phone's midpoint determines its containing word interval. Unmatched
phones, unknown labels, out-of-domain intervals and silence retain the original
recipient parameters and are counted explicitly; they are not deleted rows.

For a matched recipient phone [r0,r1) and donor phone [d0,d1), map each recipient
WORLD center t to d0 + (t-r0)*(d1-d0)/(r1-r0). At the final audio endpoint only,
allow the final closed interval. Use no global shift, dynamic time warping,
video-derived adjustment or waveform time resampling. Save paired intervals,
raw/normalized labels, matched time fraction and per-factor actual exposure.

WORLD donor voicing is nearest-center F0>0; ties choose the earlier frame.
Interpolate log F0 only within the donor's contiguous voiced run, with endpoint
holding at that run's edges. Do not interpolate across an unvoiced interval.
Interpolate AP linearly by frequency bin; clamp only numerical excursions to
[0,1]. Interpolate log normalized SP with a 1e-12 power floor and renormalize
the mapped shape to unit sum. Interpolation is bounded to the matched phone.

SP denotes the numeric spectral-envelope array returned by this pinned
CheapTrick API. Normalize by the simple sum of its equally spaced bins,
including DC/Nyquist, rather than a trapezoidal integral. Require identical
recipient/donor fs, FFT size and frequency grid; store the bin frequencies.
Save each reconstructed SP row's sum error relative to the original row.

Missing or unqualified alignment keeps the intended row, produces an explicit
failure, and blocks the full-cohort primary claim. Partial unmatched phones use
the identity policy above, so the estimand remains the intention to apply this
fixed transfer procedure to all 80 pairs. Report its actual intervention dose.

## Waveform arms per recipient

Every arm is synthesized with the same engine, trimming and global RMS policy.
The own WORLD reconstruction is the within-system baseline. First save its
identity repeat and the RAW-to-reconstruction bridge; do not subtract that
bridge and call the residual a general mediation fraction.

| Arm | F0 / voicing | AP | SP shape and frame energy |
|---|---|---|---|
| RT | own | own | own |
| EX | mapped donor | mapped donor | own |
| SP | own | own | mapped donor shape × own SP summed power |
| BOTH | mapped donor | mapped donor | mapped donor shape × own SP summed power |
| F0 | mapped donor on jointly voiced mapped frames; otherwise own | own | own |
| AP | own | mapped donor on jointly voiced mapped frames; otherwise own | own |

EX transfers F0, voiced/unvoiced pattern and AP jointly. F0 and AP sub-arms
retain recipient voicing and explicitly report jointly voiced exposure. Do
not attribute EX minus these two effects to voicing: interactions and their
different exposure domains remain. A voicing-specific follow-up, if needed,
requires its own explicit F0/AP fill policy and new freeze.

Outside qualified matched phones every parameter is exactly the recipient's
own value. SP transfer preserves each original SP row's total power; global
RMS is set after synthesis without clipping or quantization. This does not
guarantee identical local waveform energy, which is measured separately.

## Local energy controls

Keep the six unmodified synthesis arms above as the primary physical factorial
and sub-arms. Add explicitly labeled energy-matched EX and SP sensitivities,
and one energy-only arm E. They answer a separate local-envelope question and
must not replace an unfavorable primary arm after observing effects.

At 16 kHz, the envelope window has indices j=-400,...,400 and weights
`w[j]=0.5*(1+cos(pi*j/400))`, divided by their sum. Evaluate weighted squared
waveforms at centers 0,80,...,80*floor((n-1)/80), reflecting at audio boundaries.
The RMS envelope is the square root of these powers. Floor RMS at 1e-5 only
when computing ratios. Match to the own RT envelope with eight fixed iterations.
Let y0 be the length-preserved, globally FIXED synthesis and let L0[s]=0 dB.
Each iteration computes

`delta[k]=clip(20*log10(max(target[k],1e-5)/max(env(yj)[k],1e-5)),-3,3)`;

linearly interpolates delta to sample times (holding endpoint gains), and sets
`L(j+1)[s]=clip(Lj[s]+delta_interp[s],-12,12)`. Then form
`v=y0*10**(L(j+1)/20)` and `y(j+1)=v*FIXED_RMS/rms(v)`.
The +/-12 dB limit applies to the local cumulative L, after interpolation.
The final global normalization is a separately saved scalar and does not count
toward that local bound. It is recomputed from y0 and L on every iteration.
Apply the same routine to RT; identity target must retain float32 samples with
max absolute error <=1e-7 and relative RMS error <=1e-6.
Report achieved active-frame log-RMS differences, p50/p95/max residual,
bound-hit counts and gains; never describe approximate matching as exact.

E starts from RT and targets its envelope multiplied by the mapped donor's
FIXED-input envelope divided by the recipient's FIXED-input envelope. The
ratio is one outside matched phones and is bounded to +/-12 dB. Use the same
eight-iteration routine and final fixed global RMS. Save intended and achieved
targets. This changes within-utterance energy while retaining RT's spectral
source construction and recipient clock; modulation side effects are audited.

The matching implementation and its waveform-only QC must pass on calibration
before freezing evaluation production. No matching parameter may be selected
using SyncNet or visual endpoint effects. If the calibration controls fail,
repair and create a new version before physical evaluation rather than hiding
the residual mismatch.

## Generator and evaluator paths

Generate with the unchanged Wav2Lip pipeline and fixed primary ROI. Score every
physical arm first against the unchanged RAW recipient audio. This is the
generator response, on the recipient's original clock. Also cache each arm's
SyncNet A embedding so evaluator and interaction responses are measurable.

For each arm X relative to own RT, report the four cells:

- q00 = V_RT, A_RAW_recipient;
- q10 = V_X, A_RAW_recipient;
- q01 = V_RT, A_X;
- q11 = V_X, A_X.

The within-WORLD set is explicitly q00=V_RT/A_RT, q10=V_X/A_RT,
q01=V_RT/A_X and q11=V_X/A_X. Keep its summaries separate from the RAW set.

The evaluator operation here replaces RAW recipient scoring audio by the
physical X waveform. Its own reconstruction component is explicit. Also
report the within-WORLD four cells with A_RT in q00/q10, using the same V
features, to separate the reconstruction bridge from donor replacement.
RAW FIXED generation from the preceding run provides a verified cached bridge
when input/face/frontend/embedding hashes and common support agree; otherwise
regenerate the bridge under this run and record the discrepancy.

Use official JPEG visual frontend, FLOAT PCM MFCC units, raw/unit geometry,
the already calibrated global k=3 and guard20 main support. Freeze one common
recipient support across all physical arms and both scoring-reference domains.
Record full/valid/guard15 sensitivity support. No per-arm best alignment or
support deletion may replace the fixed-anchor comparison.

Cache the Wav2Lip z trajectory for each arm. As secondary descriptive links,
project X-minus-RT encoder changes onto the already frozen d, u and d's
component outside u. Use a common mask derived from the original FIXED
recipient waveform and its generation window centers, not an arm-specific
activity selection. Report clip-then-speaker means, actual exposure and
residual displacement. Do not fit a new direction, regress an axis using
evaluation outcomes, or call these projections causal mediation percentages.

## Inference and independent visual endpoint

Primary source transfer is the conjunction of four 99% confidence intervals:
EX minus RT is positive for N recipients and negative for T recipients, for
both raw guard20 C and C_anchor scored against the unchanged RAW recipient.
Use all intended 40 speaker means of two paired clips, 20,000 PCG64 bootstrap
draws with seed 20260930. Missing planned scoring artifacts or an incomplete
speaker block makes the full-cohort claim ineligible; retain estimates and
every failure. This is a directional source-coordinate claim within WORLD.

Report SP, BOTH, EX/SP factorial interaction, F0, AP, EX/SP energy-matched
sensitivities, E, raw/unit B/D/search gain, donor-map dose, active energy
residuals and RAW/RT bridge irrespective of sign. The factorial interaction is
BOTH-EX-SP+RT at the same audio-reference domain. These are prespecified
secondary effects, not additional opportunities to pass the primary gate.

Use the existing cal-frozen MediaPipe/phone rules and recipient MFA alignment.
Compute within-clip open-vowel minus bilabial aperture separation, then equal
two-clip/speaker means. Do not refit latency or phone count gates for physical
arms using evaluation effects. Report contrasts, timing diagnostics, missing
coverage and pose/velocity separately. A source-coordinate score response may
be reported even if the independent visual endpoint is unchanged; that result
describes this generator/evaluator response rather than perceptual improvement.

Seal all intended physical inputs, code and conditions before model forwards;
seal all row artifacts before effect summaries. Independent validation must
reconstruct phone maps, parameter identity/copy domains, waveform length/RMS,
scoring reference, lag matrices and intended speaker denominator.
