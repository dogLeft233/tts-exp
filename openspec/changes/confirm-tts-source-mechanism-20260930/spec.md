# TTS source direction and waveform mechanism confirmation

Status: design frozen before fresh evaluation forward passes or effect scores.
Input and code seals will bind the concrete artifacts before execution.

## Question

Separate the waveform level path, the Wav2Lip generator representation path,
and the SyncNet evaluator path behind a native TTS score advantage. Test a
positive, transferable generator mechanism using unchanged natural scoring
audio and its original clock. Analyze visible mouth motion separately.

## Inputs and separation

- Publisher: AISHELL/AISHELL-1, revision
  `bbe295d530192a4cd41644b711c9aecd087df653`.
- Calibration: S0052–S0061, two utterances per speaker, 20 utterances.
- Evaluation: S0062–S0101, two utterances per speaker, 80 utterances.
- Select the first two lexicographic utterance IDs with mono 16 kHz PCM,
  duration 3–10 seconds, and a nonempty publisher transcript containing only
  Han characters after whitespace removal. Selection never reads model scores.
- These speakers exclude the prior natural80 S0002–S0041 and the historical
  test-speaker calibration. Failures and all selection attempts remain visible.
- Qwen cloud self-clone: `qwen3-tts-vc-2026-01-22`, text is the publisher
  transcript, reference is that utterance's natural audio. Preserve the original
  cloud 24 kHz response and separately bind the canonical 16 kHz waveform.
- Two local Chinese utterances are engineering pilots and never scientific
  calibration or evaluation observations.
- Primary face: `data/aishell1_100_zh/image/neutral_01.png`; secondary face:
  `data/data/image/3.png`. Fixed faces, boxes, model weights, and clocks.

## Level four cells

FIXED rescales each utterance to RMS `0.0376838172675746`. Preserve length,
floating PCM and peaks; do not clip or quantize to simulate a level change.
For each N or T recipient, use its own clock and one common support:

| Cell | Generation waveform | Scoring waveform |
|---|---|---|
| q00 | RAW | RAW |
| q10 | FIXED | RAW |
| q01 | RAW | FIXED |
| q11 | FIXED | FIXED |

G=q10−q00, E=q01−q00, I=q11−q10−q01+q00. Report effects separately for N/T
and the T−N path contrasts. Native gap is q00(T)−q00(N); processed gap is
q11(T)−q11(N). Their difference is the controlled gap attenuation under this
operation, not a general mediation percentage.

## Calibration axes

Wav2Lip z is the nonnegative 512-dimensional audio-encoder output. Use FIXED
waveforms and the same 16-column mel windows as generation. A speech mask
is the corresponding 200 ms waveform-window RMS above `10**(-45/20)`.
For each clip, average active z; then average its two clips per speaker and
the ten speakers equally. The source direction d is mean(T)−mean(N).

The level secant u is the similarly aggregated z difference between the same
FIXED natural waveform at +3 and −3 dB, divided by 6. Both forwards use the
FIXED-natural mask. No evaluation z contributes to axis fitting.

## Generator intervention

All eight conditions start at G(FIXED N) and score against the same RAW N
audio. Inactive frames are copied exactly. Conditions are baseline,
full_plus, full_minus, orth_plus, orth_minus, random1, random2, random3.
Full shifts are `ReLU(z ± 0.5*d)`.

The strict orthogonal shifts solve, per frame:

`min_y ||y-(z ± 0.5*d)||²`, subject to `y>=0` and `u·(y-z)=0`.

Use the scalar KKT projection, and record float64 residual and float32
rounding bounds. This tests the component outside the measured level secant;
it is not a claim of orthogonality to every nonlinear level response.

Random directions use seeds 20260930, 20260931, 20260932; remove their u/d
components and normalize their intended displacement to `||0.5*d_perp||`.
Use the same constrained projection. Match the actual orth_plus displacement
norm per frame with a scalar search where feasible; explicitly record dose,
projection and infeasible cases, never silently discard a cell or observation.
Search a nonnegative multiplier on [0,64], bracket from 1 by doubling,
then bisect for 64 steps. The norm tolerance is
`max(1e-6, 1e-6*target_norm)`. A zero target copies z. If the upper bound
cannot reach the target, preserve that output and flag the dose mismatch.

## Scoring and primary inference

Use official SyncNet V2 weights and JPEG visual frontend. Preserve the source
PCM independently of mux encoding. FLOAT32 PCM reaches MFCC in float64
integer-amplitude units (`x*32768`), with no rounding, int16 cast, or clipping.
Check PCM16/float-identity, no-op/repeat, and known five-frame delay controls.

Keep one common per-recipient support across conditions. Report raw/unit
geometry and full/valid/guard20/guard15 support. Main endpoint is raw guard20.
The one global anchor uses calibration natural G(FIXED N), A_RAW N (q10).
For each clip select the exact curve argmin; exact ties favor proximity to
historical k=3 then the smaller k. Take each speaker's two-clip median, then
the median of the ten speaker values, rounded to the nearest integer; exact
half-integer ties again favor proximity to k=3 then the smaller integer.
Here k indexes A[i+k] against V[i]; the official AV offset is -k.
Evaluation cannot select its own anchor. k=3 is a sensitivity view.

Primary claim requires orth_plus−baseline and orth_plus−orth_minus to have
99% bootstrap lower bounds above zero for both C and C_anchor. Treat this as
one conjunction; retain every individual interval. Full-direction, random
mean, B/D/search, unit and secondary-face contrasts remain prespecified
secondary results and are all reported. Use 20,000 PCG64 bootstrap draws,
seed 20260930, equal speaker means of paired clip effects. Frames and tokens
are not independent observations.

Generate and seal all condition artifacts before computing effect summaries.
Preserve failures and the distinction between engineering control failure and
an estimated scientific effect. Store code/model/input hashes, A/V/z features,
videos for visual audit, calibration axes, scores and independent verification.

## Cached temporal null

For each evaluation clip, draw 100 uniform permutations of its full cached
RAW-N A row domain with PCG64 seed `20260930 + ordered_eval_index`. Use the
same permutation for all eight arms and both faces. Keep V and the frozen
guard20 query rows unchanged. Recompute the 31-lag curve, B, D, C,
search_gain and calibration-anchor C_anchor in raw and row-unit geometry.
This preserves the exact A/V row multisets and breaks representation-time
pairing; it is not a waveform intervention or a perceptual timing measure.

For orth_plus minus baseline, compute clip-paired, two-clip-equal speaker
means and the 40-speaker mean for each permutation. Report diagnostic
one-sided `p=(1+count(null_mean>=observed_mean))/101`. Also report the
99% speaker-bootstrap interval for observed minus mean-null paired effects.
All are secondary; the four primary gates remain unchanged. Precompute full
cross-distance matrices and gather permutations without extra model forwards.

## Independent visual endpoint

The CPU MediaPipe analysis in `tts_mechanism_visual_20260930.py` measures
mouth shape/motion independently of SyncNet. Where qualified phone alignment
exists, its primary endpoint is open-vowel minus bilabial aperture separation,
with speaker-level paired inference and support/missingness in the denominator.
Use all eight generator labels. The two primary visual contrasts are
orth_plus minus baseline and orth_plus minus the scalar endpoint mean of
random1/2/3; both need 99% lower bounds above zero. Compute each phone-class
mean within each clip, its aperture separation, then equal means of the two
clips per speaker. Never average trajectories to invent a random-control video.
Retain both expected clips and every arm in failure/coverage denominators.
Whole-track extrema matching and calibrated latency are exploratory timing
endpoints. A repeated-frame or shuffled trajectory cannot count as improved
timing. Keep overall aperture, velocity, pose and phonetic separation distinct.
Without qualified phone alignment, report movement descriptors without a
phonetic timing claim. Perceptual naturalness remains a separate endpoint.

## Resampling robustness

Prepare a natural 16→24→16 kHz roundtrip using the same canonical ffmpeg
binary and resampler settings as cloud-output conversion. Bind PCM hashes,
waveform differences, mel and MFCC/embedding differences before reporting its
native-score robustness contrast. This checks a conversion contribution to the
native N/T gap; it does not gate the natural-audio source-direction intervention.

## Following physical mechanism

Use calibration acoustics and the completed level/source-direction results to
freeze source/filter or local dynamic-range interventions. WORLD roundtrip is
its own baseline; its bridge to RAW must be quantified. Existing incomplete
WORLD audio-only feature caches do not supply missing visual embeddings or a
producer protocol. Do not finish the main inquiry with an exclusion list.
