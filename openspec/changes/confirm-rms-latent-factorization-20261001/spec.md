# Generator level response through encoder direction and radius

Status: informed follow-up declared after the principal read the completed
level G/E/I experiment and the independent static source-axis factorization.
The latter starts at FIXED N and factorizes orth_plus. It does not factorize
the RAW-to-FIXED RMS response. Keep these experiments distinct.

## Question and basis

Under unchanged RAW-N scoring audio, utterance RMS normalization produced
primary G=+.192, speaker99% CI[+.111,+.276], and positive secondary-face/unit
responses. Factorize the actual encoder trajectory for that operation to
locate the generator response in its direction/radius coordinates.

The static-source factorization's q10 direction-only contrast was -.013,
99%[-.035,+.010], and radius-only +.006[-.021,+.033] on80/40. These results
motivate explicitly naming each pair of endpoints, not substituting static
source-axis coordinates for a waveform-gain trajectory.

## Frozen endpoints and qualification

Use all original80 natural clips/40 male speakers, two clips each, the frozen
primary image/face box and original frame/mel clocks. No TTS waveform or
phonetic time warp is needed for this stage. Original TARGET_RMS remains
.0376838172675746; FLOAT PCM length, unclipped gain and frontend definitions
remain those of confirm-mel-level-partition-20261001.

Let z0 be the saved float32 audio-encoder output from that new mel run's
q00=frontend(RAW N), and z1 from q11=frontend(FIXED N). Use the exact arrays,
not a fitted mean level direction, +3/-3dB secant u or orth_plus trajectory.
Bind the complete mel run, planned rows, input/code seals, endpoint file and
array hashes, checkpoint, image and actual recorded encoder forward flags.
Mel q00/q11 effects need not be positive to retain these planned rows.

CAL numerical QA uses the original20 calibration feature NPZs' natural_z_raw
and natural_z_fixed arrays, with every file/array hash. It fits no axis or
threshold. Report finite/nonnegative shape checks, radii, radius ratios,
cosines, per-frame displacement and exact factorization identities before
new evaluation decoder forwards. For EVAL, seal all80 exact endpoints and
factorized inputs before the first new decoder forward.

## Four fresh decoder conditions

For every frame calculate r0=norm(z0), r1=norm(z1) in float64:

| Cell | Encoder direction | Encoder radius | Decoder input |
|---|---|---|---|
| q00 | RAW | RAW | exact z0 |
| q10 | FIXED | RAW | float32(z1*r0/r1) |
| q01 | RAW | FIXED | float32(z0*r1/r0) |
| q11 | FIXED | FIXED | exact z1 |

Require finite nonnegative arrays and positive radii. Zero radius retains a
failed planned row; never invent a replacement direction. Verify copied
endpoints bit exact and mixed-cell relative norm error<=1e-6. Apply the
factorization to every generation frame. Original FIXED N activity masks
are descriptive only; do not impose inactive identity on mixed cells.
Save radii, cosines, actual delta norms and float32 errors for all frames.

Decode all four conditions anew under identical weights, face inputs,
batching, and actual deterministic/cudnn flags recorded after detection and
at each forward. Keep cached-encoder provenance separate from fresh decoder
flags. Compare new q00/q11 pixels, V embeddings and scores to the mel run's
two endpoints and the original level experiment as engineering bridges;
never mix those older videos into the new four-cell effects.

## Scoring and statistics

Score all320 planned rows against one new common unchanged RAW-N A cache
per clip, official JPEG/MFCC units, k=+3/column18/offset=-3. Save MP4/z/A/V,
full cross and zero-boundary/31-lag matrices. Preserve raw/unit geometry and
full/valid/guard20/guard15 common support across all four conditions.

Freeze direction=q10-q00, radius=q01-q00, joint=q11-q00,
interaction=q11-q10-q01+q00, and q10-q01. Report B,D,C,C_anchor and corrected
search_gain=D0-D. Paired clip effects -> two-clip speaker means -> equal40
speaker means; PCG64 seed20260930,20,000 draws,99% percentile intervals.
Planned failures remain in intended denominators. Positive lower99% bounds
for joint C and C_anchor constitute replication of the level generator
response in the cached-encoder/fresh-decoder route. All factor contributions
are informed secondary mechanism estimates, not an exclusive-winner test.

Apply100 full-A row permutations per clip, seed20260930+ordered_eval_index,
the same permutation across all four videos. Report observed, mean null,
observed-minus-null and complete-speaker intervals in all support domains.
An independent consumer reconstructs factorization and matrices without
importing producer operations/scoring code.

## Interpretation and storage

In real-valued coordinates the latent four-cell interaction itself equals
(r1-r0)*(z1/r1-z0/r0). Thus an output interaction alone does not identify
decoder nonlinearity; this factorization already has multiplicative input
geometry. Record its latent norm descriptively. Do not call these quantities
an audio-power mediation percentage or perceived lip-sync improvement.

Use the private /root/tts-research-system-20261001/ root, fresh empty output,
controls outside output, source/input/row seals before forwards and complete
artifact seals after. Share A metadata without dropping exact arrays needed
for independent reconstruction; crop/JPEG temporaries remain outside seals.
