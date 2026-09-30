# Encoder direction and radius factorization

Status: declared from CAL geometry before the principal reads fresh EVAL
effect summaries. Keep the original source experiment unchanged. Freeze this
spec, input/code/row seals and numeric calibration QA before new forwards.

## Question and calibration basis

The source intervention is orthogonal to the measured level secant u, but
that does not hold the norm of the decoder's z input constant. Equal delta
norm random controls also need not have equal resulting z norms. Separate
direction and radius of the existing exact orth_plus trajectory to identify
their generator responses without changing scoring audio or phonetic clocks.

CAL-only independent KKT diagnostics bind all20 feature NPZs and original
axes/masks in calibration_radial_geometry.json, raw SHA256
800d0a0d040df2337249958c907b28eb13ef75173392778ce7c674eed1fce17e.
The N orth_plus mean radial cosine is -.088 and relative norm change -.145%;
displacement/norm is about1.35%. These are calibration geometry, not effect
estimates. They motivate factorization rather than treating a latent norm
change as an audio gain change.

## Inputs and four generator cells

Use the same fixed 80 natural recipients / 40 speakers, two clips each,
primary image, original d/u and recipient clocks. Read only the independently
verified saved float32 baseline z0 and exact orth_plus z1 from retry3 contexts;
bind each context and array SHA. Do not refit or reproject those trajectories.
For every generation frame, compute r0=norm(z0), r1=norm(z1) in float64:

| Cell | Direction | Radius | Decoder trajectory |
|---|---|---|---|
| q00 | own | own | exact saved z0 |
| q10 | source-plus | own | float32(z1 * r0/r1) |
| q01 | own | source-plus | float32(z0 * r1/r0) |
| q11 | source-plus | source-plus | exact saved z1 |

Require finite, nonnegative arrays and strictly positive r0/r1. A zero radius
is an explicit failed planned row, not an arbitrary replacement direction.
Check q10 norms against r0 and q01 against r1 with relative error <=1e-6;
check copied endpoints bit exact. Save radius ratios, cosine changes, actual
delta norms and u projections before/after float32 conversion. q10/q01 do not
inherit q11's strict u-orthogonality claim. Apply all-frame factorization; use
the original FIXED N activity mask only for descriptive summaries.

The cached encoder outputs are the fixed inputs of this experiment. Render
all four cells anew with identical decoder/SyncNet execution: restore
deterministic algorithms and benchmark=false after detection, record actual
flags at forwards, and do not mix old videos with newly rendered conditions.
Record the cached encoder execution provenance and compare new q00/q11 to
old artifacts as an engineering bridge without substituting them for new rows.

## Scoring and inference

Score all four videos against the unchanged original RAW N PCM, using a new
common A feature cache, official JPEG frontend and float PCM MFCC units.
Freeze common recipient support across four cells: raw/unit, full/valid,
guard20 main and guard15 sensitivity; original k=+3 / column18 / offset=-3.
Save all A/V embeddings, full cross/zero-boundary/31-lag arrays and 320 planned
score rows. No per-condition lag search replaces C_anchor.

Direction effect is q10-q00, radius effect q01-q00, joint effect q11-q00 and
interaction q11-q10-q01+q00. Their arithmetic decomposition is not a mediation
percentage. The directional confirmation gate requires all four 99% lower
bounds >0: q10-q00 and q10-q01, each for C and C_anchor, raw guard20.
Use 20,000 PCG64 speaker-bootstrap draws, seed20260930, equal40 speaker means
of two paired clips. Missing required support keeps the intended denominator
and makes the complete-cohort gate ineligible; label subset estimates.

Report all four cells and contrasts, B/D/search gain, unit geometry, original
masked z summaries and dose distributions irrespective of sign. Apply the
original 100 full-A row permutations, shared across all four videos, to report
observed versus null response and sync excess. Independent validation checks
the radius/direction identities and reconstructs every unique full cross
matrix. The existing frozen phone separation/AUC analysis is a separate
visual endpoint, with original availability and latency rules retained.
