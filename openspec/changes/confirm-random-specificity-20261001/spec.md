# Random specificity continuation

Freeze: 2026-09-30 UTC, before reading fresh EVAL scientific effect summaries.

## Reason and preserved evidence

The independent main-run consumer found that the producer stopped its scalar
search using float64 projected displacement, then cast the selected latent
to float32. Some saved random-control frames exceed the original actual-dose
tolerance after casting. The original tolerance and intended cohort remain
unchanged. Preserve the original run, arrays, scores, diagnostics and failed
validation attempts.

The original source primary requires baseline, orth_plus and orth_minus;
random means are secondary. Validate nonrandom rows using their own inputs,
arrays, interventions and scores. Record random-dose failures by arm/frame;
do not discard an otherwise qualified whole clip because a secondary random
arm failed its dose contract. The original full planned-row seal remains a
prerequisite. This fixes validation dependency scope, without adding a new
primary contrast or changing a confidence threshold.

## Corrected numerical operation

Use exact original float32 baseline z0 and orth_plus z1, activity mask,
float64 CAL d/u and three original random direction arrays/seeds.
No audio encoder forward, axis refitting, clip selection or dose change.

For each active frame, the target is
`norm(float64(z1)-float64(z0))`. Each candidate scalar uses the same
nonnegative, level-constrained float64 KKT projection. Cast the candidate to
float32 and evaluate its actual displacement in float64 from that saved
float32 array. Accept only if absolute error is at most
`max(1e-6, 1e-6*target_norm)`.

Keep multiplier bounds [0,64], doubling bracket from 1, up to 64 bisections,
zero-dose exact copy and explicit capped infeasibility. Choose the closer
candidate by actual float32 displacement, with lower multiplier breaking
ties. Record float64 and float32 norms/errors separately, KKT residual,
actual cast constraint bound, scalar and iteration count. Copy inactive
frames exactly. Failed planned cells stay represented; no threshold relaxation.

Implement this as a separately hashed operation version. Preserve the old
executed ops source. Female replication may use the corrected operation
before its first forward, binding the new source and this specification.

## Fresh matched execution

New independent auxiliary run: the original 80 male clips / 40 speakers,
both original faces, four conditions `[orth_plus, random1, random2, random3]`,
640 intended score rows. Render all four conditions afresh, restore
benchmark=false after face detection, and log actual forward flags.
Use a fresh RAW-N SyncNet A cache shared by four conditions on each face.
Do not compare newly rendered random videos with an old-runtime cached
orth_plus video. Cached z/input provenance is explicit and immutable.

Reuse original PCM, frontend, ROI policy, weights, common prefix, all
full/valid/guard20/guard15 domains, raw/unit geometry, k=3 and column18.
Save all z/A/V, videos, complete cross/zero/lag matrices and support hashes.
Bind original protocol/row plan/recovery/source-input/CAL seals and every
inherited z/mask/direction array hash, new execution source and this spec.

## Independent inference

Reconstruct all cached matrices independently and check every active frame's
actual float32 dose and every inactive frame's exact copy before inference.
Preserve the full 80/40 intended denominator and show all failures.

The estimand is the original prespecified secondary contrast:
orth_plus minus the mean of three random arm metrics within each clip.
Average the two paired clip effects within speaker, then weight 40 speakers
equally. Use 20,000 PCG64 bootstrap draws, seed20260930, 99% intervals for
C/C_anchor, B/D/search and unit geometry. Report both faces separately.
Reuse the original 100 shared full-A permutation scheme for sync_excess.
No new primary threshold, equivalence margin or mediation percentage.

If this contrast generalizes, it supports specificity of the original
source-direction response relative to these three fixed equal-dose controls.
It does not establish that the direction is unique to TTS or to all TFG models.
