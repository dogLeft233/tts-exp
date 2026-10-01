# Local WORLD response: overall endpoint and phase-dose addendum

Status: frozen before local-response EVAL extraction. This revision follows a
static contract review of the first consumer draft; no local EVAL response
values were opened. It retains the original run, cohort, phone inventory,
fixed anchor, support arrays, 100 full-A permutations, and 20,000-speaker
bootstrap.

## Overall fixed-anchor local response

For every arm, recipient direction, raw/unit geometry, and support, report the
unweighted mean of the fixed-anchor per-window response `G_i` over every row
in that support, including rows with zero selected-phone exposure. Report its
observed mean and mean over the same 100 full-A permutations, plus their
difference. Equal-weight the two clips within each speaker and then the 40
speakers. An empty support is retained as `INCOMPLETE_SUPPORT` against the
intended 80-clip / 40-speaker denominator.

Add the 256 overall-response endpoints (8 arms × 2 recipient directions × 2
geometries × 4 supports × 2 estimands: observed and time-excess) to the
original 16,384 phone-local endpoints. One shared 20,000 × 40 bootstrap and
one two-sided 99% studentized max-|t| critical value cover the complete
16,640-endpoint family. Missing endpoints remain in the declared family mask;
intervals are emitted only for estimable endpoints and the family is marked
incomplete when any intended endpoint lacks its full support.

## Phase-specific acoustic dose

The physical dose table has one aggregate for each recipient context, arm,
fixed support, exact CAL-frozen phone label, and `whole` / `early` / `middle` /
`late` phase. Use the same exact recipient phone interval, center-based mel
phase, and mel-column support multiplicities as the corresponding local score
weights. Map each included 80-Hz mel center to the nearest saved WORLD center,
with exact ties choosing the earlier frame. Preserve the mapped mel-column
indices, frame indices, and weights so every aggregate can be reconstructed.

Save compact per-context NPZ arrays for per-window/per-mel-column doses and
weights, plus their phone × phase × support summaries. The dose remains
descriptive: no phone-level randomization, regression-selected phones, or
mediation claim is introduced. All dose arrays keep unmapped or unvoiced
support explicit through masks, counts, and denominators. Report the distinct
recipient-token count with at least one exact phase-center column in each
support; keep the underlying phone/phase masks, support multiplicities, mel
centers, nearest WORLD-frame indices, and video-window column unions so these
counts and aggregates can be reconstructed from the indexed source artifacts.

## Clock identity

Every physical WORLD arm, including `EX`, must use the same stored mel-start
sequence as the fixed-recipient stream. Check this equality before deriving
any receptive-field weights. The producer already asserts this identity; the
independent local consumer must enforce the same invariant.
