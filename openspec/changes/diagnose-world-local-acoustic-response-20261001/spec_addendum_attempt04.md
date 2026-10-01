# Local WORLD response: overall endpoint and exact-token dose addendum

Status: frozen before local-response EVAL extraction. This revision follows a
static contract review of the first consumer draft; no local EVAL response
values were opened. It retains the original run, cohort, phone inventory,
fixed anchor, support arrays, 100 full-A permutations, and 20,000-speaker
bootstrap.

This follow-up fixes a dose-only join ambiguity found during static review of
the consumer. The original addendum and attempt04 source remain immutable. No
score, support, permutation, endpoint, or inference rule changes.

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

Save the single shared speaker-index matrix, its count matrix, the estimable
endpoint mask, the fixed-order 16,640 × 40 speaker-value matrix (NaN for
unestimable endpoints), and the family max-|t| draws in one compact NPZ with a
SHA-256 reference. When some endpoints are unestimable, the max-|t| draws are
computed over the estimable endpoints only and the overall family remains
marked incomplete. Do not save a 20,000 × 16,640 bootstrap matrix.

## Phase-specific acoustic dose

The physical dose table has one aggregate for each recipient context, arm,
fixed support, exact CAL-frozen phone label, and `whole` / `early` / `middle` /
`late` phase. Use the same exact recipient phone interval, center-based mel
phase, and mel-column support multiplicities as the corresponding local score
weights. Map each included 80-Hz mel center to the nearest saved WORLD center,
with exact ties choosing the earlier frame. A mel column is eligible for a
dose only if the nearest WORLD frame has the same unique recipient
`phone_index` as the mel center and that WORLD frame is in the same phone
phase. Matching only the IPA label and phase is insufficient: repeated tokens
can share a label. For each phone/phase/window, aggregate a dose only from
these exact-token WORLD frames and the mel-receptive-field support. Preserve
the mapped mel-column indices, frame indices, and weights so every aggregate
can be reconstructed.

Keep same-token, cross-token, unowned/ambiguous, wrong-phase,
transfer-unmapped, and otherwise unmatched column-window weights explicit,
with the original exposed-weight denominator. Do not select phones or
contexts based on exposure or dose availability.

Save compact per-context NPZ arrays for per-window/per-mel-column doses and
weights, plus their phone × phase × support summaries. The dose remains
descriptive: no phone-level randomization, regression-selected phones, or
mediation claim is introduced. All dose arrays keep unmapped or unvoiced
support explicit through masks, counts, and denominators. Report the distinct
recipient-token count with at least one exact phase-center column in each
support; keep the underlying phone/phase masks, support multiplicities, mel
centers, nearest WORLD-frame indices, and video-window column unions so these
counts and aggregates can be reconstructed from the indexed source artifacts.
Also save mel-center-to-WORLD exact-token and exact-token-phase masks so each
mismatch category can be recomputed.

## Clock identity

Every physical WORLD arm, including `EX`, must use the same stored mel-start
sequence as the fixed-recipient stream. Check this equality before deriving
any receptive-field weights. The producer already asserts this identity; the
independent local consumer must enforce the same invariant.
