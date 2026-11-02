# Local WORLD response: RF-boundary and STFT sensitivity addendum

Status: frozen before local-response score extraction. This addendum preserves
the original local estimand, cohort, anchor, supports, permutations, and
bootstrap. It makes the already specified boundary diagnostics executable and
adds an explicitly descriptive STFT-overlap sensitivity.

## RF and phone-boundary diagnostics

For every five-frame visual window, retain the exact set-union of its 16-column
mel chunks. Report, per intended recipient context and frozen support:

- the fraction of unique RF mel columns whose centers are within 40 ms of any
  start or end boundary in the exact matched-recipient phone map;
- whether any centered 50-ms STFT window in the RF crosses such a boundary;
- whether mel-column centers in the RF occupy more than one matched phone
  identity (a mixed-phone RF);
- RFs with no selected CAL-frozen phone identity at their mel centers; and
- the window and unique-column denominators for each quantity.

Boundary and mixed-phone summaries describe input context. They are not
eligibility gates, do not change phone weights, and do not imply that a phone
was independently randomized. Zero selected-phone exposure is not a null gate:
WORLD synthesis phase, cross-frame filtering, per-arm RMS normalization, and
overlapping STFT support can carry effects across an unmapped phone boundary.

## STFT-overlap sensitivity and multiplicity

For each of the 16 frozen labels, compute a separate `stft50` local response
using each RF mel column's fraction of its centered 50-ms interval that
overlaps the exact half-open recipient-phone interval. Keep this distinct from
the primary mel-center exposure. Report the fixed-denominator weighted
contribution, its full-domain permutation excess, exposure fraction and
per-context support. This sensitivity is descriptive only: report no
confidence interval, p-value, or confirmatory claim from it. The original
two-sided 99% studentized max-|t| family remains the four frozen mel-center
outcomes (`whole`, `early`, `middle`, and `late`), over arms, directions,
geometries, supports, labels, and observed/time-excess estimands: 16,384
planned endpoints when all intended rows are available.

No alternate anchor, phone inventory, recipient interval, mel start, support,
or analysis threshold is introduced.
