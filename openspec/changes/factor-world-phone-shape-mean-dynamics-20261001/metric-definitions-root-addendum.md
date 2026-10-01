# Prospective metric definitions for the phone shape factorial

This addendum resolves the six-metric shorthand in
`eval-design-root-freeze.md` (SHA256
`e17d257e43a78af6d7a291641ade6825b61c430ba51b8052383cab54dbea31fb`).
Root and the independent consumer reviewer resolved it before any shape
factorial EVAL waveform, GPU output, score array, or effect was produced or
read. The original design remains unchanged. No result informed this choice.

## Six metrics in the planned global family

For each frozen query support, aggregate the full lag-distance curve with
the original padding contract. Its indices run from 0 to 30 (VSHIFT=15).

1. `C_fixed_anchor`: `B_median - curve[18]`, the confidence at the CAL-frozen
   k=3 / official offset=-3 anchor. This is the single `C` endpoint in the
   3,456 global endpoint count.
2. `B_median`: median of the aggregated 31-column lag curve.
3. `D_min`: minimum of that curve.
4. `D0`: `curve[15]`, the distance at zero lag. This is distinct from the
   CAL-frozen anchor distance `curve[18]`.
5. `search_gain`: `curve[15] - D_min`.
6. `optimal_offset_frames`: `15 - argmin(curve)`, with the original first
   minimum rule on a tie.

All nine frozen contrasts, directions, geometries, supports, fixed RAW/RT
audio domains, observed/time-excess estimands, shared permutations and
20,000 speaker resamples remain. The planned joint family is still
3,456 global plus 8,320 local = 11,776 endpoints. A short empty guard20
endpoint remains in the planned family with its incomplete mask.

## Retained descriptive diagnostics

Retain `C_best = B_median - D_min` and
`D_fixed_anchor = curve[18]` for every cell/support, including their
full-audio time nulls. They are descriptive diagnostics outside the planned
11,776 simultaneous family. Do not label an unadjusted interval on these
extra diagnostics as a family-confirmed result. Search and offset retain
their diagnostic interpretation; a larger search gain is not by itself
better synchronization.

The fixed anchor is used for the family C because the mechanism question
concerns alignment at the independently calibrated lag. Retaining C_best
allows inspection of best-lag behavior without silently enlarging or
selecting the family after observing effects.
