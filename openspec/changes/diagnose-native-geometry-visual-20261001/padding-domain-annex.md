# Native score padding-domain annex

Status: additive, frozen analysis clarification for the cached native temporal
null. It preserves the original native protocol and does not change the
declared support, cohort, lag, permutation or bootstrap rules.

## Why both cutoff views are retained

The cached producer scored each common query position with audio indices
bounded by `query_n = min(V, A)`, where `V` and `A` are the video and audio
embedding row counts. The saved full cross-distance matrix still retains the
complete audio-row domain. When `A > V`, a lag position can therefore be
outside the producer's scoring cutoff while still referring to a row present
in the complete audio matrix. Since out-of-domain lag positions use the
video-to-zero boundary value, the observed curve and its temporal null must
use the same cutoff for a matched comparison.

## Frozen paired analyses

1. **Legacy common-audio cutoff (primary cached-score reproduction).**
   Reconstruct the observed lag curve and each permuted null curve with the
   original `query_n` boundary. Each permutation still permutes the complete
   audio-row multiset. This view is checked against the producer's stored
   score matrix and is the reproduction of its original scoring geometry.
2. **Full-audio cutoff (additive padding sensitivity).** Reconstruct both the
   observed curve and each null curve with boundary `len(A)`. Keep the same
   original common video query prefix, all four saved support arrays, the
   complete audio-row permutation vectors, and every other input fixed. This
   view asks how the result changes when existing audio rows beyond the
   common query cutoff remain eligible as lag destinations.

Both views use the same utterances, faces, N/T sources, four audio/video
conditions, raw and row-unit distance geometries, four supports, metrics,
permutation vectors and speaker bootstrap draws. For each comparison, its
observed and null paths share one boundary cutoff. Report both cutoffs; do
not replace the primary legacy reproduction with the full-audio sensitivity
or choose between them based on effect direction.

## Frozen evidence bindings

- Parent native protocol: `openspec/changes/diagnose-native-geometry-visual-20261001/spec.md`, SHA256
  `b11c48d185beb30a3ae3f0036ca7ba02eb4d54ef24fdc13f5f96dfd3d71810bd`.
- Independent consumer: `scripts/experiments/portable_source_axis_20260930/independent_native_cached_null_20261001_attempt05.py`, SHA256
  `cd1ab039a7a0794adb10509c6b8da859755c02190cd955a97fb8a5664c8e6d5a`.
- Completed report: `runs/recovered_20260930/independent_validation/native_cached_null_attempt05/independent_native_cached_null.json`, SHA256
  `7b4c8e96155e103bb93a149c55bc97d2bdeb7fa99986bd8d5a36467a38ddd8f9`.
- The report records the exact two matched-boundary rules under
  `padding_domain_protocol` and stores both summaries under
  `padding_cutoff_sensitivity`. The completed report is immutable; this
  annex records its interpretation and does not rewrite it.
