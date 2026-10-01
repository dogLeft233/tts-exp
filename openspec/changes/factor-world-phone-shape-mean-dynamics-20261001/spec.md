# CAL protocol: separate phone mean spectral shape and within-phone dynamics

Status: CAL-only protocol, to be sealed before new CPU output. This work
produces WORLD parameter/waveform controls on the existing 20 calibration
pairs. It does not read evaluation audio, video, SyncNet outputs, or evaluation
effects. Any EVAL use requires a separate plan and root freeze after this CAL
preflight is reviewed.

## Question

The existing physical WORLD transfer changes normalized spectral shape using
the donor trajectory mapped onto the recipient's phone clock. This diagnostic
separates the mapped donor's per-phone mean log shape from its within-phone
log-shape residual, then crosses those coordinates in a two-by-two design.
It also measures a recipient-only resampling loop on the donor's 5 ms grid so
that grid-cycle smoothing can be seen directly.

This is a WORLD acoustic engineering diagnostic. CAL responses do not establish
any TFG or SyncNet effect, and they do not identify a separately randomized
phone treatment.

## Frozen inputs and cohort

- Use the sealed 20 CAL utterances, 10 speakers, and both existing directions:
  `tts_to_natural` (recipient N, donor T) and `natural_to_tts` (recipient T,
  donor N). Keep all intended records and direction rows; any failure remains
  visible and makes the CAL run incomplete.
- Bind the frozen WORLD CAL manifest/plan, the executed WORLD producer, the MFA
  execution annex and phone table, the root CAL waveform-QC freeze, and
  `cal_phone_inventory.json` by both raw-byte and embedded/payload identities
  where present.
- The pinned WORLD plan contains both CAL and EVAL condition rows. Verify its
  exact raw hash and the independently frozen expected payload hash, but do not
  deserialize or enumerate its combined condition list. Open only the
  calibration producer manifest as a dataset.
- Read each direction's exact saved phone map and transfer audit, both
  recipient/donor WORLD feature archives, and the saved RT/SP parameter and
  waveform references. Recompute the donor mapping with the frozen WORLD
  producer and require it to agree with the saved transfer audit before
  factorization.
- Do not inspect or enumerate the EVAL split. The input loader accepts only the
  CAL manifest, checks `split=calibration`, 20 unique records and the frozen
  source bindings, and has no EVAL path argument.

## Exact per-token coordinates

Use each exact matched recipient phone token separately. Repeated occurrences
of the same label are distinct tokens; never pool their means. The token's
frame domain is the common set of 5 ms recipient WORLD centers that lie in its
half-open recipient interval and have a valid saved/recomputed donor map.
Apply the original final-audio-endpoint closed-interval rule only when the
phone ends at that endpoint. If multiple tokens claim one center, mark it
ambiguous, exclude it from every token mean, leave it at the recipient shape in
all arms, and report it. Unmatched, unmapped, ambiguous, or zero-exposure frames
remain the recipient's original shape.

For each eligible frame, normalize recipient SP and mapped donor SP rows by
their frame sums. For token `p`, with `epsilon=1e-12`, compute

```text
lR[t,k] = log(max(shapeR[t,k], epsilon))
lD[t,k] = log(max(mappedShapeD[t,k], epsilon))
muR[p,k] = mean_t(lR[t,k])
muD[p,k] = mean_t(lD[t,k])
rhoR[t,k] = lR[t,k] - muR[p,k]
rhoD[t,k] = lD[t,k] - muD[p,k]
```

Means use exactly the token's common mapped-frame domain, independently for
each token. Reconstruct each mixed row with a numerically stable softmax across
frequency bins:

| Cell | Per-frame log coordinate | Meaning |
|---|---|---|
| `q00` | `muR + rhoR` | recipient mean, recipient dynamics |
| `q10` | `muD + rhoR` | donor mean, recipient dynamics |
| `q01` | `muR + rhoD` | recipient mean, donor dynamics |
| `q11` | `muD + rhoD` | donor mean, donor dynamics |

For every cell, multiply the normalized shape by the recipient's original SP
row power. Outside eligible token frames every cell uses the recipient's
original shape, including ambiguous mapped frames. The cells change SP shape
only: recipient F0, AP, voiced/unvoiced state, WORLD time centers, FFT grid,
sample rate, and sample count stay fixed.

Retain every matched label and token in the output. The existing frozen 16-label
inventory is tagged as the only prespecified label family for any later local
inference; additional matched labels remain explicit descriptive rows. This
CAL-only run does not freeze or compute EVAL inference.

## `SELF_GRID` resampling control

For the same matched token, use only the recipient's own log normalized-SP
trajectory. Interpolate it linearly with endpoint holding from recipient
centers to the donor's original 5 ms centers using normalized phone phase, then
interpolate those intermediate rows linearly with endpoint holding back to the
recipient centers. Apply only to the token's eligible common mapped recipient
centers; keep recipient shape elsewhere. Preserve recipient row power and all
other recipient parameters exactly.

This is a two-pass WORLD-grid cycle control. It is not the donor transfer, not a
proof that one-pass donor interpolation is artifact-free, and not a claim of a
pure or invertible interpolation operation. Keep it beside the four cells and
the original RT/SP outputs, not inside the two-by-two factorial.

## CAL production and checks

Save parameter arrays and float WAVs for all four cells and `SELF_GRID` for
every record/direction. Synthesize with the pinned WORLD 0.3.5 runtime and
frozen 5 ms settings. Right-trim only to the exact recipient sample count;
short synthesis is a retained engineering failure. Apply the fixed global RMS
`0.0376838172675746` after trimming, without clipping, time warping or padding.

The q00/q11 formula-to-endpoint check uses an elementwise mixed tolerance:
`abs(actual - endpoint) <= 1e-12 + 1e-10 * abs(endpoint)`. Report maximum
absolute and relative errors separately; relative error uses
`max(abs(endpoint), 1e-12)` as its denominator floor. This accommodates
floating-point softmax reconstruction without describing the formula as
bit-exact. For q00 and q11, construct the formula on eligible token frames and
the recipient shape everywhere else, then compare that full-grid array with
the normalized sealed RT/SP endpoint. Only after this full-grid bridge passes
may the producer store an exact copy of the sealed endpoint SP array. A mapped
ambiguous frame must have exactly the recipient shape in the original SP
endpoint, or the row fails; it cannot be exempted from the endpoint bridge.
Report the log-shape epsilon-floor added mass per row as the
sum of `max(1e-12 - normalized_shape, 0)`, including observed maxima and the
per-row upper bound `frequency_bin_count * 1e-12`. F0, AP, time, frequency, and
sample-clock copies are independently checked bit-for-bit.

The preflight and output manifest record raw and embedded input hashes, code
and spec hashes, ordered pair IDs, every loaded file's path/hash/size, per-token
frame indices and `muR`/`muD` arrays, ambiguous/unmapped counts, output file
hashes, package/runtime versions, synthesis tail counts, RMS, finite-value and
SP row-sum diagnostics. No output is called PASS unless all 20 pairs and 40
directions are complete and every required cell/control passes.

Required QA gates:

1. Recomputed phone mapping and every transfer-audit array agree with the
   frozen producer result; no matched frame is silently reassigned.
2. The q00/q11 formula agrees with its endpoint within the frozen mixed
   tolerance on the full WORLD grid, including the recipient-shape rows outside
   eligible phone tokens. Mapped ambiguous frames must be exactly recipient
   shape in the original SP endpoint (or there must be no mapped ambiguous
   frames). If either full-grid bridge fails, retain the engineering failure;
   do not copy or exempt the endpoint. After both checks pass, stored q00/q11
   SP arrays are exact copies of the existing RT/SP arrays; the softmax
   reconstruction itself is not called bit-exact. Both preserve the
   recipient's F0, AP, time and frequency arrays bit-for-bit.
3. `q00` and `q11` fresh waveforms are compared against the existing RT/SP
   float-WAVs and reported with exact identity, maximum absolute and relative
   RMS differences; any difference above the frozen `1e-6` absolute waveform
   tolerance fails the CAL engineering gate.
4. Every cell and `SELF_GRID` has finite nonnegative SP, preserves each
   recipient SP row sum within `1e-10` relative error (and reports the absolute
   maximum), has finite synthesized samples, exact recipient length and fixed
   global RMS within `1e-9` absolute error.
   Report the actual epsilon-floor added mass and its per-row
   `frequency_bin_count * 1e-12` upper bound on each recipient and mapped-donor
   normalized shape row.
5. Repeated fresh synthesis of `q00` and `q11` is bit-identical to its first
   synthesis. Record right-trim samples and peaks; clipping counts are
   descriptive and never repaired by clipping.
6. Report the per-token log-shape mean gap, recipient and donor residual
   dynamics, and within-token adjacent log-shape changes for every cell and
   `SELF_GRID`. Compute adjacent pairs only within one token and only for
   consecutive recipient frame indices. Summaries retain token and label
   coverage; they do not choose a subset or alter an EVAL family.

## Interpretation boundaries

Softmax-normalized residual coordinates are not a pure smoothness operator:
per-frame normalization can couple frequency bins. `rho` is a centered log
shape trajectory, not a standalone physical source. `SELF_GRID` includes two
interpolations and may smooth the recipient trajectory. Any later EVAL
interpretation must compare the sealed two-by-two cells and `SELF_GRID` under a
newly frozen protocol, preserve the original 20 CAL / 80 EVAL cohorts, and use
the existing fixed 16-label inferential family unless a new family is frozen
before EVAL artifacts are opened.
