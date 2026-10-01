# EVAL design: phone spectral mean, internal trajectory and grid-cycle control

Status: prespecified root design before any WORLD EVAL effect is read. The CAL
engineering producer, complete physical QA, new EVAL producer/GPU plan and
independent consumer source identities must be sealed before execution. This
document sets scientific coordinates and inference; it does not claim a run.

## Causal question

At fixed recipient phone clock, WORLD source parameters, per-frame SP row
power and sentence RMS, does replacing the exact phone's average log spectral
shape or its internal centered log-shape trajectory change generated lip
response? Does that response exceed the recipient-only SELF_GRID control?
This is a causal intervention on operational WORLD coordinates. A residual
trajectory is not a pure scalar smoothness operator, and SELF_GRID has two
interpolation passes rather than the donor mapping's one pass.

## Cohort and physical treatment

Use all 80 already frozen EVAL pairs, 40 speakers with two utterances each,
and both recipient directions. Preserve the original complete physical
manifest and phone mapping identities. No EVAL selection, discarded clip,
renormalized speaker roster or effect-dependent label/support change.

Apply the CAL protocol in `spec.md` to every exact matched token on its common
mapped recipient 5-ms frame domain. Produce q00 = recipient mean/residual,
q10 = donor mean/recipient residual, q01 = recipient mean/donor residual,
q11 = donor mean/residual, and separate SELF_GRID. All five conditions preserve
recipient F0/AP/UV, SP row power, sample rate/length, clock and fixed global RMS
0.0376838172675746. Unmatched/ambiguous/unmapped frames stay recipient-owned.
Use every matched label; inference labels remain the prior CAL-only 16 exact
labels with tone/aspiration distinctions intact. Report all descriptor and
mapping exposure rows, including failures and zero exposure.

CAL engineering QA must pass before EVAL physical execution. Check q00/q11
against saved RT/SP parameters and WAVs with the prospectively frozen
absolute-plus-relative numerical tolerance in the CAL spec. Save all fresh
EVAL waveforms/parameters, token/domain indices, source hashes and runtime
identities. Independent EVAL QA covers every 80 pair/160 direction/800 output,
checks recipient invariants and bridges, and precedes GPU release.

## Generation and scoring

Generate all five videos fresh in one pinned runtime on the original fixed
primary face: 160 contexts, 800 videos. Encode exactly two fixed scoring
audio domains per context: natural/raw recipient PCM `A_RAW`, and that
recipient's saved WORLD round-trip PCM `A_RT`. SELF_GRID changes the
generation input only. Five V cells by two fixed A domains gives 1,600 full
cross matrices, each retaining both raw and unit distance geometry and its
distance-to-zero fallback vector.

The original deterministic policy applies before every model forward:
matmul/cudnn TF32 false, cudnn benchmark false, cudnn deterministic true and
deterministic algorithms true, CUBLAS_WORKSPACE_CONFIG=:4096:8. Use a compact
atomic final count/hash-chain sidecar and independent five-flag gate. No
producer or GPU model execution is allowed inside the independent consumer.

For each context freeze the common contiguous query prefix from the minimum
of all five complete V clocks and both A clocks. Keep every full V row and A
row in stored matrices; common query supports never truncate the saved A
permutation population. Use FULL/VALID/guard15/guard20 exactly as the original
scoring contract. Keep an empty short-T guard20 endpoint incomplete with its
intended 80 clips/40 speakers. Do not rescue it with 39 speakers or guard15.
Scoring CAL lag anchor remains k=3 / curve column18. The -85ms offset applies
only to the separately documented visual phone endpoint.

Compare fresh q00/q11 PCM, decoded pixels, latent, V/A and scores against
original RT/SP sealed endpoints as explicit bridges. Numerical waveform
agreement is not a claim of video/embedding bit identity, and cached endpoints
cannot replace a fresh generator condition without exact provenance.

## Global endpoints and contrasts

Recompute C, B, D, D0, search = D0-Dmin, and offset from full matrices with
the original padding/aggregation contract. C/B positive differences and D
negative differences have their stated score direction; search and offset
remain separate diagnostics. Let Y00, Y10, Y01, Y11, YS be each score with
the same fixed scoring audio domain. Retain all nine contrasts:

| Contrast | Definition |
|---|---|
| mean | Y10-Y00 |
| trajectory | Y01-Y00 |
| joint | Y11-Y00 |
| grid cycle | YS-Y00 |
| mean minus trajectory | Y10-Y01 |
| mean beyond grid | Y10-YS |
| trajectory beyond grid | Y01-YS |
| joint beyond grid | Y11-YS |
| factorial interaction | Y11-Y10-Y01+Y00 |

The interaction is a response to coordinated factor treatments; it does not
identify a standalone decoder nonlinearity. Do not subtract parameter-vector
and score interactions, whose units differ.

Use 100 PCG64 full-A permutations per context and audio domain, shared across
all five V cells, both geometries, all supports, metrics and contrasts. RAW and
RT have separate explicit seed namespaces and their respective full A lengths:
RAW seed20260930+ordered_eval_index and RT seed20270930+ordered_eval_index;
do not claim identical permutation arrays if the populations differ. Record
ordered EVAL index, seed, shape and array hash. Time excess subtracts the same
contrast's permuted-score mean from its observed contrast; neither baseline
nor null uses a truncated audio population.

The planned global family is 9 contrasts x 2 directions x 2 geometries x
4 supports x 6 metrics x 2 fixed scoring audio domains x 2 estimands
(observed/time excess) = 3,456 endpoints.

## Local response

For each of the four treated generator conditions X in q10/q01/q11/SELF_GRID,
on fixed RAW scoring audio define G_i = d(V_q00[i], A_RAW[i+3]) -
d(V_X[i], A_RAW[i+3]), including the original zero-padding rule. Positive G
means reduced distance at the fixed anchor. It is not an additive decomposition
of median Sync-C and is not a separately randomized phone treatment.

Use the previous exact Wav2Lip starts and the union of actual mel columns used
by the five video frames. Compute each selected phone's whole/begin/middle/end
half-open phase coverage weight, exact repeated-token ownership, ambiguity and
other-phone support. Preserve the full window denominator, including zero
exposure clips/windows. Report conditional means separately without replacing
the intended denominator. All cell mel clocks must agree; no alternative
clock or boundary selection after effects. Keep fixed +/-40ms boundary/mixed
phone diagnostics and 50ms STFT overlap sensitivity descriptive.

Local observed weighted G and full-A time-excess G for 4 treatments x
2 directions x 2 geometries x 4 supports x 16 exact labels x 4 phase outcomes x
2 estimands gives 8,192 endpoints. Add 4 x 2 x 2 x 4 x 2 = 128 unweighted
whole-support G endpoints. Local family is 8,320 endpoints. Dose/response
correlations are descriptive; arm contrasts provide the causal evidence.

## Joint inference and reproducibility

The complete planned family is 3,456 global + 8,320 local = **11,776**.
Use one frozen sorted 40-speaker roster, two-clip speaker averaging, 20,000
PCG64 speaker resamples with seed20260930 and the same int16 resample indices
for every estimable endpoint. Compute one 99% bootstrap max-|t| simultaneous
critical value across all estimable members of this planned family. Preserve
the planned family mask and intended counts for unestimable endpoints;
intervals cover the estimable subset, and family completeness is reported.

Save compact reconstructable indices/counts, each endpoint's 40-speaker
vector, ordered endpoint keys, estimable mask, max-|t| draws and all array/file
hashes. No huge redundant draws-by-endpoints file. Point estimates, bootstrap
SE, simultaneous intervals, support/exposure and null population identities
must be independently recomputable from sealed arrays. Do not select a smaller
family or promote a descriptive endpoint after reading results.

Interpret results using fixed RAW and RT audio separately, bidirectional
recipient changes, all support boundaries, raw/unit geometry, null excess,
own-grid comparators and independent visual tracking. Do not label SyncNet
response as human perceptual or phonetic accuracy. Keep physical failures,
incomplete support and actual scientific results distinct.

## Storage and completion

Code/control are persistent; bounded RAM output with resource guards may
overlap the old-v5 verified E/B archive. Preserve all failed attempts and sealed
inputs. Complete scientific outputs, consumers and visual tracks must be copied
to persistent storage with per-member size/hash, source/destination equality,
readback and a durable receipt before the run is reported complete.
