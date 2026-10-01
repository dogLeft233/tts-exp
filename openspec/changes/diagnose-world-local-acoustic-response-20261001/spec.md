# Local WORLD acoustic response by exact phone and model receptive field

Status: frozen before WORLD EVAL local-response extraction. The evaluation run
must first be sealed and pass its independent score audit. This protocol and
the analysis source are hash-bound in the resulting sidecar.

## Question and estimand

The primary question is where the already randomized WORLD arm changes the
generator/evaluator response on the recipient's time axis, and which acoustic
coordinates were changed at those same locations. This is a local response
analysis of the existing physical intervention. It does not add a sentence-
level gain manipulation or claim that any single phone was independently
treated.

For recipient context c, intervention arm X, score window i and audio geometry
g, define the fixed-anchor local generator response

`G[c,X,g,i] = D[g](V_RT[i], A_RAW[i+3]) - D[g](V_X[i], A_RAW[i+3])`.

Positive values mean the treated visual embedding is closer to the unchanged
RAW recipient audio embedding. `D_raw` and `D_unit` are the two stored SyncNet
distance matrices. Column 18 is fixed from calibration (k=3; official offset
−3); no per-phone lag is fitted. This is a per-window distance contrast, not
the global median-C contrast and not an additive decomposition of it.
The phone-local score clock always uses the scoring anchor `A_RAW[i+3]`.
The separate −85-ms CAL latency belongs only to MediaPipe visual endpoints and
must not shift these SyncNet score windows.

The intervention arms are the six primary physical arms RT, EX, SP, BOTH, F0,
and AP. EX_EM, SP_EM and E remain prespecified controls and are reported in a
separate stratum. EX transfers donor F0, voicing and AP together; SP transfers
normalized spectral shape at recipient SP power; BOTH combines them. F0 and AP
sub-arms update only jointly voiced frames. Their results are not an algebraic
partition of EX because the exposure domains and interactions differ.

## Frozen phone inventory

Use `cal_phone_inventory.json` exactly as sealed, with raw SHA-256
`62eeb63462c6e2d378482d1cdc7876d2e05666bac0f54c32d2a08f5e114db2c2` and
embedded inventory SHA-256
`54a51ff1a9af891271b0eaed797dc0bb7508a338ef7e38f922b8f9292fa40f5d`.
Selection used only the 20 CAL records and the frozen exact phone maps, bound
to the CAL physical-manifest SHA-256
`297d3c138fc2de75e86f32db43f7d42f94476e3275c9ef9b72235c1880af07cc` and
CAL phone-acoustic QC SHA-256
`5ff39cfd82817f56eb9cc348e2d8481432973a09db7fd9a41ed9a280d91301c5`.

The fixed rule is at least 10 distinct CAL utterances and 5 distinct CAL
speakers containing a matched token in either frozen transfer direction;
the two directions are deduplicated for utterance and speaker counts. Exact
`world-phone-identity-v1` labels retain tone, aspiration, length and other
diacritics. The resulting 16 labels are:

`a˥˩, i˥, i˥˩, i˧˥, j, n, o˥˩, o˧˥, t, ts, tɕ, w, ŋ, ɕ, ʂ, ʈʂ`.

All 16 labels are reported. No label, threshold or interval is selected using
EVAL scores, dose, visual output or support. A missing/unmatched phone has
zero exact-label exposure and remains in the fixed denominator.
That zero is a localization weight only; it is not a scientific null gate.
Unmatched WORLD phones can still affect output through synthesis phase,
cross-frame filtering, each arm's global RMS scalar, and overlapping STFT
support. EX_EM, SP_EM and E remain prespecified controls.

## Time axes and input receptive field

The EVAL recipient MFA grid and WORLD phone map are the only phone clocks.
Each SyncNet visual row i consumes five consecutive decoded video frames
`i..i+4`. Wav2Lip generated frame r consumes the stored mel columns
`mel_starts[r] .. mel_starts[r]+15`. Require the stored mel starts, latent,
decoded video, crop and SyncNet V counts to agree with the sealed producer
clock; do not infer a replacement clock from duration or resample.

For row i, define `M_i` as the **set union** of the 16-column slices used by
frames i through i+4. The primary receptive-field weight for exact matched
recipient phone token p is the fraction of unique columns in `M_i` whose
80-Hz mel-grid centers lie in p's half-open recipient interval. The center of
mel column m is m/80 seconds. Also report the 80 input slots before set
deduplication as a clock diagnostic; it does not replace the frozen set-union
weight. A column center exactly on a phone boundary belongs to the interval
starting at that boundary. The column-grid weight localizes the model input;
it is not a claim that the mel STFT has no overlapping waveform support.

The frozen Wav2Lip frontend uses an 800-sample Hann STFT window (50 ms at
16 kHz), 200-sample hop (12.5 ms), and centered constant padding. Therefore
each mel center has a 50-ms waveform analysis window; the primary weight stays
on the exact mel-column centers and is explicitly a generator mel-coordinate
support, not an exact waveform receptive field. Report a separate descriptive
STFT-support sensitivity that gives each column the fraction of its centered
50-ms interval overlapping the phone, with out-of-recording zero padding left
outside every phone. It does not replace or move the primary weight.

Mel chunk starts are reproduced from the official 80-Hz / 25-fps loop:
`start=int(frame_index*80/25)` while `start+16 <= mel_frames`; when the next
window would exceed the mel, append `mel_frames-16` once and stop. Repeated
final starts are deduplicated only when forming each visual row's set union.
The exact stored starts are checked against this rule, including the final
fallback window.

Within each token, predefine three equal normalized-time phases: early
`[0,1/3)`, middle `[1/3,2/3)`, and late `[2/3,1]`. Phase weights assign each
union column to its matched phone phase by its center time. A column without a
selected, matched exact label is assigned to the explicit other/unmatched
bucket. For every score window, selected-label weights therefore remain
bounded by one and are never renormalized over only the selected phones.

For each phone p, phase b, context, arm and support S, the primary
full-denominator contribution is

`L[c,X,p,b,S,g] = sum(i in S, w[i,p,b] * G[c,X,g,i]) / |S|`.

An absent phone contributes zero. Report the coverage `sum(w)/|S|`, distinct
matched token count, covered windows and contexts. Separately report the
conditional token mean `sum(w*G)/sum(w)` and its exposed clip/speaker count;
this is explicitly a variable-support descriptive estimand and does not
replace L or the intended 80 clips / 40 speakers.

Report the whole-support unweighted mean G for context, plus each support
`full`, `valid`, `guard20` and `guard15`, in raw and row-unit geometry. Use the
producer's fixed support arrays unchanged. Main local location is guard20;
the other domains remain visible sensitivity results. Preserve too-short and
missing rows as intended-denominator failures; never switch to guard15 to
rescue guard20.

## Matched acoustic dose on the same local clock

After local arm responses are computed, describe the actual per-frame acoustic
changes from the sealed phone map and `parameter_transfer_audit.npz`. Use each
recipient's 5-ms WORLD centers, not a time warp or score-selected alignment.
Map each mel-column center to the nearest WORLD center; exact ties choose the
earlier center. For each p/phase/window, aggregate a dose only from WORLD
frames inside that exact mapped recipient phone and its mel receptive field.
Keep unmatched columns, unavailable dose frames and transitions explicit.

The predeclared dose summaries are:

- **Spectral shape:** Hellinger distance between recipient and mapped
  normalized SP rows; report recipient versus mapped adjacent-frame mean
  absolute log-shape change within a phone, using only consecutive mapped
  frames. Retain total SP power as a separate invariant check; do not interpret
  normalized shape as energy.
- **F0 and voicing:** F0 difference in cents only where both recipient and
  donor are voiced; separately report donor/recipient voiced-state mismatch
  and 0→1 / 1→0 transitions. EX includes state mismatch; the F0 sub-arm does
  not.
- **Aperiodicity:** absolute AP difference by fixed bands 0–500, 500–1000,
  1000–2000, 2000–4000 and 4000–8000 Hz; report jointly voiced exposure
  separately from all mapped EX frames. AP sub-arm updates jointly voiced
  frames only.
- **Map boundaries:** report windows whose RF intersects a matched-phone
  boundary and the fraction of RF columns within the frozen ±40-ms MFA
  boundary tolerance. Report mixed-phone RFs explicitly; do not attribute a
  mixed window to one sole phone.

Dose/result plots and correlations are explanatory heterogeneity summaries.
The manipulated arm is the causal contrast; phone exposure is not separately
randomized. Correlation of G with a dose, or a high phone-weighted response,
does not establish an independent phone-level cause or mediation percentage.
Do not fit a regression to choose phones, bands, thresholds, lags or arms.

## Time-null and inference

For the same `i+3` anchor, recompute the local G contrast using the already
frozen 100 full-domain RAW-A permutations: NumPy PCG64 with seed
`20260930 + ordered_eval_index`, 100 successive `rng.permutation(nA)` calls,
one permutation shared by RT and X and reused for every phone and phase in
that context. Apply the same fixed window weights and support to each draw.
For local excess, subtract the mean of the 100 permuted weighted responses
from the observed response. Save permutation array hashes and valid query
counts. Do not shuffle V rows, restrict the A permutation to phone intervals,
or refit the anchor.

Equal-weight the two clips within each speaker, then equal-weight the 40
speakers for each recipient direction. Use one shared 20,000 × 40 PCG64
speaker bootstrap index matrix (seed 20260930) for every arm, support,
geometry, label, phase and observed/null-excess response. The predeclared
simultaneous 99% interval family includes all eight non-RT arms, both
recipient directions, raw/unit geometry, four supports, 16 labels, four
phone outcomes (whole label plus three phases), and both observed and
time-excess estimates. Compute a shared max-|t| critical value over this full
family; do not select families after inspecting effects. Conditional-token
means and acoustic-dose distributions are descriptive with their support
reported.

## Validation and release gate

Do not calculate or publish any EVAL local-response value until:

1. the WORLD run reports complete status for 160 contexts and 12,160 planned
   score rows, with no failed planned rows;
2. its evaluation plan, protocol, row plan, code seal, status and full
   `all_files_sha256.json` inventory pass exact-byte/hash verification;
3. the independent WORLD score consumer reports PASS and confirms full
   V×A raw/unit matrices, zero rows, support arrays, aliases and fixed anchor;
4. the physical EVAL manifest, 160 phone-map files, 160 transfer-audit NPZs,
   320 recipient/donor WORLD feature NPZs and their declared hashes validate;
5. every phone-map row is bound to the original recipient/donor TextGrid
   hashes and physical producer plan; every mel window and audio anchor is
   reproduced from stored arrays and row labels.

A failure blocks the summary and retains its exact intended 80-clip / 40-
speaker denominator. Preflight may report hashes, keys, shapes and counts but
must not compute or expose effects. The final report records every source
hash, input count, support count, missingness, output hash and analysis-source
hash. Keep local response results separate from whole-utterance C gates and
from visual phonetic-quality claims.
