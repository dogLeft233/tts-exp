# Generator level response through mel interior and clipping coordinates

Status: informed follow-up frozen after the principal read the source-axis,
RMS four-cell, resampling and corrected secondary results. It does not replace
the earlier source-axis primary. The partition rule already exists in
tts_fixed_mel_partition_feasibility_20260927.py; the old run statistics were
not recovered. Execute it on the complete fresh natural80/40 cohort.

## Question and source evidence

Fixed utterance RMS changed the generator under unchanged RAW-N scoring:
primary raw guard20 G=+.192, speaker99% CI[+.111,+.276]; secondary face and
unit geometry retained positive responses. Determine which mel coordinates
carry that response: bins interior at both levels, or bins touching a
normalization bound. This is an intervention on generator inputs, separate
from SyncNet audio scaling and from a norm of the decoder latent.

Actual runtime Wav2Lip/audio.py SHA256
71d5e927cac3d0b7103e7671a43b82b41b6a56398a092bbfa2e96d3041249c4c;
hparams.py SHA256
4a4c31dc51ebd8140da18ac24841e582b2472389f7c65e7f0a6b4447fc0c71cf.
Bind both files, the actual librosa compatibility shim, versions, defaults,
weights, face image and original canonical input manifest in the new run.

The implemented frontend uses preemphasis .97, magnitude STFT/mel projection,
20log10 amplitude, minimum amplitude 1e-5, reference20dB, minimum level-100dB
and symmetric clipping to[-4,4]. FFT/window800, hop200 at16k, 80mel bands,
55..7600Hz. Librosa default Hann/centering/padding and mel defaults must be
bound from the actual environment rather than assumed from an older server.
Away from bounds, multiplying PCM by gain g predicts normalized offset
1.6log10(g). Because clipped values lose sub-bound information, a uniform
shift of the saved clipped mel is not an exact waveform-gain surrogate.

## Frozen inputs and exact partition

Use the original evaluation80 natural clips /40 speakers /two clips each,
primary image, exact original16k decoded float32 PCM, and TARGET_RMS
.0376838172675746. Normalize in float64 then cast waveform float32 once,
without clipping, as in the sealed source experiment. No selection by gain,
mask size, audio features or prior effect. Preserve every planned failure.

Produce official complete mel endpoints M0=frontend(RAW), M1=frontend(FIXED),
saved float32. Require equal shape, finite values and legal[-4,4] bounds.
On those exact saved arrays freeze U=(-4<M0<4)&(-4<M1<4), S=~U.
No epsilon, fitted threshold or float32 boundary reassignment.

| Cell | Interior U | Boundary S | Exact mel array |
|---|---|---|---|
| q00 | RAW | RAW | exact M0 |
| q10 | FIXED | RAW | where(U,M1,M0) |
| q01 | RAW | FIXED | where(U,M0,M1) |
| q11 | FIXED | FIXED | exact M1 |

All intermediate values are copied from actual legal endpoints. Save full
M0/M1/U/S, chunk arrays, gain metadata and all array/file hashes. Verify
bit-exact endpoints, elementwise partition copies and float64 difference
closure M1-M0=(q10-M0)+(q01-M0). Do not relabel U as exclusively speech or S
as exclusively silence. S includes lower/upper bounds and zero-change bins;
amplitude floor and normalized lower clamp are different operations.

CAL20 CPU QA precedes any EVAL neural forward. Report both sources' bin
counts, lower-lock/unlock, upper-crossings, inward-one-ULP counts, delta norms,
interior offset prediction error and exact closure. Save full input arrays.
No CAL effect tuning, bin filtering or exclusion based on those diagnostics.
Then preflight the complete EVAL mask/input plan before generating videos.

## Execution and scoring

Render all four cells fresh with the same official chunking/clock and frozen
primary image. Record actual deterministic algorithms/cudnn flags before
every encoder, decoder and SyncNet forward; benchmark=false after detection.
No mixing older videos or embeddings into the new four-cell contrasts.
Save every z, MP4, decoded frame count, SyncNet A/V embedding and full cross
distance matrix. q00/q11 form a fresh bridge to the original RAW/FIXED
endpoints; report pixel/embedding differences without asserting old flags.

All four videos are scored against one new common unchanged RAW-N PCM A
cache, official JPEG frontend/MFCC units, original k=+3/column18/offset=-3.
Use actual V/A row counts and common support across all four cells; preserve
raw/unit, full/valid, guard20 and guard15. No alignment optimization or
condition-specific lag replaces fixed anchor. Intended rows80*4=320.

## Statistics and interpretation

Freeze contrasts interior=q10-q00, boundary=q01-q00,
joint=q11-q00, interaction=q11-q10-q01+q00, and q10-q01.
Report C, C_anchor, B, D, corrected search_gain=D0-D, both geometries and
all four support domains. Clip contrast, then average two clips per speaker,
then equal40 speaker means; 20,000 PCG64 bootstrap draws seed20260930,
percentile99% intervals. Missing support retains intended denominators.

The fresh joint C/C_anchor positive lower99% bounds constitute a replication
of the level generator response. Partition contrasts are declared secondary
mechanism estimates; do not create an exclusive-winner or mediation-percent
claim. Retain the original100 full-A row permutations with identical pi
within each clip across all four cells; report observed minus mean null,
permutation provenance and complete-speaker intervals.

An independent consumer must reconstruct partition identities and every
score matrix without importing producer scoring/operations. Changes in mel
coordinates identify generator responses to the specified intervention;
SyncNet endpoints alone do not establish perceived lip-sync quality.
Independent native RAW/FIXED visual diagnostics remain a separate endpoint.
New outputs and control logs go under private /root/tts-research-system-
20261001/, with controls outside an initially empty output directory; seal
sources/inputs/planned rows before forward and all final artifacts afterward.
