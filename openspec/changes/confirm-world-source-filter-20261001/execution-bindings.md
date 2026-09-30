# WORLD execution bindings

This annex makes existing operations explicit before physical evaluation
forwards. It does not change the calibration producer's frozen waveform
algorithm or overwrite its source spec. Bind this annex's SHA alongside that
spec, the successful CAL producer/QA freeze, and the evaluation row plan.

## Energy donor clock

For a qualified matched phone, recipient envelope center t in [r0,r1)
queries donor time q=d0+(t-r0)*(d1-d0)/(r1-r0). The input envelopes use the
already fixed 801-sample Hann window and 80-sample center grid. Interpolate
the donor's measured full envelope grid linearly at q, holding endpoint
values at the audio edges. The window itself may cover a phone boundary.
Divide this value by the recipient FIXED-input envelope at t, flooring both
at 1e-5 for the ratio and clipping its log gain to +/-12 dB. The ratio is
exactly one at unmatched centers. Multiply the own RT envelope by this ratio
to obtain E's target; apply the previously specified eight-iteration routine.
No samplewise division between unequal N/T waveforms is permitted.

The current producer audit's `target_rms_envelope_active_frames` means target
RMS >=1e-5, not the source-axis speech mask. Report this definition. Separate
QA sidecars also report original FIXED-recipient -45 dBFS activity residuals,
global residual bias, centered residual shape, target/achieved envelope
power and local-gain saturation. These diagnostics do not modify the target,
iteration count or main contrast and cannot turn approximate matching into
exact matching.

## Original calibration anchor

Reuse k=3, column18, official offset=-3 from the original source-axis cal20
q10 cell: G(FIXED natural), A_RAW natural. C_anchor is the median of the
31-distance lag curve minus its distance at column18. Use the same common
recipient rows as C, including the frozen main guard20 definition.

The GPU plan binds:

- `run/calibration_20260930/calibration_summary.json`, raw SHA256
  `34876249bb3463417f903ea49e11da03306b1c0785a28cb6a4f97da92d59f22a`;
- that summary's `global_anchor` and its calibration sample/speaker IDs;
- `run/calibration_20260930/calibration_axes.npz`, raw SHA256
  `e71b669a67f435d951742cf67a9683e0385b77f5b97ab7a7408851389ade9244`;
- the original independent calibration verification artifact.

Paths above are relative to the existing remote portable-source research
root. Do not refit, round differently, or select an arm-specific WORLD lag.

## TextGrid, waveform and direction identity

Each intended row binds the exact sample/speaker ID and publisher text;
natural RAW WAV/decoded PCM hashes; original Qwen 24 kHz and canonical
16 kHz WAV/decoded PCM hashes and EOF-correction provenance; normalized LAB
text/hash; both TextGrid paths/hashes; sample rates/counts; and qualification
record. It also binds MFA version, executable hash, acoustic-model version
and hash, dictionary hash, configuration hash and parser source hash.
Record raw file hashes separately from embedded manifest fingerprints.

Donor and recipient must be the existing same-ID, same-speaker, same-text
N/T pair. Direction names state donor to recipient:

- `tts_to_natural`: donor T, recipient N; the prespecified positive EX-RT
  contrasts use this recipient's unchanged RAW N scoring audio;
- `natural_to_tts`: donor N, recipient T; the prespecified negative EX-RT
  contrasts use this recipient's unchanged RAW T scoring audio.

Save these explicit donor/recipient fields even on failed rows. A label
without its waveform identities cannot establish bilateral transfer.

## Forward runtime and cached bridges

Face detection can change PyTorch cuDNN flags. The new physical runner
explicitly restores deterministic algorithms and benchmark=false after
detection and records actual flags at model forwards. Keep the original
run's actual-state correction as separate immutable provenance.

A cached original RAW-FIXED bridge requires matching inputs, model,
frontend, ROI, execution flags and feature hashes. If the original run used
different effective flags, regenerate the bridge in this physical run.
Do not call the resulting runtime difference a waveform intervention.
