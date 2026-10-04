# Shared recipient transition head V2 — preparation contract

**State:** preparation only. This is a separate future method proposal. It is not part of the frozen six-slot V5 bank, has no selected coefficients, and has not been executed, fit, or evaluated. Development and confirmation results were not used to choose settings.

## Purpose and boundary

The candidate applies a small learned frequency-band mask whose time gates are computed from the same utterance being enhanced. Its aim is to alter recipient-local spectral transitions without importing a donor's spectrum or timing. The transform receives the original canonical mono PCM and an externally sealed RMS-baseline waveform for that exact source. It does not decode, resample, time-warp, align, or select a speaker/model-specific transform.

The implementation is [shared_recipient_transition_head_v2.py](../../audio_enhancement/shared_recipient_transition_head_v2.py). Its constructor requires every signal-processing, gating, projection, RMS, peak, and clock parameter. There are no candidate settings in this contract. V1 is retained unchanged at `scripts/audio_enhancement/shared_recipient_transition_head_v1.py` (29,494 bytes; SHA-256 `25098dfcdcea733968010a132d24ff120dc300d75b7c18f0004edfffea127abd`).

## Parameterization

The module forms four disjoint one-sided frequency-band maps from five caller-supplied edges spanning 0 Hz through Nyquist. Each map has one explicitly selected reference band. Three temporal components use those same maps: a static component, a positive relative spectral-flux gate, and a negative relative spectral-flux gate.

There are nine stored trainable scalars: three reference-coded band contrasts for each of the three temporal components. The omitted reference-band coefficient is fixed at zero. All four configured bands must contain at least one STFT bin. This parameter count is not a claim of nine independent effective directions: the temporal design matrix `[1, positive_gate, negative_gate]`, band support after smoothing, and frequency-common-mode choice determine the actual design rank on each fit utterance and on the fit corpus. Report that rank before fitting; do not assume it from the parameter tensor shape. The band edges, reference band, flux normalization and floors, active-transition rule, per-utterance gate quantiles and temperature, gate/mask smoothing, edge padding, frequency-common-mode policy, smooth mask bound, RMS target/epsilon, optional uniform peak ceiling, baseline tolerance, and video-frame clock are explicit constructor inputs. They remain unselected.

The two transition gates use raw linear magnitudes from the original PCM's configured STFT. For adjacent frames `t-1,t`, they compute `positive[t] = sum_f relu(|X[f,t]| - |X[f,t-1]|) / max(sum_f |X[f,t-1]|, denominator_floor) / flux_scale`, and the matching rectified negative difference. Quantiles are computed separately over valid transitions only; a sigmoid with the explicit temperature and declared smoothing produces each gate. A transition is valid only when both endpoint frames exceed the explicit activity floor and the previous-frame magnitude sum exceeds the denominator floor. The first frame is always invalid. Invalid flux entries use finite zero placeholders paired with `valid_transition_mask=false`; they are excluded from thresholds and gates, and smoothing is masked back to that support.

The signed difference also contains local magnitude-envelope information: `positive - negative` equals the adjacent total-magnitude change divided by the floored previous total and the flux scale. Therefore these gates do not isolate articulation from local amplitude. They are audio-derived controls, not VAD, phoneme labels, ASR timing, or annotations. A future native comparison must retain physical frame-power and envelope diagnostics alongside any synchrony result.

The four bands are combined into an F-by-T log-magnitude mask. Frequency and time smoothing are explicit. The caller chooses either `retain_reference_coded` or `remove_per_frame` handling of frequency-common content. The latter subtracts each frame's frequency mean before the bound. A global smooth radial tanh projection then keeps the maximum absolute log-magnitude change below the supplied dB bound while applying one scale to the whole mask. The projection does not independently clip frequency bins or transition cells.

## Baseline, phase, and clock

For original complex STFT `X` and projected amplitude mask `M_db`, the module constructs `ΔX = X * (10 ** (M_db / 20) - 1)`, computes `Δx = ISTFT(ΔX, length=N)`, and forms the unnormalized candidate as `x + Δx`. This uses the original STFT phase when constructing the masked coefficients. It does **not** claim that reanalysis of the reconstructed waveform has the original phase; that residual is an explicit future diagnostic.

Both `x + Δx` and `x` pass through the same declared global RMS and optional uniform peak-cap operation. The returned waveform is the supplied canonical RMS baseline plus the difference between those two normalized waveforms. At zero learned contrasts, the STFT residual is exactly zero and the added difference is exactly zero, so the output samples equal the supplied baseline samples while the residual calculation remains in the differentiable graph. The caller must bind the supplied baseline to the same source PCM, sample count, target RMS, and peak policy. The module checks shape, dtype, finiteness, and the caller-supplied amplitude tolerance against its own Torch normalization; it does not calculate or certify file lineage hashes.

No audio samples are inserted, removed, or shifted. The output length is checked against the input length. This revision requires centered STFT/ISTFT with its explicitly supplied padding mode because a Hann window does not support the uncentered synthesis path used here. STFT centers and 40 ms video-frame centers are returned in original source-sample coordinates; any incomplete trailing 40 ms interval is counted explicitly. Scoring continues to use the unchanged original PCM and existing query clock.

Diagnostics return the original spectrum, constructed residual spectrum, raw/smoothed/policy/projected masks, raw signed fluxes, active-frame and valid-transition masks, both gates, band-bin counts, projection scale, RMS/peak values, and source/candidate/baseline/output physical 40 ms frame-power arrays with validity masks. Unsupported or zero-power ratios use finite zero placeholders plus a false validity mask. Frequency-mean removal does not guarantee equal physical frame power. Because the canonical baseline anchor adds the difference between two Torch-normalized waveforms to the supplied canonical samples, final RMS and peak may differ slightly from the internal normalized candidate. The returned final-output RMS and peak must be checked by a future runtime guard; internal normalization alone is not an acceptance check.

## Runtime and frontend integration

The module adds no dependency beyond Torch and has no import-time model or data loading. It is a waveform transform, not a frontend adapter. The exact integration references are:

| Role | Pinned source | SHA-256 |
|---|---|---|
| Existing full-context differentiable Wav2Lip fit runtime | `scripts/audio_enhancement/wav2lip_fullcontext_fit_runtime_v4_attempt01.py` | `dd751132bd3d8dec5abce288c2f765df78d9b7ef9150cdfb0ccff52ba72ee458` |
| Existing Wav2Lip Torch mel mirror and parity helper | `scripts/audio_enhancement/wav2lip_frontend.py` | `3480087daecf8dc295f4a3b20703f393bada2690e1f5ce7839ba1dc4fbdec8ef` |
| Existing NumPy RMS baseline implementation | `scripts/audio_enhancement/baselines.py` | `7d161970381cc7e6770a4a53063f91b2606c05490945b2b86ab41c7de4d0a3e2` |
| Existing source-clock/phase diagnostic helper | `scripts/audio_enhancement/phase_clock_diagnostics_v1.py` | `25ed6f204dea7e5c149f7c897dfccfebe972ad0ef30eda1a2b1dbdc59ae216c5` |
| Prior unfrozen local-transition scaffold, reference only | `scripts/audio_enhancement/v5_local_transition_transform_scaffold_v2.py` | `cf86b95b65b2c4a423082e0f717713983915cdc0674ee7ddb4fba2534e9906e0` |
| V1 preparation source, retained unchanged | `scripts/audio_enhancement/shared_recipient_transition_head_v1.py` | `25098dfcdcea733968010a132d24ff120dc300d75b7c18f0004edfffea127abd` |

The Wav2Lip fit path has an existing differentiable Torch mel mirror, but it must be requalified against the pinned official preprocessing in the exact execution environment before this head is trained. The known combined runtime is CPython 3.10.12 with Torch 2.5.1+cu124 in the Wav2Lip environment; a future plan must bind its actual interpreter path, model/checkpoint/source closure, and deterministic numeric controls.

The MuseTalk generator can render an enhanced waveform, and its Whisper frontend has an observed Transformers 4.39.2 runtime. That does not establish differentiability through the production feature extractor. A Torch Whisper-feature mirror requires a separate source-pinned numerical parity qualification before it can provide gradients. Ditto uses an ONNX Runtime CPU frontend; no autograd path through that ONNX graph is claimed. MuseTalk and Ditto remain essential native evaluation frontends even if Wav2Lip supplies a qualified gradient path.

No frontend-input parity, native video parity, score effect, transfer effect, or content-quality result is claimed for this source. A future evaluation must use the unchanged natural source PCM for SyncNet scoring, compare against the exactly bound RMS baseline, preserve all planned model/sample/seed denominators, and retain failures. A transform result alone is not a synchronization result.

## Future qualification gates

Before fitting, a separate immutable plan must provide every constructor argument and pin the exact module/runtime/input baseline. Engineering qualification must establish:

1. zero-parameter bitwise sample equality with the bound RMS baseline, exact output length, finite output, and a finite nonzero parameter gradient on non-silent fit audio; separately record final RMS/peak deltas and enforce only tolerances frozen in a future qualification config;
2. source STFT/reconstruction behavior and mask-bound/common-mode invariants under the pinned Torch version;
3. frontend parity for every gradient path actually used; Wav2Lip's existing mirror is the only currently available Torch path, while Whisper parity remains a prerequisite for Muse gradients and Ditto remains evaluator-only;
4. original-PCM score, lag curve, source-clock mapping, and 40 ms physical-power diagnostics on the frozen fit corpus before any development evaluation;
5. native rendered development results across Wav2Lip, MuseTalk, and Ditto with original PCM and full planned denominators. Native outcomes, not proxy loss alone, determine usefulness.

The future engineering qualification must also verify that positive and negative gate arrays differ on declared nonuniform probes, retain the first/invalid transition as unsupported, and calculate the actual temporal and band design ranks on fit data. These are diagnostics, not preconditions that can be assumed from the model definition.

The mask uses per-utterance transition statistics, but all learned band contrasts are shared. Any future speaker-folding, candidate bank, optimization budget, and acceptance rule must be specified in a separate reviewed plan; none is selected here.
