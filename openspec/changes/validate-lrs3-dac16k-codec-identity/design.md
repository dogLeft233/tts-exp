## Context

See `proposal.md` for motivation and `specs/lrs3-dac16k-codec-identity/spec.md` for the behavioral contract.

The completed LRS3 direct-resynthesis experiment provides the comparison cohort and current codec control:

- parent summary: `runs/lrs3_wavlm_hifigan_direct_20260826/summary.json`;
- parent summary SHA-256: `96a29d7355ca3916ef2f60935147d9f20a123a8a7aefa12db130bbcf07eb08d5`;
- source manifest SHA-256: `34efb614d72e183727d87d40a05807d4c6ccda954cadc14f76bc8cbd01ee255f`;
- ordered cohort: 50 records, 38 source groups;
- ordered sample-ID SHA-256: `8e47514fdf877e9a57533e8719bd16fab1b96a99d8cb47cfa007c96689326b94`;
- current reconstruction: WavLM-Large layer 6 plus the prematched bshall/knn-vc HiFi-GAN, both frozen.

The parent experiment found a strict natural-audio replacement penalty of approximately `-0.231` Sync-C benefit and `-0.232` Sync-D benefit. The narrower AISHELL-1 2-by-2 identity control also found crossed audio/video preference even though reconstructed and natural WavLM representations were similar. These results make codec-domain mismatch a plausible contributor, but they do not test an acoustic codec designed for waveform reconstruction.

## Goals / Non-Goals

**Goals:**

- Determine whether a native-16-kHz acoustic neural codec reduces the current reconstruction-induced replacement penalty.
- Determine whether DAC reconstruction is practically non-inferior to untouched natural audio when a DAC-driven video is replaced with untouched natural PCM.
- Separate audio fidelity, candidate self-consistency, fixed-video audio compatibility, and strict replacement substitutability.
- Produce a terminal decision before any TTS or alignment model is introduced.

**Non-Goals:**

- Demonstrating replacement gain over the natural baseline; an identity codec is expected to approach, not improve upon, that baseline.
- Testing TTS, MFA-linear, hard-DTW, Soft-DTW, residual injection, masking, or a learned aligner.
- Training or fine-tuning DAC, WavLM, HiFi-GAN, Wav2Lip, or SyncNet.
- Searching across codecs, bitrates, codebook counts, checkpoint versions, normalization policies, output shifts, or post-filters.
- Claiming that a negative DAC result rules out every high-fidelity codec.
- Accessing sealed validation/test media or making population-wide LRS3 generalization claims.

## Decisions

### 1. Use native-16-kHz DAC as the single new codec

The new `D` arm uses the official Descript Audio Codec release tag `0.0.5`, source commit `408235a9dcd2983684c87615a1bc2a8954f6eb47`, model type `16khz`, model bitrate `8kbps`, and all checkpoint-provided quantizers (`n_quantizers=None`). The official release asset is:

```text
https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth
```

Stage 00 downloads or locates the checkpoint without opening cohort media, records its byte size and SHA-256, and freezes that identity in the protocol manifest. Every later stage rejects a different source revision, checkpoint hash, model configuration, or dependency lock.

**Why:** DAC is an acoustic residual-vector-quantized codec with a native 16-kHz checkpoint. It avoids the 16-to-24/44.1-kHz resampling confound and preserves more waveform detail than a semantic WavLM-L6 bottleneck is designed to retain.

**Alternative considered:** DAC 44.1-kHz 16-kbps. Rejected for the first experiment because upsample/downsample behavior would become a second independent variable.

**Alternative considered:** EnCodec 24-kHz. Rejected for the same sample-rate confound and because this first protocol permits only one new codec.

### 2. Bypass the DAC CLI normalization path

Input is decoded as the exact 16-kHz mono PCM16 natural waveform and converted deterministically to float in `[-1, 1)`. The runner calls the pinned DAC model directly:

```text
encoded/decoded = DAC.forward(
    natural_float,
    sample_rate=16000,
    n_quantizers=None,
)
```

The model may right-pad internally to its hop length, but the decoded waveform is cropped only to the original sample count. The runner MUST NOT invoke the convenience CLI's `normalize(-16)` behavior. It MUST NOT apply loudness normalization, gain matching, temporal alignment, delay compensation, denoising, filtering, resampling, phase correction, silence trimming, interpolation, or sample-count padding after decode. It canonicalizes the exact-length float output once to 16-kHz mono PCM16 using the repository's audited conversion policy.

**Why:** Post-hoc normalization or alignment could make DAC appear more substitutable while changing the hypothesis from codec reconstruction to codec plus corrective processing.

### 3. Freeze the existing 50-record cohort

Stage 00 validates the complete parent summary hash, source-manifest hash, ordered 50 IDs, 38 source groups, per-record natural audio, face video, WavLM reconstruction, sample counts, and file hashes. Selection follows only the frozen parent order. No record may be removed or substituted based on DAC quality, face detection, render success, or SyncNet score.

This is a fit-only diagnostic cohort inherited from the parent experiment. The protocol does not open any additional LRS3 records and does not access sealed validation/test data.

**Why:** Reusing the cohort makes the new result directly comparable to the known WavLM/HiFi-GAN penalty while avoiding outcome-dependent resampling.

### 4. Use three audio arms and a complete 3-by-3 matrix

For every record:

```text
A_N = untouched natural PCM
A_W = bound historical WavLM-L6/HiFi-GAN reconstruction PCM
A_D = new DAC-16-kHz reconstruction PCM

V_N = frozen Wav2Lip video driven by A_N
V_W = frozen Wav2Lip video driven by A_W
V_D = frozen Wav2Lip video driven by A_D
```

The experiment freshly renders all three videos in one run with the same frozen Wav2Lip checkpoint, invocation, face video, and face geometry. Face detections or fallback boxes are computed once per source record and hash-bound, then reused across the three driver arms. It then scores all nine cells:

```text
V_N/A_N  V_N/A_W  V_N/A_D
V_W/A_N  V_W/A_W  V_W/A_D
V_D/A_N  V_D/A_W  V_D/A_D
```

Each mux copies the video stream, uses PCM s16le audio, and verifies decoded PCM byte equality with the selected source audio before official file-level SyncNet V2 scoring. Every record and all 450 cells are required for a scientific conclusion.

**Why:** The full matrix distinguishes codec self-consistency from actual substitutability and tests whether either reconstructed domain is preferred by videos driven from that domain.

### 5. Audit fidelity without correcting the waveform

For both reconstructed arms relative to `A_N`, the experiment computes per-record metrics on the unshifted exact sample grid:

- Wav2Lip 80-bin mel mean absolute error using the exact frozen Wav2Lip preprocessing;
- multi-resolution log-STFT distance with frozen window/hop/FFT settings;
- SI-SDR with no lag search or scale-estimation preprocessing beyond the metric definition;
- raw waveform correlation at zero lag;
- RMS and peak ratio;
- F0 voiced-frame error and voiced/unvoiced disagreement using one pinned extractor;
- short-time energy-envelope correlation;
- zero-lag onset and phone-boundary-window errors where existing metadata permits;
- clipping, non-finite values, DC offset, sample count, channel count, rate, and sample width.

The mechanistic fidelity contrast is:

```text
mel_improvement = mel_L1(A_N, A_W) - mel_L1(A_N, A_D)
```

Positive values mean DAC is closer to natural in the exact mel domain consumed by Wav2Lip. All metrics are reported; none may be used to shift, select, or repair output.

### 6. Separate codec improvement from identity compatibility

The main replacement benefits are defined so positive always means better:

```text
DAC_over_W_C = SyncC(V_D, A_N) - SyncC(V_W, A_N)
DAC_over_W_D = SyncD(V_W, A_N) - SyncD(V_D, A_N)

DAC_identity_C = SyncC(V_D, A_N) - SyncC(V_N, A_N)
DAC_identity_D = SyncD(V_N, A_N) - SyncD(V_D, A_N)
```

The experiment uses 10,000 source-group cluster-bootstrap draws with seed `20260904`, sampling all rows of a selected source group together. It reports paired means, two-sided percentile 95% confidence intervals, record wins, group summaries, and all per-record values.

Decision A, `DAC_REDUCES_CODEC_REPLACEMENT_PENALTY`, requires all three lower confidence bounds to be strictly greater than zero:

```text
DAC_over_W_C
DAC_over_W_D
mel_improvement
```

Decision B, `DAC_IDENTITY_COMPATIBLE`, additionally requires both DAC identity lower bounds to be strictly greater than the pre-registered non-inferiority margin `-0.10` SyncNet units:

```text
lower_CI(DAC_identity_C) > -0.10
lower_CI(DAC_identity_D) > -0.10
```

The `0.10` margin is an engineering identity tolerance, not a claim of positive gain. It is less than half the approximately `0.23` historical WavLM replacement penalty and is frozen before DAC outcomes are observed.

Possible terminal scientific decisions are:

```text
DAC_CODEC_IDENTITY_COMPATIBLE
DAC_CODEC_PARTIAL_IMPROVEMENT
DAC_CODEC_HYPOTHESIS_NOT_SUPPORTED
```

An incomplete or invalid matrix produces engineering `BLOCKED`, never a scientific negative result.

### 7. Report diagonal preference without mistaking it for success

For each reconstructed domain, report the own-audio preference on its generated video:

```text
W_own_preference_C = SyncC(V_W, A_W) - SyncC(V_W, A_N)
W_own_preference_D = SyncD(V_W, A_N) - SyncD(V_W, A_W)

D_own_preference_C = SyncC(V_D, A_D) - SyncC(V_D, A_N)
D_own_preference_D = SyncD(V_D, A_N) - SyncD(V_D, A_D)
```

Large positive values indicate codec-specific audio/video co-adaptation. High `V_D/A_D` diagonal scores MUST NOT count as identity compatibility or replacement gain. Cross-codec cells are descriptive and help determine whether `W` and `D` form separate compatibility domains.

### 8. Keep future modeling sealed

This experiment produces no TTS-conditioned audio and trains no model. Only `DAC_CODEC_IDENTITY_COMPATIBLE` may set `future_tts_alignment_experiment_eligible=true`. Eligibility does not execute or scientifically promote MFA-linear, DTW, Soft-DTW, residual modeling, or any TTS-transfer claim; each requires a new OpenSpec change.

If the result is `DAC_CODEC_PARTIAL_IMPROVEMENT`, a new codec-only protocol may test a higher-rate or continuous acoustic representation. If the hypothesis is not supported, no codec may be substituted post hoc in this run.

## Stage Layout

Use the isolated run root:

```text
runs/lrs3_dac16k_codec_identity_20260904/
  00_protocol/
  01_audio/
  02_fidelity/
  03_videos/
  04_matrix/
  05_analysis/
  06_final/
```

Every stage validates all upstream hashes and writes append-only manifests plus failure and media-access ledgers. Per-cell work uses temporary outputs followed by atomic rename. Resume accepts only cells whose complete provenance and output hashes remain valid.

## Risks / Trade-offs

- **[DAC 16-kHz 8-kbps may still be too lossy]** -> Interpret a negative result only for this exact checkpoint and configuration; do not generalize to all codecs.
- **[A more faithful codec can still change Wav2Lip-sensitive details]** -> Make strict `V_D/A_N` replacement the endpoint rather than relying on perceptual or feature metrics.
- **[The historical cohort contains face-detection fallback cases]** -> Freeze one face geometry per record across all three arms, report fallback membership, and require a complete matrix; do not remove difficult records.
- **[Natural and WavLM controls could drift if reused]** -> Freshly render `V_N`, `V_W`, and `V_D` in the same stage and bind all current model/tool hashes.
- **[Multiple matrix contrasts invite post-hoc interpretation]** -> Freeze the three co-primary improvement endpoints and two non-inferiority endpoints; label all other cells descriptive.
- **[SyncNet variation may obscure practical equivalence]** -> Use paired source-group bootstrap and a pre-registered non-inferiority margin rather than interpreting failure to reject zero as equivalence.
- **[Full 3-by-3 scoring costs 450 SyncNet cells]** -> Load frozen models once, cache one video per driver, use hash-validated resume, and do no training or TTS generation.

## Migration Plan

1. Implement the isolated experiment package and deterministic media/fidelity/statistics tests.
2. Acquire and hash-lock the DAC source, dependency environment, and 16-kHz checkpoint before reading cohort media.
3. Build and independently validate Stage 00 against the immutable 50-record parent.
4. Generate 50 DAC reconstructions, canonicalize once, and validate exact-length/QC/provenance invariants.
5. Compute the frozen fidelity panel without output correction or record selection.
6. Render all 150 frozen-Wav2Lip videos with shared per-record face geometry.
7. Strict-mux and score all 450 matrix cells, requiring decoded PCM and video-stream identity.
8. Finalize the cluster-bootstrap decisions exactly once and write the terminal artifact.
9. Preserve all parent artifacts unchanged. Rollback removes only incomplete temporary files from the new run root; completed stage evidence is never overwritten.
