# Fit40 direct-D0 and GateDC source contract — attempt02

**Status:** source-only implementation prepared and statically reviewed. No runner, model, optimizer construction or step, GPU, Dev/Confirm data, network/API, pytest, or experiment was executed. The current candidate180 protocol remains under the parent task's control. Existing Adam120 sources, checkpoints, data, and prior manifests were preserved.

## Frozen source route

- Head: `scripts/audio_enhancement/shared_recipient_transition_head_v3_d0_gatedc_attempt01.py` — fresh copy of the nine-parameter V3 recipient-transition head with a GateDC forward mode.
- Objective/native bridge: `scripts/audio_enhancement/v3_v4_native_runtime_adapter_attempt04.py` — direct D0 and zero-lag unit-cosine objective; delegates unchanged rendering/scoring to adapter attempt03.
- Strict backward wrapper: `scripts/audio_enhancement/v3_v4_native_runtime_adapter_v9_attempt01.py` — fresh wrapper that pins the new head/objective and retains the strict deterministic qualification route.
- Fit runner: `scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_d0_direction_gatedc_attempt02.py` — fresh Fit40-only clone; retains the frozen Fit40 row order and Adam120 schedule, all-40 same-GPU zero-head gate, then a full-objective backward qualification requiring a finite, nonzero [3,3] gradient before constructing the V3-head Adam.

The source pins and old-file unchanged checks are in `source_contract_manifest_attempt02.json`. The fresh strict wrapper v9 and runner attempt02 passed `ast.parse`; this confirms syntax only and does not import or execute project code. The head and objective remain the unchanged pinned sources from attempt01.

## Objective

Maximize the sum of:

- `(D0_RMS - D0_head) / max(D0_RMS, 1e-8)`
- `(cos_head - cos_RMS) / max(1 - cos_RMS, 0.05)`

Each reward coefficient is `1.0`. The explicit Cbest nonworsening penalty is `relu((Cbest_RMS - Cbest_head) / max(abs(Cbest_RMS), 0.05)) ** 2` with coefficient `1.0`. Retained regularizers are waveform-relative-MSE to the canonical unbounded RMS baseline (`0.02`) and projected log-mask total variation in natural-log gain (`0.002`). B, Dmin, far-lag contrast, and Canchor are diagnostics only. The objective module rejects weight keys or coefficients that differ from this contract.

The runner uses the existing exact native visual/audio context and source-clock query mapping. Cosine is computed with the same cached natural-audio center embeddings for RMS and candidate outputs, while the visual embeddings and lag curve come from the corresponding rendered branch.

## Head and GateDC

The fresh head keeps the same `[3,3]` zero-initialized trainable parameter, 32 ms STFT window, 8 ms hop, 3 dB smooth projection, fixed-RMS/phase restoration, and original source clock. GateDC replaces each post-smoothed valid positive/negative gate with that gate's utterance-specific mean; invalid frames remain zero. The same static terms, mask smoothing/projection, RMS policy, and synthesis then apply. GateDC is scored at the same learned theta after the 120 Fit updates and has zero optimizer updates. The all-40 zero-head preflight requires both dynamic and GateDC branches to reproduce the RMS waveform bit-for-bit.

## Qualification and scope

The V2 loader creates an unused legacy optimizer while loading the frozen runtime; the runner discards that object before qualification and never steps it. Only the V3-head Adam optimizer is created, and only after all 40 same-CUDA zero-head comparisons pass and strict deterministic full-objective backward qualification passes with a finite, nonzero [3,3] gradient. The qualification still demonstrates the frozen native forward and required backward paths. An all-zero gradient is an engineering failure and cannot qualify. This source pin does not assert that runtime qualification has passed; no GPU work was performed here.

Native frontend feature audit remains a separate post-freeze diagnostic for Wav2Lip, MuseTalk/Whisper, and Ditto/HuBERT. The runner records each frontend separately, marks the audit not run and excluded from training, and makes no cross-frontend gradient claim. The current probe uses inference mode or CPU ONNX paths; it is not a differentiable three-frontend training objective.

## Future extension cohort rule

Keep the existing 20-clip Dev denominator and all nine GUARD20 support failures intact; do not replace or relabel any clip. For a separate future cohort, use only frozen sample/speaker metadata for screening. Do not open Confirm audio or scores. Require IDs disjoint from Fit20, current Dev10, and Confirm40 speakers. Eligible clips must be mono 16-kHz IEEE float32 original PCM with at least 64,000 samples (100 source 25-fps periods, 4.00 s). From metadata, take the first 10 canonical speaker IDs with at least two eligible new sample IDs, then exactly the first two canonical sample IDs per selected speaker. Freeze the complete inclusion/exclusion inventory before model rendering/scoring. Any rendered GUARD20 shortfall remains in the preselected denominator with no replacement. If a local eligible pool is unavailable, source-authorized inventory/download is the next action; it does not block this source preparation.
