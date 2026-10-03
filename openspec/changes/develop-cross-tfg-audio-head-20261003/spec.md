# Portable waveform enhancement for multiple TFG models

## Objective

For natural audio `x`, generate video `V_m(h(x), face, seed)` with model `m`, attach the original `x`, and improve its synchronization relative to `V_m(x, face, seed)` scored with the same original `x`:

`delta_C(m,x) = C(V_m(h(x)), x) - C(V_m(x), x)`.

The deliverable is a usable audio transform, fixed configurations/weights, model integration instructions, an original-audio mux path, and a reproducible multi-model evaluation. A failed candidate initiates diagnosis and another method iteration. Goal completion requires useful method evidence and delivery, not a launched job or a single successful sample.

## Evidence and scope

Report 25 is concluded for AISHELL-1 Chinese/cloud Qwen/Wav2Lip. A sentence-level RMS transform has a positive saved-feature fixed-natural-audio response in 80 clips/40 speakers: full-A/full common support primary raw Cbest +0.1969088674, 99% CI [0.1207077477,0.2750632986]; secondary +0.1188069582 [0.0570874549,0.1858611917]. This is a motivating baseline. Its old dataset has been repeatedly inspected and is a diagnostic/regression set for this goal, not a fresh confirmation set.

Local mu/rho interventions show recipient-relative spectral/temporal compatibility. They motivate constrained residual adjustments; direct donor transplant is not a ready enhancement recipe. The historical RMS target `0.0376838172675746` is a fixed positive-control configuration, not an optimum for every frontend.

## Product contract

1. Input/output: canonical mono 16 kHz float32 waveform, exactly equal sample count; no duration warp, insertions, deletions or sample-index retiming. Support original external sample rates through a declared shared resampling policy.
2. Preserve original natural PCM separately. Enhanced PCM is used only as the generator condition; final delivery/scoring audio is original natural PCM.
3. Identity and silence paths are exact. Reject non-finite inputs; explicitly declare gain/mask bounds, peak policy, silence treatment, chunking and any algorithmic latency.
4. Default head operates outside the TFG model with a shared waveform configuration. Any backend-specific calibrated preset is a separate evaluated mode with explicit model dependence and calibration cost.
5. Preferred trainable form: smooth bounded complex-STFT magnitude modulation with residual synthesis `y=x+iSTFT((mask-1)*STFT(x))`, identity initialization and fixed input STFT phase. Sample count is invariant; this constraint does not imply all output acoustic features are unchanged.
6. Gain, local envelope and frequency masks must be separately switchable for ablation. A small deterministic parameterized family is also supported as an optimization baseline and deployable alternative.

## Data and model isolation

- Audit actual data availability before freezing sample identities. Use speaker-disjoint fit, development and confirmation sets. Record the ordered sample IDs, speaker/gender/source, input path, natural PCM hash, sample count and split provenance.
- Previously inspected data may be explicitly assigned to fitting and method development; its exposed status must be recorded. It cannot support a fresh confirmation claim. Exclude historical male CAL S0052-S0061 / EVAL S0062-S0101, female S0124-S0143 and previously inspected AISHELL test speakers/samples from confirmation. Fit, dev and confirmation speakers still remain disjoint.
- Aim for at least 40 confirmation speakers / 80 clips if inventory supports it. Final eligibility rules and sample count must be frozen before fitting or reading candidate confirmation effects. Missing data is reported, not silently replaced.
- Optimize candidates/hyperparameters on fit/dev only. Freeze selected weights/configuration before confirmation. A changed candidate after a failed confirmation requires another untouched confirmation batch for a fresh claim.
- Validate at least three TFG models spanning different audio frontend families; target five when available. Reserve at least one model from candidate selection as the primary transfer model. Source faces, pose/template clips, model version/checkpoint, seeds, generation settings and model roles are frozen.
- Wav2Lip, Ditto, MuseTalk, LatentSync and EchoMimic are inventory candidates. Presence of a directory/weight is not execution evidence. Deployed commits and actual frontend normalization decide their roles.

Inventory-derived candidate data allocation, still awaiting row freeze: historical male S0052-S0101 multi-utterance fitting; fifty uninspected official HF speaker packs S0002-S0051 divided into ten development and forty confirmation speakers (two clips per confirmation speaker). Record the immutable dataset revision and gender composition. This supports new-speaker Mandarin confirmation; other genders/languages require separate untouched cohorts.

## Stages

### Stage 0: inventory and implementation

Read-only GPU/environment/storage/data/model audit; legacy learned-head failure review; prepare waveform CLI, model adapters, fresh mux/scoring, and independent plan/result schema. Existing environments and immutable research sources are preserved. Necessary dependency repairs use isolated new environments or minimal versioned adapters.

### Stage A: real waveform baselines

Compare identity and the historical RMS transform first, then a small preregistered set of bounded local-envelope/frequency presets on development data. Every arm performs actual TFG generation, original-PCM mux and fresh SyncNet scoring. This stage establishes working interfaces and candidate direction across frontends; it does not consume confirmation data.

### Stage B: method fitting and failure-driven iteration

Fit a constrained waveform head through a frozen differentiable TFG/SyncNet path where practical; use multi-frontend representation preservation and/or multiple generators for robust optimization. Black-box low-dimensional shared-controller optimization is a parallel route for non-differentiable generators. Only development results select the candidate. Track optimization instability, frontend gain invariance, spectral/content damage, timing/cropping/codec issues and cross-model tradeoffs separately; iterate on the corresponding failure mechanism.

### Stage C: confirmation and delivery

Evaluate the frozen selected method, original direct drive and RMS baseline on untouched speakers and the reserved model. Include gain-only, spectrum-only and local-time-component ablations; report compute/latency, peak/silence behavior, content preservation and independent visible-mouth measurements. Produce a runnable CLI/configuration/checkpoint, all required integration examples, a final report and a verified portable artifact set.

## Evaluation controls and statistics

- Natural scoring PCM, resampling, timestamps, FPS, crop/ROI policy and score model are identical across paired methods. Verify decoded final-audio PCM and its sample count against the canonical original, including any declared container limitations. A PCM-preserving MKV/MOV validation artifact can accompany a distributable MP4.
- Canonical float32 PCM is preserved exactly in the validation mux. Primary `native_roi_original_pcm16` scoring uses the selected native crop's video with a direct PCM16 derivative of the canonical original audio at its declared source-clock track interval. Verify that PCM16 against the independently decoded original slice; float-to-PCM16 conversion is not bit-exact float preservation. The pinned official pipeline's intermediate AVI actually uses MP3 by default (engineering QC04), so its `native_official_mp3` score is a separately labeled benchmark diagnostic. It must not replace the original-PCM primary endpoint. Bind each selected track to its matching crop, actual frame map and audio interval; retain native and frozen common-ROI endpoints separately. No earlier frozen run is changed.
- Use one same-seed identity-repeat engineering probe per model outside efficacy rows. For deterministic execution compare exact artifacts; otherwise measure repeat variation on development data and freeze paired-seed/repeat settings before confirmation. Numerical nondeterminism is recorded and controlled rather than used to silently remove a model.
- Fresh official Sync-C uses `Cbest = median(lag_distance)-min(lag_distance)`. Also retain `Canchor`, B, Dmin, D0, the 31-lag curve, offset and predetermined FULL/guard supports. Per-window distance matrices permit independent reconstruction.
- Freeze common support from paired valid query windows, and describe each support's denominator. FULL is the primary whole-query result; guard sensitivity is reported. No best-offset audio/video retiming is applied to improve a candidate score.
- Maintain plan and result rows including all engineering failures and short/unavailable samples. Give planned, complete, incomplete and failed counts by model and method; do not select successful faces/speakers after scoring.
- Average clips within speaker, source faces within the defined model estimand, then speakers equally. Reuse deterministic speaker bootstrap draws for candidate comparisons and model aggregate; report every model and a model-equal aggregate. The independent consumer will freeze exact fields and the statistical family before first Stage A effect read.
- Primary comparisons are selected shared head versus identity and versus RMS; transfer-model comparison is separate. Predeclared 95% simultaneous speaker-bootstrap intervals address the finite confirmation family. A 0.05 mean Sync-C improvement is an initial practical-effect target, not a threshold tuned from this confirmation set.
- Useful-method acceptance targets positive transfer and improvement across at least three tested models, positive model-equal aggregate, and useful gain beyond the RMS baseline. If only calibrated presets meet the criterion, label their model dependence. A model with no effect, loss or failure remains in the results and guides the next iteration; it is not removed from the declared scope to manufacture success.
- Independent visual/content measures supplement the SyncNet endpoint. Score improvements are not automatically labeled perceptual or phonetic improvements.

## Execution and evidence

Each scientific run records Git/source revision, immutable configuration, split/plan, model/input/weight pins, actual commands/environment, result rows, failures and output manifest. Preserve failed attempts, but avoid packaging/approval layers that delay scientific iteration. New implementation belongs in `scripts/audio_enhancement/`; independent evaluation belongs in `scripts/audio_enhancement_eval/`. ROOT owns this specification. Do not modify earlier concluded evidence or unrelated dirty files.

User authorized remote execution, reruns, data/weight downloads, local/Baidu storage and Qwen when useful. Credentials stay in runtime memory. Do not add/run unit tests; actual research evaluation and artifact/QC verification are part of this goal.
