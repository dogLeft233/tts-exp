## ADDED Requirements

### Requirement: Independent real-trajectory calibration

The visual branch SHALL consume design.md inputs independently of SyncNet scores. Reuse `scripts/experiments/lrs3_tts_visual_advantage/video_features.py` create_landmarker / extract_video_features; pin MediaPipe version, model asset hash and options. Extract R and each generated video on the common 140-frame clock, with canonical_mouth[T,31,2], valid[T], timestamps_s[T]. No learned audio-video similarity score enters this route.

Before inspecting generated features, calibrate using R only. On fixed J0=`range(25,115)`, compute centered dynamic MSE of R vs itself, R shifted by +5 and −5 frames, and R reversed across the complete140 frames (index139−t). Center each compared sequence on its own valid common support. Identity MSE must≤1e-12. For each distorted condition require error>1e-6 in at least10/12 groups. A pair requires ≥90% valid support; keep all12 groups in counts. These thresholds establish implementation sensitivity only, not perceptual validity. Failure yields METRIC_NOT_CALIBRATED; keep descriptive numbers and the blind package, without automated dynamic-advantage claims or threshold changes.

#### Scenario: Metric ignores temporal order

- **WHEN** Shift or reversal does not satisfy the fixed real-only sensitivity gate
- **THEN** The metric is not calibrated even if generated candidates later score better

### Requirement: Fixed-clock dynamic advantage with missingness

The branch SHALL compare each model separately. Its common support J is the intersection of valid frames for R,N42,C42,N43,C43 within J0; exact frame indices and matching PTS (tolerance≤0.5/25 seconds) are required. No interpolation across missing frames, shift search or DTW. A group is observed when |J|≥81; otherwise mark its group result missing. R is the original moving video, never a generated natural video.

For each X on J: μ_X=mean_t(X), Xc=X−μ_X; e_dynamic_X=mean((Rc−Xc)^2), e_static_X=mean((μ_R−μ_X)^2), e_total_X=mean((R−X)^2). Verify total=static+dynamic (absolute tolerance1e-10×max(1,total)). For each seed compute b=(e_dynamic_N−e_dynamic_C)/(e_dynamic_N+e_dynamic_C+1e-12), then average the two b values per group. If both errors zero, b=0. Report raw errors, b, group support counts, static/total errors, and each arm's missingness.

Use PCG64(20260910),20000 group bootstrap draws,99%CI,linear quantiles; observed-only CI must be labeled with its observed group count. Missing b belongs to [−1,1]. Full12-group bounds are [(sum observed b−m)/12,(sum observed b+m)/12], where m is missing groups; these are finite-cohort bounds, not CIs. Static motion is not dropped merely for yielding b=0.

For each generated arm, also reverse its centered sequence along that same J and compute q=(e_reverse−e_dynamic)/(e_reverse+e_dynamic+1e-12); average q over the two seeds per condition. Natural and candidate q must each have observed99%CI lower>0 and at least10/12 positive groups. Otherwise use GENERATED_TIMING_SPECIFICITY_UNRESOLVED; a static candidate cannot pass merely by shrinking its movement amplitude. The old visual run already passed reversal sensitivity, so this is a measurement requirement, not a presumption that old metrics were broken.

Per model an automated dynamic SIGNAL requires real calibration and both generated q gates, at least11/12 observed, observed mean b>0.02, lower99%CI>0, at least10/12 groups positive (missing not counted positive), and full-cohort lower bound>0. Other statuses: MISSINGNESS_LIMITED if observed criteria pass but coverage/bounds fail; otherwise NO_DYNAMIC_ADVANTAGE_ESTABLISHED. The0.02 threshold is exploratory, not a validated perceptual margin. Failure for one model does not prevent reporting the other.

#### Scenario: Favorable complete cases hide failed candidates

- **WHEN** Candidate landmarks fail on some frozen groups
- **THEN** Those groups remain in the12-group bounds and arm failure table; observed positivity cannot automatically verify the full cohort

### Requirement: Blind human pairing

The branch SHALL prepare 24 primary pair trials:12 groups×2 models, seed42 only, each showing synchronized silent R and randomized left/right N/C panels from the same fixed frame interval25..114. Normalize panel size/crop/encoding equally; generic trial IDs and side labels carry no condition or scores. Use a separate secret mapping, seed20260911, outside the reviewer package. Reviewer prompt: “哪侧嘴部运动在开合时刻和运动轨迹上更接近真实视频？左／右／相同／无法判断”。Identity similarity and overall sharpness are not the target.

Add4 QC trials selected by the first two groups: for each group one identity-vs-+5-frame-shift pair and one identity-vs-reversal pair using R. Mix their order with the24 trials. Prepare two reviewer packages with independent side flips. Two human reviewers independently complete every trial; agents may create the package but may not invent ratings or submit on a human's behalf. No external messages are sent without explicit authorization.

A reviewer qualifies when at least3/4 QC answers select identity. Report each reviewer separately and agreement. Missing/unjudgeable answers remain visible; with fewer than2 qualified complete reviewers set human_status=insufficient. For each model with2 qualified complete reviewers, score each trial C=1,N=−1,tie/unjudgeable=0, average reviewers per group, then group mean and99%bootstrapCI. Human support requires mean>0,CI lower>0 and at least9/12 group means>0. Human disagreement does not trigger reassignment or condition-informed exclusion.

#### Scenario: No human ratings are available

- **WHEN** Automated analysis and the blind package are complete but ratings have not arrived
- **THEN** Set automated_status=complete, human_status=pending, visual_verified=false; shut down the GPU and leave only the human step pending

### Requirement: Bounded visual conclusion and independent validation

The branch SHALL set visual_verified=true only for a model passing both its automated dynamic and human support rules, while keeping replacement_confirmed=false. Combined reporting may say a model has “independent visual support for an exploratory replacement signal” only if both A and B pass for that same model. A alone is a scorer signal; B alone is a visual result without established natural-audio synchronization gain.

Its validator SHALL independently recompute calibration, J, decomposition, b, bounds, CI, anonymization completeness and available human counts, without importing the producer's analysis/gate functions. It need not rerun the landmarker. Save feature provenance and deterministic sampled overlay contact sheets (groups1,6,12,frames25,70,114) for inspecting landmark placement; these sheets do not replace human ratings.

#### Scenario: Independent routes disagree

- **WHEN** SyncNet improves but visual or human evidence does not support improvement
- **THEN** Report the disagreement and the outstanding boundary; do not authorize training or claim confirmed replacement
