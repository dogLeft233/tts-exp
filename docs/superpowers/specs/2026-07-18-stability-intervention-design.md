# Stability-Targeted Adversarial Perturbation + SyncNet Evaluation (Script 25)

**Status**: Approved design
**Date**: 2026-07-18
**Author**: tts-tfg mechanism study
**Replaces**: None (new)
**Related**: `scripts/22_dose_response_syncnet.py`, `scripts/23_identity_control_syncnet.py`, `scripts/24_identity_corrected_analysis.py`

## Motivation

Phase 2's identity-corrected dose-response (script 24) rejected all three
*acoustic* features (LUFS, spectral tilt, dynamic range) as isolable
causal mechanisms: every TTS-source intervention's Sync-C drop is
explained by the pipeline-noise baseline (ΔSync-C = −1.16 ± 0.47, n=9).
Two causal candidates survive:

1. **G×E co-adaptation** (`+7.29` Sync-C interaction) — joint property
   of Faster-Qwen3 × Ditto × SyncNet; cannot be tested by single-feature
   audio interventions by construction.
2. **HuBERT layer-11 `segment_stability`** (Phase 1 observational:
   FDR p=0.024, d=−1.045) — the strongest SSL-representation-side
   candidate. Phase 2's interventions did not target it directly.

Script 25 is the natural next test: a *stability-targeted adversarial
perturbation* that surgically perturbs the HuBERT-side property while
holding acoustic features approximately fixed, then re-runs the
Ditto + SyncNet pipeline.

## Hypothesis and Causal Logic

### The `segment_stability` cost metric (from script 16)

```python
segment_stability = mean(1 − cos_sim(h[i], h[i+1]))
                    for i in within_segment consecutive frame pairs
```

This is a **cost**: lower metric = more stable embeddings (frames stay
close in HuBERT space across the segment); higher metric = less stable.

### Phase 1 finding

| Group | layer-11 `segment_stability` mean | n |
|---|---|---|
| TTS     | 0.1506 | 9 |
| Natural | 0.1634 | 9 |
| Mean diff (`tts − natural`) | **−0.0128** | — |
| Cohen's d | −1.045 | FDR p = 0.024 |

**TTS audio produces *lower* segment_stability cost than natural**
(i.e. TTS embeddings are more temporally coherent within phoneme
segments than natural embeddings). The direction of the Phase 1 effect
is `TTS < Natural` (TTS is more stable, not less). The earlier informal
characterization "TTS produces less stable visemes" was wrong; the cost
raw value comparison is unambiguous: TTS = 0.1506 < Natural = 0.1634.

### Causal directions

If `segment_stability` is causally responsible for the +1.25 Sync-C
diagonal TTS-TTS advantage, then moving either source along this axis
*toward the other side* should move Sync-C in the same direction:

| Cell | Source | Cost-metric target | PGD direction | Expected Sync-C change vs unperturbed |
|---|---|---|---|---|
| `stability_adj_tts` | TTS | **raise** cost (destabilize TTS toward natural) | PGD maximizes `L` (i.e. minimize `−L`) | ↓ toward natural diagonal |
| `stability_adj_nat` | Natural | **lower** cost (stabilize natural toward TTS) | PGD minimizes `L` | ↑ toward TTS diagonal |
| `random_noise_tts` | TTS | untargeted (same ε budget, random direction) | n/a | ↓ (compare magnitude vs cell 1) |

### PGD `direction` parameter convention

Named after the **action on the cost metric**:

- `"raise_cost"` → PGD minimizes `−L` → moves metric upward →
  `stability_adj_tts` uses this.
- `"lower_cost"` → PGD minimizes `L` → moves metric downward →
  `stability_adj_nat` uses this.

### Sign-convention summary (memorize this)

- Higher `segment_stability` (the cost) = MORE drift between frames = LESS stable.
- Lower `segment_stability` = LESS drift = MORE stable.
- TTS produces **lower** cost → TTS is **more** stable at layer 11 than natural.
- Cell `stability_adj_tts` makes TTS **less** stable (raises cost) → expected to
  hurt Sync-C (drop toward the natural diagonal).
- Cell `stability_adj_nat` makes natural **more** stable (lowers cost) → expected
  to improve Sync-C (rise toward the TTS diagonal).

### 3-tier causal-verification ladder

| Tier | Baseline | Confound addressed |
|---|---|---|
| 1. Identity (script 23) | ΔSync-C = −1.16 ± 0.47 (n=9) | Pipeline write/Ditto/SyncNet re-run noise |
| 2. Random-direction noise at same ε (this script) | Tighter baseline: arbitrary-direction ε-noise | Any-direction noise at the same magnitude |
| 3. Stability-targeted perturbation at same ε | Most surgical intervention | Feature-specific signal beyond tiers 1 + 2 |

### Statistical tests (script 26; designed alongside, implemented separately)

All per-sample paired, n=9 (sample 9 excluded — systematic face-detection
failure, unrecoverable):

| Test | Statistic | Significance criterion | Reject causality if |
|---|---|---|---|
| 1 | Δ(`stability_adj_tts`) − Δ(`random_noise_tts`) vs 0 | paired t-test, p<0.05, |mean|>0.5 | p>0.20 OR |mean|<0.2 |
| 2 | Δ(`stability_adj_nat`) vs 0 | paired t-test, p<0.05, |mean|>0.5 | p>0.20 |
| 3 | Test 1's residual against the identity baseline | Robustness only — adds pipeline noise margin | — |

### Verdict codes (script 26)

- `STABILITY_CAUSAL`: Test 1 significant in expected direction, Test 2
  in expected direction (significant or trending, p<0.10). Stability is
  causally responsible for part of the +1.25 Sync-C diagonal advantage.
- `STABILITY_REJECTED`: Test 1 non-significant or wrong direction. G×E
  co-adaptation is the only surviving candidate.
- `INCONCLUSIVE`: Test 1 mixed signal (e.g., marginal p but wrong Test 2
  direction). Likely need higher n or different intervention design.

## Architecture

### `stability_loss(h_l11, frame_times, seg_boundaries_idx)`

Cosine-change cost over consecutive within-segment frame pairs, matching
the `segment_stability` half of `boundary_sharpness` in
`scripts/16_feature_separability.py:369` (`def boundary_sharpness`) exactly
so Phase 1 and Phase 2 metrics are commensurable.

```
emb_norm           = h_l11 / ||h_l11||_2  (per-frame)
L = mean(1 − dot(emb_norm[i], emb_norm[i+1]))
    for i in within_segment_consecutive_pairs
```

Boundary indices: token-boundary frame indices excluded from the
within-segment average (identical to script 16's logic).

### PGD loop per sample

```python
y = source_audio.float()  # (1, n_samples), in [-1, 1]
delta = torch.zeros_like(y, requires_grad=True)
sign_factor = -1.0 if direction == "lower_cost" else +1.0  # multiply loss to descend
for k in range(K):
    x_adv = (y + delta).clamp(-1, 1)
    h = huert_model(x_adv, output_hidden_states=True).hidden_states[12]
    L = stability_loss(h, frame_times, boundary_idx)
    # To lower_cost: minimize L directly -> sign_factor = -1.
    # To raise_cost: maximize L equivalently minimize -L -> sign_factor = +1.
    loss = sign_factor * L
    loss.backward()
    with torch.no_grad():
        delta -= alpha * torch.sign(delta.grad)  # gradient descent on `loss`
        delta.clamp_(-eps, +eps)                  # L_inf projection
    delta.grad.zero_()
perturbed = (y + delta).clamp(-1, 1).detach()
```

Where `loss = sign_factor * L` and `delta -= alpha * sign(delta.grad)`
together implement gradient descent on `loss`:

- `lower_cost` (`sign_factor = -1`): descend `L = stability_loss`,
  shrinking the metric.
- `raise_cost` (`sign_factor = +1`): descend `−L`, i.e. ascend `L`,
  growing the metric.

Hyperparameters (script 25 ships with these defaults):

```
eps   = 0.005   # L_inf bound, ~ -46 dB rel peak amplitude
alpha = 0.001   # per-step size (eps / 5)
K     = 50      # iterations
restarts = 1    # single run per sample (per Q4)
```

Escape-hatch args allow override: `--eps`, `--alpha`, `--pgd-steps`,
`--pgd-restarts` for later sensitivity exploration.

### Random-noise control

```python
sign = torch.randint(0, 2, y.shape, device=y.device) * 2 - 1
delta = sign * (eps * torch.rand(y.shape, device=y.device))
perturbed = (y + delta).clamp(-1, 1)
```

Same ε budget as PGD for a clean feature-specific comparison.

### Intervention registry (3 cells)

```python
Intervention("stability_adj_tts",  source="tts",     baseline_cond="tts_raw",
    transform=pgd_stability(direction="raise_cost", eps=0.005, alpha=0.001, K=50))
Intervention("stability_adj_nat",  source="natural", baseline_cond="natural_raw",
    transform=pgd_stability(direction="lower_cost", eps=0.005, alpha=0.001, K=50))
Intervention("random_noise_tts",   source="tts",     baseline_cond="tts_raw",
    transform=random_sign_noise(eps=0.005))
```

Direction mapping (matches the table above):

- `stability_adj_tts` → `raise_cost` → destabilize TTS → push Sync-C
  *down* (toward natural diagonal).
- `stability_adj_nat` → `lower_cost` → stabilize natural → push Sync-C
  *up* (toward TTS diagonal).

### Token-boundary source

- **Default**: MFA phoneme alignments from
  `data/wav2sem_analysis/alignments/{tts,natural}/{sid}.txt`
  (whitespace-delimited end-times in seconds; mirrors script 16's loader).
- **Fallback**: uniform 50ms segmentation when MFA missing for a sample
  (matches script 16's existing fallback path — keeps cells comparable
  to Phase 1 results).
- Boundary times are converted to frame indices via
  `int(np.searchsorted(frame_times, boundary_time))`, exactly like
  script 16.

### Pipeline per cell × sample (reuses script 22 infrastructure)

Each `(intervention, sample)` cell:

1. **Perturbation step**:
   a. Load source audio (TTS or natural) as `float32` mono at TARGET_SR.
   b. Load HuBERT base model with `output_hidden_states=True`, `.eval()`,
      frozen weights.
   c. Extract reference `frame_times` and `boundary_idx` for this sample.
   d. Run PGD (or random-noise generator) → perturbed waveform.
   e. Write perturbed audio to `runs/{run_id}/03_ditto/interventions/{intv}/{sid}.wav`
      as PCM_16.
2. **Post-hoc verification** (new): re-extract HuBERT layer-11
   embeddings from the perturbed audio and re-compute
   `segment_stability`. Record:
   - `stability_metric_pre` (computed on the unperturbed audio)
   - `stability_metric_post`
   - `stability_metric_delta` and direction (expected vs achieved)
   - `lufs_pre`, `lufs_post`, `Δ_lufs`
   - `tilt_pre`, `tilt_post`, `Δ_tilt`
   - `dynamic_range_pre`, `dynamic_range_post`, `Δ_dyn`
3. **Ditto inference** → MP4 video at
   `runs/{run_id}/03_ditto/interventions/{intv}/{sid}.mp4` (reuses script
   22's cached Ditto invocation logic).
4. **SyncNet evaluation** →
   `runs/{run_id}/04_eval/interventions/{intv}/{sid}/syncnet.json`.
5. **Δ Sync-C/D** vs original diagonal baseline (`tts_raw` or
   `natural_raw`) loaded from the strict-run `04_eval/` tree.

Caching flags `--no-cache-audio`, `--no-cache-video`, `--no-cache-syncnet`
mirror script 22 for re-runs.

### Module wiring

Script 25 imports (no code duplication):

- From `scripts/tfg_feature_common`: `STUDY_SAMPLES`, `TARGET_SR`,
  `load_audio_mono`, `ensure_output_dirs`.
- From `scripts/15_extract_ssl_embeddings`: `_load_model`,
  `extract_frame_embeddings`, `_compute_frame_stride`.
- From `scripts/16_feature_separability`: `compute_boundary_and_stability`
  (used in post-hoc verification).
- From `scripts/22_dose_response_syncnet`: `Intervention`,
  `run_intervention_pipeline`, `load_baseline_results`,
  `aggregate_deltas`. These are reused verbatim.
- New in script 25: `pgd_stability_transform`, `random_sign_noise_transform`,
  `stability_loss`, `load_token_boundaries`, `post_hoc_verify`.

## Data Flow

```
Inputs:
  data/data/audio/{sid}.wav                              (natural)
  runs/{run_id}/02_tts/{sid}.wav                         (TTS)
  data/wav2sem_analysis/alignments/{tts,natural}/{sid}.txt  (MFA boundaries)

→ scripts/25_stability_perturbation_syncnet.py
   → runs/{run_id}/03_ditto/interventions/{intv}/{sid}.wav   (perturbed PCM)
   → runs/{run_id}/03_ditto/interventions/{intv}/{sid}.mp4   (Ditto video)
   → runs/{run_id}/04_eval/interventions/{intv}/{sid}/syncnet.json
   → data/wav2sem_analysis/metrics/stability_intervention.json
     (per-cell × per-sample + post-hoc verification fields)
   → data/wav2sem_analysis/metrics/stability_intervention_summary.json
     (aggregated means, paired-t stats, verdict)
```

## Output JSON shape

`stability_intervention.json` (compatible with script 24's reader):

```json
{
  "run_id": "r2_faster_qwen3_20260707T145233Z",
  "samples": [1, 2, 3, 4, 5, 6, 7, 8, 10],
  "baselines": {
    "tts_raw":     {"1": {"sync_c": ..., "sync_d": ...}, ...},
    "natural_raw": {"1": {...}, ...}
  },
  "interventions": {
    "stability_adj_tts": {
      "n_samples": 9,
      "per_sample": {
        "1": {
          "sync_c": ...,
          "sync_d": ...,
          "delta_c": ...,
          "delta_d": ...,
          "post_hoc": {
            "stability_metric_pre": ...,
            "stability_metric_post": ...,
            "stability_metric_delta": ...,
            "expected_direction": "lower_cost",
            "achieved_direction": "lower_cost",
            "lufs_pre": ..., "lufs_post": ..., "delta_lufs": ...,
            "tilt_pre": ..., "tilt_post": ..., "delta_tilt": ...,
            "dyn_pre": ..., "dyn_post": ..., "delta_dyn": ...
          }
        }, ...
      },
      "mean_delta_c": ..., "mean_delta_d": ...
    },
    "stability_adj_nat": {...},
    "random_noise_tts": {...}
  },
  "failed": [],
  "elapsed_s": ...,
  "pgd_params": {"eps": 0.005, "alpha": 0.001, "K": 50, "restarts": 1},
  "verdict": "TBD"  # filled in by script 26
}
```

## Unit-test Plan (12–16 tests, target)

File: `tests/test_stability_perturbation.py`

1. `stability_loss` returns 0 for identical consecutive embeddings.
2. `stability_loss` returns the correct cosine-distance value for
   two-frame synthetic input.
3. `stability_loss` ignores boundary-straddling pairs (boundary mask).
4. PGD-projected `delta` strictly satisfies `|delta| <= eps`.
5. PGD output stays within waveform bounds `[-1, 1]`.
6. `random_sign_noise` produces `|delta| <= eps`.
7. PGD on synthetic audio + randomized HuBERT-like weights: 5 steps
   move the stability metric in the requested direction (lower_cost or
   raise_cost).
8. `_load_interventions` returns exactly 3 cells with expected names.
9. `load_token_boundaries` happy path with a synthetic MFA file.
10. `load_token_boundaries` fallback path when MFA file missing
    (returns uniform 50ms segments).
11. Post-hoc verification flags the expected-vs-achieved direction
    correctly for both `lower_cost` and `raise_cost` directions.
12. Full pipeline smoke test with `--dry-run` and a stubbed HuBERT
    model (no GPU): intervention registry is iterated and the report
    JSON shape is correct.
13. (Optional) Acoustic-feature drift sanity: on a synthetic 100ms
    sine, PGD with `eps=0.005` produces LUFS/tilt/dynamic-range
    drifts within 0.5 dB / 0.3 dB/oct / 1.0 dB respectively. This
    validates the "tight ε" assumption in a controlled input.
14. (Optional) `run_intervention_pipeline` integration: a single
    `(intervention, sample)` cell with mocked Ditto/SyncNet call
    produces a record with all required fields.

## Compute budget

| Step | Time per sample × cell | Cells × samples | Total |
|---|---|---|---|
| HuBERT forward+backward × 50 PGD iter | ~5s | 2 (PGD cells) × 9 | ~1.5 min |
| Post-hoc verification (1 forward pass) | ~0.1s | 3 × 9 | negligible |
| Ditto inference | ~3 min | 3 × 9 | ~1.5 h |
| SyncNet eval | ~10s | 3 × 9 | ~5 min |
| **Total** | | 27 cells | **~1.7 h on RTX 4080** |

Within the same order of magnitude as script 22's last run (37 min for 72
cache-cold cells). Existing HuBERT model weights cached on server from
script 15 reuse; no new downloads.

## Risks / Open Questions

- **Convergence**: HuBERT gradient wrt raw waveform may not move the
  segment_stability metric far in 50 PGD steps. Mitigation:
  post-hoc verification flags any cell whose `stability_metric_delta`
  is opposite-signed or smaller than the median per-step move. If
  more than 3/9 samples fail, escalate to `K=100` and `restarts=4`
  via CLI args.
- **L_inf vs L2 budget**: ε = 0.005 is per-sample L_inf — a tight
  bound. If the resulting visibility is below the model's sensitivity,
  try ε = 0.01 (still ~−40 dB, generally inaudible on speech).
- **Sign vs full gradient**: shipped PGD uses `sign(delta.grad)` for
  simplicity; if convergence stalls, consider upgrading to full-gradient
  or Adam on `delta`.
- **Token boundaries mismatch between MFA and segment_stability's
  "segment" definition**: this is by design — script 16 used MFA
  boundaries and script 25 reuses the same definition for Phase
  commensurability.
- **Cross-TFG validation not covered**: script 25 only tests Ditto.
  Cross-model HuBERT extraction for Wav2Lip / MuseTalk / LatentSync /
  JoyVASA is out of scope here and remains a follow-up task.

## Out of Scope (explicitly)

- Script 26 (paired-t statistical analysis, verdict classification). It
  will be a separate ~200-line script with its own unit tests, mirroring
  script 24's structure.
- Cross-TFG extension to Wav2Lip / MuseTalk / LatentSync / JoyVASA.
- Human perceptual evaluation.
- Larger-n extension (n=9 is fixed by Phase 1's available data).

## Acceptance Criteria

- [ ] All 3 intervention cells run end-to-end on 9 samples × 1 run
      (script 25 exits 0).
- [ ] Post-hoc verification confirms stability metric moves in the
      requested direction for ≥7/9 samples in each PGD cell.
- [ ] LUFS / spectral-tilt / dynamic-range drift stays within ±0.5 dB /
      ±0.3 dB/oct / ±1.0 dB respectively for ≥7/9 samples in each cell
      (otherwise flagged as acoustic-confounded).
- [ ] Output JSON passes schema check (all required fields present).
- [ ] All 12–16 unit tests pass; full pytest remains ≥219 + new tests.
- [ ] `data/wav2sem_analysis/metrics/stability_intervention.json`
      committed for downstream analysis (script 26).
- [ ] Mechanism report + Phase 2 execution log updated with results &
      verdict once script 26 lands (separate step).

## References

- Phase 1 separability script computing `segment_stability`:
  `scripts/16_feature_separability.py:388` (`compute_boundary_and_stability`)
- Phase 2 dose-response pipeline reused verbatim:
  `scripts/22_dose_response_syncnet.py:247` (`run_intervention_pipeline`)
- Identity control:
  `scripts/23_identity_control_syncnet.py` (verdict ladder vocabulary)
- Identity-corrected paired analysis:
  `scripts/24_identity_corrected_analysis.py`
- HuBERT embedding extraction (reused verbatim):
  `scripts/15_extract_ssl_embeddings.py:177` (`extract_frame_embeddings`)
- Phase 2 execution log post-identity findings:
  `docs/experiments/phase2_execution_20260718.md`
- Mechanism report (causal candidates table):
  `docs/experiments/tts_tfg_mechanism_report.md`