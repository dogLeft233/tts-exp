# Handoff — IndexTTS2 Emotion × TFG: Ditto+SyncNet Evaluation Complete

**Date**: 2026-07-08  
**Project**: `tts-exp` (TTS Talking Face Generation comparison via SyncNet)  
**Repo**: `/mnt/e/Documents/tts-audio/tts-exp`

---

## What Happened This Session

### TFG pipeline executed on :20398 for all IndexTTS2 + GSV-TTS conditions

A new SeetaCloud server was used for evaluation (the IndexTTS2 generation server :49961 doesn't have Ditto/SyncNet installed).

- **Server**: `connect.westd.seetacloud.com:20398`, RTX 4080 SUPER 32GB, CUDA 13.2, driver 595.71.05
- **Repo**: `/root/autodl-tmp/repos/tts-exp/` (already deployed from prior experiments)
- **Environments**: `/root/autodl-tmp/envs/ditto/` and `/root/autodl-tmp/envs/syncnet/` — both ready
- **Checkpoints**: Ditto TRT/PyTorch/ONNX all present in `/root/autodl-tmp/checkpoints/`

### Data uploaded

| Source | Files | Location on server |
|--------|:-----:|-------|
| IndexTTS2 baseline (13 samples) | 13 | `/root/autodl-tmp/indextts2/baseline/indextts2_output/` |
| IndexTTS2 emotion sweep P2 (40) | 40 | `/root/autodl-tmp/indextts2/emo_sweep/indextts2_emo_sweep/` |
| IndexTTS2 intensity sweep P3 (18) | 18 | `/root/autodl-tmp/indextts2/emo_intensity/indextts2_emo_intensity/` |
| GSV-TTS (13) | 13 | `/root/autodl-tmp/gsv_tts/gsv-tts/` (renamed `{n}gsv.wav` → `{n}.wav`) |

### Batch pipeline executed: 16 runs × Ditto(TRT) → SyncNet → Report

Batch script: `/root/autodl-tmp/batch_pipeline.sh`  
Log: `/root/autodl-tmp/batch_pipeline.log`

**Total**: 69m27s, 0 failures, 168 syncnet.json files produced.

The pipeline uses a neat trick: `detect_sample_ids()` reads from `data/data/audio/`, so for each run the script trims that directory to only the required sample subset (13, 5, or 3 files), runs the full Ditto→SyncNet→Report chain, then restores the full 13-file set before the next run. Original audio backed up at `data/data/audio.orig/`.

| Phase | Runs | Samples/run | SyncNet files |
|-------|:----:|:-----------:|:-------------:|
| IndexTTS2 baseline | 1 | 13 | 26 |
| P2 emotion sweep | 8 | 5 | 80 |
| P3 intensity sweep | 6 | 3 | 36 |
| GSV-TTS | 1 | 13 | 26 |
| **Total** | **16** | — | **168** |

---

## Key Results

Statistical analysis: `scripts/07_emo_analysis.py`  
Raw data: `data/r3_results/p2_emotion_sweep.csv`, `p3_intensity_sweep.csv`, `baseline_gsv.csv`  
Full documentation: `docs/indextts2.md` (updated with results section)

### Phase 2 — Emotion matters significantly

**RM-ANOVA: F(8,32)=6.585, p=0.000028**

| Rank | Condition | ΔC vs Natural | p vs baseline |
|:----:|-----------|:---:|:---:|
| 1 | **angry** | +2.80 | 0.001 ** |
| 2 | calm | +2.27 | 0.026 * |
| 3 | surprised | +1.76 | 0.195 |
| 16 | melancholic | −0.08 | 0.008 ** |

angry and calm significantly improve lip sync; melancholic/sad/afraid significantly degrade it.

### Phase 3 — Dose-response is non-linear

- **calm**: monotonic ↗ with alpha (α=1.0 best: ΔC=+2.73)
- **happy**: inverted-U ∩, peak at α=0.6 (ΔC=+2.49), drops at α=1.0

### IndexTTS2 vs GSV-TTS

IndexTTS2 baseline wins on both SyncC (+0.37) and SyncD (−0.20). Effect d=0.418, non-significant (p=0.20).

---

## Next Moves

### 1. Cross-model comparison

Integrate these results into `scripts/06_cross_model.py` so IndexTTS2 conditions appear alongside other TTS providers (Fish, Qwen3, DashScope VC, SiliconFlow).

### 2. Deeper statistical analysis

- **Correlation**: Does TTS audio duration predict SyncC? (suspected: shorter = better, since angry/surprised produce shortest audio)
- **Mixed-effects model**: Treat sample_id as random effect (current RM-ANOVA already accounts for this via paired design but doesn't model within-sample variance)
- **Visualizations**: Box plot grid (8 emotions × SyncC), interaction plot (alpha × emotion), emotion ranking bar chart

### 3. Add IndexTTS2 as formal TTS provider

Currently IndexTTS2 TTS audio is generated externally and copied into `02_tts/`. To enable `run_all.sh` → `02_tts.py` to generate IndexTTS2 directly:
- Write `scripts/tts/indextts2.py` following the factory pattern (`factory.py` + existing providers like `faster_qwen3.py`, `fish_tts.py`)
- Add to `scripts/config.yaml` under `tts:` providers

### 4. Replication

The angry/calm findings are based on n=5 per condition. A confirmatory study with n=13 (all samples) for the top 2-3 emotions would increase power. The script is ready — just change `P2_IDS` in `batch_pipeline.sh`.

### 5. Phase 1 — emotion source comparison (not yet run)

The original plan had Phase 1 comparing different emotion sourcing methods (text-inferred vs vector vs audio-prompt). This hasn't been executed yet.

---

## Artifacts

| Artifact | Path |
|----------|------|
| Analysis script | `scripts/07_emo_analysis.py` |
| Raw CSVs | `data/r3_results/p2_emotion_sweep.csv`, `p3_intensity_sweep.csv`, `baseline_gsv.csv` |
| Updated docs | `docs/indextts2.md` |
| Batch script (local copy) | records `tmp/batch_pipeline_v2.sh` |
| Batch script (server) | `/root/autodl-tmp/batch_pipeline.sh` |
| Batch log (server) | `/root/autodl-tmp/batch_pipeline.log` |
| Emotion generation script | `tmp/batch_generate_emo.py` |
| Prior handoffs | `/tmp/tts-exp-handoff-20260708-v2.md` (deployment), `/tmp/tts-exp-handoff-20260708-v3.md` (emotion design) |
| Server credentials | `tmp/servers.sh` |
| TTS outputs (local) | `tmp/indextts2_output/`, `tmp/indextts2_emo_sweep/`, `tmp/indextts2_emo_intensity/` |
| TTS outputs (server :49961) | `/root/autodl-tmp/indextts_batch/output/` |
| TFG runs (server :20398) | `/root/autodl-tmp/repos/tts-exp/runs/r3_*` |
| Config | `scripts/config.yaml` |

---

## Suggested Skills

| Skill | When to use |
|-------|-------------|
| `diagnose` | If Ditto/SyncNet pipeline fails when re-running or adding new conditions |
| `brainstorming` | Before writing Phase 1 emotion source comparison, or before deeper statistical analysis |
| `graphify` | To understand TTS factory pattern before adding IndexTTS2 as provider |
| `to-issues` | To create GitHub issues for Phase 1, cross-model integration, confirmatory replication |
| `handoff` | After completing any of the "Next Moves" items |
| `writing-plans` | Before implementing IndexTTS2 provider or cross-model integration |
