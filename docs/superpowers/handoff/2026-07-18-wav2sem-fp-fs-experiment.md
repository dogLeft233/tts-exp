# Handoff — Wav2Sem Fp/Fs Mediation Experiment Design

**Date**: 2026-07-18
**Source conversation**: English TTS replication + Wav2Sem reproduction scoping
**Next session focus**: Implement the Fp/Fs swap experiment to test whether Wav2Sem-style semantic features mediate the TTS lip-sync advantage.

---

## Background

The project has completed Phase 1 (Wav2Sem-style HuBERT separability) and Phase 2 (acoustic interventions, identity control, stability perturbation) on 12 Chinese AISHELL-1 samples. Key findings:

- TTS advantage is real but model-dependent (Ditto ΔSync-C = +1.246; LatentSync ≈ 0).
- Phase 2 identity control: all dose-response acoustic drops disappear after subtracting pipeline noise.
- Main mechanism: **G×E co-adaptation** (+7.29 interaction), not a single acoustic feature.
- HuBERT layer-11 `segment_stability` is strongest candidate (FDR p=0.024) but fails bidirectional causal test.
- Full mechanism report: `docs/experiments/tts_tfg_mechanism_report.md`

The user asked to (1) add English natural speech samples, (2) run TTS vs natural comparison, (3) reproduce Wav2Sem. This evolved into an analysis of whether Wav2Sem can explain the TTS advantage, and a clean experiment design.

---

## Key Decisions Made

| Decision | Choice |
|---|---|
| English sample source | LibriSpeech test-clean, 13 speakers, 13 utterances, ≥5s each |
| TTS backend | Existing `FasterQwen3TTSProvider` with `language="English"` |
| TFG pipeline | Ditto + SyncNet (same as Chinese experiments) |
| Face images | Existing 13 surrogate faces (LibriSpeech has no video) |
| Transcript | Use official LibriSpeech gold text; run Qwen ASR for WER baseline |
| Wav2Sem scope | Postpone full training; first validate the logical chain |
| English experiment launch | Deferred pending Fp/Fs mechanism validation |

---

## Wav2Sem Logical Chain (from Paper)

The paper claims:

```
Audio A → TCN + Transformer → sentence-level semantic feature Fs
Fs aligned to BERT text semantic space via L1 loss
Fd = FC(FC(Fs) + Fp)   (Fp = frame-level phoneme features from SSL encoder)
Decoupled Fd → reduced near-homophone averaging → better facial animation
```

The paper proves this on VOCASET/BIWI with 3D mesh metrics (LVE, MVE, FDD). It does **not** test TTS, SyncNet, or explain the natural-vs-TTS gap.

The inference chain for our context:

```
TTS audio → more regular acoustic pattern
         → Fs closer to BERT text target
         → near-homophone syllables more separable
         → reduced lip-shape averaging
         → better TFG output (Sync-C, Sync-D)
```

This is a **candidate mechanism**, not a proven conclusion.

### Model Scale

- Default Wav2Sem: **~94M parameters** (7-layer TCN with 512 channels, 12-layer Transformer, hidden_size=768)
- ~360 MB FP32 weights
- Training: LibriSpeech-960, 200 epochs, batch_size=1, Adam, lr=1e-4
- Official checkpoint: Baidu Netdisk (see Wav2Sem README)
- Official code: `https://github.com/wslh852/Wav2Sem` — two subdirectories: `Wav2Sem/` (training) and `CodeTalker_Wav2Sem/` (integration)
- Caveat: data loader has a key-collision bug (`read_data()` in `data_loader.py` — keys `f+'-'+fd` overwrite multiple utterances per chapter)

---

## The Fp/Fs Swap Experiment Design

### Definitions

```
Fp = frame-level phoneme/acoustic features (e.g., from HuBERT, Wav2Vec2, or Chinese SSL)
Fs = sentence-level semantic feature (audio → TCN+Transformer → BERT-aligned vector)
Fd = FC(FC(Fs) + Fp)   — broadcast Fs to each frame, add to Fp, then linear projection
```

### 2×2 Factorial Design

For each utterance (same text for natural N and TTS T):

| Condition | Fp source | Fs source | Label |
|---|---|---|---|
| Fp(N) only | Natural | — | N-only |
| Fp(T) only | TTS | — | T-only |
| Fp(N) + Fs(N) | Natural | Natural | N-N |
| Fp(N) + Fs(T) | Natural | TTS | N-T |
| Fp(T) + Fs(N) | TTS | Natural | T-N |
| Fp(T) + Fs(T) | TTS | TTS | T-T |

Each row is a feature vector fed to the same TFG generator and the same evaluator.

### Mechanism Interpretation

| Pattern | Interpretation |
|---|---|
| Results follow Fs (N-T ≈ T-T, T-N ≈ N-N) | Semantic feature mediates TTS advantage |
| Results follow Fp (N-N ≈ N-T, T-N ≈ T-T) | Acoustic/phonetic feature mediates advantage |
| Only N-N, T-T are high; cross-cells are low | Fp × Fs co-adaptation (like current G×E finding) |
| Fp-only ≈ Fp+Fs (within same Fp) | Fs adds no value → semantic channel irrelevant |

### Upstream Metrics (before TFG)

| Metric | What it tests |
|---|---|
| `||Fs - F_CLS||` (L1 or cosine) | Are TTS semantics closer to text target? |
| Var(Fs) across runs | Is TTS semantic representation more stable? |
| Var(Fp) within same viseme/phone | Are TTS phoneme features more consistent? |
| L2 distance between near-homophone pairs in Fd | Does Fs improve decoupling? |
| Fp swapping → Fd change magnitude | How much of Fd is driven by Fs vs Fp? |

---

## Chinese Audio Feasibility

### Three Tiers

1. **Oracle Fs (no training)**
   - Use transcript → Chinese BERT/RoBERTa CLS vector as `Fs`
   - Zero training cost, same-day implementation
   - Proves whether the semantic channel *could* help
   - Cannot claim "audio automatically produced this semantic feature"

2. **Trained Chinese Fs**
   - Train TCN+Transformer on AISHELL-1 training set (or other Chinese speech-text pairs)
   - Target: Chinese BERT/RoBERTa sentence embedding
   - 12 current test samples held out for evaluation only
   - ~94M params, feasible on RTX 4080 (32GB) with small batch size

3. **Full TFG Causal Experiment**
   - Requires TFG model that accepts external Fp/Fs instead of raw audio
   - Current Ditto pipeline only accepts raw WAV; needs feature-injection modification
   - CodeTalker (from Wav2Sem's own integration) may be easier to instrument

### Recommended Path

```
1. Oracle Fs on Chinese → validate 2×2 logic (fast)
2. If oracle shows Fs matters:
   → Train Chinese audio-only Fs on AISHELL-1
   → Run full Fp/Fs swap experiment
3. If oracle shows Fs doesn't matter:
   → Redirect to acoustic-only explanations
   → Cancel Wav2Sem training plans
```

---

## Existing Resources to Reuse

- **Pipeline scripts**: `scripts/01_asr.py`, `scripts/02_tts.py`, `scripts/03_ditto.py`, `scripts/04_eval.py`, `scripts/05_report.py`
- **TTS providers**: `scripts/tts/` (all accept `language` parameter; `faster_qwen3` primary)
- **HuBERT extraction**: `scripts/15_extract_ssl_embeddings.py` (L11 embeddings, 16kHz mono)
- **Separability**: `scripts/16_feature_separability.py`
- **Feature linking**: `scripts/17_link_features_to_tfg.py`
- **Audio transformations**: `scripts/12_transformation_experiment.py`, `scripts/20_causal_feature_interventions.py`
- **Stability intervention**: `scripts/25_stability_perturbation_syncnet.py`
- **Config**: `scripts/config.yaml` (single truth source; TTS provider switchable)
- **Mechanism report**: `docs/experiments/tts_tfg_mechanism_report.md`
- **Wav2Sem analysis**: `docs/experiments/wav2sem_style_analysis.md`
- **Wav2Sem paper**: `papers/Wav2Sem_CVPR2025.pdf`, `papers/Wav2Sem_arXiv.pdf`
- **Wav2Sem code**: `https://github.com/wslh852/Wav2Sem` (not yet cloned; trained on LibriSpeech-960, English BERT)
- **Handoff**: `HANDOFF.md` (server access, protocol, known bugs)

---

## Server Info

- **Host**: `connect.westd.seetacloud.com:26325`
- **GPU**: RTX 4080 (32GB VRAM)
- **Environments**: `/root/autodl-tmp/envs/ditto`, `/root/autodl-tmp/envs/syncnet`
- **Checkpoints**: `/root/autodl-tmp/checkpoints/`
- **Repo**: `/root/autodl-tmp/repos/tts-exp`
- **HF mirror**: `HF_ENDPOINT=https://hf-mirror.com` must be set on mainland server
- **Credentials**: see HANDOFF.md for redacted passwords; API keys in `scripts/.env`

---

## Blockers & Known Risks

| Risk | Mitigation |
|---|---|
| **Wav2Sem official checkpoint inaccessible** (Baidu Netdisk) | Use oracle Fs first; train from scratch only if needed |
| **Wav2Sem data loader bug** (key collision per chapter) | Fix before training: use full file path as key |
| **Ditto only accepts raw WAV** | Fp/Fs swap needs feature injection; may need different TFG model |
| **Chinese BERT vs English BERT dimension** | Match Chinese RoBERTa (768-d) to Wav2Sem hidden_size (768 or 1024) |
| **AISHELL-1 training set size** | 150h — may be sufficient for small-scale Fs training |
| **RTX 4080 VRAM** | 94M-param model fits; batch_size=1 used in paper |
| **Sample 9 face detection failure** | Excluded from all paired analysis |

---

## Suggested Skills

| Skill | When |
|---|---|
| `brainstorming` | Before implementing Fp/Fs swap — validate design choices |
| `codegraph_context` | To explore TFG audio entry points for feature injection |
| `remote-gpu-deploy` | When deploying training or inference to the SeetaCloud server |
| `diagnose` | If Fp/Fs pipeline produces unexpected null results |
| `deep-research` | If literature review needed on semantic mediation in speech-driven animation |
| `graphify` | If already indexed (`graphify-out/` exists), for structural queries on the codebase |

---

## Next Session Starter

> "This is a handoff continuation. Read `docs/superpowers/handoff/2026-07-18-wav2sem-fp-fs-experiment.md`. Start with Tier 1: implement oracle Fs (Chinese BERT CLS from transcript) and validate the 2×2 Fp/Fs swap logic on the existing 12 AISHELL-1 samples."
