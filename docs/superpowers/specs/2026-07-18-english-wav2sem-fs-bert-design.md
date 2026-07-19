# English Wav2Sem Fs-BERT Analysis — Design

**Date**: 2026-07-18
**Status**: Draft (awaiting user review)
**Scope**: Bilingual Wav2Sem analysis was originally planned; this spec covers only the **English** half. The Chinese half is deferred (see Future Work).

## 1. Background and Motivation

### 1.1 What already exists

The repo has completed Phase 1 (`scripts/14`–`18`) on 12 Mandarin AISHELL-1 samples, plus Phase 2 (`scripts/19`–`25`) on G×E decomposition, identity control, and stability perturbation. Key Chinese-phase findings:

- HuBERT L11 `segment_stability`: natural vs TTS paired-permutation FDR q = 0.024, Cohen's d = −1.045 (large effect).
- G×E interaction = +9.07 (Sync-C), the dominant unexplained effect after identity control.
- TTS advantage on Ditto: ΔSync-C = +1.246 (large), gate passes.

### 1.2 What the English half already has

The English multi-seed baseline already ran on the server at `/root/autodl-tmp/experiments/english-wav2sem-20260718/`:

- 13 LibriSpeech test-clean samples (matched to Chinese durations within 5 ms).
- 3 seeds (42, 43, 44) × 13 samples = 39 paired runs through Ditto + SyncNet.
- Gate 1 verdict: `FAIL_EVALUATOR_OR_INTERACTION_DRIVEN`.
  - ΔSync-C mean = +0.160, sd = 0.517, Cohen's dz = 0.310, bootstrap 95% CI [−0.106, +0.423], sign-flip p = 0.286, Holm p = 0.571.
  - G×E generator effect = −4.194 (no generator main effect).
  - Conclusion: TTS advantage does **not** replicate on English in Ditto.

### 1.3 Gap this spec fills

The Wav2Sem-style Fs analysis on the English 13 samples has **not** been run. Specifically missing:

- SSL frame embeddings (script 15) on the 26 English wavs (13 natural + 13 TTS).
- Feature separability metrics (script 16) on the English embeddings.
- Oracle Fs from BERT CLS (no equivalent script exists yet).
- Fd construction and Fs-mediated separability gain (no equivalent script exists yet).
- Per-sample separability ↔ ΔSync-C correlation report (script 30 currently does only Gate 1).

This spec adds those missing steps.

## 2. Scientific Hypotheses

The bilingual contrast was originally designed to make the Wav2Sem mediation test orthogonal: mechanism existence (English H1/H2) vs mechanism effect (Chinese H3). With Chinese deferred, this spec reduces to a single-language test on English:

- **H1 (separability existence)**: TTS audio produces more separable viseme embeddings than natural audio on the 13 English LibriSpeech test-clean samples (HuBERT L11 segment_stability and related metrics).
- **H2 (oracle Fs decoupling)**: Injecting oracle Fs (BERT CLS) into Fp via additive fusion Fd = FC(Fs⊕Fp) increases viseme separability beyond Fp-only.
- **H3 (link to TFG)**: Per-sample separability diagnostics correlate with ΔSync-C.

Interpretation logic under the single-language English setup:

- If H1 holds but H3 does not (the expected case given the Gate 1 failure) → Wav2Sem Fs decoupling mechanism exists on English but is **not** a sufficient condition for a TTS lip-sync advantage. This is a meaningful negative result that constrains the Wav2Sem theory's scope.
- If H1 fails on English → the Chinese L11 d = −1.045 finding is language-specific, plausibly tied to Mandarin homophone density.
- If H3 holds on English despite Gate 1 failing → separability captures something SyncNet sees but the G×E generator effect does not; would warrant re-examining the G×E interpretation.

## 3. Architecture: Components and Data Flow

All components communicate only via files on disk. No component passes in-memory state to another. Each is independently re-runnable; each skips if its output exists and inputs are unchanged.

```
[27] Download LibriSpeech test-clean
       └─→ data/wav2sem_analysis_en/manifest/librispeech_phn/{spk-chap}/{utt}.phn

[28] Build English alignment manifest
       in : librispeech_phn/*.phn + audio_en/manifest.json + audio_en_qwen3_tts/manifest.json
       └─→ data/wav2sem_analysis_en/manifest/alignment.json

[15] SSL frame embeddings (reuse existing script, language-agnostic)
       in : alignment.json + 26 wavs at data/data/audio_en/ + data/data/audio_en_qwen3_tts/
       └─→ data/wav2sem_analysis_en/embeddings/{id}_{cond}_{variant}_{model}.npy/.json

[16] Feature separability (reuse existing script, language-agnostic)
       in : embeddings/ + alignment tokens
       └─→ data/wav2sem_analysis_en/metrics/separability_metrics.json

[29] Oracle Fs = BERT CLS (new)
       in : alignment.json (text field) + bert-base-uncased checkpoint
       └─→ data/wav2sem_analysis_en/metrics/oracle_fs.json

[31] Fd construction + Fs-mediated separability gain (new)
       in : embeddings/ (Fp = HuBERT L11) + oracle_fs.json
       └─→ data/wav2sem_analysis_en/metrics/oracle_fd_separability.json

[30] Render report (extend existing script)
       in : separability_metrics.json + oracle_fs.json + oracle_fd_separability.json
            + existing gate1.json from server
       └─→ docs/experiments/wav2sem_bilingual_report_en.md
            data/wav2sem_analysis_en/figures/*.png
```

### 3.1 Component responsibilities

| # | Script | What it does | New or reused |
|---|---|---|---|
| 27 | `27_download_librispeech_phn.py` | Download test-clean.tar.gz from OpenSLR, extract, copy 13 sample .phn/.wrd/.txt/.flac by `librispeech_id` lookup | New |
| 28 | `28_prepare_english_alignment.py` | Parse .phn files, map ARPABET → Preston Blair 13 visemes, write alignment.json with phone+viseme tokens | New |
| 15 | `15_extract_ssl_embeddings.py` | As-is, just pointed at English manifest | Reused (no code change) |
| 16 | `16_feature_separability.py` | As-is | Reused (no code change) |
| 29 | `29_oracle_fs_bert.py` | Encode each sample's text via `bert-base-uncased`, take CLS + mean-pooled token embeddings as two Fs candidates, compute cosine distance between natural and TTS Fs | New |
| 31 | `31_oracle_fd_separability.py` | Construct Fd at HuBERT L11 with untrained orthogonal-init FC; compute separability at three levels (Fp-only / Fd_zero=Fs+Fp directly / Fd_random=FC(Fs+Fp)) | New |
| 30 | `30_analyze_english_wav2sem.py` | Extend current Gate1-only analyzer with Wav2Sem modules B (separability), C (oracle Fs), D (Fd gain), E (link to Sync-C) | Modified (add modules, keep Gate1) |

## 4. Data Product Layout

```
data/wav2sem_analysis_en/
  manifest/
    alignment.json                       # script 28
    librispeech_phn/                     # script 27
      2830-3980-0036.phn
      2830-3980-0036.wrd
      2830-3980-0036.trans.txt
      ...
  embeddings/                            # script 15
    1_natural_raw_hubert.npy
    1_natural_raw_hubert.json
    ...
    13_tts_raw_xlsr.npy
    13_tts_raw_xlsr.json
  metrics/
    separability_metrics.json            # script 16
    oracle_fs.json                       # script 29
    oracle_fd_separability.json          # script 31
  figures/
    boundary_sharpness_en.png            # script 30B
    segment_stability_en.png
    fs_gain_scatter_en.png
  report.md                              # script 30 (top-level)

docs/experiments/
  wav2sem_bilingual_report_en.md         # script 30 (copy/links; main report)

data/data/
  english_viseme_map.yaml                # ARPABET → Preston Blair 13 (new)
```

Outputs of steps that already ran (Gate 1) are not duplicated; script 30 references them in place at `/root/autodl-tmp/experiments/english-wav2sem-20260718/runs/english_wav2sem_gate1_gate1.json`.

## 5. Configuration

Extend existing `scripts/configs/english_wav2sem.yaml` with a new `wav2sem:` block. Existing fields stay unchanged.

```yaml
wav2sem:
  ssl_models: [hubert, xlsr]
  layers: [0, 6, 11, 12]
  target_bert: bert-base-uncased
  alignment_source: librispeech_phn           # new; one of: librispeech_phn, mfa
  viseme_system: preston_blair_13             # new
  oracle_fs_input: transcript_text            # new; one of: transcript_text, transcript_asr
  oracle_fs_pooling: [cls, mean]               # new
  fd_fc_layers: 2                             # new
  fd_fc_init: orthogonal                      # new; one of: orthogonal, identity, zero
  fd_random_seed: 42                          # new; for FC orthogonal init reproducibility
  fdr_alpha: 0.05
  permutation_n: 10000
```

## 6. Algorithmic Specifications

### 6.1 LibriSpeech .phn parsing (script 28)

LibriSpeech native `.phn` format (whitespace-separated, one phone per line):

```
0 470000 h#
470000 580000 k
580000 700000 ae
...
```

Fields: `start_sample end_sample phone_label`. Sample rate is 16000 Hz. Conversion to seconds: `start_s = start_sample / 16000`. Silence labels are: `h#`, `sil`, `sp`, `spn`, `""`. These are dropped from the viseme classification but kept as time markers for segment boundary analysis.

### 6.2 ARPABET → Preston Blair 13 visemes

The mapping table is defined in a new YAML file `data/data/english_viseme_map.yaml`. The 13 Preston Blair classes (with their canonical members) are:

| Viseme key | Phones (ARPABET) | Description |
|---|---|---|
| `pbmv` | P, B, M | Lips together |
| `fv` | F, V | Lower lip to teeth |
| `th` | TH, DH | Tongue between teeth |
| `cdsz` | T, D, S, Z, N, L | Tongue to alveolar ridge |
| `kg` | K, G, NG | Back of tongue to soft palate |
| `chjsh` | CH, JH, SH, ZH | Blade to post-alveolar |
| `e` | EH, AE | Relaxed open mouth |
| `o` | AA, AO, OW, UH, OY | Open round |
| `i` | IY, IH | Wide stretch |
| `u` | UW, W | Small round |
| `o` | UH, AA, AO, OW, OY | Open round  |
| `r` | R, ER, AXR | Slightly cupped |
| `ai` | AY, EY | Wide open to close |
| `aw` | AW | Round to wide |
| `sil` | h#, sil, sp, spn (silence) | Dropped from viseme metric |

Some phones are ambiguous in Preston Blair (notably UH, which some sources put with `u` and some with `o`). This spec commits to `UH → o` (lax back vowel groups with open round). The full definitive table lives in the YAML `data/data/english_viseme_map.yaml`; any phone not in the map is classified as `other` and excluded from viseme separability metrics but kept in phone-level metrics (a separate metric level `arpabet` is computed without merging).

### 6.3 Oracle Fs extraction (script 29)

```python
text = manifest_entry["text"]  # already upper-case LibriSpeech gold transcript
text_lower = text.lower()      # BERT tokenizer convention

inputs = tokenizer(text_lower, return_tensors="pt", truncation=True, max_length=128)
with torch.no_grad():
    outputs = bert(**inputs)

# Fs candidates
Fs_cls = outputs.last_hidden_state[0, 0].cpu().numpy()                # (768,) CLS
Fs_mean = outputs.last_hidden_state[0, 1:-1].mean(0).cpu().numpy()    # (768,) mean-pooled

# Sanity metrics
fs_norm = float(np.linalg.norm(Fs_cls))
fs_empty_l1 = float(np.linalg.norm(Fs_cls - Fs_baseline_empty))  # baseline = BERT("")
```

Per-sample output:

```json
{
  "sample_id": 1,
  "condition": "natural",
  "text": "WHEREVER THE MEANS OF GRACE ARE FOUND...",
  "text_lower": "wherever the means of grace are found...",
  "token_count": 14,
  "Fs_cls": [...768 floats...],
  "Fs_mean": [...768 floats...],
  "fs_cls_norm": 6.43,
  "fs_empty_l1": 2.87,
  "is_degenerate": false
}
```

Degenerate flag: `fs_cls_norm` outside ±5σ of the 26-sample distribution, or `fs_empty_l1 < 0.5`.

In addition, the script computes natural↔TTS Fs similarity per sample:

```json
{
  "sample_id": 1,
  "cosine_cls_nt": 0.971,
  "cosine_mean_nt": 0.963,
  "l1_cls_nt": 1.234
}
```

### 6.4 Fd construction and separability gain (script 31)

```python
import torch.nn as nn

# Fp = HuBERT L11 frame embeddings, shape (T, 768)
# Fs = oracle Fs_cls, shape (768,)
# Note: Fp comes from the .npy at embeddings/1_natural_raw_hubert.npy
#       layers index 2 (because layers=[0,6,11,12] → index 2 = L11)

# Two FC variants for control:
# 1. Fd_zero: Fs added directly to Fp, no projection (sanity baseline)
# 2. Fd_random: untrained orthogonal-init 2-layer MLP

Fs_broadcast = np.tile(Fs, (T, 1))           # (T, 768)
Fd_zero = Fs_broadcast + Fp                    # (T, 768)

torch.manual_seed(cfg_fd_random_seed)
fc = nn.Sequential(
    nn.Linear(768, 768),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(768, 768),
)
for m in fc.modules():
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight)
        nn.init.zeros_(m.bias)
fc.eval()

Fd_random = fc(torch.tensor(Fs_broadcast + Fp, dtype=torch.float32)).detach().numpy()
```

Then for each of `Fp`, `Fd_zero`, `Fd_random`, run the same separability metrics as script 16 at the `viseme` and `arpabet` levels. Output schema:

```json
{
  "sample_id": 1,
  "condition": "natural",          // or "tts"
  "fp_metrics": {"intra": 0.74, "silhouette": -0.09, "boundary_sharpness": 0.33, "segment_stability": 0.32, ...},
  "fd_zero_metrics": {...},
  "fd_random_metrics": {...},
  "gain_zero_vs_fp": {"intra": -0.02, "silhouette": +0.01, ...},
  "gain_random_vs_fp": {...},
  "gain_random_vs_zero": {...}    // the Fs-specific contribution
}
```

The `gain_random_vs_zero` is the key signal: positive on `silhouette`, negative on `intra_class_dist` would support H2 (Fs adds decoupling beyond what additive noise alone would give).

Paired test across 13 samples (natural vs TTS) is not done at script 31 level. Script 31 computes per-sample per-condition diagnostic. Script 30 then does the cross-condition and cross-sample statistics.

## 7. Failure Modes and Mitigations

### 7.1 Four pre-identified failure modes

| # | Failure | Detection | Mitigation |
|---|---|---|---|
| M1 | LibriSpeech .phn end time > local audio duration (samples downloaded out of order, or manifest.json librispeech_id mismatched with downloaded archive) | script 28 asserts `last_phn_end_s ≤ audio_duration_s + 0.1s` per sample; ≥ 2 violations abort the batch | When mismatch detected, log both timestamps and the manifest.json librispeech_id; continue with failed samples marked `phn_unmatched`; if ≥ 2 fail, abort and re-run script 27 with `--verify-metadata` to check IDs match |
| M2 | BERT CLS collapses on short LibriSpeech sentences (≥ 1 sample has ||Fs|| > 5σ from mean, or L1 distance to `BERT("")` baseline < 0.5) | script 29 computes distribution stats over 26 samples; flags outliers `is_degenerate` | Degenerate samples are still written but excluded from 30C aggregates; report them in a separate table; if > 20% degenerate, suggest using `Fs_mean` instead of `Fs_cls` and re-run |
| M3 | Fd_random's untrained orthogonal FC introduces spurious separability (orthogonal init alone could decorrelate embedding axes enough to look "more separable") | script 31 computes three levels: `Fp`, `Fd_zero = Fp + Fs` (no projection), `Fd_random = FC(Fp + Fs)` (orthogonal FC); H2 is supported only when the Fs-specific contribution `gain_random_vs_zero` is in the metric-direction-favorable direction **for at least 3 of the 5 core metrics** (silhouette↑, boundary_sharpness↑, segment_stability↑, intra_class_dist↓, fisher↑), as a paired sign test across 13 samples; if only `Fd_random > Fp` holds but `Fd_zero ≈ Fp`, the gain is an FC artifact not an Fs effect | If `Fd_zero` does not improve over `Fp` but `Fd_random` does, the gain is FC artifacts not Fs. Report this honestly in 30D as `M3_FC_ARTIFACT`; do not claim H2 supported |
| M4 | HuBERT/XLS-R model not cached on server, or wrong model loaded (local_files_only=True returns whatever is found) | script 15 already validates layer index vs `model.config.num_hidden_layers`; add check `model.config.model_type in {"hubert", "wav2vec2"}` and hidden_size == 768 | If model files missing, use `HF_ENDPOINT=https://hf-mirror.com` and re-download to `/root/autodl-tmp/checkpoints/`; if wrong model loaded, raise |

### 7.2 Silence handling

LibriSpeech `.phn` labels silence as `h#`, `sil`, `sp`, `spn`, and sometimes empty strings. script 28 filters silence tokens out of the viseme metric but keeps them in the alignment array as time anchors for `boundary_sharpness` (the boundary at the silence→speech transition is informative).

### 7.3 Statistical robustustness

- All tests use BH-FDR corrected q-values at α = 0.05.
- 13-sample paired permutation tests run with n = 10000 permutations.
- Effect sizes (Cohen's d, dz) are reported alongside p-values; underpowered trends (large d, non-significant p) are labeled `trend` and not declared significant.
- Per-sample separability ↔ ΔSync-C correlation uses Spearman ρ and 10000-permutation test.
- Multiple comparison across (model × layer × metric × level) is corrected by BH-FDR.

### 7.4 Re-entry and caching

All scripts honor the existing caching pattern:

- Output exists AND output mtime ≥ input mtime → skip
- `--no-cache` flag forces overwrite
- Progress is checkpointed every 5 samples (write partial JSON, atomic rename at end)
- No script depends on prior script's in-memory state

## 8. Testing Strategy

### 8.1 Test matrix

| Test target | Type | How |
|---|---|---|
| LibriSpeech .phn parsing (27/28) | Unit smoke | One sample round-trip; token count equals `len(text.split()) + 1` accounting for `h#` markers; viseme assigned for every non-silence phone |
| ARPABET → PB13 map (28) | Unit smoke | All 39 ARPABET phones have a viseme assignment; silence labels excluded |
| BERT CLS encoding (29) | Unit smoke | Same text twice → identical Fs (determinism); short text `"HELLO WORLD"` → ||Fs|| in empirical range [4, 8] |
| Fd separability gain (31) | Unit smoke | All-same-label case → all separability metrics = 0 (catches implementation bugs that mistake variance for signal) |
| Full pipeline | e2e smoke | `--smoke` runs sample 1 only through 27 → 28 → 15 → 16 → 29 → 31 → 30; outputs 5 JSON + 1 MD with all required fields |

### 8.2 Smoke commands

```bash
cd /root/autodl-tmp/experiments/english-wav2sem-20260718
export HF_ENDPOINT=https://hf-mirror.com
export PATH=/root/autodl-tmp/envs/ditto/bin:$PATH
PY=/root/autodl-tmp/envs/ditto/bin/python

$PY scripts/27_download_librispeech_phn.py --samples 1 --smoke
$PY scripts/28_prepare_english_alignment.py --samples 1 --smoke
$PY scripts/15_extract_ssl_embeddings.py \
    --manifest data/wav2sem_analysis_en/manifest/alignment.json \
    --models hubert --layers 0,11 --smoke \
    --output-dir data/wav2sem_analysis_en/embeddings
$PY scripts/16_feature_separability.py \
    --embeddings-dir data/wav2sem_analysis_en/embeddings \
    --output-dir data/wav2sem_analysis_en/metrics --smoke
$PY scripts/29_oracle_fs_bert.py --samples 1 --smoke
$PY scripts/31_oracle_fd_separability.py --samples 1 --smoke
$PY scripts/30_analyze_english_wav2sem.py \
    --runs-root runs --run-prefix english_wav2sem_gate1 \
    --seeds 42,43,44 --with-wav2sem
```

### 8.3 Acceptance criteria

| Acceptance item | Location | Required value |
|---|---|---|
| 13 English samples have .phn tokens | `data/wav2sem_analysis_en/manifest/alignment.json` | All 13 entries (26 with TTS) have non-empty `tokens[]` |
| 26 SSL embedding .npy files exist | `data/wav2sem_analysis_en/embeddings/` | 26 HuBERT + 26 XLS-R = 52 .npy files |
| separability_metrics contains HuBERT L11 + L0 | `separability_metrics.json` | `layers` array includes both 0 and 11 |
| 26 oracle Fs vectors | `oracle_fs.json` | 13 natural + 13 TTS entries with `Fs_cls` of length 768 |
| Three-tier Fd separation | `oracle_fd_separability.json` | Keys `fp_metrics`, `fd_zero_metrics`, `fd_random_metrics` present per sample |
| H1/H2/H3 verdict | `report.md` | All three explicitly stated with either PASS/FAIL/TREND/disabled |

### 8.4 No tests required for

- script 15/16 metrics implementation — already validated on Chinese samples; reused unchanged.
- BERT tokenizer — HuggingFace official implementation; trust upstream.
- LibriSpeech .phn file format — Kaldi standard, simple whitespace parsing.

## 9. Dependencies and Environment

### 9.1 Python packages

All already installed in `/root/autodl-tmp/envs/ditto/`:

- `transformers` (BERT, HuBERT, XLS-R)
- `torch` (≥ 2.0)
- `numpy`, `scipy`, `scikit-learn`
- `librosa`, `soundfile`
- `pyyaml`
- `tqdm`

No new pip installs required.

### 9.2 Model checkpoints

| Model | Size | Storage | Source |
|---|---|---|---|
| `facebook/hubert-base-ls960` | ~360 MB | already cached at server (used by gate1) | HF mirror |
| `facebook/wav2vec2-large-xlsr-53` | ~1.2 GB | already cached | HF mirror |
| `bert-base-uncased` | ~440 MB | new; `/root/autodl-tmp/checkpoints/bert-base-uncased/` | `HF_ENDPOINT=https://hf-mirror.com` |

### 9.3 Data downloads

| File | Size | Source |
|---|---|---|
| LibriSpeech test-clean.tar.gz | 346 MB | `https://www.openslr.org/resources/12/test-clean.tar.gz` |

Disk check: server `/root/autodl-tmp` has 7.8 GB free (per `df -h` in gate1 log). Adding 346 MB (LibriSpeech) + 440 MB (BERT) + ~700 MB embeddings (26 × 2 × 4 × float32 × ~100 frames × 768 dim ≈ 92 MB per condition, doubled for safety) = ~1.5 GB additional. Within budget.

## 10. Schedule Estimate

| Step | Estimate | Notes |
|---|---|---|
| 27 download + 28 alignment | 0.5 day | Mostly I/O bound |
| 15 embeddings (26 × 2 models × 4 layers) | 2 hours GPU | Single GPU, batch 1 |
| 16 separability | 1 hour CPU | Pure numpy/scipy |
| 29 oracle Fs | 0.5 hour GPU | 26 BERT forward passes |
| 31 Fd separability | 1 hour CPU | Separability on 3 variants × 26 samples |
| 30 render report | 1 hour | Markdown + matplotlib |
| Total | ~1 working day | Conservative; can be 1.5 if M1/M2/M3 fires |

## 11. Future Work (out of scope this spec)

- Chinese-side Fs-BERT analysis: same scripts (`29_oracle_fs_bert.py`, `31_oracle_fd_separability.py`) and a Chinese BERT (`bert-base-chinese`); becomes useful if H1 differs across languages.
- Training audio→Fs TCN+Transformer (true Wav2Sem): only justified if oracle Fs shows a measurable decoupling effect.
- Fp/Fs swap into the Ditto `wav2feat` injection point: implementation-ready per monkey-patch analysis (`SDK.wav2feat.wav2feat`), but only meaningful if oracle Fs is shown to matter.

## 12. References

- Wav2Sem CVPR 2025 paper: `papers/Wav2Sem_CVPR2025.pdf`
- Existing Chinese mechanism report: `docs/experiments/tts_tfg_mechanism_report.md`
- English Gate 1 output (server): `/root/autodl-tmp/experiments/english-wav2sem-20260718/runs/english_wav2sem_gate1_gate1.json`
- LibriSpeech format spec: `https://www.openslr.org/resources/12/`
- Preston Blair 13 visemes: standard animator reference; see mapping in `data/data/english_viseme_map.yaml`