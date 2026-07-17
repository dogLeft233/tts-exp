# Wav2Sem-Style Feature Analysis

## Experiment Goal

Wav2Sem-style feature separability analysis on 12 AISHELL-1 samples.
Assess whether SSL-derived speech representations (HuBERT / XLSR) and
acoustic features systematically separate natural speech from TTS outputs.
The primary research question: are natural and synthetic speech samples
distinguishable in high-dimensional embedding space, and if so, which
layers / models yield the strongest separability signal?

## Data Conventions

| Field       | Values                                       |
|-------------|----------------------------------------------|
| Sample IDs  | 1–8, 10–13 (study); 3, 6, 8, 10, 12 (pilot) |
| Condition   | `natural`, `tts`                             |
| Variant     | `raw` (untouched), `gain_matched` (LUFS)     |
| TTS engines | `faster_qwen3` (primary), `f5_tts` (validation) |
| SR          | 16000 Hz                                     |
| Target LUFS | −23.0 dB (EBU R128)                         |

Sample 9 is excluded from all analysis because its SyncNet alignment
failed during the pre-processing stage.

## Pipeline Dependencies

Scripts in `scripts/` that feed this analysis:

- **`08_audio_features_v2.py`** — extracts 35+ acoustic features (MFCC,
  formants, spectral, energy, F0) from all natural + TTS audio files;
  outputs `data/r4_results/audio_features_v2.csv`.
- **`12_transformation_experiment.py`** — runs the five-condition
  interventional experiment (raw/dur/loud/both/gain) that defines the
  `raw` and `gain_matched` variants used in Wav2Sem comparisons.
- **`04_eval.py`** — SyncNet evaluation; provides Sync-C / Sync-D
  scores used to filter out failed samples.

## Pipeline Stages

1. **Data** — assemble manifest of all natural + TTS audio files with
   condition/variant labels.
2. **Alignment** — forced alignment (Montreal Forced Aligner) to
   produce phone-level time boundaries and token-level features.
3. **Embeddings** — extract HuBERT / XLSR layer-wise embeddings for
   every frame; store as `EmbeddingRecord` objects.
4. **Separability** — compute pairwise separability metrics (silhouette,
   nearest-neighbour, CKA) between natural/tts groups within matched
   acoustic windows.
5. **Link** — join acoustic features (stage 1) with SSL embeddings
   (stage 3) via time-aligned token spans.
6. **Report** — summary statistics, permutation tests, bootstrap CIs;
   per-layer significance tables; UMAP / PCA visualisations.

Outputs land under `data/wav2sem_analysis/{manifest,alignments,
embeddings,metrics,figures}/`.
