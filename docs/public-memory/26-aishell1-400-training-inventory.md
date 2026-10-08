---
title: 26-aishell1-400-training-inventory
type: note
permalink: tts-exp/docs/experiments/26-aishell1-400-training-inventory
---

# 26 — AISHELL-1 400-pair waveform-training inventory and recovery ledger

## Purpose

This document is the canonical recovery anchor for the Chinese paired corpus used by the waveform-enhancer work. It records the artifact lineage, quality gates, split contract, and provenance needed to resume the experiment after a context compaction. The result directories named below are local working-tree artifacts; their presence is not implied by Git history. Do not use this ledger as evidence of a downstream result unless a separate fixed Ditto/SyncNet report says so.

## Corpus lineage

The recovery chain is:

```text
400 deterministic candidate pairs
    → 392 strict MFA pairs (8 quarantined)
    → 784 condition-level records (392 natural + 392 TTS)
    → 391 usable weak-target items (291 train + 50 valid + 50 test)
```

| Gate | Artifact | Result |
|---|---|---|
| Candidate selection | `results/rhythm_style_500/source_manifests/aishell1_test_400_candidates.json` | 400 pairs, 8 speakers, 50 per speaker |
| Strict MFA | `results/rhythm_style_500/source_manifests/aishell1_test_400_strict_mfa.json` | 392 accepted, 8 quarantined |
| Condition expansion | `results/rhythm_style_500/source_manifests/aishell1_test_400_condition_mfa.json` | 784 records: 392 natural + 392 TTS |
| Dataset build | `results/rhythm_style_500/aishell1_test_400/rhythm_style_dataset/dataset_manifest.json` | 391 accepted: 291 train / 50 valid / 50 test |

The condition manifest is the authoritative speaker split for the paired training inventory. The dataset-stage count is lower than 392 because one pair was rejected while constructing the weak target; this is a separate gate, not a strict-MFA quarantine.

## Exact manifest fingerprints

These SHA-256 values are recovery checks. Recompute them before treating a local artifact as the same input.

| Manifest | SHA-256 |
|---|---|
| `aishell1_test_400_candidates.json` | `44c3b7c4d880ec14600b5fbad3e14a9cd25140fd616feeac77a93c99beb98d18` |
| `aishell1_test_400_strict_mfa.json` | `c50208107c0c451aeff1cde9f1b8d3c7bb5f692f25063dec249485f477be83a5` |
| `aishell1_test_400_condition_mfa.json` | `1090a1964214810b9350cc02cf40702384d0a6226750e0d3d503e1c5374ac0ce` |
| `dataset_manifest.json` | `00b7af2e68671ba7fc586c3867521107f118ce7d7e5d1bc2f4bb81fc3761cb8b` |

## Split and leakage contract

The intended speaker-disjoint split is:

```text
train:   S0901, S0906, S0912, S0913, S0914, S0915
valid:   S0765
test:    S0770  (heldout)
```

`test` is the heldout split in the waveform trainer. It must not be used for training, normalization, prototype/statistic fitting, checkpoint selection, early stopping, or loss-weight selection. Any downstream evaluation must happen only after the checkpoint and all choices are fixed. Paired keys, source hashes, transcripts, and speaker groups must remain disjoint across splits.

The older `data/wav2sem_analysis_zh` 13-pair corpus is a separate legacy experiment and must not be merged into this AISHELL-1 inventory.

## Artifact directories

The recovered directory layout is:

```text
results/rhythm_style_500/aishell1_test_400/
├── natural/                 # 16-kHz mono natural WAVs
├── tts/                     # Faster-Qwen3 source WAVs
├── mfa/                     # independent natural/TTS MFA alignments
├── weak_targets/            # exact-length phone-local weak targets + metadata
├── embeddings_full/         # historical embedding sidecars/cache
└── training_runs/full_mvp/  # historical Chinese waveform MVP outputs
```

Additional source manifests are under:

```text
results/rhythm_style_500/source_manifests/
```

Historical feature-domain MVP outputs are under:

```text
runs/tts_feature_head_full/
```

The prior strict Chinese downstream report is separate:

```text
runs/aishell1_strict_20260707T081223Z/05_report/report.md
```

## Audio and alignment contracts

- Natural audio is 16-kHz mono.
- Faster-Qwen3 TTS source audio is 24-kHz mono before any explicit training-side resampling. The current frozen-HuBERT waveform trainer consumes 16-kHz audio, so a Chinese run must make this conversion explicit and auditable; it must not silently reinterpret a 24-kHz file as 16 kHz.
- Weak targets are 16-kHz mono and have exactly the natural waveform sample count.
- Natural and TTS MFA alignments are independent. Phone spans must be present and pass the configured confidence/coverage gate.
- Missing pairs, invalid spans, transcript mismatches, missing audio, non-finite audio, and unsupported sample rates fail closed.

## Strict-MFA quarantine

All eight candidates below were rejected by the strict-MFA gate for `phone_coverage_below_0.90`:

| source utterance | phone coverage |
|---|---:|
| `BAC009S0906W0267` | 0.8182 |
| `BAC009S0906W0284` | 0.4839 |
| `BAC009S0912W0365` | 0.6667 |
| `BAC009S0912W0399` | 0.1364 |
| `BAC009S0913W0405` | 0.3571 |
| `BAC009S0913W0465` | 0.5667 |
| `BAC009S0915W0190` | 0.5000 |
| `BAC009S0915W0390` | 0.7660 |

These are not to be repaired with uniform alignment or silently replaced by another pair.

## Separate dataset-stage rejection

`BAC009S0913W0314` (`sample_id: 276`) passed the strict-MFA inventory but was rejected during weak-target dataset construction because its weak-target coverage was `0.5897436`. This rejection explains the 392 → 391 transition and must remain distinct from the eight strict-MFA quarantines.

## Embedding-cache caveat

`embeddings_full` contains 784 sidecar files associated with the 392 natural records. It must not be described as a complete paired natural/TTS HuBERT feature cache. The current waveform trainer recomputes raw TTS HuBERT layer-6 features from the TTS waveform and aligns them through independent MFA phone spans. Cache filenames alone are not proof of paired feature completeness.

## Historical runs versus the new objective

`results/rhythm_style_500/aishell1_test_400/training_runs/full_mvp/` is a historical Chinese waveform MVP. It predates the current frozen-HuBERT explicit raw-TTS objective and must not be presented as equivalent training. `runs/tts_feature_head_full/` is a feature-domain residual adapter using train-derived phone prototypes; it is not a waveform enhancer and did not establish a Ditto/SyncNet result. The strict Chinese report at `runs/aishell1_strict_20260707T081223Z/05_report/report.md` documents an earlier downstream TTS-vs-natural baseline, not an enhancer result.

## Proximity telemetry and completed run

The first Chinese explicit-style run is archived in [`27-aishell1-hubert-waveform-training.md`](27-aishell1-hubert-waveform-training.md). It used a fresh strict rebuild of 351 pairs: 259 train, 45 valid, and 47 heldout. The fresh rebuild excluded 41 items for complete-audio alignment or coverage gates; these exclusions are recorded in `runs/hubert_waveform_aishell1_20260810/targets/target_manifest.json`. The run completed three epochs, but enhanced HuBERT layer-6 features were not stably closer to aligned raw-TTS features on train or valid. This is a training-style feasibility result only, not heldout generalization or downstream TFG evidence.

## Proximity telemetry for future runs

The new waveform trainer can report, for train and valid only, the masked distance from natural and enhanced HuBERT layer 6 features to aligned raw-TTS layer 6 features. Natural, enhanced, raw-TTS, aligned-TTS, and the MFA mask already exist in the training path, so this telemetry adds masked reductions and zero additional HuBERT encoder calls. It is descriptive only: it is not a loss, checkpoint-selection criterion, or proxy claim for Ditto/SyncNet improvement. Heldout proximity must remain a final evaluation metric after all choices are fixed.

Each run should record that the telemetry uses existing encoder outputs and `extra_encoder_calls: 0`. Train/valid aggregate summaries should be finite JSON and include matched-frame count, coverage, cosine/Smooth-L1/combined distances, progress toward TTS, and the fraction of items where enhanced is closer than natural.

## Fail-closed recovery rules

- Do not substitute a missing pair, speaker, TTS provider, or alignment.
- Do not use a uniform fallback alignment.
- Do not invent or rebalance the speaker split without recording a new manifest and hash.
- Do not merge the legacy 13-pair corpus into this inventory.
- Do not use test/heldout records for training or selection.
- Do not write credentials, tokens, raw private data, or raw audio into documentation, Git, or reports.
- Use fresh output directories for new runs; do not overwrite historical artifacts.

This tracked document is the durable recovery ledger; local result directories remain subject to working-tree cleanup and should be revalidated by the hashes above.