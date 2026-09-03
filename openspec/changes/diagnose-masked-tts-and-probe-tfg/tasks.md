## 1. Diagnose the existing result

- [x] 1.1 Load the exploratory parent manifests, per-cell losses, saved predictions, masks, natural mels, normalization, and phone centroids; verify the parent is complete and read-only.
- [x] 1.2 Produce the exact patch/weighted-velocity decomposition and the two-negative-group comparison required by `design.md`; write `00_diagnosis/analysis.json` and `report.md`.

## 2. Freeze the TFG probe

- [x] 2.1 Select and freeze one maximum-mask-coverage record from each of the eight evaluation groups with the specified hash tie-break; add focused tests for selection and mel-core placement.
- [x] 2.2 Build the 80 required natural/three-arm/three-seed mel drivers from saved predictions, preserving natural mel outside evaluated cores.

## 3. Render and score

- [x] 3.1 Implement the small direct-mel Wav2Lip wrapper, pass natural-mel chunk parity, and render all 80 cells with the frozen checkpoint.
- [x] 3.2 Strictly replace every render's audio with untouched natural audio and score the complete matrix with frozen official SyncNet V2.

## 4. Analyze and report

- [x] 4.1 Compute group-level token/modality Sync-C and Sync-D gains, bootstrap intervals, win counts, seed summaries, and the predeclared exploratory status.
- [x] 4.2 Run focused tests and `openspec validate diagnose-masked-tts-and-probe-tfg --strict`; report the output path, matrix completeness, decision, and claim boundary.
