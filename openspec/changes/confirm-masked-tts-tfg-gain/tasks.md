## 1. Freeze the confirmation

- [x] 1.1 Build the 16-record confirmation manifest by excluding the first probe's eight records; require two remaining records in each source group.
- [x] 1.2 Reuse the existing mel builder to create and validate the complete 160-cell driver matrix.

## 2. Render and score

- [x] 2.1 Render all cells with the frozen direct-mel Wav2Lip path, strictly replace audio with untouched natural audio, and score with frozen official SyncNet V2.

## 3. Decide

- [x] 3.1 Compute the predeclared record→group modality and token contrasts, assign one confirmatory status, and write the minimal outputs and claim boundary.
- [x] 3.2 Run focused tests and `openspec validate confirm-masked-tts-tfg-gain --strict`.
