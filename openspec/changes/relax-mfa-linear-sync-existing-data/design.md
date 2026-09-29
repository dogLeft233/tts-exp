## Context

See `proposal.md` and `specs/mfa-linear-sync-existing-data/spec.md` for the reason and behavior contract. The current prepared inventory has 102 eligible records across 10 source groups after the unchanged structural and MFA-linear checks. The earlier 200-record/source-group-heldout run is intentionally immutable and remains a separate blocked experiment.

This change relaxes only the evaluation unit and denominator: it uses unseen record IDs while allowing source-group overlap. It does not relax natural-audio isolation, SyncNet correctness, audio QC, split policy, or artifact immutability.

## Goals / Non-Goals

**Goals:**

- Run a fresh 94-record shared fit and evaluate 8 deterministic record-heldout rows using the assets already available.
- Preserve the exact waveform, differentiable loss, official scoring, and replacement contracts from the prior prototype.
- Make source-group overlap explicit and prevent the result from being described as source-group generalization.
- Complete the real-video gate and conditionally run the existing 72-cell strict replacement matrix.

**Non-Goals:**

- Claiming transfer to unseen source groups, speakers, or a sealed LRS3 test set.
- Reusing the four-record checkpoint or modifying the blocked 200/40 run.
- Adding a natural-audio reconstruction loss, tuning the SyncNet objective, or changing per-record thresholds.
- Recovering missing MFA/video assets, opening validation/test splits, or duplicating records.
- Running multiple seeds or changing the 100-step compute budget in this data-unit comparison.

## Decisions

### 1. Use 94 training records and 8 record-heldout evaluations

Run the existing structural and detached natural-reference preflight first. From the complete 102-row eligible inventory, select eight evaluation rows with a fixed SHA256 key, preferring one deterministic representative per source group for broad coverage; select the remaining 94 rows for training. The evaluation IDs are disjoint from training IDs, but source groups are intentionally allowed to overlap. Persist the overlap list and the exact record-level claim scope.

Alternative: preserve source-group disjointness by evaluating eight groups. Rejected because the remaining two groups contain only eight records and would leave an uninformative shared fit. Alternative: use all 102 records for training and evaluate old scores. Rejected because it would expose every evaluated ID to optimization and invalidate the new result.

### 2. Keep one fresh model and one fixed optimization budget

Instantiate the same model with the fixed seed and run exactly 100 steps. Each step computes the equal-weight mean loss across the 94 locked records using the existing differentiable MFCC/SyncNet path. Save step-zero and step-100 state and do not search checkpoints or retry. This isolates the relaxed evaluation unit from architecture and optimizer changes.

Alternative: warm-start from the prior shared-four checkpoint. Rejected because it makes the data-unit comparison depend on an earlier fit and violates the fresh-training contract.

### 3. Reuse current official real-video and replacement seams

Use the prior paired real-video evaluator for the 8 selected rows, but implement a new aggregate decision with record-heldout status names to avoid conflating it with the old source-group-heldout result. The six-of-eight and median `0.003` gate remains unchanged. If it passes, reuse the current three driver arms and 3×3 matrix implementation, with 72 cells and natural-audio-column metrics.

Alternative: relax QC or accept incomplete scoring because the run is exploratory. Rejected because that would make improvements uninterpretable and could reproduce the earlier engineering/scientific ambiguity.

### 4. Fail closed and report the changed claim boundary

Use a new run root such as `runs/lrs3_mfa_linear_sync_existing_data_20260903`. A complete engineering-valid miss is a scientific negative; missing assets, curves, or matrix cells are not evaluated. Every report must state that source groups may overlap and that this is only record-heldout within the fit-only inventory.

## Risks / Trade-offs

- **Training and evaluation can share source groups.** → Lock IDs before training, report group overlap explicitly, and prohibit unseen-group language.
- **The 8-row evaluation remains small.** → Keep the descriptive six-of-eight rule, report every row, and make no inferential claim.
- **Existing evaluation records have been seen in prior analyses.** → Use a new deterministic selection key and treat the result as a bounded follow-up, not an independent replication.
- **Candidate audio may fail QC on an unseen record.** → Keep the original QC and classify an incomplete cohort as not evaluated instead of hiding the row.
- **The full replacement matrix is expensive.** → Run only after the real-video gate passes and require all 72 cells.

## Migration Plan

1. Run focused tests and strict OpenSpec validation for this additive change.
2. Lock the current 102 eligible records into 94 training and 8 record-heldout evaluation IDs.
3. Run fresh 100-step training and candidate inference once.
4. Score all 8 real-video pairs and emit the record-heldout decision.
5. Run replacement only if the real-video decision passes.
6. Validate the terminal artifact graph and write the bounded report. Leave the 200/40 blocked run and all prior artifacts untouched.
