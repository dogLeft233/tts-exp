## Purpose

为下游实验agent提供固定输入、明确科学端点和可独立复算的实验契约，补足现有证据尚未回答的问题，区分工程验收、探索性信号和已确认的自然音轨replacement收益。

## ADDED Requirements

### Requirement: Frozen common-support cached cohort

The experiment SHALL retain the 133-record and 23-group denominator, reproduce the parent's 115 observed records and common support, and read only bound JSON and feature arrays. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: Corrupt cache is not missingness

- **WHEN** A feature file fails its bound hash or shape validation
- **THEN** The experiment reports BLOCKED rather than treating the record as scientifically missing

### Requirement: Static and dynamic decomposition

The experiment SHALL report total, static and centered dynamic MSE and the fixed time-reversal controls on identical three-arm support. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: Constant mouth-shape bias

- **WHEN** Two trajectories differ by a time-independent coordinate offset
- **THEN** Dynamic error is zero and total error equals static error within the specified numerical tolerance

### Requirement: Timing specificity precedes teacher evidence

The experiment SHALL require both natural and candidate reversal controls to pass the fixed observed-subset timing rule before labeling a dynamic teacher signal. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: No temporal sensitivity

- **WHEN** The natural or candidate reversal control fails its fixed rule, even if the candidate's dynamic benefit is positive
- **THEN** The decision is TIMING_SPECIFICITY_NOT_ESTABLISHED

### Requirement: Missingness and subset conclusions remain separate

The experiment SHALL report both original-denominator missingness bounds and explicitly labeled observed-subset estimates, following the ordered decisions in design.md. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: Observed signal is not full-cohort evidence

- **WHEN** All three observed-subset rules pass but at least one original-denominator lower bound is not positive
- **THEN** The decision is OBSERVED_DYNAMIC_SIGNAL_REQUIRES_NEW_COHORT and parent_gate_repaired and stage02_authorized remain false

### Requirement: Zero-model independent completion

The experiment SHALL use zero model calls, new videos and new scores, and independently recompute support, decomposition, statistics and decision before finalization. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: A well-signed numerical result is false

- **WHEN** A reported decomposition or bootstrap statistic is changed and re-signed
- **THEN** Validation fails from raw-array recomputation and final cannot report complete
