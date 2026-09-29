## Purpose

为下游实验agent提供固定输入、明确科学端点和可独立复算的实验契约，补足现有证据尚未回答的问题，区分工程验收、探索性信号和已确认的自然音轨replacement收益。

## ADDED Requirements

### Requirement: Immutable cohort and construction

The experiment SHALL retain all 16 records in 8 groups and use the frozen original PHONE_CORE and GENERIC_CORE transforms and input bindings. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: A parent file changes

- **WHEN** A parent hash or mask identity differs from the frozen snapshot
- **THEN** Execution reports BLOCKED with scientific_decision=not_available and generates no candidate

### Requirement: Group-level exposure

The experiment SHALL apply exposure eligibility to source groups while reporting every record's exposure and retaining zero-exposure records in the denominator. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: One record has no exposure

- **WHEN** One record has no U exposure but its paired record in the same group does, and all other groups have exposure
- **THEN** The group passes the exposure gate and both records remain in the full experiment

### Requirement: Unsupported group stops generation

The experiment SHALL stop before model calls when any whole group lacks frozen U exposure or the fixed construction is numerically degenerate. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: Both paired records lack exposure

- **WHEN** Both records of one source group have no exposure
- **THEN** The decision is INPUT_DEGENERATE, all 16 records remain documented, and new video and score counts are zero

### Requirement: Natural baseline precedes mechanism claims

The experiment SHALL apply the frozen natural-baseline gain rule before the matched generic-control mechanism rule. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: Generic comparison alone is positive

- **WHEN** PHONE_CORE beats GENERIC_CORE but fails the full natural-baseline gain rule
- **THEN** The scientific decision is NO_PHONE_CORE_GAIN_ESTABLISHED

### Requirement: Verified bounded completion

The experiment SHALL follow the common execution contract, stay within 34 new videos and 36 new scores, and finish only after independent numerical validation. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: Signed analysis is numerically wrong

- **WHEN** An analysis value is changed and its file or internal hash is recomputed
- **THEN** The validator rejects the value by recomputation; final cannot report complete
