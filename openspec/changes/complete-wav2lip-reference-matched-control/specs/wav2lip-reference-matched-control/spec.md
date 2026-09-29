## Purpose

为下游实验agent提供固定输入、明确科学端点和可独立复算的实验契约，补足现有证据尚未回答的问题，区分工程验收、探索性信号和已确认的自然音轨replacement收益。

## ADDED Requirements

### Requirement: Frozen factorial design

The experiment SHALL evaluate the frozen N/CORRECT by F0/F46 design on all 16 records without selecting new references or changing the natural evaluation audio. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: Reference main effect only

- **WHEN** Changing the reference adds the same anchor gain to N and CORRECT
- **THEN** The difference-in-differences interaction is zero and no audio-increment or replacement claim is produced

### Requirement: Reference-specific matched delay gate

The experiment SHALL validate F46's own delay control in the natural lag domain [-15,15] and delayed lag domain [-10,20], using physical offsets 15-j and 10-j respectively. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: Parent control cannot authorize F46

- **WHEN** P/F0 passes but F46 has fewer than 14 of 16 offsets in the expected difference range [-6,-4] under a correctly implemented control
- **THEN** Execution ends CONTROL_FAILED and F46_C is not generated

### Requirement: Shared natural anchor

The experiment SHALL compute the primary interaction using the F0_N anchor for all four cells and retain the original interaction thresholds. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: Positive interaction with harmful candidate

- **WHEN** The interaction is positive but g1 is negative
- **THEN** Both values are reported and replacement_confirmed remains false

### Requirement: Current runtime validation

The experiment SHALL independently verify input provenance, fresh repeat pixels, parity matrices, both delay domains and all four-cell statistics within the frozen budget of 36 new videos and 54 new scores. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: Cached PASS conceals wrong lag domain

- **WHEN** An old PASS is available but the delayed curve was evaluated with the wrong physical offset mapping
- **THEN** The validator rejects the control and candidate execution remains blocked until the specified calculation is correctly validated

### Requirement: Read-only analysis and honest completion

The experiment SHALL analyze existing complete artifacts without generating media and distinguish scientific stopping from missing or corrupt inputs. The normative execution details are in design.md and ../../../parallel-next-experiments-20260910.md relative to the change root.

#### Scenario: Analyze is missing a cell

- **WHEN** The analyze stage lacks one required factorial cell
- **THEN** The command exits nonzero, invokes no model, and does not issue an interaction conclusion
