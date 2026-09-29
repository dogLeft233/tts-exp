## 1. Protocol and implementation

- [x] Add the short-phone/nominal-support protocol proposal and design.
- [x] Add deterministic overlap-shared frame selection without changing natural rows or silence handling.
- [x] Record the ownership, support, and feature-tail policies in Stage 00 and DTW traces.

## 2. Tests and validation

- [x] Add synthetic shared-boundary and short-phone mapping tests.
- [x] Run focused DTW tests and lint.
- [x] Validate the new OpenSpec change strictly.

## 3. Frozen rerun

- [x] Build revised Stage 00 in the new run root.
- [x] Generate and validate all 133 revised candidates.
- [x] If Stage 01 is complete, continue with the existing diagonal gate; otherwise preserve a blocked terminal artifact.
