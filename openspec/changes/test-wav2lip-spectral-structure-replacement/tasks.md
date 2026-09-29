## 1. Implement and freeze (CPU first)

- [ ] 1.1 Read the four change documents and BM router/experiment instructions; search/read `Wav2Lip spectral structure replacement 2026-09-08`, mark running without altering historical conclusions.
- [ ] 1.2 Implement the small package and fixed parent joins/hash/support audit; freeze 22 IDs, U/PLUS/MINUS, formulas, runtime, stats and code/spec bindings.
- [ ] 1.3 Implement RT/MAG/ENV and diagnostics; independently check reconstruction, ENV time-invariance of log correction, exact PCM/length, non-finite rejection. Freeze candidates before score access.
- [ ] 1.4 Add focused synthetic tests: mismatched joins/hashes, wrong ENV broadcast, RT roundtrip, C/D signs, U versus Q, average-before-min, shared cluster draws, strict CI boundary, nonsignificance versus equivalence, stage gating, resume and all terminal states.

## 2. Run bounded stages

- [ ] 2.1 Verify host CUDA and >=15GiB disk; wire the existing generation/media/scoring helpers without invoking old runners or detection.
- [ ] 2.2 Execute Stage A: 66 fresh videos, 88 fresh cells. Independently validate artifacts/statistics and freeze control_validation.json.
- [ ] 2.3 If controls fail, finalize CONTROL_FAILED with zero B work. Otherwise execute exactly 44 additional videos and 44 cells for MAG/ENV.
- [ ] 2.4 Compute every registered comparison using the specified 95%/97.5% CIs, equivalence conditions and decision order; preserve full per-record data and actual counts.

## 3. Validate and close

- [ ] 3.1 Run independent construction, embedding/matrix, media-identity, statistical and final-state checks; no scientific claim on engineering BLOCKED.
- [ ] 3.2 Run focused pytest, Ruff on the new package/tests and OpenSpec strict validation. Self-review the protocol's joins, geometry, PCM, masks, CI levels, comparison signs and stopping boundaries.
- [ ] 3.3 Write result.md: historical clue, new scoped endpoint, controls, MAG/ENV versus N/RT and each other, effect sizes/CIs, limits, terminal decision and permitted next step. Do not call this independent replication or generalized replacement.
- [ ] 3.4 Update the same BM note after search/full read, append changelog and read back; mark actual result/conclusion/status and artifact paths. Stop after this spec, including scientific failure.
