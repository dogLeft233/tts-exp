# Female-speaker replication of the frozen TTS source intervention

Status: input/protocol intent declared before any fresh evaluation effect
summary has been read by the principal. Producer and input hashes must be
sealed separately before replication model forwards.

## Reason

The official AISHELL-1 `resource_aishell/speaker.info` at dataset revision
`bbe295d530192a4cd41644b711c9aecd087df653` identifies all ten calibration
speakers S0052-S0061 and all forty evaluation speakers S0062-S0101 as M.
Preserve that cohort and its results. Independently test the same frozen
generator manipulation on speakers labeled F by this publisher metadata.
The labels describe dataset metadata; they are not inferred from voice.

## Inputs

- Initial evaluation speakers: S0124-S0143, twenty F-labeled speakers.
- Same first-two-lexicographic utterance rule: mono 16 kHz PCM, 3-10 seconds,
  nonempty publisher text containing only Han characters after whitespace
  removal. Use the same transcript/revision and preserve source archive hashes.
- If a speaker lacks two eligible utterances, preserve the rejection and
  advance to the next F-labeled speaker by lexicographic ID. Exclude the
  existing scientific and engineering-pilot speakers/utterances. Freeze the
  resulting forty utterances before TTS or model effects.
- Self-clone each selected utterance with `qwen3-tts-vc-2026-01-22` using its
  publisher text and own natural reference. Preserve original 24 kHz bytes,
  EOF-decoded lengths, canonical 16 kHz PCM and the established resampling
  provenance. Use a new manifest/cache/run directory. No key in artifacts.

## Frozen intervention and scoring

Reuse the male calibration's d, u, global k=3, projection, three random seeds,
RMS target 0.0376838172675746 and source-axis/level four-cell definitions
without fitting female calibration data. This tests transfer of that exact
direction; do not fit a new direction after an unfavorable replication.

Run the existing fixed primary face and its official JPEG/SyncNet frontend.
Create all sixteen intended score rows per utterance (eight level four-cell
rows plus eight generator-axis rows): 640 rows for forty clips/twenty speakers.
Seal the complete row grid, code, input identities and external calibration
references before any forwards; keep one common per-recipient support.
Guard20 is the main endpoint. Empty support is recorded as unavailable for
that view; retain other frozen views and the intended row. No new TTS draw,
per-utterance anchor or changed guard may replace a missing main measurement.

## Analysis

Apply the same four-condition conjunction for orth_plus-baseline and
orth_plus-orth_minus C/C_anchor: all four 99% lower bounds must exceed zero,
with 20,000 PCG64 equal-speaker bootstrap draws, seed 20260930. This is an
external replication; it does not replace the original forty-speaker primary
result or turn missing original endpoints into complete measurements.

Report random-control specificity, raw/unit B/D/search, four-cell G/E/I,
native/processed gap, intervention doses, all failure rows and support counts.
All twenty speakers and their two clips remain the intended denominator;
missing required primary support makes that claim ineligible. Keep complete
pair/speaker estimates separately. Report male and female results separately;
any combined or between-group effect is secondary and labeled explicitly.

For independent visual analysis, mechanically apply the existing cal-frozen
landmark, phone-class, latency and minimum-count rules to recipient natural
MFA clocks. Do not tune female thresholds using effects. State endpoint
availability and metadata scope alongside generator and evaluator results.

Prepare waveforms and cache while the original formal run continues, then
let the sole GPU agent schedule replication after the original dual-face
artifacts are sealed. Keep this replication separate from the WORLD physical
intervention's own reconstruction and scoring-reference domains.
