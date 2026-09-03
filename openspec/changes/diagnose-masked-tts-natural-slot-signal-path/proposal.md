## Why

The independent 16-record confirmation showed that the trained reconstructor uses its TTS input and reacts to within-phone order, but it did not show a stable visual advantage for the correct TTS occurrence over a same-phone wrong occurrence. The analysis also paired reconstruction controls by record and seed instead of by mask, which changes the reported reconstruction values even though the pass status remains unchanged. Before more training, more records, or waveform decoding, locate where the wrong-instance signal disappears and test the missing oracle control: natural-audio WavLM features in the existing TTS input slot.

## What Changes

- Correct and regression-test mask-level reconstruction pairing without rerunning training, Qwen TTS, Wav2Lip, or SyncNet.
- Trace existing paired-versus-control differences through input features, predicted mel cores, record drivers, rendered mouth regions, and SyncNet scores using the completed new-record artifacts; treat rendered RGB differences as response magnitudes, not causal bottleneck proof.
- Add one frozen `NATURAL_WAVLM_IN_TTS_SLOT` oracle condition on the same 16 records and three trained checkpoints; reuse existing paired and zero-input cells.
- Preflight a better-controlled wrong-instance test using same-record or same-source-group occurrences matched by phone context and duration. Run its downstream stage only if the frozen eligible subset covers every record and all eight groups under the fixed rule.
- Keep all work diagnostic and read-only with respect to parent artifacts. Add no retraining, model change, Qwen synthesis, dataset search, or waveform decoder.

## Capabilities

### New Capabilities

- `masked-tts-natural-slot-signal-path`: Correct the reconstruction audit, localize signal loss, test natural-reference features in the TTS slot, and test a more tightly matched wrong-instance control.

### Modified Capabilities

None.

## Impact

- Reuses `runs/lrs3_masked_tts_new_confirmation_20260902/` and the hard-negative checkpoints under `runs/lrs3_masked_tts_trajectory_specificity_20260902/` as immutable inputs.
- Adds one experiment runner, focused tests, and a new run directory. Existing paired, zero-input, reversed, and original wrong-instance artifacts are reused wherever their mask support matches.
- The natural-slot condition adds 48 downstream cells. The matched-control stage is conditional: it uses three donors for cheap reconstruction robustness, then adds at most 96 downstream cells for one predeclared best-ranked donor and its common-mask paired baseline.
- No result from this change opens the waveform-decoder gate or proves audible feature retention, deployability, or population generalization.
