# Multilingual TTS-vs-Natural Run Design

## Scope

Run the existing Chinese TTS-TFG protocol on five downloaded MDC languages:
English, German, Italian, Spanish, and Korean.

Each language uses 13 deterministic samples, paired with the existing portrait
images `1..13`. The run remains the existing self-clone protocol: each natural
audio clip is both the Qwen voice-cloning reference and the natural comparison
arm. This is not a held-out-content generalization experiment.

## Protocol

- TTS: local `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, ICL voice cloning.
- TTS seed: 42.
- TTS language: the manifest's Qwen language token for each language.
- TTS reference audio: the selected natural sample.
- TTS reference text and target text: the selected manifest text, preserving
  alignment with the existing Chinese self-clone experiment.
- TFG: Ditto offline TRT checkpoint/config used by the Chinese baseline.
- Conditions: `natural_raw` and `tts_raw` only.
- Loudness normalization: disabled, matching the strict Chinese baseline.
- Evaluation: SyncNet V2, reporting Sync-C and Sync-D.
- Pairing: require complete natural/TTS pairs before reporting.
- Replicates: run one smoke sample first, then one full 13-sample run per
  language with the fixed seed.

## Sample Selection

The manifest contains 20 samples per downloaded language. The runner selects 13
samples deterministically. The English subset has only four samples at least
5 seconds long, so a hard 5-second cutoff would make the requested 13-sample
comparison impossible.

1. Sort by decoded duration descending.
2. Use manifest order as the stable tie-breaker.
3. Select the first 13 records.
4. Record the selected manifest IDs, source paths, texts, durations, speaker IDs,
   SHA256 values, and image mapping in the run directory.

Samples shorter than 5 seconds remain eligible but are explicitly flagged in the
run metadata and report. Italian is explicitly reported as a single-speaker
condition.

## Isolation

The remote execution uses a new staging directory and does not overwrite the
existing remote repository, which currently contains unrelated uncommitted
changes. The staging directory receives only the required current scripts,
MDC data, manifest, and portrait images. Existing remote `third_party`, conda
environments, and checkpoints are reused through a symlink or configured
absolute paths.

Each language receives a distinct run ID and output tree. No credentials are
stored in the repository, run metadata, or command files.

## Data Flow

1. Read `data/mdc_tts/manifest.json`.
2. Select and canonicalize each language's 13 natural clips to numeric IDs
   `1..13` in an isolated per-language input directory. Canonical audio is
   mono PCM16 WAV at 24 kHz, with no loudness normalization.
3. Create gold-text transcript files from the manifest; do not call ASR.
4. Run the existing TTS stage.
5. Run Ditto on the same image for natural and TTS audio.
6. Run SyncNet for both conditions.
7. Generate the existing paired report and a machine-readable run manifest.

## Validation and Failure Handling

- Smoke must produce one natural video, one TTS video, and two parseable SyncNet
  result files before full execution starts.
- The full run stops before the next language if smoke fails.
- Missing audio, missing images, invalid audio decoding, failed TTS, failed
  Ditto inference, incomplete pairs, and failed SyncNet parsing are recorded and
  cause the affected language run to be marked incomplete.
- Reports must preserve per-sample results and must not pool languages into a
  single inferential sample count.

## Interpretation

Reports must label the design as `self_clone`. Cross-language absolute SyncNet
rankings are not treated as validated because SyncNet and Ditto's audio encoder
are not established as language-neutral. The primary comparison is within each
language: TTS minus natural under the same image and inference configuration.
