# HDTF Paired TTS-vs-Natural Run Design

## Scope

Run the same Chinese-baseline self-clone protocol on 13 matched HDTF
audio/video samples. This is an exploratory paired-source experiment, not a
strict unseen evaluation, because HDTF is part of Ditto's published evaluation
ecosystem.

## Source Selection

- Use the 321 audio/video name intersections under HDTF `_audio_raw` and
  `_videos_raw`.
- Restrict candidates to `WRA_` and `WDA_` clips, which are talking-face source
  videos rather than radio-only records.
- Normalize trailing numeric suffixes to a speaker key and keep at most one clip
  per speaker.
- Sort by decoded audio duration ascending and select the first 13 distinct
  speakers. The current deterministic candidate list starts with Bill Cassidy,
  Joe Heck, Kristi Noem, Pat Roberts, Jon Kyl, Jaime Herrera Beutler, Olympia
  Snowe, Austin Scott, Bob Corker, Joe Pitts, Cory Gardner, John Thune, and
  Lamar Alexander.
- If a selected source fails face validation, replace it with the next unused
  candidate in this deterministic order and record the replacement.

## Preprocessing

For every selected matched pair:

1. Download the matching MP4 from the NAS to a temporary staging directory.
2. Use the matching local HDTF audio file and the video duration to select a
   deterministic centered 6-second interval.
3. Crop the audio interval to mono PCM16 WAV at 24 kHz.
4. Extract the video midpoint frame for that interval as the Ditto source image.
5. Validate that the image contains a usable face and record the source hashes,
   crop start/duration, and frame timestamp.

Raw downloaded videos remain outside the repository and may be removed after
frame extraction. The run metadata retains NAS paths and hashes for provenance.

## TTS/TFG Protocol

- ASR: Qwen3-ASR Flash through DashScope, using `DASHSCOPE_API_KEY` only in the
  process environment.
- TTS: local `Qwen/Qwen3-TTS-12Hz-0.6B-Base`, English ICL self-clone, seed 42.
- Reference audio: the cropped HDTF natural audio.
- Reference text and target text: the ASR transcript of the same crop.
- TFG: Ditto offline TRT configuration used by the Chinese baseline.
- Conditions: `natural_raw` and `tts_raw` only.
- Loudness normalization: disabled.
- Evaluation: SyncNet V2, requiring complete pairs for the primary report.

## Outputs

Each run has a distinct ID and contains:

- `00_pairs/natural/1..13.wav`;
- `00_pairs/images/1..13.png`;
- `01_transcript/1..13.txt` and `transcript.json`;
- `pair_manifest.json` with NAS provenance, hashes, crop parameters, and ASR
  text;
- standard `02_tts`, `03_ditto`, `04_eval`, and `05_report` outputs;
- `run_metadata.json` labeling the design as `hdtf_paired_self_clone` and the
  evaluation as exploratory.

## Interpretation Boundary

The primary comparison is within each matched source pair: TTS minus natural
under the same extracted HDTF frame. The result must not be described as
strictly unseen or as a clean estimate of general TTS quality. HDTF provenance
and Ditto evaluation-set overlap must remain visible in every report.
