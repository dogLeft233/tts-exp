# HDTF Fixed-Image Rerun Design

## Goal

Re-run the completed HDTF paired self-clone experiment with the same fixed
images used by the Chinese and MDC experiments, to isolate whether the prior
result was affected by using a frame extracted from each HDTF source video.

## Scope

- Keep the HDTF v2 13-sample natural audio, source videos, sample IDs, ASR
  transcripts, and generated Qwen3-TTS audio unchanged.
- Use `data/data/image/{1..13}.png` as the Ditto source image for both
  `natural_raw` and `tts_raw`.
- Keep Ditto settings, seed, no-loudnorm policy, and SyncNet V2 evaluation
  unchanged.
- Write results to a new run; do not overwrite `hdtf_paired_full_v2`.
- Re-run Ditto, SyncNet, and report generation. ASR and TTS are reused rather
  than regenerated because the only independent variable is the image.

## Data Flow

1. Create a new run config whose `audio_dir` and `tts_audio_dir` point to the
   HDTF v2 source run and whose `image_dir` is `data/data/image`.
2. Copy the source run metadata into the new run and annotate the image
   protocol as `fixed_existing_images`, preserving the source run reference.
3. Run `03_ditto.py` for `natural_raw` and `tts_raw`.
4. Run `04_eval.py` with the same SyncNet model and completeness requirement.
5. Run `05_report.py` and compare Sync-C/Sync-D direction against the HDTF v2
   source-frame result.

## Acceptance Criteria

- All 13 sample IDs use `data/data/image/{id}.png`.
- The source natural/TTS audio hashes match HDTF v2.
- Both raw conditions produce complete Ditto and SyncNet results, or every
  failure is recorded with its sample ID and cause.
- The report states the fixed-image protocol and retains the HDTF/Ditto
  overlap caveat.
- The original HDTF v2 artifacts remain unchanged.
