# Cross-TFG original-PCM evaluation contract

This contract evaluates an audio transform as a generator condition while
scoring every generated video with the paired original natural waveform. The
enhanced waveform is never muxed or passed as SyncNet audio.

## Frozen `plan.json`

The plan is frozen before candidate confirmation scores are opened. It binds:

- `schema_version=cross-tfg-audio-eval-v1`, `run_id`, `status=FROZEN`, Git
  revision, split-manifest/config SHA-256, and canonical-payload
  `plan_sha256` (self-field excluded).
- Ordered `fit_speaker_ids`, `dev_speaker_ids`, and
  `confirm_speaker_ids`, plus ordered historical-exposure speaker/sample ID
  sets and their source manifest. The three speaker splits are disjoint.
  Historical exposure may be assigned to fit or dev and must be marked on each
  row; no historically exposed speaker or sample may enter confirmation.
  Historical exposure does not justify replacing or dropping a confirmation
  speaker after the plan is frozen. Current roster eligibility is Mandarin
  male AISHELL-1: fit S0052–S0101 (50 historically exposed speakers), ten
  frozen dev speakers, and the remaining forty S0002–S0051 speakers for
  confirmation. Exact source-PCM rows remain inventory-bound and must be
  frozen before any new effects are opened. Reused canonical PCM hashes across
  split rosters are rejected.
- `models`: `model_id`, exact `audio_frontend`, `role` (`candidate` or reserved
  `primary_transfer`), model/code/score-model/preprocessing SHA-256 values and
  rational FPS.
- `methods`: `identity`, fixed RMS control, and each prespecified candidate;
  every method has a config SHA-256 and optional checkpoint SHA-256. A
  `rms_fixed` method also freezes `target_rms` and absolute
  `target_rms_tolerance`; the consumer checks generated float32 RMS against
  that target. Training seeds belong to fit/config metadata and are distinct
  from render/evaluation seeds stored on result rows. `generation_seed_family`
  separately predeclares ordered render seeds by split. Development evaluation
  uses at least three paired seeds to measure generation variability before
  model selection; the frozen plan supplies the exact seed IDs. A fit-only plan
  may omit evaluation seeds, and a dev-only plan may have an empty confirmation
  roster with no confirmation family or endpoints. Confirmation seeds are
  frozen only with the later confirmation plan. Every planned
  sample/model/face context expands to the complete split seed family crossed
  with every method. A failed seed/method remains a planned row and makes the
  paired context unavailable; it does not shrink the family. If plan rows omit
  a declared seed or method from an otherwise represented context, the consumer
  records an integrity issue and counts missing units as unestimable instead
  of dropping them or discarding other dev descriptions. Per-seed SyncNet
  metrics are computed first, then seed/face contexts are averaged within
  sample and samples within speaker. The efficacy family includes every
  planned method, including failed runs.
- `rows`: one row per evaluated
  `split × sample_id × model_id × face_id × seed × method_id`. Fit inventories
  and checkpoints may be separately bound to the plan; evaluation rows are
  for dev/confirm. Each row records `historical_exposed`, canonical
  `natural_pcm`, `score_audio`, source face/video, generation settings and
  method/model identities. `score_audio` is byte-for-byte the same canonical
  PCM reference as `natural_pcm`.
- `generation_repeat_qc`: one direct-drive identity repeat per model, separate
  from efficacy rows and all denominators. A deterministic model declares
  exact artifact/matrix equality; otherwise a repeat-noise protocol declares
  the development repeats and reports variation without silently excluding
  the model. This section is empty for a fit-only plan with no development
  evaluation rows. Each attempt binds video/file and elementary-stream hashes, a
  timestamp sidecar path/SHA, crop-policy SHA, and a sealed raw-distance NPZ.
  The consumer independently checks decoded RGB frame difference, query-map
  identity, maximum matrix difference and absolute Cbest delta against the
  baseline identity row.
- Supports, score protocol, mux/crop/ROI policy, score preprocessing,
  contrasts, endpoints, the simultaneous family and shared speaker-bootstrap
  seed/draw count. Lag offsets are fixed to `-15..15`; `Cbest`, `Canchor`,
  `B`, `Dmin`, `D0`, best lag and AV offset are retained. The lag column is
  `k=lag_offsets_frames[best_column]`; official AV offset is `-k`.
  `best_lag_frames` and `official_av_offset_frames` are both written, and
  `offset_frames` aliases the official AV offset. No lag-based video/audio
  retiming is permitted.
- `protocol.syncnet.score_reduction` pins the score reconstruction runtime and
  exact operation sequence: `backend=pytorch_cpu`, exact `torch_version`,
  `num_threads=1`, `matrix_layout=query_by_lag_float32`,
  `stacked_layout=lag_by_query_contiguous_float32`,
  `curve_operation=torch.mean(torch.stack(query_distance_vectors, dim=1), dim=1)`,
  `median_operation=torch.median(curve)`,
  `minimum_operation=torch.min(curve, dim=0)`,
  `median_semantics=lower_middle_order_statistic`, and
  `contrast_dtype=float32`. The consumer applies the same PyTorch operation
  sequence to saved distances on CPU and records the actual runtime version and
  thread count. Producer summaries are checked at the frozen tolerance; CPU/GPU
  reduction bit identity is not assumed.
- `protocol.syncnet.primary_audio_input` freezes the primary SyncNet waveform as
  a direct PCM16 conversion of the canonical natural float32 PCM, cropped to the
  exact source-video time span selected for the SyncNet face track. It records
  `sample_rate_hz=16000`, `channels=1`, `codec=pcm_s16le`, the FFmpeg binary
  SHA-256, the exact conversion argv template, and the adapter/source SHA-256.
  The source-video FPS must map each selected frame boundary to an integer
  source sample index. This primary input is assembled with the selected crop
  video into a new Matroska file whose video elementary stream is copied from
  that exact crop and whose PCM16 audio bytes equal the direct natural-source
  window. The old AVI/MP3/redecoded crop-audio route, if retained, is a separate
  `native_pipeline_diagnostic`; it never supplies the primary score matrix.

Canonical audio is non-empty mono 16 kHz IEEE-float32 little-endian. Each WAV
reference has `path`, file SHA-256, `pcm_sha256` (hash of contiguous decoded
float32 little-endian sample bytes), rate, channel count and sample count.
`natural_pcm` is the canonical natural waveform. The plan's `score_audio`
reference must exactly match it. The original external input and deterministic
resampling/downmix policy are separately recorded as provenance.

## `results.jsonl`

There is exactly one result row for each planned row, in exact plan order.
Rows repeat all identity/config hashes, natural/score PCM hashes, source
face/video hashes, and `status` (`PASS`, `FAILED`, `INCOMPLETE`). A non-PASS
row retains a non-empty `failure_reason`; the row is never omitted. A failed
row may have no generator output. A PASS row has:

- `generator_pcm`: its actual enhanced condition audio (canonical float32,
  exactly the original sample count), with file and decoded PCM hashes. The
  rendered video record binds the same generator PCM hash.
- `render`: silent video, final mux, video elementary-stream hashes, FPS,
  frame count, rational timestamp sidecar, crop/ROI sidecar, crop policy,
  selected SyncNet track ID/index, and `selected_track_crop` with the exact crop
  video file/elementary-stream refs plus a sealed frame map. The map has one row
  per crop frame under `cross-tfg-track-frame-map-v2`, with
  `crop_frame_index`, `source_clock_frame_index`, reduced `timestamp_num` /
  `timestamp_den`, `render_frame_index`, and `frame_origin`. The source-clock
  coordinate is the nominal full-video frame lattice and remains contiguous.
  The map payload also repeats `selected_crop_video_path`,
  `selected_crop_video_file_sha256`, `selected_crop_video_elementary_sha256`,
  `selected_track_id`, `selected_track_index`, `render_video_path`,
  `render_video_file_sha256`,
  `render_video_elementary_sha256`, `render_timestamps_sha256`, `fps_num`, and
  `fps_den`; the consumer checks each against the enclosing result.
  `frame_origin=decoded_render` requires `render_frame_index` to equal the
  in-range source-clock index; `frame_origin=nominal_clock_outside_render_decode`
  requires a null render index and a source-clock index beyond decoded render
  frames. This records AVI/crop tail padding without inventing a rendered frame.
  The nominal source-clock span drives the audio-window derivation.
- `mux_audio_readback`: codec `pcm_f32le`, mono 16 kHz, exact sample count and
  decoded float32 PCM hash. This hash must equal the natural PCM hash exactly;
  the mux operation must leave the video elementary stream unchanged.
- `score`: exact model/preprocessing pins and `score_audio_pcm_sha256` equal to
  the full natural float32 PCM SHA. `primary_pcm16_window` records the separately
  derived PCM16 WAV (`path`, file SHA, decoded PCM SHA, sample rate, channels,
  count), its `source_float32_pcm_sha256`, exact half-open sample bounds, source
  frame bounds (nominal source-clock coordinates), rational PTS bounds, and the
  pinned conversion command/tool/code.
  The requested end sample is derived from the selected crop's exclusive source
  frame stop. If that point lies past the available natural PCM because the
  rendered/cropped video track is longer, the actual end is clipped to the
  natural sample count. Record both `requested_stop_sample_exclusive` and the
  clipped `stop_sample_exclusive`, plus `tail_clipped_to_natural_pcm`; do not
  pad audio or retime video. Keep the full video-frame map and let the score's
  valid/common-support masks represent any resulting short tail.
  `primary_syncnet_input` records the fresh Matroska passed to the official
  evaluator: container/file SHA, copied crop-video elementary-stream SHA,
  PCM16 audio codec/hash/count/rate/channels. The consumer checks the map against
  the nominal source-video clock and decoded-render rows against the full render
  timestamp sidecar, recomputes both sample bounds from rational PTS,
  independently converts the full natural float32 WAV to
  PCM16 with the pinned FFmpeg command, slices the exact sample window, and
  checks those bytes against both the referenced PCM16 WAV and the scorer
  Matroska audio stream. `score_preprocessed_pcm16_sha256` and
  `score_preprocessed_sample_count` refer to this primary window only. An
  optional `native_pipeline_diagnostic` may record the distinct AVI/MP3-derived
  bytes and metrics, clearly outside the primary estimand. A sealed NPZ contains
  the full per-query × 31-lag matrix, exact query/video frame maps and validity
  mask, plus per-support curves/counts/metrics. Curves retain float32-reduced
  `B`, `Dmin`, `D0`, `Cbest`, `Canchor`, `best_lag_frames`, and official
  `official_av_offset_frames` (also named `offset_frames`). PCM16 is never
  described as bit-exact preservation of the original float32 waveform.

The score NPZ arrays are `query_frame_index:int32[N]`,
`query_timestamp_num:int64[N]`, `query_timestamp_den:int64[N]`,
`track_frame_index:int32[N]`, `valid_query_mask:bool[N]`,
`distance_by_query_lag:float32[N,31]`, and
`lag_offsets_frames:int32[31]`. Each array reference binds key, dtype, shape
and SHA-256 of contiguous C-order array bytes. Rational times are reduced to
canonical numerator/denominator before joining.
`query_frame_index` and `track_frame_index` use the global nominal source-clock
frame coordinate. NPZ row position `i` is the crop-local SyncNet query/window
start. For every valid query, the producer maps `i` through
`selected_track_crop.frame_map[i]`; both frame arrays and its reduced rational
timestamp must equal the mapped source-clock frame and PTS. The separately
recorded decoded render frame distinguishes real rendered frames from padded
tail frames. Matrix row count is exactly
`max(0, min(crop_video_frames, floor(primary_pcm16_sample_count/640)) - 5)`;
valid rows are checked against the frame-map entry at their row position. This
keeps paired source-time identity stable when crop starts differ.

For each sample/model/face/seed context, `FULL` is the ordered intersection
of valid `(query frame, reduced rational timestamp, track frame)` keys across
**all planned methods**. A failed/incomplete method makes that paired context
unavailable; it is not removed before intersection. `GUARD20` removes the
first 20 and last 20 FULL queries; if fewer than 41 exist, it is empty. Empty
and short supports remain recorded and ineligible. The plan freezes any common
ROI policy. If a model only supports its native ROI, native-official results
must be a separately named estimand and must not be presented as the
common-ROI comparison.

The consumer rebuilds every nonempty curve and metric from the NPZ using the
frozen CPU PyTorch float32 score reduction and compares producer summaries at
the frozen tolerance. An independent NumPy float64 reduction is retained as a
diagnostic only and never replaces the primary score.
A single PCG64 speaker-index matrix is shared across every confirmatory endpoint; the frozen
simultaneous max-absolute-studentized family is never shortened. Failed rows,
missing models and incomplete support remain in planned denominators. Stage-A
dev summaries are exploratory and do not enter the confirmatory family. A
development-only plan with no confirmatory endpoints reports the simultaneous
family as `NOT_APPLICABLE`, not as an incomplete family.

Optional visual and content guards are separately declared in the plan. When
required, the visual artifact binds a pinned landmark model/config, rational
frame clock, valid masks, aperture/width/jaw measures, aperture velocity/jerk,
face tracking and upper-face displacement. These are diagnostics, not a
“larger is better” target. Content checks compare ASR of natural and enhanced
condition audio against the same frozen reference/model; speech changes do
not count as a synchronization gain because the final mux uses natural PCM.

The result manifest binds plan payload/raw SHA, JSONL raw SHA, row totals,
failure totals, model/method counts and canonical payload `manifest_sha256`.
The independent consumer writes a new audit JSON, readable summary, all
speaker contrasts, shared bootstrap draws and an output manifest; it never
modifies producer inputs and refuses to overwrite output.
