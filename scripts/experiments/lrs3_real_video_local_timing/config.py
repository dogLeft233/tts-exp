from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
HISTORY_COHORT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json"
HISTORY_COHORT_SHA256 = "b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b"
HISTORY_SOURCE_MANIFEST = REPO / "runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json"
HISTORY_SOURCE_MANIFEST_SHA256 = "34efb614d72e183727d87d40a05807d4c6ccda954cadc14f76bc8cbd01ee255f"

PARENT_SPEC = REPO / "openspec/changes/diagnose-lrs3-real-video-local-timing/specs/lrs3-real-video-local-timing-diagnostic/spec.md"
PARENT_SPEC_SHA256 = "e133cec33ea32ac51ed034e9c4cb46c933b20defcf03fae8abe4b0147f216278"
AMENDMENT_SPEC = REPO / "openspec/changes/fix-lrs3-real-video-tail-contract/specs/lrs3-real-video-tail-contract/spec.md"
AMENDMENT_SPEC_SHA256 = "cf8397a2cf1c97401e2573dc06787313804caa5f22daec508071818bab415aec"
PARENT_BLOCKED_FINAL = REPO / "runs/lrs3_real_video_local_timing_20260905_v2/final.json"
PARENT_BLOCKED_FINAL_SHA256 = "fac113b053693551f6a07eedaa4a35c2ed0a161aacc326693df8ed4cb190a80b"
PARENT_BLOCKED_RESULT_SHA256 = "daf92d08ded41b507ab02bce0882a5d339a8f6d1824706bd59822f7c99399f49"

PROTOCOL_ID = "lrs3_real_video_local_timing"
PROTOCOL_PREFIX = "lrs3_real_video_local_timing"
PROTOCOL_REVISION = "bounded_audio_tail_v2"
EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22
EXPECTED_SAMPLE_ID_SHA256 = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
AUDIO_TAIL_MIN_SAMPLES = -SAMPLES_PER_FRAME
AUDIO_TAIL_MAX_SAMPLES = 2 * SAMPLES_PER_FRAME
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
CROP_SIZE = 224
CROP_SCALE = 0.40
DETECTION_SCALE = 0.25
MIN_FACE_SIZE = 100
MIN_TRACK = 5
TRACK_IOU_THRESHOLD = 0.5
NUM_FAILED_DET = 25

REAL_ARM = "REAL"
REPEAT_ARM = "REAL_REPEAT"
WARP_ARM = "VIDEO_WARP_120"
ARMS = (REAL_ARM, REPEAT_ARM, WARP_ARM)

SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_PYTHON = Path.home() / ".venvs/syncnet/bin/python"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")

VSHIFT = 15
WINDOW_FRAMES = 5
WARP_AMPLITUDE_FRAMES = 3.0
MIN_LOCAL_ROWS = 5
PEAK_GAP_THRESHOLD = 0.010
OFFSET_TOLERANCE_FRAMES = 1
MIN_BASELINE_RECORDS = 20
MIN_SUCCESS_RECORDS = 18

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260905
BLIND_SEED = 20260906
BLIND_RECORD_COUNT = 4

NO_SEALED_MEDIA_TOKENS = ("/test/", "/tests/", "/val/", "/validation/", "/heldout/")


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    validate_run_id(run_id)
    return REPO / "runs" / f"{PROTOCOL_PREFIX}_{run_id}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def protocol(self) -> Path:
        return self.root / "protocol.json"

    @property
    def input_audit(self) -> Path:
        return self.root / "input_audit.json"

    @property
    def media(self) -> Path:
        return self.root / "media"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def review(self) -> Path:
        return self.root / "review"

    @property
    def final(self) -> Path:
        return self.root / "final.json"

    @property
    def result(self) -> Path:
        return self.root / "result.md"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"


@dataclass(frozen=True)
class FrozenConfig:
    schema_version: int = 2
    protocol_id: str = PROTOCOL_ID
    protocol_revision: str = PROTOCOL_REVISION
    sample_rate: int = SAMPLE_RATE
    fps: int = FPS
    samples_per_frame: int = SAMPLES_PER_FRAME
    audio_tail_min_samples: int = AUDIO_TAIL_MIN_SAMPLES
    audio_tail_max_samples: int = AUDIO_TAIL_MAX_SAMPLES
    pcm_channels: int = PCM_CHANNELS
    pcm_sample_width: int = PCM_SAMPLE_WIDTH
    crop_size: int = CROP_SIZE
    crop_scale: float = CROP_SCALE
    detection_scale: float = DETECTION_SCALE
    min_face_size: int = MIN_FACE_SIZE
    min_track: int = MIN_TRACK
    vshift: int = VSHIFT
    window_frames: int = WINDOW_FRAMES
    warp_amplitude_frames: float = WARP_AMPLITUDE_FRAMES
    min_local_rows: int = MIN_LOCAL_ROWS
    peak_gap_threshold: float = PEAK_GAP_THRESHOLD
    offset_tolerance_frames: int = OFFSET_TOLERANCE_FRAMES
    min_baseline_records: int = MIN_BASELINE_RECORDS
    min_success_records: int = MIN_SUCCESS_RECORDS
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    blind_seed: int = BLIND_SEED
    blind_record_count: int = BLIND_RECORD_COUNT
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT
    expected_sample_id_sha256: str = EXPECTED_SAMPLE_ID_SHA256
    syncnet_model_sha256: str = SYNCNET_MODEL_SHA256

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "arms": list(ARMS),
                "warp_formula": "s[n]=n+3*sin(2*pi*n/(F-1)); endpoints forced; nearest-even integer j[n]",
                "offset_column_formula": "offset=15-column_index",
                "pcm_policy": "unchanged 16 kHz mono PCM16 bytes",
                "audio_tail_policy": {
                    "revision": PROTOCOL_REVISION,
                    "delta_samples": "d=N-640*F",
                    "min_samples": AUDIO_TAIL_MIN_SAMPLES,
                    "max_samples": AUDIO_TAIL_MAX_SAMPLES,
                    "comparison": "integer",
                    "extended_tail_reason": "unattributed",
                },
                "bootstrap_policy": "sorted source groups, NumPy default_rng/PCG64, percentile 2.5/97.5",
                "forbidden_operations": [
                    "audio_modification",
                    "audio_resampling",
                    "global_offset_alignment",
                    "tts_generation",
                    "wav2lip",
                    "training",
                    "score_based_retry",
                    "record_filtering",
                    "heldout_or_sealed_media_access",
                    "history_overwrite",
                ],
            }
        )
        return payload


def expected_cell_count() -> int:
    return EXPECTED_RECORD_COUNT * len(ARMS)
