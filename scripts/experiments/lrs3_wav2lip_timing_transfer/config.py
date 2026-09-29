from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]

PROTOCOL_ID = "lrs3_wav2lip_timing_transfer"
PROTOCOL_REVISION = "cross_domain_v1"
SPEC = REPO / "openspec/changes/diagnose-lrs3-wav2lip-timing-transfer/specs/lrs3-wav2lip-timing-transfer-diagnostic/spec.md"
SPEC_SHA256 = "9514383e714624d23d0a1f85ce561aa93a5621bb85daf44df5fd946dc1a8df34"
PROPOSAL = REPO / "openspec/changes/diagnose-lrs3-wav2lip-timing-transfer/proposal.md"
PROPOSAL_SHA256 = "7c68378766286dff1caf009eb5c6df78507f95ee84e6503915ae6c653264c42f"
DESIGN = REPO / "openspec/changes/diagnose-lrs3-wav2lip-timing-transfer/design.md"
DESIGN_SHA256 = "22f349daaf7fb987a0c6d17a385d17364ada0db6fcbe73592beaeb8ad164d696"

COHORT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json"
COHORT_SHA256 = "b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b"
CALIBRATION_FINAL = REPO / "runs/lrs3_local_timing_control_calibration_20260905_v5/05_final/final.json"
CALIBRATION_FINAL_SHA256 = "318054ecc86aab5e0ea9b19fa9d5a6e0bcd81181ec200ac9e23f6abca0c08d8b"
AUDIO_MANIFEST = REPO / "runs/lrs3_local_timing_control_calibration_20260905_v5/02_audio/audio_manifest.json"
AUDIO_MANIFEST_SHA256 = "0617cb895574516b77aa430c44f0480eefe4be0e0582b3e8fdec4addf46e68e1"
VIDEOS_MANIFEST = REPO / "runs/lrs3_local_timing_control_calibration_20260905_v5/03_videos/videos_manifest.json"
VIDEOS_MANIFEST_SHA256 = "5157148c2aed4f797beffa2c832784991ba474c58e86e6391b1f01040c38e941"
TAIL_FINAL = REPO / "runs/lrs3_real_video_local_timing_20260905_tail_v2/final.json"
TAIL_FINAL_SHA256 = "5df3615f7b2219e60ad62fb49792cb30eccd068441d1ab5b8006d0fb92aee24d"
TAIL_PROTOCOL = TAIL_FINAL.with_name("protocol.json")
ORDERED_SAMPLE_ID_SHA256 = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
CROP_SIZE = 224
CROP_SCALE = 0.40
WARP_AMPLITUDE_SAMPLES = 1_920
WARP_AMPLITUDE_FRAMES = 3.0

AUDIO_N = "N"
AUDIO_W = "W"
AUDIO_ARMS = (AUDIO_N, AUDIO_W)
VIDEO_R = "R"
VIDEO_GN = "G_N"
VIDEO_GW = "G_W"
VIDEO_ARMS = (VIDEO_R, VIDEO_GN, VIDEO_GW)
MAIN_CELL_SPECS = (
    (VIDEO_R, AUDIO_N),
    (VIDEO_R, AUDIO_W),
    (VIDEO_GN, AUDIO_N),
    (VIDEO_GN, AUDIO_W),
    (VIDEO_GW, AUDIO_N),
    (VIDEO_GW, AUDIO_W),
)
REPEAT_CELL_SPECS = ((VIDEO_R, AUDIO_N), (VIDEO_GN, AUDIO_N))

EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22
EXPECTED_MAIN_CELL_COUNT = EXPECTED_RECORD_COUNT * len(MAIN_CELL_SPECS)
EXPECTED_REPEAT_CELL_COUNT = EXPECTED_RECORD_COUNT * len(REPEAT_CELL_SPECS)
EXPECTED_TOTAL_CELL_COUNT = EXPECTED_MAIN_CELL_COUNT + EXPECTED_REPEAT_CELL_COUNT

VSHIFT = 15
WINDOW_FRAMES = 5
MIN_LOCAL_ROWS = 5
PEAK_GAP_THRESHOLD = 0.010
OFFSET_TOLERANCE_FRAMES = 1.0
MIN_BASELINE_RECORDS = 20
MIN_SUCCESS_RECORDS = 18
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260905

AUDIO_TAIL_MIN_SAMPLES = -SAMPLES_PER_FRAME
AUDIO_TAIL_MAX_SAMPLES = 2 * SAMPLES_PER_FRAME
NO_SEALED_MEDIA_TOKENS = ("/test/", "/tests/", "/val/", "/validation/", "/heldout/")


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    validate_run_id(run_id)
    return REPO / "runs" / f"lrs3_wav2lip_timing_transfer_{run_id}"


def cell_key(video_arm: str, audio_arm: str, repeat: bool = False) -> str:
    value = f"{video_arm}__{audio_arm}"
    return f"{value}__repeat" if repeat else value


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def input_audit(self) -> Path:
        return self.root / "input_audit.json"

    @property
    def protocol(self) -> Path:
        return self.root / "protocol.json"

    @property
    def media(self) -> Path:
        return self.root / "media"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis.json"

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
    schema_version: int = 1
    protocol_id: str = PROTOCOL_ID
    protocol_revision: str = PROTOCOL_REVISION
    sample_rate: int = SAMPLE_RATE
    fps: int = FPS
    samples_per_frame: int = SAMPLES_PER_FRAME
    pcm_channels: int = PCM_CHANNELS
    pcm_sample_width: int = PCM_SAMPLE_WIDTH
    crop_size: int = CROP_SIZE
    crop_scale: float = CROP_SCALE
    warp_amplitude_samples: int = WARP_AMPLITUDE_SAMPLES
    warp_amplitude_frames: float = WARP_AMPLITUDE_FRAMES
    vshift: int = VSHIFT
    window_frames: int = WINDOW_FRAMES
    min_local_rows: int = MIN_LOCAL_ROWS
    peak_gap_threshold: float = PEAK_GAP_THRESHOLD
    offset_tolerance_frames: float = OFFSET_TOLERANCE_FRAMES
    min_baseline_records: int = MIN_BASELINE_RECORDS
    min_success_records: int = MIN_SUCCESS_RECORDS
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT
    expected_main_cell_count: int = EXPECTED_MAIN_CELL_COUNT
    expected_repeat_cell_count: int = EXPECTED_REPEAT_CELL_COUNT
    expected_total_cell_count: int = EXPECTED_TOTAL_CELL_COUNT
    expected_sample_id_sha256: str = ORDERED_SAMPLE_ID_SHA256

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "audio_arms": list(AUDIO_ARMS),
                "video_arms": list(VIDEO_ARMS),
                "main_cell_specs": [[video, audio] for video, audio in MAIN_CELL_SPECS],
                "repeat_cell_specs": [[video, audio] for video, audio in REPEAT_CELL_SPECS],
                "warp_formula": "s[n]=n+1920*sin(2*pi*n/(L-1)); endpoints forced; float64 linear interpolation; nearest-even int16",
                "inverse_formula": "piecewise-linear interpolation of swapped coordinates (s,n)",
                "offset_column_formula": "offset=15-column_index",
                "pcm_policy": "unchanged 16 kHz mono PCM16 bytes; container hash and decoded PCM hash are separate",
                "local_support": "Q=min(F_R,F_GN,F_GW,floor(L/640)); candidate rows 15 <= r < Q-20",
                "bootstrap_policy": "sorted source groups; NumPy default_rng/PCG64; 10000 draws; percentile 2.5/97.5; seed reset per metric",
                "forbidden_operations": [
                    "tts_generation",
                    "wav2lip_generation",
                    "audio_resampling",
                    "audio_truncation",
                    "video_padding_or_looping",
                    "global_offset_alignment",
                    "score_based_retry",
                    "record_filtering",
                    "training",
                    "mfa_or_dtw",
                    "bridge_rescoring",
                    "heldout_or_sealed_media_access",
                    "history_overwrite",
                ],
            }
        )
        return payload


def spec_bindings() -> dict[str, dict[str, str]]:
    # tasks.md is a mutable execution checklist.  Binding it would make a
    # valid run fail validation as soon as completed tasks are marked.
    return {
        "spec": {"path": str(SPEC.resolve()), "sha256": SPEC_SHA256},
        "proposal": {"path": str(PROPOSAL.resolve()), "sha256": PROPOSAL_SHA256},
        "design": {"path": str(DESIGN.resolve()), "sha256": DESIGN_SHA256},
    }
