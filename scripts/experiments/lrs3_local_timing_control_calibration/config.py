from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
HISTORY_ROOT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904"
HISTORY_FINAL = HISTORY_ROOT / "04_final/final.json"
HISTORY_COHORT = HISTORY_ROOT / "00_protocol/cohort.json"
HISTORY_AUDIO = HISTORY_ROOT / "01_audio/audio_manifest.json"
HISTORY_VIDEOS = HISTORY_ROOT / "02_videos/videos_manifest.json"
HISTORY_SCORES = HISTORY_ROOT / "03_scores/scores_manifest.json"

PROTOCOL_PREFIX = "lrs3_local_timing_control_calibration"
PROTOCOL_ID = "lrs3_local_timing_control_calibration"
HISTORY_FINAL_SHA256 = "df0ca9767e70ccc384c86c1da23243c53fa609be12abd1dda20f9075b6732c6e"
HISTORY_PROTOCOL_ID = "lrs3_natural_to_tts_bridge_confirmation_20260904"
HISTORY_COHORT_SHA256 = "b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b"
HISTORY_AUDIO_SHA256 = "2c279dc5a74dbeb53ae547235347870c6664fd9caf656a594b48a8310d9bd288"
HISTORY_VIDEOS_SHA256 = "680053f3ca851cea6a2572654a228671bbe616f369b00d84a552b22b305543c2"
HISTORY_SCORES_SHA256 = "0022c07d68f859a0189833c58fd7f8daf639787d8d4ffc99e48e6ccaa1a99e5b"
EXPECTED_SAMPLE_ID_SHA256 = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"
EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22

SAMPLE_RATE = 16_000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
MIN_AUDIO_SAMPLES = 1_024
SMOOTH_WARP_AMPLITUDE_SAMPLES = 1_920
SMOOTH_WARP_MIN_SAMPLES = int(2 * 3.141592653589793 * SMOOTH_WARP_AMPLITUDE_SAMPLES) + 2

REPAIR_BRANCH = "REPAIR_ONLY"
SMOOTH_BRANCH = "SMOOTH_WARP"
LOCAL_SWAP_ARM = "LOCAL_SWAP"
SMOOTH_WARP_ARM = "LOCAL_WARP_120"
NATURAL_ARM = "N"
REPEAT_ARM = "N_REPEAT"
BRANCHES = (REPAIR_BRANCH, SMOOTH_BRANCH)
BASE_ARMS = (NATURAL_ARM, REPEAT_ARM)

WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
WAV2LIP_CHECKPOINT = WAV2LIP_ROOT / "checkpoints/wav2lip_gan.pth"
WAV2LIP_PYTHON = Path.home() / ".venvs/wav2lip/bin/python"
WAV2LIP_ENV = {"NUMBA_DISABLE_JIT": "1"}
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_PYTHON = Path.home() / ".venvs/syncnet/bin/python"
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")
MIN_TRACK = 50

WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
WAV2LIP_INFERENCE_SHA256 = "42cfc8d3060921e6714c8e08243e5508d2e637d29cd0be78760492c5d26380a9"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
WAV2LIP_PYTHON_SHA256 = "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
SYNCNET_PYTHON_SHA256 = WAV2LIP_PYTHON_SHA256
FFMPEG_SHA256 = "019d92fd5839dfeabf5c67cbdf8408e1d7bdf0f1528cece52a19e5fa89593fcd"
FFPROBE_SHA256 = "14b31fde8eb1e12e6f53fb8de706d9475604f8b74548757f978c7bedbfd3c3f2"

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260904
REPLACEMENT_MARGIN = -0.10
LOCAL_DAMAGE_THRESHOLD = 0.10
OFFSET_TOLERANCE_FRAMES = 1
MIN_OFFSET_AGREEMENT_RECORDS = 20
MIN_LOCAL_SENSITIVITY_RECORDS = 18

NO_SEALED_MEDIA_TOKENS = ("/test/", "/tests/", "/val/", "/validation/", "/heldout/")


def control_arm_for_branch(branch: str) -> str:
    if branch == REPAIR_BRANCH:
        return LOCAL_SWAP_ARM
    if branch == SMOOTH_BRANCH:
        return SMOOTH_WARP_ARM
    raise ValueError(f"unknown calibration branch: {branch}")


def arms_for_branch(branch: str) -> tuple[str, ...]:
    return (*BASE_ARMS, control_arm_for_branch(branch))


def matrix_cells(control_arm: str) -> tuple[str, ...]:
    return (
        f"V_{NATURAL_ARM}/A_{NATURAL_ARM}",
        f"V_{REPEAT_ARM}/A_{NATURAL_ARM}",
        f"V_{control_arm}/A_{control_arm}",
        f"V_{control_arm}/A_{NATURAL_ARM}",
    )


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
    def audit(self) -> Path:
        return self.root / "00_audit"

    @property
    def protocol(self) -> Path:
        return self.root / "01_protocol"

    @property
    def audio(self) -> Path:
        return self.root / "02_audio"

    @property
    def videos(self) -> Path:
        return self.root / "03_videos"

    @property
    def scores(self) -> Path:
        return self.root / "04_scores"

    @property
    def final(self) -> Path:
        return self.root / "05_final"


@dataclass(frozen=True)
class FrozenConfig:
    schema_version: int = 1
    protocol_id: str = PROTOCOL_ID
    sample_rate: int = SAMPLE_RATE
    pcm_channels: int = PCM_CHANNELS
    pcm_sample_width: int = PCM_SAMPLE_WIDTH
    min_audio_samples: int = MIN_AUDIO_SAMPLES
    smooth_warp_amplitude_samples: int = SMOOTH_WARP_AMPLITUDE_SAMPLES
    smooth_warp_min_samples: int = SMOOTH_WARP_MIN_SAMPLES
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    replacement_margin: float = REPLACEMENT_MARGIN
    local_damage_threshold: float = LOCAL_DAMAGE_THRESHOLD
    offset_tolerance_frames: int = OFFSET_TOLERANCE_FRAMES
    min_offset_agreement_records: int = MIN_OFFSET_AGREEMENT_RECORDS
    min_local_sensitivity_records: int = MIN_LOCAL_SENSITIVITY_RECORDS
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT
    wav2lip_checkpoint_sha256: str = WAV2LIP_CHECKPOINT_SHA256
    wav2lip_inference_sha256: str = WAV2LIP_INFERENCE_SHA256
    syncnet_model_sha256: str = SYNCNET_MODEL_SHA256
    wav2lip_python_sha256: str = WAV2LIP_PYTHON_SHA256
    syncnet_python_sha256: str = SYNCNET_PYTHON_SHA256
    ffmpeg_sha256: str = FFMPEG_SHA256
    ffprobe_sha256: str = FFPROBE_SHA256
    wav2lip_environment: dict[str, str] = field(default_factory=lambda: dict(WAV2LIP_ENV))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "branch_options": list(BRANCHES),
                "base_arms": list(BASE_ARMS),
                "local_swap": "x[:floor(L/4)] || x[floor(L/2):floor(3L/4)] || x[floor(L/4):floor(L/2)] || x[floor(3L/4):]",
                "smooth_warp": "s[n]=n+1920*sin(2*pi*n/(L-1)); z[n]=linear_interpolate(x,s[n]); y=round_to_nearest_ties_to_even(z)",
                "pcm_policy": "16khz_mono_pcm16_little_endian",
                "bootstrap": "sorted source-group cluster, NumPy default_rng, percentile 2.5/97.5",
                "wav2lip_numba_cache": "per-cell work_dir/numba_cache",
                "min_track": MIN_TRACK,
                "forbidden_operations": [
                    "bridge_rescoring",
                    "training",
                    "fine_tuning",
                    "tts_generation",
                    "mfa",
                    "dtw",
                    "sealed_media_access",
                    "score_based_retry",
                    "branch_switch",
                    "strength_search",
                    "record_filtering",
                    "parent_or_history_overwrite",
                ],
            }
        )
        return payload


def expected_video_count(branch: str) -> int:
    return EXPECTED_RECORD_COUNT * len(arms_for_branch(branch))


def expected_cell_count(branch: str) -> int:
    return EXPECTED_RECORD_COUNT * len(matrix_cells(control_arm_for_branch(branch)))
