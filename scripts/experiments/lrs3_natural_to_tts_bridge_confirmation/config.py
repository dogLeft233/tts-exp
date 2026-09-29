from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = "lrs3-natural-to-tts-bridge-confirmation"
PROTOCOL_ID = "lrs3_natural_to_tts_bridge_confirmation_20260904"
RUN_ROOT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904"
STAGES = {
    "00_protocol": RUN_ROOT / "00_protocol",
    "01_audio": RUN_ROOT / "01_audio",
    "02_videos": RUN_ROOT / "02_videos",
    "03_scores": RUN_ROOT / "03_scores",
    "04_final": RUN_ROOT / "04_final",
}

PARENT_PROTOCOL = REPO / "runs/lrs3_mfa_dtw_replacement_short_phone_20260904/00_protocol/manifest.json"
PARENT_REPLACEMENT = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/03_strict_replacement_face_ready_retry7/replacement_manifest.json"
DISCOVERY_FINAL = REPO / "runs/lrs3_phase_preserving_replacement_envelope_20260904/05_final/final.json"
DISCOVERY_COHORT = REPO / "runs/lrs3_phase_preserving_replacement_envelope_20260904/00_protocol/cohort.json"
EXPECTED_PARENT_PROTOCOL_SHA256 = "e8c50a459c8197c86ec11cf06a01c000e3f62def981d16d8ed9275a38a473784"
EXPECTED_PARENT_REPLACEMENT_SHA256 = "157cd678f5f03be6eae9986a82babfdfb1f1a44e611bb0b86476c6e19b7eb272"
EXPECTED_DISCOVERY_FINAL_SHA256 = "9bab2a853edda0cdb16e79e2b9a8c6d32ce3e36d721e4a1c1140306d9ffac8c1"
EXPECTED_DISCOVERY_COHORT_SHA256 = "850c224856bc7acb95aa709a18f4c0b3369ca724d8759603d1c4bf3aac74d72a"
EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22
EXPECTED_SAMPLE_ID_SHA256 = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"
PARENT_RECORD_COUNT = 133
PARENT_SOURCE_GROUP_COUNT = 23
DISCOVERY_RECORD_COUNT = 23
DISCOVERY_SOURCE_GROUP_COUNT = 23

SAMPLE_RATE = 16_000
PCM_SAMPLE_WIDTH = 2
PCM_CHANNELS = 1
MIN_AUDIO_SAMPLES = 1024
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
MAGNITUDE_FLOOR = 1e-7
PEAK_LIMIT = 0.999
BRIDGE_ALPHA = 0.75
BRIDGE_ARM = "BRIDGE_075"
ARMS = ("N", "N_REPEAT", "LOCAL_SWAP", "BRIDGE_075")

WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
WAV2LIP_CHECKPOINT = WAV2LIP_ROOT / "checkpoints/wav2lip_gan.pth"
WAV2LIP_PYTHON = Path.home() / ".venvs/wav2lip/bin/python"
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_PYTHON = Path.home() / ".venvs/syncnet/bin/python"
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")
MIN_TRACK = 50
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
WAV2LIP_PYTHON_SHA256 = "1643dacd9feaedc58f3cc581e4d22577dfe25c09b10282936186ccf0f2e61118"
SYNCNET_PYTHON_SHA256 = WAV2LIP_PYTHON_SHA256
FFMPEG_SHA256 = "019d92fd5839dfeabf5c67cbdf8408e1d7bdf0f1528cece52a19e5fa89593fcd"
FFPROBE_SHA256 = "14b31fde8eb1e12e6f53fb8de706d9475604f8b74548757f978c7bedbfd3c3f2"

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260904
REPLACEMENT_MARGIN = -0.10
PRIMARY_GAIN_THRESHOLD = 0.0
LOCAL_DAMAGE_THRESHOLD = 0.10
OFFSET_TOLERANCE_FRAMES = 1
MIN_OFFSET_AGREEMENT_RECORDS = 20
MIN_MOVEMENT_RECORDS = 20
MIN_LOCAL_SENSITIVITY_RECORDS = 18
MOVEMENT_THRESHOLD = 0.15

DRIVER_AUDIO_CELLS = (
    ("N", "N"),
    ("N_REPEAT", "N"),
    ("LOCAL_SWAP", "N"),
    ("LOCAL_SWAP", "LOCAL_SWAP"),
    ("BRIDGE_075", "N"),
    ("BRIDGE_075", "BRIDGE_075"),
)
MATRIX_CELLS = tuple(
    f"V_{video_arm}/A_{audio_arm}"
    for video_arm, audio_arm in DRIVER_AUDIO_CELLS
)
EXPECTED_VIDEO_COUNT = EXPECTED_RECORD_COUNT * len(ARMS)
EXPECTED_CELL_COUNT = EXPECTED_RECORD_COUNT * len(MATRIX_CELLS)
NO_SEALED_MEDIA_TOKENS = ("/test/", "/tests/", "/val/", "/validation/", "/heldout/")


@dataclass(frozen=True)
class FrozenConfig:
    schema_version: int = 1
    experiment: str = EXPERIMENT
    protocol_id: str = PROTOCOL_ID
    parent_protocol_sha256: str = EXPECTED_PARENT_PROTOCOL_SHA256
    parent_replacement_sha256: str = EXPECTED_PARENT_REPLACEMENT_SHA256
    discovery_final_sha256: str = EXPECTED_DISCOVERY_FINAL_SHA256
    discovery_cohort_sha256: str = EXPECTED_DISCOVERY_COHORT_SHA256
    sample_rate: int = SAMPLE_RATE
    pcm_sample_width: int = PCM_SAMPLE_WIDTH
    pcm_channels: int = PCM_CHANNELS
    min_audio_samples: int = MIN_AUDIO_SAMPLES
    n_fft: int = N_FFT
    win_length: int = WIN_LENGTH
    hop_length: int = HOP_LENGTH
    magnitude_floor: float = MAGNITUDE_FLOOR
    peak_limit: float = PEAK_LIMIT
    bridge_alpha: float = BRIDGE_ALPHA
    arms: tuple[str, ...] = ARMS
    matrix_cells: tuple[str, ...] = MATRIX_CELLS
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    replacement_margin: float = REPLACEMENT_MARGIN
    primary_gain_threshold: float = PRIMARY_GAIN_THRESHOLD
    local_damage_threshold: float = LOCAL_DAMAGE_THRESHOLD
    offset_tolerance_frames: int = OFFSET_TOLERANCE_FRAMES
    min_offset_agreement_records: int = MIN_OFFSET_AGREEMENT_RECORDS
    movement_threshold: float = MOVEMENT_THRESHOLD
    min_movement_records: int = MIN_MOVEMENT_RECORDS
    min_local_sensitivity_records: int = MIN_LOCAL_SENSITIVITY_RECORDS
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT
    expected_video_count: int = EXPECTED_VIDEO_COUNT
    expected_cell_count: int = EXPECTED_CELL_COUNT
    wav2lip_checkpoint_sha256: str = WAV2LIP_CHECKPOINT_SHA256
    syncnet_model_sha256: str = SYNCNET_MODEL_SHA256
    ffmpeg_sha256: str = FFMPEG_SHA256
    ffprobe_sha256: str = FFPROBE_SHA256

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        for key in ("arms", "matrix_cells"):
            payload[key] = list(payload[key])
        payload.update(
            {
                "stft_window": "periodic_hann",
                "stft_center": True,
                "stft_pad_mode": "reflect",
                "phase_policy": "natural_phase",
                "pcm_policy": "16khz_mono_pcm16_little_endian",
                "post_scaling": "one_global_rms_match_then_one_peak_attenuation_if_needed",
                "local_swap": "x[0:floor(L/4)] || x[floor(L/2):floor(3L/4)] || x[floor(L/4):floor(L/2)] || x[floor(3L/4):L]",
                "forbidden_operations": [
                    "resampling", "compressor", "limiter", "denoiser", "filtering",
                    "time_adjustment", "iterative_repair", "score_based_retry",
                    "record_substitution", "strength_search", "training", "fine_tuning",
                    "tts_generation", "mfa", "dtw", "sealed_media_access",
                ],
            }
        )
        return payload

    def validate(self) -> None:
        if self != FrozenConfig():
            raise ValueError("natural-to-TTS bridge configuration is frozen")


def validate_serialized_config(payload: Mapping[str, Any]) -> None:
    if dict(payload) != FrozenConfig().to_dict():
        raise ValueError("serialized natural-to-TTS bridge configuration is frozen")


def stage_manifest(stage: str, filename: str) -> Path:
    return STAGES[stage] / filename
