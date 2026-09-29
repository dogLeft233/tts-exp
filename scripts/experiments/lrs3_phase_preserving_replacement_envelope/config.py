from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = "lrs3-phase-preserving-replacement-envelope"
PROTOCOL_ID = "lrs3_phase_preserving_replacement_envelope_20260904"
RUN_ROOT = REPO / "runs/lrs3_phase_preserving_replacement_envelope_20260904"
STAGES = {
    "00_protocol": RUN_ROOT / "00_protocol",
    "01_candidates": RUN_ROOT / "01_candidates",
    "02_audio_diagnostics": RUN_ROOT / "02_audio_diagnostics",
    "03_videos": RUN_ROOT / "03_videos",
    "04_scores": RUN_ROOT / "04_scores",
    "05_final": RUN_ROOT / "05_final",
}

PARENT_PROTOCOL = REPO / "runs/lrs3_mfa_dtw_replacement_short_phone_20260904/00_protocol/manifest.json"
PARENT_REPLACEMENT = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/03_strict_replacement_face_ready_retry7/replacement_manifest.json"
EXPECTED_PARENT_PROTOCOL_SHA256 = "e8c50a459c8197c86ec11cf06a01c000e3f62def981d16d8ed9275a38a473784"
EXPECTED_PARENT_REPLACEMENT_SHA256 = "157cd678f5f03be6eae9986a82babfdfb1f1a44e611bb0b86476c6e19b7eb272"
EXPECTED_RECORD_COUNT = 23
EXPECTED_SOURCE_GROUP_COUNT = 23
EXPECTED_SAMPLE_ID_SHA256 = "c56e420d9ade7e04f5558f37fbf68ee68463d3817060baab79e8e6e0bf7e8fbd"
PARENT_RECORD_COUNT = 133
PARENT_SOURCE_GROUP_COUNT = 23

SAMPLE_RATE = 16_000
PCM_SAMPLE_WIDTH = 2
PCM_CHANNELS = 1
MIN_AUDIO_SAMPLES = 1024
SHIFT_SAMPLES = 3_200
N_FFT = 1024
WIN_LENGTH = 1024
HOP_LENGTH = 256
MAGNITUDE_FLOOR = 1e-7
PEAK_LIMIT = 0.999
INV_MIN_PCM = -32768
MAG_STRENGTHS = (0.25, 0.50, 0.75, 1.00)
ARMS = ("N", "INV", "MAG_025", "MAG_050", "MAG_075", "MAG_100", "SHIFT_200")
MAG_ARMS = ("MAG_025", "MAG_050", "MAG_075", "MAG_100")
ARM_STRENGTHS = dict(zip(MAG_ARMS, MAG_STRENGTHS))

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
OFFSET_TOLERANCE_FRAMES = 1
MIN_OFFSET_AGREEMENT_RECORDS = 21
MIN_MOVEMENT_RECORDS = 21
MOVEMENT_THRESHOLD = 0.15
SHIFT_OFFSET_CHANGE_FRAMES = 3
SHIFT_OFFSET_CHANGE_RECORDS = 18
SHIFT_MEAN_DEGRADATION = -0.10
INV_MEL_MAE_LIMIT = 1e-5

DRIVER_AUDIO_CELLS = (("N", "N"),) + tuple(
    (arm, audio_arm)
    for arm in ARMS[1:]
    for audio_arm in ("N", arm)
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
    sample_rate: int = SAMPLE_RATE
    pcm_sample_width: int = PCM_SAMPLE_WIDTH
    pcm_channels: int = PCM_CHANNELS
    min_audio_samples: int = MIN_AUDIO_SAMPLES
    shift_samples: int = SHIFT_SAMPLES
    n_fft: int = N_FFT
    win_length: int = WIN_LENGTH
    hop_length: int = HOP_LENGTH
    magnitude_floor: float = MAGNITUDE_FLOOR
    peak_limit: float = PEAK_LIMIT
    mag_strengths: tuple[float, ...] = MAG_STRENGTHS
    arms: tuple[str, ...] = ARMS
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    replacement_margin: float = REPLACEMENT_MARGIN
    offset_tolerance_frames: int = OFFSET_TOLERANCE_FRAMES
    min_offset_agreement_records: int = MIN_OFFSET_AGREEMENT_RECORDS
    movement_threshold: float = MOVEMENT_THRESHOLD
    min_movement_records: int = MIN_MOVEMENT_RECORDS
    inv_mel_mae_limit: float = INV_MEL_MAE_LIMIT
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
        for key in ("mag_strengths", "arms"):
            payload[key] = list(payload[key])
        payload.update(
            {
                "stft_window": "periodic_hann",
                "stft_center": True,
                "stft_pad_mode": "reflect",
                "phase_policy": "natural_phase",
                "pcm_policy": "16khz_mono_pcm16_little_endian",
                "post_scaling": "one_global_rms_match_then_one_peak_attenuation_if_needed",
                "forbidden_operations": [
                    "resampling", "compressor", "limiter", "denoiser", "filtering",
                    "time_shift_except_shift_200_control", "iterative_repair", "score_based_retry",
                    "record_substitution", "strength_search", "training", "fine_tuning",
                    "tts_generation", "mfa", "dtw", "sealed_media_access",
                ],
            }
        )
        return payload

    def validate(self) -> None:
        if self != FrozenConfig():
            raise ValueError("replacement-envelope configuration is frozen")


def validate_serialized_config(payload: Mapping[str, Any]) -> None:
    if dict(payload) != FrozenConfig().to_dict():
        raise ValueError("serialized replacement-envelope configuration is frozen")


def stage_manifest(stage: str, filename: str) -> Path:
    return STAGES[stage] / filename
