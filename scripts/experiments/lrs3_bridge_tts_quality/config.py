from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]

PROTOCOL_ID = "lrs3_bridge_tts_quality_v2"
PROTOCOL_REVISION = "20260913"
RUN_PREFIX = "lrs3_bridge_tts_quality"

EXPECTED_SAMPLE_IDS = (
    "lrs3_6WeS1bXRBOk_00006",
    "lrs3_6ul2TSvUDog_00007",
    "lrs3_6wk4dkYSrV0_00006",
    "lrs3_73jPh0eRPSY_00008",
    "lrs3_6qqqVwM6bMM_00007",
    "lrs3_70VZ1SOzSnc_00007",
    "lrs3_73cTNHEQhkQ_00007",
    "lrs3_796LfXwzIUk_00007",
    "lrs3_7CIq4mtiamY_00007",
    "lrs3_7DCofMA9eQA_00007",
    "lrs3_6ORDQFh0Byw_00008",
    "lrs3_6VnKV1sr5VQ_00008",
    "lrs3_6yR5OUVb2gY_00008",
    "lrs3_6ydYeyNSQVY_00008",
    "lrs3_6SdtkXAQq3k_00009",
    "lrs3_6tSlMoMNSlY_00009",
    "lrs3_6xtmm0MnaS0_00010",
    "lrs3_73rUjrow5pI_00009",
    "lrs3_6xy5pWOgeBY_00010",
    "lrs3_6weGCM3sWKc_00014",
    "lrs3_7DLzXAjscXk_00013",
    "lrs3_79tRTivyMSM_00014",
)

EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22
EXPECTED_SAMPLE_ID_SHA256 = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"

CONFIRMATION_COHORT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json"
CLOUD_TTS_METADATA = REPO / "runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts_meta.json"
HISTORICAL_CLOUD_TARGET = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825/02_candidate_audio_retry4/candidate_manifest.json"
PARENT_PROTOCOL = REPO / "runs/lrs3_mfa_dtw_replacement_short_phone_20260904/00_protocol/manifest.json"
INPUT_BINDINGS = REPO / "openspec/changes/compare-lrs3-bridge-tts-quality/input-bindings.json"
INPUT_BINDINGS_SHA256 = "d08ffb6ffe4932be1de922d8f6c14ac41a6c5e70f00ba16bb65381a61f93487e"

PARENT_HASHES = {
    "confirmation_cohort": "b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b",
    "cloud_tts_metadata": "2de0e1d377a0513862b7573770b3a7b36fde1df0307faa057aa0ab5cbce052f4",
    "historical_cloud_target": "2ff6da4923e3a2abb31463014ba69d6587c5bd8bf2da257d9aff32a23fa42d6a",
    "parent_protocol": "e8c50a459c8197c86ec11cf06a01c000e3f62def981d16d8ed9275a38a473784",
}

LOCAL_PROVIDER = "faster_qwen3"
LOCAL_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
LOCAL_CLONE_MODE = "icl"
CLOUD_PROVIDER = "dashscope_vc"
CLOUD_MODEL = "qwen3-tts-vc-2026-01-22"
LANGUAGE = "English"

SAMPLE_RATE = 16_000
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
MAX_AUDIO_SECONDS = 30.0
MIN_AUDIO_SAMPLES = 1_024
MIN_DURATION_RATIO = 0.5
MAX_DURATION_RATIO = 1.5

MFA_VERSION = "3.4.1"
MFA_DICTIONARY = "english_mfa"
MFA_ACOUSTIC_MODEL = "english_mfa"
MFA_ROOT_DIR = Path("/home/wjj/Documents/MFA/mfa3_runtime_20260815")
MFA_EXECUTABLE = Path("/home/wjj/miniconda3/envs/mfa3/bin/mfa")
MFA_ROOT_CONFIG = MFA_ROOT_DIR / "global_config.yaml"
MFA_DICTIONARY_PATH = MFA_ROOT_DIR / "pretrained_models/dictionary/english_mfa.dict"
MFA_ACOUSTIC_MODEL_PATH = MFA_ROOT_DIR / "pretrained_models/acoustic/english_mfa.zip"
MAX_FEATURE_TAIL_EXTENSION_S = 0.020

FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")

WAVLM_CHECKPOINT = Path("/home/wjj/.cache/torch/hub/checkpoints/WavLM-Large.pt")
VOCODER_CHECKPOINT = Path("/home/wjj/.cache/torch/hub/checkpoints/prematch_g_02500000.pt")
KNN_VC_SOURCE = REPO / "third_party/knn-vc"
KNN_VC_REVISION = "c616845c4e309e24d5927f15adbdf277a3d65358"
WAVLM_CHECKPOINT_SHA256 = "6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f"
VOCODER_CHECKPOINT_SHA256 = "f924c7632c6eaf99004386d62293e124419e33582573552a6ae976eb88ed2dd5"

WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
WAV2LIP_CHECKPOINT = WAV2LIP_ROOT / "checkpoints/wav2lip_gan.pth"
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
WAV2LIP_INFERENCE = WAV2LIP_ROOT / "inference.py"
WAV2LIP_TAIL_PAD_SAMPLES = 1_920

SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
SYNCNET_MIN_TRACK = 50
SYNCNET_VSHIFT = 15
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS

BRIDGE_ALPHA = 0.75
N_FFT = 1_024
WIN_LENGTH = 1_024
HOP_LENGTH = 256
MAGNITUDE_FLOOR = 1e-7
PEAK_LIMIT = 0.999

ARMS = ("N", "B0", "B_LOCAL", "B_CLOUD")
REPEATS = (0, 1)
SCORE_CELLS = (
    ("N", "N", "baseline"),
    ("B0", "N", "reconstruction_control"),
    ("B_LOCAL", "N", "replacement"),
    ("B_CLOUD", "N", "replacement"),
    ("B_LOCAL", "B_LOCAL", "own_audio_diagnostic"),
    ("B_CLOUD", "B_CLOUD", "own_audio_diagnostic"),
    ("N", "N_REV", "wrong_audio_control"),
)
MATRIX_CELLS = tuple(f"V_{video}/A_{audio}" for video, audio, _ in SCORE_CELLS)
EXPECTED_VIDEO_COUNT = EXPECTED_RECORD_COUNT * len(ARMS) * len(REPEATS)
EXPECTED_CELL_COUNT = EXPECTED_RECORD_COUNT * len(SCORE_CELLS) * len(REPEATS)
EXPECTED_QUALITY_STIMULI = EXPECTED_RECORD_COUNT * 4

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260913
QUALITY_BOOTSTRAP_SEED = 20260915
ASSOCIATION_BOOTSTRAP_SEED = 20260916
MIN_RATERS_PER_PAIR = 3
MOVEMENT_THRESHOLD = 0.15
MOVEMENT_MIN_RECORDS = 20
REPLACEMENT_MARGIN = -0.100
OFFSET_TOLERANCE_FRAMES = 1
MIN_OFFSET_AGREEMENT_RECORDS = 20
MIN_WRONG_AUDIO_DAMAGE_RECORDS = 18
WRONG_AUDIO_DAMAGE_THRESHOLD = 0.100
MIN_VALID_BOOTSTRAPS = 9_500

NO_SEALED_MEDIA_TOKENS = ("/test/", "/tests/", "/val/", "/validation/", "/heldout/")
GPU_LOCK_PATH = Path("/tmp/tts-exp-lrs3-bridge-gpu.lock")


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"{RUN_PREFIX}_{validate_run_id(run_id)}"


@dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def protocol(self) -> Path:
        return self.root / "00_protocol"

    @property
    def tts(self) -> Path:
        return self.root / "01_tts"

    @property
    def targets(self) -> Path:
        return self.root / "02_targets"

    @property
    def bridge(self) -> Path:
        return self.root / "03_bridge"

    @property
    def quality(self) -> Path:
        return self.root / "04_quality"

    @property
    def videos(self) -> Path:
        return self.root / "05_videos"

    @property
    def scores(self) -> Path:
        return self.root / "06_scores"

    @property
    def analysis(self) -> Path:
        return self.root / "07_analysis"

    @property
    def analysis_versions(self) -> Path:
        return self.analysis / "versions"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"


@dataclass(frozen=True)
class FrozenConfig:
    schema_version: int = 1
    protocol_id: str = PROTOCOL_ID
    protocol_revision: str = PROTOCOL_REVISION
    dataset: str = "lrs3"
    language: str = LANGUAGE
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT
    ordered_sample_ids_sha256: str = EXPECTED_SAMPLE_ID_SHA256
    input_bindings_sha256: str = INPUT_BINDINGS_SHA256
    sample_rate: int = SAMPLE_RATE
    pcm_channels: int = PCM_CHANNELS
    pcm_sample_width: int = PCM_SAMPLE_WIDTH
    max_audio_seconds: float = MAX_AUDIO_SECONDS
    min_duration_ratio: float = MIN_DURATION_RATIO
    max_duration_ratio: float = MAX_DURATION_RATIO
    mfa_version: str = MFA_VERSION
    mfa_dictionary: str = MFA_DICTIONARY
    mfa_acoustic_model: str = MFA_ACOUSTIC_MODEL
    knn_vc_revision: str = KNN_VC_REVISION
    bridge_alpha: float = BRIDGE_ALPHA
    n_fft: int = N_FFT
    win_length: int = WIN_LENGTH
    hop_length: int = HOP_LENGTH
    magnitude_floor: float = MAGNITUDE_FLOOR
    peak_limit: float = PEAK_LIMIT
    fps: int = FPS
    samples_per_frame: int = SAMPLES_PER_FRAME
    syncnet_min_track: int = SYNCNET_MIN_TRACK
    syncnet_vshift: int = SYNCNET_VSHIFT
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    quality_bootstrap_seed: int = QUALITY_BOOTSTRAP_SEED
    association_bootstrap_seed: int = ASSOCIATION_BOOTSTRAP_SEED
    arms: tuple[str, ...] = ARMS
    repeats: tuple[int, ...] = REPEATS
    matrix_cells: tuple[str, ...] = MATRIX_CELLS
    expected_video_count: int = EXPECTED_VIDEO_COUNT
    expected_cell_count: int = EXPECTED_CELL_COUNT
    expected_quality_stimuli: int = EXPECTED_QUALITY_STIMULI
    wav2lip_checkpoint_sha256: str = WAV2LIP_CHECKPOINT_SHA256
    wav2lip_tail_pad_samples: int = WAV2LIP_TAIL_PAD_SAMPLES
    syncnet_model_sha256: str = SYNCNET_MODEL_SHA256
    wavlm_checkpoint_sha256: str = WAVLM_CHECKPOINT_SHA256
    vocoder_checkpoint_sha256: str = VOCODER_CHECKPOINT_SHA256
    gpu_max_parallel: int = 1
    gpu_lease_path: str = str(GPU_LOCK_PATH)
    gpu_policy: str = "reject active compute processes and nonzero utilization; stages are serial"
    forbidden_operations: tuple[str, ...] = (
        "new_cloud_tts_call",
        "score_based_retry",
        "record_substitution",
        "quality_based_selection",
        "alpha_search",
        "dtw",
        "training",
        "fine_tuning",
        "loudness_normalization_in_model_audio",
        "global_waveform_stretch",
        "sealed_media_access",
    )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("arms", "repeats", "matrix_cells", "forbidden_operations"):
            value[key] = list(value[key])
        value.update(
            {
                "local_tts": {
                    "provider": LOCAL_PROVIDER,
                    "model": LOCAL_MODEL,
                    "clone_mode": LOCAL_CLONE_MODE,
                    "backend_policy": "strict_no_fallback",
                    "reference_policy": "paired_natural_audio",
                },
                "cloud_tts": {
                    "provider": CLOUD_PROVIDER,
                    "model": CLOUD_MODEL,
                    "new_calls_allowed": 0,
                    "reference_policy": "paired_natural_audio",
                },
                "bridge": {
                    "phase_policy": "natural_phase",
                    "pcm_decode_scale": "/32768",
                    "bridge_quantize_scale": "*32768",
                    "target_quantize_scale": "*32767",
                    "stft_window": "periodic_hann",
                    "stft_center": True,
                    "stft_pad_mode": "reflect",
                    "rms_policy": "one_global_match",
                    "peak_policy": "one_global_attenuation_if_needed",
                },
                "render": {
                    "box_policy": "full_frame",
                    "box_order": ["top", "bottom", "left", "right"],
                    "box": "[0,H,0,W]",
                    "nosmooth": True,
                    "face_det_batch_size": 16,
                    "wav2lip_batch_size": 16,
                    "repeat_seeds": [20260913, 20260914],
                    "driver_tail_pad_samples": WAV2LIP_TAIL_PAD_SAMPLES,
                    "driver_tail_pad_value": 0,
                    "driver_audio_policy": "fixed_zero_right_context_for_frozen_support_only; score_original_prefix",
                },
                "quality": {
                    "stimuli_per_record": 4,
                    "minimum_common_raters": MIN_RATERS_PER_PAIR,
                    "raw_and_target_stages": ["raw", "target"],
                    "listening_rms_dbfs": -26.0,
                },
                "support": {
                    "start_frame": 0,
                    "audio_samples_per_video_frame": SAMPLES_PER_FRAME,
                    "min_frames": 50,
                    "track_source": "original_face_video_once_per_record",
                    "crop_reused_for_all_arms_and_score_audio": True,
                },
            }
        )
        return value


def expected_score_cells(sample_id: str) -> list[dict[str, str]]:
    return [
        {
            "sample_id": sample_id,
            "cell": f"V_{video}/A_{audio}",
            "video_arm": video,
            "score_audio_arm": audio,
            "purpose": purpose,
        }
        for video, audio, purpose in SCORE_CELLS
    ]


def runtime_paths() -> dict[str, Path]:
    return {
        "mfa_executable": MFA_EXECUTABLE,
        "mfa_root_config": MFA_ROOT_CONFIG,
        "mfa_dictionary": MFA_DICTIONARY_PATH,
        "mfa_acoustic_model": MFA_ACOUSTIC_MODEL_PATH,
        "wavlm_checkpoint": WAVLM_CHECKPOINT,
        "vocoder_checkpoint": VOCODER_CHECKPOINT,
        "wav2lip_python": WAV2LIP_PYTHON,
        "wav2lip_checkpoint": WAV2LIP_CHECKPOINT,
        "wav2lip_inference": WAV2LIP_INFERENCE,
        "syncnet_python": SYNCNET_PYTHON,
        "syncnet_model": SYNCNET_MODEL,
        "ffmpeg": FFMPEG,
        "ffprobe": FFPROBE,
    }
