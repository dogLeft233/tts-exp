from __future__ import annotations

import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CHANGE_ROOT = REPO / "openspec/changes/test-wav2lip-spectral-structure-replacement"
PROPOSAL = CHANGE_ROOT / "proposal.md"
DESIGN = CHANGE_ROOT / "design.md"
SPEC = CHANGE_ROOT / "specs/wav2lip-spectral-structure-replacement/spec.md"

PLATEAU_ROOT = REPO / "runs/wav2lip_integer_plateau_control_20260907_all_v2"
PLATEAU_FINAL = PLATEAU_ROOT / "final.json"
PLATEAU_PROTOCOL = PLATEAU_ROOT / "protocol.json"
PLATEAU_GENERATED_VALIDATION = PLATEAU_ROOT / "generated_validation.json"
PLATEAU_AUDIO_MANIFEST = PLATEAU_ROOT / "audio/manifest.json"

CONFIRMATION_ROOT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol"
CONFIRMATION_PROTOCOL = CONFIRMATION_ROOT / "protocol.json"
CONFIRMATION_COHORT = CONFIRMATION_ROOT / "cohort.json"

ROI_ROOT = REPO / "runs/wav2lip_face_roi_replacement_20260906_host_fix1"
ROI_PROTOCOL = ROI_ROOT / "protocol.json"
ROI_MANIFEST = ROI_ROOT / "roi/manifest.json"

FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")
SYNCNET_PYTHON = Path("/home/wjj/.venvs/syncnet/bin/python")
WAV2LIP_PYTHON = Path("/home/wjj/.venvs/wav2lip/bin/python")
GENERATION_WORKER = REPO / "scripts/experiments/wav2lip_face_roi_replacement/generation_worker.py"
WAV2LIP_CHECKPOINT = REPO / "third_party/Wav2Lip/checkpoints/wav2lip_gan.pth"
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
SYNCNET_MODEL = REPO / "third_party/syncnet_python/data/syncnet_v2.model"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"

PLATEAU_FINAL_SHA256 = "2c3eb59f6a1ee25916b65b899a3b7ba462dba26b4b1d337e71a062d6e4dfd3fb"
PLATEAU_PROTOCOL_SHA256 = "483e89a5de249bccefdcfd9e376b7a612c92d3f7085f5a47168bdca77d9ad384"
PLATEAU_GENERATED_VALIDATION_SHA256 = "0f91ffa3c5a0e1eb47061af239f23dbac168fda0e6db29fe7a054b0fb9fc443b"
PLATEAU_AUDIO_MANIFEST_SHA256 = "ed5a8d028bbbfff0c2beb6b67b081fe267fad5e186be76ee8d3a16b122507fc8"
CONFIRMATION_PROTOCOL_SHA256 = "70580209b05d940ab2e73fae25a4d8a243146bc4f7ccb2f82e66a26d4ad7bef6"
CONFIRMATION_COHORT_SHA256 = "b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b"
ROI_PROTOCOL_SHA256 = "835ff09188653a61b0e1de3fdcfb8c6344ad3b6318f84ee4814c37d5cc369cc4"
ROI_MANIFEST_SHA256 = "4d005b26e3e108f534597d5cfe4832633e680e325a33332aa98fe99d7f7e6723"

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
FRAME_WIDTH = 224
FRAME_HEIGHT = 224
VSHIFT = 15
WINDOW_FRAMES = 5
MATRIX_COLUMNS = 2 * VSHIFT + 1
BATCH_SIZE = 20
TORCH_THREADS = 4
STFT_NFFT = 1024
STFT_WIN_LENGTH = 1024
STFT_HOP_LENGTH = 256
MAGNITUDE_FLOOR = 1e-7
ALPHA = 0.75
RMS_EPSILON = 1e-12
PEAK_GAP_THRESHOLD = 0.010
OFFSET_TOLERANCE_FRAMES = 1
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260908
EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22
MIN_BASELINE_RECORDS = 20
MIN_SUCCESS_RECORDS = 18
DISK_FREE_MIN_BYTES = 15 * 1024**3
PCM_CONTAINER = "mono_pcm16_16k"

ARM_N = "N"
ARM_N_REPEAT = "N_REPEAT"
ARM_RT = "RT"
ARM_MAG = "MAG"
ARM_ENV = "ENV"
ARM_P = "P"
STAGE_A_ARMS = (ARM_N, ARM_N_REPEAT, ARM_RT)
STAGE_B_ARMS = (ARM_MAG, ARM_ENV)
ALL_AUDIO_ARMS = (ARM_N, ARM_N_REPEAT, ARM_RT, ARM_MAG, ARM_ENV, ARM_P)


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    return REPO / "runs" / f"wav2lip_spectral_structure_replacement_{validate_run_id(run_id)}"


def cell_key(sample_id: str, video_arm: str, audio_arm: str) -> str:
    return f"{sample_id}__{video_arm}__{audio_arm}"


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
    def audio_manifest(self) -> Path:
        return self.root / "audio/manifest.json"

    @property
    def videos_manifest(self) -> Path:
        return self.root / "videos/manifest.json"

    @property
    def scores_manifest(self) -> Path:
        return self.root / "scores/manifest.json"

    @property
    def control_analysis(self) -> Path:
        return self.root / "control_analysis.json"

    @property
    def control_validation(self) -> Path:
        return self.root / "control_validation.json"

    @property
    def analysis(self) -> Path:
        return self.root / "analysis.json"

    @property
    def validation(self) -> Path:
        return self.root / "validation.json"

    @property
    def final(self) -> Path:
        return self.root / "final.json"

    @property
    def result(self) -> Path:
        return self.root / "result.md"

    @property
    def audio_dir(self) -> Path:
        return self.root / "audio"

    @property
    def videos_dir(self) -> Path:
        return self.root / "videos"

    @property
    def scores_dir(self) -> Path:
        return self.root / "scores"


@dataclass(frozen=True)
class FrozenConfig:
    schema_version: int = 1
    protocol_id: str = "wav2lip_spectral_structure_replacement"
    protocol_revision: str = "spectral_structure_v1"
    sample_rate: int = SAMPLE_RATE
    fps: int = FPS
    samples_per_frame: int = SAMPLES_PER_FRAME
    frame_width: int = FRAME_WIDTH
    frame_height: int = FRAME_HEIGHT
    vshift: int = VSHIFT
    window_frames: int = WINDOW_FRAMES
    matrix_columns: int = MATRIX_COLUMNS
    stft_n_fft: int = STFT_NFFT
    stft_win_length: int = STFT_WIN_LENGTH
    stft_hop_length: int = STFT_HOP_LENGTH
    magnitude_floor: float = MAGNITUDE_FLOOR
    alpha: float = ALPHA
    peak_gap_threshold: float = PEAK_GAP_THRESHOLD
    offset_tolerance_frames: int = OFFSET_TOLERANCE_FRAMES
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT
    min_baseline_records: int = MIN_BASELINE_RECORDS
    min_success_records: int = MIN_SUCCESS_RECORDS
    disk_free_min_bytes: int = DISK_FREE_MIN_BYTES
    ordered_sample_id_sha256: str = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value.update(
            {
                "stage_a_audio_arms": list(STAGE_A_ARMS),
                "stage_b_audio_arms": list(STAGE_B_ARMS),
                "all_audio_arms": list(ALL_AUDIO_ARMS),
                "stage_a_cells_per_record": [[arm, ARM_N] for arm in STAGE_A_ARMS] + [[ARM_N, ARM_P]],
                "stage_b_cells_per_record": [[arm, ARM_N] for arm in STAGE_B_ARMS],
                "stft_window": "periodic Hann",
                "stft_center": True,
                "stft_pad_mode": "reflect",
                "stft_normalized": False,
                "stft_onesided": True,
                "pcm_rounding": "np.rint; clip(-32768,32767); little-endian int16",
                "bootstrap": "source_group means; sorted groups; PCG64/default_rng; shared draws; linear quantiles",
                "u_mask": "plateau protocol U=PLUS||MINUS; never Q/source rows",
                "fixed_flags": {
                    "training_authorized": False,
                    "generalization_established": False,
                    "historical_gate_repaired": False,
                    "legacy_bridge_executed": False,
                },
            }
        )
        return value


def spec_bindings(file_sha256: Any) -> dict[str, dict[str, str]]:
    return {
        name: {"path": str(path.resolve()), "sha256": file_sha256(path)}
        for name, path in (("proposal", PROPOSAL), ("design", DESIGN), ("spec", SPEC))
    }


def environment() -> dict[str, Any]:
    return {
        "python": sys.executable,
        "python_version": sys.version,
        "syncnet_python": str(SYNCNET_PYTHON),
        "wav2lip_python": str(WAV2LIP_PYTHON),
        "syncnet_model": str(SYNCNET_MODEL.resolve()),
        "syncnet_model_sha256": SYNCNET_MODEL_SHA256,
        "wav2lip_checkpoint": str(WAV2LIP_CHECKPOINT.resolve()),
        "wav2lip_checkpoint_sha256": WAV2LIP_CHECKPOINT_SHA256,
        "generation_worker": str(GENERATION_WORKER.resolve()),
        "ffmpeg": str(FFMPEG),
        "ffprobe": str(FFPROBE),
    }
