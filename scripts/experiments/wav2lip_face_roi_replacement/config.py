from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[3]
CHANGE_ROOT = REPO / "openspec/changes/validate-wav2lip-face-roi-replacement"
PROTOCOL_ID = "wav2lip_face_roi_replacement"
PROTOCOL_REVISION = "roi_v1"

SPEC = CHANGE_ROOT / "specs/wav2lip-face-roi-replacement/spec.md"
PROPOSAL = CHANGE_ROOT / "proposal.md"
DESIGN = CHANGE_ROOT / "design.md"
INHERITED_TIMING_SPEC = REPO / "openspec/changes/diagnose-lrs3-wav2lip-timing-transfer/specs/lrs3-wav2lip-timing-transfer-diagnostic/spec.md"
INHERITED_BRIDGE_SPEC = REPO / "openspec/changes/confirm-lrs3-natural-to-tts-bridge/specs/lrs3-natural-to-tts-bridge-confirmation/spec.md"

COHORT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json"
TIMING_FINAL = REPO / "runs/lrs3_wav2lip_timing_transfer_20260905_v8/final.json"
TIMING_PROTOCOL = REPO / "runs/lrs3_wav2lip_timing_transfer_20260905_v8/protocol.json"
BRIDGE_FINAL = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904/04_final/final.json"
BRIDGE_PROTOCOL = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/protocol.json"
BRIDGE_AUDIO_MANIFEST = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904/01_audio/audio_manifest.json"
CALIBRATION_AUDIO_MANIFEST = REPO / "runs/lrs3_local_timing_control_calibration_20260905_v5/02_audio/audio_manifest.json"
TAIL_FINAL = REPO / "runs/lrs3_real_video_local_timing_20260905_tail_v2/final.json"
TAIL_PROTOCOL = REPO / "runs/lrs3_real_video_local_timing_20260905_tail_v2/protocol.json"

COHORT_SHA256 = "b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b"
TIMING_FINAL_SHA256 = "4bc4dcab90adec07ec6f3e7df34914fc5211fd28aba54f5ffb6a7678969f763f"
BRIDGE_FINAL_SHA256 = "df0ca9767e70ccc384c86c1da23243c53fa609be12abd1dda20f9075b6732c6e"
CALIBRATION_AUDIO_SHA256 = "0617cb895574516b77aa430c44f0480eefe4be0e0582b3e8fdec4addf46e68e1"
BRIDGE_AUDIO_SHA256 = "2c279dc5a74dbeb53ae547235347870c6664fd9caf656a594b48a8310d9bd288"
TAIL_FINAL_SHA256 = "5df3615f7b2219e60ad62fb49792cb30eccd068441d1ab5b8006d0fb92aee24d"
ORDERED_SAMPLE_ID_SHA256 = "5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72"

INHERITED_TIMING_SPEC_SHA256 = "9514383e714624d23d0a1f85ce561aa93a5621bb85daf44df5fd946dc1a8df34"
INHERITED_BRIDGE_SPEC_SHA256 = "984a87ce6921bc22aabb7079b1e9644ca86394315400be91f98c9519f2860080"

WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
WAV2LIP_CHECKPOINT = WAV2LIP_ROOT / "checkpoints/wav2lip_gan.pth"
WAV2LIP_PYTHON = Path.home() / ".venvs/wav2lip/bin/python"
WAV2LIP_CHECKPOINT_SHA256 = "ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8"
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_PYTHON = Path.home() / ".venvs/syncnet/bin/python"
SYNCNET_WORKER = REPO / "scripts/experiments/lrs3_real_video_local_timing/syncnet_worker.py"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")

SAMPLE_RATE = 16_000
FPS = 25
SAMPLES_PER_FRAME = SAMPLE_RATE // FPS
PCM_CHANNELS = 1
PCM_SAMPLE_WIDTH = 2
CROP_SIZE = 224
CROP_SCALE = 0.40
WAV2LIP_SIZE = 96
WAV2LIP_BATCH_SIZE = 4
FACE_DET_BATCH_SIZE = 1
SYNCNET_BATCH_SIZE = 20
VSHIFT = 15
WINDOW_FRAMES = 5
MIN_LOCAL_ROWS = 5
PEAK_GAP_THRESHOLD = 0.010
OFFSET_TOLERANCE_FRAMES = 1
MIN_BASELINE_RECORDS = 20
MIN_SUCCESS_RECORDS = 18
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260905

AUDIO_N = "N"
AUDIO_N_REPEAT = "N_REPEAT"
AUDIO_W = "W"
AUDIO_BRIDGE = "BRIDGE_075"
AUDIO_ARMS = (AUDIO_N, AUDIO_N_REPEAT, AUDIO_W, AUDIO_BRIDGE)
VIDEO_R = "R"
VIDEO_GN = "G_N"
VIDEO_GNR = "G_NR"
VIDEO_GW = "G_W"
VIDEO_GB = "G_B"
CONTROL_VIDEO_ARMS = (VIDEO_R, VIDEO_GN, VIDEO_GNR, VIDEO_GW)
ALL_VIDEO_ARMS = (*CONTROL_VIDEO_ARMS, VIDEO_GB)
CONTROL_MAIN_CELL_SPECS = (
    (VIDEO_R, AUDIO_N),
    (VIDEO_R, AUDIO_W),
    (VIDEO_GN, AUDIO_N),
    (VIDEO_GN, AUDIO_W),
    (VIDEO_GNR, AUDIO_N),
    (VIDEO_GW, AUDIO_N),
    (VIDEO_GW, AUDIO_W),
)
CONTROL_REPEAT_CELL_SPECS = ((VIDEO_R, AUDIO_N), (VIDEO_GN, AUDIO_N))
BRIDGE_CELL_SPECS = ((VIDEO_GB, AUDIO_N), (VIDEO_GB, AUDIO_BRIDGE))

EXPECTED_RECORD_COUNT = 22
EXPECTED_SOURCE_GROUP_COUNT = 22
EXPECTED_CONTROL_VIDEO_COUNT = EXPECTED_RECORD_COUNT * 3
EXPECTED_CONTROL_MAIN_CELL_COUNT = EXPECTED_RECORD_COUNT * len(CONTROL_MAIN_CELL_SPECS)
EXPECTED_CONTROL_REPEAT_CELL_COUNT = EXPECTED_RECORD_COUNT * len(CONTROL_REPEAT_CELL_SPECS)
EXPECTED_CONTROL_SCORE_COUNT = EXPECTED_CONTROL_MAIN_CELL_COUNT + EXPECTED_CONTROL_REPEAT_CELL_COUNT
EXPECTED_BRIDGE_VIDEO_COUNT = EXPECTED_RECORD_COUNT
EXPECTED_BRIDGE_CELL_COUNT = EXPECTED_RECORD_COUNT * len(BRIDGE_CELL_SPECS)

NO_SEALED_MEDIA_TOKENS = ("/test/", "/tests/", "/val/", "/validation/", "/heldout/")


def validate_run_id(run_id: str) -> str:
    if not run_id or re.fullmatch(r"[A-Za-z0-9_-]+", run_id) is None:
        raise ValueError("run-id must contain only letters, numbers, underscores, and hyphens")
    return run_id


def run_root_for(run_id: str) -> Path:
    validate_run_id(run_id)
    return REPO / "runs" / f"wav2lip_face_roi_replacement_{run_id}"


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
    def audio_manifest(self) -> Path:
        return self.root / "audio_manifest.json"

    @property
    def roi_manifest(self) -> Path:
        return self.root / "roi" / "manifest.json"

    @property
    def roi_review(self) -> Path:
        return self.root / "roi" / "review.json"

    @property
    def videos(self) -> Path:
        return self.root / "videos"

    @property
    def media(self) -> Path:
        return self.root / "media"

    @property
    def scores(self) -> Path:
        return self.root / "scores"

    @property
    def control(self) -> Path:
        return self.root / "control.json"

    @property
    def bridge(self) -> Path:
        return self.root / "bridge.json"

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
    wav2lip_size: int = WAV2LIP_SIZE
    wav2lip_batch_size: int = WAV2LIP_BATCH_SIZE
    face_det_batch_size: int = FACE_DET_BATCH_SIZE
    syncnet_batch_size: int = SYNCNET_BATCH_SIZE
    vshift: int = VSHIFT
    window_frames: int = WINDOW_FRAMES
    min_local_rows: int = MIN_LOCAL_ROWS
    peak_gap_threshold: float = PEAK_GAP_THRESHOLD
    offset_tolerance_frames: int = OFFSET_TOLERANCE_FRAMES
    min_baseline_records: int = MIN_BASELINE_RECORDS
    min_success_records: int = MIN_SUCCESS_RECORDS
    bootstrap_draws: int = BOOTSTRAP_DRAWS
    bootstrap_seed: int = BOOTSTRAP_SEED
    expected_record_count: int = EXPECTED_RECORD_COUNT
    expected_source_group_count: int = EXPECTED_SOURCE_GROUP_COUNT
    ordered_sample_id_sha256: str = ORDERED_SAMPLE_ID_SHA256
    wav2lip_checkpoint_sha256: str = WAV2LIP_CHECKPOINT_SHA256
    syncnet_model_sha256: str = SYNCNET_MODEL_SHA256

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "audio_arms": list(AUDIO_ARMS),
                "control_video_arms": list(CONTROL_VIDEO_ARMS),
                "control_main_cell_specs": [[v, a] for v, a in CONTROL_MAIN_CELL_SPECS],
                "control_repeat_cell_specs": [[v, a] for v, a in CONTROL_REPEAT_CELL_SPECS],
                "bridge_cell_specs": [[v, a] for v, a in BRIDGE_CELL_SPECS],
                "wav2lip_preprocess": "official 96x96 resize; lower-half mask; official mel chunks; no network change",
                "face_detection": "official FaceAlignment SFD; flip_input=False; pads=[0,10,0,0]; resize_factor=1; nosmooth=True; batch=1",
                "pcm_policy": "16 kHz mono PCM16; exact decoded bytes; no resampling or truncation",
                "bootstrap_policy": "sorted source groups; NumPy default_rng/PCG64; 10000 draws; percentile 2.5/97.5; seed reset per metric",
                "forbidden_operations": [
                    "full_frame_fallback",
                    "manual_or_interpolated_boxes",
                    "score_based_box_selection",
                    "audio_resampling",
                    "audio_truncation",
                    "video_looping_or_padding",
                    "global_offset_alignment",
                    "score_based_retry",
                    "record_filtering",
                    "training",
                    "mfa_or_dtw",
                    "heldout_or_sealed_media_access",
                    "history_overwrite",
                ],
            }
        )
        return payload


def spec_bindings() -> dict[str, dict[str, str]]:
    from .common import file_sha256

    expected = {
        "spec": (SPEC, None),
        "proposal": (PROPOSAL, None),
        "design": (DESIGN, None),
        "inherited_timing_spec": (INHERITED_TIMING_SPEC, INHERITED_TIMING_SPEC_SHA256),
        "inherited_bridge_spec": (INHERITED_BRIDGE_SPEC, INHERITED_BRIDGE_SPEC_SHA256),
    }
    result: dict[str, dict[str, str]] = {}
    for name, (path, frozen_hash) in expected.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = file_sha256(path)
        if frozen_hash is not None and actual != frozen_hash:
            raise ValueError(f"registered spec hash differs: {name}")
        result[name] = {"path": str(path.resolve()), "sha256": actual}
    return result


def cell_key(video_arm: str, audio_arm: str, repeat: bool = False) -> str:
    value = f"{video_arm}__{audio_arm}"
    return f"{value}__repeat" if repeat else value
