"""Frozen configuration for the masked TTS reconstruction prototype."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

SAMPLE_RATE = 16_000
NATURAL_SUPPORT_SAMPLES = 61_440
NATURAL_SUPPORT_SECONDS = NATURAL_SUPPORT_SAMPLES / SAMPLE_RATE
MEL_BINS = 80
MEL_HOP_SAMPLES = 200
MEL_FRAME_SECONDS = MEL_HOP_SAMPLES / SAMPLE_RATE
WAVLM_HOP_SAMPLES = 320
WAVLM_FRAME_SECONDS = WAVLM_HOP_SAMPLES / SAMPLE_RATE
WAVLM_DIM = 1024
WINDOW_FRAMES = 96
MASK_GUARD_FRAMES = 4
MIN_CORE_FRAMES = 4
MAX_CORE_FRAMES = 40
MIN_TTS_FRAMES = 2
HIDDEN_WIDTH = 128
CONTEXT_DILATIONS = (1, 2, 4, 8)
TTS_DILATIONS = (1, 2)
KERNEL_SIZE = 3
MAX_PARAMETERS = 1_500_000
SEEDS = (20260901, 20260902, 20260903)
BOOTSTRAP_SEED = 20260901
BOOTSTRAP_DRAWS = 10_000
TRAIN_STEPS = 600
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP = 1.0
VELOCITY_WEIGHT = 0.25
CPU_THREADS = 1

TRAIN_GROUPS = (
    "6WeS1bXRBOk",
    "6XNrzmh0EVs",
    "6XS8TA4RBog",
    "6YKYo00mFAg",
    "6ZiN9ZJT294",
    "6qTfX4U6CS0",
)
EVAL_GROUPS = (
    "6tSlMoMNSlY",
    "6tpsSu5O0Ws",
    "6wk4dkYSrV0",
    "6xtmm0MnaS0",
)
GROUP_ORDER = TRAIN_GROUPS + EVAL_GROUPS
ADMINISTRATIVE_LABELS = frozenset(
    {"", "sil", "sp", "spn", "<eps>", "<sil>", "silence"}
)

@dataclass(frozen=True)
class AssetPaths:
    asset_root: Path

    @property
    def split(self) -> Path:
        return self.asset_root / "data/splits/exp_a_split.json"

    @property
    def source_manifest(self) -> Path:
        return self.asset_root / "tmp/lrs3_policy_a1_200_20260828/policy_cohort/source_manifest.json"

    @property
    def record_dir(self) -> Path:
        return self.asset_root / "tmp/lrs3_policy_a1_200_20260828/policy_cohort/records"

    @property
    def feature_dir(self) -> Path:
        return self.asset_root / "tmp/runs/diagnostic_week1/features"

    @property
    def alignment_dir(self) -> Path:
        return self.asset_root / "tmp/runs/diagnostic_week1/week2_mfa/phone_alignments"

    def record(self, sample_id: str) -> Path:
        return self.record_dir / f"{sample_id}.json"

    def feature(self, sample_id: str, modality: str) -> Path:
        return self.feature_dir / f"{sample_id}_{modality}_wavlm_l6.npy"

    def alignment(self, sample_id: str) -> Path:
        return self.alignment_dir / f"{sample_id}.json"


def default_asset_root(repo_root: Path | None = None) -> Path:
    repo = (repo_root or Path(__file__).resolve().parents[3]).resolve()
    return repo / ".claude/worktrees/lrs3-wavlm-resynthesis-50"


def frozen_constants() -> dict[str, object]:
    return {
        "sample_rate": SAMPLE_RATE,
        "natural_support_samples": NATURAL_SUPPORT_SAMPLES,
        "mel_bins": MEL_BINS,
        "mel_hop_samples": MEL_HOP_SAMPLES,
        "wavlm_hop_samples": WAVLM_HOP_SAMPLES,
        "wavlm_dim": WAVLM_DIM,
        "window_frames": WINDOW_FRAMES,
        "mask_guard_frames": MASK_GUARD_FRAMES,
        "min_core_frames": MIN_CORE_FRAMES,
        "max_core_frames": MAX_CORE_FRAMES,
        "min_tts_frames": MIN_TTS_FRAMES,
        "hidden_width": HIDDEN_WIDTH,
        "context_dilations": list(CONTEXT_DILATIONS),
        "tts_dilations": list(TTS_DILATIONS),
        "kernel_size": KERNEL_SIZE,
        "max_parameters": MAX_PARAMETERS,
        "seeds": list(SEEDS),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "train_steps": TRAIN_STEPS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "gradient_clip": GRADIENT_CLIP,
        "velocity_weight": VELOCITY_WEIGHT,
        "cpu_threads": CPU_THREADS,
        "train_groups": list(TRAIN_GROUPS),
        "evaluation_groups": list(EVAL_GROUPS),
        "group_order": list(GROUP_ORDER),
        "administrative_labels": sorted(ADMINISTRATIVE_LABELS),
    }
