"""Frozen configuration for the MFA-linear real-video SyncNet prototype."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000
VIDEO_FPS = 25
SAMPLES_PER_VIDEO_FRAME = SAMPLE_RATE // VIDEO_FPS
SEGMENT_FRAMES = 96
SEGMENT_SAMPLES = SEGMENT_FRAMES * SAMPLES_PER_VIDEO_FRAME
SYNCNET_WINDOW_FRAMES = 5
SYNCNET_MFCC_PER_VIDEO_FRAME = 4
SYNCNET_MFCC_WINDOW = 20
VSHIFT = 15
CURVE_SIZE = 2 * VSHIFT + 1
Q = 0.001
TARGET_GAP_MIN = 2 * Q
MEL_TRUST_LIMIT = 0.10
SATURATION_FRACTION_MAX = 1e-4

SEED = 20_260_903
MODEL_CHANNELS = 32
MODEL_DILATIONS = (1, 2, 4, 8, 16, 32, 64, 128)
MODEL_RESIDUAL_SCALE = 0.05
LEARNING_RATE = 1e-4
ADAM_BETAS = (0.9, 0.999)
ADAM_EPS = 1e-8
WEIGHT_DECAY = 0.01
GRAD_CLIP_NORM = 1.0
P1_STEPS = 20
P2_STEPS = 100

P1_SAMPLE_ID = "lrs3_6ORDQFh0Byw_00004"
P2_RECORDS = (
    ("lrs3_6ORDQFh0Byw_00004", "6ORDQFh0Byw"),
    ("lrs3_70VZ1SOzSnc_00003", "70VZ1SOzSnc"),
    ("lrs3_6xtmm0MnaS0_00006", "6xtmm0MnaS0"),
    ("lrs3_73jPh0eRPSY_00002", "73jPh0eRPSY"),
)
P2_ASSET_LOCKS = {
    "lrs3_6ORDQFh0Byw_00004": {
        "policy": "65d224814b331903102985549f21605644fbecbcfe65e382710c56f9629149f1",
        "source": "04f447710c48c4a5b2f18ee8221229c27c485ff95bd652c89129e50f9ef33526",
        "crop": "5ebfbdc7bd76d8e305137b62dcd40b6b0bd2b5361117b1ef0ab86a780c0b1c46",
        "crop_audio": "6daff138aac92d566676e5d1b9355bf7af30834831fbbcd43efb9f2945ef7041",
        "natural": "b94ba97aebb0a978fe5f933faa46f9523f999430623250dd8180086c6b27496d",
        "track": "2ff3b06a49ee9b46cf0a33928a22f83e2be10433f4a8c92b83bd6c4f27befd06",
        "mfa": "191b3bfeedb63c18da94b5ef80d460442141873e77f73c81b5b085b8b2b0b245",
    },
    "lrs3_70VZ1SOzSnc_00003": {
        "policy": "fb418d12072cb17f95280eb41c3216a324bebd659879865fe02cb4228618e535",
        "source": "f51d0f98a86334f123d4ededee550b5db1c20729c32f3abd9849558ea4737f60",
        "crop": "0be946175dd51057c43646fbfe6db566a211a35c27c35222ae705dea9e466c49",
        "crop_audio": "f40375368286bb399b81ec5eb23ebb26d579807ba54a08e4f215aecf40f45706",
        "natural": "0bf6e215d4531e225b80c4b9fe581f4dd79565f3c70917aab86407ac5b0856cd",
        "track": "211bfaf9862162b3c51c71f3aa88d13b1f8ac65e9e49aae508ed625bc0c5ad7e",
        "mfa": "c39bd2012dff7082fe75b4067acd8df2aa0435c0f2016eb156be0712946704f5",
    },
    "lrs3_6xtmm0MnaS0_00006": {
        "policy": "af8e7cfbc1318f2b27b7feb15c2e5988ec60bab39b0e1e55824ac7f0ec2db032",
        "source": "cf6a730a912f0b4936e1b68cd37b3991b18a5e8e403064298f073d752452b9a3",
        "crop": "8716c0ee00c7d9c3dbe0a291f70a6c7ed5f0042be75a557fc707faf8bf65fb5e",
        "crop_audio": "4ec1aae81bfa70e17ad30b11c4a4ac5fb6da4895403e90c6ea6711bf37231cc1",
        "natural": "b8010e100315c543c53bcb99fa4a96a079b4112eb7c58b22e028e4efa401b432",
        "track": "fa84655fcc30b3995d7628a32e0b7f0e801081baa5c24fda9729a0fbdb183fad",
        "mfa": "2a6eba933670d15e165f0e94bf4606f139e13905eeb7f271713d46f9514a76b5",
    },
    "lrs3_73jPh0eRPSY_00002": {
        "policy": "6bdccc4371d2b52af318f0d4f9fd4e7d24143f3124d7b2604fe43e8486ab9301",
        "source": "a4f4830acf2cb89f0ce9771b6f6742d9b30f0ab622809c19bf2fafc7c1177130",
        "crop": "c9e4b98547406a1b982a089e6494f7564890c9cd36d178bef8bd67c34a8ac68a",
        "crop_audio": "6def05f8137a6893cc72a83b0ee0830dbe8812090f3e8e7e13f86cef79515576",
        "natural": "334f4a4cf13f44e1fa14d473646ebbf6b738c17d68ac003884397a6e6dcd52fa",
        "track": "2ed32e5dd0f6cc3b0df5c51196aa7716caf2cd81853068108e3946d0edc2b7a3",
        "mfa": "74fc418aae502fdee44881d3ac41bbe15dd09a5493b8905b1a804030227264d1",
    },
}
P1_MFA_PARENT_SHA256 = "191b3bfeedb63c18da94b5ef80d460442141873e77f73c81b5b085b8b2b0b245"
P1_STEP0_PCM16_SHA256 = "03b0b367ecb632fd9b19a30bd5f288158466a7fc2ad20528d323138441338cee"
P1_MFA_SUMMARY_SHA256 = "2c47347933b636eb59c15c112c5b91bc228da274667ac6211384e0d1da3960a4"
P1_POLICY_RECORD_SHA256 = "65d224814b331903102985549f21605644fbecbcfe65e382710c56f9629149f1"
P1_SOURCE_VIDEO_SHA256 = "04f447710c48c4a5b2f18ee8221229c27c485ff95bd652c89129e50f9ef33526"
P1_LEGACY_CROP_SHA256 = "5ebfbdc7bd76d8e305137b62dcd40b6b0bd2b5361117b1ef0ab86a780c0b1c46"
P1_CROP_AUDIO_SHA256 = "6daff138aac92d566676e5d1b9355bf7af30834831fbbcd43efb9f2945ef7041"
P1_NATURAL_AUDIO_SHA256 = "b94ba97aebb0a978fe5f933faa46f9523f999430623250dd8180086c6b27496d"
P1_TRACK_SHA256 = "2ff3b06a49ee9b46cf0a33928a22f83e2be10433f4a8c92b83bd6c4f27befd06"
P1_CROP_FRAME0_SHA256 = "3d39035d1b008b5dea9ce6d066781f35bb644e38cf506fb8a1517b49c140fc55"
P1_CROP_FRAME95_SHA256 = "b94da4624fac276aa6e1eb7a57fbdf74348568f2ffe703c2dd9340c89d5df88d"
P1_CROP_96_CONCAT_SHA256 = "5d9e1c188889320a8dae0c0d0b1d4816a8c376ee3f30e0657039064e44374f8d"
SYNCNET_MODEL_SHA256 = "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442"

TRAINING_RECORD_FIELDS = frozenset(
    {
        "mfa_linear_tts_waveform",
        "visual_embedding",
        "target_offset",
        "pristine_curve",
    }
)
MODEL_INPUT_FIELDS = frozenset({"mfa_linear_tts_waveform"})
FORBIDDEN_FIELD_TOKENS = (
    "natural",
    "identity",
    "speaker",
    "transcript",
    "text",
    "video_feature",
    "video_condition",
    "second_audio",
    "key_value",
    "cross_attention",
)


@dataclass(frozen=True)
class PrototypeConfig:
    seed: int = SEED
    device: str = "cuda"
    dtype: str = "float32"
    mfcc_internal_dtype: str = "float64"
    deterministic_algorithms: bool = True
    cudnn_benchmark: bool = False
    amp: bool = False
    scheduler: bool = False
    optimizer: str = "AdamW"
    learning_rate: float = LEARNING_RATE
    betas: tuple[float, float] = ADAM_BETAS
    epsilon: float = ADAM_EPS
    weight_decay: float = WEIGHT_DECAY
    gradient_clip_norm: float = GRAD_CLIP_NORM
    model_channels: int = MODEL_CHANNELS
    model_dilations: tuple[int, ...] = MODEL_DILATIONS
    residual_scale: float = MODEL_RESIDUAL_SCALE
    p1_steps: int = P1_STEPS
    p2_steps: int = P2_STEPS
    sample_rate: int = SAMPLE_RATE
    video_fps: int = VIDEO_FPS
    segment_frames: int = SEGMENT_FRAMES
    segment_samples: int = SEGMENT_SAMPLES
    vshift: int = VSHIFT
    q: float = Q
    mel_trust_limit: float = MEL_TRUST_LIMIT
    saturation_fraction_max: float = SATURATION_FRACTION_MAX

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["betas"] = list(self.betas)
        payload["model_dilations"] = list(self.model_dilations)
        payload["checkpoint_selection"] = [0, self.p1_steps]
        payload["trainable_input_modalities"] = sorted(MODEL_INPUT_FIELDS)
        return payload

    def validate(self) -> None:
        expected = PrototypeConfig()
        if self != expected:
            raise ValueError("scientific configuration is frozen; changed constants are not allowed")


@dataclass(frozen=True)
class AssetRoots:
    repo: Path
    policy_records: Path
    mfa_linear: Path
    syncnet_model: Path

    @classmethod
    def defaults(cls, repo: str | Path) -> "AssetRoots":
        root = Path(repo).resolve()
        return cls(
            repo=root,
            policy_records=root
            / ".claude/worktrees/lrs3-wavlm-resynthesis-50/tmp/lrs3_policy_a1_200_20260828/policy_cohort/records",
            mfa_linear=root / "runs/lrs3_cem_fixed_video_20260820/12_policy_train_mfa_linear",
            syncnet_model=root / "third_party/syncnet_python/data/syncnet_v2.model",
        )
