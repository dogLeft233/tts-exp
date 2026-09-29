from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = "lrs3-dac16k-codec-identity"
PROTOCOL_ID = "lrs3_dac16k_codec_identity_20260904"
RUN_ROOT = REPO / "runs/lrs3_dac16k_codec_identity_20260904"
STAGES = {
    "00_protocol": RUN_ROOT / "00_protocol",
    "01_audio": RUN_ROOT / "01_audio",
    "02_fidelity": RUN_ROOT / "02_fidelity",
    "03_videos": RUN_ROOT / "03_videos",
    "04_matrix": RUN_ROOT / "04_matrix",
    "05_analysis": RUN_ROOT / "05_analysis",
    "06_final": RUN_ROOT / "06_final",
}

PARENT_SUMMARY = REPO / "runs/lrs3_wavlm_hifigan_direct_20260826/summary.json"
EXPECTED_PARENT_SUMMARY_SHA256 = "96a29d7355ca3916ef2f60935147d9f20a123a8a7aefa12db130bbcf07eb08d5"
EXPECTED_SOURCE_MANIFEST_SHA256 = "34efb614d72e183727d87d40a05807d4c6ccda954cadc14f76bc8cbd01ee255f"
EXPECTED_SAMPLE_ID_SHA256 = "8e47514fdf877e9a57533e8719bd16fab1b96a99d8cb47cfa007c96689326b94"
EXPECTED_RECORD_COUNT = 50
EXPECTED_SOURCE_GROUP_COUNT = 38

DAC_TAG = "0.0.5"
DAC_SOURCE_COMMIT = "408235a9dcd2983684c87615a1bc2a8954f6eb47"
DAC_SOURCE_URL = "https://github.com/descriptinc/descript-audio-codec"
DAC_CHECKPOINT_URL = "https://github.com/descriptinc/descript-audio-codec/releases/download/0.0.5/weights_16khz.pth"
DAC_MODEL_TYPE = "16khz"
DAC_BITRATE = "8kbps"
DAC_SAMPLE_RATE = 16_000
DAC_HOP_LENGTH = 320
DAC_EXPECTED_CHECKPOINT_SIZE = 296_820_733
DAC_SOURCE_ROOT = Path.home() / ".cache" / f"descript-audio-codec-{DAC_TAG}-{DAC_SOURCE_COMMIT}"
DAC_CHECKPOINT = Path.home() / ".cache" / "descript" / "16khz" / DAC_TAG / "dac" / "weights.pth"

SAMPLE_RATE = 16_000
WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
WAV2LIP_CHECKPOINT = WAV2LIP_ROOT / "checkpoints/wav2lip_gan.pth"
WAV2LIP_PYTHON = Path.home() / ".venvs" / "wav2lip" / "bin" / "python"
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
SYNCNET_PYTHON = Path.home() / ".venvs" / "syncnet" / "bin" / "python"
FFMPEG = "ffmpeg"
FFPROBE = "ffprobe"
MIN_TRACK = 50

BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260904
IDENTITY_MARGIN = -0.10

DRIVER_ARMS = ("N", "W", "D")
AUDIO_ARMS = ("N", "W", "D")
MATRIX_CELLS = tuple(f"V_{video}/A_{audio}" for video in DRIVER_ARMS for audio in AUDIO_ARMS)

NO_SEALED_MEDIA_TOKENS = ("/test/", "/tests/", "/val/", "/validation/", "/heldout/")
