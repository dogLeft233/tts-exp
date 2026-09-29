from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
EXPERIMENT = "lrs3-mfa-dtw-replacement"
PROTOCOL_ID = "lrs3_mfa_dtw_replacement_20260904"
RUN_ROOT = REPO / "runs/lrs3_mfa_dtw_replacement_20260904"
SEED = 20260903
BOOTSTRAP_DRAWS = 10_000
BAND_RATIO = 0.5
DTW_FRAME_OWNERSHIP_POLICY = "center_exclusive_v1"
EXPECTED_RECORD_COUNT = 133
EXPECTED_ALIGNMENT_COUNT = 146
EXPECTED_COHORT_HASH = "61f8c982041cdfdded8daf8850d31e127943386ba2cb7035aa742e94cea9a973"
PARENT_ROOT = REPO / "runs/lrs3_mfa_linear_replacement_mfa3_exploratory_20260825"
STRICT_PROTOCOL = PARENT_ROOT / "03_strict_replacement_face_ready_retry7/protocol_manifest.json"
REPLACEMENT_MANIFEST = PARENT_ROOT / "03_strict_replacement_face_ready_retry7/replacement_manifest.json"
ALIGNMENT_MANIFEST = PARENT_ROOT / "01_mfa3_screen_retry1/alignment_manifest.json"
HISTORICAL_ANALYSIS = PARENT_ROOT / "04_statistics_exploratory_retry2/analysis.json"
LEGACY_STAGE00 = PARENT_ROOT / "00_protocol_lock_retry2/manifest.json"
PARENT_HASHES = {
    "strict_protocol": "d4e5db3cfdfe9a390225c72eff1111cef329942389833696a522618d5d22bf6a",
    "replacement_manifest": "157cd678f5f03be6eae9986a82babfdfb1f1a44e611bb0b86476c6e19b7eb272",
    "alignment_manifest": "111c270ed6d5c37f2318bc0b0e1becc327370c94740c9eada2228dade5d01c98",
    "historical_analysis": "880563c361948f77a62e6a3b9f507393ac6d6c253cdb2c9ced4cc8c671c05690",
    "legacy_stage00": "7db85750e2249a5cd04637f51df7ba278f7d09d094cdc0bc176792f1f068df28",
}
KNN_VC_SOURCE = Path.home() / ".cache/torch/hub/bshall_knn-vc_c616845c4e309e24d5927f15adbdf277a3d65358"
KNN_VC_REVISION = "c616845c4e309e24d5927f15adbdf277a3d65358"
WAVLM_CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/WavLM-Large.pt"
VOCODER_CHECKPOINT = Path.home() / ".cache/torch/hub/checkpoints/prematch_g_02500000.pt"
WAV2LIP_PYTHON = Path.home() / ".venvs/wav2lip/bin/python"
WAV2LIP_ROOT = REPO / "third_party/Wav2Lip"
WAV2LIP_CHECKPOINT = WAV2LIP_ROOT / "checkpoints/wav2lip_gan.pth"
SYNCNET_PYTHON = Path.home() / ".venvs/syncnet/bin/python"
SYNCNET_ROOT = REPO / "third_party/syncnet_python"
SYNCNET_MODEL = SYNCNET_ROOT / "data/syncnet_v2.model"
FFMPEG = Path("/home/wjj/miniconda3/bin/ffmpeg")
FFPROBE = Path("/home/wjj/miniconda3/bin/ffprobe")
MIN_TRACK = 50

STAGE00 = RUN_ROOT / "00_protocol"
STAGE01 = RUN_ROOT / "01_candidates"
STAGE02 = RUN_ROOT / "02_diagonal"
STAGE03 = RUN_ROOT / "03_diagonal_analysis"
STAGE04 = RUN_ROOT / "04_replacement"
STAGE05 = RUN_ROOT / "05_final"
