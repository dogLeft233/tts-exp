#!/usr/bin/env python3
"""No-forward environment/source preflight for the fit8 frontend-input probe."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np

PLAN_SHA = "1f0a290b612850b29b21233e4ffbfe5d0b48538341fdeeb853ce6cec945a99b7"
MANIFEST_SHA = "be772cbfa4b394991d889f8d5da2cbac3f07d5f8a66979d7278486bc068c8528"
RUNNER_SHA = "5c632a1885a2a3db7af9224acee602d0079ac51566dc9142998809b44150479c"
HELPER_SHA = "3480087daecf8dc295f4a3b20703f393bada2690e1f5ce7839ba1dc4fbdec8ef"
CONSTANTS_SHA = "873d525f1b1b4669594b9fbfb50fa3421f5c537da70cf967e5b3ee88ac74d449"
ROOT_QC_SHA = "e8ad71c17000254427205317c5e026a082b62327f182077a3289070b00a97edf"
ROOT_RECEIPT_SHA = "21555b138c29567dfaca357b361ebdb0fa4a75978edca6c84f8efaf13b4ef781"
W2L_AUDIO_SHA = "71d5e927cac3d0b7103e7671a43b82b41b6a56398a092bbfa2e96d3041249c4c"
W2L_HPARAMS_SHA = "4a4c31dc51ebd8140da18ac24841e582b2472389f7c65e7f0a6b4447fc0c71cf"
MUSE_PREPROCESSOR_SHA = "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666"
MUSE_AUDIO_PROCESSOR_SHA = "b7b62f3daa99678576138bf581a6394a10d346571a2530a760d1299ffd622b81"
RUNTIME_HELPER_SHA = "9e0bf54fe830e07fdeac7f63871b3e589f9b34eeebd411dbc72852476649920a"
RUNTIME_DISCOVERY_SHA = "f6122c492b2a0dcc210eb0b804fa0d42b4dd9a780e47e7e52814fc7f60ececff"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def require_file(path: Path, expected: str, label: str) -> None:
    if path.is_symlink() or not path.is_file() or sha256(path) != expected:
        raise ValueError(f"{label} path or SHA mismatch")


def load_runtime_contract_helper(path: Path):
    require_file(path, RUNTIME_HELPER_SHA, "measured Transformers runtime contract helper")
    spec = importlib.util.spec_from_file_location("frontend_gain_transformers_runtime_contract_attempt02", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import measured Transformers runtime contract helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--runner", type=Path, required=True)
    ap.add_argument("--helper", type=Path, required=True)
    ap.add_argument("--constants", type=Path, required=True)
    ap.add_argument("--root-qc", type=Path, required=True)
    ap.add_argument("--wav2lip-root", type=Path, required=True)
    ap.add_argument("--muse-workspace", type=Path, required=True)
    ap.add_argument("--muse-source-root", type=Path, required=True)
    ap.add_argument("--runtime-helper", type=Path, required=True)
    ap.add_argument("--runtime-discovery", type=Path, required=True)
    args = ap.parse_args()

    if os.environ.get("CUDA_VISIBLE_DEVICES", "") != "":
        raise ValueError("CUDA_VISIBLE_DEVICES must be empty")
    if sys.version_info[:3] != (3, 10, 12) or np.__version__ != "1.26.4":
        raise ValueError("Python/NumPy version mismatch")
    require_file(args.runtime_discovery, RUNTIME_DISCOVERY_SHA, "actual Muse Transformers runtime discovery")
    runtime_contract_helper = load_runtime_contract_helper(args.runtime_helper)
    runtime_contract = runtime_contract_helper.verify_runtime_contract(args.runtime_discovery)
    require_file(args.plan, PLAN_SHA, "frozen fit plan")
    require_file(args.manifest, MANIFEST_SHA, "fit audio transfer manifest")
    require_file(args.runner, RUNNER_SHA, "gain probe runner")
    require_file(args.helper, HELPER_SHA, "Wav2Lip frontend helper")
    require_file(args.constants, CONSTANTS_SHA, "Ditto constants audit")
    require_file(args.root_qc, ROOT_QC_SHA, "Ditto root numeric receipt")
    root_qc = json.loads(args.root_qc.read_text(encoding="utf-8"))
    if root_qc.get("receipt_sha256") != ROOT_RECEIPT_SHA:
        raise ValueError("root Ditto numeric receipt binding mismatch")

    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    payload = dict(manifest)
    embedded_payload = payload.pop("manifest_sha256", None)
    canonical = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    if canonical != embedded_payload or embedded_payload != "fdc1ab5bf25a645be584e29acae94639159bb9333ac748957c60b506f724973b":
        raise ValueError("transfer manifest canonical payload checksum mismatch")
    if any(manifest.get(k) is not False for k in ("development_audio_opened", "confirmation_audio_opened", "effects_or_scores_read")):
        raise ValueError("transfer manifest scope flags are not closed")

    models = {m.get("model_id"): m for m in plan.get("models", [])}
    fit = [r for r in plan.get("rows", []) if r.get("split") == "fit" and r.get("model_id") == "wav2lip_gan_bac9a81e" and r.get("method_id") == "identity"]
    by_id = {r.get("sample_id"): r for r in fit}
    if len(fit) != 8 or len(by_id) != 8 or len(manifest.get("rows", [])) != 8:
        raise ValueError("frozen fit8 roster mismatch")
    if str(args.wav2lip_root) != models["wav2lip_gan_bac9a81e"]["generator"]["repo"]:
        raise ValueError("Wav2Lip path differs from frozen plan")
    muse = models["musetalk_fixed_portrait_attempt16"]
    if str(args.muse_workspace) != muse["generator"]["model_workspace"]:
        raise ValueError("Muse workspace differs from frozen plan")
    if str(args.muse_source_root) != muse["generator"]["model_source_root"]:
        raise ValueError("Muse source root differs from frozen plan")

    muse_audio = args.muse_source_root / "musetalk" / "utils" / "audio_processor.py"
    muse_preproc = args.muse_workspace / "models" / "whisper" / "preprocessor_config.json"
    w2l_audio = args.wav2lip_root / "audio.py"
    w2l_hparams = args.wav2lip_root / "hparams.py"
    require_file(muse_audio, MUSE_AUDIO_PROCESSOR_SHA, "Muse AudioProcessor source")
    require_file(muse_preproc, MUSE_PREPROCESSOR_SHA, "Muse Whisper preprocessor")
    require_file(w2l_audio, W2L_AUDIO_SHA, "Wav2Lip audio frontend")
    require_file(w2l_hparams, W2L_HPARAMS_SHA, "Wav2Lip hparams")
    asset = next((a for a in muse.get("model_assets", {}).get("whisper", []) if a.get("relative_path") == "preprocessor_config.json"), None)
    if asset is None or asset.get("sha256") != MUSE_PREPROCESSOR_SHA:
        raise ValueError("Muse preprocessor asset does not match frozen model plan")

    samples = set()
    audio_bytes_checked = 0
    for row in manifest["rows"]:
        sid = row["source_manifest_row_id"]
        if sid in samples:
            raise ValueError("duplicate fit audio row")
        samples.add(sid)
        natural = by_id.get(sid, {}).get("natural_pcm", {})
        if (
            natural.get("path") != row.get("remote_path")
            or natural.get("file_sha256") != row.get("file_sha256")
            or natural.get("pcm_sha256") != row.get("pcm_float32_sha256")
            or natural.get("sample_count") != row.get("sample_count")
            or natural.get("sample_rate_hz") != 16000
            or natural.get("channels") != 1
            or natural.get("subtype") != "PCM_16"
            or row.get("sample_count", 0) >= 30 * 16000
        ):
            raise ValueError("plan/manifest source PCM binding mismatch: " + sid)
        audio = Path(row["remote_path"])
        require_file(audio, row["file_sha256"], "source fit WAV " + sid)
        audio_bytes_checked += audio.stat().st_size
    if samples != set(by_id):
        raise ValueError("manifest sample roster differs from frozen fit8")

    import librosa
    if librosa.__version__ != "0.10.2.post1":
        raise ValueError("librosa version mismatch")
    import torch
    if torch.__version__ != "2.5.1+cu124" or torch.cuda.is_initialized():
        raise ValueError("Torch version mismatch or CUDA initialized")
    from transformers import AutoFeatureExtractor
    extractor = AutoFeatureExtractor.from_pretrained(str(muse_preproc.parent), local_files_only=True)
    if type(extractor).__name__ != "WhisperFeatureExtractor":
        raise ValueError("Muse preprocessor class mismatch")

    spec = importlib.util.spec_from_file_location("cross_tfg_wav2lip_frontend_probe", args.helper)
    if spec is None or spec.loader is None:
        raise ValueError("cannot import Wav2Lip helper")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    audio_module, mel_spec, meta = module.import_official_wav2lip_audio(args.wav2lip_root)
    if meta.get("audio_sha256") != W2L_AUDIO_SHA or meta.get("hparams_sha256") != W2L_HPARAMS_SHA:
        raise ValueError("official Wav2Lip frontend mismatch")
    if "onnxruntime" in sys.modules or torch.cuda.is_initialized():
        raise ValueError("unexpected ONNX Runtime or CUDA initialization")
    print(json.dumps({
        "status": "PREFLIGHT_PASS",
        "fit_audio_rows": len(samples),
        "fit_audio_bytes_verified": audio_bytes_checked,
        "plan_raw_sha256": PLAN_SHA,
        "transfer_manifest_raw_sha256": MANIFEST_SHA,
        "runner_source_sha256": sha256(args.runner),
        "ditto_constants_audit_raw_sha256": CONSTANTS_SHA,
        "ditto_root_qc_raw_sha256": ROOT_QC_SHA,
        "runtime": {"python": sys.version.split()[0], "numpy": np.__version__, "librosa": librosa.__version__, "torch": torch.__version__, "transformers": runtime_contract["transformers_distribution_version"]},
        "runtime_discovery_raw_sha256": RUNTIME_DISCOVERY_SHA,
        "runtime_contract_helper_sha256": RUNTIME_HELPER_SHA,
        "runtime_contract": runtime_contract,
        "muse_preprocessor": type(extractor).__name__,
        "wav2lip_mel_spec": vars(mel_spec),
        "model_or_encoder_forward": False,
        "onnx_runtime_loaded": False,
        "cuda_initialized": False,
        "development_or_confirmation_audio": False,
    }, sort_keys=True, separators=(",", ":")))


if __name__ == "__main__":
    main()
