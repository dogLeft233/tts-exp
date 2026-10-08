#!/usr/bin/env python3
"""Five-argument Stage-A adapter for the qualified MuseTalk attempt16 profile.

The shim delegates to immutable attempt16 renderer/adapter sources. It writes a
sealed per-render receipt but leaves muxing and SyncNet scoring to the shared
Stage-A core, which always restores the original natural PCM.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PYTHON = "/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python"
REPO_ROOT = "/root/tts-research-system-20261001"
PROJECT = REPO_ROOT + "/cross_tfg_audio_head_20261003"
BASE_OVERLAY = PROJECT + "/model_deploy_attempt09/code/overlay_python310_attempt02"
FUTURE_OVERLAY = PROJECT + "/model_deploy_attempt10/code/future_overlay_attempt02"
MODEL_CODE = PROJECT + "/model_deploy_attempt16/code"
DETERMINISM_BOOTSTRAP = REPO_ROOT + "/scripts/experiments/cross_tfg_audio_head_20261003/runtime_determinism_attempt03"
DETERMINISM_BOOTSTRAP_SHA256 = "3099c4b2e1c650f7aa46c31145cc2c88ccda9f085288ab039c6ffed85007ab65"
SOURCE_ROOT = PROJECT + "/model_sources_attempt02/MuseTalk"
MODEL_ROOT = "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt16/output/musetalk_workspace"
QUALIFICATION_REPORT = "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt16/output/musetalk_portrait_silent.native_report.json"
QUALIFICATION_REPORT_SHA256 = "3004566cff8ce3736fcb393f14e804e5aec2d7ec529f0df8bb5cfb594ca948d0"
RENDERER = MODEL_CODE + "/render_silent_video_attempt09.py"
RENDERER_SHA256 = "b75004340fbf90b937fb516c5d8ee013f193f763f5cf61bbfc7d28cb12ce88c8"
ADAPTER = MODEL_CODE + "/musetalk_fixed_portrait_infer_attempt08.py"
ADAPTER_SHA256 = "200209c7fbaa4b2c83d571a04696bec7658e759a564494c064b220cdfdf2daaf"
OVERDUB_HELPER = MODEL_CODE + "/scripts/audio_enhancement/overdub.py"
OVERDUB_SHA256 = "0febe6f2b43ed4fb089bdf36e2e208cc1cd61681d8e0ebfd7f59cd56dd00b4b6"
FACE_SHA256 = "e3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6"
MUSE_COMMIT = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"
MUSE_CODELOAD_SHA256 = "2e1685e4bcc58531c5118c0ba485199f22012016ed4608cab725a51d131f7ca7"
MUSE_MODEL_REVISION = "2bcb936e2fddb4d86db4c62fd45b387d0c061571"
EXPECTED_DETECTED_BBOX = [164, 113, 409, 358]
EXPECTED_EXPANDED_BBOX = [121, 80, 452, 411]
EXPECTED_FACE_PARSER = "MuseTalk official FaceParsing BiSeNet + pinned parser weights"

SOURCE_PINS = {
    "scripts/inference.py": "aaf9c8289c8f07eaf00b87fb11442ac0e1005ac4e2f469966ae3dab2dc4a03aa",
    "musetalk/utils/face_parsing/__init__.py": "c2824281e80458b81b7ecdde6cfd2b5f0224b022f41aaf1c61144a52163b2494",
    "musetalk/utils/face_parsing/model.py": "05d69d1bd77fbca7b537602fe3f7851e8b2e1a3ea0279f383b4223598157e2d2",
    "musetalk/utils/audio_processor.py": "b7b62f3daa99678576138bf581a6394a10d346571a2530a760d1299ffd622b81",
    "musetalk/utils/blending.py": "f1be04977db7e6fa1551dfe66dfbdea1e2f32c8d237edaf5a75a0ca4ddd80b53",
    "musetalk/models/vae.py": "6ecde3f4417178d8a9de5ed6045e2a4832efa65189b10b0672cab68efe529d32",
    "musetalk/models/unet.py": "05d5dadfde2e726b61d8387a4876336d4dcacc696636e8dcfec69863ce819256",
    "musetalk/utils/utils.py": "3715eacedefc97d0ef97b98c2f3bf06929bf9e312393752b75b708e43a99a055",
}
EXPECTED_MODEL_ASSETS = {
    "unet": {
        "musetalk.json": (748, "5b6923aee04d71692e0e9846c471e0a4ea07a4f686d39545e472bd4ba17e1b47"),
        "unet.pth": (3400074924, "7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007"),
    },
    "vae": {
        "config.json": (547, "92d3dfb746fca211a2c9e019e285f8597412211728dce3c5bcf4eda0f2d62e7e"),
        "diffusion_pytorch_model.bin": (334707217, "1b4889b6b1d4ce7ae320a02dedaeff1780ad77d415ea0d744b476155c6377ddc"),
    },
    "whisper": {
        "config.json": (1983, "ffdccec4f3211f4c63310f2b7098f309fe70f3952cedc5e4d11e43f5b2379b98"),
        "preprocessor_config.json": (184990, "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666"),
        "pytorch_model.bin": (151095027, "9607f98a2b22d9e229ae43c52ecea79dcede9e0c5cfae67e8da6eda86d8aac1d"),
    },
    "face_parser": {
        "79999_iter.pth": (53289463, "468e13ca13a9b43cc0881a9f99083a430e9c0a38abd935431d1c28ee94b26567"),
        "resnet18-5c106cde.pth": (46827520, "5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8"),
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def _verify_source_tree() -> dict[str, str]:
    root = Path(SOURCE_ROOT)
    result: dict[str, str] = {}
    for relative, expected in SOURCE_PINS.items():
        path = root / relative
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"MuseTalk source pin mismatch: {relative}: {actual}")
        result[relative] = actual
    return result


def _read_canonical_audio(path: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    info = sf.info(str(path))
    if (info.samplerate, info.channels, info.subtype) != (16000, 1, "FLOAT"):
        raise RuntimeError(f"expected mono 16 kHz IEEE-float WAV, got {info}")
    data, sample_rate = sf.read(str(path), dtype="float32", always_2d=True)
    samples = np.ascontiguousarray(data[:, 0], dtype="<f4")
    if sample_rate != 16000 or samples.size != info.frames or samples.size == 0 or not np.isfinite(samples).all():
        raise RuntimeError("canonical audio decode has invalid clock, shape, or samples")
    return {
        "path": str(path), "file_sha256": sha256_file(path),
        "pcm_float32_sha256": hashlib.sha256(samples.tobytes(order="C")).hexdigest(),
        "sample_rate_hz": 16000, "channels": 1, "sample_count": int(samples.size),
        "subtype": "IEEE_FLOAT_32_LE",
    }


def _validate_model_report(native: dict[str, Any], audio_ref: dict[str, Any], face_path: Path,
                           source_hashes: dict[str, str]) -> None:
    if native.get("status") != "PASS_ENGINEERING_ONLY":
        raise RuntimeError("pinned Muse adapter did not report engineering success")
    if native.get("official_source", {}).get("commit") != MUSE_COMMIT:
        raise RuntimeError("Muse source commit mismatch in adapter report")
    if native.get("official_source", {}).get("codeload_sha256") != MUSE_CODELOAD_SHA256:
        raise RuntimeError("Muse source archive SHA mismatch in adapter report")
    if native.get("official_model_snapshot", {}).get("revision") != MUSE_MODEL_REVISION:
        raise RuntimeError("Muse model revision mismatch in adapter report")
    selected = native.get("official_source", {}).get("selected_source_files", {})
    for relative, expected in source_hashes.items():
        item = selected.get(relative)
        if not isinstance(item, dict) or item.get("sha256") != expected:
            raise RuntimeError(f"render report does not bind pinned Muse source {relative}")
    adapter = native.get("adapter", {})
    if (adapter.get("fixed_portrait_only") is not True
            or adapter.get("face_box_method") != "OpenCV Haar frontalface_default, expanded square"
            or adapter.get("face_parser") != EXPECTED_FACE_PARSER
            or adapter.get("mmpose_preprocessing_used") is not False
            or adapter.get("detected_face_bbox_xyxy") != EXPECTED_DETECTED_BBOX
            or adapter.get("model_crop_bbox_xyxy") != EXPECTED_EXPANDED_BBOX
            or adapter.get("blend_face_bbox_xyxy") != EXPECTED_EXPANDED_BBOX):
        raise RuntimeError("Muse report differs from the qualified fixed-portrait/Haar/official-parser geometry")
    resize = adapter.get("official_native_face_resize", {})
    if (resize.get("operation") != "cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))"
            or resize.get("face_bbox_xyxy") != EXPECTED_EXPANDED_BBOX
            or any(shape != [256, 256, 3] for shape in resize.get("input_shapes", []))
            or any(shape != [331, 331, 3] for shape in resize.get("resized_shapes", []))
            or len(resize.get("input_shapes", [])) != native.get("output", {}).get("frame_count")):
        raise RuntimeError("Muse native resize operation/geometry differs from the portrait qualification")
    runtime = native.get("runtime", {})
    if (runtime.get("torch_version") != "2.5.1+cu124" or runtime.get("cuda_version") != "12.4"
            or runtime.get("float16") is not True or runtime.get("tf32") is not False
            or runtime.get("fps") != 25):
        raise RuntimeError("Muse renderer runtime differs from the pinned engineering qualification")
    input_ref = native.get("input", {})
    if input_ref.get("audio_file_sha256") != audio_ref["file_sha256"]:
        raise RuntimeError("Muse native report audio file SHA differs from generator input")
    if input_ref.get("audio_sample_count_frontend") != audio_ref["sample_count"]:
        raise RuntimeError("Muse native report audio sample count differs from generator input")
    if input_ref.get("portrait_file_sha256") != FACE_SHA256 or sha256_file(face_path) != FACE_SHA256:
        raise RuntimeError("Muse native report did not use the frozen primary face image")
    output = native.get("output", {})
    if (output.get("audio_sample_count") != audio_ref["sample_count"]
            or output.get("audio_sample_rate_hz") != 16000
            or output.get("audio_channels") != 1
            or output.get("audio_float32_sha256") != audio_ref["pcm_float32_sha256"]
            or output.get("audio_matches_original_decoded_pcm") is not True):
        raise RuntimeError("Muse native PCM mux does not preserve its exact generator input")
    for group, expected_files in EXPECTED_MODEL_ASSETS.items():
        actual_items = native.get("models", {}).get(group)
        if not isinstance(actual_items, list):
            raise RuntimeError(f"Muse report lacks model asset group {group}")
        actual = {str(item.get("relative_path")): (item.get("bytes"), item.get("sha256"))
                  for item in actual_items if isinstance(item, dict)}
        if actual != expected_files:
            raise RuntimeError(f"Muse {group} asset pins differ from the frozen model manifest")


def _verify_qualification_receipt(source_hashes: dict[str, str]) -> None:
    path = Path(QUALIFICATION_REPORT).resolve(strict=True)
    actual_sha = sha256_file(path)
    if actual_sha != QUALIFICATION_REPORT_SHA256:
        raise RuntimeError(f"Muse portrait qualification receipt SHA mismatch: {actual_sha}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS_ENGINEERING_ONLY":
        raise RuntimeError("Muse fixed-portrait qualification report is not PASS_ENGINEERING_ONLY")
    if report.get("official_source", {}).get("commit") != MUSE_COMMIT or report.get("official_source", {}).get("codeload_sha256") != MUSE_CODELOAD_SHA256:
        raise RuntimeError("Muse qualification report source identity mismatch")
    if report.get("official_model_snapshot", {}).get("revision") != MUSE_MODEL_REVISION:
        raise RuntimeError("Muse qualification report model revision mismatch")
    selected = report.get("official_source", {}).get("selected_source_files", {})
    for relative, expected in source_hashes.items():
        if selected.get(relative, {}).get("sha256") != expected:
            raise RuntimeError(f"Muse qualification report source mismatch: {relative}")
    adapter = report.get("adapter", {})
    if (adapter.get("fixed_portrait_only") is not True
            or adapter.get("face_box_method") != "OpenCV Haar frontalface_default, expanded square"
            or adapter.get("face_parser") != EXPECTED_FACE_PARSER
            or adapter.get("mmpose_preprocessing_used") is not False
            or adapter.get("detected_face_bbox_xyxy") != EXPECTED_DETECTED_BBOX
            or adapter.get("model_crop_bbox_xyxy") != EXPECTED_EXPANDED_BBOX
            or adapter.get("blend_face_bbox_xyxy") != EXPECTED_EXPANDED_BBOX):
        raise RuntimeError("Muse qualification does not match the fixed-portrait/Haar/official-parser geometry")
    resize = adapter.get("official_native_face_resize", {})
    if (resize.get("operation") != "cv2.resize(res_frame.astype(np.uint8), (x2-x1, y2-y1))"
            or resize.get("face_bbox_xyxy") != EXPECTED_EXPANDED_BBOX
            or any(shape != [256, 256, 3] for shape in resize.get("input_shapes", []))
            or any(shape != [331, 331, 3] for shape in resize.get("resized_shapes", []))
            or len(resize.get("input_shapes", [])) != report.get("output", {}).get("frame_count")):
        raise RuntimeError("Muse qualification resize/geometry does not match the pinned crop contract")
    runtime = report.get("runtime", {})
    if (runtime.get("torch_version") != "2.5.1+cu124" or runtime.get("cuda_version") != "12.4"
            or runtime.get("float16") is not True or runtime.get("tf32") is not False
            or runtime.get("fps") != 25):
        raise RuntimeError("Muse qualification runtime differs from the frozen profile")
    for group, expected_files in EXPECTED_MODEL_ASSETS.items():
        actual_items = report.get("models", {}).get(group)
        if not isinstance(actual_items, list):
            raise RuntimeError(f"Muse qualification report lacks model asset group {group}")
        actual = {str(item.get("relative_path")): (item.get("bytes"), item.get("sha256"))
                  for item in actual_items if isinstance(item, dict)}
        if actual != expected_files:
            raise RuntimeError(f"Muse qualification {group} asset pins differ")


def _write_json_exclusive(path: Path, value: dict[str, Any]) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--face", required=True, type=Path)
    parser.add_argument("--outfile", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--fps", required=True, type=int, choices=(25,))
    args = parser.parse_args()

    audio_path = args.audio.resolve(strict=True)
    face_path = args.face.resolve(strict=True)
    output = args.outfile.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite silent video: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt_path = output.with_suffix(".adapter_receipt.json")
    renderer_report_path = output.with_suffix(".renderer_report.json")
    native_report_path = output.with_suffix(".native_report.json")
    native_mux_path = output.with_name(output.stem + "_native_mux.mkv")
    for path in (receipt_path, renderer_report_path, native_report_path, native_mux_path):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite MuseTalk attempt output: {path}")
    if not Path(PYTHON).is_file():
        raise FileNotFoundError(PYTHON)
    if sha256_file(face_path) != FACE_SHA256:
        raise RuntimeError("planned MuseTalk face file differs from the pinned primary portrait")

    renderer = Path(RENDERER)
    adapter = Path(ADAPTER)
    helper = Path(OVERDUB_HELPER)
    source_root = Path(SOURCE_ROOT)
    model_root = Path(MODEL_ROOT)
    for path, expected in ((renderer, RENDERER_SHA256), (adapter, ADAPTER_SHA256), (helper, OVERDUB_SHA256)):
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"pinned Muse inference code SHA mismatch: {path} ({actual})")
    if not source_root.is_dir() or not model_root.is_dir():
        raise FileNotFoundError("pinned Muse source or model workspace is missing")
    source_hashes = _verify_source_tree()
    _verify_qualification_receipt(source_hashes)

    bootstrap_file = Path(DETERMINISM_BOOTSTRAP) / "sitecustomize.py"
    if (not bootstrap_file.is_file() or sha256_file(bootstrap_file) != DETERMINISM_BOOTSTRAP_SHA256):
        raise RuntimeError("strict renderer bootstrap is absent or differs from the pinned source")
    expected_pythonpath = os.pathsep.join((DETERMINISM_BOOTSTRAP, BASE_OVERLAY, FUTURE_OVERLAY, MODEL_CODE))
    actual_pythonpath = os.environ.get("PYTHONPATH", "")
    actual_entries = actual_pythonpath.split(os.pathsep) if actual_pythonpath else []
    for root in (BASE_OVERLAY, FUTURE_OVERLAY, MODEL_CODE):
        if root not in actual_entries or not Path(root).is_dir():
            raise RuntimeError(f"Muse dependency/source root missing from PYTHONPATH: {root}")
    audio_ref = _read_canonical_audio(audio_path)

    temporary_root = output.parent / ".musetalk_tmp"
    temporary_root.mkdir(exist_ok=False)
    env = os.environ.copy()
    expected_strict_context = {
        "CROSS_TFG_STRICT_STARTUP_POLICY": "1",
        "CROSS_TFG_ROW_ID": output.parent.parent.name,
        "CROSS_TFG_MODEL_ID": "musetalk_fixed_portrait_attempt16",
        "CROSS_TFG_AUDIO_FRONTEND": "musetalk_whisper_fixed_portrait_haar_expanded_box_fp16",
        "CROSS_TFG_BACKEND_ENTRY": RENDERER,
        "PYTHONHASHSEED": "20261003",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, expected in expected_strict_context.items():
        actual = os.environ.get(key)
        if actual != expected:
            raise RuntimeError(f"strict native renderer context mismatch for {key}: {actual!r}")
        env[key] = actual
    for key in ("CROSS_TFG_DETERMINISM_RECEIPT_DIR", "CROSS_TFG_CONTROL_DIR"):
        value = os.environ.get(key)
        if not value or not Path(value).is_absolute():
            raise RuntimeError(f"strict native renderer context lacks an absolute {key}")
        env[key] = value
    env.update({
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "20261003",
        "TFG_DETERMINISM_SEED": str(args.seed),
        "NUMEXPR_NUM_THREADS": "1",
        "TMPDIR": str(temporary_root),
        "PYTHONPATH": expected_pythonpath,
    })
    command = [
        PYTHON, "-B", str(renderer), "--model", "musetalk",
        "--audio", str(audio_path), "--face", str(face_path), "--outfile", str(output),
        "--seed", str(args.seed), "--fps", "25", "--source-root", str(source_root),
        "--model-root", str(model_root), "--musetalk-adapter", str(adapter),
        "--overdub-helper", str(helper), "--ffmpeg", "/usr/bin/ffmpeg",
        "--ffprobe", "/usr/bin/ffprobe", "--report", str(renderer_report_path),
    ]
    stdout_path = output.with_suffix(".muse_renderer.stdout.log")
    stderr_path = output.with_suffix(".muse_renderer.stderr.log")
    with stdout_path.open("xb") as stdout, stderr_path.open("xb") as stderr:
        completed = subprocess.run(command, cwd=str(model_root), env=env, stdin=subprocess.DEVNULL,
                                   stdout=stdout, stderr=stderr, timeout=1800, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"pinned MuseTalk renderer exit {completed.returncode}; stderr={stderr_path}")
    for path in (output, renderer_report_path, native_report_path, native_mux_path):
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"MuseTalk renderer omitted required output: {path}")
    renderer_report = json.loads(renderer_report_path.read_text(encoding="utf-8"))
    native_report = json.loads(native_report_path.read_text(encoding="utf-8"))
    _validate_model_report(native_report, audio_ref, face_path, source_hashes)
    if renderer_report.get("status") != "PASS_ENGINEERING_ONLY" or renderer_report.get("model") != "musetalk":
        raise RuntimeError("combined renderer report does not show the pinned MuseTalk profile")
    if renderer_report.get("adapter_source_sha256") != ADAPTER_SHA256 or renderer_report.get("repository_commit") != MUSE_COMMIT:
        raise RuntimeError("combined renderer report source binding mismatch")
    silent_meta = renderer_report.get("silent_video", {})
    if (silent_meta.get("path") != str(output) or silent_meta.get("file_sha256") != sha256_file(output)
            or silent_meta.get("probe", {}).get("fps") != "25/1"):
        raise RuntimeError("combined renderer report does not match the actual 25 fps silent video")
    native_meta = renderer_report.get("native_report", {})
    if (native_meta.get("path") != str(native_report_path)
            or native_meta.get("file_sha256") != sha256_file(native_report_path)):
        raise RuntimeError("combined renderer report does not bind the native Muse report")
    native_mux_meta = renderer_report.get("native_audio_mux_preserved", {})
    if (native_mux_meta.get("path") != str(native_mux_path)
            or native_mux_meta.get("file_sha256") != sha256_file(native_mux_path)):
        raise RuntimeError("combined renderer report does not bind its native mux")

    temporary_removed = False
    if temporary_root.exists():
        resolved_tmp = temporary_root.resolve(strict=True)
        if temporary_root.is_symlink() or resolved_tmp.parent != output.parent.resolve(strict=True):
            raise RuntimeError("refusing to clean a Muse temporary directory outside this result row")
        shutil.rmtree(resolved_tmp)
        temporary_removed = True

    receipt = {
        "schema": "cross_tfg_musetalk_stagea_adapter_receipt_v1",
        "status": "PASS",
        "interpretation": "fresh MuseTalk fixed-portrait dev render; the Stage-A scorer separately remuxes the original natural PCM",
        "row_id": os.environ.get("CROSS_TFG_ROW_ID"),
        "model_id": os.environ.get("CROSS_TFG_MODEL_ID"),
        "audio_frontend": os.environ.get("CROSS_TFG_AUDIO_FRONTEND"),
        "seed": int(args.seed),
        "model_profile": "musetalk_native_whisper_unet_fp16_fixed_portrait_haar_expanded_box_official_face_parser_mmpose_free",
        "fps_num": 25,
        "fps_den": 1,
        "input_audio": audio_ref,
        "face": {"path": str(face_path), "file_sha256": sha256_file(face_path)},
        "adapter_sha256": sha256_file(Path(__file__).resolve()),
        "native_adapter_sha256": ADAPTER_SHA256,
        "renderer_sha256": RENDERER_SHA256,
        "overdub_helper_sha256": OVERDUB_SHA256,
        "source_commit": MUSE_COMMIT,
        "source_codeload_sha256": MUSE_CODELOAD_SHA256,
        "model_revision": MUSE_MODEL_REVISION,
        "qualification_report_path": QUALIFICATION_REPORT,
        "qualification_report_sha256": QUALIFICATION_REPORT_SHA256,
        "source_files": source_hashes,
        "native_report_path": str(native_report_path),
        "native_report_sha256": sha256_file(native_report_path),
        "renderer_report_path": str(renderer_report_path),
        "renderer_report_sha256": sha256_file(renderer_report_path),
        "silent_video": {
            "path": str(output), "file_sha256": sha256_file(output), "bytes": output.stat().st_size,
            "frame_count": int(silent_meta["probe"]["frame_count"]),
            "fps": "25/1", "video_stream_sha256": native_report["output"]["video_stream_sha256"],
        },
        "native_pcm_mux": {
            "path": str(native_mux_path), "file_sha256": sha256_file(native_mux_path),
            "decoded_audio_pcm_float32_sha256": native_report["output"]["audio_float32_sha256"],
            "audio_sample_count": native_report["output"]["audio_sample_count"],
        },
        "renderer_stdout": {"path": str(stdout_path), "file_sha256": sha256_file(stdout_path), "bytes": stdout_path.stat().st_size},
        "renderer_stderr": {"path": str(stderr_path), "file_sha256": sha256_file(stderr_path), "bytes": stderr_path.stat().st_size},
        "temporary_workspace": {"path": str(temporary_root), "removed_after_success": temporary_removed},
    }
    receipt_sha256 = _write_json_exclusive(receipt_path, receipt)
    print(json.dumps({"status": "PASS", "row_id": receipt["row_id"], "seed": args.seed,
                      "silent_video_sha256": receipt["silent_video"]["file_sha256"],
                      "adapter_receipt": str(receipt_path), "adapter_receipt_sha256": receipt_sha256},
                     ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
