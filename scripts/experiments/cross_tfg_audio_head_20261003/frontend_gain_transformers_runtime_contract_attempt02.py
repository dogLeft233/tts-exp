#!/usr/bin/env python3
"""Exact runtime contract for the frozen Muse Whisper frontend probe.

This module reads the saved, read-only runtime discovery and checks that the
same Transformers implementation is imported before frontend feature work.
It does not instantiate a feature extractor or load/run a model.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any


TRANSFORMERS_VERSION = "4.39.2"
TRANSFORMERS_OVERLAY = "/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt09/code/overlay_python310_attempt02"
TRANSFORMERS_INIT_PATH = TRANSFORMERS_OVERLAY + "/transformers/__init__.py"
TRANSFORMERS_INIT_SHA256 = "80e13341f10743e1b0b7c0b625427af62dc172c8110c599938ac2f5197e982c3"
WHISPER_MODULE_PATH = TRANSFORMERS_OVERLAY + "/transformers/models/whisper/feature_extraction_whisper.py"
WHISPER_MODULE_SHA256 = "579e4d965290295de49cd49cd1a72357b59083d6cdb5f70cd27d7d5cf3de50bf"
WHISPER_CALL_SOURCE_SHA256 = "7c22114d88d751e78d80e59f8c32ebe0f208ace244de8a5508119175b829f970"
WHISPER_CALL_FIRST_LINE = 157
EXPECTED_PYTHONPATH = (
    TRANSFORMERS_OVERLAY
    + ":/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt10/code/future_overlay_attempt02"
    + ":/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_deploy_attempt16/code"
)
DISCOVERY_JSON_SHA256 = "f6122c492b2a0dcc210eb0b804fa0d42b4dd9a780e47e7e52814fc7f60ececff"
DISCOVERY_RESPONSE_RAW_SHA256 = "99e5037abe24ccbdb885bbf1aeb176f90af187ddedda1bae3867c34186f1b82a"
NATIVE_REPORT_PATH = "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt16/output/musetalk_portrait_silent.native_report.json"
NATIVE_REPORT_SHA256 = "3004566cff8ce3736fcb393f14e804e5aec2d7ec529f0df8bb5cfb594ca948d0"
PREPROCESSOR_CONFIG_PATH = "/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt16/output/musetalk_workspace/models/whisper/preprocessor_config.json"
PREPROCESSOR_CONFIG_SHA256 = "9b5cd03a36fbb8a627c64d98a5b5b126ead95a77720723944487311f0110b666"
AUDIO_PROCESSOR_SOURCE_SHA256 = "b7b62f3daa99678576138bf581a6394a10d346571a2530a760d1299ffd622b81"


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def verify_runtime_contract(discovery_path: Path) -> dict[str, Any]:
    """Verify the imported runtime against the actual discovery evidence."""
    raw = discovery_path.read_bytes()
    discovery_sha = sha256_bytes(raw)
    _require(discovery_sha == DISCOVERY_JSON_SHA256, "runtime discovery evidence SHA mismatch")
    discovery = json.loads(raw.decode("utf-8"))
    _require(discovery.get("schema") == "frontend_gain_feature_probe_runtime_discovery_v1", "runtime discovery schema mismatch")
    _require(discovery.get("status") == "DISCOVERY_COMPLETED_WITHOUT_VERSION_GATE", "runtime discovery status mismatch")
    scope = discovery.get("scope", {})
    for field in (
        "audio_read",
        "feature_extractor_instance_created",
        "model_loaded",
        "encoder_forward",
        "onnx_runtime_imported",
        "generator_or_video",
        "syncnet",
        "asr_or_api",
        "dev_or_confirm_audio",
        "package_install",
    ):
        _require(scope.get(field) is False, "runtime discovery scope is not closed: " + field)

    _require(os.environ.get("PYTHONPATH", "") == EXPECTED_PYTHONPATH, "Muse PYTHONPATH differs from the measured runtime")
    _require(sys.version_info[:3] == (3, 10, 12), "Python runtime differs from measured Muse runtime")
    import numpy as np
    import torch
    _require(np.__version__ == "1.26.4", "NumPy runtime differs from measured Muse runtime")
    _require(torch.__version__ == "2.5.1+cu124", "Torch runtime differs from measured Muse runtime")
    _require(not bool(torch.cuda.is_initialized()), "CUDA was initialized in CPU-only frontend probe")

    dist_version = importlib.metadata.version("transformers")
    transformers = importlib.import_module("transformers")
    whisper_module = importlib.import_module("transformers.models.whisper.feature_extraction_whisper")
    feature_extractor = getattr(whisper_module, "WhisperFeatureExtractor")
    _require(dist_version == TRANSFORMERS_VERSION, "Transformers distribution version differs from measured runtime")
    _require(getattr(transformers, "__version__", None) == TRANSFORMERS_VERSION, "Transformers module version differs from measured runtime")

    init_path = str(Path(transformers.__file__).resolve())
    whisper_path = str(Path(whisper_module.__file__).resolve())
    _require(init_path == TRANSFORMERS_INIT_PATH, "Transformers import origin differs from qualified overlay")
    _require(whisper_path == WHISPER_MODULE_PATH, "Whisper feature module import origin differs from qualified overlay")
    init_sha = sha256_file(Path(init_path))
    whisper_sha = sha256_file(Path(whisper_path))
    _require(init_sha == TRANSFORMERS_INIT_SHA256, "Transformers package source SHA differs from discovery")
    _require(whisper_sha == WHISPER_MODULE_SHA256, "Whisper feature module source SHA differs from discovery")

    method = feature_extractor.__call__
    signature = inspect.signature(method)
    source_lines, first_line = inspect.getsourcelines(method)
    call_source_sha = sha256_bytes("".join(source_lines).encode("utf-8"))
    _require(first_line == WHISPER_CALL_FIRST_LINE, "Whisper __call__ source line differs from discovery")
    _require(call_source_sha == WHISPER_CALL_SOURCE_SHA256, "Whisper __call__ source SHA differs from discovery")
    _require(str(signature) == discovery["whisper_feature_extractor"]["call_signature"], "Whisper __call__ signature differs from discovery")
    expected_defaults = {
        "truncation": True,
        "padding": "max_length",
        "return_attention_mask": None,
        "sampling_rate": None,
        "do_normalize": None,
        "return_tensors": None,
    }
    for name, expected in expected_defaults.items():
        _require(signature.parameters[name].default == expected, "Whisper __call__ default differs from discovery: " + name)

    dist = importlib.metadata.distribution("transformers")
    _require(str(Path(dist.locate_file("")).resolve()) == TRANSFORMERS_OVERLAY, "Transformers distribution root differs from qualified overlay")
    actual_discovery = discovery.get("transformers_distribution", {})
    _require(actual_discovery.get("version") == dist_version, "discovery Transformers distribution version mismatch")
    _require(actual_discovery.get("distribution_root") == TRANSFORMERS_OVERLAY, "discovery Transformers distribution root mismatch")
    _require(discovery.get("transformers_module", {}).get("file_sha256", {}).get("sha256") == init_sha, "discovery Transformers module SHA mismatch")
    _require(discovery.get("transformers_module", {}).get("under_configured_pythonpath") is True, "discovery module was outside configured Muse paths")
    whisper_evidence = discovery.get("whisper_feature_extractor", {})
    _require(whisper_evidence.get("module_file_sha256", {}).get("sha256") == whisper_sha, "discovery Whisper module SHA mismatch")
    _require(whisper_evidence.get("call_source_sha256") == call_source_sha, "discovery Whisper call source SHA mismatch")
    _require(discovery.get("qualified_musetalk_native_report", {}).get("expected_sha256_matches") is True, "qualified native report did not match its frozen SHA")
    native_report = discovery.get("qualified_musetalk_native_report", {}).get("actual", {})
    _require(native_report.get("path") == NATIVE_REPORT_PATH and native_report.get("sha256") == NATIVE_REPORT_SHA256, "qualified native report binding mismatch")
    native_fields = discovery.get("qualified_musetalk_native_report", {}).get("selected_runtime_and_source_fields", {})
    _require(native_fields.get("status") == "PASS_ENGINEERING_ONLY", "frozen Muse qualification report is not a pass")
    _require(native_fields.get("schema") == "cross_tfg_musetalk_fixed_portrait_engineering_inference_v3", "frozen Muse qualification report schema mismatch")
    _require(native_fields.get("runtime", {}).get("torch_version") == "2.5.1+cu124", "native Muse report Torch binding mismatch")
    config = discovery.get("qualified_musetalk_whisper_preprocessor_config", {})
    _require(config.get("expected_sha256_matches") is True, "native report preprocessor config binding failed")
    config_actual = config.get("actual", {})
    _require(config_actual.get("path") == PREPROCESSOR_CONFIG_PATH and config_actual.get("sha256") == PREPROCESSOR_CONFIG_SHA256, "Whisper preprocessor config binding mismatch")
    _require(sha256_file(Path(NATIVE_REPORT_PATH)) == NATIVE_REPORT_SHA256, "qualified native report changed after discovery")
    _require(sha256_file(Path(PREPROCESSOR_CONFIG_PATH)) == PREPROCESSOR_CONFIG_SHA256, "Whisper preprocessor config changed after discovery")

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "torch": torch.__version__,
        "torch_cuda_initialized": False,
        "transformers_distribution_version": dist_version,
        "transformers_module_version": transformers.__version__,
        "transformers_distribution_root": str(Path(dist.locate_file("")).resolve()),
        "transformers_init_path": init_path,
        "transformers_init_sha256": init_sha,
        "whisper_feature_module_path": whisper_path,
        "whisper_feature_module_sha256": whisper_sha,
        "whisper_call_signature": str(signature),
        "whisper_call_first_line": first_line,
        "whisper_call_source_sha256": call_source_sha,
        "whisper_call_defaults": {name: repr(signature.parameters[name].default) for name in expected_defaults},
        "runtime_discovery_json_sha256": discovery_sha,
        "runtime_discovery_response_raw_sha256": DISCOVERY_RESPONSE_RAW_SHA256,
        "native_report_path": NATIVE_REPORT_PATH,
        "native_report_sha256": NATIVE_REPORT_SHA256,
        "preprocessor_config_path": PREPROCESSOR_CONFIG_PATH,
        "preprocessor_config_sha256": PREPROCESSOR_CONFIG_SHA256,
        "muse_audio_processor_source_sha256": AUDIO_PROCESSOR_SOURCE_SHA256,
        "feature_extractor_instance_created": False,
        "model_or_encoder_forward": False,
        "onnx_runtime_loaded": False,
    }
