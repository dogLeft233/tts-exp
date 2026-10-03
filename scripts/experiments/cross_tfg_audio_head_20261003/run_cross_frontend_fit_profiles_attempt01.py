#!/usr/bin/env python3
"""Fit-only multi-frontend runner for sealed shared frequency-profile plans.

This wrapper keeps the canonical Stage-A scorer/mux contract. It only extends
the Stage-A transform dispatch for the frozen shared-frequency profile family;
the pinned cross-frontend adapter remains responsible for MuseTalk/Ditto video
generation. The Wav2Lip rows continue through Stage-A's original generator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CORE_SHA256 = "30c894ea31541cdde76a5a915bf635a36838a1cdb11f761c334c8986e71a01bc"
ADAPTER_RUNNER_SHA256 = "7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc"
PROFILE_SOURCE_SHA256 = "fd7303f513fdbada80da50e5dd395e9441142f1658544765d7523f1618f55253"
PROFILE_METHOD = "shared_frequency_profile"
ADAPTER_CONTRACT = "cross_tfg_silent_video_cli_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_module(name: str, path: Path, expected_sha256: str):
    path = path.resolve(strict=True)
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"pinned source SHA mismatch for {path}: {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module, path


def _load_stagea(path: Path):
    core, _ = _load_module("cross_tfg_fit_profile_stagea_core", path, CORE_SHA256)
    return core


def _profile_transform(samples: np.ndarray, method: Mapping[str, Any], profile_module: Any) -> tuple[np.ndarray, dict[str, Any]]:
    config = method.get("config")
    if not isinstance(config, Mapping) or config.get("method") != PROFILE_METHOD:
        raise ValueError("profile transform requires the frozen shared_frequency_profile method config")
    profile_id = str(config.get("profile_id", ""))
    output, diagnostics = profile_module.transform_shared_profile(
        samples,
        profile_id,
        sample_rate_hz=int(config.get("sample_rate_hz", 16000)),
        target_rms=float(config["target_rms"]),
        peak_ceiling=float(config["peak_ceiling"]),
    )
    output = np.ascontiguousarray(output, dtype=np.float32)
    if output.shape != samples.shape or not np.isfinite(output).all():
        raise RuntimeError("shared profile must preserve finite float32 sample count")
    diagnostics = dict(diagnostics)
    diagnostics.update({
        "operation": "pinned_shared_frequency_profile_v1",
        "profile_id": profile_id,
        "profile_source_sha256": PROFILE_SOURCE_SHA256,
    })
    return output, diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stagea-core", required=True)
    parser.add_argument("--cross-frontend-adapter-runner", required=True)
    parser.add_argument("--profile-source", required=True)
    own, remaining = parser.parse_known_args(argv)

    core = _load_stagea(Path(own.stagea_core))
    adapter_runner, _ = _load_module(
        "cross_tfg_pinned_cross_frontend_adapter_runner",
        Path(own.cross_frontend_adapter_runner),
        ADAPTER_RUNNER_SHA256,
    )
    profile_source = Path(own.profile_source).resolve(strict=True)
    if _sha256(profile_source) != PROFILE_SOURCE_SHA256:
        raise RuntimeError("shared-frequency profile implementation does not match the frozen source pin")
    if str(Path(own.profile_source).parent.parent.parent) not in sys.path:
        sys.path.insert(0, str(Path(own.profile_source).parent.parent.parent))
    from scripts.audio_enhancement import shared_frequency_profile_v1 as profile_module

    original_transform = core._transform
    original_generator = core._run_generator

    def transform(samples: np.ndarray, method: Mapping[str, Any]):
        config = method.get("config", {})
        if isinstance(config, Mapping) and config.get("method") == PROFILE_METHOD:
            return _profile_transform(samples, method, profile_module)
        return original_transform(samples, method)

    def generator(**kwargs):
        model = kwargs["model"]
        model_generator = model.get("generator", {})
        if isinstance(model_generator, Mapping) and model_generator.get("contract") == ADAPTER_CONTRACT:
            return adapter_runner._adapter_generator(core, **kwargs)
        return original_generator(**kwargs)

    core._transform = transform
    core._run_generator = generator
    stagea_args = core.build_parser().parse_args(remaining)
    return core.run(stagea_args)


if __name__ == "__main__":
    raise SystemExit(main())
