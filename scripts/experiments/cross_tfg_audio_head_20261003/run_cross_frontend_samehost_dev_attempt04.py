#!/usr/bin/env python3
"""Run a frozen identity/RMS plan through the pinned cross-frontend Stage-A.

The Stage-A scorer and its natural-PCM mux/scoring path remain untouched. This
small adapter selects the source-pinned Wav2Lip native-entry runner and the
qualified silent-video runner for MuseTalk and Ditto. Host/runtime admission
belongs to the outer late-bind launcher, which must validate and record the
current-host binding before importing this dispatcher.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from typing import Any, Mapping


CORE_SHA256 = "d3bd71921dcacfa807626264b945730c9de533bc6e25dadcfc21772cc5158755"
ADAPTER_RUNNER_SHA256 = "90ba0cf3c06e998abb26913f0c2a4c7620136caf4987f6f557bbc95bc74bce2d"
ADAPTER_CONTRACT = "cross_tfg_silent_video_cli_v1"
W2L_MODEL_ID = "wav2lip_gan_bac9a81e"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(path: Path, name: str, expected_sha256: str) -> Any:
    path = path.resolve(strict=True)
    actual = sha_file(path)
    if actual != expected_sha256:
        raise ValueError(f"qualified source SHA mismatch for {path}: {actual}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load qualified source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stagea-core", required=True)
    parser.add_argument("--cross-frontend-adapter-runner", required=True)
    own, remaining = parser.parse_known_args(argv)
    core = load_source(Path(own.stagea_core), "samehost_stagea_core_attempt04", CORE_SHA256)
    adapter = load_source(Path(own.cross_frontend_adapter_runner),
                          "samehost_cross_frontend_adapter_attempt06", ADAPTER_RUNNER_SHA256)
    stagea_args = core.build_parser().parse_args(remaining)
    original_generator = core._run_generator

    def generator(**kwargs: Any) -> dict[str, Any]:
        row = kwargs.get("row", {})
        model = kwargs.get("model", {})
        if isinstance(row, Mapping) and row.get("model_id") == W2L_MODEL_ID:
            return adapter._wav2lip_native_generator(core, **kwargs)
        descriptor = model.get("generator", {}) if isinstance(model, Mapping) else {}
        if isinstance(descriptor, Mapping) and descriptor.get("contract") == ADAPTER_CONTRACT:
            return adapter._adapter_generator(core, **kwargs)
        return original_generator(**kwargs)

    core._run_generator = generator
    return core.run(stagea_args)


if __name__ == "__main__":
    raise SystemExit(main())
