#!/usr/bin/env python3
"""Run one Ditto inference with explicit deterministic seeding."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ditto-dir", type=Path, required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--cfg-pkl", required=True)
    parser.add_argument("--audio-path", required=True)
    parser.add_argument("--source-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    inference_path = args.ditto_dir / "inference.py"
    if not inference_path.exists():
        raise FileNotFoundError(inference_path)
    sys.path.insert(0, str(args.ditto_dir))
    spec = importlib.util.spec_from_file_location("ditto_inference", inference_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {inference_path}")
    inference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(inference)

    inference.seed_everything(args.seed)
    sdk = inference.StreamSDK(args.cfg_pkl, args.data_root)
    inference.run(
        sdk,
        args.audio_path,
        args.source_path,
        args.output_path,
    )


if __name__ == "__main__":
    main()
