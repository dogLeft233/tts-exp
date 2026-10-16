"""Apply the frozen per-condition RNG settings before running Wav2Lip inference."""

from __future__ import annotations

import json
import os
import random
import runpy
import sys
from pathlib import Path
from typing import Any


def seed_runtime(seed: int) -> dict[str, Any]:
    if type(seed) is not int or seed < 0:
        raise ValueError("render seed must be a nonnegative integer")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)

    import numpy as np

    np.random.seed(seed)

    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return {
        "seed": seed,
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "python_random_seed": seed,
        "numpy_seed": seed,
        "torch_seed": seed,
        "cuda_seed_all": seed,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_version": str(torch.__version__),
    }


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) < 1:
        raise SystemExit("usage: wav2lip_seeded_inference.py INFERENCE_SCRIPT [ARGS ...]")
    script = Path(args[0]).resolve()
    seed_text = os.environ.get("P1_RENDER_SEED")
    if seed_text is None or not seed_text.isdecimal():
        raise RuntimeError("P1_RENDER_SEED must be set by the reviewed render runner")
    seed = int(seed_text)
    if os.environ.get("PYTHONHASHSEED") != str(seed):
        raise RuntimeError("PYTHONHASHSEED does not match P1_RENDER_SEED")
    if not script.is_file():
        raise FileNotFoundError(f"Wav2Lip inference script is missing: {script}")
    seed_state = seed_runtime(seed)
    if seed_state["pythonhashseed"] != str(seed):
        raise RuntimeError("Python hash seed receipt does not match the requested seed")
    print("P1_RENDER_SEED_CONFIG=" + json.dumps(seed_state, sort_keys=True, separators=(",", ":")), flush=True)
    sys.path.insert(0, str(script.parent))
    sys.argv = [str(script), *args[1:]]
    runpy.run_path(str(script), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
