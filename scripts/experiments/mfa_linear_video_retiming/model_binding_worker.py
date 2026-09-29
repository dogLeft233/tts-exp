"""Reconstruct the loaded Wav2Lip parameter digest in a fresh CPU process."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def parameter_sha256(checkpoint_path: Path, wav2lip_root: Path) -> str:
    root = wav2lip_root.resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import torch
    from models import Wav2Lip

    model = Wav2Lip()
    payload = torch.load(str(checkpoint_path.resolve()), map_location="cpu")
    if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
        raise ValueError("Wav2Lip checkpoint has no state_dict")
    state = {str(key).replace("module.", ""): value for key, value in payload["state_dict"].items()}
    model.load_state_dict(state)
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--wav2lip-root", type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps({"loaded_parameter_sha256": parameter_sha256(args.checkpoint, args.wav2lip_root)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
