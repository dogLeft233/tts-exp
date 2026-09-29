from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
ROOT = Path(__file__).resolve().parents[3] / "third_party/Wav2Lip"
sys.path.insert(0, str(ROOT))

import audio
import numpy as np


def sha256_array(value: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(value, dtype="<f8").tobytes()).hexdigest()


def inspect(path: str) -> dict[str, object]:
    waveform = audio.load_wav(path, 16000)
    mel = np.asarray(audio.melspectrogram(waveform), dtype=np.float64)
    if mel.ndim != 2 or mel.shape[0] != 80 or not np.isfinite(mel).all():
        raise ValueError(f"invalid mel for {path}: {mel.shape}")
    fps = 25.0
    chunks = 0
    while True:
        start = int(chunks * 80.0 / fps)
        if start + 16 > mel.shape[1]:
            chunks += 1
            break
        chunks += 1
    return {"sample_count": int(np.asarray(waveform).size), "mel_shape": [int(x) for x in mel.shape], "mel_chunk_count": int(chunks), "mel_sha256": sha256_array(mel)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    source = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for sample_id, paths in source.items():
        if not isinstance(paths, dict):
            raise TypeError(f"malformed probe row: {sample_id}")
        arms = {arm: inspect(str(paths[arm])) for arm in ("N", "N_REPEAT", "W", "BRIDGE_075")}
        mfa = inspect(str(paths["mfa"]))
        if arms["N"]["sample_count"] != arms["N_REPEAT"]["sample_count"] or arms["N"]["sample_count"] != arms["W"]["sample_count"] or arms["N"]["sample_count"] != arms["BRIDGE_075"]["sample_count"]:
            raise ValueError(f"audio sample counts differ: {sample_id}")
        direction = np.asarray([0.0])
        # The directional metric is calculated from mel values in a second pass
        # only for BRIDGE_075; retain compact metadata for the protocol.
        natural = np.asarray(audio.melspectrogram(audio.load_wav(str(paths["N"]), 16000)), dtype=np.float64)
        target = np.asarray(audio.melspectrogram(audio.load_wav(str(paths["mfa"]), 16000)), dtype=np.float64)
        candidate = np.asarray(audio.melspectrogram(audio.load_wav(str(paths["BRIDGE_075"]), 16000)), dtype=np.float64)
        direction = (target - natural).reshape(-1)
        movement = (candidate - natural).reshape(-1)
        denominator = max(float(np.dot(direction, direction)), 1e-12)
        progress = float(np.dot(movement, direction) / denominator)
        rows.append({"sample_id": str(sample_id), "arms": arms, "mfa_linear": mfa, "bridge_movement": {"progress": progress, "orthogonal_ratio": float(np.linalg.norm(movement - progress * direction) / max(float(np.linalg.norm(direction)), 1e-12)), "mel_mae_to_natural": float(np.mean(np.abs(candidate - natural))), "mel_mae_to_mfa_linear": float(np.mean(np.abs(candidate - target)))}})
    body = {"schema_version": 1, "stage_id": "mel_diagnostics", "protocol_id": "wav2lip_face_roi_replacement", "status": "complete", "record_count": len(rows), "rows": rows, "runtime": {"wav2lip_root": str(ROOT.resolve()), "mel": "official Wav2Lip audio.melspectrogram", "fps": 25, "mel_step_size": 16}}
    body["artifact_sha256"] = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
    Path(args.output).write_text(json.dumps(body, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
