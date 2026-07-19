#!/usr/bin/env python3
"""Extract Fs from audio using pretrained Wav2Sem (Wav2Bert) weights.

Loads the pretrained Wav2Bert checkpoint (7-layer TCN + 12-layer Transformer,
trained on LibriSpeech-960 with L1 loss to BERT CLS), encodes each audio
sample, mean-pools across time to produce a 768-dim Fs vector.

Per the Wav2Sem training code (train_wav2sem.py: out_emb = out.mean(axis=1)),
Fs = mean(T-dim) of the encoder output.

Usage
-----
    python scripts/32_wav2sem_infer_fs.py [--samples IDS] [--smoke]
        [--checkpoint PATH] [--device cuda] [--output DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tfg_feature_common import ENGLISH_STUDY_SAMPLES, OUTPUT_BASE_EN, TARGET_SR

ALIGNMENT_MANIFEST = OUTPUT_BASE_EN / "manifest" / "alignment.json"
DEFAULT_CKPT = Path("/root/autodl-tmp/checkpoints/wav2sem/wav2sem.pth.tar")
OUTPUT_PATH = OUTPUT_BASE_EN / "metrics" / "wav2sem_fs.json"


def load_wav2sem(checkpoint_path, device="cpu"):
    import torch
    from importlib.util import spec_from_file_location, module_from_spec

    wav2sem_repo = Path("/root/autodl-tmp/repos/Wav2Sem")
    code_dir = wav2sem_repo / "CodeTalker_Wav2Sem"
    if str(code_dir) not in sys.path:
        sys.path.insert(0, str(code_dir))
    if str(code_dir.parent) not in sys.path:
        sys.path.insert(0, str(code_dir.parent))

    model_file = code_dir / "models" / "wav2sem.py"
    spec = spec_from_file_location("wav2sem_model", model_file)
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)

    config = mod.wav2vecconfig()
    model = mod.Wav2Bert(config)
    model.eval()

    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[32] missing keys: {len(missing)}", file=sys.stderr)
        for m in missing[:10]:
            print(f"  {m}", file=sys.stderr)
    if unexpected:
        print(f"[32] unexpected keys: {len(unexpected)}", file=sys.stderr)
        for u in unexpected[:10]:
            print(f"  {u}", file=sys.stderr)

    model.to(device)
    return model, len(missing)


def load_audio_mono(filepath, target_sr=TARGET_SR):
    import librosa
    y, _ = librosa.load(str(filepath), sr=target_sr, mono=True)
    return y.astype(np.float32)


def infer_fs(model, audio, device="cpu"):
    import torch
    waveform = torch.from_numpy(audio).unsqueeze(0).to(device)
    with torch.no_grad():
        encoded = model(waveform)
        fs = encoded.mean(dim=1).squeeze(0).cpu().numpy()
    return fs.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--samples", type=str,
        default=",".join(str(s) for s in ENGLISH_STUDY_SAMPLES),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CKPT))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--alignment", type=str, default=str(ALIGNMENT_MANIFEST))
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    sample_ids = [1] if args.smoke else [int(x) for x in args.samples.split(",") if x.strip()]
    repo_root = Path(__file__).resolve().parent.parent

    alignment_path = Path(args.alignment)
    if not alignment_path.is_absolute():
        alignment_path = repo_root / alignment_path
    if not alignment_path.exists():
        print(f"[32] alignment manifest not found: {alignment_path}", file=sys.stderr)
        return 1

    output_path = Path(args.output)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"[32] checkpoint not found: {checkpoint_path}", file=sys.stderr)
        return 1

    if output_path.exists() and not args.no_cache:
        if output_path.stat().st_mtime >= max(checkpoint_path.stat().st_mtime,
                                               alignment_path.stat().st_mtime):
            print(f"[32] cached: {output_path}")
            return 0

    print(f"[32] loading Wav2Sem from {checkpoint_path} ...")
    model, n_missing = load_wav2sem(checkpoint_path, args.device)
    if n_missing > 20:
        print(f"[32] too many missing keys ({n_missing}); checkpoint incompatible",
              file=sys.stderr)
        return 1

    alignment = json.loads(alignment_path.read_text())
    entries_all = alignment.get("manifest", alignment)

    target_entries = [e for e in entries_all if int(e["sample_id"]) in sample_ids]

    results = []
    for e in target_entries:
        sid = int(e["sample_id"])
        cond = e["condition"]
        fpath = repo_root / e["filepath"]
        if not fpath.exists():
            print(f"[32] audio missing: {fpath}", file=sys.stderr)
            continue
        audio = load_audio_mono(fpath)
        fs = infer_fs(model, audio, args.device)
        results.append({
            "sample_id": sid,
            "condition": cond,
            "text": e.get("text", ""),
            "Fs_wav2sem": fs.tolist(),
            "fs_norm": float(np.linalg.norm(fs)),
            "duration_s": float(len(audio) / TARGET_SR),
        })
        print(f"[32] sample {sid} {cond}: Fs norm={results[-1]['fs_norm']:.3f}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({
        "checkpoint": str(checkpoint_path),
        "device": args.device,
        "n_entries": len(results),
        "entries": results,
    }, ensure_ascii=False, indent=2))
    print(f"[32] wrote {len(results)} entries to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
