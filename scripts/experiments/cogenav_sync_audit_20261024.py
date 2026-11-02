"""Frozen CoGenAV Base representation diagnostic; no training or media generation.

Requires the pinned upstream and AutoAVSR preprocessing files in --run.
Paper Eq(1) is reported separately from derived lag diagnostics. Never call this
a calibrated perceptual metric. Full native clips have different clocks.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def lag_curve(a, v, maxlag=15):
    """Fixed video centers across every lag; positive lag selects later audio."""
    assert a.shape == v.shape and a.ndim == 2
    if len(a) <= 2 * maxlag:
        raise ValueError("Too short for common lag support")
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
    centers = np.arange(maxlag, len(a) - maxlag)
    local = np.stack([np.maximum(0, (a[centers + k] * v[centers]).sum(-1))
                      for k in range(-maxlag, maxlag + 1)])
    return local.mean(1), local


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def compatible_protocol(parent, current):
    """Only cohort size and explicit cache provenance may differ."""
    ignored = {"sources", "reuse"}
    return {k: v for k, v in parent.items() if k not in ignored} == {
        k: v for k, v in current.items() if k not in ignored}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--sources", type=int, default=4)
    p.add_argument("--reuse-run", help="Import verified compatible pilot caches without new forwards")
    p.add_argument("--record-long-clips", action="store_true", help="Retain >30s inputs as explicitly unmeasurable")
    args = p.parse_args()
    root = Path(args.run).resolve()
    root.mkdir(parents=True, exist_ok=True)
    protocol_path = root / "protocol.json"
    protocol = {
        "model": "CoGenAV Base", "hf_revision": "3c85c000c60fe7ffb3aaa3c63c96ba33ad8ce42c",
        "weight_sha256": "90f8c5165282c190c81c23d8fe92db2aa700862e021b1312fb96779caf9cfe61",
        "github_revision": "dbd3f835a2cc859766c421aaf3918cda2246008d",
        "sources": args.sources, "selection": "lexical first sources, all N/L06/L17 draws",
        "input_manifest_sha256": sha(args.manifest), "fps": 25,
        "video": "AutoAVSR MediaPipe detector/VideoProcess RGB 96x96; upstream weighted grayscale; Base mean0 std1; black pad to750",
        "audio": "canonical manifest PCM16 mono16k; default WhisperFeatureExtractor 80mel pad30sec; no noise",
        "representation": "separate audio-only/video-only encoder outputs BEFORE DeltaUpsampler; FP32 eval",
        "primary": "paper Eq1 mean ReLU cosine on true native length",
        "derived": "31lag +/-600ms, same video centers over lags; no wrapping; zero/max/offtarget contrast reported separately",
        "comparison": "three families N-L06,N-L17,L06-L17; paired same draw; source-level aggregation",
        "qualification": "raw audio shifts +/-160ms must move peak by induced delay within40ms in >=80% pilot clip-shift cases; peak-vs-outerlags positive median; otherwise UNVALIDATED, no quality conclusion",
        "boundary": "exploratory model agreement only; no perception or pure evaluator-bias identification",
    }
    allowed_protocols = set()
    if args.reuse_run:
        parent = Path(args.reuse_run).resolve()
        parent_protocol = json.loads((parent / "protocol.json").read_text())
        assert compatible_protocol(parent_protocol, protocol), "Changed scientific protocol"
        allowed_protocols.add(sha(parent / "protocol.json"))
        protocol["reuse"] = {"root": str(parent), "protocol_sha256": sha(parent / "protocol.json"),
                             "manifest_sha256": sha(parent / "final_manifest.json")}
    if protocol_path.exists():
        assert json.loads(protocol_path.read_text()) == protocol
    else:
        dump(protocol_path, protocol)
    allowed_protocols.add(sha(protocol_path))
    if args.reuse_run:
        records = json.loads((parent / "final_manifest.json").read_text())
        imported = []
        for record in records:
            rel = Path(record["path"])
            if rel.parts[0] not in {"embeddings", "crops"}:
                continue
            source = parent / rel
            assert sha(source) == record["sha256"]
            target = root / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert sha(target) == record["sha256"]
            else:
                shutil.copy2(source, target)
            imported.append(record)
        dump(root / "cache_import.json", {"parent": str(parent), "files": imported})
    import cv2
    import torch
    from transformers import WhisperFeatureExtractor
    from omegaconf import OmegaConf
    from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config

    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    np.random.seed(20261024)
    sys.path.insert(0, str(root / "upstream"))
    from models.cogenav import CoGenAV
    weight = root / "weights/base_cogen_mirror.pt"
    assert sha(weight) == protocol["weight_sha256"]
    # Merge versioned fairseq defaults; explicit upstream options take priority.
    cfg = OmegaConf.merge(OmegaConf.structured(Wav2Vec2Config()),
                          OmegaConf.load(root / "upstream/config/base.yaml"))
    cfg_path = root / "base_resolved.yaml"
    OmegaConf.save(cfg, cfg_path)
    model = CoGenAV(str(cfg_path), model_tensor=None)
    state = torch.load(weight, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval().cuda()
    dump(root / "model_load.json", {"strict": True, "parameters": sum(x.numel() for x in model.parameters()),
                                  "weight_sha256": sha(weight), "torch": torch.__version__,
                                  "config_sha256": sha(cfg_path), "device": torch.cuda.get_device_name()})
    detector = module("avdetector", root / "autoavsr_detector.py").LandmarksDetector()
    cropper = module("avcrop", root / "autoavsr_video_process.py").VideoProcess(convert_gray=False)
    fe = WhisperFeatureExtractor()
    rows = json.loads(Path(args.manifest).read_text())["rows"]
    sources = sorted({r["sample_id"] for r in rows})[:args.sources]
    rows = [r for r in rows if r["sample_id"] in sources]
    dump(root / "selected_inputs.json", rows)
    unmeasurable = []
    if args.record_long_clips:
        eligibility = {"rule": "No truncation or changed context; canonical audio >30sec is UNMEASURABLE under native30sec protocol",
                       "inputs": [{"id": r["id"], "duration_s": r["features"]["duration_s"],
                                   "prior_qualified": r["qualified"], "reason": "EXCEEDS_NATIVE_30SEC_INPUT"}
                                  for r in rows if r["features"]["duration_s"] > 30]}
        epath = root / "eligibility_protocol.json"
        if epath.exists():
            assert json.loads(epath.read_text()) == eligibility
        else:
            dump(epath, eligibility)
        unmeasurable = eligibility["inputs"]
        dump(root / "unmeasurable.json", unmeasurable)
    outputs = []
    new_forwards = 0
    (root / "embeddings").mkdir(exist_ok=True)
    (root / "crops").mkdir(exist_ok=True)
    for row in rows:
        if row["id"] in {r["id"] for r in unmeasurable}:
            print(json.dumps({"id": row["id"], "status": "UNMEASURABLE_NATIVE30SEC"}), flush=True)
            continue
        cache = root / "embeddings" / (row["id"] + ".npz")
        if cache.exists():
            data = np.load(cache)
            assert str(data["protocol_sha256"]) in allowed_protocols
            a, v = data["audio"], data["video"]
        else:
            for field in ["audio", "video"]:
                assert sha(row[field + "_path"]) == row[field + "_sha256"]
            cap = cv2.VideoCapture(row["video_path"])
            assert abs(cap.get(cv2.CAP_PROP_FPS) - 25) < 1e-3
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            landmarks = detector(frames)
            detected = sum(l is not None for l in landmarks)
            if detected / len(frames) < .95:
                raise ValueError(f"Insufficient detected face support: {row['id']}")
            patches = cropper(frames, landmarks)
            assert patches.shape[1:3] == (96, 96)
            grid = np.concatenate([patches[i] for i in np.linspace(0, len(patches)-1, 8).astype(int)], axis=1)
            cv2.imwrite(str(root / "crops" / (row["id"] + ".jpg")), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            gray = np.dot(patches[..., :3], [0.1140, .5870, .2989]).astype(np.uint8)
            nvideo = len(gray)
            if nvideo > 750:
                raise ValueError("Clip longer than native30sec; no silent truncation")
            padded = np.zeros((750, 96, 96), dtype=np.float32)
            padded[:nvideo] = gray / 255.
            video = torch.from_numpy(padded)[None, ..., None].permute(0, 2, 3, 4, 1).cuda()
            import soundfile as sf
            audio, sr = sf.read(row["audio_path"], dtype="float32")
            assert sr == 16000 and audio.ndim == 1
            mel = torch.from_numpy(fe(audio, sampling_rate=sr).input_features[0])[None].cuda()
            n = min(nvideo, len(audio)//640)
            with torch.inference_mode():
                a = model(None, mel, use_upsampler=False)[0, :n].cpu().numpy()
                v = model(video, None, use_upsampler=False)[0, :n].cpu().numpy()
            new_forwards += 2
            np.savez_compressed(cache, audio=a, video=v, detected=detected, nvideo=nvideo,
                                audio_samples=len(audio), protocol_sha256=sha(protocol_path))
        curve, local = lag_curve(a, v)
        an = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
        vn = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
        eq1 = float(np.maximum(0, (an * vn).sum(-1)).mean())
        lags = np.arange(-15, 16)
        null = float(curve[np.abs(lags) >= 8].mean())
        item = {k: row[k] for k in ["id", "sample_id", "arm", "draw", "qualified", "score_float32"]}
        item.update(eq1=eq1, zero_common=float(curve[15]), max_score=float(curve.max()),
                    peak_lag_frames=int(lags[curve.argmax()]), outer_mean=null,
                    zero_excess=float(curve[15]-null), peak_excess=float(curve.max()-null),
                    frames=len(a), curve=curve.tolist(), embedding_sha256=sha(cache))
        # Recompute the audio representation after a real waveform shift. This
        # is deliberately not a tautological shift of cached feature rows.
        shift_path = root / "embeddings" / (row["id"] + "_rawshifts.npz")
        if shift_path.exists():
            shifted_data = np.load(shift_path)
            shifted = {int(k): shifted_data[k] for k in shifted_data.files}
        else:
            import soundfile as sf
            audio, sr = sf.read(row["audio_path"], dtype="float32")
            assert sha(row["audio_path"]) == row["audio_sha256"] and sr == 16000
            shifted = {}
            for shift in [-4, 4]:
                samples = shift * 640
                wav = np.pad(audio, (max(samples, 0), max(-samples, 0)))
                wav = wav[max(-samples, 0):max(-samples, 0)+len(audio)]
                smel = torch.from_numpy(fe(wav, sampling_rate=sr).input_features[0])[None].cuda()
                with torch.inference_mode():
                    sa = model(None, smel, use_upsampler=False)[0, :len(v)].cpu().numpy()
                shifted[shift] = sa
                new_forwards += 1
            np.savez_compressed(shift_path, **{str(k): val for k, val in shifted.items()})
        item["raw_shift_controls"] = []
        for shift, sa in sorted(shifted.items()):
            sc, _ = lag_curve(sa, v)
            peak = int(lags[sc.argmax()])
            error = peak - item["peak_lag_frames"] - shift
            item["raw_shift_controls"].append({"shift_frames": shift, "peak_lag_frames": peak,
                                                "peak_movement_error_frames": error,
                                                "pass_within_one_frame": abs(error) <= 1,
                                                "curve": sc.tolist()})
        outputs.append(item)
        dump(root / "scores.json", outputs)
        print(json.dumps({k:item[k] for k in ["id", "eq1", "peak_lag_frames", "zero_excess"]}), flush=True)
    dump(root / "extraction_complete.json", {"clips": len(outputs), "planned_clips": len(rows),
                                           "unmeasurable": len(unmeasurable), "sources": sources,
                                           "new_forwards": new_forwards,
                                           "protocol_sha256": sha(protocol_path), "status": "EXTRACTED_NOT_YET_QUALIFIED"})


if __name__ == "__main__":
    main()
