"""AVS_u paper-definition replication; explicit checkpoint and reduction choices.

No author-provided AVS scorer is available. Uses upstream AV-HuBERT encoder,
dataset audio frontend and visual transform, with an AutoAVSR mouth crop adapter.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def shift_pcm(audio, samples):
    import numpy as np
    out = np.zeros_like(audio)
    if samples > 0:
        out[samples:] = audio[:-samples]
    elif samples < 0:
        out[:samples] = audio[-samples:]
    else:
        out[:] = audio
    return out


def similarities(audio, video):
    import numpy as np
    a, v = np.asarray(audio, dtype=np.float64), np.asarray(video, dtype=np.float64)
    if a.shape != v.shape or a.ndim != 2 or len(a) == 0:
        raise ValueError("matching nonempty TxD features required")
    den = np.linalg.norm(a, axis=1) * np.linalg.norm(v, axis=1)
    if not np.all(den > 0) or not np.all(np.isfinite(a)) or not np.all(np.isfinite(v)):
        raise ValueError("invalid embedding")
    return {"frame_mean": float(np.mean(np.sum(a * v, axis=1) / den)),
            "flatten_cosine": float(np.sum(a * v) / (np.linalg.norm(a) * np.linalg.norm(v)))}


def prepare(repo, root, deploy):
    base = repo / "runs/cloud_av_preference20_20261027_attempt05"
    old = read(repo / "runs/cogenav_sync_full20_20261024_attempt01/selected_inputs.json")
    naturals = sorted((r for r in old if r["arm"] == "N"), key=lambda r: r["sample_id"])
    cloud = read(base / "cogenav_inputs.json")
    if isinstance(cloud, dict):
        cloud = cloud["rows"]
    cloud = sorted(cloud, key=lambda r: r["id"])
    assert len(naturals) == 20 and len(cloud) == 60
    records = [{**{k: r[k] for k in ["id", "sample_id", "audio_path", "video_path", "audio_sha256", "video_sha256"]},
                "arm": "N"} for r in naturals]
    records += [{**{k: r[k] for k in ["id", "sample_id", "audio_path", "video_path", "audio_sha256", "video_sha256"]},
                 "arm": "C"} for r in cloud]
    for r in records:
        for kind in ["audio", "video"]:
            assert sha(r[kind + "_path"]) == r[kind + "_sha256"], r["id"]
    protocol = {
        "status": "FROZEN_BEFORE_MODEL_SCORES", "metric": "AVS_u paper-definition replication; NOT official AVS package",
        "language": "Mandarin Chinese; English-trained model out-of-domain pilot",
        "checkpoint_url": read(root / "setup.json")["weight_url"],
        "checkpoint": "AV-HuBERT Base LRS3 VSR433h; explicit replication choice, exact paper checkpoint unconfirmed",
        "encoder": "strict-load all retained encoder tensors from public fine-tuned checkpoint; final transformer layer; FP32 eval",
        "missing_modality": "upstream extract_finetune None -> zero FEATURES for absent modality, not raw silence",
        "audio": "upstream logfbank PCM16/16k, stack_order_audio and normalize from checkpoint; match video frame count as dataset",
        "visual": "AutoAVSR MediaPipe/VideoProcess RGB96 -> OpenCV RGB2GRAY -> official AV-HuBERT eval transform from checkpoint",
        "primary": "mean per-frame cosine, no ReLU, no best-lag search, no DTW",
        "secondary": "flattened sequence cosine; frozen because paper reduction is not specified",
        "support": "whole native video frames, audio feature tail pad/truncate exactly upstream dataset; no temporal resampling",
        "controls": "20 N fixed videos, real PCM +/-6400 samples, fixed length zero fill, rerun audio encoder",
        "qualification": "primary score must decrease in >=32/40 +/-400ms controls; otherwise stop before cloud preference",
        "statistics": "20 sources independent; source-mean T-N with 10000 source bootstraps seed20261027, strict12 separate",
        "caveats": ["control pass does not establish perceptual truth", "not three independent AVS judges", "no Chinese fine-tuning"],
        "script_sha256": sha(__file__), "source_tar_sha256": sha(deploy / "official_source.tar.gz"),
        "cloud_pairs_sha256": sha(base / "paired_scores.json"),
    }
    for name, obj in [("protocol.json", protocol), ("inputs.json", records)]:
        path = root / name
        if path.exists():
            frozen = read(path)
            if name == "protocol.json" and frozen != obj:
                repair = read(root / "execution_repair.json")
                assert frozen["script_sha256"] == repair["original_script_sha256"]
                assert obj["script_sha256"] == repair["new_script_sha256"]
                assert {k: v for k, v in frozen.items() if k != "script_sha256"} == {k: v for k, v in obj.items() if k != "script_sha256"}
            else:
                assert frozen == obj
        else:
            write(path, obj)
    write(root / "status.json", {"status": "PREPARED_WAITING_WEIGHT", "inputs": len(records)})
    return records


def load_encoder(deploy, root, device):
    import torch
    from omegaconf import OmegaConf
    source = Path((deploy / "source_path.txt").read_text())
    sys.path[:0] = [str(deploy / "vendor"), str(source)]
    import avhubert
    from avhubert.hubert import AVHubertModel, AVHubertConfig
    from avhubert.hubert_pretraining import AVHubertPretrainingConfig
    from fairseq.dataclass.utils import convert_namespace_to_omegaconf
    from fairseq.models.wav2vec.wav2vec2 import Wav2Vec2Config
    receipt = read(deploy / "download_receipt.json")
    weight = deploy / "base_lrs3_433h.pt"
    assert weight.stat().st_size == receipt["bytes"] and sha(weight) == receipt["sha256"]
    # Trusted public PyTorch checkpoint contains OmegaConf, not just tensor data.
    state = torch.load(weight, map_location="cpu", weights_only=False)
    cfg = state.get("cfg")
    if cfg is None:
        cfg = convert_namespace_to_omegaconf(state["args"])
    if not OmegaConf.is_config(cfg):
        cfg = OmegaConf.create(cfg)
    wcfg = cfg.model.w2v_args
    if not OmegaConf.is_config(wcfg):
        wcfg = convert_namespace_to_omegaconf(wcfg)
    # The released 2022 checkpoint has legacy keys absent from current dataclasses.
    # Merge unstructured copies of defaults, preserving every released setting.
    model_cfg = OmegaConf.merge(OmegaConf.to_container(OmegaConf.structured(Wav2Vec2Config())),
                               OmegaConf.to_container(OmegaConf.structured(AVHubertConfig())), wcfg.model)
    task_defaults = OmegaConf.to_container(OmegaConf.structured(AVHubertPretrainingConfig()))
    task_cfg = OmegaConf.merge(task_defaults, wcfg.task)
    # VSR fine-tuning used video-only and did not serialize audio stacking again.
    # Inherit the pretrained 4x26 frontend, then apply saved fine-tune task options.
    eval_task = OmegaConf.merge(task_defaults, wcfg.task, cfg.task)
    assert int(eval_task.stack_order_audio) * 26 == int(model_cfg.audio_feat_dim)
    # Keep the original task root for lazy config interpolation. Do not resolve
    # unused legacy aliases such as task.input_modality, absent upstream as well.
    merged = OmegaConf.create({"model": OmegaConf.to_container(model_cfg, resolve=False),
                              "task": OmegaConf.to_container(task_cfg, resolve=False)})
    model = AVHubertModel(merged.model, merged.task, [None])
    model.remove_pretraining_modules()
    prefixes = ["encoder.w2v_model.", "w2v_encoder.w2v_model."]
    prefix = next(p for p in prefixes if any(k.startswith(p) for k in state["model"]))
    retained = {k[len(prefix):]: v for k, v in state["model"].items() if k.startswith(prefix)}
    # Fine-tuning retains an unused pretraining target table. Restore it too,
    # without requiring the original cluster-label dictionary on local disk.
    if "label_embs_concat" in retained and not hasattr(model, "label_embs_concat"):
        model.register_parameter("label_embs_concat", torch.nn.Parameter(
            torch.empty(retained["label_embs_concat"].shape, dtype=torch.float32)))
    model.load_state_dict(retained, strict=True)
    write(root / "model_load.json", {"strict_encoder": True, "weight": receipt,
          "encoder_keys": len(retained), "excluded_keys": [k for k in state["model"] if not k.startswith(prefix)],
          "parameters": sum(p.numel() for p in model.parameters()), "torch": torch.__version__,
          "checkpoint_cfg": OmegaConf.to_container(cfg, resolve=False, enum_to_str=True),
          "encoder_cfg_with_task_root": OmegaConf.to_container(merged, resolve=False, enum_to_str=True), "device": device})
    del state, retained
    return model.eval().to(device), eval_task, source


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def score_all(repo, root, deploy, records, device):
    import cv2
    import numpy as np
    import torch
    from scipy.io import wavfile
    torch.set_num_threads(4)
    torch.manual_seed(20261027)
    model, task, source = load_encoder(deploy, root, device)
    from python_speech_features import logfbank
    # avhubert.__init__ wildcard exports can shadow this name with fairseq.utils.
    avutils = importlib.import_module("avhubert.utils")
    crop_root = repo / "runs/cogenav_sync_audit_20261024_attempt01"
    detector = module("avs_detector", crop_root / "autoavsr_detector.py").LandmarksDetector()
    cropper = module("avs_crop", crop_root / "autoavsr_video_process.py").VideoProcess(convert_gray=False)
    transform = avutils.Compose([avutils.Normalize(0, 255),
        avutils.CenterCrop((int(task.image_crop_size), int(task.image_crop_size))),
        avutils.Normalize(float(task.image_mean), float(task.image_std))])
    cache = root / "embeddings"
    cache.mkdir(exist_ok=True)
    scores = read(root / "scores.json") if (root / "scores.json").exists() else {}

    def audio_embedding(pcm, frames):
        feats = logfbank(pcm, samplerate=16000).astype(np.float32)
        order = int(task.stack_order_audio)
        if len(feats) % order:
            feats = np.pad(feats, ((0, order - len(feats) % order), (0, 0)))
        feats = feats.reshape(-1, feats.shape[1] * order)
        # Match upstream dataset load_feature, THEN optional per-frame layer norm.
        feats = np.pad(feats[:frames], ((0, max(0, frames - len(feats))), (0, 0)))
        x = torch.from_numpy(feats)
        if task.normalize:
            x = torch.nn.functional.layer_norm(x, x.shape[1:])
        with torch.inference_mode():
            out = model.extract_finetune({"audio": x.T[None].to(device), "video": None}, mask=False)[0]
        return out[0].cpu().numpy()

    def one(row, controls=False):
        key = row["id"]
        dest = cache / (key + ".npz")
        if dest.exists():
            data = np.load(dest)
            a, v = data["audio"], data["video"]
        else:
            cap = cv2.VideoCapture(row["video_path"])
            assert abs(cap.get(cv2.CAP_PROP_FPS) - 25) < 1e-5
            frames = []
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            marks = detector(frames)
            detected = sum(m is not None for m in marks)
            assert detected / len(frames) >= .95
            patches = cropper(frames, marks)
            gray = np.stack([cv2.cvtColor(p, cv2.COLOR_RGB2GRAY) for p in patches])
            x = torch.from_numpy(np.ascontiguousarray(transform(gray))).float()[None, None].to(device)
            sr, pcm = wavfile.read(row["audio_path"])
            assert sr == 16000 and pcm.ndim == 1 and pcm.dtype == np.int16
            a = audio_embedding(pcm, len(gray))
            with torch.inference_mode():
                v = model.extract_finetune({"audio": None, "video": x}, mask=False)[0][0].cpu().numpy()
            np.savez_compressed(dest, audio=a, video=v, detected=detected,
                protocol_sha256=sha(root / "protocol.json"), input_video_sha256=row["video_sha256"],
                input_audio_sha256=row["audio_sha256"])
            cv2.imwrite(str(cache / (key + "_mouth.jpg")), np.concatenate([gray[i] for i in np.linspace(0, len(gray)-1, 6).astype(int)], axis=1))
        result = {"id": key, "source": row["sample_id"], "arm": row["arm"],
                  "frames": len(v), "embedding_sha256": sha(dest), **similarities(a, v)}
        if controls:
            sr, pcm = wavfile.read(row["audio_path"])
            control_cache = cache / (key + "_controls.npz")
            shifted = dict(np.load(control_cache)) if control_cache.exists() else {}
            for samples in [-6400, 6400]:
                label = str(samples)
                if label not in shifted:
                    shifted[label] = audio_embedding(shift_pcm(pcm, samples), len(v))
            np.savez_compressed(control_cache, **shifted)
            result["controls"] = [{"shift_samples": int(s), **similarities(aa, v)} for s, aa in shifted.items()]
        scores[key] = result
        write(root / "scores.json", scores)
        write(root / "status.json", {"status": "RUNNING", "scored": len(scores), "last": key})
        print(json.dumps({"scored": len(scores), **result}), flush=True)

    for row in records:
        if row["arm"] == "N":
            one(row, controls=True)
    nrows = [r for r in scores.values() if r["arm"] == "N"]
    lower = sum(c["frame_mean"] < r["frame_mean"] for r in nrows for c in r["controls"])
    controls = {"primary_lower": lower, "total": 40, "required": 32,
                "status": "PASS" if lower >= 32 else "CONTROL_FAILED"}
    write(root / "controls.json", controls)
    if lower < 32:
        write(root / "status.json", {"status": "CONTROL_FAILED", "cloud_preference_run": False, **controls})
        return
    for row in records:
        if row["arm"] == "C":
            one(row)
    paired = read(repo / "runs/cloud_av_preference20_20261027_attempt05/paired_scores.json")
    natural = {r["source"]: r for r in nrows}
    pairs = [{"id": p["id"], "source": p["source"], "strict_reference": p["strict_reference"],
              "N": natural[p["source"]]["frame_mean"], "T": scores[p["id"]]["frame_mean"],
              "delta": scores[p["id"]]["frame_mean"] - natural[p["source"]]["frame_mean"],
              "delta_sync_c": p["delta"]["SyncC"]} for p in paired]
    out = {"status": "EXPLORATORY_COMPLETE_NOT_PERCEPTUAL_VALIDATION", "controls": controls, "views": {}}
    for view, ps in [("all20", pairs), ("strict12", [p for p in pairs if p["strict_reference"]])]:
        sources = sorted({p["source"] for p in ps})
        d = np.array([np.mean([p["delta"] for p in ps if p["source"] == s]) for s in sources])
        rng = np.random.default_rng(20261027)
        out["views"][view] = {"sources": len(sources), "pairs": len(ps), "delta_source_mean": float(d.mean()),
             "ci95_descriptive": np.quantile(d[rng.integers(0, len(d), (10000, len(d)))].mean(1), [.025, .975]).tolist(),
             "tts_higher": sum(p["delta"] > 0 for p in ps), "natural_higher": sum(p["delta"] < 0 for p in ps),
             "conflicts_sync_c": sum(p["delta"] * p["delta_sync_c"] < 0 for p in ps)}
    write(root / "pairs.json", pairs)
    write(root / "summary.json", out)
    write(root / "status.json", {"status": "COMPLETE", "scored": len(scores), "controls": controls})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--deploy", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    records = prepare(args.repo, args.run, args.deploy)
    if not args.prepare_only:
        if not (args.deploy / "download_receipt.json").exists():
            raise RuntimeError("weight download incomplete; no model scoring attempted")
        score_all(args.repo, args.run, args.deploy, records, args.device)


if __name__ == "__main__":
    main()
