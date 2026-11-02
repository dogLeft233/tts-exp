"""Offline, frozen Qwen3 speech-tokenizer roundtrip. No language-model inference.

Only this run's new output directory is written. Raw decoder length is preserved.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback

import numpy as np
import soundfile as sf


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def validate_waveform(values: np.ndarray, sample_rate: int, *, expected_sr: int = 24000) -> np.ndarray:
    x = np.asarray(values)
    if int(sample_rate) != expected_sr:
        raise ValueError(f"expected prequalified {expected_sr} Hz input; got {sample_rate}")
    if x.ndim != 1 or x.size == 0:
        raise ValueError("waveform must be nonempty mono")
    if not np.isfinite(x).all():
        raise ValueError("waveform has non-finite samples")
    return x.astype(np.float32, copy=False)


def waveform_stats(x: np.ndarray, sr: int) -> dict:
    y = x.astype(np.float64)
    return {"samples": int(x.size), "sample_rate": int(sr), "duration_s": float(x.size / sr),
            "peak": float(np.max(np.abs(y))), "rms": float(np.sqrt(np.mean(y * y))),
            "mean": float(np.mean(y)), "samples_abs_ge_1": int(np.sum(np.abs(y) >= 1)),
            "float32_le_sha256": hashlib.sha256(x.astype("<f4").tobytes()).hexdigest()}


def preserve_float_wav(path: Path, x: np.ndarray, sr: int) -> dict:
    """FLOAT avoids clipping/PCM quantization; neither crop nor pad nor stretch."""
    sf.write(path, x, sr, subtype="FLOAT")
    readback, observed_sr = sf.read(path, dtype="float32")
    if observed_sr != sr or not np.array_equal(readback, x):
        raise RuntimeError("FLOAT WAV readback differs from raw array")
    return {"path": str(path.resolve()), "file_sha256": sha256(path), **waveform_stats(x, sr)}


def load_input(row: dict) -> tuple[np.ndarray, int, Path]:
    name = next((key for key in ("input24_path", "input24k_path", "input_24k_path", "source_path") if key in row), None)
    if name is None:
        raise ValueError("manifest needs input24_path or source_path")
    p = Path(row[name])
    expected = next((row[key] for key in ("input24_sha256", "input24k_sha256", "input_24k_sha256", "source_sha256") if row.get(key)), None)
    if not expected or sha256(p) != expected:
        raise ValueError(f"input SHA mismatch or absent: {p}")
    x, sr = sf.read(p, dtype="float32", always_2d=False)
    return validate_waveform(x, sr), int(sr), p


def validate_codes(codes: np.ndarray) -> np.ndarray:
    if codes.ndim != 2 or codes.shape[1] != 16 or codes.shape[0] == 0:
        raise ValueError(f"expected (T,16) all-codebook codes, got {codes.shape}")
    if not np.issubdtype(codes.dtype, np.integer):
        raise ValueError("codes must be integers")
    if np.min(codes) < 0 or np.max(codes[:, 0]) >= 4096 or np.max(codes[:, 1:]) >= 2048:
        raise ValueError("codes outside configured semantic/acoustic vocabulary")
    return codes


def run(args: argparse.Namespace) -> None:
    # Before loading transformers: force all HF operations offline, including feature-extractor loading.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    started = time.time()
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    events = out / "events.jsonl"

    def event(kind: str, **fields: object) -> None:
        row = {"event": kind, "utc_unix": time.time(), "elapsed_s": time.time() - started, **fields}
        with events.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        print(json.dumps(row, ensure_ascii=False, allow_nan=False), flush=True)

    def timeout_handler(signum: int, frame: object) -> None:
        raise TimeoutError(f"codec hard timeout after {args.timeout_seconds}s")

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout_seconds)
    event("started", pid=os.getpid(), command=sys.argv, executable=sys.executable)
    try:
        manifest_path = Path(args.manifest).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not manifest["samples"]:
            raise ValueError("empty manifest")
        ids = [str(row["sample_id"]) for row in manifest["samples"]]
        if len(set(ids)) != len(ids) or any(Path(i).name != i or i in (".", "..") for i in ids):
            raise ValueError("duplicate or unsafe sample IDs")
        # Validate every input before model construction, not selectively after seeing outputs.
        qualified = [(row, *load_input(row)) for row in manifest["samples"]]
        model_dir = Path(manifest["model_dir"]).resolve()
        config = json.loads((model_dir / "config.json").read_text())
        if (config["model_type"] != "qwen3_tts_tokenizer_12hz" or config["encoder_valid_num_quantizers"] != 16
                or config["input_sample_rate"] != 24000 or config["output_sample_rate"] != 24000):
            raise ValueError("unexpected tokenizer contract")
        model_files = {name: {"sha256": sha256(model_dir / name), "bytes": (model_dir / name).stat().st_size}
                       for name in ("config.json", "preprocessor_config.json", "model.safetensors")}
        expected_model_sha = manifest.get("model_sha256", manifest.get("model_weights_sha256"))
        if not expected_model_sha or model_files["model.safetensors"]["sha256"] != expected_model_sha:
            raise ValueError("tokenizer checkpoint SHA mismatch")
        seed = int(manifest.get("seed", 20261022))
        random.seed(seed)
        np.random.seed(seed)
        import torch
        from qwen_tts import Qwen3TTSTokenizer
        torch.manual_seed(seed)
        torch.set_num_threads(2)
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        source_paths = [Path(inspect.getfile(Qwen3TTSTokenizer)), Path(__file__).resolve()]
        runtime = {"pid": os.getpid(), "seed": seed, "dtype": "float32", "attention": "eager", "device": args.device,
                   "python": sys.version, "manifest_path": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                   "model_dir": str(model_dir), "model_files": model_files,
                   "packages": {n: importlib.metadata.version(n) for n in ("torch", "transformers", "qwen-tts", "numpy", "soundfile", "librosa")},
                   "policy": "official Qwen tokenizer encode/decode, all 16 codebooks, no LM/text, no normalization/crop/stretch/pad of decoded waveform"}
        event("loading_model", checkpoint_sha256=model_files["model.safetensors"]["sha256"])
        tokenizer = Qwen3TTSTokenizer.from_pretrained(str(model_dir), local_files_only=True,
                    device_map=args.device, dtype=torch.float32, attn_implementation="eager")
        tokenizer.model.eval().requires_grad_(False)
        source_paths.append(Path(inspect.getfile(type(tokenizer.model))))
        runtime["source_files"] = {str(p): sha256(p) for p in source_paths}
        runtime["input_sample_rate"] = tokenizer.get_input_sample_rate()
        runtime["output_sample_rate"] = tokenizer.get_output_sample_rate()
        runtime["encode_downsample_rate"] = tokenizer.get_encode_downsample_rate()
        runtime["decode_upsample_rate"] = tokenizer.get_decode_upsample_rate()
        write_json(out / "runtime.json", runtime)
        rows = []
        for index, (row, x, sr, source_path) in enumerate(qualified):
            sample_id = str(row["sample_id"])
            directory = out / sample_id
            directory.mkdir()
            input_saved = preserve_float_wav(directory / "input_24k_float.wav", x, sr)
            event("encoding", index=index, sample_id=sample_id, input_samples=int(x.size))
            t0 = time.monotonic()
            with torch.inference_mode():
                encoded = tokenizer.encode(x, sr=sr)
                codes = validate_codes(encoded.audio_codes[0].detach().cpu().numpy())
                reconstructed, reconstructed_sr = tokenizer.decode(encoded)
            if len(reconstructed) != 1:
                raise RuntimeError("single input must produce one reconstructed waveform")
            y = validate_waveform(reconstructed[0], reconstructed_sr)
            codes_path = directory / "codes.npy"
            np.save(codes_path, codes, allow_pickle=False)
            output_saved = preserve_float_wav(directory / "reconstructed_raw_24k_float.wav", y, reconstructed_sr)
            result = {"sample_id": sample_id, "source_manifest_row": row, "source_path": str(source_path),
                      "input": input_saved, "output": output_saved,
                      "codes": {"path": str(codes_path), "shape": list(codes.shape), "dtype": str(codes.dtype), "sha256": sha256(codes_path)},
                      "length_difference_samples_24k": int(y.size - x.size),
                      "length_difference_ms": float((y.size - x.size) / 24),
                      "latency_status": "NOT_INFERRED_FROM_LENGTH_OR_SYNCNET; root timing analysis required",
                      "compute_elapsed_s": time.monotonic() - t0}
            write_json(directory / "record.json", result)
            rows.append(result)
            event("sample_complete", sample_id=sample_id, output_samples=int(y.size), code_shape=list(codes.shape),
                  length_difference_samples=int(y.size - x.size))
        # Write the observations before releasing the process, retaining full untrimmed media.
        write_json(out / "results.json", {"status": "COMPLETE", "count": len(rows), "rows": rows})
        write_json(out / "codec_manifest.json", {"status": "COMPLETE", "count": len(rows), "samples": [
            {"sample_id": r["sample_id"], "speaker_id": r["source_manifest_row"].get("speaker_id"),
             "input24_path": r["input"]["path"], "input24_sha256": r["input"]["file_sha256"],
             "reconstruction_path": r["output"]["path"], "sha256": r["output"]["file_sha256"],
             "samples": r["output"]["samples"], "sample_rate_hz": r["output"]["sample_rate"],
             "record_path": str(out / r["sample_id"] / "record.json")}
            for r in rows]})
        del tokenizer
        gc.collect()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
            peak = int(torch.cuda.max_memory_allocated())
            torch.cuda.empty_cache()
        else:
            peak = 0
        write_json(out / "completion.json", {"status": "COMPLETE", "count": len(rows), "elapsed_s": time.time() - started,
                   "max_cuda_memory_allocated_bytes": peak, "results_sha256": sha256(out / "results.json"), "gpu_model_released": True})
        event("complete", count=len(rows), gpu_model_released=True)
    except BaseException as exc:
        write_json(out / "failure.json", {"status": "FAILED", "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        event("failed", error_type=type(exc).__name__, message=str(exc))
        raise
    finally:
        if hasattr(signal, "SIGALRM"):
            signal.alarm(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    run(parser.parse_args())
