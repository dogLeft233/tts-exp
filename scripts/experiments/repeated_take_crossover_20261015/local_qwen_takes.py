#!/usr/bin/env python3
"""Generate the T1/T2 local Qwen3-TTS arm for the repeated-take pilot.

This runner uses the already-installed qwen-tts PyTorch implementation with
the cached 0.6B Base checkpoint. It deliberately requires a frozen protocol
addendum ID and SHA before it can write any TTS output.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import random
import re
import sys
import time
import traceback
import unicodedata
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np


MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
EXPECTED_MODEL_FILES = {
    "model.safetensors": "180b3b10eb1c9f1b4db7806d5475bae3071c0243c299d49926bab1da3b6946f6",
    "speech_tokenizer/model.safetensors": "836b7b357f5ea43e889936a3709af68dfe3751881acefe4ecf0dbd30ba571258",
}
DEFAULT_MODEL_DIR = Path(
    "/home/wjj/tts-audio/tts-exp/models/Qwen3-TTS-12Hz-0.6B-Base"
)
DEFAULT_MANIFEST = Path(
    "/home/wjj/tts-audio/tts-exp/runs/repeated_take_crossover_20261015_attempt01/"
    "assets_discovery/aishell_repeated_realizations/manifest_v3.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/wjj/tts-audio/tts-exp/runs/repeated_take_crossover_20261015_attempt01/"
    "local_qwen_attempt02_fp32"
)
ALL_SPEAKERS = (
    "S0003",
    "S0006",
    "S0007",
    "S0008",
    "S0014",
    "S0015",
    "S0016",
    "S0020",
)
T1_SEED_BASE = 2026101501
T2_SEED_BASE = 2026101502
GENERATION_KWARGS: dict[str, Any] = {
    "do_sample": True,
    "temperature": 0.9,
    "top_k": 50,
    "top_p": 1.0,
}
MODEL_LOAD_KWARGS: dict[str, Any] = {
    "device_map": "cuda:0",
    "torch_dtype": "float32",
    "attn_implementation": "sdpa",
    "local_files_only": True,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def remove_whitespace(text: str) -> str:
    return "".join(ch for ch in text if not ch.isspace())


def strip_punctuation_for_check(text: str) -> str:
    return "".join(
        ch
        for ch in text
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )


def resolve_asset(asset_root: Path, item: dict[str, Any]) -> Path:
    relative = item.get("staged_path_relative")
    if relative:
        return (asset_root / str(relative)).resolve()
    staged = Path(str(item.get("staged_path", "")))
    if staged.is_absolute():
        return staged
    return (asset_root / staged).resolve()


def load_and_validate_inputs(manifest_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw.decode("utf-8"))
    asset_root = manifest_path.parent
    speakers = manifest.get("speakers")
    if not isinstance(speakers, list):
        raise ValueError("manifest must contain a speakers list")

    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for speaker in speakers:
        speaker_id = str(speaker["speaker"])
        if speaker_id in seen:
            raise ValueError(f"duplicate speaker in input manifest: {speaker_id}")
        seen.add(speaker_id)
        if speaker_id not in ALL_SPEAKERS:
            raise ValueError(f"unexpected speaker in manifest: {speaker_id}")

        target_raw = str(speaker["target_transcript"])
        target_text = remove_whitespace(target_raw)
        if not target_text:
            raise ValueError(f"empty target transcript for {speaker_id}")
        pair = speaker.get("target_pair")
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError(f"expected N1/N2 pair for {speaker_id}")
        pair_items: list[dict[str, Any]] = []
        pair_texts: list[str] = []
        for item in pair:
            path = resolve_asset(asset_root, item)
            if not path.is_file():
                raise FileNotFoundError(path)
            actual_sha = sha256_file(path)
            expected_sha = str(item.get("raw_sha256", ""))
            if expected_sha and actual_sha != expected_sha:
                raise ValueError(f"natural-audio hash mismatch: {path}")
            pair_texts.append(str(item["transcript"]))
            pair_items.append(
                {
                    "role": str(item["role"]),
                    "utterance_id": str(item["utterance_id"]),
                    "path_relative": Path(item.get("staged_path_relative", path.name)).as_posix(),
                    "path": str(path),
                    "raw_sha256": actual_sha,
                    "duration_s": float(item["duration_s"]),
                    "transcript": str(item["transcript"]),
                }
            )
        if strip_punctuation_for_check(pair_texts[0]) != strip_punctuation_for_check(
            pair_texts[1]
        ):
            raise ValueError(f"N1/N2 transcript mismatch for {speaker_id}")
        if strip_punctuation_for_check(pair_texts[0]) != strip_punctuation_for_check(
            target_raw
        ):
            raise ValueError(f"target transcript differs from N1/N2 for {speaker_id}")

        reference = dict(speaker.get("reference") or {})
        reference_for_clone = speaker.get("reference_for_clone") or {}
        if not isinstance(reference_for_clone, dict):
            raise ValueError(f"reference_for_clone must be an object for {speaker_id}")
        reference.update(reference_for_clone)
        if not reference:
            raise ValueError(f"missing reference metadata for {speaker_id}")
        ref_path = resolve_asset(asset_root, reference)
        if not ref_path.is_file():
            raise FileNotFoundError(ref_path)
        ref_sha = sha256_file(ref_path)
        expected_ref_sha = str(reference.get("raw_sha256", ""))
        if expected_ref_sha and ref_sha != expected_ref_sha:
            raise ValueError(f"reference-audio hash mismatch for {speaker_id}: {ref_path}")

        source_parts = reference.get("transcript_source_parts")
        if source_parts:
            ref_text_raw = "".join(str(part["transcript"]) for part in source_parts)
            ref_source_ids = [str(part["utterance_id"]) for part in source_parts]
        else:
            ref_text_raw = str(reference["transcript"])
            ref_source_ids = [str(reference.get("utterance_id", ""))]
        ref_text = remove_whitespace(ref_text_raw)
        if not ref_text:
            raise ValueError(f"empty reference transcript for {speaker_id}")

        ref_wav, sample_rate = read_wave_header(ref_path)
        if sample_rate != 16000 or ref_wav <= 0:
            raise ValueError(f"invalid reference WAV format for {speaker_id}: {ref_path}")
        expected_active = reference.get("active_speech_seconds_10ms_rms_gt_minus50dBFS")

        numeric_speaker_id = int(speaker_id.removeprefix("S"))
        rows.append(
            {
                "speaker": speaker_id,
                "target_transcript_source": target_raw,
                "target_text_sent": target_text,
                "target_transcript_whitespace_rule": "remove whitespace characters only; retain punctuation/content",
                "natural_pair": pair_items,
                "reference_audio_path": str(ref_path),
                "reference_audio_path_relative": Path(
                    reference.get("staged_path_relative", ref_path.name)
                ).as_posix(),
                "reference_audio_sha256": ref_sha,
                "reference_transcript_source_parts": source_parts or [
                    {"utterance_id": ref_source_ids[0], "transcript": ref_text_raw}
                ],
                "reference_text_sent": ref_text,
                "reference_utterance_ids": ref_source_ids,
                "reference_duration_s": float(reference["duration_s"]),
                "reference_active_speech_seconds_10ms_rms_gt_minus50dBFS": expected_active,
                "take_seeds": {
                    "T1": T1_SEED_BASE + 100 * numeric_speaker_id,
                    "T2": T2_SEED_BASE + 100 * numeric_speaker_id,
                },
            }
        )

    if set(seen) != set(ALL_SPEAKERS):
        missing = sorted(set(ALL_SPEAKERS) - seen)
        raise ValueError(f"manifest does not contain all eight frozen speakers: {missing}")
    rows.sort(key=lambda row: ALL_SPEAKERS.index(row["speaker"]))
    manifest_meta = {
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "asset_root": str(asset_root.resolve()),
        "manifest_status": manifest.get("status"),
        "speaker_ids": [row["speaker"] for row in rows],
    }
    return manifest_meta, rows


def read_wave_header(path: Path) -> tuple[int, int]:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"expected mono PCM16 input: {path}")
        return handle.getnframes(), handle.getframerate()


def package_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in (
        "torch",
        "qwen-tts",
        "faster-qwen3-tts",
        "transformers",
        "accelerate",
        "torchaudio",
        "safetensors",
        "numpy",
        "soundfile",
    ):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def model_asset_inventory(model_dir: Path, use_hashes: bool) -> list[dict[str, Any]]:
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)
    files = sorted(path for path in model_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"model directory is empty: {model_dir}")
    inventory = []
    for path in files:
        relative = path.relative_to(model_dir).as_posix()
        item: dict[str, Any] = {
            "relative_path": relative,
            "size_bytes": path.stat().st_size,
            "mtime_ns": path.stat().st_mtime_ns,
        }
        if use_hashes:
            item["sha256"] = sha256_file(path)
        inventory.append(item)
    if use_hashes:
        for relative, expected in EXPECTED_MODEL_FILES.items():
            found = next((item for item in inventory if item["relative_path"] == relative), None)
            if found is None or found.get("sha256") != expected:
                raise ValueError(f"checkpoint pin mismatch for {relative}")
    return inventory


def make_plan_spec(
    manifest_meta: dict[str, Any],
    rows: list[dict[str, Any]],
    model_dir: Path,
    protocol_id: str,
    protocol_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "run_name": "local_qwen_attempt02_fp32",
        "created_by": "scripts/experiments/repeated_take_crossover_20261015/local_qwen_takes.py",
        "runner_source_sha256": sha256_file(Path(__file__).resolve()),
        "protocol_addendum": {
            "id": protocol_id,
            "sha256": protocol_sha256,
        },
        "source_manifest": manifest_meta,
        "provider": "qwen_tts_native_pytorch_local",
        "model_id": MODEL_ID,
        "model_dir": str(model_dir.resolve()),
        "model_load_kwargs": MODEL_LOAD_KWARGS,
        "generation_kwargs": GENERATION_KWARGS,
        "rng_scope": {
            "python_random": True,
            "numpy_global": True,
            "torch_cpu": True,
            "torch_cuda_all_devices_restored": True,
            "seed_formula": {
                "T1": "2026101501 + 100 * numeric speaker ID",
                "T2": "2026101502 + 100 * numeric speaker ID",
            },
            "seeds": {row["speaker"]: row["take_seeds"] for row in rows},
        },
        "package_versions": package_versions(),
        "rows": rows,
        "output_format": {
            "sample_rate_expected_hz": 24000,
            "channels": 1,
            "container": "WAV",
            "subtype": "FLOAT",
            "canonical_16k_output": "produced by the matrix runner",
        },
    }


def freeze_plan(
    output_dir: Path,
    spec: dict[str, Any],
    model_dir: Path,
) -> tuple[dict[str, Any], str]:
    plan_path = output_dir / "input_manifest.frozen.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if plan_path.exists():
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        if frozen.get("plan_spec") != spec:
            raise ValueError("output directory is frozen to a different protocol or input manifest")
        current_meta = model_asset_inventory(model_dir, use_hashes=False)
        frozen_meta = [
            {k: item[k] for k in ("relative_path", "size_bytes", "mtime_ns")}
            for item in frozen["model_assets"]
        ]
        if current_meta != frozen_meta:
            raise ValueError("model files changed after they were pinned")
        return frozen, str(frozen["plan_sha256"])

    existing = list(output_dir.iterdir())
    if existing:
        raise ValueError(f"refusing non-empty output dir without a frozen input manifest: {output_dir}")
    model_assets = model_asset_inventory(model_dir, use_hashes=True)
    frozen = {
        "plan_spec": spec,
        "plan_spec_sha256": canonical_sha256(spec),
        "model_assets": model_assets,
        "model_assets_sha256": canonical_sha256(model_assets),
        "frozen_before_tts": True,
        "frozen_at_utc": utc_now(),
    }
    frozen["plan_sha256"] = canonical_sha256(frozen)
    write_json_atomic(plan_path, frozen)
    return frozen, str(frozen["plan_sha256"])


def load_model(model_dir: Path, device: str = "cuda:0") -> tuple[Any, dict[str, Any]]:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch
    from qwen_tts import Qwen3TTSModel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing CPU fallback for the TTS run")
    if torch.cuda.device_count() < 1:
        raise RuntimeError("no CUDA device found")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    capability = torch.cuda.get_device_capability(0)
    if capability < (7, 0):
        raise RuntimeError(f"unsupported CUDA capability for the selected fp32 plan: {capability}")
    model = Qwen3TTSModel.from_pretrained(
        str(model_dir),
        device_map=device,
        torch_dtype=torch.float32,
        attn_implementation="sdpa",
        local_files_only=True,
    )
    info = {
        "device": device,
        "device_name": torch.cuda.get_device_name(0),
        "compute_capability": list(capability),
        "cuda_version": torch.version.cuda,
        "torch_version": str(torch.__version__),
        "torch_dtype": "float32",
        "attention": "sdpa",
        "flash_attention_used": False,
        "cuda_memory_total_bytes": int(torch.cuda.get_device_properties(0).total_memory),
        "cuda_memory_free_bytes_after_model_load": int(torch.cuda.mem_get_info(0)[0]),
        "cuda_memory_allocated_bytes_after_model_load": int(torch.cuda.memory_allocated(0)),
    }
    return model, info


@contextlib.contextmanager
def seeded_rng_scope(seed: int, torch: Any, cuda_devices: list[int]) -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def write_provider_wav(path: Path, audio: np.ndarray, sample_rate: int) -> dict[str, Any]:
    import soundfile as sf

    values = np.asarray(audio, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("model returned empty, non-finite, or non-mono audio")
    if sample_rate != 24000:
        raise ValueError(f"expected provider-native 24 kHz output, got {sample_rate} Hz")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.stem}.{os.getpid()}.tmp.wav")
    sf.write(str(temporary), values, sample_rate, format="WAV", subtype="FLOAT")
    decoded, decoded_sr = sf.read(str(temporary), dtype="float32", always_2d=False)
    if decoded_sr != sample_rate or decoded.shape != values.shape or not np.array_equal(decoded, values):
        temporary.unlink(missing_ok=True)
        raise ValueError("provider WAV did not round-trip the generated float32 PCM exactly")
    os.replace(temporary, path)
    pcm_bytes = np.asarray(decoded, dtype="<f4").tobytes(order="C")
    return {
        "path": str(path.resolve()),
        "raw_wav_sha256": sha256_file(path),
        "decoded_pcm_sha256": hashlib.sha256(pcm_bytes).hexdigest(),
        "sample_rate_hz": int(decoded_sr),
        "sample_count": int(decoded.size),
        "duration_s": float(decoded.size / decoded_sr),
        "channels": 1,
        "sample_format": "float32",
        "peak_abs": float(np.max(np.abs(decoded))),
        "rms": float(np.sqrt(np.mean(decoded.astype(np.float64) ** 2))),
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--speakers", nargs="+", choices=ALL_SPEAKERS, required=True)
    parser.add_argument("--takes", nargs="+", choices=("T1", "T2"), default=("T1", "T2"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    started = time.monotonic()
    process_id = os.getpid()
    output_dir = args.output_dir.resolve()
    event_path = output_dir / "events.jsonl"
    event: dict[str, Any] = {
        "event": "process_start",
        "pid": process_id,
        "ppid": os.getppid(),
        "started_at_utc": utc_now(),
        "argv": sys.argv,
        "return_code": None,
    }
    try:
        if not re.fullmatch(r"[0-9a-f]{64}", args.protocol_sha256):
            raise ValueError("--protocol-sha256 must be 64 lowercase hexadecimal characters")
        if len(set(args.speakers)) != len(args.speakers) or len(set(args.takes)) != len(args.takes):
            raise ValueError("duplicate speaker or take selection")

        manifest_meta, rows = load_and_validate_inputs(args.manifest.resolve())
        model_dir = args.model_dir.resolve()
        spec = make_plan_spec(
            manifest_meta,
            rows,
            model_dir,
            args.protocol_id,
            args.protocol_sha256,
        )
        frozen, plan_sha = freeze_plan(output_dir, spec, model_dir)
        results_path = output_dir / "results.json"
        if results_path.exists():
            results = json.loads(results_path.read_text(encoding="utf-8"))
            if results.get("plan_sha256") != plan_sha:
                raise ValueError("existing results are bound to a different frozen plan")
        else:
            results = {
                "plan_sha256": plan_sha,
                "protocol_addendum": spec["protocol_addendum"],
                "provider": spec["provider"],
                "model_id": MODEL_ID,
                "records": [],
            }
            write_json_atomic(results_path, results)

        completed = {
            (str(row.get("speaker")), str(row.get("take")))
            for row in results["records"]
            if row.get("status") == "ok"
        }
        print(
            json.dumps(
                {
                    "stage": "frozen",
                    "plan_sha256": plan_sha,
                    "speakers_requested": args.speakers,
                    "takes_requested": args.takes,
                    "checkpoint_hashes": {
                        item["relative_path"]: item.get("sha256")
                        for item in frozen["model_assets"]
                        if item["relative_path"] in EXPECTED_MODEL_FILES
                    },
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        selected = {row["speaker"]: row for row in rows}
        pending = [
            (selected[speaker], take)
            for speaker in args.speakers
            for take in args.takes
            if (speaker, take) not in completed
        ]
        if not pending:
            event.update({"event": "process_complete", "return_code": 0, "no_op": True})
            return 0

        load_started = time.monotonic()
        model, runtime = load_model(model_dir)
        runtime["model_load_seconds"] = time.monotonic() - load_started
        runtime["model_asset_pins_sha256"] = frozen["model_assets_sha256"]
        write_json_atomic(output_dir / "runtime.json", runtime)
        import torch

        cuda_devices = list(range(torch.cuda.device_count()))
        for row, take in pending:
            seed = int(row["take_seeds"][take])
            take_id = f"{row['speaker']}__{take}"
            out_path = output_dir / "provider_native_24k" / f"{take_id}.wav"
            if out_path.exists():
                raise FileExistsError(f"refusing to overwrite existing take file: {out_path}")
            record: dict[str, Any] = {
                "speaker": row["speaker"],
                "take": take,
                "take_id": take_id,
                "seed": seed,
                "status": "running",
                "pid": process_id,
                "started_at_utc": utc_now(),
                "reference_audio_path": row["reference_audio_path"],
                "reference_audio_sha256": row["reference_audio_sha256"],
                "reference_text_sent": row["reference_text_sent"],
                "reference_utterance_ids": row["reference_utterance_ids"],
                "target_text_sent": row["target_text_sent"],
                "generation_kwargs": GENERATION_KWARGS,
                "model_id": MODEL_ID,
                "model_load_kwargs": MODEL_LOAD_KWARGS,
                "return_code": None,
            }
            take_started = time.monotonic()
            results["records"].append(record)
            record_index = len(results["records"]) - 1
            write_json_atomic(results_path, results)
            try:
                with seeded_rng_scope(seed, torch, cuda_devices):
                    audio_list, sample_rate = model.generate_voice_clone(
                        text=row["target_text_sent"],
                        language="Chinese",
                        ref_audio=row["reference_audio_path"],
                        ref_text=row["reference_text_sent"],
                        **GENERATION_KWARGS,
                    )
                audio = audio_list[0] if isinstance(audio_list, (list, tuple)) else audio_list
                audio = np.asarray(audio, dtype=np.float32)
                output_info = write_provider_wav(out_path, audio, int(sample_rate))
                record.update(output_info)
                record["status"] = "ok"
                record["return_code"] = 0
            except Exception as exc:
                record["status"] = "failed"
                record["return_code"] = 1
                record["error"] = f"{type(exc).__name__}: {exc}"
                record["traceback"] = traceback.format_exc(limit=12)
                raise
            finally:
                record["elapsed_seconds"] = time.monotonic() - take_started
                record["finished_at_utc"] = utc_now()
                results["records"][record_index] = record
                write_json_atomic(results_path, results)
                append_jsonl(
                    event_path,
                    {
                        "event": "take_complete" if record["status"] == "ok" else "take_failed",
                        "pid": process_id,
                        "take_id": take_id,
                        "started_at_utc": record["started_at_utc"],
                        "finished_at_utc": record["finished_at_utc"],
                        "elapsed_seconds": record["elapsed_seconds"],
                        "return_code": record["return_code"],
                    },
                )
            print(
                json.dumps(
                    {
                        "take_id": take_id,
                        "status": record["status"],
                        "duration_s": record.get("duration_s"),
                        "elapsed_seconds": record["elapsed_seconds"],
                        "raw_wav_sha256": record.get("raw_wav_sha256"),
                        "return_code": record["return_code"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

        event.update({"event": "process_complete", "return_code": 0})
        return 0
    except Exception as exc:
        event.update(
            {
                "event": "process_failed",
                "return_code": 1,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=16),
            }
        )
        print(json.dumps(event, ensure_ascii=False), file=sys.stderr, flush=True)
        return 1
    finally:
        event["finished_at_utc"] = utc_now()
        event["elapsed_seconds"] = time.monotonic() - started
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            append_jsonl(event_path, event)
        except Exception:
            # Keep the original process error if the event sink itself fails.
            pass


if __name__ == "__main__":
    raise SystemExit(main())
