#!/usr/bin/env python3
"""Transfer the Chinese phoneme-separability analysis to LRS3 English.

This runner is deliberately a new protocol wrapper around the project's
existing numerical helpers.  It reuses the Wav2Sem-style metrics, paired
permutation/bootstrap statistics, English phone-to-viseme map, and the
per-phone KLD/AUC implementation, while rebuilding English MFA alignments for
every audio arm that does not already have a hash-verified alignment.

The run compares five paired conditions on the same 100 LRS3 utterances:

* natural
* qwen_cloud (the existing cloud Qwen output)
* qwen_local (local faster_qwen3)
* index_tts2
* cosyvoice2 (SiliconFlow FunAudioLLM/CosyVoice2-0.5B)

The command is staged so a failed MFA or model extraction never silently
turns into a uniform-span fallback::

    python scripts/experiments/lrs3_english_phoneme_transfer.py all \
        --device cuda

Individual stages are ``prepare``, ``mfa``, ``extract`` and ``metrics``.
Artifacts are written under ``runs/lrs3_english_phoneme_transfer_n100_20260921``
by default.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.experiments.lrs3_phone_rules_metrics import (  # noqa: E402
    SILENCE_LABELS,
    normalize_phone,
    pool_phone_tokens,
)
from scripts.experiments.lrs3_phone_rules_worker import (  # noqa: E402
    extract_features,
    frontend_config_from_model,
    load_ssl_bundle,
    parse_textgrid,
    read_json,
    read_pcm16,
    sha256_file,
    write_json_atomic,
)
from scripts.prepare_mdc_english_alignment import phone_to_viseme  # noqa: E402


RUN_ID = "lrs3_english_phoneme_transfer_n100_20260921"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / RUN_ID
DEFAULT_COHORT = REPO_ROOT / "runs/lrs3_local_tts_n100_20260921/00_inputs/cohort_n100.json"
OLD_FEATURE_RUN = REPO_ROOT / "runs/lrs3_phone_rules_full_20260921_proxy"
OLD_FEATURE_ROOT = OLD_FEATURE_RUN / "01_features"
OLD_COHORT = OLD_FEATURE_RUN / "00_audit/cohort.json"
SAMPLE_RATE = 16_000
CONDITIONS = ("natural", "qwen_cloud", "qwen_local", "index_tts2", "cosyvoice2")
TTS_CONDITIONS = CONDITIONS[1:]
MODELS: dict[str, dict[str, Any]] = {
    "hubert": {
        "model_name": "facebook/hubert-base-ls960",
        "processor_name": "facebook/hubert-base-ls960",
        "revision": "dba3bb02fda4248b6e082697eee756de8fe8aa8a",
        "layers": [0, 6, 11],
        "probe": False,
        "probe_layers": [],
    },
    "xlsr": {
        "model_name": "facebook/wav2vec2-large-xlsr-53",
        "processor_name": "facebook/wav2vec2-large-xlsr-53",
        "revision": "c3f9d884181a224a6ac87bf8885c84d1cff3384f",
        "layers": [10],
        "probe": False,
        "probe_layers": [],
    },
}
METRIC_LEVELS = ("phoneme", "viseme")
METRIC_NAMES = (
    "intra_class_dist",
    "inter_class_dist",
    "fisher_ratio",
    "silhouette",
    "boundary_sharpness",
    "segment_stability",
)
P_PERMUTATIONS = 2_000
BOOTSTRAP_DRAWS = 5_000
PER_PHONE_MIN_SAMPLES = 20
PER_PHONE_PCA_DIM = 30
MFA_ALIGNMENT_CONFIG = {"beam": 100, "retry_beam": 400}


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


SEP = _load_module(REPO_ROOT / "scripts/16_feature_separability.py", "lrs3_sep_metrics")
KLD = _load_module(REPO_ROOT / "scripts/38_per_phoneme_kld.py", "lrs3_kld_metrics")
DUR = _load_module(REPO_ROOT / "scripts/36_duration_jsd.py", "lrs3_duration_metrics")


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json_atomic(path, value)


def _jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{line_number}")
        rows.append(value)
    return rows


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _resolved(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_result_map(path: Path) -> dict[str, dict[str, Any]]:
    payload = read_json(path)
    results = payload.get("results") if isinstance(payload, Mapping) else None
    if isinstance(results, Mapping):
        return {str(key): dict(value) for key, value in results.items() if isinstance(value, Mapping)}
    if isinstance(results, list):
        return {str(row["sample_id"]): dict(row) for row in results if isinstance(row, Mapping) and row.get("sample_id")}
    raise ValueError(f"unsupported result manifest schema: {path}")


def _audio_record(path: Path) -> dict[str, Any]:
    pcm, meta = read_pcm16(path, sample_rate=SAMPLE_RATE)
    return {
        "path": str(path.resolve()),
        "sha256": meta["container_sha256"],
        "pcm_sha256": meta["pcm_sha256"],
        "sample_count": int(pcm.size),
        "duration_s": float(pcm.size / SAMPLE_RATE),
        "sample_rate": SAMPLE_RATE,
        "channels": 1,
        "sample_width": 2,
    }


def _condition_audio_paths() -> dict[str, dict[str, Path]]:
    """Return explicit provider paths; no path inference from position."""

    local = _load_result_map(REPO_ROOT / "runs/lrs3_local_tts_n100_20260921/01_qwen_local/tts_manifest.json")
    index = _load_result_map(REPO_ROOT / "runs/lrs3_local_tts_n100_20260921/02_index_tts2/tts_manifest.json")
    cosy = _load_result_map(REPO_ROOT / "runs/lrs3_local_tts_n100_20260921/03_siliconflow_cosyvoice2/tts_manifest.json")
    providers = {
        "qwen_local": (local, "canonical_audio"),
        "index_tts2": (index, "canonical_audio"),
        "cosyvoice2": (cosy, "canonical_audio"),
    }
    result: dict[str, dict[str, Path]] = {key: {} for key in CONDITIONS}
    cloud_meta = _load_result_map(REPO_ROOT / "runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts_meta.json")
    for sid, row in cloud_meta.items():
        if row.get("canonical_16k_audio"):
            result["qwen_cloud"][sid] = _resolved(str(row["canonical_16k_audio"]))
    for condition, (rows, key) in providers.items():
        for sid, row in rows.items():
            if row.get(key):
                result[condition][sid] = _resolved(str(row[key]))
    return result


def _build_manifest(run_dir: Path, cohort_path: Path) -> dict[str, Any]:
    cohort = read_json(cohort_path)
    records = cohort.get("records") if isinstance(cohort, Mapping) else None
    if not isinstance(records, list) or len(records) != 100:
        raise ValueError(f"expected exactly 100 cohort records: {cohort_path}")
    provider_paths = _condition_audio_paths()
    old_textgrid_root = REPO_ROOT / "runs/lrs3_qwen_cloud_n500_20260817/03_rhythm_data_mfa_fixed/mfa"
    output_records: list[dict[str, Any]] = []
    for raw in records:
        sid = str(raw["sample_id"])
        stem = sid.removeprefix("lrs3_")
        natural_path = _resolved(str(raw["natural_audio_path"]))
        paths: dict[str, Path] = {"natural": natural_path}
        for condition in TTS_CONDITIONS:
            path = provider_paths[condition].get(sid)
            if path is None:
                if condition == "qwen_cloud":
                    path = REPO_ROOT / "runs/lrs3_qwen_cloud_n500_20260817/02_tts/tts" / f"{sid}.wav"
                else:
                    raise FileNotFoundError(f"{condition} audio missing for {sid}")
            paths[condition] = path
        arms: dict[str, dict[str, Any]] = {}
        for condition, audio_path in paths.items():
            audio = _audio_record(audio_path)
            if condition == "natural":
                textgrid = old_textgrid_root / "natural" / f"{sid}.TextGrid"
                alignment_source = "reused_existing_lrs3_mfa_fixed"
            elif condition == "qwen_cloud":
                textgrid = old_textgrid_root / "tts" / f"{sid}.TextGrid"
                alignment_source = "reused_existing_lrs3_mfa_fixed_cloud_qwen"
            else:
                textgrid = run_dir / "01_mfa" / "aligned" / condition / f"{sid}.TextGrid"
                alignment_source = "independent_english_mfa_required"
            arms[condition] = {
                "audio": audio,
                "textgrid": str(textgrid.resolve()),
                "alignment_source": alignment_source,
            }
        output_records.append(
            {
                "sample_id": sid,
                "source_group": str(raw.get("source_group", stem.split("_")[0])),
                "selection_index": int(raw.get("selection_index", len(output_records))),
                "transcript": str(raw.get("transcript") or raw.get("tts_transcript") or "").strip(),
                "transcript_sha256": str(raw.get("transcript_sha256", "")),
                "speaker_id": raw.get("speaker_id"),
                "arms": arms,
            }
        )
    manifest = {
        "schema_version": 1,
        "protocol": "lrs3_english_phoneme_separability_condition_transfer_v1",
        "dataset": "LRS3 English",
        "sample_rate": SAMPLE_RATE,
        "sample_count": len(output_records),
        "conditions": list(CONDITIONS),
        "cohort_path": str(cohort_path.resolve()),
        "cohort_sha256": _sha256(cohort_path),
        "selection_policy": "exactly the existing first-100 local-TTS/IndexTTS/CosyVoice2 cohort order",
        "phone_label_policy": "NFC plus surrounding whitespace only; English MFA IPA labels retain diacritics",
        "viseme_map_policy": "reuse scripts/prepare_mdc_english_alignment.py English phone_to_viseme map",
        "records": output_records,
    }
    _json(run_dir / "00_inputs/manifest.json", manifest)
    return manifest


def _prepare_mfa_corpus(manifest: Mapping[str, Any], run_dir: Path, condition: str) -> dict[str, Any]:
    corpus = run_dir / "01_mfa" / "corpus" / condition
    corpus.mkdir(parents=True, exist_ok=True)
    rows = []
    for record in manifest["records"]:
        sid = str(record["sample_id"])
        audio = Path(record["arms"][condition]["audio"]["path"])
        wav = corpus / f"{sid}.wav"
        lab = corpus / f"{sid}.lab"
        if wav.exists() or wav.is_symlink():
            wav.unlink()
        wav.symlink_to(audio)
        lab.write_text(str(record["transcript"]) + "\n", encoding="utf-8")
        rows.append({"sample_id": sid, "audio": str(audio), "wav": str(wav), "lab": str(lab)})
    result = {
        "condition": condition,
        "corpus_dir": str(corpus.resolve()),
        "record_count": len(rows),
        "records": rows,
    }
    _json(run_dir / "01_mfa" / f"corpus_{condition}.json", result)
    return result


def _run_mfa_for_condition(run_dir: Path, condition: str) -> dict[str, Any]:
    corpus = run_dir / "01_mfa" / "corpus" / condition
    aligned = run_dir / "01_mfa" / "aligned" / condition
    aligned.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "01_mfa" / "mfa_alignment_config.yaml"
    config_path.write_text(
        "beam: 100\nretry_beam: 400\n",
        encoding="utf-8",
    )
    command = [
        "mfa",
        "align",
        str(corpus),
        "english_us_mfa",
        "english_mfa",
        str(aligned),
        "--config_path",
        str(config_path),
        "--clean",
        "--overwrite",
    ]
    log_path = run_dir / "01_mfa" / f"mfa_{condition}.log"
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=str(REPO_ROOT), stdout=handle, stderr=subprocess.STDOUT, check=False)
    result = {
        "condition": condition,
        "command": command,
        "config": MFA_ALIGNMENT_CONFIG,
        "log_path": str(log_path.resolve()),
        "returncode": int(completed.returncode),
        "aligned_dir": str(aligned.resolve()),
        "textgrid_count": len(list(aligned.glob("*.TextGrid"))),
    }
    _json(run_dir / "01_mfa" / f"mfa_{condition}.json", result)
    if completed.returncode != 0:
        raise RuntimeError(f"MFA failed for {condition}; see {log_path}")
    result["status"] = "complete" if result["textgrid_count"] == 100 else "partial"
    result["missing_sample_ids"] = sorted(
        path.stem
        for path in (run_dir / "01_mfa" / "corpus" / condition).glob("*.lab")
        if not (aligned / f"{path.stem}.TextGrid").is_file()
    )
    _json(run_dir / "01_mfa" / f"mfa_{condition}.json", result)
    return result


def _audit_alignments(manifest: Mapping[str, Any], run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for record in manifest["records"]:
        for condition in CONDITIONS:
            arm = record["arms"][condition]
            path = Path(str(arm["textgrid"]))
            row = {
                "sample_id": record["sample_id"],
                "condition": condition,
                "audio_sha256": arm["audio"]["sha256"],
                "textgrid": str(path),
                "alignment_source": arm["alignment_source"],
                "exists": path.is_file(),
            }
            if not path.is_file():
                failures.append({**row, "reason": "MISSING_TEXTGRID"})
                rows.append(row)
                continue
            try:
                tokens = parse_textgrid(path)
            except Exception as exc:  # preserve all per-arm failures
                failures.append({**row, "reason": "INVALID_TEXTGRID", "error": str(exc)})
                rows.append(row)
                continue
            speech = [token for token in tokens if bool(token.get("speech", False))]
            row.update(
                {
                    "textgrid_sha256": _sha256(path),
                    "token_count": len(tokens),
                    "speech_token_count": len(speech),
                    "phone_labels": sorted({normalize_phone(token.get("label", "")) for token in speech}),
                }
            )
            rows.append(row)
    condition_counts = {
        condition: sum(1 for row in rows if row["condition"] == condition and row.get("exists"))
        for condition in CONDITIONS
    }
    result = {
        "schema_version": 1,
        "status": "complete" if not failures else "partial",
        "expected": 500,
        "completed": len(rows) - len(failures),
        "condition_counts": condition_counts,
        "failures": failures,
        "records": rows,
    }
    _json(run_dir / "01_mfa/alignment_manifest.json", result)
    return result


def prepare(run_dir: Path, cohort_path: Path, *, run_mfa: bool = True) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "00_inputs/manifest.json"
    manifest = read_json(manifest_path) if manifest_path.is_file() else _build_manifest(run_dir, cohort_path)
    for condition in ("qwen_local", "index_tts2", "cosyvoice2"):
        _prepare_mfa_corpus(manifest, run_dir, condition)
        if run_mfa:
            _run_mfa_for_condition(run_dir, condition)
    if run_mfa:
        return _audit_alignments(manifest, run_dir)
    result = {
        "schema_version": 1,
        "status": "staged",
        "conditions": ["qwen_local", "index_tts2", "cosyvoice2"],
        "record_count": len(manifest["records"]),
        "next_stage": "mfa",
    }
    _json(run_dir / "01_mfa/staging_manifest.json", result)
    return result


def _old_audio_hashes() -> dict[tuple[str, str], str]:
    if not OLD_COHORT.is_file():
        return {}
    payload = read_json(OLD_COHORT)
    result: dict[tuple[str, str], str] = {}
    for row in payload.get("records", []):
        sid = str(row.get("sample_id", ""))
        for condition, old_key in (("natural", "natural"), ("qwen_cloud", "tts")):
            audio = row.get(old_key, {}).get("audio", {})
            if sid and audio.get("container_sha256"):
                result[(sid, condition)] = str(audio["container_sha256"])
    return result


def _old_feature_path(model: str, sample_id: str, condition: str) -> Path:
    old_condition = "natural" if condition == "natural" else "tts"
    return OLD_FEATURE_ROOT / model / f"{sample_id}__{old_condition}.npz"


def _load_reused_features(model: str, sample_id: str, condition: str, audio_sha256: str, layers: Sequence[int]) -> tuple[dict[int, np.ndarray], np.ndarray] | None:
    old_hashes = _old_audio_hashes()
    if old_hashes.get((sample_id, condition)) != audio_sha256:
        return None
    path = _old_feature_path(model, sample_id, condition)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as archive:
        if any(f"layer_{layer}" not in archive.files for layer in layers) or "frame_times" not in archive.files:
            return None
        selected = {int(layer): np.asarray(archive[f"layer_{layer}"], dtype=np.float32) for layer in layers}
        frame_times = np.asarray(archive["frame_times"], dtype=np.float64)
    return selected, frame_times


def _phone_tokens(path: Path) -> list[dict[str, Any]]:
    tokens = parse_textgrid(path)
    output: list[dict[str, Any]] = []
    for token in tokens:
        row = dict(token)
        raw_label = normalize_phone(token.get("label", token.get("token", "")))
        row["label"] = raw_label
        row["token"] = raw_label
        row["viseme"] = phone_to_viseme(raw_label)
        row["speech"] = bool(token.get("speech", raw_label.lower() not in {"", "sil", "sp", "spn", "pau", "noise"}))
        row["silence"] = not row["speech"]
        output.append(row)
    return output


def _boundary_metrics(frame_times: np.ndarray, frame_embeddings: np.ndarray, tokens: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    boundaries = sorted(
        {
            float(token["end_s"])
            for token in tokens
            if bool(token.get("speech", False)) and float(token["end_s"]) < float(frame_times[-1])
        }
    ) if len(frame_times) else []
    boundary, stability = SEP.boundary_sharpness(frame_times, frame_embeddings, boundaries)
    return _safe_float(boundary), _safe_float(stability)


def _extract_model(manifest: Mapping[str, Any], run_dir: Path, model_key: str, device: str) -> dict[str, Any]:
    config = MODELS[model_key]
    layers = [int(value) for value in config["layers"]]
    bundle = load_ssl_bundle(
        str(config["model_name"]),
        processor_name=str(config["processor_name"]),
        revision=str(config["revision"]),
        device=device,
    )
    model_dir = run_dir / "02_embeddings" / model_key
    model_dir.mkdir(parents=True, exist_ok=True)
    vector_lists: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    token_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    reused_counts: dict[str, int] = defaultdict(int)
    for record_index, record in enumerate(manifest["records"], 1):
        for condition in CONDITIONS:
            sid = str(record["sample_id"])
            arm = record["arms"][condition]
            try:
                audio_path = Path(str(arm["audio"]["path"]))
                pcm, audio_meta = read_pcm16(audio_path, sample_rate=SAMPLE_RATE)
                if audio_meta["container_sha256"] != arm["audio"]["sha256"]:
                    raise ValueError("audio hash changed after input manifest was frozen")
                tokens = _phone_tokens(Path(str(arm["textgrid"])))
                reused = _load_reused_features(model_key, sid, condition, str(arm["audio"]["sha256"]), layers)
                if reused is not None:
                    selected, frame_times = reused
                    feature_source = "reused_hash_verified_lrs3_phone_rules_cache"
                    reused_counts[condition] += 1
                else:
                    selected, frame_times, feature_meta = extract_features(
                        bundle["model"],
                        bundle["processor"],
                        pcm.astype(np.float32) / 32768.0,
                        SAMPLE_RATE,
                        layers,
                        device=device,
                        frontend=bundle["frontend"],
                    )
                    feature_source = "fresh_frozen_ssl_extraction"
                for layer in layers:
                    embeddings = np.asarray(selected[layer], dtype=np.float32)
                    pooled = pool_phone_tokens(
                        embeddings,
                        frame_times,
                        tokens,
                        sample_id=sid,
                        source_group=str(record["source_group"]),
                        condition=condition,
                    )
                    base_index = len(vector_lists[layer])
                    for token in pooled:
                        token_index = int(token["token_index"])
                        source_token = tokens[token_index]
                        vector = token.get("embedding")
                        row = {
                            "sample_id": sid,
                            "source_group": str(record["source_group"]),
                            "condition": condition,
                            "layer": int(layer),
                            "token_index": token_index,
                            "label": normalize_phone(token.get("label", "")),
                            "viseme": str(source_token.get("viseme", "other")),
                            "start_s": float(token["start_s"]),
                            "end_s": float(token["end_s"]),
                            "duration_s": float(token["duration_s"]),
                            "speech": bool(token.get("speech", False)),
                            "valid": bool(token.get("valid", False)),
                            "frame_count": len(token.get("frame_indices", [])),
                            "vector_index": None,
                        }
                        if vector is not None and bool(token.get("valid", False)):
                            row["vector_index"] = base_index
                            vector_lists[layer].append(np.asarray(vector, dtype=np.float32))
                            base_index += 1
                        token_rows.append(row)
                    boundary, stability = _boundary_metrics(frame_times, embeddings, tokens)
                    sample_rows.append(
                        {
                            "sample_id": sid,
                            "source_group": str(record["source_group"]),
                            "condition": condition,
                            "model": model_key,
                            "layer": int(layer),
                            "audio_sha256": str(arm["audio"]["sha256"]),
                            "feature_source": feature_source,
                            "frame_count": int(embeddings.shape[0]),
                            "duration_s": float(audio_meta["duration_s"]),
                            "speech_token_count": int(sum(bool(token.get("speech", False)) for token in pooled)),
                            "valid_token_count": int(sum(bool(token.get("valid", False)) for token in pooled)),
                            "boundary_sharpness": boundary,
                            "segment_stability": stability,
                        }
                    )
            except Exception as exc:  # keep audit trail and continue other pairs
                failures.append({"sample_id": sid, "condition": condition, "model": model_key, "error": str(exc)})
        if record_index % 10 == 0:
            print(f"[{model_key}] processed {record_index}/{len(manifest['records'])} sample pairs", flush=True)
    vector_payload: dict[str, np.ndarray] = {}
    vector_counts: dict[str, int] = {}
    for layer in layers:
        if vector_lists[layer]:
            matrix = np.stack(vector_lists[layer]).astype(np.float32)
        else:
            matrix = np.empty((0, int(bundle["hidden_size"])), dtype=np.float32)
        vector_payload[f"layer_{layer}"] = matrix
        vector_counts[str(layer)] = int(matrix.shape[0])
    np.savez_compressed(model_dir / "pooled_vectors.npz", **vector_payload)
    _jsonl(model_dir / "token_records.jsonl", token_rows)
    _jsonl(model_dir / "sample_metrics.jsonl", sample_rows)
    meta = {
        "schema_version": 1,
        "model_key": model_key,
        "model_name": bundle["model_name"],
        "processor_name": bundle["processor_name"],
        "revision": config["revision"],
        "layers": layers,
        "device": device,
        "frontend": bundle["frontend"],
        "frame_time_convention": "frontend_receptive_field_centers",
        "pooling": "mean_pool_half_open_mfa_phone_span_then_l2_normalize",
        "token_record_count": len(token_rows),
        "sample_metric_count": len(sample_rows),
        "vector_counts": vector_counts,
        "reused_audio_counts": dict(reused_counts),
        "failures": failures,
    }
    _json(model_dir / "feature_meta.json", meta)
    return meta


def extract(run_dir: Path, *, device: str = "cuda") -> dict[str, Any]:
    manifest = read_json(run_dir / "00_inputs/manifest.json")
    alignment = read_json(run_dir / "01_mfa/alignment_manifest.json")
    if alignment.get("status") not in {"complete", "partial"}:
        raise RuntimeError("extract requires a successful alignment audit")
    summaries = {}
    for model_key in MODELS:
        summaries[model_key] = _extract_model(manifest, run_dir, model_key, device)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    result = {
        "schema_version": 1,
        "status": "complete" if not any(summary.get("failures") for summary in summaries.values()) else "partial",
        "models": summaries,
    }
    _json(run_dir / "02_embeddings/summary.json", result)
    return result


def _load_store(run_dir: Path, model_key: str) -> tuple[dict[int, np.ndarray], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = run_dir / "02_embeddings" / model_key
    meta = read_json(root / "feature_meta.json")
    with np.load(root / "pooled_vectors.npz", allow_pickle=False) as archive:
        vectors = {int(key.removeprefix("layer_")): np.asarray(archive[key], dtype=np.float32) for key in archive.files}
    tokens = _read_jsonl(root / "token_records.jsonl")
    samples = _read_jsonl(root / "sample_metrics.jsonl")
    for row in tokens:
        index = row.get("vector_index")
        layer = int(row["layer"])
        if index is not None:
            row["embedding"] = vectors[layer][int(index)]
        else:
            row["embedding"] = None
    return vectors, tokens, samples, meta


def _pool_metric(embeddings: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, do_probe: bool, boundary: float | None, stability: float | None) -> dict[str, Any]:
    if embeddings.size == 0:
        return {name: None for name in (*METRIC_NAMES, "probe_accuracy", "probe_f1_macro", "probe_f1_weighted")}
    result = {
        "intra_class_dist": _safe_float(SEP.intra_class_variance(embeddings, labels)),
        "inter_class_dist": _safe_float(SEP.inter_class_separation(embeddings, labels)),
        "fisher_ratio": _safe_float(SEP.fisher_ratio(embeddings, labels)),
        "silhouette": _safe_float(SEP._silhouette_cosine(embeddings, labels)),
        "boundary_sharpness": boundary,
        "segment_stability": stability,
    }
    if do_probe:
        probe = SEP.linear_probe_cv(embeddings, labels, groups, cv=5)
        result.update(
            {
                "probe_accuracy": _safe_float(probe["accuracy"]),
                "probe_f1_macro": _safe_float(probe["f1_macro"]),
                "probe_f1_weighted": _safe_float(probe["f1_weighted"]),
            }
        )
    else:
        result.update({"probe_accuracy": None, "probe_f1_macro": None, "probe_f1_weighted": None})
    return result


def _tokens_for_level(rows: Sequence[Mapping[str, Any]], level: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vectors: list[np.ndarray] = []
    labels: list[str] = []
    groups: list[str] = []
    for row in rows:
        if not bool(row.get("speech", False)) or row.get("embedding") is None:
            continue
        label = normalize_phone(row.get("label", "")) if level == "phoneme" else str(row.get("viseme", "other"))
        if not label or label.lower() in SILENCE_LABELS:
            continue
        vectors.append(np.asarray(row["embedding"], dtype=np.float32))
        labels.append(label)
        groups.append(str(row["sample_id"]))
    if not vectors:
        return np.empty((0, 0), dtype=np.float32), np.asarray([], dtype=str), np.asarray([], dtype=object)
    return np.stack(vectors), np.asarray(labels, dtype=str), np.asarray(groups, dtype=object)


def _sample_metric_rows(tokens: Sequence[Mapping[str, Any]], samples: Sequence[Mapping[str, Any]], model: str, layer: int, level: str) -> dict[tuple[str, str], dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in tokens:
        if int(row["layer"]) == layer:
            grouped[(str(row["condition"]), str(row["sample_id"]))].append(row)
    sample_scalars = {
        (str(row["condition"]), str(row["sample_id"])): row
        for row in samples
        if str(row.get("model")) == model and int(row["layer"]) == layer
    }
    result: dict[tuple[str, str], dict[str, Any]] = {}
    for key, rows in grouped.items():
        embeddings, labels, groups = _tokens_for_level(rows, level)
        scalar = sample_scalars.get(key, {})
        metric = _pool_metric(
            embeddings,
            labels,
            groups,
            do_probe=False,
            boundary=_safe_float(scalar.get("boundary_sharpness")),
            stability=_safe_float(scalar.get("segment_stability")),
        )
        result[key] = metric
    return result


def _paired_comparison(natural: Mapping[str, Any], candidate: Mapping[str, Any], *, seed: int) -> dict[str, Any] | None:
    common = sorted(set(natural) & set(candidate))
    rows: list[dict[str, Any]] = []
    for sid in common:
        nrow, trow = natural[sid], candidate[sid]
        for metric in METRIC_NAMES:
            nv, tv = nrow.get(metric), trow.get(metric)
            if nv is None or tv is None or not math.isfinite(float(nv)) or not math.isfinite(float(tv)):
                continue
            rows.append({"sample_id": sid, "metric": metric, "natural": float(nv), "tts": float(tv)})
    return {"common_sample_count": len(common), "rows": rows} if common else None


def _compare_metric_values(per_sample: Mapping[tuple[str, str], Mapping[str, Any]], condition: str, model: str, layer: int, level: str) -> list[dict[str, Any]]:
    natural = {sid: value for (cond, sid), value in per_sample.items() if cond == "natural"}
    candidate = {sid: value for (cond, sid), value in per_sample.items() if cond == condition}
    common = sorted(set(natural) & set(candidate))
    comparisons: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(METRIC_NAMES):
        pairs = [
            (float(candidate[sid][metric]), float(natural[sid][metric]))
            for sid in common
            if candidate[sid].get(metric) is not None
            and natural[sid].get(metric) is not None
            and math.isfinite(float(candidate[sid][metric]))
            and math.isfinite(float(natural[sid][metric]))
        ]
        if len(pairs) < 3:
            continue
        tts = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        nat = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
        seed = int(20260921 + metric_index * 101 + layer * 1009 + sum(ord(char) for char in condition) + sum(ord(char) for char in model) + sum(ord(char) for char in level))
        p_value, observed, _ = SEP.paired_permutation_test(tts, nat, n_permutations=P_PERMUTATIONS, random_seed=seed)
        low, high, mean_diff = SEP.bootstrap_paired_ci(tts, nat, n_bootstrap=BOOTSTRAP_DRAWS, random_seed=seed)
        d_value = SEP.cohens_d_paired(tts, nat)
        comparisons.append(
            {
                "model": model,
                "layer": layer,
                "level": level,
                "condition": condition,
                "metric": metric,
                "direction": "tts_minus_natural",
                "natural_mean": float(np.mean(nat)),
                "tts_mean": float(np.mean(tts)),
                "delta": float(mean_diff if math.isfinite(mean_diff) else observed),
                "ci_95": [float(low), float(high)],
                "p_perm": float(p_value),
                "cohens_d": _safe_float(d_value),
                "n_pairs": len(pairs),
                "common_candidate_pairs": len(common),
            }
        )
    return comparisons


def _duration_jsd(manifest: Mapping[str, Any]) -> dict[str, Any]:
    alignments: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in manifest["records"]:
        for condition in CONDITIONS:
            path = Path(record["arms"][condition]["textgrid"])
            if path.is_file():
                alignments[(condition, str(record["sample_id"]))] = _phone_tokens(path)
    output: dict[str, Any] = {}
    for condition in TTS_CONDITIONS:
        phone_dur: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"natural": [], "tts": []})
        for record in manifest["records"]:
            sid = str(record["sample_id"])
            if ("natural", sid) not in alignments or (condition, sid) not in alignments:
                continue
            sides = {key: [token for token in alignments[(key, sid)] if bool(token.get("speech", False))] for key in ("natural", condition)}
            for side, key in (("natural", "natural"), ("tts", condition)):
                phones = sides[key]
                durations = np.asarray([float(token["end_s"]) - float(token["start_s"]) for token in phones], dtype=np.float64)
                if durations.size == 0 or float(durations.mean()) <= 0:
                    continue
                for token, duration in zip(phones, durations, strict=True):
                    phone_dur[normalize_phone(token["label"])][side].append(float(duration / durations.mean()))
        per_phone, summary = DUR._compute_per_phone_jsd(
            phone_dur,
            min_phone_samples=PER_PHONE_MIN_SAMPLES,
            n_bins=6,
            n_perm=P_PERMUTATIONS,
            label=f"natural_vs_{condition}_rate_normalized",
        )
        output[condition] = {"summary": summary, "per_phone": per_phone}
    return output


def _per_phone_kld(tokens: Sequence[Mapping[str, Any]], *, model: str, layer: int) -> dict[str, Any]:
    by_condition: dict[str, dict[str, dict[str, list[Any]]]] = defaultdict(lambda: defaultdict(lambda: {"vecs": [], "utterances": []}))
    for row in tokens:
        if int(row["layer"]) != layer or row.get("embedding") is None or not bool(row.get("speech", False)):
            continue
        label = normalize_phone(row.get("label", ""))
        if not label:
            continue
        condition = str(row["condition"])
        by_condition[condition][label]["vecs"].append(np.asarray(row["embedding"], dtype=np.float32))
        by_condition[condition][label]["utterances"].append(str(row["sample_id"]))
    results: dict[str, Any] = {}
    for condition in TTS_CONDITIONS:
        classes: dict[str, Any] = {}
        excluded: dict[str, str] = {}
        for label in sorted(set(by_condition["natural"]) | set(by_condition[condition])):
            natural = by_condition["natural"].get(label)
            candidate = by_condition[condition].get(label)
            if not natural or not candidate:
                excluded[label] = "missing_condition_support"
                continue
            natural_raw, candidate_raw = len(natural["vecs"]), len(candidate["vecs"])
            if natural_raw < PER_PHONE_MIN_SAMPLES or candidate_raw < PER_PHONE_MIN_SAMPLES:
                excluded[label] = "raw_support_below_threshold"
                continue
            nat_by_id: dict[str, list[np.ndarray]] = defaultdict(list)
            cand_by_id: dict[str, list[np.ndarray]] = defaultdict(list)
            for vector, sid in zip(natural["vecs"], natural["utterances"], strict=True):
                nat_by_id[sid].append(vector)
            for vector, sid in zip(candidate["vecs"], candidate["utterances"], strict=True):
                cand_by_id[sid].append(vector)
            x_list: list[np.ndarray] = []
            y_list: list[np.ndarray] = []
            groups: list[str] = []
            for sid in sorted(set(nat_by_id) & set(cand_by_id)):
                count = min(len(nat_by_id[sid]), len(cand_by_id[sid]))
                x_list.extend(nat_by_id[sid][:count])
                y_list.extend(cand_by_id[sid][:count])
                groups.extend([sid] * count)
            if len(x_list) < PER_PHONE_MIN_SAMPLES:
                excluded[label] = "matched_support_below_threshold"
                continue
            x, y = np.stack(x_list), np.stack(y_list)
            kld = KLD._gaussian_kld_sym(x, y, PER_PHONE_PCA_DIM)
            grouped = KLD._grouped_binary_metrics(x, y, np.asarray(groups + groups, dtype=object), PER_PHONE_PCA_DIM)
            if kld is None or grouped.get("auc") is None:
                excluded[label] = "nonfinite_kld_or_auc"
                continue
            classes[label] = {
                "natural_raw": natural_raw,
                "tts_raw": candidate_raw,
                "matched": int(len(x)),
                "n_utterances": int(len(set(groups))),
                "kld_sym": float(kld),
                "accuracy": _safe_float(grouped.get("accuracy")),
                "auc": _safe_float(grouped.get("auc")),
                "accuracy_std": _safe_float(grouped.get("accuracy_std")),
                "auc_std": _safe_float(grouped.get("auc_std")),
            }
        klds = [row["kld_sym"] for row in classes.values()]
        aucs = [row["auc"] for row in classes.values()]
        results[condition] = {
            "model": model,
            "layer": layer,
            "min_samples": PER_PHONE_MIN_SAMPLES,
            "pca_dim": PER_PHONE_PCA_DIM,
            "n_classes_tested": len(classes),
            "n_classes_excluded": len(excluded),
            "mean_kld_sym": float(np.mean(klds)) if klds else None,
            "median_kld_sym": float(np.median(klds)) if klds else None,
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "classes": classes,
            "exclusions": excluded,
        }
    return results


def metrics(run_dir: Path) -> dict[str, Any]:
    manifest = read_json(run_dir / "00_inputs/manifest.json")
    all_results: list[dict[str, Any]] = []
    comparisons: list[dict[str, Any]] = []
    per_phone: dict[str, Any] = {}
    duration = _duration_jsd(manifest)
    support: dict[str, Any] = {}
    model_metadata: dict[str, Any] = {}
    for model_key in MODELS:
        _, token_rows, sample_rows, model_meta = _load_store(run_dir, model_key)
        model_metadata[model_key] = model_meta
        for layer in MODELS[model_key]["layers"]:
            layer_sample_rows = [row for row in sample_rows if int(row["layer"]) == int(layer)]
            for level in METRIC_LEVELS:
                per_condition: dict[str, Any] = {}
                for condition in CONDITIONS:
                    rows = [row for row in token_rows if int(row["layer"]) == int(layer) and str(row["condition"]) == condition]
                    embeddings, labels, groups = _tokens_for_level(rows, level)
                    scalar = [row for row in layer_sample_rows if str(row["condition"]) == condition]
                    boundary = [row["boundary_sharpness"] for row in scalar if row.get("boundary_sharpness") is not None]
                    stability = [row["segment_stability"] for row in scalar if row.get("segment_stability") is not None]
                    pooled = _pool_metric(
                        embeddings,
                        labels,
                        groups,
                        do_probe=bool(
                            MODELS[model_key]["probe"]
                            and int(layer) in set(MODELS[model_key].get("probe_layers", []))
                        ),
                        boundary=float(np.mean(boundary)) if boundary else None,
                        stability=float(np.mean(stability)) if stability else None,
                    )
                    per_condition[condition] = {
                        **pooled,
                        "n_token_vectors": int(embeddings.shape[0]),
                        "n_utterances": int(len(set(groups.tolist()))) if groups.size else 0,
                        "n_classes": int(len(set(labels.tolist()))) if labels.size else 0,
                        "class_counts": {str(label): int(count) for label, count in zip(*np.unique(labels, return_counts=True))} if labels.size else {},
                        "boundary_sample_count": len(boundary),
                        "stability_sample_count": len(stability),
                    }
                    support[f"{model_key}_{layer}_{level}_{condition}"] = per_condition[condition]
                all_results.append({"model": model_key, "layer": int(layer), "level": level, "conditions": per_condition})
                per_sample = _sample_metric_rows(token_rows, sample_rows, model_key, int(layer), level)
                for condition in TTS_CONDITIONS:
                    comparisons.extend(_compare_metric_values(per_sample, condition, model_key, int(layer), level))
            if model_key == "hubert" and int(layer) == 6:
                per_phone[f"{model_key}_layer_{layer}"] = _per_phone_kld(token_rows, model=model_key, layer=int(layer))
    finite_p = [float(row["p_perm"]) for row in comparisons if row.get("p_perm") is not None and math.isfinite(float(row["p_perm"]))]
    q_values = SEP.fdr_bh_correction(np.asarray(finite_p, dtype=np.float64)) if finite_p else np.asarray([], dtype=np.float64)
    q_index = 0
    for row in comparisons:
        if row.get("p_perm") is not None and math.isfinite(float(row["p_perm"])):
            row["q_fdr_global"] = float(q_values[q_index])
            row["significant_fdr"] = bool(q_values[q_index] < 0.05)
            q_index += 1
        else:
            row["q_fdr_global"] = None
            row["significant_fdr"] = False
    result = {
        "schema_version": 1,
        "protocol": "lrs3_english_phoneme_separability_condition_transfer_v1",
        "scope": "100 paired LRS3 English utterances; pooled separability plus paired per-utterance deltas",
        "directions": {
            "intra_class_dist": "lower_is_more_compact",
            "inter_class_dist": "higher_is_more_separated",
            "fisher_ratio": "higher_is_more_separated",
            "silhouette": "higher_is_more_separated",
            "probe_accuracy": "higher_is_more_separable",
            "boundary_sharpness": "higher_boundary_change",
            "segment_stability": "lower_within_segment_change",
            "comparison_delta": "tts_minus_natural",
        },
        "conditions": list(CONDITIONS),
        "models": model_metadata,
        "pooled_results": all_results,
        "support": support,
        "comparisons": comparisons,
        "duration_jsd": duration,
        "per_phone_kld_auc": per_phone,
        "statistics": {
            "paired_permutation_draws": P_PERMUTATIONS,
            "paired_bootstrap_draws": BOOTSTRAP_DRAWS,
            "fdr_family": "all_phoneme_and_viseme_paired_metric_comparisons",
            "fdr_test_count": len(finite_p),
            "per_phone_min_samples": PER_PHONE_MIN_SAMPLES,
            "per_phone_matching": "same exact phone label; occurrence order within paired sample; each arm own MFA",
        },
        "linear_probe": {
            "status": "not_run",
            "reason": "reused full-token logistic probe is prohibitively expensive at 768 dimensions; all distance, boundary, stability, duration, KLD and AUC metrics are retained",
        },
        "claim_boundary": "representation-level metrics only; no direct SyncNet/TFG causality claim",
    }
    _json(run_dir / "03_metrics/metrics.json", result)
    _write_report(run_dir, result)
    return result


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.4f}" if isinstance(value, (int, float)) else str(value)


def _write_report(run_dir: Path, result: Mapping[str, Any]) -> None:
    lines = [
        "# LRS3 English TTS 音素可区分度条件迁移（n=100）",
        "",
        "本报告把 AISHELL-1 中文 n=100 实验的指标与配对统计迁移到 LRS3 English；不迁移中文数值、最佳层位、音素规则或 TextGrid。五臂均使用相同 sample_id/transcript；natural/cloud-Qwen 仅复用 hash 可核验的既有 MFA/SSL 资产，local-Qwen、IndexTTS-2、CosyVoice2 使用各自独立英文 MFA。",
        "",
        "## 协议边界",
        "",
        "- phoneme label 只做 NFC/空白规范化，保留英语 MFA IPA 的长度、送气、腭化等符号；viseme 使用既有 MDC English map。",
        "- pooled metrics 是全 cohort 操作性表征指标；paired delta 是逐 utterance 的 TTS − natural，含 95% bootstrap CI、paired sign-flip permutation p 和全比较 BH-FDR。历史 logistic probe 在本次 768 维全 token 迁移中不运行，JSON 保留空字段以兼容旧 schema。",
        "- per-phone KLD/AUC 每个 TTS arm 都用自己的 MFA span，并按 paired utterance 内同标签 occurrence 顺序匹配；support 不足的 phone 显式排除。",
        "- LRS3 当前本地资产没有可用的完整 speaker_id，因此不作 speaker-controlled 结论；这些指标也不等同于 Sync-C 或 TFG 因果收益。",
        "",
        "## HuBERT layer 6 pooled phoneme metrics",
        "",
        "|metric|natural|qwen_cloud|qwen_local|index_tts2|cosyvoice2|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    target = next((row for row in result["pooled_results"] if row["model"] == "hubert" and row["layer"] == 6 and row["level"] == "phoneme"), None)
    if target:
        for metric in ("intra_class_dist", "inter_class_dist", "fisher_ratio", "silhouette", "boundary_sharpness", "segment_stability"):
            values = [metric, *[_fmt(target["conditions"].get(condition, {}).get(metric)) for condition in CONDITIONS]]
            lines.append("|" + "|".join(values) + "|")
    lines += ["", "## HuBERT layer 6 paired deltas", "", "|level|condition|metric|Δ TTS−natural|95% CI|p|q(FDR)|n|", "|---|---|---|---:|---|---:|---:|---:|"]
    for row in result["comparisons"]:
        if row["model"] == "hubert" and row["layer"] == 6:
            lines.append(f"|{row['level']}|{row['condition']}|{row['metric']}|{row['delta']:+.4f}|[{row['ci_95'][0]:+.4f}, {row['ci_95'][1]:+.4f}]|{row['p_perm']:.4g}|{row['q_fdr_global']:.4g}|{row['n_pairs']}|")
    lines += ["", "## Duration JSD and per-phone KLD/AUC", "", "|condition|duration phones tested|duration mean JSD|KLD phones tested|mean symmetric KLD|mean AUC|", "|---|---:|---:|---:|---:|---:|"]
    kld_block = result["per_phone_kld_auc"].get("hubert_layer_6", {})
    for condition in TTS_CONDITIONS:
        duration_summary = result["duration_jsd"].get(condition, {}).get("summary", {})
        kld_summary = kld_block.get(condition, {})
        lines.append(f"|{condition}|{duration_summary.get('n_phones_tested', 0)}|{_fmt(duration_summary.get('mean_jsd'))}|{kld_summary.get('n_classes_tested', 0)}|{_fmt(kld_summary.get('mean_kld_sym'))}|{_fmt(kld_summary.get('mean_auc'))}|")
    lines += [
        "",
        "## 迁移结论的解读限制",
        "",
        "中文 AISHELL-1 的正/负方向只能作为方法学参照；LRS3 English 的绝对分数不能与中文或 MDC n=50 直接比较。层位曲线的峰值是本英文 cohort 的描述性结果，不应回填为中文最佳层。若后续要解释 Sync-C，需要再以同一 sample_id 做独立的音频表征—视频分数关联，不能由本报告单独推出 TFG 因果。",
        "",
        "完整 token support、每条样本 boundary/segment 统计、MFA provenance、audio/feature hash 和失败清单见相邻 JSONL/JSON artifacts。",
    ]
    (run_dir / "03_metrics/report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_protocol(run_dir: Path, cohort_path: Path) -> None:
    code_paths = [
        Path(__file__),
        REPO_ROOT / "scripts/16_feature_separability.py",
        REPO_ROOT / "scripts/36_duration_jsd.py",
        REPO_ROOT / "scripts/38_per_phoneme_kld.py",
        REPO_ROOT / "scripts/experiments/lrs3_phone_rules_metrics.py",
        REPO_ROOT / "scripts/experiments/lrs3_phone_rules_worker.py",
        REPO_ROOT / "scripts/prepare_mdc_english_alignment.py",
    ]
    protocol = {
        "schema_version": 1,
        "protocol": "lrs3_english_phoneme_separability_condition_transfer_v1",
        "created_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "run_id": run_dir.name,
        "cohort": str(cohort_path.resolve()),
        "cohort_sha256": _sha256(cohort_path),
        "sample_rate": SAMPLE_RATE,
        "conditions": list(CONDITIONS),
        "models": MODELS,
        "mfa": {"executable": "mfa", "dictionary": "english_us_mfa", "acoustic_model": "english_mfa", "alignment_config": MFA_ALIGNMENT_CONFIG, "independent_new_alignment_conditions": list(("qwen_local", "index_tts2", "cosyvoice2"))},
        "reused_cache": {"run": str(OLD_FEATURE_RUN.resolve()), "eligibility": "audio container SHA-256 plus model/layer file presence"},
        "metrics_reused": ["scripts/16_feature_separability.py", "scripts/36_duration_jsd.py", "scripts/38_per_phoneme_kld.py", "scripts/experiments/lrs3_phone_rules_metrics.py"],
        "code_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in code_paths if path.is_file()},
        "git_head": subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True, capture_output=True, check=False).stdout.strip(),
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain"], cwd=str(REPO_ROOT), text=True, capture_output=True, check=False).stdout.strip()),
    }
    _json(run_dir / "protocol.json", protocol)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "mfa", "audit", "extract", "metrics", "all"))
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--cohort", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--no-mfa", action="store_true", help="only stage MFA corpus and audit existing alignments")
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    cohort_path = args.cohort.resolve()
    if args.stage in {"prepare", "mfa", "all"}:
        _write_protocol(run_dir, cohort_path)
        if args.stage == "prepare":
            prepare(run_dir, cohort_path, run_mfa=False)
        else:
            prepare(run_dir, cohort_path, run_mfa=not args.no_mfa)
    if args.stage == "audit":
        manifest = read_json(run_dir / "00_inputs/manifest.json")
        _audit_alignments(manifest, run_dir)
        return 0
    if args.stage == "extract":
        extract(run_dir, device=args.device)
    elif args.stage == "metrics":
        metrics(run_dir)
    elif args.stage == "all":
        extract(run_dir, device=args.device)
        metrics(run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
