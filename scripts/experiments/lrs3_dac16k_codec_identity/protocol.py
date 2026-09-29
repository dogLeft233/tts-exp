from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

from . import config
from .common import (
    assert_run_root_compatible,
    canonical_json_sha256,
    file_sha256,
    read_json,
    verify_self_hashed_json,
    write_self_hashed_json,
)


class ProtocolError(RuntimeError):
    pass


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.download")
    request = Request(url, headers={"User-Agent": "lrs3-dac16k-codec-identity/1"})
    with urlopen(request, timeout=120) as response, temporary.open("wb") as handle:
        while True:
            chunk = response.read(1 << 20)
            if not chunk:
                break
            handle.write(chunk)
    os.replace(temporary, destination)


def ensure_dac_source(source_root: Path = config.DAC_SOURCE_ROOT) -> dict[str, Any]:
    marker = source_root / "source_revision.json"
    if marker.is_file():
        binding = verify_self_hashed_json(marker)
        if binding.get("commit") != config.DAC_SOURCE_COMMIT:
            raise ProtocolError(f"DAC source revision mismatch: {source_root}")
        for relative, metadata in binding.get("files", {}).items():
            path = source_root / str(relative)
            if not path.is_file() or file_sha256(path) != metadata.get("sha256"):
                raise ProtocolError(f"DAC source file changed: {path}")
        if "path" not in binding:
            body = {key: value for key, value in binding.items() if key != "artifact_sha256"}
            body["path"] = str(source_root.resolve())
            write_self_hashed_json(marker, body)
            binding = verify_self_hashed_json(marker)
        return binding
    if source_root.exists() and any(source_root.iterdir()):
        raise ProtocolError(f"DAC source directory is populated but unbound: {source_root}")

    api_url = f"https://api.github.com/repos/descriptinc/descript-audio-codec/git/trees/{config.DAC_SOURCE_COMMIT}?recursive=1"
    request = Request(api_url, headers={"Accept": "application/vnd.github+json", "User-Agent": "lrs3-dac16k-codec-identity/1"})
    tree = json.loads(urlopen(request, timeout=60).read())
    paths = [
        str(item["path"])
        for item in tree.get("tree", [])
        if item.get("type") == "blob" and (str(item["path"]).startswith("dac/") or str(item["path"]) in {"setup.py", "pyproject.toml"})
    ]
    if not paths:
        raise ProtocolError("pinned DAC source tree is empty")

    source_root.mkdir(parents=True, exist_ok=True)
    files: dict[str, dict[str, Any]] = {}
    for relative in paths:
        destination = source_root / relative
        raw_url = f"https://raw.githubusercontent.com/descriptinc/descript-audio-codec/{config.DAC_SOURCE_COMMIT}/{relative}"
        _download(raw_url, destination)
        files[relative] = {"sha256": file_sha256(destination), "size": destination.stat().st_size}
    binding = {
        "source": config.DAC_SOURCE_URL,
        "path": str(source_root.resolve()),
        "tag": config.DAC_TAG,
        "commit": config.DAC_SOURCE_COMMIT,
        "acquisition": "github_raw_tree_snapshot",
        "files": files,
    }
    write_self_hashed_json(marker, binding)
    return verify_self_hashed_json(marker)


def ensure_dac_checkpoint(checkpoint: Path = config.DAC_CHECKPOINT) -> dict[str, Any]:
    if not checkpoint.is_file():
        _download(config.DAC_CHECKPOINT_URL, checkpoint)
    size = checkpoint.stat().st_size
    if size != config.DAC_EXPECTED_CHECKPOINT_SIZE:
        raise ProtocolError(f"DAC checkpoint size mismatch: {size}")
    return {
        "url": config.DAC_CHECKPOINT_URL,
        "path": str(checkpoint.resolve()),
        "size": size,
        "sha256": file_sha256(checkpoint),
    }


def _dependency_lock() -> dict[str, Any]:
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze", "--all"],
        capture_output=True,
        text=True,
        check=True,
    )
    freeze = result.stdout.splitlines()
    versions: dict[str, str | None] = {}
    for name in ("descript-audio-codec", "descript-audiotools", "torch", "torchaudio", "numpy", "soundfile", "librosa", "einops", "argbind"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    missing = [name for name, version in versions.items() if version is None]
    if missing:
        raise ProtocolError(f"DAC dependency lock is incomplete: {missing}")
    return {
        "python": platform.python_version(),
        "executable": str(Path(sys.executable).resolve()),
        "freeze": freeze,
        "freeze_sha256": hashlib.sha256(("\n".join(freeze) + "\n").encode()).hexdigest(),
        "required_package_versions": versions,
    }


def _sample_ids_hash(sample_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(sample_ids).encode()).hexdigest()


def _resolve_parent_source_manifest(parent: Mapping[str, Any]) -> Path:
    selection = parent.get("selection")
    if not isinstance(selection, Mapping):
        raise ProtocolError("parent selection is missing")
    path = Path(str(selection.get("manifest", "")))
    if not path.is_absolute():
        path = config.REPO / path
    return path.resolve()


def load_frozen_cohort(parent_summary: Path = config.PARENT_SUMMARY) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if file_sha256(parent_summary) != config.EXPECTED_PARENT_SUMMARY_SHA256:
        raise ProtocolError("parent summary hash mismatch")
    parent = read_json(parent_summary)
    if parent.get("status") != "complete" or parent.get("dataset") != "lrs3":
        raise ProtocolError("parent summary is not a complete LRS3 run")
    selection = parent.get("selection", {})
    ids = [str(value) for value in selection.get("sample_ids", [])]
    if len(ids) != config.EXPECTED_RECORD_COUNT or _sample_ids_hash(ids) != config.EXPECTED_SAMPLE_ID_SHA256:
        raise ProtocolError("parent sample-ID binding mismatch")
    if int(selection.get("source_groups", -1)) != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ProtocolError("parent source-group binding mismatch")

    source_manifest = _resolve_parent_source_manifest(parent)
    if not source_manifest.is_file() or file_sha256(source_manifest) != config.EXPECTED_SOURCE_MANIFEST_SHA256:
        raise ProtocolError("parent source manifest hash mismatch")
    source = read_json(source_manifest)
    source_by_id = {str(row.get("sample_id")): row for row in source.get("records", [])}
    parent_rows = parent.get("reconstructions")
    if not isinstance(parent_rows, list) or len(parent_rows) != config.EXPECTED_RECORD_COUNT:
        raise ProtocolError("parent reconstruction list is incomplete")

    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, parent_row in enumerate(parent_rows):
        sample_id = str(parent_row.get("sample_id", ""))
        if sample_id != ids[index] or sample_id in seen:
            raise ProtocolError(f"parent record order mismatch at {index}: {sample_id}")
        seen.add(sample_id)
        source_row = source_by_id.get(sample_id)
        if source_row is None:
            raise ProtocolError(f"sample is absent from source manifest: {sample_id}")
        if str(source_row.get("source_group")) != str(parent_row.get("source_group")):
            raise ProtocolError(f"source group mismatch: {sample_id}")
        natural = Path(str(parent_row.get("natural_audio"))).resolve()
        video = Path(str(parent_row.get("video"))).resolve()
        wavlm = Path(str(parent_row.get("resynthesis", {}).get("output_path"))).resolve()
        paths = (("natural_audio", natural, parent_row.get("natural_audio_sha256")), ("video", video, parent_row.get("video_sha256")), ("wavlm_audio", wavlm, parent_row.get("resynthesis", {}).get("output_sha256")))
        hashes: dict[str, str] = {}
        for name, path, expected in paths:
            if not path.is_file():
                raise ProtocolError(f"missing bound {name}: {path}")
            actual = file_sha256(path)
            if actual != str(expected):
                raise ProtocolError(f"bound {name} hash mismatch: {sample_id}")
            hashes[name] = actual
        if hashes["natural_audio"] != str(source_row.get("natural_audio_sha256")) or hashes["video"] != str(source_row.get("video_sha256")):
            raise ProtocolError(f"source manifest asset mismatch: {sample_id}")
        sample_count = int(parent_row.get("natural_sample_count", -1))
        expected_natural = (config.REPO / str(source_row["natural_audio_path"])).resolve()
        expected_video = (config.REPO / str(source_row["video_local_path"])).resolve()
        if natural != expected_natural or video != expected_video:
            raise ProtocolError(f"source manifest path mismatch: {sample_id}")
        if sample_count <= 0 or int(source_row.get("natural_audio_samples", -1)) != sample_count:
            raise ProtocolError(f"invalid natural sample count: {sample_id}")
        for path in (natural, video, wavlm):
            lowered = str(path).lower()
            if any(token in lowered for token in config.NO_SEALED_MEDIA_TOKENS):
                raise ProtocolError(f"sealed media path crossed: {path}")
        records.append({
            "sample_id": sample_id,
            "source_group": str(parent_row.get("source_group")),
            "clip_id": str(parent_row.get("clip_id")),
            "transcript": str(parent_row.get("transcript", "")),
            "natural_audio": str(natural),
            "natural_audio_sha256": hashes["natural_audio"],
            "wavlm_audio": str(wavlm),
            "wavlm_audio_sha256": hashes["wavlm_audio"],
            "video": str(video),
            "video_sha256": hashes["video"],
            "natural_sample_count": sample_count,
            "parent_record_sha256": canonical_json_sha256(parent_row),
            "source_manifest_record_sha256": canonical_json_sha256(source_row),
        })
    if len({row["source_group"] for row in records}) != config.EXPECTED_SOURCE_GROUP_COUNT:
        raise ProtocolError("cohort source-group count mismatch")
    return records, {
        "path": str(source_manifest),
        "sha256": file_sha256(source_manifest),
        "record_count": len(records),
        "source_group_count": len({row["source_group"] for row in records}),
        "sample_ids_sha256": _sample_ids_hash(ids),
    }


def _protocol_binding(source: Mapping[str, Any], checkpoint: Mapping[str, Any], dependencies: Mapping[str, Any], source_manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "experiment": config.EXPERIMENT,
        "dataset": "lrs3",
        "source": source,
        "checkpoint": checkpoint,
        "dac": {
            "model_type": config.DAC_MODEL_TYPE,
            "bitrate": config.DAC_BITRATE,
            "sample_rate": config.DAC_SAMPLE_RATE,
            "n_quantizers": None,
            "quantizer_policy": "all_checkpoint_provided_quantizers",
            "hop_length": config.DAC_HOP_LENGTH,
            "model_input_padding": "right_pad_to_one_hop_beyond_next_multiple_before_forward",
            "postprocess": "crop_only_to_original_sample_count_then_canonical_pcm16_once",
            "forbidden_operations": ["resampling", "loudness_normalization", "gain_matching", "temporal_shift", "delay_compensation", "lag_alignment", "interpolation", "filtering", "denoising", "phase_correction", "trimming", "output_repair"],
        },
        "parent": {
            "summary": str(config.PARENT_SUMMARY.resolve()),
            "summary_sha256": config.EXPECTED_PARENT_SUMMARY_SHA256,
            "source_manifest": source_manifest,
        },
        "dependencies": dependencies,
        "sealed_scope": {
            "validation_test_access": False,
            "tts_access": False,
            "mfa_access": False,
            "dtw_access": False,
            "training": False,
        },
    }


def run_stage00() -> dict[str, Any]:
    assert_run_root_compatible(config.RUN_ROOT, config.PROTOCOL_ID)
    config.RUN_ROOT.mkdir(parents=True, exist_ok=True)
    failures: list[dict[str, Any]] = []
    try:
        source = ensure_dac_source()
        checkpoint = ensure_dac_checkpoint()
        dependencies = _dependency_lock()
        records, source_manifest = load_frozen_cohort()
        binding = _protocol_binding(source, checkpoint, dependencies, source_manifest)
        protocol_path = config.STAGES["00_protocol"] / "protocol.json"
        if protocol_path.is_file():
            existing = verify_self_hashed_json(protocol_path)
            existing_body = dict(existing)
            existing_body.pop("artifact_sha256", None)
            if canonical_json_sha256(existing_body) != canonical_json_sha256(binding):
                parent_keys = ("parent", "source", "checkpoint", "sealed_scope")
                immutable_match = all(existing_body.get(key) == binding.get(key) for key in parent_keys)
                downstream_started = any((config.RUN_ROOT / name).exists() for name in config.STAGES if name != "00_protocol")
                if immutable_match and not downstream_started:
                    write_self_hashed_json(protocol_path, binding)
                else:
                    raise ProtocolError("existing Stage 00 protocol binding differs")
        else:
            write_self_hashed_json(protocol_path, binding)
        cohort = {
            "schema_version": 1,
            "stage_id": "00_protocol",
            "protocol_id": config.PROTOCOL_ID,
            "status": "complete",
            "parent_summary_sha256": config.EXPECTED_PARENT_SUMMARY_SHA256,
            "source_manifest": source_manifest,
            "record_count": len(records),
            "source_group_count": len({row["source_group"] for row in records}),
            "sample_ids_sha256": source_manifest["sample_ids_sha256"],
            "records": records,
        }
        write_self_hashed_json(config.STAGES["00_protocol"] / "cohort.json", cohort)
        media_access = {
            "schema_version": 1,
            "stage_id": "00_protocol",
            "protocol_id": config.PROTOCOL_ID,
            "cohort_audio_decode_count": 0,
            "cohort_video_decode_count": 0,
            "cohort_audio_hash_read_count": config.EXPECTED_RECORD_COUNT * 2,
            "cohort_video_hash_read_count": config.EXPECTED_RECORD_COUNT,
            "score_read_count": 0,
            "sealed_media_accessed": False,
            "selection_used_scores": False,
        }
        write_self_hashed_json(config.STAGES["00_protocol"] / "media_access.json", media_access)
        write_self_hashed_json(config.STAGES["00_protocol"] / "failures.json", {"schema_version": 1, "stage_id": "00_protocol", "protocol_id": config.PROTOCOL_ID, "failures": failures})
        return {"status": "complete", "record_count": len(records), "source_group_count": source_manifest["source_group_count"]}
    except Exception as exc:
        failures.append({"error_type": type(exc).__name__, "error": str(exc)})
        write_self_hashed_json(config.STAGES["00_protocol"] / "media_access.json", {"schema_version": 1, "stage_id": "00_protocol", "protocol_id": config.PROTOCOL_ID, "cohort_audio_decode_count": 0, "cohort_video_decode_count": 0, "score_read_count": 0, "sealed_media_accessed": False, "selection_used_scores": False})
        write_self_hashed_json(config.STAGES["00_protocol"] / "failures.json", {"schema_version": 1, "stage_id": "00_protocol", "protocol_id": config.PROTOCOL_ID, "status": "blocked", "failures": failures})
        raise
