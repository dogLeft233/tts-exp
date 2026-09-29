"""Configuration, hashing, resource gates, and atomic run-artifact I/O."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import MEASUREMENT_VERSION, PROTOCOL_ID

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def path_binding(path: str | Path) -> dict[str, Any] | None:
    """Hash a file or a directory without losing relative member identity."""

    target = Path(path).resolve()
    if target.is_file():
        return {"path": str(target), "kind": "file", "sha256": file_sha256(target)}
    if target.is_dir():
        digest = hashlib.sha256()
        members = []
        for member in sorted(item for item in target.rglob("*") if item.is_file()):
            relative = str(member.relative_to(target))
            member_hash = file_sha256(member)
            digest.update(relative.encode("utf-8"))
            digest.update(member_hash.encode("ascii"))
            members.append({"relative": relative, "sha256": member_hash})
        return {"path": str(target), "kind": "directory", "sha256": digest.hexdigest(), "members": members}
    return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("non-finite numeric value cannot be serialized")
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if hasattr(value, "detach") and callable(value.detach):
        value = value.detach().cpu()
        if int(value.numel()) != 1:
            raise TypeError("non-scalar tensor cannot be written to JSON")
        return _json_safe(value.item())
    if hasattr(value, "item") and callable(value.item):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError, RuntimeError):
            pass
    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _json_safe(value.tolist())
        except (TypeError, ValueError):
            pass
    return value


def write_json(path: str | Path, payload: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(target)
    return file_sha256(target)


def read_json(path: str | Path) -> Any:
    def reject(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=reject)


def write_jsonl(path: str | Path, rows: list[Mapping[str, Any]]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(_json_safe(dict(row)), ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(target)
    return file_sha256(target)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                if not isinstance(item, dict):
                    raise ValueError(f"JSONL row is not an object: {path}")
                rows.append(item)
    return rows


def load_yaml(path: str | Path) -> dict[str, Any]:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def resolve_path(value: str | Path, repo_root: str | Path = REPO_ROOT) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else Path(repo_root) / path


def validate_run_id(run_id: str) -> str:
    if not RUN_ID_RE.fullmatch(str(run_id)):
        raise ValueError("invalid run id")
    return str(run_id)


def state_dict_hash(state_dict: Mapping[str, Any]) -> str:
    import torch

    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key].detach().cpu().contiguous()
        digest.update(str(key).encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(repr(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _gpu_query(query: str) -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    names = [part.strip() for part in query.split(",")]
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) == len(names):
            rows.append(dict(zip(names, values)))
    return rows


def _gpu_processes() -> list[dict[str, str]]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory", "--format=csv,noheader,nounits"],
            text=True,
            capture_output=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    rows: list[dict[str, str]] = []
    for line in result.stdout.splitlines():
        values = [part.strip() for part in line.split(",")]
        if len(values) == 4:
            rows.append(dict(zip(("gpu_uuid", "pid", "process_name", "used_memory_mib"), values)))
    return rows


def resource_snapshot(*, gpu_index: int = 0, ignored_process_names: list[str] | tuple[str, ...] = ()) -> dict[str, Any]:
    disk = shutil.disk_usage(REPO_ROOT)
    available_ram = None
    try:
        meminfo = Path("/proc/meminfo").read_text(encoding="ascii")
        available_ram = next(float(line.split()[1]) * 1024 for line in meminfo.splitlines() if line.startswith("MemAvailable:"))
    except (OSError, StopIteration, ValueError):
        pass
    gpus = _gpu_query("index,name,memory.used,memory.total,utilization.gpu")
    gpu = next((row for row in gpus if int(row.get("index", -1)) == int(gpu_index)), None)
    processes = _gpu_processes()
    own_pid = str(os.getpid())
    ignored = tuple(str(value) for value in ignored_process_names if str(value))
    foreign = [row for row in processes if row.get("pid") != own_pid and not any(pattern in str(row.get("process_name", "")) for pattern in ignored)]
    ignored_processes = [row for row in processes if row.get("pid") != own_pid and any(pattern in str(row.get("process_name", "")) for pattern in ignored)]
    return {
        "timestamp": time.time(),
        "pid": os.getpid(),
        "disk_available_bytes": int(disk.free),
        "ram_available_bytes": int(available_ram) if available_ram is not None else None,
        "gpu": gpu,
        "gpu_compute_processes": processes,
        "ignored_gpu_compute_processes": ignored_processes,
        "ignored_process_name_patterns": list(ignored),
        "foreign_gpu_compute_processes": foreign,
        "gpu_ready_without_foreign_process": bool(gpu is not None and not foreign and float(gpu.get("utilization.gpu", "100")) <= 10.0),
    }


def resource_decision(config: Mapping[str, Any], *, stage: str, gpu_required: bool) -> dict[str, Any]:
    runtime = config.get("runtime", {})
    snap = resource_snapshot(gpu_index=int(runtime.get("gpu_index", 0)), ignored_process_names=tuple(runtime.get("gpu_process_allowlist", [])))
    reasons: list[str] = []
    min_disk = float(runtime.get("min_free_disk_gib", 4.0)) * (1 << 30)
    min_ram = float(runtime.get("min_available_ram_gib", 8.0)) * (1 << 30)
    if snap["disk_available_bytes"] < min_disk:
        reasons.append("DISK_BELOW_MINIMUM")
    if snap["ram_available_bytes"] is not None and snap["ram_available_bytes"] < min_ram:
        reasons.append("RAM_BELOW_MINIMUM")
    if gpu_required:
        gpu = snap.get("gpu")
        default_gpu = 5.0 if stage in {"render", "score"} else 12.0
        configured_gpu = runtime.get("min_free_gpu_gib")
        if configured_gpu is None:
            configured_gpu = runtime.get(f"min_free_gpu_gib_{stage}", default_gpu)
        min_gpu = float(configured_gpu)
        if gpu is None:
            reasons.append("GPU_UNAVAILABLE")
        else:
            free_mib = float(gpu.get("memory.total", 0)) - float(gpu.get("memory.used", 0))
            if free_mib < min_gpu * 1024:
                reasons.append("GPU_MEMORY_BELOW_MINIMUM")
            if not bool(snap.get("gpu_ready_without_foreign_process")):
                reasons.append("GPU_OCCUPIED")
    return {"stage": stage, "decision": "WAIT" if reasons else "PASS", "reasons": reasons, "snapshot": snap}


@contextmanager
def proxy_environment(proxy: str | None):
    """Bind every common proxy variable for model downloads in this run only."""
    if not proxy:
        yield
        return
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy", "ALL_PROXY", "all_proxy")
    old = {key: os.environ.get(key) for key in keys}
    try:
        for key in keys:
            os.environ[key] = str(proxy)
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def protocol_snapshot(config: Mapping[str, Any], config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path).resolve()
    source = dict(config.get("source", {}))
    input_hashes: dict[str, Any] = {}
    for key in ("parent_registry", "parent_support", "parent_support_jsonl", "parent_lock", "parent_pilot", "parent_direct_dir"):
        value = source.get(key)
        candidate = resolve_path(value) if value else None
        input_hashes[key] = path_binding(candidate) if candidate and candidate.exists() else None
    old_run = resolve_path(source.get("old_run")) if source.get("old_run") else None
    old_manifest = old_run / "04_audio/manifest.json" if old_run else None
    input_hashes["old_audio_manifest"] = path_binding(old_manifest) if old_manifest and old_manifest.exists() else None
    code_root = Path(__file__).resolve().parent
    code_hashes = {str(item.relative_to(REPO_ROOT)): file_sha256(item) for item in sorted(code_root.glob("*.py"))}
    shared_files = [
        REPO_ROOT / "scripts/experiments/static_image_bridge/images.py",
        REPO_ROOT / "scripts/experiments/static_image_bridge/detect_worker.py",
        REPO_ROOT / "scripts/experiments/static_image_bridge/render_worker.py",
        REPO_ROOT / "scripts/experiments/phone_separability_enhancement/audio.py",
        REPO_ROOT / "scripts/experiments/phone_separability_enhancement/teacher.py",
        REPO_ROOT / "scripts/experiments/phone_separability_enhancement/metrics.py",
    ]
    for item in shared_files:
        if item.is_file():
            code_hashes[str(item.relative_to(REPO_ROOT))] = file_sha256(item)
    model_bindings: dict[str, Any] = {}
    for section in ("models", "tfg"):
        values = config.get(section, {})
        if isinstance(values, Mapping):
            for key, value in values.items():
                candidate = resolve_path(value) if isinstance(value, (str, Path)) else None
                if candidate is not None and candidate.is_file():
                    model_bindings[f"{section}.{key}"] = {"path": str(candidate), "sha256": file_sha256(candidate)}
    portrait_bindings: dict[str, Any] = {}
    for key, value in (config.get("portraits", {}) or {}).items():
        if isinstance(value, (str, Path)):
            binding = path_binding(resolve_path(value))
            if binding is not None:
                portrait_bindings[str(key)] = binding
        elif isinstance(value, list):
            portrait_bindings[str(key)] = [path_binding(resolve_path(item)) for item in value]
    return {
        "schema_version": 2,
        "protocol_id": str(config.get("protocol_id", PROTOCOL_ID)),
        "measurement_version": str(config.get("measurement_version", MEASUREMENT_VERSION)),
        "config_path": str(path),
        "config_sha256": file_sha256(path),
        "input_hashes": input_hashes,
        "code_hashes": code_hashes,
        "model_bindings": model_bindings,
        "portrait_bindings": portrait_bindings,
        "config": config,
    }


def ensure_protocol_unchanged(previous: Mapping[str, Any], current: Mapping[str, Any]) -> None:
    for key in ("protocol_id", "measurement_version", "config_sha256", "input_hashes", "code_hashes", "model_bindings", "portrait_bindings"):
        if previous.get(key) != current.get(key):
            raise ValueError(f"resume rejected: protocol field changed: {key}")


__all__ = [
    "MEASUREMENT_VERSION",
    "PROTOCOL_ID",
    "REPO_ROOT",
    "canonical_hash",
    "ensure_protocol_unchanged",
    "file_sha256",
    "load_yaml",
    "read_json",
    "read_jsonl",
    "resolve_path",
    "resource_decision",
    "resource_snapshot",
    "state_dict_hash",
    "validate_run_id",
    "write_json",
    "write_jsonl",
]
