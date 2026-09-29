from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np

from . import config


class ProtocolError(RuntimeError):
    """An input, resource, or artifact contract violation."""


class InputInvalidError(ProtocolError):
    """A frozen input or transitive evidence binding is invalid."""


class DependencyBlockedError(ProtocolError):
    """A required external model or executable is not provenance-bound."""


class ResourceWaitError(ProtocolError):
    """The requested cell must wait for disk, VRAM, or a shared device lease."""


class ImplementationIncompleteError(ProtocolError):
    """A requested stage has an explicitly unfinished implementation."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return bytes_sha256(canonical_json(value))


def assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, (float, np.floating)) and not math.isfinite(float(value)):
        raise ProtocolError(f"non-finite value at {path}")
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_finite(item, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            assert_finite(item, f"{path}[{index}]")


def read_json(path: str | Path) -> Any:
    target = Path(path)

    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant: {value}")

    try:
        value = json.loads(target.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON: {target}") from exc
    assert_finite(value)
    return value


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    assert_finite(value)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def write_self_hashed_json(path: str | Path, value: Mapping[str, Any]) -> dict[str, Any]:
    body = dict(value)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_sha256(body)
    write_json(path, body)
    return body


def read_self_hashed_json(path: str | Path) -> dict[str, Any]:
    value = read_json(path)
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    recorded = value.get("artifact_sha256")
    body = dict(value)
    body.pop("artifact_sha256", None)
    if not isinstance(recorded, str) or recorded != canonical_sha256(body):
        raise ProtocolError(f"artifact hash mismatch: {path}")
    return value


def require_file(path: str | Path, label: str) -> Path:
    target = Path(path)
    if not target.is_file():
        raise ProtocolError(f"{label} is missing: {target}")
    return target


def resolve_repo_path(value: str | Path) -> Path:
    target = Path(value)
    if not target.is_absolute():
        target = config.REPO / target
    return target.resolve()


def asset_binding(path: str | Path, *, expected: str | None = None, label: str = "asset") -> dict[str, str]:
    target = resolve_repo_path(path)
    require_file(target, label)
    actual = file_sha256(target)
    if expected and actual != expected:
        raise ProtocolError(f"{label} hash mismatch: {target}: {actual} != {expected}")
    return {"path": str(target), "sha256": actual}


def verify_binding(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ProtocolError(f"{label} binding is malformed")
    return asset_binding(str(value.get("path", "")), expected=str(value.get("sha256", "")), label=label)


def copy_atomic(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, target)


def run_command(command: Sequence[str], *, cwd: Path | None = None, timeout: float | None = None) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(list(command), cwd=str(cwd) if cwd else None, capture_output=True, check=True, timeout=timeout)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", b"")
        if isinstance(detail, bytes):
            detail = detail.decode("utf-8", errors="replace")
        raise ProtocolError(f"command failed: {' '.join(map(str, command))}: {str(detail)[-1000:]}") from exc


def ffprobe_json(path: Path) -> dict[str, Any]:
    executable = executable_path(config.FFPROBE, "ffprobe")
    result = run_command((str(executable), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)))
    value = json.loads(result.stdout.decode("utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"ffprobe output is malformed: {path}")
    return value


def executable_path(configured: Path, name: str) -> Path:
    """Resolve a configured executable, retaining the actual path in audits."""

    if configured.is_file():
        return configured.resolve()
    found = shutil.which(name)
    if found:
        return Path(found).resolve()
    raise ProtocolError(f"{name} executable is unavailable: {configured}")


def decode_media_pcm16(path: str | Path, *, ffmpeg: Path | None = None) -> np.ndarray:
    """Use the historical worker's exact 16 kHz mono PCM decode contract."""

    target = require_file(path, "media")
    executable = executable_path(ffmpeg or config.FFMPEG, "ffmpeg")
    result = run_command(
        (
            str(executable),
            "-v",
            "error",
            "-i",
            str(target),
            "-async",
            "1",
            "-ac",
            "1",
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(config.SAMPLE_RATE),
            "-f",
            "s16le",
            "pipe:1",
        )
    )
    payload = result.stdout
    if len(payload) == 0 or len(payload) % 2:
        raise ProtocolError(f"ffmpeg returned invalid PCM16 payload: {target}")
    values = np.frombuffer(payload, dtype="<i2").copy()
    if values.ndim != 1 or values.size == 0:
        raise ProtocolError(f"ffmpeg returned empty PCM16 payload: {target}")
    return values


def csv_write(path: str | Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    import csv

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, target)


def csv_read(path: str | Path) -> list[dict[str, str]]:
    import csv

    try:
        with Path(path).open(encoding="utf-8", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError as exc:
        raise ProtocolError(f"cannot read CSV: {path}") from exc


def gpu_snapshot() -> dict[str, Any]:
    try:
        result = subprocess.run(
            ("nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"),
            capture_output=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return {"available": False, "rows": []}
    rows = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        fields = [item.strip() for item in line.split(",")]
        if len(fields) != 6:
            continue
        try:
            rows.append({"index": int(fields[0]), "name": fields[1], "memory_total_mib": int(fields[2]), "memory_used_mib": int(fields[3]), "memory_free_mib": int(fields[4]), "utilization_gpu_percent": int(fields[5])})
        except ValueError:
            continue
    return {"available": bool(rows), "rows": rows}


def gpu_compute_pids() -> list[int]:
    try:
        result = subprocess.run(("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"), capture_output=True, check=True, timeout=10)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return []
    pids = []
    for line in result.stdout.decode("utf-8", errors="replace").splitlines():
        try:
            pids.append(int(line.strip()))
        except ValueError:
            continue
    return pids


def process_tree_pids(pid: int) -> set[int]:
    """Return a best-effort PID set for a subprocess created by this run."""

    result = {int(pid)}
    changed = True
    while changed:
        changed = False
        try:
            entries = list(Path("/proc").iterdir())
        except OSError:
            break
        for entry in entries:
            if not entry.name.isdigit():
                continue
            child = int(entry.name)
            try:
                status = (entry / "status").read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            parent = next((line.split("\t", 1)[1] for line in status.splitlines() if line.startswith("PPid:\t")), None)
            if parent is not None and int(parent) in result and child not in result:
                result.add(child)
                changed = True
    return result


def run_monitored(command: Sequence[str], *, cwd: Path | None, log_path: Path, interval_seconds: float = 30.0) -> dict[str, Any]:
    """Run an owned GPU subprocess while recording resource observations."""

    log_path.parent.mkdir(parents=True, exist_ok=True)
    snapshots: list[dict[str, Any]] = []
    environment = os.environ.copy()
    ffmpeg = executable_path(config.FFMPEG, "ffmpeg")
    environment["PATH"] = f"{ffmpeg.parent}{os.pathsep}{environment.get('PATH', '')}"
    process: subprocess.Popen[bytes] | None = None
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(list(command), cwd=str(cwd) if cwd else None, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            while process.poll() is None:
                snapshot = {"time": time.time(), "gpu": gpu_snapshot(), "compute_pids": gpu_compute_pids()}
                snapshots.append(snapshot)
                owned = process_tree_pids(process.pid)
                foreign = [pid for pid in snapshot["compute_pids"] if pid not in owned]
                if foreign:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
                    raise ResourceWaitError(f"RESOURCE_WAIT: foreign compute PIDs appeared during owned command: {foreign}")
                time.sleep(max(0.1, float(interval_seconds)))
            returncode = int(process.returncode or 0)
    except ResourceWaitError:
        if process is not None:
            snapshots.append({"time": time.time(), "gpu": gpu_snapshot(), "compute_pids": gpu_compute_pids()})
            write_json(log_path.with_suffix(log_path.suffix + ".resource.json"), {"command": list(command), "snapshots": snapshots, "returncode": int(process.returncode or 0), "status": "RESOURCE_WAIT"})
        raise
    snapshots.append({"time": time.time(), "gpu": gpu_snapshot(), "compute_pids": gpu_compute_pids()})
    write_json(log_path.with_suffix(log_path.suffix + ".resource.json"), {"command": list(command), "snapshots": snapshots, "returncode": returncode})
    if returncode != 0:
        raise ProtocolError(f"owned command failed with exit code {returncode}: {log_path}")
    return {"command": list(command), "log": str(log_path), "snapshots": snapshots, "returncode": returncode}


def disk_snapshot(path: Path = config.REPO) -> dict[str, int]:
    usage = shutil.disk_usage(path)
    return {"total_bytes": int(usage.total), "used_bytes": int(usage.used), "free_bytes": int(usage.free)}


def resource_plan(
    *,
    gpu_peak_bytes: int = config.GPU_PEAK_BUDGET_BYTES,
    disk_temp_bytes: int = config.CELL_TEMP_BUDGET_BYTES,
    disk_persistent_bytes: int = config.PERSISTENT_BUDGET_BYTES,
) -> dict[str, Any]:
    gpu_peak_bytes = int(gpu_peak_bytes)
    disk_temp_bytes = int(disk_temp_bytes)
    disk_persistent_bytes = int(disk_persistent_bytes)
    return {
        "disk_before": disk_snapshot(),
        "disk_persistent_budget_bytes": disk_persistent_bytes,
        "disk_temp_budget_bytes": disk_temp_bytes,
        "gpu_peak_budget_bytes": gpu_peak_bytes,
        # Keep the old names in the artifact for readers of the parent run.
        "persistent_budget_bytes": disk_persistent_bytes,
        "single_cell_temp_budget_bytes": disk_temp_bytes,
        "reserve_bytes": config.RESERVE_BYTES,
        "required_disk_free_bytes": disk_persistent_bytes + disk_temp_bytes + config.RESERVE_BYTES,
        "required_gpu_free_bytes": gpu_peak_bytes + config.RESERVE_BYTES,
        "required_free_bytes": disk_persistent_bytes + disk_temp_bytes + config.RESERVE_BYTES,
        "gpu_before": gpu_snapshot(),
        "compute_pids_before": gpu_compute_pids(),
        "lease_path": str(config.GPU_LOCK),
        "lease_scope": "all tts-exp GPU stages and experiments on this host",
        "lease_policy": "exclusive shared lock; no foreign compute PID; three 5-second idle samples; serial GPU stages",
        "current_process_may_not_terminate_foreign_pids": True,
    }


def resource_gate(
    *,
    gpu_peak_bytes: int | None = None,
    disk_temp_bytes: int | None = None,
    disk_persistent_bytes: int | None = None,
    # Backward-compatible aliases used by the parent implementation.  New
    # callers should use the explicitly typed budget names above.
    estimated_persistent: int | None = None,
    estimated_temp: int | None = None,
    require_gpu: bool = True,
    allowed_compute_pids: set[int] | None = None,
) -> dict[str, Any]:
    if disk_persistent_bytes is None:
        disk_persistent_bytes = 0 if estimated_persistent is None else int(estimated_persistent)
    if disk_temp_bytes is None:
        disk_temp_bytes = 0 if estimated_temp is None else int(estimated_temp)
    if gpu_peak_bytes is None:
        gpu_peak_bytes = config.GPU_PEAK_BUDGET_BYTES
    before = resource_plan(
        gpu_peak_bytes=int(gpu_peak_bytes),
        disk_temp_bytes=int(disk_temp_bytes),
        disk_persistent_bytes=int(disk_persistent_bytes),
    )
    errors: list[str] = []
    allowed = set() if allowed_compute_pids is None else {int(pid) for pid in allowed_compute_pids}
    before["compute_pids_before_all"] = list(before["compute_pids_before"])
    before["compute_pids_before"] = [pid for pid in before["compute_pids_before"] if pid not in allowed]
    free = int(before["disk_before"]["free_bytes"])
    required = int(disk_persistent_bytes) + int(disk_temp_bytes) + config.RESERVE_BYTES
    if free < required:
        errors.append(f"disk free {free} < required {required}")
    if require_gpu:
        if not before["gpu_before"].get("available"):
            errors.append("nvidia-smi is unavailable")
        if before["compute_pids_before"]:
            errors.append(f"foreign/unknown compute PIDs present: {before['compute_pids_before']}")
        idle_samples = []
        for index in range(3):
            snapshot = gpu_snapshot()
            idle_samples.append(snapshot)
            if index < 2:
                time.sleep(5)
        before["gpu_idle_samples"] = idle_samples
        for snapshot in idle_samples:
            for row in snapshot.get("rows", []):
                if int(row.get("utilization_gpu_percent", 100)) > 5:
                    errors.append("GPU utilization exceeded 5% during preflight")
                if int(row.get("memory_free_mib", 0)) * (1 << 20) < int(gpu_peak_bytes) + config.RESERVE_BYTES:
                    errors.append("free GPU memory is below estimated GPU peak plus reserve")
    before["gate"] = "PASS" if not errors else "RESOURCE_WAIT"
    before["errors"] = errors
    before["gpu_peak_bytes"] = int(gpu_peak_bytes)
    before["disk_temp_bytes"] = int(disk_temp_bytes)
    before["disk_persistent_bytes"] = int(disk_persistent_bytes)
    before["estimated_persistent_bytes"] = int(disk_persistent_bytes)
    before["estimated_temp_bytes"] = int(disk_temp_bytes)
    return before


@contextmanager
def gpu_lease(
    *,
    gpu_peak_bytes: int | None = None,
    disk_temp_bytes: int | None = None,
    disk_persistent_bytes: int | None = None,
    estimated_persistent: int | None = None,
    estimated_temp: int | None = None,
) -> Iterator[dict[str, Any]]:
    config.GPU_LOCK.parent.mkdir(parents=True, exist_ok=True)
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover
        raise ProtocolError("fcntl is required for the GPU lease") from exc
    with config.GPU_LOCK.open("a+") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ResourceWaitError("RESOURCE_WAIT: shared GPU lease is held") from exc
        try:
            gate = resource_gate(
                gpu_peak_bytes=gpu_peak_bytes,
                disk_temp_bytes=disk_temp_bytes,
                disk_persistent_bytes=disk_persistent_bytes,
                estimated_persistent=estimated_persistent,
                estimated_temp=estimated_temp,
                require_gpu=True,
                allowed_compute_pids={os.getpid()},
            )
            if gate["gate"] != "PASS":
                raise ResourceWaitError("RESOURCE_WAIT: " + "; ".join(gate["errors"]))
            yield gate
        finally:
            # The lease must be released even when a model, decoder, or cell
            # raises.  Leaving the advisory lock held would make resume look
            # like a foreign resource conflict until the process exits.
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def pcm_hash(values: np.ndarray) -> str:
    array = np.asarray(values)
    if array.dtype != np.int16 or array.ndim != 1:
        raise ProtocolError("PCM hash requires one-dimensional int16 samples")
    return bytes_sha256(np.asarray(array, dtype="<i2").tobytes())


def sample_ids_hash(values: Sequence[int | str]) -> str:
    return bytes_sha256("\n".join(str(item) for item in values).encode("utf-8"))


def finite_array(value: Any, *, label: str, ndim: int | None = None) -> np.ndarray:
    array = np.asarray(value)
    if ndim is not None and array.ndim != ndim:
        raise ProtocolError(f"{label} must have ndim {ndim}, got {array.shape}")
    if array.size == 0 or not np.isfinite(array).all():
        raise ProtocolError(f"{label} is empty or non-finite")
    return array
