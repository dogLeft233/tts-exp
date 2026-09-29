"""Provenance-bound request/response contract for the official LeapTalk family.

The experiment deliberately does not embed a second copy of LeapTalk.  The
official checkout remains the model owner, while this module makes the
boundary explicit and auditable: one JSON request enters the adapter and one
response proves what was consumed.  A command template is still useful for a
remote or separately managed checkout, but it is executed as argv with
``shell=False`` and cannot make an incomplete model look ready.
"""

from __future__ import annotations

import hashlib
import os
import platform
import shlex
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from . import config
from .common import (
    DependencyBlockedError,
    ProtocolError,
    canonical_sha256,
    file_sha256,
    read_json,
    read_self_hashed_json,
    write_self_hashed_json,
)

REQUIRED_COMPONENTS = (
    "base",
    "lora",
    "audio_projection",
    "audio_encoder",
    "decoder",
    "vae",
    "tae",
)
REQUIRED_RUNTIME = (
    "mode",
    "sampling",
    "dtype",
    "frontend",
    "resolution",
    "fps",
    "steps",
    "guidance",
    "attention_backend",
    "compile",
    "python",
    "torch",
    "cuda",
    "driver",
    "hardware",
)


def _component_file_rows(component: Mapping[str, Any], root: Path) -> list[dict[str, Any]]:
    raw_files = component.get("files")
    path_value = component.get("path")
    if not isinstance(path_value, str) or not path_value:
        raise DependencyBlockedError("LeapTalk component has no path")
    path = Path(path_value).expanduser()
    if not path.is_absolute():
        path = (root / path).resolve()
    if path.is_file():
        rows = [{"path": str(path), "relative_path": path.name}]
    elif path.is_dir():
        if not isinstance(raw_files, list) or not raw_files:
            raise DependencyBlockedError(
                f"directory component {path} must enumerate every loaded file"
            )
        rows = []
        for item in raw_files:
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                raise DependencyBlockedError(f"malformed file row in component {path}")
            child = (path / str(item["path"])).resolve()
            try:
                child.relative_to(path.resolve())
            except ValueError as exc:
                raise DependencyBlockedError(f"component file escapes its root: {child}") from exc
            rows.append({"path": str(child), "relative_path": str(item["path"])})
    else:
        raise DependencyBlockedError(f"LeapTalk component is missing: {path}")
    return rows


def _bind_component(name: str, raw: Any, root: Path) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise DependencyBlockedError(f"LeapTalk component {name} is not configured")
    if raw.get("loaded") is not True:
        raise DependencyBlockedError(f"LeapTalk component {name} is not proven loaded")
    revision = raw.get("revision")
    if not isinstance(revision, str) or revision in {"", "UNKNOWN"}:
        raise DependencyBlockedError(f"LeapTalk component {name} has no pinned revision")
    rows = _component_file_rows(raw, root)
    bound_files: list[dict[str, Any]] = []
    for item in rows:
        path = Path(str(item["path"]))
        if not path.is_file():
            raise DependencyBlockedError(f"LeapTalk component file is missing: {path}")
        hashes = raw.get("file_hashes")
        if not isinstance(hashes, Mapping) or str(item["relative_path"]) not in hashes:
            raise DependencyBlockedError(f"LeapTalk component {name} has no hash for {item['relative_path']}")
        expected = hashes[str(item["relative_path"])]
        actual = file_sha256(path)
        if expected is not None and actual != str(expected):
            raise DependencyBlockedError(
                f"LeapTalk component hash mismatch: {name}/{item['relative_path']}"
            )
        bound_files.append(
            {
                "path": str(path),
                "relative_path": str(item["relative_path"]),
                "bytes": int(path.stat().st_size),
                "sha256": actual,
            }
        )
    declared = raw.get("sha256")
    component_hash = canonical_sha256(
        [{"relative_path": item["relative_path"], "sha256": item["sha256"], "bytes": item["bytes"]} for item in bound_files]
    )
    if declared is not None and str(declared) != component_hash:
        raise DependencyBlockedError(f"LeapTalk component aggregate hash mismatch: {name}")
    return {
        "name": name,
        "path": str(Path(str(raw["path"])).expanduser().resolve() if Path(str(raw["path"])).is_absolute() else (root / str(raw["path"])).resolve()),
        "revision": revision,
        "loaded": True,
        "files": bound_files,
        "sha256": component_hash,
    }


def _command_from_config(payload: Mapping[str, Any]) -> str | list[str] | None:
    command = payload.get("command")
    if isinstance(command, str) and command.strip():
        return command
    if isinstance(command, list) and all(isinstance(item, str) and item for item in command):
        return list(command)
    environment = os.environ.get("LEAPTALK_COMMAND")
    return environment if environment and environment.strip() else None


def _load_config(path: Path | None = None) -> tuple[dict[str, Any], Path | None]:
    configured = path
    if configured is None:
        value = os.environ.get("LEAPTALK_CONFIG")
        configured = Path(value).expanduser().resolve() if value else None
    if configured is None:
        raise DependencyBlockedError(
            "LEAPTALK_CONFIG is required; a bare command/checkpoint is insufficient"
        )
    if not configured.is_file():
        raise DependencyBlockedError(f"LeapTalk model config is missing: {configured}")
    payload = read_json(configured)
    if not isinstance(payload, dict):
        raise DependencyBlockedError("LeapTalk model config must be a JSON object")
    return payload, configured


def discover(path: Path | None = None) -> dict[str, Any]:
    """Validate a complete LeapTalk binding and return immutable provenance."""

    try:
        payload, config_path = _load_config(path)
        root_value = payload.get("root") or os.environ.get("LEAPTALK_ROOT")
        if not isinstance(root_value, str) or not root_value:
            raise DependencyBlockedError("LeapTalk root is missing")
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise DependencyBlockedError(f"LeapTalk root is missing: {root}")
        url = payload.get("official_url")
        commit = payload.get("repo_commit")
        if not isinstance(url, str) or not url.startswith("https://"):
            raise DependencyBlockedError("LeapTalk official_url must be recorded")
        if not isinstance(commit, str) or len(commit) < 7 or commit == "UNKNOWN":
            raise DependencyBlockedError("LeapTalk repo_commit must be pinned")
        command = _command_from_config(payload)
        if command is None:
            raise DependencyBlockedError("LeapTalk adapter command is missing")
        runtime = payload.get("runtime")
        if not isinstance(runtime, Mapping):
            raise DependencyBlockedError("LeapTalk runtime settings are missing")
        missing_runtime = [key for key in REQUIRED_RUNTIME if key not in runtime or runtime[key] in (None, "", "UNKNOWN")]
        if missing_runtime:
            raise DependencyBlockedError(f"LeapTalk runtime settings are unknown: {missing_runtime}")
        raw_components = payload.get("components")
        if not isinstance(raw_components, Mapping):
            raise DependencyBlockedError("LeapTalk loaded components are missing")
        missing_components = [name for name in REQUIRED_COMPONENTS if name not in raw_components]
        if missing_components:
            raise DependencyBlockedError(f"LeapTalk required components are missing: {missing_components}")
        components = {name: _bind_component(name, raw_components[name], root) for name in REQUIRED_COMPONENTS}
        loaded = payload.get("loaded_components")
        if not isinstance(loaded, list) or sorted(str(item) for item in loaded) != sorted(REQUIRED_COMPONENTS):
            raise DependencyBlockedError("loaded_components does not prove the full LeapTalk stack")
        command_text = " ".join(command) if isinstance(command, list) else command
        return {
            "family": "LeapTalk",
            "status": "NEW_LEAPTALK_CONFIGURATION" if payload.get("historical_replication_verified") is not True else "HISTORICAL_REPLICATION_VERIFIED",
            "official_url": url,
            "root": str(root),
            "repo_commit": commit,
            "config_path": str(config_path),
            "config_sha256": file_sha256(config_path),
            "command": list(command) if isinstance(command, list) else command,
            "command_template_sha256": hashlib.sha256(command_text.encode()).hexdigest(),
            "components": components,
            "runtime": dict(runtime),
            "historical_replication_verified": bool(payload.get("historical_replication_verified", False)),
            "historical_evidence": payload.get("historical_evidence", "UNKNOWN"),
            "training_distribution": payload.get("training_distribution", "TRAINING_DISTRIBUTION_UNKNOWN"),
            "no_wav2lip_substitution": True,
        }
    except DependencyBlockedError:
        raise
    except (OSError, ProtocolError, TypeError, ValueError) as exc:
        raise DependencyBlockedError(f"LeapTalk provenance validation failed: {exc}") from exc


def request_payload(
    *,
    cell_id: str,
    role: str,
    sample_id: int,
    source: str,
    driver: str,
    seed: int,
    portrait: Mapping[str, Any],
    audio: Mapping[str, Any],
    config_hash: str,
    output_dir: Path,
) -> dict[str, Any]:
    if not isinstance(audio.get("path"), str) or not isinstance(audio.get("pcm_sha256"), str):
        raise ProtocolError("LeapTalk request requires a bound PCM path/hash")
    if not isinstance(portrait.get("path"), str) or not isinstance(portrait.get("file_sha256"), str):
        raise ProtocolError("LeapTalk request requires a bound portrait path/hash")
    audio_file_hash = audio.get("file_sha256")
    if not isinstance(audio_file_hash, str):
        raise ProtocolError("LeapTalk request requires the audio container hash")
    return {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "cell_id": cell_id,
        "role": role,
        "sample_id": int(sample_id),
        "source": source,
        "driver": driver,
        "seed": int(seed),
        "portrait": {"path": str(portrait["path"]), "sha256": str(portrait["file_sha256"])},
        "audio": {"path": str(audio["path"]), "file_sha256": audio_file_hash, "pcm_sha256": str(audio["pcm_sha256"]), "sample_rate": config.SAMPLE_RATE},
        "config_hash": config_hash,
        "output_dir": str(output_dir),
        "output_path": str(output_dir / "video.mp4"),
        "response_path": str(output_dir / "response.json"),
        "required_response_evidence": [
            "consumed_audio",
            "frontend",
            "rng_reset",
            "chunk_to_frame_timing",
            "loaded_components",
        ],
    }


def command_argv(command: str | Sequence[str], values: Mapping[str, Any]) -> list[str]:
    tokens = shlex.split(command) if isinstance(command, str) else list(command)
    try:
        return [token.format(**values) for token in tokens]
    except (KeyError, ValueError) as exc:
        raise ProtocolError(f"LeapTalk command template is invalid: {exc}") from exc


def validate_response(
    response_path: Path,
    request: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    response = read_self_hashed_json(response_path)
    if response.get("status") != "COMPLETE":
        raise ProtocolError(f"LeapTalk adapter response is incomplete: {response_path}")
    request_body = dict(request)
    request_body.pop("artifact_sha256", None)
    expected_request_hash = canonical_sha256(request_body)
    if response.get("request_sha256") != expected_request_hash:
        raise ProtocolError("LeapTalk response is bound to a different request")
    if response.get("config_sha256") != request.get("config_hash"):
        raise ProtocolError("LeapTalk response is bound to a different model configuration")
    if response.get("output_path") != request.get("output_path"):
        raise ProtocolError("LeapTalk response output path differs from the requested cell")
    consumed = response.get("consumed_audio")
    if not isinstance(consumed, Mapping) or consumed.get("path") != request["audio"]["path"] or consumed.get("file_sha256") != request["audio"].get("file_sha256") or consumed.get("pcm_sha256") != request["audio"]["pcm_sha256"]:
        raise ProtocolError("LeapTalk did not prove consumption of the requested PCM")
    frontend = response.get("frontend")
    if not isinstance(frontend, Mapping):
        raise ProtocolError("LeapTalk frontend consumption proof is missing")
    for key in ("path", "sha256", "tensor_sha256", "tensor_shape", "tensor_dtype", "pcm_sha256", "sample_start", "sample_end"):
        if key not in frontend or frontend[key] in (None, "", "UNKNOWN"):
            raise ProtocolError(f"LeapTalk frontend proof is missing {key}")
    if frontend.get("pcm_sha256") != request["audio"]["pcm_sha256"] or int(frontend["sample_start"]) < 0 or int(frontend["sample_end"]) <= int(frontend["sample_start"]):
        raise ProtocolError("LeapTalk frontend tensor is bound to another PCM")
    frontend_path = Path(str(frontend["path"]))
    if not frontend_path.is_file() or file_sha256(frontend_path) != str(frontend["sha256"]):
        raise ProtocolError("LeapTalk frontend artifact is missing or tampered")
    rng = response.get("rng_reset")
    required_rng_streams = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if not isinstance(rng, Mapping) or rng.get("all_streams_reset") is not True or not rng.get("initial_state_sha256") or not required_rng_streams.issubset(set(rng.get("streams", []))):
        raise ProtocolError("LeapTalk RNG reset evidence is incomplete")
    timing = response.get("chunk_to_frame_timing")
    if not isinstance(timing, Mapping) or timing.get("path") in (None, "", "UNKNOWN") or timing.get("mapping_sha256") in (None, "", "UNKNOWN") or timing.get("sha256") in (None, "", "UNKNOWN") or not timing.get("rows"):
        raise ProtocolError("LeapTalk chunk-to-frame timing evidence is missing")
    timing_path = Path(str(timing["path"]))
    if not timing_path.is_file() or file_sha256(timing_path) != str(timing["sha256"]):
        raise ProtocolError("LeapTalk chunk-to-frame timing artifact is missing or tampered")
    loaded = response.get("loaded_components")
    expected_loaded = sorted(str(key) for key in provenance.get("components", {}))
    if not isinstance(loaded, list) or sorted(str(item) for item in loaded) != expected_loaded:
        raise ProtocolError("LeapTalk response did not load the frozen component set")
    return response


def runtime_snapshot() -> dict[str, Any]:
    """Record the process runtime used by a preflight or adapter launcher."""

    try:
        import torch

        torch_version = torch.__version__
        cuda = torch.version.cuda
    except Exception:  # pragma: no cover - optional dependency in CPU tests  # noqa: BLE001
        torch_version = "UNAVAILABLE"
        cuda = "UNAVAILABLE"
    return {
        "python": sys.version.split()[0],
        "torch": torch_version,
        "cuda": cuda,
        "driver": _driver_snapshot(),
        "hardware": platform.platform(),
    }


def _driver_snapshot() -> dict[str, Any]:
    try:
        from .common import gpu_snapshot

        return gpu_snapshot()
    except Exception:  # pragma: no cover - diagnostic only  # noqa: BLE001
        return {"available": False, "error": "gpu snapshot unavailable"}


def write_request(path: Path, request: Mapping[str, Any]) -> dict[str, Any]:
    return write_self_hashed_json(path, dict(request))
