from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import yaml

from . import __version__
from .common import ProtocolError, canonical_json_sha256, file_sha256


RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,79}$")
PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[2]


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).resolve()
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ProtocolError("configuration must be a YAML object")
    root = config_path.parents[2]
    raw["config_path"] = str(config_path)
    raw["repo_root"] = str(root)
    paths = raw.get("paths")
    if not isinstance(paths, dict):
        raise ProtocolError("configuration is missing paths")
    for key, value in list(paths.items()):
        if value is not None:
            candidate = Path(str(value)).expanduser()
            if key in {"wav2lip_python", "syncnet_python", "mfa_python"}:
                # Keep the venv's bin/python symlink as the invocation path.
                # Resolving it can launch the base interpreter outside its venv.
                candidate = candidate if candidate.is_absolute() else root / candidate
                paths[key] = str(candidate.absolute())
            else:
                paths[key] = str((root / candidate).resolve() if not candidate.is_absolute() else candidate.resolve())
    validate_config(raw)
    return raw


def validate_config(config: dict[str, Any]) -> None:
    protocol = config.get("protocol")
    if protocol not in {"mfa_linear_video_retiming_v1", "mfa_linear_video_retiming_v2"}:
        raise ProtocolError("unsupported protocol name")
    if int(config.get("schema_version", 0)) != 1:
        raise ProtocolError("unsupported protocol schema_version")
    paths = config.get("paths", {})
    required = (
        "portrait_registry", "wav2lip_python", "wav2lip_root", "wav2lip_checkpoint",
        "syncnet_python", "syncnet_root", "syncnet_model", "ffmpeg", "ffprobe",
        "render_worker", "legacy_score_worker", "official_score_adapter",
    )
    required += (("mfa_summary", "clean_manifest", "mini_manifest", "legacy_eval_dir") if protocol.endswith("_v1")
                 else ("strict_source_manifest", "cohort_audio_root", "mfa_python"))
    missing = [key for key in required if not paths.get(key)]
    if missing:
        raise ProtocolError(f"missing required paths: {missing}")
    for key in (("wav2lip_python", "syncnet_python") if protocol.endswith("_v1") else
                ("wav2lip_python", "syncnet_python", "mfa_python")):
        executable = Path(str(paths[key]))
        if not executable.is_file() or not executable.stat().st_mode & 0o111:
            raise ProtocolError(f"model environment interpreter is missing or not executable: {key}:{executable}")
    ids = [str(value) for value in config.get("sample_ids", [])]
    if protocol.endswith("_v1"):
        expected_ids = ["1", "101", "201"]
    else:
        from .cohort_v2 import SAMPLE_IDS
        expected_ids = list(SAMPLE_IDS)
    if ids != expected_ids:
        raise ProtocolError(f"formal cohort must remain exactly {len(expected_ids)} fixed sample IDs")
    if config.get("portraits") != ["3", "6", "9"]:
        raise ProtocolError("portrait registry selection must remain exactly 3, 6, 9")
    search = config.get("search", {})
    if protocol.endswith("_v1"):
        if int(search.get("max_candidates", 0)) != 256 or float(search.get("max_seconds_per_sample", 0)) != 1200.0:
            raise ProtocolError("search budget differs from the frozen specification")
        if int(config.get("budget", {}).get("total_gpu_seconds", 0)) != 10800:
            raise ProtocolError("total GPU budget differs from the frozen specification")
    else:
        if search.get("engine") != "embedding_beam_v2" or search.get("proxy_anchor") != "window_center":
            raise ProtocolError("v2 search engine/anchor mismatch")
        if (int(search.get("proxy_unique_maps", 0)), int(search.get("true_candidate_forwards", 0)),
                float(search.get("max_seconds_per_sample", 0))) != (20000, 96, 1200.0):
            raise ProtocolError("v2 search budget differs from frozen specification")
        if (float(search.get("stage_a_wall_deadline_seconds", 0)),
                float(search.get("stage_b_wall_deadline_seconds", 0))) != (420.0, 880.0):
            raise ProtocolError("v2 stage time reservations differ from frozen specification")
        if int(config.get("budget", {}).get("total_gpu_seconds", 0)) != 129600:
            raise ProtocolError("v2 total GPU budget differs from frozen specification")


def validate_run_id(value: str) -> str:
    if not RUN_ID_RE.fullmatch(value):
        raise ProtocolError("run ID must contain 1-80 safe filename characters")
    return value


def code_bindings(package_root: Path = PACKAGE_ROOT) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    for path in sorted(package_root.glob("*.py")):
        result.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
    return result


def build_run_fingerprint(config: dict[str, Any], asset_sha256: str) -> tuple[str, dict[str, Any]]:
    paths = config["paths"]
    bindings = {
        "schema_version": 1,
        "protocol": config["protocol"],
        "package_version": __version__,
        "config": {key: value for key, value in config.items() if key not in ("config_path", "repo_root")},
        "asset_manifest_sha256": asset_sha256,
        "code": code_bindings(),
        "model_call_sources": [
            {"path": str(path.resolve()), "sha256": file_sha256(path)}
            for path in (
                Path(paths["wav2lip_root"]) / "audio.py",
                Path(paths["wav2lip_root"]) / "hparams.py",
                Path(paths["wav2lip_root"]) / "inference.py",
                Path(paths["wav2lip_root"]) / "models" / "__init__.py",
                Path(paths["wav2lip_root"]) / "models" / "wav2lip.py",
                Path(paths["wav2lip_root"]) / "models" / "conv.py",
                Path(paths["wav2lip_root"]) / "face_detection" / "api.py",
                Path(paths["syncnet_root"]) / "SyncNetModel.py",
                Path(paths["syncnet_root"]) / "SyncNetInstance.py",
                Path(paths["syncnet_root"]) / "run_pipeline.py",
                Path(paths["syncnet_root"]) / "run_syncnet.py",
            )
        ],
        "models": {
            "wav2lip_checkpoint_sha256": file_sha256(paths["wav2lip_checkpoint"]),
            "syncnet_model_sha256": file_sha256(paths["syncnet_model"]),
            "render_worker_sha256": file_sha256(paths["render_worker"]),
            "legacy_score_worker_sha256": file_sha256(paths["legacy_score_worker"]),
            "official_score_adapter_sha256": file_sha256(paths["official_score_adapter"]),
        },
    }
    return canonical_json_sha256(bindings), bindings
