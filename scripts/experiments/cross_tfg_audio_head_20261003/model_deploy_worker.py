#!/usr/bin/env python3
"""Download official MuseTalk and Ditto inference code/checkpoints in isolation.

Workers never install or modify PyTorch. Small Hub/gdown utilities, if missing,
are installed into a task-specific overlay outside the existing Torch env.
Weights are written only to the task tmpfs namespace.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


BASE = Path("/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt01")
CODE_ROOT = Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_sources")
OVERLAY_ROOT = Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/overlay")
TORCH_PYTHON = Path("/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python")
HF_ENDPOINT = "https://hf-mirror.com"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def child_env(overlay: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["HF_ENDPOINT"] = HF_ENDPOINT
    env["HF_HUB_DISABLE_XET"] = "1"
    env["PYTHONPATH"] = str(overlay) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run(argv: list[str], *, cwd: Path | None, log_root: Path, name: str, timeout: int = 3600, env: dict[str, str] | None = None) -> dict[str, Any]:
    log_root.mkdir(parents=True, exist_ok=True)
    stdout = log_root / f"{name}.stdout.log"
    stderr = log_root / f"{name}.stderr.log"
    started = time.time()
    with stdout.open("wb") as out, stderr.open("wb") as err:
        try:
            p = subprocess.run(argv, cwd=str(cwd) if cwd else None, env=env, stdin=subprocess.DEVNULL, stdout=out, stderr=err, timeout=timeout, check=False)
            code = p.returncode
        except subprocess.TimeoutExpired:
            code = 124
    return {"argv": argv, "cwd": str(cwd) if cwd else None, "returncode": code,
            "stdout_path": str(stdout), "stdout_sha256": sha256_file(stdout), "stdout_bytes": stdout.stat().st_size,
            "stderr_path": str(stderr), "stderr_sha256": sha256_file(stderr), "stderr_bytes": stderr.stat().st_size,
            "started_unix": started, "ended_unix": time.time()}


def tree_manifest(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache/huggingface" not in path.as_posix():
            records.append({"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("musetalk", "ditto"), required=True)
    parser.add_argument("--output-root", type=Path, default=BASE)
    parser.add_argument("--code-root", type=Path, default=CODE_ROOT)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    code_root = args.code_root.resolve()
    overlay = (code_root / "overlay" / f"download_tools_{args.job}").resolve()
    job_root = output_root / args.job
    logs = job_root / "logs"
    status_path = job_root / "deployment_status.json"
    status: dict[str, Any] = {
        "schema_version": "cross_tfg_model_deployment_v1", "job": args.job,
        "status": "RUNNING", "started_unix": time.time(), "finished_unix": None,
        "exit_code": None, "official_source": None, "weights_root": None,
        "steps": [], "errors": [], "torch_modified": False,
        "storage": {"output_root": str(output_root), "bytes_free_at_start": shutil.disk_usage(output_root if output_root.exists() else output_root.parent).free},
    }
    atomic_json(status_path, status)
    try:
        if not TORCH_PYTHON.is_file():
            raise FileNotFoundError(f"expected existing Torch interpreter not found: {TORCH_PYTHON}")
        torch_before = subprocess.check_output([str(TORCH_PYTHON), "-c", "import torch;print(torch.__version__)"] , text=True).strip()
        status["torch_version_before"] = torch_before
        status["torch_interpreter"] = str(TORCH_PYTHON)
        status["hf_endpoint"] = HF_ENDPOINT
        status["xet_disabled"] = True
        code_root.mkdir(parents=True, exist_ok=True)
        target_weights = output_root / "models" / ("musetalk" if args.job == "musetalk" else "ditto_pytorch")
        target_weights.mkdir(parents=True, exist_ok=False)
        if args.job == "musetalk":
            source_url = "https://github.com/TMElyralab/MuseTalk.git"
            code_path = code_root / "MuseTalk"
            patterns = [
                ("TMElyralab/MuseTalk", ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"], target_weights),
                ("stabilityai/sd-vae-ft-mse", ["config.json", "diffusion_pytorch_model.bin"], target_weights / "sd-vae"),
                ("openai/whisper-tiny", ["config.json", "pytorch_model.bin", "preprocessor_config.json"], target_weights / "whisper"),
                ("yzd-v/DWPose", ["dw-ll_ucoco_384.pth"], target_weights / "dwpose"),
                ("ByteDance/LatentSync", ["latentsync_syncnet.pt"], target_weights / "syncnet"),
            ]
        else:
            source_url = "https://github.com/antgroup/ditto-talkinghead.git"
            code_path = code_root / "ditto-talkinghead"
            patterns = [("digital-avatar/ditto-talkinghead",
                         ["ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", "ditto_pytorch/aux_models/**", "ditto_pytorch/models/**"],
                         target_weights)]
        if code_path.exists():
            raise FileExistsError(f"refusing to overwrite existing official source checkout: {code_path}")
        clone = run(["git", "clone", "--depth", "1", source_url, str(code_path)], cwd=code_root, log_root=logs, name="official_git_clone", timeout=1200)
        status["steps"].append({"name": "official_git_clone", **clone})
        atomic_json(status_path, status)
        if clone["returncode"] != 0:
            raise RuntimeError(f"official GitHub clone failed; see {clone['stderr_path']}")
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=code_path, text=True).strip()
        status["official_source"] = {"url": source_url, "path": str(code_path), "git_head": head}

        overlay.mkdir(parents=True, exist_ok=True)
        missing = []
        for module, distribution in (("huggingface_hub", "huggingface_hub==0.30.2"), ("gdown", "gdown")):
            if importlib.util.find_spec(module) is None:
                missing.append(distribution)
        if missing:
            packages = run([str(TORCH_PYTHON), "-m", "pip", "install", "--target", str(overlay), "--no-deps", *missing],
                           cwd=code_path, log_root=logs, name="install_download_only_overlay_extras", timeout=1200)
            status["steps"].append({"name": "install_download_only_overlay_extras", **packages,
                                    "packages": missing, "torch_requested": False})
            atomic_json(status_path, status)
            if packages["returncode"] != 0:
                raise RuntimeError(f"download utility overlay install failed; see {packages['stderr_path']}")
        sys.path.insert(0, str(overlay))
        from huggingface_hub import HfApi, snapshot_download
        import gdown  # noqa: F401 - preflight the Google Drive utility before a background fetch.
        env = child_env(overlay)
        resolved_revisions = {}
        for repo_id, allow_patterns, local_dir in patterns:
            local_dir.mkdir(parents=True, exist_ok=True)
            sha = HfApi(endpoint=HF_ENDPOINT).model_info(repo_id, revision="main").sha
            resolved_revisions[repo_id] = sha
            status["current_hf_repo"] = repo_id
            status["hf_revisions"] = resolved_revisions
            atomic_json(status_path, status)
            started = time.time()
            downloaded = snapshot_download(repo_id=repo_id, repo_type="model", revision=sha,
                                           local_dir=str(local_dir), allow_patterns=allow_patterns,
                                           endpoint=HF_ENDPOINT)
            status["steps"].append({"name": "snapshot_download", "repo_id": repo_id,
                                    "revision": sha, "allow_patterns": allow_patterns,
                                    "local_dir": str(local_dir), "snapshot_path": str(downloaded),
                                    "started_unix": started, "ended_unix": time.time(), "returncode": 0})
            atomic_json(status_path, status)

        if args.job == "musetalk":
            face_dir = target_weights / "face-parse-bisent"
            face_dir.mkdir(parents=True, exist_ok=True)
            gdown_env = child_env(overlay)
            face_download = run([str(TORCH_PYTHON), "-m", "gdown", "--id", "154JgKpzCPW82qINcVieuPH3fZ2e0P812", "-O", str(face_dir / "79999_iter.pth")],
                                cwd=code_path, log_root=logs, name="download_face_parse_bisent", timeout=1800, env=gdown_env)
            status["steps"].append({"name": "download_face_parse_bisent", **face_download})
            atomic_json(status_path, status)
            if face_download["returncode"] != 0:
                raise RuntimeError(f"official gdown face parse weight download failed; see {face_download['stderr_path']}")
            resnet = run(["curl", "-fL", "--retry", "5", "--continue-at", "-", "https://download.pytorch.org/models/resnet18-5c106cde.pth", "-o", str(face_dir / "resnet18-5c106cde.pth")],
                         cwd=code_path, log_root=logs, name="download_resnet18", timeout=1200, env=child_env(overlay))
            status["steps"].append({"name": "download_resnet18", **resnet})
            atomic_json(status_path, status)
            if resnet["returncode"] != 0:
                raise RuntimeError(f"resnet18 weight download failed; see {resnet['stderr_path']}")
            link = code_path / "models"
            if link.exists() or link.is_symlink():
                raise FileExistsError(f"refusing to replace MuseTalk models path: {link}")
            os.symlink(target_weights, link, target_is_directory=True)
            status["models_symlink"] = {"path": str(link), "target": str(target_weights)}

        status["weights_root"] = str(target_weights)
        status["hf_revisions"] = resolved_revisions
        status["weight_files"] = tree_manifest(target_weights)
        torch_after = subprocess.check_output([str(TORCH_PYTHON), "-c", "import torch;print(torch.__version__)"] , text=True).strip()
        status["torch_version_after"] = torch_after
        if torch_after != torch_before:
            raise RuntimeError(f"Torch changed unexpectedly: {torch_before} -> {torch_after}")
        status["torch_modified"] = False
        status["status"] = "DOWNLOAD_COMPLETE"
        status["exit_code"] = 0
    except BaseException as exc:
        status["status"] = "DOWNLOAD_FAILED"
        status["exit_code"] = 1
        status["errors"].append({"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    status["finished_unix"] = time.time()
    status["storage"]["bytes_free_at_end"] = shutil.disk_usage(output_root).free if output_root.exists() else None
    atomic_json(status_path, status)
    print(json.dumps({"job": args.job, "status": status["status"], "exit_code": status["exit_code"], "status_path": str(status_path)}, ensure_ascii=False, sort_keys=True))
    return int(status["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
