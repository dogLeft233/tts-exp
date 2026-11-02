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
import tarfile
import time
import traceback
import urllib.request
import zipfile
from pathlib import Path
from typing import Any


BASE = Path("/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt02")
CODE_ROOT = Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_sources_attempt02")
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


def download_github_source(repo_id: str, code_path: Path, logs: Path) -> dict[str, Any]:
    owner, repo = repo_id.split("/", 1)
    request = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "cross-tfg-audio-head-deployer"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        repo_info = json.load(response)
    branch = repo_info["default_branch"]
    request = urllib.request.Request(
        f"https://api.github.com/repos/{owner}/{repo}/commits/{branch}",
        headers={"Accept": "application/vnd.github+json", "User-Agent": "cross-tfg-audio-head-deployer"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        commit_info = json.load(response)
    sha = commit_info["sha"]
    archive_path = logs / f"{repo}-{sha}.tar.gz"
    url = f"https://codeload.github.com/{owner}/{repo}/tar.gz/{sha}"
    with urllib.request.urlopen(url, timeout=60) as response, archive_path.open("wb") as stream:
        shutil.copyfileobj(response, stream, length=1 << 20)
    archive_sha = sha256_file(archive_path)
    code_path.mkdir(parents=True, exist_ok=False)
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        roots = {member.name.split("/", 1)[0] for member in members if "/" in member.name}
        if len(roots) != 1:
            raise RuntimeError(f"unexpected GitHub archive top-level entries: {sorted(roots)}")
        prefix = next(iter(roots)) + "/"
        base = code_path.resolve()
        for member in members:
            if not member.name.startswith(prefix):
                continue
            relative = Path(member.name[len(prefix):])
            if not relative.parts:
                continue
            target = (code_path / relative).resolve(strict=False)
            if not target.is_relative_to(base):
                raise RuntimeError(f"unsafe path in official GitHub archive: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RuntimeError(f"missing archive payload for {member.name}")
                with source, target.open("wb") as output:
                    shutil.copyfileobj(source, output, length=1 << 20)
                target.chmod(member.mode & 0o777)
    archive_path.unlink()
    return {"repo_id": repo_id, "url": f"https://github.com/{owner}/{repo}",
            "default_branch": branch, "git_head": sha,
            "archive_url": url, "archive_sha256": archive_sha,
            "path": str(code_path), "checkout_method": "official_codeload_at_resolved_commit"}


def ensure_download_tools_overlay(overlay: Path, logs: Path) -> dict[str, Any]:
    overlay.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {"overlay": str(overlay), "pip_bootstrap": None, "packages": []}
    bootstrap = overlay / "_pip_bootstrap"
    if importlib.util.find_spec("pip") is None:
        bootstrap.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request("https://pypi.org/pypi/pip/25.2/json",
                                         headers={"User-Agent": "cross-tfg-audio-head-deployer"})
        with urllib.request.urlopen(request, timeout=30) as response:
            metadata = json.load(response)
        wheel_info = next(item for item in metadata["urls"] if item["filename"] == "pip-25.2-py3-none-any.whl")
        wheel_path = bootstrap / wheel_info["filename"]
        with urllib.request.urlopen(wheel_info["url"], timeout=60) as response, wheel_path.open("wb") as stream:
            shutil.copyfileobj(response, stream, length=1 << 20)
        wheel_sha = sha256_file(wheel_path)
        if wheel_sha != wheel_info["digests"]["sha256"]:
            raise RuntimeError("official PyPI pip wheel SHA-256 mismatch")
        with zipfile.ZipFile(wheel_path) as wheel:
            wheel.extractall(bootstrap)
        status["pip_bootstrap"] = {"version": "25.2", "url": wheel_info["url"], "sha256": wheel_sha,
                                    "bootstrap_path": str(bootstrap)}
    env = dict(os.environ)
    env["PYTHONPATH"] = str(bootstrap) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    modules = (("huggingface_hub", "huggingface_hub==0.30.2"), ("gdown", "gdown==5.2.0"),
               ("bs4", "beautifulsoup4==4.13.4"), ("soupsieve", "soupsieve==2.7"))
    missing = [distribution for module, distribution in modules if importlib.util.find_spec(module) is None]
    if missing:
        argv = [str(TORCH_PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
                "--index-url", "https://pypi.org/simple", "--target", str(overlay), "--no-deps", *missing]
        result = run(argv, cwd=overlay, log_root=logs, name="install_download_only_overlay_extras", timeout=1200, env=env)
        status["packages"] = missing
        status["pip_install"] = result
        if result["returncode"] != 0:
            raise RuntimeError(f"download utility overlay install failed; see {result['stderr_path']}")
    status["pip_bootstrap"] = status["pip_bootstrap"] or {"status": "existing_pip_importable"}
    return status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("musetalk", "ditto"), required=True)
    parser.add_argument("--output-root", type=Path, default=BASE)
    parser.add_argument("--code-root", type=Path, default=CODE_ROOT)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    code_root = args.code_root.resolve()
    overlay = (code_root / "overlay" / f"download_tools_attempt02_{args.job}").resolve()
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
    logs.mkdir(parents=True, exist_ok=True)
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
            github_repo = "TMElyralab/MuseTalk"
            code_path = code_root / "MuseTalk"
            patterns = [
                ("TMElyralab/MuseTalk", ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"], target_weights),
                ("stabilityai/sd-vae-ft-mse", ["config.json", "diffusion_pytorch_model.bin"], target_weights / "sd-vae"),
                ("openai/whisper-tiny", ["config.json", "pytorch_model.bin", "preprocessor_config.json"], target_weights / "whisper"),
                ("yzd-v/DWPose", ["dw-ll_ucoco_384.pth"], target_weights / "dwpose"),
                ("ByteDance/LatentSync", ["latentsync_syncnet.pt"], target_weights / "syncnet"),
            ]
        else:
            github_repo = "antgroup/ditto-talkinghead"
            code_path = code_root / "ditto-talkinghead"
            patterns = [("digital-avatar/ditto-talkinghead",
                         ["ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", "ditto_pytorch/aux_models/**", "ditto_pytorch/models/**"],
                         target_weights)]
        if code_path.exists():
            raise FileExistsError(f"refusing to overwrite existing official source checkout: {code_path}")
        status["current_phase"] = "official_source_archive_download"
        atomic_json(status_path, status)
        source_info = download_github_source(github_repo, code_path, logs)
        status["steps"].append({"name": "official_codeload_at_resolved_commit", **source_info})
        status["official_source"] = source_info
        atomic_json(status_path, status)

        tool_overlay = ensure_download_tools_overlay(overlay, logs)
        status["steps"].append({"name": "isolated_download_tool_overlay", **tool_overlay,
                                "torch_requested": False, "numpy_requested": False})
        atomic_json(status_path, status)
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
