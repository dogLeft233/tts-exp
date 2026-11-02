#!/usr/bin/env python3
"""Resume pinned model-weight downloads from attempt02's official source trees.

This worker installs only download clients into a new per-model overlay. It
reuses the exact resolved source archives from attempt02 and never modifies
the existing Torch environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any


OUT = Path("/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt03")
CODE02 = Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_sources_attempt02")
CODE03 = Path("/root/tts-research-system-20261001/cross_tfg_audio_head_20261003/model_sources_attempt03")
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


def tree_manifest(root: Path) -> list[dict[str, Any]]:
    rows = []
    for p in sorted(root.rglob("*")):
        if p.is_file() and ".cache/huggingface" not in p.as_posix():
            rows.append({"path": str(p), "bytes": p.stat().st_size, "sha256": sha256_file(p)})
    return rows


def pip_env(bootstrap: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(bootstrap) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


def run_logged(argv: list[str], *, cwd: Path, env: dict[str, str], logs: Path, name: str, timeout: int = 3600) -> dict[str, Any]:
    logs.mkdir(parents=True, exist_ok=True)
    out = logs / f"{name}.stdout.log"
    err = logs / f"{name}.stderr.log"
    start = time.time()
    with out.open("wb") as stdout, err.open("wb") as stderr:
        try:
            result = subprocess.run(argv, cwd=str(cwd), env=env, stdin=subprocess.DEVNULL,
                                    stdout=stdout, stderr=stderr, timeout=timeout, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    return {"argv": argv, "cwd": str(cwd), "returncode": code,
            "stdout_path": str(out), "stdout_bytes": out.stat().st_size, "stdout_sha256": sha256_file(out),
            "stderr_path": str(err), "stderr_bytes": err.stat().st_size, "stderr_sha256": sha256_file(err),
            "started_unix": start, "ended_unix": time.time()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=("musetalk", "ditto"), required=True)
    args = parser.parse_args()
    outroot = OUT / args.job
    logs = outroot / "logs"
    status_path = outroot / "deployment_status.json"
    outroot.mkdir(parents=True, exist_ok=False)
    logs.mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {
        "schema_version": "cross_tfg_model_deployment_v1", "attempt": "attempt03_resume_weights",
        "job": args.job, "status": "RUNNING", "started_unix": time.time(), "finished_unix": None,
        "exit_code": None, "hf_endpoint": HF_ENDPOINT, "torch_interpreter": str(TORCH_PYTHON),
        "torch_modified": False, "source_reused_from_attempt02": True,
        "steps": [], "errors": [], "weight_files": [],
        "storage": {"output_root": str(OUT), "bytes_free_at_start": shutil.disk_usage(OUT if OUT.exists() else OUT.parent).free},
    }
    atomic_json(status_path, status)
    try:
        source_status = json.loads((Path("/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt02") / args.job / "deployment_status.json").read_text(encoding="utf-8"))
        if source_status.get("official_source", {}).get("checkout_method") != "official_codeload_at_resolved_commit":
            raise RuntimeError("attempt02 did not preserve an official source archive at its resolved commit")
        source = Path(source_status["official_source"]["path"]).resolve(strict=True)
        if not source.is_dir():
            raise FileNotFoundError(f"attempt02 pinned source checkout missing: {source}")
        status["official_source"] = source_status["official_source"]
        status["torch_version_before"] = subprocess.check_output([str(TORCH_PYTHON), "-c", "import torch;print(torch.__version__)"], text=True).strip()
        status["source_reuse"] = {"attempt02_status_path": str(Path("/dev/shm/cross_tfg_audio_head_20261003/model_deploy_attempt02") / args.job / "deployment_status.json"),
                                  "source_path": str(source), "source_git_head": source_status["official_source"]["git_head"],
                                  "source_archive_sha256": source_status["official_source"]["archive_sha256"]}
        atomic_json(status_path, status)

        old_overlay = CODE02 / "overlay" / f"download_tools_attempt02_{args.job}"
        bootstrap = old_overlay / "_pip_bootstrap"
        if not (bootstrap / "pip").is_dir():
            raise FileNotFoundError(f"attempt02's verified pip bootstrap is missing: {bootstrap}")
        overlay = CODE03 / "overlay" / f"download_tools_attempt03_{args.job}"
        if overlay.exists():
            raise FileExistsError(f"refusing to overwrite attempt03 download overlay: {overlay}")
        overlay.mkdir(parents=True)
        packages = ["huggingface_hub==0.30.2", "gdown==5.2.0", "beautifulsoup4==4.13.4", "soupsieve==2.7", "PyYAML==6.0.2"]
        pip_result = run_logged([str(TORCH_PYTHON), "-m", "pip", "install", "--disable-pip-version-check",
                                 "--index-url", "https://pypi.org/simple", "--target", str(overlay), *packages],
                                cwd=overlay, env=pip_env(bootstrap), logs=logs, name="install_resume_download_tools", timeout=1200)
        status["steps"].append({"name": "attempt03_isolated_download_overlay", "packages": packages,
                                "pip_bootstrap_reused_from_attempt02": str(bootstrap), "torch_requested": False,
                                "numpy_requested": False, **pip_result})
        atomic_json(status_path, status)
        if pip_result["returncode"] != 0:
            raise RuntimeError(f"isolated download client install failed; see {pip_result['stderr_path']}")

        env = dict(os.environ)
        env["HF_ENDPOINT"] = HF_ENDPOINT
        env["HF_HUB_DISABLE_XET"] = "1"
        env["PYTHONPATH"] = str(overlay) + os.pathsep + str(bootstrap) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        sys.path.insert(0, str(overlay))
        from huggingface_hub import HfApi, snapshot_download
        import gdown  # noqa: F401
        import yaml  # noqa: F401
        preflight = subprocess.check_output([str(TORCH_PYTHON), "-c",
            "import yaml,requests,filelock,packaging,tqdm,fsspec,typing_extensions; from huggingface_hub import HfApi,snapshot_download; import gdown; print('download_dependencies_ok')"],
            env=env, text=True).strip()
        status["steps"].append({"name": "download_dependency_preflight", "result": preflight,
                                "torch_requested": False, "numpy_requested": False})
        atomic_json(status_path, status)

        target = OUT / "models" / ("musetalk" if args.job == "musetalk" else "ditto_pytorch")
        target.mkdir(parents=True, exist_ok=False)
        if args.job == "musetalk":
            repos = [("TMElyralab/MuseTalk", ["musetalkV15/musetalk.json", "musetalkV15/unet.pth"], target),
                     ("stabilityai/sd-vae-ft-mse", ["config.json", "diffusion_pytorch_model.bin"], target / "sd-vae"),
                     ("openai/whisper-tiny", ["config.json", "pytorch_model.bin", "preprocessor_config.json"], target / "whisper"),
                     ("yzd-v/DWPose", ["dw-ll_ucoco_384.pth"], target / "dwpose"),
                     ("ByteDance/LatentSync", ["latentsync_syncnet.pt"], target / "syncnet")]
        else:
            repos = [("digital-avatar/ditto-talkinghead",
                      ["ditto_cfg/v0.4_hubert_cfg_pytorch.pkl", "ditto_pytorch/aux_models/**", "ditto_pytorch/models/**"], target)]
        revisions: dict[str, str] = {}
        for repo_id, patterns, dest in repos:
            dest.mkdir(parents=True, exist_ok=True)
            revision = HfApi(endpoint=HF_ENDPOINT).model_info(repo_id, revision="main").sha
            revisions[repo_id] = revision
            status.update({"current_hf_repo": repo_id, "hf_revisions": revisions, "weights_root": str(target)})
            atomic_json(status_path, status)
            started = time.time()
            result_dir = snapshot_download(repo_id=repo_id, repo_type="model", revision=revision,
                                           local_dir=str(dest), allow_patterns=patterns, endpoint=HF_ENDPOINT)
            status["steps"].append({"name": "snapshot_download", "repo_id": repo_id, "revision": revision,
                                    "allow_patterns": patterns, "local_dir": str(dest), "snapshot_path": str(result_dir),
                                    "started_unix": started, "ended_unix": time.time(), "returncode": 0})
            atomic_json(status_path, status)

        if args.job == "musetalk":
            face_dir = target / "face-parse-bisent"
            face_dir.mkdir()
            face = run_logged([str(TORCH_PYTHON), "-m", "gdown", "--id", "154JgKpzCPW82qINcVieuPH3fZ2e0P812", "-O", str(face_dir / "79999_iter.pth")],
                              cwd=source, env=env, logs=logs, name="download_face_parse_bisent", timeout=1800)
            status["steps"].append({"name": "download_face_parse_bisent", **face})
            atomic_json(status_path, status)
            if face["returncode"] != 0:
                raise RuntimeError(f"MuseTalk face-parse weight download failed; see {face['stderr_path']}")
            resnet_path = face_dir / "resnet18-5c106cde.pth"
            resnet = run_logged(["curl", "-fL", "--retry", "5", "--continue-at", "-",
                                 "https://download.pytorch.org/models/resnet18-5c106cde.pth", "-o", str(resnet_path)],
                                cwd=source, env=env, logs=logs, name="download_resnet18", timeout=1200)
            status["steps"].append({"name": "download_resnet18", **resnet})
            atomic_json(status_path, status)
            if resnet["returncode"] != 0:
                raise RuntimeError(f"MuseTalk resnet18 weight download failed; see {resnet['stderr_path']}")

        status["weight_files"] = tree_manifest(target)
        status["hf_revisions"] = revisions
        status["torch_version_after"] = subprocess.check_output([str(TORCH_PYTHON), "-c", "import torch;print(torch.__version__)"], text=True).strip()
        if status["torch_version_after"] != status["torch_version_before"]:
            raise RuntimeError("the existing Torch environment changed during model download")
        status["torch_modified"] = False
        status["weights_root"] = str(target)
        status["status"] = "DOWNLOAD_COMPLETE"
        status["exit_code"] = 0
    except BaseException as exc:
        status["status"] = "DOWNLOAD_FAILED"
        status["exit_code"] = 1
        status["errors"].append({"error_type": type(exc).__name__, "error": str(exc), "traceback": traceback.format_exc()})
    status["finished_unix"] = time.time()
    status["storage"]["bytes_free_at_end"] = shutil.disk_usage(OUT).free
    atomic_json(status_path, status)
    print(json.dumps({"job": args.job, "status": status["status"], "exit_code": status["exit_code"],
                      "status_path": str(status_path), "weights_root": status.get("weights_root")}, sort_keys=True))
    return int(status["exit_code"])


if __name__ == "__main__":
    raise SystemExit(main())
