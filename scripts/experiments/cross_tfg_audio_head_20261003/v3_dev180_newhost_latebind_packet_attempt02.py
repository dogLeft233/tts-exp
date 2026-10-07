#!/usr/bin/env python3
"""Late-bind preflight, execute, and collect the fixed step-120 Dev180 run.

This packet binds the actual METHOD_CANDIDATE config and materializer receipt
to the sealed 180-row plan. It reuses the reviewed V3 dispatcher, Stage-A
renderer/scorer, and 540-row evidence joiner. It never calls ASR, reads
confirmation inputs, or implements audio/video generation or scoring.

All host paths and resource thresholds are explicit inputs. `run` requires
`--execute-dev180` plus a preflight attestation made on the same host. `collect`
copies hash-verified Stage-A raw score matrices into the joiner's required
content-addressed asset layout, then invokes the reviewed joiner.

This attempt also gates RAM against both host MemAvailable and the active
process cgroup's finite memory limit, so large host RAM cannot mask a smaller
container limit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import posixpath
from pathlib import Path
import platform
import re
import stat
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[3]
PLAN_BUILDER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/build_v3_materialized_dev180_plan_attempt01.py"
PLAN_BUILDER_BYTES = 17_392
PLAN_BUILDER_SHA256 = "b236b73dc570214588200744516dc90732d06ca0c821e368ac253570a76bc633"
QUALIFIED_BUILDER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/build_v3_candidate_only_dev_plan_attempt03.py"
STAGEA_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_attempt03.py"
STAGEA_BYTES = 67_141
STAGEA_SHA256 = "d12c863ffcd5eacde7b0d09b0abc097429d0318ccea3b059d022d5b2e52e2a94"
CROSS_ADAPTER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt03.py"
CROSS_ADAPTER_BYTES = 7_505
CROSS_ADAPTER_SHA256 = "7f78a14485eebaa20e7afd893a46b2ead26db5544194d988d7d9890d6a43ebbc"
V3_DISPATCHER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_v3_candidate_attempt01.py"
V3_DISPATCHER_BYTES = 12_597
V3_DISPATCHER_SHA256 = "228d465dc2091d728e172787d77ca0fbef21cd08f5e67a08cc09c66078d95cac"
JOINER_REL = "scripts/experiments/cross_tfg_audio_head_20261003/join_v3_candidate_540_dev_attempt02.py"
JOINER_BYTES = 59_742
JOINER_SHA256 = "6bc65f80a2a3b2159ea38d9b93f6b704472236d0eb4da14ef52abc96bbd64e63"
PCM16_REL = "scripts/experiments/cross_tfg_audio_head_20261003/stagea_pcm16_lineage.py"
PCM16_BYTES = 4_528
PCM16_SHA256 = "f634bad9c287a2f7f8cd8ffa62674bdd9e97dbf4b0f00dbf44725e4ca4ca36a0"
BASELINE_ARCHIVE_BYTES = 18_784_301
BASELINE_ARCHIVE_SHA256 = "349c5b62f09c1a775013039d2abb3bd72d7c6b6233531fb7f87e6398fff981d4"
MODELS = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
METHOD_ID = "shared_recipient_transition_head_v3_fit40_step120"
SEEDS = (20261003, 20261004, 20261005)
SCHEMA = "cross_tfg_v3_dev180_newhost_latebind_attestation_v2"
PYTHON_DISTRIBUTIONS = (
    "numpy", "torch", "torchvision", "torchaudio", "scipy", "librosa",
    "opencv-python", "opencv-python-headless", "transformers", "diffusers",
    "onnxruntime", "onnxruntime-gpu", "python-speech-features", "soundfile",
)


def sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def pretty(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                       allow_nan=False) + "\n").encode("utf-8")


def payload_sha(value: Mapping[str, Any], field: str) -> str:
    return sha_bytes(canonical({key: item for key, item in value.items() if key != field}))


def read_regular(path: Path, label: str, limit: int = 32 * 1024 * 1024) -> bytes:
    lexical = Path(os.path.abspath(os.fspath(path)))
    info = lexical.stat(follow_symlinks=True)
    if not stat.S_ISREG(info.st_mode) or info.st_size < 0 or info.st_size > limit:
        raise ValueError(f"{label} is not a bounded regular file: {lexical}")
    raw = lexical.read_bytes()
    after = lexical.stat(follow_symlinks=True)
    if (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or len(raw) != info.st_size:
        raise ValueError(f"{label} changed while it was read")
    return raw


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, label: str, expected_sha256: str | None = None) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"{label} is not a regular file: {resolved}")
    digest = sha_file(resolved)
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(f"{label} SHA-256 differs from its frozen pin: {resolved}")
    return {"path": str(resolved), "bytes": info.st_size, "sha256": digest}


def directory_identity(path: Path, label: str) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    info = resolved.stat()
    if not stat.S_ISDIR(info.st_mode):
        raise ValueError(f"{label} is not a directory: {resolved}")
    return {"path": str(resolved), "device": info.st_dev, "inode": info.st_ino,
            "mtime_ns": info.st_mtime_ns}


def path_from_root(root: Path, value: str, label: str) -> Path:
    candidate = Path(value)
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate


def load_json(path: Path, label: str, limit: int = 32 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    raw = read_regular(path, label, limit)
    value = json.loads(raw.decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return value, raw


def load_bindings(path: Path) -> tuple[dict[str, Any], bytes]:
    bindings, raw = load_json(path, "runtime bindings", 2 * 1024 * 1024)
    forbidden = {"token", "password", "secret", "api_key", "access_key", "private_key"}

    def reject_credentials(value: Any, trail: str = "runtime_bindings") -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                key_text = str(key).lower().replace("-", "_")
                if any(part in key_text for part in forbidden):
                    raise ValueError(f"runtime bindings must not contain credential fields ({trail}.{key})")
                reject_credentials(child, f"{trail}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                reject_credentials(child, f"{trail}[{index}]")

    reject_credentials(bindings)
    required = (
        "project_root", "stagea_python", "stagea_core", "cross_frontend_adapter_runner",
        "v3_dispatcher", "v3_cli", "generator_child", "wav2lip_python", "wav2lip_root",
        "syncnet_root", "syncnet_python", "syncnet_checkpoint", "syncnet_evaluator",
        "ffmpeg", "ffprobe", "pcm16_lineage", "baseline_evidence_archive", "output_root",
    )
    missing = [key for key in required if not isinstance(bindings.get(key), str) or not bindings[key].strip()]
    if missing:
        raise ValueError(f"runtime bindings are missing explicit path fields: {', '.join(missing)}")
    return bindings, raw


def _load_materialization_validator(project_root: Path):
    wrapper_path = project_root / PLAN_BUILDER_REL
    file_identity(wrapper_path, "materialized Dev plan builder", PLAN_BUILDER_SHA256)
    spec = importlib.util.spec_from_file_location("latebind_materialized_dev180_builder", wrapper_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the pinned materialized Dev plan validator")
    wrapper = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = wrapper
    spec.loader.exec_module(wrapper)
    qualified_path = project_root / QUALIFIED_BUILDER_REL
    builder_path, _builder_raw, builder = wrapper._load_builder(project_root)
    if builder_path.resolve(strict=True) != qualified_path.resolve(strict=True):
        raise RuntimeError("materialized Dev validator loaded an unexpected qualified builder")
    return wrapper, builder


def validate_frozen_inputs(project_root: Path, plan_path: Path, rows_path: Path,
                           plan_manifest_path: Path, config_path: Path,
                           receipt_path: Path, baseline_archive: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    plan, plan_raw = load_json(plan_path, "frozen Dev plan", 32 * 1024 * 1024)
    rows_raw = read_regular(rows_path, "frozen Dev planned rows", 32 * 1024 * 1024)
    manifest, manifest_raw = load_json(plan_manifest_path, "Dev plan manifest", 2 * 1024 * 1024)
    expected_plan_sha = payload_sha(plan, "plan_sha256")
    if (plan.get("status") != "FROZEN" or plan.get("plan_sha256") != expected_plan_sha
            or plan.get("run_id") != "cross_tfg_audio_head_20261003_v3_fit40_dev180_attempt02"):
        raise ValueError("plan status, canonical seal, or fixed run identity is invalid")
    if (manifest.get("schema") != "cross_tfg_v3_candidate_only_dev_plan_v1"
            or manifest.get("manifest_sha256") != payload_sha(manifest, "manifest_sha256")
            or manifest.get("plan_sha256") != plan["plan_sha256"]
            or manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or manifest.get("planned_rows_sha256") != sha_bytes(rows_raw)
            or manifest.get("planned_rows") != 180):
        raise ValueError("plan manifest seal or plan/row byte bindings are invalid")
    if manifest.get("confirm_audio_read") is not False or manifest.get("confirm_effects_read") is not False:
        raise ValueError("frozen plan does not certify closed confirmation inputs")
    rows = [json.loads(line) for line in rows_raw.decode("utf-8").splitlines() if line.strip()]
    plan_rows = plan.get("rows")
    if not isinstance(plan_rows, list) or len(rows) != 180 or len(plan_rows) != 180:
        raise ValueError("candidate Dev plan must contain exactly 180 rows")
    if rows_raw != b"".join(canonical(row) + b"\n" for row in plan_rows):
        raise ValueError("planned-row ledger is not the canonical ordered view of the frozen plan")
    if [row.get("row_id") for row in rows] != [row.get("row_id") for row in plan_rows]:
        raise ValueError("candidate row order differs from the frozen plan")
    if tuple(model.get("model_id") for model in plan.get("models", [])) != MODELS:
        raise ValueError("candidate model order differs from the fixed Wav2Lip/MuseTalk/Ditto roster")
    if tuple(method.get("method_id") for method in plan.get("methods", [])) != (METHOD_ID,):
        raise ValueError("candidate method is not the fixed terminal step-120 method")
    if plan.get("generation_seed_family", {}).get("dev") != list(SEEDS):
        raise ValueError("candidate plan does not retain the exact three frozen Dev seeds")
    if plan.get("execution", {}).get("confirm_audio_read") is not False or plan.get("execution", {}).get("confirm_effects_read") is not False:
        raise ValueError("candidate execution metadata does not certify closed confirmation inputs")
    keys = {(row.get("sample_id"), row.get("speaker_id"), row.get("model_id"), row.get("seed")) for row in rows}
    samples = {row.get("sample_id") for row in rows}
    speakers = {row.get("speaker_id") for row in rows}
    counts = {model_id: sum(row.get("model_id") == model_id for row in rows) for model_id in MODELS}
    expected = {(sample, next(row["speaker_id"] for row in rows if row["sample_id"] == sample), model_id, seed)
                for sample in samples for model_id in MODELS for seed in SEEDS}
    if len(keys) != 180 or keys != expected or len(samples) != 20 or len(speakers) != 10 or any(count != 60 for count in counts.values()):
        raise ValueError("candidate Dev row grid is not 20 clips × 3 seeds × 3 models")
    if any(row.get("split") != "dev" or row.get("method_id") != METHOD_ID for row in rows):
        raise ValueError("candidate row contains a non-Dev or non-candidate method")
    control = plan.get("execution", {}).get("baseline_controls", {})
    if (control.get("control_rows", {}).get("identity") != 180
            or control.get("control_rows", {}).get("rms_fixed") != 180
            or control.get("archive_raw_sha256") != BASELINE_ARCHIVE_SHA256
            or control.get("archive_bytes") != BASELINE_ARCHIVE_BYTES):
        raise ValueError("frozen plan does not bind the exact protected 540-row control archive")
    baseline_pin = file_identity(baseline_archive, "protected 540-row baseline archive", BASELINE_ARCHIVE_SHA256)
    if baseline_pin["bytes"] != BASELINE_ARCHIVE_BYTES:
        raise ValueError("protected baseline archive byte count differs from the frozen evidence")

    refs = plan.get("source_reference_plans", {})
    config_ref = refs.get("v3_cli_config", {})
    receipt_ref = refs.get("materialization_receipt", {})
    config_path = config_path.resolve(strict=True)
    receipt_path = receipt_path.resolve(strict=True)
    config_expected_path = path_from_root(project_root, str(config_ref.get("path", "")), "candidate config reference").resolve(strict=True)
    receipt_expected_path = path_from_root(project_root, str(receipt_ref.get("path", "")), "materializer receipt reference").resolve(strict=True)
    if config_expected_path != config_path or receipt_expected_path != receipt_path:
        raise ValueError("supplied candidate config/receipt paths differ from the sealed plan references")
    wrapper, builder = _load_materialization_validator(project_root)
    actual = wrapper.validate_actual_materialization(project_root, config_path, receipt_path, builder)
    if (actual.get("config_raw_sha256") != config_ref.get("raw_sha256")
            or actual.get("receipt_raw_sha256") != receipt_ref.get("raw_sha256")
            or actual.get("checkpoint_sha256") != manifest.get("candidate_checkpoint_sha256")
            or actual.get("checkpoint_sha256") != plan.get("execution", {}).get("candidate_checkpoint_sha256")
            or actual.get("checkpoint_sha256") != plan.get("methods", [{}])[0].get("checkpoint_sha256")
            or actual.get("optimizer_steps") != 120):
        raise ValueError("actual step-120 config/receipt does not match the sealed plan and materializer manifest")
    return {
        "project_root": str(project_root),
        "plan": {"path": str(plan_path.resolve(strict=True)), "bytes": len(plan_raw), "raw_sha256": sha_bytes(plan_raw),
                 "plan_sha256": plan["plan_sha256"], "run_id": plan["run_id"], "planned_rows": 180,
                 "row_ledger_raw_sha256": sha_bytes(rows_raw), "manifest_raw_sha256": sha_bytes(manifest_raw)},
        "candidate_config": {"path": str(config_path), "raw_sha256": actual["config_raw_sha256"]},
        "materialization_receipt": {"path": str(receipt_path), "raw_sha256": actual["receipt_raw_sha256"],
                                    "optimizer_steps": actual["optimizer_steps"],
                                    "candidate_checkpoint_sha256": actual["checkpoint_sha256"]},
        "baseline_archive": baseline_pin,
        "row_counts_by_model": counts,
        "confirm_audio_read": False,
        "confirm_effects_read": False,
    }


def model_file_pins(models: Iterable[Mapping[str, Any]]) -> list[dict[str, str]]:
    found: dict[str, str] = {}

    def add(value: str, digest: str, label: str, roots: list[str]) -> None:
        candidate = Path(value)
        if not candidate.is_absolute():
            if not roots:
                return
            candidate = Path(roots[0]) / candidate
        key = str(candidate)
        prior = found.get(key)
        if prior is not None and prior != digest:
            raise ValueError(f"model dependency has conflicting frozen digests: {key}")
        found[key] = digest

    def scan(node: Any, roots: list[str], trail: str = "") -> None:
        if isinstance(node, Mapping):
            local_roots = list(roots)
            for key in ("model_source_root", "ditto_source_root", "model_code_root", "model_workspace", "ditto_weights_root", "repo"):
                value = node.get(key)
                if isinstance(value, str) and value:
                    local_roots.insert(0, value)
            for path_key in ("path", "adapter_path", "musetalk_adapter_path", "overdub_helper_path",
                             "renderer_path", "checkpoint", "ditto_config_path", "config_path",
                             "qualification_report_path", "header_report_path", "import_report_path"):
                if isinstance(node.get(path_key), str):
                    hash_candidates = [
                        f"{path_key}_sha256",
                        path_key.removesuffix("_path") + "_sha256",
                        "file_sha256", "raw_sha256", "sha256",
                    ]
                    for hash_key in hash_candidates:
                        digest = node.get(hash_key)
                        if is_sha256(digest):
                            add(node[path_key], digest.lower(), f"{trail}.{path_key}", local_roots)
                            break
            for key, child in node.items():
                if key in {"source_sha256", "face_parser_source_sha256"} and isinstance(child, Mapping):
                    root = next((node.get(name) for name in ("model_source_root", "ditto_source_root", "model_code_root")
                                 if isinstance(node.get(name), str)), roots[0] if roots else None)
                    for relative, record in child.items():
                        if not isinstance(relative, str):
                            continue
                        if isinstance(record, Mapping):
                            value = record.get("path", relative)
                            digest = record.get("sha256") or record.get("file_sha256")
                            candidate_roots = local_roots
                        else:
                            value = relative
                            digest = record
                            candidate_roots = [root] if isinstance(root, str) else local_roots
                        if isinstance(value, str) and is_sha256(digest):
                            add(value, digest.lower(), f"{trail}.{key}", candidate_roots)
                elif key not in {"env", "rows", "natural_pcm", "source_face", "source_video"}:
                    scan(child, local_roots, f"{trail}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                scan(child, roots, f"{trail}[{index}]")

    for model in models:
        generator = model.get("generator", {})
        roots = []
        if isinstance(generator, Mapping):
            for key in ("model_source_root", "ditto_source_root", "model_code_root",
                        "model_workspace", "ditto_weights_root", "repo"):
                value = generator.get(key)
                if isinstance(value, str) and value:
                    roots.append(value)
        scan(model, roots, str(model.get("model_id", "model")))
    return [{"path": path, "sha256": digest} for path, digest in sorted(found.items())]


def probe_python(path_text: str) -> dict[str, Any]:
    # Keep the venv launcher path for invocation. Resolving a POSIX venv's
    # `bin/python` symlink first can launch the system interpreter instead.
    executable = Path(os.path.abspath(path_text))
    if not executable.is_file():
        raise ValueError(f"Python executable is absent: {executable}")
    identity = file_identity(executable, "Python interpreter")
    code = (
        "import importlib.metadata as m, json, platform, sys\n"
        f"wanted = {list(PYTHON_DISTRIBUTIONS)!r}\n"
        "installed = {name: m.version(name) for name in wanted if any(d.metadata.get('Name','').lower()==name.lower() for d in m.distributions())}\n"
        "print(json.dumps({'version': list(sys.version_info[:3]), 'implementation': platform.python_implementation(), 'machine': platform.machine(), 'distributions': installed}, sort_keys=True))\n"
    )
    proc = subprocess.run([str(executable), "-I", "-c", code], stdin=subprocess.DEVNULL,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=60)
    if proc.returncode != 0:
        raise RuntimeError(f"Python metadata query failed for {executable}: {proc.stderr.decode('utf-8', 'replace')[-1000:]}")
    data = json.loads(proc.stdout.decode("utf-8"))
    return {**identity, "invocation_path": str(executable),
            "version": data["version"], "implementation": data["implementation"],
            "machine": data["machine"], "distributions": data["distributions"]}


def _proc_unescape(value: str) -> str:
    return re.sub(r"\\([0-7]{3})", lambda match: chr(int(match.group(1), 8)), value)


def _cgroup_memory_snapshot() -> dict[str, Any]:
    """Read the current process memory controller limits without invoking models."""
    membership_rows: list[tuple[str, str, str]] = []
    for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            membership_rows.append((parts[0], parts[1], _proc_unescape(parts[2])))

    mounts: list[dict[str, str]] = []
    for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
        if " - " not in line:
            continue
        before, after = line.split(" - ", 1)
        left, right = before.split(), after.split()
        if len(left) < 5 or not right:
            continue
        fstype = right[0]
        if fstype not in {"cgroup2", "cgroup"}:
            continue
        super_options = right[2] if len(right) > 2 else ""
        mounts.append({
            "fstype": fstype,
            "root": _proc_unescape(left[3]),
            "mountpoint": _proc_unescape(left[4]),
            "super_options": super_options,
        })

    def cgroup_leaf(mount: Mapping[str, str], member_path: str) -> Path | None:
        root = posixpath.normpath(mount["root"])
        member = posixpath.normpath(member_path)
        if not root.startswith("/") or not member.startswith("/"):
            return None
        if member != root and not member.startswith(root.rstrip("/") + "/"):
            return None
        relative = posixpath.relpath(member, root)
        if relative == ".":
            relative = ""
        if relative == ".." or relative.startswith("../"):
            return None
        return Path(mount["mountpoint"]) / relative

    candidates: list[tuple[int, str, Mapping[str, str], Path]] = []
    for hierarchy, controllers, member_path in membership_rows:
        is_v2 = hierarchy == "0" and controllers == ""
        has_memory = "memory" in controllers.split(",")
        if not is_v2 and not has_memory:
            continue
        for mount in mounts:
            if is_v2 and mount["fstype"] != "cgroup2":
                continue
            if not is_v2 and (mount["fstype"] != "cgroup" or "memory" not in mount["super_options"].split(",")):
                continue
            leaf = cgroup_leaf(mount, member_path)
            if leaf is None:
                continue
            current_name = "memory.current" if is_v2 else "memory.usage_in_bytes"
            limit_name = "memory.max" if is_v2 else "memory.limit_in_bytes"
            if (leaf / current_name).is_file() and (leaf / limit_name).is_file():
                candidates.append((0 if is_v2 else 1, "v2" if is_v2 else "v1", mount, leaf))
    if not candidates:
        raise RuntimeError("Linux cgroup memory controller current/limit files were not found for this process")

    _, version, mount, leaf = sorted(candidates, key=lambda item: item[0])[0]
    current_name = "memory.current" if version == "v2" else "memory.usage_in_bytes"
    limit_name = "memory.max" if version == "v2" else "memory.limit_in_bytes"
    current_raw = (leaf / current_name).read_text(encoding="ascii").strip()
    limit_raw = (leaf / limit_name).read_text(encoding="ascii").strip()
    if not current_raw.isdigit():
        raise RuntimeError("cgroup memory current usage is not a nonnegative integer")
    current = int(current_raw)
    if version == "v2" and limit_raw == "max":
        limit: int | None = None
    elif not limit_raw.isdigit():
        raise RuntimeError("cgroup memory limit is not an integer or the v2 unlimited marker")
    else:
        parsed_limit = int(limit_raw)
        # The v1 kernel reports a very large sentinel when no memory limit applies.
        limit = None if version == "v1" and parsed_limit >= (1 << 60) else parsed_limit
    available = None if limit is None else max(0, limit - current)
    identity = {
        "version": version,
        "mount_target": mount["mountpoint"],
        "mount_root": mount["root"],
        "membership_path": next((path for _, controllers, path in membership_rows
                                  if (version == "v2" and controllers == "")
                                  or (version == "v1" and "memory" in controllers.split(","))), None),
        "leaf_path": str(leaf),
        "finite_limit": limit is not None,
        "limit_bytes": limit,
    }
    return {"version": version, "current_bytes": current, "limit_bytes": limit,
            "available_bytes": available, "finite_limit": limit is not None,
            "identity": identity}


def _memory_bytes() -> dict[str, Any]:
    meminfo = Path("/proc/meminfo").read_text(encoding="ascii")
    parsed: dict[str, int] = {}
    for line in meminfo.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields = value.split()
        if fields and fields[0].isdigit():
            parsed[key] = int(fields[0]) * 1024
    if "MemTotal" not in parsed or "MemAvailable" not in parsed:
        raise RuntimeError("Linux RAM inventory lacks MemTotal or MemAvailable")
    cgroup = _cgroup_memory_snapshot()
    cgroup_available = cgroup["available_bytes"]
    effective_available = parsed["MemAvailable"] if cgroup_available is None else min(parsed["MemAvailable"], cgroup_available)
    return {"total_bytes": parsed["MemTotal"], "available_bytes": effective_available,
            "host_available_bytes": parsed["MemAvailable"], "cgroup": cgroup}


def _mount_identity(target: Path) -> dict[str, str]:
    resolved = target.resolve(strict=False)
    best: tuple[int, str, str] | None = None
    mountinfo = Path("/proc/mounts")
    if mountinfo.exists():
        for line in mountinfo.read_text(encoding="utf-8", errors="replace").splitlines():
            fields = line.split()
            if len(fields) < 3:
                continue
            mount_target = Path(fields[1].replace("\\040", " ").replace("\\011", "\t"))
            try:
                resolved.relative_to(mount_target)
            except ValueError:
                continue
            item = (len(str(mount_target)), str(mount_target), fields[2])
            if best is None or item[0] > best[0]:
                best = item
    if best is None:
        raise RuntimeError(f"could not identify the output path mount: {resolved}")
    return {"target": best[1], "filesystem": best[2]}


def disk_inventory(path: Path) -> dict[str, Any]:
    probe = path
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists() or not probe.is_dir():
        raise ValueError(f"output path has no existing directory ancestor: {path}")
    usage = os.statvfs(probe)
    mount = _mount_identity(probe)
    return {"requested_output_root": str(Path(os.path.abspath(path))),
            "existing_ancestor": str(probe.resolve(strict=True)),
            "target": mount["target"], "filesystem": mount["filesystem"],
            "total_bytes": usage.f_blocks * usage.f_frsize,
            "available_bytes": usage.f_bavail * usage.f_frsize}


def gpu_inventory() -> list[dict[str, Any]]:
    proc = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total,memory.free", "--format=csv,noheader,nounits"],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        check=False, timeout=30,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"nvidia-smi resource query failed: {proc.stderr.decode('utf-8', 'replace')[-1000:]}")
    rows = []
    for values in csv.reader(proc.stdout.decode("utf-8").splitlines(), skipinitialspace=True):
        if len(values) != 5:
            raise RuntimeError("nvidia-smi returned an unrecognized GPU inventory row")
        rows.append({"index": int(values[0]), "uuid": values[1].strip(), "name": values[2].strip(),
                     "memory_total_mib": int(values[3]), "memory_free_mib": int(values[4])})
    if not rows:
        raise RuntimeError("no NVIDIA GPU was reported on the new host")
    return rows


def _stable_resources(resources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "host": resources["host"],
        "gpus": [{"index": item["index"], "uuid": item["uuid"], "name": item["name"], "memory_total_mib": item["memory_total_mib"]}
                 for item in resources["gpus"]],
        "ram_total_bytes": resources["ram"]["total_bytes"],
        "ram_cgroup_limit_identity": resources["ram"]["cgroup"]["identity"],
        "shm_total_bytes": resources["shm"]["total_bytes"],
        "output_mount": {key: resources["output_disk"][key]
                         for key in ("target", "filesystem", "total_bytes")},
    }


def resource_snapshot(output_root: Path, thresholds: Mapping[str, int]) -> dict[str, Any]:
    mem = _memory_bytes()
    shm_stat = os.statvfs("/dev/shm")
    shm = {"total_bytes": shm_stat.f_blocks * shm_stat.f_frsize,
           "available_bytes": shm_stat.f_bavail * shm_stat.f_frsize}
    disk = disk_inventory(output_root)
    gpus = gpu_inventory()
    if min(item["memory_free_mib"] for item in gpus) < thresholds["min_gpu_free_mib"]:
        raise RuntimeError("at least one current GPU is below the supplied free-memory minimum")
    if mem["available_bytes"] < thresholds["min_ram_available_bytes"]:
        raise RuntimeError("effective available RAM (minimum of host and finite cgroup availability) is below the supplied preflight minimum")
    if shm["available_bytes"] < thresholds["min_shm_free_bytes"]:
        raise RuntimeError("current /dev/shm capacity is below the supplied preflight minimum")
    if disk["available_bytes"] < thresholds["min_output_free_bytes"]:
        raise RuntimeError("selected output mount capacity is below the supplied preflight minimum")
    return {"host": platform.node(), "gpus": gpus, "ram": mem, "shm": shm,
            "output_disk": disk, "thresholds": dict(thresholds)}


def _resolve_binding(bindings: Mapping[str, Any], name: str) -> Path:
    value = bindings.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"runtime binding {name} must be an explicit nonempty path")
    return Path(value)


def runtime_identity(project_root: Path, plan: Mapping[str, Any], bindings: Mapping[str, Any]) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    refs = plan.get("source_reference_plans", {})
    script_bindings = {
        "stagea_core": ("stagea_core", STAGEA_REL, STAGEA_BYTES, STAGEA_SHA256, "stagea_core"),
        "cross_frontend_adapter_runner": ("cross_frontend_adapter_runner", CROSS_ADAPTER_REL,
                                           CROSS_ADAPTER_BYTES, CROSS_ADAPTER_SHA256, "cross_frontend_adapter_runner"),
        "v3_dispatcher": ("v3_dispatcher", V3_DISPATCHER_REL, V3_DISPATCHER_BYTES, V3_DISPATCHER_SHA256,
                          "v3_candidate_dispatcher"),
        "v3_cli": ("v3_cli", "scripts/audio_enhancement/enhance_v3_attempt02.py", None, None, "v3_cli"),
    }
    scripts: dict[str, Any] = {}
    for label, (binding_key, relative, fixed_size, fixed_hash, ref_key) in script_bindings.items():
        chosen = _resolve_binding(bindings, binding_key).resolve(strict=True)
        if label != "v3_cli":
            expected_path = project_root / relative
            if chosen != expected_path.resolve(strict=True):
                raise ValueError(f"runtime binding {binding_key} is not the reviewed repository source")
        identity = file_identity(chosen, label, fixed_hash)
        if fixed_size is not None and identity["bytes"] != fixed_size:
            raise ValueError(f"{label} byte length differs from the reviewed source")
        reference = refs.get(ref_key, {})
        expected_plan_hash = reference.get("raw_sha256")
        if not isinstance(expected_plan_hash, str) or identity["sha256"] != expected_plan_hash.lower():
            raise ValueError(f"{label} source identity differs from the frozen plan")
        scripts[label] = identity
    if scripts["v3_cli"]["path"] != str((project_root / "scripts/audio_enhancement/enhance_v3_attempt02.py").resolve(strict=True)):
        raise ValueError("V3 CLI path differs from the frozen repository entrypoint")

    configured_cli = Path(str(plan.get("source_reference_plans", {}).get("v3_cli_config", {}).get("path", "")))
    if not configured_cli.is_absolute():
        configured_cli = project_root / configured_cli
    cli_config, _cli_raw = load_json(configured_cli, "materialized V3 candidate config", 8 * 1024 * 1024)
    cli_pin = cli_config.get("source_pins", {}).get("cli", {})
    if cli_pin.get("sha256") != scripts["v3_cli"]["sha256"]:
        raise ValueError("materialized V3 config CLI pin differs from the frozen V3 CLI source")

    pcm = file_identity(_resolve_binding(bindings, "pcm16_lineage"), "PCM16 lineage implementation", PCM16_SHA256)
    if pcm["bytes"] != PCM16_BYTES or pcm["path"] != str((project_root / PCM16_REL).resolve(strict=True)):
        raise ValueError("PCM16 lineage binding differs from the reviewed source")
    joiner_path = project_root / JOINER_REL
    joiner = file_identity(joiner_path, "540-row evidence joiner", JOINER_SHA256)
    if joiner["bytes"] != JOINER_BYTES:
        raise ValueError("540-row evidence joiner byte length differs from reviewed source")

    role_paths: dict[str, Any] = {}
    for key in ("stagea_python", "wav2lip_python", "syncnet_python"):
        executable = _resolve_binding(bindings, key)
        role_paths[key] = probe_python(str(executable))
    models = plan.get("models", [])
    per_model_python: dict[str, Any] = {}
    per_model_directories: dict[str, Any] = {}
    for model in models:
        generator = model.get("generator", {})
        python = generator.get("python") if isinstance(generator, Mapping) else None
        if isinstance(python, str) and python:
            per_model_python[str(model.get("model_id"))] = probe_python(python)
        if isinstance(generator, Mapping):
            directories = {}
            for key in ("cwd", "repo", "model_source_root", "ditto_source_root", "model_code_root",
                        "model_workspace", "ditto_weights_root"):
                value = generator.get(key)
                if isinstance(value, str) and value:
                    directories[key] = directory_identity(Path(value), f"{model.get('model_id')} {key}")
            env = generator.get("env", {})
            pythonpath = env.get("PYTHONPATH") if isinstance(env, Mapping) else None
            if isinstance(pythonpath, str):
                for index, value in enumerate(item for item in pythonpath.split(os.pathsep) if item):
                    directories[f"PYTHONPATH[{index}]"] = directory_identity(
                        Path(value), f"{model.get('model_id')} PYTHONPATH entry"
                    )
            per_model_directories[str(model.get("model_id"))] = directories

    candidate_runtime = cli_config.get("runtime", {})
    stage_runtime = role_paths["stagea_python"]
    if (stage_runtime.get("version") != candidate_runtime.get("python_version")
            or stage_runtime.get("distributions", {}).get("numpy") != candidate_runtime.get("numpy_version")
            or stage_runtime.get("distributions", {}).get("torch") != candidate_runtime.get("torch_version")):
        raise ValueError("stagea_python Python/NumPy/PyTorch identity differs from the actual V3 candidate config")
    for model in models:
        profile = model.get("profile", {})
        expected_torch = profile.get("torch_version") if isinstance(profile, Mapping) else None
        observed = per_model_python.get(str(model.get("model_id")), {}).get("distributions", {}).get("torch")
        if isinstance(expected_torch, str) and observed != expected_torch:
            raise ValueError(f"{model.get('model_id')} renderer PyTorch version differs from its frozen model profile")

    wav2lip_record = next((model for model in models if model.get("model_id") == MODELS[0]), None)
    if wav2lip_record is not None and isinstance(wav2lip_record.get("generator"), Mapping):
        generator = wav2lip_record["generator"]
        planned_python = generator.get("python")
        if isinstance(planned_python, str) and Path(planned_python).resolve(strict=True) != _resolve_binding(bindings, "wav2lip_python").resolve(strict=True):
            raise ValueError("runtime wav2lip_python differs from the frozen Wav2Lip generator interpreter")
        planned_repo = generator.get("repo")
        if isinstance(planned_repo, str) and Path(planned_repo).resolve(strict=True) != _resolve_binding(bindings, "wav2lip_root").resolve(strict=True):
            raise ValueError("runtime wav2lip_root differs from the frozen Wav2Lip generator repository")

    paths: dict[str, Any] = {}
    expected_runtime_files = bindings.get("expected_file_sha256", {})
    if not isinstance(expected_runtime_files, Mapping):
        raise ValueError("runtime expected_file_sha256 must be a path-role-to-SHA-256 object")
    for key in ("generator_child", "syncnet_evaluator"):
        if not is_sha256(expected_runtime_files.get(key)):
            raise ValueError(f"runtime bindings must provide the reviewed SHA-256 for {key}")
    for key in ("generator_child", "syncnet_evaluator", "ffmpeg", "ffprobe", "syncnet_checkpoint"):
        expected = expected_runtime_files.get(key)
        if expected is not None and not is_sha256(expected):
            raise ValueError(f"runtime expected_file_sha256.{key} must be a 64-character hex digest")
        paths[key] = file_identity(_resolve_binding(bindings, key), key, expected)
    paths["wav2lip_root"] = directory_identity(_resolve_binding(bindings, "wav2lip_root"), "Wav2Lip model root")
    paths["syncnet_root"] = directory_identity(_resolve_binding(bindings, "syncnet_root"), "SyncNet source root")
    score_hashes = {model.get("score_model_sha256") for model in models}
    if len(score_hashes) != 1 or paths["syncnet_checkpoint"]["sha256"] != next(iter(score_hashes)):
        raise ValueError("current SyncNet checkpoint differs from the frozen model evaluator identity")
    expected_ffmpeg = plan.get("protocol", {}).get("syncnet", {}).get("primary_audio_input", {}).get("ffmpeg_sha256")
    if paths["ffmpeg"]["sha256"] != expected_ffmpeg:
        raise ValueError("current FFmpeg executable differs from the frozen PCM16 lineage pin")

    dependencies: list[dict[str, Any]] = []
    dependencies_by_model: dict[str, list[dict[str, Any]]] = {}
    for model in models:
        model_id = str(model.get("model_id"))
        pins = model_file_pins([model])
        if not pins:
            raise ValueError(f"frozen {model_id} descriptor contains no path/hash backend dependency pins")
        identities = [file_identity(Path(pin["path"]), f"{model_id} frozen backend dependency", pin["sha256"])
                      for pin in pins]
        dependencies.extend(identities)
        dependencies_by_model[model_id] = identities
    model_records = {str(model["model_id"]): sha_bytes(canonical(model)) for model in models}
    return {"scripts": scripts, "pcm16_lineage": pcm, "joiner": joiner,
            "runtime_paths": paths, "python_environments": role_paths,
            "model_python_environments": per_model_python,
            "model_backend_directories": per_model_directories,
            "candidate_runtime_requirements": {
                "python_version": candidate_runtime.get("python_version"),
                "numpy_version": candidate_runtime.get("numpy_version"),
                "torch_version": candidate_runtime.get("torch_version"),
                "device": candidate_runtime.get("device"),
                "torch_intraop_threads": candidate_runtime.get("torch_intraop_threads"),
                "torch_interop_threads": candidate_runtime.get("torch_interop_threads"),
            },
            "model_record_canonical_sha256": model_records,
            "model_backend_dependency_files": dependencies,
            "model_backend_dependencies_by_model": dependencies_by_model}


def _attestation_payload(project_root: Path, plan_path: Path, rows_path: Path,
                         manifest_path: Path, config_path: Path, receipt_path: Path,
                         bindings_path: Path, bindings: Mapping[str, Any], thresholds: Mapping[str, int]) -> dict[str, Any]:
    baseline = _resolve_binding(bindings, "baseline_evidence_archive")
    validation = validate_frozen_inputs(project_root, plan_path, rows_path, manifest_path,
                                        config_path, receipt_path, baseline)
    plan, _raw = load_json(plan_path, "frozen Dev plan", 32 * 1024 * 1024)
    resources = resource_snapshot(_resolve_binding(bindings, "output_root"), thresholds)
    identity = runtime_identity(project_root, plan, bindings)
    return {
        "schema": SCHEMA,
        "created_unix": time.time(),
        "input_validation": validation,
        "runtime_bindings": {"path": str(bindings_path.resolve(strict=True)),
                             "raw_sha256": sha_file(bindings_path.resolve(strict=True))},
        "runtime_identity": identity,
        "resources": resources,
        "stable_resource_identity": _stable_resources(resources),
        "execution_authorized_by_packet": False,
        "api_calls_enabled": False,
    }


def write_new(path: Path, raw: bytes) -> None:
    target = Path(os.path.abspath(path))
    target.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(target, flags, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def preflight(args: argparse.Namespace) -> int:
    thresholds = {
        "min_gpu_free_mib": args.min_gpu_free_mib,
        "min_ram_available_bytes": args.min_ram_available_bytes,
        "min_shm_free_bytes": args.min_shm_free_bytes,
        "min_output_free_bytes": args.min_output_free_bytes,
    }
    if any(not isinstance(value, int) or value <= 0 for value in thresholds.values()):
        raise ValueError("all four resource minima must be positive integers")
    bindings, _bindings_raw = load_bindings(args.runtime_bindings)
    project_root = _resolve_binding(bindings, "project_root")
    packet = _attestation_payload(project_root, args.plan, args.planned_rows, args.plan_manifest,
                                  args.candidate_config, args.materialization_receipt,
                                  args.runtime_bindings, bindings, thresholds)
    raw = pretty(packet)
    write_new(args.attestation_out, raw)
    print(json.dumps({"status": "PREFLIGHT_PASS", "attestation": str(args.attestation_out.resolve()),
                      "attestation_raw_sha256": sha_bytes(raw), "model_dependency_files": len(packet["runtime_identity"]["model_backend_dependency_files"]),
                      "gpu_free_mib_by_index": {str(item["index"]): item["memory_free_mib"]
                                                 for item in packet["resources"]["gpus"]},
                      "ram_available_bytes": packet["resources"]["ram"]["available_bytes"],
                      "shm_available_bytes": packet["resources"]["shm"]["available_bytes"],
                      "output_mount": packet["resources"]["output_disk"]["target"],
                      "output_available_bytes": packet["resources"]["output_disk"]["available_bytes"]},
                     ensure_ascii=False, sort_keys=True))
    return 0


def _load_attestation(path: Path, expected_sha: str) -> tuple[dict[str, Any], bytes]:
    packet, raw = load_json(path, "preflight attestation", 8 * 1024 * 1024)
    if sha_bytes(raw) != expected_sha.lower() or packet.get("schema") != SCHEMA:
        raise ValueError("preflight attestation bytes/schema do not match the caller pin")
    return packet, raw


def run_dev180(args: argparse.Namespace) -> int:
    if not args.execute_dev180:
        raise ValueError("run requires explicit --execute-dev180")
    bindings, _bindings_raw = load_bindings(args.runtime_bindings)
    packet, packet_raw = _load_attestation(args.preflight_attestation, args.expected_preflight_sha256)
    project_root = _resolve_binding(bindings, "project_root").resolve(strict=True)
    expected_bindings_sha = packet.get("runtime_bindings", {}).get("raw_sha256")
    if expected_bindings_sha != sha_file(args.runtime_bindings.resolve(strict=True)):
        raise ValueError("runtime bindings changed after the host preflight")
    thresholds = packet.get("resources", {}).get("thresholds", {})
    required_thresholds = ("min_gpu_free_mib", "min_ram_available_bytes", "min_shm_free_bytes", "min_output_free_bytes")
    if any(not isinstance(thresholds.get(key), int) or thresholds[key] <= 0 for key in required_thresholds):
        raise ValueError("preflight attestation lacks explicit current-host capacity minima")
    current_resources = resource_snapshot(_resolve_binding(bindings, "output_root"), thresholds)
    if _stable_resources(current_resources) != packet.get("stable_resource_identity"):
        raise ValueError("GPU, host RAM class, /dev/shm size, or output mount changed after preflight")
    plan, _plan_raw = load_json(args.plan, "frozen Dev plan", 32 * 1024 * 1024)
    identity = runtime_identity(project_root, plan, bindings)
    if canonical(identity) != canonical(packet.get("runtime_identity")):
        raise ValueError("software, generator/model dependency, or backend identity changed after preflight")
    current_inputs = validate_frozen_inputs(
        project_root, args.plan, args.planned_rows, args.plan_manifest,
        args.candidate_config, args.materialization_receipt,
        _resolve_binding(bindings, "baseline_evidence_archive"),
    )
    if canonical(current_inputs) != canonical(packet.get("input_validation")):
        raise ValueError("candidate config, materializer receipt, plan, rows, or baseline evidence changed after preflight")
    output_root = _resolve_binding(bindings, "output_root")
    if output_root.exists() and (not output_root.is_dir() or any(output_root.iterdir())):
        raise FileExistsError(f"Dev180 output root already exists and is not empty: {output_root}")

    cli_sha = identity["scripts"]["v3_cli"]["sha256"]
    ffmpeg_sha = identity["runtime_paths"]["ffmpeg"]["sha256"]
    pcm_sha = identity["pcm16_lineage"]["sha256"]
    argv = [
        str(_resolve_binding(bindings, "stagea_python")), "-B", str(_resolve_binding(bindings, "v3_dispatcher")),
        "--stagea-core", str(_resolve_binding(bindings, "stagea_core")),
        "--cross-frontend-adapter-runner", str(_resolve_binding(bindings, "cross_frontend_adapter_runner")),
        "--v3-cli", str(_resolve_binding(bindings, "v3_cli")), "--expected-v3-cli-sha256", cli_sha,
        "--plan", str(args.plan.resolve(strict=True)), "--planned-rows", str(args.planned_rows.resolve(strict=True)),
        "--expected-plan-raw-sha256", sha_file(args.plan.resolve(strict=True)),
        "--output-root", str(output_root), "--project-root", str(project_root),
        "--wav2lip-python", str(_resolve_binding(bindings, "wav2lip_python")),
        "--wav2lip-root", str(_resolve_binding(bindings, "wav2lip_root")),
        "--generator-child", str(_resolve_binding(bindings, "generator_child")),
        "--syncnet-root", str(_resolve_binding(bindings, "syncnet_root")),
        "--syncnet-python", str(_resolve_binding(bindings, "syncnet_python")),
        "--syncnet-checkpoint", str(_resolve_binding(bindings, "syncnet_checkpoint")),
        "--syncnet-evaluator", str(_resolve_binding(bindings, "syncnet_evaluator")),
        "--ffmpeg", str(_resolve_binding(bindings, "ffmpeg")),
        "--ffprobe", str(_resolve_binding(bindings, "ffprobe")),
        "--ffmpeg-sha256", ffmpeg_sha, "--pcm16-lineage-sha256", pcm_sha,
        "--generator-timeout", str(args.generator_timeout),
        "--syncnet-timeout", str(args.syncnet_timeout), "--ffmpeg-timeout", str(args.ffmpeg_timeout),
    ]
    launch = {"schema": "cross_tfg_v3_dev180_latebind_execution_receipt_v1",
              "status": "RUNNING", "started_unix": time.time(),
              "preflight_attestation_raw_sha256": sha_bytes(packet_raw),
              "plan_raw_sha256": sha_file(args.plan.resolve(strict=True)),
              "candidate_config_raw_sha256": current_inputs["candidate_config"]["raw_sha256"],
              "materialization_receipt_raw_sha256": current_inputs["materialization_receipt"]["raw_sha256"],
              "runtime_identity_sha256": sha_bytes(canonical(identity)),
              "output_root": str(output_root.resolve(strict=False)),
              "argv_sha256": sha_bytes(canonical(argv)), "api_calls_enabled": False,
              "confirm_audio_read": False, "confirm_effects_read": False}
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False)
    tail = proc.stdout.decode("utf-8", "replace")[-12_000:]
    launch.update({"ended_unix": time.time(), "returncode": proc.returncode,
                   "status": "PASS" if proc.returncode == 0 else "INCOMPLETE_OR_FAILED",
                   "result_manifest_path": str(output_root / "result_manifest.json"),
                   "result_manifest_raw_sha256": sha_file(output_root / "result_manifest.json")
                   if (output_root / "result_manifest.json").is_file() else None,
                   "stdout_tail": tail})
    write_new(args.execution_receipt_out, pretty(launch))
    print(json.dumps({"status": launch["status"], "returncode": proc.returncode,
                      "output_root": str(output_root),
                      "result_manifest_raw_sha256": launch["result_manifest_raw_sha256"],
                      "execution_receipt": str(args.execution_receipt_out.resolve())}, ensure_ascii=False, sort_keys=True))
    return 0 if proc.returncode == 0 else proc.returncode


def _copy_verified(source: Path, target: Path, expected_sha256: str) -> dict[str, Any]:
    source = source.resolve(strict=True)
    before = source.stat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"candidate raw-score matrix is not a regular file: {source}")
    if before.st_size > 64 * 1024 * 1024:
        raise ValueError(f"candidate raw-score matrix exceeds the joiner's 64 MiB read limit: {source}")
    digest = hashlib.sha256()
    target.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as src, target.open("xb") as dst:
        for block in iter(lambda: src.read(1024 * 1024), b""):
            digest.update(block)
            dst.write(block)
        dst.flush()
        os.fsync(dst.fileno())
    after = source.stat()
    actual = digest.hexdigest()
    if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns
    ) or actual != expected_sha256.lower():
        target.unlink(missing_ok=True)
        raise ValueError(f"candidate matrix changed or failed its result-ledger hash: {source}")
    return {"source_path": str(source), "asset_path": str(target), "bytes": before.st_size, "sha256": actual}


def collect_and_join(args: argparse.Namespace) -> int:
    bindings, _bindings_raw = load_bindings(args.runtime_bindings)
    project_root = _resolve_binding(bindings, "project_root").resolve(strict=True)
    baseline_archive = _resolve_binding(bindings, "baseline_evidence_archive")
    validation = validate_frozen_inputs(project_root, args.plan, args.planned_rows, args.plan_manifest,
                                        args.candidate_config, args.materialization_receipt,
                                        baseline_archive)
    plan, plan_raw = load_json(args.plan, "frozen Dev plan", 32 * 1024 * 1024)
    rows_path = args.output_root / "results.jsonl"
    result_manifest_path = args.output_root / "result_manifest.json"
    result_manifest, result_manifest_raw = load_json(result_manifest_path, "Stage-A result manifest", 4 * 1024 * 1024)
    results_raw = read_regular(rows_path, "Stage-A results ledger", 64 * 1024 * 1024)
    if (result_manifest.get("schema_version") != "cross-tfg-audio-eval-v1"
            or result_manifest.get("run_id") != plan.get("run_id")
            or result_manifest.get("plan_sha256") != plan.get("plan_sha256")
            or result_manifest.get("plan_raw_sha256") != sha_bytes(plan_raw)
            or result_manifest.get("results_jsonl_sha256") != sha_bytes(results_raw)
            or result_manifest.get("manifest_sha256") != payload_sha(result_manifest, "manifest_sha256")
            or result_manifest.get("planned_rows") != 180):
        raise ValueError("Stage-A result manifest does not bind the frozen 180-row plan and results ledger")
    result_rows = [json.loads(line) for line in results_raw.decode("utf-8").splitlines() if line.strip()]
    if len(result_rows) != 180 or [row.get("row_id") for row in result_rows] != [row.get("row_id") for row in plan["rows"]]:
        raise ValueError("Stage-A result ledger is not the complete frozen 180-row order")
    counts: dict[str, int] = {"PASS": 0, "FAILED": 0, "INCOMPLETE": 0}
    for row in result_rows:
        status = row.get("status")
        if status not in counts:
            raise ValueError(f"Stage-A ledger has an unknown row status: {status}")
        counts[status] += 1
    if result_manifest.get("status_counts") != counts or result_manifest.get("result_rows") != 180:
        raise ValueError("Stage-A result manifest status or row denominators differ from results.jsonl")
    if result_manifest.get("status") not in {"PASS", "INCOMPLETE"}:
        raise ValueError("Stage-A result manifest has an unsupported terminal status")
    if (result_manifest["status"] == "PASS" and (counts["FAILED"] or counts["INCOMPLETE"])):
        raise ValueError("Stage-A manifest claims PASS despite failed/incomplete rows")

    assets_root = args.candidate_assets_root
    assets_dir = assets_root / "assets"
    if assets_root.exists() and (not assets_root.is_dir() or any(assets_root.iterdir())):
        raise FileExistsError(f"candidate matrix asset directory is not empty: {assets_root}")
    assets_dir.mkdir(parents=True, exist_ok=False)
    out_root = args.output_root.resolve(strict=True)
    asset_refs = []
    seen: set[str] = set()
    for row in result_rows:
        if row["status"] != "PASS":
            continue
        descriptor = row.get("score", {}).get("raw_matrix_npz", {})
        digest = descriptor.get("file_sha256")
        source_text = descriptor.get("path")
        if not is_sha256(digest) or not isinstance(source_text, str):
            raise ValueError(f"PASS result row lacks a raw score matrix pin: {row.get('row_id')}")
        source = Path(source_text).resolve(strict=True)
        try:
            source.relative_to(out_root)
        except ValueError as exc:
            raise ValueError("Stage-A matrix reference escaped the Dev output root") from exc
        if digest in seen:
            continue
        asset_refs.append(_copy_verified(source, assets_dir / f"{digest}.npz", digest))
        seen.add(digest)
    index = {"schema": "cross_tfg_v3_dev180_candidate_matrix_asset_index_v1",
             "result_manifest_raw_sha256": sha_bytes(result_manifest_raw),
             "results_jsonl_sha256": sha_bytes(results_raw), "asset_count": len(asset_refs),
             "asset_bytes": sum(item["bytes"] for item in asset_refs), "assets": asset_refs,
             "matrix_files_are_score_evidence_only": True}
    asset_index_path = assets_root / "asset_index.json"
    write_new(asset_index_path, pretty(index))

    joiner_path = project_root / JOINER_REL
    joiner = file_identity(joiner_path, "540-row evidence joiner", JOINER_SHA256)
    stagea_python = _resolve_binding(bindings, "stagea_python")
    argv = [
        str(stagea_python), "-B", str(joiner_path), "--project-root", str(project_root),
        "--candidate-plan", str(args.plan.resolve(strict=True)),
        "--candidate-plan-raw-sha256", sha_bytes(plan_raw),
        "--candidate-planned-rows", str(args.planned_rows.resolve(strict=True)),
        "--candidate-planned-rows-sha256", validation["plan"]["row_ledger_raw_sha256"],
        "--candidate-result-manifest", str(result_manifest_path.resolve(strict=True)),
        "--candidate-result-manifest-raw-sha256", sha_bytes(result_manifest_raw),
        "--candidate-results", str(rows_path.resolve(strict=True)),
        "--candidate-results-sha256", sha_bytes(results_raw),
        "--candidate-cli-config", str(args.candidate_config.resolve(strict=True)),
        "--candidate-cli-config-raw-sha256", validation["candidate_config"]["raw_sha256"],
        "--candidate-assets-root", str(assets_root.resolve(strict=True)),
        "--baseline-evidence-archive", str(baseline_archive.resolve(strict=True)),
        "--output-dir", str(args.join_output_dir),
    ]
    proc = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          check=False)
    output = proc.stdout.decode("utf-8", "replace")[-12_000:]
    receipt = {
        "schema": "cross_tfg_v3_dev180_latebind_collection_receipt_v1",
        "status": "JOINED" if proc.returncode == 0 else "JOIN_FAILED",
        "returncode": proc.returncode,
        "candidate_plan_raw_sha256": sha_bytes(plan_raw),
        "candidate_result_manifest_raw_sha256": sha_bytes(result_manifest_raw),
        "candidate_results_sha256": sha_bytes(results_raw),
        "candidate_matrix_asset_index_sha256": sha_file(asset_index_path),
        "candidate_matrix_asset_count": len(asset_refs),
        "baseline_archive_sha256": validation["baseline_archive"]["sha256"],
        "joiner_sha256": joiner["sha256"], "join_output_dir": str(args.join_output_dir),
        "confirm_audio_read": False, "confirm_effects_read": False,
        "method_selected": False, "api_calls_enabled": False,
        "stdout_tail": output,
    }
    write_new(args.collection_receipt_out, pretty(receipt))
    print(json.dumps({"status": receipt["status"], "returncode": proc.returncode,
                      "candidate_matrix_assets": len(asset_refs),
                      "join_output_dir": str(args.join_output_dir),
                      "collection_receipt": str(args.collection_receipt_out.resolve())},
                     ensure_ascii=False, sort_keys=True))
    return 0 if proc.returncode == 0 else proc.returncode


def add_frozen_inputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runtime-bindings", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--planned-rows", type=Path, required=True)
    parser.add_argument("--plan-manifest", type=Path, required=True)
    parser.add_argument("--candidate-config", type=Path, required=True)
    parser.add_argument("--materialization-receipt", type=Path, required=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pre = subparsers.add_parser("preflight", help="bind sealed plan, actual receipt, current host, models, and capacity")
    add_frozen_inputs(pre)
    pre.add_argument("--attestation-out", type=Path, required=True)
    pre.add_argument("--min-gpu-free-mib", type=int, required=True)
    pre.add_argument("--min-ram-available-bytes", type=int, required=True)
    pre.add_argument("--min-shm-free-bytes", type=int, required=True)
    pre.add_argument("--min-output-free-bytes", type=int, required=True)
    pre.set_defaults(func=preflight)

    run = subparsers.add_parser("run", help="execute the fixed 180-row Dev slate through the existing Stage-A runner")
    add_frozen_inputs(run)
    run.add_argument("--preflight-attestation", type=Path, required=True)
    run.add_argument("--expected-preflight-sha256", required=True)
    run.add_argument("--execution-receipt-out", type=Path, required=True)
    run.add_argument("--execute-dev180", action="store_true", help="explicitly authorize the local 180-row render/score invocation")
    run.add_argument("--generator-timeout", type=float, default=900)
    run.add_argument("--syncnet-timeout", type=float, default=1800)
    run.add_argument("--ffmpeg-timeout", type=float, default=300)
    run.set_defaults(func=run_dev180)

    collect = subparsers.add_parser("collect", help="package verified score matrices and run the reviewed 540-row evidence join")
    add_frozen_inputs(collect)
    collect.add_argument("--output-root", type=Path, required=True)
    collect.add_argument("--candidate-assets-root", type=Path, required=True)
    collect.add_argument("--join-output-dir", type=Path, required=True)
    collect.add_argument("--collection-receipt-out", type=Path, required=True)
    collect.set_defaults(func=collect_and_join)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        return int(args.func(args))
    except Exception as exc:
        print(json.dumps({"status": "BLOCKED", "error_type": type(exc).__name__, "error": str(exc)},
                         ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
