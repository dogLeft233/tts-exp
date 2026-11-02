"""Bounded two-arm experiment controller; data -> phones -> sync gates -> TTS.

Never stops other GPU users, never changes the frozen training budget, and never
uses TTS preference to choose checkpoints. A failed stage stops the chain.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    p = path.with_suffix(path.suffix + ".tmp")
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    p.replace(path)


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def rows_from(path):
    text = Path(path).read_text(encoding="utf-8")
    if Path(path).suffix == ".jsonl":
        return [json.loads(x) for x in text.splitlines() if x.strip()]
    value = json.loads(text)
    return value if isinstance(value, list) else value["rows"]


def validate_data(rows):
    ids, speakers, seconds = set(), {"train": set(), "val": set()}, {"train": 0., "val": 0.}
    for r in rows:
        if r["id"] in ids:
            raise ValueError("duplicate data IDs")
        ids.add(r["id"])
        if r["split"] not in {"train", "val"}:
            raise ValueError("only official train/val may enter pilot; test locked")
        speakers[r["split"]].add(r["speaker"])
        seconds[r["split"]] += int(r["frame_count"]) / 25
        if int(r.get("sample_rate", 16000)) != 16000 or float(r.get("fps", 25)) != 25:
            raise ValueError("invalid canonical timebase")
        for key in ("audio_path", "video_path", "mfcc_path"):
            if not Path(r[key]).is_file():
                raise FileNotFoundError(r[key])
    if speakers["train"] & speakers["val"]:
        raise ValueError("speaker leakage")
    if seconds["train"] < 20 * 3600 or seconds["val"] < 2 * 3600:
        raise ValueError(f"requested preparation budget incomplete: {seconds}")
    return {"hours": {k: v / 3600 for k, v in seconds.items()},
            "speakers": {k: len(v) for k, v in speakers.items()}, "clips": len(rows)}


def gpu_foreign_processes():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv,noheader,nounits"],
                            text=True, capture_output=True, check=True, timeout=20)
    # The desktop remote-display daemon is small and persistent; actual CUDA
    # workloads, including scoring jobs owned by other chats, block this queue.
    return [line for line in result.stdout.splitlines() if line.strip()
            and "gnome-remote-desktop" not in line]


def terminate_owned_group(child):
    """A stage has its own session; clean grandchildren even if leader exited."""
    try:
        os.killpg(child.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    # The leader may already be gone while a worker ignores SIGTERM. Reap the
    # whole owned group rather than using leader exit as proof of group exit.
    try:
        os.killpg(child.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    child.wait()


def run(args):
    repo, root = Path(args.repo).resolve(), Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "controller_status.json"
    if status_path.exists():
        raise FileExistsError("Controller output exists; do not silently retry or overwrite a previous chain")
    started = time.time()
    state = {"pid": os.getpid(), "started": utc(), "run_root": str(root),
             "data_ready": str(args.ready), "authorized_budget": {"seeds": [1729],
             "conditions": ["N0", "N1"], "steps_per_arm": 5000, "batch_size": 16,
             "lr": .0001, "margin": 10., "no_acoustic_augmentation": True}}
    scripts = root / "frozen_source"
    scripts.mkdir(exist_ok=False)
    for filename in ("chinese_lips_phones_20261029.py", "chinese_lips_syncnet_train_20261029.py",
                     "chinese_lips_preference_20261029.py", "chinese_lips_controller_20261029.py"):
        shutil.copy2(repo / "scripts" / "experiments" / filename, scripts / filename)
    shutil.copy2(repo / "third_party/syncnet_python/SyncNetModel.py", scripts / "SyncNetModel.py")
    write(root / "frozen_sources.json", {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                        for p in sorted(scripts.glob("*.py"))})

    def update(stage, **extra):
        state.update(stage=stage, updated=utc(), elapsed_seconds=time.time() - started, **extra)
        write(status_path, state)
        print(stage, json.dumps(extra, ensure_ascii=False), flush=True)

    def execute(stage, argv, timeout):
        if shutil.disk_usage(root).free < 1_500_000_000:
            raise OSError("Less than 1.5GB free on experiment output disk")
        update(stage, command=argv)
        write(root / f"{stage}_command.json", {"argv": argv, "timeout_seconds": timeout})
        with (root / f"{stage}.log").open("w", encoding="utf-8") as log:
            child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, cwd=repo, start_new_session=True,
                                     env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1"))
            try:
                begin = time.time()
                while child.poll() is None:
                    update(stage, child_pid=child.pid, child_elapsed_seconds=time.time() - begin,
                           log_bytes=(root / f"{stage}.log").stat().st_size)
                    if time.time() - begin > timeout:
                        raise TimeoutError(f"hard timeout of owned child {stage}")
                    time.sleep(20)
                if child.returncode:
                    raise RuntimeError(f"{stage} exited {child.returncode}; see its log; no automatic retry")
            finally:
                terminate_owned_group(child)

    def wait_gpu():
        begin, idle = time.time(), None
        while True:
            foreign = gpu_foreign_processes()
            if foreign:
                idle = None
            elif idle is None:
                idle = time.time()
            update("WAIT_GPU", foreign_cuda_processes=foreign,
                   idle_seconds=0 if idle is None else time.time() - idle)
            if idle is not None and time.time() - idle >= 60:
                return
            if time.time() - begin > args.wait_hours * 3600:
                raise TimeoutError("GPU did not become idle within bounded waiting budget")
            time.sleep(20)

    try:
        update("WAIT_DATA")
        deadline = time.time() + args.wait_hours * 3600
        while True:
            if args.ready.is_file():
                try:
                    ready = read(args.ready)
                    break
                except json.JSONDecodeError:
                    # A receipt arriving via SCP can briefly exist before EOF.
                    # This polls publication, never retries a failed experiment.
                    update("WAIT_DATA", receipt_incomplete=True)
            if time.time() > deadline:
                raise TimeoutError("data preparation did not become ready within waiting budget")
            time.sleep(20)
            update("WAIT_DATA")
        if ready.get("status") not in {"READY", "COMPLETE"}:
            raise ValueError("data-ready receipt not complete")
        manifests = ([ready["manifest"]] if "manifest" in ready else
                     [ready["train_manifest"], ready["val_manifest"]])
        rows = [row for path in manifests for row in rows_from(path)]
        audit = validate_data(rows)
        write(root / "data_manifest.json", {"rows": rows, "audit": audit, "upstream_ready": ready})
        update("DATA_READY", data_audit=audit)
        execute("ALIGN_PHONES", [sys.executable, str(scripts / "chinese_lips_phones_20261029.py"),
                "--manifest", str(root / "data_manifest.json"), "--out", str(root / "phones"),
                "--dictionary", str(args.dictionary), "--jobs", "4"], 12 * 3600)
        execute("PREPARE_PAIRS", [sys.executable, str(scripts / "chinese_lips_syncnet_train_20261029.py"),
                "prepare-pairs", "--manifest", str(root / "phones" / "manifest.json"),
                "--output", str(root / "pairs")], 3600)
        pool_audit = read(root / "pairs" / "weight_audit.json")
        if not pool_audit.get("quality_gate", {}).get("passed", False):
            update("POOL_CONTROL_FAILED", pool_quality=pool_audit.get("quality_gate"),
                   interpretation="No training: negative intervention lacks frozen common support")
            return
        # Trainer itself enforces the frozen common-support capability of pool.
        source = scripts / "SyncNetModel.py"
        for condition in ("N0", "N1"):
            wait_gpu()
            execute("TRAIN_" + condition, [sys.executable, str(scripts / "chinese_lips_syncnet_train_20261029.py"),
                    "train", "--pairs", str(root / "pairs"), "--output", str(root / condition),
                    "--model-source", str(source), "--condition", condition, "--seed", "1729",
                    "--steps", "5000", "--batch-size", "16", "--lr", "0.0001", "--margin", "10"], 8 * 3600)
        import torch
        models, gates = [], []
        for condition in ("N0", "N1"):
            checkpoint = root / condition / "best.pt"
            stored = torch.load(checkpoint, map_location="cpu", weights_only=False)
            gate = stored["validation"]
            gate_path = root / condition / "selected_gate.json"
            write(gate_path, gate)
            gates.append(bool(gate["gate_pass"]))
            models.append({"id": condition, "checkpoint": str(checkpoint), "gate_path": str(gate_path)})
        write(root / "models.json", models)
        if not all(gates):
            update("CAPABILITY_FAILED", gate_pass_by_condition=dict(zip(("N0", "N1"), gates)),
                   interpretation="No TTS preference conclusion; basic synchronization gate failed")
            return
        preference = root / "preference"
        execute("FREEZE_PREFERENCE", [sys.executable, str(scripts / "chinese_lips_preference_20261029.py"),
                "freeze", "--repo", str(repo), "--inputs", str(repo / "runs/common_window_curves_cloud_20261028_attempt01/inputs.json"),
                "--models", str(root / "models.json"), "--out", str(preference)], 600)
        wait_gpu()
        execute("SCORE_PREFERENCE", [sys.executable, str(scripts / "chinese_lips_preference_20261029.py"),
                "run", "--repo", str(repo), "--out", str(preference)], 3 * 3600)
        update("COMPLETE", result=str(preference / "summary.json"))
    except Exception as exc:
        update("FAILED", error=repr(exc), traceback=traceback.format_exc())
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ready", type=Path, required=True)
    p.add_argument("--dictionary", type=Path, required=True)
    p.add_argument("--wait-hours", type=float, default=24)
    run(p.parse_args())


if __name__ == "__main__":
    main()
