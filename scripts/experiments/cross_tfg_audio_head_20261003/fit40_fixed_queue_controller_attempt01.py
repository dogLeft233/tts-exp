#!/usr/bin/env python3
"""Fair one-shot runner for the frozen three-model Fit40 six-row blocks.

The scheduler accepts only the frozen plan/source allowlist, requires the
existing live resource gate and an absent output leaf before every block,
captures the real child wait status, and advances only on a complete six-row
PASS terminal. A failure pauses that model and keeps its output leaf intact;
other ready models continue in round-robin order. Failed leaves are never
replayed automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import time
import types
from typing import Any, Callable, Mapping, Sequence


SCHEMA = "fit40_fixed_queue_controller_v1"
EVENT_SCHEMA = "fit40_fixed_queue_event_v1"
PROCESS_SCHEMA = "fit40_queue_process_terminal_v1"
PLAN_SCHEMA = "cross-tfg-audio-eval-v1"
METHODS = (
    "identity",
    "rms_fixed",
    "fit40_native_transition_head_v3_attempt11_step030",
    "fit40_native_transition_head_v3_attempt11_step060",
    "fit40_native_transition_head_v3_attempt11_step090",
    "fit40_native_transition_head_v3_attempt11_step120",
)
BLOCK_ROWS = 6
TOTAL_ROWS = 240
BLOCKS_PER_MODEL = TOTAL_ROWS // BLOCK_ROWS
MODEL_ORDER = (
    "wav2lip_gan_bac9a81e",
    "musetalk_fixed_portrait_attempt16",
    "ditto_hubert_fixed_portrait_attempt18",
)
ZERO_SHA256 = "0" * 64


class QueueError(RuntimeError):
    """A frozen queue pin, lease, or block terminal proof is invalid."""


@dataclass(frozen=True)
class SourcePin:
    path: str
    bytes: int
    sha256: str


@dataclass(frozen=True)
class ModelPinSpec:
    model_id: str
    plan: SourcePin
    planned_rows: SourcePin
    runtime_binding: SourcePin
    launch_sources: tuple[SourcePin, ...]


PINNED_MODELS: tuple[ModelPinSpec, ...] = (
    ModelPinSpec(
        model_id="wav2lip_gan_bac9a81e",
        plan=SourcePin(
            ".codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/project_payload_attempt05/fit40_w2l240_plan.json",
            902628,
            "25f49cf6e87800da57fc797210b5d0c4bb6bb3396af04f9f9e9ff1aec0146832",
        ),
        planned_rows=SourcePin(
            ".codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/project_payload_attempt05/planned_rows.jsonl",
            678368,
            "54a69d99b34cf70b02f58b4a4bae032bcf5446fb9777d7a5594f6cc45ffeffbe",
        ),
        runtime_binding=SourcePin(
            ".codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/project_payload_attempt05/control/w2l_runtime_binding.json",
            2884,
            "a6638095c8ca05ffc6c48fe3d1017e9fd2f3d5da01b3279c01ca97b3e0a67d58",
        ),
        launch_sources=(SourcePin(
            ".codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/project_payload_attempt05/scripts/experiments/cross_tfg_audio_head_20261003/run_stagea_dev_w2l_block_attempt02.py",
            96537,
            "6969c6e952d46b6753d279b0e2447aa71c7b8e92967cd42ee57dc8122da5616e",
        ),),
    ),
    ModelPinSpec(
        model_id="musetalk_fixed_portrait_attempt16",
        plan=SourcePin(
            ".codex/scratch/v100_musetalk_fit40_240_attempt01/project_payload_attempt02/musetalk_fit40_240_plan.json",
            888737,
            "ae79df59ef5c3f05ba99d7850a1e0158c9fefb07e6b0e0fee8cc557af9f52dc4",
        ),
        planned_rows=SourcePin(
            ".codex/scratch/v100_musetalk_fit40_240_attempt01/project_payload_attempt02/planned_rows.jsonl",
            653408,
            "2c69885517a36e571406fa7a864097a5eb593eb8cb4006d35dcf6ade2c9d0fb0",
        ),
        runtime_binding=SourcePin(
            ".codex/scratch/v100_musetalk_fit40_240_attempt01/project_payload_attempt02/control/musetalk_v100_runtime_binding_attempt02.json",
            4438,
            "ee474826c0cea8b2f039f4b7dac986d7da98fa88fa1f0dcff85620c12a3f18de",
        ),
        launch_sources=(
            SourcePin(
                ".codex/scratch/v100_musetalk_fit40_240_attempt01/launch_musetalk_fit40_block_attempt01.py",
                13822,
                "acd5955fafe1c8708bb0358c11d02c38e28b6936365244658bad5d5e45c58b49",
            ),
            SourcePin(
                ".codex/scratch/v100_musetalk_fit40_240_attempt01/project_payload_attempt02/run_musetalk_fit40_block_attempt02.py",
                57751,
                "589184880821522d4f90609244b6cfc0230616079bb927b1cea7b259d87fb4d3",
            ),
            SourcePin(
                ".codex/scratch/v100_musetalk_fit40_240_attempt01/project_payload_attempt02/scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt09.py",
                23249,
                "04adf397a2edc45c6e3bc1e67d2265ea1c6ad2fcdac3df268b290938e5bf74df",
            ),
        ),
    ),
    ModelPinSpec(
        model_id="ditto_hubert_fixed_portrait_attempt18",
        plan=SourcePin(
            ".codex/scratch/v100_ditto_fit40_plan_attempt05/build_output_attempt05/fit40_ditto240_plan_attempt05.json",
            895194,
            "82ceb8ebd93f541bee9ae3ac9f93f3a3e657ff4979dcb451d03732cd2ad52475",
        ),
        planned_rows=SourcePin(
            ".codex/scratch/v100_ditto_fit40_plan_attempt05/build_output_attempt05/planned_rows_attempt05.jsonl",
            660128,
            "f386441f3ed54da8d69f38610f1cb7c8d79b0320a214ce20d1421fda3c232164",
        ),
        runtime_binding=SourcePin(
            ".codex/scratch/v100_ditto_fit40_plan_attempt05/build_output_attempt05/ditto_runtime_binding_attempt05.json",
            22530,
            "275c8ac529653ab4d066320b65a25dd3c3fb181ca89f09f6928d092013acfc2b",
        ),
        launch_sources=(SourcePin(
            ".codex/scratch/v100_ditto_fit40_block_attempt05/launch_ditto_fit40_block_attempt05.py",
            26097,
            "0c0ef0c656e91e3602976cdb064bc42cd80e498a67815973aae7195f8ba7b6f1",
        ),),
    ),
)


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def linux_process_start_ticks(pid: int) -> int:
    """Read Linux /proc start time ticks, failing closed off the target host."""
    if type(pid) is not int or pid <= 0:
        raise QueueError("process PID must be a positive integer")
    stat_path = Path(f"/proc/{pid}/stat")
    try:
        raw = stat_path.read_text(encoding="ascii")
    except OSError as exc:
        raise QueueError(f"cannot read Linux process identity {stat_path}") from exc
    # comm is parenthesized and may itself contain spaces or ')'; fields after
    # its last closing parenthesis begin with proc stat field 3 (state).
    close = raw.rfind(")")
    if close < 0:
        raise QueueError("Linux process stat has no command terminator")
    fields = raw[close + 1:].split()
    if len(fields) <= 19:
        raise QueueError("Linux process stat is truncated before starttime")
    try:
        ticks = int(fields[19])  # field 22, indexed from field 3
    except ValueError as exc:
        raise QueueError("Linux process starttime is not an integer") from exc
    if ticks <= 0:
        raise QueueError("Linux process starttime must be positive")
    return ticks


def current_process_identity() -> tuple[int, int]:
    pid = os.getpid()
    return pid, linux_process_start_ticks(pid)


def _require_current_owner(owner_pid: int, owner_start_ticks: int) -> None:
    current_pid, current_ticks = current_process_identity()
    if (owner_pid, owner_start_ticks) != (current_pid, current_ticks):
        raise QueueError("queue lease PID/start ticks do not identify this live coordinator process")


def _write_exclusive_json(path: Path, value: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists() or path.is_symlink():
        raise QueueError(f"refusing to overwrite queue evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(value)
    body["self_sha256"] = _self_hash(body, "self_sha256")
    raw = canonical(body) + b"\n"
    try:
        fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise QueueError(f"refusing to overwrite queue evidence: {path}") from exc
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha256(raw),
            "self_sha256": body["self_sha256"]}


def _self_hash(record: Mapping[str, Any], field_name: str) -> str:
    body = {key: value for key, value in record.items() if key != field_name}
    return sha256(canonical(body))


def _read_regular_pin(root: Path, pin: SourcePin, label: str) -> bytes:
    path = root / pin.path
    try:
        st = path.lstat()
    except FileNotFoundError as exc:
        raise QueueError(f"missing pinned {label}: {path}") from exc
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISREG(st.st_mode):
        raise QueueError(f"pinned {label} must be a regular non-symlink file: {path}")
    raw = path.read_bytes()
    if len(raw) != pin.bytes or sha256(raw) != pin.sha256:
        raise QueueError(f"pinned {label} raw size/SHA mismatch: {path}")
    return raw


@dataclass(frozen=True)
class FrozenBundle:
    spec: ModelPinSpec
    plan: Mapping[str, Any]
    rows: tuple[Mapping[str, Any], ...]
    row_ids: tuple[str, ...]
    sample_ids: tuple[str, ...]
    scientific_input_pins_sha256: str


def _source_identity(row: Mapping[str, Any]) -> dict[str, Any]:
    natural = row.get("natural_pcm")
    scored = row.get("score_audio")
    face = row.get("source_face")
    video = row.get("source_video")
    if not all(isinstance(x, Mapping) for x in (natural, scored, face, video)):
        raise QueueError("row is missing frozen natural-audio/face/media identity objects")
    fields = ("file_sha256", "pcm_sha256", "sample_count", "sample_rate_hz", "channels", "subtype")
    natural_identity = {key: natural.get(key) for key in fields}
    score_identity = {key: scored.get(key) for key in fields}
    if natural_identity != score_identity:
        raise QueueError("score_audio is not the exact natural PCM source")
    return {
        "split": row.get("split"),
        "sample_id": row.get("sample_id"),
        "speaker_id": row.get("speaker_id"),
        "seed": row.get("seed"),
        "face_id": row.get("face_id"),
        "source_media_kind": row.get("source_media_kind"),
        "natural_pcm": natural_identity,
        "source_face_sha256": face.get("file_sha256"),
        "source_video_sha256": video.get("file_sha256"),
    }


def load_pinned_bundles(repo_root: Path) -> tuple[FrozenBundle, ...]:
    """Verify the exact three frozen 240-row plans and their launch sources."""
    if tuple(spec.model_id for spec in PINNED_MODELS) != MODEL_ORDER:
        raise QueueError("source allowlist model order changed")
    bundles: list[FrozenBundle] = []
    shared_roster: tuple[str, ...] | None = None
    shared_science: Any = None
    shared_rows: dict[str, dict[str, Any]] | None = None
    for spec in PINNED_MODELS:
        plan_raw = _read_regular_pin(repo_root, spec.plan, f"{spec.model_id} plan")
        rows_raw = _read_regular_pin(repo_root, spec.planned_rows, f"{spec.model_id} planned rows")
        _read_regular_pin(repo_root, spec.runtime_binding, f"{spec.model_id} runtime binding")
        for source_pin in spec.launch_sources:
            _read_regular_pin(repo_root, source_pin, f"{spec.model_id} launch source")
        try:
            plan = json.loads(plan_raw.decode("utf-8"))
            row_lines = [json.loads(line) for line in rows_raw.splitlines() if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QueueError(f"invalid frozen JSON/JSONL for {spec.model_id}") from exc
        if not isinstance(plan, dict) or plan.get("schema_version") != PLAN_SCHEMA or plan.get("status") != "FROZEN":
            raise QueueError(f"{spec.model_id} plan is not frozen Stage-A")
        if plan.get("plan_sha256") != _self_hash(plan, "plan_sha256"):
            raise QueueError(f"{spec.model_id} plan self hash mismatch")
        models = plan.get("models")
        methods = plan.get("methods")
        plan_rows = plan.get("rows")
        if not isinstance(models, list) or len(models) != 1 or models[0].get("model_id") != spec.model_id:
            raise QueueError(f"{spec.model_id} frozen plan model descriptor mismatch")
        if not isinstance(methods, list) or tuple(item.get("method_id") for item in methods) != METHODS:
            raise QueueError(f"{spec.model_id} frozen six-method roster mismatch")
        if not isinstance(plan_rows, list) or len(plan_rows) != TOTAL_ROWS or len(row_lines) != TOTAL_ROWS:
            raise QueueError(f"{spec.model_id} must have exactly {TOTAL_ROWS} planned rows")
        if tuple(row.get("row_id") for row in row_lines) != tuple(row.get("row_id") for row in plan_rows):
            raise QueueError(f"{spec.model_id} planned_rows JSONL does not match the frozen plan order")
        if tuple(row.get("row_id") for row in row_lines) != tuple(dict.fromkeys(row.get("row_id") for row in row_lines)):
            raise QueueError(f"{spec.model_id} planned row IDs are not unique")
        samples: list[str] = []
        identities: dict[str, dict[str, Any]] = {}
        for offset in range(0, TOTAL_ROWS, BLOCK_ROWS):
            block = row_lines[offset:offset + BLOCK_ROWS]
            if len(block) != BLOCK_ROWS:
                raise QueueError("incomplete six-method block in frozen plan")
            if tuple(row.get("method_id") for row in block) != METHODS:
                raise QueueError(f"{spec.model_id} row order is not the exact six-method block")
            sample_ids = {row.get("sample_id") for row in block}
            model_ids = {row.get("model_id") for row in block}
            if len(sample_ids) != 1 or model_ids != {spec.model_id}:
                raise QueueError(f"{spec.model_id} block mixes samples or models")
            sample_id = next(iter(sample_ids))
            if not isinstance(sample_id, str) or not sample_id:
                raise QueueError(f"{spec.model_id} block has an empty sample ID")
            samples.append(sample_id)
            identity = _source_identity(block[0])
            for row in block:
                if _source_identity(row) != identity:
                    raise QueueError(f"{spec.model_id} methods disagree on source identity for {sample_id}")
                if row.get("split") != "fit" or row.get("seed") != 20261003:
                    raise QueueError(f"{spec.model_id} plan includes a non-Fit row or wrong renderer seed")
            identities[sample_id] = identity
        if len(set(samples)) != 40:
            raise QueueError(f"{spec.model_id} planned sample roster is not 40 unique clips")
        science = plan.get("scientific_input_pins")
        science_sha = plan.get("scientific_input_pins_sha256")
        if not isinstance(science, Mapping) or not isinstance(science_sha, str):
            raise QueueError(f"{spec.model_id} plan is missing shared scientific input pins")
        if sha256(canonical(science)) != science_sha:
            raise QueueError(f"{spec.model_id} scientific input pins self digest mismatch")
        if shared_roster is None:
            shared_roster = tuple(samples)
            shared_science = (science_sha, science)
            shared_rows = identities
        elif tuple(samples) != shared_roster or (science_sha, science) != shared_science or identities != shared_rows:
            raise QueueError("three model plans do not share the exact Fit40 natural-input roster")
        bundles.append(FrozenBundle(
            spec=spec,
            plan=plan,
            rows=tuple(row_lines),
            row_ids=tuple(str(row["row_id"]) for row in row_lines),
            sample_ids=tuple(samples),
            scientific_input_pins_sha256=science_sha,
        ))
    return tuple(bundles)


@dataclass
class ModelProgress:
    next_start: int = 0
    paused: bool = False
    pause_reason: str | None = None


@dataclass
class QueueState:
    progress: dict[str, ModelProgress]
    last_served_model: str | None = None
    active_claim: dict[str, Any] | None = None
    used_output_leaves: set[str] = field(default_factory=set)
    event_count: int = 0
    last_event_sha256: str = ZERO_SHA256


def _event_file_pin(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise QueueError(f"expected a regular evidence file: {path}")
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "bytes": len(raw), "sha256": sha256(raw)}


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], dict[str, Any]]:
    pin = _event_file_pin(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QueueError(f"invalid {label} JSON: {path}") from exc
    if not isinstance(value, dict):
        raise QueueError(f"{label} must be a JSON object: {path}")
    return value, pin


def _write_terminal_receipt(
    receipt_dir: Path,
    claim: Mapping[str, Any],
    *,
    child_pid: int | None,
    child_start_ticks: int | None,
    exit_code: int | None,
    output_dir: Path,
    launch_receipt: Path | None = None,
    launch_error: str | None = None,
    logs: Mapping[str, Path] | None = None,
) -> Path:
    nonce = str(claim["claim_nonce"])
    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise QueueError("claim nonce is not safe for a terminal receipt filename")
    path = receipt_dir / f"{nonce}_process_terminal.json"
    status = "EXITED" if child_pid is not None else "FAILED_BEFORE_SPAWN"
    body: dict[str, Any] = {
        "schema": PROCESS_SCHEMA,
        "claim_nonce": nonce,
        "model_id": claim["model_id"],
        "block_index": claim["block_index"],
        "block_start_index": claim["block_start_index"],
        "plan_raw_sha256": claim["plan_raw_sha256"],
        "output_leaf": str(Path(str(claim["output_leaf"]))),
        "terminal_status": status,
        "child_pid": child_pid,
        "child_start_ticks": child_start_ticks,
        "exit_code": exit_code,
    }
    if launch_error is not None:
        body["launch_error"] = launch_error[:2000]
    if launch_receipt is not None and launch_receipt.exists() and not launch_receipt.is_symlink():
        body["launch_receipt"] = _event_file_pin(launch_receipt)
    for filename, key in (("run_status.json", "run_status_raw_sha256"),
                           ("result_manifest.json", "result_manifest_raw_sha256"),
                           ("results.jsonl", "results_jsonl_raw_sha256")):
        artifact = output_dir / filename
        if artifact.exists() and not artifact.is_symlink() and artifact.is_file():
            body[key] = _event_file_pin(artifact)["sha256"]
    for label, log_path in (logs or {}).items():
        if log_path.exists() and not log_path.is_symlink() and log_path.is_file():
            body[f"{label}_log"] = _event_file_pin(log_path)
    _write_exclusive_json(path, body)
    return path


class _CapturingSubprocess:
    """Narrow proxy used to wait on the real Popen handle in frozen launchers."""

    DEVNULL = subprocess.DEVNULL

    def __init__(self):
        self.children: list[subprocess.Popen[Any]] = []
        self.start_ticks: dict[int, int] = {}

    def Popen(self, *args: Any, **kwargs: Any) -> subprocess.Popen[Any]:
        if self.children:
            raise QueueError("frozen one-shot launcher attempted more than one child process")
        child = subprocess.Popen(*args, **kwargs)
        self.children.append(child)
        self.start_ticks[child.pid] = linux_process_start_ticks(child.pid)
        return child


def _verify_launcher_source(path: Path, expected_sha256: str) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise QueueError(f"one-shot launcher source is missing or symlinked: {path}")
    raw = path.read_bytes()
    if sha256(raw) != expected_sha256:
        raise QueueError(f"one-shot launcher source raw SHA differs from its frozen pin: {path}")
    return raw


def _read_single_json_line(text: str, label: str) -> dict[str, Any]:
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise QueueError(f"{label} emitted no machine-readable launch record")
    try:
        value = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise QueueError(f"{label} launch record is not valid JSON") from exc
    if not isinstance(value, dict):
        raise QueueError(f"{label} launch record must be a JSON object")
    return value


def _verify_outer_launch_record(
    launch: Mapping[str, Any], claim: Mapping[str, Any], bundle: FrozenBundle,
    captured_pid: int, captured_ticks: int, launcher_sha256: str,
) -> Path:
    start = int(claim["block_start_index"])
    if (launch.get("status") != "STARTED_DETACHED"
            or launch.get("pid") != captured_pid
            or launch.get("start_ticks") != captured_ticks
            or launch.get("block_start", launch.get("block_start_index")) != start
            or launch.get("block_index") != start // BLOCK_ROWS
            or launch.get("block_rows") != BLOCK_ROWS
            or launch.get("row_ids") != list(claim["row_ids"])
            or str(Path(str(launch.get("output_root", "")))) != str(Path(str(claim["output_leaf"])) )):
        raise QueueError("frozen launcher receipt does not match the claimed child/block/output leaf")
    launch_pin = launch.get("launcher")
    if isinstance(launch_pin, Mapping) and launch_pin.get("sha256") not in (None, launcher_sha256):
        raise QueueError("frozen launcher receipt names a different launcher source SHA")
    plan_pin = launch.get("plan")
    if not isinstance(plan_pin, Mapping) or plan_pin.get("sha256", plan_pin.get("raw_sha256")) != bundle.spec.plan.sha256:
        raise QueueError("frozen launcher receipt does not bind the exact plan bytes")
    rows_pin = launch.get("planned_rows", launch.get("rows"))
    if not isinstance(rows_pin, Mapping) or rows_pin.get("sha256", rows_pin.get("raw_sha256")) != bundle.spec.planned_rows.sha256:
        raise QueueError("frozen launcher receipt does not bind the exact planned-row bytes")
    receipt_path = launch.get("launch_receipt")
    if not isinstance(receipt_path, str) or not Path(receipt_path).is_absolute():
        raise QueueError("frozen launcher did not return an absolute launch-receipt path")
    path = Path(receipt_path)
    if path.is_symlink() or not path.is_file():
        raise QueueError("frozen launcher receipt path is missing or symlinked")
    raw = path.read_bytes()
    if (len(raw) != launch.get("launch_receipt_bytes")
            or sha256(raw) != launch.get("launch_receipt_sha256")):
        raise QueueError("frozen launch receipt bytes differ from its stdout pin")
    try:
        doc = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QueueError("frozen launch receipt is malformed") from exc
    if (not isinstance(doc, Mapping) or doc.get("status") != "STARTED_DETACHED"
            or doc.get("pid") != captured_pid or doc.get("start_ticks") != captured_ticks
            or doc.get("block_index") != start // BLOCK_ROWS
            or doc.get("block_start", doc.get("block_start_index")) != start
            or doc.get("output_root") != str(Path(str(claim["output_leaf"])) )
            or doc.get("row_ids") != list(claim["row_ids"])):
        raise QueueError("frozen launch receipt body does not match the child/block claim")
    return path


def make_detached_main_launcher(source_path: Path, source_sha256: str,
                                receipt_dir: Path) -> "PinnedOneShotLauncher":
    """Import a frozen Muse/Ditto launcher and wait on its actual spawned child."""
    source_path = Path(source_path)
    receipt_dir = Path(receipt_dir)

    def invoke(bundle: FrozenBundle, claim: Mapping[str, Any]) -> tuple[Path, Path]:
        capture = _CapturingSubprocess()
        output = Path(str(claim["output_leaf"]))
        leaf_match = re.fullmatch(r"block_(\d{3})_attempt(\d{2})", output.name)
        if leaf_match is None or int(leaf_match.group(1)) != int(claim["block_index"]):
            raise QueueError("claimed output leaf is not a frozen block/attempt leaf")
        attempt = int(leaf_match.group(2))
        try:
            _verify_launcher_source(source_path, source_sha256)
            module_name = f"_fit40_queue_launcher_{source_sha256[:12]}_{claim['claim_nonce']}"
            spec = importlib.util.spec_from_file_location(module_name, source_path)
            if spec is None or spec.loader is None:
                raise QueueError("cannot create module spec for pinned launcher")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if not callable(getattr(module, "main", None)):
                raise QueueError("pinned detached launcher has no main() entry point")
            module.subprocess = capture
            old_argv = sys.argv
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            sys.argv = [str(source_path), "--block-start", str(claim["block_start_index"]),
                        "--attempt", str(attempt)]
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture
            launch_error: str | None = None
            launch_record: dict[str, Any] | None = None
            try:
                module.main()
                launch_record = _read_single_json_line(stdout_capture.getvalue(), "pinned launcher")
            except Exception as exc:
                launch_error = f"{type(exc).__name__}: {exc}"
            finally:
                sys.argv = old_argv
                sys.stdout = old_stdout
                sys.stderr = old_stderr
            if len(capture.children) > 1:
                raise QueueError("pinned launcher created more than one child")
            if not capture.children:
                receipt_path = _write_terminal_receipt(
                    receipt_dir, claim, child_pid=None, child_start_ticks=None,
                    exit_code=None, output_dir=output, launch_error=launch_error or "launcher returned without a child",
                )
                return receipt_path, output
            child = capture.children[0]
            ticks = capture.start_ticks[child.pid]
            launch_receipt_path = None
            if launch_error is None and launch_record is not None:
                try:
                    launch_receipt_path = _verify_outer_launch_record(
                        launch_record, claim, bundle, child.pid, ticks, source_sha256
                    )
                except Exception as exc:
                    launch_error = f"{type(exc).__name__}: {exc}"
            exit_code = child.wait()
            terminal_path = _write_terminal_receipt(
                receipt_dir, claim, child_pid=child.pid, child_start_ticks=ticks,
                exit_code=exit_code, output_dir=output, launch_receipt=launch_receipt_path,
                launch_error=launch_error,
            )
            return terminal_path, output
        except Exception as exc:
            # If import/verification failed before spawn, record that fact with
            # null PID/exit status. Never turn PID absence into a fabricated rc.
            if capture.children:
                child = capture.children[0]
                ticks = capture.start_ticks[child.pid]
                rc = child.wait()
                terminal_path = _write_terminal_receipt(
                    receipt_dir, claim, child_pid=child.pid, child_start_ticks=ticks,
                    exit_code=rc, output_dir=output,
                    launch_error=f"{type(exc).__name__}: {exc}",
                )
            else:
                terminal_path = _write_terminal_receipt(
                    receipt_dir, claim, child_pid=None, child_start_ticks=None,
                    exit_code=None, output_dir=output,
                    launch_error=f"{type(exc).__name__}: {exc}",
                )
            return terminal_path, output

    return PinnedOneShotLauncher(source_sha256=source_sha256, invoke=invoke)


def make_w2l_runner_launcher(receipt_dir: Path) -> "PinnedOneShotLauncher":
    """Build the exact W2L Stage-A argv from its frozen plan and run it as child."""
    spec = next(spec for spec in PINNED_MODELS if spec.model_id == MODEL_ORDER[0])
    source_pin = spec.launch_sources[0]

    def invoke(bundle: FrozenBundle, claim: Mapping[str, Any]) -> tuple[Path, Path]:
        execution = bundle.plan.get("execution", {})
        runtime_ref = execution.get("w2l_runtime_binding") if isinstance(execution, Mapping) else None
        if not isinstance(runtime_ref, Mapping) or not isinstance(runtime_ref.get("path"), str):
            raise QueueError("W2L plan lacks its exact staged runtime-binding path")
        runtime_binding_path = Path(runtime_ref["path"])
        project_root = runtime_binding_path.parent.parent
        stage_root = project_root.parent
        plan_path = project_root / Path(bundle.spec.plan.path).name
        rows_path = project_root / Path(bundle.spec.planned_rows.path).name
        runtime_raw = runtime_binding_path.read_bytes()
        if sha256(runtime_raw) != bundle.spec.runtime_binding.sha256:
            raise QueueError("staged W2L runtime binding differs from the frozen raw pin")
        plan_raw = plan_path.read_bytes()
        rows_raw = rows_path.read_bytes()
        if (len(plan_raw) != bundle.spec.plan.bytes or sha256(plan_raw) != bundle.spec.plan.sha256
                or len(rows_raw) != bundle.spec.planned_rows.bytes
                or sha256(rows_raw) != bundle.spec.planned_rows.sha256):
            raise QueueError("staged W2L plan or planned rows differ from their frozen raw pins")
        suffix = Path(*Path(source_pin.path).parts[Path(source_pin.path).parts.index("scripts"):])
        runner_path = project_root / suffix
        _verify_launcher_source(runner_path, source_pin.sha256)
        model = bundle.plan["models"][0]
        generator = model.get("generator")
        protocol = bundle.plan.get("protocol", {})
        syncnet = protocol.get("syncnet", {}) if isinstance(protocol, Mapping) else {}
        primary_audio = syncnet.get("primary_audio_input", {}) if isinstance(syncnet, Mapping) else {}
        if not isinstance(generator, Mapping) or not isinstance(primary_audio, Mapping):
            raise QueueError("W2L plan lacks the frozen generator or SyncNet audio protocol")
        python_path = generator.get("python")
        wav2lip_root = generator.get("repo")
        generator_child = generator.get("child_script_path")
        project_repo = generator.get("repo")
        if not all(isinstance(value, str) and Path(value).is_absolute()
                   for value in (python_path, wav2lip_root, generator_child, project_repo)):
            raise QueueError("W2L plan lacks exact absolute runtime/source paths")
        repository_root = Path(project_repo).parents[1]
        syncnet_root = repository_root / "third_party" / "syncnet_python"
        evaluator = project_root / "scripts/experiments/cross_tfg_audio_head_20261003/stagea_syncnet_evaluator.py"
        checkpoint = syncnet_root / "data/syncnet_v2.model"
        ffmpeg_sha = primary_audio.get("ffmpeg_sha256")
        pcm16_sha = primary_audio.get("code_sha256")
        if any(not isinstance(value, str) or len(value) != 64 for value in (ffmpeg_sha, pcm16_sha)):
            raise QueueError("W2L plan has no exact FFmpeg/PCM16 preprocessing pins")
        output = Path(str(claim["output_leaf"]))
        argv = [
            python_path, str(runner_path),
            "--plan", str(plan_path), "--planned-rows", str(rows_path),
            "--expected-plan-raw-sha256", bundle.spec.plan.sha256,
            "--output-root", str(output), "--project-root", str(repository_root),
            "--wav2lip-python", python_path, "--wav2lip-root", wav2lip_root,
            "--generator-child", generator_child,
            "--syncnet-root", str(syncnet_root), "--syncnet-python", python_path,
            "--syncnet-checkpoint", str(checkpoint), "--syncnet-evaluator", str(evaluator),
            "--ffmpeg", "/usr/bin/ffmpeg", "--ffprobe", "/usr/bin/ffprobe",
            "--ffmpeg-sha256", ffmpeg_sha, "--pcm16-lineage-sha256", pcm16_sha,
            "--generator-timeout", "900", "--syncnet-timeout", "1800", "--ffmpeg-timeout", "300",
            "--block-start", str(claim["block_start_index"]), "--block-count", str(BLOCK_ROWS),
        ]
        nonce = str(claim["claim_nonce"])
        stdout_path = Path(receipt_dir) / f"{nonce}_stdout.log"
        stderr_path = Path(receipt_dir) / f"{nonce}_stderr.log"
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with stdout_path.open("xb") as stdout_stream, stderr_path.open("xb") as stderr_stream:
                child = subprocess.Popen(argv, cwd=str(repository_root), stdin=subprocess.DEVNULL,
                                         stdout=stdout_stream, stderr=stderr_stream, close_fds=True)
                ticks = linux_process_start_ticks(child.pid)
                exit_code = child.wait()
        except Exception as exc:
            terminal_path = _write_terminal_receipt(
                Path(receipt_dir), claim, child_pid=None, child_start_ticks=None,
                exit_code=None, output_dir=output,
                launch_error=f"{type(exc).__name__}: {exc}",
                logs={"stdout": stdout_path, "stderr": stderr_path},
            )
            return terminal_path, output
        terminal_path = _write_terminal_receipt(
            Path(receipt_dir), claim, child_pid=child.pid, child_start_ticks=ticks,
            exit_code=exit_code, output_dir=output,
            logs={"stdout": stdout_path, "stderr": stderr_path},
        )
        return terminal_path, output

    return PinnedOneShotLauncher(source_sha256=source_pin.sha256, invoke=invoke)


class FixedFitQueueController:
    """Append-only claim/terminal state machine for the fixed model queue."""

    def __init__(self, bundles: Sequence[FrozenBundle], state_dir: Path):
        if tuple(bundle.spec.model_id for bundle in bundles) != MODEL_ORDER:
            raise QueueError("controller requires the frozen W2L→Muse→Ditto allowlist order")
        self.bundles = {bundle.spec.model_id: bundle for bundle in bundles}
        self.order = MODEL_ORDER
        self.state_dir = Path(state_dir)
        self.events_dir = self.state_dir / "events"
        self.claims_dir = self.state_dir / "claims"
        self.lock_path = self.state_dir / "exclusive.lock"

    def _replay(self) -> QueueState:
        progress = {model: ModelProgress() for model in self.order}
        state = QueueState(progress=progress)
        if not self.events_dir.exists():
            return state
        files = sorted(self.events_dir.glob("*.json"))
        expected_seq = 1
        for path in files:
            event, pin = _read_json(path, "queue event")
            self_hash = event.get("self_sha256")
            if (event.get("schema") != EVENT_SCHEMA or event.get("seq") != expected_seq
                    or event.get("previous_sha256") != state.last_event_sha256
                    or self_hash != _self_hash(event, "self_sha256")):
                raise QueueError(f"queue event chain/self seal is invalid: {path}")
            kind = event.get("event")
            model = event.get("model_id")
            if model not in self.bundles:
                raise QueueError(f"queue event names a model outside the allowlist: {path}")
            if kind == "CLAIM":
                if state.active_claim is not None:
                    raise QueueError("queue journal has concurrent active claims")
                offset = state.progress[model].next_start
                if (state.progress[model].paused or event.get("block_start_index") != offset
                        or offset < 0 or offset >= TOTAL_ROWS or offset % BLOCK_ROWS):
                    raise QueueError(f"claim is out of sequence for {model}")
                if event.get("plan_raw_sha256") != self.bundles[model].spec.plan.sha256:
                    raise QueueError("claim changed frozen plan identity")
                launcher_sha = event.get("launcher_source_sha256")
                if launcher_sha not in {pin.sha256 for pin in self.bundles[model].spec.launch_sources}:
                    raise QueueError("claim launcher source is outside the frozen model allowlist")
                expected_rows = self.bundles[model].row_ids[offset:offset + BLOCK_ROWS]
                if event.get("row_ids") != list(expected_rows):
                    raise QueueError("claim row IDs differ from the frozen six-row block")
                eligible = event.get("eligible_model_ids")
                if (not isinstance(eligible, list) or len(set(eligible)) != len(eligible)
                        or any(candidate not in self.bundles for candidate in eligible)):
                    raise QueueError("claim eligibility snapshot is malformed")
                eligible = [candidate for candidate in self.order if candidate in eligible
                            and not state.progress[candidate].paused
                            and state.progress[candidate].next_start < TOTAL_ROWS]
                cursor = self.order.index(state.last_served_model) + 1 if state.last_served_model in self.order else 0
                expected_model = next((self.order[(cursor + delta) % len(self.order)]
                                       for delta in range(len(self.order))
                                       if self.order[(cursor + delta) % len(self.order)] in eligible), None)
                if expected_model != model:
                    raise QueueError("claim violates fixed round-robin order among ready models")
                gate = event.get("resource_gate")
                absent = event.get("absent_leaf_proof")
                if not isinstance(gate, Mapping) or not isinstance(absent, Mapping):
                    raise QueueError("claim omits live resource/absent-leaf evidence")
                self._validate_gate(model, self.bundles[model], offset, gate)
                self._validate_absent_leaf(model, self.bundles[model], offset, absent)
                leaf = event.get("output_leaf")
                if not isinstance(leaf, str) or leaf in state.used_output_leaves:
                    raise QueueError("claim leaf is empty or previously used")
                state.used_output_leaves.add(leaf)
                state.active_claim = event
            elif kind in {"BLOCK_PASS", "BLOCK_PAUSED_FAILURE"}:
                active = state.active_claim
                if active is None or event.get("claim_nonce") != active.get("claim_nonce"):
                    raise QueueError("terminal event has no matching active claim")
                if model != active.get("model_id") or event.get("block_start_index") != active.get("block_start_index"):
                    raise QueueError("terminal event changed active block identity")
                state.last_served_model = model
                if kind == "BLOCK_PASS":
                    if state.progress[model].paused:
                        raise QueueError("paused model cannot advance")
                    state.progress[model].next_start += BLOCK_ROWS
                else:
                    state.progress[model].paused = True
                    state.progress[model].pause_reason = str(event.get("failure_reason") or "terminal block failure")
                state.active_claim = None
            else:
                raise QueueError(f"unknown queue event kind: {kind!r}")
            state.event_count = expected_seq
            state.last_event_sha256 = str(self_hash)
            expected_seq += 1
        return state

    @staticmethod
    def _validate_gate(model_id: str, bundle: FrozenBundle, start: int, gate: Mapping[str, Any]) -> None:
        digest = gate.get("evidence_raw_sha256")
        if (gate.get("status") != "PASS" or gate.get("model_id") != model_id
                or gate.get("block_start_index") != start
                or gate.get("plan_raw_sha256") != bundle.spec.plan.sha256
                or not isinstance(digest, str) or len(digest) != 64):
            raise QueueError("existing per-block live resource gate is missing, stale, or bound to another block")

    @staticmethod
    def _validate_absent_leaf(model_id: str, bundle: FrozenBundle, start: int,
                              proof: Mapping[str, Any]) -> str:
        leaf = proof.get("output_leaf")
        digest = proof.get("evidence_raw_sha256")
        if (proof.get("exists") is not False or proof.get("status") != "PASS"
                or proof.get("model_id") != model_id or proof.get("block_start_index") != start
                or proof.get("plan_raw_sha256") != bundle.spec.plan.sha256
                or not isinstance(leaf, str) or not leaf.startswith("/")
                or not isinstance(digest, str) or len(digest) != 64):
            raise QueueError("output-leaf absent proof is missing, stale, or not bound to this block")
        expected_base = bundle.plan.get("execution", {}).get("output_root")
        if not isinstance(expected_base, str):
            runtime_key = {
                MODEL_ORDER[0]: "w2l_runtime_binding",
                MODEL_ORDER[1]: "muse_runtime_binding",
                MODEL_ORDER[2]: "ditto_runtime_binding",
            }[model_id]
            runtime_ref = bundle.plan.get("execution", {}).get(runtime_key)
            runtime_path = runtime_ref.get("path") if isinstance(runtime_ref, Mapping) else None
            if not isinstance(runtime_path, str) or not Path(runtime_path).is_absolute():
                raise QueueError("frozen plan has no absolute output namespace binding")
            expected_base = str(Path(runtime_path).parents[2] / "output")
        leaf_path = Path(leaf)
        if (not leaf_path.is_absolute() or str(leaf_path.parent) != str(Path(expected_base))
                or not re.fullmatch(rf"block_{start // BLOCK_ROWS:03d}_attempt\d{{2}}", leaf_path.name)):
            raise QueueError("output leaf does not match the frozen model output namespace/block")
        return leaf

    def _append_event(self, state: QueueState, event_body: Mapping[str, Any]) -> dict[str, Any]:
        self.events_dir.mkdir(parents=True, exist_ok=True)
        event = dict(event_body)
        event.update({
            "schema": EVENT_SCHEMA,
            "seq": state.event_count + 1,
            "previous_sha256": state.last_event_sha256,
        })
        event["self_sha256"] = _self_hash(event, "self_sha256")
        path = self.events_dir / f"{event['seq']:06d}_{event['event'].lower()}_{event.get('claim_nonce','')}.json"
        raw = canonical(event) + b"\n"
        try:
            fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise QueueError(f"refusing to overwrite existing queue event: {path}") from exc
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
        except Exception:
            try:
                path.unlink()
            except OSError:
                pass
            raise
        return event

    def claim_next(
        self,
        *,
        ready_model_ids: Sequence[str],
        resource_gates: Mapping[str, Mapping[str, Any]],
        absent_leaf_proofs: Mapping[str, Mapping[str, Any]],
        owner_pid: int,
        owner_start_ticks: int,
        launcher_sha256_by_model: Mapping[str, str] | None = None,
        nonce: str | None = None,
    ) -> dict[str, Any] | None:
        """Create one serialized claim; do not launch the model process here."""
        if type(owner_pid) is not int or owner_pid <= 0 or type(owner_start_ticks) is not int or owner_start_ticks <= 0:
            raise QueueError("queue owner PID/start ticks must be positive integers")
        _require_current_owner(owner_pid, owner_start_ticks)
        state = self._replay()
        if state.active_claim is not None or self.lock_path.exists():
            raise QueueError("an active or stale exclusive queue lock exists; no automatic recovery")
        ready = set(ready_model_ids)
        unknown = ready - set(self.order)
        if unknown:
            raise QueueError(f"ready set includes models outside frozen allowlist: {sorted(unknown)}")
        start_index = self.order.index(state.last_served_model) + 1 if state.last_served_model in self.order else 0
        eligible: list[str] = []
        valid_proofs: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        for model_id in self.order:
            progress = state.progress[model_id]
            if model_id not in ready or progress.paused or progress.next_start >= TOTAL_ROWS:
                continue
            gate = resource_gates.get(model_id)
            absent = absent_leaf_proofs.get(model_id)
            if not isinstance(gate, Mapping) or not isinstance(absent, Mapping):
                continue
            try:
                self._validate_gate(model_id, self.bundles[model_id], progress.next_start, gate)
                leaf = self._validate_absent_leaf(model_id, self.bundles[model_id], progress.next_start, absent)
            except QueueError:
                continue
            if leaf in state.used_output_leaves:
                continue
            eligible.append(model_id)
            valid_proofs[model_id] = (gate, absent)
        selected = next((self.order[(start_index + delta) % len(self.order)]
                         for delta in range(len(self.order))
                         if self.order[(start_index + delta) % len(self.order)] in eligible), None)
        if selected is None:
            return None
        if nonce is None:
            nonce = secrets.token_hex(16)
        if (not isinstance(nonce, str) or len(nonce) != 32
                or any(char not in "0123456789abcdef" for char in nonce)):
            raise QueueError("claim nonce must be 32 lowercase hexadecimal characters")
        bundle = self.bundles[selected]
        start = state.progress[selected].next_start
        allowed_launcher_shas = {pin.sha256 for pin in bundle.spec.launch_sources}
        launcher_sha = (launcher_sha256_by_model or {}).get(selected)
        if launcher_sha is None:
            launcher_sha = bundle.spec.launch_sources[0].sha256
        if launcher_sha not in allowed_launcher_shas:
            raise QueueError("one-shot launcher is outside the frozen model source allowlist")
        gate = dict(valid_proofs[selected][0])
        absent = dict(valid_proofs[selected][1])
        rows = list(bundle.row_ids[start:start + BLOCK_ROWS])
        claim = {
            "event": "CLAIM",
            "claim_nonce": nonce,
            "model_id": selected,
            "block_index": start // BLOCK_ROWS,
            "block_start_index": start,
            "block_rows": BLOCK_ROWS,
            "row_ids": rows,
            "plan_raw_sha256": bundle.spec.plan.sha256,
            "planned_rows_sha256": bundle.spec.planned_rows.sha256,
            "runtime_binding_sha256": bundle.spec.runtime_binding.sha256,
            "launch_source_sha256s": [pin.sha256 for pin in bundle.spec.launch_sources],
            "launcher_source_sha256": launcher_sha,
            "resource_gate": gate,
            "output_leaf": absent["output_leaf"],
            "absent_leaf_proof": absent,
            "eligible_model_ids": eligible,
            "owner_pid": owner_pid,
            "owner_start_ticks": owner_start_ticks,
        }
        self.state_dir.mkdir(parents=True, exist_ok=True)
        lock_body = {
            "schema": "fit40_queue_exclusive_lock_v1",
            "claim_nonce": nonce,
            "model_id": selected,
            "block_start_index": start,
            "owner_pid": owner_pid,
            "owner_start_ticks": owner_start_ticks,
        }
        lock_raw = canonical(lock_body) + b"\n"
        try:
            fd = os.open(str(self.lock_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as exc:
            raise QueueError("queue exclusive lock raced another controller") from exc
        with os.fdopen(fd, "wb") as stream:
            stream.write(lock_raw)
            stream.flush()
            os.fsync(stream.fileno())
        self._append_event(state, claim)
        return claim

    def _validate_process_receipt(self, claim: Mapping[str, Any], path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
        receipt, pin = _read_json(path, "terminal process receipt")
        if receipt.get("schema") != PROCESS_SCHEMA or receipt.get("self_sha256") != _self_hash(receipt, "self_sha256"):
            raise QueueError("terminal process receipt schema/self hash is invalid")
        for key, expected in (
            ("claim_nonce", claim.get("claim_nonce")),
            ("model_id", claim.get("model_id")),
            ("block_index", claim.get("block_index")),
            ("block_start_index", claim.get("block_start_index")),
            ("plan_raw_sha256", claim.get("plan_raw_sha256")),
            ("output_leaf", claim.get("output_leaf")),
        ):
            if receipt.get(key) != expected:
                raise QueueError(f"terminal process receipt changed {key}")
        status = receipt.get("terminal_status")
        if status == "EXITED":
            if (type(receipt.get("child_pid")) is not int or receipt["child_pid"] <= 0
                    or type(receipt.get("child_start_ticks")) is not int or receipt["child_start_ticks"] <= 0
                    or type(receipt.get("exit_code")) is not int):
                raise QueueError("terminal process receipt does not identify an exited child")
        elif status == "FAILED_BEFORE_SPAWN":
            if (receipt.get("child_pid") is not None or receipt.get("child_start_ticks") is not None
                    or receipt.get("exit_code") is not None
                    or not isinstance(receipt.get("launch_error"), str)
                    or not receipt.get("launch_error")):
                raise QueueError("pre-spawn failure receipt must preserve the absence of child/exit evidence")
        else:
            raise QueueError("terminal process receipt has an unsupported terminal status")
        return receipt, pin

    def _verify_pass_outputs(self, claim: Mapping[str, Any], receipt: Mapping[str, Any],
                             output_dir: Path) -> dict[str, Any]:
        model_id = str(claim["model_id"])
        bundle = self.bundles[model_id]
        start = int(claim["block_start_index"])
        expected_ids = list(bundle.row_ids[start:start + BLOCK_ROWS])
        claimed_output = Path(str(claim.get("output_leaf", "")))
        try:
            actual_output = Path(output_dir).resolve(strict=True)
            expected_output = claimed_output.resolve(strict=True)
        except OSError as exc:
            raise QueueError("claimed output leaf is not an existing readable directory") from exc
        if (actual_output != expected_output or receipt.get("output_leaf") != str(expected_output)
                or actual_output.is_symlink() or not actual_output.is_dir()):
            raise QueueError("terminal outputs are outside the exact claimed output leaf")
        run_status, status_pin = _read_json(output_dir / "run_status.json", "run status")
        manifest, manifest_pin = _read_json(output_dir / "result_manifest.json", "result manifest")
        ledger_path = output_dir / "results.jsonl"
        ledger_pin = _event_file_pin(ledger_path)
        if receipt.get("exit_code") != 0:
            raise QueueError("renderer child exited nonzero")
        for obj, label in ((run_status, "run status"), (manifest, "result manifest")):
            if (obj.get("block_index") != claim["block_index"]
                    or obj.get("block_start_index") != start or obj.get("block_rows") != BLOCK_ROWS
                    or obj.get("planned_rows") != TOTAL_ROWS
                    or obj.get("plan_raw_sha256") != bundle.spec.plan.sha256):
                raise QueueError(f"{label} is not bound to the claimed plan/block")
        if run_status.get("schema_version") != "cross_tfg_audio_head_stagea_v1":
            raise QueueError("run status schema is not the frozen Stage-A block status")
        if run_status.get("status") != "BLOCK_COMPLETE" or run_status.get("completed_rows") != BLOCK_ROWS:
            raise QueueError("run status is not a completed six-row block")
        if run_status.get("planned_rows_sha256") != bundle.spec.planned_rows.sha256:
            raise QueueError("run status planned-row source pin mismatch")
        if run_status.get("pid") != receipt.get("child_pid"):
            raise QueueError("terminal child PID differs from run status")
        if run_status.get("result_manifest_sha256") != manifest_pin["sha256"]:
            raise QueueError("run status does not bind exact result manifest bytes")
        if receipt.get("run_status_raw_sha256") != status_pin["sha256"]:
            raise QueueError("terminal receipt does not bind exact run status bytes")
        if receipt.get("result_manifest_raw_sha256") != manifest_pin["sha256"]:
            raise QueueError("terminal receipt does not bind exact result manifest bytes")
        if receipt.get("results_jsonl_raw_sha256") != ledger_pin["sha256"]:
            raise QueueError("terminal receipt does not bind exact result ledger bytes")
        if manifest.get("schema_version") != PLAN_SCHEMA or manifest.get("status") != "BLOCK_COMPLETE":
            raise QueueError("result manifest is not a completed Stage-A block")
        if (manifest.get("parent_planned_rows") != TOTAL_ROWS or manifest.get("planned_rows") != TOTAL_ROWS
                or manifest.get("result_rows") != BLOCK_ROWS):
            raise QueueError("result manifest parent/block row counts are not exact")
        if manifest.get("planned_rows_sha256") not in (None, bundle.spec.planned_rows.sha256):
            raise QueueError("result manifest planned-row pin mismatch")
        if manifest.get("results_jsonl_sha256") != ledger_pin["sha256"]:
            raise QueueError("result manifest does not bind the exact six-row JSONL")
        pass_counts = {"FAILED": 0, "INCOMPLETE": 0, "PASS": BLOCK_ROWS}
        if run_status.get("status_counts") != pass_counts or manifest.get("status_counts") != pass_counts:
            raise QueueError("six-row terminal status counts are not all PASS")
        if manifest.get("block_sample_id") != bundle.rows[start].get("sample_id"):
            raise QueueError("result manifest block sample differs from frozen plan")
        counts = manifest.get("counts_by_model_method")
        expected_counts = [
            {"method_id": method, "model_id": model_id, "planned_rows": 1,
             "result_rows": 1, "status_counts": {"FAILED": 0, "INCOMPLETE": 0, "PASS": 1}}
            for method in METHODS
        ]
        if counts != expected_counts:
            raise QueueError("result manifest method/model counts differ from the frozen six-method block")
        try:
            results = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise QueueError("result JSONL is malformed") from exc
        if len(results) != BLOCK_ROWS or [row.get("row_id") for row in results] != expected_ids:
            raise QueueError("result JSONL does not contain the exact ordered six claimed rows")
        for row, planned in zip(results, bundle.rows[start:start + BLOCK_ROWS]):
            if (row.get("status") != "PASS" or row.get("model_id") != model_id
                    or row.get("method_id") != planned.get("method_id")
                    or row.get("sample_id") != planned.get("sample_id")
                    or row.get("seed") != planned.get("seed")):
                raise QueueError("result row does not match frozen plan identity or PASS status")
        return {
            "run_status": status_pin,
            "result_manifest": manifest_pin,
            "results_jsonl": ledger_pin,
            "result_rows": BLOCK_ROWS,
            "status_counts": pass_counts,
        }

    def finish_claim(self, claim: Mapping[str, Any], *, owner_pid: int, owner_start_ticks: int,
                     process_receipt_path: Path, output_dir: Path) -> dict[str, Any]:
        """Verify one terminal leaf, append PASS or pause-on-failure, then release its lock."""
        _require_current_owner(owner_pid, owner_start_ticks)
        state = self._replay()
        active = state.active_claim
        if (active is None or active.get("claim_nonce") != claim.get("claim_nonce")
                or active.get("owner_pid") != owner_pid or active.get("owner_start_ticks") != owner_start_ticks):
            raise QueueError("claim is not the current exclusive owner")
        try:
            lock = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise QueueError("exclusive lock is missing or malformed") from exc
        if (lock.get("claim_nonce") != active.get("claim_nonce") or lock.get("owner_pid") != owner_pid
                or lock.get("owner_start_ticks") != owner_start_ticks):
            raise QueueError("exclusive lock owner does not match the claim")
        process, process_pin = self._validate_process_receipt(active, process_receipt_path)
        failure_reason: str | None = None
        proof: dict[str, Any] = {"process_receipt": process_pin}
        if process["terminal_status"] == "FAILED_BEFORE_SPAWN":
            failure_reason = f"launcher failed before spawn: {process['launch_error']}"
        elif process.get("launch_error"):
            failure_reason = f"launcher failed after spawn: {process['launch_error']}"
        elif process["exit_code"] == 0:
            try:
                proof.update(self._verify_pass_outputs(active, process, output_dir))
            except QueueError as exc:
                failure_reason = str(exc)
        else:
            failure_reason = f"renderer exited with code {process['exit_code']}"
            for name in ("run_status.json", "result_manifest.json", "results.jsonl"):
                path = output_dir / name
                if path.exists() and not path.is_symlink() and path.is_file():
                    proof[name] = _event_file_pin(path)
        outcome = "BLOCK_PASS" if failure_reason is None else "BLOCK_PAUSED_FAILURE"
        terminal = self._append_event(state, {
            "event": outcome,
            "claim_nonce": active["claim_nonce"],
            "model_id": active["model_id"],
            "block_index": active["block_index"],
            "block_start_index": active["block_start_index"],
            "output_leaf": active["output_leaf"],
            "failure_reason": failure_reason,
            "process_terminal": {
                "child_pid": process["child_pid"],
                "child_start_ticks": process["child_start_ticks"],
                "exit_code": process["exit_code"],
            },
            "verified_output_pins": proof,
        })
        self.lock_path.unlink()
        return terminal

    def snapshot(self) -> dict[str, Any]:
        state = self._replay()
        return {
            "schema": SCHEMA,
            "model_order": list(self.order),
            "next_start_by_model": {model: state.progress[model].next_start for model in self.order},
            "paused_models": {
                model: state.progress[model].pause_reason
                for model in self.order if state.progress[model].paused
            },
            "last_served_model": state.last_served_model,
            "active_claim": state.active_claim,
            "used_output_leaf_count": len(state.used_output_leaves),
            "event_count": state.event_count,
            "last_event_sha256": state.last_event_sha256,
        }


def run_ready_blocks(
    controller: FixedFitQueueController,
    *,
    owner_pid: int,
    owner_start_ticks: int,
    available_model_ids: Callable[[], Sequence[str]],
    live_resource_gate: Callable[[str, int], Mapping[str, Any] | None],
    absent_output_leaf: Callable[[str, int], Mapping[str, Any] | None],
    launchers: Mapping[str, "PinnedOneShotLauncher"],
    max_blocks: int | None = None,
) -> list[dict[str, Any]]:
    """Run pinned one-shot adapters in fair order until none is ready.

    ``launch_and_wait`` must invoke only the current bundle's pinned launcher,
    wait for its child to exit, and return (terminal process receipt,
    readback block directory).  This scheduler rechecks the existing resource
    gate before every block.  An adapter exception is not an implicit retry:
    it leaves the active claim/lock intact for manual inspection.
    """
    terminal_events: list[dict[str, Any]] = []
    while max_blocks is None or len(terminal_events) < max_blocks:
        ready = tuple(available_model_ids())
        gates: dict[str, Mapping[str, Any]] = {}
        leaves: dict[str, Mapping[str, Any]] = {}
        state = controller._replay()
        for model in ready:
            launcher = launchers.get(model)
            if (model not in controller.bundles or launcher is None
                    or launcher.source_sha256 not in {
                        pin.sha256 for pin in controller.bundles[model].spec.launch_sources
                    }):
                continue
            start = state.progress[model].next_start
            if state.progress[model].paused or start >= TOTAL_ROWS:
                continue
            gate = live_resource_gate(model, start)
            leaf = absent_output_leaf(model, start)
            if gate is not None:
                gates[model] = gate
            if leaf is not None:
                leaves[model] = leaf
        claim = controller.claim_next(
            ready_model_ids=ready,
            resource_gates=gates,
            absent_leaf_proofs=leaves,
            owner_pid=owner_pid,
            owner_start_ticks=owner_start_ticks,
            launcher_sha256_by_model={model: launchers[model].source_sha256 for model in ready
                                      if model in launchers},
        )
        if claim is None:
            break
        try:
            launcher = launchers[claim["model_id"]]
            process_receipt_path, output_dir = launcher.invoke(controller.bundles[claim["model_id"]], claim)
        except Exception:
            # The claim and lock remain, preventing an accidental replay or a
            # competing launcher.  A human must inspect and close this leaf.
            raise
        terminal_events.append(controller.finish_claim(
            claim,
            owner_pid=owner_pid,
            owner_start_ticks=owner_start_ticks,
            process_receipt_path=process_receipt_path,
            output_dir=output_dir,
        ))
    return terminal_events


@dataclass(frozen=True)
class PinnedOneShotLauncher:
    """One model-specific adapter pinned to a source in that model's allowlist."""

    source_sha256: str
    invoke: Callable[[FrozenBundle, Mapping[str, Any]], tuple[Path, Path]]
