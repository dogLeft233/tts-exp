from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import subprocess
from pathlib import Path

from .protocol import (
    REPO_ROOT,
    ProtocolError,
    candidate_rows,
    canonical_hash,
    file_sha256,
    inspect_clip,
    load_self_hashed,
    select_cohort,
    validate_visual_audit,
    write_json,
)

CHANGE_ROOT = REPO_ROOT / "openspec/changes/probe-fresh-source-cross-generator"
UNBLOCK_CHANGE_ROOT = REPO_ROOT / "openspec/changes/unblock-fresh-source-inputs"
DIRECT_AUDIO_CODE = REPO_ROOT / "scripts/experiments/lrs3_direct_audio/run_wavlm_hifigan_lrs3.py"
KNN_VC_REVISION = "c616845c4e309e24d5927f15adbdf277a3d65358"


def _body_hash(value: dict) -> str:
    body = dict(value)
    body.pop("artifact_sha256", None)
    return hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _manifest(paths: list[Path]) -> dict:
    """Hash the exact spec/code files used by a P run."""
    expanded: list[Path] = []
    for path in paths:
        if path.is_file():
            expanded.append(path)
        elif path.is_dir():
            expanded.extend(
                item for item in path.rglob("*")
                if item.is_file() and "__pycache__" not in item.parts and ".git" not in item.parts
            )
    rows = []
    for path in sorted(set(expanded)):
        resolved = path.resolve()
        try:
            label = str(resolved.relative_to(REPO_ROOT.resolve()))
        except ValueError:
            label = str(resolved)
        rows.append({"path": label, "sha256": file_sha256(path)})
    return {"files": rows, "sha256": canonical_hash(rows)}


def _torch_hub_dir() -> Path:
    torch_home = os.environ.get("TORCH_HOME")
    if torch_home:
        return Path(torch_home).expanduser() / "hub"
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return cache_root / "torch" / "hub"


def _model_identity() -> dict:
    """Return stable model/checkpoint identity without loading torch."""
    hub = _torch_hub_dir()
    source = hub / f"bshall_knn-vc_{KNN_VC_REVISION}"
    checkpoints = [hub / "checkpoints" / name for name in ("WavLM-Large.pt", "prematch_g_02500000.pt")]
    checkpoint_rows = []
    for path in checkpoints:
        row = {"path": str(path), "exists": path.is_file()}
        if path.is_file():
            row.update({"sha256": file_sha256(path), "size": path.stat().st_size})
        checkpoint_rows.append(row)
    source_manifest = _manifest([source]) if source.is_dir() else {"files": [], "sha256": None}
    return {
        "revision": KNN_VC_REVISION,
        "source": str(source),
        "source_manifest": source_manifest,
        "checkpoints": checkpoint_rows,
    }


def _execution_contract(history: dict, rows: list[dict], visual_audit: Path | None, *, run_root: Path | None = None, pool: dict | None = None) -> dict:
    """Build the identity required before a blocked run may be resumed."""
    spec_manifest = _manifest([CHANGE_ROOT / "design.md", CHANGE_ROOT / "specs", UNBLOCK_CHANGE_ROOT / "design.md", UNBLOCK_CHANGE_ROOT / "specs"])
    code_manifest = _manifest([
        Path(__file__),
        REPO_ROOT / "scripts/experiments/fresh_source_inputs/protocol.py",
        REPO_ROOT / "scripts/experiments/fresh_source_inputs/validate.py",
        REPO_ROOT / "scripts/experiments/fresh_source_inputs/acquire.py",
        REPO_ROOT / "scripts/experiments/fresh_source_inputs/visual_audit.py",
        DIRECT_AUDIO_CODE,
    ])
    visual_identity = {
        "path": str(visual_audit.resolve()) if visual_audit else None,
        "sha256": file_sha256(visual_audit) if visual_audit else None,
    }
    acquisition_identity: dict[str, object] = {}
    if run_root is not None:
        acquisition = run_root / "acquisition"
        for name in ("plan.json", "source_manifest.json", "pool.json", "ledger.jsonl", "visual_audit.json", "clip_audit.jsonl"):
            path = acquisition / name
            if path.is_file():
                acquisition_identity[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
        if pool is not None:
            acquisition_identity["pool_body_sha256"] = canonical_hash(pool)
        if visual_audit and visual_audit.is_file():
            try:
                visual_value = json.loads(visual_audit.read_text(encoding="utf-8"))
                evidence: list[dict[str, str]] = []
                for item in (visual_value.get("groups", {}) if isinstance(visual_value, dict) else {}).values():
                    if not isinstance(item, dict):
                        continue
                    for key in ("landmark_array_path", "input_review_path", "preview_path"):
                        path = Path(str(item.get(key, ""))) if item.get(key) else None
                        if path and path.is_file():
                            evidence.append({"path": str(path.resolve()), "sha256": file_sha256(path)})
                acquisition_identity["visual_evidence"] = sorted(evidence, key=lambda row: row["path"])
            except (OSError, ValueError, TypeError):
                acquisition_identity["visual_evidence"] = "unreadable"
    input_identity = {
        "history_manifest_sha256": history.get("scan_file_manifest_sha256"),
        "history_groups": history.get("groups", []),
        "candidate_rows": rows,
        "visual_audit": visual_identity,
        "acquisition": acquisition_identity,
    }
    contract = {
        "schema_version": 1,
        "spec_manifest": spec_manifest,
        "code_manifest": code_manifest,
        "model_identity": _model_identity(),
        "input_sha256": canonical_hash(input_identity),
    }
    contract["contract_sha256"] = canonical_hash(contract)
    return contract


def _assert_shared_matches(run_root: Path, names: tuple[str, ...] = ("history_groups.json", "candidate_audit.json", "cohort.json", "inputs.json", "validation.json")) -> None:
    shared = run_root / "run" / "shared"
    for name in names:
        root_path = run_root / name
        shared_path = shared / name
        if not root_path.is_file() or not shared_path.is_file():
            raise ProtocolError(f"root/shared artifact missing: {name}")
        if _body_hash(load_self_hashed(root_path)) != _body_hash(load_self_hashed(shared_path)):
            raise ProtocolError(f"root/shared artifact mismatch: {name}")


def _source_inside_data_root(source: Path, group: str) -> Path:
    data_root = (REPO_ROOT / "data/dataset_samples/lrs3/pretrain").resolve()
    resolved = source.resolve()
    try:
        relative = resolved.relative_to(data_root)
    except ValueError as exc:
        raise ProtocolError(f"candidate source outside frozen data root: {source}") from exc
    if not relative.parts or relative.parts[0] != group or not resolved.is_file():
        raise ProtocolError(f"candidate source/group mismatch: {source} != {group}")
    return resolved


def _decoded_pcm_identity(path: Path) -> tuple[str, int]:
    """Return the exact serialized PCM16 hash and sample count for a WAV."""
    import numpy as np
    import soundfile as sf

    values, sample_rate = sf.read(path, dtype="int16", always_2d=False)
    values = np.asarray(values)
    if int(sample_rate) != 16000 or values.ndim != 1:
        raise ProtocolError(f"natural PCM decode contract failed: {path}")
    pcm = values.astype("int16", copy=False).tobytes()
    return hashlib.sha256(pcm).hexdigest(), int(values.size)


def run_prepare(run_root: Path, *, visual_audit: Path | None = None, resume: bool = False) -> int:
    freeze_path = run_root / "run" / "shared" / "freeze.json"
    if freeze_path.exists():
        raise RuntimeError(f"frozen run is immutable; create a new run instead: {run_root}")
    existing = (run_root / "cohort.json").exists()
    if existing and not resume:
        raise RuntimeError(f"frozen run already exists; pass --resume explicitly: {run_root}")
    previous = {}
    previous_inputs = {}
    if existing and resume:
        previous = load_self_hashed(run_root / "cohort.json")
        previous_inputs = load_self_hashed(run_root / "inputs.json") if (run_root / "inputs.json").is_file() else {}
        if previous.get("status") == "GO" or previous_inputs.get("status") == "GO":
            raise RuntimeError("cannot resume a run after a GO cohort or GO inputs; create a new run")
    audit = None
    pool = None
    acquisition = run_root / "acquisition"
    plan_path = acquisition / "plan.json"
    pool_path = acquisition / "pool.json"
    if plan_path.is_file() or pool_path.is_file():
        if not plan_path.is_file() or not pool_path.is_file():
            raise RuntimeError("acquisition plan and pool must be supplied together")
        plan = load_self_hashed(plan_path)
        pool = load_self_hashed(pool_path)
        if pool.get("source_plan_sha256") != file_sha256(plan_path):
            raise RuntimeError("acquisition pool is not bound to the current plan")
        if plan.get("run_root") and Path(str(plan["run_root"])).resolve() != run_root:
            raise RuntimeError("acquisition plan belongs to a different run root")
    if visual_audit:
        visual_audit = visual_audit.resolve()
        audit = validate_visual_audit(load_self_hashed(visual_audit))
    rows, history = candidate_rows(REPO_ROOT, visual_audit=audit, ignore_run_root=run_root, pool=pool)
    coverage_blockers = list(history.get("coverage_blockers", []))
    coverage_blockers.extend(history.get("pool_blockers", []))
    if pool is not None and pool.get("status") != "READY":
        coverage_blockers.append(str(pool.get("status")))
    if pool is not None and pool.get("status") != "READY":
        # A partially inventoried pool is evidence for a blocked handoff, not
        # a way to sneak unbounded local groups back into P.
        rows = []
    execution_contract = _execution_contract(history, rows, visual_audit, run_root=run_root, pool=pool)
    if existing and resume:
        previous_contract = previous.get("execution_contract") or previous_inputs.get("execution_contract")
        if not isinstance(previous_contract, dict):
            raise RuntimeError("cannot resume a run without an execution contract; create a new run")
        if previous_contract != execution_contract:
            raise RuntimeError("resume execution identity mismatch (spec/code/model/input); create a new run")
    cohort = select_cohort(rows, coverage_blockers=coverage_blockers)
    if pool is not None and pool.get("status") in {"BLOCKED_SOURCE_ACCESS", "BLOCKED_ACQUISITION_BUDGET"}:
        cohort["status"] = str(pool.get("status"))
        cohort["blockers"] = sorted(set(cohort.get("blockers", []) + [str(pool.get("blocked_reason") or pool.get("status"))]))
    cohort["readiness"] = "COHORT_READY" if cohort["status"] == "GO" else cohort["status"]
    cohort["acquisition_pool"] = str(pool_path.resolve()) if pool is not None else None
    cohort["execution_contract"] = execution_contract
    run_root.mkdir(parents=True, exist_ok=True)
    shared = run_root / "run" / "shared"
    prior = {}
    if existing and resume:
        for name in ("history_groups.json", "candidate_audit.json", "cohort.json", "inputs.json"):
            path = run_root / name
            if path.is_file():
                try:
                    prior[name] = file_sha256(path)
                except OSError:
                    pass
    write_json(run_root / "history_groups.json", history)
    write_json(shared / "history_groups.json", history)
    candidate_audit = {
        "schema_version": 1,
        "rows": rows,
        "fresh_group_count": len(rows),
        "visual_audit": str(visual_audit.resolve()) if visual_audit else None,
        "visual_audit_sha256": file_sha256(visual_audit) if visual_audit else None,
        "history_coverage_blockers": coverage_blockers,
        "acquisition_pool": str(pool_path.resolve()) if pool is not None else None,
        "acquisition_pool_sha256": file_sha256(pool_path) if pool_path.is_file() else None,
        "execution_contract": execution_contract,
    }
    write_json(run_root / "candidate_audit.json", candidate_audit)
    write_json(shared / "candidate_audit.json", candidate_audit)
    write_json(run_root / "cohort.json", cohort)
    write_json(shared / "cohort.json", cohort)
    inputs = {
        "schema_version": 1,
        "status": cohort["status"] if cohort["status"] != "GO" else "PENDING_CANDIDATE",
        "readiness": cohort.get("readiness"),
        "records": [],
        "formal_count": len(cohort["formal"]),
        "smoke_count": len(cohort["smoke"]),
        "blockers": cohort["blockers"],
        "candidate_audit": "candidate_audit.json",
        "resume_from": prior or None,
        "execution_contract": execution_contract,
    }
    write_json(run_root / "inputs.json", inputs)
    write_json(shared / "inputs.json", inputs)
    validation = {
        "schema_version": 1,
        "status": cohort["status"],
        "history_group_count": len(history["groups"]),
        "fresh_group_count": len(rows),
        "eligible_fresh_group_count": cohort["accepted_count"],
        "required_formal": cohort["required_formal"],
        "required_smoke": cohort["required_smoke"],
        "blockers": cohort["blockers"],
        "visual_screening": (
            "not_run" if visual_audit is None else
            "blocked_or_insufficient" if not audit else
            "pass_available" if cohort["accepted_count"] >= 14 else "run_but_insufficient"
        ),
        "acquisition_pool": str(pool_path.resolve()) if pool is not None else None,
        "resume_from": prior or None,
        "execution_contract": execution_contract,
    }
    write_json(run_root / "validation.json", validation)
    write_json(shared / "validation.json", validation)
    print(f"P status={cohort['status']} historical={len(history['groups'])} fresh={len(rows)} eligible={cohort['accepted_count']}")
    return 0 if cohort["status"] == "GO" else 2


def _ffmpeg(command: list[str]) -> None:
    subprocess.run(command, check=True, timeout=120, capture_output=True)


def _make_cropped_video(source: Path, output: Path, clip: dict) -> dict:
    """Create the fixed 140-frame, face-box-centered reflection crop."""
    import cv2

    geometry = clip.get("visual_geometry") or {}
    box = geometry.get("face_box")
    if not isinstance(box, list) or len(box) != 4:
        raise ProtocolError("visual audit did not provide a face box")
    x, y, width, height = (float(value) for value in box)
    if width <= 0 or height <= 0:
        raise ProtocolError("visual audit face box is empty")
    side = math.ceil(max(width, height) * 1.5)
    center_x = x + width / 2.0
    center_y = y + height / 2.0
    left = round(center_x - side / 2.0)
    top = round(center_y - side / 2.0)
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise ProtocolError(f"cannot open source video: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"FFV1"), 25.0, (512, 512))
    if not writer.isOpened():
        capture.release()
        raise ProtocolError(f"cannot open FFV1 writer: {output}")
    count = 0
    try:
        while count < 140:
            ok, frame = capture.read()
            if not ok:
                raise ProtocolError(f"source has fewer than 140 frames: {source}")
            frame_height, frame_width = frame.shape[:2]
            pad_left, pad_top = max(0, -left), max(0, -top)
            pad_right = max(0, left + side - frame_width)
            pad_bottom = max(0, top + side - frame_height)
            if pad_left or pad_top or pad_right or pad_bottom:
                frame = cv2.copyMakeBorder(frame, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
            crop = frame[top + pad_top : top + pad_top + side, left + pad_left : left + pad_left + side]
            if crop.shape[0] != side or crop.shape[1] != side:
                raise ProtocolError(f"face crop is incomplete: {source}")
            writer.write(cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LANCZOS4))
            count += 1
    finally:
        capture.release()
        writer.release()
    media = inspect_clip(output)
    if media["video_frames"] != 140 or abs(float(media["fps"]) - 25.0) > 0.01 or not media["pts_ok"]:
        raise ProtocolError(f"cropped video contract failed: {output}")
    return {
        "face_box": [x, y, width, height],
        "crop_side_px": side,
        "crop_left": left,
        "crop_top": top,
        "resize_interpolation": "INTER_LANCZOS4",
        "frames_written": count,
        "pts_ok": media["pts_ok"],
    }


def run_candidate(run_root: Path, *, device: str = "cpu") -> int:
    cohort = load_self_hashed(run_root / "cohort.json")
    if (run_root / "run" / "shared" / "freeze.json").exists():
        raise RuntimeError("cannot regenerate candidate inputs after freeze")
    if cohort.get("status") != "GO":
        blocked = {
            "schema_version": 1, "status": cohort.get("status", "BLOCKED_NEW_SOURCE"), "records": [],
            "formal_count": len(cohort.get("formal", [])), "smoke_count": len(cohort.get("smoke", [])),
            "blockers": cohort.get("blockers", []),
            "execution_contract": cohort.get("execution_contract"),
        }
        write_json(run_root / "inputs.json", blocked)
        write_json(run_root / "run" / "shared" / "inputs.json", blocked)
        print(f"P candidate status={blocked['status']}")
        return 2
    shared = run_root / "run" / "shared"
    media_root = shared / "media"
    rows = cohort.get("formal", []) + cohort.get("smoke", [])
    records: list[dict] = []
    try:
        import numpy as np
        import torch

        from scripts.experiments.lrs3_direct_audio.run_wavlm_hifigan_lrs3 import (
            DEFAULT_KNN_VC_SOURCE,
            KNN_VC_REVISION,
            load_model,
            model_checkpoint_metadata,
            resynthesize,
        )
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        model_metadata = model_checkpoint_metadata()
        model = load_model(device, DEFAULT_KNN_VC_SOURCE)
    except Exception as exc:  # noqa: BLE001 - model backends fail with heterogeneous exception types
        failure_payload = {"schema_version": 1, "status": "BLOCKED_DIRECT_CANDIDATE", "failures": [{"error": f"direct_v1 unavailable: {type(exc).__name__}: {exc}"}], "record_count": 0, "expected_count": len(rows)}
        write_json(run_root / "failure_ledger.json", failure_payload)
        write_json(run_root / "run" / "shared" / "failure_ledger.json", failure_payload)
        blocked = {
            "schema_version": 1, "status": "BLOCKED_DIRECT_CANDIDATE", "records": [],
            "formal_count": len(cohort.get("formal", [])), "smoke_count": len(cohort.get("smoke", [])),
            "blockers": [f"direct_v1 unavailable: {type(exc).__name__}: {exc}"],
            "execution_contract": cohort.get("execution_contract"),
        }
        write_json(run_root / "inputs.json", blocked)
        write_json(run_root / "run" / "shared" / "inputs.json", blocked)
        return 2
    failures: list[dict] = []
    audit = load_self_hashed(run_root / "candidate_audit.json")
    audit_rows = {str(row.get("source_group")): row for row in audit.get("rows", [])}
    for row in rows:
        group = str(row["source_group"])
        try:
            audit_row = audit_rows.get(group)
            if not audit_row or audit_row.get("accepted_clip") is None:
                raise ProtocolError(f"selected group is not accepted by candidate audit: {group}")
            clip = row["accepted_clip"]
            if clip.get("path") != audit_row["accepted_clip"].get("path"):
                raise ProtocolError(f"cohort/audit clip mismatch: {group}")
            source = _source_inside_data_root(Path(clip["path"]), group)
            if file_sha256(source) != clip.get("sha256"):
                raise ProtocolError(f"source SHA mismatch: {source}")
        except (OSError, ProtocolError, KeyError, TypeError) as exc:
            failures.append({"source_group": group, "error": f"{type(exc).__name__}: {exc}"})
            continue
        sample_id = f"lrs3_{group}_{clip['clip_id']}"
        audio_path = media_root / "natural_audio" / f"{sample_id}.wav"
        real_path = media_root / "real_video" / f"{sample_id}.mkv"
        ref_path = media_root / "reference" / f"{sample_id}.png"
        candidate_path = media_root / "direct_audio" / f"{sample_id}.wav"
        try:
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            real_path.parent.mkdir(parents=True, exist_ok=True)
            ref_path.parent.mkdir(parents=True, exist_ok=True)
            candidate_path.parent.mkdir(parents=True, exist_ok=True)
            natural_transcode_command = [
                "ffmpeg", "-loglevel", "error", "-y", "-i", str(source),
                "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(audio_path),
            ]
            _ffmpeg(natural_transcode_command)
            geometry = _make_cropped_video(source, real_path, clip)
            _ffmpeg(["ffmpeg", "-loglevel", "error", "-y", "-i", str(real_path), "-frames:v", "1", "-vf", "scale=512:512:flags=lanczos", str(ref_path)])
            import soundfile as sf
            natural, sample_rate = sf.read(audio_path, dtype="float32", always_2d=False)
            if int(sample_rate) != 16000 or natural.ndim != 1 or not (96000 <= natural.size <= 160000):
                raise ProtocolError(f"natural PCM length/rate failed: {sample_id} {sample_rate} {natural.size}")
            natural_pcm_sha256, natural_pcm_sample_count = _decoded_pcm_identity(audio_path)
            result = resynthesize(model, natural, candidate_path)
            adjustment = result["length_adjustment"]["adjustment_sample_count"]
            if adjustment > 640:
                raise ProtocolError(f"direct_v1 tail adjustment exceeds 640 samples: {sample_id}")
            if result.get("audio", {}).get("rms", 0.0) <= 0.0:
                raise ProtocolError(f"direct_v1 candidate has zero RMS: {sample_id}")
            records.append({
                "sample_id": sample_id, "source_group": group, "clip_id": clip["clip_id"],
                "source_video": str(source.resolve()), "source_video_sha256": file_sha256(source),
                "transcript": clip["transcript"], "transcript_sha256": clip["transcript_sha256"],
                "natural_audio": str(audio_path.resolve()), "natural_audio_sha256": file_sha256(audio_path),
                "natural_pcm_sha256": natural_pcm_sha256,
                "natural_pcm_sample_count": natural_pcm_sample_count,
                "real_video": str(real_path.resolve()), "real_video_sha256": file_sha256(real_path),
                "reference_image": str(ref_path.resolve()), "reference_image_sha256": file_sha256(ref_path),
                "source_media_contract": {
                    "format_start_time": clip.get("format_start_time"),
                    "stream_start_times": clip.get("stream_start_times"),
                    "first_video_pts": clip.get("first_video_pts"),
                    "first_audio_pts": clip.get("first_audio_pts"),
                    "pts_ok": clip.get("pts_ok"),
                    "transcode_command": natural_transcode_command,
                },
                "direct_audio": result,
                "model_contract": {
                    "revision": KNN_VC_REVISION,
                    "source": str(DEFAULT_KNN_VC_SOURCE),
                    "checkpoints": model_metadata,
                    "wavlm_layer": 6,
                    "seed": 42,
                },
                # The historical metadata may not expose a speaker ID.  Keep
                # the candidate rather than silently treating unknown as a
                # hard failure; the frozen record makes that uncertainty
                # explicit and downstream reports must not overclaim it.
                "speaker_independence": "verified" if clip.get("speaker_id") else "unverified",
                "geometry_contract": geometry,
            })
        except Exception as exc:  # noqa: BLE001 - keep one failure ledger row per selected group
            failures.append({"source_group": group, "sample_id": sample_id, "error": f"{type(exc).__name__}: {exc}"})
    if failures or len(records) != len(rows):
        failure_payload = {"schema_version": 1, "status": "BLOCKED_DIRECT_CANDIDATE", "failures": failures, "record_count": len(records), "expected_count": len(rows)}
        write_json(run_root / "failure_ledger.json", failure_payload)
        write_json(shared / "failure_ledger.json", failure_payload)
        blocked = {
            "schema_version": 1, "status": "BLOCKED_DIRECT_CANDIDATE", "records": records,
            "formal_count": len(cohort.get("formal", [])), "smoke_count": len(cohort.get("smoke", [])),
            "blockers": ["direct candidate generation did not produce a complete cohort"],
            "execution_contract": cohort.get("execution_contract"),
        }
        write_json(run_root / "inputs.json", blocked)
        write_json(shared / "inputs.json", blocked)
        return 2
    payload = {
        "schema_version": 1, "status": "GO", "formal_count": len(cohort["formal"]), "smoke_count": len(cohort["smoke"]),
        "cohort_sha256": file_sha256(run_root / "cohort.json"),
        "candidate_audit_sha256": file_sha256(run_root / "candidate_audit.json"),
        "execution_contract": cohort.get("execution_contract"), "records": records,
    }
    write_json(run_root / "inputs.json", payload)
    write_json(shared / "inputs.json", payload)
    print(f"P candidate status=GO records={len(records)}")
    return 0


def run_freeze(run_root: Path) -> int:
    if (run_root / "run" / "shared" / "freeze.json").exists():
        raise RuntimeError("freeze manifest already exists; frozen inputs are immutable")
    inputs = load_self_hashed(run_root / "inputs.json")
    cohort = load_self_hashed(run_root / "cohort.json")
    if inputs.get("status") != "GO":
        print(f"P freeze status={inputs.get('status')}")
        return 2
    records = inputs.get("records", [])
    expected = int(inputs.get("formal_count", 0)) + int(inputs.get("smoke_count", 0))
    if expected != 14 or len(records) != expected or len({record.get("sample_id") for record in records}) != expected:
        print(f"P freeze status=BLOCKED_INCOMPLETE_INPUTS records={len(records)} expected={expected}")
        return 2
    _assert_shared_matches(run_root)
    audit = load_self_hashed(run_root / "candidate_audit.json")
    if inputs.get("cohort_sha256") != file_sha256(run_root / "cohort.json") or inputs.get("candidate_audit_sha256") != file_sha256(run_root / "candidate_audit.json"):
        print("P freeze status=BLOCKED_BINDING_MISMATCH")
        return 2
    if inputs.get("execution_contract") != cohort.get("execution_contract"):
        print("P freeze status=BLOCKED_EXECUTION_IDENTITY")
        return 2
    audit_rows = {str(row.get("source_group")): row for row in audit.get("rows", [])}
    expected_records = {}
    for row in cohort.get("formal", []) + cohort.get("smoke", []):
        clip = row.get("accepted_clip") or {}
        expected_records[f"lrs3_{row.get('source_group')}_{clip.get('clip_id')}"] = (str(row.get("source_group")), str(clip.get("clip_id")), str(clip.get("path")))
    for record in records:
        key = str(record.get("sample_id"))
        expected_record = expected_records.get(key)
        audit_row = audit_rows.get(str(record.get("source_group")))
        if expected_record is None or not audit_row or (audit_row.get("accepted_clip") or {}).get("path") != record.get("source_video") or expected_record[:2] != (str(record.get("source_group")), str(record.get("clip_id"))):
            print(f"P freeze status=BLOCKED_COHORT_BINDING sample={key}")
            return 2
        if record.get("speaker_independence") not in {"verified", "unverified"}:
            print(f"P freeze status=BLOCKED_SPEAKER_INDEPENDENCE sample={key}")
            return 2
    frozen = {
        "schema_version": 1, "status": "FROZEN", "inputs_sha256": file_sha256(run_root / "inputs.json"),
        "cohort_sha256": file_sha256(run_root / "cohort.json"), "record_count": len(inputs.get("records", [])),
    }
    write_json(run_root / "run" / "shared" / "freeze.json", frozen)
    print(f"P freeze status=FROZEN records={frozen['record_count']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "candidate", "freeze", "tts"), default="prepare")
    parser.add_argument("--visual-audit", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.stage == "prepare":
        return run_prepare(args.run_root.resolve(), visual_audit=args.visual_audit, resume=args.resume)
    if args.stage == "candidate":
        return run_candidate(args.run_root.resolve())
    if args.stage == "freeze":
        return run_freeze(args.run_root.resolve())
    raise SystemExit("tts is gated by a successful frozen cohort and is not available in this blocked run")


if __name__ == "__main__":
    raise SystemExit(main())
