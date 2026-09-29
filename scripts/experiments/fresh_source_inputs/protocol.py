from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_GROUP_RE = re.compile(r"^lrs3_([A-Za-z0-9]{11})_\d{5}(?:[^/]*)$")
SAMPLE_GROUP_PREFIX_RE = re.compile(r"lrs3_([A-Za-z0-9]{11})_(?=\d{5}(?:\D|$))")
GROUP_ID_RE = re.compile(r"^[A-Za-z0-9]{11}$")
FIELD_GROUP_RE = re.compile(
    r"[\"'](source_group|source_video_id)[\"']\s*:\s*[\"']([A-Za-z0-9]{11})[\"']"
)
TOKEN_RE = re.compile(
    r"lrs3_[A-Za-z0-9]{11}_(?=\d{5}(?:\D|$))|[\"']source_group[\"']\s*:\s*[\"'][A-Za-z0-9]{11}[\"']|[\"']source_video_id[\"']\s*:\s*[\"'][A-Za-z0-9]{11}[\"']"
)
VISUAL_METHOD = "mediapipe_face_mesh_v1"


class ProtocolError(RuntimeError):
    """A frozen-input contract violation."""


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    body = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    body = dict(payload)
    body.pop("artifact_sha256", None)
    body["artifact_sha256"] = canonical_hash(body)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(body, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return body


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"invalid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"expected JSON object: {path}")
    return value


def load_self_hashed(path: Path) -> dict[str, Any]:
    value = read_json(path)
    actual = value.get("artifact_sha256")
    body = dict(value)
    body.pop("artifact_sha256", None)
    if not isinstance(actual, str) or actual != canonical_hash(body):
        raise ProtocolError(f"self-hash mismatch: {path}")
    return value


def source_group_from_sample(sample_id: str) -> str | None:
    match = SAMPLE_GROUP_RE.fullmatch(str(sample_id))
    return match.group(1) if match else None


def _is_under(path: Path, root: Path) -> bool:
    try:
        # Scan paths are already rooted under repo_root; avoid resolving every
        # metadata path through the filesystem (which is prohibitively slow on
        # large run trees). Security-sensitive candidate paths use resolve()
        # separately in runner/validator.
        path.absolute().relative_to(root.absolute())
        return True
    except ValueError:
        return False


def _extract_tokens(text: str, path: Path) -> list[tuple[str, str]]:
    """Extract provenance tokens from JSON-like text and unquoted CSV fields."""
    tokens: list[tuple[str, str]] = []
    for match in TOKEN_RE.finditer(text):
        token = match.group(0)
        sample = SAMPLE_GROUP_PREFIX_RE.search(token)
        if sample:
            tokens.append(("sample_id", sample.group(1)))
            continue
        field_match = FIELD_GROUP_RE.search(token)
        if field_match:
            tokens.append(field_match.groups())
    if path.suffix.lower() == ".csv":
        try:
            reader = csv.DictReader(io.StringIO(text))
            fields = {field for field in (reader.fieldnames or []) if field}
            for row in reader:
                for field in ("source_group", "source_video_id", "sample_id", "video_id"):
                    if field not in fields:
                        continue
                    value = str(row.get(field) or "").strip()
                    sample = SAMPLE_GROUP_RE.fullmatch(value)
                    if sample:
                        tokens.append((field, sample.group(1)))
                    elif GROUP_ID_RE.fullmatch(value) and field in {"source_group", "source_video_id"}:
                        tokens.append((field, value))
        except (csv.Error, UnicodeError):
            pass
    return tokens


def _rg_history(repo_root: Path, *, ignore_run_root: Path | None = None) -> tuple[set[str], dict[str, dict[str, Any]], list[str]]:
    """Extract group provenance from all JSON/JSONL/CSV metadata under runs.

    This deliberately scans text rather than interpreting producer schemas.  Old
    runs use several incompatible schemas; the stable evidence is the group ID
    itself and the file that contained it.  ``rg`` makes the full scan bounded and
    reproducible while retaining a fallback for environments without ripgrep.
    """
    root = repo_root / "runs"
    if not root.is_dir():
        return set(), {}, []
    files: list[Path]
    try:
        listed = subprocess.run(
            ["rg", "-l", "--glob", "*.json", "--glob", "*.jsonl", "--glob", "*.csv", "source_group|source_video_id|lrs3_[A-Za-z0-9_-]{11}_", str(root)],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        files = [Path(item) for item in listed if Path(item).is_file() and not (ignore_run_root and _is_under(Path(item), ignore_run_root))]
    except (FileNotFoundError, subprocess.CalledProcessError):
        files = [path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".json", ".jsonl", ".csv"} and not (ignore_run_root and _is_under(path, ignore_run_root))]

    groups: set[str] = set()
    provenance: dict[str, dict[str, Any]] = {}
    file_rows: list[str] = []
    for path in sorted(set(files)):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        matches = _extract_tokens(text, path)
        if not matches:
            continue
        relative = path.resolve().relative_to(repo_root.resolve()).as_posix()
        file_rows.append(f"{relative}\t{file_sha256(path)}")
        for field, group in matches:
            if not GROUP_ID_RE.fullmatch(group):
                continue
            groups.add(group)
            row = provenance.setdefault(group, {"fields": set(), "files": []})
            row["fields"].add(field)
            if relative not in row["files"]:
                row["files"].append(relative)
            row["file_count"] = int(row.get("file_count", 0)) + 1
    for row in provenance.values():
        row["fields"] = sorted(row["fields"])
        row["files"] = sorted(row["files"])
    return groups, provenance, file_rows


def scan_history(repo_root: Path = REPO_ROOT, *, ignore_run_root: Path | None = None) -> dict[str, Any]:
    groups, provenance, file_rows = _rg_history(repo_root, ignore_run_root=ignore_run_root)
    archive_rows: list[dict[str, Any]] = []
    archive_groups: set[str] = set()
    for archive in sorted((repo_root / "data/lrs3/ellipsis-lrs3-raw/ainncy").glob("pretrain.*.tar")):
        try:
            listing = subprocess.run(["tar", "-tf", str(archive)], check=True, capture_output=True, text=True, timeout=120).stdout.splitlines()
            members = sorted({parts[1] for line in listing if len(parts := line.split("/")) >= 3 and parts[2].endswith(".mp4") and GROUP_ID_RE.fullmatch(parts[1])})
            archive_groups.update(members)
            member_hash = hashlib.sha256(("\n".join(members) + "\n").encode("utf-8")).hexdigest()
            archive_rows.append({"path": str(archive.resolve()), "size": archive.stat().st_size, "mtime_ns": archive.stat().st_mtime_ns, "member_group_count": len(members), "member_groups_sha256": member_hash})
        except (OSError, subprocess.SubprocessError) as exc:
            archive_rows.append({"path": str(archive.resolve()), "error": f"{type(exc).__name__}: {exc}"})
    manifest_text = "".join(f"{row}\n" for row in sorted(file_rows)).encode("utf-8")
    lock_command = ["find", str(repo_root / "runs"), "-type", "f", "-name", "test_lock.json"]
    if ignore_run_root:
        lock_command.extend(["-not", "-path", f"{ignore_run_root.as_posix()}/*"])
    lock_command.extend(["-print", "-quit"])
    lock_result = subprocess.run(lock_command, capture_output=True, text=True, check=False)
    multiset_result = subprocess.run(
        ["rg", "-l", "--glob", "*.json", "--glob", "*.jsonl", "--glob", "*.csv", "multiset", str(repo_root / "runs")],
        capture_output=True, text=True, check=False,
    )
    multiset_paths = [Path(item) for item in multiset_result.stdout.splitlines() if item]
    if ignore_run_root:
        multiset_paths = [path for path in multiset_paths if not _is_under(path, ignore_run_root)]
    required_markers = {
        "n500_manifest": (repo_root / "runs/lrs3_qwen_cloud_n500_20260817/00_manifest/manifest.json").is_file(),
        "source_pool": (repo_root / "runs/lrs3_data_supplement_20260903/09_policy_cohort_retry2/source_pool.json").is_file(),
        "supplement_manifest": (repo_root / "runs/lrs3_data_supplement_20260903/00_manifest.json").is_file(),
        "multiset250_metadata": bool(multiset_paths),
        "test_locks": bool(lock_result.stdout.strip()),
        "latest_parent_runs": any("20260910" in path for path in file_rows),
    }
    return {
        "schema_version": 1,
        "scan_root": str((repo_root / "runs").resolve()),
        "scan_file_count": len(file_rows),
        "scan_file_manifest_sha256": hashlib.sha256(manifest_text).hexdigest(),
        "groups": sorted(groups),
        "provenance": provenance,
        "archive_inputs": archive_rows,
        "archive_group_count": len(archive_groups),
        "archive_groups": sorted(archive_groups),
        "coverage": required_markers,
        "coverage_blockers": [name for name, present in required_markers.items() if not present],
    }


def _ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-count_frames", "-print_format", "json", "-show_format", "-show_streams", str(path)
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=30)
    value = json.loads(result.stdout)
    if not isinstance(value, dict):
        raise ProtocolError(f"ffprobe did not return an object: {path}")
    return value


def _duration(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def inspect_clip(path: Path) -> dict[str, Any]:
    probe = _ffprobe(path)
    streams = probe.get("streams", [])
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    duration = _duration((probe.get("format") or {}).get("duration"))
    fps_text = str((video or {}).get("r_frame_rate", "0/1"))
    try:
        numerator, denominator = fps_text.split("/", 1)
        fps = float(numerator) / float(denominator)
    except (ValueError, ZeroDivisionError):
        fps = float("nan")
    format_data = probe.get("format") or {}
    start_time = _duration(format_data.get("start_time"))
    stream_starts = {
        str(stream.get("codec_type")): _duration(stream.get("start_time"))
        for stream in streams
        if stream.get("codec_type") in {"audio", "video"}
    }
    first_video_pts = _first_stream_pts(path, "v:0")
    first_audio_pts = _first_stream_pts(path, "a:0")
    decoded_pcm_samples = _decoded_pcm_samples(path) if audio is not None else 0
    finite_starts = bool(
        np.isfinite(start_time)
        and all(np.isfinite(value) for value in stream_starts.values())
        and np.isfinite(first_video_pts)
    )
    # Source clips are required to carry audio, so their contract compares
    # decoded audio/video PTS directly.  The normalized R asset is deliberately
    # video-only; for that asset the zero-PTS contract is the first video frame
    # itself.  Treating a missing audio stream as an automatic failure would
    # make every valid R asset impossible to freeze.
    if audio is None:
        pts_ok = bool(finite_starts and abs(first_video_pts) <= 0.020)
    else:
        pts_ok = bool(
            finite_starts
            and np.isfinite(first_audio_pts)
            and abs(first_audio_pts - first_video_pts) <= 0.020
        )
    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "duration_s": duration,
        "fps": fps,
        "width": int((video or {}).get("width") or 0),
        "height": int((video or {}).get("height") or 0),
        "video_frames": int((video or {}).get("nb_frames") or (video or {}).get("nb_read_frames") or 0),
        "has_video": video is not None,
        "has_audio": audio is not None,
        "audio_sample_rate": int((audio or {}).get("sample_rate") or 0),
        "audio_channels": int((audio or {}).get("channels") or 0),
        "format_start_time": start_time,
        "stream_start_times": stream_starts,
        "first_video_pts": first_video_pts,
        "first_audio_pts": first_audio_pts,
        "decoded_pcm_samples": decoded_pcm_samples,
        "pts_ok": pts_ok,
    }


def _first_stream_pts(path: Path, selector: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", selector, "-show_entries", "frame=best_effort_timestamp_time", "-of", "csv=p=0", str(path)],
        check=True, capture_output=True, text=True, timeout=30,
    )
    for line in result.stdout.splitlines():
        try:
            value = float(line.strip().split(",", 1)[0])
            if np.isfinite(value):
                return value
        except ValueError:
            continue
    return float("nan")


def _decoded_pcm_samples(path: Path) -> int:
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:a:0", "-ac", "1", "-ar", "16000", "-f", "s16le", "pipe:1"],
        check=True, capture_output=True, timeout=60,
    )
    if len(result.stdout) % 2:
        raise ProtocolError(f"decoded PCM byte count is odd: {path}")
    return len(result.stdout) // 2


def transcript_for(path: Path) -> str:
    text_path = path.with_suffix(".txt")
    if not text_path.is_file():
        return ""
    lines = text_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    text = next((line for line in lines if line.strip().lower().startswith("text:")), "")
    text = re.sub(r"^\s*Text:\s*", "", text, flags=re.IGNORECASE).strip()
    return " ".join(text.split())


def _validate_visual_array(item: dict[str, Any], *, group: str, model_sha: str) -> dict[str, Any] | None:
    """Verify the optional raw landmark binding emitted by the visual worker."""

    array_path_value = item.get("landmark_array_path")
    array_sha = item.get("landmark_array_sha256")
    if array_path_value is None and array_sha is None:
        return None  # legacy producer artifacts remain readable; new audits bind it
    if not isinstance(array_path_value, str) or not Path(array_path_value).is_file():
        raise ProtocolError(f"visual landmark array is missing for {group}")
    if not isinstance(array_sha, str) or file_sha256(array_path_value) != array_sha:
        raise ProtocolError(f"visual landmark array SHA mismatch for {group}")
    try:
        with np.load(array_path_value, allow_pickle=False) as arrays:
            points = np.asarray(arrays["landmarks"])
            valid = np.asarray(arrays["valid"])
            timestamps = np.asarray(arrays["timestamps_s"], dtype=np.float64)
            metadata = json.loads(str(arrays["metadata_json"].item()))
    except (OSError, KeyError, ValueError, json.JSONDecodeError) as exc:
        raise ProtocolError(f"visual landmark array is malformed for {group}") from exc
    if points.ndim != 3 or points.shape[0] != 140 or points.shape[2] != 2 or valid.shape != (140,) or valid.dtype != np.bool_ or timestamps.shape != (140,) or not np.isfinite(timestamps).all() or (timestamps.size > 1 and not np.all(np.diff(timestamps) > 0)):
        raise ProtocolError(f"visual landmark array shape is invalid for {group}")
    if not np.isfinite(points).all() or not isinstance(metadata, dict):
        raise ProtocolError(f"visual landmark array contains nonfinite data for {group}")
    if metadata.get("source_group") != group or metadata.get("clip_sha256") != item.get("clip_sha256") or metadata.get("model_sha256") != model_sha:
        raise ProtocolError(f"visual landmark array provenance mismatch for {group}")
    valid_sha = item.get("valid_array_sha256")
    if valid_sha and valid_sha != hashlib.sha256(valid.astype(np.bool_).tobytes()).hexdigest():
        raise ProtocolError(f"visual valid-array SHA mismatch for {group}")
    nested = metadata.get("metadata") if isinstance(metadata.get("metadata"), dict) else metadata
    width = float(nested.get("width", item.get("width", 0)) or 0)
    height = float(nested.get("height", item.get("height", 0)) or 0)
    if width <= 0 or height <= 0:
        raise ProtocolError(f"visual landmark array dimensions are missing for {group}")
    first = points[0]
    finite = np.isfinite(first).all(axis=1)
    first_face = bool(valid[0] and finite[33] and finite[263])
    eye = float(np.linalg.norm((first[263] - first[33]) * np.asarray([width, height]))) if first_face else 0.0
    box = None
    if first_face and finite.any():
        cloud = first[finite] * np.asarray([width, height])
        low, high = cloud.min(axis=0), cloud.max(axis=0)
        box = [float(low[0]), float(low[1]), float(high[0] - low[0]), float(high[1] - low[1])]
    return {"valid_fraction": float(valid.sum() / 140.0), "frames_covered": int(valid.sum()), "first_frame_face": first_face, "eye_distance_px": eye, "face_box": box}


def validate_visual_audit(value: Any) -> dict[str, dict[str, Any]]:
    """Validate visual evidence, retaining ordinary quality FAIL records.

    A blocked audit may have no model file and no groups; this is a valid
    engineering handoff, not a visual PASS.  For an ``unblock_v1`` audit,
    PASS records additionally bind raw arrays and an explicit input review.
    """

    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ProtocolError("visual audit must be schema_version=1 object")
    if value.get("method") != VISUAL_METHOD:
        raise ProtocolError(f"visual audit method must be {VISUAL_METHOD}")
    audit_status = value.get("status", "READY")
    if audit_status not in {"READY", "BLOCKED_VISUAL_SCREENING", "BLOCKED_VISUAL_ASSET", "BLOCKED_SOURCE_POOL"}:
        raise ProtocolError(f"invalid visual audit status: {audit_status}")
    model_sha = value.get("model_sha256")
    model_path = value.get("model_path")
    if audit_status in {"BLOCKED_VISUAL_ASSET", "BLOCKED_SOURCE_POOL"} and not model_sha:
        return {}
    if not isinstance(model_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", model_sha) or set(model_sha) == {"0"}:
        raise ProtocolError("visual audit model_sha256 is required")
    if not isinstance(model_path, str) or not Path(model_path).is_file() or file_sha256(model_path) != model_sha:
        raise ProtocolError("visual audit model_path must exist and match model_sha256")
    thresholds = value.get("thresholds")
    try:
        threshold_valid = float((thresholds or {}).get("min_valid_fraction", 0.0))
        threshold_eye = float((thresholds or {}).get("min_eye_distance_px", 0.0))
        threshold_frames = int((thresholds or {}).get("min_frames", 0))
    except (AttributeError, TypeError, ValueError) as exc:
        raise ProtocolError("visual audit thresholds are malformed") from exc
    if not isinstance(thresholds, dict) or threshold_valid < 0.95 or threshold_eye < 40.0 or threshold_frames < 140:
        raise ProtocolError("visual audit thresholds do not meet the frozen gate")
    groups = value.get("groups")
    if not isinstance(groups, dict):
        raise ProtocolError("visual audit groups must be an object")
    validated: dict[str, dict[str, Any]] = {}
    strict_review = value.get("audit_mode") == "unblock_v1"
    for group, item in groups.items():
        if not GROUP_ID_RE.fullmatch(str(group)) or not isinstance(item, dict):
            raise ProtocolError(f"invalid visual audit group: {group!r}")
        status = item.get("status")
        if status not in {"PASS", "FAIL"}:
            raise ProtocolError(f"invalid visual status for {group}")
        clip_sha = item.get("clip_sha256")
        if not isinstance(clip_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", clip_sha):
            raise ProtocolError(f"visual audit clip_sha256 is required for {group}")
        clip_path = item.get("clip_path")
        if clip_path is not None:
            if not isinstance(clip_path, str) or not Path(clip_path).is_file() or file_sha256(clip_path) != clip_sha:
                raise ProtocolError(f"visual audit clip binding is invalid for {group}")
        elif value.get("audit_mode") == "unblock_v1":
            raise ProtocolError(f"visual audit clip_path is required for {group}")
        try:
            valid_fraction = float(item.get("valid_fraction", 0.0))
            eye_distance = float(item.get("eye_distance_px", 0.0))
            frames_covered = int(item.get("frames_covered", 0))
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"visual audit numeric fields missing for {group}") from exc
        if not (np.isfinite(valid_fraction) and np.isfinite(eye_distance) and 0.0 <= valid_fraction <= 1.0 and eye_distance >= 0.0 and frames_covered >= 0):
            raise ProtocolError(f"visual audit geometry fields invalid for {group}")
        face_box = item.get("face_box")
        if face_box is not None:
            try:
                box_values = [float(v) for v in face_box] if isinstance(face_box, list) else []
            except (TypeError, ValueError) as exc:
                raise ProtocolError(f"visual audit face_box is malformed for {group}") from exc
            if len(box_values) != 4 or any(not np.isfinite(v) or v < 0 for v in box_values) or box_values[2] <= 0 or box_values[3] <= 0:
                raise ProtocolError(f"visual audit face_box is required for {group}")
        if status == "PASS":
            if valid_fraction < threshold_valid or eye_distance < threshold_eye or frames_covered < threshold_frames:
                raise ProtocolError(f"visual audit geometry gate failed for {group}")
            if item.get("first_frame_face") is not True or item.get("mouth_visible") is not True:
                raise ProtocolError(f"visual audit first-frame face/mouth gate failed for {group}")
            if face_box is None:
                raise ProtocolError(f"visual audit face_box is required for {group}")
            if strict_review:
                if not item.get("landmark_array_path") or not item.get("landmark_array_sha256"):
                    raise ProtocolError(f"visual landmark array binding is missing for {group}")
                review_path = item.get("input_review_path")
                review_sha = item.get("input_review_sha256")
                if not isinstance(review_path, str) or not Path(review_path).is_file() or not isinstance(review_sha, str) or file_sha256(review_path) != review_sha:
                    raise ProtocolError(f"visual input review binding is missing for {group}")
                try:
                    review = load_self_hashed(Path(review_path))
                except ProtocolError as exc:
                    raise ProtocolError(f"visual input review is invalid for {group}") from exc
                if review.get("clip_sha256") != clip_sha or review.get("status") != "PASS" or not review.get("reviewer") or review.get("mouth_visible") is not True:
                    raise ProtocolError(f"visual input review does not support PASS for {group}")
                evidence_path = review.get("evidence_path")
                evidence_sha = review.get("evidence_sha256")
                if not isinstance(evidence_path, str) or not Path(evidence_path).is_file() or not isinstance(evidence_sha, str) or file_sha256(evidence_path) != evidence_sha:
                    raise ProtocolError(f"visual input review evidence is missing for {group}")
        elif not item.get("reason"):
            raise ProtocolError(f"visual FAIL must preserve a reason for {group}")
        if status == "PASS":
            measured = _validate_visual_array(item, group=str(group), model_sha=model_sha)
            if measured is not None:
                for key in ("valid_fraction", "eye_distance_px"):
                    try:
                        if abs(float(item.get(key)) - float(measured[key])) > 1e-6:
                            raise ProtocolError(f"visual audit {key} does not match raw array for {group}")
                    except (TypeError, ValueError) as exc:
                        raise ProtocolError(f"visual audit {key} is malformed for {group}") from exc
                if int(item.get("frames_covered", -1)) != measured["frames_covered"] or item.get("first_frame_face") is not measured["first_frame_face"]:
                    raise ProtocolError(f"visual audit frame measurements do not match raw array for {group}")
                expected_box = measured.get("face_box")
                actual_box = item.get("face_box")
                if expected_box is None or not isinstance(actual_box, list) or any(abs(float(left) - float(right)) > 1e-6 for left, right in zip(expected_box, actual_box)):
                    raise ProtocolError(f"visual audit face box does not match raw array for {group}")
        elif strict_review and item.get("landmark_array_path"):
            measured = _validate_visual_array(item, group=str(group), model_sha=model_sha)
            if measured is not None and int(item.get("frames_covered", -1)) != measured["frames_covered"]:
                raise ProtocolError(f"visual FAIL frame measurements do not match raw array for {group}")
        validated[str(group)] = dict(item, model_sha256=model_sha, model_path=model_path, audit_verified=status == "PASS")
    return validated


def _pool_clip_rows(pool: dict[str, Any], data_root: Path) -> tuple[list[tuple[str, list[dict[str, Any]]]], list[str]]:
    """Normalize and sanity-check the current-run bounded pool."""

    errors: list[str] = []
    groups = pool.get("groups")
    if not isinstance(groups, list):
        raise ProtocolError("source pool groups must be a list")
    if len(groups) > 24 or int(pool.get("max_groups", 24)) > 24:
        raise ProtocolError("source pool exceeds the 24-group acquisition cap")
    normalized: list[tuple[str, list[dict[str, Any]]]] = []
    for expected_order, entry in enumerate(groups):
        if not isinstance(entry, dict):
            errors.append("malformed source pool group entry")
            continue
        group = str(entry.get("source_group", ""))
        if not GROUP_ID_RE.fullmatch(group):
            errors.append(f"invalid source pool group: {group}")
            continue
        if entry.get("group_order", expected_order) != expected_order:
            errors.append(f"source pool group order field is not stable: {group}")
        clips = entry.get("clips")
        if not isinstance(clips, list):
            errors.append(f"source pool clips missing: {group}")
            continue
        clips = sorted((dict(clip) for clip in clips if isinstance(clip, dict)), key=lambda item: str(item.get("clip_id", "")))[:8]
        local_dir = data_root / group
        local_ids = sorted(path.stem for path in local_dir.glob("*.mp4"))[:8] if local_dir.is_dir() else []
        pool_ids = [str(clip.get("clip_id", "")) for clip in clips]
        if local_ids and pool_ids != local_ids:
            errors.append(f"source pool is not the original first-eight clip order: {group}")
        for clip in clips:
            clip_path = Path(str(clip.get("mp4_path", "")))
            if not _is_under(clip_path, data_root):
                errors.append(f"source pool clip outside data root: {clip_path}")
            txt_path = clip.get("txt_path")
            if txt_path and not _is_under(Path(str(txt_path)), data_root):
                errors.append(f"source pool transcript outside data root: {txt_path}")
        normalized.append((group, clips))
    group_order = [group for group, _ in normalized]
    if group_order != sorted(group_order) or len(group_order) != len(set(group_order)):
        errors.append("source pool group order is not stable and unique")
    return normalized, errors


def candidate_rows(
    repo_root: Path = REPO_ROOT,
    *,
    visual_audit: dict[str, dict[str, Any]] | None = None,
    ignore_run_root: Path | None = None,
    pool: dict[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    data_root = repo_root / "data/dataset_samples/lrs3/pretrain"
    history = scan_history(repo_root, ignore_run_root=ignore_run_root)
    historical = set(history["groups"])
    pool_errors: list[str] = []
    if pool is None:
        local_groups = sorted(path.name for path in data_root.iterdir() if path.is_dir() and GROUP_ID_RE.fullmatch(path.name))
        group_clips = [(group, [{"clip_id": path.stem, "mp4_path": str(path.resolve()), "txt_path": str(path.with_suffix(".txt").resolve())} for path in sorted((data_root / group).glob("*.mp4"))[:8]]) for group in local_groups]
    else:
        group_clips, pool_errors = _pool_clip_rows(pool, data_root)
    fresh_groups = [(group, clips) for group, clips in group_clips if group not in historical]
    rows: list[dict[str, Any]] = []
    for group, pool_clips in fresh_groups:
        clips: list[dict[str, Any]] = []
        paths = [Path(str(clip.get("mp4_path"))) for clip in pool_clips]
        for path in paths:
            try:
                media = inspect_clip(path)
                transcript = transcript_for(path)
                media["clip_id"] = path.stem
                media["transcript"] = transcript
                media["transcript_sha256"] = hashlib.sha256(transcript.encode("utf-8")).hexdigest()
                media["duration_ok"] = bool(np.isfinite(media["duration_s"]) and 6.0 <= media["duration_s"] <= 10.0)
                media["pts_ok"] = bool(media.get("pts_ok"))
                media["duration_pcm_ok"] = bool(96000 <= int(media["decoded_pcm_samples"]) <= 160000)
                visual = (visual_audit or {}).get(group, {})
                media["visual_status"] = visual.get("status", "UNVERIFIED")
                if visual:
                    media["visual_geometry"] = {
                        "face_box": visual.get("face_box"),
                        "first_frame_face": visual.get("first_frame_face"),
                        "mouth_visible": visual.get("mouth_visible"),
                        "frames_covered": visual.get("frames_covered"),
                        "eye_distance_px": visual.get("eye_distance_px"),
                        "valid_fraction": visual.get("valid_fraction"),
                        "resize_interpolation": visual.get("resize_interpolation", "lanczos"),
                    }
                media["visual_audit_verified"] = bool(
                    visual.get("audit_verified")
                    and visual.get("clip_sha256") == media["sha256"]
                    and float(visual.get("valid_fraction", 0.0)) >= 0.95
                    and float(visual.get("eye_distance_px", 0.0)) >= 40.0
                    and int(visual.get("frames_covered", 0)) >= 140
                    and visual.get("first_frame_face") is True
                    and visual.get("mouth_visible") is True
                )
                media["input_ok"] = bool(
                    media["has_video"] and media["has_audio"] and media["duration_ok"]
                    and media["duration_pcm_ok"] and media["pts_ok"] and transcript
                    and media["visual_status"] == "PASS" and media["visual_audit_verified"]
                )
                media["pool_clip"] = next((dict(item) for item in pool_clips if str(item.get("clip_id")) == path.stem), None)
                clips.append(media)
            except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
                clips.append({"path": str(path.resolve()), "clip_id": path.stem, "input_ok": False, "error": f"{type(exc).__name__}: {exc}"})
        accepted = next((clip for clip in clips if clip.get("input_ok") is True), None)
        duration_candidates = [clip for clip in clips if clip.get("has_video") and clip.get("has_audio") and clip.get("duration_ok") and clip.get("duration_pcm_ok") and clip.get("pts_ok") and clip.get("transcript")]
        rows.append({
            "source_group": group,
            "historical": False,
            "clips_examined": clips,
            "accepted_clip": accepted,
            "duration_candidates": duration_candidates,
            "quality_status": "candidate" if accepted else ("VISUAL_UNVERIFIED" if duration_candidates else "NO_6_TO_10S_COMPLETE_CLIP"),
            "pool_clip_ids": [str(item.get("clip_id")) for item in pool_clips],
        })
    if pool_errors:
        history = dict(history)
        history["pool_blockers"] = sorted(pool_errors)
    return rows, history


def select_cohort(
    rows: Iterable[dict[str, Any]],
    *,
    formal: int = 12,
    smoke: int = 2,
    coverage_blockers: Iterable[str] = (),
) -> dict[str, Any]:
    accepted = [row for row in rows if row.get("accepted_clip")]
    accepted.sort(key=lambda row: str(row["source_group"]))
    coverage_blockers = sorted({str(item) for item in coverage_blockers})
    blockers = list(coverage_blockers)
    if len(accepted) < formal + smoke:
        blockers.extend([
            f"need {formal + smoke} fully screened fresh groups, found {len(accepted)}",
            "MediaPipe visual screening is not PASS for the duration-only candidates" if accepted else "no fully screened candidate is available",
        ])
    status = "GO" if len(accepted) >= formal + smoke and not coverage_blockers else ("BLOCKED_HISTORY_COVERAGE" if coverage_blockers else "BLOCKED_NEW_SOURCE")
    selected_formal = accepted[:formal] if status == "GO" else []
    selected_smoke = accepted[formal : formal + smoke] if status == "GO" else []
    return {
        "schema_version": 1,
        "formal": selected_formal,
        "smoke": selected_smoke,
        "accepted_count": len(accepted),
        "required_formal": formal,
        "required_smoke": smoke,
        "status": status,
        "blockers": blockers,
    }


def delay_pcm(natural: np.ndarray, *, samples: int = 3200) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(natural, dtype=np.float32).reshape(-1)
    if samples < 0 or samples >= len(values):
        raise ValueError("delay must be non-negative and shorter than the waveform")
    if samples == 0:
        return values.copy(), np.arange(len(values), dtype=np.int64)
    result = np.zeros_like(values)
    result[samples:] = values[:-samples]
    source_index = np.full(len(values), -1, dtype=np.int64)
    source_index[samples:] = np.arange(len(values) - samples, dtype=np.int64)
    return result, source_index
