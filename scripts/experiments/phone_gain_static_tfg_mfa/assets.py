"""Read-only import of P2 assets and strict no-video manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
import wave
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from scripts.experiments.lrs3_phone_rules_metrics import normalize_phone

from .config import REPO_ROOT, canonical_hash, file_sha256, read_json, write_json, write_jsonl

FORBIDDEN_FIELDS = frozenset({"video_path", "video_sha256", "landmarks", "frame_sequence", "face_track", "target_video", "video_features"})
IMAGE_IDS = ("3", "6", "9")


def read_pcm16(path: str | Path, *, sample_rate: int = 16000) -> tuple[np.ndarray, dict[str, Any]]:
    target = Path(path)
    with wave.open(str(target), "rb") as handle:
        rate = int(handle.getframerate())
        channels = int(handle.getnchannels())
        width = int(handle.getsampwidth())
        count = int(handle.getnframes())
        raw = handle.readframes(count)
    if rate != int(sample_rate) or channels != 1 or width != 2:
        raise ValueError(f"expected mono {sample_rate} Hz PCM16: {target}")
    pcm = np.frombuffer(raw, dtype="<i2").copy()
    if pcm.size != count or pcm.ndim != 1:
        raise ValueError(f"invalid PCM payload: {target}")
    return pcm, {"sample_rate": rate, "sample_count": int(pcm.size), "pcm_sha256": hashlib.sha256(pcm.tobytes()).hexdigest(), "container_sha256": file_sha256(target), "path": str(target.resolve())}


def write_pcm16(path: str | Path, pcm: Sequence[int] | np.ndarray, *, sample_rate: int = 16000) -> dict[str, Any]:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(pcm, dtype=np.int16).reshape(-1)
    with wave.open(str(target), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(values.astype("<i2", copy=False).tobytes())
    return {"path": str(target.resolve()), "sample_rate": int(sample_rate), "sample_count": int(values.size), "pcm_sha256": hashlib.sha256(values.tobytes()).hexdigest(), "container_sha256": file_sha256(target)}


def _token_speech(token: Mapping[str, Any]) -> bool:
    return bool(token.get("speech", not token.get("silence", False)))


def normalize_tokens(tokens: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    previous = 0.0
    for index, raw in enumerate(tokens):
        start = float(raw["start_s"])
        end = float(raw["end_s"])
        if not np.isfinite([start, end]).all() or start < 0 or end <= start or start < previous - 1e-6:
            raise ValueError(f"invalid/non-monotonic token at index {index}")
        label = normalize_phone(raw.get("label", raw.get("token", "")))
        result.append({**dict(raw), "token_id": str(raw.get("token_id", f":{index}")), "token_index": index, "label": label, "speech": _token_speech(raw), "start_s": start, "end_s": end, "duration_s": end - start})
        previous = end
    return result


def _check_unknown(payload: Mapping[str, Any], allowed: set[str], name: str) -> None:
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise ValueError(f"{name} contains forbidden/unknown fields: {unknown}")


def _portrait_meta(path: Path, detection: Mapping[str, Any] | None = None, *, allow_full_frame: bool = False) -> dict[str, Any]:
    png_magic = b"\x89PNG\r\n\x1a\n"
    if not path.is_file() or path.read_bytes()[:8] != png_magic:
        raise ValueError(f"portrait is not a PNG: {path}")
    decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if decoded is None or decoded.ndim != 3 or decoded.shape[2] != 3:
        raise ValueError(f"portrait is not a decodable single image: {path}")
    rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    if not isinstance(detection, Mapping):
        raise ValueError(f"static portrait detection is required; refusing full-frame fallback: {path}")
    from scripts.experiments.static_image_bridge.images import generation_box, score_box, select_detection

    candidates = detection.get("detections")
    if not isinstance(candidates, list):
        raise ValueError(f"portrait detection has no candidate list: {path}")
    selected = select_detection(candidates)
    generation = generation_box(selected, int(decoded.shape[1]), int(decoded.shape[0]))
    scoring = score_box(selected)
    if generation == [0, 0, int(decoded.shape[1]), int(decoded.shape[0])] and not allow_full_frame:
        raise ValueError(f"detector returned a full-frame generation box: {path}")
    return {
        "portrait_id": path.stem,
        "path": str(path.resolve()),
        "container_sha256": file_sha256(path),
        "rgb_pixel_sha256": hashlib.sha256(np.ascontiguousarray(rgb).tobytes()).hexdigest(),
        "width": int(decoded.shape[1]),
        "height": int(decoded.shape[0]),
        "generation_box_xyxy": generation,
        "score_box": scoring,
        "source_frame_indices": [0],
        "detector": {
            "threshold": float(detection.get("threshold", 0.9)),
            "device": detection.get("device"),
            "selected": selected,
            "detections": candidates,
        },
        "geometry_contract": "sfd_selected_face_v1",
    }


def _detect_portraits(config: Mapping[str, Any], root: Path, portrait_paths: Mapping[str, Path]) -> dict[str, Any]:
    """Run the static PNG detector once and bind its output to pixel hashes.

    This adapter deliberately calls ``detect_worker.py`` instead of the bridge
    preparation path: the latter can decode an LRS3 video frame.  A cached
    detection is accepted only when every requested PNG hash and detector
    implementation hash still match.
    """
    detector = REPO_ROOT / "scripts/experiments/static_image_bridge/detect_worker.py"
    python = Path(str(config.get("tfg", {}).get("wav2lip_python", "/home/wjj/.venvs/wav2lip/bin/python")))
    output = root / "00_protocol/portrait_detections.json"
    request = root / "00_protocol/portrait_detection_requests.json"
    expected_images = {
        portrait_id: {
            "sample_id": portrait_id,
            "image": str(path.resolve()),
            "image_sha256": hashlib.sha256(
                cv2.cvtColor(cv2.imread(str(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).tobytes()
            ).hexdigest(),
        }
        for portrait_id, path in sorted(portrait_paths.items())
    }
    binding = {"detector_sha256": file_sha256(detector), "python": str(python.resolve()), "images": expected_images, "threshold": 0.9}
    if output.is_file():
        try:
            cached = read_json(output)
            if cached.get("binding") == binding and cached.get("status") == "complete":
                return cached
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    if not python.is_file():
        raise FileNotFoundError(f"static detector Python executable missing: {python}")
    request.parent.mkdir(parents=True, exist_ok=True)
    write_json(request, list(expected_images.values()))
    command = [str(python), str(detector), "--requests", str(request), "--output", str(output), "--threshold", "0.9"]
    completed = subprocess.run(command, cwd=str(REPO_ROOT), text=True, capture_output=True, check=False)
    if completed.returncode != 0 or not output.is_file():
        raise RuntimeError(f"static portrait detection failed ({completed.returncode}): {completed.stderr[-2000:]}")
    payload = read_json(output)
    records = payload.get("records")
    if payload.get("status") != "complete" or not isinstance(records, list) or len(records) != len(expected_images):
        raise RuntimeError("static portrait detection output is incomplete")
    by_id = {str(row.get("sample_id")): row for row in records}
    if set(by_id) != set(expected_images):
        raise RuntimeError("static portrait detection keyed join is incomplete")
    for portrait_id, expected in expected_images.items():
        row = by_id[portrait_id]
        if row.get("image_sha256") != expected["image_sha256"]:
            raise RuntimeError(f"static portrait detection input hash mismatch: {portrait_id}")
    payload["binding"] = binding
    payload["command"] = command
    write_json(output, payload)
    return payload


def _pair_row(pair: Mapping[str, Any], *, direct_root: Path) -> dict[str, Any]:
    natural = dict(pair["sides"]["natural"])
    tts = dict(pair["sides"]["tts"])
    n_pcm, n_meta = read_pcm16(natural["audio_path"])
    t_pcm, t_meta = read_pcm16(tts["audio_path"])
    if int(natural.get("sample_count", n_pcm.size)) != int(n_pcm.size) or int(tts.get("sample_count", t_pcm.size)) != int(t_pcm.size):
        raise ValueError(f"registry sample count mismatch: {pair['pair_id']}")
    direct = direct_root / f"{pair['pair_id']}.wav"
    output: dict[str, Any] = {
        "pair_id": str(pair["pair_id"]),
        "sample_id": str(pair.get("sample_id", pair["pair_id"])),
        "source_group": str(pair["source_group"]),
        "analysis_split": str(pair["analysis_split"]),
        "transcript": str(pair.get("transcript", natural.get("transcript", ""))),
        "natural": {"audio_path": str(Path(natural["audio_path"]).resolve()), "pcm_sha256": n_meta["pcm_sha256"], "container_sha256": n_meta["container_sha256"], "sample_count": int(n_pcm.size), "tokens": normalize_tokens(natural.get("tokens", []))},
        "tts": {"audio_path": str(Path(tts["audio_path"]).resolve()), "pcm_sha256": t_meta["pcm_sha256"], "container_sha256": t_meta["container_sha256"], "sample_count": int(t_pcm.size), "tokens": normalize_tokens(tts.get("tokens", []))},
        "direct": {"audio_path": str(direct.resolve()), "exists": bool(direct.is_file())} if direct.is_file() else {"audio_path": None, "exists": False},
    }
    _check_unknown(output, {"pair_id", "sample_id", "source_group", "analysis_split", "transcript", "natural", "tts", "direct"}, "inference manifest row")
    return output


def import_parent_assets(config: Mapping[str, Any], run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    source = config["source"]
    registry = read_json(REPO_ROOT / source["parent_registry"])
    pairs = registry.get("pairs", [])
    if len(pairs) != int(config.get("cohort", {}).get("expected_pairs", 240)):
        raise ValueError(f"P2 pair count mismatch: expected 240, got {len(pairs)}")
    direct_root = REPO_ROOT / source.get("parent_direct_dir", "runs/phone_separability_enhancement_audit_v2_20260921/05_evaluation/direct/OPT_DYNAMIC_6")
    rows = [_pair_row(pair, direct_root=direct_root) for pair in pairs]
    counts: dict[str, int] = {}
    groups: dict[str, set[str]] = {}
    for row in rows:
        counts[row["analysis_split"]] = counts.get(row["analysis_split"], 0) + 1
        groups.setdefault(row["analysis_split"], set()).add(row["source_group"])
    expected_counts = config.get("cohort", {}).get("split_counts", {"fit": 135, "dev": 65, "e_seen": 40})
    if counts != {str(k): int(v) for k, v in expected_counts.items()}:
        raise ValueError(f"cohort split mismatch: expected {expected_counts}, got {counts}")
    portrait_config = config.get("portraits", {})
    portrait_paths = {
        portrait_id: (REPO_ROOT / str(value)).resolve()
        for portrait_id, value in {
            "3": portrait_config.get("primary", "data/data/image/3.png"),
            "6": (portrait_config.get("sensitivity") or ["data/data/image/6.png"])[0],
            "9": (portrait_config.get("sensitivity") or ["data/data/image/6.png", "data/data/image/9.png"])[-1],
        }.items()
    }
    if bool(config.get("static_geometry", False)):
        detected = _detect_portraits(config, root, portrait_paths)
        detected_by_id = {str(row["sample_id"]): row for row in detected["records"]}
        portraits = {portrait_id: _portrait_meta(path, detected_by_id[portrait_id]) for portrait_id, path in portrait_paths.items()}
        write_json(root / "00_protocol/portrait_geometry.json", {"status": "COMPLETE", "binding": detected.get("binding"), "portraits": portraits})
    else:
        # Kept solely for reading historical v1 manifests.  New v2 configs must
        # set static_geometry=true and therefore cannot silently use this path.
        portraits = {}
        for portrait_id, path in portrait_paths.items():
            decoded = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"portrait cannot be decoded: {path}")
            portraits[portrait_id] = _portrait_meta(path, {"detections": [{"x1": 0, "y1": 0, "x2": decoded.shape[1] - 1, "y2": decoded.shape[0] - 1, "score": 0.9}], "threshold": 0.9, "device": "legacy"}, allow_full_frame=True)
    training = [{"pair_id": row["pair_id"], "source_group": row["source_group"], "analysis_split": row["analysis_split"], "audio_path": row["natural"]["audio_path"], "sample_count": row["natural"]["sample_count"], "transcript": row["transcript"], "tokens": row["natural"]["tokens"]} for row in rows if row["analysis_split"] == "fit"]
    inference = [{"pair_id": row["pair_id"], "source_group": row["source_group"], "analysis_split": row["analysis_split"], "audio_path": row["natural"]["audio_path"], "sample_count": row["natural"]["sample_count"], "transcript": row["transcript"], "tokens": row["natural"]["tokens"], "direct_audio_path": row["direct"]["audio_path"]} for row in rows]
    render = [{"pair_id": row["pair_id"], "source_group": row["source_group"], "analysis_split": row["analysis_split"], "portraits": portraits, "audio_arms": {"N": row["natural"]["audio_path"], "T": row["tts"]["audio_path"], "D": row["direct"]["audio_path"]}} for row in rows]
    for item in training + inference + render:
        if FORBIDDEN_FIELDS.intersection(item):
            raise ValueError("video field leaked into a worker manifest")
    payload = {"schema_version": 1, "parent_registry": str((REPO_ROOT / source["parent_registry"]).resolve()), "parent_registry_sha256": file_sha256(REPO_ROOT / source["parent_registry"]), "rows": rows, "split_counts": counts, "split_groups": {key: sorted(value) for key, value in groups.items()}, "portraits": portraits, "manifest_hash": canonical_hash(rows)}
    write_json(root / "00_protocol/registry.json", payload)
    write_json(root / "00_protocol/portraits.json", portraits)
    write_json(root / "00_protocol/training_manifest.json", {"schema_version": 1, "rows": training, "manifest_hash": canonical_hash(training)})
    write_json(root / "00_protocol/inference_manifest.json", {"schema_version": 1, "rows": inference, "manifest_hash": canonical_hash(inference)})
    write_json(root / "00_protocol/render_manifest.json", {"schema_version": 1, "rows": render, "manifest_hash": canonical_hash(render)})
    write_jsonl(root / "00_protocol/assets.jsonl", rows)
    return payload


def load_registry(run_dir: str | Path) -> dict[str, Any]:
    return read_json(Path(run_dir) / "00_protocol/registry.json")


def rows_by_split(registry: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    return [dict(row) for row in registry.get("rows", []) if str(row.get("analysis_split")) == str(split)]


__all__ = ["FORBIDDEN_FIELDS", "IMAGE_IDS", "import_parent_assets", "load_registry", "normalize_tokens", "read_pcm16", "rows_by_split", "write_pcm16"]
