#!/usr/bin/env python3
"""SOURCE_ONLY reader for the frozen six-row output-owned MuseTalk screen.

This is derived from response_readout_executor_attempt05.py.  It accepts one
input manifest with exactly two samples x MuseTalk x {zero, plus, minus}.
SyncNet uses each row's original H.264 video with the original natural WAV.
Primary mouth geometry reads the retained pre-codec PNG sequence directly,
using the qualified 483f IMAGE detector and unchanged trajectory guard.

Input manifest schema (paths are absolute on the execution host):
  schema = "output_owned_musetalk_readout_input_v1", status = "PASS"
  protocol = {path, sha256}
  renderer = {runner:{path,sha256}, source_manifest:{path,sha256},
              input_manifest:{path,sha256}, controller_receipt:{path,sha256}}
  operator = {cpu_input_manifest:{path,sha256}, root_array_qa:{path,sha256},
              owned_output_frames:{path,sha256}}
  sources = {canonical_scorer, features, visual_guard, landmarker,
             evaluator, overdub, syncnet_checkpoint, syncnet_instance}
            where every value is {path, sha256}
  audio_contracts[sample_id] = {natural_path, natural_wav_sha256,
      natural_pcm_float32_sha256, sample_count}
  owned_output_frame_indices[sample_id] = sorted unique frame indices
  rows = six entries with row_id, sample_id, arm (zero/plus/minus),
      model="musetalk", seed=20261003, expected_frames, fps_num=25,
      fps_den=1, render_receipt:{path,sha256}, run_status:{path,sha256,
      status,returncode,pid,start_ticks}, silent_video:{path,sha256},
      precodec_frames:[{frame_index,path,sha256}, ...]

Future invocation (only after independent ROOT authorization):
  python -B output_owned_musetalk_readout_executor_attempt01.py
      --input-manifest <qualified-six-row-input.json>
      --output-root <fresh-remote-readout-root> --execute

No renderer, model, baseline, or test is started by source preparation.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "output_owned_musetalk_readout_input_v1"
PROTOCOL_SHA256 = "561e81a747b62c02af7a2465eac9457436013ffba45286be15d8c6cf40c9123e"
RENDERER_RUNNER_SHA256 = "484f08bac841513a754eb94910b91b8fbc863bc9ca080d74746f3b221e80eeb0"
EXTRACTOR_SHA256 = "483f361f571feee56eb714b4b15c9cc778eaafbf21041418400257e233cae1da"
FIXED_SOURCE_SHA256 = {
    "canonical_scorer": "c8753e5d3a89b39dfbc90732bf233e30d86d6c2206d9327c222706f8c6a059f8",
    "features": EXTRACTOR_SHA256,
    "visual_guard": "e154a46aa70e3c434ecc3ae9618e82e072d06fe36126eca19571404f6ca6968f",
    "landmarker": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
    "evaluator": "b15762e1837fd28eaba8fe9e261ffb35b164246da555cb3a7e3bbc7c645f818b",
    "overdub": "0febe6f2b43ed4fb089bdf36e2e208cc1cd61681d8e0ebfd7f59cd56dd00b4b6",
    "syncnet_checkpoint": "961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442",
    "syncnet_instance": "bd722bf844700d6a923dec2f10d7cacd65c0c6b22d09aa5d44d2509660e900d6",
}
SAMPLE_IDS = ("BAC009S0022W0123", "BAC009S0024W0122")
ARMS = ("zero", "plus", "minus")
ARM_LABELS = {
    "zero": "native_height_zero",
    "plus": "native_height_plus",
    "minus": "native_height_minus",
}
EXPECTED_FRAMES = {"BAC009S0022W0123": 80, "BAC009S0024W0122": 88}
SEED = 20261003
FPS = 25


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_new(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def ref_path(ref: dict[str, Any], label: str, base: Path) -> Path:
    raw = ref.get("path")
    expected = ref.get("sha256")
    if not isinstance(raw, str) or not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"input reference is malformed: {label}")
    path = Path(raw)
    if not path.is_absolute():
        path = (base / path).resolve()
    path = path.resolve(strict=True)
    actual = sha_file(path)
    if actual != expected:
        raise ValueError(f"input reference SHA mismatch: {label}={actual}")
    return path


def load_inputs(manifest_path: Path) -> tuple[dict[str, Any], dict[str, Path], dict[str, Any]]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != SCHEMA or manifest.get("status") != "PASS":
        raise ValueError("input manifest schema/status is not qualified for this reader")
    base = manifest_path.parent

    protocol_ref = manifest.get("protocol")
    protocol_path = ref_path(protocol_ref, "protocol", base)
    if protocol_ref.get("sha256") != PROTOCOL_SHA256:
        raise ValueError("output-owned protocol pin differs from the frozen protocol")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (protocol.get("schema") != "output_owned_musetalk_articulatory_probe_v1"
            or protocol.get("status") != "FROZEN_SOURCE_AND_CPU_PREPARATION_ONLY"
            or protocol.get("cohort") != list(SAMPLE_IDS)
            or protocol.get("model") != "musetalk"
            or protocol.get("seed") != SEED
            or protocol.get("future_gpu_cap", {}).get("arms") != list(ARMS)
            or protocol.get("p_by_sample") != {sample: 4 for sample in SAMPLE_IDS}):
        raise ValueError("frozen output-owned protocol identity or six-row scope differs")

    refs: dict[str, Path] = {"protocol": protocol_path}
    renderer = manifest.get("renderer")
    if not isinstance(renderer, dict):
        raise ValueError("renderer provenance object is missing")
    for name in ("runner", "source_manifest", "input_manifest", "controller_receipt"):
        refs[f"renderer_{name}"] = ref_path(renderer.get(name, {}), f"renderer.{name}", base)
    if renderer["runner"].get("sha256") != RENDERER_RUNNER_SHA256:
        raise ValueError("pre-codec MuseTalk runner pin differs from the qualified source")

    operator = manifest.get("operator")
    if not isinstance(operator, dict):
        raise ValueError("operator provenance object is missing")
    for name in ("cpu_input_manifest", "root_array_qa", "owned_output_frames"):
        refs[f"operator_{name}"] = ref_path(operator.get(name, {}), f"operator.{name}", base)

    source_refs = manifest.get("sources")
    if not isinstance(source_refs, dict):
        raise ValueError("runtime/source pin map is missing")
    for name, expected_sha in FIXED_SOURCE_SHA256.items():
        path = ref_path(source_refs.get(name, {}), f"sources.{name}", base)
        if source_refs[name].get("sha256") != expected_sha:
            raise ValueError(f"qualified source SHA differs: {name}")
        refs[name] = path

    audio_contracts = manifest.get("audio_contracts")
    if not isinstance(audio_contracts, dict) or set(audio_contracts) != set(SAMPLE_IDS):
        raise ValueError("natural PCM contract roster differs from the two frozen samples")
    for sample in SAMPLE_IDS:
        contract = audio_contracts[sample]
        wav = Path(contract.get("natural_path", ""))
        if not wav.is_absolute():
            wav = (base / wav).resolve()
        if not wav.is_file() or sha_file(wav) != contract.get("natural_wav_sha256"):
            raise ValueError(f"natural PCM WAV file pin mismatch: {sample}")
        if type(contract.get("sample_count")) is not int or contract["sample_count"] <= 0:
            raise ValueError(f"natural PCM sample count is missing: {sample}")
        if not isinstance(contract.get("natural_pcm_float32_sha256"), str):
            raise ValueError(f"canonical float32 PCM binding is missing: {sample}")
        contract["natural_path"] = str(wav.resolve(strict=True))

    rows = manifest.get("rows")
    expected = {(sample, arm) for sample in SAMPLE_IDS for arm in ARMS}
    observed: set[tuple[str, str]] = set()
    for row in rows if isinstance(rows, list) else ():
        sample, arm = row.get("sample_id"), row.get("arm")
        if (sample not in SAMPLE_IDS or arm not in ARMS or row.get("model") != "musetalk"
                or row.get("seed") != SEED or row.get("expected_frames") != EXPECTED_FRAMES[sample]
                or row.get("fps_num") != FPS or row.get("fps_den") != 1):
            raise ValueError(f"unexpected renderer row identity/clock: {row.get('row_id')}")
        key = (sample, arm)
        if key in observed:
            raise ValueError(f"duplicate renderer row: {key}")
        observed.add(key)
        if row.get("row_id") != f"{sample}/musetalk/{arm}":
            raise ValueError(f"renderer row ID differs from frozen identity: {key}")
        for name in ("render_receipt", "run_status", "silent_video"):
            row[name]["_resolved_path"] = str(ref_path(row.get(name, {}), f"{row['row_id']}.{name}", base))
        status = row["run_status"]
        if (status.get("status") not in ("PASS", "PASS_RENDER_ONLY")
                or status.get("returncode") != 0 or type(status.get("pid")) is not int
                or not isinstance(status.get("start_ticks"), (str, int))
                or row["render_receipt"].get("sha256") != sha_file(Path(row["render_receipt"]["_resolved_path"]))):
            raise ValueError(f"render process/receipt is not a terminal successful row: {key}")
        frame_rows = row.get("precodec_frames")
        if not isinstance(frame_rows, list) or len(frame_rows) != EXPECTED_FRAMES[sample]:
            raise ValueError(f"pre-codec PNG roster length differs: {key}")
        for frame_index, frame in enumerate(frame_rows):
            if frame.get("frame_index") != frame_index:
                raise ValueError(f"pre-codec PNG indices are not canonical: {key}/{frame_index}")
            frame_path = Path(frame.get("path", ""))
            if not frame_path.is_absolute():
                frame_path = (base / frame_path).resolve()
            if frame_path.name != f"{frame_index:08d}.png":
                raise ValueError(f"pre-codec PNG filename differs from the frozen frame contract: {key}/{frame_index}")
            if not frame_path.is_file() or sha_file(frame_path) != frame.get("sha256"):
                raise ValueError(f"pre-codec PNG pin mismatch: {key}/{frame_index}")
            frame["_resolved_path"] = str(frame_path.resolve(strict=True))
    if observed != expected:
        raise ValueError(f"six-row render roster mismatch: missing={sorted(expected-observed)} extra={sorted(observed-expected)}")

    owned = manifest.get("owned_output_frame_indices")
    if not isinstance(owned, dict) or set(owned) != set(SAMPLE_IDS):
        raise ValueError("explicit owned-output-frame map is required for the direct control")
    for sample in SAMPLE_IDS:
        indices = owned[sample]
        if (not isinstance(indices, list) or any(type(i) is not int for i in indices)
                or sorted(set(indices)) != indices
                or not indices or len(indices) >= EXPECTED_FRAMES[sample]
                or any(i < 0 or i >= EXPECTED_FRAMES[sample] for i in indices)):
            raise ValueError(f"owned-output-frame indices are malformed: {sample}")
    owner_doc = json.loads(refs["operator_owned_output_frames"].read_text(encoding="utf-8"))
    if (owner_doc.get("schema") != "output_owned_musetalk_edit_frame_map_v1"
            or owner_doc.get("owned_output_frame_indices_by_sample") != owned):
        raise ValueError("declared non-owned RGB control mask differs from the bound operator frame map")
    return manifest, refs, protocol


def compute_metrics(matrix: np.ndarray, lag: np.ndarray) -> dict[str, Any]:
    if matrix.ndim != 2 or matrix.shape[1] != lag.size or matrix.dtype != np.float32:
        raise RuntimeError(f"unexpected SyncNet matrix contract: {matrix.shape}/{matrix.dtype}")
    zero_cols = np.flatnonzero(lag == 0)
    if zero_cols.size != 1:
        raise RuntimeError(f"lag axis must contain exactly one zero column: {lag.tolist()}")
    import torch
    torch.set_num_threads(1)
    curve = torch.mean(torch.from_numpy(np.ascontiguousarray(matrix.T)), dim=1)
    dmin, idx_t = torch.min(curve, dim=0)
    baseline = torch.median(curve)
    d0 = curve[int(zero_cols[0])]
    idx = int(idx_t.item())
    return {
        "B": float(np.float32(baseline.item())),
        "Dmin": float(np.float32(dmin.item())),
        "D0": float(np.float32(d0.item())),
        "Cbest": float(np.float32((baseline - dmin).item())),
        "Canchor": float(np.float32((baseline - d0).item())),
        "best_lag_frames": int(lag[idx]),
        "official_av_offset_frames": int(-lag[idx]),
        "query_count": int(matrix.shape[0]),
        "lag_count": int(matrix.shape[1]),
        "lag_curve": curve.numpy().astype(np.float32).tolist(),
    }


def save_full_guard20(score_npz: Path, row_out: Path) -> dict[str, Any]:
    with np.load(score_npz, allow_pickle=False) as bundle:
        matrix = np.ascontiguousarray(bundle["distance_by_query_lag"], dtype=np.float32)
        lag = np.ascontiguousarray(bundle["lag_offsets_frames"], dtype=np.int32)
    if lag.tolist() != list(range(-15, 16)):
        raise RuntimeError(f"unexpected SyncNet lag axis: {lag.tolist()}")
    guarded = np.ascontiguousarray(matrix[20:-20, :], dtype=np.float32)
    if guarded.shape[0] <= 0:
        raise RuntimeError("guard20 leaves no SyncNet query rows")
    indices = np.arange(20, matrix.shape[0] - 20, dtype=np.int32)
    guard_path = row_out / "distance_by_query_lag_guard20.npz"
    np.savez_compressed(guard_path, distance_by_query_lag=guarded, lag_offsets_frames=lag,
                        original_query_row_indices=indices)
    return {
        "matrix_npz_path": str(score_npz), "matrix_npz_sha256": sha_file(score_npz),
        "matrix_array_sha256_le_f32": sha_bytes(matrix.astype("<f4", copy=False).tobytes(order="C")),
        "matrix_shape": list(matrix.shape), "matrix_dtype": matrix.dtype.str,
        "lag_offsets_frames": lag.tolist(), "zero_lag_column": int(np.flatnonzero(lag == 0)[0]),
        "metrics": compute_metrics(matrix, lag),
        "guard20": {
            "rule": "query rows [20:N-20), all 31 lag columns; original clock, no lag selection or remux shift",
            "query_row_indices": [int(indices[0]), int(indices[-1])],
            "query_count": int(guarded.shape[0]), "matrix_npz_path": str(guard_path),
            "matrix_npz_sha256": sha_file(guard_path),
            "matrix_array_sha256_le_f32": sha_bytes(guarded.astype("<f4", copy=False).tobytes(order="C")),
            "matrix_shape": list(guarded.shape), "matrix_dtype": guarded.dtype.str,
            "metrics": compute_metrics(guarded, lag),
        },
    }


def precodec_sequence(row: dict[str, Any], features: Any, landmarker_handle: tuple[Any, Any]):
    import cv2

    mp, landmarker = landmarker_handle
    landmarks: list[np.ndarray] = []
    body_hashes: list[str] = []
    expected_size: tuple[int, int] | None = None
    for entry in row["precodec_frames"]:
        path = Path(entry["_resolved_path"])
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"pre-codec PNG is not RGB8-decodable: {path}")
        if expected_size is None:
            expected_size = (int(bgr.shape[1]), int(bgr.shape[0]))
        if (bgr.shape[1], bgr.shape[0]) != expected_size:
            raise ValueError(f"pre-codec frame size changed within row: {row['row_id']}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if rgb.dtype != np.uint8 or not rgb.flags.c_contiguous:
            rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
        body_hashes.append(sha_bytes(rgb.tobytes(order="C")))
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        if result.face_landmarks:
            points = np.asarray([(point.x, point.y) for point in result.face_landmarks[0]], dtype=np.float64)
            if points.shape != (478, 2):
                points = np.full((478, 2), np.nan, dtype=np.float64)
        else:
            points = np.full((478, 2), np.nan, dtype=np.float64)
        landmarks.append(points)
    if not landmarks or expected_size is None:
        raise ValueError(f"pre-codec frame sequence is empty: {row['row_id']}")
    landmark_array = np.stack(landmarks, axis=0)
    geom = features.mouth_features(landmark_array)
    landmark_array = np.nan_to_num(landmark_array, nan=0.0, posinf=0.0, neginf=0.0)
    frame_count = len(landmarks)
    times = np.arange(frame_count, dtype=np.float64) / FPS
    sequence = features.VisualSequence(
        landmarks=landmark_array, canonical_mouth=geom["canonical_mouth"],
        valid=geom["valid"], mouth_open=geom["mouth_open"], mouth_width=geom["mouth_width"],
        mouth_height=geom["mouth_height"], mouth_area=geom["mouth_area"],
        timestamps_s=times,
        metadata={
            "schema_version": features.SCHEMA_VERSION,
            "video_path": str(row["precodec_frames"][0]["_resolved_path"]),
            "input_kind": "ordered_precodec_png_sequence",
            "declared_frame_count": frame_count, "decoded_frame_count": frame_count,
            "read_frame_count": frame_count, "decode_errors": 0, "fps": float(FPS),
            "width": expected_size[0], "height": expected_size[1], "fourcc": None,
            "landmarker_asset": str(row["_landmarker_path"]),
            "coordinate_system": "mediapipe_normalized_xy_then_eye_similarity_canonical",
            "landmark_confidence": "unavailable_from_face_landmarker_tasks_api",
            "mouth_indices": list(features.MOUTH_INDICES),
            "valid_fraction": float(geom["valid"].mean()),
            "timestamps_monotonic": bool(np.all(np.diff(times) > 0)),
        },
    )
    return sequence, body_hashes


def save_visual(row: dict[str, Any], row_out: Path, features: Any, guard: Any,
                landmarker_handle: tuple[Any, Any]) -> dict[str, Any]:
    sequence, body_hashes = precodec_sequence(row, features, landmarker_handle)
    exact_times = [Fraction(index, FPS) for index in range(len(sequence.valid))]
    metrics, arrays = guard._trajectory_summary(sequence, exact_times)
    arrays_path = row_out / "mouth_geometry_aperture_velocity_precodec.npz"
    np.savez_compressed(arrays_path, **arrays)
    array_meta = {}
    for key, value in arrays.items():
        array = np.asarray(value)
        array_meta[key] = {
            "dtype": array.dtype.str, "shape": list(array.shape),
            "array_sha256": sha_bytes(np.ascontiguousarray(array).tobytes(order="C")),
        }
    receipt = {
        "schema": "output_owned_musetalk_precodec_visual_receipt_v1",
        "status": "PASS", "row_id": row["row_id"], "sample_id": row["sample_id"],
        "model": "musetalk", "arm": ARM_LABELS[row["arm"]], "seed": SEED,
        "source_kind": "original retained pre-codec PNG frames; no video decode or temporal tracking",
        "frame_count": len(sequence.valid), "fps": FPS,
        "timestamps": {"rule": "frame_index/25", "count": len(sequence.valid),
                       "last_s": float(sequence.timestamps_s[-1])},
        "extractor": "remote_video_features_483f.py::create_landmarker(IMAGE)+detect(SRGB)+mouth_features",
        "extractor_sha256": EXTRACTOR_SHA256,
        "visual_guard_path": str(row["_guard_path"]),
        "visual_guard_sha256": row["_guard_sha256"],
        "landmarker_path": str(row["_landmarker_path"]),
        "landmarker_sha256": row["_landmarker_sha256"],
        "precodec_rgb_body_sha256_by_frame": body_hashes,
        "precodec_frames": [
            {
                "frame_index": int(item["frame_index"]),
                "timestamp_num": int(item["frame_index"]),
                "timestamp_den": FPS,
                "path": item["_resolved_path"],
                "png_file_sha256": item["sha256"],
                "decoded_rgb_body_sha256": body_hashes[index],
            }
            for index, item in enumerate(row["precodec_frames"])
        ],
        "valid_frame_count": int(sequence.valid.sum()),
        "valid_fraction": float(sequence.valid.mean()),
        "metrics": metrics,
        "arrays_npz": {"path": str(arrays_path), "sha256": sha_file(arrays_path),
                       "arrays": array_meta},
        "feature_metadata": sequence.metadata,
    }
    receipt_path = row_out / "mouth_geometry_aperture_velocity_precodec.json"
    write_new(receipt_path, receipt)
    return {
        "status": "PASS", "receipt": str(receipt_path),
        "receipt_sha256": sha_file(receipt_path), "arrays_npz": str(arrays_path),
        "arrays_npz_sha256": sha_file(arrays_path), "metrics": metrics,
    }


def precodec_control(rows_by_key: dict[tuple[str, str], dict[str, Any]],
                     owned_by_sample: dict[str, list[int]]) -> dict[str, Any]:
    import cv2

    def read_rgb(entry: dict[str, Any]) -> np.ndarray:
        bgr = cv2.imread(entry["_resolved_path"], cv2.IMREAD_COLOR)
        if bgr is None or bgr.dtype != np.uint8 or bgr.ndim != 3 or bgr.shape[2] != 3:
            raise ValueError(f"pre-codec control PNG is not RGB8-decodable: {entry['_resolved_path']}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    results = []
    for sample in SAMPLE_IDS:
        arms = {arm: rows_by_key[(sample, arm)] for arm in ARMS}
        owned = set(owned_by_sample[sample])
        outside = [index for index in range(EXPECTED_FRAMES[sample]) if index not in owned]
        mismatches = []
        for arm in ("plus", "minus"):
            for index in outside:
                zero_rgb = read_rgb(arms["zero"]["precodec_frames"][index])
                arm_rgb = read_rgb(arms[arm]["precodec_frames"][index])
                if zero_rgb.shape != arm_rgb.shape:
                    raise ValueError(f"non-owned RGB control frame dimensions differ: {sample}/{arm}/{index}")
                if not np.array_equal(zero_rgb, arm_rgb):
                    mismatches.append({
                        "sample_id": sample, "arm": arm, "frame_index": index,
                        "zero_rgb_sha256": sha_bytes(zero_rgb.tobytes(order="C")),
                        "arm_rgb_sha256": sha_bytes(arm_rgb.tobytes(order="C")),
                    })
        results.append({
            "sample_id": sample, "owned_output_frame_indices": sorted(owned),
            "nonowned_frame_count_checked_per_candidate": len(outside),
            "candidate_frames_compared": 2 * len(outside),
            "mismatch_count": len(mismatches), "mismatches": mismatches,
            "status": "PASS" if not mismatches else "CONTROL_FAILED",
        })
    return {
        "schema": "output_owned_musetalk_precodec_direct_exposure_control_v1",
        "status": "PASS" if all(row["status"] == "PASS" for row in results) else "CONTROL_FAILED",
        "comparison": "RGB8 arrays decoded from retained pre-codec PNGs; each plus/minus frame compared with same-index zero frame only outside frozen owned-output-frame map",
        "rows": results,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true",
                        help="source is inert without explicit flag and separate ROOT authorization")
    args = parser.parse_args(argv)
    if not args.execute:
        print("SOURCE_ONLY_NOT_EXECUTED")
        return 0
    output_root = args.output_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"create-only readout root already exists: {output_root}")
    manifest, refs, protocol = load_inputs(args.input_manifest)

    scorer = load_module("output_owned_musetalk_canonical_scorer", refs["canonical_scorer"])
    features = load_module("output_owned_musetalk_features_483f", refs["features"])
    guard = load_module("output_owned_musetalk_visual_guard", refs["visual_guard"])
    landmarker_path = refs["landmarker"]
    scorer.EVALUATOR = refs["evaluator"]
    scorer.OVERDUB = refs["overdub"]
    scorer.FEATURES = refs["features"]
    scorer.VISUAL_GUARD = refs["visual_guard"]
    scorer.LANDMARKER = refs["landmarker"]
    scorer.SYNC_ROOT = refs["syncnet_instance"].parent
    if ((scorer.SYNC_ROOT / "SyncNetInstance.py").resolve(strict=True) != refs["syncnet_instance"]
            or (scorer.SYNC_ROOT / "data" / "syncnet_v2.model").resolve(strict=True) != refs["syncnet_checkpoint"]):
        raise ValueError("canonical scorer SyncNet root does not resolve to the pinned instance and checkpoint")
    landmarker_handle = features.create_landmarker(landmarker_path)
    output_root.mkdir(parents=True, exist_ok=False)
    operator_refs = manifest["operator"]
    renderer_refs = manifest["renderer"]
    write_new(output_root / "readout_started.json", {
        "schema": "output_owned_musetalk_readout_attempt01_start_v1",
        "status": "RUNNING", "protocol_sha256": PROTOCOL_SHA256,
        "renderer_runner_sha256": RENDERER_RUNNER_SHA256,
        "feature_extractor_sha256": EXTRACTOR_SHA256,
        "operator_inputs": {name: value for name, value in operator_refs.items()},
        "renderer_inputs": {name: value for name, value in renderer_refs.items()},
        "source_pins": {name: {"path": str(refs[name]), "sha256": sha_file(refs[name])}
                        for name in sorted(FIXED_SOURCE_SHA256)},
        "row_count": 6, "models": ["musetalk"], "samples": list(SAMPLE_IDS),
        "arms": list(ARMS), "seed": SEED, "precodec_geometry": True,
        "original_natural_pcm_syncnet": True,
        "h264_secondary_geometry": True,
    })

    rows_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in manifest["rows"]:
        sample, arm = row["sample_id"], row["arm"]
        rows_by_key[(sample, arm)] = row
        row["_landmarker_path"] = str(landmarker_path)
        row["_landmarker_sha256"] = FIXED_SOURCE_SHA256["landmarker"]
        row["_guard_path"] = str(refs["visual_guard"])
        row["_guard_sha256"] = FIXED_SOURCE_SHA256["visual_guard"]

    row_records = []
    try:
        for sample in SAMPLE_IDS:
            contract = manifest["audio_contracts"][sample]
            for arm in ARMS:
                row = rows_by_key[(sample, arm)]
                row_id = row["row_id"]
                video = Path(row["silent_video"]["_resolved_path"])
                natural = Path(contract["natural_path"])
                row_out = output_root / sample / "musetalk" / ARM_LABELS[arm]
                row_out.mkdir(parents=True, exist_ok=False)
                scorer_row = {
                    "row_id": row_id, "sample_id": sample, "speaker_id": sample[6:11],
                    "model_id": "musetalk", "condition": ARM_LABELS[arm],
                    "source_silent": video, "natural_wav": natural,
                }
                crop_meta = scorer.crop_and_mux(scorer_row, row_out)
                crop_path = row_out / "crop_mux_receipt.json"
                write_new(crop_path, {
                    "schema": "output_owned_musetalk_original_pcm_crop_receipt_v1",
                    "row_id": row_id, "sample_id": sample, "model": "musetalk",
                    "arm": ARM_LABELS[arm], "seed": SEED,
                    "render_receipt_path": row["render_receipt"]["_resolved_path"],
                    "render_receipt_sha256": row["render_receipt"]["sha256"],
                    "run_status_path": row["run_status"]["_resolved_path"],
                    "run_status_sha256": row["run_status"]["sha256"],
                    "render_pid": row["run_status"]["pid"],
                    "render_start_ticks": row["run_status"]["start_ticks"],
                    "render_returncode": row["run_status"]["returncode"],
                    "natural_wav_path": str(natural),
                    "natural_wav_sha256": contract["natural_wav_sha256"],
                    "natural_pcm_float32_sha256": contract["natural_pcm_float32_sha256"],
                    **crop_meta,
                })
                score = scorer.score_row(scorer_row, row_out, crop_meta)
                score_meta = save_full_guard20(row_out / "distance_by_query_lag.npz", row_out)
                visual = save_visual(row, row_out, features, guard, landmarker_handle)
                h264_visual_root = row_out / "h264_geometry"
                h264_visual_root.mkdir(parents=False, exist_ok=False)
                h264_visual = scorer.visual_row(scorer_row, h264_visual_root)
                record = {
                    "schema": "output_owned_musetalk_readout_row_v1",
                    "status": "PASS", "row_id": row_id, "sample_id": sample,
                    "speaker_id": sample[6:11], "model": "musetalk",
                    "arm": ARM_LABELS[arm], "seed": SEED,
                    "audio_role": "original natural PCM used for final SyncNet mux/scoring; fixed-RMS driver is not rescored",
                    "clock": "source video frame j owns j/25 seconds; original clock retained",
                    "render_receipt_path": row["render_receipt"]["_resolved_path"],
                    "render_receipt_sha256": row["render_receipt"]["sha256"],
                    "render_run_status_path": row["run_status"]["_resolved_path"],
                    "render_run_status_sha256": row["run_status"]["sha256"],
                    "render_pid": row["run_status"]["pid"],
                    "render_start_ticks": row["run_status"]["start_ticks"],
                    "render_returncode": row["run_status"]["returncode"],
                    "silent_video_path": str(video),
                    "silent_video_sha256": row["silent_video"]["sha256"],
                    "natural_pcm_float32_sha256": contract["natural_pcm_float32_sha256"],
                    "precodec_frame_directory": str(Path(row["precodec_frames"][0]["_resolved_path"]).parent),
                    "precodec_frame_count": len(row["precodec_frames"]),
                    "crop_mux_receipt_path": str(crop_path),
                    "crop_mux_receipt_sha256": sha_file(crop_path),
                    "syncnet": {
                        "status": score["status"], "evaluator_receipt": score["evaluator_receipt"],
                        "evaluator_receipt_sha256": score["evaluator_receipt_sha256"],
                        **score_meta,
                    },
                    "mouth_motion": visual,
                    "mouth_motion_h264": {
                        **h264_visual,
                        "source_kind": "renderer H.264 silent video decoded through canonical scorer.visual_row; descriptive secondary geometry",
                        "source_silent_path": str(video),
                        "source_silent_sha256": row["silent_video"]["sha256"],
                    },
                    "extractor_source_sha256": EXTRACTOR_SHA256,
                    "mouth_interpretation": "pre-codec MediaPipe motion geometry; descriptive only, no accuracy ground truth",
                }
                receipt_path = row_out / "row_receipt.json"
                write_new(receipt_path, record)
                row_records.append({
                    "row_id": row_id, "row_receipt_path": str(receipt_path),
                    "row_receipt_sha256": sha_file(receipt_path),
                })
    finally:
        landmarker_handle[1].close()

    control = precodec_control(rows_by_key, manifest["owned_output_frame_indices"])
    control_path = output_root / "precodec_nonowned_rgb_control.json"
    write_new(control_path, control)
    final = {
        "schema": "output_owned_musetalk_readout_manifest_v1",
        "status": "PASS" if control["status"] == "PASS" else "CONTROL_FAILED",
        "row_count": len(row_records), "protocol_sha256": PROTOCOL_SHA256,
        "renderer_runner_sha256": RENDERER_RUNNER_SHA256,
        "controller_receipt_path": str(refs["renderer_controller_receipt"]),
        "controller_receipt_sha256": sha_file(refs["renderer_controller_receipt"]),
        "input_manifest_path": str(args.input_manifest.resolve(strict=True)),
        "input_manifest_sha256": sha_file(args.input_manifest.resolve(strict=True)),
        "operator": {name: {"path": str(refs[f"operator_{name}"]),
                            "sha256": sha_file(refs[f"operator_{name}"])}
                     for name in ("cpu_input_manifest", "root_array_qa", "owned_output_frames")},
        "precodec_nonowned_rgb_control": {
            "path": str(control_path), "sha256": sha_file(control_path),
            "status": control["status"],
        },
        "rows": row_records,
        "score_endpoint_rule": "full rows [0:N); guard20 query rows [20:N-20); all 31 lag columns; D0 uses explicit lag_offsets_frames==0 column",
        "geometry_source": "primary: retained pre-codec PNG frames via IMAGE detector at frame_index/25; secondary mouth_motion_h264: pinned canonical scorer.visual_row on renderer H.264 silent video; descriptive only",
        "mouth_interpretation": "MediaPipe geometry and velocity describe observed track motion only; no accuracy ground truth",
    }
    write_new(output_root / "readout_manifest.json", final)
    print(json.dumps({"status": final["status"], "rows": len(row_records),
                      "control": control["status"], "output": str(output_root)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
