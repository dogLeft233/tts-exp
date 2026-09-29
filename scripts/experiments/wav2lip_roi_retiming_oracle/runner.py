from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.wav2lip_roi_peak_recheck.worker import SyncNetScorer

from . import config
from .analysis import analyze
from .common import (
    OracleError,
    bytes_sha256,
    file_sha256,
    read_pcm16_wav,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from .media import (
    audit_parent_stream,
    build_oracle_indices,
    encode_video,
    forward_map,
    materialize_audio,
    mux_audio,
    reconstruct_warped_pcm,
)


def _fixed_parent() -> dict[str, Any]:
    try:
        from scripts.experiments.wav2lip_roi_control_diagnostic.analysis import (
            load_parent,
        )

        parent = load_parent()
    except Exception as exc:
        raise OracleError(f"fixed parent audit failed: {exc}") from exc
    fixed_paths = {
        "final": config.PARENT_ROOT / "final.json",
        "protocol": config.PARENT_ROOT / "protocol.json",
        "control": config.PARENT_ROOT / "control.json",
        "media_manifest": config.PARENT_ROOT / "media/control/manifest.json",
        "score_manifest": config.PARENT_ROOT / "scores/control/manifest.json",
    }
    for name, expected in config.PARENT_HASHES.items():
        if file_sha256(fixed_paths[name]) != expected:
            raise OracleError(f"fixed parent hash changed: {name}")
    recheck_final = verify_self_hashed_json(
        config.RECHECK_ROOT / "final.json", config.RECHECK_HASHES["final"]
    )
    recheck_validation = verify_self_hashed_json(
        config.RECHECK_ROOT / "validation.json", config.RECHECK_HASHES["validation"]
    )
    if recheck_final.get("diagnostic_decision") != "PEAKS_REPRODUCED" or recheck_validation.get("valid") is not True:
        raise OracleError("fixed independent peak recheck is not complete and valid")
    return parent


def _source_bindings(parent: Mapping[str, Any]) -> dict[str, dict[str, str]]:
    package = Path(__file__).resolve().parent
    paths: list[tuple[str, Path]] = [(f"new/{path.name}", path) for path in sorted(package.glob("*.py"))]
    paths.extend(
        [
            ("official/SyncNetModel.py", config.SYNCNET_ROOT / "SyncNetModel.py"),
            ("official/SyncNetInstance.py", config.SYNCNET_ROOT / "SyncNetInstance.py"),
            ("official/syncnet_v2.model", config.SYNCNET_MODEL),
            ("parent_audit/analysis.py", config.REPO / "scripts/experiments/wav2lip_roi_control_diagnostic/analysis.py"),
            ("parent_audit/common.py", config.REPO / "scripts/experiments/wav2lip_roi_control_diagnostic/common.py"),
            ("parent_audit/config.py", config.REPO / "scripts/experiments/wav2lip_roi_control_diagnostic/config.py"),
        ]
    )
    for name, item in parent.get("inherited_specs", {}).items():
        paths.append((f"inherited/{name}", Path(str(item["path"]))))
    result: dict[str, dict[str, str]] = {}
    for name, path in paths:
        if not path.is_file():
            raise OracleError(f"source binding is missing: {path}")
        result[name] = {"path": str(path.resolve()), "sha256": file_sha256(path)}
    return result


def _environment(device: str, threads: int) -> dict[str, Any]:
    try:
        import cv2
        import python_speech_features
        import torch

        packages = {
            "torch": torch.__version__,
            "numpy": np.__version__,
            "cv2": cv2.__version__,
            "python_speech_features": getattr(python_speech_features, "__version__", "unknown"),
        }
        cuda_available = bool(torch.cuda.is_available())
        cuda_name = torch.cuda.get_device_name(0) if cuda_available else None
    except Exception as exc:
        raise OracleError(f"scoring environment import failed: {exc}") from exc
    ffmpeg = subprocess.run([str(config.FFMPEG), "-version"], capture_output=True, text=True, check=False)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "device_requested": device,
        "torch_threads": int(threads),
        "cuda_available": cuda_available,
        "cuda_device": cuda_name,
        "ffmpeg_first_line": (ffmpeg.stdout.splitlines() or [""])[0],
        "syncnet_python": str(config.SYNCNET_PYTHON),
        "syncnet_model": str(config.SYNCNET_MODEL.resolve()),
        "syncnet_model_sha256": config.SYNCNET_MODEL_SHA256,
    }


def _media_row(parent: Mapping[str, Any], sample_id: str) -> Mapping[str, Any]:
    for row in parent["media_manifest"]["rows"]:
        if isinstance(row, Mapping) and str(row.get("sample_id")) == sample_id:
            return row
    raise OracleError(f"parent media row is missing: {sample_id}")


def _derive_masks(sample_count: int, frame_count: int) -> dict[str, Any]:
    mapped = forward_map(sample_count)
    q = min(frame_count, sample_count // config.SAMPLES_PER_FRAME)
    candidate = list(range(config.VSHIFT, q - 20))
    plus: list[int] = []
    minus: list[int] = []
    d_by_row: dict[str, float] = {}
    a_by_row: dict[str, float] = {}
    coordinates = np.arange(sample_count, dtype=np.float64)
    for row in candidate:
        center = float(config.SAMPLES_PER_FRAME * (row + 2))
        d = (float(np.interp(center, coordinates, mapped)) - center) / config.SAMPLES_PER_FRAME
        a = (center - float(np.interp(center, mapped, coordinates))) / config.SAMPLES_PER_FRAME
        d_by_row[str(row)] = d
        a_by_row[str(row)] = a
        if d >= 2.5 and a >= 2.5:
            plus.append(row)
        if d <= -2.5 and a <= -2.5:
            minus.append(row)
    if len(plus) < config.MIN_LOCAL_ROWS or len(minus) < config.MIN_LOCAL_ROWS:
        raise OracleError("derived local masks contain fewer than five rows")
    return {
        "sample_count": int(sample_count),
        "q_frames": int(q),
        "candidate_rows": candidate,
        "common_window_rows": list(range(q - config.WINDOW_FRAMES)),
        "plus_rows": plus,
        "minus_rows": minus,
        "d_by_row": d_by_row,
        "a_by_row": a_by_row,
        "plus_expected_offset": float(np.mean([a_by_row[str(row)] for row in plus])),
        "minus_expected_offset": float(np.mean([a_by_row[str(row)] for row in minus])),
        "plus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in plus])),
        "minus_expected_video_response": float(-np.mean([d_by_row[str(row)] for row in minus])),
        "forward_mapping_sha256": bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()),
        "forward_mapping_length": int(mapped.size),
    }


def _mapping_matches(parent_masks: Mapping[str, Any], derived: Mapping[str, Any]) -> bool:
    for key in ("sample_count", "q_frames", "candidate_rows", "common_window_rows", "plus_rows", "minus_rows"):
        if parent_masks.get(key) != derived.get(key):
            return False
    for key in ("d_by_row", "a_by_row"):
        left = parent_masks.get(key)
        right = derived.get(key)
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
            return False
        if any(abs(float(left[item]) - float(right[item])) > 1e-12 for item in left):
            return False
    return True


def _audit_parent(parent: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    source_ids = [str(record["sample_id"]) for record in parent["records"]]
    if bytes_sha256("\n".join(source_ids).encode("utf-8")) != config.ORDERED_SAMPLE_ID_SHA256:
        raise OracleError("parent ordered sample-id hash changed")
    from .common import extract_pcm_from_media

    for record in parent["records"]:
        sample_id = str(record["sample_id"])
        media = _media_row(parent, sample_id)
        stream = media.get("streams", {}).get("G_N")
        if not isinstance(stream, Mapping):
            raise OracleError(f"parent G_N stream is missing: {sample_id}")
        stream_path = Path(str(stream["output"])).resolve()
        if not stream_path.is_file() or file_sha256(stream_path) != str(stream["output_sha256"]):
            raise OracleError(f"parent G_N stream hash changed: {sample_id}")
        frame_count = int(stream.get("frame_count", -1))
        frames, frame_evidence = audit_parent_stream(stream_path, frame_count)
        natural = record.get("natural_audio")
        if not isinstance(natural, Mapping):
            raise OracleError(f"natural audio binding is missing: {sample_id}")
        natural_path = Path(str(natural["path"])).resolve()
        if not natural_path.is_file() or file_sha256(natural_path) != str(natural["sha256"]):
            raise OracleError(f"natural audio hash changed: {sample_id}")
        natural_pcm, natural_values, natural_params = read_pcm16_wav(natural_path)
        if natural_values.size != int(natural["sample_count"]) or bytes_sha256(natural_pcm) != str(natural["decoded_pcm_sha256"]):
            raise OracleError(f"natural PCM binding changed: {sample_id}")
        warped_expected, mapped = reconstruct_warped_pcm(natural_pcm)
        w_cell = media.get("cells", {}).get("G_N__W")
        if not isinstance(w_cell, Mapping):
            raise OracleError(f"parent G_N/W media cell is missing: {sample_id}")
        w_media = Path(str(w_cell["output"])).resolve()
        if not w_media.is_file() or file_sha256(w_media) != str(w_cell["output_sha256"]):
            raise OracleError(f"parent G_N/W media hash changed: {sample_id}")
        warped_actual = extract_pcm_from_media(w_media)
        if warped_actual != warped_expected or bytes_sha256(warped_actual) != str(w_cell["audio_pcm_sha256"]):
            raise OracleError(f"parent W PCM reconstruction differs: {sample_id}")
        masks = record.get("masks")
        if not isinstance(masks, Mapping):
            raise OracleError(f"parent timing masks are missing: {sample_id}")
        derived_masks = _derive_masks(int(natural_values.size), frame_count)
        if not _mapping_matches(masks, derived_masks):
            raise OracleError(f"parent timing masks differ from the frozen formula: {sample_id}")
        indices = build_oracle_indices(int(natural_values.size), frame_count)
        rows.append(
            {
                "sample_id": sample_id,
                "source_group": str(record["source_group"]),
                "frame_count": frame_count,
                "sample_count": int(natural_values.size),
                "natural_audio": {
                    "path": str(natural_path),
                    "container_sha256": file_sha256(natural_path),
                    "decoded_pcm_sha256": bytes_sha256(natural_pcm),
                    "sample_count": int(natural_values.size),
                    "sample_rate": int(natural_params["sample_rate"]),
                },
                "parent_stream": {
                    "path": str(stream_path),
                    "sha256": file_sha256(stream_path),
                    "frame_count": frame_count,
                    "timeline": frame_evidence["timeline"],
                },
                "parent_w_media": {
                    "path": str(w_media),
                    "sha256": file_sha256(w_media),
                    "audio_pcm_sha256": str(w_cell["audio_pcm_sha256"]),
                },
                "warp": {
                    "mapping_sha256": bytes_sha256(np.asarray(mapped, dtype="<f8").tobytes()),
                    "mapping_length": int(mapped.size),
                    "strictly_increasing": True,
                    "expected_pcm_sha256": bytes_sha256(warped_expected),
                    "observed_pcm_sha256": bytes_sha256(warped_actual),
                    "pcm_exact": True,
                },
                "indices": indices,
                "masks": dict(masks),
                "parent_cells": {
                    "G_N__N": dict(media["cells"]["G_N__N"]),
                    "G_N__W": dict(media["cells"]["G_N__W"]),
                },
                "_frames": frames,
                "_natural_pcm": natural_pcm,
                "_warped_pcm": warped_actual,
            }
        )
    audit = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "status": "complete",
        "record_count": len(rows),
        "source_group_count": len({row["source_group"] for row in rows}),
        "stream_count": len(rows),
        "parent_g_n_streams_checked": len(rows),
        "parent_w_pcm_reconstructions_checked": len(rows),
        "q_indices_checked": len(rows),
        "masks_checked": len(rows),
        "new_generated_videos": 0,
        "new_score_cells": 0,
        "records": [{key: value for key, value in row.items() if not key.startswith("_")} for row in rows],
    }
    return audit, rows


def _protocol(parent: Mapping[str, Any], audit: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], run_id: str, device: str, threads: int) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    selected_cells: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row["sample_id"])
        parent_record = next(item for item in parent["records"] if str(item["sample_id"]) == sample_id)
        records.append(
            {
                "sample_id": sample_id,
                "source_group": str(row["source_group"]),
                "protocol_split": "seen_fit_diagnostic",
                "frame_count": int(row["frame_count"]),
                "sample_count": int(row["sample_count"]),
                "masks": row["masks"],
                "indices": row["indices"],
                "natural_audio": row["natural_audio"],
                "parent_stream": row["parent_stream"],
                "parent_w_media": row["parent_w_media"],
                "parent_cells": row["parent_cells"],
                "historical_parent_record_hashes": parent_record.get("parent_record_hashes"),
            }
        )
        for video, audio in config.CELL_SPECS:
            selected_cells.append(
                {
                    "sample_id": sample_id,
                    "source_group": str(row["source_group"]),
                    "video_arm": video,
                    "audio_arm": audio,
                    "media": None,
                    "media_sha256": None,
                    "audio": None,
                    "audio_container_sha256": None,
                    "audio_pcm_sha256": None,
                }
            )
    return {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "protocol_revision": config.PROTOCOL_REVISION,
        "run_id": run_id,
        "status": "frozen",
        "classification": "seen_fit_diagnostic",
        "parent": {
            "root": str(config.PARENT_ROOT.resolve()),
            "fixed_hashes": dict(config.PARENT_HASHES),
            "scientific_decision": "CONTROL_FAILED",
            "record_count": config.EXPECTED_RECORD_COUNT,
            "source_group_count": config.EXPECTED_SOURCE_GROUP_COUNT,
        },
        "recheck": {
            "root": str(config.RECHECK_ROOT.resolve()),
            "fixed_hashes": dict(config.RECHECK_HASHES),
            "decision": "PEAKS_REPRODUCED",
        },
        "change_bindings": config.spec_bindings(),
        "source_bindings": _source_bindings(parent),
        "frozen_config": config.FrozenConfig().to_dict(),
        "environment": _environment(device, threads),
        "audit_sha256": file_sha256(config.run_root_for(run_id) / "input_audit.json"),
        "records": records,
        "selected_cells": selected_cells,
        "expected_record_count": config.EXPECTED_RECORD_COUNT,
        "expected_stream_count": config.EXPECTED_STREAM_COUNT,
        "expected_media_count": config.EXPECTED_MEDIA_COUNT,
        "expected_score_cell_count": config.EXPECTED_SCORE_CELL_COUNT,
    }


def _result_markdown(analysis: Mapping[str, Any]) -> str:
    lines = [
        "# Wav2Lip ROI retiming oracle 2026-09-06",
        "",
        "本实验从现有 G_N 的无损 ROI 像素构造 V_ID/V_ORACLE，未重新生成 Wav2Lip 视频。",
        "",
        f"- 科学终态：[{analysis['decision']}]",
        f"- 记录：{analysis['record_count']}/{analysis['expected_record_count']}；新评分：{analysis['score_cell_count']}/{analysis['expected_score_cell_count']}",
        f"- 恒等臂复现：{analysis['identity_reproduction']['pass']}；最大矩阵差：{analysis['identity_reproduction']['max_abs_difference']:.9f}",
        f"- 基线可解释：{analysis['baseline_count']}/{analysis['record_count']}；B/C_oracle/O_oracle：{analysis['timing_counts']['B']}/{analysis['timing_counts']['C_oracle']}/{analysis['timing_counts']['O_oracle']}",
        f"- own CI 下界：C={analysis['own_bootstrap']['c']['ci95'][0]:.6f}，D={analysis['own_bootstrap']['d']['ci95'][0]:.6f}；offset agreement：{analysis['own_gate']['offset_agreement_count']}/{analysis['record_count']}",
        f"- damage CI 下界：C={analysis['damage_bootstrap']['c']['ci95'][0]:.6f}，D={analysis['damage_bootstrap']['d']['ci95'][0]:.6f}；both positive：{analysis['damage_gate']['both_positive_count']}/{analysis['record_count']}",
        "",
        "V_ORACLE 是已知像素重定时参照，不能替代实际 G_W 生成响应；本轮任何结果都不改写历史 CONTROL_FAILED。",
        "",
        "| sample | historical C | baseline | B | C_oracle | O_oracle | own offset | damage positive |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in analysis["per_record"]:
        lines.append(
            f"| {row['sample_id']} | {row['historical_c_pass']} | {row['baseline']} | "
            f"{row['checks']['B']['passes']} | {row['checks']['C_oracle']['passes']} | "
            f"{row['checks']['O_oracle']['passes']} | {row['own']['offset_agreement']} | "
            f"{row['damage']['both_positive']} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "即使 ORACLE_CONTROL_SUPPORTED，也只说明当前 22 条生成画面的已知重定时参照和控制判据可用；不得称父 G_W 控制通过、replacement effect 成立、已获得训练授权或跨模型泛化。",
            "",
        ]
    )
    return "\n".join(lines)


def _write_blocked(paths: config.RunPaths, reason: str, completed_cells: int = 0) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    paths.result.write_text(
        "# Wav2Lip ROI retiming oracle 2026-09-06\n\n"
        f"终态：[BLOCKED]。已完成新评分：{completed_cells}/{config.EXPECTED_SCORE_CELL_COUNT}。\n\n"
        f"原因：{reason}\n",
        encoding="utf-8",
    )
    write_self_hashed_json(
        paths.final,
        {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "protocol_revision": config.PROTOCOL_REVISION,
            "status": "blocked",
            "engineering_decision": "BLOCKED",
            "diagnostic_decision": None,
            "historical_scientific_decision": "CONTROL_FAILED",
            "parent_own_audio_retested": False,
            "oracle_own_audio_tested": completed_cells > 0,
            "bridge_executed": False,
            "training_authorized": False,
            "reference_conditioned_audio_head_spec_eligible": False,
            "generalization_established": False,
            "new_generated_videos": 0,
            "new_video_stream_count": 0,
            "new_media_count": 0,
            "new_score_cells": completed_cells,
            "blocked_reason": reason,
            "result_sha256": file_sha256(paths.result),
        },
    )


def run_oracle(run_id: str, device: str = "cpu", threads: int = config.TORCH_THREADS) -> dict[str, Any]:
    if device != "cpu":
        raise OracleError("this diagnostic is frozen to CPU")
    paths = config.RunPaths(config.run_root_for(run_id))
    if paths.root.exists() and any(paths.root.iterdir()):
        raise OracleError(f"run root already contains artifacts; use a new run-id: {paths.root}")
    paths.root.mkdir(parents=True, exist_ok=True)
    try:
        parent = _fixed_parent()
        audit, rows = _audit_parent(parent)
        audit_sha = write_self_hashed_json(paths.input_audit, audit)
        protocol = _protocol(parent, audit, rows, run_id, device, threads)
        protocol_sha = write_self_hashed_json(paths.protocol, protocol)
        frame_rows: list[dict[str, Any]] = []
        media_rows: list[dict[str, Any]] = []
        score_rows: list[dict[str, Any]] = []
        scorer = SyncNetScorer(config.SYNCNET_MODEL, device=device, batch_size=config.BATCH_SIZE, threads=threads)
        for index, row in enumerate(rows, 1):
            sample_id = str(row["sample_id"])
            frames = row["_frames"]
            indices = row["indices"]
            video_dir = paths.media / "streams"
            id_video = video_dir / f"{sample_id}__V_ID.mkv"
            oracle_video = video_dir / f"{sample_id}__V_ORACLE.mkv"
            id_frames = frames[np.asarray(indices["q_id"], dtype=np.int64)]
            oracle_frames = frames[np.asarray(indices["q_oracle"], dtype=np.int64)]
            id_evidence = encode_video(id_frames, id_video)
            oracle_evidence = encode_video(oracle_frames, oracle_video)
            frame_rows.extend(
                [
                    {"sample_id": sample_id, "source_group": row["source_group"], "video_arm": "V_ID", "source_stream": row["parent_stream"], "q": indices["q_id"], "evidence": id_evidence, "pixel_identity_verified": True},
                    {"sample_id": sample_id, "source_group": row["source_group"], "video_arm": "V_ORACLE", "source_stream": row["parent_stream"], "q": indices["q_oracle"], "evidence": oracle_evidence, "pixel_identity_verified": True},
                ]
            )
            audio_meta = materialize_audio(
                Path(row["natural_audio"]["path"]),
                row["_warped_pcm"],
                paths.root / "audio",
                artifact_stem=sample_id,
            )
            muxes: dict[tuple[str, str], dict[str, Any]] = {}
            for video, video_path in (("V_ID", id_video), ("V_ORACLE", oracle_video)):
                for audio, audio_info in audio_meta.items():
                    mux_path = paths.media / "cells" / f"{config.cell_key(sample_id, video, audio)}.mkv"
                    mux_evidence = mux_audio(
                        video_path,
                        Path(str(audio_info["path"])),
                        mux_path,
                        row["_natural_pcm"] if audio == "N" else row["_warped_pcm"],
                    )
                    sidecar = dict(mux_evidence)
                    sidecar.update({"sample_id": sample_id, "video_arm": video, "audio_arm": audio, "protocol_sha256": protocol_sha})
                    write_self_hashed_json(mux_path.with_suffix(".json"), sidecar)
                    muxes[(video, audio)] = {**mux_evidence, "audio": audio_info}
            media_rows.append(
                {
                    "sample_id": sample_id,
                    "source_group": row["source_group"],
                    "streams": {
                        "V_ID": {"output": str(id_video.resolve()), "output_sha256": file_sha256(id_video), "frame_count": row["frame_count"]},
                        "V_ORACLE": {"output": str(oracle_video.resolve()), "output_sha256": file_sha256(oracle_video), "frame_count": row["frame_count"]},
                    },
                    "cells": {
                        f"{video}__{audio}": {**{key: value for key, value in evidence.items() if key != "video"}, "audio": evidence["audio"]}
                        for (video, audio), evidence in muxes.items()
                    },
                }
            )
            for video, audio in config.CELL_SPECS:
                item = muxes[(video, audio)]
                cell_dir = paths.scores / "cells" / config.cell_key(sample_id, video, audio)
                print(f"CELL_START {index}/{config.EXPECTED_RECORD_COUNT} {sample_id} {video}/{audio}", flush=True)
                worker = scorer.score(Path(item["output"]), Path(item["audio"]["path"]), cell_dir, item["output_sha256"], item["audio_pcm_sha256"])
                worker.update({"protocol_id": config.PROTOCOL_ID, "protocol_sha256": protocol_sha, "sample_id": sample_id, "source_group": row["source_group"], "video_arm": video, "audio_arm": audio, "new_forward": True})
                worker_path = cell_dir / "worker.json"
                worker_sha = write_self_hashed_json(worker_path, worker)
                score_rows.append(
                    {
                        "schema_version": 1,
                        "stage_id": "fresh_syncnet_forward",
                        "protocol_id": config.PROTOCOL_ID,
                        "protocol_sha256": protocol_sha,
                        "sample_id": sample_id,
                        "source_group": row["source_group"],
                        "video_arm": video,
                        "audio_arm": audio,
                        "media": item["output"],
                        "media_sha256": item["output_sha256"],
                        "media_pcm_sha256": item["audio_pcm_sha256"],
                        "audio": item["audio"]["path"],
                        "audio_container_sha256": item["audio"]["container_sha256"],
                        "audio_pcm_sha256": item["audio"]["decoded_pcm_sha256"],
                        "cell_dir": str(cell_dir.resolve()),
                        "worker_result": str(worker_path.resolve()),
                        "worker_result_sha256": worker_sha,
                        "visual": worker["visual"],
                        "visual_sha256": worker["visual_sha256"],
                        "audio_embedding": worker["audio_embedding"],
                        "audio_embedding_sha256": worker["audio_embedding_sha256"],
                        "matrix": worker["matrix"],
                        "matrix_sha256": worker["matrix_sha256"],
                        "matrix_shape": worker["matrix_shape"],
                    }
                )
                print(f"CELL_DONE {index}/{config.EXPECTED_RECORD_COUNT} {sample_id} {video}/{audio}", flush=True)
        frame_sha = write_self_hashed_json(paths.frame_manifest, {"schema_version": 1, "stage_id": "frame_materialization", "protocol_id": config.PROTOCOL_ID, "protocol_sha256": protocol_sha, "status": "complete", "record_count": len(rows), "stream_count": len(frame_rows), "rows": frame_rows})
        audio_sha = write_self_hashed_json(paths.audio_manifest, {"schema_version": 1, "stage_id": "audio_binding", "protocol_id": config.PROTOCOL_ID, "protocol_sha256": protocol_sha, "status": "complete", "record_count": len(rows), "policy": "N copied byte-identical; W exact registered PCM reconstruction", "rows": [{"sample_id": row["sample_id"], "natural_audio": row["natural_audio"], "audio_dir": str((paths.root / "audio").resolve())} for row in rows]})
        media_sha = write_self_hashed_json(paths.media_manifest, {"schema_version": 1, "stage_id": "media", "protocol_id": config.PROTOCOL_ID, "protocol_sha256": protocol_sha, "status": "complete", "record_count": len(media_rows), "stream_count": len(frame_rows), "cell_count": len(score_rows), "expected_stream_count": config.EXPECTED_STREAM_COUNT, "expected_cell_count": config.EXPECTED_MEDIA_COUNT, "rows": media_rows})
        score_manifest_sha = write_self_hashed_json(paths.score_manifest, {"schema_version": 1, "stage_id": "score", "protocol_id": config.PROTOCOL_ID, "protocol_sha256": protocol_sha, "media_manifest_sha256": media_sha, "status": "complete", "record_count": len(rows), "cell_count": len(score_rows), "expected_cell_count": config.EXPECTED_SCORE_CELL_COUNT, "scores": score_rows})
        analysis = analyze(parent, protocol, score_rows)
        analysis.update({"input_audit_sha256": audit_sha, "protocol_sha256": protocol_sha, "frame_manifest_sha256": frame_sha, "audio_manifest_sha256": audio_sha, "media_manifest_sha256": media_sha, "score_manifest_sha256": score_manifest_sha})
        analysis_sha = write_self_hashed_json(paths.analysis, analysis)
        paths.result.write_text(_result_markdown(analysis), encoding="utf-8")
        final = {
            "schema_version": 1,
            "protocol_id": config.PROTOCOL_ID,
            "protocol_revision": config.PROTOCOL_REVISION,
            "status": "complete",
            "engineering_decision": "GO",
            "diagnostic_decision": analysis["decision"],
            "historical_scientific_decision": "CONTROL_FAILED",
            "parent_own_audio_retested": False,
            "oracle_own_audio_tested": True,
            "bridge_executed": False,
            "training_authorized": False,
            "reference_conditioned_audio_head_spec_eligible": False,
            "generalization_established": False,
            "record_count": len(rows),
            "expected_record_count": config.EXPECTED_RECORD_COUNT,
            "new_generated_videos": 0,
            "new_video_stream_count": len(frame_rows),
            "new_media_count": len(score_rows),
            "new_score_cells": len(score_rows),
            "protocol_sha256": protocol_sha,
            "input_audit_sha256": audit_sha,
            "analysis_sha256": analysis_sha,
            "result_sha256": file_sha256(paths.result),
            "media_manifest_sha256": media_sha,
            "score_manifest_sha256": score_manifest_sha,
        }
        write_self_hashed_json(paths.final, final)
        return final
    except Exception as exc:
        _write_blocked(paths, str(exc))
        raise


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the Wav2Lip ROI known-pixel-retiming oracle")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu",))
    parser.add_argument("--threads", type=int, default=config.TORCH_THREADS)
    args = parser.parse_args(argv)
    try:
        result = run_oracle(args.run_id, args.device, args.threads)
    except Exception as exc:  # noqa: BLE001 - blocked final is written for inspection
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
