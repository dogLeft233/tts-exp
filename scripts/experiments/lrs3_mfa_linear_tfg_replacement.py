"""Evaluate MFA-linear as a TFG driver with and without natural-audio replacement.

The cohort and runtime bindings are inherited from the frozen LRS3 bridge
confirmation cohort.  MFA-linear is used to generate the video once; the two
scored media cells differ only in the final muxed audio stream:

* ``V_MFA_LINEAR/A_MFA_LINEAR``: native MFA-linear track;
* ``V_MFA_LINEAR/A_N``: the same generated frames with the natural track.

This is deliberately not a LOCAL_SWAP experiment.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation import (
    config as base_config,
)
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.common import (
    canonical_json_sha256,
    file_sha256,
    verify_self_hashed_json,
    write_self_hashed_json,
)
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.render import (
    shared_face_geometry,
)
from scripts.experiments.lrs3_natural_to_tts_bridge_confirmation.scoring import (
    score_syncnet,
    strict_mux,
)

REPO = Path(__file__).resolve().parents[2]
SOURCE_ROOT = REPO / "runs/lrs3_natural_to_tts_bridge_confirmation_20260904"
RUN_ROOT = REPO / "runs/lrs3_mfa_linear_tfg_replacement_20260913"
COHORT_PATH = SOURCE_ROOT / "00_protocol/cohort.json"
SOURCE_SCORES = SOURCE_ROOT / "03_scores/scores"
EXPECTED_RECORD_COUNT = 22
PROTOCOL_ID = "lrs3_mfa_linear_tfg_replacement_20260913"
VIDEO_ARM = "MFA_LINEAR"
AUDIO_ARMS = ("MFA_LINEAR", "N")
BOOTSTRAP_DRAWS = 10_000
BOOTSTRAP_SEED = 20260913


def _run(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        result = subprocess.run(command, cwd=str(cwd), stdout=handle, stderr=subprocess.STDOUT, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {log_path}")


def _cohort() -> dict[str, Any]:
    payload = verify_self_hashed_json(COHORT_PATH)
    records = payload.get("records")
    if payload.get("status") != "complete" or not isinstance(records, list) or len(records) != EXPECTED_RECORD_COUNT:
        raise ValueError("frozen bridge cohort is incomplete")
    seen: set[str] = set()
    for record in records:
        sample_id = str(record.get("sample_id", ""))
        if not re.fullmatch(r"lrs3_[A-Za-z0-9]+_\d{5}", sample_id) or sample_id in seen:
            raise ValueError(f"invalid or duplicate sample id: {sample_id}")
        seen.add(sample_id)
        for key in ("natural_audio", "mfa_linear_audio", "face_video"):
            path = Path(str(record[key]["path"]))
            if not path.is_file():
                raise FileNotFoundError(path)
    return payload


def _protocol(cohort: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "frozen",
        "source_cohort": str(COHORT_PATH),
        "source_cohort_sha256": file_sha256(COHORT_PATH),
        "record_count": EXPECTED_RECORD_COUNT,
        "video_arm": VIDEO_ARM,
        "audio_arms": list(AUDIO_ARMS),
        "generation": {
            "driver": "MFA-linear",
            "wav2lip_checkpoint": str(base_config.WAV2LIP_CHECKPOINT),
            "wav2lip_checkpoint_sha256": file_sha256(base_config.WAV2LIP_CHECKPOINT),
            "wav2lip_python": str(base_config.WAV2LIP_PYTHON),
            "wav2lip_python_sha256": file_sha256(base_config.WAV2LIP_PYTHON),
            "registered_confirmation_wav2lip_python_sha256": base_config.WAV2LIP_PYTHON_SHA256,
            "runtime_binding_note": "The registered confirmation Python executable is no longer present; current executable hash is recorded explicitly.",
            "face_geometry": "constant_full_frame_fallback_from_each_source_video",
            "face_det_batch_size": 16,
            "wav2lip_batch_size": 16,
            "nosmooth": True,
            "workers": 1,
        },
        "replacement": {
            "native": "MFA-linear audio remains attached to MFA-linear-generated frames",
            "natural": "only the final audio stream is replaced with the paired natural PCM",
            "video_stream_copy": True,
            "audio_codec": "pcm_s16le",
            "sample_rate": 16000,
            "channels": 1,
        },
        "forbidden_condition": "LOCAL_SWAP is not part of this experiment",
        "selection": "fixed 22-record confirmation cohort; no score-based selection",
    }


def _render_with_current_runtime(
    face: Path,
    audio: Path,
    output: Path,
    geometry: dict[str, Any],
    work_dir: Path,
    log_path: Path,
) -> dict[str, Any]:
    """Render with the available runtime after recording its binding drift."""
    binding = {
        "wav2lip_python": file_sha256(base_config.WAV2LIP_PYTHON),
        "wav2lip_checkpoint": file_sha256(base_config.WAV2LIP_CHECKPOINT),
        "wav2lip_inference": file_sha256(base_config.WAV2LIP_ROOT / "inference.py"),
    }
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "temp").mkdir(exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.partial{output.suffix}")
    top, bottom, left, right = (str(int(value)) for value in geometry["box"])
    command = [
        str(base_config.WAV2LIP_PYTHON),
        str(base_config.WAV2LIP_ROOT / "inference.py"),
        "--checkpoint_path", str(base_config.WAV2LIP_CHECKPOINT),
        "--face", str(face),
        "--audio", str(audio),
        "--outfile", str(temporary),
        "--box", top, bottom, left, right,
        "--face_det_batch_size", "16",
        "--wav2lip_batch_size", "16",
        "--nosmooth",
    ]
    _run(command, work_dir, log_path)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        raise RuntimeError(f"Wav2Lip render produced no output: {output}")
    temporary.replace(output)
    return {
        "output": str(output),
        "output_sha256": file_sha256(output),
        "face_sha256": file_sha256(face),
        "audio_sha256": file_sha256(audio),
        "geometry_sha256": canonical_json_sha256(geometry),
        "checkpoint_sha256": binding["wav2lip_checkpoint"],
        "command": command,
        "command_sha256": canonical_json_sha256(command),
        "runtime_bindings": binding,
        "registered_wav2lip_python_sha256": base_config.WAV2LIP_PYTHON_SHA256,
        "runtime_binding_drift": binding["wav2lip_python"] != base_config.WAV2LIP_PYTHON_SHA256,
        "log": str(log_path),
    }


def _render_one(record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    sample_id = str(record["sample_id"])
    face = Path(str(record["face_video"]["path"]))
    audio = Path(str(record["mfa_linear_audio"]["path"]))
    output = output_dir / "raw" / f"{sample_id}.mp4"
    sidecar = output.with_suffix(".json")
    geometry = shared_face_geometry(face)
    if output.is_file() and sidecar.is_file():
        prior = verify_self_hashed_json(sidecar)
        if (
            prior.get("protocol_id") != PROTOCOL_ID
            or prior.get("sample_id") != sample_id
            or prior.get("face_sha256") != file_sha256(face)
            or prior.get("audio_sha256") != file_sha256(audio)
            or prior.get("output_sha256") != file_sha256(output)
        ):
            raise ValueError(f"existing MFA render identity changed: {sample_id}")
        return prior
    if output.exists() or sidecar.exists():
        raise ValueError(f"partial MFA render cannot be resumed: {sample_id}")
    rendered = _render_with_current_runtime(
        face,
        audio,
        output,
        geometry,
        output_dir / "work" / sample_id,
        output_dir / "logs" / f"{sample_id}.log",
    )
    row = {
        "schema_version": 1,
        "stage_id": "01_videos",
        "protocol_id": PROTOCOL_ID,
        "sample_id": sample_id,
        "source_group": str(record["source_group"]),
        "face": str(face),
        "face_sha256": file_sha256(face),
        "mfa_linear_audio": str(audio),
        "mfa_linear_audio_sha256": file_sha256(audio),
        "geometry": geometry,
        **rendered,
    }
    write_self_hashed_json(sidecar, row)
    return row


def render_stage(cohort: dict[str, Any]) -> dict[str, Any]:
    output_dir = RUN_ROOT / "01_videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, record in enumerate(cohort["records"], 1):
        row = _render_one(record, output_dir)
        rows.append(row)
        print(f"VIDEO {index}/{EXPECTED_RECORD_COUNT} {record['sample_id']}", flush=True)
    result = {
        "schema_version": 1,
        "stage_id": "01_videos",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "record_count": len(rows),
        "rows": rows,
    }
    write_self_hashed_json(output_dir / "manifest.json", result)
    return result


def _score_sidecar(path: Path) -> dict[str, Any]:
    return verify_self_hashed_json(path)


def score_stage(cohort: dict[str, Any], videos: dict[str, Any]) -> dict[str, Any]:
    output_dir = RUN_ROOT / "02_scores"
    output_dir.mkdir(parents=True, exist_ok=True)
    video_rows = {str(row["sample_id"]): row for row in videos["rows"]}
    score_rows: list[dict[str, Any]] = []
    mux_rows: list[dict[str, Any]] = []
    for index, record in enumerate(cohort["records"], 1):
        sample_id = str(record["sample_id"])
        video_row = video_rows[sample_id]
        video = Path(str(video_row["output"]))
        audio_paths = {
            "MFA_LINEAR": Path(str(record["mfa_linear_audio"]["path"])),
            "N": Path(str(record["natural_audio"]["path"])),
        }
        for audio_arm in AUDIO_ARMS:
            cell = f"V_{VIDEO_ARM}/A_{audio_arm}"
            cell_id = f"{sample_id}__{VIDEO_ARM}_{audio_arm}"
            mux_path = output_dir / "mux" / VIDEO_ARM / audio_arm / f"{sample_id}.mkv"
            mux_sidecar = mux_path.with_suffix(".json")
            if mux_path.is_file() and mux_sidecar.is_file():
                mux = verify_self_hashed_json(mux_sidecar)
                if (
                    mux.get("protocol_id") != PROTOCOL_ID
                    or mux.get("cell") != cell
                    or mux.get("video_source_sha256") != file_sha256(video)
                    or mux.get("audio_source_sha256") != file_sha256(audio_paths[audio_arm])
                    or mux.get("sha256") != file_sha256(mux_path)
                ):
                    raise ValueError(f"existing mux identity changed: {cell_id}")
            elif mux_path.exists() or mux_sidecar.exists():
                raise ValueError(f"partial mux cannot be resumed: {cell_id}")
            else:
                mux = strict_mux(video, audio_paths[audio_arm], mux_path, output_dir / "logs" / "mux" / f"{cell_id}.log")
                mux = {"schema_version": 1, "stage_id": "02_scores", "protocol_id": PROTOCOL_ID, "sample_id": sample_id, "cell": cell, **mux}
                write_self_hashed_json(mux_sidecar, mux)
            mux_rows.append(mux)

            score_path = output_dir / "scores" / VIDEO_ARM / audio_arm / f"{sample_id}.json"
            if score_path.is_file():
                score = _score_sidecar(score_path)
                if score.get("protocol_id") != PROTOCOL_ID or score.get("cell") != cell or score.get("media_sha256") != file_sha256(mux_path):
                    raise ValueError(f"existing score identity changed: {cell_id}")
            elif score_path.exists():
                raise ValueError(f"partial score cannot be resumed: {cell_id}")
            else:
                score = score_syncnet(mux_path, output_dir, cell_id)
                score = {"schema_version": 1, "stage_id": "02_scores", "protocol_id": PROTOCOL_ID, "sample_id": sample_id, "source_group": str(record["source_group"]), "cell": cell, **score}
                write_self_hashed_json(score_path, score)
            score_rows.append(score)
            print(f"SCORE {index}/{EXPECTED_RECORD_COUNT} {cell}", flush=True)
    result = {
        "schema_version": 1,
        "stage_id": "02_scores",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "record_count": EXPECTED_RECORD_COUNT,
        "cell_count": len(score_rows),
        "cells": [f"V_{VIDEO_ARM}/A_{arm}" for arm in AUDIO_ARMS],
        "muxes": mux_rows,
        "scores": score_rows,
    }
    write_self_hashed_json(output_dir / "manifest.json", result)
    return result


def _existing_score(sample_id: str, video_arm: str, audio_arm: str) -> dict[str, Any]:
    path = SOURCE_SCORES / video_arm / audio_arm / f"{sample_id}.json"
    return verify_self_hashed_json(path)


def _bootstrap(values: np.ndarray, seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(BOOTSTRAP_DRAWS, len(values)))
    means = values[indices].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def _summary(label: str, scores: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    delta_c = np.asarray([float(row["sync_c"]) - float(ref["sync_c"]) for row, ref in zip(scores, baseline)], dtype=np.float64)
    delta_d = np.asarray([float(ref["sync_d"]) - float(row["sync_d"]) for row, ref in zip(scores, baseline)], dtype=np.float64)
    offset_change = np.asarray([int(row["av_offset"]) != int(ref["av_offset"]) for row, ref in zip(scores, baseline)], dtype=bool)
    return {
        "label": label,
        "record_count": len(scores),
        "delta_sync_c": {"mean": float(delta_c.mean()), "ci95": list(_bootstrap(delta_c)), "positive_count": int((delta_c > 0).sum())},
        "delta_sync_d_improvement": {"mean": float(delta_d.mean()), "ci95": list(_bootstrap(delta_d, BOOTSTRAP_SEED + 1)), "positive_count": int((delta_d > 0).sum())},
        "offset_change_count": int(offset_change.sum()),
        "per_record_delta_c": [float(value) for value in delta_c],
        "per_record_delta_d_improvement": [float(value) for value in delta_d],
    }


def analysis_stage(cohort: dict[str, Any], score_manifest: dict[str, Any]) -> dict[str, Any]:
    score_by_key = {(str(row["sample_id"]), str(row["cell"]).split("/")[-1].removeprefix("A_")): row for row in score_manifest["scores"]}
    records = list(cohort["records"])
    baseline = [_existing_score(str(record["sample_id"]), "N", "N") for record in records]
    b075_native = [_existing_score(str(record["sample_id"]), "BRIDGE_075", "BRIDGE_075") for record in records]
    b075_replaced = [_existing_score(str(record["sample_id"]), "BRIDGE_075", "N") for record in records]
    mfa_native = [score_by_key[(str(record["sample_id"]), "MFA_LINEAR")] for record in records]
    mfa_replaced = [score_by_key[(str(record["sample_id"]), "N")] for record in records]
    summaries = [
        _summary("B075 native (existing)", b075_native, baseline),
        _summary("B075 with natural replacement (existing)", b075_replaced, baseline),
        _summary("MFA-linear native", mfa_native, baseline),
        _summary("MFA-linear with natural replacement", mfa_replaced, baseline),
    ]
    result = {
        "schema_version": 1,
        "stage_id": "03_analysis",
        "protocol_id": PROTOCOL_ID,
        "status": "complete",
        "record_count": len(records),
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED, "unit": "paired_record", "confidence": 0.95},
        "summaries": summaries,
        "source_b075_run": str(SOURCE_ROOT),
        "mfa_score_manifest_sha256": file_sha256(RUN_ROOT / "02_scores/manifest.json"),
    }
    write_self_hashed_json(RUN_ROOT / "03_analysis.json", result)
    _write_report(result, records)
    return result


def _write_report(result: dict[str, Any], records: list[dict[str, Any]]) -> None:
    lines = [
        "# LRS3 MFA-linear TFG generation / natural replacement",
        "",
        "固定确认轮的22条 LRS3 动态视频，使用同一张 source video、同一 Wav2Lip checkpoint 和 SyncNet V2。MFA-linear 只用于生成视频；随后分别保留 MFA-linear 音轨，或仅将最终音轨换回配对 natural。没有 LOCAL_SWAP 条件。",
        "",
        "注意：确认轮登记的 Wav2Lip Python 可执行文件 hash 已不再存在，本 run 使用当前同一路径下可用的 Python 并显式记录 hash；checkpoint 与 inference.py 保持登记版本。该运行时差异是与历史 B075 分数合并解释时的一个复现边界。",
        "",
        "## 条件",
        "",
        "|条件|视频帧来源|最终音轨|状态|",
        "|---|---|---|---|",
        "|natural baseline|Wav2Lip + natural|natural|沿用确认轮已核验分数|",
        "|B075 native|Wav2Lip + B075|B075|沿用确认轮已核验分数|",
        "|B075 natural replacement|Wav2Lip + B075|natural|沿用确认轮已核验分数|",
        "|MFA-linear native|Wav2Lip + MFA-linear|MFA-linear|本 run 新测|",
        "|MFA-linear natural replacement|Wav2Lip + MFA-linear|natural|本 run 新测|",
        "",
        "## 相对 natural baseline 的结果",
        "",
        "|条件|ΔSync-C 均值 [95% CI]|C 增加条数|ΔSync-D 改善均值 [95% CI]|D 改善条数|offset变化|",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for summary in result["summaries"]:
        c = summary["delta_sync_c"]
        d = summary["delta_sync_d_improvement"]
        lines.append(
            f"|{summary['label']}|{c['mean']:+.3f} [{c['ci95'][0]:+.3f},{c['ci95'][1]:+.3f}]|{c['positive_count']}/{summary['record_count']}|"
            f"{d['mean']:+.3f} [{d['ci95'][0]:+.3f},{d['ci95'][1]:+.3f}]|{d['positive_count']}/{summary['record_count']}|{summary['offset_change_count']}/{summary['record_count']}|"
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "B075 的 native/replacement 两条来自既有确认轮，并非 LOCAL_SWAP。MFA-linear 两条只回答同一 TFG 生成链中：保留驱动音轨，或把同一生成视频的最终音轨替换为 natural，会得到什么 SyncNet 变化。它不把 replacement 后的分数解释成嘴型运动被因果改变，也不代表 held-out 泛化。",
            "",
            f"样本数：{len(records)}；bootstrap：{BOOTSTRAP_DRAWS} 次、seed={BOOTSTRAP_SEED}。详细 JSON、视频和评分日志见本 run。",
            "",
        ]
    )
    (RUN_ROOT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("protocol", "render", "score", "analysis", "all"), default="all")
    args = parser.parse_args()
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    cohort = _cohort()
    protocol_path = RUN_ROOT / "protocol.json"
    expected_protocol = _protocol(cohort)
    if not protocol_path.is_file():
        write_self_hashed_json(protocol_path, expected_protocol)
    else:
        prior_protocol = verify_self_hashed_json(protocol_path)
        prior_body = dict(prior_protocol)
        prior_body.pop("artifact_sha256", None)
        if prior_body != expected_protocol:
            write_self_hashed_json(protocol_path, expected_protocol)
    if args.stage == "protocol":
        return 0
    videos = verify_self_hashed_json(RUN_ROOT / "01_videos/manifest.json") if args.stage in {"score", "analysis"} else None
    if args.stage in {"render", "all"}:
        videos = render_stage(cohort)
    if args.stage == "render":
        return 0
    scores = verify_self_hashed_json(RUN_ROOT / "02_scores/manifest.json") if args.stage == "analysis" else None
    if args.stage in {"score", "all"}:
        if videos is None:
            videos = verify_self_hashed_json(RUN_ROOT / "01_videos/manifest.json")
        scores = score_stage(cohort, videos)
    if args.stage == "score":
        return 0
    if args.stage in {"analysis", "all"}:
        if scores is None:
            scores = verify_self_hashed_json(RUN_ROOT / "02_scores/manifest.json")
        analysis_stage(cohort, scores)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
