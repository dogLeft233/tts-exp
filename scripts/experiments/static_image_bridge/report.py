from __future__ import annotations

import html
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import ProtocolError, file_sha256, verify_self_hashed_json


def _mux_preview(paths: config.RunPaths, sample_id: str, video_arm: str, audio_arm: str, video_path: Path, audio_path: Path) -> Path:
    output = paths.playback / f"{sample_id}__{video_arm}_{audio_arm}.mp4"
    if output.is_file():
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    # Keep the final container extension on the temporary path as ffmpeg
    # infers the muxer from the output suffix.
    temporary = output.with_name(f".{output.stem}.partial.mp4")
    command = [
        str(config.FFMPEG), "-y", "-v", "error", "-i", str(video_path), "-i", str(audio_path),
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "64k", "-shortest", str(temporary),
    ]
    result = subprocess.run(command, capture_output=True, check=False)
    if result.returncode != 0:
        temporary.unlink(missing_ok=True)
        raise ProtocolError(f"playback mux failed: {result.stderr.decode(errors='replace')[-1000:]}")
    temporary.replace(output)
    return output


def _number(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def build_playback(paths: config.RunPaths, inputs: Mapping[str, Any], audio_manifest: Mapping[str, Any], video_manifest: Mapping[str, Any], scores: Mapping[tuple[str, str, str], Mapping[str, Any]]) -> list[dict[str, Any]]:
    audio_by_id = {str(row["sample_id"]): row for row in audio_manifest.get("rows", [])}
    video_by_key = {(str(row["sample_id"]), str(row["video_arm"])): row for row in video_manifest.get("rows", [])}
    cells = (("N", "N"), ("RT", "N"), ("B", "N"), ("N", "S"), ("S", "N"), ("S", "S"))
    entries = []
    for item in inputs["records"]:
        sid = str(item["sample_id"])
        audio_row = audio_by_id[sid]
        for video_arm, audio_arm in cells:
            video_row = video_by_key.get((sid, video_arm))
            if video_row is None:
                continue
            audio_path = Path(str(audio_row["arms"][audio_arm]["path"]))
            preview = _mux_preview(paths, sid, video_arm, audio_arm, Path(str(video_row["output"])), audio_path)
            score = scores.get((sid, video_arm, audio_arm))
            entries.append({"sample_id": sid, "video_arm": video_arm, "audio_arm": audio_arm, "path": str(preview.relative_to(paths.playback)), "score": score})
    return entries


def write_report(paths: config.RunPaths) -> dict[str, Any]:
    verify_self_hashed_json(paths.protocol)
    inputs = verify_self_hashed_json(paths.inputs)
    audio = verify_self_hashed_json(paths.audio_manifest)
    videos = verify_self_hashed_json(paths.video_manifest)
    scores = {(str(row["sample_id"]), str(row["video_arm"]), str(row["audio_arm"])): row for row in verify_self_hashed_json(paths.scores_manifest).get("rows", [])}
    final = verify_self_hashed_json(paths.final)
    analysis = verify_self_hashed_json(paths.analysis) if paths.analysis.is_file() else None
    playback = build_playback(paths, inputs, audio, videos, scores)
    lines = [
        "# 静态图片 Natural-to-TTS bridge 实验报告",
        "",
        f"- 协议：`{config.PROTOCOL_ID}` / `{config.PROTOCOL_REVISION}`",
        f"- 样本：确认轮冻结的 {len(inputs['records'])} 条、{len({row['source_group'] for row in inputs['records']})} 个 source group；`seen_fit=true`，不代表新来源泛化确认。",
        "- 输入：每条只使用源视频第 0 帧 PNG；所有 arm 共用同一 PNG、固定生成框和固定评分 crop。",
        "- 评分：官方 SyncNet V2 模型与 MFCC frontend；先在共同有效窗口 W 上求距离曲线均值，再取 D/C/k。",
        "",
        "## 历史背景",
        "",
        "| 队列 | C 提升 | mean ΔC | C 95% CI |",
        "|---|---:|---:|---|",
        "| discovery MAG_075（23 条） | 18/23 (78.3%) | +0.078 | [+0.016,+0.136] |",
        "| confirmation BRIDGE_075（22 条） | 9/22 (40.9%) | +0.031 | [−0.034,+0.105] |",
        "",
        "历史动态视频结果与本实验不做因果差分：静态实验同时固定了输入视觉时序、生成 ROI、评分 crop 和支持范围。",
        "",
        "## 实验臂与门槛",
        "",
        "| arm | 音频 | 用途 |",
        "|---|---|---|",
        "| N | natural | 静态自然基线 |",
        "| N_REPEAT | natural（独立生成） | 重复性控制 |",
        "| RT | natural 的 alpha=0 STFT 往返 | 重建控制 |",
        "| B | 历史 BRIDGE_075 | 唯一 bridge 候选 |",
        "| S | 历史 LOCAL_SWAP | 局部时序诊断 |",
        "",
        "A 阶段必须先通过独立重复性和 +200 ms 延迟控制；否则 B 不运行，也不能把结果写成 bridge 阴性。B 的收益必须同时胜过 N 和 RT，C 均值大于 0.05 且 95% CI 下界大于 0。",
        "",
        "## 结果",
        "",
        f"- measurement：`{final.get('measurement')}`",
        f"- bridge_gain：`{final.get('bridge_gain')}`",
        f"- swap_transfer：`{final.get('swap_transfer')}`",
        f"- human_review：`{final.get('human_review')}`",
        f"- training_authorized：`{final.get('training_authorized')}`",
    ]
    if analysis:
        lines.extend(["", "### Bridge 对比摘要", "", "| 比较 | C 均值 | C 95% CI | D 均值 | D 95% CI |", "|---|---:|---|---:|---|"])
        for name, bundle in analysis["bridge_gain"]["comparisons"].items():
            lines.append(f"| {name} | {_number(bundle['C']['mean'])} | [{_number(bundle['C']['ci95'][0])}, {_number(bundle['C']['ci95'][1])}] | {_number(bundle['D']['mean'])} | [{_number(bundle['D']['ci95'][0])}, {_number(bundle['D']['ci95'][1])}] |")
        lines.extend(["", "### 结论边界", "", "静态条件下的 score gain（如果成立）只说明固定 PNG + 固定 crop 协议中的测量结果；它不证明嘴型泄漏、不证明动态视频中的历史效应已修复、不授权训练 audio head，也不构成新来源泛化。"])
    lines.extend(["", "## 可离线播放", "", "以下文件是有损播放副本，不参与评分；评分使用的是固定 crop 下的独立矩阵。", "", "| 样本 | 视频 arm | 音频 arm | 播放 | SyncNet cell |", "|---|---|---|---|---|"])
    for entry in playback:
        score = entry.get("score") or {}
        lines.append(f"| {entry['sample_id']} | {entry['video_arm']} | {entry['audio_arm']} | [{entry['path']}]({entry['path']}) | C={_number(score.get('C', 'n/a')) if isinstance(score, Mapping) else 'n/a'} |")
    paths.report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    html_rows = []
    for entry in playback:
        html_rows.append(f"<tr><td>{html.escape(entry['sample_id'])}</td><td>{entry['video_arm']}</td><td>{entry['audio_arm']}</td><td><video controls preload='metadata' src='{html.escape(entry['path'])}'></video></td></tr>")
    paths.playback.mkdir(parents=True, exist_ok=True)
    (paths.playback / "index.html").write_text("<!doctype html><meta charset='utf-8'><title>static bridge playback</title><style>video{width:240px}</style><table><tr><th>sample</th><th>video</th><th>audio</th><th>preview</th></tr>" + "".join(html_rows) + "</table>\n", encoding="utf-8")
    return {"report": str(paths.report), "playback_index": str((paths.playback / "index.html").resolve()), "preview_count": len(playback), "report_sha256": file_sha256(paths.report)}
