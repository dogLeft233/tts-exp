"""Plain-language report and final-state assembly for the attribution run."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from . import config
from .common import (
    ProtocolError,
    file_sha256,
    read_json,
    write_self_hashed_json,
)


def _status(path: Path, default: str = "NOT_RUN") -> str:
    if not path.is_file():
        return default
    try:
        return str(read_json(path).get("status", default))
    except ProtocolError:
        return "INVALID"


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return str(value)


def _fmt_precise(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        return f"{float(value):.3e}"
    except (TypeError, ValueError):
        return str(value)


def _interval(summary: Mapping[str, Any], key: str) -> str:
    value = summary.get(key)
    if not isinstance(value, list) or len(value) != 2:
        return "NA"
    return f"[{_fmt(value[0])}, {_fmt(value[1])}]"


def _primary_table(analysis: Mapping[str, Any]) -> str:
    lines = ["| 对比 | 定义 | 均值 | 95%描述区间 | 99.166667%校正区间 | 状态 |", "|---|---|---:|---|---|---|"]
    labels = {
        "A_N_DENOISE_E": "固定 V_N：谱抑制后评分音频 − A0",
        "A_T_NOISE_E_HARM": "固定 V_T：A0 − 加噪后评分音频",
        "A_R_DENOISE_E": "固定真实视频：谱抑制后评分音频 − A0",
        "B_N_DENOISE_G": "B 自然源：谱抑制生成视频 − 基线生成视频",
        "B_T_NOISE_G_HARM": "B TTS 源：基线生成视频 − 加噪生成视频",
        "B_NATIVE_FRESH": "B 新鲜生成：TTS 原生 − 自然原生",
    }
    for name in ("A_N_DENOISE_E", "A_T_NOISE_E_HARM", "A_R_DENOISE_E", "B_N_DENOISE_G", "B_T_NOISE_G_HARM", "B_NATIVE_FRESH"):
        summary = analysis.get("primary", {}).get("summaries", {}).get(name, {})
        if summary.get("status") != "COMPLETE":
            lines.append(f"| `{name}` | {labels[name]} | NA | NA | NA | **{summary.get('evidence_status', 'INCOMPLETE')}** |")
        else:
            lines.append(f"| `{name}` | {labels[name]} | {_fmt(summary.get('mean'))} | {_interval(summary, 'ci95')} | {_interval(summary, 'ci99_166667_bonferroni')} | **{summary.get('evidence_status', 'INCONCLUSIVE')}** |")
    return "\n".join(lines)


def _hypothesis_status(analysis: Mapping[str, Any], crossed: Mapping[str, Any]) -> dict[str, str]:
    primary = analysis.get("primary", {}).get("summaries", {})
    def evidence(name: str) -> str:
        return str(primary.get(name, {}).get("evidence_status", "INCOMPLETE"))
    a_effects = [evidence(name) for name in ("A_N_DENOISE_E", "A_T_NOISE_E_HARM", "A_R_DENOISE_E")]
    if any(item == "INCOMPLETE" for item in a_effects):
        h1 = "INCONCLUSIVE"
    elif any(item == "POSITIVE_EVIDENCE" for item in a_effects):
        h1 = "SUPPORTED_FOR_THIS_INTERVENTION"
    else:
        h1 = "INCONCLUSIVE"
    b_n = evidence("B_N_DENOISE_G")
    b_t = evidence("B_T_NOISE_G_HARM")
    if crossed.get("status") != "COMPLETE":
        h2 = "INCONCLUSIVE"
    elif crossed.get("control_gate") != "PASS":
        h2 = "CONTROL_LIMITED"
    elif b_n == "INCOMPLETE" or b_t == "INCOMPLETE":
        h2 = "INCONCLUSIVE"
    elif b_n == "POSITIVE_EVIDENCE" and b_t == "POSITIVE_EVIDENCE":
        h2 = "SUPPORTED_FOR_THIS_INTERVENTION"
    else:
        h2 = "NOT_SUPPORTED_FOR_REGISTERED_OPERATIONS"
    return {"H1": h1, "H2": h2, "H3": "DESCRIPTIVE_ONLY", "H4": "NOT_CAUSALLY_IDENTIFIED", "PERCEPTION": "NOT_ASSESSED"}


def assemble_final(paths: config.RunPaths, *, validation_status: str | None = None) -> dict[str, Any]:
    a = read_json(paths.fixed_video / "a_manifest.json") if (paths.fixed_video / "a_manifest.json").is_file() else {}
    generation = read_json(paths.generation / "manifest.json") if (paths.generation / "manifest.json").is_file() else {}
    crossed = read_json(paths.crossed / "manifest.json") if (paths.crossed / "manifest.json").is_file() else {}
    analysis = read_json(paths.analysis / "summary.json") if (paths.analysis / "summary.json").is_file() else {}
    perception = read_json(paths.perception / "analysis.json") if (paths.perception / "analysis.json").is_file() else {}
    stages = {
        "audit": _status(paths.audit / "assets.json"),
        "audio": _status(paths.audio / "manifest.json"),
        "fixed_video": _status(paths.fixed_video / "manifest.json"),
        "A_scoring": _status(paths.fixed_video / "a_manifest.json"),
        "generation": _status(paths.generation / "manifest.json"),
        "crossed_scores": _status(paths.crossed / "manifest.json"),
        "analysis": _status(paths.analysis / "summary.json"),
        "perception_package": _status(paths.perception / "package.json"),
        "perception_analysis": _status(paths.perception / "analysis.json"),
    }
    automatic = all(stages[key] == "COMPLETE" for key in ("audit", "audio", "fixed_video", "A_scoring", "generation", "crossed_scores", "analysis", "perception_package", "perception_analysis")) and validation_status == "valid"
    if automatic:
        engineering_status = "AUTOMATIC_COMPLETE"
    elif "RESOURCE_WAIT" in stages.values():
        engineering_status = "RESOURCE_WAIT"
    elif "DEPENDENCY_BLOCKED" in stages.values():
        engineering_status = "DEPENDENCY_BLOCKED"
    else:
        engineering_status = "PARTIAL"
    final = {
        "schema_version": 1,
        "protocol_id": config.PROTOCOL_ID,
        "engineering_status": engineering_status,
        "status": engineering_status,
        "automatic_complete": automatic,
        "stage_status": stages,
        "counts": {
            "A_science_expected": config.EXPECTED_A_SCIENCE,
            "A_control_expected": config.EXPECTED_A_CONTROLS,
            "A_science_observed": a.get("science_cell_count"),
            "A_control_observed": a.get("control_cell_count"),
            "B_videos_expected": config.EXPECTED_B_VIDEOS,
            "B_repeats_expected": config.EXPECTED_B_REPEATS,
            "B_science_scores_expected": config.EXPECTED_B_SCIENCE,
            "B_control_scores_expected": config.EXPECTED_B_CONTROLS,
            "B_videos_observed": generation.get("science_video_count"),
            "B_scores_observed": crossed.get("science_cell_count"),
            "perception_sync_formal": config.EXPECTED_SYNC_PAIRS,
            "perception_quality_formal": config.EXPECTED_QUALITY_PAIRS,
        },
        "hypotheses": _hypothesis_status(analysis, crossed),
        "primary": analysis.get("primary", {}),
        "historical_v15_is_context_only": True,
        "fresh_native_reestablished": str(analysis.get("primary", {}).get("summaries", {}).get("B_NATIVE_FRESH", {}).get("evidence_status", "INCOMPLETE")) == "POSITIVE_EVIDENCE",
        "perception": {"sync": perception.get("sync", {}).get("status", "PERCEPTION_NOT_ASSESSED"), "quality": perception.get("quality", {}).get("status", "QUALITY_NOT_ASSESSED")},
        "training_distribution": "TRAINING_DISTRIBUTION_UNKNOWN",
        "validation_status": validation_status,
        "no_wav2lip_substitution": True,
    }
    return final


def write_report(paths: config.RunPaths, *, validation_status: str | None = None) -> dict[str, Any]:
    audit = read_json(paths.audit / "assets.json") if (paths.audit / "assets.json").is_file() else {}
    audio = read_json(paths.audio / "manifest.json") if (paths.audio / "manifest.json").is_file() else {}
    fixed = read_json(paths.fixed_video / "manifest.json") if (paths.fixed_video / "manifest.json").is_file() else {}
    a = read_json(paths.fixed_video / "a_manifest.json") if (paths.fixed_video / "a_manifest.json").is_file() else {}
    generation = read_json(paths.generation / "manifest.json") if (paths.generation / "manifest.json").is_file() else {}
    crossed = read_json(paths.crossed / "manifest.json") if (paths.crossed / "manifest.json").is_file() else {}
    analysis = read_json(paths.analysis / "summary.json") if (paths.analysis / "summary.json").is_file() else {}
    perception = read_json(paths.perception / "analysis.json") if (paths.perception / "analysis.json").is_file() else {}
    final = assemble_final(paths, validation_status=validation_status)
    replay = a.get("v15_replay", {}) if isinstance(a.get("v15_replay"), Mapping) else {}
    control_statuses: dict[str, int] = {}
    for row in a.get("controls", []) if isinstance(a.get("controls"), list) else []:
        status = str(row.get("control", {}).get("status", "UNKNOWN"))
        control_statuses[status] = control_statuses.get(status, 0) + 1
    control_summary = ", ".join(f"{key}={control_statuses[key]}" for key in sorted(control_statuses)) or "NONE"
    audio_rows = audio.get("records", []) if isinstance(audio.get("records"), list) else []
    audio_contract_pass = sum(
        bool(row.get("condition") in {"GAIN", "NOISE"} and (
            (row.get("condition") == "GAIN" and abs(float(row.get("operation", {}).get("measured_gain_db", 999.0)) + 6.0) <= 0.01)
            or (row.get("condition") == "NOISE" and abs(float(row.get("operation", {}).get("measured_snr_db", 999.0)) - 20.0) <= 0.1)
        ))
        for row in audio_rows
    )
    unit = analysis.get("unit_geometry", {}) if isinstance(analysis.get("unit_geometry"), Mapping) else {}
    content = analysis.get("content_retrieval", {}) if isinstance(analysis.get("content_retrieval"), Mapping) else {}
    rhythm = analysis.get("rhythm", {}) if isinstance(analysis.get("rhythm"), Mapping) else {}
    generation_reason = ""
    if isinstance(generation.get("failures"), list) and generation["failures"]:
        generation_reason = str(generation["failures"][0].get("reason", ""))
    lines = [
        f"# TTS 原生 Sync-C 增益来源实验（{paths.root.name}）",
        "",
        "## 研究问题",
        "",
        "TFG 模型主要在自然说话视频上训练，但历史 LRS3 结果显示 TTS 音频配合生成视频时可能有更高 Sync-C。本实验把“生成口型”和“评价音频”拆开，判断增益来自哪一条路径。历史 v15 只作为已冻结背景；本 run 的 A/B 结果单独计算。",
        "",
        "## 实验流程",
        "",
        "1. 固定 ID 151–162 和 12 个 source group。CPU 审计 v15 的 142 个绑定文件、24 个历史 crop/matrix、原始 LRS3 manifest、真实视频、转写和 SyncNet 权重。真实视频的人脸轨迹按最长连续轨迹、最早起始帧、文件名字典序选择；选择不看 Sync-C。",
        "2. 从固定 crop（N/T）和真实视频（R）解码 16 kHz 单声道 PCM16。每个 ID 的自然音频低能帧估计有色噪声谱；N/T/R 各自保留自己的样本时钟。",
        "3. 对每条源音频产生 ORIGINAL、共同 headroom 基线 A0、−6 dB、20 dB 有色噪声和固定谱抑制 DENOISE。所有处理在 float64，最后用 ties-to-even 量化 PCM16；变体不删静音、不重采样、不改变长度。",
        "4. A 阶段固定视频像素、crop 和 PTS，只换评价音轨，得到 180 个科学 cell 和 18 个延迟/恒等控制 cell。A 的差值是 evaluator-input effect，不能直接叫口型改善。",
        "5. B 阶段要求可核验的 LeapTalk 推理配置。每个 ID、N/T、三种 driver、seed 42/43 生成 V0/Vnoise/Vdenoise，再用同一音频和同一冻结 ROI 做七格评分。四格公式为 q00=(V0,A0)、q01=(V0,H)、q10=(Vh,A0)、q11=(Vh,H)，因此 G=q10−q00，E=q01−q00，I=q11−q10−q01+q00，total=q11−q00。",
        "6. 先对时间求均值，再对 31 个 lag 求 D=min、B=median、C=B−D；主端点使用共同 INTERIOR 支持。另存 D0、offset、FULL/EQUAL_COUNT、单位范数特征、错误内容检索和节奏描述。",
        "7. 六个主对比都以 12 个 source group 等权。使用同一套 PCG64 seed 20260915、20000 次 bootstrap 和六项 Bonferroni 校正；校正区间完全在 ±0.200 内才可称小幅。",
        "",
        "## 本 run 的工程状态",
        "",
        f"- audit: **{audit.get('status', 'NOT_RUN')}**；audio: **{audio.get('status', 'NOT_RUN')}**；fixed video: **{fixed.get('status', 'NOT_RUN')}**；A scoring: **{a.get('status', 'NOT_RUN')}**。",
        f"- B generation: **{generation.get('status', 'NOT_RUN')}**；crossed scoring: **{crossed.get('status', 'NOT_RUN')}**。LeapTalk 缺少本机可核验的 repo/checkpoint 时，B 必须是 `DEPENDENCY_BLOCKED`，不能用 Wav2Lip 替代。",
        f"- final engineering status: **{final['engineering_status']}**；validation: **{validation_status or 'PENDING'}**。",
        f"- A 重放审计：{replay.get('status', 'NOT_RUN')}；比较 {replay.get('comparison_count', 'NA')} 个 N/T 矩阵，最大绝对误差 {_fmt_precise(replay.get('max_abs_error'))}，容差 {_fmt_precise(replay.get('tolerance_max_abs'))}。A 控制状态：{control_summary}。",
        f"- 音频契约：{len(audio_rows)} 条逻辑音频记录；GAIN/NOISE 的 72 个有效检查通过 {audio_contract_pass} 个。逻辑特征条目为 {a.get('feature_forward_counts', {})}，本次实际模型前向为 {a.get('model_forward_counts', {})}，缓存/ORIGINAL-A0 别名为 {a.get('feature_cache_counts', {})}；距离矩阵在 CPU 上生成。",
        f"- CPU 诊断：单位范数几何 {unit.get('status', 'NOT_RUN')}（{len(unit.get('rows', [])) if isinstance(unit.get('rows'), list) else 'NA'} 行）；错误内容检索 {content.get('status', 'NOT_RUN')}（{content.get('pair_count', 'NA')}/{content.get('expected_pair_count', config.EXPECTED_WRONG_CONTENT)} 对，转写 {content.get('transcript_status', 'NA')}）；节奏 {rhythm.get('status', 'NOT_RUN')}。",
        f"- B 生成错误/阻塞原因：{generation_reason or '未记录'}；固定计划分母为 {config.EXPECTED_B_VIDEOS} 个科学视频、{config.EXPECTED_B_REPEATS} 个重复视频、{config.EXPECTED_B_SCIENCE} 个科学评分和 {config.EXPECTED_B_CONTROLS} 个控制评分；实际完成数见机器清单。",
        "",
        "## 六个预注册主对比",
        "",
        _primary_table(analysis),
        "",
        "正值是预期方向。A 的三个量测量固定视频的评价音频响应；B 的两个干预量测量重新生成的口型响应；`B_NATIVE_FRESH` 只判断本次 LeapTalk 配置是否重现新鲜原生优势。区间跨 0 是不确定，不能写成“没有机制”。",
        "",
        "## 如何解释结果",
        "",
        f"- H1（SyncNet 评价端几何/声学可辨认度）：**{final['hypotheses']['H1']}**。固定视频若 C 改变，只能定位评价音频路径。单位范数和错误内容检索用于区分尺度变化与内容区分度，不能把 Sync-C 本身当感知质量。",
        f"- H2（干扰较少的音频更适合 TFG）：**{final['hypotheses']['H2']}**。谱抑制可能同时去掉语音能量，加噪也可能损伤发音线索，所以结果只属于已注册的两种操作。",
        "- 当前可定位的 A 证据：`A_T_NOISE_E_HARM` 的方向和区间状态见上表；它表示同一固定 V_T 在评价端替换音频后 Sync-C 如何变化，属于 evaluator-input 响应，不能推出重新生成口型会获得同样增益。`A_N_DENOISE_E` 与 `A_R_DENOISE_E` 分别只支持表中标出的状态，不能合并成一个一般性的“音质越好越同步”结论。",
        "- 若 B 仍为 `DEPENDENCY_BLOCKED`，本 run 不能判断音频变化是否通过 LeapTalk 生成器改变了口型，也不能确认原生 TTS 与自然音频的 fresh native 差异；需要同一模型、冻结 ROI、成对 seed 的 B 数据后再计算 G/E/I。",
        "- H3（节奏/局部动态复杂度）：**DESCRIPTIVE_ONLY**。本轮只做音频摘要和预注册 Spearman 描述，不做时间伸缩因果干预，也不做中介分析。",
        "- H4（训练筛选或 checkpoint 选择造成共享偏好）：**NOT_CAUSALLY_IDENTIFIED**。训练分布、frontend 预训练和实际 checkpoint 选择证据不完整，统一标为 `TRAINING_DISTRIBUTION_UNKNOWN`。",
        f"- 真人评价：同步 **{perception.get('sync', {}).get('status', 'PERCEPTION_NOT_ASSESSED')}**，音质 **{perception.get('quality', {}).get('status', 'QUALITY_NOT_ASSESSED')}**。已构建 96 对同步和 48 对音质包及隐藏重复题；空模板没有返回评分，不能由 Sync-C 代填。",
        "",
        "## 产物位置",
        "",
        f"- P0/P1：`{paths.audit}`、`{paths.audio}`；A：`{paths.fixed_video}`；B：`{paths.generation}`、`{paths.crossed}`；分析：`{paths.analysis}`；人工包：`{paths.perception}`。",
        f"- 机器可读结论：`{paths.final}`；独立校验：`{paths.validation}`。",
        "",
        "本报告保留原始矩阵、特征、音频 hash、视频像素/PTS hash、支持集合和失败记录。任何缺失 cell 都保留在固定分母中，不用更高分样本替换。",
        "",
    ]
    paths.report.parent.mkdir(parents=True, exist_ok=True)
    temporary = paths.report.with_name(f".{paths.report.name}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(paths.report)
    final["report_sha256"] = file_sha256(paths.report)
    return write_self_hashed_json(paths.final, final)
