"""Human-readable report generation with explicit partial/blocking states."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .config import write_json


def write_report(run_dir: str | Path, *, config: Mapping[str, Any], statuses: Mapping[str, Any], analysis: Mapping[str, Any] | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    analysis = dict(analysis or {})
    expected = {"audit", "calibrate", "infer", "quality", "phone", "render", "score", "official", "analyze", "report"}
    if str(config.get("protocol_id", "")).endswith("repair_v2"):
        expected.update({"train", "lock"})
    complete_states = {"COMPLETE", "NOT_APPLICABLE"}
    complete = bool(statuses) and all(stage in statuses and str(statuses[stage].get("state", "")) in complete_states for stage in expected)
    report_status = "COMPLETE" if complete else "PARTIAL"
    lines = ["# LRS3 音素增强静态肖像 TFG 与 MFA 条件实验", "", f"- protocol: `{config.get('protocol_id')}`", f"- run_status: **{report_status}**", "- 本轮允许阴性结果；任何未完成分支都不作为科学结论。", "", "## 阶段状态", "", "| stage | state | reason |", "|---|---|---|"]
    for stage, value in statuses.items():
        if isinstance(value, Mapping):
            lines.append(f"| {stage} | {value.get('state', 'UNKNOWN')} | {', '.join(value.get('reasons', []))} |")
    lines.extend(["", "## 结论边界", "", "- 推理输入白名单只允许静态 PNG 与指定 PCM；本报告不把静态输入契约扩大为所有预训练数据独立。", "- C 的 MFA 身份条件是已知转写辅助条件，不是无文本音频系统。", "- 人工听感若未收集，状态为 `HUMAN_NOT_ASSESSED`。", "", "## 统计产物", "", f"```json\n{analysis}\n```", ""])
    report_path = root / "09_report/report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    decision = {"status": report_status, "human_quality": "HUMAN_NOT_ASSESSED", "phone": analysis.get("phone", "NOT_RUN"), "tfg": analysis.get("tfg", "NOT_RUN"), "tts_comparison": analysis.get("tts_comparison", "NOT_RUN"), "report": str(report_path.resolve())}
    write_json(root / "09_report/decision.json", decision)
    return decision


__all__ = ["write_report"]
