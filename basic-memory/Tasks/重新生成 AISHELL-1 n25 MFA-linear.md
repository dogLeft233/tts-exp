---
title: 重新生成 AISHELL-1 n25 MFA-linear
type: task
permalink: tts-exp/tasks/重新生成-aishell-1-n25-mfa-linear
assigned_to: claude
current_step: 5
priority: high
started: '2026-08-13'
status: done
steps:
- 核对现有 frozen cohort、strict TTS metadata、tokens、模型 provenance，以及新 rerun 目录不存在或为空。
- 在独立新目录重新运行 MFA-linear 生成脚本，不覆盖既有 mfa_linear_strict 结果。
- 检查 25/25 输出、0 failures、audio hash、exact natural length、finite/provenance 字段和旧目录未变化。
- 运行相关测试并确认没有回归。
- 更新本任务和实验记忆，记录新 rerun 路径、检查结果及限制。
tags:
- task
- aishell1
- mfa-linear
- rerun
completed: '2026-08-13'
---

# 重新生成 AISHELL-1 n25 MFA-linear

## Observations
- [description] 在不删除或覆盖既有 strict MFA-linear 结果的前提下，重新生成一份独立的 MFA-linear 产物并完成测试与逐步完整性检查。
- [status] done
- [assigned_to] claude
- [current_step] 5
- [completed] 2026-08-13
- [result] 最终 provenance-corrected rerun 位于 `runs/aishell1_mfa_linear_n25_20260813/mfa_linear_rerun_fixed_20260813/`：25/25、0 failures、exact natural length、finite audio、valid hashes、5 speakers x 5。
- [result] Existing `mfa_linear_strict` 和第一份 rerun 目录均未修改。
- [result] 针对性测试通过 `24 passed, 3 warnings`；修复后聚焦测试通过 `15 passed, 3 warnings`。
- [problem] 首次直接调用 `.venv/bin/python -m pytest` 因 `.venv` 未安装 pytest 失败；改用 `uv run --with pytest --python .venv/bin/python` 后通过，未修改环境。

## Steps
1. [x] 核对现有 frozen cohort、strict TTS metadata、tokens、模型 provenance，以及新 rerun 目录不存在或为空。
2. [x] 在独立新目录重新运行 MFA-linear 生成脚本，不覆盖既有 mfa_linear_strict 结果。
3. [x] 检查 25/25 输出、0 failures、audio hash、exact natural length、finite/provenance 字段和旧目录未变化。
4. [x] 运行相关测试并确认没有回归。
5. [x] 更新本任务和实验记忆，记录新 rerun 路径、检查结果及限制。

## Context
旧的有效结果位于 `runs/aishell1_mfa_linear_n25_20260813/mfa_linear_strict/`，未被覆盖。最终 rerun 使用冻结 cohort `runs/aishell1_mfa_linear_n25_20260813/data_boundary/aishell1_n25_speaker_balanced_cohort.json`、strict TTS metadata `runs/aishell1_mfa_linear_n25_20260813/tts_strict/tts_meta.json` 和 frozen strict token manifest `runs/aishell1_mfa_linear_n25_20260813/mfa/tokens.json`。`scripts/pilot_generate_mfa_linear.py` 已修复为记录实际 cohort manifest 路径及 SHA。模型链保持冻结 WavLM-Large L6 + prematched HiFi-GAN。未重新运行 Wav2Lip/SyncNet。

## Relations
- project [[MFA-linear 冻结 VC 重合成与下游验证]]
