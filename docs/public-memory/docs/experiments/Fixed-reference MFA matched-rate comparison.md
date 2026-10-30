---
title: Fixed-reference MFA matched-rate comparison
type: report
permalink: tts-exp/docs/experiments/fixed-reference-mfa-matched-rate-comparison
status: blocked
date: '2026-08-14'
sample_id: '1'
mfa_version: 2.2.17
alignment_gate: failed
tags:
- mfa
- fixed-reference
- mfa-linear
- negative-result
---

# Fixed-reference MFA matched-rate comparison

## Result

2026-08-14: 用同一 MFA 命令比较历史 strict-gate TTS 与 fixed-reference TTS。两条音频都转为 16 kHz、mono，并使用同一 transcript（sample 1，S0765）：`楼市地市交相升温房价会不会再度暴涨`。

命令：

```bash
PATH="$HOME/miniconda3/envs/mfa/bin:$PATH" "$HOME/miniconda3/envs/mfa/bin/mfa" align <input> mandarin_mfa mandarin_mfa <output> --single_speaker --clean --num_jobs 4
```

MFA 环境：`[redacted-local-path]`，已验证版本 2.2.17。

输入：

- 历史 strict-gate TTS：`results/rhythm_style_500/aishell1_test_400/tts/0001.wav`，原始 24 kHz；使用 `scipy.signal.resample_poly` 转为 16 kHz；原历史 TextGrid 曾包含 34 个 speech phones。
- fixed-reference TTS：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/fixed_reference_tts_20260814/tts/1.wav`，原生 16 kHz。

输出：`runs/aishell1_mfa_linear_n25_resample_poly_20260814/mfa_compare_known_strict16k_vs_fixed_20260814/`

## Observations

- 两次 MFA 命令均返回成功并导出 TextGrid。
- 两条音频均得到相同结构：3 个 phone intervals，前后为空，中间整段为 `spn`。
- 两条结果 speech phone count 都是 0。
- 两条日志都有同一 warning：80125 pronunciations ignored because they contain one of 40 phones not present in trained acoustic model。
- 匹配到 16 kHz 后，结果仍然完全相同；sampling rate 不是 fixed-reference 失败原因。
- 这也发现：仅依赖当前 MFA 2.2.17 重跑，历史上通过的 TTS 也不能复现；历史 TextGrid 来自不同的历史对齐环境、模型版本或配置。

## Decision

fixed-reference control 仍不能进入 strict MFA-linear。不能伪造 phones，也不能复用历史 TextGrid/tokens。下一步应审计历史成功 MFA 的 acoustic model、dictionary、MFA 版本和调用配置，与当前环境逐项比较；不应继续把问题归因于 fixed-reference 音频或重新安装 MFA。
