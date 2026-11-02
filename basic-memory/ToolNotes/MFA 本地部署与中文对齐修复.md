---
title: MFA 本地部署与中文对齐修复
type: note
permalink: tts-exp/tool-notes/mfa-本地部署与中文对齐修复
tags:
- mfa
- tooling
- alignment
- spn
- mfa-linear
---

# MFA 本地部署与中文对齐修复

## Context

本地 MFA 中文对齐全部输出 `spn`(整段 speech noise),排查发现是模型版本错配 + 中文分词缺失。修复 2.2.17 后,又按官方方式安装了 MFA 3.4.1 对比。线程终点:两套环境都可用了,2.2 + 逐字预分词效果最好;并揭示了服务器缓存 TextGrid 的真实来源与缺陷。

## 模型地址与环境

### 2026-08-16 rerun runtime distinction

- `/home/wjj/Documents/MFA/mfa3_runtime_20260815` is an English-only MFA 3.4.1 runtime (`english_mfa` only); it must not be used for Chinese alignment.
- `/home/wjj/Documents/MFA/mfa3_chinese_rerun_20260815` is a separate plain-YAML MFA 3.4.1 runtime with linked `mandarin_china_mfa` dictionary, `mandarin_mfa` acoustic model, and `mandarin_china_mfa` g2p asset. It produced complete Chinese TextGrids for the 2026-08-16 RAMC/AliMeeting cloud-Qwen rerun.
- The old MFA 2.2.17 environment remains available for historical compatibility, but the rerun showed substantially more local `spn` and six missing RAMC natural TextGrids; those outputs remain diagnostic and were not used for MFA-linear.

| 项 | 值 |
|---|---|
| 2.2.17 env | `/home/wjj/miniconda3/envs/mfa` (py3.10), mfa 命令 `~/.local/bin/mfa` |
| 3.4.1 env | `/home/wjj/miniconda3/envs/mfa3` (py3.14), 官方 conda-forge 安装 |
| 模型根目录 | `~/Documents/MFA/` (MFA_ROOT_DIR) |
| 词典 2.x | `~/Documents/MFA/pretrained_models/dictionary/mandarin_mfa.dict` (v2.0.0a) |
| 声学 2.x | `~/Documents/MFA/pretrained_models/acoustic/mandarin_mfa.zip` (v2.0.0a, 133 phones) |
| 词典 3.x | `mandarin_china_mfa.dict` |
| 声学 3.x | `mandarin_mfa.zip` (v3.0.0, 2024-02-24, 145 phones) |
| g2p 3.x | `g2p/mandarin_china_mfa.zip` (拼音转写用, 不处理汉字 OOV) |
| 备份 | `~/Documents/MFA/backup_3x/` (旧 3.x 格式模型) |

## 使用方式

```bash
# 2.2.17 (需要 PATH 含 env bin 才能找到 initdb)
PATH="/home/wjj/miniconda3/envs/mfa/bin:$PATH" mfa align <corpus> mandarin_mfa mandarin_mfa <out> --single_speaker --clean

# 3.4.1 (内置词级分词, 无需预处理中文)
mfa align <corpus> mandarin_china_mfa mandarin_mfa <out> --single_speaker --clean
```

中文转录 `.lab` 必须字符级预分词 (`楼 市 地 市`) 否则整句当单个 OOV 词 → spn。MFA 2.2 无内置中文分词;3.x 有 (pkuseg) 但词级分词生僻词仍 OOV。

## 修复内容 (2026-08-14)

### 根因:模型版本错配

本地装的是 MFA 2.2.17,但 `mandarin_mfa.dict` 是 **MFA 3.x 格式**(带概率列,163066 行全部),2.2.17 把概率数字当 phones → 80125 条发音被忽略 → 词典报废 → 全 OOV → 全 spn。用 `mfa model download dictionary mandarin_mfa --version v2.0.0a` 重下 2.x 兼容版修复。旧 3.x 词典曾备份。

### 根因:中文未分词

`clean_lab_text` 原实现把非中文字符(含空格)全删 → 整句一个 word → OOV。修复为逐字空格分隔。

### 代码修复

- `scripts/run_aishell1_n25_mfa.py` 与 `scripts/pilot_run_mfa.py` 的 `clean_lab_text` 同步升级:
  - CJK 连串 → 逐字空格分隔
  - 阿拉伯数字 → 中文读法 (`2013` → `二千零一十三`,支持到 9999 万)
  - `<digits>%` → `百分之<读法>` (`30%` → `百分之三十`)
  - 拉丁 token 保留原样 (会 OOV → 局部 spn)
- 新增 `tests/test_mfa_clean_lab_text.py` (33 passed)
- 数字读法注意:10-19 首字符 "十" 不加 "一";中间零位补 "零"

## 验证结果 (3 样本 × natural/tts)

| 方式 | phones | spn |
|---|---|---|
| 2.2 + 逐字预分词 | 44/50/47 | **0** |
| 3.x + raw 中文 (内置词级分词) | 38/50/42 | 1 (交相/市管国管/东地产 OOV) |
| 3.x + 逐字预分词 | 42/56/49 | **0** |
| 2.2 + raw 未分词 | 全 spn | 100% |

## 重大发现:缓存 TextGrid 有缺陷

- 服务器缓存 `results/rhythm_style_500/aishell1_test_400/mfa/{natural,tts}/*.TextGrid` 是 **MFA 3.x + 内置词级分词**生成的 (本地 3.x raw 输出与之逐字节一致),不是独立门控过的干净对齐
- 缓存 words tier 词级分词 (楼市/交相/东地产),"交相" 等未登录词 → spn
- `strict_natural.tokens` 把 spn 静默标成 `sil` → mfa_linear 在"交相"处按静音处理,丢失真实发音 → 可能部分解释 n25 结果差
- RAMC/AliMeeting pilot 的 tokens.json 全 spn + coverage 0.936 是 spn↔spn 匹配的假象,那轮自发域结论需重新审视

## spn 判断标准

spn 不必然坏 (官方定义为 unknown word 兜底 phone):
- **整段 spn** = 系统性配置错误 (词典不兼容/未分词) → 数据不可用
- **局部 spn** (外来词如 "IG" 0.5s) = 真实 OOV → 可接受但该段无音素对齐
- **坏的是把 spn 静默当 sil**:有实际语音却按静音处理,下游丢内容

## Observations

- [decision] 中文 MFA 转录必须逐字预分词,2.2 与 3.x 下均适用 #mfa #alignment
- [decision] 本地 mfa 环境保留 2.2.17 (现有脚本依赖),mfa3 环境供 3.x 实验 #mfa #tooling
- [insight] 服务器缓存 TextGrid 由 MFA 3.x 词级分词生成,非干净门控,含 spn 缺口 #mfa #cache
- [insight] 缓存把 spn 静默降级为 sil,导致 mfa_linear 丢失"交相"等字发音 #mfa-linear
- [insight] RAMC/AliMeeting pilot 覆盖率 0.91-0.94 是 spn 匹配假象,非真实对齐质量 #pilot
- [solution] 数字/百分号在词典无条目,须预转中文读法 #mfa #text-normalization
- [problem] MFA 3.x g2p mandarin_china_mfa 只处理拼音转写,不兜底汉字词 OOV #mfa
- [learning] MFA 2.2.17 依赖链脆弱 (joblib<1.4, setuptools<81, postgresql),3.x 已重构 #tooling
- [learning] 3.x 安装需 `conda tos accept` + 中文 tokenizer 依赖 (spacy-pkuseg dragonmapper hanziconv) #tooling

## Relations

- relates_to [[MFA-linear 冻结 VC 重合成与下游验证]]
- relates_to [[多样域中文试点：MFA-linear 在自发对话域失败]]
- relates_to [[Montreal Forced Aligner (McAuliffe et al., 2017)]]
## 2026-08-14 verification correction

A later on-disk audit found the claimed 2.x dictionary repair is not currently true. `~/Documents/MFA/pretrained_models/dictionary/mandarin_mfa.dict` still begins with MFA 3.x probability columns (`<eps> 1.0 0.0 0.0 0.0 sil`) and has the same size/content pattern as `backup_3x/mandarin_mfa.dict`. Fresh MFA 2.2.17 runs still log `80125 pronunciations ... ignored` because 40 dictionary phones are absent from the acoustic model. Thus recent `fixed_reference_mfa_sameenv`, `multireference_mfa_style_audit`, and related reruns used a mismatched dictionary despite command success; local `spn` outputs are not a clean repaired baseline.

Code audit also found `scripts/run_aishell1_n25_mfa.py` and `scripts/knn_vc_retrieval.py` still classify `spn` as silence. `_token_label` collapses every `spn` to `<sil>`, and `matched_span_map` can align unrelated unknown-speech spans as equal silence. This can contaminate coverage, phone-local interpolation, pause statistics, and speaker-dependent artifacts.

Correct next action: stop model training; use MFA 3.4.1 with matched `mandarin_china_mfa` dictionary + `mandarin_mfa` acoustic model and character-segmented transcript in a small fresh gate, freeze asset hashes, separate `spn` from silence/non-speech, and rerun clean baseline before interpreting prior mapper failures.
## 2026-08-14 clean MFA 3 gate result

Created isolated MFA 3.4.1 root `/home/wjj/Documents/MFA/mfa3_clean_20260814` with plain YAML config and symlinked pretrained assets. Verified dictionary `mandarin_china_mfa` and acoustic model `mandarin_mfa`.

Three sample gate (`S0765/1`, `S0901/101`, `S0912/201`) used character-segmented `.lab`, natural + canonical 16 kHz self-clone TTS:

- MFA commands: 3/3 complete;
- all six TextGrids: 0 `spn`;
- speech phone counts natural/TTS: 39/39, 50/50, 47/47.

Clean MFA-linear output: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/mfa3_clean_mfa_linear_gate_fix2_20260814/`.

Fixed-face Wav2Lip + SyncNet output: `runs/aishell1_mfa_linear_n25_resample_poly_20260814/mfa3_clean_gate_wav2lip_syncnet_20260814/`.

Clean MFA 3 MFA-linear scores: sample 1 `C=6.794/D=7.037`; 101 `5.350/7.235`; 201 `5.839/7.284`. Historical mismatched chain scores: 1 `6.406/7.477`; 101 `4.986/7.147`; 201 `5.583/7.473`. Mean clean change: Sync-C approximately `+0.336`, Sync-D approximately `-0.180` across three samples. Clean MFA still leaves 101/201 below natural, but impact is not negligible.

Decision: do not train renderer yet. Expand clean MFA 3.4.1 + `mandarin_china_mfa` + character segmentation to full n25, fix `spn` semantics in parser/retrieval, regenerate MFA-linear and audio QC, then reassess prior training conclusions.
