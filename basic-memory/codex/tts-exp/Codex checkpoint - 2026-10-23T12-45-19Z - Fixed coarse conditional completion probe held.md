---
title: Codex checkpoint - 2026-10-23T12-45-19Z - Fixed coarse conditional completion
  probe held
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-23-t12-45-19-z-fixed-coarse-conditional-completion-probe-held
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-23T12:45:19Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16144-7517-77a1-b372-027fba42da62
model: gpt-6-astra
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-23T12-45-19Z - Fixed coarse conditional completion probe held

## Summary
固定粗编码的0.6B/1.7B条件补全已完成单句技术验证：接口和精确自重放通过，但内容与局部时间保持未通过，停在音频检查。

## Story
研究原目标是解释TTS原生Sync-C优势。此前同10人统一RMS下1.7B对0.6B+0.656，但不能归因纯参数量或人类嘴型质量。用户指出过去停顿、有声响度、音素时长和局部能量干预多为损伤，要求先查BM再找更可靠思路。已核对包络四格、原生U时钟、WORLD时钟/谱形、39输入因子、43处理链预测及独立视觉/共同负池结果：不少破坏性干预能降低TTS却不能正向迁移，WORLD正结果有桥接失败边界，不能重新将这些列为未做机制。因而提出模型内部完整条件补全以寻找正向救援，但共享码本并不保证内容/时序纯分解。

用户最新明确授权最小技术验证，不完整评分、不训练。按旧云端10人ID首条BAC009S0002W0122、draw1、seed2026102401，两个模型各自的Talker.generate添加LogitsProcessor固定粗码与EOS，保留原采样调用/RNG消耗与各自隐藏状态、参考和其余15码本补全。未修改安装包或旧封存run。两个模型均通过旧完整码解码、原参数自由生成、固定自身粗码生成的逐码/逐波形精确重放，再做两个交叉组合。

两次交叉完整粗码和EOS、总长精确保持；0.6粗码58帧4.640秒、1.7粗码47帧3.760秒，细码改变92.759%/92.199%。但Whisper-small编辑距离0.6原组2→1.7补全7，1.7原组3→0.6补全4；原组已有识别错，内容损坏不能当人工事实。修复分词的MFA各35对应phone/0unknown，音素中心差中位/p95/max为25/120/130ms和20/201/245ms；两方向未过预设时间门。强制给定文字的标签一致不证明内容正确，自动边界也非人工真实移位。技术可行性通过，干净机制干预资格HOLD。

## Working State
权威run为runs/qwen_fixed_coarse_20261023_attempt01/，report.md及checks/mfa_lexical_repair02/summary.json是当前结论。旧checks/summary.json属于无分词OOV/spn首轮，必须保留区别。全部4模型子进程actual exit0，合计6次生成和2次单独旧码解码；4音频ASR、首轮4无效MFA和修复4MFA。无新TFG/SyncNet/训练/云调用，GPU进程已退出，仅原桌面313MiB进程保留。

完整run在Ubuntu /home/wjj/tts-audio/tts-exp与本机E runs。216成员248399764B已逐SHA读回；manifest SHA 1a75ca8dcbcfe9b0aac93c8aa2262e2d67d1a7c3ae1ca03e1d7c86f303b8fdf0。C压缩包C:/Users/dogle/.codex/tts-exp-artifacts/qwen_fixed_coarse_20261023_attempt01.tar.gz SHA 0e540ddaa0919a0b4ec3529c022ab065432c426b060f8330ccb54e58de314ae9；同目录_readback.json为运输回执，不声称云备份。E仍约1.8GB可用，勿大扩数据。

工作区继承大量dirty/untracked；本轮未提交或清理。当前分支codex/cross-tfg-audio-head-20261003，本地HEAD 2b71639c3b555293375d248350999377168d515d，未确认此SHA远端可解析，不构造commit链接。GitHub只读确认canonical repo，当前分支PR查询为空。

## Changed Files
新增scripts/experiments/qwen_fixed_coarse_20261023/下probe.py、driver.py、audio_checks.py、verify.py、finalize.py，及tests/test_qwen_fixed_coarse_20261023.py。完整执行源码在run/final_sources/与先前source版本保留。更新HANDOFF.md，BM实验笔记与研究地图已读回。新脚本/测试为untracked，不能仅依Git恢复。

## Verification
4项unittest通过：强制序列/RNG不变、非法输入拒绝、异常后恢复hook、空/unknown/尾部异常时间门拒绝。独立读存储NPY/WAV复核两模型三种重放与两交叉粗码/EOS/样本数，复算时间统计通过；这不是独立第二对齐器或人工听辨。216文件完整SHA读回PASS。

失败修复全部保留：首次audio_checks在ASR环境导入soundfile失败，尚未forward；改懒加载和stdlib wave，预处理使用Qwen环境。首次MFA无空格整句被词典视为OOV/spn，process0不代表有效测量；在新mfa_lexical_repair02用逐字词典覆盖通过的文本，仅重做对齐、不改音频或重合成。首轮readback遇空phone数组IndexError，修正空支持处理后审计新有效summary。execution_notes.json与历史source/产物承载这些差异。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- runs/qwen_fixed_coarse_20261023_attempt01/report.md
- runs/qwen_fixed_coarse_20261023_attempt01/independent_readback.json
- runs/qwen_fixed_coarse_20261023_attempt01/checks/mfa_lexical_repair02/summary.json

## Observations
- [result] 两模型精确自重放通过，固定粗码跨模型补全可运行；内容与局部时序保持未确认，技术门HOLD。
- [decision] 保留单句/单seed范围，不扩10人、不调门限、不进入TFG/SyncNet或训练；不将共享码本解释为纯语义/声学因果拆分。
- [blocker] ASR基线自身有误识别，MFA自动边界变化可能混合发音变化与测量误差，缺盲听及独立局部边界复核。
- [next_step] 若继续本路线，先复核已保存四格中疑似异常词段的可懂度与局部边界，判明实际变化或测量失效，再决定是否值得扩大；当前不启动更多计算。

## Relations
- continues [[Codex checkpoint - 2026-10-23T12-11-40Z - Qwen17 normalized gain mechanism interpretation]]
- summarizes [[固定粗编码的双模型条件补全技术验证 2026-10-23]]
- relates_to [[研究主线与证据地图]]
