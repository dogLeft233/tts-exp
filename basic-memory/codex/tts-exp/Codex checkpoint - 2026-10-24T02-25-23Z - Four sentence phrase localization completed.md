---
title: Codex checkpoint - 2026-10-24T02-25-23Z - Four sentence phrase localization
  completed
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-24-t02-25-23-z-four-sentence-phrase-localization-completed
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-24T02:25:23Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16144-7517-77a1-b372-027fba42da62
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-24T02-25-23Z - Four sentence phrase localization completed

## Summary
四句28音频的短语定位及父侧原数组复核已完成，发现一处稳定模型间上升和同一句内两处下降，尚不能归纳跨来源的语音共性或原因。

## Story
目标是解释统一音量后1.7B相对0.6B的SyncNet优势为何随来源改变，而非先训练方法。此前独立复核已确认旧10来源平均+0.656、新10−0.201，简单声学特征及熵相关未提供跨批稳健解释。用户先要求试听案例，随后授权子代理部署FunASR，再明确短语分析只做刚试听4句话/28条音频，不扩到20来源。父侧预先冻结9文字短语、三个比较、全部3draw和五边界规则；子代理完成CPU对齐与已有官方距离数组分析，父侧独立复算。未新生成TTS/TFG、未新SyncNet前向或训练。

权威结果runs/qwen_phrase_local_20261024_attempt02/：S0002“抑制作用最大的限购”L17−L06为+1.743/+1.864/+0.332；S0014“折合楼面价”为−2.441/−2.052/−0.392，“三点八万元”为−0.622/−0.732/−1.497，均三draw与五边界同号。但S0014两个TTS各自仍高于自然；S0002该短语0.6B相对自然稳定负、1.7B相对自然混合。S0013整句三draw负，拆出的两短语却各自混合。S0014“每平方米”1.7B draw1仅2窗，不能完整比较。三个第一短语ASR不一致仅标内容待核，不能称实际错读。

最新用户问稳定短语有没有共性，还是数据不足。应回答目前没有可靠跨句/跨说话人共性：一个正短语与同一句两个负短语，非三个独立来源；28条音频也只有4来源/4文本，三次生成不增加文本多样性；来源与内容混淆，四句为按已知整句表现挑选的事后案例。可提出待检验的发音/韵律猜测，但不能把数字词、短句、模型更清晰等说成已发现规律。先核听及看三个稳定片段，再冻结具体候选并在其它已有来源检验，比盲目扩大特征搜索更有价值。

## Working State
本机E:/Documents/tts-exp与Ubuntu /home/wjj/tts-audio/tts-exp均有该run；index.html为全部63WAV+63MP4入口，comparison_table.md与comparisons.json完整27比较，parent_raw_array_audit.json为父侧凭据。protocol在runs/qwen_phrase_local_design_20261024_attempt01/protocol.json，SHA601f1fabcc48a184b6d64976ebd639cc1674160fb5b0096d5f6179121e8a1646。子代理全219文件SHA读回0差异，父侧未重复全部媒体运输验证。无该任务遗留进程。大量继承dirty/untracked，本轮直接git status读取且未清理或提交。

FunASR独立CPU环境/home/wjj/.venvs/funasr-cpu-20261024，模型/home/wjj/tts-audio/model_cache/funasr_20261024；部署凭据runs/funasr_deployment_20261024_attempt01/及BM部署funasr。用paraformer识别内容、fa-zh给定文字对齐；原生Paraformer时间戳8条中6条越界，不能直接拿来切片。部署8条fa结构通过；短语阶段复用8条、新补20条，28fa结构通过但未人工验证。

BM路由使用project tts-exp和确切permalink；旧项目UUID b913836f-0910-4a4c-839b-986d19f12b6f曾误路由main，不能复用。短语实验及证据地图已记录完成，但实验笔记前半含旧“尚无结果”阶段描述，读取时以最终交付节及run为准。HANDOFF.md未追加本轮短语结果。

## Changed Files
新增未跟踪scripts/funasr_deployment_smoke.py、tests/test_funasr_deployment_smoke.py、scripts/experiments/qwen_phrase_local_20261024.py、tests/experiments/test_qwen_phrase_local_20261024.py、scripts/experiments/qwen_prediction_mechanism_20261023/root_phrase_audit_20261024.py。BM部署funasr、新短语experiment、研究主线与证据地图已维护。当前本地HEAD仍2b71639c3b555293375d248350999377168d515d，GitHub该SHA未验证可解析，不构造commit链接。

## Verification
父侧逐SHA绑定28原音频/视频/官方score/activesd，原矩阵整句C/D误差≤0.000496056；独立从FA字符时间、真实视频frame及音频215ms/视频200ms感受野重建315支持配置，912局部标量及1155配对标量误差0。所有31lag要求实际支持，排除padding和尾部补视频帧；实测28音轨均1105采样点处理延迟，84固定片块同峰值，一尾块相关0.797单列限制。时间映射非人工边界证明。
子代理三项RF/lag/padding/边界与缺失测试函数直接通过；Windows无numpy、远端无pytest，未运行完整pytest或改环境。FunASR部署两测试通过、pip check通过。20新ASR/fa三CPU批exit0。63展示MP4 ffmpeg exit0。attempt01取错transcript字段失败，仅写协议未算效果；attempt02修元数据入口完成，未按结果变更规则。原canonical音频运行后SHA不变。
本次恢复后重新读canonical BM、report.md及Git状态；GitHub只读确认仓库并查当前head的PR为空，无新科学实验。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- runs/qwen_phrase_local_20261024_attempt02/report.md
- runs/qwen_phrase_local_20261024_attempt02/index.html
- runs/qwen_phrase_local_20261024_attempt02/parent_raw_array_audit.json

## Observations
- [result] 三处跨draw和边界一致的模型间局部差异只涉及两个来源；其中下降的两处两个TTS仍各自优于自然音频。
- [decision] 局部量使用整track固定lag下的背景距离中位数减匹配距离，不称官方短语Sync-C、整句可加贡献、感知质量或因果机制。
- [decision] 保持四句事后案例范围和无训练状态，不据三次同号声称显著性或数字发音类别效应。
- [blocker] 缺跨文本复现、人工内容与边界核验及独立口型事件证据，来源/内容混淆未解除。
- [next_step] 先回答用户哪些共性可说及证据不足之处；若继续实验，优先人工核听/观看三个稳定片段，形成一个明确候选再用其它已有来源检验。

## Relations
- continues [[Codex checkpoint - 2026-10-24T01-27-11Z - Qwen paired gain heterogeneity and completed parent review]]
- summarizes [[Qwen双规模与自然音频的四句短语局部定位 2026-10-24]]
- informs [[研究主线与证据地图]]
