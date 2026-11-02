---
title: PC-Talk中文20来源Sync-C与官方对照2026-10-31
type: task
permalink: tts-exp/codex/tts-exp/pc-talk-中文-20-来源-sync-c-与官方对照-2026-10-31
status: done
current_step: 4
steps:
- 冻结旧中文80输入及PC-Talk默认无表情配置
- 生成并验收80逻辑/41唯一视频
- 只运行SyncNet并独立核验
- 与官方I2V分数及协议比较并交付
completed: '2026-10-31'
---

# PC-Talk中文20来源Sync-C与官方对照2026-10-31

## Observations
- [status] done
- [description] 用户授权中文PC-Talk生成与评分，Sync-C后先检查官方差距；本任务只交付该检查点，其余八项未启动。HDTF TalkLikeYou仍PAUSED_BY_USER。
- [current_step] 4
- [result] 全20自然7.266、TTS7.756、Δ+.490、14/20提高；主要12为7.182/7.639/+.457、9/12。边际CI不是Holm9，无九项显著性结论。

## Steps
1. [x] 80 WAV/3.png与FlashHead输入逐SHA相同、RMS−24；PC-Talk100源码/12权重、SyncNet18源码/权重/guard钉住。
2. [x] 80逻辑/41唯一推理，512²/25fps/57–193帧；全部PCM/PTS/非静态验证，首对目视主体脸。
3. [x] SyncNet41唯一评分、80逻辑映射完成；原矩阵独立FP64最大误差2.563889e−6、stdout全吻合；本机6478内容文件SHA、41矩阵和三个来源分层/CI独立重算PASS。
4. [x] 正式报告81、逐视频/配对CSV、HTML视频对比和80标准播放媒体已交付；所有任务PID退出，GPU727回2MiB/0%。

## Context
权威报告[[81-pctalk-chinese20-syncnet-official-comparison]]。官方I2V HDTF9.370/MEAD8.190，本次自然低2.104/.924、TTS低1.614/.434；语料语言/脸/片长/电平与风格配置不同，不可直接归因评分错误。原生生成少194–237ms画面尾部，原始驱动PCM不动。默认lip1、推荐person192、无EMC，face-index1避免3.png背景误检，seed42每唯一WAV新进程；无新云调用/训练/按分调参。

重要边界：官方SyncNet run_pipeline默认内部AVI转MP3；首smoke N/T相对原PCM最大相关均+1105样点=69.0625ms，corr .986/.966；首N145原生帧→147官方处理帧，41最佳offset全−2。入口PCM精确不等于全链无损；未做内部强制PCM评分对照，不能量化此行为对C的贡献或解释论文差距。平均框几何全41覆盖主体，实际首crop目视通过。全部41轨迹59–195帧，其中14长度≤100；项目min_track25不同上游默认100。公开PC-Talk无作者评估脚本，未声称完整论文复现或感知真值。

run=/home/data2/wangjiajun/av-evaluation/jobs/pctalk_chinese20_rms24_20261031_attempt01；本机results/pctalk/chinese20_rms24_20261031_attempt01，local_audit.json/terminal_receipt.json为终局入口。证据包133628381B，SHA3604e0a57c20f4b5c5a680d4bd9f50a19059fb3c8cbd2fd4cfc9692b242b8c98。报告81是PC-Talk，不是暂停HDTF最终报告。档内generation_status RUNNING80是历史，不能重启。

源码scripts/experiments/pctalk_chinese20_20261031，测试tests/test_pctalk_chinese20_20261031.py，2PASS/compileall。生成前修复官方SHA清单前缀解析与浮点帧取整边界，未重跑科学前向。官方commit a0392744cbac857d96db1b99ceab70cf7b5f6c83，权重e6220816118bd8714ff57dc83e5906122b4e9168；SyncNet907c0b579c2e2d83f0eae1b2ac9e720cde4e5623。已退出PID19124/23870/23871/30362；不再自动launch任一阶段。

本轮MCP应用身份实际读取1048手册后运行，个人data2/nohup/单GPU727/CPU4与affinity0–3/生成Torch45%与评分60%上限；未改共享环境、驱动、他人进程。41生成+封装校验635.197秒，PID显存采样峰2350MiB。未提交，无PR，无网盘归档声明。

## Relations
- uses [[PC-Talk 192服务器部署]]
- produces [[81-pctalk-chinese20-syncnet-official-comparison]]
- compares_with [[79-talklikeyou-chinese20-nine-scorers]]
