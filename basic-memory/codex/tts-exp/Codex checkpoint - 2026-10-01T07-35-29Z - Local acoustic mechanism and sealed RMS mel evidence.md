---
title: Codex checkpoint - 2026-10-01T07-35-29Z - Local acoustic mechanism and sealed
  RMS mel evidence
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t07-35-29-z-local-acoustic-mechanism-and-sealed-rms-mel-evidence
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 2b78a03dfe32c4702dde15b573087563ee1e2cff
started: 2026-10-01 07:35:29.228573+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-01T07-35-29Z - Local acoustic mechanism and sealed RMS mel evidence

## Summary

独立复核已定位句级RMS响应主要经过两端均未裁剪的mel坐标；研究重点继续转到音素内谱包络、F0/清浊音、非周期性与局部能量的双向移植。

## Story

用户要求从本地BM、Git和百度盘恢复TTS增强TFG的研究路线并自主完成机制结论。root负责理论、证据边界与BM，三个gpt-6-luna/max代理负责科学代码与实验。最新用户指出音量/能量实验已有，要求寻找局部声学原因。保留已有RMS作为因果基线，重点推进已经冻结的WORLD音素内声源/谱形双向移植及局部分析；不能只停在排除项、实现或作业启动。

权威新结果已完成root下载、SHA和全文复核并写入BM：女性40句/20新说话人的生成端RMS G raw +.250266、unit+.034505；full-A时间null excess分别+.230148、+.031209，99%CI均正。女性原生T−N差距配对缩小的时间excess raw+.245392、unit+.035909均正。报告 `runs/recovered_20260930/independent_validation/female_cached_null_addendum_attempt02/independent_female_cached_null.json` SHA1bfa675eafce3595ac07b8a8d88d6d7f4efe75901ad092a3b8b447ddaf3da20a，女性实验BM状态concluded。

MEL独立07报告 `runs/recovered_20260930/independent_validation/mel_partition_consumer_20261001_attempt07/independent_mel_report.json` SHA112ff916b3c6b1790f3cf3db1b75721f08f84486aca0e16d828ec432bf4bc6fb，80句/40人/320四格完整。guard20 raw C：interior +.171139[+.101257,+.245350]、joint+.189832[+.116516,+.273877]、interior−boundary+.179846[+.109355,+.253011]；三者时间null excess也为正。raw/unit×四supports×C/C_anchor的16组均重复，boundary与I各支持99%CI跨0。interior定义两端均在(-4,4)，不是语音段；U/S剂量与坐标数不等，不解释成每坐标内在重要性或中介比例。MEL实验BM已concluded。

端点桥接320行完整，报告 `runs/recovered_20260930/independent_validation/endpoint_bridge_20261001_attempt01/endpoint_bridge.json` SHA4fcb2aae718a65cf7bb6c808c6c48b4fe8ad2d6e08312ece3b0e83c6d7386aea。旧axis→norm latent各80/80逐位一致，旧RMS→MEL latent有小差异；帧数一致但各新run pixels/V/A非逐位一致。分别引用每次run，不能宣称原始run精确重播。

RMS09真实潜变量方向/半径四格已生成80行/320 cells/1680 artifacts/0fail；consumer06因把metric float当support record失败，07仅修schema取值并由root及机制代理审阅，所有数学/门槛不变。07源码SHA327ee3ef38498ec2fe5004244910a9330ce960f9b27678a72de5b944fdc6af79，实际stage PASS、PID455527完成但consumer_exitcode=2；28,494,928B报告SHA6371c540e15ff9f1a0ba8a0bda21f7836be5d92305d100e20712c02bdaa9afe6未通过，正在只读错误诊断，未读效应。WORLD physicalQA04 1920WAV/1920NPZ/160contexts完整；GPUv4在任何前向前因run_path丢失package context失败。freshv5 wrapper3使用verified find_spec+run_module，CUBLAS在CUDA前、五flag实际prehook，worker SHAe9aac4b51fe3ecc90aab1ca0ee12614881de691bb9894ab1e5c9370e8e379528、wrapperSHA001b5eefa0233ab70dd4187e1d65365c004e35aba23290faee4ccc3f28271456；root实际upload返回hash PASS后dispatch启动。完整独立consumer和视觉/音素局部贡献仍待完成。

## Working State

权威图谱为项目tts-exp，已读Startup Router、实验指令与Coding Session。主Task [[解释 TTS 的 TFG 与 SyncNet 优势]] 活跃；女性/MEL结论、axis03、RMS09生成及端点桥接已同步，报告25尚未写。现有cache精确白名单已实际释放2692578028B；环境、权重、科学数据保留。AutoDL探查未发现现有百度/FUSE/rclone连接，本地B: CloudDrive约1.27TB可用，必要时完整归档/hash/读回/restore后转存。服务器root@connect.weste.seetacloud.com:12349为授权入口；密码/API key只运行时使用，不写入持久笔记。

脆弱本地状态：authenticated PTY50274、SSH daemon93533/port3019/token内存；dispatch helper保存每个.response.json，读取实际返回才推进依赖。8MiB下载上限，大报告gzip+7340032B分片。run和scratch均本地/ignored或untracked；已执行或ready bundle不可原地修改。CONTEXT91行、ignoredHANDOFF已更新但还需成功状态；既有dirty tests保留，未运行pytest。当前HEAD本地未推、无PR。

## Changed Files

科学源码consumer MEL07、female cachednull02、RMS07、endpoint bridge、WORLD wrapper3/GPU05/independent consumer；最终PNG见 `runs/recovered_20260930/independent_validation/mechanism_figures_20261001_attempt05/png_manifest.json`。图仅复制封存数值，root尚待视觉检查。BM女性、MEL、axis、RMS、主Task与storage Decision已更新读回。旧失败attempt保留。

## Verification

root实际检查上述完整报告SHA/全文、RMS07 stage和launch、WORLDv5三项stage实际返回hash，consumer修复diff及worker静态审核；科学结果对应独立矩阵/时间null/说话人bootstrap。新checkpoint前samechat18条/无更多、legacy0并直接readR身份，git root/branch/SHA/status/用户名主机现查；GitHub仓库、baseline commit与当前分支无PR现查。没有新增或运行pytest。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 本地未推HEAD `2b78a03dfe32c4702dde15b573087563ee1e2cff`。

## Observations

- [result] 男性和独立女性复现RMS生成端正向时间响应与配对native差距缩小，MEL U-only与U−S在16组支持/geometry/anchor/null重复为正。
- [decision] 用户最新重点为局部声学机制；以WORLD音素内SP、EX/F0/AP/UV及局部能量移植继续主线，RMS作为已完成基线。
- [decision] 独立视觉运动与SyncNet评分分开报告；不同run非逐位端点不得宣称exact replay；输入I与输出I单位不同，不直接相减。
- [blocker] RMS07独立科学报告、WORLDv5完整GPU/独立null consumer/视觉及音素局部定位仍待完成。
- [next_step] 读取RMS07实际结果并完成WORLDv5和独立局部声学分析，再以全部封存结果完成报告25及BM/Git交接。

## Relations

- continues [[Codex checkpoint - 2026-10-01T06-46-17Z - Completed RMS generation and independent mechanism consumers]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[句级音量的生成响应经过哪些 mel 坐标]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]