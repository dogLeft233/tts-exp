---
title: Codex checkpoint - 2026-10-01T01-00-59Z - Static factorization audited and
  RMS mechanism prepared
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t01-00-59-z-static-factorization-audited-and-rms-mechanism-prepared
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 70f124129147c8f5a8ca7a50ce55242752cea977
started: '2026-10-01T01:00:59Z'
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

# Codex checkpoint - 2026-10-01T01-00-59Z - Static factorization audited and RMS mechanism prepared

## Summary
静态来源方向拆分已完成独立复核；WORLD 的 1440 条双向主波形和 mel 的 320 个输入单元已经生成，下一轮要实际运行生成器，把已确认的音量响应追到内部坐标。

## Story
用户要求从本地代码、Git、BM 与百度盘恢复 TTS 增强 TFG 的研究路线，并自行推进到全面明确的机制结论。用户授权服务器、Qwen API、下载和历史数据，要求 root 维护理论与主线，实验代码交由 gpt-6-luna/max 子代理。本地 tts-exp BM 为研究记录，main 不重复项目笔记。Qwen 自克隆 male100/female40 已成功，不需要重复生成。

主实验 2560 行和独立二次分析已经确认：在评分音频及自然音频时钟不变时，固定 RMS 对自然语音生成路径带来 raw guard20 ΔC=+0.192（99% CI +0.111,+0.276），第二脸 +0.135，unit 距离也为正。这个生成端响应值得追踪。静态的 CAL 来源均值轴本身没有达到增益标准；完整与中心窗口的原生 N/T 比较方向不同，故最终报告必须同时呈现 full/valid/guard15/guard20，不能挑选正窗口。原生可见口型、配对 null、音量 mel 分区、真实 RMS 编码轨迹及 WORLD 双向物理移植承担后续解释。

本次核对明确旧角度/范数实验是 natural_z_fixed→axis_z_orth_plus，不是 RAW→FIXED 音量轨迹。320 行独立消费者完成；方向、范数、joint 与 interaction 的 raw/unit 四支持域区间均跨零。另冻结真实音量编码轨迹的独立方案，使用 mel q00/q11 保存的 encoder 数组。输入的 direction×radius 本身有乘法交互，输出四格交互不能单独证明解码器非线性。

## Working State
Durable science: `runs/recovered_20260930/independent_validation/norm_consumer_attempt07/independent_norm_reanalysis.json`（9442167B，SHA256 1fca337e66c6924d241040e666e6873a7f3a033c48e4b405dc697acb6a4d204b）由 root 解压校验并直接读完关键估计。80 clips/40 speakers/320 rows，36756 all-files 项、320 score 重建，零失败；四项方向主门槛全未通过。方向 q10−q00 C=-0.013（99% CI -0.035,+0.010），范数 +0.006（-0.021,+0.033），joint -0.004（-0.024,+0.015）。追加 all-domain 缓存 null 与旧/新 q00/q11 pixel/V/score 工程桥接是单独只读补充，07 不修改。

WORLD CPU 输出 `/root/tts-research-system-20261001/world_phone_transfer_20261001_overlay/out/world_physical_eval_attempt03` 完成 exit0、stderr 空。80 样本、160 recipient directions、9 arms，每 arm160个 ok；原始 MFA79/80、仅物理 overlay80/80，1440主波形+320QA+160固定源=1920WAV，约6.6GiB。manifest raw SHA 27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3，plan f92dbcfc71ecb775d98c3cd33ec19b4dd99d206c27d155eba8ae9bc51204861e，root freeze-v3 0a5b694975efeb07b9d1e9c0beb10f0142bb4203ba4a026d7ff965477991ab9d。还没有 WORLD 模型响应。GPU 设计为每recipient10视频/36唯一V-A矩阵、76显式标签，总1600视频/5760矩阵/12160标签；RAW 与 within-WORLD 两四格族加 fresh unsynthesized FIXED 生成桥接。bridge q00=(V_FIXED,A_RAW), q10=(V_RT,A_RAW), q01=(V_FIXED,A_RT), q11=(V_RT,A_RT)，2个新矩阵+2个既有矩阵别名。

mel CPU producer `tts_mel_level_partition_20261001.py` SHA 02d3f858c623c6930b95b366729c511856491b16f61ca685c88c69f2f52e80a4。CAL20 pairs/40 waveforms/160cells exit0 QA PASS。EVAL80自然样本/320输入格亦完成 CPU-only，164文件ledger精确，80 masks/partitions独立复算，零失败、没有forward/effect。私有根 `mel_level_partition_20261001/{calibration_attempt01,evaluation_attempt01}`；EVAL plan ab6d713246ed6a9f11ecfcea08be2aae791a23e240ee4ab5791e31745be17f66，manifest4f4b9bfbb1b51cb0e75585126ba3bde1c87d40bb444ea0581dc9512e0568a3f9，ledgerd9d52ced138ce13df0602b1a14ecf7d9646df41db80af1d86e80e1530a970172，independent audit e71b71be4ec8b6e34bb4c8f2da8f7390725c1e1836225dfd78a12da980047f41。GPU runner 尚待实现。必须保存 q00/q11 exact encoder z及实际encoder flags，以接真实 RMS 方向/范数实验。

Native320视频 local C档案完整校验；MediaPipe 提取320/320、零失败。`native_geometry_visual_20261001_attempt02` 的 v2 availability 在读取 aperture 前冻结；原MFA79/80，四格全部80/40，phone主可用52clips/16完整speakers（T_FIXED contrast53/17）。代理已提交 v2 report72e7f68df2920a062f8703855bdc6d994f59223851f266e5a94713356e9ac341：所有99% paired-phone separation/AUC区间跨零，root尚待直接读数。首版 clock map-key raw/baseline问题仅静态阶段发现并保留，v2源7311aecb3ef0d6e622b40254f595003d60d7964dee17a4d207b7012d0403e4b2。缓存 native full-A null/padding诊断仍待做。

Aux640 corrected-random runner8b809abd1e67afdda8e0d2f2f570f8a82c63d95c8fe149c368638e4f622452e6独立静态PASS，F32 QA80/240/22428，零剂量miss。Root 实际执行 preflight04 exit0 PASS；用正确 inputs/primary_face.png 与 secondary_face.png，原code不变。启动04失败且未运行模型：df awk科学计数值无法shell整数比较，execution_assets Python缺python_speech_features。remote代理正在新05启动封装修复纯python包隔离，保留04；不能说auxGPU已运行。Female40clips/20speakers/640rows一主脸adapter1bad9ac0f5dbde450eeaba89784c3aeddfe226de9976a88a4f7adb7d1d6ec363静态PASS，package8e8320c1f4c9bf9f57581155eda71d33eae3e705c939b4c8fe6d0611b87528a9及CPU preflight请求已准备，未dispatch。

Machine-local/fragile: root 的 authenticated PowerShell PTY exec50274 由 root 操作，子代理无法跨task访问；bridge daemon13803/loopback8228，秘密仅runtime/original user message，不写文件。准确 endpoint=root@connect.weste.seetacloud.com:12349，BM已有无密码引用。非秘密 request JSON 在 `.codex/scratch/`，下载/上传 keys remote/local，payload≤7.5MB。GPU4080SUPER32760MiB；preflight04时GPU1MiB/idle，overlay约8.9GB空闲。所有新输出只在 `/root/tts-research-system-20261001/`0700，旧sealed数据不删除；仅验证的两处纯conda下载cache可按hash/path边界清理，尚未清理。新增本地档案放 `C:/Users/dogle/.codex/tts-exp-artifacts/20261001`，repo留transfer mapping。Core301shards最新已知267/301，不能默认全部到齐。

## Changed Files
- Local unpushed HEAD70f124129147c8f5a8ca7a50ce55242752cea977 冻结 `openspec/changes/confirm-rms-latent-factorization-20261001/spec.md`，100行，SHA3eaf572afd83682e5dc9d09dc325313bc2e16b969183fc8f89d391424b861e23；独立静态PASS。
- `openspec/changes/confirm-mel-level-partition-20261001/spec.md` 已在local326bb7bad35f198855766b57b46e2f0e90b03ee0，abeced0f6f50a94fe49fc085af40358071dacc5091c7c796c26c5edafbe63ee8。
- 多个消费者、aux/female/native/mel/WORLD scripts仍dirty/untracked；旧tests dirty没有运行或添加新测试。不要stage全部。`CONTEXT.md`91行，状态应刷新；`HANDOFF.md`缺失，最终写current snapshot。
- BM 主线任务及各实验已有笔记，其中 WORLD/random/female/native/mel 的当前状态仍需按实际结果更新；旧immutable checkpoint不改。

## Verification
Root核对git status/branch/root/HEAD、BM Router/实验指令/schema、same-chat9个checkpoint/legacy0、前一个直接读回身份。GitHub只读API确认canonical repo与head PR_COUNT0；pushed c4a199890bd529928ec34b9be6da86a10a737c97可解析。Root直接审计norm完整consumer和aux04 preflight。代理完成mel CAL/EVAL数组与ledger QA、WORLD九arm×160资格以及native320提取和v2 availability冻结；后两类仍须root读取报告。未运行pytest。两次aux启动资产/依赖失败是工程失败，不是科学阴性。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- main independent consumer: `runs/recovered_20260930/independent_validation/source_axis_consumer_attempt08/independent_consumer_summary.json`
- corrected full/support/search: `runs/recovered_20260930/independent_validation/secondary_reanalysis_20261001_attempt03/independent_secondary_reanalysis.json`

## Observations
- [result] 自然语音固定RMS的生成路径响应在两张脸及unit geometry复现；静态来源轴与其角度/范数拆分均未达到预设正增益门槛，两个问题必须分别记录。
- [decision] WORLD保留9arms双向、RAW/within-WORLD完整四格和fresh FIXED桥接，优化共享缓存而不删科学单元；记录36unique/76alias perrecipient。
- [decision] 真实 RMS latent factorization使用保存的RAW/FIXED实际z，不替换为CAL来源d/u；input交互与decoder非线性不混称。
- [blocker] Aux05启动修复还未执行；WORLD/mel/RMS/female GPU响应及native cached-null尚未完成。旧/新norm engineeringbridge与all-domainnull补充还在处理。
- [next_step] 修复并实际启动aux640 GPU，同时完成WORLD evaluator/mel runner代码，随后依序跑满所有机制单元并独立读取复核。

## Relations
- continues [[Codex checkpoint - 2026-10-01T00-01-01Z - Sealed norm experiment and physical mechanism execution]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[编码器方向与范数对生成响应的分离]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]
- relates_to [[句级音量的生成响应经过哪些 mel 坐标]]
- relates_to [[句级音量在编码器中的方向与范数响应]]