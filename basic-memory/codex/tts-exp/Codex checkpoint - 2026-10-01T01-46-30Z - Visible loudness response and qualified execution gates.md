---
title: Codex checkpoint - 2026-10-01T01-46-30Z - Visible loudness response and qualified
  execution gates
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t01-46-30-z-visible-loudness-response-and-qualified-execution-gates
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 2b78a03dfe32c4702dde15b573087563ee1e2cff
started: 2026-10-01 01:46:30+00:00
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

# Codex checkpoint - 2026-10-01T01-46-30Z - Visible loudness response and qualified execution gates

## Summary
全 80 条样本的独立视觉配对分析确认句级音量改变实际嘴部运动幅度；随机剂量辅助实验的完整 CPU 材料化已通过，下一步实际执行 GPU 并把响应追到 mel、编码器与物理声源坐标。

## Story
用户要求恢复本地代码、Git、BM 与百度盘上的 TTS 增强 TFG 研究路线，并自主推进到全面明确的结论。root 负责理论与实验主线，三名 gpt-6-luna/max 子代理负责代码、远程执行与独立审计。male100/female40 Qwen 自克隆完成；原用户消息中的服务器及 API 凭据只用于运行时。最新授权允许用百度盘解决空间问题，当前服务器入口始终是 root@connect.weste.seetacloud.com:12349。

主实验已确认评分音频和自然时钟不变时，固定 RMS 的生成路径 raw guard20 ΔC=+0.192，99% CI [+0.111,+0.276]，第二脸 +0.135，unit 也为正。静态 CAL 来源轴与其角度/范数拆分没有达到预设增益门槛；它们不是 RAW→FIXED 的真实音量编码轨迹。完整与中心支持窗口的原生 N/T 比较方向不同，必须共同报告 full/valid/guard15/guard20，不能挑选窗口。当前推进正向链条：音量→mel 分区→实际 encoder z 的方向/范数→生成嘴部运动；另用 WORLD 双向移植定位声源、谱形与局部能量。

本轮新增的可见运动证据来自全部 80 clips/40 speakers：自然 FIXED−RAW 的归一化开合均值 +0.002899，99% CI [+0.001844,+0.003940]；绝对开合速度/秒 +0.022588，[+0.014555,+0.030767]。TTS FIXED−RAW 开合 -0.001254，[-0.002108,-0.000396]；速度 -0.007101，[-0.013010,-0.001405]。原生 TTS−N 的开合 +0.010824、速度 +0.074732，均为正。音量确实影响实际输出运动；该证据不等于音素判别或主观同步质量。phone separation/AUC 在预先冻结的 52 clips/16 完整 speakers 子集（T_FIXED 53/17）中所有 99% CI 跨零；原始 MFA79/80，不混用 WORLD 80/80 overlay。v2 median 字段因聚合键不匹配没有测得，原报告不静默修补。

## Working State
Durable evidence:
- `runs/recovered_20260930/independent_validation/native_geometry_visual_20261001_attempt02/analysis_v2/native_geometry_contrast_supplement.json` SHA b713a1f0bfc5d38ce8684d52b4dccd70f9153757283e37cb3c00ba48d7923c39；原始 phone report 72e7f68df2920a062f8703855bdc6d994f59223851f266e5a94713356e9ac341；root 直接读取了完整关键估计与来源封印。
- 主实验 core 301/301 分片现已下载并校验，总 1587388518B，missing_or_mismatched=[]。`runs/recovered_20260930/remote/evaluation_20260930_retry3_analysis_20261001/analysis_transfer_manifest_verification.json` SHA a0fce1e7e62d95d6ac15a70677a70c859d428b07e8f51046e5a6ae1a751172ce；manifest 98350b737fcb40434da7fb653b9560869adaf4c9d85cc070f6c55939df47dd73。尚未全量解包，source_path 为恢复映射。
- 静态 norm 独立结果 `norm_consumer_attempt07/independent_norm_reanalysis.json` SHA1fca337e66c6924d241040e666e6873a7f3a033c48e4b405dc697acb6a4d204b：80/40/320，36756 文件与 320 score 重建，0失败。全支持域补充 attempt01 的 null 错用了 guard20 全 lag invariant，保留失败；v2 be02e69cb45bd71d763034589ff124bb078ed95174527a0be2815eb1b745cd04 的 attempt02 已实际启动 CPU PID863274，6 seals/320 scored metadata 预检 PASS，待状态和结果。原 launch_record 可能含字面反斜杠 n，需保留原件并写 SHA 绑定解析侧车，不能改科学原件。
- WORLD CPU `world_phone_transfer_20261001_overlay/out/world_physical_eval_attempt03` 已完成1440主波形+480QA/固定源，80 pairs/160 recipients/9arms 全部ok。manifest27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3，plan f92dbcfc71ecb775d98c3cd33ec19b4dd99d206c27d155eba8ae9bc51204861e。GPU全网格尚未执行；别名设计冻结在 `openspec/changes/confirm-world-source-filter-20261001/evaluation-aliases.md`，SHA d14260e47869003756a0efd6e84719011c0a2550b3af07ead5f62b902fb22f3f：1600 videos/5760 unique V-A matrices/12160 aliases。RAW和 within-WORLD 四格均保留，bridge 是 fresh FIXED→WORLD RT，不是 RAW→FIXED RMS。
- Mel CAL40 waves/160cells 与 EVAL80 N/320cells CPU QA均完成，无模型 forward。EVAL planab6d713246ed6a9f11ecfcea08be2aae791a23e240ee4ab5791e31745be17f66，manifest4f4b9bfbb1b51cb0e75585126ba3bde1c87d40bb444ea0581dc9512e0568a3f9，array audite71b71be4ec8b6e34bb4c8f2da8f7390725c1e1836225dfd78a12da980047f41。GPU runner 当前子代理报告 SHA c4241fbc8961e920f89ede812193a51789138070663b526c3b1077b6df348b5a，需 root 核验并独立静态审查。root 刚发现流式 visual_features 可能产生 N−4 窗口而官方/旧 helper 使用 N−5，须检查/修复再放行。应保存 q00/q11 exact z 与实际 flags，后接真实 RMS factorization（spec3eaf572afd83682e5dc9d09dc325313bc2e16b969183fc8f89d391424b861e23），其 code 正在写。
- Aux v1 worker 在 context JSON 写前求 SHA 而退出，0视频/forward。v2 `random_specificity_aux_20261001_v2.py` SHA792e0522572f5d02f476519d85139849af472348f723bd3c730a090a530a43e7，只修写/hash顺序并加 CPU-only materialization，数学不变。完整材料化 attempt06 已 exit0：80contexts/640row refs/880array hashes/0integrity；QA487bedd831e000012ebbf2524e5897454d57c098322868a35ac09c412948bae2b，inventory4fdd955d6e9b759211035bb030bd474a3799a43ed0233426e1b5ad8e1e0ebd46。root 聚合3随机 arms noncapped dose misses=0、capped=0，独立静态PASS。fresh GPU attempt03 launcher42cf0664f25aaea44aac4cfd1ee4a2a96d782c965ab809ada395cc2dcaca57e3 ready，upload/dispatch requests `.codex/scratch/random_specificity_aux_attempt07_*launch_v2.json` 尚未执行，不能称GPU已跑。
- Female40 clips/20 speakers/640 rows 单主脸adapter1bad9ac0f5dbde450eeaba89784c3aeddfe226de9976a88a4f7adb7d1d6ec363静态PASS，package8e8320c1f4c9bf9f57581155eda71d33eae3e705c939b4c8fe6d0611b87528a9待CPU preflight/GPU。Native全 A cached null/padding独立消费者正在实现。

Machine-local/fragile: root authenticated PTY50274、loopback8228 bridge13803仅 root可操作，秘密不入笔记。纯 PSF0.6 private parent `/root/tts-research-system-20261001/deps/python_speech_features_only_attempt05/` 仅3个源py文件、treeSHA a419b997acfd21da8f4c7324e5be22e32357d085e37ac3376165e676c1e67c17，排除 pycache，实际np1.26.4；不可把 SyncNet np2全site-packages塞入W2环境。primary face 是 `/root/autodl-tmp/tts-research-20260930/portable_source_axis_20260930/inputs/primary_face.png` SHAe3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6；norm08 face metadata8ada80c9c7a3628edef7428e9f218b475839a8a4f03079dcd9794e45ad0c4bbb，generation box[168,88,381,385]，score box[59,16,489,446]。新输出只在 `/root/tts-research-system-20261001/`0700，8.9GB左右空闲。旧科学产物/活跃输入不删；2.6GB已验证纯下载cache尚未清理。

本机 `B:/tts-exp-project/archives` 可访问；本轮只读发现 weste 无匹配网盘挂载/客户端/PATH工具，未读取密钥或安装客户端。可经本机B挂载归档 immutable shards，保存 SHA/member/source_mapping/recovery manifest，消费者用完且归档验证后再释放副本；挂载写入不直接等于云端持久化确认。新大归档放 C:/Users/dogle/.codex/tts-exp-artifacts/20261001/，repo保留映射。

## Changed Files
`CONTEXT.md`91行与 WORLD alias annex 已提交 local/unpushed 2b78a03；`HANDOFF.md`56行忽略文件、不是Git提交。多个独立消费者、aux/female/mel/RMS/WORLD脚本仍dirty/untracked，旧tests dirty未动/未跑，避免stage all。BM主任务/native/mel/serverguide已更新，random/WORLD/norm Observations尚需按实际执行刷新；此前 checkpoint保持immutable。

## Verification
Root核验branch/root/HEAD/status，加载Router/实验指令/CodingSession schema，same-chat10个checkpoint、legacy0，直接确认前件身份。GitHub只读API返回head PR=[]，canonical仓库与pushed baseline可解析；PowerShell `@(Invoke-RestMethod [])` 曾误计1，直接JSON纠正0。Root直接审计native paired report、core301验证与aux材料化剂量计数，实际执行norm02 CPU启动。未运行pytest；工程启动失败分别保留，不充当科学阴性。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local unpushed current commit `2b78a03dfe32c4702dde15b573087563ee1e2cff`
- Main source consumer: `runs/recovered_20260930/independent_validation/source_axis_consumer_attempt08/independent_consumer_summary.json`, SHA55988932b41c0410f10acdb87db43ac30d21315d9ef1a5e06f423462c5403f1c
- Corrected search/support consumer: `secondary_reanalysis_20261001_attempt03/independent_secondary_reanalysis.json`, SHA768934c1ab011bf12dfb53d0ec49592802895d74018c594b1f4f088e39ccc86a

## Observations
- [result] 固定RMS不仅改变SyncNet评分，还在全80/40配对中改变实际嘴部开合/速度；音素判别与主观质量单独取证。
- [decision] 使用真实RAW/FIXED编码轨迹定位音量路径，输入direction×radius乘法交互不混称decoder非线性；静态来源轴单独记录。
- [decision] 百度盘作为可验证归档通道，保留恢复索引与活跃输入，未确认持久化前不释放唯一副本。
- [blocker] Aux/mel/RMS/WORLD/female模型响应、native cached null、norm all-domain补充及工程桥接尚未完成；mel visual窗口边界需审查。
- [next_step] 实际启动通过完整CPU剂量门槛的aux640 GPU，同时修正并审查mel frontend，随后跑满机制网格并独立读取结果。

## Relations
- continues [[Codex checkpoint - 2026-10-01T01-00-59Z - Static factorization audited and RMS mechanism prepared]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[原生 TTS 优势的时间配对与可见口型诊断]]
- relates_to [[随机对照的实际 float32 剂量复核]]
- relates_to [[句级音量的生成响应经过哪些 mel 坐标]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]