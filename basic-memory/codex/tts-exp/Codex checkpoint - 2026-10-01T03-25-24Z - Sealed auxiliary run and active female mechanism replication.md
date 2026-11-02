---
title: Codex checkpoint - 2026-10-01T03-25-24Z - Sealed auxiliary run and active female
  mechanism replication
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t03-25-24-z-sealed-auxiliary-run-and-active-female-mechanism-replication
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 2b78a03dfe32c4702dde15b573087563ee1e2cff
started: 2026-10-01 03:25:24+00:00
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

# Codex checkpoint - 2026-10-01T03-25-24Z - Sealed auxiliary run and active female mechanism replication

## Summary

随机剂量 GPU 复跑全部封存，女性队列已实际完成 400/640 行；mel/WORLD 执行实现获独立静态复核，两个 CPU 消费者的实际字段适配失败已定位并保留，继续沿真实音量响应拆分机制。

## Story

用户要求从代码、Git、本地 BM 和百度盘恢复未同步的研究路线，持续研究 TTS 增强 TFG 的原因；root 负责理论与主线，代码和实验由 gpt-6-luna/max 子代理承担。已授权使用 Qwen TTS、下载数据、复跑实验和百度盘处理空间。服务器为 root@connect.weste.seetacloud.com:12349，密码/API key 只留运行时，永不写文件或记忆。男性100和女性40自克隆语音已经完成，不再生成。

已确立的正向证据是句级 RMS 干预在同一 RAW 自然评分音频、相同时钟下提高生成路径 C，并在单位化特征和第二张脸保留正向响应；独立视觉测量显示开口幅度与速度变化。静态 CAL TTS 均值方向的全域置换结果不支持该静态轴的主 C/anchor/search 正效应，不能把它外推为全部 TTS 声源或音素机制。当前用 mel 的严格内部/饱和坐标划分、真正 RMS 引起的编码器方向/范数分解，以及固定 RMS/音素时钟的 WORLD 双向移植确定路径。全面结论要依赖真实运行与独立消费，不能在排除项、预检或启动任务后结束。

随机 aux03 GPU 已完成80 clips/640 rows、零失败，实际模型 forward flags 全部 deterministic=True/cuDNN=True/benchmark=False；2889/2889 文件哈希独立核对通过。元数据检查器的 dose aggregate 为0 clips，因为它读取了错误字段；这不是零剂量错误的证据，完整 NPZ 剂量重建由独立消费者承担。消费者真实启动后 exit2，错误为 missing evaluation input manifest: /root，保留失败并交机制审阅修正新attempt。

女性 CPU02 预检 PASS，40 clips/20 speakers/640 planned rows。GPU attempt03 已实际运行，最近只读状态25 clips/400 rows/0 failed、1836个forward满足合同，视频/crop/JPEG计数一致、lip windows=N−5。服务器连接曾断开，后台 GPU 作业继续；新 SSH bridge 已恢复并核实实际产物。原生 cached-null CPU attempt01 的 top-level anchor 字段误读已改为 protocol.method.global_anchor_column=18；attempt02 又在行覆盖检查 exit1，保留原attempt，尚无科学统计结果。

mel 旧 preflight03 路径错误和 preflight04 CAL QA 缺 model_forward_performed 键失败均保留。root 实际下载读回 CAL QA/manifest 和 EVAL manifest：QA 无该键，但 manifest 明确 no-forward/effects=False；CAL40音频/160cell，EVAL80/320，payload/identity seals 一致。新 runner attempt04 只修正实际 schema 绑定，数学、严格N−5、完整V×A及逐forward确定性合同不变，独立静态复核 PASS。新源码与3个worker已上传；registry04和CPU preflight05尚待执行。WORLD最新 GPU evaluator同样静态 PASS，PSF无dist-info时只记录版本unknown，仍绑定精确3个源文件路径/tree SHA；CPU-only staging包 ready，尚未dispatch。WORLD physical EVAL的1920WAV/1920NPZ已经完成，仍需独立EVAL物理QA。

真正RMS producer冻结于adf42b...d808，CAL06实际20clips/10speakers、几何QA PASS、no-forward/effectsFalse；仅校准几何，不能解读为decoder贡献。独立RMS consumer第一版静态发现完成状态误把失败意向行算PASS，并缺少独立baseline bridge、逐forward flags、nonnegative/target/latent interaction报告；local agent正在出新版本，不能执行旧消费者作为最终验证。

## Working State

持久入口：CONTEXT.md为91行；Task“解释 TTS 的 TFG 与 SyncNet 优势”与相关Experiments笔记已更新并读回。分支codex/tts-mechanism-20260930，HEAD `2b78a03dfe32c4702dde15b573087563ee1e2cff` 为本地未推送。GitHub API实际确认仓库canonical URL，当前分支PR列表为空。未创建新线程、automation或goal。

机器本地脆弱状态：根实验目录 `/root/tts-research-system-20261001/`。新bridge端口3019，daemon exec93533，认证PowerShell exec50274的BRIDGE_TOKEN_PRIVATE环境变量已设置；token不落文件，旧8228 transport inactive。以client.py --json-file逐个请求，严格查ok/remote code再继续，SFTP upload不会创建父目录。女性GPU parent75809/child75811，control job_control/female_replication_attempt03，run female_replication/female_replication_20261001_attempt03；最新free6.484GB，仅单GPU排程female→mel→trueRMS→WORLD。大归档在本机C盘tts-exp-artifacts和B:\tts-exp-project\archives，云端同步不能由挂载copy自动推断。

aux03 producer run source_axis/random_specificity_aux_20261001_attempt03，独立元数据本地 `runs/recovered_20260930/independent_validation/aux03_metadata_attempt01/aux03_metadata_inventory.json`，50132B/SHA `1f2fdd9d9d4346fa77275c45c8686c02ea98de13c447c81134a2d6fafd3bf960`；results SHA `0e783920d5a7fae1883b005177ad467149c1896df263aab5d6973f76cf1977e3`。64位 materialization QA正确SHA `487bedd831e00012ebbf2524e5897454d57c098322868a35ac09c412948bae2b`。

失败消费者：aux03_attempt01 source `3f6646f4def5f3b3923b7291f93ddd7bd957e2d59468998ca4dd1acf6ff3c5a5` PID170665 exit2/control job_control/independent_aux03_attempt01；native_cached_null_20261001_attempt02 source `9f228083e3cc20bcf0f8f87e1db18b50d6e89e8646c8e166c98f343ebb9cb0dc` PID172971 exit1/control其attempt02/control。状态请求 `.codex/scratch/consumer_dispatch_root_20261001/05_status.json`，科学结果尚未形成。

mel请求索引 `.codex/scratch/mel_level_partition_20261001_attempt04/request_bundle_index.json`，runner SHA `3838790a8e4978d618fce4ca35308cac05f62c86c8f88d05ab79d2b14451eb94`。04 verify→05 registry build→06 status→07 CPU preflight→08 status，只有female640完整完成且GPU空闲才09 run；新registry04必须由新源码创建。namespace仍attempt03，control/gpu_preflight_attempt03和gpu_attempt03_artifacts原失败发生前都不存在，不覆盖旧失败control。

WORLD四请求 `.codex/scratch/world_gpu_attempt01_{mkdir,bundle_upload,stage_script_upload,stage_preflight}_request.json`，latest evaluator `f75e3cba9c40ea24807ba225ffcc29b4fa6a50bbdf9560e12c87b31fdecb2ddc`，bundle `7e6e7d85397ddd8de6829e12f1e771b6de355857cad0491d487cbd9ed334845f`；CPU预检必须隐藏CUDA。所有PSF调用使用isolated3-file parent PYTHONPATH，不用带numpy2的SyncNet site-packages。CUBLAS环境在CUDA前设置，worker LF-only+bash-n。

## Changed Files

root本轮只维护BM/操作请求/HANDOFF，没有改科学算法。HANDOFF.md为ignored本地快照，aux/native启动后最新失败尚未写入；需刷新。`.codex/scratch/consumer_dispatch_root_20261001/`六个root操作请求真实namespace/upload/launch均通过；agents科学源码多数untracked，旧tests也有dirty文件，不要stage-all。主要新实现位于scripts/experiments/tts_mel_level_partition_gpu_20261001_attempt04.py、tts_rms_latent_factorization_20261001.py、independent_rms_latent_factorization_consumer_20261001.py及portable_source_axis_20260930/的aux/native/WORLD/female代码。

## Verification

root实际读回各BM更改、CAL/EVAL schema字节与SHA、aux2889文件哈希、女性GPU状态与flags、两个CPU exit/stderr。agent独立静态复核mel04/WORLD f75 PASS；RMS消费者第一版审阅失败需新版本。源码compile、LF、worker bash-n及部署SHA checks已实际通过；没有运行pytest或新增tests。`gh`不在当前PATH，改用只读GitHub REST，canonical URL和PR空列表实际成功。

既有完成报告：norm全域02 `independent_norm_reanalysis.json` SHA `6659867e237adf065d87b6bfaaa7039518460960861458502ec44e751a9e97f4`，80/40/320/36756hash零失败，200个CI全部C/anchor/search excess跨0，10个B/D次要端点非零；native视觉supplement SHA `b713a1f0bfc5d38ce8684d52b4dccd70f9153757283e37cb3c00ba48d7923c39`，Traw−Nraw aperture+.010824、velocity+.074732，Nfixed−raw+.002899/+.022588，99CI均正。主RMS生成G raw guard20 C+.192073[+.111268,+.276076]，unit+.026961[+.016045,+.037865]；不把mouth motion说成音素/感知同步改善。

## References

[Repository](https://github.com/dogLeft233/tts-exp)。当前SHA为本地未推送，不构造GitHub commit link。实际实验产物与spec指针以本地/服务器目录为准。

## Observations

- [result] aux640已实际完成并封存，女性400/640正在运行；独立消费者实际字段适配失败，未产生合格统计结论。
- [decision] 保留failed attempts并新版本修schema，不能放宽剂量/时钟/旗标/完成性门限；单GPU顺序执行。
- [decision] 正向主线是实际音量→生成器→嘴部运动响应；mel和真实RMS latent分解回答路径，WORLD双向移植回答剩余物理差异。
- [blocker] aux/native consumers需新attempt修实际manifest路径/行覆盖；RMS consumer需独立工程QA补齐；mel/WORLD需CPU预检及后续真实GPU/消费。空间预算需沿百度归档和已核验可再生缓存解决，不删除科学证据。
- [next_step] 完成mel04 registry/CPU预检与WORLD CPU预检，同时修复aux/native独立消费者；女性640封存后按单GPU队列执行mel、trueRMS、WORLD，独立重建全部统计后写25综合报告。

## Relations

- continues [[Codex checkpoint - 2026-10-01T02-31-34Z - Completed null audit and active mechanism execution]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[随机对照的实际 float32 剂量复核]]
- relates_to [[女性说话人上的 TTS 来源方向独立复核]]
- relates_to [[句级音量在编码器中的方向与范数响应]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]