---
title: Codex checkpoint - 2026-10-05T03-28-20Z - Verified complete ASR and staged
  all40 production while preserving frame-map failure
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-05-t03-28-20-z-verified-complete-asr-and-staged-all40-production-while-preserving-frame-map-failure
git_sha: 16adacdf2a12d8e3c40d313c7a2d3c1cc8e57f6b
trigger: compact
capture: deliberate
agent: codex
model: gpt-6.1-sol
project: tts-exp
username: dogle
status: open
hostname: LAPTOP-E133GSLE
repo_root: E:/Documents/tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
cwd: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
started: '2026-10-05T03:28:20.2676833+00:00'
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-05T03-28-20Z - Verified complete ASR and staged all40 production while preserving frame-map failure

## Summary

完整核验了 60 条增强音频的 Qwen ASR 和 540 条视频的局部运动数据，保留评分汇总的实际格式失败，并将全 40 条原音缓存生产器密封部署到 GPU 服务器，继续推进可跨 TFG 使用的自然音频增强头。

## Story

用户最初要求恢复 TTS 增强 TFG 效果的研究路线，随后明确要求产出可用的自然音频增强头：增强 PCM 驱动视频生成，最终换回逐样本完全相同的自然 PCM 和时间轴，以自然音频评价。研究不能只停留于排除项或失败候选。主代理维护理论、协议、独立复核和 Basic Memory；实验与代码交给 GPT-6 Luna/max 子代理。阿里云 Qwen ASR/TTS、重跑、下载和网盘存储均获授权；凭据仅在私有运行时使用。

当前方法将频带增强分为局部稳定、上升、下降三种声学状态，保持相位构造和原音时钟，并投影全局 RMS。9 参数训练路线需要全 40 条自然音频缓存、RMS 对照几何、同一 GPU 的零参数/对照复现后，才执行冻结的 120 步优化。固定共享频率候选路线另有 540 条三模型开发视频，连同 identity/RMS 共 900 条待汇总。两条路线保持各自预算与来源。未见 80 条、40 位说话人与 LatentSync 的确认集继续封存，不能用来调参。

本轮把共享候选的 ASR05 六个完整文件实际下载并逐字节验 SHA。独立重算全部 60 个 API 结果：三个配置各 20 条、每配置 10 位说话人，CER 与相对自然 ASR 转写差异均为零，540 个计划行恰好映射一次。原自然 ASR 派生 summary/score 文件缺失，但原始转写存在且已独立计算；不需要重跑 API。60 个真实 PCM16 WAV 尚待只读下载核验。视觉档案的 551 个载荷成员、540 个 NPZ、22 个数值已完整核验；新增三方共同时间戳分析也全 540 条通过。这是局部运动描述，尚不是感知质量或最终方法获益结论。

CPU 汇总 attempt06 通过实际资源预检并启动 PID 626610，约三秒后 exit 1。实际错误是旧 baseline frame-map 与 crop_frame_count 的格式约束不符，尚未产生 900 行数值。子代理从旧 writer 和全 360 条基线元数据确认：时间戳、ROI、解码视频长度都是 N，frame-map 是 N+2，其中两行明确标记 nominal_clock_outside_render_decode。旧汇总器已支持这些行的语义，却错误要求 map 总长度=N。下一版必须保留完整时钟、源哈希与尾帧语义校验，并先收集全 900 条模式/元数据问题；不能删行或放宽来源约束掩盖失败。

全 40 条生产器和外层资源监督器经完整源码、静态语法、封装成员与请求字节复核，sealed build03 的 20 个源文件实际部署成功。预检/运行仍未执行，因为 /dev/shm 只有约 1.846 GB 空闲，而生产器要求实际 4 GiB。实测 cgroup 有约 29.7 GB RAM 余量，但没有 CAP_SYS_ADMIN，不能通过 remount 扩大 tmpfs。优先核验本地 E 盘已有完整归档对应的三处旧输出，再做精确退休；V4 未归档输出需要新只读状态/进程证据和无损备份。RAM memfd 原始字节归档是待实现的存储路线，尚未运行。其他约 13 GB 非项目共享内存不能动。

## Working State

持续任务是 [[开发跨 TFG 的自然音频增强头]]，研究笔记为 [[跨 TFG 自然音频增强头的训练与原音换轨验证]]；任务 active、方法 IMPLEMENTED，尚无冻结训练权重、跨模型确认结果或最终报告 26。Task60/Experiment59 已写入并直接读回，随后修复 ASR result-manifest 的一个 SHA 抄录字符错误并读回；旧不可变 checkpoint 未修改。

Git 分支 codex/cross-tfg-audio-head-20261003；HEAD 16adacdf2a12d8e3c40d313c7a2d3c1cc8e57f6b 为本地未推送提交。仓库直接 GitHub 查询证实当前无 PR；已推送可解析提交 c4a199890bd529928ec34b9be6da86a10a737c97。CONTEXT.md 为本轮修改，93 行；HANDOFF.md 更新为 48 行但属本地交接。历史五个已修改源码/测试不属于本轮；不要宽泛 git add。大量 scratch/源包尚未追踪，原始数据在 runs/ 与既有本地归档中。

机器局部私有 PowerShell PTY 50274 保有 SSH/API SecureString、bridge token 与端口；bridge06 PID252640，端口10887。私有调度脚本 .codex/scratch/dispatch_requests_root_attempt02.ps1 只有 -RequestPaths 参数，不接受 -Port。服务器 root@connect.weste.seetacloud.com:12349，host pin SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；远端工作根 /root/tts-research-system-20261001/cross_tfg_audio_head_20261003 与 /dev/shm/cross_tfg_audio_head_20261003。密码和 API key 不进入笔记、文件、命令行或日志。每次实际 SSH exec 八字段、下载五字段、原始字节长度与 SHA 均须复核。失败 prefix 不重用；只读响应覆盖前复制保留旧原字节。

三名 Luna/max 子代理继续运行：cross_tfg_execution 修复汇总07与原生 encoder168 的依赖兼容；local_recovery 准备三处已有归档旧 run 的只读退休审计和 V4 无损存储路线；mechanism_audit 准备 ASR WAV 只读包与 fit120 实际几何晚绑定封装。子代理没有私有调度权限。C 盘当前约 11.78 GB 空闲可作归档/恢复校验暂存，B 映射 CloudDrive 约1.259 TB；B 的大文件本轮未重新完成整文件哈希，不把目录存在当作备份证据。

## Changed Files

- scripts/experiments/cross_tfg_audio_head_20261003/build_fit40_natural_cache_rms_geometry_attempt02.py：62404 B，SHA 83a503da4b5576df5caed7cd4e4e73da00846d137c5cdec70cf9de941e7c42dd；封存生产器修复 qualified v2 导入、subprocess 与资源度量。
- 同目录 run_fit40_natural_cache_rms_geometry_attempt02.py：35021 B，SHA 165b382927791009322846c9f5125f0968507b631992bdb8c1f740112943e7ec；40 条失败分母、资源准入与实际状态外层监督。
- 同目录 run_fit40_native_transition_head_v3_attempt03.py：72032 B，SHA e0b56abfce15390cd7be7ed8facc9ac016a77f1b20d0dbabd8559096a016b021；只改一个 final-producer SHA 字面量，源码已验证，未执行。
- .codex/scratch/cross_tfg_audio_head_20261003/shared_profile_900row_join_attempt06/：资源修复源码、密封请求与实际失败证据；不覆盖这版产物。

## Verification

- runs/cross_tfg_audio_head_20261003/shared_profile_asr05_actual_readback_attempt01/downloads/：六个真实 ASR 文件全部传输/hash PASS。content_result_manifest.json 1939 B / SHA 3e02545b228924db73eadb97207c3c510928df18f79c53f0f24906ec1df326f4。
- 同 run 的 qc_attempt09/independent_asr_qc.json：42863 B / SHA ac69ca21205af699384440bb87fa6038f7da90e1803f255c21a957957596b3fa；numeric_qc_status PASS，60/60 内容通过；自然派生文件缺失单独保留。
- shared_profile_visual540_actual_readback_attempt01/downloads/candidate_visual_evidence.tar.gz：7825861 B / SHA e6e1067c1019fc1c0216356576e915b07ebd5feec3290fd93c8a3e6334cf7b5a；全成员/数组数值复核 PASS。qc_attempt01/independent_visual540_qc.json SHA d74c1826dc49d0d943bc947c0567bb49d574f11c8445a3c859dbae2d8c4de198。
- shared_profile_visual_pair_join_attempt03/triple_attempt02/triple_method_common_support_report.json SHA 93faabfb6948f2d6cabfa0c36952e6d0de4106c08a2b4eb349a6232f7c3b1352：同原音/画像/seed/模型、25 fps 的候选/identity/RMS 有效时间戳交集，540/540 PASS，无插值。
- join06 实际预检 SHA 023086c85dcef9550147f0253c544422fce141e9395c5af1c2b1979f625f9bc96；最新状态响应 SHA 38b8ffb39bd57569bb4d7a44362cabc3e182aaaadd2fc09fae97a1283d4f8752 为 FAILED/exit1/进程已退出，完整失败保留。
- .codex/scratch/fit40_native_rms_geometry_packet_attempt02/sealed_packet_attempt02_build03/request_index_attempt02.json SHA 7c4fa696f1e321b3620a4becd4dc73c0aaea894b9f338ad739e336b81bf53607；20 成员原始 SHA、7 请求 shell/Python 静态检查 PASS；真实 stage receipt 9665 B / SHA 9c908bf1ccdaac82a9e2ee5d5ef81957155f5a657b3a93895e4de33758122831，全 00–04 步通过，05–06 未跑。
- .codex/scratch/root_tmpfs_capacity_probe_attempt01/request.json.response.json SHA d079e121a0cfe7b1b81de625bc70865217ed7f19fb62c9d723aba52457aff93c：只读实际检查，无 CAP_SYS_ADMIN，内存与 tmpfs 额度分别记录。没有运行 pytest 或新增单元测试。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 本地未推送 HEAD `16adacdf2a12d8e3c40d313c7a2d3c1cc8e57f6b`；openspec/changes/develop-cross-tfg-audio-head-20261003/ 与两个已提交 executed replay ZIP 保留此前 native07/CLI02 的真实零参数资格证据。

## Observations

- [result] 60 条增强音频 ASR 实际完成且独立数值复算全通过；540 条视频局部运动数据已完整字节与数值核验。
- [decision] 保持自然 PCM/时钟终点评价、完整失败分母、开发与确认隔离；三方运动幅度不作为选择嘴型质量的替代指标。
- [decision] 旧帧映射的显式 nominal 尾行按真实 writer 契约验证；保留实际失败并使用 fresh attempt，不通过删行改变支持集。
- [decision] 原始 run 只有在完整本地归档 SHA、严格恢复、当前远端清单与消费者静止均通过后才精确退休；不清理其他项目的共享内存。
- [blocker] 900 行汇总07 尚待全元数据 census 和实际运行；完整 fit40 生产仍被实际 4 GiB tmpfs 准入门槛阻挡，训练120和确认未运行。
- [next_step] 审核并执行三处已归档旧 run 的只读退休审计，安全释放空间后运行已部署的 fit40 生产器和固定120步训练；同时完成 fresh join07 的全900行模式校验及实际汇总。

## Relations

- continues [[Codex checkpoint - 2026-10-05T02-46-07Z - Verified complete visual evidence and preserved the join storage failure]]
- advances [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]