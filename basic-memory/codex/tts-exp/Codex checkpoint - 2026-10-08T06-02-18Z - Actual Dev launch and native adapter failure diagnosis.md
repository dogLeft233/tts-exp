---
title: Codex checkpoint - 2026-10-08T06-02-18Z - Actual Dev launch and native adapter
  failure diagnosis
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t06-02-18-z-actual-dev-launch-and-native-adapter-failure-diagnosis
trigger: compact
branch: codex/cross-tfg-audio-head-20261003
hostname: LAPTOP-E133GSLE
git_sha: 6126c7d1aeba5498f1f61042a49afe8f1e9ff59f
cwd: E:/Documents/tts-exp
started: '2026-10-08T06:02:18.5099711+00:00'
repo_root: E:/Documents/tts-exp
agent: codex
status: open
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
project: tts-exp
username: dogle
capture: deliberate
repository: dogLeft233/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T06-02-18Z - Actual Dev launch and native adapter failure diagnosis

## Summary
Dev360 对照实验已真实启动，首批60行中 Wav2Lip 20行通过、MuseTalk与Ditto40行快速失败；正在核对实际逐行错误并修复原生renderer的环境接线。

## Story
用户要求把中文TTS对TFG的增强机制转化为适配多个TFG的自然音频增强头，持续实验直到形成完整可用方法。增强PCM仅用于视频生成，最终必须换回原自然F32 PCM及原时钟；独立评分使用另有绝对时钟映射的索引ROI。此前Fit40候选已冻结，当前工作是独立Dev20/10说话人、3种子、Wav2Lip/MuseTalk/Ditto上的360行identity/RMS对照，再运行180行候选；Confirm80与LatentSync仍密封。

本轮修复最后一个CPU预检consumer错误：controls17只对QC06的三个原始收据做嵌套验证，另外两个源码/策略SHA保留顶层核验。collector10、builder16、candidate12、rebinder11仅重定向一致的不可变源码pins。实际recovery07全部8项请求通过，产生真实binding08/plan08/manifest/360行与preflight08。根代理和独立子代理核对完整原始字节、自哈希、61个唯一依赖、360有序Cartesian行及20份原F32 PCM身份，未改变科学协议或重新做native资格。

launcher06实际启动controls17，PID105669、start_ticks329309991，收据状态STARTED_DETACHED；这表示工程启动，不能当作科学结果。status07从binding namespace与controls17 helper正确派生实际嵌套leaf，首个真实快照见60行、20PASS/40FAILED。独立源码审核发现frontend08未在controls17清除环境之后补CROSS_TFG_BACKEND_ENTRY，两套M/D shim都严格要求它等于各自钉住的原生render_entry_path；Muse还没有CONTROL_DIR，Ditto则沿用旧共享路径。实际逐行诊断封包已审查并开始派发，等待输出确认原因，再做一次完整环境/收据契约修复。运行中的frontend08与图不修改，所有失败分母保留。

## Working State
持久锚点是BM active Task103、Experiment102及Git分支codex/cross-tfg-audio-head-20261003。HEAD `6126c7d1aeba5498f1f61042a49afe8f1e9ff59f`为本地未发布；16项本轮源码及真实CPU/启动证据已明确暂存、尚未提交。五个无关的tracked脏文件和大量既有untracked文件保留，禁止全量add/reset。HANDOFF最新入口尚需更新。BM Task/Experiment当前节已有CPU与启动，未加入本次首批失败。

易失运行仍在用户迁移后的服务器root@connect.westd.seetacloud.com:18625；密码和API仅私有runtime，永不写文件或BM。当前私有PTY57588与bridge07有效，诊断请求正在运行。实际controls leaf是 `/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360/controls360`，上一层是namespace；candidate leaf应是该namespace下candidate180。启动收据在job_control/dev360_phase2_recovery_attempt07/output/controls360_fit08_controls360_attempt17_launch01.launch_receipt.json。空间与进程必须实时读，归档前不清理SHM。

## Changed Files
当前源码链见 `openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/dev360_samehost_source_snapshot_attempt05.json`。已暂存collector09/10、builder15/16、controls16/17、candidate11/12、rebinder10/11、CONTEXT、精准.gitattributes规则和四份原字节provenance。`.codex/scratch/dev360_detached_status_attempt07/read_dev360_detached_status_attempt07.py`已真实部署、只读；诊断worker `.codex/scratch/dev360_failure_diagnostic_attempt01/read_dev360_first_backend_failures_attempt01.py` 为21191字节SHA55fb2bf3970235273df56f69e8346f8cc37dea73dc09a73dba11987d7109357d，只读日志/元数据、不打开音视频。

## Verification
recovery07全部8项实际请求PASS；六份实际下载raw pins一致，五份payload selfhash一致，完整360行跨文件关联、20 PCM及61依赖审核PASS。根QC记录 `dev360_binding_plan08_root_qc_attempt01.json` 1328字节SHA5ada7583530eb4148074e614668cac99bfd385a629da5a4dfa7ffb13f1582843。CPU summary4106字节SHA6484dfde2791ea7dbdd651200f764544f2fc63364249a95809a910e2eee12c7d。

launcher06全部4项实际请求PASS。完整启动收据5836字节SHA06578e7984a577056dbb040104a3075e197a29bcc954edc2dc79120d607ee484、self331dcbe6416455196abe9d32f2f12f51cb3ce3a57a272b4b6bea926335c23393已读回。status07首个请求PASS，PID/ticks匹配、execution身份valid、没有final manifest，20PASS/40FAILED。诊断封包index3983字节SHA3070d589715d394dd1b266677c88cfdd92dac3c13ee2bf662ea1f181fc39ef64；所有request/上传字节及Bash UTF8 stdin -n已检查，q00创建/q01上传已实际PASS、q02尚待读取。

暂存diff--check PASS；14份暂存.py/.json与工作树原字节一致。无通用测试、无新Dev科学结论、无候选/ASR API调用、无push/PR。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)，本次直接只读HTTP200核验。
- [已推送历史提交c4a1998](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)，本次HTTP200确认。
- 当前HEAD仅本地；本次GitHub分支PR搜索无匹配。

## Observations
- [result] 真实CPU07预检通过、GPU Dev360已启动，首批Wav2Lip20PASS、M/D40FAILED；尚无完整Dev或候选结果。
- [decision] 修复接线而保持shim严格断言、冻结候选、原音/时钟、全部计划行和失败分母；不覆盖运行源码或历史失败。
- [blocker] M/D实际错误日志尚待读回；源码已发现共同BACKEND_ENTRY缺失及CONTROL_DIR配置问题。
- [next_step] 读回实际M/D首个失败行与日志，确认后让Luna/max实现并独立审查新frontend09及一致fresh执行图，继续完整360→180与内容/视觉/GUARD20验证。

## Relations
- continues [[Codex checkpoint - 2026-10-08T05-28-23Z - Actual Dev binding and plan awaiting QC06 consumer correction]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]