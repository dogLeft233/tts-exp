---
title: Codex checkpoint - 2026-10-08T14-57-59Z - Bound Python repair and complete
  controls
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t14-57-59-z-bound-python-repair-and-complete-controls
hostname: LAPTOP-E133GSLE
branch: codex/cross-tfg-audio-head-20261003
started: '2026-10-08T14:57:59.9052502+00:00'
project: tts-exp
username: dogle
capture: deliberate
repository: dogLeft233/tts-exp
trigger: compact
model: gpt-6.1-sol
status: open
git_sha: 0d3d675ce397718506801a984ad31408759464c7
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
agent: codex
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T14-57-59Z - Bound Python repair and complete controls

## Summary
360 条同机对照已完整读回并通过核验；候选启动的两次环境检查失败均有原始证据，新的启动器按实际 Python 身份修复并待最终审阅。

## Story
目标是把 TTS 的局部声学效应变成适配多个 TFG 模型的自然音频增强头。增强音频驱动生成，最终视频必须恢复完整自然 F32 PCM 与原始时间轴。用户授权持续实验、重跑、云 ASR/TTS 和网盘归档，代码与实验交给 GPT-6 Luna/max 子代理，ROOT 维护机制推理、协议和研究主线。

当前冻结方法仍是 Fit40/20 说话人的 Adam120 权重；Dev20/10 说话人、3 seeds、Wav2Lip/MuseTalk/Ditto。identity/RMS 共 360 条对照已全部 PASS，候选 180 条仍未启动。完整六产物、360 行身份及严格运行证明已由 ROOT 与独立核验器核对；全部终端输入 14 文件、16,044,666 字节也在 F 盘与百度盘本地 B 挂载两处完成双次完整字节读回。云同步状态尚未确认，媒体另有存储。

候选实际启动请求先被旧的计划自哈希常量拦住，再被启动器错误拒绝已合格的 venv Python 符号链接。两个请求均 FAILED_BEFORE_LAUNCH/code3，原始包与响应保持不变。实际只读探针确认 lexical venv 调用、解析后的 /usr/bin/python3.10 大小/哈希、strict12 与 bootstrap 全部匹配；唯一拒绝原因是 python_is_symlink_true。新的 launcher10 保留 lexical venv 调用并核对绑定的解析文件路径、字节数和 SHA；其五请求包在本地封存，尚未发出。修复不会改变冻结科学输入或重跑已经完成的对照。

RMS 对照的描述均值显示三模型相对 identity 的 ΔSync-C 分别为 Ditto +0.165、MuseTalk +0.053、Wav2Lip +0.238，等模型均值 +0.152。新建的局部分析正在独立核验置信区间与分解，初步分解显示改善主要来自负例基线 B，而非 Dmin 下降。当前 20 条 Dev 中一个短片段 BAC009S0022W0123 不满足 GUARD20，影响 9 个 seed/model 配对；保留完整 180 配对及全部 10 说话人，不以删短片段或九说话人敏感性分析替代原始完整支持。后续须安排时长合格的新验证批次，原 Dev 仍按冻结方案完成。

## Working State
持久锚点是本地未推送 Git HEAD `0d3d675ce397718506801a984ad31408759464c7`，分支 codex/cross-tfg-audio-head-20261003；58 个文件提交保存了全部对照终端核验证据与第一次候选启动失败，54 个复制产物已经按 worktree/index/HEAD 全字节核对。目录 `openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/controls360_attempt10_terminal_and_candidate_metadata04/` 的旧清单不可修改；新增 3 个备份回执附件仍未跟踪，需独立附件清单后精选提交。其他五个已有改动保持原样。

机器局部状态：当前服务器是 root@connect.westd.seetacloud.com:18625，host key SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0，RTX4080 SUPER。凭证只在私有运行时；不写入任何文件。私有 PowerShell PTY 57588、bridge07 与其私有环境值是脆弱状态。remote candidate180 输出叶子在两次启动前实际不存在；新的包再次检查。E 盘空间有限，较大归档使用 F/B。Confirm80/40 未见说话人与 LatentSync 仍封存，未运行 ASR API 或候选重复生成诊断。

## Changed Files
- `CONTEXT.md` 保持 94 行，反映完整对照与候选输入修复；已入 HEAD。
- `.codex/scratch/dev360_candidate180_detached_launch_attempt10/` 保存新 launcher/status、最小 diff 和五请求 index；SOURCE_ONLY，待 ROOT/独立审阅。
- `.codex/scratch/dev360_candidate180_python_identity_probe_attempt01/` 保存实际三请求响应与独立 QC。
- `.codex/scratch/dev360_controls360_rms_diagnostic_attempt01/` 保存协议、源码与局部分析产物，尚待 ROOT 数值核验。
- Task117 为 active，增强头实验116为 implemented；当前 phase 为 dev_controls360_full_readback_candidate_launch_guard_repair，需在实际候选启动后刷新。

## Verification
controls360 六产物的完整原始哈希、自哈希、14 源输入、360 精确行身份、6×60 设计、严格进程证明、正常退出以及空 NOT_RUN_SUBRUN repeat ledger 均 PASS。备份 F/B 各两次完整读回 PASS。候选 plan13/manifest14 实际读回，metadata04 实际 PASS_METADATA_ONLY/180。两个失败启动无自动重放、无候选 PID 或启动回执；第二次拒绝发生在元数据预检和 Popen 之前。实际 Python 探针 response SHA 为 `73e612143a12ac669dcec7ce44d5befb8b474d77945be2c859843ca438b6cc05`。尚未运行候选 GPU、ASR API、repeat6；未运行 pytest 或新增测试。

## References
[项目仓库](https://github.com/dogLeft233/tts-exp) 已用只读 GitHub 页面核对；当前分支查询无匹配 PR。当前 SHA 为本地未推送提交，不构造云端提交链接。

## Observations
- [result] 当前完整对照 360/360 PASS，可直接用于候选对比，无需重复生成。
- [decision] 允许已经绑定并实际核验的 venv Python 符号链接，始终保留 lexical venv 调用；解析路径仅用于绑定的文件身份检查。
- [decision] 保留原完整 20 clips/10 speakers 支持及短片段失配计数，GUARD20 不放宽。
- [blocker] 候选 180 运行尚无实际 launch receipt，新的 launcher10 仅本地封存；完整跨模型增强、原音换轨、ASR、独立视觉及重复生成端点仍待运行。
- [next_step] 完成 launcher10 五请求包的 ROOT 与独立审阅后，在私有 bridge 单次发出，读回实际 launch receipt，再启动正确引用 JOB10 的 observer 并维护 BM 任务。

## Relations
- continues [[Codex checkpoint - 2026-10-08T14-17-41Z - Full controls verified and candidate packet ready]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]