---
title: Codex checkpoint - 2026-10-07T14-15-13Z - Exported same GPU context and restoring
  full method execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t14-15-13-z-exported-same-gpu-context-and-restoring-full-method-execution
cwd: E:/Documents/tts-exp
trigger: compact
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
username: dogle
hostname: LAPTOP-E133GSLE
project: tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: c246d94a5bfabcaac2977fba6dadccd990031837
model: gpt-6.1-sol
started: '2026-10-07T14:15:13.8252552+00:00'
agent: codex
repo_root: E:/Documents/tts-exp
capture: deliberate
repository: dogLeft233/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T14-15-13Z - Exported same GPU context and restoring full method execution

## Summary
迁移后的确定性 context 已实际导出，CPU 局部声学构造与重复控制通过独立复核，正按新 GPU 的真实产物恢复完整训练和三模型验证。

## Story
用户要求持续做成适配多种 TFG 的自然音频增强头；增强音频驱动生成，最终音轨与评分恢复精确原自然 PCM 和原时钟。根代理维护机制推理、科学协议、BM/Git 与独立复核，代码和实验由 GPT-6 Luna/max 子代理承担。固定 Fit40/20spk、120 Adam 更新，以及 Dev20/10spk×三种子×Wav2Lip/MuseTalk/Ditto；接受要求三模型均正、等模型平均相对 RMS 再增0.050，保留 D0/Canchor、内容、视觉、完整失败分母和 GUARD20。Confirm80/40未见说话人与 LatentSync 继续封存至方法冻结。

服务器迁移到 root@connect.westd.seetacloud.com:18625，物理 GPU 为 GPU-d09622d0-3535-e662-bb45-3df2b049fcea。旧 SHM 消失，持久输入和 Wav2Lip/SyncNet 依赖完整。QC05 的真实重复失败保留；QC06 严格确定性设置通过完整视频、裁剪、轨迹和138×31评分矩阵精确重复。ADR015要求候选180与重新运行的自然/RMS对照360在这张新 GPU 上执行，旧归档只保留历史来源。

本轮根代理将 QC06 全27项证据复制到本地 runs，逐项核对字节/SHA，并实际加载 NPZ 复核4309个标量精确重复。另实际运行不导入 worker/head 的 physical_power_readback03：320项物理40ms摘要与861个共同正功率帧/单元全部通过；shape-only物理功率残差约0.035–0.147%，小于 envelope-only 的0.454–1.839%。这是局部声学构造测量，仍待 TFG 效果检验。

Exporter04及revision02 supervisor完整审阅后已实际上传并运行，13.764秒返回0、PASS_ENGINEERING_ONLY。实际导出141帧几何、25fps时钟和91520样本精确评分PCM。10项小文件已取回；15,622,636字节 context.npz 的整文件传输真实失败于8MiB bridge上限，失败响应保留。已完整审阅字节分块工具，下一步执行三块取回，拼接核对整文件SHA与五个数组seal，再晚绑定producer05和training09。不得将远端导出成功写成本地NPZ独立读回成功。

实际检查服务器46项后端依赖：24项存在且SHA匹配，缺12项Ditto资产、9项Muse资产和1项历史Muse报告。固定HF版本的Ditto恢复源码已完整审阅，尚未运行；Muse报告原字节已在本地找回，固定来源由子代理核对，恢复请求仍在准备。Wav2Lip训练可继续。当前无zero-head40、优化器更新、学习权重、terminal120重评分或新Dev/Confirm结果；后续若候选失败继续方法迭代。

## Working State
本地未发布 HEAD `c246d94a5bfabcaac2977fba6dadccd990031837`，分支codex/cross-tfg-audio-head-20261003。CONTEXT已提交QC06与ADR015；Task81 active、Experiment80 implemented已直接读回，阶段cpu11_and_qc06_qualified_new_context_pending。完整dirty/untracked证据见`.codex/scratch/checkpoint_EV_git_status.txt`，原五项dirty文件保留。忽略的HANDOFF、runs、scratch、私密PTY57588/bridge07及远端SHM属本地或易失状态；密码和API凭证只在运行时持有，绝不入文件/BM。

权威本地证据：`runs/cross_tfg_audio_head_20261003/local_power_signal_qualification_attempt10/` 的 numeric_readback04、physical_power_readback03；`runs/cross_tfg_audio_head_20261003/newgpu_deterministic_control_qualification_attempt06/` 的27文件及root_independent_repeat_npz_readback_attempt01.json；`runs/cross_tfg_audio_head_20261003/newgpu_engineering_context_attempt04/` 的已取回小文件。新context远端终态响应`.codex/scratch/fit40_newgpu_producer_attempt05_context_export_attempt04_revision02/04_read_context_export_status_attempt04_revision02.json.response.json` SHA `c0a51ae847b557313015c041ce17c1ce3a4b75c58f1e5886c6048e742782cbd9`；NPZ预期SHA `6678dce21f71693b2c3da7cc2eaf4f405d0983b293ff95165b4e43257dfbbc2a`。

cgroup约64.91GB可用、SHM约33.23GB、overlay约7.66GB、autodl-tmp仅188.76MB；按实际配额准入。B:为本地CloudDrive挂载，可作原字节备份，尚无远端可写云盘证明。producer05的历史540来源适配器仅用作来源验证，保留历史FAILED join与未执行repeat，不用旧GPU行作为新配对对照。training09的08→09差异已完整审阅，配置/binder/闭包等待实际context独立读回与producer最终source pin。

## Changed Files
- `CONTEXT.md`：实际QC06与新360对照决策，已提交c246d94。
- `scripts/experiments/cross_tfg_audio_head_20261003/export_engineering_context_attempt04.py`：未提交，已实际执行；SHA `cd5103349f8b0bd40ea455464065eaf4bc6202b5a568582b844b1188a4d12e1a`。
- 新训练09、producer05、chunk工具与固定权重恢复包仍为本地未提交源码；执行前根代理复核最终版本及请求。

## Verification
本轮实际执行physical_power_readback03返回0并读完整结果；QC06全27项持久副本SHA与NPZ精确重复独立PASS；context04实际supervisor/child返回0，原PCM seal通过，10项小文件取回校验。backend presence02实际返回完整46行/24哈希匹配。整NPZ超限失败保留；分块、本地数组读回、训练与权重下载尚未执行。未运行通用测试或新API调用。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
直接GitHub查询当前分支无匹配PR；c246d94及本轮其他提交为本地未发布。

## Observations
- [result] context04已实际导出并通过原PCM/时钟工程门槛，15.6MB NPZ仍待分块独立取回。
- [result] 独立物理40ms复核与确定性评分矩阵重复均通过。
- [decision] 新候选与360对照必须共享新物理GPU和严格确定性设置，保留原科学cohort和所有保护门槛。
- [blocker] context本地数组读回、Fit40缓存/zero-head40/120步以及Ditto/Muse权重恢复尚未完成。
- [next_step] 实际分块导出取回context.npz，核对整文件和五数组seal，以通过的实际产物绑定producer05并启动完整Fit40缓存。

## Relations
- continues [[Codex checkpoint - 2026-10-07T13-38-29Z - Qualified local power and same GPU deterministic controls]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[015-same-gpu-deterministic-controls]]