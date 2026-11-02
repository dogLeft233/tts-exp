---
title: Codex checkpoint - 2026-10-07T12-31-54Z - Recovered migrated inputs and rebuilding
  same GPU training cache
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t12-31-54-z-recovered-migrated-inputs-and-rebuilding-same-gpu-training-cache
username: dogle
repository: dogLeft233/tts-exp
trigger: compact
hostname: LAPTOP-E133GSLE
capture: deliberate
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
model: gpt-6.1-sol
git_sha: 4ac55960f205cd2bfc0f99cae6698f1afc304528
started: '2026-10-07T12:31:54.8604583+00:00'
agent: codex
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
project: tts-exp
status: open
repo_root: E:/Documents/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T12-31-54Z - Recovered migrated inputs and rebuilding same GPU training cache

## Summary
服务器迁移后原始40条音频和模型文件仍完整，08训练源码已保留，但共享内存缓存丢失且GPU变化；主线正在按原设计重建缓存并完成局部谱形与功率包络资格实验。

## Story
用户最初要求恢复TTS增强TFG的原因研究，随后要求交付适配多个TFG模型的自然音频增强头，并持续迭代到完整可用的方法。增强PCM只驱动生成，评分及成片恢复精确原始自然PCM和原始时钟。根代理维护理论、协议、BM、Git、私密派发和独立核验；三个GPT-6 Luna/max子代理分别负责训练恢复、十一参数信号资格和Dev/ASR依赖。用户已授权重跑、Qwen ASR/TTS及网盘使用。

已有Dev RMS基线在三个模型等权平均提升+0.153602，简单共享谱形尚未达成相对RMS再增+0.050的目标。继续固定Fit40/20说话人的120步九参数训练，以及区分局部谱形和功率包络的十一参数探索。Dev20/10说话人×三种子×Wav2Lip/MuseTalk/Ditto；Confirm80/40未见说话人及LatentSync在方法冻结前封存。保留完整失败分母、D0/Canchor和内容/视觉保护、GUARD20短片完整cohort空值。

新服务器root@connect.westd.seetacloud.com:18625已实际认证，主机密钥与原机一致；物理GPU变为GPU-d09622d0-3535-e662-bb45-3df2b049fcea。RTX4080 SUPER报告32760MiB，实际查询没有计算任务。整个旧SHM命名空间在迁移后不存在，不能把旧GPU的几何/自然缓存当作新机资格。40/40持久源WAV完整文件SHA、61469字节Fit40清单及9项模型/源码/原始portrait/ffmpeg依赖全部实际通过；Wav2Lip和SyncNet checkout HEAD一致。六个旧context文件实际缺失。子代理正在严格归档查找原字节；如未存档，则以原portrait重新导出合格context并保留新来源，再按原40条设计重建producer。

08训练源包22成员、实际stage和CPU preflight均已部署并在新机完整SHA核验保留。绑定、launch和completion08实际不存在；尚无zero-head40、优化器更新、末步权重、终态重评分或新Dev结果。CPU十一参数packet08已实际上传三文件，解包因ZIP条目仅有权限位而被类型检查误拒；根代理独立读取23条目，全部无符号链接。attempt09未执行，子代理又发现其旧schema guard并封存，正准备attempt10四请求续接；worker03、config05和科学常量保持不变。部署失败不属于科学阴性。

## Working State
权威状态为Task79、Experiment78、CONTEXT.md及服务器reference；本地HEAD `4ac55960f205cd2bfc0f99cae6698f1afc304528`未发布。CONTEXT.md为93行，HANDOFF.md为ignored快照。五个既有dirty文件保持原样；其他未提交源版本、runs和scratch见 `.codex/scratch/checkpoint_ES_git_status.txt`，均属于本机状态。

迁移实际读回：`.codex/scratch/remote_migration_20261007_root_attempt01/01_read_migrated_current_state.json.response.json`，SHA-256 `3913d11cad83e6d0f560b3d0d9df0f7106dde85d81bd606af507bc6e40527f42`。40条及依赖实际核验：`.codex/scratch/fit40_newgpu_producer_attempt05_inventory_attempt01/00_readonly_inventory_attempt01_request.json.response.json`，SHA-256 `c74a2ded7b8dfde1de5330095579b5e036756ff773936912ae6432073decfc23`。

08源入口 `.codex/scratch/fit40_transition_head_training_packet_attempt08_revision06/`；CPU信号入口 `.codex/scratch/cross_tfg_audio_head_20261003/local_power_spectral_head_cpu_signal_attempt03/`。私密桥接为易失PTY57588/port12249，已审阅bridge07只改endpoint，凭证运行时持有、永不入文件/BM。远端持久NS `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`；共享内存必须新建。已闭合26项存储回收不再重做，overlay中的严格备份是恢复来源。

## Changed Files
- `CONTEXT.md`：记录迁移、新GPU、SHM丢失及重建路线，已提交4ac。
- `HANDOFF.md`：ignored当前快照，更新私密桥接和三个子代理分工。
- `basic-memory/docs/reference/server-access.md`：实际认证的新endpoint和旧机历史状态，不含密码。
- 已提交的末步候选物化及540行Dev联合分析工具保留；实际末步候选仍待训练。

## Verification
根代理完整审阅08训练源变更、ZIP全22成员及20未变成员；实际stage10261字节和CPU preflight574字节在迁移后SHA保持一致。新机40条原WAV及9项依赖完整哈希均通过。CPU08原始失败响应SHA `bb6c70ddd7f8a534609af03618f14595e7c7b56017138074608bb70d5100e608`；23条ZIP元数据均为权限0o600、S_IFMT0、无symlink，worker尚未执行。此前168原生编码产物和861完整40ms功率窗独立数值核验保留；它们不代替新机训练资格。未运行通用测试。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
当前分支GitHub查询无匹配PR；4ac/e5为本地提交。

## Observations
- [result] 新服务器40条源WAV及9项模型/源码/portrait/ffmpeg依赖完整，08训练源码与CPU preflight保留，实际08训练尚未启动。
- [decision] 在新物理GPU重新生产40条几何及自然缓存；仅恢复可证明原字节的旧context，缺失时重新导出并记录新来源。
- [decision] CPU解包修复只兼容无文件类型位的普通ZIP条目，仍拒绝symlink/危险路径；使用新receipt/control/output，不覆盖失败执行。
- [blocker] 六项旧context及几何缓存丢失，CPU十一参数实际资格和固定120步终态训练尚未完成；方法未冻结。
- [next_step] 审阅并实际派发CPU续接attempt10，同时恢复或重新导出context，在新GPU重建原Fit40 producer，然后完成zero-head40、固定120步训练与独立末步重评分。

## Relations
- continues [[Codex checkpoint - 2026-10-05T15-27-45Z - Verified factory content identity and fixed terminal candidate route]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]