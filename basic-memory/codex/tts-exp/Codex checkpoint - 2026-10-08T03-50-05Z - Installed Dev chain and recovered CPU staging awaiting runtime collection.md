---
title: Codex checkpoint - 2026-10-08T03-50-05Z - Installed Dev chain and recovered
  CPU staging awaiting runtime collection
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t03-50-05-z-installed-dev-chain-and-recovered-cpu-staging-awaiting-runtime-collection
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
agent: codex
project: tts-exp
hostname: LAPTOP-E133GSLE
trigger: compact
cwd: E:/Documents/tts-exp
model: gpt-6.1-sol
branch: codex/cross-tfg-audio-head-20261003
git_sha: 96b47a3a3e530ebfc34510abeb4dc23b8799b132
repo_root: E:/Documents/tts-exp
started: '2026-10-08T03:50:05.2050178+00:00'
username: dogle
repository: dogLeft233/tts-exp
capture: deliberate
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T03-50-05Z - Installed Dev chain and recovered CPU staging awaiting runtime collection

## Summary
冻结的三模型 Dev 执行链已实际安装，CPU staging 33 个源文件逐一核对通过；当前宿主模板下载核对完成，正准备实际绑定和 360 条对照启动。

## Story
用户要求从 TTS 增强中文 TFG 的机制，推进到适配多模型的自然音频增强头，并持续实验直到方法完备。root 管理理论、冻结协议、BM、Git、私有 SSH 和独立复核，代码及实验委托 GPT-6 Luna/max。增强 PCM 驱动视频，最终成片恢复完整原始自然 F32 PCM 与源时钟。ADR007/016 授权带绝对源时钟映射的 selected-track PCM16 评分切片。

冻结 Fit40/20 speakers、Adam120/3 passes 候选。Dev20/10 speakers ×3 seeds×Wav2Lip/MuseTalk/Ditto，共360 identity/RMS控制及180候选；要求三模型正向、等模型相对RMS均值+.050，并保留D0/Canchor、内容、视觉、GUARD20与完整失败分母。Confirm80/40未见说话人及LatentSync保持封存。strict12 Wav2Lip 和各自strict09 MuseTalk/Ditto 原生重复、20条Dev原音及原生帧映射均已root独立复核，不需要重复资格实验。

ADR017 保留实测pilot与容量外推的区别：17,604,091,672 B 显式场景加独立5 GiB，初始540准入、剩余阶段及逐行实时GPU/RAM/SHM门；最终完整可恢复归档仍必要。本轮完成core06、binder06、controls13、candidate guard08等一致源码审查。发现并修复controls13严格资格分支的未定义record/read_regular引用，以及controls13/candidate08应绑定Wav2Lip原生native_entry_path的约定；工程修复没有改变候选、评分或Dev选择规则。

最终source_snapshot_attempt01.json封存24个closure角色，21个本地唯一源码root实际SHA/大小核对通过；signal独立静态源代码及调用合同审查通过。phase1原包未派发；第二包实际q00建立namespace后，q01上传因远端数字父目录不存在而失败。保留该失败回执，随后recovery03仅在已有空namespace创建33个数字目录，实际顺序执行全部37请求PASS。安装器核对33个源目标，仅新增20个文件，对已有目标要求字节一致；未覆盖不一致旧文件。

安装后binder prepare-template实际返回PENDING_CURRENT_HOST_RUNTIME_VALUES_THREE_QUALIFICATION_LINEAGES_AND_FRAME_BINDING，58833 B/raw0ece646bf5a0341af44543671f85b694c8c186e7e6a41cfb66dad5751900efa4。root从服务器下载原始模板、验证传输回执和完整原始SHA，并读其顶层和三个模型生成器路径。模板不是已收集runtime binding；StageA应使用已资格的Wav2Lip环境词法路径，当前sharedface/video/Dev路径需来自恢复来源，不能沿用不存在的legacy face_path。旧容量描述须与ADR017一致。

科学结论仍以已完成Fit诊断为准：actual相对RMS Cbest+.057816875、D0-.046865845；静态频谱贡献大部分Cbest，正确局部门控相对±512ms约额外改善D0 .010。Cbest=B−Dmin，不能把分数变化直接称为视觉改善。41文件及分析annex实际持久保存，机制笔记concluded。Dev本轮尚无评分，不用工程进度代替方法验收。

## Working State
服务器为root@connect.westd.seetacloud.com:18625，ROOT=/root/tts-research-system-20261001，NS=cross_tfg_audio_head_20261003，GPU GPU-d09622d0-3535-e662-bb45-3df2b049fcea。private PTY57588及bridge07在本机运行；凭据仅私有内存/环境，永不入文件。SHM为易失，本地.codex/scratch、未跟踪源码及未推送提交均需后续归档。

实际CPU阶段控制目录NS/job_control/dev360_cpu_bind_plan_preflight_attempt01。phase2填值seed、collector、builder12及controls13 metadata-only preflight尚未实际执行；360对照与180候选GPU未启动。dev_method_resume正在准备该小包和detached启动器。fit08_resume/signal11_resume仅审查终态compact durability工具，发现detached launch链接、候选controls ancestry、GPU attestation及canonical PASS发布边界仍需修复；源代码未实际copy。durability工具不阻塞首个GPU阶段。

Task98 active/Experiment97 implemented尚未更新本轮实际staging状态。CONTEXT94行旧96/95入口和忽略HANDOFF需要随后同步；本轮没有Git提交，HEAD96b47a3a3e530ebfc34510abeb4dc23b8799b132为本地未推送。保留五个无关dirty脚本/测试，不清理历史失败证据。

## Changed Files
已安装源码以scripts/experiments/cross_tfg_audio_head_20261003/与source_snapshot_attempt01.json为准。实际请求/失败/恢复回执在.codex/scratch/dev360_cpu_bind_plan_preflight_attempt01/、attempt02/、attempt03/；原始下载模板在attempt03/downloads/runtime_binding_template_attempt07.json。source_manifest_attempt01.json raw1dd552d7cb2f7fcd9a95edf5dcd6d11d7321bc760dc4867acf31f4d573a0bf10，recovery03 request index raw9fbc57d61b0587a1db22799efc30e5cce18e3089fcaa2c0e7c166de9fe47f1d0。

## Verification
root逐一读取37恢复响应、验证所有上传原始pin与exec envelope，最终安装/模板响应raw5163bf1a20772324087249772d51593201cce0a267a635410a2ca5da6fe8f424；实际download响应rawe880226851bd010e688cd7e71e16186fe64294638ee4bd380f82279c04f5a4776。最终source snapshot13237 B/raw0e58475eeef4b46d095220118e891e8774e39220fa26acdbb0fa243d1b14c3ec；root source receipt2815 B/raw26ac4f9f976ddbb5a53e7da037655efc343862fd16ae90eed6735c50e07787c8。执行请求Bash语法审查通过。没有通用pytest、没有Dev GPU渲染、没有打开Confirm。本地BCloudFS历史1.28GB归档回执曾校验，未重新读取完整payload，不声称云提供商已同步。

## References
[仓库](https://github.com/dogLeft233/tts-exp)与[已验证远端提交](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)本次只读HTTP200；分支PR搜索无匹配。当前HEAD仅本地未推送。

## Observations
- [result] 最终Dev源码链已实际安装；33目标pin核对、20新增安装、37恢复请求与模板原始下载全部PASS。
- [decision] 保持冻结120步候选、全原音换轨、原生帧边界、完整失败分母及ADR017资源门；当前修复只改变可执行性。
- [blocker] 实际runtime值填入、同宿主collector、builder及preflight尚待执行，Dev正式GPU尚未启动。
- [next_step] 将实际模板填入已确认宿主路径，收集runtime binding并完成plan/preflight，启动360对照随后180候选和完整配对评价。

## Relations
- continues [[Codex checkpoint - 2026-10-08T03-15-46Z - Refined capacity gates and coherent Dev chain awaiting actual launch]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]
- relates_to [[017-measured-pilot-scenario-and-live-capacity-gates]]