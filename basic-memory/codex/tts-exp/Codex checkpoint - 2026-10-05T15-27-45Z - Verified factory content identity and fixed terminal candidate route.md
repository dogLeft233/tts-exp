---
title: Codex checkpoint - 2026-10-05T15-27-45Z - Verified factory content identity
  and fixed terminal candidate route
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-05-t15-27-45-z-verified-factory-content-identity-and-fixed-terminal-candidate-route
repository: dogLeft233/tts-exp
username: dogle
agent: codex
repo_root: E:/Documents/tts-exp
model: gpt-6.1-sol
git_sha: e5f175dca66b569a4ec602adfe38c0b096d4fe14
capture: deliberate
project: tts-exp
status: open
hostname: LAPTOP-E133GSLE
trigger: compact
started: '2026-10-05T15:27:45.0174999+00:00'
branch: codex/cross-tfg-audio-head-20261003
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
cwd: E:/Documents/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-05T15-27-45Z - Verified factory content identity and fixed terminal candidate route

## Summary
实际训练07通过物理 GPU 身份校验后，在工厂文件绝对路径比较处停止；独立远端读回证明两处工厂文件内容相同，正在以严格内容身份修复启动08。

## Story
用户最初要求恢复 TTS 增强 TFG 的机制研究，随后将目标扩展为可用于多个 TFG 模型的自然音频增强头。增强波形只驱动生成，评分与交付恢复原始自然 PCM 和时钟；最终要求未见说话人及保留模型验证。根代理维护理论与实验主线，GPT-6 Luna/max 子代理负责代码和实验。Qwen ASR/TTS、重跑和网盘使用已经授权。

目前 RMS 基线在三个 Dev 模型的等权平均提升为 +0.153602；简单共享谱形尚未满足相对 RMS 再增 +0.050 的目标。因此并行推进固定120步的九参数局部过渡头，以及区分谱形和局部功率包络的十一参数头。保持 Fit40/20说话人、Dev20/10说话人×三种子×三个模型；Confirm80/40未见说话人和 LatentSync 在最终方法冻结前封存。

训练07已实际完成22文件源包 staging、CPU preflight、实际绑定独立核验并启动 GPU 子进程。UUID 表示差异已修复。模型加载的13项身份检查中前11项通过，随后 `runtime_factory_path` 因生产几何与训练使用不同绝对 checkout 路径而停止，尚无 zero-head40 或优化器更新。根代理独立读取两处实际工厂文件，均为20085字节、SHA-256 `7903fbd9dfebf95bcb388add26a1967bc39ad855d3d995a4730d316dd0e806c4`。08只修复路径来源比较：保留两个绝对路径，并要求两处实际文件与源记录的大小和内容哈希严格一致，其余12个身份字段和科学常量保持严格检查。

第120步候选配置物化工具和540行共同支持集分析工具已经提交，前者固定取末步权重，后者联合180条新候选和360条已核验 identity/RMS 控制。旧720范围已纠正为540；局部压缩仍是独立探索。新ASR适配器将每个确定性候选音频仅识别一次，并验证九条种子/模型行的WAV哈希一致；保留旧CER和字符差异的描述性报告，不新增阈值。

## Working State
持久入口为当前 Task78、Experiment77 和 CONTEXT.md。本地 HEAD `e5f175dca66b569a4ec602adfe38c0b096d4fe14` 尚未发布，新增末步物化与联合分析代码。五个既有修改文件保持原样；大量 scratch、源版本和 runs 为本机易失产物，详见 `.codex/scratch/checkpoint_ER_git_status.txt`。

实际07根核验：`.codex/scratch/fit40_transition_head_training_packet_attempt07_revision01/root_actual_binding_qc_attempt01.json`。实际失败与源身份读回：同目录 `root_status_attempt01/` 和 `root_factory_content_readback_attempt01/`；后者原始响应 SHA-256 `5880ea0121945d2217442250037400affb1aeb8ffac292ce5d2155cd2ed67ae58`。07子进程675943已经释放，没有训练任务占用GPU。08源包正在准备，必须新路径而不能覆盖07。

十一参数头CPU资格 worker03已完整审阅；修复了未运行02的缺失数组键，并增加全部160次相位检查、实际STFT实现来源、输出容量限制和终态CUDA未初始化核查。具体 packet06 已准备但尚未执行，入口 `.codex/scratch/cross_tfg_audio_head_20261003/local_power_spectral_head_cpu_signal_attempt03/dispatch_attempt06/request_index_attempt06.json`。这只是信号与梯度资格，不代表SyncNet提升。

远端主机为 root@connect.weste.seetacloud.com:12349，命名空间 `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`。私密凭证只在根代理运行时，绝不写文件或BM。私密桥接PTY50274、port10887为易失状态；使用已审阅 `.codex/scratch/root_dispatch_reviewed_packet_attempt02.ps1`，每次核对实际源字节与请求哈希。已闭合的26项存储转移与删除不要重做；SHM约5.014GB空闲，overlay保持512MiB预留。

## Changed Files
- `scripts/experiments/cross_tfg_audio_head_20261003/materialize_v3_step120_method_candidate_attempt02.py`：固定末步、绑定实际训练来源，尚未物化权重。
- `scripts/experiments/cross_tfg_audio_head_20261003/join_v3_candidate_540_dev_attempt02.py`：三方法共同有效查询支持、说话人等权bootstrap、Cbest及D0/Canchor对比；实际联合分析待候选生成。
- `CONTEXT.md`：记录实际07失败位置与当前路线，93行。

## Verification
根代理核验07全部22个源成员及相对06的20个未变文件、40个原始PCM、几何缓存与固定120步范围，实际绑定PASS。启动后状态明确0更新，在runtime_model_load失败。两处工厂实际完整SHA相同，失败属于路径来源检查而非学习结果。此前已独立重算168个原生特征产物和861个完整40ms功率窗，均通过数值核验；RMS功率比4.6252291331566475，对应5.939439364175865dB。未运行通用测试。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
当前分支没有匹配PR；本地e5提交不能构造远端链接。

## Observations
- [result] 实际训练07已启动并停止于同内容工厂文件的绝对路径身份比较；zero-head40、120优化更新、终态重评分及新Dev结果均未完成。
- [decision] 仅将工厂路径改为严格双文件内容身份校验，保留12项其他身份检查和全部科学常量，使用全新08产物。
- [decision] 固定第120步权重，不以Dev选择迭代；540行主分析保留失败分母、GUARD20短片完整cohort空值和原音原时钟评分。
- [blocker] 08启动修复、十一参数实际CPU资格以及末步独立重评分仍待完成，最终可学习方法尚未冻结。
- [next_step] 完整审阅并实际派发新08源包与CPU资格packet06，先取得zero-head40和全部11坐标梯度资格，再完成固定120步训练及终态重评分。

## Relations
- continues [[Codex checkpoint - 2026-10-05T14-50-21Z - Verified physical GPU identity and local power qualification route]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]