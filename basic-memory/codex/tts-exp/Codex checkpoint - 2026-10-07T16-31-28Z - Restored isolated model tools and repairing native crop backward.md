---
title: Codex checkpoint - 2026-10-07T16-31-28Z - Restored isolated model tools and
  repairing native crop backward
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t16-31-28-z-restored-isolated-model-tools-and-repairing-native-crop-backward
username: dogle
started: '2026-10-07T16:31:28.3200016+00:00'
branch: codex/cross-tfg-audio-head-20261003
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
hostname: LAPTOP-E133GSLE
repository: dogLeft233/tts-exp
git_sha: 292106070e1e1c2600c7f5080fac17675c881498
project: tts-exp
repo_root: E:/Documents/tts-exp
capture: deliberate
cwd: E:/Documents/tts-exp
status: open
model: gpt-6.1-sol
agent: codex
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T16-31-28Z - Restored isolated model tools and repairing native crop backward

## Summary

完整回读了首次训练失败的归档，实际修复了 MuseTalk 的隔离下载依赖，并发现确定性反向适配器没有覆盖真实裁剪模块；正在以修正后的源码包启动训练资格化。

## Story

用户的目标从解释中文 TTS 增强 TFG 的现象发展为可使用的跨模型自然音频增强头。增强音频只用于驱动视频，最终成片恢复原自然 PCM 与时钟，再用原音评分。root 维护机制推理、冻结协议和独立复核，GPT-6 Luna/max 子代理实现实验。历史共享谱形方案未达到相对 RMS 的额外增益标准，因此当前学习九参数、三种局部过渡门控乘三频带的谱形增强头，幅度界限 ±3 dB。

迁移后 Fit40 的缓存真实完成，training09 的全部零头门控通过，首次反向计算因 CUDA MaxPool3d 严格确定性限制退出，实际优化更新数仍为零。前一 checkpoint 的完整失败 tar 尚待 root 回读；本次已逐一核验全部 87 个成员、29,758,336 字节及归档 SHA，通过。失败证据可以持久恢复，不能将这次工程失败解释为方法效果。

修复保留模型、GPU 前向、损失与样本顺序，仅为 MaxPool3d 和 bilinear resize 提供 CPU 确定性反向。源码检查纠正了缺少 torch 导入、反向 recorder 引用、完成回执被覆盖和训练路径过重等问题。随后 root 沿真实调用链发现 bilinear 操作来自 differentiable_native_crop 模块，单独替换 adapter.F 不会触及该操作。旧封装未上传或执行；子代理已准备 revision02，字节固定原裁剪源码并在隔离进程中替换两个模块的同一个 F 引用，同时核验函数 globals 身份。该修正包仍需 root 全量差异审阅和实际 GPU 资格化，尚无成功 backward 或学习参数。

模型恢复也有实际新结果。Muse05 在三个固定 wheel 下载完成后因目标 Python 无 pip 失败，原失败保留。Muse06 用标准库解包固定 wheel 到新隔离目录，真实安装成功：gdown 5.2.2、beautifulsoup4 4.12.3、soupsieve 2.10 的路径与版本和既有 requests/tqdm/filelock 均读回核验，76 个文件，未修改模型环境。资产 worker 尚未运行。root 又发现预检内的 WORKER_PIN 与封装实际 worker 相差一字节及 SHA，要求保留旧包并修正，不能把外部索引校验通过当作内部绑定正确。Ditto 12 个权重已真实全部下载重算通过。

## Working State

冻结 Fit40/20 speakers、Adam120/三遍，Dev20/10 speakers × 三种子 × Wav2Lip/MuseTalk/Ditto，迁移后同 GPU 新做 360 identity/RMS 控制与 180 候选。接受条件为三个模型方向均正，等模型相对 RMS 额外至少 +0.050，并通过 D0/Canchor、内容、视觉、GUARD20 与完整失败分母。Confirm80/40 未见说话人与 LatentSync 保持密封。没有新 Dev 或 Confirm 结果。

当前服务器为 root@connect.westd.seetacloud.com:18625，SSH 指纹 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0，GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。密码和云 API key 仅保留私有运行时，永不入文件。持久命名空间 /root/tts-research-system-20261001/cross_tfg_audio_head_20261003，临时数据 /dev/shm/cross_tfg_audio_head_20261003。私有 PTY57588/bridge07 端口12249 是机器本地状态，恢复后先核验存活。

HEAD 292106070e1e1c2600c7f5080fac17675c881498 为本地未推送提交，保存 runner10、binder10 与 v5 joint adapter 的初版；该提交包含上述未覆盖 crop.F 的错误，不可直接派发。修正源码和新封装为未提交机器本地状态。五个之前存在的无关修改保留。CONTEXT 94 行与 ignored HANDOFF 尚未同步 Muse06 的实际新阶段；Task84/Experiment83 仍记录上一阶段，实际结果需下一次更新。Fit40 producer05 的 4,979 文件/287,367,187 字节完整树仍在 SHM，独立解码和持久备份包已由子代理准备，尚未派发。项目持久空间上次实测约7.529 GB，SHM约30.583 GB，执行前重新测量。

## Changed Files

- scripts/audio_enhancement/v3_v4_native_runtime_adapter_v5_attempt01.py：GPU 前向、CPU 确定性反向适配器，revision02 待审阅。
- scripts/experiments/cross_tfg_audio_head_20261003/run_fit40_native_transition_head_v3_attempt10.py 与 late_bind_fit40_native_transition_head_v3_attempt10.py：完整单次 backward gate、资格化独立输出、固定120步训练入口。
- .codex/scratch/fit40_transition_head_v3_attempt10_recovery_attempt02_patch01/：新修正 ZIP/manifest/config 与 stage/bind/qualification 请求，未派发。
- .codex/scratch/musetalk_asset_restore_attempt06/：隔离安装实际回执和待修复内部 worker pin 的资产恢复包。
- .codex/scratch/root_training09_archive_independent_readback_request_attempt01.json：实际完整归档回读请求与 response。

## Verification

training09 tar SHA256 05d9c2da8a71cb4e4e30b5a2399f9e07e753f52544955871dca9eaf0c0833a04，root实际全部87成员重新读回并核验唯一文件名、完整 SHA 和29,758,336字节，PASS；response SHA256 97074c14fce35b529468d77e6b4226334d54fa22e4660ed77d183ae7ec411f5f6。之前独立比较40×31评分曲线与40×27×1024视觉值完全相等的结果保留。

旧 training10 revision01 ZIP 的23成员与20个沿用09源码字节均由 root核验PASS，但该包因真实裁剪模块遗漏而停用。revision02 子代理声明本地语法和交叉 pin 通过，root 尚未复核，不视为执行结果。没有新增或运行通用 pytest。

Muse06 实际隔离安装耗时0.866秒，状态 PASS_ISOLATED_PINNED_GDOWN_RUNTIME，receipt4009字节/SHA256 cf36a39cc49a42501faeffac823977d32a2beb64881e45d67db3595688784997；response SHA256 285433c82a0573fd56be68b02591ee9b4ad47d1c8299bd5085ab691facf70301。没有新模型资产下载或 Muse 推理结果。Dev runner04、builder05、collector03/rawmedia02、join03/bootstrap02 的完整差异已审阅；还需修正 rebinder 指向自身、join要求旧schema等问题并实际进行两个重复/后端共六行的工程资格化。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)；当前 HEAD 为本地未推送提交。
- GitHub 对当前分支的只读 PR 搜索返回无匹配。
- .codex/scratch/fit40_native_rms_geometry_attempt05_root_raw_readback_attempt01/request_index_attempt01.json：待审阅的 producer 完整解码/归档包。

## Observations

- [result] training09 完整87文件归档经 root独立重读通过；训练更新数仍为零。
- [result] Muse06 隔离下载依赖真实安装并读回通过，模型资产恢复尚未运行。
- [decision] 严格确定性策略和冻结科学协议保留，修复应覆盖真实 native crop 模块且保持该源码字节不变。
- [decision] 新 GPU 三模型控制必须真实重建；六行后端重复资格化只提供工程控制，不替代 Dev 效果评价。
- [blocker] 修正后的 joint backward 资格化尚未实际运行；Muse 资产预检存在内部 worker pin 错误，正在修正。
- [next_step] 审阅 revision02 的真实 crop.F 修复与封装并执行完整单次 backward 资格化，通过后运行冻结 Fit40/120 和独立终端评分，再推进同 GPU 三模型 Dev。

## Relations

- continues [[Codex checkpoint - 2026-10-07T16-03-27Z - Verified zero head arrays and repairing strict training backward]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[015-same-gpu-deterministic-controls]]