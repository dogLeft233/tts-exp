---
title: Codex checkpoint - 2026-10-18T01-35-21Z - FP16 adapter replay evidence and
  isolated official environment staging
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t01-35-21-z-fp16-adapter-replay-evidence-and-isolated-official-environment-staging
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-18 01:35:21.055470+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6-astra
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-18T01-35-21Z - FP16 adapter replay evidence and isolated official environment staging

## Summary
用户授权的FP16适配/官方环境审计已捕获独立重放的首个非有限梯度，但原非有限loss尚未复现；官方旧版Python/Torch已真实CUDA运行，Ubuntu18/GCC7/CUDA11.3扩展对照仍待完成。

## Story
原目标仍是四身份两seed三训练档24完整模型及三轴27格；原Brown/+6/ep5失败并不撤销数据准入，也不构成科学阴性。最新用户明确要求先核查新增适配并在官方原环境对照、先不改FP32、保留失败，已覆盖旧等待诊断范围答复。新工作仅独立诊断，不重启原supervisor105372/observer133984/job610686，也不将诊断更新并入旧attempt。

通过CodeGraph/固定官方源审计了controlled_training、execution_state、train_epoch、runtime_common、audio_context、build_datasets。实际旧runtime只有4个CUDA backend编译标志改动及provider缓存AVE/分割改动；原network/renderer/utils训练数学未改。新增域RNG、epoch状态恢复和额外验证仍需逐项资格，不能以源未改代替证明。原官方环境是Ubuntu18/Python3.8.8/Torch1.12.1+cu113，失败环境为Ubuntu24/Python3.10.20/Torch2.5.1+cu121。

先各跑2步controlled/direct原train_step对照，第一步loss一致、第二步分岔，未通过逐位资格。一次PowerShell批次未阻断后续启动导致长诊断在资格失败后仍启动；已主动仅SIGINT该新诊断PID688584，319完整步/第320步backward中止，保留全部记录。其receipt因KeyboardInterrupt未被Exception捕获仍标running但有ended，须以进程和日志为准，不改旧receipt掩盖。v2合同将该运行保守计320，current新重放5680、official6000、资格4，总12004次更新尝试上限不增加。

12个零更新FP16前后向（3固定frame×controlled/direct×2重复）显示相同输入和loss、梯度hash在同实现重复间也不同，relL2约8e-5至2.4e-4；全部scaledgrad有限，optimizer/scaler无更新且模型恢复。已证实反向非逐位确定性，但尚未定位具体kernel，不能把长重放称为精确历史轨迹。current_environment_replay_v2已完成5680次，firstbadscaledgradient global26042、loss .0031228854786604643仍有限、scale16777216；density/sigma/audio等梯度NaN/Inf，render输出全有限。global27883首次scale<1。末global29680/cum29639，37skip、minscale0.0001220703125，未出现原非有限loss。该结果支持数值不稳定但尚不足以确定原失败根因。

## Working State
权威新目录FROOT/controller_qualification/fp16_adapter_environment_audit_attempt01；ROOT=/dev/shm/synctalk-multispeaker-data-20261017，FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017，SSH wjj@192.168.50.2。原失败目录nonfinite_partial_epoch_failure_attempt01原路径/20成员和archive SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72保留。ep4输入checkpoint SHA0e2223445603b688d6c8feae30e8717bb90dacf66226599bced99306f2eb67f7前后未变；此state不是丢失的原失败state。

新current replay事件state两份各27MB：first_nonfinite_scaled_gradient.pth SHA339589c6845cc88cfe63d9e3d285894f19841f0b90fc2de860c8aae348f0c444，first_scale_below_one.pth SHA1a0808b5bfd427ff4a54d4fc1cbb173af494befe36a3c01661f125392a583435。新重放目录已scp到F，但独立全成员SHA核验尚待完成。新source diagnose_fp16_failure.py SHA6a5dd24f5d62124a3bcc90e346c41e521d8c56f885af5cc1834467956a25a3ee。v1/v2合同、qualification_controlled/direct、zero_update_parity、被中止current_environment_replay全部保留。

机器局部官方环境：Windows C:/Users/dogle/.codex/synctalk-official-env-20261018；Linux /run/user/1000/synctalk-official-env-20261018含Python3.8.8/site Torch1.12.1+cu113，810MB cuDNN库在ROOT/official_env_audit_torchlibs经symlink连接。torchwheel SHA4adf483ac2d047534a7d023f0022bd8694d87627068ad6dddf186cb3273bbfa2。Pythonlibffi3.3/openssl1.1.1w独立提取修复_ctypes加载后，已真实torch.ones(cuda)通过。rootfs/GCC7/CUDA11.3/proot在/home/wjj/tts-audio/synctalk-official-env-audit-20261018，官方包SHA核验；nvcc最初错误版本URL404后改官方Packages索引11.3.109，不重跑整个安装。proot可执行ldd通过，尚未实际在Ubuntu18用户空间运行torch。宿主kernel/driver仍现有7.0/580.178.04，不冒称完全旧硬件环境。仍缺匹配torchvision、torch_ema/ninja/lpips/tensorboardX/mcubes/rich/librosa/trimesh等；autoavsr现有Python3.8依赖可只读借用但cv2 5不保证旧glibc兼容。旧Torch没有register_step_post_hook和weights_only参数，需仅诊断接口兼容shim并资格，不改原冻结源。

根盘在CUDA安装前360MB、SHM195MB、run tmpfs554MB；CUDA约160MB后余量更少，真实df优先。只用本任务独立目录，不清原数据/旧May缓存/共享venv。正式训练停机，独立诊断也已完成，没有需重启的长job。synctalk-27仍ACTIVE且prompt已前置最新用户授权/FP16约束；旧尾部等待字样被新范围覆盖。Task父卡接近50KB，仍需拆新审计任务并更新当前头部/研究地图，而非追加epoch日志。

## Changed Files
未提交的新脚本scripts/experiments/synctalk_multispeaker_full_20261017/diagnose_fp16_failure.py、qualify_fp16_replay.py及fp16_audit_contract_v1.json/v2.json；.codex/scratch中的独立环境下载/提取/证据脚本。Git共1037dirty项，包含其他工作，不全stage/clean。没有新commit/PR。

## Verification
新脚本py_compile通过；两步轨迹逐位资格失败并保留；12零更新对照模型恢复和optimizer/scaler不变通过；5680步独立FP16重放完成且loss有限，捕获scaledgradient非有限现场。原失败archive复核SHA保持。Python3.8.8/Torch1.12.1+cu113真实CUDA张量通过，原环境扩展编译/训练对照尚未执行。低scale、源码exp和CUDA原子操作只是定位线索，尚非根因证明。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)，GitHub只读确认当前branch无openPR。
- 本地SHA `2b71639c3b555293375d248350999377168d515d`，未确认已推送，不构造commit链接。
- [固定官方源](https://github.com/ZiqiaoPeng/SyncTalk/tree/43079fb3f349675f6cb33031bb07262f7566cdd5)

## Observations
- [decision] 保持FP16，原失败不可变，独立诊断预算12004attempts；不将新重放视为旧attempt续训。
- [result] 同state同输入FP16的forward loss可一致而backward梯度非逐位确定，原轨迹未被精确重建。
- [result] 新重放先出现梯度非有限而loss/render有限，5680次内未出现原forward非有限loss。
- [blocker] 官方Ubuntu18旧依赖和原CUDA扩展尚未完成资格；原非有限首发算子/归因未知。
- [next_step] 完成官方旧版隔离环境及接口兼容资格，用同state/input的FP16对照和捕获梯度现场定位首发非有限算子，再区分适配与环境影响。

## Relations
- continues [[Codex checkpoint - 2026-10-18T01-16-13Z - FP16 failure preserved and official environment comparison authorized]]
- tracks [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]