---
title: SyncTalk FP16 非有限训练失败的适配与官方环境核查
type: experiment
permalink: tts-exp/docs/experiments/sync-talk-fp16-非有限训练失败的适配与官方环境核查
status: concluded
tags:
- synctalk
- fp16
- numerical-diagnosis
---

# SyncTalk FP16 非有限训练失败的适配与官方环境核查

本轮核查定位了原训练失败的高置信度终末机制：GradScaler 缩放因子降至 2^-128 后，反缩放倒数溢出，有限 scaled gradients 可在 unscale 中变成 NaN/Inf，而非有限检测仍返回0，从而允许一次破坏参数的 optimizer 调用。新旧 PyTorch 的实际 CUDA 核均复现了这一行为；原失败日志末步恰好呈现同一阈值和调用顺序。原失败时完整 state 未保留，因此这是日志、源码与独立复现交叉支持的归因，不能冒称精确恢复了原失败步。

## 范围与实验地位

用户要求先核查新增适配、在官方原环境对照，先不改FP32并保留失败。本轮始终使用原FP16训练政策，没有修复或重启原正式attempt。四身份24模型/27格仍未完成，数据准入不撤销，也没有关于音量效果、嘴型质量或数据充分性的科学结论。既有0/−6完整ep5不能弥补+6失败。

原失败archive `nonfinite_partial_epoch_failure_attempt01/failure.tar.gz` SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72再次核验不变。原4636记录/4484实际optimizer调用/152AMP skips保留，最后global28636/cum28480。失败断言在原step/update之后、trace写入之前，原未记录失败行的具体NaN/Inf类型和optimizer delta仍未知。

## 适配审计与环境边界

固定SyncTalk commit43079fb3f349675f6cb33031bb07262f7566cdd5。已核查controlled_training、execution_state、train_epoch、runtime_common、audio_context、build_datasets。原运行tree相对官方有4个CUDA backend编译flags和provider的缓存AVE/切分适配；network、renderer、utils的数学代码与固定官方源一致。独立全state恢复资格核验model/optimizer/scheduler/scaler/EMA及非注册model counters。数据/whole-recording音频上下文、域RNG、epoch调度与额外验证是实验适配，不能因核心数学未改而宣称整条流程完全等价于官方默认训练。

原失败环境是Python3.10.20/Torch2.5.1+cu121/CUDA12.1。现独立恢复Ubuntu18.04.5用户空间、glibc2.27、Python3.8.8、Torch1.12.1+cu113、cuDNN8302、GCC7.3和CUDA11.3.109；37所需官方源文件逐字节核验，shencoder/gridencoder/raymarching用原C++14 flags真实编译并CUDA运行。宿主kernel7.0/driver580.178.04仍保留，这不是完整旧机器虚拟化。Py1.12诊断桥只处理weights_only参数与step计数API差异，不改AdamW、GradScaler、loss或训练精度。

环境重放固定同一+6 ep4完整state和本实验数据/采样适配，用来隔离依赖环境；直接原Trainer.train_step及原density probe另行绕过ControlledTrainer和loader。两个环境3980条共同前缀的frame/source_audio/pixel/rays/lip/rect全部相同，只有global24172一行LR出现最大8.673617379884035e-19差异。梯度重复本身也非逐位确定，故重放不是历史轨迹的逐位再现。当前环境5680步未出现非有限loss，不能据此声称新环境更稳定。

## 已验证的数值机制

| 证据 | 结果 | 能支持的判断 |
|---|---|---|
| 同捕获state直接原train_step，新旧环境 | scale2^24时MmBackward0 NaN；unit multiplier时FP16梯度有限 | 首个捕获梯度异常受loss scaling影响；不是仅由当前适配train_step或新版依赖产生 |
| 官方旧环境独立重放 | 完成3980次；global27981 loss NaN，在backward前停下 | 原版数学和旧版依赖也能出现同类失败 |
| 首密度非有限输入与完整state | global26624，有限x/encoded_audio/eye，密度logit104.0625；两环境均sigma Inf | 官方无界exp在该捕获条件下确实溢出；FP16 autocast内部exp按原行为返回FP32，这不是切换训练到FP32 |
| 独立CUDA unscale标量fixture，两环境 | scale2^-127时结果[0,1,-1]；2^-128时逆数Inf、结果[NaN,Inf,-Inf]、found_inf仍0 | 反缩放产生的非有限结果可绕过这次检测；不涉及模型optimizer/scaler update |
| 新重放末步与失败state | global27980 finite loss、scaledgrad全部finite、optimizer计数+1；下一state大量参数全NaN | 与反缩放溢出放行破坏性更新机制一致 |
| 原封存日志末步 | global28635 scale降到2^-128；28636 finite loss且optimizer实际调用；随后非有限loss断言 | 原失败终末签名与复现吻合，高置信度支持同机制；原state丢失限制精确追溯 |

2^-128 = 2.938735877055719e-39，其倒数2^128超过float32最大有限值。GradScaler使用double reciprocal再转float32；CUDA unscale先检查输入梯度，再乘倒数，未对乘积二次检查。已读回并封存[PyTorch1.12.1 GradScaler](https://github.com/pytorch/pytorch/blob/v1.12.1/torch/cuda/amp/grad_scaler.py)、[1.12.1 CUDA核](https://github.com/pytorch/pytorch/blob/v1.12.1/aten/src/ATen/native/cuda/AmpKernels.cu)、[2.5.1 CUDA核](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/AmpKernels.cu)。独立标量fixture直接执行相应安装版本的CUDA核，验证源码路径的实际行为。

官方重放首梯度异常global26028，首density_grid Inf为26624，首scale<1为27043，末27980。累计152skips；从ep4累计23996至27824，新增3828次optimizer调用。原失败也有152skips，但发生帧和时刻不同。密度缓存中的Inf、后续梯度非有限和scaler下降同时存在；尚未通过干预隔离证明“首个density_grid Inf导致每一个后续skip”，也未隔离+6为什么将模型推入该轨迹，不能把这部分写成已证实因果链。

## 我们的代码问题与未实施的修复

原ControlledTrainer有限loss检查位置过晚，且有限的更新前loss不能保证更新后的参数有限。它在已有NaN被下一步观察到时终止，并非制造exp或unscale溢出的运算。现有“successful_update”计数证明optimizer被实际调用，不证明数值有效；原历史计数不改写，但不能把它当参数健康证明。

另有本轮诊断编排错误：一次两步资格失败后，PowerShell没有自动中止后续命令，提前启动了新诊断。仅该新job已中止，319完整步/第320次backward中断，日志保留并保守计320；后续启动显式检查exit code。没有把此错误删除或并入原attempt。

未修改FP16、GradScaler参数、sigma公式、LR或正式预算。下一步若进入修复，需要另立版本，在optimizer前验证unscale结果和参数健康、阻止倒数Inf；仅阻断损坏不等于恢复稳定训练。密度参数化/梯度路径稳定性需要单独资格，不能偷偷钳制、调scaler或重跑旧attempt。本轮结论已足以反对“只删除断言继续跑”或“只恢复旧依赖就能解决”。

## 验证、预算与持久交付

总optimizer尝试上限12004，本轮保守计9984=资格4+中止320+current5680+official3980，另外一个非有限前向在backward前终止。24个模型零更新case已全部执行；最后两个case同时各执行两个独立标量unscale fixture，共4次库运算、0模型/optimizer/scaler更新。没有增加正式训练更新。

新证据位于 `F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/controller_qualification/fp16_adapter_environment_audit_attempt01`。`preserved_snapshot_02/verification.json`独立逐SHA核验68文件、174419530B，包含原环境重放4个完整事件state、精确密度输入、各probe、执行源码和环境日志；manifestSHA2bd65e00cebe72fd5f0aa233edbb4073c51340db4b9107255a6c6da6f9d744f1。先前25文件资格也保留。`analysis_02/evidence_summary.json`SHAe68ba8418b3d9e787b15148386741201fb82c5ee04edf759c5051dcf0e360c23记录完整prefix QA、原日志尾部、参数状态检查及探针结果；上游源码另有URL/SHA。全部原失败、失败安装和不通过资格都保留。

## Observations
- [result] 原版依赖也复现NaN loss；exp溢出与GradScaler反缩放检查盲区均有新旧环境直接算子证据。
- [result] 原日志同阈值末步支持高置信度终末归因；原失败步完整state与上游稳定性因果仍有证据边界。
- [decision] 保持FP16和原失败，不把诊断重放合并进正式训练，不将控制失败写为音量科学阴性。
- [next_step] 后续修复须另立版本，先资格更新前数值保护及稳定性机制，再决定正式三档统一重训政策。

## Relations
- relates_to [[核查 SyncTalk FP16 适配与官方环境]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[研究主线与证据地图]]