## Context

本实验承接 BM 的「LRS3 Wav2Lip 局部时间传递诊断结果」和「LRS3 natural-to-TTS bridge confirmation result」。memory-continue 恢复的约束是：先建立生成端控制，再解释 bridge；历史失败不被覆盖。

用户选择 Wav2Lip-first 的开发顺序。Ditto/LeapTalk 部署 spec 保留但不属于本实验。22 条记录已用于多轮诊断，所以新的成功也只是已见 fit 数据上的 pilot，不能称为独立确认。

## Goals / Non-Goals

目标：一次有界实验回答「明确的人脸生成区域下，Wav2Lip 是否响应局部音频时序；固定 bridge 驱动后换回原始自然音频，是否优于自然音频直接驱动」。

不做生成头、loss 设计、参数扫描、新 TTS/MFA/DTW、全帧/ROI 多臂消融、跨模型或 heldout 实验。全帧历史结果仅作背景，不用于本轮配对统计；不能据此证明全帧框是唯一原因。

## Decisions

### 1. 生成框与评分框分开

历史 `lrs3_natural_to_tts_bridge_confirmation/render.py::shared_face_geometry` 返回整帧框，calibration renderer 也强制验证整帧。本轮不能直接调用这两个 geometry 路径。

新增一个薄的 Wav2Lip 适配器，从原视频预提取、缓存逐帧脸框，并将这些框用于官方输入预处理和输出回贴。固定 checkpoint、96×96 输入、mel 分块、下半脸 mask、推理与回贴逻辑；不改网络。优先在实验目录提供适配器，避免原地修改历史绑定的 third_party 文件。

评分框继续读取 tail_v2 的 `crop.processed_track`，坐标在原视频画布上。生成输出必须回贴到原画布，不能将局部 96×96 脸图直接交给历史评分轨迹。

### 2. 一个协议，两道门

`prepare → control → [bridge，仅 control PASS] → report → validate`

prepare 冻结两个阶段全部参数与音频身份。A 用 N、N_REPEAT、W 新生成 66 个视频；B 仅增加 22 个 BRIDGE_075 视频。A 失败后，B 标为 `NOT_RUN_CONTROL_FAILED`，不得把旧 bridge 分数补进来。

A 的局部 gate 继承 timing-transfer spec，不新增 offset 估计方法；附加独立生成重复与控制 own-audio/替换损伤门槛。B 延续 bridge 的 movement、natural-replacement 非劣性与正收益门槛。全部全局比较取同一共同窗口，原生官方全局分数另列为描述指标。

### 3. 单卡顺序执行

面向已有 16GB 卡，检测 batch 固定 1、Wav2Lip batch 固定 4、SyncNet batch 固定 20；不并发驻留生成器与评分器。先做协议内第 1 条 N 的工程 smoke，完整且绑定一致可作为正式 cell 复用；不可按分数挑 smoke。CPU 缓存尽量分批，OOM/缺权重记 BLOCKED，不自动租卡、量化或改配置。

建议提供以下新 CLI（这是待实现接口，不是现有命令）：

```bash
python -m scripts.experiments.wav2lip_face_roi_replacement.runner --run-id <id> --stage all
python -m scripts.experiments.wav2lip_face_roi_replacement.validate --run-root runs/wav2lip_face_roi_replacement_<id>
```

runner 支持 `prepare|control|bridge|report|all`，直接调用 bridge 也必须验证 A 的完整通过证据。实现以小模块为主，无需建设通用实验框架。退出码：0=完整科学终态且验证通过（包括科学失败）；2=工程阻塞/验证失败。调用方必须读取科学终态，不能把退出码 0 当成功效应。

## Risks / Trade-offs

- 检测不到脸或检测对象不明确可能阻塞；不允许整帧 fallback 或人工按生成结果挑框。
- 固定 W 的线性插值仍会改变声学内容；控制失败只说明该条件组合未通过，不证明模型本质无效。
- ROI 改变和既有数据重复使用限制推断强度；统计 CI 是探索性精度描述。
- 不要求这一轮同时证明所有模型。未来使用同一候选 PCM 测至少一个不同架构 TFG，才能开始讨论可迁移性；Wav2Lip mel movement 不是泛用生成头接口。

## Migration Plan

仅新增实验目录与新 run，保留所有旧 runs、spec 和未相关工作区修改。执行完成后按 BM 实验指令更新同一份本实验笔记，记录成功、负结果或阻塞；本 spec 编写阶段不生成结果笔记。
