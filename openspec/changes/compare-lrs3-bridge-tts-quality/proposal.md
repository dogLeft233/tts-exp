## Why

自然音频保留相位、向 TTS 频谱移动的 bridge，曾得到小幅 replacement Sync-C 信号：发现轮约 +0.078，22 条确认轮约 +0.031；后者置信区间跨零，两轮均有控制失败，尚未科学确认。值得继续检验的是：换一个 TTS 声学目标，是否会改变这种信号？

已核对上轮 LRS3 数据链使用 **云端** `dashscope_vc / qwen3-tts-vc-2026-01-22`。本地 `faster_qwen3 / Qwen/Qwen3-TTS-12Hz-0.6B-Base` 是另一个待配齐的来源。不能把历史 bridge 误标为本地，也不能用 AISHELL-1 的本地结果与 LRS3 的云端结果直接相减。

“云端质量较高”是本实验要验证的假设。两种 provider 同时改变音色、韵律、发音和克隆方式；即使结果阳性，也只能支持质量关联，不能单独证明参数量或音质的因果作用。

## What Changes

- 固定上轮 22 条 LRS3 / 22 source groups，逐条配对本地与云端 TTS，文本和自然参考音频完全相同。
- 双方使用同一 MFA-linear 目标生成流程和同一 `alpha=0.75` bridge 构造；重新生成目标，消除两套历史处理脚本带来的混杂。
- 四个驱动臂 `N / B0 / B_LOCAL / B_CLOUD`，每臂两次独立 Wav2Lip 渲染；主要评分全部配原始自然音轨。共 176 个新视频、308 个评分 cell。
- 主端点是同记录、同重复下的 `replacement gain(CLOUD) − replacement gain(LOCAL)`，另报告各自是否实际优于自然基线。
- 原始 TTS 和 MFA-linear 目标分别进行匿名听评；质量结果不参与样本筛选或参数调整。
- 产出可复算协议、逐项 provenance、独立 validator、分开的来源比较与质量解释状态。

## Capabilities

### New Capabilities

- `lrs3-bridge-tts-quality`: 固定自然音轨的双 TTS 来源 bridge 配对比较及独立音质验证。

### Modified Capabilities

无。历史协议、结果与 CONTROL_FAILED 状态不改写。

## Impact

实现目录 `scripts/experiments/lrs3_bridge_tts_quality/`，测试目录 `tests/experiments/lrs3_bridge_tts_quality/`，产物目录 `runs/lrs3_bridge_tts_quality_<run_id>/`。优先复用现有 TTS、MFA、WavLM/HiFi-GAN、Wav2Lip、SyncNet 基础组件，不创建通用实验框架。

## Non-goals

本次交付是 spec，不执行模型推理或云端调用。下游实验不训练、不搜索 alpha、不访问 sealed 数据、不合并不同语言/数据集、不宣称已经证明音质因果关系。22 条既有 fit 样本的结果属于配对机制探索，不是新的独立泛化确认。
