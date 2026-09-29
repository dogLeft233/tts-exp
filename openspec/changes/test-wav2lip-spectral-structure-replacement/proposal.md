## Why

首轮 natural-phase MAG_075 的 replacement ΔSync-C=+0.078，18/23 条改善，未校正 CI=[+0.016,+0.136]；固定 0.75 的确认轮缩小为 +0.032，CI=[−0.034,+0.105]，两轮控制均未完整通过。这个信号值得一次直接的声学对照，但尚不足以训练生成头。

本轮区分：逐时刻 MFA-linear 谱迁移是否优于整句平均谱修正，及观察到的收益是否超过重建处理和重复生成波动。暂停的生成链时间控制分支保持暂停。

## What Changes

- 固定原 22 条 seen-fit 数据、α=0.75；四个科学臂 N / RT / MAG / ENV，加 N_REPEAT 复现控制。
- RT 只做 STFT 重建；MAG 迁移逐时刻 log magnitude；ENV 只迁移整句平均 log-spectrum 差，保留 natural 的时变谱结构。
- 所有收益比较均对未修改的 natural PCM；固定原 face/ROI 和整数平台的 U 评分窗口。
- 使用 V_N/P 错音轨控制检查当前评分窗口的错配敏感性。P 仅用于 mux，不驱动 Wav2Lip；这不是修复旧 generated-own gate。
- 先跑控制阶段（66 个 GPU 视频、88 个 CPU cells），通过独立验收后再加 MAG/ENV（44 个视频、44 个 cells）；最多 110 视频、132 cells。
- 输出单一、有止损的机制试验结果和独立验收；不做训练或额外参数搜索。

## Capabilities

### New Capabilities

- `wav2lip-spectral-structure-replacement`: 固定 natural 参考下，比较逐时刻谱迁移与平均谱修正的收益、复现性和等效性。

### Modified Capabilities

无。旧 CONTROL_FAILED / GENERATED_PLATEAU_UNRESOLVED 均不改变。

## Impact

新增小型实验包 `scripts/experiments/wav2lip_spectral_structure_replacement/` 和对应 tests，复用已有音频、无损媒体、GPU worker 和 CPU SyncNet。无需新依赖、模型、TTS、MFA/DTW 或云卡。当前交付为设计，未执行实验。
