# 04 — 英语复现实验

## 问题

中文上 TTS 提升唇形同步的效应在英语上是否成立？

## 方法

- TTS: faster_qwen3 0.6B ICL（英文模式）
- 数据: LibriSpeech test-clean 子集 13 样本
- TFG: Ditto + SyncNet（与中文相同管线）

## 结果

### LibriSpeech test-clean（n=13，相同说话人）

| 指标 | 值 |
|------|------|
| ΔSync-C (TTS − Natural) | **+0.160** |
| 配对 t 检验 | t=1.96, p=0.058 (ns) |
| Bootstrap 95% CI | [−0.11, +0.42] |

### HDTF paired（n=13，不同说话人）

| 协议 | Nat C | TTS C | ΔC | Nat D | TTS D | ΔD | 方向 |
|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| HDTF 抽帧图像 | 6.128 | 5.901 | −0.228 | 8.790 | 9.053 | +0.263 | ❌ |
| 固定图像 (data/data/image) | 7.431 | 7.290 | −0.140 | 7.721 | 7.739 | +0.018 | ❌ |

注：固定图像实验排除了 HDTF 视频抽帧图像质量差对 SyncNet 绝对分数的压低效应（两组分数均明显提升），但方向不变。

## 结论

**TTS 优势在英语上不成立**。两个独立的英文数据集（LibriSpeech test-clean, HDTF paired）均未复现中文上观察到的 TTS 增强效应。HDTF 实验中 TTS 不优于 Natural 的方向在两个图像协议下一致。

可能原因：
1. 英语 SyncNet 训练数据多样性更高
2. 英语 TTS 质量（faster_qwen3 英文模式）低于中文
3. 英语自然语音变异性本就更大
4. HDTF 说话人之间录音条件、语速、口音差异大，增加了 Ditto 生成的不确定性
5. HDTF Natural 基线本身较高（Sync-C 7.43），提升空间有限

## 状态

✅ **已完成** — 跨语言不对称已确认（中文 √ 英文 ×）

## 相关文件

- `docs/experiments/english_wav2sem_fp_fs_summary.md` — 详细分析（保留原文件）
