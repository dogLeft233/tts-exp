# 03 — 因果机制分析

## 问题

TTS 提升唇形同步的**机制**是什么？

## 实验链

### Phase 1: 观察性分析
- 提取 36 个声学/Wav2Sem 特征 (n=13×2 条件)
- Wav2Sem Fp/Fs 分离度分析
- 发现: 无单一特征能解释 TTS 优势

### Phase 2: 因果干预
- 逐样本反事实：调节 LUFS、频谱倾斜、动态压缩/扩展
- **全部 4 个声学特征被身份控制拒绝**（p>0.15）
- HuBERT L11 segment_stability: 单方向显著（d=−3.40），但双向测试失败

### Phase 3: G×E 矩阵
- Generator (TTS/Natural) × Evaluator (SyncNet) 4×4 矩阵
- 交互项 +7.29：**TTS 和 SyncNet 之间存在共适应**
- 这解释了为什么单一声学特征无法解释全部效应

## 结论

TTS→TFG 的利好最可能是 **Generator×Evaluator 共适应**：
- SyncNet 训练分布偏向 TTS 般的"干净"语音
- TTS 音素边界更清晰，匹配 SyncNet 的判别特征
- 不是任何单一声学特征驱动

## 次要发现

1. TTS **不损害** HuBERT 帧级音素信息（PER 差异 <1%）
2. TTS 略微提升 L2 类间分离度（+7%）
3. Natural 和 TTS 在 HuBERT L6 t-SNE 中完全重叠
4. 英文上 TTS 优势不成立（ΔC=+0.16, ns）

## 状态

✅ **已完成** — 因果链未完全闭合，但核心机制已识别

## 相关文件

- `docs/experiments/tts_tfg_mechanism_report.md` — 详细报告（保留原文件）
- `docs/experiments/summary.md` — 多实验综合分析（保留原文件）
