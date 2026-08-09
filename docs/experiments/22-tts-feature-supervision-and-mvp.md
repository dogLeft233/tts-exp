# 22 — TTS feature supervision 与 feature-domain enhancement head MVP

## 目的与范围

前序实验已经确认，在固定的 TFG/SyncNet 评测中，TTS 输入相对于 natural 输入存在稳定的 Sync-C 优势，但还没有找到一个经过验证、可直接作为 enhancement-head ground truth 的单一声学或 SSL 指标。本实验把监督问题收窄为一个可审计的 feature-domain MVP：输入 natural HuBERT frame features，输出一个受约束的 TTS-like representation correction。它不是 waveform enhancer，也没有把输出接入 Ditto、TFG 或 SyncNet。

本报告记录监督设计、数据边界、实现契约和验证标准。representation metric 的改善不能写成 lip-sync 改善；后续必须使用独立、固定配置的 downstream 评测验证任何 TFG/SyncNet 假设。

## 当前证据与边界

### TFG 结果

AISHELL-100 的 paired 结果显示 TTS 的 Sync-C 比 natural 高约 `+1.022`，Cohen's `d=1.24`，约 92% 的 paired samples 改善。Sync-D 没有同样明确的显著改善，因此不能把结果简化为整体 AV offset 变好。

对 104 个去重后的表征/韵律 delta 做关联分析后，没有一个指标在 BH-FDR 校正后显著预测 Sync-C。少数未校正候选不能作为 confirmatory target，更不能直接当作训练 ground truth。最新的 Phase-2 机制结果还表明 TTS、Ditto 与 SyncNet 之间存在较强的耦合，HuBERT layer-11 segment stability 虽然是最强候选信号之一，但定向 intervention 不能单独重现 TTS diagonal 的 downstream 优势，因果结论仍为 inconclusive。

### SSL、音素与韵律结果

- HuBERT 的 phone/viseme 信息在中间层较强；TTS-natural 差异在早层更明显，但差异并不局限于输入层。
- AISHELL-100 的 L6 cross-condition frame-level PER 约为 natural→TTS `38.97%`、TTS→natural `40.70%`；两方向较对称，说明不能把某一侧当作无条件 teacher。
- MDC English 的独立 MFA、HuBERT/XLS-R 审查显示 TTS 在多个层和 phone/viseme 指标上更易迁移识别；TTS 整体更快、句内 duration CV 更低，但该数据集没有完成 TFG/SyncNet 验证。
- duration 结果支持“全局缩短 + 句内相对时长更规整”，不支持一个 phone-specific duration reallocation target。

### 已否定或降级的监督

实验 16–18 说明 phone-local waveform warp、严格保持 natural timing 的 waveform control，以及单独的 loudness、energy、spectral-timbre 和 duration control 都不能恢复 raw TTS 的 Sync-C 优势。phone-local warp 同时改变 pitch、formant、phase、spectral envelope、transient、coarticulation、timing 和 prosody，因此只能称为 weak pseudo-target，不能称为 disentangled ground truth。

实验 21 还发现旧 MVP sidecar 的 natural sample count 与当前 natural 文件不一致。该 provenance mismatch 阻塞旧 MVP 的逐样本 acoustic-to-SyncNet 解释，也不允许继续把旧 waveform output 当作严格 paired supervision。

## 监督方案的优先级

长期可比较的约束集合为：

1. MFA 对齐的 phone-conditioned TTS SSL representation distribution；
2. phone/viseme teacher 或 nearest-centroid consistency；
3. natural content、timing 与 identity preservation；
4. 低维 duration/pause/energy statistics；
5. 条件 adversarial matching；
6. 最后才考虑需要稳定 downstream differentiability 的 Sync-C reward。

全局 KLD、单一 spectral feature 或 raw waveform L1 都不足以覆盖已观察到的跨语言、跨层和 TFG 耦合现象。任何 distribution target 都必须用 train-only statistics 估计，并记录 support gate。

## 本次 MVP：feature-domain residual adapter

### 输入和输出

```text
natural HuBERT features [batch, frames, dim]
    -> small temporal residual adapter
    -> enhanced HuBERT-like features [batch, frames, dim]
```

MVP 使用预先缓存的 HuBERT features，不在训练时下载或微调 SSL 模型。默认实现支持 768 维 HuBERT features，并可按 metadata 选择保存的 layer 6 或 layer 11；禁止假设 NPY 第一维就是某个固定层。zero-initialized output projection 使 fresh adapter 是 exact identity，便于和 natural identity baseline 做严格比较。

### 训练 target

对每个 train natural frame，依据该 natural 音频自己 MFA TextGrid 的 phone label，构造 train-only prototypes：

```text
mu_natural(phone) = train natural phone centroid
mu_tts(phone)     = train TTS phone centroid
delta(phone)      = mu_tts(phone) - mu_natural(phone)
x_shift           = x + alpha * delta(phone)
```

默认 `alpha=0.5`。只保留 natural/TTS 两侧 support 都达到最低门槛的 phone。validation/test 不参与 prototype、normalization、support vocabulary 或模型选择。TTS 和 natural 使用各自的 MFA spans；不复制 timestamp，也不使用 uniform fallback alignment。

总损失为：

```text
L = 1.00 * prototype_shift_smooth_l1
  + 0.20 * tts_centroid_mse
  + 0.05 * identity_mse
  + 0.01 * delta_l2
```

其中 identity 项限制不必要的漂移，delta 项约束 residual 幅度；centroid attraction 是辅助项，不代表真实 TFG target。后续若加入 layer-11 segment stability，应作为有边界的辅助 regularizer，并单独做 ablation，不能在此 MVP 中把 stability 写成已验证因果 target。

### 训练数据边界

优先使用本地公开/授权的 AISHELL-1 paired natural/TTS 数据与独立 MFA：

- `results/rhythm_style_500/aishell1_test_400/`；
- train/valid/test 由 condition-level manifest 的 speaker split 决定；
- paired key 和 speaker 在 split 之间严格不重叠；
- train 只使用 train split 建立 prototypes；
- test 只在 checkpoint 固定后做最终 held-out evaluation；
- 不使用旧 `weak_target` waveform 作为监督；
- 不混入 MDC English 训练，不下载需要认证的数据，不把密码或 token 写入任何 artifact。

缺失 pair、missing layer、embedding metadata mismatch、非 MFA alignment、NaN/Inf 或不一致的 natural/TTS occurrence 都 fail closed，而不是以零向量、uniform fallback 或另一条件的 alignment 替代。

## 调试与 provenance 契约

每次训练至少保存：

- manifest、embedding sidecar/NPY 和 prototype SHA-256；
- seed、Python/PyTorch 版本、model config、loss weights；
- train/valid/test paired keys 和 speaker IDs；
- 每个 phone 的两侧 support、coverage、被丢弃的 tokens/frames；
- 每 epoch 的分项 loss、identity baseline、输入/输出统计、residual norm、gradient norm、parameter norm；
- finite、NaN、Inf 和 missing-frame counters；
- checkpoint 的 target definition、split fingerprint 和 warning。

所有 JSON 使用严格 finite serialization，禁止用 NaN/Inf 占位。debug 日志只保存稳定 ID、统计量和 hash，不保存 raw audio 或完整 embedding。

## 评估与成功标准

held-out evaluator 比较：

1. natural identity；
2. adapter output；
3. TTS phone-prototype reference（仅作为 representation upper-reference，不是 ground truth）。

报告 phone-conditioned distance to natural/TTS centroids、within-phone variance、nearest-centroid frame accuracy/PER、delta norm、identity drift、coverage 和 per-phone rows，并保存 held-out keys 与 leakage checks。成功标准是训练可复现、无泄露、无 NaN/Inf，并且 adapter 在冻结的 held-out representation metrics 上向 train-derived TTS phone-conditioned distribution 移动，同时 identity/content drift 受控。单独降低训练 loss 不算成功。

本 MVP **不能**声称改善 Ditto/TFG/SyncNet。只有将 representation adapter 接到实际 downstream audio consumer，并用固定视频、固定 seed、identity/no-op 和 random-noise controls 做独立 paired SyncNet ledger 后，才可以检验 downstream hypothesis。

## 相关文件

- 模型：`scripts/tts_feature_head.py`
- 训练与 prototype 构造：`scripts/train_tts_feature_head.py`
- held-out evaluator：`scripts/eval_tts_feature_head.py`
- 测试：`tests/test_tts_feature_head.py`
- 前序 target 失败：`docs/experiments/16-phone-local-warp-validation.md`、`docs/experiments/18-tts-feature-control-validation.md`
- provenance 阻塞：`docs/experiments/21-mvp-audio-forensics.md`
- TFG 关联边界：`docs/experiments/15-tfg-link.md`
