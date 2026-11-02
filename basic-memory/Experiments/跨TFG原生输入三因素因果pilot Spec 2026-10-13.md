---
title: 跨TFG原生输入三因素因果pilot Spec 2026-10-13
type: note
permalink: tts-exp/experiments/跨tfg原生输入三因素因果pilot-spec-2026-10-13
status: frozen
protocol: native_axis_factor_pilot_v1
scientific_status: SPEC_ONLY
tags:
- tts
- tfg
- native-feature-space
- phoneme-separation
- segment-stability
- local-envelope
- visual-motion
---

# 跨TFG原生输入三因素因果pilot

本页保存冻结设计。实际执行、修正和结果见 [[tts-exp/experiments/native-axis-factor-pilot-20261013]]，综合判断见 [[研究交接总览]]；不以本规格页的状态代替实验结果。

## 问题与研究边界

现有主线已经把共同方向收敛为三类：音素间表示分离、片段内稳定性、局部能量/谱形。历史结果没有给出一个跨模型可复制的数值坐标系。本 pilot 不训练增强头，也不把任何一个坐标预先称为原因；它只回答每个 TFG 的原生输入空间中，三类单因素操纵是否能在保持相同音频 RMS、样本时钟和内容的前提下改变 SyncNet，并且是否同时改变可解释的嘴部运动量。

评分音频始终是未修改的自然 PCM，最终视频若有音轨也换回同一自然 PCM。增强/操纵特征只进入生成器。SyncNet、嘴部开合和嘴部速度是并列终点；口部运动量本身不是真值标签。

## 最小预算与停止门

- 先做 CPU/原生前端资格检查：2 个未见说话人、3 个 TFG、每个因素各 1 个固定强度，确认帧数、有限值、纯度和恢复路径。
- 资格通过后只渲染 2 个未见说话人 × 3 个 TFG × 4 臂（RMS-identity、phoneme-separation、segment-stability、local-envelope/spectral-shape），共 24 个视频；不打开旧四中心、Dev/Confirm 或训练。
- 若任一模型的三个操纵全部不能通过原生前端或 purity 门，停止该模型，保留其它模型；不以工程失败写成科学阴性。
- 扩展门：只有至少 2/3 模型、至少 1 个因素在两个说话人上同方向，且共同口部运动端点未恶化，才允许另立预算做 6–10 说话人的确认。否则回到坐标设计，不扩大样本。
- pilot 的逐说话人差值只做方向和可行性；不报告跨说话人显著性，不用单 pilot 选择生产路由。

## 冻结不变量

1. 输入为同一自然 WAV、同一文本和同一参考图；每个 TFG 都使用同一 16 kHz mono canonical PCM，目标 RMS 固定为 `0.0376838172675746`，样本数逐样本相同。
2. 特征序列的帧数、时间边界和 renderer seed 固定。MFA/TextGrid 只用于预先冻结 phone/segment mask，不按结果改边界。
3. 生成器输入分别为各自原生 frontend 的最早 renderer-consumed tensor：Wav2Lip 的 80-bin mel chunk；MuseTalk 的 pinned Whisper/audio-encoder sequence；Ditto 的 pinned HuBERT/ONNX sequence。不得把 Wav2Lip 的 mel 坐标直接冒充另外两种模型的坐标。
4. 变换在独立 FIT 说话人上估计方向和尺度；两个 pilot 说话人只用于执行。所有音频、视频和 SyncNet 产物写入新 run，旧原件只读。

## 三个单因素坐标

### A. 音素间表示分离（SEP）

在 FIT 前端特征上按 phone 计算冻结原型和最近竞争 phone。对 pilot 的每个 phone segment，只沿 `prototype(phone)-prototype(nearest_rival)` 方向移动 segment 的所有帧，并以预注册的半个 FIT 内部标准差为最大剂量；逐帧范数和全局特征 RMS 记录为 QC。该操作保持 segment 内时间顺序，目标是只增加 segment centroid 的 phone 间距离。若 segment stability 或 frame-norm CV 的变化超过其 FIT 参考分布的 2 个标准差，SEP 标记为非纯坐标，不进入正向解释。

### B. 局部片段稳定性（STAB）

在每个 phone segment 内对去均值残差做固定 25% 的时间平滑，再重建原 segment centroid；不跨 phone、不改帧数。目标是降低相邻帧 cosine cost，同时尽量保持 phone centroid separation 和 frame-norm trajectory。segment centroid 位移、phone separation 和 frame-norm CV 都必须单独报告；任一越过 FIT 纯度门即标记为耦合操纵。

### C. 局部能量/谱包络（ENV）

只改变 native sequence 的每帧尺度包络，保持每帧方向/cosine 不变。Wav2Lip 的 80-bin mel 和 MuseTalk 的 Whisper log-mel/audio-encoder 输入可同时报告真正的频带包络；Ditto 的 HuBERT/ONNX hidden sequence 没有可解释的频率轴，只报告局部 frame-norm energy envelope，禁止在隐藏维度上伪造“频带”或 PCA 频谱。方向由 FIT 中 TTS−natural 的中位局部轨迹预先固定，幅度为 FIT 中位绝对变化的 50%；序列总特征能量和输入音频 RMS 保持。若 phone separation 或 segment stability 改变超过 FIT 纯度门，ENV 降级为“局部包络与其它坐标耦合”；若 Ditto 不能在其原生序列入口保持该约束，则该模型的 ENV 为 `UNAVAILABLE`，不改用伪频谱坐标。

每个因素都保留一个 no-op RMS-identity 控制。若实现支持，额外保存反号 CPU-only sanity，不进入主分母，用于确认方向不是数值符号错误。

## 视觉端点

从每个生成视频逐帧提取稳定的人脸/嘴唇 landmarks，计算：

- `aperture_t = distance(upper_lip, lower_lip) / interocular`；
- `aperture_velocity_t = abs(diff(aperture_t)) * fps`；
- phone/segment 边界附近的 aperture velocity 与 native feature energy/transition 的预定窗口相关；
- 全片段均值、标准差、95%分位和有效帧率。

SyncNet 提升只有在视频生成成功且嘴部运动端点没有质量门失败时才标记为 `motion-consistent`。运动变化不等于真实嘴型准确；没有人工或真实视频标签时，结论只能是“与口部运动同步变化/未同步变化”。同时保存 SyncNet 的 B、D、固定 lag 距离分解，避免把 B 上升写成真实匹配改善。

## 主表与预先判定

每行固定字段：`speaker_id, sample_id, model_id, factor, audio_pcm_sha256, native_feature_sha256, frame_count, rms, sample_count, C, D, B, lag, aperture_mean, aperture_velocity_mean, phone_separation, segment_stability, envelope_cv, purity_status, render_status`。

相对 RMS-identity 的主差值为 `delta_C`, `delta_D`, `delta_B`, `delta_aperture_velocity`。一个因素的 pilot 支持条件是：操纵目标按预定方向移动至少 FIT 参考标准差的 0.5 倍；两个非目标指标各不超过 2 个 FIT 标准差；两个说话人均无音频/视频质量失败。否则保留完整结果但标记 `NO_COORDINATE` 或 `COUPLED`。

只有同时满足 `delta_C > 0`、`D` 或固定 lag 距离不恶化、以及口部运动端点为 `motion-consistent`，才允许把它列为“值得确认的方向”。C 单独上升、B 上升而 D 不改善，归入“评分响应”而不是“真实口部运动改善”。

## 复用与禁止

复用 `scripts/16_feature_separability.py` 的 segment stability 定义、`scripts/20_causal_feature_interventions.py` 的 RMS/包络 QC、`scripts/audio_enhancement/` 的 canonical PCM 和 original-audio mux 合同，以及已有 10-speaker holdout 的独立计划。禁止复用旧四中心、旧 trainable Confirm 或按 SyncNet 分数挑句。不要运行 pytest；先做资格 probe，再按停止门决定是否渲染 24 行。

## 预期结果分支

- 三因素在不同模型方向一致且 motion-consistent：只说明存在共同方向，仍需新的跨说话人确认；不声称已有统一坐标。
- 只有 ENV 在 Wav2Lip 成功：保留为模型特定坐标，不扩展到 MuseTalk/Ditto。
- SEP/STAB 改变目标但 C 不变：说明 native representation 方向可操纵但不是当前生成/评分瓶颈。
- C 上升而口部运动不随之改善：优先解释为 SyncNet/评价器响应，暂停增强头训练。
- 三者都 COUPLED 或资格失败：坐标系尚未可复制，转向更低维、模型特定的 adapter，不增加样本规模。

## Changelog

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| 根据主线与证据地图把“共同方向”收敛成三个原生输入单因素坐标，冻结最小 24 行预算、纯度门、视觉端点和停止规则；尚未执行科学评分 | 2026-10-13 | user request; root theory |

## Eligibility amendment (2026-10-14)

MFA qualification completed on the initial scorer clips. Phone masks may cover only known non-silence phone intervals. `spn`, empty, and silence intervals remain the RMS identity input and are excluded from SEP/STAB. A row is eligible only when known-phone time covers at least 70% of the non-silence phone region; every row records the excluded intervals.

S0044 (`BAC009S0044W0266`) passes with 95.357143% known-phone coverage of its non-silence phone region. S0045 (`BAC009S0045W0283`) has 50.000000% and is retained as a valid Wav2Lip scorer qualification baseline only, not as a SEP/STAB factor row. The two-speaker factor cohort is S0044 plus `BAC009S0052W0386` (S0052), which passes with 84.905660% known-phone coverage; its one `spn` interval remains excluded and is not relabeled.

This amendment preserves the two-speaker budget and makes the mask eligibility explicit. It does not change ENV eligibility or the fixed RMS, clock, content, native frontend, SyncNet, and visual endpoints.
