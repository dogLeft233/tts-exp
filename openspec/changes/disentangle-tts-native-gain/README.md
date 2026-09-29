# TTS 原生增益来源：下游实验入口

状态：**IMPLEMENTED；PARTIAL**。A、CPU 诊断、人工包和独立校验已完成；B 因本机缺少可核验的 LeapTalk repo/checkpoint/adapter 而 `DEPENDENCY_BLOCKED`。运行日期：2026-09-15。

研究问题：TFG 在真实说话视频上训练，为什么合成语音反而取得更高的原生 Sync-C？本实验分别测量评估音频的影响、生成口型的影响，以及二者交互；同时检查声学干扰、节奏和特征几何的解释。

原生比较是 `SyncNet(G(T), T) − SyncNet(G(N), N)`，同时改变生成输入和评分输入。分数提高本身不能定位是哪一环节改善。

本次运行目录为 [`implementation_20260915_v1`](../../../runs/tts_native_gain_attribution_implementation_20260915_v1/report.md)。A 完成 180 个科学 cell 和 18 个控制 cell；v15 的 24 个 ORIGINAL 矩阵重放最大绝对误差为 `1.144e-05`（容差 `1e-04`），6 个恒等控制为 `IDENTITY_PASS`，12 个延迟控制为 `DELAY_DETECTED`。三项 A 主对比中，`A_T_NOISE_E_HARM` 的校正区间为 `[0.158, 1.011]`，`A_N_DENOISE_E` 为 `[-0.161, 0.021]`，`A_R_DENOISE_E` 为 `[-0.213, 0.059]`；这些是固定视频的评价端响应，不能直接称为生成口型增益。B 的固定分母仍保留 144 个视频、336 个科学评分和 8 个控制评分，未用其他生成器替代。独立校验为 `valid`，工程状态为 `PARTIAL`。

## 阅读与执行顺序

1. [proposal.md](proposal.md)：已知结论、研究动机、范围。
2. [design.md](design.md)：假设、因果比较、计算公式与结论边界。
3. [protocol.md](protocol.md)：固定样本、音频算法、矩阵、资源与统计。
4. [规范](specs/tts-native-gain-attribution/spec.md)：强制验收场景。
5. [tasks.md](tasks.md)：按依赖实现的任务卡。
6. [input-bindings.json](input-bindings.json)：本次写 spec 时核验的既有输入及 SHA-256。

先完成 A：现有 LeapTalk 视频与真实视频的固定视频实验，不生成新 TTS/TFG。然后完成 B：**同一 LeapTalk 模型家族**的新生成交叉实验。A/B 均为自动实验必需阶段；只完成 A 应报告 `PARTIAL`。B 缺模型或资源须准确标为 `DEPENDENCY_BLOCKED` / `RESOURCE_WAIT`，继续交付 A 和 CPU 分析。

**不能将 Wav2Lip 替换进 B 后称为 LeapTalk 机制解释。** 当前本机有既有 LeapTalk 视频，但未定位到可直接运行的 LeapTalk 环境/权重；历史部署记录指向已关闭的远端。B 的部署与 checkpoint 来源审计是实际工作，不是已经满足的前提。

## 最重要的执行约束

- 固定 LRS3 ID 151–162，共 12 个 source groups；这些样本已被观察，属于机制诊断，不是新样本确认。
- N 与 T 各自保留自己的时钟。交叉矩阵只在 `A0` 与其**同长度、同时间轴**的处理版之间建立，禁止直接将未对齐的 N/T 互换。
- 所有阶段保存完整距离矩阵及音频/视频特征，先时间均值再计算最小值和中位数。`C = B − D` 的代数分解不能充当因果机制。
- A 必做响度 −6 dB、20 dB 加噪、固定谱抑制三个干预。B 必做加噪和谱抑制，两次配对生成 seed。固定视频评分和重新生成严格分开。
- 谱抑制不自动等于“音质改善”；人工听评未返回时只能描述算法干预、声学变化和评分响应。
- 原生 advantage、固定音轨生成效应、人工同步判断分别报告。没有人工评分时 `PERCEPTION_NOT_ASSESSED`，不得假造感知结论。
- 不训练增强头、不搜索最优参数、不新调用云端 TTS、不重复旧 bridge sweep。两种 TTS provider 的既有 bridge spec 仍是独立问题。

## 预计工作量与终态

A：180 个科学评分配对 + 18 个控制配对；包括原始音轨、共同 headroom 基线及其特征重算。B：144 个新视频 + 336 个科学评分配对 + 8 个控制配对、4 个重复生成视频。另有 A 的 396 个错误内容特征配对，复用特征在 CPU 上计算，不能计为 396 次模型前向。

同一音视频特征可缓存复用，**评分配对数不等于前向进程数**。禁止为凑数重复渲染。人工包的构建与分析程序必交；实际人工评分不是自动可完成项。

自动终态必须同时给出 `engineering_status`、逐阶段状态、各假设证据与未解决项。A/B 数据完整且独立验证通过才是 `AUTOMATIC_COMPLETE`；它不等于全部机制已解释，也不等于感知改善。

## 已确认的资源状态

编写时本机 Tesla V100 16 GiB，约 5 MiB 使用、0% 利用率；根盘仅约 1.9 GiB 空闲。此状态只是一份快照。A 可流式复用已有 crop，B 的模型部署不能假设现有空间足够。每一 GPU 阶段重新做资源检查，详见 protocol。

## 上次实验的正确入口

有效完成 run 是 [`review_v15`](../../../runs/lrs3_tts_gain_mechanism_review_v15/report.md)，不是资源等待的 v13，也不是元数据不完整的 v14。旧 spec 的 README 曾保留 v13 状态，以 v15 的报告、final 和独立 validation 为准。
