# 下游任务卡

本文件的勾选表示实际实现/执行/验收，不因本 spec 写完而勾选。先读 README、design、protocol 和 normative spec；数值以 protocol v1 为准，发现文档矛盾先修正并冻结，不选择有利解释。

## 1. 输入、接口与可运行骨架

- [x] 1.1 用 CodeGraph 查看可复用的 v15 worker/analysis、GPU lease 和历史媒体审计；记录实际代码位置，不直接复制旧结果判定逻辑。
- [x] 1.2 新建 `scripts/experiments/tts_native_gain_attribution/`，提供 runner 与独立 validate，CLI 包含 `audit/audio/fixed-video/generate/crossed-score/analyze/perception-pack/perception-analyze/report/all`，run-id 与 resume 支持。
- [x] 1.3 核验 input-bindings、v15、12组、24历史crop/原媒体与12真实视频；冻结 assets/protocol/claim_registry/代码快照。
- [x] 1.4 核查实际 LeapTalk checkpoint/部署依赖、肖像规则与资源预算，区分已知训练证据和 UNKNOWN；不能让 B 部署阻塞 A 的 CPU 准备。

## 2. 等时钟音频与可追溯数据

- [x] 2.1 实现P1的 ORIGINAL/A0/GAIN/NOISE/DENOISE、共同headroom、PCG64 noise 和固定 STFT/iSTFT。
- [x] 2.2 保存每变体的长度、PCM hash、峰值、RMS、有效SNR、谱抑制操作效果和时间操纵检查，完成恒等/已知噪声/无裁剪测试。
- [x] 2.3 固定 A 的 crop/ROI/PTS 和解码像素，生成完整逻辑cell清单，不让按需循环隐式遗漏条件。

## 3. A：已有视频的评估输入干预

- [x] 3.1 实现原始 audio/video embeddings 与 `[T,31]` 导出，冻结 common INTERIOR，确认 ORIGINAL 重放与v15一致。
- [x] 3.2 串行执行180科学/18控制逻辑cell，复用可信特征，报告新前向量、缓存量与失败量。
- [x] 3.3 完成raw/unit范数几何、396规划错配与文本核验、固定视频6个控制家族，保存解释限制。
- [x] 3.4 用独立validator验证A并交付中间报告；B没跑时必须PARTIAL。

## 4. B：同一LeapTalk的生成/评分交叉

- [ ] 4.1 `DEPENDENCY_BLOCKED`：本机没有可核验的 LeapTalk repo、checkpoint 和 provenance-bound adapter；资源门禁本身通过时仍不绕过身份要求。
- [ ] 4.2 `DEPENDENCY_BLOCKED`：未生成 144+4 视频，避免用其他生成器或不明 checkpoint 填补。
- [ ] 4.3 `DEPENDENCY_BLOCKED`：未执行 336+8 评分，固定分母与 blocked rows 已保存。
- [ ] 4.4 `DEPENDENCY_BLOCKED`：fresh native 与 B 重复性只能待同一 LeapTalk 配置恢复后计算。

## 5. 统计、独立证据与报告

- [x] 5.1 完成六主对比及统一六项校正、四格G/E/I、C=B−D、次要几何/优势收缩和预定声学/节奏描述；保存bootstrap索引。A 三项已计算，B 三项按固定分母保留为 `INCOMPLETE`。
- [x] 5.2 输出科学图：A的固定视频效应、B四格与分解、ORIGINAL/headroom/fresh native并列、raw/unit几何与内容检索、逐组响应；B 图在数据不可用时保留空/不适用状态。
- [x] 5.3 构建96对同步包与48对音质包、匿名映射/评分模板/人工分析程序；无人评分时正确NOT_ASSESSED。
- [x] 5.4 编写面对项目外读者的report：动机→假设→操作→公式→结果/区间→路径结论→尚未识别的问题。不能只给GO/NO_GO。
- [x] 5.5 完成独立validator及P7的篡改/置换/缺失/资源负例测试；审阅到spec与数据身份/统计一致。
- [x] 5.6 按真实状态更新Basic Memory实验笔记和本README。只有A/B及全部自动必需产物通过才记AUTOMATIC_COMPLETE；实际人类评分单列。

## 明确不计入本 change 完成条件

音素节奏因果干预、新TTS provider合成、两provider bridge重跑、Wav2Lip替代、第二生成器复现、增强头训练、新来源确认。它们可由本轮证据引出后续spec，不能由本轮执行者临时搜索或借其结果挽救主检验。
