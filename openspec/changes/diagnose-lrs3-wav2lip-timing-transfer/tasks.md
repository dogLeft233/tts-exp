## 1. 冻结输入与诊断规则

- [x] 1.1 新建独立实验模块及 runner 阶段入口，实现 22 条历史资产 join、入口/媒体 hash 与 PCM/PTS/帧数审计；验证错配/缺失输入会汇总到 input_audit 并 BLOCKED，N 与 cohort natural PCM 相同、W 可精确复建。
- [x] 1.2 实现原音频周期的正/逆映射、共同候选行、PLUS/MINUS 和四项预期 offset；用合成坐标与错符号测试验证方向，验证生成视频较短仍保留原长 PCM、每段不足 5 行时阻塞。

## 2. 媒体与统一评分

- [x] 2.1 复用 tail_v2 轨迹生成 66 份裁后视频流和 132 个交叉 mux，适配官方 worker 产出 132+44 个新评分 cell；通过小型媒体测试核验 N/W 视频流一致、PCM 正确、时间轴不变、重复评分独立以及缓存/部分结果保护。
- [x] 2.2 实现矩阵与日志一致性检查、四项局部计数、固定优先级终态、共同窗口 C/D 与 bootstrap 报告；用合成矩阵验证所有终态、峰间距/边界、误差容差、17/18 与 19/20 计数边界。

## 3. 验证、执行与交付

- [x] 3.1 实现离线 validator 和篡改测试：从媒体/矩阵独立复算，能拒绝缺 cell、换音轨、PCM/PTS/hash/掩码/终态篡改；运行对应 pytest、Ruff 和本 change 的 OpenSpec strict validation 全部通过后再执行完整实验。
- [x] 3.2 在新 run 完成 prepare→media→score→report 与独立验证，交付 22 条/132 主 cell/44 重复 cell 的分析、final、result、validation；科学失败照实交付，不调参数重试。若工程 BLOCKED，保留失败证据并说明未完成项，不将完整运行任务勾为完成。
- [x] 3.3 按 Startup Router 将结果收录到 BM `tts-exp` 的同一实验笔记，搜索去重并写后读取验证状态一致；交付 run 与 BM 指针，明确历史 CONTROL_FAILED 未被改写、音频头 eligibility=false，并更新本清单实际完成项。
