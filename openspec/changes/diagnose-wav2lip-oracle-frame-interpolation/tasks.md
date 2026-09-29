## 1. 恢复与冻结

- [ ] 1.1 全文读取本 change、design 指定 BM 笔记和 spec 指定继承规则；读取/更新同一 BM `Wav2Lip oracle frame interpolation 2026-09-07`，准备执行时置 running。
- [ ] 1.2 实现父九文件 hash/自哈希/实际引用审计；复算全部 22 条历史指标并重现 ORACLE_OWN_AUDIO_UNRESOLVED；冻结本轮协议、全部映射/掩码/44-cell 清单和执行身份。

## 2. 最小实现与测试

- [ ] 2.1 实现新 runner 的单一 V_LINEAR 像素公式、22 流/44 mux 身份校验和 44-cell fresh forward，显式保留 88 个历史 cell 引用。
- [ ] 2.2 实现 common/local 统计、四种科学终态和独立 gain 判断；实现离线 validator，不共享 producer 的核心计算。
- [ ] 2.3 完成 spec 指定的公式/边界/证据篡改/门槛测试，通过 focused pytest、ruff 和 OpenSpec strict validation。

## 3. 一次正式实验与交付

- [ ] 3.1 使用现有 SyncNet CPU 环境完成 22 条、44 个新 cell；独立运行 validator；发生工程错误时留下证据并按修复规则处理。
- [ ] 3.2 输出中文 result.md：父/本轮 timing、own/damage/gain 的均值和 CI、唯一终态、缓存/新评分数量、验证结果、混帧解释局限。
- [ ] 3.3 全文读取后更新同一 BM 笔记为实际状态并读回核验；交付 run 路径与 BM permalink。科学失败不循环搜索参数，不自动开启下一实验。
