## 1. 输入与静态协议

- [x] 1.1 实现独立 prepare 入口并冻结父 cohort、N/B/S decoded PCM、模型及执行契约；验证 22/22 keyed join、原顺序及篡改 hash 拒绝测试。
- [x] 1.2 提取每条第 0 帧 PNG、固定生成/评分框和输入 source-index；交付 22 张框图并通过动态帧混入、换图、整帧兜底的拒绝测试。
- [x] 1.3 构造 RT 和 ND、验证原 S 四分之一交换，冻结生成长度、W 与感受野映射；通过 PCM 逐样本、+3200 补零、尾部及拼接边界测试。

## 2. 生成、评分与测量校准

- [x] 2.1 实现固定 PNG 五臂渲染、独立 repeat 和无损审计输出；单元测试确认音频 arm/参考图绑定、resume 契约及失败重试计数，禁止成功 cell 重跑。
- [x] 2.2 实现固定 crop 的官方 SyncNet 矩阵、共同 W、C/D/k 与 D_anchor；通过官方相同支持 parity、lag 符号、先均值后取最小值及绝对索引测试。
- [x] 2.3 实现独立 validator 和 A 门槛计算；用篡改矩阵、错绑定音轨、伪造 repeat、缺失记录负例证明 validator 会拒绝，不能仅检查 runner pass 字段。
- [x] 2.4 执行 A 的 44 视频/66 cells 并独立验收；交付 PASS 或 MEASUREMENT_CONTROL_FAILED 全量报告，失败时验证 B 为零。

## 3. Bridge 与 LOCAL_SWAP

- [x] 3.1 A 通过后执行 RT/B/S，新增 66 视频/132 cells；核对总计 110/198，无候选、参考或强度扩展。
- [x] 3.2 分别计算 B−N、B−RT 的组 bootstrap、收益/安全门槛、未舍入胜负比例及 mel 诊断；用可手算 fixture 验证符号、分母和 CI 跨零的未建立收益状态。
- [x] 3.3 分段计算 p_N/p_S、D0 与支持掩码，独立给出 swap_transfer；验证两个段不被合并、支持不足不筛样、swap 未通过不覆盖 bridge_gain。

## 4. 完整交付

- [x] 4.1 全量独立重算验收并交付 final/validation、CSV、矩阵及可离线播放页；逐条核对 22 份 N/RT/B 同音轨对照和 N/S 四配对，人工未审则明确标记。
- [x] 4.2 交付面向新读者的报告并更新同一 BM 实验实体；保留历史结果边界，明确静态收益不证明嘴型泄漏。运行对应 focused tests、格式检查和 OpenSpec strict validation，记录结果。
