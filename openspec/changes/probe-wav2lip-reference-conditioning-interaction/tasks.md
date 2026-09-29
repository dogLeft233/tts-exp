## 1. 实现与锁定

- [x] 阅读公共契约/本change及BM，更新本实体running；锁定四cell ID连接。
- [x] 实现F46固定帧/box提取、输入退化检查和合成测试；锁定所有参考后再评分。
- [x] 实现本包adapter、四cell分析和独立validator，不修改共享worker。

## 2. 执行与验收

- [x] GPU锁内完成F0 replay和F46 N/repeat，执行parity与F46 delay控制。
- [x] control未通过，按门禁未生成/评分16个F46_C。
- [x] 由于control gate失败，保存CONTROL_FAILED终态、限制和self-review，未计算主I。
- [x] 跑本包pytest及 `openspec validate probe-wav2lip-reference-conditioning-interaction --strict --no-interactive`。
- [x] 更新/读回本BM实体，报告实际预算，不扫更多参考图。
