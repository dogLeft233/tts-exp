## 1. 输入与视频控制

- [ ] 1.1 实现 prepare：冻结历史 cohort、同源时间映射和官方单人 crop 轨迹；以错误 hash、缺记录、轨迹不完整和错误时间基准测试确认 BLOCKED，交付 protocol.json。
- [ ] 1.2 实现三臂视频帧映射与独立编码；测试首尾、单调、±3 帧平台、nearest-even、帧数/PTS、逐帧来源和 mux PCM 恒等，确认 REAL_REPEAT 不复制完成产物。

## 2. 官方曲线与判定

- [ ] 2.1 实现官方 SyncNet wrapper 与矩阵导出；验证 scorer 实际音频 PCM、31 列形状、window 坐标、日志重建 parity、缺失/非有限输出拒绝及 cell 临时目录隔离。
- [ ] 2.2 实现固定 PLUS/MINUS 窗口与两段 offset 差；用已知嵌入序列经官方距离函数验证 +3 视频读取对应 −3 offset，确认边缘 padding 排除、三臂掩码一致、不足五行拒绝。
- [ ] 2.3 实现判定与描述性 bootstrap；合成用例覆盖重复性失败、基线模糊、17/18 条恢复边界、全局 C 不降但局部恢复成功及固定 22 条分母。

## 3. 可复核交付

- [ ] 3.1 实现阶段 runner、hash 续跑检查和终态保护；通过临时目录集成测试验证 clean `--stage all`、配置变化拒绝复用、已有终态不覆盖。
- [ ] 3.2 生成固定前四条匿名盲看包和表单；测试随机顺序可复现、无条件名泄露、无审阅者时 PENDING 且数值报告可以完成。
- [ ] 3.3 实现离线 validator 与 result.md；篡改帧映射、PCM、矩阵、计数或 final 判定时均应被拒绝，完整与 BLOCKED 产物均能说明已验证范围。

## 4. 执行与收尾

- [ ] 4.1 运行新包 pytest、受影响复用模块的相关测试、Ruff 和 `openspec validate diagnose-lrs3-real-video-local-timing --strict`；记录通过结果后启动真实实验。
- [ ] 4.2 按固定协议执行一次 22 条诊断并运行离线 validator；交付有效终态、实际完成 cell 数、完整曲线与盲看包。若 BLOCKED，交付证据与未完成项，保留未执行任务；科学失败则正常完成实验，不调参重试。
