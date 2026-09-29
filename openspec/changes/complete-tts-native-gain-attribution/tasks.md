# 下游任务与完成证据

实现快照：2026-09-16。CPU 修复、父证据导入、A 复验、B 成功 fixture、阻塞态全链路和独立 validator 已执行；正式 B 仍因缺少可核验 `LEAPTALK_CONFIG` 阻塞。因此本 change 当前是 `PARTIAL/DEPENDENCY_BLOCKED`，不能标为 `AUTOMATIC_COMPLETE`。勾选只表示该条有真实产物或测试，完整实验条目仍须保留准确的未完成边界。

## 1. CPU修复与父证据

- [x] 1.1 核验本change evidence-bindings及父传递资产；创建新run，保存只读parent/reuse manifest和组序，验证A180+18、操纵、重放、诊断。
- [x] 1.2 逐项审查design缺口，实现adapter请求/响应、严格provenance、消费输入现场证明、独立repeat路径与原子续跑。
- [x] 1.3 实现冻结基线ROI、候选轨迹复用、双seed共同支持、完整七格评分与8控制；CPU fixture 已覆盖成功路径。
- [x] 1.4 实现来源统计、96四格分解、B主对比、fresh native和基于B的H2状态；修复all聚合/非0退出码。
- [x] 1.5 补齐盲评实际音轨、随机题序、隐私字段、缓存、非破坏重建和真实评分bootstrap/验证；B 阻塞时媒体如实为 PARTIAL_MEDIA。
- [x] 1.6 实现独立B验收及design全部篡改负例，成功fixture和中断续跑/GPU竞争/磁盘不足测试通过。已覆盖成功 fixture、矩阵/自哈希/资源/配置负例，以及中断媒体隔离和外部 GPU 竞争的 typed `RESOURCE_WAIT` 日志测试；其余完整 B 成功态仍受外部依赖阻塞。

## 2. 部署与完整smoke

- [ ] 2.1 核实已有授权环境/磁盘，固定官方repo/revisions，核算所有持久和临时空间；分开GPU峰值预算。资源快照和分离预算已落盘，LeapTalk revision 尚未绑定。
- [ ] 2.2 完整绑定实际加载权重/config/patch/环境，核验V100或所选设备兼容性；新配置标签正确。
- [x] 2.3 实现共享设备锁和实时资源监控；不抢占其他进程。共享锁、foreign PID 检查、三次空闲采样和资源等待状态已实现并记录。
- [ ] 2.4 独立smoke目录完成ID151两source、两seed、三driver、两个repeat及交叉/控制/分析/盲评媒体链；修复后冻结正式代码和模型。

## 3. 正式实验

- [ ] 3.1 生成144科学视频和4独立重复，完整校验PCM/frontend/时间映射/肖像/模型/RNG证明。
- [ ] 3.2 完成336科学评分和8控制、矩阵/特征/ROI/共同支持绑定，控制通过。
- [ ] 3.3 重算六主对比（12组、seed先平均、同一20000次bootstrap、六项校正），96分解和规定次要诊断/图/逐组表。
- [ ] 3.4 产出106个可播放同步题与53个音质题；无人工评分保持未评估，有则独立分析。

## 4. 自审与交付

- [ ] 4.1 独立validator从原始证据通过完整成功路径，并拒绝伪完成和所有身份篡改。当前 validator 已独立通过真实阻塞态，完整 B 成功态待外部依赖。
- [x] 4.2 完成implementation_audit问题—修复—验证映射；修复全部影响科学有效性/完成度的问题后再审。
- [x] 4.3 交付中文报告、实际命令、final/validation、模型/资源记录；明确阳性/阴性/不确定/未识别边界。
- [x] 4.4 复核父run字节未变、当前代码/新spec快照一致；更新BM原笔记及下游入口。父run传递 hash、当前代码/spec快照、BM原实验笔记、blocked任务和Task Board均已核对。
- [x] 4.5 只有真实全量数据、控制与验收满足时标AUTOMATIC_COMPLETE；否则保留准确阻塞和未勾选任务。本 run 保持 `DEPENDENCY_BLOCKED`，未伪造完成。
