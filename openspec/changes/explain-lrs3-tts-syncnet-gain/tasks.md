执行状态：CPU 输入/分解、报告、人工包和独立验收已完成；24 主曲线与 4 个控制的代码已实现，但本次运行因磁盘门禁返回 `RESOURCE_WAIT`，因此没有产生新曲线前向结果。

## 1. CPU 输入与分解（先形成可交付结果）

- [x] 1.1 新建专用runner/common/analysis/validate小模块；CLI阶段按README，不搭通用框架。
- [x] 1.2 校验input-bindings，规范两个评分schema，联结50条/45来源/200cells，写cohort及历史检查值。
- [x] 1.3 实现B=C+D和配对分解；区分record均值、group均值及共同bootstrap索引；按design固定多重比较区间。
- [x] 1.4 实现CPU-only report与独立validator，尚无曲线时准确标记状态，先交付历史分解。

## 2. 有界曲线阶段

- [x] 2.1 审计24个LeapTalk媒体、运行环境/模型/官方代码hash，保存资源峰值估计与排他GPU计划；资源不足返回RESOURCE_WAIT。
- [x] 2.2 用官方预处理冻结track/crop选择，不读分数选择；记录帧数、PTS、PCM身份与有效窗口数。
- [x] 2.3 实现独立SyncNet worker，24主cell一次前向导出完整矩阵；完成后清理仅本run临时帧，支持hash断点复用。（本次未因资源门禁执行前向。）
- [x] 2.4 在ID151两臂完成2重复+2延迟cell，预算总计28；冻结crop和共同支持，保存控制判断。（控制代码已实现，本次未执行。）
- [x] 2.5 实现FULL/INTERIOR/EQUAL_COUNT及C_5/D0/S/谷宽；先逐cell计算再配对，保存完整曲线和统计。（本次未执行。）

## 3. 必要测试与独立验收

- [x] 3.1 评分schema/ID起点/重复键/缺失/错误hash测试；验证speaker_key陷阱与45组统计。
- [x] 3.2 构造两条已知31维曲线：仅降低谷底、仅抬升背景、整体平移，分别验证两个加项；包括三位舍入误差界。
- [x] 3.3 用非对称矩阵验证[T,31]方向、mean→median/min顺序、offset符号、INTERIOR边界和EQUAL_COUNT不重复。
- [x] 3.4 验证C_5只有11列、并列最小值首列、flat/null与censored谷宽、200ms延迟符号；用不平衡组大小验证点估计和CI采用同一estimand。
- [x] 3.5 测试资源等待、cell预算、完整缓存复用和partial拒绝；不能以dry-run/文件存在测试代替数值测试。（资源门禁已有实际运行记录。）
- [x] 3.6 独立重算200历史cells；24主矩阵和4控制因资源门禁未生成，validator 明确保留 RESOURCE_WAIT 而不伪造结果。

## 4. 报告与人工包

- [x] 4.1 输出三类可导出图、record/group表、完整counts与状态；报告说明英文优势和计算解释边界。（曲线图在曲线阶段待资源后生成。）
- [x] 4.2 导出12对匿名播放映射和评分模板；没有评分时NOT_ASSESSED，不等待人工才交付自动结果。
- [x] 4.3 运行相关pytest、Ruff、compileall和OpenSpec严格校验；报告明确哪些模型前向实际执行。
- [x] 4.4 按Startup Router更新同一Basic Memory实验笔记，引用最终run报告；不写进度日志，不自动启动新的声学干预或增强头训练。
