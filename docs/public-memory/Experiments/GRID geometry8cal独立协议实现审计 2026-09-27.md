---
title: GRID geometry8cal独立协议实现审计 2026-09-27
type: experiment
permalink: tts-exp/experiments/grid-geometry8cal-独立协议实现审计-2026-09-27
status: concluded
---

# GRID geometry8cal独立协议实现审计 2026-09-27

| Notes | Modification Date | Approved By |
|:--|:--|:--|
| root授权只读协议/实现审计与纯合成数值核验 | September 27, 2026 | user/root |
| v1/v2审计修订留存，v3独立PASS并按授权交接8cal | September 27, 2026 | user/root |
| root追加8cal结果独立复核，等待执行方主动交付 | September 27, 2026 | user/root |
| 结果审计PASS，确认校准3/8 FAIL、空间16项失败与RAW-only lag，保持eval锁定 | September 27, 2026 | user/root |
| 新r测量开发审计planned，冻结共同支持SD/噪声门与r破坏控制，eval仍锁 | September 27, 2026 | user/root |
| 新开发门7/8独立复核PASS；旧3/8 FAIL保留，独立eval仍需另冻 | September 27, 2026 | user/root |

## Observations

- [r_development_status] concluded：新固定配对aperture形状相关测量开发静态/结果独立审计均PASS，开发门7/8 PASS。8cal明确为已暴露开发集；旧3/8 FAIL保留，本轮不解锁16eval，不计算处理差或运行模型/GPU。
- [r_development_result] 从432旧科学数组用独立complex-LS重算，864hash核验，3296标量max差2.42e−13。仅s5 FIXED SCALE1.1 r=.9497408965<.95失败；三域共同41帧SD .0375173–.0697538，SD/max8空间RMSE=2.7163–9.7853，全部过新运动门；REVERSE/两WARP的repeat r下降 .133231–1.227612，72/72过.05。其余保留门全过。
- [r_development_conclusion] 通过仅支持下一步设计更窄的形状相关留出研究，不是独立效度验证，不能称纯时间/幅度/完整嘴型保真改善。新门在新聚合前冻结但旧cal数值已被揭示，保留开发选择历史。条件相关测量偏差仍未排除，未来同扰动处理差稳定性及lag0唯一主端点须root另冻。
- [r_development_report] runs/grid_geometry_protocol_audit_20260927/r_development_review/report.md SHA24ddedc1fc0c8dbe7531d73cde02142ae5d0741030181565095b151664a7892d；receipt.json SHA58ae6a60e1e60a47b6332265042f5b1149cef9e451ff1492fb6e796df7ce9d40。新增审计约140KiB，整个audit树约584KiB≤1MiB。

- [followup_status] 结果独立复核 concluded，审计PASS/科学校准FAIL。收到封存产物后获授权读取8cal科学数组；复数最小二乘独立重算2688标量max差1.11e−14，864科学文件hash一致，原协议审计报告/hash保持。
- [followup_result] 固定8源仅s2/s7/s8即3/8通过完整门（要求≥6）；全部16/192失败为空间控制，均centered RMSE>.01，另s5 FIXED SCALE1.1 r=.949741<.95。时间控制216/216、自域shift96/96、RAW跨域8/8及relative shift32/32通过；共同支持全部41帧；24个FROZEN r按预设NA，无其他NA。
- [followup_result] RAW-only8源选lag−3：real[t]对generated[t−3]；同支持source均值r由lag0 .169412到校正.760316。科学数组seal→diagnostic_lock→lag_lock→diagnostics顺序核实。校正r属于选lag所用cal描述，不是独立泛化验证，不指定某生成器/音轨独占物理延迟。
- [followup_conclusion] 当前检测/插值/配准组合未达到冻结空间鲁棒性，不能单指MediaPipe某层或断言所有几何测量无效；时间可测性与空间鲁棒性分开。REVERSE/WARP≥.10是经验灵敏度门而非数学必然，本次均通过。不调门、不选3源继续，16eval保持锁定，无处理效应汇总。
- [followup_engineering] Path.with_suffix小数视图文件名截断在评分前暂停；独立22对映射/hash证明无覆盖，2对WARP无损rename。v5相对/绝对依赖hash矛盾在resume前阻断，v6 canonical统一后PASS；科学值/参数未改，版本保留。
- [followup_report] runs/grid_geometry_protocol_audit_20260927/result_review/report.md，SHA361986c53998176cbf979c3127c0a7377cccb7f60fc21661a42567622ae5e747；receipt.json SHAb996fe8a9a67e378b7f322c8ce1841dd12efd91831f641772bb9aa1f23d3fe5a。审计树+结果checker约452KiB，≤1MiB；GPU=0。

- [status] concluded。
- [scope] CodeGraph先定位，独立只读代码/协议/元数据与纯合成数值审计；GPU=0，真实媒体/处理效应读取=0，不编辑visual产物。16eval模型分析仍锁定。
- [result] v3协议0522013aeb2b898a0f8c0f8f56c403a77b6f98646b2dc8a30a32a3ce52ec0621及代码快照/hash PASS；双运行环境独立匹配。44个global/relative shift合成组合符号/闭域PASS；实际lag_lock AST八源选global+3，FIXED访问0。
- [result] 准周期正aperture反例的E(−4/−2/+2/+4)=−0.956/−2.358/−2.470/−0.865，但四shift唯一精确恢复。root据纯合成证据在所有forward前批准新GRID幅度单调性仅diagnostic；符号/≤1帧/33支持门不变，旧LRS3不修改。
- [result] v2零动态RAW在空E比较处TypeError，v3相同合成fixture完整结束且source failed；checker补齐独立lag源资格/计数、生成BASE QC、lag0、None-aware比较和runtime/hash入口。v1/v2原件保留。
- [conclusion] 协议/实现审计PASS，已按root授权通知visual开始8cal；不预先保证真实输入/科学门通过，不构成16eval解锁或嘴型改善结论。
- [limits] 41帧interior14..54、≥33valid、源0..68避尾钳位；canonical acquisition PCM16桥接，target0.0376838172675746，RAW跨域E仅描述。执行预算20MiB持久、tmpfs64MiB、自身峰RSS+childRSS+tmpfs≤1GiB、磁盘≥5GiB，实际峰值仍由运行门判断。审计约64KiB（不含BM），≤1MiB。
- [report] runs/grid_geometry_protocol_audit_20260927/report.md；SHA256 985fc174f6126e9afbda34a9099b8baf805fbb641e934d5a2c26c81f2b09b045。final_audit.json、artifact_hashes.json及修订/合成证据同目录。

## Relations

- relates_to [[FIXED mel未饱和与饱和区域生成四格 2026-09-27]]