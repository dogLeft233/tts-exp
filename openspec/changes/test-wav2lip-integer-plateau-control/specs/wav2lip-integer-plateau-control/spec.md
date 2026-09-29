## ADDED Requirements

### Requirement: Freeze one bounded diagnostic and its evidence

实验 SHALL 固定 design 的父文件 hashes、全部 22 records / 22 source groups 和顺序，ordered-ID hash `5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`，分类 `seen_fit_diagnostic`。SHALL 核对父 validation 与 final 绑定、JSON 自哈希及实际使用资产。缺失/不符为工程 BLOCKED，不换 cohort、run 或分母。

评分前 SHALL 冻结 A/B 全部细胞清单、映射、支持掩码、阈值、模型/源码/环境和本 proposal/design/spec hashes。tasks 可更新而不进入冻结 hash。SHALL 读取父 ROI 和 oracle capability spec 的媒体/评分契约；本 spec 明确替换干预、掩码和比较基线，其余配置沿用。不把旧正弦 W 混入 P。

#### Scenario: Historical failure remains historical

- **WHEN** 父状态是 CONTROL_FAILED / ORACLE_OWN_AUDIO_UNRESOLVED / LINEAR_OWN_AUDIO_UNRESOLVED
- **THEN** 只要资产完整就可执行本轮新控制，但不修改旧状态，也不把本轮结果计为旧 gate 通过

### Requirement: Construct sample exact integer plateaus

令 N 为父原长 L 的 mono 16k PCM16，X 为 V_ID 的 F 帧 uint8 BGR24；m=floor(F/2)，b=640*m。SHALL 用整数数组直接索引：

```text
s_v[i] = +5 if i<m else -5                 (i=0..F-1)
q[i]   = i+s_v[i]
V_P[i] = X[q[i]]
s_a[n] = +3200 if n<b else -3200           (n=0..L-1)
P[n]   = N[n+s_a[n]]
PLUS   = arange(25, m-25)                  # stop exclusive
MINUS  = arange(m+25, F-25)
U      = concatenate(PLUS, MINUS)
Q      = U+s_v[U]                          # same ordered row correspondence
```

SHALL 检查所有源索引合法，输出保持 F 帧/L samples；不 clip、padding、重采样、混帧、淡入淡出、归一化或补尾。本干预明确允许中点处重复源片段和首尾源内容变化；它是合成控制而非用户可听候选。最终 replacement 的 N 始终是完整原 PCM。

每段 SHALL 至少 5 行。对 U 及 Q，SHALL 独立验证五视频帧、全部 31 列候选音频窗口及 MFCC 原始采样支持（含预加重前驱）合法；涉及 P 的支持必须完全位于同一平台，不能跨 b。SHALL 同样审核 Stage B 的 16-mel chunk 与 STFT/预加重支持不接触拼接或首尾填充。任何不满足为 BLOCKED，不再缩掩码、删样本或只保留部分距离列。

SHALL 保存 q、PCM 源索引规则、逐帧/PCM hash 和逐行支持证据；FFV1 无损完整解码逐字节等于 X[q]，25fps、PTS 从0且与 i/25 误差≤1ms。mux 只 copy 视频+完整 PCM s16le；禁止 AAC、-shortest 或调 offset。

#### Scenario: A row sees a splice through the offset search

- **WHEN** 中心行没有跨拼接，但任一 ±15 列的 MFCC 支持跨拼接
- **THEN** 输入审计 BLOCKED；不能只检查中心行就接受该 record

### Requirement: Run the oracle stage with transported content references

A SHALL 新建 22 个 V_P 视频流、66 个 mux，并 fresh forward `V_ID/P, V_P/N, V_P/P`，每条3个共66 cell；只缓存引用父22个 `V_ID/N`。V_ID 本身引用父流。所有使用的矩阵 SHALL 为完整 float32 [T,31]，同时保留 embeddings 和 worker日志；不通过移动旧 embedding 代替 fresh forward。

SHALL 使用固定 SyncNet 权重 SHA-256 `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`，CPU batch20/threads4/vshift15。距离为 `sqrt(sum((visual-audio+1e-6)^2))`，列 c 对应音频行 `r+c-15`，offset=`15-argmin`。只用已验证的真实支持，不利用补零边界。

记缓存距离为 M0。SHALL 比较 fresh `M(V_P/P)[r,c]` 与 `M0[q[r],c]` 的全部 U×31 元素，最大差≤0.001 为 transport parity，要求22/22；它是共同平移的等价性检查，不是独立神经网络复现。完整媒体无误但 parity 不符，记科学 ORACLE_PLATEAU_UNRESOLVED；篡改、支持/环境错误则工程 BLOCKED。

任意行集合 S 的曲线 SHALL 为矩阵对应行的逐列均值 z，`D=min(z), C=median(z)-min(z), off=15-argmin(z)`。峰清晰为最小与次小差严格>0.010且 |off|<15；并列 argmin 取首列。不能逐行算 C 后平均冒充曲线 C。

每条两段分别计算目标行 R 和源行 R+s（PLUS s=+5，MINUS s=−5），并按下表检查 offset 误差≤1且相关峰均清晰。B/C/O 每项至少18/22，每条必须两段都成功：

| 检查 | 实测 offset 差 | 预期 |
|---|---|---|
| B | off(V_ID/P,R)−off(V_ID/N,R) | s |
| C | off(V_P/N,R)−off(V_ID/N,R+s) | −s |
| O | off(V_P/P,R)−off(V_ID/N,R+s) | 0 |

baseline gate SHALL 要求 V_ID/N 的 PLUS/MINUS 与各自源行集合四个峰均清晰，且目标两段 offset 差≤1、源两段 offset 差≤1，至少20/22。SHALL 使用 U（目标时间）与 Q（对应源内容）定义：

```text
matched_own_C = C(V_P/P,U)-C(V_ID/N,Q)
matched_own_D = D(V_ID/N,Q)-D(V_P/P,U)
damage_C(V)  = C(V/P,U)-C(V/N,U)
damage_D(V)  = D(V/N,U)-D(V/P,U)
```

oracle own gate SHALL 为 matched_own 两项 CI 下界均>−0.10，且 common offset 与 V_ID/N,Q 相差≤1的记录≥20/22。oracle damage gate SHALL 为 V=V_P 的两项 CI 下界均>0.10且两项逐条同时正向≥18/22。A 通过要求 parity、baseline、B/C/O、own、damage 全通过；否则唯一科学终态 `ORACLE_PLATEAU_UNRESOLVED`，B不运行。

SHALL 另报 chronological own（用 V_ID/N,U）和原生 global 指标，仅描述。新内容匹配 gate 不得被称为旧 own gate 修复。

#### Scenario: Oracle scores are compared to different source content

- **WHEN** V_P/P 在输出 r 使用源 r+5 或 r−5 的内容
- **THEN** matched own 的 reference 使用 M0 对应源行 Q；报告同时给出 chronological 版本并明确二者不同

### Requirement: Test the actual generator only after independent oracle acceptance

仅当 A 所有 gate 通过且 `oracle_validation.json` valid=true、绑定证据未变，B SHALL 运行。SHALL 使用父原 face、缓存逐帧框、冻结 Wav2Lip GAN 权重 `ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8` 和官方 mel/回贴 worker，batch4，生成 `G_P=TFG(face,P)` 共22视频。SHALL 在目标时刻 i 使用原face/框i，不把face改成q[i]。

SHALL 复用原评分轨迹在目标时刻裁成224×224，无损存为E_P；预测生成帧数必须等于F，实际亦相同。不重检测、不静态脸替代或重复帧补长。随后 fresh `E_P/N, E_P/P` 共44cell。全实验最多110fresh+22cached评分、22个GPU生成视频；A失败时只有66fresh+22cached且GPU生成0。单独列出oracle派生视频22、评分ROI派生流0或22，避免混称新生成。

SHALL 比较 E_P 与 V_P 在相同目标行和相同音频下的结果：每条两段上，`E_P/N` 对 `V_P/N`、`E_P/P` 对 `V_P/P` 的峰均清晰且offset差≤1，两项各≥18/22。generated own 定义为 `C(E_P/P,U)-C(V_P/P,U)` 与 `D(V_P/P,U)-D(E_P/P,U)`，CI下界均>−0.10，common offset差≤1至少20/22。generated damage 使用前述 V=E_P，要求与A相同。四类要求全通过才记 `INTEGER_PLATEAU_CONTROL_SUPPORTED`，否则 `GENERATED_PLATEAU_UNRESOLVED`。

SHALL 同时报 E_P/P 相对 V_ID/N 的 Q 与 U 两种 own 差值，不能只报更有利的一个。不根据stageA结果调整stageB规则。

#### Scenario: Oracle passes but the actual chain does not

- **WHEN** A验收通过但生成链的timing、own或damage失败
- **THEN** 输出 GENERATED_PLATEAU_UNRESOLVED，暂停当前生成链分支；不能将失败单独归因于网络，也不能调用旧bridge阶段

### Requirement: Deliver fixed statistics independent validation and memory

所有差值 SHALL 先逐条计算，D方向正值代表改善；固定全22条、按排序source groups、PCG64/default_rng seed20260905、每指标重置seed、10000次有放回抽22组、线性2.5/97.5分位数。完整精度判定，Sync-C显示3位小数。不按峰、分数、时长或阶段结论删样本。

SHALL 写新 `runs/wav2lip_integer_plateau_control_<id>/`，至少包含 protocol/input_audit、audio/frames/media/scores manifests、fresh worker证据、oracle_analysis/oracle_validation、条件性generated_analysis、analysis/final/result/validation。工程不完整为 `engineering_decision=BLOCKED, scientific_decision=null`，列出已完成数量和准确阻塞；A科学失败但66cell完整可concluded。A通过B尚未完成只记中间状态，不伪报完整成功。

独立validator SHALL 从实际PCM/像素/支持与保存embeddings重建矩阵（容差0.0001），从矩阵独立复算映射、曲线、统计、计数、parity和终态（统计容差1e-6，离散判定完全一致）。不得调用producer的核心映射/统计/gate函数，可共享I/O；不声称重复神经forward。validation绑定final文件hash；A验收绑定A证据，不与final形成循环。

SHALL 测试平移符号、半开区间、中点拼接全31列支持、源行Q与目标行U区别、PCM/像素篡改、N/P交换、缺/重复cell、D符号、阈值等号、计数边界，以及A未通过却启动GPU的拒绝行为。正式执行前完成一次代码对spec自审，A和最终阶段各做独立产物验收，发现工程问题用新run，不覆盖父或已终态产物；只可恢复协议/代码/输入身份完全相同的完整cell。

final SHALL 总含 `historical_scientific_decision=CONTROL_FAILED`、`historical_gate_repaired=false`、`bridge_executed=false`、`training_authorized=false`、`reference_conditioned_audio_head_spec_eligible=false`、`generalization_established=false`。成功只代表当前已见数据上的平台控制可用；最终报告明确失败门槛及上述解释边界。

下游 SHALL 按 Startup Router/实验指令更新同一BM实验 `Wav2Lip integer plateau control 2026-09-07`，每次显式project="tts-exp"；搜索后读取全文再编辑，planned→running→concluded，工程阻塞保留running并注明execution_status=blocked。保留changelog，唯一status observation与frontmatter一致，写后读回。BM不可用则报告未同步，不伪报写入。

一次预注册方案完成 SHALL 停止；不搜索位移/拼接/掩码/阈值、不执行候选bridge、训练、其他TFG或sealed数据。下一步只有新的决策/spec，不能因本轮通过自动扩展任务。

#### Scenario: An informative negative result is complete

- **WHEN** 所需cell完整、独立验收valid，但科学终态为任一UNRESOLVED
- **THEN** 交付负结论和暂停建议，BM记concluded，禁止为追求通过继续调参
