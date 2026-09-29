## 1. 下游入口与边界

依次读 proposal、本文、spec、tasks。BM先读 `Startup Router`、`实验指令`，再读 `Wav2Lip natural content residual continuation 2026-09-09`、`Wav2Lip historical shift rescore 2026-09-08`、`Wav2Lip spectral structure replacement 2026-09-08`、`TTS 视觉教师到 replacement-safe 音频控制链路`。

研究目标仍是 `TFG(face,C)+N` 优于 `TFG(face,N)+N`。本轮只检查一种潜在教师的视觉依据，视觉相似度不是replacement收益。先停止当前固定content-residual构造；不重新调幅度、偏移、窗口或训练步数。

只复用已见fit缓存。设计期已读历史结果和部分视觉记录/组汇总，核对下列资产及文件数量，尚未计算本轮共同支持界限。因此必须标记 `analysis_type=retrospective_diagnostic`，不能称为独立预注册确认。

## 2. 固定输入与只读预检

路径均相对仓库根：

```text
P = runs/lrs3_tts_visual_control_20260825/01_visual_teacher_audit_retry4
L = runs/lrs3_tts_visual_control_20260825/00_protocol_lock_retry2
R = _reviews/lrs3_tts_visual_control/01_visual_teacher_audit/artifact_review_retry4.json
输出 = runs/lrs3_visual_teacher_missingness_<run-id>
```

| 文件 | 文件字节SHA-256 |
|---|---|
| P/manifest.json | 139ab35e89f837de6f32b4f4934c49d94577b0216a622b416cc6523bda6d23be |
| P/summary.json | 0d080bfb7bc49467d19296aa674ebc2e79a9541ff86c896dcb6acc43c5fef8ff |
| P/decision.json | 3e82b583937ee655b714f1e19ebca1327d8828c70dfcd229df5129583fd07ba9 |
| L/manifest.json | e1c970542eb90a2dac787463ebe4e8708ad3f2c4b4f1c4379f22e8382a29ec38 |
| L/test_lock.json | 295ccba71fb50d873a3ddf1b5cfd895418fd5bd95d242ec6441c57ef7e9660d4 |
| R | 8525d2959888bd72ed296ebbce107254a7e1da58c5dbc7b2cc37899e2a489349 |

P/records与P/features共有532文件（133 JSON + 399 NPZ）。整体摘要为 `da8faef0c9432d81dee9556f2c7435230e75b6a01aa0aa0192ab0100ddeba57c`。精确算法：从仓库根枚举这两个目录全部文件，以仓库相对POSIX路径按字节排序，每文件形成UTF-8行 `sha256 + 两个空格 + 路径 + LF`，将所有行拼接后SHA-256。与 `rg --files <两目录> | LC_ALL=C sort | xargs sha256sum | sha256sum` 一致。不能随意增加/忽略文件或用绝对路径算此摘要。

prepare首先核对摘要，再按 `(source_group,sample_id)` 建133条一对一ID连接；record名单和全部23组分母来自L/cohort，不从eligible子集反推。核对17 fit-train+6 fit-selection、sealed_unvisited、不访问其他split。只读取上述JSON、NPZ及实现/测试/spec；不解码或hash旧媒体、不加载landmarker、不启动任何模型。原媒体溯源依赖已绑定历史review；报告说明本轮未重新验证原始视频。

NPZ含 `canonical_mouth[frames,31,2]`、`valid[frames]`、`timestamps_s[frames]` 等，允许只加载需要字段，`allow_pickle=False`。参考 `scripts/experiments/lrs3_tts_visual_advantage/{video_features.py,visual_metrics.py}` 的格式和时间配对规则；实现小型纯NumPy读取/计算，不调用旧runner或其生成/提取入口。

所有字段shape/dtype/有限性、无效帧派生值为0、时间严格递增均需验收。缓存文件损坏、缺失、hash冲突是工程BLOCKED，**不能当成科学缺失值填[-1,1]继续**。科学缺失只允许由完整有效缓存上的冻结eligibility规则产生。

## 3. 三臂共同时间支持

G=real，N=natural，M=candidate（MFA3 natural-clock候选，不是最新CORRECT臂）。先复现旧pairwise exact-time结果及eligibility，作为缓存契约检查：

1. G/N与G/M各在原始时间戳上用双指针单调一对一配对。容差是两序列各自中位帧间隔的较小值×0.5；长度≤1时G步长取0.02秒、另一序列步长沿用G。若 `abs(t_X[j]-t_G[i])<=tol` 配对并同时前进；差小于−tol时仅X前进，否则仅G前进。不补帧、不平移、不DTW。
2. 各pair仅保留两端valid=true。每帧距离为 `sqrt(mean((Gmouth-Xmouth)**2))`（31点×xy、float64）；旧记录distance是帧距离的**中位数**，coverage分母是两序列原始帧数最大值。与 `metrics.exact_time.{natural,candidate}` 相比，距离/coverage容差1e-10、计数严格相等；没有有效pair则旧distance应为null，不以0代替。
3. 重算三臂valid fraction≥0.90、两pair coverage≥0.85、旧primary有限性；secondary有限性只检查已绑定JSON内distance/coverage/path_length/warp_burden/benefit（本轮不重跑DTW）。逐条核对旧eligibility字段；必须复现115/133合格、21/23覆盖及两个缺组 `6ZiN9ZJT294`、`79tRTivyMSM`。
4. 按相同G帧索引连接两pair，取三臂均有效的交集J。共同coverage=`len(J)/max(n_G,n_N,n_M)`；记录成为本轮observed，当且仅当旧eligible=true且共同coverage≥0.85且J非空。未通过者标明具体原因，仍保留在133条及其源组分母中。

对observed记录，N/M都在同一J上分别求帧距离中位数，记为d_N、d_M。定义本轮唯一主量：

```text
b = (d_N - d_M) / (d_N + d_M + 1e-12)     # [-1,1]，正值=TTS更接近真实口型
```

保存J的G/N/M索引、d_N/d_M、原单位差和b。两距离皆0时b=0。禁止按距离/正负选择帧、记录、组或把不合格记录已有的有限分数纳入主量。原pairwise分数和DTW只用于复验/背景；不与新共同支持分数混算。

这个量衡量合格共同支持上的canonical嘴部几何相似度，不是完整视频所有帧的误差、人的感知质量、音素语义信息或SyncNet收益。规范化依旧可能包含landmark测量误差与口部静态形状差异。

## 4. 原分母上的最好／最坏情况

对每组g，n_g为L中的全部记录数，O_g为observed记录，m_g=n_g−|O_g|。未观测记录的b只假定在[-1,1]；不假定随机缺失，不填总体/组均值：

```text
L_g = (sum(b_i for i in O_g) - m_g) / n_g
U_g = (sum(b_i for i in O_g) + m_g) / n_g
L = mean(L_g over ALL 23 groups)
U = mean(U_g over ALL 23 groups)
```

整组无观测时界限必须为[-1,1]。同时输出原133条数量、observed数量、缺失原因计数、所有23组n/observed/missing/L_g/U_g。按冻结17/6 split分别输出同定义界限，但不用于选择主结论。observed-only均值可附录展示，不能替代23组界限。

这是已见有限队列的**缺失值敏感性界限，不是95%置信区间**；不需要bootstrap/p值。它不推断其他记录、说话人或TFG。所有计算float64，不先舍入；决策比较L/U与0的容差边界固定为1e-12：

| 条件（按顺序） | diagnostic_decision | 建议 |
|---|---|---|
| 资产/实现/独立验收不一致 | not_available，engineering=BLOCKED | 修工程，不给视觉结论 |
| L > 1e-12 | VISUAL_SIGNAL_ROBUST_IN_CACHED_COHORT | 值得另设计独立队列视觉确认；只写建议 |
| U < -1e-12 | NO_POSITIVE_MEAN_UNDER_MISSINGNESS_BOUND | 当前教师缺少投入依据，停止此教师构造 |
| 其余（包括与0相等） | INCONCLUSIVE_UNDER_MISSINGNESS | 现有缓存不足，暂停本构造；不调阈值追阳性 |

三种科学诊断终态均为已完成审计。父Stage01仍BLOCKED、minimum=122；即使L正也不能宣布父科学GO。固定 `parent_gate_repaired=false`、`stage02_authorized=false`、`replacement_confirmed=false`、`waveform_head_authorized=false`、`generalization_established=false`。下游不得自动补采、重提landmark、启动旧Stage02/训练、开卡或关机。

## 5. 最小实现、独立验收和BM

新包只需runner、analysis、validate及必要I/O；无需搭通用框架。预期接口（尚待实现）：

```bash
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.lrs3_visual_teacher_missingness.runner --run-id 20260909_v1 --stage all
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.lrs3_visual_teacher_missingness.validate --run-root runs/lrs3_visual_teacher_missingness_20260909_v1
```

runner支持prepare/analyze/all及hash一致的工程--resume。analyze只能消费完整prepare；不隐式生成/提取。protocol在新指标计算前锁定输入树摘要、逐文件hash、spec/代码hash、记录分组、规则与零模型预算。新目录写入；parent不覆盖；代码/spec/输入变化要求新run。

最少产物：`protocol.json`、`input_audit.json`、`records.json`、`support_indices.npz`、`analysis.json`、`validation.json`、`review.json`、`final.json`、`result.md`。独立validator从父缓存独立重建时间pair、J、eligibility、b、组界限与终态，不能调用producer对应算法；可共享I/O。验收float误差≤1e-10、索引/ID/计数/标签精确相同，结束重验父摘要。final最后绑定validation/review，避免循环hash。

关键测试：不等长尾部仍进coverage分母；两pair有效帧不同必须取三臂交集；整组缺失保留[-1,1]；不等组大小等权；双零距离；共同coverage恰好0.85；缓存坏掉不是missing；ID乱序/漏条；界限跨0不标阳性；父BLOCKED和授权false不可被改写。使用合成小数组手算期望，不跑新模型。

阶段后保存诚实标注的self-review，独立数值validator不冒充子agent审查。执行下游须用同一BM实体 `LRS3 visual teacher missingness audit 2026-09-09`，planned→running→concluded（工程未完成保持blocked）；先Router/实验指令、搜索并全文读取，保留changelog、更新后读回。记录父历史限制、实际observed数、L/U、唯一诊断、零模型预算和run指针。最终说明“视觉证据能否支持下一轮投入”，不以本审计宣称发现replacement。
