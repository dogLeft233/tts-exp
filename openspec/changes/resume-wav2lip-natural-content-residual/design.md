## 1. 下游先读与继承范围

依次读本 proposal → 本文 → 本 change 的 spec → tasks；再完整读取 `../probe-wav2lip-natural-content-residual/design.md` 和其 `specs/wav2lip-natural-content-residual/spec.md`。原设计的资产、四臂构造、媒体、统计与结论边界全部继承；本修订仅覆盖控制搜索坐标、只读缓存续跑、跨会话检查与预算。

BM先读 Startup Router、实验指令，再读：`Wav2Lip delay search support audit 2026-09-09`、`Wav2Lip natural content residual probe 2026-09-08`、`Wav2Lip historical shift rescore 2026-09-08`。历史含义：控制恢复不等于 replacement；masked PAIRED 胜 NAT_ONLY 不等于胜完整 natural；WRONG 为同音素错实例，不能把内容条件称为句义语义。

固定父 run `P=runs/wav2lip_natural_content_residual_20260908_v2`，诊断 `D=runs/wav2lip_delay_search_support_20260909_v1`。新 run 建议 `runs/wav2lip_natural_content_residual_continuation_20260909_v1`。P/D及所有间接引用资产只读，禁止对父 run 执行旧 `--resume`，禁止修改父 final 中的失败/授权字段。

以下为文件字节 SHA-256，不是 JSON 内部 artifact_sha256：

| 文件 | SHA-256 |
|---|---|
| P/protocol.json | e0ad38110883bc7079e66866b5187f68636ee2ec8882ddba94879b203a5a2485 |
| P/drivers/manifest.json | 4ded4bc6c468cafa4990db92d326d823be63f63e15629b5b858d1986bc565ecf |
| P/control_scores/manifest.json | 143e7b7e72eb4bc14d10cfa88afa87decb59f87da48eed7581bf93780dfda5b2 |
| P/control_analysis.json | 068bf6ec3d43e432032ad6bfde8ce7cc284669b0c020575eb4ccf77bb30f9041 |
| P/control_validation.json | 9ae69db31e1f4b636a0b09e2793a86b373b58f183d8589f8a10f60f11d9138c6 |
| P/final.json | a9a423d96cddd774a3b4ac570f1d1cdceddd6e6eb700dd0b668c652faeb473aa |
| D/protocol.json | 4b8fbd105c38edc09a03b9adbfeb01b90812adc44c150a583dfa9d5da4f87968 |
| D/analysis.json | c6db175908c7be7964d0b9d2cf71d870f48a1466b57d3b729d286492ba675a41 |
| D/validation.json | 7a3e46012d15fa22cb25ccfb21d0e5b3ab3ffc7d5d9400432c61816dce761c2e |
| D/final.json | 8e7fb08dcbf10c688d87a14f5e131ca1d8e96fb8259f399a9e6810c30a178f26 |
| 原 change/design.md | 1b06fc44d5ecc96a28eeb907d3692517e8bacd3cafb2ed3d9f08fb96465a3286 |
| 原 change/specs/wav2lip-natural-content-residual/spec.md | d512c5ed4ef4153d536b16d1ca74ff367df02f5bead97f2014d713a763039cce |

缺失或冲突为 BLOCKED。设计时只核对了上述资产和历史证据，未运行候选。下游必须重验引用的PCM、face、driver、checkpoint、embeddings、矩阵与媒体；不能只信 final=PASS。新 protocol 在新 forward 前锁定 `protocol_revision=matched_delay_continuation_v1`、上述父链、本修订及实际调用代码hash、样本、参数和预算。

## 2. 修订门禁：offset补偿与固定anchor必须分开

固定 U=range(30,58)，真实embedding为[88,1024]。沿用诊断的 float32 距离 `sqrt(sum((V[r]-A[q]+1e-6)**2))`，先按U求float64均值曲线，再求首个argmin。j=0..30：

```text
N:       q=r+j-15, physical_lag=j-15, offset_N=15-argmin(z_N)
A_DELAY: q=r+j-10, physical_lag=j-10, offset_DELAY=10-argmin(z_DELAY_matched)
```

N的q=15..72，delay的q=20..77，全部须有真实MFCC/PCM支持。禁止padding、roll、复制自然曲线或择峰居中。自然和所有候选仍用原lag=-15..15；只有已知延迟控制用lag=-10..20。

独立重算旧13/16、配对16/16与三条异常恢复，核对诊断的 `SEARCH_SUPPORT_RECOVERED` 和资格true。诊断复现一致是缓存资格检查；原灵敏度阈值仍为至少14/16的offset差位于[-6,-4]，没有把门槛降低或改成新的16/16科学门槛。

固定anchor损伤仍为 `z_DELAY_legacy[k_N]-z_N[k_N]`，其中k_N来自N原31列。不能用已补偿的 `z_DELAY_matched[k_N]`。原8组bootstrap下95%CI下界>0且至少7/8组正；历史mean=1.162567632539、CI=[0.844646827451,1.482466255980]、8/8正，须由数组重算。矩阵容差1e-4、U曲线/统计1e-6，offset严格按整数比较。

父parity与N_REPEAT证据也重验。D/final的 `stage_b_authorized=false` 是诊断范围，保持原样；只有本 run 的控制重算与独立验收均通过，才写本 run 的 `stage_b_authorized=true`。不得手填一个授权字典绕过检查。

## 3. 最小跨会话检查与候选执行

prepare只读导入P的16条/8组及已构造N/CORRECT/WRONG/SHUFFLE；ID连接、按(source_group,sample_id)排序，独立按原公式从三个seed缓存重构四臂并逐值验收。继续固定K、排列盐、a和clip后范数比[0.95,1.05]；不因新控制结果重构或挑选候选。

controls先完成CPU复验，再做两项当前运行链路检查：

1. 按父固定规则重跑历史rescore v7 `parity.json` 中两个 `v4.cells`，对参考矩阵、原support端点容差1e-4，offset相同。它们不进入科学统计。
2. 对排序前两条，使用P的N driver与静态face各独立forward一次，命名N_REPLAY；参数完全沿用原设计。解码视频像素须与P对应N逐值一致，PCM/PTS一致；各评分一次，矩阵误差<=1e-4，U上的C/D/自然anchor误差<=1e-4、offset相同。这两个重复只检查运行链路，不替换主分析的父N，也不当新样本。

使用 `/home/wjj/.venvs/wav2lip/bin/python` 在宿主命名空间执行真实CUDA kernel预检；16G卡串行batch=4、空闲至少5GiB。普通沙箱缺/dev透传时使用获准的宿主执行机制，不据此重装驱动。SyncNet用 `/home/wjj/.venvs/syncnet/bin/python` CPU batch=20、threads=4。记录实际版本/设备/确定性设置。复跑不一致为BLOCKED，不放宽容差或换缓存基线。

controls独立validator PASS之后，candidates才可生成16×3=48个视频。所有入口（all、candidates、resume）都检查当前协议/输入hash与控制验收，而非只读取一个布尔值。analyze只读已有完整候选，缺cell时拒绝分析，不能隐式启动生成。

继承93帧/25fps/224×224静态头像、FFV1、整条原N PCM16 mux、零起点、不删尾；特征仅前61440 samples。评分完整画面，保存fresh[88,31]矩阵与embeddings，U及k_N不变。禁止对候选应用delay配对域或将其自由最优offset改作anchor。

预算：复用P的18视频/36评分；新增N_REPLAY 2视频/2评分、parity 2评分、候选48视频/48评分，新增总上限50视频/52评分，含P累计68视频/88评分。D的32个缓存cell复算不是fresh评分。相较原总预算66/84，新增2视频/4评分只用于跨会话检查。记录fresh/reused/失败尝试及实际forward次数，工程resume复用已通过且hash一致的cell，不追加科学重复。训练/TTS/vocoder均0。

## 4. 固定分析与停止决策

对X/Y定义正向收益：ΔC=C_X-C_Y，ΔD=D_Y-D_X，ΔA=z_Y[k_N]-z_X[k_N]。仅分析CORRECT/N、CORRECT/WRONG、CORRECT/SHUFFLE；每组两条先均值、8组等权；PCG64(20260909)的10000×8有放回索引和linear quantile 95%CI，所有contrast共享索引。保存未舍入数值，报告Sync-C三位小数。

pass(X,Y)：C/D/A三项CI下界全>0且至少7/8组三项同正。主假设额外要求CORRECT/N平均ΔC>0.05。按以下顺序确定唯一科学终态：

| 条件 | 结果与下一步 |
|---|---|
| 工程、缓存复现或独立验收不一致 | BLOCKED；scientific_decision=not_available，保留具体失败阶段 |
| 原构造退化 | INPUT_DEGENERATE；不删样继续 |
| 有效实现下灵敏度门禁未通过 | CONTROL_FAILED；候选0，停止本轮，不扫域/窗口 |
| CORRECT/N未通过 | NO_INCREMENT_ESTABLISHED；停止此固定辅助增量读出，不再调幅度求正分 |
| 主假设通过、范数有效、两个额外contrast均pass | CONTENT_RESIDUAL_SIGNAL；仅建议冻结同一构造做独立source-group确认 |
| 主假设通过但上项不成立 | NATURAL_GAIN_MECHANISM_UNRESOLVED；记录具体未通过项，下一阶段需确认增益和解释对照 |

即使CORRECT胜过WRONG，也不能越过natural主假设。阴性不是所有语义路线无效，阳性不是句义因果机制。所有终态保持 `replacement_confirmed=false`、`waveform_head_authorized=false`、`generalization_established=false`、`historical_shift_gate_repaired=false`；报告这是seen-record短支持探索。本轮到终态停止，不自动进入新的实验或关机。

## 5. 实现接口与验收

建议小包 `scripts/experiments/wav2lip_natural_content_residual_continuation/`，只增加runner、独立validate及必要辅助函数；复用父纯driver/media/scoring/候选分析及现有worker。不要调用父runner的all流程或父诊断CLI写入旧目录；需要抽取接口时显式传paths/control证据，不用全局monkeypatch。新validator可复用只读I/O，核心重建/门禁/bootstrap/终态必须独立计算，不能调用producer对应函数。

待下游实现的CLI：

```bash
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_natural_content_residual_continuation.runner --run-id 20260909_v1 --stage all
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_natural_content_residual_continuation.validate --run-root runs/wav2lip_natural_content_residual_continuation_20260909_v1 --stage all
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m pytest -q tests/experiments/wav2lip_natural_content_residual_continuation/
openspec validate resume-wav2lip-natural-content-residual --strict --no-interactive
```

runner支持prepare/controls/candidates/analyze/all及仅工程中断的--resume；validator支持controls/all。resume核对全部绑定；代码/协议变更需新run。验收失败进程非零退出；科学阴性是已完成实验，须写明确科学标签。不得把尚未运行的CLI当作已有接口。

最少产物：protocol、input_audit、reuse_manifest、control_analysis、control_validation、candidate_scores/manifest、analysis、validation、review、final（JSON）与result.md；另保留driver/媒体/生成manifest、复跑/parity证据与bootstrap索引。manifest记录唯一cell键、source path/hash、新旧run归属和fresh/reused。最终验收重算全部16条×3候选的媒体身份、矩阵、C/D/A统计、组正数、范数和唯一决策；final最后绑定validation与review，避免循环hash。两父run锁定文件结束时再次核对。

聚焦回归：旧k=27/28峰恢复且offset符号正确；anchor误用补偿曲线被拒；候选改lag/U被拒；缺独立验收/篡改授权时all和直接candidates均不生成；N_REPLAY/ID乱序/缺cell/hash变化被拒；仅胜错配、仅自由分数正、缺一对照时不能升格；resume不重复已完成cell。检查原父包相关测试仍通过。

BM使用本续跑实体 `Wav2Lip natural content residual continuation 2026-09-09`，planned→running→concluded；保留两个父实体的历史终态并建立follows关系。先读Router/实验指令、搜索、全文读已有笔记，更新后读回并保留changelog。每个阶段写状态、数字、run指针和限制；阶段自审保存review.json并标明self-review，独立数值validator不冒充子agent审查。最终回报工程/科学状态、三组C/D/A、实际预算及下一步边界。
