# 下一轮并行实验交接（2026-09-09）

## 先读结论

目标仍是 `TFG(face,C)+N` 优于 `TFG(face,N)+N`，最终希望得到能跨TFG使用的波形生成头。当前没有已确认的泛用replacement收益。值得继续有限机制探索；不宜继续放大同一TTS谱迁移/偏移/固定内容残差构造或直接训练头。

本次只交付设计，未执行下面的新实验。历史数字与问题见 [历史证据复核](replacement-history-review-20260909.md)。最近A/B/C的旧PASS不满足独立数值验收要求；必须区分“报告了阴性”与“已按spec独立复验”。

## 五路任务

| 路线 | change | 问题 | 新TFG视频 / 新SyncNet cells上限 |
|---|---|---|---|
| E0 | [audit-wav2lip-parallel-evidence-contract](changes/audit-wav2lip-parallel-evidence-contract/design.md) | A/B/C数字与实现是否符合原spec；B延迟域遗漏有什么影响？ | 0 / 0 |
| E1 | [audit-lrs3-visual-teacher-missingness](changes/audit-lrs3-visual-teacher-missingness/design.md) | 真实嘴部几何是否支持现有TTS视觉教师？ | 0 / 0 |
| E2 | [probe-wav2lip-waveform-gain](changes/probe-wav2lip-waveform-gain/design.md) | 仅改变驱动波形标量增益，固定原N评分能否改善？ | 34 / 36 |
| E3 | [probe-wav2lip-phone-core-shrinkage](changes/probe-wav2lip-phone-core-shrinkage/design.md) | 保留音素边界、压缩音素内部变化，是否胜过同支持普通平滑？ | 34 / 36 |
| E4 | [probe-wav2lip-reconstruction-base-interaction](changes/probe-wav2lip-reconstruction-base-interaction/design.md) | 内容增量失效是否与读取它的自然/重建基底有关？ | 50 / 52 |

E1是此前尚未实现的现有spec，直接执行它，不新建同义实验。E0是查错，E1是另一种视觉证据，E2/E3/E4是三个不同的固定候选问题，并非五种已有效的头。
合计最多118个新TFG视频、124个fresh评分；训练、TTS、vocoder、新模型下载均0。

五路可并行实现、测试、CPU prepare。E2/E3/E4独立从P/Q/D重建所需控制，不把A/B/C的PASS当输入门禁，不等待其他新科学结果；若E0发现共享P/Q/D证据损坏，调度者暂停依赖它的分支。不得因某路阳性调其他路参数。

## 共同资产与坐标（E2/E3/E4必读）

读取上一轮公共契约 `openspec/parallel-replacement-probes-20260909.md` 中固定P/Q/D哈希表与媒体规则。本页明确覆盖其“任务、候选、预算、统计与验收”；不继承其A/B/C runner或validator实现。P/Q/D分别是：

- P = `runs/wav2lip_natural_content_residual_20260908_v2`
- Q = `runs/wav2lip_natural_content_residual_continuation_20260909_v1`
- D = `runs/lrs3_masked_tts_new_confirmation_20260902`

前页13项文件字节SHA须逐项核对；间接输入从manifest/worker沿既存hash绑定，不用当前文件自hash冒充历史来源。额外绑定D/mask_manifest与reconstruction见E3/E4。按(source_group,sample_id)排序，以ID连接全部16条/8组每组2条；不得删样，seed不作独立样本。本批已被观察多次，所有新结果都是探索，非独立确认。

完整N PCM为16k mono PCM16，mel M只编码其前61440 samples，M形状[80,308]。官方chunk起点int(t*80/25)，16列，尾部按官方最后16列规则，共93帧；25fps、静态P/F0、224×224、FFV1。mux完整原N，保留长音轨尾部，起点0，不-shortest、不加帧。SyncNet固定整224画面入口（包括父worker既存JPEG提帧约定），不更换检测/裁脸/frontend；矩阵[88,31]，U=range(30,58)。这不是整条长音频实验。

float32距离 `sqrt(sum((V[r]-A[q]+1e-6)**2))`，自然/候选q=r+j-15；U中全部q必须真实存在，无padding。先以float64平均U为z，再C=median(z)-min(z)，D=min(z)，k0=argmin(z_N)，并列首列。所有比较固定P/N的k0。正向收益C_X-C_Y、D_Y-D_X、z_Y[k0]-z_X[k0]。候选不能自行改lag或anchor。

## 每个GPU分支自己的控制

1. 独立复算P自然/延迟32缓存cell：旧域13/16、配对域16/16。已知delay +3200 samples的自然lag=-15..15、delay lag=-10..20，offset分别15-j、10-j。不得把延迟曲线仍交给offset=15-j的旧helper。
2. 未补偿损伤取legacy delay曲线在natural k0处减N。复现mean=1.162567632539、95%CI=[0.844646827451,1.482466255980]、8/8组正，父PCG64(20260909)、10000次组bootstrap。14/16 offset、95%下界>0、7/8组正为冻结敏感性门槛；控制规则不换成新科学99%区间。
3. 两个fresh scorer parity使用历史rescore v7 `parity.json/v4.cells`，SHA见旧公共契约/原P design；前2条P/F0/N各fresh生成评分一次。与P/N解码像素逐值一致，矩阵误差≤1e-4、端点≤1e-6、offset一致。不能仅检查U行列表相同就写replay_pass。
4. 独立validator从原embeddings/媒体重算以上控制；`all/candidates/resume`都只接受绑定当前输入、代码、spec的control_validation。控制科学失败停止候选；实现或绑定错误为BLOCKED。

使用宿主 `/home/wjj/.venvs/wav2lip/bin/python`，真实CUDA kernel预检；Wav2Lip GAN父权重hash不变，batch4、eval/no_grad、seed20260909、确定性开、TF32关。scorer解释器 `/home/wjj/.venvs/syncnet/bin/python`，CPU batch20、threads2。可复用旧generation_worker及SyncNetScorer纯模型接口，不能复制A/B/C的控制/验收逻辑。

## 科学筛选与状态

每记录先计算收益，再组内两条均值，8组等权。PCG64(20260911)、20000×8共享组重采样，保存实际索引；NumPy linear quantile，双侧99%CI。E2两主contrast、E3一个主contrast、E4一个主contrast；控制和机制对照不变成额外候选搜索。99%是探索筛选约定，不声称在8组和长期反复探索中保证严格错误率。

统一 `gain(X,Y)`：三项C/D/A的99%CI下界全>0，至少7/8组三项同时正，mean ΔC>0.05。机制比较沿各design单独定义；胜错配/退化基底不能代替胜完整N。
工程不一致→BLOCKED/science=not_available；有效输入无辨识力→INPUT_DEGENERATE；有效控制不敏感→CONTROL_FAILED；阴性是完成，不用零效应或等效性措辞。任何阳性最多SIGNAL_TO_CONFIRM，`replacement_confirmed=false`、`waveform_head_authorized=false`、`generalization_established=false`、`historical_shift_gate_repaired=false`。
若有筛选信号，下轮才设计新source-group、冻结构造、独立视觉/同步验证；E3/E4随后还需波形可达性。最终泛用性需至少另一TFG验证，本轮Wav2Lip结果不能替代。

## 执行接口、隔离与资源

E2/E3/E4包名对应去掉probe后的下划线名，各自目录 `scripts/experiments/<package>/`、`tests/experiments/<package>/`、`runs/<package>_<run-id>/`，只改自己change/tasks和BM实体。
待实现：
`PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.<package>.runner --run-id 20260910_v1 --stage all`。
支持prepare/controls/candidates/analyze/all及hash相同的工程--resume。analyze只消费已有完整产物，绝不隐式生成或评分；缺产物非零退出。validator同包 `validate --run-root ... --stage all`。

GPU生成必须持公共OS flock `/tmp/tts-exp-wav2lip-gpu0.lock`，锁内空闲≥5GiB，同一时间仅一个GPU模型。忙则QUEUED，调度重试，不杀其他进程。CPU总并发最多3个worker，每个最多2线程，E0/E1优先；E2→E3→E4生成排队，每批结束释放锁。OOM只可原配置有限工程恢复，不改科学参数；每个cell最多1次失败后重试，仍失败则BLOCKED，所有失败尝试记账。

最少文件：protocol/input_audit、drivers/support、media/score manifests、control_analysis/control_validation、analysis、validation、review、final及result.md。protocol冻结输入/代码/spec/环境/预算后才能新评分；代码变更新run，旧产物只读。新final只在独立验收结束后形成单向绑定（final→validation/review/analysis hash），不先写complete再覆盖validation。
独立validator只共享I/O和模型输出格式，不导入producer构造、支持映射、统计或判定。独立重算driver、矩阵、曲线、抽样、所有contrast及终态；矩阵≤1e-4、统计≤1e-6、ID/索引/计数/布尔精确。没有该证据不能称独立验收。

最低反例测试：故意改analysis数字并重签JSON仍失败；漏/重复ID、错offset符号/延迟域、改父hash、不同replay像素、自由峰好而anchor坏、7个边际正组但非7个联合正组、analyze缺产物、resume代码变更均拒。只测试计算和实际门禁，不堆通用框架。阶段self-review要诚实标注，与独立数值validator分开。

BM每路唯一实体见各design；先Startup Router/实验指令、搜索、全文读取再更新，状态planned→running→concluded或blocked。使用memory-notes格式，正文与frontmatter只保留一个当前status，保留Changelog表逐行历史；结果写Observations，不追加到表中。任务完成勾tasks，strict OpenSpec和数值验收分别报告。
