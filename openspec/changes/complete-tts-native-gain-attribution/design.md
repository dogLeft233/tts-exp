# 实现与实验设计

## 1. 契约优先级与不可变证据

科学参数、音频算法、样本和假设继承 `../disentangle-tts-native-gain/protocol.md` P0–P7。本文件对成功路径、缓存、资源、人工交付和验证作更严格约束；不改变原六项主对比。若无法满足科学协议，保持未完成，不能临时换样本或处理算法。

读取 `evidence-bindings.json` 并校验全部 SHA-256，再展开父 manifests 的传递文件绑定：PCM、原视频、肖像、crop、像素/PTS、特征、矩阵和 bootstrap 索引。绑定文件清单不是传递审计的替代。

绑定清单中的源码/测试是本次审查的修复前版本。下游先验证并将其字节归档为 `pre_repair_snapshot/`，再修改工作区；续跑时针对这份归档验证旧hash，并分别验证新代码快照，不能要求修复后的工作区仍匹配修复前hash。父run和原科学协议的hash始终保持不变。

父 run 只读；新 run 保存 `parent_evidence.json`、`reuse_manifest.json` 和新代码/文档快照。只读引用或复制均可，禁止通过可写硬链接改动父产物。导入 A 时记录原生产代码 hash 与新验证代码 hash，不强行要求父代码等于当前代码。改变路径需要 location map，文件内容身份仍靠 hash。旧文档/代码变更后不能通过刷新父快照把旧产物伪装成新生产结果。

A 的 180+18 逻辑 cell、72 个音频操纵检查、24 个重放和 396 个错内容诊断均须验收；兼容时零额外 GPU 前向。若发现父 A 证据缺陷，保留原结果并单独重算受影响部分，解释与旧值差异。写 spec 时记录的结论不能代替机器重算。

## 2. 已确认实现缺口与最小修复

| 位置 | 当前缺口 | 必需修复/回归 |
|---|---|---|
| `generation.py:crossed_score_stage` | 环境变量齐全后仍抛出占位异常 | 实现 ROI、特征、矩阵、支持、控制和清单；测试真实成功分支 |
| `_run_one_generation` | repeat 使用同一 `seed42.mp4`，可覆盖基线及 frontend/log | 将 role/repeat_index 纳入完整 cell ID 和每个产物路径；重复独立推理，禁止复用基线 |
| `discover_leaptalk` | 有 repo 路径及单个 checkpoint 就可声明可用，其他组件可 UNKNOWN | 明确官方来源、commit/patch、所有实际加载组件与文件；检查目录/shards/config，不只单文件 |
| frontend sidecar | 存在且自哈希不等于实际消费证明 | 在读取音频、frontend 输出、模型加载与 RNG 重置现场生成证明并核对请求 |
| `validate.py` | B COMPLETE 分支主要检查计数；独立主统计只重算 A | 从 B 文件、矩阵、支持及组表独立重算 B、G/E/I、控制和状态 |
| `perception.py` | 直接硬链接生成 MP4 未保证同音轨；公开 hidden_repeat/source_pair_id；总数与 cache 判断不符 | 生成统一 A0 播放媒体/播放器、随机全题顺序、隐藏重复身份、内容级缓存 |
| 人工分析/验证 | 只有 `source_group_bootstrap_required` 标志；validator 假定永远空评分 | 实现来源 bootstrap、隐藏题一致性与真实评分分支；保留无人评分合法状态 |
| `report.py:_hypothesis_status` | H2 依据 A 对比和 B COMPLETE 判断，未用 B 的 G | H2 必须逐项依据 B_N_DENOISE_G/B_T_NOISE_G_HARM 与控制；A 阳性不能触发生成端结论 |
| `common.py:resource_gate` | GPU 显存检查复用磁盘临时空间估计；锁路径为实验专属 | 分离 gpu_peak_bytes/disk_temp_bytes/disk_persistent_bytes，接入实际共用的设备锁 |
| `runner.py:main`、generation 异常 | all 无顶层 status 时可能打印 COMPLETE 并返回0；资源错误可被吞为生成失败 | 明确 aggregate/退出码，保留 RESOURCE_WAIT/DEPENDENCY_BLOCKED/IMPLEMENTATION_INCOMPLETE/INPUT_INVALID |

以上是针对未执行路径的审查，不表示 A 数值已经被这些缺口污染。下游仍须检查相邻调用链，不限于逐条打补丁。

## 3. 环境恢复与资源

官方入口于 2026-09-16 核对：[LeapTalk README](https://github.com/zhangrongxiang/LeapTalk)。它列出 SoulX-FlashHead 基座、wav2vec2、LeapTalk LoRA/audio projection/TAE，并提供 Lite、单步、compile-off 配置。版本和实际参数以执行时锁定 commit 的代码为准；这里不承诺 V100 兼容性或最低显存。

先审计已经授权的本地目录、环境和可用远端，不反复访问部署笔记中已关闭端口。不复原历史 checkpoint 也可使用已冻结的官方新配置，标 `NEW_LEAPTALK_CONFIGURATION`；全 148 视频使用同一个配置。不能只运行 SoulX 基座冒充 LeapTalk。

`model_provenance.json` 至少保存：官方 URL、repo commit、dirty diff/hash、各模型 revision、实际加载文件相对路径/大小/SHA-256、base/LoRA/audio projection/audio encoder/decoder及所需配置、模式、分辨率、fps、步数、guidance、dtype、attention backend、compile、Python/Torch/CUDA/驱动和硬件。非使用组件显式 `NOT_USED` 并给加载依据，必需字段不允许 UNKNOWN。历史等价性和训练分布可为 UNKNOWN，不妨碍新配置实验，但不能宣称历史重放或已识别训练机制。

部署前按真实挂载点估算权重+环境+缓存+148视频+特征/矩阵+盲评包+最大单cell临时空间+至少1GiB余量；已存在的可验证文件不重复下载。不得用单cell 256MiB 假装总持久预算。空间不足时转向已授权大盘/机器；不得删除历史产物。下载和安装用任务独立缓存，记录可清理临时文件。

V100 的精度/attention/kernel 能力必须实际核对，不照抄其他卡上的 bf16 或 FlashAttention 设置。候选配置的兼容性 smoke 可以调整，最终选择只依据资源/兼容性；正式配置冻结前不看评分优劣。模型、精度、分辨率改变必须另建正式 run，不能混入已有结果。租机仅在已有费用授权范围内进行，否则先做好可执行部署包、准确预算和待执行命令再请求批准。

每 GPU 任务使用同设备、跨实验共享且参与方确实遵守的锁；同时检查外部 compute PID、3次间隔5秒利用率≤5%、空闲显存≥独立估计峰值+1GiB。采样记录无法替代锁，也不保证阻止非协作进程。生成/SyncNet 串行；遇外部负载停止启动下一cell并落盘。不能终止他人进程。

## 4. 推理 adapter 与续跑

建议新增 `leaptalk_adapter.py`，以请求/响应 JSON 或显式 argv 接口接入实际官方入口，不把任意 shell 字符串当可信模型身份。请求绑定 `cell_id,role,id,source,driver,seed,portrait,pcm,config_hash,output_dir`；路径带空格可用，命令不得 shell=True。

每cell重置 Python/NumPy/Torch CPU/CUDA RNG、模型流式缓存和相关状态；同 source/seed 的三个 driver 使用同初态。保存初态或状态摘要及重置证据。frontend sidecar 必须来自实际输入读取和模型特征产生处，记录已消费 PCM hash/样本范围、重采样/归一化、frontend tensor shape/dtype/hash及可验证特征产物、chunk 样本到帧映射、padding、初态 seed、肖像 hash、实际加载模型身份。验收不接受手填“must reset”字符串或仅复制请求参数。

输出检查真实可解码帧数、25fps、PTS、非零持续时间、所有有效帧，不能以存在 MP4 或进程退出0为完成。保留官方内部必要 padding 及映射，但 padding、重复补帧不进入科学支持；不对输出挑分数，不人为 retime。科学评分读取独立 PCM，不读模型封装的有损音轨。

主视频和重复视频分目录，例如 `.../151/N/A0/seed42/main/` 与 `.../151/N/A0/seed42/repeat1/`；二者相同像素 hash 在确定性模型下合法，但必须有不同执行记录且不可互相覆盖/缓存替代。

先按确定顺序建立完整预期台账，再逐cell原子提交。中断不保留 COMPLETE；完成项需重新验证文件、参数和依赖 hash 才可复用。缺失/失败项仍保留，不能仅收集成功结果。依赖缺失不能触发148次相同失败，资源等待不转成科学失败。每条错误含原因、影响cell、可重试动作；报告 planned/completed/blocked/failed 数，分母不冒充完成数。

## 5. 完整实验矩阵和 ROI

固定 ID151–162，12 个 source_group，N/T 两种音频来源，seed42/43；A0/NOISE/DENOISE 完全复用父音频。每 ID 肖像采用父审计绑定的真实视频第一帧，所有臂相同。N/T 保留自己的时钟，不直接互换未对齐音轨。

- 科学视频：12×2×3×2=144。
- 重复视频：ID151/152×N/T×seed42×A0=4。
- 每 ID/source/seed 七格：`(V0,A0),(V0,An),(Vn,A0),(Vn,An),(V0,Ad),(Vd,A0),(Vd,Ad)`，共336。
- 控制：4个重复视频评A0；对应4个主V0评真实 PCM +200ms 的A0，共8。

每 ID/source/seed 只在 V0 用原协议官方人脸跟踪规则选 ROI，将同一逐帧轨迹/裁剪/缩放应用于 Vn、Vd和该seed的重复视频。保存未经候选修正的轨迹、track规则和hash。候选若出框或无法应用，标 ROI_INVALID；不能重新挑脸。可复用仓库 SyncNet 管线，但需写成可执行的 crop adapter，不能要求下游再另找不存在的 LEAPTALK_CROP_COMMAND。

同 ID/source 两seed的全部14个科学评分共同冻结 U：按真实帧/音频时间索引的有效行交集，再剔除官方lag搜索两端各15行，至少25行。禁止只按最小数组长度忽略起点/PTS，或各cell独立挑窗口。两seed可分别冻结 ROI，但同seed三视频必须同空间变换。延迟控制另建剔除±5帧污染的共同缩减支持，重新在该支持计算基线；重复也和对应基线用共同合法支持，不改变科学支持。

保存原始音视频特征、完整 `[T,31]` lag矩阵、ROI/PTS/hash和支持。视觉特征按视频复用，音频按PCM/frontend复用，矩阵在CPU计算；评分数不等于模型前向数。算 `curve=mean_time(matrix[U,:])`，`D=min(curve)`、`B=median(curve)`、`C=B−D`，offset采用既有官方符号，不逐帧先取极值。

控制门：repeat 的 |ΔC|、|ΔD|≤0.050，offset差≤1帧；+200ms应使offset相对基线−5帧（容差1帧）且基线最优列距离升高。越界无信息/控制失败均保留，不无限重复挑通过实例；相应结论 CONTROL_LIMITED/GENERATION_UNSTABLE，不能自动完成。

## 6. 计算与可作结论

对每 ID/source/seed/干预h，令 `q00=C(V0,A0), q01=C(V0,Ah), q10=C(Vh,A0), q11=C(Vh,Ah)`。

`G=q10−q00` 是固定评价音频时的生成路径响应；`E=q01−q00` 是固定视频时的评价路径响应；`I=q11−q10−q01+q00`；总响应 `G+E+I=q11−q00`。NOISE/DENOISE各一套，共96个四格分解。C的生成路径响应仍是指标证据，不自动等于人类口型改善。C、−D、背景B、D0、offset及unit诊断分开。

主对比严格继承六项：A_N_DENOISE_E、A_T_NOISE_E_HARM、A_R_DENOISE_E；B_N_DENOISE_G=`q10_d−q00`；B_T_NOISE_G_HARM=`q00−q10_n`；B_NATIVE_FRESH=`C(V0_T,T0)−C(V0_N,N0)`。fresh native 的N/T各用自身共同支持，不能伪造跨时钟对齐；另报原协议描述性支持敏感性。

先seed内差、再同记录两seed平均、再同source_group记录平均、最后12组等权。保存完整逐cell/seed/来源表和显式组序。复用父PCG64 seed20260915、20000×12 bootstrap索引；六项共享同组序和索引。95%描述区间，99.166667% Bonferroni区间，线性分位数；缺B不缩小检验族。任何必要cell/seed缺失，相关主检验 INCOMPLETE，完整对子集只能单列探索性分析。

每个B生成主对比单独报告阳性/反向/小幅/不确定；阳性需校正下界>0且对应操纵、控制有效，±0.200等效需整个校正区间落入该范围。若生成主对比至少一项阳性，只能说“该操作在相应来源下影响生成路径”，不可把另一项自动判阳性。H2汇总列出二者状态，不能依赖A的阳性判H2。

只有 fresh native 重现阳性，才可讨论本配置原生优势与干预响应的关系；否则标 NATIVE_NOT_REESTABLISHED，其他有效干预结果仍可报告。不把干预响应除以历史+1.253算解释比例，不把加噪损伤直接等同于自然/TTS差异的原因。H3仍为描述，H4仍未识别训练因果性；本任务也不识别两种TTS provider的质量因果效应。

## 7. 人工包与分析

补齐96同步正式对+10隐藏重复、48音质正式对+5隐藏重复。同步双方必须实际播放相同的A0 PCM，可用无损音频容器或可验证的静音视频+同步播放器；原MP4内音轨必须被移除或禁用，禁止仅在说明里要求“听同一声音”。验证左右解码像素与绑定视频相同，音频时间起点一致，不能用-shortest裁掉有效尾部。

正式题与重复题一起按已定seed随机顺序；私有映射保存hidden_repeat/source_pair_id及source/driver/seed，公开包不暴露这些字段、私有文件、条件路径或元数据标签。匿名编号不可按连续ID/条件泄漏设计，发布清单不得包含private_mapping路径。质量比较保持组内共同播放缩放、不逐clip响度归一化。

包完整指媒体可播放、hash正确、映射可追溯；B阻塞时只有台账，标 PARTIAL_MEDIA，不可包装为完整同步包。生成后的缓存绑定generation/audio manifest和媒体hash，不沿用缺文件空包；重复构建不得覆盖已有真实评分。

无人参与时保持 PERCEPTION_NOT_ASSESSED/QUALITY_NOT_ASSESSED，不阻止其他条件齐全后的自动实验完成。若返回评分，至少3位共同评审完成每种包全部正式题；严格校验重复(rater,pair)、未知ID、枚举、有限数值/范围、缺失和不可判断。平局0.5，左右方向映射后按配对内评审平均、source内seed平均、来源等权bootstrap给95%描述区间；输出获胜比例与相对0.5差分且名称准确。不可判断造成的缺口须报有效分母并限制推断，不能把“填了行”当完整有效评分。隐藏题只报一致性，不按偏好剔除评审，不重复计入正式统计。音质与同步/伪影分开报告。

## 8. 实施顺序与目标 CLI

沿用 `python -m scripts.experiments.tts_native_gain_attribution.runner`。下游新增以下接口并提供 `--help`；这些是待实现接口：

```bash
python -m scripts.experiments.tts_native_gain_attribution.runner --run-id completion_YYYYMMDD_v1 --stage import-parent --parent-run runs/tts_native_gain_attribution_implementation_20260915_v1 --completion-spec openspec/changes/complete-tts-native-gain-attribution
python -m scripts.experiments.tts_native_gain_attribution.runner --run-id completion_YYYYMMDD_v1 --stage preflight --model-config /absolute/path/leaptalk.json
```

先在独立 smoke run 用ID151完整12科学视频+N/T两个重复验证全链及资源（小样本不作科学结论、不记作正式完成）。smoke发现代码缺口时修复并重新冻结；正式run不借用开发中挑选的高分输出。新命令可加入 `--smoke` 明确单独目录和预期分母。

冻结后依次执行已有stage名称 `generate → crossed-score → analyze → perception-pack → perception-analyze → report`，续跑通过hash验证跳过已完成项。最终执行：

```bash
python -m scripts.experiments.tts_native_gain_attribution.validate --root runs/tts_native_gain_attribution_completion_YYYYMMDD_v1
pytest -q tests/experiments/tts_native_gain_attribution
openspec validate complete-tts-native-gain-attribution --strict
```

CPU fixture必须先覆盖B成功路径；正式执行不能由fixture替代。阶段成功退出0；RESOURCE_WAIT退出75；其他未完成/缺依赖/验证失败非0（例如2），--stage all按最终聚合状态返回且打印真实状态。错误不能被空dict默认COMPLETE掩盖。

## 9. 独立验收与交付

验证器可共享I/O和hash，不导入runner/analysis的科学计算或状态判定。它必须从实际矩阵独立重算全部六对比及96组G/E/I、原始曲线、控制、bootstrap和结论，核验模型/frontend/PCM/肖像/像素/PTS/ROI/U/seed的传递身份。只检查JSON自哈希、计数或汇总字段不够。保持父A的生产快照与新B代码快照可独立追溯。

负例必须覆盖：q01/q10交换且重写自哈希；B矩阵改值；支持换行；同路径伪repeat；缺seed/cell；重复来源冒充12组；不加载LoRA；frontend消费另一音频；控制失败；公开隐藏题；左右播放不同声音；伪造COMPLETE。验证器均拒绝。另用真实B成功fixture证明非永远失败；mock只用于工程测试。

交付 `implementation_audit.md`（最终问题/修复/验证映射，非进度日志）、完整新run、中文report、final/validation、资源/模型provenance和实际命令。报告解释流程、计算、逐组和整体数值、结论边界，不依赖读者了解项目。Sync-C显示3位，机器精度保留。

AUTOMATIC_COMPLETE仅当A证据有效、B148视频和344评分全部有效、控制通过、六项分析和全部诊断齐全、盲评媒体齐全、独立验证通过。科学结果可以阴性或不确定。PARTIAL/RESOURCE_WAIT/DEPENDENCY_BLOCKED允许诚实交付但不算本spec完成；列出准确剩余工作而非“已完美完成”。更新BM原实验笔记保留历史changelog，区分前轮部分结果与本轮终态，避免继续传播“B只缺权重”或“英文不成立”的过时泛化。
