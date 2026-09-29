# 下一轮实验：补完未回答的问题（2026-09-10）

## 推荐与历史依据

先完成两项被实现门禁提前终止的实验，并行做一项零模型的视觉动态诊断。当前不进入头训练、同批调参或大规模泛化。本轮是已见数据上的探索/纠错续验，不是独立确认。

| 最新证据 | 当前结论 | 本轮决策 |
|---|---|---|
| [波形增益](../runs/wav2lip_waveform_gain_20260910_v1/result.md) | GAIN_PLUS ΔSync-C=+0.017，但固定 anchor 收益为负；完整门槛未过 | 关闭固定 ±3 dB 探针 |
| [重建基底](../runs/wav2lip_reconstruction_base_interaction_20260910_v1/result.md) | BASE_CONTENT ΔSync-C=+0.027，99%区间跨0；交互也未过联合门槛 | 关闭固定基底/内容增量调参 |
| [phone-core](../runs/wav2lip_phone_core_shrinkage_20260910_v1/input_audit.json) | 两条因 no public U exposure 停止，0视频/0评分 | 原 design 要求每组曝光，construct_candidates 却逐记录拒绝；先验证统计单位，再补测 |
| [E0](../runs/wav2lip_parallel_evidence_contract_20260910_v1/result.md) | P/F0 matched-domain 16/16；f46_boundary_restored=false | 不能把 P/F0 通过数当 F46 通过数；F46_C 仍待控制后生成 |
| [视觉教师](../runs/lrs3_visual_teacher_missingness_20260909_v1/result.md) | 115/133 observed；缺失界限[-0.221,+0.214]跨0 | 先检验动态证据及指标辨识力，再决定是否值得补采 |

长期历史：中文 TTS diagonal 优势不等于换回自然音轨后的 replacement；MFA-linear、固定谱桥、完整 natural 内容残差均未建立稳定 replacement 收益。见 [历史复核](replacement-history-review-20260909.md)。该旧综述中的“E1尚未执行”等状态已被上表更新，历史数字仍保留其原端点限制。

设计期核对了最新 BM 全文、上述结果与关键源码；没有运行本轮候选、F46正确域统计或视觉动态统计。phone-core 的组级覆盖是否通过尚待执行；这是源码偏差发现，不能预先宣布旧门禁已修复。

## 三路交接

| 路线 / 建议优先级 | change（每个独立交给一个 Luna） | 实验包 | 新视频 / 新评分上限 |
|---|---|---|---|
| A / 第一 | [complete-wav2lip-phone-core-group-support](changes/complete-wav2lip-phone-core-group-support/design.md) | wav2lip_phone_core_shrinkage | 34 / 36 |
| B / 第二 | [complete-wav2lip-reference-matched-control](changes/complete-wav2lip-reference-matched-control/design.md) | wav2lip_reference_conditioning_interaction | 36 / 54 |
| C / 同时启动CPU | [probe-lrs3-visual-dynamic-specificity](changes/probe-lrs3-visual-dynamic-specificity/design.md) | lrs3_visual_teacher_missingness | 0 / 0 |

三路不等待对方科学结果。CPU准备/测试/分析可并发；单卡先A后B。总上限70视频/90评分；控制或支持失败可提前结束。零训练、零TTS、零vocoder、零下载。不要启动旧 runner 的 all。

## Luna 的最短阅读与实施路径

1. 读本页、本 change 的 proposal/design/spec/tasks；C可跳过“A/B公共生成、坐标与对照”，无需通读全部实验史。
2. 只读取 [输入快照](next-experiments-20260910-inputs.json) 中自己 branch 的键。路径均相对仓库根，SHA是文件字节摘要，区别于JSON内部 artifact_sha256。间接数组/媒体沿已绑定manifest逐项核验；缺失历史绑定就报告，不能用当前自hash冒充历史证据。
3. 先实现纯计算与边界测试，再完成 prepare；支持/控制验收通过才运行候选。每个包用 runner.py、analysis.py、validate.py 和必要的小adapter即可，不建新框架。
4. 独立 validate 从原始数组重算核心量；不能导入producer的候选构造、统计或终态函数。允许共享I/O、模型输出格式及文件hash工具。无需另造全套审计系统。
5. tasks 只勾实际完成项；工程通过的科学阴性或有效提前停止是完成。缺资产、坏hash、验收不一致是 BLOCKED，science=not_available；不要伪造PASS。

三路仅写 `scripts/experiments/<package>/`、`tests/experiments/<package>/`、`runs/<package>_<run-id>/`、自己的 change/tasks 和自己的BM实体。公共helpers、旧run、旧spec、CONTEXT和Task Board只读。共享helper需要改动时在本包写小adapter。

待实现CLI（package取上表；本轮默认run-id为20260910_v1）：

```bash
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.<package>.runner --run-id 20260910_v1 --stage all
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.<package>.validate --run-root runs/<package>_20260910_v1
pytest tests/experiments/<package>/
openspec validate <change-name> --strict --no-interactive
```

A/B runner支持prepare/controls/candidates/analyze/all，C支持prepare/analyze/all；A/B validator支持--stage prepare|controls|all，C支持--stage prepare|all，默认all。analyze只消费已有产物，缺失就非零退出，不隐式调用模型。--resume仅接受输入/代码/spec完全一致的run；变化用新run-id。prepare冻结本页、inputs.json、自己的proposal/design/spec、代码及所用helpers的hash、样本与预算；tasks勾选状态不参与运行身份，避免正常勾选破坏resume。

最少输出：protocol.json（含输入audit）、support.npz或drivers/manifest.json、analysis.json、validation.json、final.json、result.md；A/B另有control_validation.json及media/scores manifests。final最后写，绑定analysis/validation文件字节hash，保留实际新建/复用/失败尝试数量。验收失败非零退出；科学阴性退出0。QUEUED表示资源忙，保留未完成状态。

## A/B 公共生成、坐标与对照

P=`runs/wav2lip_natural_content_residual_20260908_v2`；Q=`runs/wav2lip_natural_content_residual_continuation_20260909_v1`；D=`runs/lrs3_masked_tts_new_confirmation_20260902`。16条/8个source_group、每组2条；按(source_group,sample_id)排序，按ID连接。P/control_scores中N为基线，Q中的CORRECT供B使用。

自然mel [80,308]来自原N前61440 samples；25fps，chunk起点int(t*80/25)、宽16，末块取最后16列，共93视频帧。每帧重复固定224×224 face，FFV1。mux完整原N、16k mono PCM16，零PTS、保留音轨尾部，不用-shortest。评分沿父整224画面/JPEG提帧入口，不再检测或裁脸。完整评分矩阵[88,31]，公共U=range(30,58)，所有lag的索引必须真实支持，禁止padding。

从embedding按float32 `sqrt(sum((V[r]-A[q]+1e-6)**2))`重建矩阵，自然q=r+j-15。U内先float64平均得曲线z，再C=median(z)-min(z)、D=min(z)。k0=argmin(z_P_N)，并列首列；所有科学比较固定这个k0。收益ΔC=C_X-C_Y、ΔD=D_Y-D_X、ΔA=z_Y[k0]-z_X[k0]。不得选择候选峰代替固定anchor。

每条分支独立执行公共控制：

- P/N与P/DELAY缓存：自然lag=-15..15、延迟+3200 samples的lag=-10..20；offset分别15-j、10-j。保存两个域，复现legacy 13/16与matched 16/16。未补偿损伤仍用legacy delay曲线在自然k0处减N；均值约1.162567633、95%CI约[0.844646827,1.482466256]、8/8组正，容差1e-6。父bootstrap为PCG64(20260909)、10000次组抽样。
- 两个fresh scorer parity：输入快照的 historical parity.json 中 v4.cells；按该parity记录原有support重算，矩阵/端点误差均≤1e-4、offset相同，不强改为本轮U。前两条P/F0/N各fresh forward+评分一次；解码像素与P/N逐值一致，矩阵≤1e-4、U上端点≤1e-6、offset相同。
- 独立validator核验控制后才能生成候选；所有入口及resume都检查当前control_validation。控制计算正确但敏感性未过为CONTROL_FAILED；实现不一致为BLOCKED。

模型纯接口：`wav2lip_face_roi_replacement/generation_worker.py` 的 load_model/render_arm/encode_video，`wav2lip_roi_peak_recheck/worker.py` 的 SyncNetScorer；可用 `wav2lip_probe_gpu.py`/`wav2lip_probe_score.py` 作薄worker，自己组装plan。旧reference/phone runner与validator仅作结构参考，不能直接继承门禁。

Wav2Lip GAN权重sha=ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8；SyncNet权重sha=961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442。生成解释器`/home/wjj/.venvs/wav2lip/bin/python`，batch4、eval/no_grad、seed20260909、确定性开、TF32关；评分解释器syncnet，CPU batch20、threads2。

生成进程持同一OS flock `/tmp/tts-exp-wav2lip-gpu0.lock`，锁内预检真实CUDA kernel及空闲≥5GiB，busy即QUEUED。不要杀其他进程。每分支CPU≤2线程。每cell最多一次工程重试，仍失败BLOCKED，计入失败尝试，不换科学参数。

## 统计、结束与下一轮

A/B先逐记录，再组内两条均值，最后8组等权；所有contrast同一组bootstrap索引。A用PCG64(20260911)，B保留原PCG64(20260910)，各20000次、NumPy linear quantile、双侧99%CI。C见自己的design。99%仅为已见小队列探索筛选，不提供反复探索的严格错误率保证。输出JSON不先舍入，报告Sync-C三位小数。

统一gain(X,Y)：C/D/A三项99%CI下界均>0、至少7/8组三项同时正、mean ΔC>0.05。只用于A；B只检验参考交互。所有终态 replacement_confirmed=false、waveform_head_authorized=false、generalization_established=false、historical_shift_gate_repaired=false。C额外保持parent_gate_repaired=false、stage02_authorized=false。

A阳性→另设计新source-group同构造确认，再考虑波形可达性与第二TFG；A阴性→关闭固定phone-core算子。B交互明确→多参考的稳健性确认；B阴性→关闭该参考交互探针。C只在有动态且时间特异证据时建议新的完整视觉队列；不靠observed-only结果放行旧教师。三路均无信号时暂停同批微调式探索，整理负结果后重新定义方法假设。

BM使用memory-notes格式：执行前Router/实验指令，搜索并更新design指定的唯一实体，planned→running→concluded或blocked；保留Changelog，写后读回。无需并发更新总看板。

格式依据为仓库 OpenSpec 1.11.0 的 spec-driven instructions，并核对 [官方schema](https://github.com/Fission-AI/OpenSpec/blob/main/schemas/spec-driven/schema.yaml)：proposal/design/specs/tasks，Requirement使用SHALL和四级Scenario。设计文件用于解除实施歧义，不建立生产平台。
