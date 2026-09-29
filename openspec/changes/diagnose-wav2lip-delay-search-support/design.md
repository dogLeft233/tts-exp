## 1. 问题和固定输入

唯一问题：上一轮三个offset异常是否可由已知+200ms音频延迟将峰值移出原搜索域解释？这是已观察数据的诊断，不是独立确认。

BM先读Startup Router和实验指令，再读 `Wav2Lip natural content residual probe 2026-09-08`。历史判断依据为 `Wav2Lip historical shift rescore 2026-09-08`、`Wav2Lip spectral structure replacement 2026-09-08`、`LRS3 masked TTS trajectory-specificity diagnosis`：自由offset得分、直接TTS谱迁移、masked重建辅助收益是不同证据，不能合并成replacement确认。

父目录 `P=runs/wav2lip_natural_content_residual_20260908_v2/`。下表为文件字节SHA-256，不是JSON内部artifact_sha256：

| 相对P的路径 | SHA-256 |
|---|---|
| protocol.json | e0ad38110883bc7079e66866b5187f68636ee2ec8882ddba94879b203a5a2485 |
| control_scores/manifest.json | 143e7b7e72eb4bc14d10cfa88afa87decb59f87da48eed7581bf93780dfda5b2 |
| control_analysis.json | 068bf6ec3d43e432032ad6bfde8ce7cc284669b0c020575eb4ccf77bb30f9041 |
| control_validation.json | 9ae69db31e1f4b636a0b09e2793a86b373b58f183d8589f8a10f60f11d9138c6 |
| final.json | a9a423d96cddd774a3b4ac570f1d1cdceddd6e6eb700dd0b668c652faeb473aa |

按 `(sample_id,video_arm,audio_arm)` 连接manifest，取16组 `(N,N)` 和 `(N,A_DELAY)`，从父protocol恢复8个source_group、每组2条，按 `(source_group,sample_id)` 排序。parity与N_REPEAT仅验证父证据绑定，不进入32-cell分析或16条统计。嵌套 `score.worker` 指向worker.json，其中 `visual`、`audio_embedding`、`matrix`及对应hash是缓存入口，文件名分别为visual.npy、audio.npy、distance.npy。

锁定全部输入、引用媒体、PCM、embeddings、代码和spec hash。检查32个cell的embedding为有限float32 `[88,1024]`，旧矩阵为 `[88,31]`；同一记录两cell的visual必须逐值一致。独立解码检查同一视频像素及PTS一致、93帧25fps、A/V起点0、音轨与source_audio逐样本一致；确认 `A_DELAY[:3200]=0; A_DELAY[3200:]=N[:-3200]` 且长度不变。复用完整原N，不能裁成3.84s。缺资产或绑定冲突为BLOCKED，不能补生成或换样。

## 2. 先复现旧结果

固定 `U=range(30,58)`。视觉行r对应frames[r:r+5]，音频embedding行q对应MFCC[:,4*q:4*q+20]；16k PCM每视频帧640 samples。定义物理lag `l=q-r`、报告offset `o=-l`。旧列 `k=l+15`，故 `o=15-k`。

按父实现float32距离契约独立重建旧矩阵：`d(r,l)=sqrt(sum((V[r]-A[r+l]+1e-6)**2))`；只有重现旧FULL矩阵时，越界q才使用父零embedding padding。矩阵最大误差<=1e-4；U均值曲线及C/D/anchor误差<=1e-6，argmin并列取最小列。先float64平均U行，再计算端点，不能逐行求峰后平均。

复现旧13/16与如下完整异常集；从数组计算后再与表比对，不能将表当计算结果：

| sample_id | k_N | 预期延迟列k_N+5 | 旧offset差 |
|---|---:|---:|---:|
| lrs3_7JVTirBEfho_00039 | 27 | 32 | +9 |
| lrs3_7PwvGfs6Pok_00003 | 27 | 32 | +1 |
| lrs3_7c5t6FkvUG0_00001 | 28 | 33 | -2 |

输出全部16条的 `expected_column`、`expected_in_legacy_domain` 和旧失败标志。若失败集与预期越界集不相同，保留实际差异，不能宣称三条均由边界解释。

## 3. 唯一诊断干预：平移已知控制的搜索域

音频真实延迟5帧时，内部支持上应有 `A_DELAY[q+5]≈A_N[q]`，所以 `d_DELAY(r,l+5)≈d_N(r,l)`。近似关系须从真实embeddings验证；不要假定MFCC/forward完全平移等变。

对每条一律计算以下两张 `[28,31]` 矩阵，行顺序为U，列j=0..30：

```text
l_N(j) = j - 15                 # natural搜索域[-15,15]
l_DELAY(j) = j - 10             # delay搜索域[-10,20]，精确平移+5
B[r,j] = distance(V_N[r], A_N[r+j-15])
T[r,j] = distance(V_DELAY[r], A_DELAY[r+j-10])
z_N = mean_U(B, dtype=float64)
z_DELAY_matched = mean_U(T, dtype=float64)
j_N = argmin(z_N)
j_DELAY = argmin(z_DELAY_matched)
offset_N = 15 - j_N
offset_DELAY_matched = 10 - j_DELAY
offset_difference_matched = offset_DELAY_matched - offset_N
```

这里两个argmin索引相同才对应物理offset差-5。不能继续给delay套用 `15-j`；也不能用roll、复制自然曲线、插值或零padding伪造新增lag。保留natural搜索域；对delay采用整个31列平移后的域，不以旧峰为中心择优。只对已知A_DELAY控制使用该坐标变换，候选音频没有已知延迟，不能采用此操作。

合法支持可预先确定：B的q在15..72，T的q在20..77，全部在0..87内。逐行核对实际MFCC窗口、前置preemphasis所需样本和音频支持；不依赖padding，也不移动U。只计算这两个域，不扩大全局vshift或扫描范围。

除offset外，逐条保存真实 `A_DELAY[q+5]-A_N[q]`（q=15..72）的max/RMS误差、`T-B`的max/RMS误差、两条均值曲线最大差、最小/次小值差。这些连续量用于解释是否仍有特征边界或峰歧义，不能据此调阈值、排除样本或挑选窗口。

固定anchor损伤仍使用旧延迟矩阵原物理lag：`z_DELAY_legacy[j_N]-z_N[j_N]`。不能误用 `z_DELAY_matched[j_N]`，后者已经补偿了已知错时。按父8组等权均值、PCG64(20260909)、10000次8组有放回索引、linear quantile的95%CI重算，保存抽样索引；预期mean=+1.162568、CI=[+0.844647,+1.482466]、8/8组正，比较未四舍五入原值。

## 4. 固定判定和停止条件

分别输出两个问题的答案：

- `corrected_control_pass`：沿用原anchor规则（95%CI下界>0且至少7/8组正）及至少14/16条offset差在[-6,-4]。原门槛不降为13/16。
- `boundary_explanation_complete`：旧失败集恰为三条预期越界记录，修订后这三条均通过且原13条仍全部通过，即16/16。连续误差和峰歧义同时报告；该标志支持搜索域解释，不证明其他数值影响严格为零。

独立验收通过后：两项都true为 `SEARCH_SUPPORT_RECOVERED`，`content_probe_revision_eligible=true`；否则为 `CONTROL_UNRESOLVED`，详细标明14/16门槛是否单独通过及剩余异常，停止此诊断，不继续搜索域/窗口/延迟扫参。工程或独立验收失败为 `BLOCKED`、资格false。更严格的16/16只用于完整解释标签，不替换原控制通过门槛。

所有终态保持 `stage_b_authorized=false`、`replacement_confirmed=false`、`historical_shift_gate_repaired=false`、`waveform_head_authorized=false`、`generalization_established=false`。父run的CONTROL_FAILED是按原协议得到的有效历史终态，保持不变。

若SEARCH_SUPPORT_RECOVERED，下一步仅建议给natural-content-residual写显式续跑修订：绑定本次诊断、沿用原16条/四臂/幅度/U/31列候选端点/固定anchor/统计，修订已知delay控制的配对搜索域，再运行原48个候选视频。该后续仍是seen-record探索。本轮不改父runner或自动执行候选。若CONTROL_UNRESOLVED，暂缓该内容残差实验，报告剩余可观测性限制，不再无界修门禁。

## 5. 小实现、验证和交付

建议只建 `runner.py`、`analysis.py`、`validate.py`及必要的 `__init__.py`。复用父纯I/O、媒体读取与hash工具；核心坐标/距离/判定在本包实现，避免修改共享scorer或复制整个父流水线。若.codegraph存在，定位现有代码先用CodeGraph。

CPU解释器 `/home/wjj/.venvs/syncnet/bin/python`。下列CLI是待实现接口：

```bash
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_delay_search_support.runner --run-id 20260909_v1
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_delay_search_support.validate --run-root runs/wav2lip_delay_search_support_20260909_v1
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m pytest -q tests/experiments/wav2lip_delay_search_support/
openspec validate diagnose-wav2lip-delay-search-support --strict --no-interactive
```

仅工程中断可 `--resume`，须逐一核对绑定；科学终态不追加重复。复用不标记fresh forward。预算：32个缓存cell、32张派生U矩阵；新生成/新forward/训练/GPU均0。新增输入或代码/spec hash变化需新run，父run只读。

产物：`protocol.json`、`input_audit.json`、`matrices/`及其manifest、`per_record.json`、`analysis.json`、`validation.json`、`review.json`、`final.json`、`result.md`。protocol在计算新域前冻结；记录parent hash、lag数组、U、容差、预算及代码/spec。final最后写，绑定单向引用validation，不能循环hash。

validator从原embeddings独立索引重建旧/新矩阵及bootstrap/判定，不调用producer的分析或判定函数；检查媒体及所有hash、完整16条、自然anchor未变、零fresh预算，数值容差同第2节。review明确为self-review，不能把独立数值validator描述为subagent审查。

必要测试：合成embedding唯一峰落在旧k=27/28，延迟后旧域截断而配对域恰好offset差-5；内部峰亦正确；符号错误/零padding/用补偿曲线算anchor被拒；argmin并列取首列；13/14/16条通过时两个布尔量与终态不同；乱序ID连接、缺cell、篡改embedding/lag/U、resume绑定不一致均拒绝。合成测试检验数学实现，不能替代真实16条诊断。

BM同一笔记 `Wav2Lip delay search support audit 2026-09-09`：planned→running→concluded；工程中断保留running并写阻塞原因。每阶段更新前全文读、保留changelog、更新后读回；记录旧13/16、新通过数、三条恢复情况、连续误差、资格和run指针。执行完成后tasks按实际勾选，最终交付报告上述数字及下一步资格；尚未运行不得填写成功结论。
