## 阅读与问题

阅读 proposal → 本文 → specs/wav2lip-natural-content-residual/spec.md → tasks。BM先读 Startup Router、实验指令，然后读 `Wav2Lip historical shift rescore 2026-09-08`、`LRS3 masked TTS trajectory-specificity diagnosis`、`LRS3 Wav2Lip phone-aligned mel causal NO-GO`。

唯一新假设：**自然重建模型的辅助条件增量，在未遮挡的 natural 基底上仍有下游价值。**旧 phone-aligned TTS-mel residual 是 `TTS_mel-N_mel`，已有显著负结果。本轮是同一自然重建模型的 `prediction(correct)-prediction(zero)`，自然 mel 不被重建结果替换。它是新读出方式的有限检验，不能借此重开旧频谱/偏移扫参。

复用数据已被观察过。本轮为 seen-record exploratory probe，成功也不是独立 confirmation。旧 masked 模型内部依然使用 masked context；本轮增加的是未遮挡 natural bypass，不能称为已训练了全上下文生成头。

## 固定资产

父目录 `P=runs/lrs3_masked_tts_new_confirmation_20260902/`，以下SHA-256已只读核对：

| 相对 P 的路径 | SHA-256 |
|---|---|
| `00_cohort/manifest.json` | `8ec7a2ec0d37c6abf51075d57f1467658fc0b3de9574799a1b8f08f9a2bf21b5` |
| `03_data/natural_mels.npz` | `b48628ee3f09c0e46abc3d121d717bc72af10384cf7f20ea8caa2f5a7aaff44f` |
| `03_data/mask_manifest.json` | `956d7eb5bd7fe6b12a658191ac0ab80a9a8a6af77a0fce2aebcec30be561deec` |
| `04_reconstruction/reconstruction.json` | `79f79c0e1a5a439324dce1d35c0e2861ce428ece7f87f6c78eb195871f757df6` |
| `05_drivers/drivers.json` | `ad5bc9412833ac30a017f27e77499e4a60e1f38eb9c01c02f5db9c909bb92410` |
| `06_renders/box_manifest.json` | `1a1304633e2eae0b525e5b7f4a06391cf8c28a52668d7ea6e1598fc1f68a17be` |

使用全部16条、8组每组2条，按 `(source_group,sample_id)` 排序；以ID连接 cohort、drivers、boxes，禁止按数组位置zip。冻结所有间接资产hash，包括完整N PCM、face、boxes、144个driver文件及三个父checkpoint的来源绑定。只读预检已确认144个文件hash与父manifest一致、natural mel均为 `[80,308]`，正确和错配增量均非零。仍需下游重做完整结构检查。

driver键为 `(sample_id,seed,condition)`；三个固定seed为 `20260901,20260902,20260903`，使用条件 `PAIRED_TTS`、`SAME_PHONE_WRONG_INSTANCE`、`NAT_ONLY`。三条件必须来自同一个seed对应的hard-negative checkpoint；从 `04_reconstruction/reconstruction.json` 追溯，不能使用单独训练的NAT_ONLY模型。不得按seed的旧评分择优。

已核对每seed的checkpoint SHA分别为 `42f8b8ac7595219c9336e427311747dd71bf589f14e42cfc7a0cfb0d1ebb7def`、`1aac2221d5b586a9e66896c3e4e4f38f08275a08f34f001723d826137c824d96`、`fc9dd8199494dd8dabbc08172ace7d1849a1849ec523747242470a208ddce920`。原文件位于 `runs/lrs3_masked_tts_trajectory_specificity_20260902/04_training/<seed>/hard_negative/checkpoint.pt`，本轮仅hash核对，不加载训练。

所有父数据都是允许复用的已观察记录；除下文明确绑定的2个已有scorer parity cell外，不打开其他源组媒体或 sealed 数据。parity不进入16条科学统计。这里不把父字段 `new_confirmation` 误判为本轮新独立数据，也不把它改写为fit。

## 时间范围：必须明确的缓存限制

父 natural mel 只编码原N的前 **61440 samples = 3.84s**，不是整条长录音。`N`基线是这段支持内完整、未遮挡的natural mel；不在支持内挖洞、不用NAT_ONLY重建作为基线。

每条使用官方 `audio.melspectrogram(float32(N[:61440]))` 复算，与缓存逐值最大误差<=1e-6且chunk完全相同；不对整条mel插值成308列。最终评分音轨仍是**整条**原N的PCM，长度可超过短视频。报告必须注明“前3.84s特征支持的短视频实验”，不得声称验证整段长音频。

## 四臂精确定义

`M`是缓存natural mel；`P_s,W_s,Z_s`分别为三个条件的缓存record-level driver（已经denormalize/clip的float32数组）。采用这些存盘数组本身，不重新训练或重新生成预测，不把标准化空间和Wav2Lip mel空间混用。

`K`是父driver.used_masks里 `[global_start_frame,global_end_frame)` 的有序并集。所有seed/条件的mask集合、区间、覆盖计数必须一致；比较mask身份和位置字段，不比较本来就应不同的prediction_sha256。父driver在K外必须逐值等于M。mask只界定增量写入位置，不删除自然上下文。先转float64：

```text
R_correct = mean_s(P_s - Z_s)
R_wrong   = mean_s(W_s - Z_s)
```

仅沿K内的时间列排列 `R_correct[:, K]`：用 `PCG64(int.from_bytes(SHA256("natural-content-residual-v1\0"+sample_id)[:8],"little"))` 的 `permutation(len(K))` 得到S；若恰为恒等排列则固定交换前两列，不重新抽。K外为零。这保留打乱前每频率的值集合与整体L2范数，破坏精确时间对应；不是随机生成新噪声或音频shift。

将R_wrong全局乘 `||R_correct[:,K]||_F / ||R_wrong[:,K]||_F`，得到W；C=R_correct。任一范数<=1e-12或K不足2列则 `INPUT_DEGENERATE`，全队列停止，不删样。每条三个候选共用一个保守幅度系数：

```text
a = min(0.25, 0.5 / max(abs(C), abs(W), abs(S)))
N       = M（原数组逐值复制）
CORRECT = float32(clip(float64(M) + a*C, -4, 4))
WRONG   = float32(clip(float64(M) + a*W, -4, 4))
SHUFFLE = float32(clip(float64(M) + a*S, -4, 4))
```

`max`跨三臂全部元素。此固定上限限制任一preclip改动<=0.5 mel单位；不改变a、符号或掩码寻找正分。K外必须与M bit-exact，所有候选shape/dtype/有限性相同。记录a、clip比例、pre/postclip L2/RMS、逐频率统计及输出hash。

裁剪会破坏实际等范数。独立设 `norm_control_valid`：每条W和S的postclip扰动L2与C相比均在 `[0.95,1.05]`，且C/W/S均非零；不满足仍可报告相对N收益，但不得给内容特异性结论或自动调参。WRONG是**同音素错实例**，不是错误句义；即使胜过WRONG，也还受实例/说话人/声道差异限制。SHUFFLE也可能引入时间不连续，不能单凭它证明纯语义因果机制。

## 生成与评分接口

为避免动态face把原口型带入，四臂固定同一静态头像：解码父face第0帧，取对应父boxes第0项 `[x1,y1,x2,y2]`，无padding地裁剪，`cv2.resize(...,(224,224),INTER_LINEAR)`，将该画面重复为全部视频帧。原始box必须在帧内，非法则工程BLOCKED，不重新检测或选帧。

复用 `masked_tts_tfg_probe.direct_mel.chunk_mels(M,25)` 的官方chunk公式（80Hz mel、16列窗口），本轮308列应产生93个chunk/93帧。小worker可复用 `wav2lip_face_roi_replacement.generation_worker.load_model/render_arm/encode_video`；传93张静态帧与93个生成box `[top,bottom,left,right]=[0,224,0,224]`。注意它与父boxes的xyxy格式不同！不要直接调用旧direct_mel.render：它包含有损DIVX/H264编码，且frames_rendered统计不等于实际写出的循环帧数。

Wav2Lip checkpoint `third_party/Wav2Lip/checkpoints/wav2lip_gan.pth` SHA=`ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`；宿主解释器 `/home/wjj/.venvs/wav2lip/bin/python`，CUDA真实kernel预检，batch=4、eval、no_grad；不静默CPU fallback。本次会话宿主GPU权限问题不等于驱动损坏。

每个生成worker固定torch seed=20260909、deterministic_algorithms=true、cudnn.benchmark=false、cudnn.deterministic=true，并禁用TF32；记录这些设置。N_REPEAT使用同样配置独立forward。遇到不支持的确定性算子则报告工程原因，不删除重复检查。

保存93帧25fps 224×224 FFV1视频，以video stream copy与**完整16k mono PCM16 N** mux为Matroska；A/V起点都是0，禁止-shortest、删尾、音频归一化、补帧或重采样。独立核对解码PCM逐样本一致、帧数、像素hash与PTS。

SyncNet复用 `wav2lip_roi_peak_recheck.worker.SyncNetScorer` 的完整224×224画面入口；不再次裁嘴、不运行S3FD、不混入历史tracked分数。CPU解释器 `/home/wjj/.venvs/syncnet/bin/python`，batch=20、threads=4；权重SHA=`961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`。保存fresh embeddings与 `[88,31]` 距离矩阵。矩阵行数遵循当前worker的 `video_frames-5` 契约，不自行改成89。

固定主窗口 `U=range(30,58)`，即88行两侧各去30行；所有臂同一绝对时间和全部31列有效。先验证每行的视觉/MFCC及offset邻域都在真实支持内。先在U上float64求平均曲线z，然后 `D=min(z)`、`C=median(z)-min(z)`、`offset=15-argmin(z)`，并列取最小列。`k_N=argmin(z_N)`一经基线得到冻结，所有候选与控制使用同一个k_N：

```text
gain_C(X,Y) = C_X-C_Y
gain_D(X,Y) = D_Y-D_X
gain_A(X,Y) = z_Y[k_N]-z_X[k_N]
```

全部正值更好。FULL仅作附录描述，不用于选择窗口或推翻U结论；不先逐行求C再平均。必须报告U上哪些Wav2Lip chunk接触K及覆盖比例，覆盖不足不能事后移动U。

## 两阶段与预算

Stage A先做资产锁定、四臂CPU构造及全体mel/chunk parity，随后：

1. 重跑历史rescore v7的 `parity.json` 中 **v4.cells** 绑定的同两个v4评分cell，与其原矩阵最大绝对误差<=1e-4、在该parity记录原有support上的端点误差<=1e-4、offset一致；只复用明确parity资产和选择规则，不运行历史实验父runner。顶层rows的46旧cell不是这2个scorer parity；parity的支持也不强改成本轮U。
2. 生成16个N；排序前2条分别再forward一次N_REPEAT，不能复制N视频。mel/chunk必须bit-exact；重复视频解码像素必须一致，C/D/anchor绝对差<=1e-4、offset相同。失败记工程BLOCKED并保留现场，不能放松为“均值差不显著”。
3. 对每条同一个V_N加一个错时评分控制：`A_DELAY[:3200]=0; A_DELAY[3200:]=N[:-3200]`，长度不变，只替换音轨，不新生成视频。此cell的source_audio和PCM期望必须绑定A_DELAY，而非N。

parity父manifest SHA=`56faf1727b6091e1dec7c2c8f9e37b330007e9fd195fdf151bfaa0e4a10057a5`；两cell均为 `lrs3_6ORDQFh0Byw_00008/DYNAMIC`，video_arm分别V_N/V_DELAY_200，audio_arm均N。从其worker.json解析准确媒体/source_audio和hash，参考矩阵/support绑定也写入本轮protocol。

固定自然anchor距离损伤 `z_DELAY[k_N]-z_N[k_N]`：组bootstrap 95%下界>0且至少7/8组正；至少14/16条的 `offset_DELAY-offset_N=-5±1` 帧。该符号由距离矩阵列约定确定，应有人工合成嵌入移位单测。此控制只要求固定anchor检测错时；**不要求自由offset的C/D下降**，因为搜索可能补偿全局延迟。失败为 `CONTROL_FAILED`，不进入B；已通过的接口不能宣称“replacement已确认”。

Stage A独立验收valid且上述控制通过后，Stage B运行每条CORRECT/WRONG/SHUFFLE共48个新视频和48个N评分cell。新视频预算为18+48=66；评分预算为2 parity+18 N/repeat+16 delay+48 candidate=84。派生静态face素材单列，不算TFG生成。无训练、无新特征提取模型、无TTS请求、无vocoder。允许hash绑定一致的工程resume，不追加科学重复。

## 统计与唯一决策

seed已经在driver层均值融合，不当作独立样本。每个contrast先算record收益，再在每组2条取均值；8组等权平均，按source_group排序，以 `PCG64(20260909)` 保存10000个8组有放回索引，所有contrast共用索引；NumPy linear quantile的双侧95%CI。不得把28个窗口行或16个相关record当作独立推断单位。

定义 `pass(X,Y)`：C、D、A的组均值CI下界**全部>0**，且至少7/8组C/D/A三项都正。唯一主假设为CORRECT优于N，额外要求mean ΔC>0.05；这不是零效应检验，也不是等效性结论。

科学标签依序：

1. 工程/独立验收失败：`BLOCKED`，科学未判定。
2. 结构退化：`INPUT_DEGENERATE`，没有下游收益判定。
3. Stage A敏感性失败：`CONTROL_FAILED`，B未执行。
4. CORRECT/N主假设未通过：`NO_INCREMENT_ESTABLISHED`，停止该固定增量读出，不宣称全部语义路线无效。
5. 主假设通过、norm_control_valid为true、`pass(CORRECT,WRONG)`且`pass(CORRECT,SHUFFLE)`：`CONTENT_RESIDUAL_SIGNAL`。
6. 主假设通过但第5条不满足：`NATURAL_GAIN_MECHANISM_UNRESOLVED`，如范数控制不合格需显式标注。

第5条为预注册的合取机制检验，不能只挑胜过的控制；第4条即使CORRECT胜WRONG也不能提升结论。第5/6条最多建议独立source-group确认同一构造，其中第6条不能声称内容特异性。所有结果 `replacement_confirmed=false`（此轮探索）、`waveform_head_authorized=false`、`generalization_established=false`、`historical_shift_gate_repaired=false`。不以无显著差异证明等效，不将旧masked收益合并进本轮统计。

## 小实现与交付

建议新建 `runner.py`、`drivers.py`、`worker.py`、`analysis.py`、`validate.py`；共享纯I/O和上述冻结模型函数即可，不复制父流水线。下游实现CLI（当前尚不存在）：

```bash
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_natural_content_residual.runner --run-id <id> --stage all
# 仅工程中断后恢复，必须核对全部绑定
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_natural_content_residual.runner --run-id <id> --stage all --resume
PYTHONPATH=. /home/wjj/.venvs/syncnet/bin/python -m scripts.experiments.wav2lip_natural_content_residual.validate --run-root runs/wav2lip_natural_content_residual_<id>
```

阶段至少支持 `prepare/controls/candidates/analyze/all`。产物为 `protocol.json`、`input_audit.json`、drivers/videos/scores manifests、`control_analysis.json`、`control_validation.json`、`analysis.json`、`validation.json`、`final.json`、`result.md`、`review.json`。manifest记录唯一cell键、代码/spec/输入hash、命令、环境与真实执行计数；阶段成功后原子落盘，验证完整且绑定相同才复用。

独立validator可共享文件读取，但不能调用producer的构造、bootstrap或门禁函数：从父数组重算增量/排列/裁剪（float32成品须逐值一致）、从embeddings重建矩阵（abs<=1e-4）、重算U曲线/全部统计和状态（abs<=1e-6），解码验证媒体PCM/视频/PTS以及静态face来源。验证后final绑定validation，避免双向循环hash。

聚焦测试：ID乱序join/漏cell；seed先融合而非伪重复；零辅助严格回到M；K外一致及等范数/clip标志；xyxy到生成box坐标；93帧/88行/U支持；delay offset符号；自由C正而anchor不通过；胜错配却未胜N；缺一对照不能标记内容信号；tamper/restart拒绝。测试、独立验收及OpenSpec strict都通过才勾完成项。

GPU前至少5GiB空闲，串行生成，OOM允许原配置清理本次进程后resume，不改batch或科学参数。缺权重/资产先给可复现错误，不下载新模型或删旧runs。BM同一实验笔记 `Wav2Lip natural content residual probe 2026-09-08` planned→running→concluded；失败写真实阶段与原因。每次更新先全文读、保留changelog、写后读回，记录数字、结论边界及run指针。

阶段后自审并保存review.json：实现自审要标明self-review，不冒充独立subagent审查。最终回答包含工程状态、科学状态、CORRECT/N及两对照的C/D/A、预算实数与报告路径。
