# 执行设计（公共契约，先读本文件）

## 1. 固定问题与最小实现

N=自然音频；C=冻结 WavLM→HiFi-GAN 直接重合成；R=真实视频。比较 `G(C)+N` 与 `G(N)+N`，G 分别为 Wav2Lip / Ditto。C 不使用 TTS、不改特征顺序；解码器仍可能改变声学事件，所以“自然时间坐标”只指构造与评估时钟，没有宣称发音事件绝对不动。

旧 direct 实验 Wav2Lip replacement ΔSync-C=−0.231，95%CI[−0.294,−0.170]。本轮只问该固定变换是否有模型依赖；任何结果不证明全部候选有效/无效。先取 n=12 source groups，每组一条；这是探索预算，不是功效计算后的确认样本量。

只做inputs/replacement/visual/timing四个小包。D的TTS构造与统计分别放inputs/replacement包，以独立文件组织。复用模型加载、音视频I/O、landmark提取；不调用带旧样本数、旧门禁和隐式生成的父runner。analyze/validate只读已有产物。新CLI在实施时创建。

阶段顺序：P冻结基础输入 → A生成natural控制、B校准真实轨迹 → C整帧延迟诊断与A直接重建并行分析 → 测量控制通过且剩余预算足够才进入D的TTS输入/生成。D不以A是否阳性选模型；A/B/C各自的有效阴性都是完成。D的两个模型保持完整设计；某模型测量失败则D暂不运行，报告测量瓶颈。具体D门槛见其spec。

## 2. P：新来源与冻结输入

候选原始来源：本地 `data/lrs3/ellipsis-lrs3-raw/ainncy/pretrain.00.tar`、`pretrain.01.tar`，及 `data/dataset_samples/lrs3/pretrain/`。只读取 pretrain 媒体。设计时归档成员检查：两包分别 45 / 46 组；相对 `runs/lrs3_data_supplement_20260903/09_policy_cohort_retry2/source_pool.json` 的 58 组，分别还有 6 / 27 组不在该池。**这些数字未完成全历史去重或质量检查，不是已冻结样本。**

按以下顺序完成 P；任何一步缺条件都给具体缺项，不由 Luna 改科学规则：

1. 从所有既有 run 的 JSON/JSONL/CSV 协议、manifest、分割锁与样本清单提取明确的 source_group / source_video_id；补取记录中 LRS3 路径和 `lrs3_<11字符原视频ID>_<clip>`。同一原视频不同片段/不同目录仍是一组。仅扫元数据，不打开封存媒体。至少覆盖 n500 清单、上述补充池、三路最新 run 的父链、multiset250 和全部 `test_lock.json`；疑似清单无法解析则报告歧义，解决后才冻结。
2. 输出 `history_groups.json`，逐组保留来源文件与字段，绑定所读文件 SHA。所有历史已用或封存组都进入排除集；仅进入未使用原始下载归档不算做过实验。已知 speaker ID 也排除重合；未知则最终写 `speaker_independence=unverified`。
3. 剩余组按原视频ID字典序；每组clip按文件名排序，每组最多查看前八条。使用完整时长6–10秒的原片，附带完整官方英文Text转写；不裁成半句用于TTS。原片须有完整音视频覆盖、单个讲话人、无切镜/配音、嘴部无遮挡。只依据输入审查保存接纳/拒绝理由；不看生成结果或SyncNet分数。
4. 用冻结 MediaPipe 提取 R。前 140 个 25fps 帧的 valid fraction ≥0.95，首帧有脸；首帧双眼距离 ≥40 像素（原分辨率）、嘴可见。这里只检查可测性；不按“动态越大越好”挑样本。按上述排序取最先通过的 12 组，另取随后 2 组作 smoke-only。少于 14 组就 `BLOCKED_NEW_SOURCE`，不混回旧组、不动封存组、不在本轮自动下载新包。
5. N从完整真实片段一次性转为16kHz mono PCM16，实际长度L在[96000,160000]；音频原始PTS与视频首帧差≤20ms，否则输入拒绝。保留源时间戳、转码命令和文件/PCM SHA；此后N不再变。R建立25fps零PTS的完整副本，记录首帧映射；若原输入覆盖不足，拒绝而非循环或补静音。
6. 参考图取 R 的首帧。按首帧人脸框中心，边长为人脸框最大边的 1.5 倍，裁正方形（越界反射填充），缩放至 512²。R 每帧使用同一框；两生成器使用同一 512² PNG。固定裁剪覆盖不足是输入拒绝，不为某个生成器另选参考帧。记录人脸框和 resize 插值。
7. 写cohort.json（12正式+2smoke，分开数组），先冻结样本；构造C后写inputs.json，含每条N/C/R/ref路径与SHA、L、source_group、完整官方转写及SHA、原clip、裁剪框、PTS。A/B/C只按ID join。候选失败保留原队列，工程阻塞，不用备用组替换。D启动后另写tts_inputs.json，引用inputs SHA，不改已冻结基础输入。

适用范围：输入质量合格的6–10秒LRS3独白、静态参考图、前140帧评估，不外推到任意长片或原视频驱动。

## 3. 唯一候选：direct_v1

复用 `scripts/experiments/lrs3_direct_audio/run_wavlm_hifigan_lrs3.py` 的 `load_model` / `resynthesize` 与其 `exact_length`；只调用函数，不运行其 main。固定本地 kNN-VC revision `c616845c4e309e24d5927f15adbdf277a3d65358`、WavLM-Large layer 6、prematched `prematch_g_02500000.pt`，16kHz、hop320、1024维、eval/inference_mode、seed42。权重已在 `/home/wjj/.cache/torch/hub/checkpoints/`，执行前核验并冻结 SHA。

N → get_features(vad_trigger_level=0) → vocode；无检索、插值、MFA、增益归一化或混合强度。复用原PCM_16保存规则；记录raw decoder长度。只允许右侧裁剪/零填充到L，差额须≤640 samples，否则BLOCKED。评分只用前140帧，远离末尾修补。C必须有限、RMS>0、无幅度溢出；保存原浮点输出、features、最终PCM SHA。两个生成器接收逐字节相同的C；声学时延不后验修正。

## 4. A：生成、公共媒体与预算

每组每模型四条正式视频：N42、C42、N43、C43。42/43 是预定 seed，N/C 在同 seed 下成对；同一模型的其他参数一致。生成顺序按 source_group，并在偶数组先 N 后 C、奇数组先 C 后 N，每次调用重设 seed。另在队列前两组每模型重新 forward 一次 N42（N42_repeat），保留独立进程/日志。复制 MP4 不算 repeat。

模型差异包含架构、checkpoint 和各自固定推理实现；不将交互单独解释为架构因果。Wav2Lip 使用固定图片静态模式、25fps、GAN checkpoint、batch4；Ditto 仅用该实例上 smoke 通过的一种后端，固定 cfg/权重/源码 hash，整轮不切换 TRT/PyTorch。先用第一个 smoke group 的 N/C 各跑两个模型（共4视频），第二个 smoke group 仅保留给输入/候选诊断，不扩大科学样本。

Ditto实际核查：根目录 `/root/autodl-tmp/ditto-talkinghead`，commit=`c3e47eee2e626500017a0556b470d6d4182f85e8`；Python=`/root/autodl-tmp/envs/ditto/bin/python`，torch2.5.1+cu121且CUDA可见，安装元数据为TensorRT8.6.1/onnxruntime-gpu1.20.2。ffmpeg/ffprobe在/usr/bin。存在 `checkpoints/ditto_trt_Ampere_Plus/` 十二个engine；offline cfg=`checkpoints/ditto_cfg/v0.4_hubert_cfg_trt.pkl`，SHA=`be6729ca19e25269c4447d8ff4062e18de053b0f6dd38fdb6ef8166de6e8f3e3`。未实际反序列化engine或推理，smoke仍待执行。

用本地 `scripts/ditto_seeded_inference.py` 接口薄适配上述入口；上游inference.py主函数的seed调用被注释，必须用seeded wrapper。上游还将音频写为AAC且以os.system执行ffmpeg，因此不能信其退出码代表科学mux合格；本轮仍须验证视频后另行mux完整N PCM。不要沿用03_ditto.py默认samples.ids=1..10或旧online cfg。

正式输出要求至少140帧、25fps、首帧PTS=0、无中间丢帧；固定取前140帧，不按条件选择时段。缩放到相同512²，保存规范视频 FFV1/MKV；科学 mux 用 stream copy + 完整 N PCM，不用 `-shortest`。保存原始生成视频；规范化的 resize/encode 完全同策略。逐条验证 mux 解码 PCM=N，规范视频 mux 前后像素一致。评分输入将整幅规范512图固定缩到224²（同一插值），不重新检测/追踪/择脸。该固定脸框评分是本轮端点，不冒充旧官方整段流水线复现。

基础A/B预算：正式12×2×4=96视频，4同seed repeat，4smoke；100正式评分cells。A评分器的synthetic shift控制由已有embedding重算；C另增加24条真实DELAY生成/评分，不能混同两种shift。D再增加48正式M视频/评分及2条M smoke。完整总上限178视频/172正式评分cells；失败cell最多一次工程重试，全轮最多8次另计，科学阴性不重试。真实视频/landmark校准不产生TFG视频数。

## 5. B/C/D：各自回答问题

B读取冻结输入与A视频。真实轨迹校准和测试先做，不等A分数；模型视频到达后立即提取，规则见[B spec](specs/fresh-source-visual/spec.md)。C在相同新组上做生成时间传递，见[C spec](specs/fresh-source-timing-transfer/spec.md)。D在测量门槛后做M对N/C的双重比较，见[D spec](specs/fresh-source-tts-increment/spec.md)。A/B/C任何候选结果都不能改变队列或参数。

## 6. 三个 Luna 的文件边界与接口

| owner | 写入范围 | 完成的接口 |
|---|---|---|
| P | fresh_source_inputs 包/测试、run/shared/ | cohort.json、inputs.json、history_groups.json、validation.json |
| A | fresh_source_replacement 包/测试、run/A/ | videos.json、embeddings/、analysis.json、validation.json、final.json |
| B | fresh_source_visual 包/测试、run/B/ | features/、analysis.json、blind/、validation.json、final.json |
| C | fresh_source_timing 包/测试、run/C/ | delay_inputs.json、analysis.json、validation.json、final.json |

D由P写run/shared/tts_inputs.json，A写run/D/统计/验收/终态；其M视频仍由A生成并放videos.json。先开P/A/B三个worker；P结束后调度C，不超过三worker。D需补输入时短暂复用空闲worker，不挤占正在跑的GPU。

run 根统一为 `runs/fresh_source_cross_generator_<run-id>/`。P 冻结后 A/B 只读 shared。A 的 videos.json 逐条原子更新，key=`sample_id/model/arm/seed/repeat_index`，值含 path、sha256、frame_count、fps、status；B 只消费 status=complete 且 hash 通过的条目。A/B 不互写文件；tasks 和最终 BM 由主 agent 汇总，避免并发改看板。

每个runner提供 `--run-root PATH --stage ... [--resume]`；P stages=prepare/candidate/freeze/tts，A=smoke/generate/score/analyze/tts-analyze，B=calibrate/extract/analyze/blind，C=prepare/analyze。A generate/score显式接收 `--arms N,C|DELAY|M`，默认只N,C；各入口检查对应冻结manifest。四包validate接受 `--run-root PATH`，A增加 `--branch A|D`。freeze后输入/参数不改；resume校验spec/代码/模型/输入hash，不匹配用新run-id；修复续跑仍算同次探索。

## 7. 最小验收和终态

对应测试至少覆盖：按 ID join/重复 source 拒绝；N PCM/mux不一致；非法尾部修补；repeat不能复制；固定评分索引与 shift 符号；缺失不缩分母；篡改分析数值即使重新签 hash 也被拒绝；analyze 不调用模型。使用小合成数组和假模型，不写镜像式测试。

每条路线validator从原数组独立重算主端点、CI和终态，不导入producer统计/判定函数；允许共享I/O。距离矩阵容差1e-4、聚合/CI1e-6、PCM完全相等。A先独立写control_validation.json，B先写calibration.json及对应validation，D入口绑定这些文件SHA，不靠口头说门槛已过。无需另造审计框架。

A/B 各自先写 analysis→validation→final，final绑定前两者文件SHA；科学阴性退出0，工程不一致非零。人工未回收时 B 自动部分可 complete，`human_status=pending`、`visual_verified=false`，整体不得声称人工验证完成。固定 `replacement_confirmed=false`、`training_authorized=false`、`generalization_established=false`；可以给探索性后续建议。

## 8. 资源与关机

CPU准备/测试/统计本地完成，线程各≤2。各宿主同一GPU用OS flock，远端只有主 agent 调度和关机，B/子agent不发关机命令。远端先确认实例身份、已有任务、磁盘≥15GiB，记录GPU实际型号。账号密码通过 SSHPASS 环境变量，日志/配置不写凭据。

本次用户已经授权用完关机。下游恢复执行需目标实例可连接；端口可能变化，不能自行创建新付费实例。给定实例开机后，设置2小时总墙钟截止；smoke计时后预测总时长，超预算保存 BLOCKED_BUDGET，不自动扩预算。安全收尾须覆盖成功、失败和超时：远端保留日志/产物/退出码，回传并核验本地SHA，然后调用 `/usr/bin/shutdown`。回传失败仍保存远端可恢复证据、停止任务并关机，记录 `transfer_pending`；不要等待人工盲评而开着GPU。

本次观察到 `shutdown --help` 也关闭实例。下游把该路径视为执行关机接口，查文件/文档，不试运行帮助参数。关机后有现成 AUTODL_TOKEN 时按[官方实例状态接口](https://www.autodl.com/docs/instance_pro_api/#_5)核实 stopped；没有就如实报告“已请求关机、SSH不可达、控制台状态未核实”，不声称已停止计费。不要释放实例或删除权重/旧产物。
