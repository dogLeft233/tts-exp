# 新来源 × 两生成器 × 独立视觉验证

状态：⚠️ P 阶段已执行并严格阻塞；A/B/C 未打开科学队列，D 延后。面向 GPT Luna；2026-09-10。

解除 P 阻塞的下游入口：[新来源补充与视觉审计修复 spec](changes/unblock-fresh-source-inputs/design.md)（2026-09-11，已实施；当前 r2 演练因 92 个历史排除、0 个新组保持 blocked）。先取得 `COHORT_READY`，再接回本轮 candidate/freeze 与 A/B/C/D；旧 run 和科学门槛保留。

交给下游的入口：[design.md](changes/probe-fresh-source-cross-generator/design.md)。只读公共设计和自己负责的 spec；任务拆分见 [tasks.md](changes/probe-fresh-source-cross-generator/tasks.md)。不要求通读旧实验历史。

本轮固定 12 个新 source groups，每组一条完整 6–10 秒片段；同一参考图、同一候选 WAV，分别驱动 Wav2Lip 与 Ditto。结合9月10日最新审计，安排四个共享产物的问题，按门槛分阶段执行：

| 路线 | 问题 | 下游阅读 |
|---|---|---|
| A | 换回原自然音轨后，候选能否改善同步；收益是否随生成器变化？ | [A spec](changes/probe-fresh-source-cross-generator/specs/fresh-source-replacement/spec.md) |
| B | 候选嘴部动态是否更接近真实视频，且得到盲评支持？ | [B spec](changes/probe-fresh-source-cross-generator/specs/fresh-source-visual/spec.md) |
| C / 优先 | 音频延迟能否传递到嘴部？另一生成器是否比Wav2Lip更敏感？ | [C spec](changes/probe-fresh-source-cross-generator/specs/fresh-source-timing-transfer/spec.md) |
| D / 第二阶段 | 自克隆TTS经音素对齐后，是否比natural和仅解码重建都更好？ | [D spec](changes/probe-fresh-source-cross-generator/specs/fresh-source-tts-increment/spec.md) |

第一阶段的C音频固定为冻结 WavLM layer 6 → prematched HiFi-GAN直接重合成，同时作为D的解码处理对照。旧Wav2Lip replacement ΔSync-C=−0.231，不能把它当正向候选依据。D增加唯一M音频：自克隆TTS经MFA-linear映射到自然音素时钟，再走相同解码器。M−N回答有无绝对收益，M−C回答TTS路径有无超出解码处理的增量；两者分开判定。

最新证据决定了优先级：phone-core −0.013、参考交互+0.004均未建立收益，暂不继续缩放/参考帧网格；视觉旧轮的两种反转控制已通过，动态优势仍未建立，因此B补新队列和人工验证；9月5日时间传递诊断虽真实/生成域评分敏感，生成响应仍0/22且结论UNRESOLVED，因此C用新来源、第二生成器、无插值的整帧延迟和视觉轨迹复核。

三条边界：

1. “新”按原视频/source group 去重，排除历史 train/fit/dev/validation/test 组；不解除旧封存。
2. 原自然 WAV 是唯一科学评分音轨；冻结时间支持和 natural anchor，保留真正的重复 forward。
3. 视觉路线先验证真实轨迹的时间敏感性；缺失保留分母；人工意见由真实评审填写。

下游先并行P输入、A生成评分、B视觉实现；P结束后并行A（含D增量统计）、B、C时间传递分析，最多3个worker加主agent。所有生成由A执行，C只交付固定DELAY驱动请求并读产物。B/C不等A科学结论；D只在工程/测量门槛通过后启动。主agent核验终态，人工评审可单独pending。

第一阶段A+B+C上限128生成视频、124正式评分cells；D通过门槛后再加50视频、48正式评分cells。完整上限178视频、172正式评分cells，另最多8次有原因的失败重试。P最多14次direct、14次TTS、14次对齐重合成及28份MFA对齐。共享同一12组，不视为多个独立确认队列。整轮GPU总墙钟上限2小时，按smoke实测速率决定是否能在上限内进入D；超预算留断点关机，不扩容。

本次执行：P 全量扫描 runs 并记录 18,475 个含有效来源 token 的元数据文件，历史 LRS3 group 86 个；本地原始归档 92 组，其中 6 组未出现在历史元数据，但没有任何组完成 MediaPipe 视觉门禁，因此 fully-screened=0，cohort=`BLOCKED_NEW_SOURCE`。Ditto TRT smoke 首次因 `libcudnn.so.8` 运行时路径失败；一次仅补充已有 cuDNN8/TensorRT 库路径的说明性重试成功，产出 512²、25fps、239 帧，mux 后 16kHz mono PCM 与输入逐样本相等。A/B/C 仅写入上游阻塞终态，D 写入 `DEFERRED_MEASUREMENT`；没有科学评分 cells。候选路径、媒体/转写 SHA、视觉硬门禁、root/shared 和 freeze 绑定逻辑均经独立校验。最终确认GPU无任务后执行了 `/usr/bin/shutdown`，连接关闭；无API token，控制台stopped/停止计费尚未核实。凭据仅走环境变量。

最新三路 `final.json` 均显示工程完成、未建立新收益：

- [phone-core](../runs/wav2lip_phone_core_shrinkage_a_20260910_r3/final.json)
- [参考交互](../runs/wav2lip_reference_conditioning_interaction_b_20260910_r2/final.json)
- [视觉动态](../runs/lrs3_visual_teacher_missingness_20260910_v5/final.json)

历史依据：[replacement 复核](replacement-history-review-20260909.md)、[跨数据集 diagonal](../basic-memory/Experiments/跨数据集%20TFG%20测评（5×50%20multiset）.md)、[direct 旧结果](../runs/lrs3_wavlm_hifigan_direct_20260826/summary.json)。跨数据集 Ditto +0.980 / LeapTalk +1.256 不等于固定自然音轨收益，也不足以概括每一种语言的结果。

新增依据：[时间传递未解决](../runs/lrs3_wav2lip_timing_transfer_20260905_v8/result.md)、[重建基底交互未建立](../runs/wav2lip_reconstruction_base_interaction_20260910_v1/result.md)、[响度增益未过完整门槛](../runs/wav2lip_waveform_gain_20260910_v1/result.md)。这些结果支持切换问题和测量路径，不支持再跑同批小步调参。

执行产物：`runs/fresh_source_cross_generator_20260910/`。`validation_independent.json` 的 `integrity=GO`，科学状态仍为 `BLOCKED_NEW_SOURCE`；A/B/C/D 终态分别见 `run/A/final.json`、`run/B/final.json`、`run/C/final.json`、`run/D/final.json`。Ditto 工程 smoke 证据见 `engineering/ditto_smoke/smoke.json`。人工视觉评审未创建，因为没有通过新来源 cohort 门禁。
