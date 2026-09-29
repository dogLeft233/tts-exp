## Purpose

在 Wav2Lip 上先建立有效生成控制，再检验固定候选音频的 natural-audio replacement 收益；成功后仅进入跨模型验证设计。

## ADDED Requirements

### Requirement: Freeze a seen fit-only cohort and inherited contracts

实验 SHALL 使用 `runs/lrs3_natural_to_tts_bridge_confirmation_20260904/00_protocol/cohort.json`，文件 SHA-256 `b55403fdc508d359f6ee0063c2728cba2362ea583bc80cf8208fb3f16f4db33b`，保持 22 records / 22 source groups 和原顺序，ordered-ID hash `5999f6bae8ba6b17c2e316b5f1e1f9e6d73b402a34cc8e4cfe482bcc714f5b72`。分类 SHALL 为 `seen_fit_pilot`，不另选样本。

SHALL 绑定 timing-transfer v8 的 `final.json`（文件 SHA-256 `4bc4dcab90adec07ec6f3e7df34914fc5211fd28aba54f5ffb6a7678969f763f`）与 bridge confirmation 的 `04_final/final.json`（`df0ca9767e70ccc384c86c1da23243c53fa609be12abd1dda20f9075b6732c6e`），沿父 manifest 校验原视频、N、W、BRIDGE_075、MFA-linear target、评分轨迹与模型身份。历史文件 hash 与 JSON 内部自哈希分开验证。

下列仓库文档为规范性继承，prepare SHALL 读取全文并冻结文件 hash；本 spec 的显式变更优先，其余公式/预处理不重设计：

- `openspec/changes/diagnose-lrs3-wav2lip-timing-transfer/specs/lrs3-wav2lip-timing-transfer-diagnostic/spec.md`：W 复建、时间轴、评分轨迹、距离矩阵、共同窗口、PLUS/MINUS、A/B/C/O 与峰判定。
- `openspec/changes/confirm-lrs3-natural-to-tts-bridge/specs/lrs3-natural-to-tts-bridge-confirmation/spec.md`：BRIDGE_075 构造、movement、replacement 定义。

SHALL 在任何本轮评分前冻结全部源码/环境/权重/spec hashes、音频、ROI、两个阶段阈值与矩阵；缺失或身份不匹配为 BLOCKED，不采用另一个历史 run。

#### Scenario: Historical control failed

- **WHEN** 父科学终态为 CONTROL_FAILED 或 GENERATED_RESPONSE_UNRESOLVED，且资产身份有效
- **THEN** 允许本轮新 ROI 实验，但保留历史结论；父失败本身不是本轮资产审计失败

### Requirement: Change only the generation face region

SHALL 使用既有 `wav2lip_gan.pth`，SHA-256 `ca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8`。从每条原视频完整帧序列一次性提取官方 Wav2Lip face_detection 的逐帧框，固定 `flip_input=False`、`pads=[0,10,0,0]`、`nosmooth=True`、resize_factor=1、不旋转、不额外裁画布、检测 batch=1。

SHALL 保存检测器代码/权重 hash、未加 padding 框和最终 `(top,bottom,left,right)` 框；框裁剪至原画布边界，必须非空。全部音频臂按绝对帧号使用同一缓存框。生成适配器保持官方 96×96 resize、下半脸 mask、mel 分块、checkpoint forward 和原位置回贴，batch=4。不能把 `--box` 当作逐帧框接口；当前 CLI 不支持缓存框输入时，只新增实验内薄适配器，并测试其预处理/回贴与官方逻辑一致。

SHALL 在评分前保存全部记录首/中/尾帧的框叠加图并完成几何审核：框覆盖说话者面部与口部、不是整帧 fallback；记录审核者、结论和图像 hash。无法完成审核为 BLOCKED。检测缺失、错误对象或审核失败时不插值补框、不手调框、不剔除记录、不换检测器。几何审核仅检查输入，不按生成效果或评分筛选。

评分 SHALL 独立复用 tail_v2 原视频轨迹和 crop_scale=0.40、224×224 裁图；禁止重新检测生成视频。N/W/bridge 输出都回贴原画布，所有帧来自原视频前缀，不循环/补帧，不用静态图替代原视频。

#### Scenario: A face cannot be located reliably

- **WHEN** 任一源帧缺框、框越界/为空或几何审核不通过
- **THEN** 写明 sample/frame 和证据，终态 BLOCKED；不退回 constant_full_frame_fallback

### Requirement: Preserve fixed waveform candidates and media identity

SHALL 使用 16kHz mono PCM16、同长度的 N、N_REPEAT、W、BRIDGE_075；N_REPEAT 解码 PCM 与 N 完全一致。W 读取已绑定 calibration v5 的 LOCAL_WARP_120，并按继承公式独立复建校验。BRIDGE_075 读取已绑定 confirmation 的音频，并复建验证 natural phase、alpha=0.75、固定 STFT/RMS/peak 处理。复建只审计，不静默替换旧 PCM。

SHALL 在 prepare 完成 BRIDGE_075 的 movement 诊断，即使 B 最后不执行；禁止选 alpha、新 TTS、重跑 MFA/DTW 或依据 mel/SyncNet 分数修候选。候选导出契约为 waveform + sample_rate + length + hash，Wav2Lip mel 仅为诊断字段。

所有 mux SHALL copy 同一视频流并使用原指定 PCM，分别核对视频流身份和 decoded PCM hash，不使用生成 MP4 的 AAC 音轨。保留完整 PCM 和视频；禁止 `-shortest`、全局 offset 对齐、截尾隐藏失败。25fps、PTS/真实视频尾差沿用 timing-transfer 契约。

prepare SHALL 从绑定 mel 分块逻辑及音频长度预测各臂帧数，要求 N/N_REPEAT/W/bridge 相等且不超过原视频；生成后核对实际解码数，不照搬旧生成视频帧数。共同 Q 使用该新帧数；控制与 bridge 全程使用同一共同有效行集合和 PLUS/MINUS，不按分数择窗。

#### Scenario: Audio identity or generated frame count differs

- **WHEN** 换回 N 后 PCM 不再相同、生成帧数不符合预测或出现视频循环
- **THEN** 工程 BLOCKED，不缩短音频或统计子集使其通过

### Requirement: Pass control stage before rendering the bridge

A 阶段 SHALL 新生成 G_N、G_NR（独立 N_REPEAT 进程/目录）、G_W，共 66 视频。R 为原真实视频。每条执行 `R/N, R/W, G_N/N, G_N/W, G_W/N, G_W/W, G_NR/N` 七个主 cell，并在独立评分进程/目录重新评分 `R/N, G_N/N`，合计 154 主 cell + 44 重复 cell = 198 次评分。不得复制历史分数或重复矩阵。

SHALL 固定官方 SyncNet V2 权重 hash `961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442`、batch=20、vshift=15；导出完整未舍入距离矩阵并重建官方 C/D/offset，数值误差 ≤0.001。跨 cell 的 C/D/offset SHALL 来自共同有效行；原生官方全局指标另列。

SHALL 按以下顺序判定 A；失败时仍报告所有可计算项，但禁止进入 B：

1. 继承的评分重复性 22/22；R/N、G_N/N 两段峰可解释性各 ≥20/22。
2. G_NR/N 相对 G_N/N 的 C 差与正向 D 差（D_N−D_NR），各 95% CI 完全落在开区间 (-0.10,0.10)，共同窗口 offset 差 ≤1 帧的记录 ≥20/22。
3. 继承 A/B/C/O 的双段预测、峰间距 >0.010、误差 ≤1 帧和固定分母，每项 ≥18/22。
4. 控制 own-audio 非劣性：C(G_W/W)−C(G_N/N)、D(G_N/N)−D(G_W/W) 的 CI 下界均 >−0.10，offset 差 ≤1 帧 ≥20/22。
5. 控制 replacement 损伤：C(G_W/W)−C(G_W/N)、D(G_W/N)−D(G_W/W) 的 CI 下界均 >0.10，且两项都 >0 的记录 ≥18/22。

所有 CI SHALL 为 source-group bootstrap：排序 22 个组、有放回抽 22 组、10,000 draws、NumPy default_rng/PCG64 seed=20260905、每个指标重置 seed、线性分位数 2.5/97.5。分母固定 22，不剔除不清晰峰。上述门槛是本 pilot 的预注册决策规则，不宣称统计功效充分。

#### Scenario: Local response passes but own-audio validity fails

- **WHEN** A/B/C/O 达标但第 4 或第 5 项失败
- **THEN** 仍输出 CONTROL_FAILED；局部时序响应不代替 replacement 控制有效性，不生成 bridge 视频

### Requirement: Require natural-audio replacement improvement

仅当 A 完整通过且控制证据经离线 validator 校验，B 阶段 SHALL 新生成 G_B=G_BRIDGE_075，共 22 视频；每条新增 `G_B/N, G_B/BRIDGE_075` 两个评分 cell，共 44。全程最多 88 新生成视频、242 次评分，不重新挑 N baseline。

SHALL 以换回未改动 N 的端点判定，定义：

```text
gain_C = C(G_B/N) - C(G_N/N)
gain_D = D(G_N/N) - D(G_B/N)
```

B 通过 SHALL 同时满足：movement progress ≥0.15 的记录 ≥20/22 且其均值 CI 下界 >0.15；gain_C/gain_D 的 CI 下界均 >−0.10；G_B/N 与 G_N/N 的共同窗口 offset 差 ≤1 帧 ≥20/22；主收益 gain_C 的 CI 下界严格 >0。使用 A 的相同 bootstrap 约定，完整精度判定、Sync-C 展示三位小数。

自身配对 G_B/BRIDGE_075 及其与 G_B/N 的差 SHALL 报告，但仅作描述，不替代上述主端点。SHALL 保存按原 cohort 顺序前 3 条 N/bridge 配 N 的可播放对照，供观察可见伪影；没有人工审核时明确 `visual_quality=UNREVIEWED`，数值 pilot 成功不得声称主观画质成功。

#### Scenario: Candidate only preserves baseline quality

- **WHEN** movement 和非劣性通过，但 gain_C CI 下界 ≤0
- **THEN** 输出 REPLACEMENT_NOT_ESTABLISHED；“没有明显变差”不叫 replacement 收益

### Requirement: Deliver one bounded pilot and transfer-ready artifacts

SHALL 使用新 `runs/wav2lip_face_roi_replacement_<run_id>/`，交付 `protocol.json`、输入审计、ROI/审核记录、audio/video/score manifests、距离矩阵/日志、逐条统计、`control.json`、`final.json`、`result.md`、`validation.json`。final SHALL 绑定全部证据 hashes 和期望/实际数量。

终态 SHALL 唯一：工程缺陷为 `engineering_decision=BLOCKED, scientific_decision=null`；工程完整时科学终态为 `CONTROL_FAILED`、`REPLACEMENT_NOT_ESTABLISHED` 或 `WAV2LIP_REPLACEMENT_PILOT_PASS`。A 失败的完整终态只要求 66 新视频/198 次评分及 B 未执行证据；成功走完两阶段才要求 88/242。

离线 validator SHALL 独立从矩阵/媒体/ROI 复核共同支持、PCM、配对、计数、bootstrap 与终态，不调用 producer 的 gate/统计函数或相信 pass 字段。SHALL 测试缺 cell、交换音轨、框坐标顺序错误、整帧 fallback、分母缩小、门槛边界、控制未过却进入 B、final 篡改及失败终态的合法数量。

只允许身份完全一致的完整 cell resume；部分产物/已终态 run 不覆盖，工程修复另起 run 并引用旧失败。科学失败不调框、改幅度/alpha/门槛、搜种子或重新选 run。所有训练、其他 TFG、sealed 数据访问、租卡和旧 run 修改均不在本实验范围。

SHALL 始终 `training_authorized=false`、`reference_conditioned_audio_head_spec_eligible=false`、`generalization_established=false`。仅 pilot PASS 且 validator valid 时 `cross_model_spec_eligible=true`：交付同一批 N/BRIDGE_075 waveform hashes、配对 ID 和构造配置，建议另立至少一个不同架构 TFG 的固定候选验证 spec；本轮不执行迁移，不按新模型重新优化候选。未来生成头训练还需要独立数据与跨模型证据。

执行结束后 SHALL 按 Startup Router 和实验指令，在 BM `tts-exp` 搜索后新建/更新同一份实验笔记，记录真实终态、数字、局限和报告指针；不覆盖历史失败。无实验结果时不得写“已通过”。

#### Scenario: Wav2Lip pilot passes

- **WHEN** 两阶段全部通过且独立验证 valid
- **THEN** 报告「已见 22 条数据上的 Wav2Lip replacement pilot 成功」，交付可移植 waveform 包并停止；只建议跨模型验证 spec，不宣称通用生成头已可行
