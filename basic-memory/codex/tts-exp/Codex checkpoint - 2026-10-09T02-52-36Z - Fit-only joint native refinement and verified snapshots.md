---
title: Codex checkpoint - 2026-10-09T02-52-36Z - Fit-only joint native refinement
  and verified snapshots
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t02-52-36-z-fit-only-joint-native-refinement-and-verified-snapshots
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T02:52:36Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-09T02-52-36Z - Fit-only joint native refinement and verified snapshots

## Summary

确认旧 step120 与 DirectD0 都只通过 Wav2Lip 训练，冻结 Fit 内三模型实际分数优化方案；四个历史快照已真实提取，联合选择/SPSA代码通过 ROOT 的12项合成测试，尚无新三模型实验成绩。

## Story

原问题是解释 TTS 为什么改变 TFG 同步分数，用户随后要求开发可适配多个 TFG 的自然音频增强头：增强波形只用于生成视频，成片必须换回完整原自然音及原时钟。用户要求失败后继续迭代，ROOT维护理论、协议、BM及私有远程调度，实验代码交给 GPT-6 Luna/max。最新提问是增强头是否已超过 RMS；已完成候选只在 Wav2Lip 上有明确点估计收益，MuseTalk近零、Ditto下降，尚未达到跨模型目标。

这一轮从真实训练代码追溯参与优化的模型。step120 和 DirectD0 都只冻结 Wav2Lip 生成器/SyncNet 权重并对输入增强头回传梯度；MuseTalk和Ditto仅参与后续评估，特征探针的 inference/ONNX 路径也不是训练损失。这个结构能解释为什么单模型收益未自然转移，但仍是机制推断，不能据源审查声称跨模型训练已经改善结果。

ADR022冻结下一轮只使用 Fit40 的设计：先比较30/60/90/120四个实际训练快照，再用同一组9参数通过三模型的原生生成、原音评分联合优化。旧W2L代理和真实编码视频的官方评分严格分开。新Dev20和Confirm继续封闭，完整分母和失败保留。快照提取、选择与优化核心已有可执行代码，下一步需要实际三模型Fit评分，不以合成测试或方案代替科学结果。

## Working State

持久状态在 Task《开发跨 TFG 的自然音频增强头》（active）、Experiment《跨 TFG 自然音频增强头的训练与原音换轨验证》（implemented）与新ADR《022-fit-only-cross-tfg-native-score-refinement》。Task/Experiment和HANDOFF当前保留上一轮正确比较，本轮新增ADR与实现由本checkpoint保存；它们尚未同步本轮全部工程产物。

已完成候选180行及对照360行的 FULL Cbest candidate−RMS 等模型均值+0.011149：Wav2Lip+0.068454、MuseTalk+0.000900、Ditto−0.035908。D0距离越低越好，正确 reference−candidate 改善为−0.033662（逐模型+0.030824、−0.028086、−0.103724）。之前把raw D0正值解释成有利已明确更正；冻结政策/analyzer原本方向正确。180组三臂query clock/mask/lag/support身份在ledgers中一致；本地尚无完整NPZ body/媒体联合独立复算。GUARD20完整端点171适格/9不适格，9组均来自BAC009S0022W0123的三模型×三seed，保留完整分母，不能用子集救活端点。

ADR022方案：Fit40/20说话人，四候选+identity/RMS，三模型及固定seed20261003，720个唯一生成评分行；同一context按全部六方法实际valid query keys取共同FULL支持，不预设N。每模型Cbest对两参考为正、D0/Canchor不恶化，等模型Cbest比RMS至少+0.050才可选。完整但未合格的候选只可按冻结违例函数确定后续优化初值；缺失/失败的完整计划行会阻断选择和初值。SPSA20更新，四说话人×两片段批次；每更新±两提案，官方原音NPZ共同支持评分；缺失/不匹配返回NO_UPDATE。固定k=5/10/15/20的完整Fit中心才参与最终选择，批次探针不能准入。完整预期2160个唯一物理评分行。具体超参数/目标/排序在ADR中；这些是前瞻设计，没有实际收敛或效果结论。

官方Stage-A端点的distance_by_query_lag是finite float32[N,31]，lag −15..15；query_frame_index、约分有理timestamp、track_frame_index、valid_query_mask都必须来自真实producer。旧W2L Fit代理[320,31,27]没有这些官方身份字段，只作诊断，不能当Muse/Ditto或三模型实际视频证据。新核心以NumPy float32归约，不声称与PyTorch归约bitwise相同；后续需实际数值容差/交叉运行检查。

四个快照实际已从legacy `fit40_native_transition_head_v3_attempt11`完整120步ledger提取，index status为EXTRACTED_UNSELECTED_NATIVE_ENDPOINT_UNBOUND。保存于machine-local `.codex/scratch/fit40_training_snapshots_attempt01/legacy_native_attempt11/`；snapshot_set_index.json为33141B，raw SHA `4a21bad297b0ec76aa733fc75132f3b2afef40f3a57ecef20ee7a25ab5bd0808`，canonical index SHA `a0d69b201bf5029f8455534ccc72cb2a26daa6c624fa7cb0c68ea1fd0d5bdd56`。ROOT核对实际raw文件；子代理执行完整源/geometry/constructor/module/checkpoint/terminal验证。四份snapshot没有伪标为terminal120，DirectD0家族另有独立profile且尚无终局文件可提取。三后端plan/wrapper还在实现，qualification/runtime/capacity均保持late-bound，不编造未来pins。

远端✅ `root@connect.westd.seetacloud.com:18625`，project `/root/tts-research-system-20261001`。DirectD0唯一actual STARTED_DETACHED receipt仍是supervisor573383/start_ticks334637264，control `cross_tfg_audio_head_20261003/job_control/fit40_d0_gatedc_training_attempt03`；没有启动后snapshot。status04 packet `.codex/scratch/fit40_d0_gatedc_status04_readback_attempt01/request_index_attempt01.json`两请求均无response，勿重放launch。私有PTY57588非空只读请求被host自动审批拒绝（AskForApproval Never）；bridge实际127.0.0.1:12249/PID29868监听不能代表远端任务进度。密码/API仅私有运行时，永不入文件或argv，不读取其他进程环境/内存。只读严格SSH替代因没有已知ED25519 host key而失败，未到认证，不关闭StrictHostKeyChecking。

合法本地执行替代正在调查：Windows/WSL实际识别RTX4060 Laptop，8188MiB显存（探测时约5.3GiB可用），Ubuntu-24.04已有Python3.12.3、NumPy/SciPy/Pillow及ffmpeg6.1.1，但无Torch等模型运行依赖。Windows Miniconda base/myenv也无Torch/NumPy；bundled primaryruntime有NumPy无Torch。未安装依赖、未运行本地TFG。已授权Baidu挂载可查权重/源数据，本地新平台必须重新执行同平台对照与资格检查；不能把服务器资格凭证复用成4060资格。

联合540 CPU包仍是 `.codex/scratch/dev540_joint540_cpu_consumer_attempt02/`，43pins已通过、未派发；旧checkpoint误写的dev540_joint_media_readback_attempt02目录不存在。readback含MediaPipe描述性视觉推理，无TFG/SyncNet重评分。未来独立Dev20 natural/RMS/natural重复ASR60/60成功，20/20归一化转写全相同，candidate20未运行；ADR021保持不变，Confirm80/40新说话人及LatentSync封闭。

Git branch仍codex/cross-tfg-audio-head-20261003，HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`本地未push。当前GitHub canonical repo已只读验证，branch PR返回[]、远端branch ref不存在，不构造本地commit URL。五处原有tracked dirty保持；大量untracked实验/scratch与活动进程都是machine-local状态。无新commit/push/PR。

## Changed Files

- BM `docs/adr/022-fit-only-cross-tfg-native-score-refinement`：前瞻选择/联合优化设计，created并直接读回PASS，未改旧ADR或冻结未来Dev政策。
- `scripts/experiments/cross_tfg_audio_head_20261003/fit_only_spsa_selection_attempt01.py`：CPU-only官方端点聚合、完整支持/身份检查、快照准入、SPSA ask/tell和full-center请求，56803B/SHA `69e596ad9ed6de1e23ca3befa353b3650a0c99b64528465030ff34b3f84fe722`。
- `tests/experiments/test_fit_only_spsa_selection_attempt01.py`：实际18587B/SHA `2e0ee0d208c944fbf7ca1ee1e3166e5172309d8654d703832000cc1ff9aa81de`，子代理口头18724B是长度误报，以ROOT实际为准。
- `.codex/scratch/dev540_joint540_local_export_attempt03/`：离线导出器与7项synthetic tests，保留02旧版；不解码NPZ和媒体，不运行模型。尚无actual join/readback输入，未执行真实导出。
- `.codex/scratch/fit40_cross_tfg_training_scope_audit_attempt03/cross_tfg_fit120_directd0_model_scope_audit_attempt03.json`：训练模型范围源码审查receipt，12853B/raw SHA `39e553e8fda4927294f16f1eadd250b919995230043f898b2b07c2c63839aaff`。

## Verification

ROOT独立执行bundled Python `-B -m unittest discover -s tests/experiments -p test_fit_only_spsa_selection_attempt01.py -v`，12/12 PASS，真实测试覆盖六方法共同支持、完整行失败、严格零值不准入、源/theta/batch不匹配、确定性ask/tell、NO_UPDATE和完整中心身份。它们使用合成官方形状输入，未运行TFG/SyncNet或证明新方法效果。ROOT核对core/test实际字节及SHA。

ROOT此前独立跑离线exporter03的7项synthetic tests PASS，receipt `.codex/scratch/root_exporter03_synthetic_check_attempt01/receipt.json`为517B/raw SHA `fc0af20fd9532a337e33ad4008925074a0cf66cd3738f85f3db062f20254acf8`；source SHA `d1152913d4c6a5e9c124c513aae3822935718e81f2e600c242fb6b30c238c0bb`，tests SHA `ca0c9e254af07be79403d11350bacf0d6f63fb5ca603409f86e24111c900e6b3`。无actual join数据，不能扩大成端到端验证。

ROOT核对scope audit03与实际legacy snapshot index raw SHA，ADR022直接readback确认typed decision/statusopen。相同agent/session的coding/codex及legacy BM查询已检索；coding第1页200条、第2页0条，codex和legacy无结果，最新checkpoint直接读回确认started、agent/session。只读Git/username/hostname/UTC与GitHub repository/PR/ref均重新验证。严格SSHread-only失败于host key检查，无新远端访问/重启、云API或依赖安装。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)，canonical URL由本轮GitHub API确认，本地SHA不链接。
- [Spall SPSA primary paper](https://www.jhuapl.edu/spsa/pdf-spsa/spall_an_overview.pdf)：联合优化无需TFG梯度；每次两个目标测量仍各含多模型/多片段视频，不能称只需两个视频。
- `.codex/scratch/root_record_joint_fit_decision_022_attempt01/actual_readback.json`：冻结设计实际读回。
- `.codex/scratch/fit40_training_snapshots_attempt01/legacy_native_attempt11/snapshot_set_index.json`：真实legacy源快照身份及未绑定官方端点。
- `.codex/scratch/root_exporter03_synthetic_check_attempt01/receipt.json`：已执行工程测试边界。

## Observations

- [result] 旧候选对RMS的FULL Cbest平均+0.011149，Ditto下降；D0正向改善平均−0.033662，仍未跨TFG准入。
- [result] step120及DirectD0的可微训练均仅使用Wav2Lip，MuseTalk/Ditto评估不能当三模型训练信号。
- [result] 四个实际legacy快照提取完成但三模型原生评分端点未绑定；联合选择/SPSA核心ROOT12项合成测试PASS，尚无实际新效果。
- [decision] ADR022冻结Fit-only三模型官方分数优化，保留源时钟/RMS/3dB限制、完整失败分母；未来Dev20与Confirm封闭。
- [blocker] 私有PTY被host自动审批拒绝，DirectD0启动后状态与540联合读回未知；严格SSH因无已知host key失败，本地GPU依赖/资产尚待资格检查。
- [next_step] 完成独立代码复核和实际Fit-only三后端执行绑定，在合格可用执行平台评分真实四快照并按ADR022推进联合优化。

## Relations

- continues [[Codex checkpoint - 2026-10-09T02-15-20Z - Candidate RMS comparison and distance sign]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]