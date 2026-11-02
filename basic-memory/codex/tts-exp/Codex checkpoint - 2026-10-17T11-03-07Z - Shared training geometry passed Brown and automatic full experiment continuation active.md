---
title: Codex checkpoint - 2026-10-17T11-03-07Z - Shared training geometry passed Brown
  and automatic full experiment continuation active
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t11-03-07-z-shared-training-geometry-passed-brown-and-automatic-full-experiment-continuation-active
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 11:03:07.992946+00:00
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-17T11-03-07Z - Shared training geometry passed Brown and automatic full experiment continuation active

## Summary
修复并版本化了跨录制坐标实现，Brown全量v2几何已通过；其余身份预处理和持久交付运行中，当前聊天已安排自动接续到24模型完整训练及27格，但新TFG训练尚未开始。

## Story
用户从May电平推理、训练与评分器三轴先导扩展到四新身份自然素材，同人不同原录制测试，并明确授权全部资格通过后自动开始官方完整训练及完整27格。报告50/51是已完成旧先导，不能当本轮新结果。新spec保持4身份×seed42/2026×三训练增益=24模型、65536rays/FP16、60k基础后累计102k完整epochs，严格同完整初始化与实际optimizer更新预算；三档driver和三档score音频分别控制。

本轮原材料和BFM全部完成独立资格，但ADR038/v1全局BFM→BA相似映射在Brown失败：p95旋转6.675°、平移14.896mm。诊断证实原版BA对每帧相机/自由3D形状的细化不能只当一个全局坐标变换。没有放宽旧门槛或改写失败；新增ADR039并在全量新方法前冻结v2，使用原版训练BA/flow三角化固定训练参考帧2三维点，heldout原NeuralNRT匹配同一训练点，冻结3D点仅拟合heldout相机。最终直接处于原训练BA坐标。相似变换仅作初值；相机恢复控制仍用2°/5mm，landmark5/10px、flow2/5px和条件数1e6保持。方法诊断使用heldout图像资格但没有使用TFG/SyncNet处理效应来挑条件，必须透明报告这一实现适配。

Brown完整train6000/val750/test1500通过v2，最后1200训练内相机恢复p95 .150°/.185mm；全val flow median/p95 1.627/3.113px，全test1.149/1.962px，固定点SHA不变、原六帧flow与heldout重复控制精确。Brown AVE六组整录制×增益缓存与监督合成也通过，已轮到Patrick原版train tracker。十个实际Ubuntu单元测试通过，包括官方Euler符号与真实整录制attention边界/录制隔离。完整数据准入、loader/ray预算及训练控制器仍未闭合，不能宣称已开训。

## Working State
持久权威根F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017。Task《准备并核验四个新身份的 SyncTalk 完整训练数据》已更新为当前v2状态，包含所有receipts、已知限制、活跃进程和未完成工作；优先读该Task、spec、ADR038/039。四人各420秒自然镜像处理视频，原PCM逐样本、42000帧RGB/PTS、增益/split/上下文/重复检查通过，60文件4001168448字节F验证。Patrick源test482平方上采样、Cathy原test4满幅样本限制保留。授权BFM转换通过，无凭据入文件。

Brown/Patrick stage1各8250JPEG/FAN/BiSeNet和F各41251文件逐SHA通过，Brown五个副脸框已审为背景/手、主preds[0]未改，Patrick无flags。Debbie全部关键点无flags且全分割技术通过，F解包SHA进行；Cathy stage1进行。远端易失ROOT=/dev/shm/synctalk-multispeaker-data-20261017，SSH wjj@192.168.50.2；root盘约536MB，SHM16GiB。stage1_worker pid222945 attempt04在Cathy；stage2_v2_worker pid233261 attempt02在Patrick train_tracker；score_reference_worker pid238330先后三人已完成，Cathy进行。以实时JSON/ps为准，不能重复启动。v2 worker每身份结束等待F全归档验证和GT/torso缓存退役后才继续。

F观察器按stage1、train刚性flow缓存、derived包、qualification包、canonical视频分别独立校验及回收。Brown derived包16587成员/1956283487原字节全SHA验证后释放SHM GT/torso1827740276字节，训练加载前必须恢复；新v2/scorer另有39MB qualification包全成员验证。Brown canonical两个SHM视频780156620字节在F新鲜SHA且消费者结束后退役，F原数保留，音频/AVE/JPEG/几何仍在。资格包SHAe6e6259cf25a27bb528b4e5fa999823f02155ff5b609a89c5b56427196c75699。May50旧SHM6.05GB此前在F和百度盘完整独立读回后回收；未触碰CloudDrive配置或未验证原数。F现约10.8GB，后续模型/视频仍需严格管理历史重复/云验证空间。

评分参考source-owned ROI与六个固定10秒片段三评价轴已为Brown/Patrick/Debbie准备；0音频原PCM16精确，±6以float64原PCM数值单位乘gain并DOUBLE WAV读回×32768精确；模型/视觉音频重复forward通过。Brown/Patrick原声六clip offset均0。Patrick第一次Sync环境OpenCV解码与原RGB哈希不一致失败保留；第二次通过原material QA OpenCV4.10运行时的原始RGB管道，所有帧SHA相等。原哈希并非FFmpeg RGB，勿重复早先推测。无按SyncNet结果排除片段。

当前聊天心跳《SyncTalk 四说话人完整训练与27格》id synctalk-27已实际创建ACTIVE，每10分钟接续；用户已获告知。它读取Task/spec/ADR和实际receipts，继续实现未完成runtime、通过所有门再完整训练/生成/评分/QA，状态不变保持安静、关键完成/故障/需输入才通知，全部27格交付后停用而不归档聊天。不要重复创建心跳。当前预处理worker本身不会训练。

## Changed Files
scripts/experiments/synctalk_multispeaker_full_20261017新增v2合同、prepare_geometry_v2.py、训练内/共享参考几何诊断、audio_context.py、build_datasets.py、runtime_common.py、原RGB decoder/score_reference准备与worker、持久归档和精确回收工具。spec/ADR039/Task已保存并回读。源码/tests/docs共56文件当前F execution_sources/full_preparation_attempt03全部字节SHA回读通过；早先execution_sources/geometry_v2_attempt01保留47文件旧版本。新目录及test均untracked，当前共享树还有大量无关dirty，未提交/清理。active Windows archive observer已经import早一版persist_prepared，默认derived路径语义相同；资格观察器用新kind接口，两版本均封存。

audio_context把provider/renderer/utils原get_audio_features的split-local索引映射到完整原录制aud_id，保留aligned N帧供occupancy索引；测试验证开头/末尾真实上下文不填零、不跨录制。build_datasets使用原版float32 Torch转置/bmm写相机变换，官方train不portrait、test才portrait；runtime_common保留-O/65536/16、head+lip设置。dataset/runtime尚未执行loader资格，需先恢复GT/torso缓存；GT恢复工具、lazy0/1逐像素/ray等价、完整ray显存/初始化/更新控制、训练/渲染/最终27格独立评分控制器仍待实现。不得把已写源码当已测试工作。

## Verification
实际Ubuntu10个unittest全通过；全新目录Python源码AST解析与F source snapshot回读通过。固定官方ZIP审计68Python：network/renderer/utils和全部预处理exact；仅4CUDA backend兼容和provider已知3项绑定差异。v1合同SHAf472a5a9ef0bf50f2d85bb535407eb83025efe2eca86be2cf294f4e685851e51；v2SHA37f0714cecd41cca0f531bb062b6029bad2e96fe5bc19bd3ad0827d61f3dd005。原flow float32比较错误保留原数组，六帧fresh NNRT/前2500轨迹/选点独立重验通过。初版诊断Euler符号不对，roundtrip断言抓出，attempt01诊断不是有效资格；v2控制用正确Rx(theta)Ry(phi)Rz(-psi)。Windows首次SciPy缺依赖失败保留。

同chat/legacy检索无后页，上一checkpoint agent/session/started已全文验证；Git本地root/branch/SHA直接核验，GitHub只读再次确认本分支无PR。当前SHA此前GitHub返回422不能解析，仍为本地未推送，不构造commit链接。准备完成不等于整个实验完成。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)
- 本地未推送SHA：`2b71639c3b555293375d248350999377168d515d`
- scripts/experiments/synctalk_multispeaker_full_20261017/protocol.json
- F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/qualification_state/
- F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/execution_sources/full_preparation_attempt03/source_manifest.json

## Observations
- [result] Brown全量v2共坐标、监督、AVE与评分参考通过；四人原材料通过，余三人完整资格进行，新生成器训练数仍0。
- [decision] 保留v1失败，版本化共享训练参考点方法，训练原版BA和误差阈值数量不变；原数据、预算与27格合同不变。
- [decision] 仅验证归档后退役SHM缓存，监督须恢复才能加载，原RGB解码按封存哈希权威一致。
- [blocker] 余三人全量资格与持久门、GT恢复/loader等价、官方全ray预算/实际更新验收以及训练渲染评分控制器未完成。
- [next_step] 从真实worker receipts继续完成余三身份预处理及完整训练准入，再由当前聊天自动接续完成24模型和27格交付。

## Relations
- continues [[Codex checkpoint - 2026-10-17T10-34-05Z - Four speaker preprocessing qualified partly and coordinate gate failed]]
- relates_to [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[039-synctalk-shared-training-flow-geometry]]