---
title: Codex checkpoint - 2026-10-09T06-25-24Z - Native repeat qualification and formal
  Fit execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t06-25-24-z-native-repeat-qualification-and-formal-fit-execution
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T06:25:24Z'
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

# Codex checkpoint - 2026-10-09T06-25-24Z - Native repeat qualification and formal Fit execution

## Summary

正式Wav2Lip已完成两个六方法Fit样本；MuseTalk同输入重复生成和原音评分完全一致，Ditto两次生成成功后改用官方人脸裁剪评分，继续三模型联合优化。

## Story

用户要探明TTS增强TFG的声学机制，并产出适配多模型的自然音频增强头。增强音频只驱动视频；成片和评分必须使用完整原自然PCM与原时钟。ROOT维护理论、协议与BM，代码/实验由GPT-6 Luna/max执行，失败后继续推进。最新用户确认新增Ubuntu V100用户名wjj，公钥可用；下载、API和网盘使用已授权，凭据不入文件。

旧step120在Wav2Lip改善、MuseTalk近零、Ditto下降；FULL Cbest对RMS等模型仅+0.011149，D0正向改善为−0.033662。因此继续ADR022的Fit-only三模型真实视频联合优化。不同前端对谱形与局部时序的响应是当前机制方向；尚未出现新方法跨模型超过RMS的完整结果。工程资格和部分Fit评分不代替效果结论。

## Working State

权威任务《开发跨 TFG 的自然音频增强头》active，实验《跨 TFG 自然音频增强头的训练与原音换轨验证》implemented；最新BM与HANDOFF保存可恢复运行状态。ADR022：Fit40/20人×identity/rms_fixed/旧快照30/60/90/120×三模型=720唯一物理行；六方法共同FULL支持，说话人等权后模型等权。每模型Cbest对两参考正、D0/Canchor不恶化，平均比RMS至少+0.050；缺行失败不选，完整不合格才用预定违例排序作初值。随后9参数20步SPSA和k5/10/15/20中心，共2160计划行；canonical Torch CPU归约。新Dev20、Confirm80、LatentSync封闭。

ADR023冻结每个query五个真实decoded原生帧/有理PTS的机械支持。原raw matrix与分母保持完整，名义尾帧不能进入共同FULL。ADR024实测当前进程资源容量，不把临时SSH cgroup路径当稳定身份，固定5GiB reserve。SSD只剩约57MB高于reserve，正式packet与媒体走SHM、有界六方法context块；归档全byte读回后才清理已验证临时媒体。

执行机✅ wjj@192.168.50.2:22，repo `/home/wjj/tts-audio/tts-exp`，V100-SXM2-16GB UUID GPU-d12b3309-c299-da83-064d-a3cd619bd560。W2L/SyncNet runtime `/home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python`实际symlink链有效，sys.prefix保留隔离音频overlay；先前“target不可用”的解释已纠正。shared Muse/Ditto runtime `/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01`为Torch2.5.1cu121/NumPy1.26.4/ORT CPU。资产和source实际hashlocked部署已完成，旧模型dirty源码保留。

正式W2L packet08在 `.codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/project_payload_attempt05/`，230files/55,302,272B。ROOT逐文件/树/自封印/40原音PCM/160候选波形/240行科学身份QC实际PASS，receipt `independent_packet_qc_attempt01.json` raw f88b4c2a0c9ee288685e79ee2b9acd7db35fe8b3eecf7fc3390b9ee79b2c4cef。remote stage `/dev/shm/fit40_w2l240_stage_attempt05/project`实际230file全SHA和原子rename PASS，stage receipt self65424c6c3b3ba5573f8d9c3c162745046e2fdd3c991df6dd580f2ab8a0e5d4a8。共同science hash ae96def4c16fecabc6e49b2aeabea0a76a0308ea159fa6c7688f7908c62c7180不变；各模型runtime/source/assets另绑。

W2L实际block0（BAC009S0052W0121，原N62639/native95帧）6/6PASS、0fail；block1（BAC009S0052W0495）6/6PASS、0fail，agent回报manifest raw5af08f747d3b8b256a26d228fb4a6ec2381986f1056b5095e923a9ab29225cab。只完成12/240，不归约效果。block0本地71file/3,835,631B读回 `block0_readback_attempt01/independent_readback_receipt_attempt01.json` raw92b64806c498ef2aa1738ad3022453706f745d1a141cdb32b561862ec0d94670，agent remote/local71SHA PASS；ROOT即将实际打开NPZ、clock、支持mask，不宣称已独立完成该检查。block2已真实RUNNING；PID/session是fragile观察，按六行边界交GPU给Ditto评分。

Muse profile固定 `musetalk_native_whisper_unet_fp16_fixed_portrait_haar_expanded_box_official_face_parser_mmpose_free`，九资产和108源文件全SHA部署PASS。row01 144native帧，原自然F32 N92225；score01/02分别guard与三值unpack失败，score03官方矩阵已成但终metadata失败，score04实际PASS_ENGINEERING_ROW_ONLY。raw139×31/Cbest4.204721451/lag−3；原PCM全部保留。25file完整小媒体读回 `.codex/scratch/v100_musetalk_gpu_smoke_attempt04/actual_readback_attempt01/`，agent QC rawc79084e45e8f1bb12558409514d23414fea3ace4d4df505ff46a4f874b34100a。

Muse row02 `muse_v100_natural1_identity_repeat_20261009_02`实际render和官方原音score terminal PASS；agent读回报告 MP4、official2-array NPZ、七数组逐byte与row01一致，139/139 decoded支持。22file QC `.codex/scratch/v100_musetalk_gpu_smoke_attempt05/actual_readback_attempt01/identity_repeat_readback_qc_attempt01.json` raw89699ecf0f9b30c2365682a98c7b0090baccc0993d75df4333d13b65f5690326。ROOT待实际数组/支持复核；Signal已释放GPU，在同40原音/160候选波形/6methods科学绑定上准备Muse240薄adapter，不变更profile/frontend数学。

Ditto12PyTorch权重/完整CPU StreamSDK import/Cython实际PASS。controller01两次在模型前缺importpath失败；controller02修checkout PYTHONPATH后两次实际render PASS，145帧25fps、同silent MP4 SHA9e8e0bc13207b2fceaeae9ffd0a536cf391c9d05a238e20d82f83e08914e4563。之后direct评分把512×512整图送入SyncNet，tensor(20×51200 and512×512)失败，0NPZ/0scoreJSON。保留两次成功视频，禁止为裁剪评分问题rerender。

新score-only wrapper `.codex/scratch/v100_ditto_identity_repeat_score_only_attempt03/score_ditto_attempt02_with_stagea_core06_attempt03.py` 40,893B/SHA4d668a7d2300ae45b2e81edbf5e77dfebe7e6c6ba24cde3a75cc97ec0fda5736，ROOT完整源码和exactSHA已读，并检查pinnedcore06 helpers。复用官方 `_run_pipeline`人脸tracking/crop→`_make_primary_input`自然PCM16源时钟→`_score_direct`官方evaluator；保留完整F32成片，实际矩阵确定query数量，receipt与NPZ/PCM/视频严格读回，工程repeat不准入。Dev已获授权create-only stage并在下一W2L块边界执行，不需额外ROOT许可；成功后立即准备Ditto240同science薄adapter。

selector06 `.scripts`实际路径为 `scripts/experiments/cross_tfg_audio_head_20261003/fit_only_spsa_selection_attempt06.py`，SHA dda610b985026be7f9328c479358e96c983a582c511964e8c6e72a6d398692e2。typed staticportrait允许sourcevideo SHA None，但真实face/naturalPCM/mediakind需跨模型跨方法全局一致；video仍要求真实SHA。新增whole-model漂移拒绝，不改数学/门槛/SPSA。ROOT实际WSL marker CUDA_VISIBLE_DEVICES= unittest 24/24PASS/0skip，日志 `.codex/scratch/root_selector06_cpu_tests_attempt01.log`；05另22/22PASS。未运行真实720selector或任何新SPSA更新。

旧AutoDL唯一DirectD0启动后的终局仍未知，两请求0response，私有PTY自动审批拒绝保持；不绕过、不重放launch。V100是当前可持续路线。Git 5688784dd5c6c1272074f0c4b6c5ff79f117c907 local未push，原五tracked dirty保留，实验代码untracked；BM、HANDOFF、scratch、SHM/PID/runtime均machine-local或fragile，须准确归档；没有commit/push/PR。

## Changed Files

- `scripts/experiments/cross_tfg_audio_head_20261003/fit_only_spsa_selection_attempt06.py`与对应tests：staticportrait支持及跨模型原源身份校验，旧attempt保留。
- `.codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/`：完整科学packet、实际stage与首两个正式块、block0读回。
- `.codex/scratch/v100_musetalk_gpu_smoke_attempt04/`、`attempt05/`：首条评分、重复工程资格及实际小媒体/NPZ证据。
- `.codex/scratch/v100_ditto_identity_repeat_run_attempt02/`、`v100_ditto_identity_repeat_score_only_attempt03/`：两次原生生成及评分裁剪修复；后者刚reviewed未执行。
- BM Task/Experiment/HANDOFF已同步block0、Muse第二行STARTED、Ditto裁剪失败、selector06；此checkpoint补充Muse第二行terminal与W2L block1完成。

## Verification

ROOT实际运行230file packetQC、stage receipt读回、selector05 22tests/06 24tests全部PASS，完整审查score-only最新source和helper契约。agents实际执行W2L两块12rows、Muse两次同输入生成+原音评分、Ditto两次生成并记录失败原因。当前ROOT尚待两模型actual NPZ/decodedmask独立复核，没有三模型增强结论。Startup Router/实验指令/Coding Session/Task已重新完整读取；same-chat coding/codex/legacy分页检索并核对前checkpoint exact身份；write_note(overwrite=False,output_format=json)经CLI源码确认，要求created并直接读回host元数据。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；branch PR实际[]，当前SHA local未push，不构造commitURL。
- ADR022 `tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement`、ADR023 `tts-exp/docs/adr/023-decoded-video-window-support-for-native-syncnet`、ADR024 `tts-exp/docs/adr/024-live-process-resource-capacity`。
- `tts-exp/docs/reference/server-access`：无凭据的V100连接和运行位置。
- Task与实验保存活跃路线；具体source/arrays/receipts用上述路径恢复。

## Observations

- [result] W2L正式Fit首两个context各6/6PASS，共12/240；Muse两个同输入实际视频/官方原音矩阵agent读回逐byte一致，均工程资格。
- [result] Ditto两次原生render145帧成功且视频相同，评分整图尺寸错误0score；官方裁剪score-only源码已ROOT完整review，继续执行。
- [verification] selector06 ROOT实际24/24PASS/0skip；230file完整payloadQC与remote stage实际PASS。
- [decision] 保持ADR022联合门槛与完整分母、ADR023真实decoded支持、ADR024 live容量；评分裁剪修复复用已成功的视频。
- [blocker] Ditto官方裁剪repeat评分、三模型完整720及新SPSA尚未完成；旧AutoDL终局未知但V100持续运行。
- [next_step] 完成Ditto官方裁剪原音评分与ROOT原矩阵支持复核，持续执行三模型40样本六方法完整720行并启动ADR022联合优化。

## Relations

- continues [[Codex checkpoint - 2026-10-09T05-47-53Z - V100 native execution and shared Fit packet]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[023-decoded-video-window-support-for-native-syncnet]]
- relates_to [[024-live-process-resource-capacity]]