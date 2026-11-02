---
title: Codex checkpoint - 2026-10-09T07-07-36Z - Three native models qualified and
  shared Fit execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t07-07-36-z-three-native-models-qualified-and-shared-fit-execution
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T07:07:36Z'
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

# Codex checkpoint - 2026-10-09T07-07-36Z - Three native models qualified and shared Fit execution

## Summary

三种TFG的原生重复运行、完整原音换轨与真实帧支持均已实际核验；Wav2Lip正式Fit完成48/240行，正在接齐MuseTalk和Ditto的同批六方法运行。

## Story

用户要求解释TTS增强TFG的局部声学机制，并进一步产出适配多个TFG的自然音频增强头。增强PCM只驱动生成，交付和SyncNet评分使用完整原自然PCM及源时钟。ROOT维护理论、协议和本地BM；代码与实验由GPT-6 Luna/max执行，失败后继续。用户新确认Ubuntu V100用户名wjj，公钥可直接登录，历史数据、网盘、下载和API使用已授权。凭据不入文件。

旧step120仅经Wav2Lip训练，FULL Cbest对RMS等模型+0.011149，D0正向改善−0.033662；跨模型目标尚未达到。现执行ADR022：同40条Fit原音/20人、四legacy快照30/60/90/120加identity/rms_fixed，在三种实际生成器上取得720唯一物理评分行，再用实际原音分数做9参数20步SPSA和完整中心，共2160计划行。工程资格或部分Fit行不作为新方法效果结论。

## Working State

权威Task《开发跨 TFG 的自然音频增强头》active、实验《跨 TFG 自然音频增强头的训练与原音换轨验证》implemented。ADR022保持六方法共同FULL、说话人等权后模型等权，每模型Cbest对两参考正、D0/Canchor不恶化且平均对RMS≥+0.050。缺行/失败不选；完整但不合格才按预定违例排序初始化。新Dev20、Confirm80、LatentSync封闭。ADR023按五个真实decoded帧/有理PTS机械判定支持，不删raw分母。ADR024检查live RAM/cgroup/SHM，固定5GiB reserve。

✅执行机 `wjj@192.168.50.2:22`，repo `/home/wjj/tts-audio/tts-exp`，V100-SXM2-16GB UUID GPU-d12b3309-c299-da83-064d-a3cd619bd560。W2L/SyncNet `/home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python`有效symlink链与隔离prefix已验证；共享Muse/Ditto `/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01`为Torch2.5.1cu121/NumPy1.26.4/ORT CPU。旧模型dirty源码保持。SSD只剩约57MB高于reserve，packet/媒体走SHM六方法块，B/F归档；本机网盘文件存在不等于云上传确认。

W2L完整230file/55,302,272B packet及remote全SHA stage实际PASS，位于 `.codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt08/project_payload_attempt05/` 和 `/dev/shm/fit40_w2l240_stage_attempt05/project`。共享science SHA ae96def4c16fecabc6e49b2aeabea0a76a0308ea159fa6c7688f7908c62c7180；plan/binding各模型runtime单独绑定。实际blocks0–7每块6/6PASS、0failure，共48/240。block7 manifest raw845a3505ca991954bed1351baae938eb4c19b06fb186b1b21e8f33ca08429959。ASR继续后续块；block8尚待live资源复测与实际启动回执。GPU按六行边界交接给已reviewed/staged就绪的Muse/Ditto，不因CPU包装准备闲置。

ROOT实际read-only QC `.codex/scratch/root_native_actual_arrays_readback_qc_attempt02.json` raw929e4e0dcdff06984411eeb968081b60615b6bb31b2ba71e9f3d60f1f68ecf53/self391fe1a5d2135ac5543841f6320361ec79d03c60afd0c813a39c2f2c79939356：Muse47文件与W2L block0的83body全SHA、actual raw/七数组dtype/shape/finite/datahash/queryclock、机械五帧支持、完整F32源及实际媒体解码均PASS。W2L block0 native95帧/raw92/共同FULL91，原N62639；本机ffmpeg重建PCM16 source-slice逐byte一致。

Muse固定profile `musetalk_native_whisper_unet_fp16_fixed_portrait_haar_expanded_box_official_face_parser_mmpose_free` 两次实际生成和官方原音评分PASS，144 native帧/25fps、原F32 N92225、139×31矩阵/139真实五帧query。ROOT实际全文MP4、矩阵和七数组逐byte相同，两个F32 mux解码与完整源一致。权威本地 `.codex/scratch/v100_musetalk_gpu_smoke_attempt04/actual_readback_attempt01/` 和 `attempt05/actual_readback_attempt01/`，repeat QC实际文件 `identity_repeat_readback_qc_attempt01.json`。score01 symlink guard、02 unpack、03末metadata失败保留，不再重复工程render。

Ditto两次native生成145帧成功，整512图评分shape错误后复用已有视频，官方tracking/224crop→自然PCM16时钟→official score-only04实际terminal PASS_ENGINEERING_ONLY_EXACT_REPEAT，非追加render。03缺9源码closure的preGPU失败保留；exactcreate-only安装完成。04 source40893B/SHAe766a4c7d0ba0b98d16847f4eba5ff59ef501a468774dc0436972f00387c1f70。完整readback `.codex/scratch/v100_ditto_identity_repeat_score_only_attempt04/readback_attempt01/ditto_v100_identity_repeat_score_only_attempt04/`，runreceipt rawd2a869edf0faa9a95bc1b286bad7aad2f418f7b3bad7e7972f2115f5d79c9441。

ROOT实际WSL独立QC `.codex/scratch/root_ditto_actual_repeat_readback_qc_attempt01.json` raw11a5e054fcb0893a494a8d75d098947b5e87e8fb9111f1db6dd10b3fd7350d50/self533d16c8ee10bfed9e58e9d0aac7de546782778ceca6630c754d78044ff0cd71：两个139×31原矩阵/七数组和video elementary完全一致，139/139五真实帧支持，实际ffprobe145帧精确25fps rationalPTS，两个full F32 mux N92225 byteequal源，自然PCM16完整转换和评分mux音频byteequal。无效果归约/选择，未加载pickle。

当前三个模型资格已完成。Signal正在修复Muse240 builder/overlay的profile dict/string、aggregate asset/UNet SHA、源closure、有效SyncNet symlink及evidence路径等字段错误；draft `.codex/scratch/v100_musetalk_fit40_240_attempt01/{build_musetalk_fit40_240_attempt01.py,run_musetalk_fit40_block_attempt01.py}` 尚未冻结/stage/执行。Dev准备Ditto240薄wrapper，复用pinnedW2L block02 core（96537B/SHA6969c6e952d46b6753d279b0e2447aa71c7b8e92967cd42ee57dc8122da5616e），只改generator/runtime/resource/parent scope与native ceil(N×25/16000)，共同评分/原PCM/decoded支持不变。两adapter对应CPU测试已要求，源完整后ROOT审查，再接GPU；不用新engineer identity重复。

ASR还在CPU准备严格late-bound720 consumer：三模型×40六行块、原声明路径到实际local restore映射、先fullhash NPZ再np.load，验证数组meta/science/artifact/binding/roster，缺失/失败slot保留CONTROL_FAILED，完成720才canonical Torch CPU selector06。selector06 source `scripts/experiments/cross_tfg_audio_head_20261003/fit_only_spsa_selection_attempt06.py` SHA dda610b985026be7f9328c479358e96c983a582c511964e8c6e72a6d398692e2，ROOT实际24/24CPU tests PASS/0skip；未实际720选择或SPSA更新。

blocks0–5媒体与canonical raw/support已agent逐byte归档到 `B:/tts-exp-project/archives/fit40_w2l240_{media_readback_attempt01,media_readback_attempt02,raw_support_readback_attempt02,raw_support_readback_attempt03}`；frame-map/PCM16 supplements在 `.codex/scratch/v100_fit_stagea_formal_w2l_fit_slice_attempt09/`。只在全byte读回后清理temporary media，canonical arrays/clock/support/ledgers完整保留，cloudbackend上传未验证。

已重新读既有机制结论《冻结增强头的门控时序与频谱贡献》《音素平均谱形与内部轨迹的因果分解》：冻结head的Cbest主要通过背景距离B拉开，门控正确时序给D0/单位方向较小额外收益；512ms错位保留Cbest，gate DC约0.5，使静态谱形和gate均值混合。μ/ρ直接TTS移植有recipient兼容性限制。当前9参数head以32ms窗/8ms步、频带静态+正负flux gate、3dB平滑mask和全局RMS匹配继续检验三个前端的响应；不重跑音量因果族，不将源码推理写成新实验证明。

旧AutoDL root@connect.westd.seetacloud.com:18625唯一DirectD0 launch终局仍未知，私有PTY自动审批拒绝保持，不绕过或重放。V100可持续路线继续。Git 5688784dd5c6c1272074f0c4b6c5ff79f117c907 local未push、原五tracked dirty保留，新实验code untracked，BM/HANDOFF/scratch/SHM/runtime是machine-local或fragile；没有commit/push/PR。

## Changed Files

- selector06与对应tests：typed staticportrait及跨模型原源身份校验；实际24项PASS。
- `.codex/scratch/root_native_actual_arrays_readback_qc_attempt02.py`、`root_ditto_actual_repeat_readback_qc_attempt01.py`及JSON：实际原数组/媒体/PCM/支持独立核验。
- Ditto score-only04及exact closure installers：原生视频保留、官方crop score复用，实际repeat PASS。
- Muse/Ditto240 adapter draft与720consumer尚在准备；不作为已完成实现。
- BM Task/Experiment/HANDOFF此前同步W2L30/240，最新48/240及Ditto04 terminal还须更新可变状态，既有immutable checkpoint保持原文。

## Verification

ROOT实际运行packet230file QC、remote stage readback、selector06 24CPU tests、Muse/W2L actualbody/NPZ/decodedmask/fullnaturalPCM QC，以及Ditto两官方评分row的actualNPZ/PTS/fullF32/PCM16重复核验，全部PASS。agents实际完成W2L48rows和Muse/Ditto各两engineering repeats。所有历史失败保留，当前无三模型完整720及新方法效应。Router/实验指令/Coding Session/Task已加载；same-chat coding/codex/legacy分页检索、previous exact身份、freshGit/canonical GitHub仓库及branch PR[]实际读回，write_note显式overwrite=False/output_format=json经CLI契约执行，要求created并直接核对host元数据。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)，当前local SHA不构造GitHubcommit链接；branch PR实际[]。
- ADR022 `tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement`、ADR023 `tts-exp/docs/adr/023-decoded-video-window-support-for-native-syncnet`、ADR024 `tts-exp/docs/adr/024-live-process-resource-capacity`。
- `tts-exp/docs/reference/server-access`保存无凭据连接和runtime位置；Task、实验与上列实际readback路径为恢复入口。

## Observations

- [result] 三模型native重复、原音换轨和真实五帧支持资格已实际通过ROOT独立数组/媒体检查；W2L正式blocks0–7各6/6PASS，共48/240。
- [verification] ROOT Ditto04实际两139×31矩阵/七数组和video elementary完全一致，完整F32 N92225和PCM16评分mux逐byte匹配自然源，native145PTS/139支持PASS。
- [decision] 保持ADR022完整共同支持与分母、ADR023decoded支持、ADR024live容量，六行边界公平交GPU；只补薄frontend/runtime adapter和严格actualNPZ消费。
- [blocker] Muse/Ditto240 packet及完整720尚未完成，不能选择初值；旧AutoDL终局未知但V100持续推进。
- [next_step] 接齐并实际执行MuseTalk/Ditto同40条六方法240薄adapter，持续W2L完整块，完成三模型720真实原音评分后按ADR022选择初值并优化。

## Relations

- continues [[Codex checkpoint - 2026-10-09T06-25-24Z - Native repeat qualification and formal Fit execution]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]
- relates_to [[023-decoded-video-window-support-for-native-syncnet]]
- relates_to [[024-live-process-resource-capacity]]