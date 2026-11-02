---
title: Codex checkpoint - 2026-10-03T08-40-18Z - Full context preflight and usable
  native audio CLI
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t08-40-18-z-full-context-preflight-and-usable-native-audio-cli
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
project: tts-exp
started: '2026-10-03T08:40:18.1890479+00:00'
cwd: E:/Documents/tts-exp
status: open
git_sha: 17ef390b5f4a678adf8e145eeef30e377c278c6d
trigger: compact
hostname: LAPTOP-E133GSLE
username: dogle
branch: codex/cross-tfg-audio-head-20261003
repository: dogLeft233/tts-exp
model: gpt-6.1-sol
capture: deliberate
repo_root: E:/Documents/tts-exp
agent: codex
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T08-40-18Z - Full context preflight and usable native audio CLI

## Summary

首个实际训练头的四方法开发实验已173/240行PASS；完整上下文训练包和预检、48k双声道命令行原轨换回均已实际核验，跨TFG方法目标继续ACTIVE。

## Story

用户从恢复TTS增强TFG机制研究，进一步要求产出适配较多TFG的可用自然音增强头，失败尝试后持续迭代。旧机制报告25已经完成并封存，本目标独立继续。增强音只驱动视频，最终音轨和主SyncNet评分用原自然音。ROOT负责理论、主线、BM、Git和实际请求核验；实现与实验由三个gpt-6-luna/max子代理完成。拟合或工程PASS并不代表跨模型收益，当前仍须完整独立消费、模型迁移、冻结和未见确认。

方法是有界STFT幅度残差、identity初始化、16k mono F32、样本数和源时钟不变。分离整体RMS、局部包络和频谱比例；比较原音、固定RMS、局部压缩和训练头。历史训练/新开发/未见确认按说话人分开，LatentSync和保留人脸用于冻结后的检验。跨前端均有效的共享配置优先；各模型专用配置只能作为单独方法报告。

## Working State

持久协议入口 `openspec/changes/develop-cross-tfg-audio-head-20261003/`，data-protocol raw SHA `cf615529edc8ae0b6ffff0a34246db71b3ec8ce5ac17d9f0daf2618ac624cf87` 不变。分支codex/cross-tfg-audio-head-20261003，local/unpushed HEAD `17ef390b5f4a678adf8e145eeef30e377c278c6d`，无PR。10个核心模块、fit_cli和首版runtime已提交。CLI/__main__/评估器/生产器/v2源码未提交；旧legacy/tests脏文件不可stage-all。ROOT改了CLI README的实际训练、native原轨和候选证据边界，并新增小型设计文件 `transfer-iteration.md`，未更改功能代码。CONTEXT93行，HANDOFF ignored仅机器本地。Task快照attempt08和Experiment快照attempt07均直接读回PASS，尚未完成。

首候选fit07使用Wav2Lip环境、历史8条S0052–55×2、22400sample中心片段、24step，实际exit0且24次非零更新。权重 SHA `bc5aeec15f9197ca0c1d0d606eebe41ff2dbf6620767eee5bc37ca4b9395aaed`。权威readback `.codex/scratch/root_read_fit07_attempt02.json.response.json` 和 `cross_tfg_fit07_root_actual_readback.json`；本地封存 `runs/cross_tfg_audio_head_20261003/fit07_actual_archive/run02.tar.gz` SHA `55ae75e8159d90d17ecbe261fde25801e8af30b6a4a505bed095af5c2342dde3`。训练实际解释器是 `/root/autodl-tmp/wav2lip-syncnet/envs/wav2lip/bin/python`，不是没有librosa的SyncNet环境；BJ中旧建议已在BK纠正。

新开发20条来自10名新男声、各2条，AISHELL revision bbe295d530192a4cd41644b711c9aecd087df653，PCM双独立核验及Git seal完成；确认40名未读取。StageB为20clip×3seed×4方法240行，科学runner SHA `30c894ea31541cdde76a5a915bf635a36838a1cdb11f761c334c8986e71a01bc`，plan raw `43f10f334ea904d071836195733a1d1bdabe479a38061156ac38d5083d7d8632`。实际launcher04起始2026-10-03T07:31:46Z，PID92924/92967，OUT `/dev/shm/cross_tfg_audio_head_20261003/stageb_dev_attempt03/run`。最新真实progress04响应 `.codex/scratch/cross_tfg_audio_head_20261003_stageb_dev_attempt03_attempt04_progress_probe_attempt04_request.json.response.json` 为173/240 PASS、0失败；tmpfs尚约10.4GB。首四配对的20引用文件、原F32/PCM16和视频流/时钟已经ROOT核验，见 `stageb03_first_four_root_readback_attempt01.json`。不得把未完成计数或训练代理写成收益结论。

重要纠正：BK基于部分结果误写生产器没有三组summary字段。ROOT已读实际冻结源：运行结束后 `_finish_supports` 会统一生成 lag_curve_by_support/support_by_support/metrics_by_support 并原子写最终results.jsonl。最终240行优先使用原consumer SHA `a693b2ab71876396fb3fd73f9dda40a665f8cd43981c31e6e61afadd17b3942e`，consumer_attempt02作为未执行备选保留。主端点仍为selected official ROI直接装原PCM16后真实evaluate；FULL/guard20、31lag、规划分母和不可估计项保留，不做offset补偿。completion_readback_attempt01请求已准备，终态后才消费。

完整上下文v2 scientific02已实际上传、提取和CPU预检PASS：同8条全长历史PCM，共553919samples，96step/12epoch；整段MFCC上下文缓存，音频embedding只取official floor(N/640)-5实际数量，中心32帧27query的31lag均有效。科学plan raw `e491d0bb210fe78647f4845543a88c629d37cecc09a0823c287560fbb1e71a06`，runtime `d086c61cdb993bffa5f9ff7a55bf4b66b1b9127028f8b80eb3b5f1be74464d7b`，bundle `aab9ed983d946924e101117ce6c14f16d92889f8296d33074d3ca42b73345d3b`。ROOT独立核验33成员、32payload、18源码闭包、8PCM和所有counts。实际preflight report raw `f822ea7b7c0c51aeb73a0a54f3509e18de19cbc634e7eef69a21a1964d555c8f`，status FULLCONTEXT_PREFLIGHT_PASS_NO_FORWARD；`.codex/scratch/cross_tfg_audio_head_fit_v2_fullcontext_attempt02/04_read_preflight_request.json.response.json` 为执行权威。launch03仅启动即返回的receipt缺陷在执行前被发现，仍未运行；fresh04 watcher已交付待ROOT最终读请求，写durable started并等待child实际exit后写finished/真实退出码和report指针，只要求空闲显存≥16GiB。worker SHA `f779a8129c0f095d9a7d76492e698f2988e8f2196ba9b43c42299bebbd067a70`，index `fcf85762e83036b721aa68ecd9838ab0f9b4676d5f260bfe7e51b510777abd6a`。尚未实际启动，不写成96次更新。

v2开发将新增60条候选视频；在原240完全封存后建立新300行五方法派生分析，原240四方法估计保持独立。显式复用source plan/result/manifest/repeat-QC/行hash和原产物，只在五方法共同支持上重算，不伪造新生成或把旧四方法支持直接当五方法。draft v3 raw `8bd6be8db176a7887ef6676ee3a7f95d6e7997409e848d8d329ffd4e6d170943`，新builder v2 `9dfdd79b505812b3ab4380c86dab1f8e8967a82bd121e3c59c29fb2359b67bef` 已交付待ROOT完整review和运行；source失败上下文保留分母，分析配置沿用原bootstrap/aggregation，新增v2 vs identity/RMS为开发主对比。fit24与full96差异同时涉及上下文和步数，不能归因于一个因素。

公共CLI原音换回实际工程QC PASS：48k stereo PCM16输入源file SHA a22e800743d9130927aa9909427482660ab34284f50138123a7740beb85b85f6，经canonical16k mono/8000sample生成驱动后，mux仍保留原48k/2ch/24000sample逐样本PCM，原视频ES/141frames/PTS一致。权威 `.codex/scratch/cross_tfg_cli_io_qc_packet_attempt02/12_read_cli_io_qc_status_attempt01_request.json.response.json`，结果persistent NS/cli_io_qc_attempt01/control/cli_io_qc_result.json raw `08b713d95e3bb485cfa1bad1ec86118e972e04ea61d6fe15951e302714975faa`。实际fit07权重经CLI的CPU应用QC新packet也已交付待review： `.codex/scratch/cross_tfg_cli_checkpoint_qc_attempt01/request_index_attempt01.json` SHA `5ebd9b981d630b5336cdfe14a5497f4fccfac001b77171c6bd7b3c918aa203c4`，不生成新视频或读取效应。通过后选择提交CLI和README。

Ditto/Muse权重实际已有，模型CPU06失败是Wav2Lip解释器无pip，未安装/未推理；只读probe发现base miniconda Python pip24支持--python目标literalWav2Lip解释器。fresh07实际目标pip dryrun工作，但onnxruntime-gpu1.20.1未提供，exit1、0安装，Muse尚未进入，不能写科学阴性。实际diagnostic `.codex/scratch/root_model07_pip_error_read_attempt01_request.json.response.json`。已查官方ORT CUDA文档/PyPI确认1.20.2 CUDA12/cuDNN9兼容Torch2.5.1cu124族，新08准备1.20.2锁wheel及每模型独立失败捕获，使Ditto失败仍执行Muse。07漏放nested overdub helper已经additive09/10上传+hash实际PASS，不改原packet。新overlay禁重装Torch/NumPy/SciPy，CPU准备可与GPU并行；后续推理仍须实际receipt。

机器易失状态：ROOT privatePTY50274/bridge2133，只通过 `.codex/scratch/dispatch_requests_root_attempt02.ps1` 实际派发完整review的请求。服务器 `root@connect.weste.seetacloud.com:12349`，ED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。密码和API仅runtime，永不入文件/BM。NS `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`；root余约1.25GB、autodltmp381MB、/dev/shm noexec。本地BCloudDrive可用。实际只读mount/recognizedCLI探查未找到服务器百度挂载，不能据此断言其他连接不存在。禁止删旧研究源。已交付源/请求不得覆盖，actual `.response.json` 才是执行权威。

## Verification

本次实际验证为StageB173行计数/0失败、v2预检全source/PCM/counts及no-forward、CLI原48k双声道精确换轨、模型07pip失败的真实stderr和nested helper上传SHA。ROOT README/transfer设计文档未引入功能实现；未运行pytest或单元测试。新头独立效应、多TFG及未见确认尚待实跑。本checkpoint先加载bm-checkpoint/bm-writing、配置、Router、实验指令/Coding Session/Task，再搜同chat与legacy并直接读BK，核验Git和GitHubcanonical/baseline/PR状态。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local/unpushed HEAD `17ef390b5f4a678adf8e145eeef30e377c278c6d`.
- [ORT CUDA compatibility](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html), [PyPI ORT1.20.2](https://pypi.org/project/onnxruntime-gpu/1.20.2/).
- `openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md`.

## Observations

- [result] StageB实际173/240 PASS且0失败；完整上下文v2预检和native48k stereo原轨CLI工程QC已实际通过。
- [decision] 最终生产器会生成三组共同支持summary，240终态优先原consumer；五方法派生分析重新计算共同支持并保留原240分析。
- [decision] 跨TFG共享方法优先，局部特征机制需分离RMS/包络/频谱，并把fit24对full96列为复合迭代比较。
- [blocker] 独立完整开发效应、多模型实际推理与共享方法冻结/确认仍未完成；模型依赖08在修复，工程失败保持原证据。
- [next_step] 完整读回并实际启动v2 watcher04及fit07 CLI CPU应用QC，同时推进模型08独立CPU准备，240封存后立即运行原独立consumer并据结果继续方法迭代。

## Relations

- continues [[Codex checkpoint - 2026-10-03T07-58-26Z - Live paired development and full context model preparation]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[006-portable-waveform-enhancement-head]]
- relates_to [[007-original-pcm-scoring-for-waveform-head]]