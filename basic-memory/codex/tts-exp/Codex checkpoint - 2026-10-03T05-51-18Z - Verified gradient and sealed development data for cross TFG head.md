---
title: Codex checkpoint - 2026-10-03T05-51-18Z - Verified gradient and sealed development
  data for cross TFG head
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t05-51-18-z-verified-gradient-and-sealed-development-data-for-cross-tfg-head
branch: codex/cross-tfg-audio-head-20261003
agent: codex
repository: dogLeft233/tts-exp
username: dogle
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
model: gpt-6.1-sol
project: tts-exp
git_sha: c2a17771047ddb7a44ef640c0393b50b3610b1f4
repo_root: E:/Documents/tts-exp
started: '2026-10-03T05:51:18.1354791+00:00'
hostname: LAPTOP-E133GSLE
status: open
capture: deliberate
trigger: compact
cwd: E:/Documents/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T05-51-18Z - Verified gradient and sealed development data for cross TFG head

## Summary

有界等长波形增强头已通过真实GPU梯度和参数更新，20条新开发音频已双重独立复核封存；当前进入候选训练及实际多模型视频开发验证。

## Story

用户最初要求从本地BM/Git/百度盘恢复TTS增强TFG研究，报告25和便携交付已完成。最新要求是做出适配较多TFG模型的可用自然音增强头：增强音驱动视频，成片和评分用原自然音，失败后继续迭代至完备方法。create_goal ACTIVE，无预算。ROOT负责理论、主线、BM与Git；local_recovery、mechanism_audit、cross_tfg_execution三代理均gpt-6-luna/max，负责实现及实验。

方法保持16k mono F32、源时钟和样本数，独立控制全局RMS、局部包络和有界平滑STFT幅度残差，zero-init从identity开始。历史暴露说话人仅fit，新dev选择，未见确认说话人及未参与优化的TFG确认。主指标是增强生成视频对原自然音的Sync-C增量，并与RMS基线比较；分数和独立视觉运动分别验证。当前没有新方法跨模型科学收益，不能将工程one-step或旧pilot当泛化结论。

## Working State

权威设计在 `openspec/changes/develop-cross-tfg-audio-head-20261003/{spec.md,tasks.md,data-protocol.json,development-data-seal.json}`。分支codex/cross-tfg-audio-head-20261003，local/unpushed HEAD c2a17771047ddb7a44ef640c0393b50b3610b1f4；无PR。已提交10个实际执行波形/训练核心文件，source raw SHA与工程运行包一致；CONTEXT93行，任务勾选真实生成/换轨/评分与梯度检查。CLI、fit runtime、independent consumer和新producer仍hot/untracked，别stage-all或覆盖legacy/tests脏文件。

官方AISHELL revision bbe295d530192a4cd41644b711c9aecd087df653。fit S0052–S0101，fresh dev10男声S0005,S0022,S0024,S0037,S0038,S0039,S0041,S0044,S0049,S0050，confirm40其余S0002–S0051未下载/读取。冻结data-protocol raw cf615529edc8ae0b6ffff0a34246db71b3ec8ce5ac17d9f0daf2618ac624cf87不要改。十归档及按lex固定2clip/sp、3.2–8sec的20F32 WAV实际完成：`B:/tts-exp-audio-head-20261003/dev_pcm_rows_attempt01/dev_pcm_rows_manifest.json` raw a553c7dff561c724f5ae653161703b9da498b8ef71a7182425cd1640993c6ad8，20 distinct PCM。mechanism metadata/wave QA276 checks PASS receipt958c821b…72d9；local十归档完整选中成员bytes回读PASS receipt39ce8ba1…277f。Git development-data-seal记录完整20行与两receipt。3698797B输入ZIP raw778fd10f…8622已实际上传远程persistent namespace，待StageA解包。

实际oldpilot工程one-step PASS：`/dev/shm/cross_tfg_audio_head_20261003/oldpilot_one_step_attempt01/run/one_step_gradient_smoke.json` raw dae9be2a9bdcffb1eed4c12e521685bef6e394663cf124d5784861fe86733aef。官方与Torch音频前端maxabs3.5762786865e-6，原atol1e-4/rtol1e-5未放宽；audio gradL2 1116.64102、head gradL2 .822993975、参数更新L2 .001473091、输出变化maxabs1.1112541e-5，6梯度tensor有限、末层2tensor217值非零，前层零梯度符合zero-init。工程head_after_one_step.pt rawdc707844…c8fe，不能当候选。32帧prefix浮点视频/straight-through uint8代理支持不同于官方全视频，工程anchor+2不能带入科学fit。ROOT实际完整响应 `.codex/scratch/cross_tfg_execution_attempt02/root_read_one_step01_attempt01.json.response.json` exit0。

新tiny fit使用历史S0052–55×2共8条源音22400 sample/1.4sec crop；plan `runs/cross_tfg_audio_head_20261003/fit_subset_attempt02/historical_fit_subset_plan.json` raw34b474fb…d956，payload3bad1111…7e12。科学候选RMS_fixed+bounded±3dB residual+post_rms_target .0376838172675746，24step首轮，目标median(D31)−normalized softmin τ.1。实际fit尚未发起。最新runtime pre-run修复：20480 PCM16前缀匹配实际32frame track，full crop另保完整hash；PSF task-local overlay必须在SyncNet模块import前插入并pin完整py树。当前runtime c993302eb2736ddac357a38413b4b47f65bbe55f8507607f3860394c8c791bfe，fit_cli a91aac1fe67c2100503e938b39a96df601a29e1a27a5045d46ec1306d0371400；旧c811/5e等不能dispatch。step label已改proxy_median_minus_normalized_softmin，未改数学。新bundle/receipt由代理继续准备。

ADR007实际created/readback PASS：官方run_pipeline中间AVI默认mp3，实际ffprobe已确认；F32无损最终换轨虽精确，旧official cropPCM16不同于直接source量化。新主端点native_roi_original_pcm16：保留官方选中ROI/video，直接canonical original F32→FFmpeg PCM16精确源clock window，无损Matroska remux→实际evaluate(self,opt,videofile)。旧native MP3只作诊断，固定ROI和训练proxy分开。Clip tail保留requested stop并clip min(N)，不填音频/移轴。Consumer与producer map v2 `{crop_frame_index,source_clock_frame_index,timestamp_num,timestamp_den,render_frame_index,frame_origin}`，NPZ query/track用nominal original clock；转换pad超出真实render时render_frame_index=null，不虚构帧。FULL原音支持主、guard20敏感性、31lag/B/Dmin/D0/offset以及缺失规划分母保留。

StageA正在构建20clip×3seed20261003/04/05×identity/RMS/RMS+localratio.8共180真实W2L开发行，尚未跑。source静态portrait reference512×512/200frame/25fps实际生成并读回SHA c76617141188acfea5f4c122513b982997ec45ebaa35a2b6ed9f67101e9c40de；ffmpeg实际36d94a605d612e4090d1b8aec889d0c0801c6eafb1593c90f5c0dfd2e2966a45。consumer静态parse PASS但尚在匹配map v2，不能用旧interimhash当finalpin。

额外模型download03实际Ditto COMPLETE约2.314GB/37files，HFdigital-avatar/ditto-talkinghead e4a2f60328ee7c32af585ac4b3cce299e4c8e254。Muse HF模型约5.1GB完成，最后Google faceparse DOWNLOAD_FAILED，复用已完成weights修复单项；不要重新整包下载。源Muse0a89dec45a0192b824e3cf4daf96c239440c5ed8/Dittoc3e47eee2e626500017a0556b470d6d4182f85e8。二者实际inference依赖/新dev尚未运行，LatentSync留为heldoutTFG，Museaux SyncNet不等于LatentTFG完成。

远程root@connect.weste.seetacloud.com:12349，hostED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；密码/API只runtime，不写文件。privatePTY50274桥接2133、tmpfs部署/run与ignored HANDOFF属于易失状态。正常迭代已获用户授权，ROOT检查request/source后dispatch并以实际exit/status核验，不重建旧巨型审批框架。根/autodltmp紧张，新包在tmpfs/noexec purepy/model可用，B盘约1.266TB可存新产物，旧源不删/退役。Python venv executable禁止resolve丢venv。

BM mutable Task/Experiment update05 exactreadback PASS，statusACTIVE/IMPLEMENTED partial，正文已one-step/dev20/ADR007；旧frontmatter blocker/next-step仍有yaml/QA待修，应下次实际里程碑替换。ADR006/007 OPEN，报告25既有结论不变。HANDOFF需更新currenthead/one-step/latestcheckpoint。CodeGraph先前实际不可用，rg fallback，不重建索引。

## Verification

实际one-step GPU完整报告/exit0、源pins与10文件bundle核验、20dev波形两独立PASS及完整输入ZIP哈希/传输、portrait/ffmpeg实际哈希、MP3中间codec、模型03下载状态已读回。没有运行pytest/unit tests。新fit与180dev尚未执行。Checkpoint前Router/实验指令/Coding Session实际读取，59同chat/0legacy分页完整并核验BG身份；canonicalGitHub/无PR/已推送baseline/本地HEAD由实际只读API与Git核验。gh未安装，改用GitHub REST而不推断。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local/unpushed HEAD `c2a17771047ddb7a44ef640c0393b50b3610b1f4`.
- `openspec/changes/develop-cross-tfg-audio-head-20261003/` and `scripts/audio_enhancement/`.

## Observations

- [result] 真实模型梯度与参数更新已通过，fresh20dev音频双独立封存并已传远程；尚无候选头跨TFG收益。
- [decision] 新科学评分用官方native ROI配直接原音PCM16而非默认MP3中间轨；训练proxy和最终原音F32成片分开记录。
- [blocker] tinyfit/180dev实际运行、额外TFG推理依赖、跨模型选择和untouched确认及视觉guards待完成；Muse单项faceparse下载失败可继续修复。
- [next_step] 执行最新版tinyfit24step候选与180行Wav2Lip开发比较，独立原音评分后迭代增强头并扩展Ditto/Muse及heldout确认，持续到可复现交付。

## Relations

- continues [[Codex checkpoint - 2026-10-03T04-48-03Z - Actual waveform generation and multi model training path]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[006-portable-waveform-enhancement-head]]
- relates_to [[007-original-pcm-scoring-for-waveform-head]]
- relates_to [[25-tts-tfg-mechanism]]