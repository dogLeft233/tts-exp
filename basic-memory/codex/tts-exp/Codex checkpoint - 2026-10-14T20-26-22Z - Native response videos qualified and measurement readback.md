---
title: Codex checkpoint - 2026-10-14T20-26-22Z - Native response videos qualified
  and measurement readback
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t20-26-22-z-native-response-videos-qualified-and-measurement-readback
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 20:26:22.458864+00:00
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-14T20-26-22Z - Native response videos qualified and measurement readback

## Summary
两句、三模型的18个同路线zero/正/负原生构音响应视频已完成，ROOT实际每帧时钟与完整解码自然PCM检查通过；唯一18行评分/口部测量已由原Luna报告rc0，完整数组正在读回，尚未判断效应。

## Story
原目标是解释TTS驱动TFG的优势并实现可跨模型的自然音频增强头。用户最新要求固定RMS、内容、时钟，隔离原生分离度、稳定性、能量/谱形，用独立口部开合/速度验证，并控制审计和规模。报告39已关闭native18与physical6屏筛；报告40用旧144轨迹形成固定参考相位下的共同输出功能坐标候选，但原head仍没有共同收益。这些既有科学数组不重算。

当前只用标签/时间支持选出的S0022W0123、S0024W0122，以自然9FIT均衡anchors定义高度方向，在token core作正负小正交旋转。ROOT实际恢复六anchors与42个token-arm操作，norm/Gram/原dtype/原clock/非core均通过冻结控制。为避免CPU编码及注入路线差异，18视频包括六个同featurehook的zero。旧RMS仅背景，不继续追溯完整旧环境。原GPT-6 Luna/max写代码和执行，ROOT只做一次实际科学产物核验。

18渲染唯一有效controller2534565/startticks49267493，于UTC19:57:49–20:06:01Z完成rc0/18PASS。ROOT直接读实际媒体每帧PTS，并完整解码18条mux音轨，全部原自然float32 PCM SHA/N相同；这不是只信manifest。之后仅测量这18条，不再渲染、重编码或MFA。原Wav Luna报告readout05 worker2542617/49386476，UTC20:17:39–20:20:38Z rc0，目前正在完整矩阵/口部NPZ/receipts读回；ROOT尚未收到完整科学数组，因此不写效应结论。

## Working State
当前权威入口为 `HANDOFF.md`、Task `basic-memory/Tasks/native-axis-factor-pilot-20261013.md` step13、实验 `构音方向的原生输入响应探针` 及研究证据地图。最新Task/实际receipt优先于本不可变快照。服务器wjj@192.168.50.2:22保持运行；关机授权已取消。scratch、远端SHM和未提交笔记均machine-local/fragile，不是完整灾备。

测量远端root：`/home/wjj/tts-audio/tts-exp/.codex/scratch/native_attribute_response_probe_20261015_readout_attempt02`，控制目录同名加`_control`；渲染root为同父级`native_attribute_response_probe_20261015_render_output_attempt03`。不可重复launch这些已结束job。readout05 source、ROOT准入、冻结协议均在本机 `.codex/scratch/native_attribute_response_probe_20261015_attempt01/`；有效renderer source08/input04在 `native_attribute_response_probe_20261015_render_stage_attempt01/`。

局部工程修复保留失败原件：两个renderer前置import/API错误均0完成视频；readout04手抄guard SHA末位缺f导致静态检查0处理退出。ROOT只核对13条明确执行来源，唯一pin差异修正为05；没有扩大环境审计或改NN/候选。所有旧训练/中心/Dev/Confirm/关机归档/cloud donor/visual teacher停放。

原Muse Luna已复核geometry consumer02科学算法，03仅将旧source-manifest07 CLI/metadata改成通用render-source-manifest，仍SOURCE_ONLY，尚未授权效果执行。后续只读实际readback，保持绝对17lags[-8,8]、原S17支持、旧RMS逐句p和同三臂commonvalid normalizer。不得误用p+[-8,8]或重新选峰。raw/z Cp及Hp需要新zero两句都为正，正>zero>负两句/三模型均成立才有资格另行规划波形可达性。support损失和UNMEASURABLE显式报告；口部速度描述性，SyncNet矩阵与功能指标分开。

## Changed Files
- HANDOFF.md和 `basic-memory/Experiments/构音方向的原生输入响应探针.md` 已压缩为当前恢复状态；Task step13、地图、CONTEXT99行已同步到唯一测量job，下一次更新应写终局及readback。
- probe scratch包含实际CPU6/旋转数组、一次ROOT QA、冻结协议、renderer来源/input/准入及测量05；stage含18真实媒体readback与ROOT完整PCM/clock QA。
- Git直接只读确认9条tracked dirty与1003条untracked状态；本轮tracked只触及CONTEXT，其余是既有变更。不删除、回退或归因于本轮。

## Verification
- ROOT旋转实际QA `root_actual_rotation_qa.json`：六NPZ/自然FIT/MFA全公式maxerror0；f32 norm相对误差最大4.324e-9，Gram2.314e-9，zero/noncore/clock逐字节一致。无需重复CPU资格。
- ROOT媒体QA `native_attribute_response_probe_20261015_render_stage_attempt01/root_actual_render_pcm_clock_qa.json`：18实际silent视频j/25、count匹配；mux25fps/原起点、18完整PCM解码SHA/N全匹配。W/M/D两句帧数分别77/80/81及85/88/89。
- 13明确执行来源实际逐项readout identity检查，除已修复的guard手抄末位外全部相同。remote实际483f extractor沿用报告39合格来源，不覆盖为本机2ce版本。
- readout05终局目前由原执行Luna观察，完整scientific readback和ROOT原数组QA尚待；不把rc0写成效果成立。
- 无实现测试、pytest、compileall、依赖安装、云API或新训练。GitHub repository API200，exact-head PR为空，commit API422；HEAD只作为local/unpushed SHA。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- local/unpushed HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`，没有可核验远端commit链接或PR。

## Observations
- [result] 18条受控视频已真实完成并通过原自然PCM及原clock核验，尚无完整评分/视觉效应结论。
- [decision] 仅两句×三模型×zero/正/负、一seed一剂量；用同路线zero消除输入路线混杂，不扩矩阵。
- [decision] native方向可控不等于波形可达，也不等于真实嘴型准确；若预设响应门失败，不翻符号、改剂量或追踪扩展该候选，改用显式不同假设。
- [blocker] 真实效果判定只等待完整18行读回、generic consumer03绑定及一次ROOT科学数组核验；没有需要用户解决的权限/凭据问题。
- [next_step] 接收唯一readout02完整18矩阵和geometry NPZ，绑定已冻结readout与renderer source08，单次CPU计算固定S17/p响应，ROOT核验并关闭本小屏筛。

## Relations
- continues [[Codex checkpoint - 2026-10-14T19-21-45Z - Functional coordinate closed and native response controls qualified]]
- tracks [[native-axis-factor-pilot-20261013]]
- relates_to [[研究主线与证据地图]]
- supported_by [[40-functional-response-coordinate]]
- relates_to [[构音方向的原生输入响应探针]]

## Artifact Pins

- `.codex/scratch/native_attribute_response_probe_20261015_attempt01/root_actual_rotation_qa.json` SHA256 `0e73e3b59a695c967fe0f290587b6e76843f504157d0c50056cfb6d389abcdf2`.
- `.codex/scratch/native_attribute_response_probe_20261015_render_stage_attempt01/root_actual_render_pcm_clock_qa.json` SHA256 `3d4b46d780b13599609db8a55afbcc8e86c1f0ac3da0b4e8f93ee994421de30a`.
- `.codex/scratch/native_attribute_response_probe_20261015_attempt01/response_readout_protocol.json` SHA256 `9b4c3f4150e0a4275925304a3a542230896de3dff82ecdb38dd6d89ecdcfe0d9`.
- `.codex/scratch/native_attribute_response_probe_20261015_attempt01/response_readout_executor_attempt05.py` SHA256 `08e453046bd9ca29bc85bf51563258e42fa7a3f96eac9344b5b98df37b1bf4d6`.
- `.codex/scratch/native_attribute_response_probe_20261015_attempt01/root_readout_admission_attempt02.json` SHA256 `5455e3fdfb1a539107c8a19bfb8ed87e12d799e4327c4c1c183e14de692d7819`.