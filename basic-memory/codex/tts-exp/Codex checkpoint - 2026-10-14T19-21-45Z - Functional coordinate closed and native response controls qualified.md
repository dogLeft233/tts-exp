---
title: Codex checkpoint - 2026-10-14T19-21-45Z - Functional coordinate closed and
  native response controls qualified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t19-21-45-z-functional-coordinate-closed-and-native-response-controls-qualified
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-14T19:21:45.728719+00:00'
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

# Codex checkpoint - 2026-10-14T19-21-45Z - Functional coordinate closed and native response controls qualified

## Summary
已完成原自然时钟功能坐标报告40，并从两份真实fixed-RMS驱动完成六组原生CPU基线；新的构音方向小响应探针已冻结CPU控制门，GPU仍未启动。

## Story
用户最初要解释TTS的TFG优势并产出通用自然音频增强头，最近明确要求固定RMS/内容/时钟，分别测试原生分离度、稳定性和局部能量/谱形，同时检查独立口部运动，并控制审计和实验规模。报告39的原生18因子行与物理谱形6行已关闭，均未通过双人跨模型扩展门。当前策略转为可解释的构音响应坐标，先复用旧轨迹，再只做两句三模型正负小干预。

报告40 `basic-memory/docs/experiments/40-functional-response-coordinate.md` 已保存17点lag曲线和人级表。借用旧RMS逐句参考p后，height/width的六个model×attribute格Cp/Hp均过描述门；原head仍无共同收益。共同输出功能测量坐标是当前候选，尚不能升级为共同输入控制或真人准确度证据。原clock/PCM没有移动。

新probe以time/label支持选择S0022W0123与S0024W0122，seed20261003，不按分数或视觉幅度选样本。自然9FIT均衡anchors定义高度轴，token core的低秩正负正交旋转保持帧半径、core内部Gram和时钟。为避免CPU编码/注入路线混杂，未来18短视频包含6个相同featurehook的zero；旧RMS仅作背景，不继续追溯旧完整环境。原GPT-6 Luna/max负责代码和执行，ROOT一次科学数组QA。

## Working State
权威导航为 `Research/研究主线与证据地图`、Task `native-axis-factor-pilot-20261013` step11、实验 `构音方向的原生输入响应探针` 及HANDOFF。Task与最新实际receipt优先于本不可变快照。

本机scratch、未提交BM、远端SHM均machine-local fragile，不是完整灾备。服务器wjj@192.168.50.2:22保持运行；用户已取消关机。最近GPU实际快照为UTC16:14:39Z idle，未来dispatch必须fresh核对，不能据旧快照断言当前空闲。旧训练/中心/Dev/Confirm/关机归档/cloud donor支线保持停放。

CPU6 source03唯一rc1因Ditto二返回值解包，0保存数组，已终止；04仅派生第三返回值后单次rc0，六数组实际读回通过。旋转source01手抄helper SHA多一个b，尚未执行；02仅修pin，正确执行入口仍放原attempt01父目录，另目录副本不用。ROOT `root_rotation_cpu_admission_attempt02.json` SHA1a19531a270f72d8b474e9b53689ac7c77534f4276bcd657bab7a130b629242a已准入一次本地CPU物化，原Muse Luna承接执行；结果尚待实际QA。原angular Luna准备18行renderer源码，仍SOURCE_ONLY。

## Changed Files
- `basic-memory/docs/experiments/40-functional-response-coordinate.md`、assets及README索引：正式关闭功能/lag诊断。
- `basic-memory/Research/研究主线与证据地图.md`、`basic-memory/Tasks/native-axis-factor-pilot-20261013.md`、两篇构音/功能Experiments、HANDOFF.md和CONTEXT.md：当前地图和恢复入口；CONTEXT99行。
- `.codex/scratch/native_attribute_response_probe_20261015_attempt01/`：真实RMS驱动、accepted CPU6 readback04、CPU旋转源码02和独立准入；新源码/笔记多为untracked。9条tracked dirty中其余既有变更未归本轮。

## Verification
- 功能ROOT actual QA `phone_conditioned_visual_response_20261015_attempt03/root_actual_effect_qa.json` SHA9debfb49…：144几何/288normalizers/6678观测/3717数值检查一致。
- Lag ROOT actual QA `functional_lag_response_20261015_attempt01/root_actual_lag_qa.json` SHA4dd93b5e56a24c7b4998904aeeebf5d1e59534a11d38dbfc54702286610e894c：144实际轨迹、132S17tokens、全部17raw/z曲线与Cp/Hp，7926检查maxerror0；不重跑。
- 两份fixed-RMS原WAV body/PCM/N/RMS匹配canonical；CPU6实际source04 PID2526284/startticks48901942，UTC18:56:53–18:57:17Z rc0并已退出。实际receipt892720030a6e78b109607bda0f447af7faeb9080283f8258a6a1bd0cd2dcf12b。
- ROOT `root_actual_cpu6_qa.json` SHAe9735787aa9d6c33ce591feec104864319e4c51a425ff3beec6ff668b47e5b72：六NPZ实际hash/数组bytes/dtype/finite/clock，Wav rawmel到chunks逐值一致。Wav[257,80]/[283,80]j80，Muse[80,50,384]/[88,50,384]j25，Ditto[81,1024]/[89,1024]j25。不重编码。
- ROOT通过CodeGraph阅读旋转source/math/anchors/输出dtype后控制，02实际SHA6f782f2f89467914f3e9c70e483c2efd28b160789b3e64f357e0256902acdcd7，确认唯一字节pin修复。CPU数学门在执行前冻结，未看输出。
- 无实现测试、依赖安装、云API、新训练或GPU渲染。Git直接只读确认9条tracked dirty、同branch/HEAD；GitHub exact-head PR为空，commit API422。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 当前HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907` local/unpushed，不构造commit链接；没有PR。

## Observations
- [result] 原head没有跨模型超过RMS，固定参考相位后的共同输出功能坐标已得到描述支持；输出测量与输入控制是两个端点。
- [decision] 固定两句/三模型/一seed/±一小剂量，先CPU控制，未来最多18三臂短render；不用大矩阵或旧完整环境审计替代效应判断。
- [decision] zero/plus/minus共用CPU featurehook/dtype/decoder，最终自然PCM/原clock不改；native旋转不是波形增强方法。
- [blocker] GPU准入仍缺新正负实际数组QA及renderer实际源码/输入绑定和fresh进程核对；这是限定的执行依赖，不是科学阴性。
- [next_step] 接收Muse Luna唯一CPU旋转输出并做一次实际norm/Gram/margin/noncore/clock QA，通过后读angular Luna18行renderer差异并启动有界响应screen。

## Relations
- continues [[Codex checkpoint - 2026-10-14T18-04-52Z - Natural clock support qualified and functional effect handoff]]
- tracks [[native-axis-factor-pilot-20261013]]
- relates_to [[研究主线与证据地图]]
- supported_by [[40-functional-response-coordinate]]
- relates_to [[构音方向的原生输入响应探针]]