---
title: Codex checkpoint - 2026-10-14T22-23-34Z - Output ownership yields bidirectional
  mouth control
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t22-23-34-z-output-ownership-yields-bidirectional-mouth-control
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 22:23:34.035573+00:00
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

# Codex checkpoint - 2026-10-14T22-23-34Z - Output ownership yields bidirectional mouth control

## Summary
MuseTalk将干预映射到真实输出所有者后，两句都通过冻结双向口部功能门，但正方向正式full Sync-C下降；下一个便宜问题是编码传播与Whisper切片一致性。

## Story
用户要解释TTS增强TFG的原因，并产出跨模型自然音频增强头。最新要求同RMS、同内容、原时间轴，分别隔离原生分离/稳定性/能量谱形，并用独立开合和速度验证；同时控制审计和无依据扩规模。ROOT维护理论与一次实际科学QA，原GPT-6 Luna/max负责代码实验。报告39/40/41已关闭，不重新训练、重跑旧数学或扩旧矩阵。

报告41的Muse干预在audio core I，而固定功能端点读取I+4；真实逐row UNet的直接暴露为0/14。此次只修条件所有权：在I+4行用同natural9FIT高度轴和同对称剂量旋转，原clock/PCM/seed不动，并保留全臂precodec PNG。六新短Muse视频、六读出和冻结consumer已经真实完成，ROOT已独立检查实际原数组、原PCM/PTS、12评分矩阵、primary及H264几何和504RGB帧。

新坐标两句均满足plus>zero>minus的raw/z Cp与Hp，2/2功能screen通过。S22 plus raw Cp/Hp均+0.014518，S24均+0.009598；z Cp分别+0.395881/+0.267726。保持H264 observer重新计算同冻结端点也2/2通过，旧operator+H264为0/2；不能把观察方式更换单独当作全部原因。正方向正式full ΔSync-C却为−0.169/−0.012，guard20 C为+0.118/+0.398且D0均变差。因此已找到限定的native口部功能控制坐标，还未形成原自然音轨的增强头。无真人准确度/泛化推断。

逐帧precodec非owned RGB完全相同；代理的新六H264单次CPU解码显示非owned RGB实际改变：S22 plus73/73、minus64/73，S24 plus81/81、minus68/81。由此定位编码传播，不归因模型跨帧响应。另一原Muse代理已报告一次Whisper共享primitive的AA+ CPU投影：源映射精确重建zero；约61.3%扰动能量不一致，owned能量保留约17.5%，每臂24非owned帧受影响，signed方向仍保留。投影结果尚待ROOT一次实际检查；这仅是切片一致性必要条件，不是waveform可达证明。

## Working State
权威当前文件：Task native-axis-factor-pilot-20261013、HANDOFF、研究主线与证据地图及专题[[接收端局部构音响应坐标]]。它们暂时仍写readout运行中，必须收束更新；最终报告42尚未创建。最新实际receipt优先于旧状态文字。

全部新artifact根为`.codex/scratch/native_attribute_response_probe_20261015_attempt01/`（probe）。CPUQA `root_actual_output_owned_qa.json` SHA0a77ad77de5801f1c73c686e11c8b55b80e4f24d12a7bc89b52755919621c0fd；原PCM/PTS QA `root_actual_output_owned_pcm_clock_qa.json` SHA563e81724232c35cb4e52e6957d2177ed2b701549a06a1f3b6f7ce897645a3e7；响应QA `root_actual_output_owned_response_qa.json` SHAf470d1aa4a926cafb65f73555e6957797bf3b7176b7bbaf8a30348593d3f32d9。effect `output_owned_geometry_effect_attempt01.json` SHAbb5fa06e5fd6b1d8a8f1340752b1fdad4c4d6cc733434cb0ba95a20e06fc4967；H264功能比较 `root_h264_functional_comparison.json` SHA39703dfebf283ce098186a031354ebf4003ac4c31f6036b212f28e7e86e7d159。

完整媒体根probe/output_owned_musetalk_render_readback_attempt01/output：543文件197713752B，504PNG/6H264/6自然PCM mux。紧凑读回probe/output_owned_musetalk_readout_attempt02_readback/readout：67文件887654B，manifest1b4f4544…；注意必须有额外readout子目录。renderer2555964/49915805 UTC21:45:52–21:50:39Z rc0/6PASS；readout02 worker2560257/50011173 UTC22:01:45–22:03:25Z rc0/6PASS，均已退出。失败readout01 root.arms查错导致0processed rc1，03仅修future_gpu_cap.arms；consumer一次CLI根路径错误与wrapper错误都保留，既有成功effect不能重算。当前无新TFG执行授权。

新像素诊断结果 `h264_rgb_pixel_diagnostic_attempt01/results/h264_rgb_pixel_diagnostic_attempt01.json` SHAd33b68df7cfacd84bbd46df74b1845ca6f6c25e3869e80a5851da726ea533d08，六ffmpeg rc0；source386d988d…/execution39178f3a…。新Whisper投影result `whisper_primitive_projection_attempt01.json` SHA0ee1129a7201932ee1b99ab8ea0dcdb6cc4fb44660c442c56e8ccdce1121e6d4，source4fdd61b1…/executionc4c3ae3b…；ROOT尚未检查。Wav代理SOURCE_ONLY准备matched codec诊断script，512FFV1/yuv420p→同224Lanczos FFV1→原PCM/原scorer，未来至多6短SyncNet、0TFG；目前未freeze或运行，FFV1保留YUV420转换，不称RGB无损。

服务器wjj@192.168.50.2:22运行，公钥SSH；用户取消关机。原训练/四中心/Dev/Confirm/归档/cloud donor/visual teacher均停放。机器scratch、SHM、未提交BM仍fragile，不视为完整灾备。密码/API key永不文件/argv/BM/log，无tests/pytest/compileall/依赖安装。

## Changed Files
本轮新增输出所有者CPU/source/controller/renderer/readout/consumer及一次ROOT QA和codec/primitive诊断scratch；报告42及地图/Task/HANDOFF收束待做。angular原代理正准备小canonical科学assets/CSV，不复制大媒体。当前9项tracked dirty中本研究触及CONTEXT，其余8项继承，1003untracked不删除/回退。

## Verification
- ROOT真实6数组与2clock、12token-arm旋转maxerror0，norm≤1.419e-9、Gram≤7.097e-10；14Cp直接覆盖，zero/非有效行逐位不变。
- ROOT六原PCM全解码和全部实际每帧j/25 PTS通过；504PNG实际body、非owned跨臂完全一致，12矩阵/6primary+6H264轨迹/2884比较maxerror0。float64重建列均值与官方float32曲线差≤2.152e-6，B/D/C精确由官方curve得到。不重复。
- 原consumer一次成功rc0；两句冻结S17、17lags、p4、raw/z Cp/Hp均双向通过。H264sameobserver端点单次CPU结果亦2/2，通过不代表正式full评分提高。
- fresh Git只读确认branch/HEAD/root；GitHub repository API200返回canonicalURL、exact-head PR为空、commitAPI422，所以HEAD为local/unpushed不构造commit链接。无实现测试或安装。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- local/unpushed HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`，无PR。

## Observations
- [result] 正确输入行→输出帧所有权使Muse冻结口部功能screen从旧0/2变成新2/2；真实功能响应与正式Sync-C收益分离。
- [decision] 不后验翻号/改dose/phase/support；不扩大TFG/GPU训练。先现有媒体codec小诊断及primitive一致性，避免盲目waveform优化。
- [blocker] 原生独立prompt行修改并非共享Whisper frontend必然可实现的波形；正式plus full Sync-C两句均下降，增强头有效性未建立。
- [next_step] ROOT核对新像素/primitive实际产物，将报告42与小science bundle收束，先维护证据地图再更新Task/HANDOFF，并决定至多六次无TFG的codec评分诊断。

## Relations
- continues [[Codex checkpoint - 2026-10-14T21-24-58Z - Output-owned Muse CPU candidate admitted]]
- tracks [[native-axis-factor-pilot-20261013]]
- relates_to [[研究主线与证据地图]]
- supported_by [[41-native-articulatory-response]]
- relates_to [[接收端局部构音响应坐标]]