---
title: Codex checkpoint - 2026-10-14T21-24-58Z - Output-owned Muse CPU candidate admitted
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-14-t21-24-58-z-output-owned-muse-cpu-candidate-admitted
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: 2026-10-14 21:24:58.761901+00:00
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

# Codex checkpoint - 2026-10-14T21-24-58Z - Output-owned Muse CPU candidate admitted

## Summary
已审阅输出所有者坐标的CPU源码和保留预编码PNG的Muse渲染器差异，准许一次CPU候选物化；尚未执行，也未准许GPU。

## Story
用户要解释TTS相对自然语音的TFG优势，并发展跨模型自然音频增强头；最新要求固定RMS、内容和原时钟，分离原生表示/片段稳定性/局部谱形，再用独立口部运动核验，控制审计和无依据扩规模。原GPT-6 Luna/max继续承担代码及实验，ROOT设计并做一次实际科学QA。

报告39三因素/物理筛查、40功能坐标及41原生构音正负18条筛查已经关闭，不重跑。41冻结门为1/6格、0模型双人；Muse的SyncNet增益与冻结口部功能对比不一致。随后实际forward及mask定位出明确混杂：Muse逐prompt row生成同index帧，原干预audio核心j而Cp测j+4，直接暴露0/14。IMAGE检测无跨帧tracker；旧预编码PNG已删除，H264可传播差异，因此旧结果不等于已隔离Muse decoder轴失败。

本轮只修干预所有者：原MFA core I映射到固定p=4的prompt行I+p，在这些行的真实baseline方向上做同FIT构音轴、同对称剂量旋转。相位、原音频/视频时钟、RMS、PCM、seed、portrait均固定。先CPU，若实际数组纯度通过，再最多两句×zero/正/负的6条Muse短视频；保留预编码PNG以分开真实逐帧响应和H264传播。仍无统一head胜RMS结论。

## Working State
权威入口：Task `native-axis-factor-pilot-20261013` step18、HANDOFF、研究主线与证据地图、报告41和[[接收端局部构音响应坐标]]。同chat上一checkpoint身份已直接read_note确认。机器本地scratch、SHM、未提交BM均fragile；不当作完整备份。

probe根为 `.codex/scratch/native_attribute_response_probe_20261015_attempt01/`。`root_output_owned_cpu_admission_attempt01.json` SHA256 72dc6ff866d51abcac23f6390d3f1bfad9d07db79edb09b876bb9bde9ff3cc1c，状态ADMIT_ONE_CPU_MATERIALIZATION_ONLY。`output_owned_rotation_candidate_source_attempt01.py` f8ab4058292b208363871b2f0f1297c1e3bc8cb05da9d43d81665dfa12a3338d、协议`output_owned_musetalk_protocol_attempt01.json` 561e81a747b62c02af7a2465eac9457436013ffba45286be15d8c6cf40c9123e。输出`output_owned_candidates_attempt01/`已查不存在，必须create-only。

原Muse代理待接执行：bundled Python `C:/Users/dogle/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe -B`，只给 `--output-dir .codex/scratch/native_attribute_response_probe_20261015_attempt01/output_owned_candidates_attempt01 --execute`；没有--admission参数。沿默认算术环境，不强制OMP/BLAS线程变量。仅既有CPU6 Muse数组、natural9 FIT及原MFA；不启动encoder/MFA/GPU/API/测试。

数值门：f32行norm相对误差及单位core Gram最大误差各≤2e-6，plus signed margin>1e-7、minus<−1e-7，14个Cp核心帧直接覆盖；zero、非有效修改行和clock逐位相同。SOURCE中的mapped_prompt_row_indices包含跳过的单帧token提案行，不能当实际修改mask；有效mask必须取status=ROTATED_OUTPUT_OWNED token行并与实际数组变化核对。S22有效行应[28,29,44,45,57,58,59]，S24应[31,32,42,43,53,54,55]。单帧跳过行仍identity。

原angular代理已提交SOURCE_ONLY `native_height_musetalk_runner_attempt02_precodec.py` 484f08bac841513a754eb94910b91b8fbc863bc9ca080d74746f3b221e80eeb0；ROOT只读diff确认变化为保留相同PNG、逐帧索引/时钟/hash receipt及create-only输出guard，没有改NN/latent/seed/PCM。接口7386f6e4959a4b67c0ce7df019fbc80a0de91c01dc01927f1a14a6bfb109ca2c。CPU QA后请其准备最小6行controller与实际输入绑定，不开展新环境审计。

原Wav代理正在SOURCE_ONLY准备6行动态readout：直接每PNG复用483f FaceLandmarker.detect/mouth_features构造同VisualSequence，再用既有guard；无需额外无损转码。预编码PNG几何是primary，H264几何为codec对照，原自然PCM的H264 full/guard20 SyncNet仍用原矩阵合同。需decoded RGB body hash和非有效owned行跨臂逐像素一致性。原S17、绝对[-8,8]、固定p4、raw/z Cp/Hp及commonvalid normalizer全部保留。

服务器wjj@192.168.50.2:22保持运行，用户已取消关机。下一GPU启动前只做一次fresh进程/显卡核查和实际输入来源绑定；旧空闲snapshot不能代替。旧训练/中心/Dev/Confirm/归档支线停放；秘密永不写入文件、argv、BM或日志。

## Changed Files
- 新增CPU准入JSON；CPU/source-only/precodec/readout准备留在probe scratch，未作为已执行结果。
- 既有报告41、研究地图、Task18、专题、HANDOFF保留最新0/14解释修正。当前tracked dirty9项中本研究只触及CONTEXT，其他8项继承；untracked1003条，不删除或回退。

## Verification
- 完整阅读CPU source01及关键helper，确认filtered Muse加载、I+p映射、mapped行baseline m、同轴旋转、clock和非mapped不变及create-onlyCLI；已查看预编码renderer的精确差异。没有执行新候选，实际数组QA待做。
- 旧18条实际科学QA已完成：36矩阵/18geometry/7146比较maxerror0；PCM和PTS已验证；不重复这些资格。参考probe/root_actual_response_qa.json 72ef059b13a3a0cfd4f398b2f203224ec104a2072e18635fa5f124a26ad2e538。
- 当前Git只读确认branch/HEAD/root；GitHub repository API200、exact-head PR空、commitAPI422，HEAD为local/unpushed。未运行实现测试/pytest/compileall、依赖安装、云API、新MFA/encoder或新GPU实验。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- local/unpushed HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907`，无可验证远端commit链接或PR。

## Observations
- [decision] 先一次CPU实际科学QA，再判断最多6Muse短row；不扩模型/剂量/cohort，不改旧phase或补选峰。
- [decision] 预编码逐帧几何和有效mask之外RGB相同用于识别条件作用域，H264评分/几何单独记录，避免把codec传播归因UNet跨帧响应。
- [blocker] 新CPU候选尚未执行，6行stage/controller/readout也未完成实际输入绑定；没有需要用户提供的凭据或批准。
- [next_step] 将已准许的单次CPU物化交回原Muse Luna，收到6Npy/2clock/receipt后ROOT核对实际有效mask、映射和数学，再推进6行controller/readout。

## Relations
- continues [[Codex checkpoint - 2026-10-14T21-11-24Z - Native screen closed and output ownership gap identified]]
- tracks [[native-axis-factor-pilot-20261013]]
- relates_to [[研究主线与证据地图]]
- supported_by [[41-native-articulatory-response]]
- relates_to [[接收端局部构音响应坐标]]