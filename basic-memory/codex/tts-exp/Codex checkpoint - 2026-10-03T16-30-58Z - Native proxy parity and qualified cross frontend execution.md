---
title: Codex checkpoint - 2026-10-03T16-30-58Z - Native proxy parity and qualified
  cross frontend execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t16-30-58-z-native-proxy-parity-and-qualified-cross-frontend-execution
username: dogle
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
status: open
branch: codex/cross-tfg-audio-head-20261003
model: gpt-6.1-sol
capture: deliberate
repository: dogLeft233/tts-exp
started: '2026-10-03T16:30:58.5438429+00:00'
project: tts-exp
cwd: E:/Documents/tts-exp
trigger: compact
git_sha: a66c915023ad72493f2d79e415ab849b6ec1d675
repo_root: E:/Documents/tts-exp
hostname: LAPTOP-E133GSLE
agent: codex
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T16-30-58Z - Native proxy parity and qualified cross frontend execution

## Summary

360次原音评分已经独立复核并提交文档，Ditto完整import实跑成功；新完成的fit parity把代理差异定位到视频后处理/编码链，下一步实际执行MuseTalk240条共享开发并修正可训练代理。

## Story

用户最初要求从BM、代码、Git与百度盘恢复TTS增强TFG机制，随后授权服务器、重跑、数据下载和Qwen服务，要求持续推进直到产出适配较多TFG的可用自然音增强头。增强音h(x)只驱动生成，最终音轨和主SyncNet评分始终恢复原自然x，不重定时。报告25机制已完成；当前增强头目标ACTIVE，不能停在准备、启动或失败候选。ROOT负责理论/协议/BM/Git/运输和独立复核，gpt-6-luna/max子代理负责代码与实验。

历史8fit与10名新男性dev各2片段、3seed分开；40名confirm及heldout模型效果保持未读。共享STFT幅度残差保持相位/样本数，RMS和局部压缩为真实基线。实际Wav2Lip360行说明两个端点不同：v3相对RMS固定零时刻D0改善0.264、Canchor提高0.153，但Cbest下降0.105且背景B下降0.111；局部压缩Cbest较RMS提高0.033。因此保留原始Cbest和固定时刻指标，不能把零时刻改善包装成Cbest成功。报告25表明盲目移植TTS谱形会损害接收者，继续保留自然音局部结构。

新增fit parity attempt05实际exit0，使用一个fit/RMS片段、27中心查询、原音PCM16精确一致；代理和官方Wav2Lip输入/预测最大差约1e-6、最佳lag均+2，而视觉嵌入/评分差异仍较大。当前直接float帧评分与官方DIVX AVI→mux→JPEG链不同；需要同帧同编码复核后改可训练代理，不能把codec差异推断成模型机制。224上下文重裁与冻结face96差异是单独记录的诊断，官方此次确实使用相同face96。对失败的静态bbox假设做了新attempt04代码/attempt05运行修复，保留原失败。

Muse16实际工程资格已完成：扩大裁剪与贴回框一致、官方parser/Whisper、原音PCM与静音视频packet复制校验、本地图片复核。共享240开发方案4方法各60已冻结；ROOT发现未执行03的解包rmdir和receipt adapter SHA两项启动错误，子代理生成全新04方案/包，尚待ROOT审核后实际启动。Ditto isolated headers14和完整StreamSDK import12实际PASS，编译SO281080B、overlay与三个include目录可复用；实际portrait17 packet02准备完毕，未运行。工程成功不等于共享效果。

## Working State

持久协议在openspec/changes/develop-cross-tfg-audio-head-20261003/；CONTEXT93行与transfer-iteration已提交本地未推送HEAD a66c915023ad72493f2d79e415ab849b6ec1d675，仅两份文档。实际v3 fit/dev及Muse冻结包仍绑定历史d512876881fd630d4c8fb89920a031bdff72a144，不能追改运行来源。其他脏文件、旧tests和未跟踪实验代码保持原状。

机器局部：root@connect.weste.seetacloud.com:12349，固定hostkey SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；凭据仅私有PTY50274运行时，不入文件。ROOT dispatch脚本.codex/scratch/dispatch_requests_root_attempt02.ps1使用审核后request JSON，逐次读response判断实际执行。literal W2L/SyncNet venv不同，不解析其python symlink。GPU RTX4080SUPER32760MiB，NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003，SHM=/dev/shm/cross_tfg_audio_head_20261003。

当前packet权威指针：Muse .codex/scratch/cross_tfg_musetalk_stageb_attempt04/request_index_attempt04_retry02.json；Ditto .codex/scratch/cross_tfg_model_deploy_attempt17/packet_attempt02/request_index_attempt02.json；parity .codex/scratch/cross_tfg_audio_head_20261003/proxy_native_parity_attempt04/packet_attempt04/09_readonly_status_and_summary_request.json.response.json。parity summary18033B SHA90c2c2125cc266ff7c33fc76abb1d10f2b738d525a8288af085aa39e4bec6f2a、NPZ16453262B SHAe198cc3faacc00a0b39cbe5df9e6d5181596b3097487f9fd11e6545525d4f3d2，子代理已读，ROOT尚待独立取回。

旧300嵌入replay实际COMPLETE_PASS/300矩阵exact/0失败，尚需归档和V方向/范数分解。实际只读storage probe：源302文件238625360B，JOB free379387904B，SHM free8936124416B；旧pack同时生成archive和chunks约477MB，不适合JOB，已要求新SHM包，禁止仅凭准备声明归档。W2L新v3独立motion60将与旧W2L identity60匹配，绝不与Muse视频混配；v3还需要Qwen内容保真，旧100 CER0不能覆盖新v3。交付还需≥3不同前端/模型、heldout、新人confirm、权重CLI/时延/音频内容与独立运动护栏、报告26。

## Changed Files

- 已提交：CONTEXT.md；openspec/changes/develop-cross-tfg-audio-head-20261003/transfer-iteration.md。
- 本机未跟踪候选：Muse shim/launcher/builder04，Ditto portrait17 packet02，proxy parity04 packet03/source。HANDOFF.md为ignored快照。
- BM既有可变Task/Experiment快照15/14仍active/implemented，下次应更新完整import与实际parity/共享运行；immutable BV保持原样。

## Verification

ROOT实际核验joint归档14regular成员、输入绑定、3600条curve→metric与28个seed→clip→speaker端点，误差<1e-12；新60原始query矩阵尚未ROOT本地重算。360结果SHA8b55aa79d4fb518ea053508405ed3951423ba24c6324bd2ddc2773cc622c898e；归档SHA966fbfb6d6b04a39a5436f87821c5117524e95bf7c3aba570e21b871fa99d4cd。guard20保持原计划不完整，不插值/默默缩分母。

ROOT实际读header14全164文件canonical tree和import12报告：PASS_FULL_STREAMSDK_IMPORT_ONLY，报告14900B SHA2ba259045dd79c797f7c0e34403f600ab1d0accc40fa03ca2aa820c8c80f7806；编译warning非失败。Muse16三文件bytes/SHA/PCM与packet流、几何和本地图像实际复核PASS_ENGINEERING_ONLY。Parity14source/8fit preflight实际PASS，05进程已exit0，数值摘要由子代理取回。未运行pytest或单测；科学运行/QC是用户已授权实验。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)，当前分支无PR；[verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)。当前HEAD `a66c915023ad72493f2d79e415ab849b6ec1d675`仅本地未推送。
- .codex/scratch/root_joint360_actual_curve_and_hierarchy_readback_attempt01.json、root_v3_fit_terminal_readback_attempt01.json、root_muse16_three_files_geometry_and_decode_readback_attempt01.json。

## Observations

- [result] 真实v3改善固定时刻同步，但Cbest弱于RMS；局部压缩仍是当前Cbest最佳共享候选，跨模型效果尚待实际运行。
- [result] Ditto完整import12和Muse16原音换轨工程已实跑成功；fit parity05发现同预测的视频后处理/codec差异。
- [decision] 同帧同编码复核代理再训练v4；共享方案冻结且维持原音评分/未读confirm；不会将工程失败或准备状态当科学结果。
- [blocker] Muse240与Ditto portrait尚未实际运行；代理输出链差异及共享效果是当前需要解决的问题，空间问题通过SHM新包继续推进。
- [next_step] 审核并实际启动Muse04共享240开发，同时执行Ditto17 portrait、取回parity NPZ并完成同编码诊断，依据真实多前端结果迭代共享方法。

## Relations

- continues [[Codex checkpoint - 2026-10-03T15-50-57Z - Dual endpoint head results and qualified native transfer]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[008-fixed-lag-and-rms-reference-head-objective]]