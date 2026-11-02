---
title: Codex checkpoint - 2026-10-01T13-33-32Z - Complete spectral results and independent
  local controls
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t13-33-32-z-complete-spectral-results-and-independent-local-controls
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 5f2ed0dbe11f8aedfcfc7303b898c0405650a017
started: 2026-10-01 13:33:32.074500+00:00
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

# Codex checkpoint - 2026-10-01T13-33-32Z - Complete spectral results and independent local controls

## Summary

完整WORLD结果已经逐字节取回并读完：谱形移植稳定改变音视频匹配，下一阶段以音素均值、内部残差与插值对照定位局部原因。

## Story

用户要求恢复并自主完成TTS增强TFG机制研究，最新重点是局部声学特征，已有音量实验不重复。root维护理论、设计和BM，三个gpt-6-luna/max代理承担实验代码。已完成的真实encoder RMS四格、mel分区、男女复制指出输入表征方向轨迹的稳定作用；当前WORLD实验固定受体时钟和其他成分，把谱形、F0/AP及能量分别交换，随后分解音素平均log谱形与内部轨迹。

本次完整WORLD独立报告实际下载、验封和全文解析。SP、SP_EM、BOTH的生成端G在每臂56个完整组合的99%点态区间全部为负，8个short-T guard20组合保留INCOMPLETE。RAW guard15固定anchor的SP G：T→N −0.224938124418、N→T −0.283170330524；四格交互分别+0.426002812386、+0.482327532768，均为正。因此谱形的具体协调方式影响生成与评分流匹配，不能把全部SP移植当成单向通用增强。EX并非全部零效应，F0有两个unit/RAW/guard20的微小正区间；AP56完整组合均跨零。所有确切数值、来源和适用范围已经写入WORLD实验与主Task并实际读回。下一步新五格比较均值、残差和SELF_GRID，避免将插值平滑混同TTS轨迹。

最终报告25仍需要完整局部16640联合端点、新800视频/11776联合端点、1600旧WORLD视频独立视觉测量，以及新RAM产物和必要输入的F盘持久归档。不能在启动、失败排查或排除项处结束。

## Working State

权威结果已在`runs/recovered_20260930/independent_validation/world_v6_consumer05_attempt02_root_readback/`，`root_readback_receipt.json` FULL_FIVE_MEMBER_SEAL_READBACK_PASS；全statistics SHA `b11f5530820ac82d654d77240bbb05b93ef2b6ab554a8e2c05e1b34fb9313b88`、13183106B，17024 observed和2432 null记录，20k×40speaker bootstrap为点态99%区间。所有五成员raw SHA由archive manifest和output seal/inventory独立绑定。机器身份以`.codex/scratch/world_v6_root_seal_metadata_attempt01_request.json.response.json`为准，不用旧checkpoint X手抄哈希；gate10全10887成员PASS。原科学consumer05退出0，packer02退出0；原失败21B包保持不变。WORLD v6 RAM run `/dev/shm/tts-research-20261001/world_phone_transfer_gpu_attempt07_v6/run`：160contexts/12160rows/5760matrix/1600videos/零失败，1043511 Module forwards flags全合格。

旧局部response06预检实际PASS后PID563429立即退出2：EX_EM没有parameters键，属于实际schema错误，不是科学阴性。真实80记录/160方向/9臂参数schema全量取回为`.codex/scratch/world_local_response_attempt07/actual_parameter_schema.json`。fresh source `scripts/experiments/independent_world_local_acoustic_response_20261001_attempt07.py` SHA `42081f3cf00de856c5a2d5bbfbeca786a2708d7ac2f822603b345ec2aadff920` 将EX_EM→EX、SP_EM→SP、E→RT，明确这些是能量处理前WORLD坐标；最终波形RMS、gain audit及独立物理QA另验，16640评分/CI数学不改。mechanism正准备fresh07包，root待审派发。

CAL五格40方向/200WAV producer已完整验封，报告`runs/recovered_20260930/independent_validation/world_phone_shape_mean_dynamic_cal_20261001_attempt04/calibration_manifest.json` raw `dc23a765794b0dd6e735be26955dbcb39e4ce06dff4fbe1be9f1c6b926ab7efa`。独立QA03/06实际先后因undefined payload常量、将原输入transfer audit误当新输出root失败，未产生完整科学QA。fresh独立QA07 source `scripts/experiments/world_phone_shape_mean_dynamic_cal_independent_qa_20261001_attempt07.py` SHA `7a47a27ceb827d6422100820d148a057980b5ca9d79de53efcce4ec3727e47b6`，使用准确input_inventory角色、原路径/size/SHA并独立重建数组；机制代理对exact source复核PASS。`.codex/scratch/world_phone_shape_mean_dynamic_cal_independent_qa_20261001_attempt07/INDEX.json` SHA `f453c7b747d800ba4f2bb2df10598e12711716e68b00179db07187e693c9ddb5` 已就绪尚未派发。200独立重合成PASS后再跑EVAL800及全独立QA。

新EVAL physical80/160所需实际meta绑定已全部读回`.codex/scratch/world_eval_root_metadata_bindings_attempt01_request.json.response.json`，物理manifest raw `27cb2cdf2246920c0667dd5bcc5d744253b3c6b7d407270c1254ffaf986aa3b3`、payload `eb77432936c3454a241eb2505a966ffa0d05aa5be8514451cc2da66aa77ed9de8`，原79+S0064物理overlay1，原视觉资格不变。新五格完整设计在读取WORLD效应前已冻结：`openspec/changes/factor-world-phone-shape-mean-dynamics-20261001/eval-design-root-freeze.md`及metric addendum，3456global+8320local=11776 joint max-|t|，20k×40speaker共享重抽样、100 full-A置换、短T guard20不救援。GPU/wrapper及完整独立consumer仍开发。

视觉隔离installer03实际因Wav2Lip Python没有pip退出，目标无文件。fresh installer04 `.codex/scratch/world_visual_runtime_attempt04/request_index_attempt04.json`及source SHA `0d40927eb6b578a46bbcf490ba81f981b1ded0948e73533c570cabf1819052cd` 已root全文静态审阅，尚未派发。它通过官方PyPI JSON和files.pythonhosted.org校验16个wheel并安全解压到RAM runtime08，不改原环境。先00/01/02实际stagePASS再03launch；完整receipt合格后旧10臂1600 tracker和独立视觉consumer。口开合/速度/−85ms音素时钟保持冻结。

旧v5 4.04GB E+B全读回及E全restore PASS，远端源保留。额外F复制旧v5已明确暂停，agent停止只读verify03之前未创建F目标，避免占新研究归档空间。新F归档尚未开始，remote先算精确文件字节和restore占用，目标`F:/tts-research-archives/20261001/`；需要v6 run/control/source/externalplan/QA/consumer及不可重生的自然/TTS实际音频、物理WORLD/mapping输入。冗余norm tree只保存实际引用的face registry/image等文件。全部RAM保留至消费者与完整archive/restore PASS；RMS已迁移和cache cleanup不重复。

易失机器状态：PTY50274持有私密bridge ENV，port2133，bridge05 PID217452；dispatch_requests_root_attempt02.ps1。服务器root@connect.weste.seetacloud.com:12349，ED25519 `SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0`。密码和API key只在运行时内存，永不文件/BM/argv/输出。三个既有代理先确认状态再继续，不重复启动。

## Changed Files

本地未推HEAD `5f2ed0dbe11f8aedfcfc7303b898c0405650a017` 保存已资格的CAL source/spec/plan、WORLDconsumer05/local06及LF规则；原dirty测试/旧consumer/validator不动。freshQA07/local07/新GPU/完整consumer/tracker仍untracked。BM WORLD与主Task已更新完整效应并readback PASS；HANDOFF仍旧X/运行中阶段，CONTEXT91行仍旧阶段，需更新当前快照，最终报告尚未创建。

## Verification

完整五成员tar安全解包到freshdir、所有member raw大小/SHA、独立audit/seal/inventory→statistics绑定、写入后再读回全PASS。全文读取WORLD统计和所有支持集/geometry/null状态，未用缩略摘要替代。局部06真实preflightPASS但分析schema失败；CAL独立QA06真实工程失败；视觉installer03真实依赖失败，均保留且不作科学结论。fresh07静态复核/compile PASS，未声称运行PASS。Router/实验指令/CodingSession/bm-writing已加载；同chat与legacy搜索完并直接核前Y exact agent/session；GitHub只读核canonical仓库/pushedbaseline/无PR，未pytest。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Local unpushed HEAD `5f2ed0dbe11f8aedfcfc7303b898c0405650a017`。
- `.codex/scratch/mechanism_synthesis_outline_20261001.md`：完整WORLD后理论主线与报告25纲要。
- `.codex/scratch/bm_world_complete_results_readback.json`、`.codex/scratch/bm_task_world_results_readback.json`：已写入并直接复核的记忆。

## Observations

- [result] 完整WORLD谱形移植G在56完整组合稳定负向，大正四格交互说明生成和评分流的匹配作用；不是通用单向谱形增强。
- [decision] 后续将音素均值谱形、内部残差与SELF_GRID分开，使用冻结11776 joint family，不按已见效应改端点。
- [decision] 工程错误保留原尝试并在fresh namespace修复；剂量别名明确处理阶段，不能冒充后处理最终波形参数。
- [blocker] 局部16640、独立CALQA、新EVAL800/11776、视觉1600及F持久归档未完成。
- [next_step] 派发已复核的CAL QA07和visual installer04，并完成实际stage、运行及独立结果验封。

## Relations

- continues [[Codex checkpoint - 2026-10-01T12-52-52Z - WORLD consumer passed and local spectral controls]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[音素平均谱形与内部轨迹的因果分解]]
- relates_to [[WORLD 音素局部响应与声学剂量]]
- relates_to [[研究服务器缓存清理与可恢复归档]]