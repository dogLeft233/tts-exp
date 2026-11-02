---
title: Codex checkpoint - 2026-10-26T06-33-38Z - Single source replaced and full fit
  held on five ASR cases
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-26-t06-33-38-z-single-source-replaced-and-full-fit-held-on-five-asr-cases
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-26T06:33:38Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a175f1-7164-7510-ae4b-f5af8e9d0fde
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-26T06-33-38Z - Single source replaced and full fit held on five ASR cases

## Summary
用户批准的同人单句替换及完整拟合集复验已完成：300重建和1200对齐通过，完整配对ASR有5条超原逐句门，继续HOLD。

## Story
原目标是逐音素静态/动态/联合TTS差迁移的11臂自动验证，先在100fit完成技术校准。前轮仅S0168W0257重建0.749654未过0.750，且固定镜像来源存在完整文字覆盖问题。用户现明确授权按推荐规则换一条、保留原失败、补齐TTS/MFA再完整复验；这只修订旧不剔除约定中的这一条，未授权按分数不断换样。开发及确认仍不动。

权威入口 runs/phoneme_direction_20261026_repair04/report.md、technical_status.json。固定从W0257之后升序查未用同人WAV，先冻结selection_contract，双独立识别核全文。W0259、W0262识别有文字分歧，保留为NOT_VERIFIED，不宣称真实读错；W0263“中国一直以来被称为世界工厂”是第一条Whisper-small与Paraformer全文均精确一致的候选。选择先于重建分数，原600ID无重合。新版input_manifest_v2.json保留1196旧记录完全相同，仅换自然+3次TTS，仍100人各3句。原600冻结清单不改，原W0257波形和旧失败均SHA核实保留。

补齐同权重Qwen1.7B full self-ICL三draw，原种子、temperature.9/top_k50/top_p1/max512/non_streaming_false保留。生产GPU及流水线锁正被其他聊天使用，没有改动或停止。将生产synthesize函数做明示CPU派生：LM/codec device_map cpu、GPU runtime字段None、torch线程4；原函数/派生脚本SHA及差异在cpu_tts_runtime_amendment.json，未证明GPU逐码等价。三个draw均EOS，时长比.972423/1.166907/.923802；新TTS配对参考识别均0错。没有训练TTS或使用云API。

继续同一普通HiFiGAN+peak95+固定L6有界后校正候选，复用299不变L6及peak解码资产，替代句新提取/解码。五折同人共折、每折80人240条拟合20人60条检查，全部五个头重新拟合；旧失败来源不进入任何训练目标。全部300输出重新生成/独立保存WAV测量，STOI中位0.9003049702382695、最低0.7567519989844749，零基础失败；替代句0.8950050851731165。原长度/finite/削波/RMS/lag等全部通过。小型Ridge实际拟合，WavLM和声码器未训练，候选仍未准入共享正式适配器。

新MFA603条：300自然A+300重建R+3新TTS，旧897TTS grid复用SHA再解析；完整1200输入结构零unknown/边界/缺失/非单调错误。300对A/R音素匹配.9963039963039964、中心偏移绝对值中位0/p9515ms，通过原.95/20/40门；自动MFA一致性仍非人工真值。

首次把原先20条配对ASR检查扩到全300对，5380参考字符，A154错/R179错，CER.02862453531598513→.03327137546468401，增量.004646840148698882通过总体.02门。但S0014W0121/S0015W0252/S0023W0389/S0095W0408/S0236W0338新增6/6/3/3/4，超每条2字门。三例出现sander/define/bang式英语拼音输出，原norm保留拉丁字母计数，不临时同音归一化；这不是5条STOI失败，也尚非人工确认丢字。因此当前HOLD_PAIRED_ASR_PER_SOURCE，未冻结、不再换样。

## Working State
远端repo /home/wjj/tts-audio/tts-exp，run同名repair04，CPU所有本聊天作业均完成。没有后台监控或新自动化。其他聊天生产任务未干预，进程参数显示其仍处理自己数据，但本实验未读dev/confirm媒体或效果。

本机交付1319文件/22960015字节逐SHA通过，包含完整新TTS媒体、三条查验候选、603新TextGrid、603 ASR单条输出、五折头、源码/报告/JSON。其余299重建WAV、旧特征和MFA数据库留Ubuntu。delivery_manifest SHA 7d788bacd4806a7ca532b8dcd96911649b874c3533c36ac3a4916be0f00b4e9d；本地delivery.tar.gz 6924793字节 SHA6caccae8fd3acabd9e2b29bd4d119e6b069dcb6f4601117e2702e125d103aea1。完整源/模型读取证据在model_source_readback.json。旧repair01/02/03共101交付SHA未改，old_failure_final_readback.json核实原失败完整WAV仍在未变。

仓库1279条dirty/untracked，很多来自其他聊天；本聊天源码也未跟踪，未commit/PR。不要reset、清理或批量提交。当前HEAD本地验证，未核实GitHub可解析。GitHub只读确认仓库和当前分支无PR。open_in_codex报告返回queued，不能声称已显示。

## Changed Files
- scripts/experiments/phoneme_direction_20261026/replace_fit_source.py：严格四单元替换、新清单、替代重建、完整五折重拟合和300读回。
- scripts/experiments/phoneme_direction_20261026/validate_replacement_fit.py：603新MFA/1200结构再核、300对时序、600配对ASR+新3TTS。
- tests/test_phoneme_fit_replacement_20261026.py：保留1196记录、禁止跨split/错speaker、缺draw和文字不一致拒绝。
- runs/phoneme_direction_20261026_repair04/：权威结果和source_snapshots；运行时CPU派生合成与内容核验脚本同目录。
- basic-memory/Research/逐音素差方向算法与验证方案.md、Research/研究主线与证据地图.md、Experiments/逐音素静态动态差的拟合集技术校准与40人验证.md：新版规则、最新数字及HOLD，MCP写后读回通过。

## Verification
远端相关23测试通过，本机17通过/2依赖模块跳过；本机compileall和限定diff检查通过。实际执行CPU三TTS、300重建、603MFA及全1200结构重读、600对ASR再加3TTS，所有执行退出0，科学/技术总门仍因ASR失败。新1319交付SHA、旧101交付及原失败WAV SHA通过。WavLM checkpoint 6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f、普通HiFiGAN f98b760e0e5fd0019cffd3a9d22bab5c4c2fe38491532d7680a8ad07eaf3e8dd再次实际读回。未新TFG/SyncNet，无模型训练或参数冻结。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 当前本地HEAD `2b71639c3b555293375d248350999377168d515d`；未构造未验证远端commit链接。
- 最新结果 runs/phoneme_direction_20261026_repair04/report.md、input_manifest_v2.json、full_reconstruction_quality_v2.json、alignment_full_fit_v2.json、roundtrip_timing_full_fit_v2.json、asr_full_fit_v2.json、local_delivery_readback.json。

## Observations
- [decision] 用户明确批准一次同人内容问题替换；原门、失败凭据和dev/confirm限制保留，不按分数再换。
- [result] 替代句W0263及三TTS完整，300重建零失败/中位.900305；1200结构与300对自动时序通过。
- [blocker] 全300对ASR有5条新增编辑超过2字门，HOLD；拉丁输出形式影响未分离于声学损失。
- [next_step] 核对这5条局部原音与重建及独立ASR响应，区分声码器声学损伤与识别器输出形式；不改门、不换更多样本、不推进开发/确认。

## Relations
- continues [[Codex checkpoint - 2026-10-26T06-15-00Z - Single fit failure confirmed and source replacement under discussion]]
- relates_to [[逐音素静态动态差的拟合集技术校准与40人验证]]
- relates_to [[逐音素差方向算法与验证方案]]
