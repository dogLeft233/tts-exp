---
title: Codex checkpoint - 2026-10-01T06-10-58Z - Female RMS replication and qualified
  physical transfer
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t06-10-58-z-female-rms-replication-and-qualified-physical-transfer
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 2b78a03dfe32c4702dde15b573087563ee1e2cff
started: 2026-10-01 06:10:58.023962+00:00
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

# Codex checkpoint - 2026-10-01T06-10-58Z - Female RMS replication and qualified physical transfer

## Summary

女性独立复核确认音量路径在新说话人上复现，WORLD物理波形已通过完整重建，继续完成真实音量潜空间分解与声源／声道因果迁移的评分。

## Story

用户从恢复本地Git/BM/百度盘历史继续到全面解释TTS增强TFG，授权实验重跑、Qwen TTS、下载和存储搬迁。root维护理论主线与BM，三个gpt-6-luna/max代理执行科学代码和实验；不要在排除项或启动作业后结束，报告25仍待完整科学结果。服务器准确入口root@connect.weste.seetacloud.com:12349，密码和API key只在运行时，永不入文件。AutoDL只读挂载/工具探查未发现百度/FUSE/rclone连接，本地B: CloudDrive仍可用约1.27TB。

前驱P的男性80句/40说话人双脸RMS结果、配对原生差距缩小及随机方向时间null结果保持有效，权威指针在既有native/aux独立报告。女性fresh consumer04只修实际seal字段名，完整通过40句/20说话人/640行、40context、120随机臂F32剂量/KKT与全V×A重建。root实际下载、SHA和读报告 `runs/recovered_20260930/independent_validation/female_consumer_attempt04/independent_female_summary.json`：3,504,578B/SHA cd31ca9541d690ebd34b557215209e094d3a0431aaedaf9a2af4fe45a51062d6。guard20 natural生成端G raw C +.250266[+.116708,+.417378]、unit +.034505[+.017046,+.056129]，C_anchor相同；TTS G raw -.043149 CI跨0。原生T−N raw +.530425，q11 +.258817，配对缩小+.271608[+.137158,+.424423]；unit缩小+.039661[+.018510,+.060905]。来源方向plus−baseline主raw -.004892 CI跨0，plus−minus unit +.006876[+.000233,+.012397]为次级条件响应，不能忽略也不能改称普遍增强。正式四门槛未满足且eligible true。female02/03实际失败保留，不以exit0替代科学PASS。代理继续计算female全支持native/G/time-null配对补充；generic旧search_gain=C错误须以独立D0−D侧车纠正。

WORLD physicalQA04实际exit0、80pair/160recipient/1,920物理WAV/1,920gainNPZ、0fail，全RAW与RT/EX/SP/F0/AP/BOTH/EX_EM/SP_EM/E按冻结producer逐字节重建。报告 `runs/recovered_20260930/remote/world_physical_eval_qa_attempt04/independent_physical_waveform_qa.json` 669,127B/SHA e78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d，root读回PASS。04仅修consumer E臂生成RT包络与固定source-phone ratio契约，producer与容差不变，02/03失败保留。E local energy residual median-context p95abs1.52dB，pooled-context p95 6.75/max20.71；EX_EM .198dB/SP_EM .101dB。E为冻结有限迭代transfer recipe，不能称精确包络匹配或据残差筛样。WORLD05 CPU preflight160context/1600V/5760matrix/12160alias完整；GPU尚未启动，代理准备唯一worker/control和storageguard，排在真实RMS之后。

MEL producer80/320完整，独立consumer03因字符串双q前缀库存判断失败：旧打印16键是symmetric difference，实际所有CAL40/EVAL80 NPZ都只有13个精确键M0/M1/U/S/mel_starts/q00/q01/q10/q11/chunks_q00/q01/q10/q11；前驱P的别名解释已由Task纠正。consumer04修prefix后实际遇RAW PCM hash字段缺失。05只读实际前4/split契约证明file SHA同时绑定plan/artifact，decoded int16严格16k/mono/framecount，与recorded float32 waveform/fixedwave SHA精确一致；CPU plan/artifact从未seal source_pcm_array_sha256，04错误比较缺失字段。fresh05须绑定实际file/waveSHA并严格逐sample验证PCM→wave映射，把derived PCM SHA仅记诊断。全120行readonly01实际heredoc缺换行SyntaxError，未执行科学checks，代理生成新02。所有冻结mel/clock/array/gain检查保留。

真实RMS07 CAL20/prepare80 actualPASS，root已下载报告SHA/read，GPU07却在输出manifest前KeyError face_registry_raw_sha256，没有模型完成结果。source03 SHA120edb032594ce3aacde6f1bf4bcc2da36443de8b88a4990a0d72b37858ef85b只改该字段从raw assets→normalized assets，root实际diff一行，数学不变。08source/workers/stageactualPASS，但CPU08launcher PID331812瞬时exit2：prepare worker QA_OUT误指cal_geometry_attempt08，launcher/stage/status绑定attempt09，已有路径检查提前退出。实际08status files/pid/start/exitcode/PSF provenance存在，no science completed。代理需fresh09全path一致并审阅launch/worker，CAL20/prepare80完整PASS再GPU320。RMS独立consumer须适配实际freshpaths/源seal，并验证MELq00/q11 z0/z1 bitexact；旧MAIN→MEL和旧axis→norm端点桥接仍待做。

新axis sensitivity02 C/anchor/B/D实际全支持读取：plus−baseline raw/unit两脸各支持99CI均跨0；secondary raw plus−minus g15 +.028479[+.008061,+.050145]保留。02抄旧wrong search_gain=C，不能用于search结论；fresh03源码7b28e641a43c5ff3e2912c5243f07bbbbfff6b473c4686efcb37b319fb705f04已实现480 NPZ/fullcross重建并独立D0−D，待新immutable CPU执行。padding-domain-annex.md SHA5adad4efc3e4411e335222bf4625a03538ae2d6fc5b0cae245fdec658ac2eba2追加legacy/full-A匹配null契约，原spec与报告只读。

## Working State

fragile runtime：SSH daemon exec93533/port3019、root authenticated PowerShell PTY50274；token只内存。dispatch_requests_root.ps1逐请求保存.response.json，遇失败必须停止依赖mutation。8MiB下载限制，大JSON需gzip/7340032B分片。当前PTY正在cache03 source/plan/worker与preflight-only launch队列，尚未root读preflight，严禁直接apply。GPU空闲，顺序RMS→WORLD。CONTEXT91行、HANDOFF ignored且部分旧；原dirty tests保留，未新增/运行pytest。科学attempt与scratch大量untracked，HEAD2b78a03本地未推；本次直接git/GitHub核验canonical repo、baseline与当前分支无PR。

storage stage1已按18精确白名单释放807,202,840B，rootreceiptSHA/readPASS。repodata structure03实际完成readonly：28已识别JSON1,885,375,188B符合conda schema；28未知infoJSON13,031B保留，无parseerror/missingsibling。严格plan02 SHAea8ec160ff174f1ebf8789d5dc5ba53610e4429f560391d3955ec84c9914f1be实际root逐member核对pathregex/bytes/hash/count against scannerPASS/no-deletion。purger03 SHA361b97d6d9bb503c6de74327150809084b3e1939af253feff47d65c075973add先只读preflight，必须读实际PREFLIGHT_PASS28后单独apply并读receipt。当前overlay约4.08GB，预期cache释放后约6GB；WORLD持久产物预算1.67–1.82GB+transient，无需先删NPZ。cache-migration两树已完成，原路径为symlink，禁止再次finalize/rmtree。科学/环境/权重未删除，尚无归档。若用B盘，tar/memberSHA/readback/restore完整通过前不能删源。

## Changed Files

root更新BM native gap/random/typedstorage Decision/主Task与ignoredHANDOFF，纠正Task的mel库存误述；female/world/RMS最新成功失败仍需追加BM。新科学consumer05/RMS03/QA04/axis03、paddingannex未跟踪；`.codex/scratch/mechanism_synthesis_outline_20261001.md`为最终理论提纲，报告25未写。最终报告必须同时保留正向RMS路径、评分/可见运动边界、selectiveaxis响应、energy残差和历史英语/codec纠正。

## Verification

root实际SHA/read female04、WORLDQA04/energy诊断、RMS07 CAL/preflight、RMS07启动失败、cache结构03与28项plan；已核验08stage及退出status。编译仅代理实际py_compile，未pytest。科学完成与操作完成分开；旧失败attempt不删，不把实现/启动冒充结论。BM Startup Router→实验指令→CodingSession及writing skill重新加载，sameagent/session搜索16条无更多、legacy0，直接读P精确metadata。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 本地未推送HEAD `2b78a03dfe32c4702dde15b573087563ee1e2cff`。

## Observations

- [result] 女性20新说话人的生成端RMS效应与配对T/N差距缩小重复成立，支持音量路径为主要可复现因果机制。
- [result] WORLD160recipient冻结物理波形重建全部通过；E残差高于EX_EM/SP_EM，后续解释需保留这一区别。
- [decision] 严格绑定实际file/array/schema契约；每次freshattempt保留失败产物，不松数学或选择门槛。
- [decision] 28项repodata只在实际preflight通过并root读取后apply；未知infoJSON及科学NPZ保留。
- [blocker] RMS08worker路径提前退出与MELfullreadonly01heredoc失败已定位，fresh修复和真实GPU/独立统计尚待完成。
- [next_step] 修复并执行MEL05/真实RMS09完整独立验证，读cache03preflight→apply释放空间，再WORLD GPU/consumer和female全域null、端点桥接，汇总报告25。

## Relations

- continues [[Codex checkpoint - 2026-10-01T05-30-34Z - Paired mechanism evidence and fresh independent consumers]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[女性说话人上的 TTS 来源方向独立复核]]
- relates_to [[研究服务器缓存清理与可恢复归档]]