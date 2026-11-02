---
title: Codex checkpoint - 2026-10-01T08-24-06Z - RMS mechanism confirmed and local
  acoustic response underway
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t08-24-06-z-rms-mechanism-confirmed-and-local-acoustic-response-underway
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 3566ac45eead923ae7ec1f765cff654652a256c8
started: 2026-10-01 08:24:06.909904+00:00
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

# Codex checkpoint - 2026-10-01T08-24-06Z - RMS mechanism confirmed and local acoustic response underway

## Summary

RMS方向/范数四格与mel分区已独立完成，主线转到WORLD音素内声源、谱形移植和逐窗局部口型响应。

## Story

用户要求从本地BM/Git/百度盘恢复并自主完成TTS增强TFG的机制研究；root维护理论与BM，三个gpt-6-luna/max代理实施实验。最新用户提醒音量/能量实验已有，要求局部声学原因。已有音量结果作为已完成基线，当前推进固定受体音素时钟和全局RMS的WORLD声源/谱形/局部能量双向移植，以及独立局部距离和视觉运动测量。最终报告25仍待这些结果，不能以代码或作业启动代替结论。

RMS07完整独立报告 `runs/recovered_20260930/independent_validation/rms_consumer_attempt07_pass_readback/independent_rms_factorization_report.json`，28494928B，SHA6371c540e15ff9f1a0ba8a0bda21f7836be5d92305d100e20712c02bdaa9afe6，root完整读回PASS，80句40人320四格零失败。guard20 raw C：direction+.161457[+.094298,+.232449]，radius+.019834[-.005550,+.046192]，joint+.189832[+.116516,+.273877]，direction−radius+.141623[+.076968,+.212289]；方向、联合、方向减半径在raw/unit×4supports×C/anchor全部16组观测及时间null excess均正。半径仅unit guard15两组为正，其余14组及所有输出I区间跨0。RMS与MEL q00/q11各80个音频特征/潜变量/视频像素及视频SHA逐位相同；该一致性不扩展到旧MAIN run。RMS BM已concluded。

纠正前checkpoint S的实际状态误读：RMS consumer退出值为0；worker包装器退出1来自最后检查错误的顶层字段。真实字段位于reconstruction_protocol且model_forward_performed_by_consumer=false，全80行均false。S将2字节exitcode文件误写为退出2；S保持不可变，本次明确纠正。独立源不导入producer或模型前向，最终报告与全部seal通过。compact sidecar v3 SHA06176334245bfe97925f027dceccac35a3f1214844b3553658d14aaf5f400249。

MEL07报告SHA112ff916b3c6b1790f3cf3db1b75721f08f84486aca0e16d828ec432bf4bc6fb，U-only raw+.171139、U−S+.179846，全部16组及时间excess为正；U定义两端均未裁剪，不等同语音段。女性独立null02报告SHA1bfa675eafce3595ac07b8a8d88d6d7f4efe75901ad092a3b8b447ddaf3da20a：生成G excess raw+.230148/unit+.031209、配对native差距缩小excess raw+.245392/unit+.035909，99%CI均正。三张已封存PNG由root实际视觉检查，manifest SHA c4b6df4411dda89cb40a3668932a6b851f0853959e4f35db82773a25f21e4eaa。

## Working State

WORLD GPU fresh v5位于 `/root/tts-research-system-20261001/world_phone_transfer_gpu_20261001_attempt06_v5/run`，worker PID460837/child460884，最新已核124/160 contexts、9424/12160 rows、0fail。五runtime flags通过prehook、CUBLAS前置；运行中初始sidecar零计数不是最终前向计数。源05 SHA275a3169c7c7531a01d5942bd06278843faae70724f0e9ae5b7cf8c90ac7a6ac，wrapper3 SHA001b5eefa0233ab70dd4187e1d65365c004e35aba23290faee4ccc3f28271456。最终独立consumer02待实际全产物；已修D0=curve[15]、search_gain=D0−Dmin，producer旧错误search字段原样保留；raw plan SHA与embedded SHA分开绑定，protocol selfhash独立核对。只读协议核查01因路径漏20261001失败，正在fresh02修复，未读效应。

局部spec/source `openspec/changes/diagnose-world-local-acoustic-response-20261001/` 与 `scripts/experiments/independent_world_local_acoustic_response_20261001.py` 在实现。CAL20仅预选16个exact音素标签（至少10clips/5speakers），保留薄支持标签分母。局部G为RAW固定anchor上RT视频距离减移植视频距离，音素权重按五个lip frames实际存储的16mel-column chunks并集计算；mel输入支持不等于STFT波形RF，不把各音素贡献称为global median C分解。shared100 full-A时间null、20k speaker bootstrap和多重比较门槛冻结后再读效果。MediaPipe −85ms仅视觉音素endpoint，评分仍anchor i+3。

CAL已核APQC SHA5ff39cfd82817f56eb9cc348e2d8481432973a09db7fd9a41ed9a280d91301c5：active∩jointvoiced映射AP差各频段接近0，旧全轨AP差混合了清浊音覆盖，不能直接解释为每个有声音素更干净。补充exactphone AP/SP频带与谱形动态CPU源SHA6b2f28d6062bdd3f65a27cf75f3b7aa4ce0a4c52647e8d5330321444b111bb5e，bundle `.codex/scratch/world_cal_phone_spectral_detail_20261001_attempt01/INDEX.json` 待root审阅派发。WORLD visual inventory03真实PASS，160grids/160maps/0errors，SHA7b3e879b9a34a6ea20159e770d6530fc77b19fd013381ce4017e87cbbc19203b；S0064 overlay N/T仅物理分析，原始保留。服务器无MediaPipe；隔离CPU runtime与模型资产部署待完成，禁止更改W2/WORLD环境。

本地/脆弱状态：authenticated PTY50274、SSH daemon93533/port3019/token仅内存；dispatch helper逐项保存response，必须核实际返回。8MiB下载上限，大报告gzip/分片。服务器授权入口root@connect.weste.seetacloud.com:12349，密码/API只运行时使用，不入笔记。AutoDL未发现现成百度挂载；本地B: CloudDrive可用于完整可恢复归档。已精确释放2692578028B缓存，不能重做删除；最新private空闲约2.17GB，需要预算CPU环境和后续输出。

## Changed Files

本地未推commit3566ac45eead923ae7ec1f765cff654652a256c8保存17项已验证科学源、CONTEXT及padding annex。`.gitattributes`固定科学Python LF；已执行plot SVG源码原始CRLF作单文件例外保持SHA，勿自动归一化已执行源。CONTEXT91行，ignored HANDOFF待本次checkpoint更新。既有dirty tests与其他失败/未完成源保留，不盲目stage。BM RMS/女性/MEL结论、主Task及storage Decision已写回验证。局部新spec/consumer和WORLD消费者仍untracked/未执行。

## Verification

root完整下载/SHA/全文审阅RMS07、MEL07、female02、endpoint bridge及三张PNG，校验RMS-MEL逐位端点与独立consumer不做模型前向；Git提交前保留源SHA并静态diff检查PASS。没有新增或运行pytest。checkpoint前现查root/branch/HEAD/status、whoami/hostname；同chat19条无更多/legacy0，直接读S核exact agent/session。GitHub public API核canonical仓库/已推baseline/当前branch无PR；本机gh不可用，使用只读API。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 本地未推HEAD `3566ac45eead923ae7ec1f765cff654652a256c8`。
- 局部设计 `openspec/changes/diagnose-world-local-acoustic-response-20261001/`；完整结果入口 `runs/recovered_20260930/independent_validation/`。

## Observations

- [result] RMS实测响应主要经过编码器方向改变；方向减半径在16组观测/时间excess均正，独立报告完整PASS。
- [result] S退出码误读已在本checkpoint纠正：consumer0/reportPASS，包装器1为错误字段位置断言，不是科学失败。
- [decision] 用户重点为局部谱包络、F0/UV/AP和能量动态；RMS基线已完成，继续WORLD移植及音素局部贡献。
- [decision] exact音素标签仅CAL选定；local G不等同global C可加分解；WORLD重合窗/相位/全局RMS使零参数覆盖窗不必零响应。
- [blocker] WORLD全GPU/独立consumer、CAL详细谱形、局部距离与MediaPipe视觉尚待实际结果，报告25未完成。
- [next_step] 完成并独立复核WORLD全产物、局部声学与视觉响应，再形成报告25及BM结论。

## Relations

- continues [[Codex checkpoint - 2026-10-01T07-35-29Z - Local acoustic mechanism and sealed RMS mel evidence]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[句级音量在编码器中的方向与范数响应]]
- relates_to [[WORLD 声源谱形与局部能量的双向移植]]