---
title: Codex checkpoint - 2026-10-11T03-05-00Z - Ditto center running and raw payload
  QA prepared
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-11-t03-05-00-z-ditto-center-running-and-raw-payload-qa-prepared
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-11T03:05:00Z'
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

# Codex checkpoint - 2026-10-11T03-05-00Z - Ditto center running and raw payload QA prepared

## Summary

k05 的 Wav2Lip 与 MuseTalk 已实际正常退出，同一中心的 Ditto 正在运行；只读终局核验来源已完整上传回读，后续独立原数组和原音窗口捕获正在准备。

## Story

目标是解释 TTS 的局部声学增强机制，做出跨 TFG、未见说话人超过固定 RMS 的自然音频增强头。增强音频只驱动视频，完整原自然 PCM 和原时钟用于评分与交付。用户取消关机并明确恢复四个中心评估，服务器继续运行；归档18/fresh07/final04/关机分支停放，tfg heartbeat 保持 ACTIVE。ROOT 维护理论、冻结协议、BM 和独立实际 QA，代码与实验沿用原 GPT-6 Luna/max 代理。

20-pair 联合 SPSA 的121链事件、20次 UPDATED、全原数组与 tell 数学已核验，journal 已封存。k05 的120计划行/40样本/20说话人/真实 tell5 已独立重建，唯一 execute 仍自然执行。Wav2Lip 和 MuseTalk 已有实际整数0进程收据；Ditto 仍活跃，因此不能宣称完整中心通过，更不能选择中心或宣称超过 RMS。当前工作使后续完整科学核验具备准确来源与字节捕获路径，未改变科学端点。

只读 QA helper 调用冻结 finalingest07/nativeadapter02，远端实际打开既有 NPZ/maps 并进行逐行 canonical Torch CPU 归约；它不调用模型、不更新 journal，也不做最终共同支持集选择。输出只嵌入终局 JSON，NPZ/map/WAV 仍是 pins 与窗口元数据。ROOT 必须独立收齐正文，重算七数组、官方时钟、原 float32 PCM 和120个原音 PCM16 窗口，不能把 helper PASS 当独立正文 QA。

## Working State

权威入口为 HANDOFF.md、active Task《开发跨 TFG 的自然音频增强头》和封存科学原件。主机 wjj@192.168.50.2:22，JOB=/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/job_control，session09 ID fit40_spsa_bfa027430723_6800c380298d。原 QUEUE cwd/source_root=/dev/shm/fit40_queue_remote_package_attempt02/project；package12 实际 package11 runner08；lexical Python=/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01/bin/python。保持命令的 lexical 路径，不改写 symlink。

唯一 k05 execute 由原 fit08_resume 持有 local exec94568，PS269032/SSH268012，启动2026-10-11T01:44:31.5270127Z。不要 ROOT 并发 poll 同一 handle、kill 或重复 launch。dispatcher1417095/PPID1417089/ticks16787748，dispatch=/dev/shm/fit40_joint_spsa_full_centers_attempt01，point=centers/fit40_d0_spsa_center_k05。packet=JOB/fit40_spsa_center_packets_attempt01/k05_attempt01/packet_index.json。实际整体 execute 返回码尚未捕获，整体 terminal 最后实际缺席。

W2L wrapper1417327/ticks16802975，01:47:04.011828Z→01:59:05.550919Z，actual child_exit_code 整数0/timed_outfalse，terminal raw360bf161110f4e9e562bc939298651dfaeb85cd8950a3df26f7bb729ebd91efe。MuseTalk wrapper1438655/ticks16875214，01:59:06.405559Z→02:43:37.552466Z，actual 整数0/timed_outfalse；terminal738B/raw352ee7c236709210554bd3b179d0e3e28d41c9e251fbcf38565acfdf444c9607/self4f01411f325189b189ed609474e3c4fadd7f19544e35a54810acd61b8a04e33a。Ditto wrapper1467248/PPID1417095/ticks17142423，于02:43:38.494178Z开始；ROOT02:52:18.706206Z 实见 renderer1474904/PPID1467248/ticks17193610/stateR，样本 BAC009S0057W0495。样本标签只表进程进度，不是完整评分计数。

ROOT 实际 captures06/07 在 `.codex/scratch/root_k05_current_readonly_attempt06/`、`attempt07/`；07 stdout27371B/raw17de14827ce19c68f0cc0f2abf268ba74f23d3f17add79614ba4d7d9eeec24e9，actualSSH0/stderr0/完整response及内嵌body核验。阶段 QC `.codex/scratch/root_k05_muse_complete_ditto_running_actual_qc_attempt02.json` self8c041389baee8e9eb5d368b454e7fe243bb59433354ab5b71d4f3ddaccc5c803，独立核验两模型实际0、开始结束及同一dispatcher/Ditto身份。QC01本地字段误写导致输出前 KeyError；正确键 started_sha256，未影响实验。

只读 helper `.codex/scratch/fit40_center_k05_terminal_qa_attempt01/readonly_point_qa_attempt01.py`29481B/raw5338df719ca610ac4d4a24da8dddc1765d0ff13eea91acfbe6328ca0ec2324e4；wrapper476B/raw9f659c5d205281aa8739b97e0549266c674cef81ae0215515a726d12edbc3b68。已在精确新 JOB/fit40_center_k05_terminal_qa_attempt01 create-only stage，02:20:43实际SSH0；02:21:21另读回两个完整base64 body，actualSSH0/stderr0。ROOT staged-body QC selfd450b6a819844d1c1ba63abce4448b71ea4f2b9b5388ccb14b35f7fff8dfa6f0。wrapper 显式 cd/PYTHONPATH=QUEUE/PYTHONDONTWRITEBYTECODE=1，按 bash 调用600模式文件。已条件释放单次只读QA：仅同一 k05 COMPLETE、原 execute 实际0、dispatcher及所有children退出后运行；当前未执行。

原 ASR reviewer 正准备窄 raw-reader SOURCE_ONLY；复用 accepted family03 body（2584596B/raw9f7914033f9e7e89a60ae2618f7d429860d11de902a8400855fe57e37eb216a2/selfb071434696b2c094180473677d484934d7e3d04c808b7915000c8cfc6c6abb0c），从 scientific_input_pins natural_pcm_* 获取精确源 WAV bytes/hash，不新增无必要 SSH bootstrap。准确容量=sum唯一payload+实际副本+metadata reserve 后选择 E 或 C 新 create-only task dir；已接受旧body按 exact bytes/hash 引用。reader 需实际 rc/stdout/stderr、每正文完整 bytes/SHA，必要时仅实际超大文件4MiB有界分块。当前无 reader 执行准入。

Dev source-only QA 合同 `.codex/scratch/fit40_center_k05_raw_payload_qa_contract_attempt01.json`15829B/raw5e6f7cc4b17d68fa4440d0c4c5a1f4f6e159829319fcca4caa54d8cbb0fe0faf/self6afb33cd7dbe1cba8373cf03c97039a2e96eaba74cb3f96aa071f7a6c5272037 已全文读，未声称实际数组QA。查询 N=max(0,min(map frames,window_N//640)-5)，矩阵必须恰好[N,31]，所有query数组恰好[N]；时钟=source_frame/25，PCM start=source_start*640，stop=min(source_stop*640,原N)。七成员 dtype/arrayhash、FULL/guard20/anchor15、全部window body须实际重建。

k10 prepare 仅 SOURCE_ONLY：`.codex/scratch/fit40_center_k10_prepare_attempt01/sealed_source_attempt01/` shell2925B/rawaba0662f7e3b58b9a107a61f63b4e58afa0c189a3e83629db9f525a1fed7db60，harness4116B/rawfc75d76cf6e2957be4b6a683df5d94c2a0f668f90d0709f3b6145b8bc660e00a；manifest02 raw42ac8de6586c48ae5f49bb44377967b0198fbb9c2680dca4f820ad95e764a04a。ROOT按原k05全部替换独立重建完整bytes通过，QC self335df3de8b292908269ef82d7aa3397b3deb06ddc8bd6fdadbd557fb9a9cd251。未promotion/远端create/prepare/execute。未来先 k05 全正文QA通过，再create-only复制两原文件到parent（harness实际从parent读取shell）、全body回读、fresh证明远端k10目标缺席，另释放prepare和实际packet QA。

封存journal=JOB/fit40_durable_spsa_controller_journal_attempt01.json，16698298B/raw2dd1c1ac08e439496c1f7753ab4bb8ccdd8727d2fc6a53aebeffd145b2167231/selfb731f233eea49ca533f4f21114eff09e342036b7287f63d172b112fe2f8d3b4d；本地 `.codex/scratch/fit40_full20_math_qa_attempt01/journal_full_readback_attempt01.json`。全20数学QC self7660be339b3ada267901fe9d97bf54a6f71afa29d782d5b5c17f1c4ff53c98c3；1200unique NPZ/1920uses/20tell/SPSA/PCG64/9theta误差0。禁止重跑或rewrite。原 coordinator OS rc 未捕获，不补造。

后续冻结 k5/10/15/20各full120→final720→portable→sealed540新Dev/Confirm。Fit准入后修 Dev draft06/builder05/analyzer04/nativeadapter03 固定旧引用，封实际candidate/runtime/base360 inputs，不能复用旧360或提前读Confirm。原PCM/钟/cohort/policy01/ADR031/ASR/视觉/bootstrap门不变；完整9门失败继续独立11谱形/局部功率。旧AutoDL DirectD0 UNKNOWN 不重放。秘密永不文件/argv/BM/log；不改sudoers/CloudDrive/安全设置，不绕过删除拒绝。

## Changed Files

本轮机内 QA、两个完整staged-source回读、geometry分析、k10 SOURCE_ONLY 和 raw-payload合同位于 `.codex/scratch/`，均属 ignored/local、未push；活动process、SHM/source/runtime也是脆弱机内状态。Task/HANDOFF已同步Muse0/Ditto运行、精确stage和geometry；本checkpoint只更新HANDOFF最新immutable指针。原科学文件未变，未commit/push/PR，未运行实现测试。

## Verification

ROOT完整重开实际SSH0流、base64正文、raw/self，确认W2L/Muse两个真实整数0、no timeout、start/terminal PID/ticks与bindings、Ditto及dispatcher固定身份。stage与独立source读回两次SSH0/stderr0、两个正文bytes/hash和本地原件一致；不能仅把manifest当正文。来源/导入路径审阅和k10 exact-source派生核验已通过，无额外GPU launch。

参数 geometry `.codex/scratch/fit40_theta_geometry_analysis_attempt01.json`22444B/raw91d577a6afd96322eeadfcf03878103ffb304ce9eccb0deadcc34f9e548162db/self5b255da03b3e1b5c2cc23c201ed721ac5d583424fae8b4d8f48075d837f30645；ROOT独立五输入body/raw/self/entries[k-1]/9theta复算，QC self58eb4478ad60ff96b7e0559b612c0461abc1d9fbffce052a7a32b29099674039。k5/10/15/20相对init参数L∞=.050033/.087398/.088751/.094557 dB，只是参数geometry。静态谱权重也会因频带组成变化改变局部功率；STFT无权logmask去均值与全局RMS不等于保持局部能量。11方法显式功率分支需在谱common-mode投影外且为时变包络。尚需实际40ms波形读出和因果对照，不能把该推理当已确认机制。

Startup Router、实验指令、Coding Session/Task schemas已加载；本checkpoint按准确 agent/session metadata检索同会话全部分页、直接读最新有效started并校验身份。GitHub规范仓库、branch PR和commit可解析性本次只读确认，dirty列表随checkpoint保留。

## References

仓库 [dogLeft233/tts-exp](https://github.com/dogLeft233/tts-exp)；HEAD `5688784dd5c6c1272074f0c4b6c5ff79f117c907` 为 local/unpushed，不构造未push commit链接。当前branch无PR；完整Git/规范仓库查询保存在本checkpoint的scratch evidence。HANDOFF与active Task是恢复入口，source pins和执行收据可沿上列路径核验。

## Observations

- [result] k05 W2L和MuseTalk实际正常0，Ditto在同一dispatcher下执行；120行科学终局尚待完整正文QA。
- [result] 只读终局helper/wrapper已精确create-only stage并完整独立body回读，显式原QUEUE导入路径保持。
- [decision] 用户恢复四中心，保持服务器和heartbeat运行；不重跑已完成20/math/seal，备份关机停放。
- [decision] helper逐行归约与pins索引不替代ROOT读取全部新数组、官方map、原音及PCM16窗口body；k10仍未释放。
- [blocker] 正等待k05实际整体终局/原execute0/所有children退出及完整独立科学QA；最终跨TFG方法仍未准入。
- [next_step] 保持唯一exec94568自然完成，在真实终局后运行已stage单次只读QA并核验输出，准确收齐新body完成ROOT独立QA后释放k10。

## Relations

- continues [[Codex checkpoint - 2026-10-11T02-21-12Z - Wav2Lip completed and MuseTalk center running]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]