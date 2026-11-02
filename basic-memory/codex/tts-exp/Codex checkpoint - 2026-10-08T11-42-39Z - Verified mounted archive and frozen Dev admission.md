---
title: Codex checkpoint - 2026-10-08T11-42-39Z - Verified mounted archive and frozen
  Dev admission
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t11-42-39-z-verified-mounted-archive-and-frozen-dev-admission
trigger: compact
project: tts-exp
hostname: LAPTOP-E133GSLE
repository: dogLeft233/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
status: open
repo_root: E:/Documents/tts-exp
capture: deliberate
username: dogle
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
agent: codex
model: gpt-6.1-sol
git_sha: cbd47e2cd49d38653fa1f41db56c5afd4cf9890f
started: '2026-10-08T11:42:39.6349796+00:00'
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T11-42-39Z - Verified mounted archive and frozen Dev admission

## Summary

冻结跨 TFG 对照267/360全PASS且仍运行；完整旧失败媒体在F和挂载B实际复制、全字节校验完成，源码修复已与冻结生产者格式核对。

## Story

用户最初要求从本地BM、Git与代码恢复TTS增强TFG的路线，随后把目标扩为可用的跨模型自然音频增强头。增强单声道F32 PCM只驱动视频，成片恢复完整原自然F32 PCM与原时钟。ROOT负责理论、冻结协议、BM/Git和独立证据，代码与实验交给GPT-6 Luna/max。失败尝试不终止研究，且不能把工程失败记成科学阴性。

当前候选固定Fit40/20 speakers、Adam120；Dev20片段/10 speakers、三seed、Wav2Lip/MuseTalk/Ditto共有360 identity/RMS对照和180候选行。11:27:21UTC观察267/360全PASS，ROOT核对原始响应、自哈希、PID和六组分母；候选GPU尚未启动。实际候选180 CPU绑定与ASR静态32成员准备已完成。RMS仍是必须超越的强基线，所有三模型正向且等权平均超过RMS .050为冻结效应门槛，另保留D0/Canchor/内容/视觉/GUARD20与全部失败分母，Confirm80/40 unseen及LatentSync封存。

完整旧失败媒体归档已真实下载104分块并组装653472590B tar；两个独立全tar扫描核对26331成员、26328媒体682851546B、5413 compact别名和旧360的120PASS/240FAILED。挂载B复制05实际完成12文件660116676B，worker和ROOT分别全读回PASS；CloudFS挂载复制不等于远端云同步确认，compact blobs仍另在服务器，需要分别保留。没有源清理或媒体提取。上述小型证据已精确按字节提交，HEAD cbd47e2本地未推送，25文件/21JSON179422B；5项无关旧改动未动。

ASR client03修复生产者保存内层argv、消费者误检查外层flags及退出状态等待deadline；ROOT全diff/上下文/pins/AST与独立审查02均PASS，审查03用AST及实际文件SHA纠正旧报告strict12 pin抄写错误。没有SSH/API调用。联合readback04最小修复media_eligible在PCM上下文检查后更新，ROOT完整diff与14来源pins静态PASS，未运行。重复wrapper02已调用冻结真实V3 dispatcher并验证original输入PCM→生成器实际PCM、step120/checkpoint/config/训练/selection链；独立源码审查通过变换修复。ROOT另发现q00要求candidate leaf不存在，而诊断需在180终端后执行，正在独立审查终态与容量门控；不能用仅metadata preflight代替真实候选完成。candidate detached launcher/status04新源码包已封存待ROOT和独立完整审查，严格180身份、每模型60PASS、零malformed、三键计数与实际COMPLETE/rc0/strict attestation；动态preflight和终态pins仍null，不伪造未来哈希。

## Working State

稳定锚点：本地未推送HEAD `cbd47e2cd49d38653fa1f41db56c5afd4cf9890f`，分支codex/cross-tfg-audio-head-20261003。Task113与Experiment112已实际更新和读回；ADR018保留字面NOT_RUN_SUBRUN，后端重复资格、实际完整ledger/终态及逐行原生回执合成准入，另运行候选特定六输出重复，不改变540主行或用于选择。

机器本地/脆弱状态：服务器root@connect.westd.seetacloud.com:18625，hostkey SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0，GPU UUID GPU-d09622d0-3535-e662-bb45-3df2b049fcea。凭据仅私有PTY57588内存，绝不入文件/BM。桥07 PID10892/端口12249，只读observer03 PID87568，controls PID344830/start_ticks329924069，SHM同级controls360/candidate180。观察间隔900秒。F完整tar在F:/tts-exp-archive-transfer/controls360_attempt17_failedterminal_media_complement_attempt04/assembly/，B实际12文件在B:/tts-exp-archives/controls360_attempt17_failedterminal_media_complement_attempt04/。scratch、未提交源码、私有PTY与忽略HANDOFF均非Git持久状态。

## Verification

ROOT实际全读104分块/响应、全tar成员与B12文件，检查实际字节SHA/自哈希/恢复映射/失败分母；独立verifier03实际exit0。Git逐blob原始字节核对25文件/21JSON、parent c93e2a和staged empty。观察11原始响应02b9193ef9b88539344a739c94b82d79599f4f47b77e99e1989b524c48708d5f核对267行全PASS，仍RUNNING。client03及readback04只做静态全diff与来源/AST检查，无目标运行、通用测试、API、push或PR。本checkpoint重读Router/实验指令/CodingSession/skill，搜索精确共享及legacy身份并直接读前一checkpoint，GitHub只读仓库与当前branch PR页200核对无PR，immutable创建后直接读回。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- 本地未推送HEAD：`cbd47e2cd49d38653fa1f41db56c5afd4cf9890f`
- `openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/controls360_attempt17_failedterminal_media_complement_attempt04/`：已提交实际归档/挂载B小型原始证据与proof_copy_manifest。
- `.codex/scratch/root_archive04_actual_commit_qc_attempt01.json`：实际Git blob精确读回。
- `.codex/scratch/root_actual_observer_poll11_qc_attempt01.json`：267/360原始独立QC。
- `.codex/scratch/root_asr_client03_source_review_qc_attempt01.json`与`root_readback04_delta_source_qc_attempt01.json`：源码静态准入证据。
- `.codex/scratch/dev360_candidate180_detached_launch_attempt04/pin_manifest_attempt04.json`：封存候选launcher/status04，动态runtime pins为空。
- `.codex/scratch/candidate_generation_repeat_diagnostic_attempt02/`：真实V3重复wrapper修复/完整diff/独立审查，终态顺序门控审查中。

## Observations

- [result] 控制267/360全部PASS仍运行，候选CPU/ASR静态准备完成，候选GPU/API尚未运行。
- [result] 完整旧失败媒体F两次全tar与B两次全文件实际校验PASS，cloud_sync_confirmed=false，compact数据仍分开保留。
- [decision] 冻结step120及540主行，候选特定重复额外6输出不改变选择；所有评分使用成片完整原自然音轨。
- [decision] 保留GUARD20至少81个原共同源帧要求、全部失败分母和未见Confirm封存。
- [blocker] 对照360尚未实际终端，candidate launcher04及repeat终态/容量门控待完整独立审查。
- [next_step] 完整审查候选launcher/status04并修复重复诊断的终态顺序门控；实际360终端读回与新容量检查通过后启动冻结180候选。

## Relations

- continues [[Codex checkpoint - 2026-10-08T11-08-09Z - Full local archive and running frozen Dev]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[018-native-repeat-basis-and-candidate-diagnostic]]