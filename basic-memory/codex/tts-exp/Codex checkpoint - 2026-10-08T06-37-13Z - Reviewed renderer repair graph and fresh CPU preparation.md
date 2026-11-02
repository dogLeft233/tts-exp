---
title: Codex checkpoint - 2026-10-08T06-37-13Z - Reviewed renderer repair graph and
  fresh CPU preparation
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t06-37-13-z-reviewed-renderer-repair-graph-and-fresh-cpu-preparation
branch: codex/cross-tfg-audio-head-20261003
git_sha: 7746b01f34b5850f11e322a0abc6934095f95f64
started: '2026-10-08T06:37:13.6400184+00:00'
agent: codex
repository: dogLeft233/tts-exp
hostname: LAPTOP-E133GSLE
trigger: compact
cwd: E:/Documents/tts-exp
username: dogle
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
status: open
project: tts-exp
model: gpt-6.1-sol
repo_root: E:/Documents/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T06-37-13Z - Reviewed renderer repair graph and fresh CPU preparation

## Summary
已从真实 Dev 失败日志确认 MuseTalk/Ditto 的原生渲染入口环境缺失，完整修复源码链通过两方审核并提交；新 CPU 模板封包正接受审查。

## Story
用户要求把 TTS 增强机制做成适配多种 TFG 的自然音频增强头，持续实验与迭代到可复现方法。增强 PCM 只驱动视频，最终换回完整原自然 F32 PCM 与原时钟。冻结 Fit40/20 说话人的 Adam120 候选，以独立 Dev20/10 说话人×3 种子×Wav2Lip、MuseTalk、Ditto 的 360 行 identity/RMS 对照，再运行180行候选及内容、视觉、GUARD20。Confirm80 与 LatentSync 保持密封；当前尚未产生完整 Dev 科学结果。

旧 controls17 批次真实启动后，M/D 首行 stderr 都报 `strict native renderer context mismatch for CROSS_TFG_BACKEND_ENTRY: None`，发生在 native inference 前。ROOT 核对原字节、计划行和完整日志，确认 frontend08 在 controls17 清理环境后未恢复入口；Muse 没有 CONTROL_DIR、Ditto 用旧共享路径。Luna/max 子代理实现 frontend09，只修改 M/D adapter 分支，在 descriptor overlays 后显式绑定严格入口/种子/源码/收据上下文及每行独立 work/control 目录；W2L函数保持原字节。冻结方法、原音协议和科学阈值没有变。新 collector11、builder17、controls18、candidate13、rebinder12 和 joiner08 一起重定向，ROOT 与独立 Signal 审核完整图及窄差分，通过全部61个本地pins。

源码与三个原字节 QC/snapshot 已提交本地 `7746b01f34b5850f11e322a0abc6934095f95f64`。精准 gitattributes 规则保留 CRLF QC 字节，13份 py/json staged blob 与工作树 raw hash 一致，diff-check 通过。旧批次最新真实 snapshot04 为357/360、120PASS/237FAILED，W2L120全部通过，M/D失败；仍RUNNING，不能视作terminal或科学阴性。新批次 attempt10 将使用新目录，不覆盖旧失败。新 CPU template-stage packet 尚未执行，独立审核已发现 q24 的 worker SHA 参数陈旧，正在用 fresh request/index02修正，不能消费 index01。

## Working State
持久锚点为 active Task「开发跨 TFG 的自然音频增强头」、Experiment「跨 TFG 自然音频增强头的训练与原音换轨验证」与本地 Git；当前 HEAD 未发布，GitHub 分支 PR 搜索无匹配。Task105/Experiment104保存实际错误和早期187行快照，下一次更新应加入完整源码审核、提交与实际新 CPU结果。无关五份 tracked 脏文件和历史 untracked 产物保留。

易失运行位于 `root@connect.westd.seetacloud.com:18625`；密码/API只在私有runtime，不写文件或BM。私有PTY57588、bridge07 port12249仍可用。旧 owned PID105669/start_ticks329309991，实际 controls leaf `/dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360/controls360`。新 namespace 是同一 latebind 根下 `controls360_repair_attempt10`，新 leaf 为其 controls360。必须确认旧360完整终态与owned PID退出后才启动修复后的GPU批次；CPU准备可先做。SHM空间与身份需现场复查，归档/内容守卫前不清理。

## Changed Files
正式锚点为 `openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/dev360_repaired_source_snapshot_attempt07.json`（21753B SHA f3429fc863854f52ccaad49fa275d02e3d96bb23e213b43cca0dcf4c78d8e844）、`dev360_repaired_source_graph_root_qc_attempt01.json`（2586B SHA218f677260cca2a3c365b92a8479bda318a00b3ff3620b5f5b9f65f5dae8050c）和 `dev360_attempt09_partial_failure_root_qc_attempt01.json`（1445B SHA4e90d6dc46aac8264d4df45268d0e18990a0a2b07468f692baa214a4c57c0f5b）。源码十份路径与pins由 snapshot07 的source_chain给出；CONTEXT入口已同步。

诊断原始response `.codex/scratch/dev360_failure_diagnostic_attempt01/requests_attempt01/02_read_first_failed_backend_rows.json.response.json` 为86656B、SHA0d68365d012e8c174c7e3e3ffa3696316e5355e892810c5b826e4f35b9e509ef。最新 status04 response在 `.codex/scratch/dev360_detached_status_attempt07/requests_attempt04/`。这些 scratch/SHM是易失证据，完整终态需归档。遗留 compact archive 的source_snapshot06是从已执行05衍生的metadata，尚需ROOT验证parent-chain等价；compact不等同完整媒体可恢复归档。

## Verification
实际 CPU07 全8请求、六下载与ROOT绑定/计划QC通过；旧 GPU启动收据通过。诊断worker、三请求、原日志/selfhash与status02由ROOT独立QC通过。status03/04实际请求通过，04为357行；未运行候选或调用ASR API。新 frontend09完整源/窄差分和snapshot07所有61pins通过ROOT与Signal审核，严格shim/native策略保持。新CPU封包仍在静态审核，发现陈旧worker arg pin并要求fresh修订；尚未派发。仅做授权实验/provenance检查，无通用测试、push或PR。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)，本次直接HTTP200核验canonical URL。
- [已推送历史提交c4a1998](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)，本次HTTP200确认。
- 当前HEAD `7746b01f34b5850f11e322a0abc6934095f95f64` 仅本地未发布。

## Observations
- [result] M/D共同失败是执行环境入口缺失；修复完整源码链独立审核通过并提交，尚无修复后的实际TFG结果。
- [decision] 保留旧完整失败分母；用新attempt10目录重跑360，再180，冻结候选与多模型验收标准保持。
- [blocker] 新CPU模板封包陈旧worker参数pin待修订，旧批次仍未读到360终态。
- [next_step] 审核fresh CPU封包并实际派发/readback，取得真实template08后衍生seed03与binding09/plan09/preflight09，再在旧owned进程终止后启动修复批次。

## Relations
- continues [[Codex checkpoint - 2026-10-08T06-02-18Z - Actual Dev launch and native adapter failure diagnosis]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[冻结增强头的门控时序与频谱贡献]]