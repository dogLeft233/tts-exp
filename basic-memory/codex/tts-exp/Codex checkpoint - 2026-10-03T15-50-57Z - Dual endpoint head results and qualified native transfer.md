---
title: Codex checkpoint - 2026-10-03T15-50-57Z - Dual endpoint head results and qualified
  native transfer
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t15-50-57-z-dual-endpoint-head-results-and-qualified-native-transfer
repo_root: E:/Documents/tts-exp
git_sha: d512876881fd630d4c8fb89920a031bdff72a144
project: tts-exp
trigger: compact
branch: codex/cross-tfg-audio-head-20261003
hostname: LAPTOP-E133GSLE
repository: dogLeft233/tts-exp
status: open
agent: codex
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
started: '2026-10-03T15:50:57.7536086+00:00'
username: dogle
model: gpt-6.1-sol
cwd: E:/Documents/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T15-50-57Z - Dual endpoint head results and qualified native transfer

## Summary

六方法360条实际开发结果已独立复核：新v3改善固定零lag距离，却降低Cbest；MuseTalk16工程与原音换轨完成核验，Ditto Python头文件修复成功，共享多前端方法验证继续进行。

## Story

用户先要求恢复TTS增强TFG的机制研究，随后把可用、适配较多TFG模型的自然音增强头作为新目标，并要求失败后持续迭代。增强音h(x)只驱动视频，最终音轨与主要SyncNet评分始终使用原始自然音x。机制报告25已完成，增强头目标仍ACTIVE：至少三种前端、目标五模型、未读heldout与新说话人确认、内容和运动保护、权重CLI、时延和最终报告26。ROOT负责理论、协议、BM/Git与独立复核，代码和实验由用户指定gpt-6-luna/max子代理完成。Qwen ASR/TTS和数据下载已获授权；凭据只在私有运行时存在。

旧局部压缩在Wav2Lip开发中高于RMS的Cbest约.033。v3按ADR008直接优化固定零lag距离D0与单位方向余弦，实际开发呈现更强零lag对齐，但距离曲线背景B同时下降，使目标Cbest回退。因此保留local compression作为目前Cbest较优的共享候选，v3作为固定时刻对齐候选并行对照。接下来的机制修复针对同一fit音频/人脸/上下文下proxy与官方生成器的逐层差异，而不是再重复音量排除实验。

## Working State

### Durable evidence

分支codex/cross-tfg-audio-head-20261003，HEAD `d512876881fd630d4c8fb89920a031bdff72a144` 本地未推送、无PR。协议在 `openspec/changes/develop-cross-tfg-audio-head-20261003/`，fit8/4人、dev20/10新男性/3seed、confirm40新男性与LatentSync heldout效应保持未读。历史失败、旧300和v3新60均不改写。

六方法joint360已于14:48:40Z完成，verified360/0issues/四input bindings和四consumer输出seal全部通过。FULL14端点COMPLETE；GUARD20全部INCOMPLETE，因为规划S0022W0123短支持仍保留2clip分母；exit2对应分析不完整规则。开发CI NOT_APPLICABLE。归档 `runs/cross_tfg_audio_head_20261003/joint_v3_analysis_attempt03/persistent_joint_analysis_attempt03.tar.gz` 1168105B SHA966fbfb6d6b04a39a5436f87821c5117524e95bf7c3aba570e21b871fa99d4cd。ROOT receipt `.codex/scratch/root_joint360_actual_curve_and_hierarchy_readback_attempt01.json` 为ROOT_JOINT360_SAFE14_ARCHIVE_ALL_CURVE_METRICS_AND28_ENDPOINTS_PASS：14安全成员、3600曲线指标关系、28端点seed→clip→speaker重算误差<1e-12。新60 rawquery矩阵由远程consumer重算，ROOT本地未再次重减原始矩阵。

FULL等说话人mean：identity Cbest6.710668715；RMS6.944804462；local6.977630949；fit24 6.966648459；v2 6.928120931；v3 6.839444486。v3−RMS D0−.263630533218、Canchor+.153098575274、B−.110531965892、Dmin−.005172022184、Cbest−.105359975497。D0下降与Canchor上升均10/10说话人；Cbest只有2/10上升。最佳lag+1/+0：identity58/2、RMS51/9、local50/10、v3 22/38；没有实际音频重定时。v3−local D0−.245190159480/Canchor+.120591394107/Cbest−.138186462720。完整分数与逐说话人表在归档和同目录 `readback_attempt01/`。

300原音embedding replay已COMPLETE_PASS：300 captured/300 matrix exact/0failed，13:50:59Z结束。actualresponse `.codex/scratch/cross_tfg_audio_head_20261003/syncnet_embedding_replay_attempt03/packet_attempt04/10_readonly_status_request.json.response.json`；manifest2a0d992b7713cb8263d596451ed1efb0687b2e1d732f8ab57508aa5612264b7d，ledgerf2640a4387854f686263bb2cc8a82908d3bafc6631f504121d4311085b934031。后续需terminal归档、共同A embedding逐行相同核验和Vnorm/unitdirection四cell代数分解，不叫因果中介。

v3 fit96步实际通过并独立数学核验，checkpoint286d085674fe1eb10fb4fff51559bc9923a947327d971a9ab67738a2d1b9cecc。fit_report有效plan raw为4b3e2551b70685953af1013b47f1a542a30022ff32f6c6c4cdcbf5d505465ad8；parent e491为不同文件。fitclock02实测8条proxy最佳lag+3七条/+2一条，而原生主要0/+1；居中27query和RMS都不能解释差异。fresh parity03 source49750B SHA4ea1faabb9e5e040e06483bf94a3b73501887b406ed6a3cee07573066a69791e已ROOT全文/差异审阅，使用官方DIVX→PCM16 mux→SyncNet JPEG路径、原音评分、32frame/27query/31lag；只运行1个fit RMS row smoke。proxy FullContextSyncAdapter.sync_distance_curve仅video与context，当前调用正确。packet `.codex/scratch/cross_tfg_audio_head_20261003/proxy_native_parity_attempt03/packet_attempt01/request_index_attempt01.json` SHA f0a68e1e68288f9c66a553314576369840cbd08f61ac41db203c6319e44cb778；4 shell scripts/9 requests待ROOT全文审阅和实际preflight→launch，source本身已审阅。

### Qualified native frontends

MuseTalk16实际工程PASS，144frame/25fps/512x512、原音PCM92225sample与copystream一致、expanded crop/paste同框[121,80,452,411]。adapter08 SHA200209c7fbaa4b2c83d571a04696bec7658e759a564494c064b220cdfdf2daaf，renderer09 b75004340fbf90b937fb516c5d8ee013f193f763f5cf61bbfc7d28cb12ce88c8；资格为静态portrait/Haar+官方FaceParsing/fp16 CUDA/TF32false，MMPosefalse。三文件实际下载到 `.codex/scratch/cross_tfg_model_deploy_attempt16/portrait_readback_attempt01/`，ROOT receipt `.codex/scratch/root_muse16_three_files_geometry_and_decode_readback_attempt01.json` PASS；contactsheet已实际view_image，抽样未见旧明显矩形贴回硬边。工程与抽样观察不代表方法效应或感知通过。cross_tfg_execution准备五参数shim、四共享方法identity/RMS/local/v3×20clip×3seed=240实际dev；保持已核验renderer每行加载，约2.8小时可接受，不引入未核验常驻新接口。

Ditto full StreamSDK CPU import09找不到setuptools，10通过invalidate_caches修复该层后失败Python.h。headers14实际 apt download libpython3.10-dev=3.10.12-1~22.04.18，仅隔离抽取164include headers并gcc syntax PASS，无baseenv安装。report CONTROL/python310_headers_attempt14.json SHA b997b9e98ed61be480788442237bf1008f4a5cef24e74dd55bdb71ea55b1f0cd；package c058675011ec4b64f5ad803d74b5fe7de0c720c3700683040da68d1fb5b44ed9；tree e9dcde923150324ed29f3e8104d18f9ca61f61b7b32b07a6a1874e47c15a2a6e。必须三include根：CODE/python310_headers_attempt14/overlay/usr/include、其中python3.10、x86_64-linux-gnu/python3.10。local_recovery准备report实际下载/成员核验和fresh fullimport12，之后实际portrait与共享dev。

Task15与Experiment14 whole快照已实际更新并直接读回PASS，phase actual_joint360_dual_endpoint_result_replay_complete_and_native_transfer；receipts `cross_tfg_task_snapshot_receipt_attempt15.json`/`cross_tfg_experiment_snapshot_receipt_attempt14.json`。Experiment仍implemented，Taskactive/current_step2。旧100ASR CER0与300motion不覆盖新v3：新60motion使用旧identity60 sealed reuse，内容guard必要时20v3真实ASR。portable旧300447MB全部7532成员/61parts/restore ROOT PASS，E权威B副本远程源保留。

### Machine-local state

NS=/root/tts-research-system-20261001/cross_tfg_audio_head_20261003；SHM=/dev/shm/cross_tfg_audio_head_20261003；CONTROL=NS/job_control/model_deploy_attempt13；CODE=NS/model_deploy_attempt13/code。服务器root@connect.weste.seetacloud.com:12349，host fingerprint SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。密码/API只能私有PTY50274/bridge2133/环境SecureString，绝不文件/BM/argv。requests经 `.codex/scratch/dispatch_requests_root_attempt02.ps1`，actualresponse顶层code/ok为证据。E约5GB/SHM约8GB；不删除远程源。

## Changed Files

本轮CONTEXT93行一个段落修改未提交；HANDOFF30行ignored已更新到360/replay/Muse16/header14，最新BM15/14 actual。spec transfer-iteration仍需更新actual360双端点结果。保留legacy/tests/untracked不stageall，不运行pytest/unit tests。.codegraph目录存在但codegraph explore实际报未索引，已按规则尝试后rg fallback。

## Verification

本轮actual360下载、14成员/seals/input bindings、全曲线指标与28分层端点复算、Muse三文件SHA/geometry/PCM/copystream和实际contactsheet查看、headers14抽取与gcc syntax已执行。proxy/native parity、Muse240、Ditto完整import/portrait、v3新guard、heldout/confirm和报告26仍未完成。加载config/Router/实验指令/CodingSession/Task/bm-writing，fresh同chat73与legacy0、直接BU/合法started、Git/GitHub canonical/noPR/pushedbaseline核验；新checkpoint只接受created和直接正文/metadata相等读回。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current local/unpushed HEAD `d512876881fd630d4c8fb89920a031bdff72a144`.
- `openspec/changes/develop-cross-tfg-audio-head-20261003/` is the shared method and cohort protocol.

## Observations

- [result] v3实际原生D0改善.264且10/10方向一致，但Cbest较RMS下降.105；local保持当前较优Cbest共享候选。
- [decision] 保留Cbest与固定时刻D0/Canchor双端点；增强音只用于生成，原音时钟与评分不变；Muse工程资格允许进入多方法实际dev。
- [blocker] proxy原生时序尚不一致，parity smoke待实跑；Muse/Ditto共享效果与未读confirm尚未测得，不能提前选择跨模型方法。
- [next_step] 实际运行同fit parity smoke并冻结启动Muse240共享开发，同时把Ditto完整import修复推进到portrait与dev，依据真实跨前端结果迭代可用增强头。

## Relations

- continues [[Codex checkpoint - 2026-10-03T15-04-20Z - Completed head development and proxy repair evidence]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[008-fixed-lag-and-rms-reference-head-objective]]