---
title: Codex checkpoint - 2026-10-01T09-57-39Z - Phone-local spectral controls and
  guarded WORLD rerun
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-01-t09-57-39-z-phone-local-spectral-controls-and-guarded-world-rerun
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/tts-mechanism-20260930
git_sha: 18e15f3d19fe6a08e1c0c703f1a39fdfef3affd0
started: 2026-10-01 09:57:39.286631+00:00
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

# Codex checkpoint - 2026-10-01T09-57-39Z - Phone-local spectral controls and guarded WORLD rerun

## Summary

CAL音素谱形诊断已完成并记录插值影响；局部16640端点协议已提交，全量WORLD v6通过实际预检并在受控tmpfs启动。

## Story

用户要求恢复项目历史并自主完成TTS增强TFG的原因研究，root维护理论、BM与主线，三个gpt-6-luna/max代理负责代码和实验。最新强调音量、能量实验已有，寻找局部声学原因。RMS与mel分区、男/女性复制及可见口部运动均已有独立完整结果，当前进一步在固定受体音素时钟和句级RMS下双向移植WORLD谱形、F0/清浊/AP与局部能量，并在实际五帧唇部查询覆盖的mel列并集上定位响应。完成报告25仍需要完整九臂消费者、局部/视觉分析及针对性谱形控制。

新完成的CAL诊断发现T→N映射后16个预选音素的谱形动态全低，但原始时钟下whole-clip active∩voiced仅13/20低，均值差−.009767861 natlog，无speaker CI；音素native仅14/16低。反向N→T映射后12/16也较低，因此线性log-SP插值自身会平滑。该诊断是CAL-only描述，不是TFG因果结论。主线已增加独立phone mean log-shape × within-phone residual四格移植及SELF_GRID往返网格控制，代码/物理CAL准备中，未读取EVAL效应。

## Working State

权威CAL报告 `runs/recovered_20260930/independent_validation/world_cal_phone_spectral_detail_20261001_attempt03/world_cal_phone_spectral_detail_20261001.json` raw SHA635a7907f4dd949c6704d6aedd0df014f6d727f1fd2e6d8963f188b28cb9e443，payload a4ec6bb4045d62ee1f5a9e5e6e0b8272aa1602503807375ce7ef3720931e587f；20 CAL clips、10 speakers、160 arm rows、170 phone×direction summaries、EVAL read false。source03 SHA5212bc3f682a185ab687fcbf61cfa1b820640085bc3523f8b40306f5b970d5aa。raw/payload双身份修复后完整运行与根读回PASS；01 preflight误比较、02未执行NameError尝试均保留。

局部冻结文件位于 `openspec/changes/diagnose-world-local-acoustic-response-20261001/`，16 exact phone labels固定CAL选择，8 arms×2 directions×2 geometries×4 supports×16 labels×4 phases×2 estimands=16384加256whole-G，同16640 max-|t| family及20k×40共享speaker draws。最新addendum04 SHAf9fb38658c05cdd6dcae714f64ad61560f1bd125f81954b28c7773637199d8f6，规定精确phone_index/phase映射与可重建bootstrap。consumer05 SHA766d7141037475c80bf70e28ef17ae88ecf86adff4e6bdfd5489b316ee1f6add静态就绪，尚未数据执行。root正在核查其common_support长度等于full V长度的断言是否符合producer实际schema；需只读159旧context的shape/count，不读效果。short guard20继续INCOMPLETE、计划分母40，不39救援。

旧WORLD v5仍是158/160 ENOSPC失败，不能拼成完整run，科学effects未读取。RMS09 1680文件/1806611203B已迁移data分区并在原路径symlink，独立全成员读回PASS；旧v5 forward log已精确恢复。空间决策、CAL结果与局部设计BM均已写/readback。

旧v5可恢复双副本归档02正在本地hidden PID212932运行，runtime `.codex/scratch/world_v5_archive_runtime/Scripts/python.exe`，client `.codex/scratch/world_v5_stream_archive_attempt02/stream_world_v5_archive.py` SHAfc8e839e3bd3faa893d0a3309a614885cc658db1c5d34e5ba56d49a801f6b7cb。E归档在同目录archive_e，B归档 `B:/experiment-archive/tts_world_v5_attempt06_20261001_attempt02`。最新进度1073741824B/约4.1GB，源10796 files/11439 members/4023374101 filebytes，inventory19ee1715c6fbfc495e6cddea3ec406f117e6e1c5685ac5df4b4b7778f39765ae。必须完整tar成员hash、E+B整文件读回、源before/after一致、E完整restore通过才退役精确旧run+JOB。attempt01因吞吐问题受控停止，136529920B双partial与abort receipt保留。retirement attempt02已准备，未执行，root尚待审核真实最终归档receipt。

root实际读资源：root约1.808GB、data .873GB，tmpfs33.286GB free、cgroup max66.57GB/current7.99GB、host available约482GB，GPU idle。已记录BM决策采用受控tmpfs并行科学计算，避免等待慢网盘归档；最终成果必须全成员copy/hash到持久存储后才结束。worker03 `.codex/scratch/world_gpu_attempt07_v6_tmpfs_worker03.sh` SHA2d5d81d84210bb913bf68c74b849982e1e5b0a8e54f74c9303c220c4c1b05a1c，request index `.codex/scratch/world_gpu_attempt07_v6_tmpfs_attempt01/request_index.json` SHA9c36b99b66b5d943e9b494eb0d7c9185071d25d5b97cbfb032236d9545330704。实际mkdir/upload/stage/launch response均0/PASS，launch PID9429；这只代表启动，不代表科学完成。

v6 run `/dev/shm/tts-research-20261001/world_phone_transfer_gpu_attempt07_v6/run`，JPEG sibling，JOB `/root/autodl-tmp/tts-research-storage-20261001/job_control/world_phone_transfer_gpu_attempt07_v6_tmpfs_worker03`。原science05 source275a3169c7c7531a01d5942bd06278843faae70724f0e9ae5b7cf8c90ac7a6ac仍在原attempt05绝对路径，plan rawbba8bf428a1e5bc0a7e4eafa203976ea2c988d4332d2f82a02d2015e90f81d6d/payload1410aface9e20f23421b42ec871f00344c424d56ea4a0d87c921b670bd4b11bc不变，physical QAe78e11587ace920003163d04010670aa4d273f80f5b7df657d0b44eb143fd39d完整1920/160 PASS。wrapper5 SHA708a33d67a1e734f3247ffb3cccd6605a0a596621fac6adff818bf61cd623afa每forward核五flag/count/hashchain并原子compact final。启动tmpfs≥8GiB/cgroup余量≥16GiB/host available≥12GiB，30s监控tmpfs≤2GiB/cgroup≤10GiB/host≤12GiB或probe失败则USR1受控停止。不能缺最终count、五flag、完整seals。

Consumer04 SHA9feaf39eb6cad01c9dc76d86535fae4ce95379a8ed775a2e70e0bc93286eb056仍保持完整160×76/full V×A独立重算，但新的RAM request bundle及local05的确切audit/stats路径正在代理准备；必须使用worker03新runtime flags sidecar。MediaPipe独立CPU环境/1600视频视觉追踪还未安装执行，不能修改W2/WORLD/SyncNet环境。新谱形四格CAL只读取冻结CAL物理输入，EVAL及其统计family需root再次事先封存。

机内易失状态：root authenticated PTY50274、daemon93533/port3019仍为凭证调度；password父ENV已在hidden归档child启动后清除。服务器授权入口root@connect.weste.seetacloud.com:12349，密码/API只运行时传递，绝不入文件/BM/argv。已核ED25519 host pin SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0。B是本地CloudDrive，AutoDL未发现既有网盘挂载。

## Changed Files

当前本地未推HEAD18e15f3d19fe6a08e1c0c703f1a39fdfef3affd0提交7项合格源/协议/CONTEXT，上一3566保存17项已合格RMS/MEL等。`.gitattributes`将实验Python与局部spec规定LF，CAL inventory原50448B CRLF raw hash固定、plot SVG源CRLF单独保留。原dirty tests/independent consumer/validator与未跟踪旧失败attempt均留原状，未stage all。CONTEXT91行；ignored HANDOFF与BM主Task需要更新最新HEAD/CAL完成/归档02/RAM启动。

## Verification

root实际验CAL03 raw及canonical payload、cohort、false EVAL标记；读取native/mapped全16摘要。五个封存输入staged blob与worktree SHA一致，scoped git whitespace check PASS。archive02 client pycompile、worker03与请求bash-n/JSON静态PASS，实际阶段响应均核hash/资源/input/runtime门控。BM结果与决策readback PASS。当前git root/branch/HEAD/status/username/hostname已直接读取；加载Startup Router/实验指令/Coding Session。samechat21、legacy0、无更多，直接读U确认agent/session。GitHub只读API核canonical仓库/pushed baseline/当前无PR。没有新增或运行pytest。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- 本地未推HEAD `18e15f3d19fe6a08e1c0c703f1a39fdfef3affd0`。
- CAL权威报告及 `.codex/scratch/root_cal_native_vs_mapped_phone_dynamics.json`。
- `.codex/scratch/world_gpu_attempt07_v6_tmpfs_attempt01/` 实际阶段响应。
- `.codex/scratch/world_v5_stream_archive_attempt02/INDEX.json` 与 archive_e/archive_progress.json。

## Observations

- [result] CAL exactphone映射谱形变化减少存在明显重采样贡献，native与mapped结果已分开保存。
- [decision] 将phone平均log谱形与内部残差轨迹分别因果移植，并并列SELF_GRID往返网格控制，读取EVAL前封存。
- [decision] 受控tmpfs推进固定协议九臂run，持久控制文件与内存guard保障失败可追溯，最终必须完整copy/hash归档。
- [blocker] WORLD v6尚未完整160/12160/5760/1600与独立验证；local schema审查、MediaPipe、新谱形CAL/EVAL和旧v5完整归档未完成。
- [next_step] 核实v6实际运行状态与五flag侧车，同时完成local common-support schema审查并封存RAM消费者包。

## Relations

- continues [[Codex checkpoint - 2026-10-01T09-13-52Z - Local acoustic protocol and verified storage recovery]]
- relates_to [[解释 TTS 的 TFG 与 SyncNet 优势]]
- relates_to [[校准音素内谱形动态与清浊覆盖]]
- relates_to [[WORLD 音素局部响应与声学剂量]]
- relates_to [[研究服务器缓存清理与可恢复归档]]