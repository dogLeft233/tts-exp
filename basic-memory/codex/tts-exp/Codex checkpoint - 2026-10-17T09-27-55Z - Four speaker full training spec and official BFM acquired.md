---
title: Codex checkpoint - 2026-10-17T09-27-55Z - Four speaker full training spec and
  official BFM acquired
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t09-27-55-z-four-speaker-full-training-spec-and-official-bfm-acquired
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 09:27:55.832626+00:00
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-17T09-27-55Z - Four speaker full training spec and official BFM acquired

## Summary
May 先导及评价音量27格已完成；新一轮四身份官方完整训练 spec 已保存，四人自然素材和官方 BFM 已取得，尚未通过新数据训练资格门。

## Story
用户最初检验音量是否改变 SyncTalk 生成及 SyncNet，随后要求 May 三档训练、三档驱动并补三档评分。正式报告50与51保存已完成结果。最新意图扩至四个新身份、每人6–8分钟、同人不同录制测试，先核验，再按官方完整流程同初始化同预算训练并输出27条目。新任务已授权后续训练，但训练只能在数据与完整预算资格门通过后启动。

新 spec 固定官方43079fb3f349675f6cb33031bb07262f7566cdd5的AVE/head+lip+portrait流程，四身份×共同seed42/2026×训练增益三档，共24个完整模型；May生成器不作warm start。每人原料420秒，监督train6000/val750/test1500帧，25fps512平方，六个固定10秒测试clip。训练/验证同录制时间留出，测试来自另一原录制。官方100000是累计目标，6000训练帧意味着60k基础后累计102k完整epoch attempts；65536 rays与FP16不以旧4096/FP32先导替代，实际optimizer更新及AMP skips必须配对核验。

HDTF官方元数据与固定镜像revision共同追溯自然录制，镜像处理版不冒称官方原始字节。四身份为SherrodBrown、PatrickLeahy、DebbieDingell、CathyMcMorrisRodgers；每人两原视频ID不同，contact sheet人工核对同人及录制差异，仍需音频/视频重复和跨录制坐标核验。Patrick测试482平方会向上采样到512，须保留质量差异限制。±6评价采用float64原PCM数值单位增益，避免新素材削波；0评价仍精确原PCM16，与报告51的舍入PCM16处理区别明确，不混池。

用户提供官方申请邮件凭据后，BFM官方tgz和01_MorphableModel.mat成功下载，凭据仅 transient环境变量，未写文件或记忆。官方许可PDF与AD-NeRF原始配套3DMM及face parser资源亦已取得。当前仍缺新人物完整预处理与独立数据QA；不得把资产下载成功等同训练资格通过。

## Working State
持久根为F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017，含17候选source/、source_probe.json、source_provenance.json、provenance/、selection.json和assets/；已选四人各420秒。BFM mat SHA256为37b1f0742db356a3b1568a8365a06f5b0fe0ab687ac1c3068c803666cbd4d8e2，tgz SHA256为d6cf2dffb74404f4f7ee2cd56f611e6fdcb22081ebf993f8b01edf6207f6cce9。资产receipt无账户与密码。官方HDTF元数据commit bbde889bbab954c9a9ebcabe9962885c29f2165e，镜像revision6f9ed465f761b8378c690be4f86f090634281584。

易失工作根为wjj@192.168.50.2:/dev/shm/synctalk-multispeaker-data-20261017。SherrodBrown 298+122秒 canonical FFV1/PCM已完成，7450+3050帧，两个scene flags均0；media_preparation_passed=true而training_admitted=false。本地scp会话20529正在持久化Brown；Patrick准备会话86763已启动，未读完成结果。其余两人尚待转换。Ubuntu SHM约5.1GB空闲、root仅545MB，F可用约23GB。每身份完成后立即持久化和SHA核验；需要空间时仅删除已核验F与云备份的自有旧May50 SHM缓存，不动旧CUDA/GCC/deps工具链。

新官方工作副本、BFM转换、face_alignment/lpips/PyTorch3D私有依赖层尚未完成；现有audit环境Torch2.5.1+cu121，face_alignment/pytorch3d/lpips缺失。旧官方运行根/dev/shm/synctalk-may-gain-inference-20261016可复用只读编译工具链，禁止改共享venv或已封存旧代码。远端没有rg，应使用find，不能把失败rg空输出当未找到证据。跨录制必须固定训练拟合的身份几何和训练背景，不能把独立身份重拟合后简单合并当相容。

报告51 F持久原数已核验；其百度盘Linux独立读回visibility timeout，云备份仍UNVERIFIED。报告50历史归档云读回已通过，可用于清理其自有易失缓存。新实验无训练、渲染或科学分数结果。

Git共享树大量既有dirty/untracked，最新代码和测试未跟踪，未stage/commit/push。当前分支无PR，GitHub解析本地SHA返回422，不构造commit链接。

## Changed Files
- scripts/experiments/synctalk_multispeaker_full_20261017/：protocol.json、source_inventory.py、collect_provenance.py、probe_sources.py、prepare_materials.py、download_bfm.py、bootstrap_assets.py。
- tests/experiments/test_synctalk_multispeaker_full_20261017.py：素材/录制/上下文合同拒绝测试。
- basic-memory/Experiments/SyncTalk 四新身份官方完整训练电平实验 Spec.md，Tasks/准备并核验四个新身份的 SyncTalk 完整训练数据.md，Task Board及研究地图计划入口。
- .codex/scratch/：BM写入辅助，机器本地可丢弃，不作为实验原数。

## Verification
3个unittest已运行通过，覆盖同原录制、时长不足/上下文不足、身份缺失重复拒绝。17来源ffprobe与contact sheet完成；全部25fps、源音视频start0、48kHz立体声，时长符合官方注释。Brown canonical全部帧PTS网格、原PCM样本数、三档DOUBLE WAV精确增益、帧SHA检查通过；全体独立核验尚未运行。BFM与配套资产下载均有SHA receipt，尚未转换或验证官方跟踪。Startup Router、实验指令、研究地图及技能已加载；checkpoint同chat/legacy搜索无后页，上一checkpoint agent/session精确验证。只读GitHub API确认仓库及无PR，本地SHA未推送。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)
- 本地未推送SHA：`2b71639c3b555293375d248350999377168d515d`
- [HDTF官方来源](https://github.com/MRzzm/HDTF)
- scripts/experiments/synctalk_multispeaker_full_20261017/protocol.json
- F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/selection.json

## Observations
- [result] 四身份素材和官方授权BFM已经取得；Brown基础媒体整理完成，整体数据仍未准入训练。
- [decision] 同身份不同原录制测试；完整官方基础加lip流程，两个共同seed、相同完整随机初始化和实际更新预算。
- [decision] ±6评价保持无削波float64 PCM单位，0评价精确原PCM16；新处理不与报告51混池。
- [blocker] 新人物官方预处理依赖、固定训练身份的跨录制坐标及独立重复/音画QA尚待闭合；SHM需分批持久化管理。
- [next_step] 完成四身份canonical素材的持久SHA核验，并在隔离官方副本转换BFM、补齐依赖后实施固定训练身份的跨录制预处理资格检查。

## Relations
- continues [[Codex checkpoint - 2026-10-17T08-25-43Z - May fixed test scoring and sample size verified]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- relates_to [[51-synctalk-may-eval-gain]]