---
title: Codex checkpoint - 2026-10-03T12-56-42Z - Completed fixed lag fit and content
  guard
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t12-56-42-z-completed-fixed-lag-fit-and-content-guard
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
cwd: E:/Documents/tts-exp
trigger: compact
agent: codex
capture: deliberate
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
model: gpt-6.1-sol
username: dogle
hostname: LAPTOP-E133GSLE
branch: codex/cross-tfg-audio-head-20261003
project: tts-exp
status: open
git_sha: 4a2422c7b12afaacda9b702d3321682e1bcac6bb
started: '2026-10-03T12:56:42.5956225+00:00'
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T12-56-42Z - Completed fixed lag fit and content guard

## Summary

固定零时差v3增强头实际完成96步训练并取回权重归档，100条ASR内容检查实际结束，300条运动保护已独立重算；跨前端资格与新开发集仍在推进。

## Story

用户最初要求从本地BM、代码、Git和百度盘恢复未同步的TTS增强TFG路线，授权服务器重跑、Qwen API和下载。最新目标是产出适配较多TFG的可用自然音增强头，持续改进直到方法完备。增强音只驱动视频，成片恢复原始自然音，主要SyncNet评分使用原始自然PCM。ROOT维护理论、设计、BM/Git、运输和独立复核，gpt-6-luna/max子代理承担代码与实验。旧机制报告25已完成；新方法目标ACTIVE，不能停在失败候选、import或launch。

历史8片段fit与10个新dev说话人各2clip隔离，三seed配对，FULL/GUARD20和31lag，按seed、clip、speaker聚合；40个confirm说话人与heldout LatentSync保持未读。交付目标至少三种前端、尽量五模型，加内容/运动保护、权重、CLI、时延和可恢复来源。

## Working State

### 科学路线与实际结果

300开发视频的权威本地证据在 `runs/cross_tfg_audio_head_20261003/stageb_joint_attempt02_readback/`，ROOT全七文件/300行/成员与分层聚合复核 receipt `.codex/scratch/root_stageb_consumer05_actual_readback_attempt01.json`。FULL等说话人ΔCbest：RMS−identity +.23413574695587167、local−identity +.26696223417917886、fit24−identity +.2559797445933024、v2fit96−identity +.21745221614837645。local−RMS +.03282648722330729且D0改善−.01844037374；fit24−RMS +.02184399764主要来自B+.02102354368，D0近零；v2−RMS −.01668353081且D0恶化+.05701052348。GUARD20保留S0022短片34query<41的规划2clip分母，九端点INCOMPLETE，无dev CI。旧学习目标训练更久并未带来方法改进，因此ADR008改为固定0lag目标和有界RMS参考。

typed ADR `tts-exp/docs/adr/008-fixed-lag-and-rms-reference-head-objective` actualcreated/readback PASS。v3保持8历史fit/32center/27query/31lag/96step/Adam1e−4/12epoch/seed20261003/±3dB幅度残差。R为相同stack关闭head/local、RMS .0376838172675746与stage/finalpeak .99的bounded_RMS_reference，区别旧rms_fixed peak=null。零初始化h==R，奖励固定D0相对改善与cos0改善各.5，罚log视觉范数比²(.1)、归一化波形MSE(.02)、logmask时间/频率平滑(.002)。自然audio1024维固定，R detached cache，h视频保留梯度；无B奖励、bestlag选择或timewarp。

v3源/18KB overlay/18成员bundle已ROOT全文/哈希/payload/optimizer审阅，receipt `.codex/scratch/root_v3_fit_source_review_attempt01.json`。CLI `scripts/audio_enhancement/fit_cli_v3_attempt03.py` SHA5e93b0731c60a55174b6ce5555ef4bcadb7c0d0d8b4a6ce8de542f8e97675c9c，runtime `wav2lip_fullcontext_fit_runtime_v3_attempt03.py` SHA8327eab43d74a6d73f979af419e08df96a0c80e379addc19ee12a87224417eac，builder e7d998146e7ba4b13fe3825864cb32b61676f948c4bd0d5d039d81a5c3f6cc41。操作修复分别保留失败：extract漏manifest使0step失败，PSF包路径验证错误使第二次0step失败；freshworker04只修依赖验证/控制路径，科学不改。实际worker107098/tick760236185最终FIT_ONLY_COMPLETE96step/8row/12epoch、PID消失，八零初始化波形/曲线/视觉严格0差且奖励/罚项0，后期音频/head梯度和参数更新非零。权重 `shared_audio_head.pt` / `checkpoint_latest.pt` SHA286d085674fe1eb10fb4fff51559bc9923a947327d971a9ab67738a2d1b9cecc，report rawea74900f7cfc2d5517f8f1982ec98a663b16baa9a672514ada6aef1a1b940925，stepsaf0fc4a545088d0023b68ceed7530605b3c734855120c447edb75c623f05356a。

实际终态归档25payload+manifest已ROOT全文审collector、实际创建/下载：`runs/cross_tfg_audio_head_20261003/fit_v3_attempt03_readback/fit_terminal_archive_attempt03.tar.gz` 658731B SHA5ea47f6483242b30796b8ab2d0867d7991e5db54dd873775b64237cd8b6da1a5。整包独立成员复核尚待做，训练奖励不能当开发效果。代理从真实300计划派生v3新60，再构造360六方法联合；主dev对比v3−identity/RMS，诊断v3−local/fit24/v2。所有candidate分数未读，fresh包需ROOT审阅后实际派发。

前BQ对显存“16GB导致16GiB门槛不可能”的叙述有误：当前实际nvidia-smi total32760MiB。12GiB空闲门槛仍是运行门槛，科学设计未变；本checkpoint明确纠正，旧不可变BQ不修改。

### 已完成保护和仍待读回内容

300 visual/speech guards实际PASS/300NPZ，两240配对集合完成。归档2523260B SHA12ab5f893104d447aceeb005076d578f23548153ad3b667b151033b32e497de6已本地下载、安全提取与ROOT全部315文件/318成员/306manifest记录验证，receipt `.codex/scratch/root_stageb_guards_archive_review_attempt01.json`。ROOT实际seed→clip→10speaker重算 receipt `.codex/scratch/root_stageb_guard_hierarchical_descriptives_attempt01.json`：五方法tracking均1；zero-lag原音RMS−aperture correlation相对identity RMS+.01771564497、local+.03139328972、fit24+.01322311712、v2+.02101373026。local−RMS+.01367764475；v2开口/速度较大仍非最高关联/评分。完整时钟/hash PASS不自动表示质量提高，独立描述无dev CI。

ASR实际100 unique sample×method waveform完成，三seed各组PCM相同；官方20 refs全文精确ID来自AISHELL transcript revisionbbe295d530192a4cd41644b711c9aecd087df653，10,091,431B SHAb5f33b9e0b47548e20a5ea4e504297f80df41d559133924c4e5b7b544c15b5c4。Consumer02 SHAe380ee51a6de42a3909d5b661020cf267f1c8cdc5d29db9ab0aec9868ee943df，Beijing qwen3-asr-flash/zh/ITNfalse/base64wav，无reference hint，无重试。私有SecureString经本地Paramiko5客户端→SSH channel stdin→remote ENV注入，key/password不进文件/argv/log；client02 SHA29a55347624e3bc395076d678667048120c8ef625b2708149adbc78aa1dedcde。一次actualAPI run worker107610/tick760287166 start12:37:49.093616Z finish12:41:38.503248Z、exit0/PIDgone、100planned、stderr空。summary9936B SHAbe93562daf408a2c7a25edfcca90e605a4e8aae79301c12d8f2ab118f31ba57f。

旧status01/terminal02误期望content_results.jsonl/execution_manifest.json，ROOT读源码在terminal派发前发现并保留未执行请求。真正六输出为content_items.jsonl/content_input_manifest.json/asr_results.jsonl/content_scores.jsonl/content_summary.json/content_result_manifest.json。fresh状态/7成员terminal reader在 `.codex/scratch/cross_tfg_audio_head_20261003/stageb_asr_content_attempt05/requests_attempt02/`，requestSHA4a8d27f393a820c0d765e273baa809942e83e61e29eb7c5544f3368db84ebf31 /20003d09002831e4f5ca7028b4d7a8aea5787a18e5416e2485cbf83cf2006720，尚未派发。科学CER和speaker描述尚未独立读回，不重复API。

embedding replay02已ROOT全文审，SHA4874c2469ee51bf9c29486ff96119ce6878919255dec8791d1d62d2c0f418936；实际官方1024维捕获/31lagrawmatrix parity待sealedpacket/GPU。单独NPZ sidecar不覆盖primary，nominaltail不删、不移时。

### 跨前端实际状态和可恢复归档

Ditto首次portrait实际4.14s失败缺filetype、0output。独立filetype1.2.0 wheel/overlay+loader CPUimport actualPASS，但非完整入口。ROOT实际读核心load_model发现TRT导入在.engine/.trt函数分支，修正静态graph“顶层必需TRT”误解；PyTorch/ONNX native路径可继续。wav2feat/hubert source直接16kfloat分块/零pad/25fps无显式全局标准化，内部ONNX invariance仍待测。fullStreamSDK CPUprobe03 actualprobeexit2、外层wrapper0，report827B SHA897dbdfd57f3fa1e4627f9b7ee04449efbd0de4325b3467ade0691f55c5c56ae/stderr2c07b02934311250d7d903a05fa65c0bc5b4e679667870b164255b5395cd66c8。fresh只读诊断 `.codex/scratch/cross_tfg_model_deploy_attempt13/ditto_streamsdk_import_probe_diagnostics_attempt01/` 待ROOT全文审/派发；不能写成科学阴性。

MuseTalk官方UNetcfg748B SHA5b6923aee04d71692e0e9846c471e0a4ea07a4f686d39545e472bd4ba17e1b47与3.4GBUNet7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007已21项实际asset PASS。实际Muse12因native加载VAE相对cwd错误15s失败/0output，freshMuse14 adapter06在已准备model_root加载、source SHA50df8eba4110ec02e8b2ba3099c102d40faf3b94f9b2680a6ba2c2fbb5cd4e44。实际八请求PASS、worker107359/tick760271064启动，最后真实snapshot RUNNING614MiB、VAE已本地load、0MP4。freshstatus16 index `.codex/scratch/cross_tfg_model_deploy_attempt14/muse_packet_attempt02/request_index_attempt16.json` SHA c8e19f69e441c688b4bd6aad91b630d8a26a4e7e4b339f1b2df01131206832ca，requestac3b590b8115b8c6dc464ded2197cb8fcae19ebaec997e3280f30d83661fa5b7待读/派发。实际32GiB GPU，先Muse终态释放再v3新dev，依赖修复持续推进。

portable300真实归档447534356B/7531files/SHA05260729197c87262b2f97402b196be78776e6cf2c6ef4914dbb6e292f328eb4已拆61块并实际ALL_61_CHUNKS_VERIFIED/0失败，后台PID284460 driver结束12:49:23.963917Z；`.codex/scratch/cross_tfg_audio_head_20261003/portable_archive_attempt02/download_requests_attempt01/download_progress_attempt01.json` actualread。整包拼接/7531member恢复及sourcebeforeafter尚待独立验证，remote source未退休。localBaidu可用，root~250MB/SHM~9GB/localE6.7GB，无删除必要。

### 持久入口与易失状态

Task `tts-exp/codex/tts-exp/开发跨-tfg-的自然音频增强头` 与Experiment `tts-exp/experiments/跨-tfg-自然音频增强头的训练与原音换轨验证` wholeupdate12/11 actualreadback，但guardsRUNNING/7refs缺失/v3未fit已过时，需下一次整体刷新。CONTEXT93行/HANDOFF40行ignored/README与transferiteration本地未提交。branchcodex/cross-tfg-audio-head-20261003、HEAD4a2422c7b12afaacda9b702d3321682e1bcac6bb local/unpushed，无PR；legacy/tests dirty保留，无stageall/push/pytest。无.codegraph目录。

私有PTY50274/bridge2133仍活；dispatcher `.codex/scratch/dispatch_requests_root_attempt02.ps1`，每request先ROOTread/SHA，actual.response.json的nestedexit权威。服务器root@connect.weste.seetacloud.com:12349，hostpinED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；凭据仅runtime。NS `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`，literalvenv不resolve symlink；nativeSO/root、资产/输出SHM。代理cross_tfg_execution(v3/portable)、local_recovery(Muse/Ditto)、mechanism_audit(guards/ASR/replay)均Luna/max。

## Verification

本turn实际v3Fit96完成、八零init与非零梯度训练记录、终态归档创建/本地下载；300运动全成员/manifest/分层描述ROOT复核；ASR一次100调用终态PASS实际read，旧collector名字错误已在派发前发现；Muse14实际启动和DittoCPUprobe实际失败；portable61真实chunk校验完成。v3整包独立审、ASR科学CER、跨前端native视频、新dev/heldout/confirm仍未完成。无pytest/unit tests。BM Router/实验指令/schema/config、完整samechat/legacy分页与直接前note身份、freshGit/GitHub后不可变created/readback。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current HEAD `4a2422c7b12afaacda9b702d3321682e1bcac6bb` local/unpushed。

## Observations

- [result] v3固定零lag实际96步训练完成；ASR100波形实际PASS终态，科学读回待做。
- [result] local相对RMS ΔC+.033、D0−.018、原音RMS−开口zero-lag关联+.01368；旧fit24主要提高B，v2延长训练减益。
- [decision] 保持开发集探索/confirm未读；v3直接固定D0/cos0及norm保护，候选与local直接比较；控制失败修复用fresh不可变入口。
- [blocker] 多前端native视频、新v3dev效果、ASR/embedding独立保护和40confirm尚未全部完成，方法目标ACTIVE。
- [next_step] 独立核验v3 fit和ASR完整终态产物，同时读Muse/Ditto真实状态并派发fresh v3新60开发渲染，继续跨前端方法迭代。

## Relations

- continues [[Codex checkpoint - 2026-10-03T12-07-15Z - Fixed lag source approval and completed motion guards]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[008-fixed-lag-and-rms-reference-head-objective]]