---
title: Codex checkpoint - 2026-10-03T12-07-15Z - Fixed lag source approval and completed
  motion guards
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-03-t12-07-15-z-fixed-lag-source-approval-and-completed-motion-guards
trigger: compact
project: tts-exp
branch: codex/cross-tfg-audio-head-20261003
status: open
repository: dogLeft233/tts-exp
agent: codex
repo_root: E:/Documents/tts-exp
git_sha: 4a2422c7b12afaacda9b702d3321682e1bcac6bb
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
capture: deliberate
hostname: LAPTOP-E133GSLE
model: gpt-6.1-sol
cwd: E:/Documents/tts-exp
username: dogle
started: '2026-10-03T12:07:15.0163194+00:00'
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-03T12-07-15Z - Fixed lag source approval and completed motion guards

## Summary

300个开发视频评分已复核，新固定零时差训练源与18成员bundle通过独立审查，300条视觉/原音运动保护实际完成；MuseTalk实际资产预检通过，新的推理和训练请求待派发。

## Story

用户最初要求恢复未同步的TTS增强TFG路线，授权本地BM/Git/百度盘、服务器重跑、Qwen API及下载；最新目标是产出适配较多TFG的可用自然音增强头，持续迭代直到方法完备。增强音只用于生成视频，成片恢复原始自然音，主要SyncNet评价使用原始自然PCM。ROOT维护理论、设计、BM/Git/运输和独立复核，gpt-6-luna/max子代理承担代码与实验。旧机制报告25已完成，新方法目标ACTIVE；不得在失败候选、import、prepared或launch停下。

历史fit与10个新dev说话人各2clip分离，40个confirm说话人及heldout LatentSync未读。三seed配对，FULL/guard20与31lag原音评价；先seed、clip再speaker聚合。目标至少三种前端、尽量五模型，含内容/运动保护、权重/CLI、方法冻结后确认性效果。当前比较提示共享局部动态处理仍有收益，旧学习目标延长训练反而弱于RMS，因此正在修正目标并实际检验跨前端。

## Working State

### 科学依据与新训练方法

权威本地 `runs/cross_tfg_audio_head_20261003/stageb_joint_attempt02_readback/` 全部300行与七文件、严格tar、hash、分层聚合已ROOT复核，receipt `.codex/scratch/root_stageb_consumer05_actual_readback_attempt01.json`。FULL等说话人ΔCbest：RMS−identity +.23413574695587167，local−identity +.26696223417917886，fit24−identity +.2559797445933024，fit96 v2−identity +.21745221614837645；v2−RMS −.01668353080749511，D0 +.057010523478190044。guard20保留S0022短片34query低于41导致9端点INCOMPLETE，规划2clip分母不删，无dev CI。不是生成故障。

typed ADR008已created/readback PASS：`tts-exp/docs/adr/008-fixed-lag-and-rms-reference-head-objective`，receipt `.codex/scratch/fixedlag_head_decision_receipt_attempt01.json`。v3以同一EnhancementStack的bounded_RMS_reference（head/local关、global和post RMS .0376838172675746、stage/finalpeak .99）为对照，零初始化h==R ≤1e−7；此R明确区别旧rms_fixed peak=null。历史8clip/32center/27query/31lag/96step/Adam1e−4/12epoch/±3dB不改。奖励为固定D0相对改善与固定cos0改善各.5，另罚log视频范数比²(.1)、normalized waveform MSE(.02)、时间/频率logmask平滑(.002)；无B奖励、bestlag选择或timewarp。自然音audio1024维固定、R cache detached、h视频保留梯度。

fresh v3 attempt03源、compact18KB overlay、18成员71538B bundle已ROOT逐一SHA/大小/成员/原plan row/full_context/规范payload重建核验，receipt `.codex/scratch/root_v3_fit_source_review_attempt01.json` 为ROOT_SOURCE_AND_ARCHIVE_REVIEW_PASS_READY_FOR_ACTUAL_FIT。CLI `scripts/audio_enhancement/fit_cli_v3_attempt03.py` SHA5e93b0731c60a55174b6ce5555ef4bcadb7c0d0d8b4a6ce8de542f8e97675c9c；runtime `wav2lip_fullcontext_fit_runtime_v3_attempt03.py` SHA8327eab43d74a6d73f979af419e08df96a0c80e379addc19ee12a87224417eac；builder e7d998146e7ba4b13fe3825864cb32b61676f948c4bd0d5d039d81a5c3f6cc41。overlay raw4b3e2551b70685953af1013b47f1a542a30022ff32f6c6c4cdcbf5d505465ad8/payload72c6e1c2bc8053d6050fa1ea2790166623fd30b433f9a92d9c60634419409a69/effectivec6dc22a06160c3968201b6137f5b72b19211a5cbabd2be41c43ad69cc955bb6a；bundle d1b89f160b4f5490c22df0ef87772e18e960b452a10beb9507f5f32c842f7f7c。

未执行01大plan超50KB，02overlay删除键/RMSE ndarray.square/容差键bug均保留；03 fresh修复已全文审阅与编译。launcher01 minfree16GiB会超过实际16GB卡可用容量，ROOT改为freshworker02 minfree12GiB/util≤90，单行diff/AST PASS，worker SHA9df8fd2880b7e3b231e03b553e238e721699c797b16e2c75a24af6aecad14b1a。request index `.codex/scratch/cross_tfg_audio_head_20261003/fit_v3_attempt03/request_index_attempt02.json` 已准备五请求，尚未派发。科学不改；manifest里的parent17ef HEAD标清继承，实际execution receipt记录当前4a242 HEAD。

### 真实跨模型推理

future10/02实际222文件1549546B CPU导入资格完成；base09不改，资格不是TFG收益。Ditto实际首portrait worker104905/tick759861991终态FAILED/rendererexit1/0输出，4.14秒，stderr1880B SHAe346fbfbe7809f973b0e19cb3e86e998290b857efc8a64981dbd7e583db3f7da，core.atomic_components.loader.py缺filetype。PID已终止、GPU空。progress权威 `.codex/scratch/ditto_status_refresh_attempt01/read_ditto_progress_attempt01_request.json.response.json`，request73168121966105cf74c0c13c7097b197573c325001353ef0f60a9eefea235a6a。控制失败不能写成科学负结果。

filetype1.2.0 pure wheel19970B SHA7ce71b6880181241cf7ac8697a2f1eb6a8bd9b429f7ad6d27b8db9ba5f1c2d25已本地；独立stage13/probe请求尚未派发。ROOT已审并实际dispatch只读AST import graph，33文件/无truncation/code0；StreamSDK链顶层TensorRT/CUDA插件调用，需要审native PyTorch依赖闭包；loader-only CUDAhidden/noTorch/ORT是窄资格，不能冒充完整入口。代理正在定位frontend wave normalization和真实native路径。

MuseTalk官方UNet配置必须`models/musetalkV15/musetalk.json`，748B SHA5b6923aee04d71692e0e9846c471e0a4ea07a4f686d39545e472bd4ba17e1b47；不能用VAE/config.json。官方source0a89dec45a0192b824e3cf4daf96c239440c5ed8/HF2bcb936e2fddb4d86db4c62fd45b387d0c061571，UNet3.4GB LFSSHA7ebf6c98c181e20838e4c0054e96e944ac60d5d692cc01db42839fe11b787007。fresh12 packet02实际七请求成功，21资产/source检查PASS，无model forward。preflight7347B rawa2ad398e44bc249437b7bba06b7431521604968753b071c3f1437069d376df3e/payloadd60372546beb790f3ed73eb44342b27e5e6784f898d45ae3890a9cbbe60365b8。

Muse launch04未执行，ROOT抓住OUT已建却断言不存在、错误base路径、status递归读3.4GB权重三个控制bug。fresh05保留worker04/render04/adapter05科学源，修正existingemptyOUT/absoluteBASE09+FUTURE10/只读已知产品。index `.codex/scratch/cross_tfg_model_deploy_attempt12/muse_packet_attempt05/request_index_attempt05.json` SHA39940ce274b6a7bb381cb2d48ec759b2c569c2e3edae2f5a43fab0b9b7ba794f，两个请求已ROOT全文/SHA/大小审阅，尚未派发。source worker467e03cbf8fdc6410328e2f08c4521c469b97cd59ec05d50b100cabe1a42f664、renderer43c38a280ecdf6f311aebdbfcf8b8f4e25137b7ec9357ff8c616fcd6dc3efd6b、adapteredbdadcabc98c0fd9d44985e3f18a6cbeb76ae4ccf785b33f66e94d596b72ba3。GPU先Muse actual，完成释放后v3 actual96step fit，不能同时争抢。

### 运动、内容、特征与归档

visual/speech guards02 ACTUAL PASS：worker100556/759733901 start11:05:36Z finish11:24:11Z、exit0/PIDgone。300/300 visual和speech planned全完成、300NPZ，两manifest/sealedJSONL；status rawda9c8634be0103780886db3fed9ab7f6cfb4bef683d4372bd3d173a27b33b176。visual audit1920B SHA9b155d391bd4779113a64d93b91c1ec6578ce7afad1b0a2fe6f29c2fe316f0dc，speech1857B SHAb2f3c29704055df92fc747b8b603f53b116c15b5584f59991755a7d232e91818。CPU提取/XNNPACK、EGL初始化，无TFG forward；PASS是hash/clock/支持完整，不是质量阈值，开口变大不能自动称更好。原音40msRMS与嘴部25fps zero-lag关联、±15lag仅描述，30commonframe门槛/all5共享支持，需独立分析。

ROOT实际terminal tar归档2523260B SHA12ab5f893104d447aceeb005076d578f23548153ad3b667b151033b32e497de6，remoteNS/stageb_posthoc_guards_attempt02_terminal.tar.gz。fresh≤8MiB下载请求 `.codex/scratch/cross_tfg_audio_head_20261003/stageb_posthoc_guards_attempt06/01_download_terminal_archive_request.json` SHA0906131fad95f94723c218a5645ac24af730d71f528f7c4f6cc623691cdf54已ROOT全文审，未派发。目标 `runs/.../stageb_posthoc_guards_attempt02_readback/`，下一步逐成员/manifest与运动统计复核。

ASR拟100 unique sample×method waveform（20natural/80enhanced，300seed渲染各组三seed PCM一致），直接Qwen3-ASR-flash/Beijing/base64WAV/zh/ITNfalse，无reference text hint。官方AISHELL transcript实际本地10,091,431B SHAb5f33b9e0b47548e20a5ea4e504297f80df41d559133924c4e5b7b544c15b5c4，141600 uniqueIDs；HF revisionbbe295d530192a4cd41644b711c9aecd087df653。subset20exactID，13旧refs13/13规范化一致；subsetmanifest10352B rawb0a08345a35f61a910a03ea5098b525541157c322b5faf69046c6227ec34803d/payloadf72b258cf2627bde99d986b384500e59fc31ca8e108a0b994d3e0cb50a87c28c。ROOT fullsource01捕获100calls后scope.sample_meta NameError，未API调用；fresh02唯一修复len(unique sampleIDs)，SHAe380ee51a6de42a3909d5b661020cf267f1c8cdc5d29db9ab0aec9868ee943df/AST PASS/rootdiff PASS。agent准备key-freebundle/launcher，密钥仅runtime注入，不在请求文件/argv/BM/log；尚未API调用。

embedding replay02源SHA4874c2469ee51bf9c29486ff96119ce6878919255dec8791d1d62d2c0f418936已ROOT全文/diff approve，修正确nestedPCMfield与exactsourceclock；不改nominal tail/窗口，nullable仅render index。待sealedpacket/实际GPU捕获official1024维与原31lagrawmatrix parity，独立sidecar不替代primary。

portable300 closure archive447534356B/7531files/SHA05260729197c87262b2f97402b196be78776e6cf2c6ef4914dbb6e292f328eb4已真实copy-only生成在JOINTROOT/stageb_dev_artifact_bundle_attempt02.tar.gz；agent准备≤8MiB chunk/完整成员restore/sourcebeforeafter。源未退休，root~250MB/SHM~9.5GB/localBaidu可用。

### 持久入口和本机易失状态

Task `tts-exp/codex/tts-exp/开发跨-tfg-的自然音频增强头` actualwholeupdate12、Experiment `tts-exp/experiments/跨-tfg-自然音频增强头的训练与原音换轨验证` actualwholeupdate11已readback，receipts `.codex/scratch/cross_tfg_task_snapshot_receipt_attempt12.json` /`cross_tfg_experiment_snapshot_receipt_attempt11.json`。两snapshot phaseactual_joint300_v2_negative_fixedlag_v3_and_cross_frontend_inference，ACTIVE/IMPLEMENTED，guardsRUNNING/7refs缺失已过时，下次整体刷新。CONTEXT93行本turn已更新300/v3/Dittofiletype/Musecfg，HANDOFF40行ignored；README/transferiteration未提交。branchcodex/cross-tfg-audio-head-20261003 HEAD4a2422c7b12afaacda9b702d3321682e1bcac6bb local/unpushed，无PR；legacy/tests dirty保留，不stageall/push/pytest。无.codegraph目录，rg可用。

privatePTY50274/bridge2133仍活；dispatcher `.codex/scratch/dispatch_requests_root_attempt02.ps1`，每请求先ROOTread+hash、实际response.ok/code权威。服务器root@connect.weste.seetacloud.com:12349，ED25519 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；password/API只runtime，不写文件。NS `/root/tts-research-system-20261001/cross_tfg_audio_head_20261003`，literalW2L/syncnet venv不resolve symlink；compiledSO root/assetsweightsoutputsSHM。当前团队cross_tfg_execution(v3/portable)、local_recovery(Ditto/Muse)、mechanism_audit(guards/ASR/replay)均Luna/max。

## Verification

实际300评价独立重算、v3 fullsource/compactoverlay18members/hash/payload/optimizerstep/单行worker12GiB修复、Muse21实际preflight、Ditto实际终态失败+33fileAST、300运动终态与2.5MB实际archive、ASR全部官方refs与02一行修复、Task/Experimentwholeupdate/directreadback、CONTEXT/HANDOFF实际更新均已完成。Muse推理/v3 fit/ASR API/embedding replay/跨TFG dev/40confirm尚未执行，不能记成效果。无pytest/unit tests。Router/实验指令/CodingSession/config/samechat与legacy完整page/前note直接身份与freshGit/GitHub后不可变created/readback。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed baseline](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current HEAD `4a2422c7b12afaacda9b702d3321682e1bcac6bb` local/unpushed。

## Observations

- [result] v2较RMS ΔC−.017/D0+.057，local较RMS+.033；新学习目标应直接约束固定时刻与norm，而不是扩大B。
- [result] 300条实际视觉和原音运动关联已完成封存；Muse21项资产预检PASS，Ditto实际缺filetype未生成。
- [decision] v3有界RMS参考与旧rms_fixed分别记账；同8历史fit和96steps、40confirm保持未读，control修复fresh不可变namespace。
- [blocker] 实际Muse/Ditto资格、v3 fit/newdev、独立保护分析及heldout/confirm还未完成；新方法目标仍ACTIVE。
- [next_step] 派发已复核Muse05和guard下载06，实际读回推理/运动数据；Muse释放GPU后立即实际运行v3 fit并推进跨前端开发验证。

## Relations

- continues [[Codex checkpoint - 2026-10-03T11-29-47Z - Actual development decline and fixed lag objective]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[008-fixed-lag-and-rms-reference-head-objective]]