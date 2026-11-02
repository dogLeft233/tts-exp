---
title: Codex checkpoint - 2026-10-09T04-24-23Z - Native Wav2Lip verified and cross-model
  Fit execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t04-24-23-z-native-wav2-lip-verified-and-cross-model-fit-execution
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T04:24:23Z'
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

# Codex checkpoint - 2026-10-09T04-24-23Z - Native Wav2Lip verified and cross-model Fit execution

## Summary

新局域网V100实际完成原音/RMS各两次Wav2Lip生成及原音SyncNet工程验证，四行均PASS；已核验160候选波形、49文件远端输入和F/B双份归档，正在准备同平台三模型Fit内联合优化。

## Story

原问题是TTS为什么增强TFG。用户将目标推进为适配较多TFG的自然音频增强头：增强音频只驱动视频，成片与评分换回完整自然PCM和原时钟；需要超过RMS且失败后继续产出完备方法。ROOT维护理论、协议与BM；实验/代码由GPT-6 Luna/max承担。旧step120只经过Wav2Lip训练，在Wav2Lip获益、MuseTalk近零、Ditto下降，因此当前ADR022在Fit40内用三个模型的真实生成视频/原音官方SyncNet联合选择和优化同一组9参数。

最新用户确认wjj是192.168.50.2的SSH用户名。现有公钥已实际打通新16GB V100。此前旧AutoDL状态读取问题不再阻止研究执行；不重放旧DirectD0未知launch。新平台四条工程控制通过，说明生成、换回原音和原生评分可运行，尚不能代替增强头超过RMS的正式比较。

## Working State

任务《开发跨 TFG 的自然音频增强头》仍active；实验《跨 TFG 自然音频增强头的训练与原音换轨验证》仍implemented。新Dev20/10说话人、Confirm80/40及LatentSync封闭，ADR021准入不改。

权威设计是`tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement`：四快照30/60/90/120+identity+rms_fixed，40Fit×3模型=720物理行，共同FULL查询支持和完整分母；每模型Cbest优于两参考、D0/Canchor不恶化、等模型RMS收益至少+0.050。完整却不合格才允许预定违例规则选优化初值；缺失/失败不能删行救活。随后20步真实三模型SPSA，k5/10/15/20完整Fit中心选择，总2160物理评分行。canonical torch_cpu归约，NumPy诊断禁止选择/更新。selector04含模块级tuple[...]别名，不能在W2L的Python3.8直接import；在Python3.10共享环境或本地marker运行，不为此无谓重写。

160实际候选波形已由ROOT逐文件byte/PCM/RIFF/finite/clock/RMS复核PASS：`.codex/scratch/fit40_snapshot_audio_application_attempt04/application_output/`，45,261,808B，4快照×40条/20人，最大RMS误差1.080632340555443e-8、掩码2.7626898288726807dB。receipt106427B/SHA74369d24f09a38e47f830d724bb2bb896f33d47791a7d435bbf4a0d482c60743；ledger555030B/SHA656b25fff7622be257719a53161421656a24c9f829dadd45c599538b13437874。仍未选择且无候选视频分数，执行是RTX4060/WSL marker。latest真实snapshot index为`.codex/scratch/fit40_training_snapshots_attempt05/legacy_native_attempt11/snapshot_set_index.json`，34139B/SHAf6fe8ea2f447194921ae1b505bc4dd8b7dd40fe57363e2cee1b13980a233e338；不是DirectD0终局。

ROOT新建176-member ZIP，52,545,709B/SHA914eec35802ff40d4a8ba34f84faac0771d431889656d5641ea7293dcb01892d，封装160波形、输入和回执/代码。实际F/B两份归档每成员全量SHA再读PASS，源文件保留：`F:/tts-exp-archive-transfer/fit40_snapshot_audio160_verified_attempt01/`与`B:/experiment-archive/cross_tfg_audio_head_20261003/fit40_snapshot_audio160_verified_attempt01/`；archive_readback_receipt self d6c5bea198f2598861e65dc2e273ba2d4f84e20d49a5fdc2f166dd8eda46db8a。B是挂载端实际读回，网盘后端上传持久状态未检查。E仅约434MB，后续大下载到F；远端和B缓存都需真实容量检查。

✅执行机`wjj@192.168.50.2:22`，公钥BatchMode/StrictHostKeyChecking=yes；hostwjj-Default-string、Tesla V100-SXM2-16GB、UUIDGPU-d12b3309-c299-da83-064d-a3cd619bd560、driver580.178.04。项目`/home/wjj/tts-audio/tts-exp`。GAN/S3FD/SyncNet/512 portrait exact byte/SHA与既有dirty源码已核验并保留。输入workspace`/home/wjj/tts-audio/fit40_w2l_syncnet_qualification_attempt01`已49文件6,161,313B实际传输及40WAV PCM/快照远端读回PASS；证据`.codex/scratch/fit40_v100_snapshot_stagea_attempt01/`，remote receipt25874B/SHA272bcc09e1112d3af1fdfb2686dad4402b3fe7b25c246047158e261c1470acfb，独立QC1824B/SHAd6d9551a9911608eb02b42fef0b5df5c7874f7e7800b8edbab83ca0e575be02b。

✅W2L fresh runtime`/home/wjj/.venvs/w2l_syncnet_v100_attempt01/bin/python`：Python3.8.20/Torch2.1.2+cu121，隔离18个音频包，librosa0.10.2.post1、numba0.58.1等；pipcheck/import/help与CUDA1MiB/sync/release实际PASS，原autoavsr保留。工程smoke02 runnerPID24896已exit0、COMPLETE_ENGINEERING_QC、4/4PASS。固定BAC009S0052W0495/S0052/j15/100992自然采样，identity/rms_fixed各rep01/02，RMS0.0376838172675746。同方法重复的全部官方数组、距离矩阵文件、generator PCM和视频elementary stream身份一致。输出`/home/wjj/tts-audio/fit40_w2l_syncnet_qualification_outputs_attempt02`总41,767,131B；代码`.../fit40_w2l_syncnet_engineering_code_attempt02`。

工程scope `engineering_wav2lip_gan_v100`并非ADR022 StageA native-frame资格，不将别名伪装正式适格模型。smoke01因protocol缺字段/官方源码重设cuDNN benchmark在dispatch前停止，保留NOT_RUN；02追加精确源码pins和SFD/W2L actual-forward flag guard并通过。评分自然PCM16窗口0–100480，157名义时钟/track帧和152×31官方queries；render timestamps只有155 decoded frames，末2帧map明确`nominal_clock_outside_render_decode`。这个源时钟/实际视频区别需纳入正式资格审查，不能声称157 decoded render帧或盲拷该规则。原完整自然PCM并未用增强PCM替换。

ASR下载35终局文件657,966B至`.codex/scratch/v100_fit_stagea_smoke_attempt02_terminal_readback_attempt01/`，index02 39146B/SHA221f68e731b2badddb095195df3297245a095063bae879c409cabf34a192db67，self122809877cfc76b71d2336cde6aefc3bfcf387fea5c69b5e74091ac9cfe9679c。ASR独立terminal QC实际PASS，24722B/SHA72ad7aed36f687357bee3d24ae92c44e12c30c5cb13be82eead2931cc857a98f，self7d572d26c1ff3a92904fb4a6dc709469ec854f1c1ff5962f68166f93ecd951a4，核验35pins/各行/flags/PCM/clock/meta重复。ROOT尚需独立打开四NPZ实际数组，视频body41MB未下载，重复media身份来自实际producer hash与终局元数据，不伪称ROOT本地video字节读回。

✅第二fresh共享环境`/home/wjj/.venvs/cross_tfg_muse_ditto_shared_torch251_cu121_attempt01`已实际安装Torch2.5.1+cu121、vision0.20.1、audio2.5.1，Python3.10.20，5,181,177,856B。ASR作为exclusive GPUowner刚实际1MiB allocation/fill/sync/read/delete/releasePASS，V100cap7.0/cuDNN9.1，allocator0→1048576→0。源mfa不改。当前homefree12,732,755,968B；SHM16,769,675,264B，但RAM/cgroup还需测量。Muse+Ditto pending assets6,300,900,762B，保留5GiB后剩约1.063GB依赖/源码空间；不复制大wheel/cache/archives。可采用共同环境或精确overlay，childvenv system-site不是自动继承parentvenv已装包。

Ditto用Strict09冻结PyTorch12assets而非旧TRT，官方HFrev e4a2f60328ee7c32af585ac4b3cce299e4c8e254、composite dc29eacae64c53a741e58a28e94081c5b1d0a04714f0b478fc7114c6a50a8478。Dev已实际新建官方source clean commitc3e47eee2e626500017a0556b470d6d4182f85e8，2,592,768B，三关键源码匹配Strict09pins。路径`/home/wjj/tts-audio/tts-exp/runs/cross_tfg_audio_head_20261009_v100_native_deployment_attempt01/model_sources/ditto-talkinghead`。CPUonnxruntime1.20.1/TorchCUDA方案；冻结renderer有旧host import报告/Cython.so pins，需透明新V100路径/编译资格，数学/权重不变。

Muse冻结实际V1.5（此前临时1.0假设已纠正），mmpose-free官方Whisper/UNetfp16/face-parser/固定人像Haar框，5weights+4config总3,986,182,419B；manifest`.codex/scratch/v100_musetalk_asset_prepare_attempt01/asset_manifest_attempt01.json`8255B/SHA53d556e02da47af1b52c26aa8bc53102af75686b5bff7e50fd74a41c6cffd9b6，self0d9605eef17019577eee77bd38b3a34d824f66abc48e10ddf4a12dbf40024d3f。MuseOfficialrev2bcb936e2fddb4d86db4c62fd45b387d0c061571、VAEa3d9d3447e87215009ccbbd178356a9fcf0b982d、Whisperaaf41dd3aafcff61dad6f603d5de0ce1317324dc、parser0073b233a5a3c4b1377d4dbf49245017938a72b5。Signal准备direct-to-final worker，逐块byte ceiling、same st_dev和5GiB reserve；旧attempt保留，当前无model inference。Muse目标`.../model_assets/musetalk_v15`，Ditto目标`.../model_assets/ditto_pytorch`，新目录互不覆盖。

真实每行工程media约10.4MB，完整720或2160无法堆SSD。ASR应测RAM/cgroup后使用SHM有界working rows，或六方法单clip归档并完整bytereadback后清理已验证临时media；canonical NPZ/ledger/clock/mediapins持久可访问，全planned denominator、failure证据保留。远端`/home/wjj/百度网盘/tts-exp-project`属挂载clouddevice85，subdir mountpointfalse不是本地盘。旧Ditto3.376GBarchive实际扫描是ONNX+Ampere_Plus TRT、无.pth，不能充当V100 PyTorch weights。

旧step120180candidate/360controls：FULL等模型candidate−RMSCbest+0.011149，W2L+0.068454、Muse+0.000900、Ditto−0.035908；D0正向reference−candidate−0.033662。GUARD20仅171/180适格、9同一clip三模型三seed不足41queries，完整端点INCOMPLETE，保持分母。仍需新联合方法，未跨模型超过RMS。

Git codex/cross-tfg-audio-head-20261003/`5688784dd5c6c1272074f0c4b6c5ff79f117c907`仍本地未push，freshcanonicalGitHub/空branchPR已读回。既有五tracked dirty保留；新untracked experiment代码、ignored BM/runs/scratch、远端venv/process和F/B挂载是machine-local或外部资产，不能当已提交仓库状态。BM Task/Experiment与HANDOFF的160波形/49输入已同步，四条smoke与共享环境进展待ROOT新读回后同步。

## Changed Files

- `scripts/experiments/cross_tfg_audio_head_20261003/fit_only_spsa_selection_attempt04.py`64975B/SHA1a9e75f32d4434d009d2b8a96dc1736ebf16054e43608581e3503079d8e7aa8d及对应test；来源核验、Fit/anchor15/canonicalTorch已经实现。
- `.codex/scratch/v100_fit_stagea_smoke_attempt02/`：runner32900B/SHAf11b49e89ac900170c7f6fd8c00c43aba4bb4cf630c26e4d94c508f6b685dcb2，child14150B/SHA4341b002cb217118c12b1efc7ef97d1afeb8bb47bad7b2ce1230c6e6d35fac5d，protocol11493B/SHA180f6ac0f5c29b144934fbd1827853f66c93ec6053816d83a7e60667406aa243，phaseA stage9458B/SHA7006cfaefe56e979008d7e3bbffb853b972d9fbf074f407a10f71cda3e400d1d。
- `.codex/scratch/root_archive_fit160_verified_attempt01.py`：ROOT只做归档，实际F/B manifest/ZIP/readback已经完成。
- `.codex/scratch/v100_fit_stagea_smoke_attempt02_terminal_readback_attempt01/`：35 terminal artifacts+independent QC，ROOT原始NPZ数据复核待续。

## Verification

此前ROOT实际selector04 19/19和snapshot工具5/5工程测试PASS，合成arrays不当科学效果。此前ROOT160WAV全量byte/PCM/RIFF/finite/clock/RMS PASS，本轮保留实际回执。新F/B两份176ZIPmembers由ROOT实际再读SHA PASS。Signal49文件远端全量byte/40PCM PASS。ASR新W2Lruntime pip/import/CUDA和四条实际生成/原音评分4/4PASS；ASRterminal35文件元数据/QC PASS；共享TorchCUDA tinyprobePASS。ROOT尚未完成四NPZ全数据独立复核，候选240/720评分没有完成。

Startup Router、实验指令与CodingSession/Task schemas已重新加载。freshsame-chat coding/codex/legacy搜索分页去重并read上一checkpoint，核对exact agent/session_id/started。CLI source确认write_note overwrite默认False且显式转传output_format=json，新checkpoint只接受actioncreated并直接readback核对hostidentity；不改旧checkpoint。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)；SHA`5688784dd5c6c1272074f0c4b6c5ff79f117c907`本地未push，不虚构commitURL；freshbranchPR[]。
- `tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement`与`tts-exp/docs/reference/server-access`。
- `.codex/scratch/v100_fit_stagea_smoke_attempt02_terminal_readback_attempt01/download_request_index_attempt02.json`：实际35终局下载白名单/pins。
- [Ditto official repository](https://github.com/antgroup/ditto-talkinghead)：官方PyTorch/旧GPU路径，实际冻结源码优先。

## Observations

- [result] 新V100完成identity/rms_fixed各两次真实W2L视频+原音SyncNet，4/4工程PASS且同方法byte/pin重复一致，尚未评价增强头收益。
- [result] 160真实Fit候选音频ROOT逐一QC通过；49输入远端全量读回与176-member F/B归档均通过。
- [result] 新共享Python3.10/Torch2.5.1+cu121环境安装及actualV100 tinyCUDA probePASS，MuseV1.5与DittoPyTorch冻结资产正在部署。
- [decision] 继续ADR022联合native原音优化及原完整分母；155decoded/157nominal帧必须显式审查，新runtime需要新资格，不将engineering alias作准入。
- [blocker] Muse/Ditto固定权重、精简依赖及V100路径/Cython资格仍待完成，正式W2L240slice容量/实际帧资格待执行；旧AutoDL终局仍未知且不重放launch。
- [next_step] 完成W2L终局原始NPZ/实际帧与容量独立核验，在V100启动40Fit六方法240行W2L slice并接齐MuseTalk/Ditto，以完整720行按ADR022选择及联合优化。

## Relations

- continues [[Codex checkpoint - 2026-10-09T03-34-43Z - Verified Fit audio and LAN V100 execution route]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]