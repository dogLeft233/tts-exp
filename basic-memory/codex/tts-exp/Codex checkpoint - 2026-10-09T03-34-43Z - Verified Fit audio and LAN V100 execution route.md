---
title: Codex checkpoint - 2026-10-09T03-34-43Z - Verified Fit audio and LAN V100 execution
  route
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-09-t03-34-43-z-verified-fit-audio-and-lan-v100-execution-route
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 5688784dd5c6c1272074f0c4b6c5ff79f117c907
started: '2026-10-09T03:34:43Z'
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

# Codex checkpoint - 2026-10-09T03-34-43Z - Verified Fit audio and LAN V100 execution route

## Summary

四快照×40条Fit原音的160次实际增强应用已完成并由ROOT逐文件复核；新增局域网V100已通过公钥登录和模型资产核验，正建立隔离运行环境与白名单输入传输。

## Story

用户先要求恢复TTS增强TFG的研究路线，随后要求做可适配多个TFG的自然音频增强头：增强音频只驱动生成视频，成片换回完整原自然PCM和原时钟。ROOT负责机制、协议、BM与结论；代码和实验交给GPT-6 Luna/max，失败后继续迭代。最新用户补充了Ubuntu/V100研究机并确认用户名wjj，已有公钥可直接连接。

旧step120在Wav2Lip有收益而MuseTalk近零、Ditto下降。源码追溯确认训练只经过Wav2Lip，因此ADR022冻结Fit40内用三模型实际视频/原音SyncNet分数选择四个快照并优化同一组9参数。新实现已解决Fit分区、anchor15、真实NPZ身份与归约后端等工程问题；160次波形应用也已真实完成。现在进入新机器的实际原生视频生成与评分阶段，工程验证尚未成为新的科学效果结论。

## Working State

持久任务《开发跨 TFG 的自然音频增强头》仍active，实验《跨 TFG 自然音频增强头的训练与原音换轨验证》仍implemented，ADR022仍open。新Dev20/10人、Confirm80/40人及LatentSync保持封闭，ADR021准入阈值不变。四快照30/60/90/120来自legacy fit40_native_transition_head_v3_attempt11的真实连续120步，绝不是未知DirectD0的终局。

实际输出在machine-local `.codex/scratch/fit40_snapshot_audio_application_attempt04/application_output/`：160/160，40条原音/20说话人，总45,261,808B，mono16kHz float32 RIFF。receipt106427B/rawSHA74369d24f09a38e47f830d724bb2bb896f33d47791a7d435bbf4a0d482c60743，selfc37bd0e3d0ee2fcab1fe983b94e6e8692cc75c20616b8c05a498bd37e11a639c；ledger555030B/SHA656b25fff7622be257719a53161421656a24c9f829dadd45c599538b13437874。最大RMS误差1.080632340555443e-8，最大谱掩码2.7626898288726807dB，长度/原时钟/finite检查全部通过。正实STFT乘法保留乘法时相位，不保证iSTFT后重算相位逐点相同。状态COMPLETE_AUDIO_APPLICATION_ONLY_UNSELECTED_UNSCORED，没有视频/SyncNet/选择/优化更新。执行环境为WSL marker Python3.11.15/Torch2.7.1+cu118/NumPy2.4.4/RTX4060；不复用为V100资格。

失败保留：application02在roster投影检查前停止、03在plain-int flux计数.item()钩子处失败且0写入行；04只修两个计数转换后完成。源WAV manifest61469B/SHA0894f33df8504f9db7f3074ea901bcd987de23d533c029431b9150325e0cd020，固定RMS0.0376838172675746。最新snapshot index `.codex/scratch/fit40_training_snapshots_attempt05/legacy_native_attempt11/snapshot_set_index.json`34139B/rawSHAf6fe8ea2f447194921ae1b505bc4dd8b7dd40fe57363e2cee1b13980a233e338，canonical921d1552c564795ec0da189440ead48874421506055594136bf6b293e9f4a04f；仍EXTRACTED_UNSELECTED_NATIVE_ENDPOINT_UNBOUND。

局域网✅ `wjj@192.168.50.2:22`，公钥BatchMode/StrictHostKeyChecking=yes已实际登录；hostwjj-Default-string，V100-SXM2-16GB，UUIDGPU-d12b3309-c299-da83-064d-a3cd619bd560，driver580.178.04，探测时free16140MiB且无compute任务。rootdisk最近18.61GB、SHM16.77GB可用，每次重测真实容量。项目`/home/wjj/tts-audio/tts-exp`，root git clean；Wav2Lip HEADbac9a81e63ecc153202353372e5724b83d9e6322有三处旧tracked修改，SyncNet HEAD907c0b579c2e2d83f0eae1b2ac9e720cde4e5623有旧mode修改，保留并固定源码。GAN435801865B/SHAca9ab7b7b812c0e80a6e70a5977c545a1e8a365a6c49d5e533023c034d7ac3d8，S3FD89843225B/SHA619a31681264d3f7f7fc7a16a42cbbe8b23f31a256f75a366e5a1bcd59b33543，SyncNet54573114B/SHA961e8696f888fce4f3f3a6c3d5b3267cf5b343100b238e79b2659bff2c605442逐字节匹配。512×512 RGB portrait253479B/SHAe3a695edf22a89cac40184453131019874a65e15592593d5be30e7f71373d9c6。连接写入BM server-access，密码/API永不入文件/argv/日志。

ASR子代理已创建隔离`/home/wjj/.venvs/w2l_syncnet_v100_attempt01`（system-site-packages来自autoavsr Python3.8.20/Torch2.1.2+cu121）；底层实际CUDA分配1MiB/sync/release成功，干跑依赖解析通过，计划仅添加音频依赖，不替换原Torch/NumPy/SciPy/OpenCV。安装/完整import及模型资格仍待实际回执。Signal子代理已核验40/40原WAV字节/PCM/时钟、四快照、人像和代码，49文件6,161,313B白名单已审查，通过deterministic archive + safe extractor转入新workspace `/home/wjj/tts-audio/fit40_w2l_syncnet_qualification_attempt01`；封包/传输/远端读回尚在执行。Dev子代理并行寻找/部署MuseTalk和Ditto；此前bounded目录未找到不等于全机不存在，不假设V100支持BF16或已有TRT兼容。任何大下载需实测容量与5GiB余量。

ADR022仍要求实际240行identity/rms_fixed基线、四候选合计720唯一行和三后端真实readiness/capacity绑定。六方法共同FULL支持、每模型Cbest对两参考正、D0/Canchor不恶化、均值比RMS至少+0.050。完整但不合格可按预设违例函数确定优化初值；缺行/失败阻断。20步SPSA对实际视频官方原音分数优化，k5/10/15/20完整Fit中心才参与选择，总2160物理评分行。canonical torch_cpu归约；numpy_diagnostic禁止准入/初值/更新。

旧candidate180/controls360 FULL Cbest candidate−RMS等模型+0.011149（W2L+0.068454、Muse+0.000900、Ditto−0.035908）；D0正确reference−candidate改善−0.033662。GUARD20为171适格/9不适格，保留180完整分母，不能用子集救活端点。该方法尚未达到跨TFG目标。旧AutoDL root@connect.westd.seetacloud.com:18625的DirectD0仍仅唯一STARTED_DETACHED回执，status04两请求0response；私有PTY非空只读遭host自动审批拒绝，绝不重放launch。新V100提供可继续执行路径，不因旧远端问题停止目标。

Git仍codex/cross-tfg-audio-head-20261003，HEAD`5688784dd5c6c1272074f0c4b6c5ff79f117c907`本地未push；canonical GitHub仓库与该branch空PR已本轮重新核验。五处既有tracked dirty保留，大量新实验代码、忽略BM/runs/scratch及活动环境均属本机状态，当前无新commit/push/PR。HANDOFF中application03执行中段落待同步为04完成；本checkpoint保存最新确证状态。

## Changed Files

- `scripts/experiments/cross_tfg_audio_head_20261003/fit_only_spsa_selection_attempt04.py`：64975B/SHA1a9e75f32d4434d009d2b8a96dc1736ebf16054e43608581e3503079d8e7aa8d，实际rms_fixed身份不改写，强制Fit/anchor15/raw NPZ metadata，canonical torch_cpu。
- `tests/experiments/test_fit_only_spsa_selection_attempt04.py`：34034B/SHA71c134372eb2c22d6781686da2082d5d6fe74b6a3e0c5dd6922f3cb7168af2df。
- `fit40_training_snapshots_attempt02.py`、`build_fit40_snapshot_stagea_plan_attempt02.py`、`run_fit40_snapshot_stagea_attempt02.py`位于同一scripts子目录；`tests/test_fit40_snapshot_stagea_attempt02.py`覆盖源提取/官方绑定要求，旧attempt/index均保留。
- `.codex/scratch/fit40_snapshot_audio_application_attempt04/run_fit40_snapshot_audio_application_attempt04.py`：33954B/SHA5140ccaaf011193c748603ea9f6e4531fbd825c47dcbd754ebb4448ae9afc0f2，160实际输出和全量ledger/receipt为权威产物。
- BM `tts-exp/docs/reference/server-access`新增已验证局域网V100入口与资产，不含凭据；实际reference readback receipt在`.codex/scratch/root_record_lan_v100_access_attempt01/`。

## Verification

ROOT独立实际运行WSL marker `CUDA_VISIBLE_DEVICES= python -B -m unittest tests.experiments.test_fit_only_spsa_selection_attempt04 -v`19/19 PASS；另实际snapshot工具5/5 PASS。测试使用合成官方[N,31]端点，覆盖完整720行rms_fixed、Fit分区、anchor、raw-array身份、缺失NO_UPDATE、完整中心、NumPy诊断禁止更新及Torch数值检查；没有生成新科学分数。

ROOT实际用独立NumPy/RIFF只读解析逐一复核160WAV：receipt raw/self和160row self hashes、每文件字节SHA/PCM SHA、RIFF fmt(3,1,16000,64000,4,32)、finite float32、源N/clock、实际RMS与记录≤1e−14及基线≤1e−6、四方法相同40sample/speaker/sourcePCM全部PASS。已执行完整工具结果恢复成`root_160_audio_readback_preserved_receipt.json`，保存事实并非再运行模型。Signal另报告40源WAV和49文件allowlist独立校验PASS；远端传输、native qualification及三模型评分未完成。

ROOT实际SSH公钥登录和V100容量探测通过，子代理逐字节读取GAN/S3FD/SyncNet/person portrait与dirty源码。Startup Router、实验指令、Coding Session/Task schemas重新读完；same-chat coding/codex/legacy检索分页、去重并直接核对前一checkpoint的exact agent/session/started。新checkpoint由CLI已验证的write_note(overwrite=False,output_format=json)创建，要求actioncreated后直接读回精确host身份。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)，当前local SHA不构造GitHub链接，branch PR实际[]。
- `tts-exp/docs/adr/022-fit-only-cross-tfg-native-score-refinement`：冻结Fit内联合目标与完整分母。
- `.codex/scratch/fit40_snapshot_audio_application_attempt04/application_output/application_receipt.json`：160次实际波形应用的权威receipt。
- `tts-exp/docs/reference/server-access`：V100持久连接信息与执行机器边界。

## Observations

- [result] 四个真实Fit快照在40条原音上完成160/160增强应用，ROOT逐文件byte/PCM/clock/RMS复核PASS；尚无新视频/SyncNet效果。
- [result] selector04 ROOT19项、snapshot工具5项实际工程测试PASS；canonical Torch和rms_fixed身份修复已经落实。
- [result] wjj@192.168.50.2:22公钥登录、16GB V100和精确W2L/S3FD/SyncNet/portrait资产已核验，新隔离runtime与输入stage正在准备。
- [decision] 继续ADR022 Fit-only三模型真实评分联合优化，保留RMS/3dB/原时钟和完整分母，新Dev/Confirm封闭；不同GPU必须重新资格与同平台对照。
- [blocker] V100完整运行依赖、W2L原生资格及MuseTalk/Ditto部署尚待实际执行；旧AutoDL DirectD0启动后状态仍未知，禁止重放launch。
- [next_step] 在新V100完成隔离运行环境和白名单输入的实际读回，先运行W2L原生生成/原音SyncNet资格，再接齐MuseTalk/Ditto实际Fit评分并按ADR022优化。

## Relations

- continues [[Codex checkpoint - 2026-10-09T02-52-36Z - Fit-only joint native refinement and verified snapshots]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[022-fit-only-cross-tfg-native-score-refinement]]