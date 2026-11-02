---
title: Codex checkpoint - 2026-10-07T18-29-08Z - Running fixed120 rescore with restored
  model assets
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-07-t18-29-08-z-running-fixed120-rescore-with-restored-model-assets
project: tts-exp
agent: codex
git_sha: cd6039cdeca1a7ee3b3ca2e6efd2de2731cc4ef9
status: open
cwd: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
capture: deliberate
repo_root: E:/Documents/tts-exp
trigger: compact
model: gpt-6.1-sol
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
started: '2026-10-07T18:29:08.6598404+00:00'
branch: codex/cross-tfg-audio-head-20261003
hostname: LAPTOP-E133GSLE
username: dogle
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-07T18-29-08Z - Running fixed120 rescore with restored model assets

## Summary

冻结 Fit40/Adam120 的九参数局部谱形头已完成训练与独立更新链核验；保持最终权重重跑终端评分，同时九份 MuseTalk 资产和完整增强头 CLI 已实际恢复。

## Story

用户要求从 TTS 提升 TFG 的机制研究推进到适配多个 TFG 模型的可用自然音频增强头，并持续自主执行。增强音频只用于视频驱动，最终精确恢复原自然 PCM 与时钟。root 负责理论、协议、Basic Memory、私有 SSH 和独立复核，GPT-6 Luna/max 子代理负责实验与代码。当前头通过九参数调整稳定、上升和下降局部谱形，固定全局 RMS、采样时钟、平滑与 ±3 dB；40ms局部能量仍可变化，需作为机制路径分析。

attempt11 已真实完成 Fit40/20说话人、三遍同序、Adam120。root 读取七份原始终端，独立核验全部120步连续更新、九参数有限非零梯度及最终 checkpoint。训练末遍平均 ΔC +0.055 仅为训练诊断。no-grad rescore09 在0行因 GPU UUID 比较退出；实际只读探针证明 torch _CUuuid 字符串缺少 GPU- 前缀，而 nvidia-smi 有此前缀，两者128位标识相同。新 rescorer07 仅严格验证 UUID 格式后规范化前缀，保持全部评分、权重与120步绑定。root 检查27源包、实际绑定的17个关键字段、64项预检，fresh rescore10 已实际启动；尚未读取其终端结果。

MuseTalk07 已真实恢复全部九份固定权重、3,986,182,419字节，root 对照原 manifest 独立核验全部哈希、六个直接别名和九项别名成员。完整 CLI 七模块闭包也已真实安装。materializer04 元数据重试正在派发，不能将它提前描述为候选生成成功。Wav2Lip/Ditto 重复资格化 runner04 的17个路径模板在内存中展开，原始模板固定不变；root已读完整源码差异，校正后的请求 packet04 尚未派发。

producer05 已持久归档，root 独立核验 tar 全部4,979项与整包哈希通过。旧解码在 MKV 缺少 stream duration 时失败。实际单行探针证明 PCM 精确一致，音频62639样本、视频95帧/25fps、1ms timebase；格式时长是四舍五入后的最长轨时长。新 rawdecode-only02 以实际样本和 packet/frame PTS 核验，增加完整4979项库存绑定、精确25fps、timebase≤1ms；尚待 root 审阅并派发，不重新归档。

## Working State

冻结目标仍为同 GPU Dev20/10说话人×三seed×Wav2Lip/MuseTalk/Ditto：fresh identity/RMS360与candidate180。验收为三个模型均正、等模型相对 RMS 额外至少 +0.050，并通过 D0/Canchor、内容、视觉、GUARD20及完整失败分母。Confirm80/40未见说话人与LatentSync密封。尚无新 Dev 或确认结果；研究目标保持 active。

Git HEAD cd6039cdeca1a7ee3b3ca2e6efd2de2731cc4ef9 为本地未推送，branch codex/cross-tfg-audio-head-20261003。Task87/Experiment86及CONTEXT/HANDOFF比本次实际资产恢复和rescore10启动旧，需在取得终端后更新。五个无关旧修改保持不动。scratch、runs、忽略HANDOFF、本地Basic Memory及活动进程属机器本地或易失状态。

服务器 root@connect.westd.seetacloud.com:18625，指纹 SHA256:EPstg8jNN+u3Bs1JyQOqlBeq3w+a4VynoJAe48YnIV0；GPU GPU-d09622d0-3535-e662-bb45-3df2b049fcea。密码与API密钥仅在私有运行时。持久 NS /root/tts-research-system-20261001/cross_tfg_audio_head_20261003；易失 SHM /dev/shm/cross_tfg_audio_head_20261003。私有PTY57588/bridge07:12249仅由root控制。rescore10 PID52682/start_ticks325210527占用GPU；GPU作业串行，CPU安装/解码可同时进行。本地E约1GB可用，不下载整个归档。

## Changed Files

- .codex/scratch/fit120_final_rescore_packet_attempt10_revision03/：27文件固定包、rescorer07 UUID最小修复、实际绑定和launch；requests_attempt10_revision04保留正确单反斜线printf。
- .codex/scratch/root_fixed120_rescore_arrays_qc_attempt01.py：准备的独立NumPy终端统计验证，尚未执行。
- .codex/scratch/musetalk_asset_restore_attempt07/：实际全资产与别名通过响应及root独立QC。
- .codex/scratch/fit40_native_rms_geometry_attempt05_timing_probe_attempt01/：revision03实际单行PCM/packet/frame计时探针。
- .codex/scratch/fit40_native_rms_geometry_attempt05_root_rawdecode_only_attempt02/：新只读解码与库存/时钟门控，尚未派发。
- runs/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/：CLI闭包02实际安装，materializer04派发中，repeat corrected packet04待派发；不存在新Dev科学结果。

## Verification

训练独立120步完整链receipt SHA52ff06de9be9381126733204be9237194a3ce25174dceaac142775759c3e499c；checkpoint8d5eb607f77963291372b33245f13f9b9578ebf71657acf1b43243746c538eb8；ledger7f6d054993a96ca8db4808e6bd06b5751abd87a0c778ac7653404d379a1bec8d。保留rescore09失败response27cd70431a625d67ac401a0ab92917ddff08843aaf91ccb33e304f237367a4fa8，actual UUID probe93bd3334babc5d26139e1a80a0cd57a707082576f58181d44d887836ee3875ebb。

rescore10实际binding0020a84bcd95db2278c9c60f5dd4e902d2becce3b140827f2d6422377832c694、preflight6a38e06fa177489658ac519f62b2d226ac1a1ee9f2ff16b961ef565950693776、launchreceiptb2fe193e9fbb045685316ea96eb35b374f752cf302c6200f59b49ceca7025172；actual launchresponsece1155d7bf7673121361ddcbbf9cfe3c1ce494e96bf1d44990cd947da856341c7。root_rescore10_status_index_attempt01.json准备好，未派发。terminal成功时需下载七份原始文件，包含最终数组与RMSbaseline，独立重算40行/20说话人统计。

Muse07实际terminalresponse70f5ca9207df2882d57b41a15ab85aee526233620e9ff0f3355576c7a21da128b，PASS_ASSETS_REPORT_AND_ALIASES_EXACTLY_PINNED；root独立receipt位于该packet。CLI q08 actualresponse65c356caba55341174a722d13d583f8707de7fef3c6f601176f71ec6fcaaf836，PASS_CLI_SOURCE_CLOSURE_INSTALLED_OR_IDENTICAL。producer05整包c35092a0478e0cc9a4738f9918596465d00944ab17a8322171faae08f1d92b1a，全库存独立response21d1e6b024fba60ac54e8a7854610b757c1542eee3d14148857b0f92ac9fe7bad；tiny clock response78fcecf1488fc6fb0489519cf9d5bb8358ba98609fcd097cc1d1d6d6d98ad2b8e。未运行通用pytest；执行的是用户授权实验与原始证据核验。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)；cd6039c仅本地。
- 当前branch的GitHub只读PR查询无匹配。
- .codex/scratch/root_materializer04_retry_index_attempt01.json及root_rescore10_status_index_attempt01.json是下一实际派发入口；已存在response的请求禁止重复派发。

## Observations

- [result] Fit40/Adam120真实完成且120步独立链通过；固定最终权重rescore10已启动。
- [result] MuseTalk九资产与别名及完整增强头CLI实际恢复完成。
- [decision] 将UUID同一128位标识规范化，只修复工程表示；固定权重、评价与sealed确认门槛不变。
- [decision] 保留原PCM/时钟、旧失败和新目录；MKV计时使用样本/packet/frame证据，既有归档不重写。
- [blocker] 终端评分、全树raw decode、后端重复资格化和候选物化仍需取得实际终端证据。
- [next_step] 先读取rescore10终端并独立核验七份原始文件，再接续fresh三模型重复资格化及Dev540。

## Relations

- continues [[Codex checkpoint - 2026-10-07T18-02-31Z - Verified Adam120 chain and repairing fixed terminal rescore]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[015-same-gpu-deterministic-controls]]