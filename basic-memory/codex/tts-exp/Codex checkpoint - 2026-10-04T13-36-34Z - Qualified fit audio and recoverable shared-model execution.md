---
title: Codex checkpoint - 2026-10-04T13-36-34Z - Qualified fit audio and recoverable
  shared-model execution
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-04-t13-36-34-z-qualified-fit-audio-and-recoverable-shared-model-execution
capture: deliberate
username: dogle
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
agent: codex
branch: codex/cross-tfg-audio-head-20261003
git_sha: 0597b3dfbfd6bd5df93c96d7cb70be12ce476e59
started: '2026-10-04T13:36:34.1948576+00:00'
status: open
hostname: LAPTOP-E133GSLE
model: gpt-6.1-sol
trigger: compact
project: tts-exp
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-04T13-36-34Z - Qualified fit audio and recoverable shared-model execution

## Summary

跨 TFG 增强头继续研发，新增历史 fit 的40条端点音频已独立核验，failed-fit 输出已完成双盘可恢复归档，共享900行分析包已实际部署。

## Story

用户要求从恢复 TTS 增强机制研究推进到可用的自然音频增强头。增强音频驱动视频，随后恢复原自然 PCM 和时钟，primary 仍为 SyncNet Cbest；目标覆盖 Wav2Lip、MuseTalk、Ditto 三类前端，并做内容、视觉与保留模型迁移检查。用户授权服务器、百度网盘、下载及 Qwen TTS/ASR；root 维护理论和独立核验，实验代码由 GPT-6-LUNA/MAX 子代理承担。最新授权是在 ASR 质量不足时使用 Qwen ASR，凭据始终只走运行时。

已合格的 V4 在完整20 clip/10 speaker×3 seed×3模型开发集上，相对 identity 等权 Cbest +0.109，相对 RMS −0.044，不能满足跨模型超过 RMS 的方法目标。局部谱结构的独立结果仍支持下一轮研究局部谱变化和时序兼容性；相关量与掩膜范数不当作因果或物理功率。三个历史 fit-only 共享候选保持冻结，dev540 尚未运行，确认80 clip/40 speaker及保留模型 LatentSync 仍封存。

本次独立重算20个源 tar 的728,484,140字节哈希和6944个 WAV 头，验证320个规范槽位映射及40个唯一 PCM16/F32 端点。所有端点符合 ADR013 的原始 j=0/15 选择，无替换、无 dev/confirm 重叠。这为新增20 speaker 的分折拟合提供了真实音频证据。scanner 非端点的 endpoint_payload_status 标签有歧义，已要求未来准备版本修正，当前合格范围是规范映射和40个端点而非该标签。

failed-fit03 的1,291,222,198字节归档已实际完成 C 下载和全323块独立复算，F镜像及46089文件严格恢复亦实际通过。源输出尚未删除；新鲜只读退休审计 attempt02 已冻结待派发。随后处理旧 Ditto 输出，恢复原定 shared540 的5GiB SHM 入场条件。共享900行 join04 科学包通过四段桥接上传和完整哈希核验，已实际解包；分析本身待候选540终态。根代理澄清了执行阶段：packet07 的04仅CPU预检，候选WAV由05生成，ASR60必须绑定05实际输出，不能把04称为音频生产。

## Working State

Durable: 本地未发布 HEAD `0597b3dfbfd6bd5df93c96d7cb70be12ce476e59`，branch `codex/cross-tfg-audio-head-20261003`，无PR。BM Task42/Experiment41及 ADR007/009/011/012/013保留当前协议；Task/Experiment快照早于最新 fit40 和900实际stage，下一次更新需补入这些结果。历史 dirty 文件保留，报告26及最终方法尚待完成。

Machine-local: 私有PTY50274和bridge06端口10887/PID252640仍承载运行时访问，服务器 root@connect.weste.seetacloud.com:12349。秘密不写笔记。C/F failed-fit归档与F全恢复、ignored runs/scratch必须保留。SHM仍不足5GiB，failed-fit和旧Ditto的具体输出需经过归档审计后分别精确退休。

Ditto ONNX实际元数据表明输入Sub/Div在Conv前，但常量尚无实际解码结果；collector attempt04 当前冻结，支持 initializer 和 Constant.value TensorProto，先全模型独立哈希再有限读取。CPU增益/前端特征探针正在准备，无新视频或科学效应结果。

## Changed Files

- `scripts/experiments/cross_tfg_audio_head_20261003/scan_v5_fit_expansion_members_attempt02.py` 与 `runs/cross_tfg_audio_head_20261003/v5_fit_expansion_member_scan_attempt02/member_scan.json`：实际40端点资格；root独立收据 `.codex/scratch/v5_fit40_root_raw_member_qc_attempt01.json`。
- `.codex/scratch/recoverable_output_failedfit_transfer_attempt04/`：C323块/全包核验、F镜像和严格恢复收据，全部实际完成。
- `.codex/scratch/failedfit_output_retirement_audit_attempt02/`：只读具体输出审计包，尚未执行或删除。
- `.codex/scratch/cross_tfg_audio_head_20261003/shared_profile_900row_join_attempt04/`：四段上传 transport03 完成，科学包原样解包，join未运行。
- `scripts/experiments/cross_tfg_audio_head_20261003/inspect_ditto_hubert_input_constants_attempt04.py`：当前常量审计代码，待root审核派发。

## Verification

Fit scanner报告183939B/SHA1b4fd8472842c9c30c4611aa760b3c2b2d9f01193d3aaf7f487f5f8096b0a140。root独立20包全哈希、6944头、320槽位、40原始PCM和规范F32均通过，无音频替换或模型运行。

failed-fit归档 SHA234006a3cc5efef762f1a943b79529fe4443751fad7f56a76cf718579e888638；C323块及全包、F全包及46089文件逐成员 SHA 实际通过，严格恢复收据52551B/SHAe4ea0cef5056e92b5f1e1455d0183ce66a530e8af107228ac951c5f34cd65302。远端源 run和重复tar仍保留。

900科学ZIP18725992B/SHA45170129879fa501723876535652f5dcf86d558da29651cd9bdf8bc1f9b8d7d9，四段实际传输、重组及8个payload解包通过。曾有单文件超过桥接8MiB失败和旧transport变量引号错误，均保持旧包，改用新鲜transport03，无科学payload修改。

没有运行pytest或新增单元测试。常量审计、退休审计、shared540、900join和新的增强效应均尚待执行。

## References

- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)

## Observations

- [result] 40条新增历史fit端点已真实取得并独立核验，20speaker规范分折可用于后续有界候选拟合。
- [decision] 保持原音换轨 Cbest、等speaker/model权重、失败行保留及确认集封存；ASR60必须绑定05生成的实际60个唯一WAV。
- [decision] C/F双包及F全恢复合格后才允许新鲜审计与具体旧输出退休；原始 recovery04 包继续保留。
- [blocker] shared540 仍等待SHM达到原定5GiB；常量数值、候选dev结果和最终跨模型方法尚无新结果。
- [next_step] 派发已审核的 Ditto 常量读取与failed-fit只读审计，合格后精确回收旧输出，再启动冻结shared540。

## Relations

- continues [[Codex checkpoint - 2026-10-04T12-57-03Z - Qualified spectral evidence and recoverable cross-model continuation]]