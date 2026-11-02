---
title: Codex checkpoint - 2026-10-08T07-48-32Z - Actual repaired Dev run and offline
  candidate delivery
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-08-t07-48-32-z-actual-repaired-dev-run-and-offline-candidate-delivery
hostname: LAPTOP-E133GSLE
branch: codex/cross-tfg-audio-head-20261003
agent: codex
repo_root: E:/Documents/tts-exp
started: '2026-10-08T07:48:32.1514662+00:00'
username: dogle
git_sha: 7746b01f34b5850f11e322a0abc6934095f95f64
capture: deliberate
session_id: 01a0f26b-a3e7-7f91-b1e8-46813543042f
trigger: compact
status: open
cwd: E:/Documents/tts-exp
model: gpt-6.1-sol
project: tts-exp
repository: dogLeft233/tts-exp
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-08T07-48-32Z - Actual repaired Dev run and offline candidate delivery

## Summary
修复后的 Dev360 已实际启动，最新 14/360 全 PASS；离线增强头已打包，旧失败批次 compact 索引已生成并待持久复制。

## Story
用户希望把中文 TTS 的同步增强机制转为可迁移的自然音频增强头，并持续实验到形成可用方法。增强 PCM 只用于驱动视频，成片恢复完整原自然 F32 PCM 和原时钟；ROOT 维护理论、冻结协议、BM 和独立复核，GPT-6 Luna/max 子 agent 编码与实验。当前固定 Fit40/20 speakers、Adam120/三遍候选，不使用 Dev/Confirm 选择参数。Dev20/10 speakers ×3 seeds ×Wav2Lip/MuseTalk/Ditto 的 360 identity/RMS 控制及 180 candidate 是当前主要科学检验，三模型方向、等模型 +0.050 over RMS、D0/Canchor、内容、视觉和 GUARD20 均要检查。Confirm80 与 LatentSync 保持封存。

上一 attempt09 的 360 行已完整结束：120 Wav2Lip PASS、240 MuseTalk/Ditto 在 native 推理前接口失败。该失败用于修复工程链路，不作候选科学阴性。frontend09 在环境覆盖后恢复每行 native entry/control context，完整修复源码在本地 HEAD7746。现在 CPU worker10 的 12 请求已全部实际通过：22 输入原始 SHA、sealed seed03、binding09、plan09/360、preflight09 与 summary09 八产物已下载，ROOT 和 Signal 两份独立 QC 均 PASS。随后 launcher07/status08 的四请求通过，真正启动 controls18 新运行；最新只读 snapshot02 在 07:36:16 UTC 记录 14/360 全 PASS，六个 model×method cell 均有成功。MuseTalk/Ditto 的 native 生成和评分已实际走通，当前尚无完整 Dev 科学结果。

候选增强头的七源码、配置与三 provenance 文件已按原始字节打成中文说明的离线 CLI ZIP02；这是可执行候选交付包，状态仍 NOT_SELECTED，不能把打包完成称为跨模型效应已成立。Fit 诊断支持谱形贡献及正确门控对固定原时钟 D0 的额外改善；global RMS 固定不等于逐帧能量守恒，STFT phase 保留也不保证外部 F0 提取数值完全不变。

旧失败 run 的 compact indexer04 错读了 selected_track_crop 的新 schema；indexer05 仅补四个已有 render 字段到规定证据角色的映射，要求集合未放宽。实际 index05 已生成：6,430,409 B、1,966 unique blobs/80,023,309 B、完整 360 分母，其中 120 PASS/240 FAILED。索引超过桥接单块限制，Fit 已准备两块 byte-exact splitter，待实际返回 part SHA 后下载重组及执行已审阅 copier03。compact 不包含全部 JPEG/AVI/成片，不可据此删除 SHM。

## Working State
Durable：branch codex/cross-tfg-audio-head-20261003，HEAD 7746b01f34b5850f11e322a0abc6934095f95f64 为本地未推送提交，无当前分支 PR。BM Task107/Experiment106 已记录实际 CPU 与 GPU 启动状态；冻结候选与实验协议不变。

Machine-local/fragile：服务器 root@connect.westd.seetacloud.com:18625，GPU GPU-d09622d0-3535-e662-bb45-3df2b049fcea；根 /root/tts-research-system-20261001。密码/API 仅私有 runtime，不入文件或 BM。PTY57588/bridge07:12249 可用；新 owned PID344830/start_ticks329924069。输出 namespace /dev/shm/cross_tfg_audio_head_20261003/samehost_dev360_latebind_attempt01/controls360_repair_attempt10，physical controls leaf 追加 /controls360，candidate 追加 /candidate180。launcher07 receipt.output_root 是 physical leaf；旧 launcher06 同名字段是 namespace，消费者必须按实际版本区分，禁止修改已消费收据。最新 SHM 可用 25,284,501,504 B；后续启动需重新实测。

关键指针：CPU 八下载在 .codex/scratch/dev360_phase2_repair_attempt10/cpu_bind_plan_preflight_attempt02/downloads/；ROOT QC root_qc_actual_binding_plan09_attempt05.json、Signal root_actual_cpu09_independent_qc_attempt01.json。实际 launch receipt 在 .codex/scratch/dev360_detached_status_attempt08/receipt_readback_attempt01/，5,899 B/raw cd5e1fa2c37d615bb3427947fa1e9a2948af56dced57c5d3d1b6e10b404a02b5。status snapshot02 request/response 在同级 requests_attempt02/，response 8,530 B/raw 6789a3d589fae396ecef039c521c0ec31ed59ceab967d210ea19dc002079c47a。已消费请求不得重放。

旧 compact 索引远端 job_control/dev360_compact_snapshot_attempt02/output/source_index_attempt02.json，whole SHA 2817fc317971a7404b8014a1b5fb56d6d9ad0f281196cfbb460ad3c7a98fe675、self 17bca9e8ab0f517c610dee04651f9ee0ade3d7fc50bed0a046fbc7894b6f6b0a；本地实际响应 requests_attempt02/04_prepare_source_index_attempt02.json.response.json。splitter 新包在 dev360_compact_snapshot_packet_attempt05/index_readback_attempt01/requests_attempt01/request_index_attempt01.json，尚未 dispatch。新 run 的 compact consumer06 仅源码，未来 terminal pins 保持未绑定。

## Changed Files
- 实际修复入口为 scripts/experiments/cross_tfg_audio_head_20261003/run_cross_frontend_dev_attempt09.py；proof 在 openspec/changes/develop-cross-tfg-audio-head-20261003/reproducibility/。前一 FV checkpoint 两处路径描述有误，本 checkpoint 与实际 Git 路径为准；FV 不可变、不编辑。
- CONTEXT.md 当前 93 行，已修改为实际 CPU09 两 QC 与新 GPU 已启动；HANDOFF.md ignored 首节待更新。其余五个历史 dirty 文件保留。
- .codex/scratch/root_qc_dev360_attempt10_launch_and_status02.json 实际验证 launch 与首五行 metadata，非全媒体 QC。
- .codex/scratch/v3_step120_portable_method_bundle_attempt02.zip 为 44,326 B/raw 32145ea20d18602d2fda2b66ca6fcc7256c215ad9fd70ca23da5a91534a33700，13 members；ROOT 仍需独立 ZIP bytes/source closure 核验及 durable placement。

## Verification
实际完成：CPU12 envelopes、八 raw/self hash、22 输入、ordered360 与旧计划字节相同、63 dependency entries/61 unique files、24 source closure、原 PCM metadata；ROOT/Signal 独立 PASS。launcher07 四请求、实际 receipt/owned PID/argv、status08 两次只读响应通过；snapshot02 的 14 行当前全 PASS、无 stderr。indexer05 五实际请求 PASS、SOURCE_INDEX_PREPARED。Signal 已逐 ZIP 13 members reread size/SHA、安全路径与原文件校验；ROOT 只完成说明/manifest/receipt 全读，独立 ZIP 检查是下一工程动作。未运行通用 pytest；旧 NPZ 实际 bytes 未另下载独立复算；新完整360、180 candidate、ASR/content、visual、Confirm 尚未完成。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Verified historical pushed commit](https://github.com/dogLeft233/tts-exp/commit/c4a199890bd529928ec34b9be6da86a10a737c97)
- Current HEAD `7746b01f34b5850f11e322a0abc6934095f95f64` is local/unpublished; fresh current-branch PR query returned no match.

## Observations
- [result] 新 controls360 attempt10 已实际运行，14/360 全 PASS，三个模型 native entry 与评分均有成功；不是完整科学验收。
- [decision] 固定候选、保持 Dev/Confirm 分离；完整原 F32 PCM/clock 用于换轨，generator PCM 另存并做内容校验，transform 每行只调用一次。
- [decision] 旧失败完整分母、源码、收据和媒体保留；compact 仅规定小证据归档，不声称全媒体恢复，不删除 SHM。
- [blocker] 当前完整 Dev 数据尚在产生；没有需要用户批准的额外阻塞。
- [next_step] 继续实际 controls360，完成后执行冻结 candidate180、完整原音换轨、generator PCM 内容与视觉/GUARD20及完整分母统计；并行完成旧索引分块读回/持久复制和离线包独立核验。

## Relations
- continues [[Codex checkpoint - 2026-10-08T07-09-36Z - Actual failed terminal and repaired CPU packet ready]]
- relates_to [[开发跨 TFG 的自然音频增强头]]
- relates_to [[跨 TFG 自然音频增强头的训练与原音换轨验证]]