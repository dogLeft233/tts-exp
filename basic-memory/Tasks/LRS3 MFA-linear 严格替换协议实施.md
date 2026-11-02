---
title: LRS3 MFA-linear 严格替换协议实施
type: task
permalink: tts-exp/tasks/lrs3-mfa-linear-严格替换协议实施
status: done
priority: high
current_step: 4
assigned_to: claude
protocol_id: lrs3_mfa_linear_replacement_20260824
tags:
- task
- mfa-linear
- replacement
- lrs3
- protocol
---

# LRS3 MFA-linear 严格替换协议实施

本任务执行独立的 LRS3 MFA-linear replacement protocol，不修改旧 raw-TTS visual audit，也不重跑其 blocked fresh confirmation。研究终点是 strict file-level replacement：冻结 Wav2Lip 用 natural audio 生成 Baseline，用 exact-natural-length MFA-linear candidate 生成 Target，随后两者都替换回同一份 untouched natural PCM；只有 replacement 后 Sync-C 上升且 Sync-D 下降才算科学改进。candidate 与自身音轨的 self-consistency 只作诊断。

任务以 engineering BLOCKED + scientific not_available 收口: 两侧 MFA 对齐在严格 unknown-phone gate 下仍有 OOV 记录, candidate 生成与 GPU 渲染未授权执行。任何后续策略需要单独预注册的新协议版本。

## Steps

1. [x] 建立独立 source/test 目录并完成 protocol、alignment、candidate、strict mux、evaluation、statistics modules
2. [x] 完成 synthetic tests、py_compile 和 code review；不打开 sealed media
3. [x] 生成 immutable Stage00 protocol lock 并完成独立 artifact review；engineering GO
4. [x] Stage01 双侧 MFA 执行完成（MFA2 retry4 与 MFA3 retry4 均因 `spn` OOV gate BLOCKED 在 candidate 之前）
5. [ ] Stage02 单一 frozen Wav2Lip 的 48 renders、四-cell strict mux 和 authoritative SyncNet（未授权）
6. [ ] Stage03 独立重算统计、artifact review、decision 和 memory closeout（因 Stage01 blocker 未授权）

## MFA2 → MFA3 对齐尝试

- MFA 2.2.17 / `english_us_mfa` retry4: 24+24 TextGrids 产出, 13 unique fit records 被 `spn` OOV 阻断; candidate/GPU 未启动
- 历史英文对齐用 MFA 3.4.1 / `english_mfa` — 工具链不等价, 触发 MFA3 复核
- MFA3 运行时必须 `MFA_ROOT_DIR=/home/wjj/Documents/MFA/mfa3_runtime_20260815`（默认全局配置含 SafeLoader 拒绝的 PyYAML PosixPath tag, 不用未 pin 的命令）
- 新协议 `lrs3_mfa_linear_replacement_mfa3_20260825` Stage00 绑定 MFA 3.4.1 + pinned root + `english_mfa`; Stage01 retry1 因 `fstcompile` 不在 PATH 中断（已修复: subprocess PATH 前置 pinned 目录, Stage01 入口校验 MFA executable/root/model hashes）
- MFA3 Stage01 retry4: 24+24 TextGrids 完成, strict parser 保留 16 个 side-specific `spn` failures / 8 unique records; blocked 13 → 8, 未放松 unknown-phone gate
- 剩余 lexical/OOV: `didn`, `acr's`, `damselfish`, `snappers`, `groupers`, `kean`, `maximization`, `beser`, `uk`, `vermiculture` — 不允许当 silence
- LRS3 `normalize_transcript` 只 uppercases + collapse whitespace; `spn` intervals 对应数字/序数、缩写、技术词、`[laughter]`; 版本切换需配套英文 token normalization audit

## Observations

- [status] done
- [protocol] lrs3_mfa_linear_replacement_20260824
- [cohort] 消费旧 visual Stage00 已冻结且从未打开媒体的 24 条 fit-only fresh_confirmation 列表，不重新选择，不访问 internal-dev、validation、test
- [primary_endpoint] benefit_C = SyncC(G_M_E_N) - SyncC(G_N_E_N); benefit_D = SyncD(G_N_E_N) - SyncD(G_M_E_N)
- [engineering_gate] 任一 cohort、provenance、alignment、candidate、render、mux 或 authoritative SyncNet gate 不完整则 BLOCKED，不写成 scientific NO_GO
- [decision] protocol 收口为 engineering BLOCKED、scientific not_available; 不跑 reduced subset, 不改 `spn` 政策
- [next] 任何 dictionary/pronunciation augmentation、contraction/compound rewrite 或 `spn` remap 需要单独预注册的新协议版本
- [constraint] 不写入 key/password；不创建 commit 或 push；GPU 运行前检查占用与 23:00–08:00 禁用窗口
- [legacy_boundary] 旧 raw-TTS retry7 与调查 artifacts immutable；本任务不覆盖其目录
- [reviews] MFA3 Stage00/Stage01 retry4 code/artifact reviews PASS, 见 `_reviews/lrs3_mfa_linear_replacement_mfa3/`

## Relations

- supersedes [[MFA-linear 原样基线实验]] only for this new strict replacement audit
- constrained_by [[Raw TTS strict replacement NO-GO]]
- informed_by [[Full-frame Wav2Lip training audit]]