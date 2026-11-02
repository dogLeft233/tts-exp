---
title: 052-integer-frame-native-window-preselection
type: decision
permalink: tts-exp/docs/adr/052-integer-frame-native-window-preselection
status: open
decided: '2026-10-29T03:20:16Z'
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
run_id: ears40_w2l_nine_scorers_20261029_attempt01
---

# 052 — 同步分类器按整数帧预选冻结的原生窗口

## Decision

本轮 EARS40 实验的 Synchformer 与 SparseSync 以冻结的整数 `center_start_5s=i` 作为唯一评价窗口坐标。在 25fps / 16kHz 条件下，选择 `video[i:i+125]` 与 `audio[i*640:(i+125)*640]`，再运行作者原样的官方 5 秒变换，局部 `v_start_i_sec=0`。物理时移先施加到完整 PCM，再选择窗口。该选择落实已冻结的中央 5 秒支持，不改变生成媒体、模型权重、评价端点或样本集合。

## Context and rationale

2026-10-29，用户授权 EARS40 自然／云 TTS → Wav2Lip → 九评分器比较。作者 `EqualifyFromRight` 默认先截取前 10 秒，导致最长自然音频 p046 的冻结中央 5 秒（整数帧 171:296，音频起点 109440）越界。作者 `sec2frames=int(sec*fps)` 的浮点往返还使四条视频起点早一帧，而音频起点未改变，形成确定的 40ms A/V 索引失配。

全 80 条静态审计定位 p006_N_0、p017_T_0、p060_T_0、p081_T_0；不是按分数或胜负挑选。故障前 Synchformer 已完成 24 条，其中后两条尚未评分。48 组原生／物理时移输入的 CPU 张量检查显示：22 条媒体的 44 组输入逐值及 SHA 一致；另外两条的四组输入仅视频存在预期偏差，音频全部一致。最长 p046 的三种时移均通过精确索引检查。首条新 buffer 与保存旧 logits 对照最大差 0。

因此保留全部旧 24 条 JSON 为 legacy 原件，复用经证实等价的 22 条，只补算 p006_N 的原生及两个控制、p017_T 的原生，共四次必要前向；其余未完成条目及尚未启动的 SparseSync 使用整数窗口。原 protocol、原源码快照及失败回执不回写。

## Alternatives considered

- 在作者前 10 秒内重新居中：会改变已冻结的物理支持，未采用。
- 直接扩大作者 Equalify 的 cap：需要更改作者配置；精确预选已经能保留完整官方后续变换，未采用。
- 全局把作者 int 改为 round 或用浮点微调：影响范围更大，也不能为两路采样点提供同一整数坐标，未采用。
- 全组重新评分：22 条输入已有逐值等价证据，无需重复前向；保留原分数并明确缓存来源。

## Consequences and affected work

新适配代码是显式的项目输入封装，不宣称原长媒体调用与作者默认 10 秒截取流程完全相同。官方网络、变换、5 秒输入长度及零时移概率端点保持原样；对可比较样本保留输入与模型输出一致性证据。Synchformer 依作者配置使用 autocast，结果转 float32 后求 softmax；SparseSync 维持既定 FP32，具体环境由实际 receipt 记录。

只读核查旧中文 28 条可测媒体，发现同类一条自然基线 BAC009S0031W0121_N_0（视频名义起点 29 帧，旧实际 28 帧，音频仍 18560 样本），关联该来源的三组历史配对。旧 Synchformer／SparseSync 统计应保留 legacy 支持说明；本轮没有重跑或篡改旧实验，历史修正另行跟踪。

代码：`scripts/experiments/ears40_extra_scorers_20261029.py` 及对应测试。执行证据：`runs/ears40_w2l_nine_scorers_20261029_attempt01/execution_repairs/01_classifier_integer_center.json`；张量 SHA、精确差值、首次 logits 对照、静态索引表及旧 JSON 在 `synchformer/repairs/01_integer_center_before_equalify10/`。原适配器 SHA f3873e287f9abb21082c7ab257ca73422192189138efefd7497a88bae0722d91，新 SHA c5f1f99d4ebcac8e068e6464cd3ffc9c89adecbc2908da4689247501b6179868。

| 修改日期 | 修改 | 记录者 |
|---|---|---|
| October 29, 2026 | 记录当前授权实验中的工程修复、等价证明和历史适用边界 | Codex |

## Observations

- [decision] 25fps/16kHz 的冻结中央 5 秒评价以整数帧与每帧 640 个样本绑定，不通过秒数浮点往返恢复起点。
- [evidence] 旧 24 条的 48 组输入中 44 组逐值及 SHA 相同；四组已知视频偏差保留原输出后补算；首条 logits 差为 0。
- [limitation] 当前修复验证不等于人类感知同步效度；历史中文一条自然基线的同类风险尚未重评分。
- [branch] codex/cross-tfg-audio-head-20261003；本地基线 commit 2b71639c3b555293375d248350999377168d515d，新实验代码为未提交工作。

## Relations

- relates_to [[003-immutable-execution-repair-provenance]]
- qualifies [[68-cloud-mtd-synchformer-native-preference]]
- qualifies [[72-sparsesync-official-native-cloud-preference]]
