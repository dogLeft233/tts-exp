## Context

动机见 [proposal.md](proposal.md)。依据为 BM `tts-exp/experiments/lrs3-真实视频局部时间敏感性诊断` 与 `runs/lrs3_real_video_local_timing_20260905_v2/final.json`：3 条超旧上界，未运行新评分。父 spec 与[本修订](specs/lrs3-real-video-tail-contract/spec.md)共同规定本次执行。

## Goals / Non-Goals

**Goals:** 在现有包中修改输入契约、补足审计证据与独立验证，让下游 agent 能继续一次固定诊断。

**Non-Goals:** 通用媒体归一化框架、音频裁剪、定位 codec padding 根因、放宽裁脸或科学判据。

## Decisions

1. 选择有界长尾容忍。80 ms 是看到输入审计后、评分前明确选定的工程界限，覆盖当前最大 56 ms；并非证明该尾部无语音或无影响。短尾维持旧 40 ms 限制。裁音频会破坏 PCM 恒等，直接把绝对阈值放宽到两帧还会扩大未遇到的短尾情形，故不采用。
2. `config.py` 冻结修订和 samples 上下界；`protocol.py` 逐记录收集审计，`media.prepare` 在裁脸前落盘。使用 ffprobe 逐帧 best-effort PTS/time_base，按有理数比较；核对已有历史生成/提取 manifest 或命令记录以确定配对起点。来源不明就报告，不用音频相关性估 offset。不要重建历史媒体。
3. `input_audit.json` 是新增的唯一独立产物，存实际 PTS 或绑定保存的原始 probe 数据，并保存可供离线验证的历史来源引用。区分源文件 SHA 与解码 PCM SHA，不能用 WAV 容器 hash 代替 PCM 字节 hash。`protocol.json` / `final.json` 引用审计；prepare 失败也保留审计。其余目录与 CLI 沿用父实现。
4. 复用现有局部内部行，不因长尾增加可评分行；逐行核验 visual/audio 支持范围。官方全局距离矩阵继续使用其原始全部行，边缘影响在报告中作为限制说明。无需改第三方模型。
5. `validate.py` 独立重算审计规则，可复用底层读文件/probe 功能，不复用 producer 的资格判定；新 BLOCKED 也需验证审计证据。保持旧 schema 可读，拒绝跨修订恢复。修改代码后使用全新 run-id。

## Risks / Trade-offs

- 长尾可能包含真实语音 → 完整保留，报告原因未知；局部内部行排除边缘，全局分数仅描述。
- 旧实现只核验了视频 stream start，来源或逐帧 PTS 可能不齐 → 用真实证据补审计；无法确认时仍 BLOCKED。
- 解开尾差后还可能遇到 crop、窗口或基线问题 → 按父 spec 终态交付，不能将输入通过写为科学成功。

## Migration Plan

在当前实验包内实现并完成针对性测试，以新 run 执行一次原命令 `python -m scripts.experiments.lrs3_real_video_local_timing.runner --run-id <new-id> --stage all`；随后运行 `python -m scripts.experiments.lrs3_real_video_local_timing.validate --run-root runs/lrs3_real_video_local_timing_<new-id>`。检查父 spec 的完整交付要求，不能因已有代码而默认后续阶段正确。

在实施阶段读取 BM Startup Router/实验指令，全文读取并更新原实验笔记，保留旧 BLOCKED 事实及新结果指针。科学终态无需必须阳性；若工程阻塞则交付证据，任务只勾选已完成项。无需迁移旧 run。
