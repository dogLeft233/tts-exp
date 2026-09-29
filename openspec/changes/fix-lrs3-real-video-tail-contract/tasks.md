## 1. 输入契约与审计

- [x] 1.1 阅读父 spec 与本修订，在现有包冻结版本及整数尾差规则；测试 d=-641/-640/0/640/768/896/1280/1281，验证仅长尾上界改变且不存在浮点边界误判。
- [x] 1.2 在裁脸前生成完整 22 条 input_audit.json，核验 hash、历史配对来源、起点、逐帧 PTS 和共同支持；测试多条失败仍完整落盘，以及来源缺失、非连续 PTS、起点超界被阻断。

## 2. 集成与独立验证

- [x] 2.1 将审计和修订 hash 绑定 protocol/final，保留完整 PCM 和既有局部行；测试 1.4 帧长尾经 mux/scorer 提取后 PCM 不变、局部行无越界，以及旧终态保护和跨版本恢复拒绝。
- [x] 2.2 实现 validator 独立重算输入审计，覆盖正常与 BLOCKED；测试篡改 N、PTS、源 hash、缺失记录均验证失败，旧 final 只能按旧 schema 验证。
- [x] 2.3 运行 `python -m pytest -q tests/experiments/lrs3_real_video_local_timing`、对应目录 Ruff 和 `openspec validate fix-lrs3-real-video-tail-contract --strict`，全部通过后执行真实诊断。

## 3. 一次执行与交付

- [x] 3.1 使用全新 run-id 执行原 runner `--stage all` 并运行独立 validator；验收完整输入审计及父协议要求的 66 cells/报告，或有证据的 BLOCKED。保持全部科学参数；阻塞后不调参重试，未执行部分保留未完成。
- [x] 3.2 按 BM Startup Router 和实验指令更新原 `LRS3 真实视频局部时间敏感性诊断` 笔记并读回核对：保留旧阻塞、记录修订/新 run/validator/科学终态或 null；向用户交付结果路径和剩余限制。
