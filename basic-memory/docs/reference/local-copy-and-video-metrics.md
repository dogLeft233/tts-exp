---
title: 本地项目入口、历史副本恢复与视频评估代码
type: note
status: active
tags:
- storage
- recovery
- video-metrics
permalink: tts-exp/docs/reference/local-copy-and-video-metrics
---

# 本地项目入口、历史副本恢复与视频评估代码

> 路径说明：`${USER_HOME}`代指原记录对应的个人主目录，不表示接收者已有该文件。定位归档时继续核对资产映射与原运行凭据；示例中的相对仓库路径从仓库根解释。

## 当前入口

✅ 当前工作区为 `./`。代码、配置与可继续使用的数据以此目录为入口。

`./` 是历史独立工作副本。其完整快照已经保存到
`${USER_HOME}/tts-exp-archives/local-copy-retirement-20261023-attempt03/old-checkout.tar`。
恢复依据同目录的 `backup_verification.json`、`members.jsonl`、`source_inventory.json`。
普通文件 96,925 个、链接 507 个；普通文件内容合计 56,113,407,704 字节。
逐成员内容 SHA、链接目标及整包 SHA 已完成独立复读核对。

此备份位于本机 C: 逻辑卷，**不是网盘备份**；没有确认 C:/E: 是否属于独立物理设备。
外部链接目标的内容不计入覆盖范围；历史 `/tmp/r2_audio` 只保留链接本身。

## 视频评估代码

✅ 已从历史副本合并五个脚本及原有测试，保持原计算口径：

| 文件 | 用途 |
|---|---|
| `scripts/41_video_quality_metrics.py` | 解码视频，计算画质、感知、时序与分布特征 |
| `scripts/42_build_video_metric_manifests.py` | A1 LeapTalk、A2 多模型、B May 增益/归一化、C 后归一化尺度的视频清单 |
| `scripts/43_aggregate_video_metrics.py` | 逐视频结果聚合、配对统计、分布距离与 Sync-C 关联 |
| `scripts/44_extract_sync_scores.py` | 提取各实验族的 Sync-C 字典 |
| `scripts/45_render_video_metric_report.py` | 从已聚合结果生成报告表格 |

可选依赖列在 `scripts/requirements-video-quality.txt`。完整指标推理还需要相应
PyTorch、torchvision、LPIPS、pyiqa 环境与权重；此次只完成合并和 CPU 单元/解码验证，未重新跑完整 GPU 指标。

视频清单中的相对视频路径默认相对于当前脚本所在工作区解析，不依赖启动时的当前目录。
`41` 支持 `--repo-root`；`42` 与 `44` 支持 `--repo-root` 和 `--data-root`。
B/C 的默认大数据入口仍是 `F:/Datasets/tts-exp`。

在当前仓库根目录生成 A2 清单与提取评分：

```powershell
python scripts/42_build_video_metric_manifests.py --families a2 --out runs/video_quality/manifests
python scripts/44_extract_sync_scores.py --family a2 --out runs/video_quality/sync_a2.json
python scripts/41_video_quality_metrics.py --manifest runs/video_quality/manifests/manifest_a2.json --out runs/video_quality/a2
python scripts/43_aggregate_video_metrics.py --run-dir runs/video_quality/a2 --manifest runs/video_quality/manifests/manifest_a2.json --out runs/video_quality/a2/report --sync-json runs/video_quality/sync_a2.json
```

原脚本的 InceptionV3 FID 与 r3d_18 FVD proxy 口径保持不变；不能直接与其它实现的绝对值比较。
历史报告保存在恢复包的 `docs/experiments/30-video-quality-metric-suite.md`，此次没有改写其科学结论。

## 旧路径处理

`scripts/project_paths.py` 集中识别历史 Windows 和 WSL 本地根，将其转换到调用方的当前工作区。
转换在检查旧文件是否存在之前执行；不会再因副本文件存在而优先读取副本。
另外三个 VFHQ 下载脚本根据 `__file__` 确定仓库根。
Ubuntu 执行机的 `/home/wjj/tts-audio/tts-exp` 是另一台机器上的正式路径，未改写。
冻结 manifest、历史报告、来源证明中的旧路径保留原样。

## 恢复和核验

恢复包目录中的 `verify_archive.py` 可再次逐成员及整包核验，运行会重新读取完整大包。
需要找回冲突版本时，按 `members.jsonl` 或 `migration_actions.jsonl` 的相对路径从 TAR 提取到新建空目录，
不要直接覆盖当前工作区。TAR 含 WSL 符号链接，建议在 WSL/Linux 中恢复；原始重解析数据也保存在成员清单中。
此快照是数据与来源恢复点，不将旧 Git worktree 的失效绝对路径视为已修复。

## 整合结果与待清理项

✅ 91,435 项通过同卷移动并入当前工作区，普通文件逻辑大小合计 55,220,635,384 字节；
506 个内部 WSL 链接已重定向到当前工作区。146 项与当前文件内容相同；
568 项因同路径内容或类型冲突未覆盖当前版本，原版完整保存在恢复包中。
其余旧代码、文档、Git/工作树元数据等保存在完整恢复包内。

验证：新代码与路径测试 31 项通过；迁移后真实 MFA cohort 集成测试 4 项通过、
1 项因既有生成 cohort 不可用按原规则跳过。没有重新运行完整 GPU 画质评估。

⚠ **旧目录尚未物理删除**。自动审批拒绝递归删除，只返回 `blocked by policy`；
可撤销的目录重命名也因 Windows 文件占用失败，旧 `.git` 元数据的重命名返回 access denied。
因此保留原路径，并用 `RETIRED.md` 和替换后的 `AGENTS.md` 标明退役。
原始 AGENTS 内容已在完整恢复包内；退役提示是归档之后添加的元数据。

⚠ 自动审批同样阻止清除本次失败的中间 TAR：

| 待清理文件 | 字节数 | 状态 |
|---|---:|---|
| `${USER_HOME}/tts-exp-archives/local-copy-retirement-20261023-attempt01/old-checkout.tar` | 17,703,449,600 | 中断包，不能用于恢复 |
| `${USER_HOME}/tts-exp-archives/local-copy-retirement-20261023-attempt02/old-checkout.tar` | 14,479,984,640 | 中断包，不能用于恢复 |

两个中断包合计约 32.2 GB；**只有 attempt03 是成功且已完整核验的恢复包，必须保留**。
旧目录删除及这两个中断包清除仍待人工处理或解除相关阻塞。
最终状态以 `retirement_result.json` 为准，不以清理前 `retirement_predelete.json` 的 READY 状态代替实际删除结果。

工程证据位于 `.codex/scratch/local-copy-consolidation-20261023/`；
`code_merge_receipt.json` 记录原始代码 SHA 与合并来源，`code_before/` 保存路径修复前的当前代码。
`code_validation.json` 记录测试范围。历史网盘音频清单与现存源 SHA 对应 1,460 项、366,710,380 字节，
此次未重新下载那些云端 payload，删除依据是本次完整本地恢复包。
