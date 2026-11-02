---
title: 017-measured-pilot-scenario-and-live-capacity-gates
type: decision
permalink: tts-exp/docs/adr/017-measured-pilot-scenario-and-live-capacity-gates
repository: dogLeft233/tts-exp
decided: '2026-10-08T02:51:25.9499092+00:00'
project: tts-exp
status: open
---

# 017-measured-pilot-scenario-and-live-capacity-gates

## Choice and context

新Dev360自然/RMS控制与Dev180冻结step120候选的输出使用同一fresh SHM命名空间下的controls360与candidate180分离叶目录，保留全部计划行和实际失败账。容量元数据采用MEASURED_PILOT_WITH_EXPLICIT_SCENARIO_AND_LIVE_GATES、media_bytes_estimated=true：实测pilot、外推场景与独立运行余量分别记录。启动先检查整个540规划；每阶段按未完成的阶段场景检查，每行执行前再检查输出/SHM余量、cgroup与GPU可用量。不得在后续阶段重新要求已经消耗的控制输出预算。

用户授权持续开发、重跑实验和处理空间。root已下载原57746 B当前资源/完整六pilot树回执，SHA164bff9e05094428af4de0901e4d38e1139754cbfcfa7e693a80251be5479bf7。实际overlay余7021514752 B、SHM26444845056 B，cgroup可用57921232896 B、GPU空闲32230MiB。最大allocated行按各模型180复制为4986961920 B；正式Stage-A新增WAV/裁剪等879246792 B，enriched base5866208712 B。加2×显式场景余量及单份5465536 B原PCM为17604091672 B；独立5GiB余量5368709120 B，共22972800792 B，probe时SHM还余3472044264 B。所有未来视频/JPEG与临时峰值均不能据此称为硬上界。

## Rationale and alternatives

完整保留树含JPEG和AVI，避免只按MP4/NPZ估算。新的容量receipt区分精确源长度算术、AVI到MKV的大小代理、目录allocated实测和全体540场景。旧MEASURED_OR_EXPLICITLY_BOUNDED与media_bytes_estimated=false字段不能真实表示这份证据，因此窄改容量语义并加入当前资源检查，不伪造已测未来字节或最大临时峰值。

普通overlay不足承载场景；放宽余量或先删除未完成原始证据不采用。等待完整慢网盘归档会增加停机，且当前挂载未核验。已授权的SHM空间允许运行；代价是重启易失，必须把完成的完整成果另行归档并逐成员恢复核验后才能交付，保留原source、参数、原音/clock、评分和失败账，不把SHM或小receipt称为持久副本。

## Consequences and affected work

工程代理修订runtime binder、阶段guard和Stage-A行前资源检查；实际CPU绑定/运行尚待，source-only审查不等于执行完成。空间失败按工程失败保留完整分母，不降低评分支持或挑选成功子集。原生执行策略、20clip/10speaker×3seed×3model、360+180分母、原音完整换轨、ADR016评分ROI和GUARD语义、候选参数以及Confirm封存均保持。

完成阶段后保留SHM原树供raw/media/visual独立审查，并复制必要证据到持久分区；最终完整产物使用满足实际容量、全成员SHA/类型/路径与恢复核验的本地/服务器/网盘归档。任何原树退役继续遵守既有可恢复归档门槛。

## Evidence

- .codex/scratch/dev360_resource_footprint_probe_attempt01/downloads/resource_footprint_receipt_attempt01.json：实际完整6PASS树和当前宿主只读回执，57746B，SHA164bff9e05094428af4de0901e4d38e1139754cbfcfa7e693a80251be5479bf7。
- .codex/scratch/dev360_allocated_capacity_analysis_attempt01/capacity_analysis_attempt01.json：19072B，SHA0e8de8f39ff72ada8e61ebab6950604020c6674a5bb9c61e0ede221fe0745315，13source pins及全部算术root独立PASS，explicit scenario/proxy/NOT_MEASURED peak。
- scripts/experiments/cross_tfg_audio_head_20261003/：新binder/guard/core正在修复一致接口，尚未生成正式Dev分数。

## Observations

- [decision] 用fresh SHM叶目录承载360+180，capacity字段明确实测pilot加外推场景，whole/start-stage/row live gates分别检查，保留5GiB运行余量和全部失败分母。
- [rationale] overlay不足而SHM实测可容纳17.604GB场景及5GiB余量；未来压缩媒体/临时峰值没有硬上界证据。
- [alternative] 普通磁盘全输出、降低余量、未归档证据删除或伪称media_bytes_estimated=false均不采用。
- [consequence] SHM只暂存，完整可恢复持久归档是最终交付依赖；源码修订和运行通过须由实际receipt另行证明。

## Relations

- affects [[开发跨 TFG 的自然音频增强头]]
- affects [[跨 TFG 自然音频增强头的训练与原音换轨验证]]
- relates_to [[015-same-gpu-deterministic-controls]]
- relates_to [[研究服务器缓存清理与可恢复归档]]