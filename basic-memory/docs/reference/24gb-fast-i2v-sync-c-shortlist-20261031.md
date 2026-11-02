---
title: 24GB快速I2V模型的Sync-C公开榜单筛选
type: note
permalink: tts-exp/docs/reference/24gb-fast-i2v-sync-c-shortlist-20261031
tags:
- TFG
- I2V
- Sync-C
- model-survey
---

# 24GB快速I2V模型的Sync-C公开榜单筛选

检索日期：2026-10-31。状态：✅公开资料核查完成；🔄候选资格待本项目实测。本次没有下载权重或运行GPU。范围为单图+外部语音生成说话视频；“快速”优先采用实时或接近实时的官方吞吐证据。以下为论文作者的测试结果，不是本项目九评分结果，也不是统一总榜。LSE-C与Sync-C均按原论文名称保留。

## 同表排名与来源

| 一手比较表 | 数据集 | 按同步置信分数排列的相关模型 |
|---|---|---|
| [PC-Talk Table1](https://arxiv.org/html/2503.14295v3#S3.T1)，仅Image行 | HDTF / MEAD-Neutral | PC-Talk 9.370/8.190；Sonic 8.640/8.030；Hallo-v2 7.530/6.280；SadTalker 7.150/6.360 |
| [TalkLikeYou Table1](https://arxiv.org/html/2610.06658v1#S5.T1) | HDTF / CelebV-HQ | TalkLikeYou 8.730/7.970；Sonic 8.620/7.680；SadTalker 7.660/6.360 |
| [DyStream Table1](https://arxiv.org/html/2512.24408v1#S4.T1) | HDTF-100 | Sonic 8.495；DyStream offline 8.136；INFP复现6.894；Hallo3 6.814；SadTalker 6.704 |
| [IMTalker Table2](https://arxiv.org/html/2511.22167v1#S4.T2) | HDTF / CelebV | IMTalker 7.711/7.364；Hallo 7.497/6.843；AniTalker 6.855/6.501；FLOAT 6.854/6.843；Ditto 6.093/5.154 |
| [MoChaBench公开榜](https://github.com/congwei1230/MoChaBench#-mochabench-leaderboard) | 英语单人独白 | MoCha6.333；Hallo3 4.866；SadTalker4.727；AniPortrait1.740 |

不同论文的HDTF子集、参考图、裁脸、音轨及SyncNet执行不同，禁止跨表比较绝对高低。MoCha榜可用于看复杂场景，但本次没有查得MoCha满足单24GB快速推理的证据。

## 候选及硬件证据

1. ✅ **DyStream：优先新增候选**。[官方代码](https://github.com/RobinWitch/DyStream)给出单图单人语音路径、最低10GB显存，以及[公开权重](https://huggingface.co/robinwitch/DyStream/tree/main)。[论文附录C](https://arxiv.org/html/2512.24408v1#A3)报告4090+i9-13900KF、5步58.48FPS、显存2.37G；性能依赖CPU和同步开销。注意该速度段承接在线lookahead配置，不能把离线8.136与58.48FPS称为同一配置实测；论文Table4在线5步为7.371（H200速度29.24FPS），离线与在线应分开冻结。适合本项目离线候选，但离线配置速度仍需实测。
2. 🔄 **PC-Talk：高分优先，24GB条件待确认**。[官方仓库](https://github.com/BQ-Wang0511/PC-Talk)链接[公开权重](https://huggingface.co/doubi-killer/PC-Talk/tree/main)。论文Table4报告30.13FPS，训练采用RTX4090，但没有明确给出测速卡与峰值显存；不能声称24GB实时已验证。纯图路径固定头姿、生成嘴部与表情；不要为凑头动额外引入未冻结参考视频。其preset style、lip-scale与emotion需要预先固定，不能按评分挑参数。
3. ✅ **IMTalker：补九评分器的实用候选**。[官方仓库](https://github.com/bigai-nlco/IMTalker)报告512×512音频驱动42FPS/RTX4090，并已发布[权重](https://huggingface.co/cbsjtu01/IMTalker/tree/main)。没有单列峰值显存，但4090运行证据满足单24GB设备资格。项目已有核心模型历史，不应称首次部署。
4. ✅ **TalkLikeYou：已测高分参照**。[官方实现](https://github.com/BQ-Wang0511/TalkLikeYou)报告3090上30+FPS、低于4GB，稳态FPS不含初始化、注册和mux。项目报告79已完成九评分器，本次无需重复执行。图像默认借Ditto生成头姿，论文实现与公开组合版本须记录。
5. 🔄 **FLOAT：储备**。[论文](https://arxiv.org/html/2412.01064v3)自表HDTF8.222、V100上默认10步41.37FPS，[官方仓库](https://github.com/deepbrainai-research/float)公开权重。V100容量未明确，不能据此保证24GB；另有论文4090 OOM，需先检查chunk与解码配置，本轮不列确定合格。

## 暂不优先

- **SoulX-LiveAct**：高分不等于24GB快速。[官方仓库](https://github.com/Soul-AILab/SoulX-LiveAct)支持4090/5090通过FP8 KV与CPU卸载运行，公布5090单卡6FPS；20FPS为双H100/H200。命令参数--fps24是输出时钟，不是推理吞吐。不同精度/卸载也不能默认保留榜单分数。
- **Sonic**：多表高分且权重公开，但现有速度证据明显慢于上述运动空间模型；若放宽批量时间预算再纳入。不能只凭高Sync-C选它做最快一轮。
- **READ**：[项目页](https://readportrait.github.io/READ/)展示高分，但核查时代码按钮指向占位YOUR_REPO_HERE，不列已核实开放权重候选。
- MuseTalk/LatentSync的榜单Video行属于重配音，不能直接冒充原生单图I2V排名；如用重复静帧适配须明确记录任务差异。
- FlashHead Pro、EchoMimicV3-Flash、TalkLikeYou已有项目九评分器报告76、77、79，本次不自动重跑。

## 推荐执行顺序

DyStream离线配置 → PC-Talk单图资格探测 → IMTalker补齐九评分器；TalkLikeYou复用已测结果。先同输入短片测端到端耗时、稳态FPS、峰值显存和成功率，再决定完整20人矩阵。自然/TTS双方共享配置；保留原PCM、原时钟和现行九评分协议。公开高分不保证中文队列高分，也不能替代感知同步证据。

## Observations

- [recommendation] DyStream具备最清晰的新候选24GB硬件证据，离线高分与在线测速配置需分开。
- [recommendation] PC-Talk值得优先探测，但尚无公开峰值显存资格。
- [limitation] 论文自表只能支持对应测试集内的排名，不能组成统一Sync-C总榜。
- [status] 本次仅资料筛选，无新推理或实验结论。

## Relations

- relates_to [[研究交接总览]]
- relates_to [[79-talklikeyou-chinese20-nine-scorers]]
- relates_to [[76-flashhead-chinese20-nine-scorers]]

- extends [[I2V口型同步模型调研 2026-10-30]]

本页补充前一日调研中的PC-Talk/DyStream与24GB资格证据；旧LiveTalk/StreamEval综合分建议不能当Sync-C优先顺序。TalkLikeYou的项目状态以报告79为准，已完成九评分器，不再是未测新增候选。
