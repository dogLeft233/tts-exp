---
title: Codex checkpoint - 2026-10-30T02-05-11Z - Chinese20 FlashHead nine scorer experiment
  complete
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-30-t02-05-11-z-chinese20-flash-head-nine-scorer-experiment-complete
status: completed
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-30T02:05:11Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1880e-73d5-7863-a38b-20bcdf7d5ebc
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-30T02-05-11Z - Chinese20 FlashHead nine scorer experiment complete

## Summary
中文20人80条FlashHead Pro视频与九评分器完整实验已交付，原始归约、2234内容文件SHA和25组来源统计本地复算全部通过，本人GPU任务已退出。

## Story
用户最初要求按192服务器手册在个人目录部署SoulX-FlashHead Pro、评分器和既有数据；随后明确仅中文20人，授权完整自然/TTS视频生成、九评分器配对比较，又追加要求音量归一化到可听水平。本轮先重读server-192 skill及飞书1048手册，检查账号、空间和GPU，再冻结中文20N+60既有云端TTS协议，统一RMS−24dBFS，另存派生音频且不覆盖规范原件。生成与评分绑定相同PCM，41种唯一音频真实生成，完全相同输入/图片/seed42的重复项保留alias，80逻辑媒体全部保留。

[[76-flashhead-chinese20-nine-scorers]]是权威最终报告；[[FlashHead Pro 中文20人可听音量九评分器配对实验]]已CONCLUDED，[[server192-nine-scorers]]和SYNC-C-SCORES已更新。主要同句参考12人九项均未通过Holm9；全20人Sync-C自然6.143、TTS6.355，差+.212、CI[−.399,+.873]，10/20提高。仅全20人PEAVS偏自然（差−.452971，Holm9 p=.04131）；batch20重放与主64+16相同到6位小数。评价器证据不作感知结论。

## Working State
持久远端 /home/data2/wangjiajun/av-evaluation/jobs/flashhead_chinese20_rms24_20261030_attempt01，已完成；不重launch。80视频全部512x512/25fps、固定3.png/Pro/stream/seed42，生成峰值6.146GiB；规范PCM逐点比对、无削波。两个中心5秒分类器33/80媒体、7人完整配对；其他七项80/80、20人配对。SyncReward沿用官方内部短片片段补齐，另列原生完整4.75秒8人子集，不再使用最初过严的4.75秒loader门槛。SyncNet保留旧中文PCM16 MKV、min_track25和FP32距离矩阵，MTD/Vocal共同完整帧，所有兼容修改留在个人目录。

本地交付 results/soulx_flashhead_pro/chinese20_rms24_20261030_attempt01/：report.md、summary.json、per_video_scores.csv、paired_scores.csv（154来源配对行）、video_index.json、paired_differences.png/.pdf及evidence/。标准播放视频在results/soulx_flashhead_pro/{natural_raw,tts_raw}/<source>/rms24_20261030_<id>.mp4，评分无损ALAC及驱动WAV在evidence/videos和audio。

evidence.tar.gz为309272893字节/SHA256 `7cf9916be529e8f0f4099bd346a4cae168bb590bd254af9266cfee213ee7a700`，2235成员包括manifest；local_readback.json核对2234内容文件PASS。归档是完成时快照，末次packaging输出另有delivery_receipt.json。机器本地SSH工具 C:/Users/dogle/.codex/tmp/server192_run.py用DPAPI，不复制凭证。

## Changed Files
新增 scripts/experiments/flashhead_chinese20_20261030/（11个Python文件及README），含prepare/generate/controller/batch_adapters/analyze/audit/package、PEAVS敏感性与本地deliver/verify/plot。新增tests/test_flashhead_chinese20_20261030.py。CONTEXT.md更新报告76入口且仍100行；旧部署README补全完成入口及短片更正。全局server-192参考文件补齐SyncNet CLI、官方内部padding、输入重复复用和归一化证据，quick_validate通过。

这些新代码未提交，现有大量dirty/untracked改动保持原状，没有PR。源代码和环境快照已在证据包，不能把未提交状态误作无可复现来源。旧HANDOFF有其他任务快照，本轮优先读报告76和本checkpoint。

## Verification
4个针对归一化安全/时序、人级三draw配对和Holm的测试通过；所有11个Python文件编译通过，PNG图已实际查看。80规范原件/派生/视频SHA、RMS和PCM核验；七种原始矩阵/embedding/logit归约PASS，SyncNet FP64对照最大差2.4485e−6。PEAVS/SyncReward为实际模型输出和身份完整性验证，不宣称独立重实现。

本地独立重算25非空指标分层的来源均值、10000bootstrap、100000signflip、Holm9和胜负计数，最大差0。AVS_u、MTD、Vocal的82真实±400ms控制均下降；Synch40/40，Sparse39/40；CoGen82/82峰移动误差在1帧内且峰超额中位正，六项控制通过。SyncNet、PEAVS和SyncReward未在本队列重做控制，PEAVS/SyncReward旧中文控制失败边界保留。

最后ps/nvidia-smi确认本人无推理或后处理进程，显存清单无本人PID，其他用户任务保留。Git分支/SHA/status已读；GitHub只读确认仓库及当前head PR_COUNT0。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 当前本地SHA `2b71639c3b555293375d248350999377168d515d`，未验证远程commit可解析，不构造commit链接。
- [最终报告76](../../../docs/experiments/76-flashhead-chinese20-nine-scorers.md)
- [1048手册](https://gcnadw6o1ien.feishu.cn/wiki/Vq5gwxt51i2DVdkDhobcP4iqnkh?fromScene=spaceOverview)

## Observations
- [result] 用户授权的中文完整实验已完成，80视频、九评分器、原始证据与本地统计复核已交付，资源已释放。
- [decision] 以20来源为统计单位，三T先平均；短片支持和参考12/8分层明确，不插补、不合成总分。
- [decision] 官方SyncReward内部片段补齐和外部延长输入分开；PEAVS保留固定批次并用同特征做旧batch20敏感性。
- [next_step] 需要审阅时从报告76、逐视频/逐人CSV及不可变证据包进入；无需重跑或部署。

## Relations
- continues [[Codex checkpoint - 2026-10-30T00-55-00Z - Chinese20 full experiment and audible normalization]]
- documents [[76-flashhead-chinese20-nine-scorers]]
- relates_to [[server192-nine-scorers]]
