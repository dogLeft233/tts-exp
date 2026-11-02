---
title: Codex checkpoint - 2026-10-29T17-12-00Z - FlashHead ready and nine scorer migration
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-29-t17-12-00-z-flash-head-ready-and-nine-scorer-migration
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-29T17:12:00Z'
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

# Codex checkpoint - 2026-10-29T17-12-00Z - FlashHead ready and nine scorer migration

## Summary
FlashHead Pro 已在192个人目录生成完整测试视频，英文40人和中文20源共160条既有音频已逐文件SHA验证迁入；九评分器中的四项已完成真实前向，其余正在验收。

## Story
用户先要求按更新后的server-192 skill部署SoulX-FlashHead Pro并测试，继而明确目的是用新TFG复用以前的九评分器实验，要求同时部署评分器与数据、遵守服务器规定，并允许记录可复用工作流。已读取飞书1048手册，检查公共checkpoint，以共享conda创建个人环境。所有代码、权重、缓存、输出位于 /home/data2/wangjiajun；没有改共享base、系统驱动、其他用户进程。后台nohup、单UUID GPU和CPU四线程执行。

FlashHead在独立env中使用torch2.7.1/cu128、SDPA、关闭compile，先完成2.24秒测试，再生成p002自然音频10.44秒/262帧完整视频。评分版保留视频流并用原始WAV重封装ALAC，167092个PCM样点逐一相同。九项按旧实验定义迁移，原始源码和历史路径凭据冻结，新增部署适配器单独保存。数据从本地原始归档恢复；中文20条自然音频经原准备流程重建后全文件SHA全部相同，60条TTS直接匹配。没有调用新TTS或替换缺失样本。

最有用的工程边界是隔离Python安装：最初.pth复用site-packages导致pip卸载FlashHead NumPy，已恢复1.26.4并用完整FlashHead视频验证；之后改为conda clone独立基础环境，再创建system-site-packages venv。CuPy/torchcodec通过进程级system libffi和CUDA12.1 NVRTC解决兼容，不改系统链接。系统FFmpeg4不能mux ALAC，使用私人FFmpeg7。MTD作者Dataset不接受262帧但仅261完整音频帧的输入，适配器显式采用261帧共同前缀，保留原片和排除尾帧凭据，不做偏移校准。

## Working State
远端根 /home/data2/wangjiajun/av-evaluation。权威数据凭据 deployment_input_manifest.json、data_validation.json（160行、271文件全PASS）。scorers/source_manifest.json验证476源码blob，peavs_weight_manifest.json验证6权重Git blob。各下载完成必须以verified.json和完整SHA为准，稀疏文件stat不是完成证据。

机器本地未提交脚本在 .codex/scratch/server192_eval_20261029/（下载器、数据打包/恢复、score_retrieval.py、score_classifier.py、score_avs_u.py）。私有SSH运行器 C:/Users/dogle/.codex/tmp/server192_run.py 通过本机DPAPI凭据登录并固定主机指纹；凭证不复制到checkpoint。全局skill新增 references/personal-deployment.md 且quick_validate通过。原仓库大量dirty/untracked文件未清理；本任务没有修改既有科学脚本。

环境 envs/eval-base是FlashHead独立clone；syncreward、synchformer、cogenav、peavs各自venv。错误初版eval-base已改名eval-base_incomplete_pth，禁止使用。当前A6000 UUID GPU-de7ff5ff-3510-739f-917c-cb98892de759，启动前重查占用。每次单GPU阶段；torch显存60%、CPU4。不要并发重复下载相同目标。

脆弱后台状态（需重查）：Synchformer下载PID760293约816/1079块；VocaLiST ZIP PID727067约496/842块，proxy-vocalist.log。SparseSync653MB、AVHu1.928GB、SyncReward6.636GB、MTD、PEAVS/VGGish均已完成校验。SparseSync cfg可能尚未下载。AVS_u PID760295已退出，缺hubert_pretraining路径；刚在本地及远端score_avs_u.py增加官方avhubert内部目录到sys.path，尚未重跑。PEAVS smoke_peavs.sh已就绪，尚未运行。CoGenAV、SyncNet、SyncReward、MTD已PASS。不要将尚未运行的五项写成成功。

## Verification
FlashHead完整视频和ALAC PCM逐点相等；全部160条音频以及冻结上下文文件SHA验证通过。CoGenAV严格load、Eq1 .5179972053；SyncNet官方pipeline全FP32 Sync-C7.914、D7.354、offset1；SyncReward BF16 encode1 reward1.6484375，峰值7.15GB；MTD严格load、K5 .67578125，±400ms真实平移控制K5 .015625/.0234375，内部偏移相对追踪±10帧，原31候选矩阵独立CPU归约一致。不是新总体实验结论，CoGenAV extraction也不等于整体qualification。

CuPy RawKernel和torch同进程GPU运行通过。fairseq0.12.2官方tag ZIP构建、hydra1.3.2导入通过；PyPI sdist缺clib，不可复用失败包。Wav2Lip S3FD和SyncNet S3FD参数键不同，分别下载正确官方权重且记录哈希。源码/模型严格加载，不静默跳权重。旧V100/torch2.5与当前A6000/torch2.7没有证明数值逐位一致；本次是部署兼容验收。git branch/SHA/status已读，GitHub确认仓库且当前head无PR。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)，当前本地SHA `2b71639c3b555293375d248350999377168d515d`，未验证该SHA远程可解析。
- [服务器手册](https://gcnadw6o1ien.feishu.cn/wiki/Vq5gwxt51i2DVdkDhobcP4iqnkh?fromScene=spaceOverview)
- 旧英语实验 runs/ears40_w2l_nine_scorers_20261029_attempt01；中文 runs/cloud_av_preference20_20261027_attempt05。
- 远端 smoke/{mtd,cogenav_p002,syncnet,avs_u}、smoke/syncreward_p002.jsonl、logs/ 是实际证据；新原片 smoke/flashhead_p002_N_0/。

## Observations
- [result] FlashHead完整视频和160条规范音频已就绪，4/9评分器完成真实前向与原始结果复核。
- [decision] 继续使用旧九项定义；AVS_u明确是论文口径复现，Synch/Sparse保留21类原始logits，PEAVS保留官方batch64口径及尾批说明。
- [decision] 环境使用独立conda clone加venv，禁止.pth跨环境pip写入；兼容修复仅在个人目录与进程级环境变量进行。
- [blocker] 其余5项尚未完成前向；Synch/Vocal权重未下载完成，AVS_u刚修复导入路径待重试；旧192.168.50.2 SSH在banner前关闭，已用本地原始归档绕开数据依赖。
- [next_step] 重跑AVS_u路径修复，随后串行验收PEAVS、VocaLiST和Synch/Sparse；补齐通用启动入口、验证总表和部署文档后交付。

## Relations
- continues [[Codex checkpoint - 2026-10-29T14-57-00Z - FlashHead Pro private deployment installing]]
