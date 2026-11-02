---
title: Codex checkpoint - 2026-10-29T17-21-00Z - FlashHead nine scorer deployment
  verified
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-29-t17-21-00-z-flash-head-nine-scorer-deployment-verified
status: completed
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-29T17:21:00Z'
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

# Codex checkpoint - 2026-10-29T17-21-00Z - FlashHead nine scorer deployment verified

## Summary
192个人目录上的FlashHead Pro、九评分器和160条既有中英文规范音频已完成部署与真实单片验收，本人后台任务全部退出。

## Story
用户要求按更新server-192 skill遵守1048手册，在个人目录部署新TFG、跑测试，然后迁入以前的多评分器和数据，为后续新TFG的同协议比较做准备。本次部署目标完成：先读手册，使用共享conda创建个人环境，私有数据区保管所有代码、模型、cache和输出；UUID单GPU、CPU4线程、nohup/互斥运行，没有修改共享base、系统驱动或他人任务。正式完整队列实验尚未运行。

最新权威说明为 [[server192-nine-scorers]] 与仓库 results/soulx_flashhead_pro/server192_nine_scorers_20261030/README.md，命令和科学口径边界均在该处。原FlashHead部署记录 [[soulx-flashhead]] 已增加九评分器入口及changelog，历史内容保留。前checkpoint的下载/导入阻塞已经解决，不能继续按4/9状态工作。

## Working State
远端TFG /home/data2/wangjiajun/SoulX-FlashHead-Pro；评分/数据 /home/data2/wangjiajun/av-evaluation，入口 run_scorer.sh、prepare_cogenav_run.py。当前部署汇总 deployment_validation.json、数据 deployment_input_manifest.json/data_validation.json；源码scorers/source_manifest.json验证476blob，PEAVS权重另有blob清单，下载asset旁verified.json记录来源、大小和哈希。不是所有作者发布了whole-file SHA，已区分自行计算与已知期望值。

完整证据归档 deployment_evidence.tar.gz 已下载到上述本地results目录，SHA256 78d6ab354ed8ccd0f34a55bc46c1882814aa3f74b4b6b2be3d92fc95626e2298；解包evidence/含新片、原始矩阵/embeddings/logits、代码、环境freeze和日志，本地再验9项PASS与所有脚本哈希通过。代码开发副本 .codex/scratch/server192_eval_20261029/ 未提交，属于机器本地状态；不要清理其它大量已有dirty/untracked文件。没有PR，GitHub只读head查询为空，仓库身份已确认。

基础eval-base是FlashHead独立conda clone，冲突包以cogenav/peavs/syncreward/synchformer venv隔离。错误旧eval-base_incomplete_pth保留但禁止使用；.pth事故后FlashHead NumPy1.26.4已恢复，并用完整新片生成证明可用。SSH使用C:/Users/dogle/.codex/tmp/server192_run.py及本机DPAPI凭证，固定主机指纹；不复制凭证。

## Changed Files
新增部署脚本 run_scorer.sh（UUID/线程/显存/锁）、prepare_cogenav_run.py、三类通用score适配器和collect_validation.py；均在scratch、远端和交付evidence中有副本。全局 C:/Users/dogle/.codex/skills/server-192/references/personal-deployment.md 增加环境隔离、断点哈希、私人FFmpeg7、libffi/NVRTC兼容、权重cache和整数窗口经验，SKILL.md链接该参考，quick_validate再次PASS。

## Verification
新FlashHead生成512x512/25fps/262帧，驱动10.44325秒；评分版视频流未改，ALAC音轨167092样点与原PCM16完全相等。全部160条规范音频/271文件SHA PASS，英语40人80条、中文20源80条；中文恢复自然20条和已有TTS60条全文件SHA匹配原清单，未新合成替换。

九项全已实际运行：SyncNet官方pipeline C7.914/D7.354/offset1；CoGenAV Eq1 .5179972053；PEAVS 2.9043863（官方batch64配置但本验收仅1条）；SyncReward BF16 encode1 1.6484375；AVS_u .0809920499；MTD K5 .67578125；VocaLiST K5 .6328125；Synchformer P0 .0002663291；SparseSync P0 .0000040338。这些是单片部署值，不是跨模型排名或TTS偏好结论。

MTD/Vocal原31候选矩阵独立CPU归约一致，真实全音频±400ms偏移使内部偏移1→11/−9帧；分类器argmax .2→.6/−.2秒，logits保留。AVS_u论文定义复现的真实时移分数也下降。CoGenAV仍标EXTRACTED_NOT_YET_QUALIFIED，不冒充总体qualification。MTD/Vocal按261共同完整帧支持，原视频/音频保留、排除尾帧明确记录；分类器中心起点68帧/43520样点，先平移完整音频再裁窗。

Synch/Sparse/Vocal最后下载均完成；旧AVHu argv==1触发notebook导入，适配器进入正常package分支后strictload/前向PASS；PEAVS链接源码指定VGGishcache后实际全pipeline PASS。CuPy+torch同进程、私人FFmpeg7 ALAC和进程级libffi/CUDA12.1NVRTC均已验证。旧torch2.5/V100与新torch2.7/A6000未证明逐位一致，matmul TF32 false/cudnn TF32 true记录在总receipt。

末次ps显示wangjiajun无推理或下载进程，显存任务列表无本人PID；他人工作未终止。bash语法、Python compile、skill验证与本地归档SHA全部通过；无科学生产代码改动，未跑无关项目pytest。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- 当前本地SHA `2b71639c3b555293375d248350999377168d515d`，未证明远程commit可解析。
- 本地交付 results/soulx_flashhead_pro/server192_nine_scorers_20261030/。
- 旧协议 runs/ears40_w2l_nine_scorers_20261029_attempt01 与 runs/cloud_av_preference20_20261027_attempt05。

## Observations
- [result] FlashHead、九评分器、160条规范音频全部就绪；部署验收及凭据回读完成，GPU任务已释放。
- [decision] 旧科学定义/资产/清单冻结，新机器路径和兼容适配单独绑定；只宣称部署兼容，完整实验须保留原batch/窗口/精度及来源统计单位。
- [next_step] 若继续新TFG比较，先固定生成种子/条件、资格和失败规则，再在已就绪的中英文队列批量生成评分；不要重下载或重复部署。

## Relations
- continues [[Codex checkpoint - 2026-10-29T17-12-00Z - FlashHead ready and nine scorer migration]]
- documents [[server192-nine-scorers]]
- relates_to [[soulx-flashhead]]
