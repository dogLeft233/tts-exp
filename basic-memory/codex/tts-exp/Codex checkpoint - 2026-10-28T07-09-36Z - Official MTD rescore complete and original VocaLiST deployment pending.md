---
title: Codex checkpoint - 2026-10-28T07-09-36Z - Official MTD rescore complete and
  original VocaLiST deployment pending
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-28-t07-09-36-z-official-mtd-rescore-complete-and-original-voca-li-st-deployment-pending
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-28T07:09:36Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16cf2-be54-7ba0-bcfb-be8433abd951
model: gpt-6.1-sol
trigger: compact
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-28T07-09-36Z - Official MTD rescore complete and original VocaLiST deployment pending

## Summary
官方MTD重评已完成；找到原版VocaLiST官方Release后，下载因SSL中断，正在恢复，再按用户新授权重评同组80媒体。

## Story
研究目标是辨别TTS驱动视频优势与SyncNet评价偏好。部署MTDVocaLiST和Synchformer后，自定义MTD零时移均值控制失败；审计证明官方输入和前向一致，所有20自然样本原生峰值约−3帧，40个已知时移变化均正确追踪，问题主要是端点定义。原作者K5–K15重评已完成，不能继续引用旧自定义0/60作为官方结论。用户随后询问完整VocaLiST；我们纠正漏查GitHub Releases的错误，确认作者2022发布独立002 ZIP含LRS2 5f原版权重。最新授权明确：部署验证通过后重评此前云TTS与自然驱动视频，统计TTS胜率、平均差、Sync-C冲突率。

## Working State
正式MTD结果在 runs/mtd_official_cloud_rescore_20261028_attempt01，80媒体、60配对、20来源。K5 T胜21/60=35%，平均T−N=−0.005656421126（−0.566个百分点），冲突33/60=55%（SyncT/MTDN30，反向3），来源bootstrap区间跨0。K7..15见报告70。Synchformer仅18可用配对/6来源，T胜15/18，P0平均差+.0129367；52媒体不足官方5秒，不能外推全组。MTD官方低绝对K5也不构成感知真值。
Ubuntu部署根 /home/wjj/av-sync-eval-20261028，源码 vocalist commit c265b59bcf7cd8559a265aaf454a9333ec24cb0f；原模型80106561参数CPU初始化已通过。官方002 ZIP总882753219字节，唯一成员VocaLiST_Weights/vocalist_5f_lrs2.pth，解压961635703字节。权重文件尚未完成，真实权重加载/GPU验证未通过。2026-10-28T07:09Z实查原依赖链FAILED（resumed download process ended before completion），日志SSL EOF。稀疏.parallel实际约214MiB，名义842MiB不是完成。新48worker续传PID63300，用cogenav独立env、512KiB块、复用已写extent，日志vocalist_download_attempt03.log；活进程属脆弱机器状态，需复查。旧chain脚本可重跑但其自动resume只有一次，未启动新chain。本轮发现smoke AST Dataset使用os却未注入env，需先修复。
原版必须cv2 BGR/255，MTD是torchvision RGB0–255，不可共用归一化。可复用正式MTD的全512 JPEG人脸裁剪，并用原版Dataset重新处理及全部新前向。用户已授权完整80评分，不必再确认。生产训练/交叉音轨维持暂停，不杀外来GPU任务、不更改旧env、不做大文件额外SHA、不泄露密钥。

## Changed Files
未提交的前序文件：scripts/experiments/{cloud_mtd_synchformer_20261028,mtd_official_audit_20261028,mtd_official_cloud_rescore_20261028,vocalist_official_smoke_20261028}.py及同名tests/test_*.py。原版批量脚本尚未编写。项目有大量继承dirty/untracked，均保留，未提交或推送。

## Verification
前序官方MTD Dataset图像/PCM/Mel 20/20一致；官方两类模型真实窗口最大logit差0；40/40时移变化零帧误差。正式80矩阵480官方端点及360配对指标独立CPU复核PASS，对应2测试通过。原版仅CPU实例化和CUDA可用性已验证，下载和真实GPU烟测未完成；不可记部署成功。本轮直接读取链状态、下载日志与已分配大小确认失败，启动PID63300恢复。Git只读确认branch/SHA；GitHub REST canonical repo成立，当前分支open PR=[]，SHA未远端核验不链接。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [Official VocaLiST release](https://github.com/vskadandale/vocalist/releases/tag/interspeech2022)
- basic-memory/docs/experiments/68-cloud-mtd-synchformer-native-preference.md
- basic-memory/docs/experiments/69-mtd-official-parity-and-shift-control-audit.md
- basic-memory/docs/experiments/70-mtd-official-native-cloud-preference.md
- runs/cloud_mtd_synchformer_20261028_attempt01/{inputs,sync_pairs}.json：冻结80媒体与60原Sync-C配对
- /home/wjj/av-sync-eval-20261028/{vocalist_deploy_chain.json,vocalist_deploy_chain.log,weights/vocalist/official_release_zip_index.json}

## Observations
- [result] 官方MTD K5胜率35%、平均−0.566pp、冲突55%，代替旧自定义rawmean0/60表述。
- [decision] 原版使用官方独立002 ZIP，不需要下载仅含歌唱权重的001包，不与蒸馏版混名。
- [decision] 官方K5主端点，K7–K15全报，无最佳时移校正；结果不是视觉真值投票。
- [blocker] 原版权重下载因SSL EOF退出，尚无真实权重加载/GPU部署通过证据。
- [next_step] 修复并完成原版续传及真实GPU验证，随后用原版Dataset/calc_pdist/eval_model对冻结80媒体新前向，独立复核并报告60配对统计。

## Relations
- continues [[Codex checkpoint - 2026-10-28T02-19-00Z - AVS pilot complete and synchronization scorer mechanisms compared]]
- relates_to [[VocaLiST原版官方Release权重补齐 2026-10-28]]
