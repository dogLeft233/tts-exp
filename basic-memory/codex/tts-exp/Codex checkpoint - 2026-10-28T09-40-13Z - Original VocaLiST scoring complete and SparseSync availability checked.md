---
title: Codex checkpoint - 2026-10-28T09-40-13Z - Original VocaLiST scoring complete
  and SparseSync availability checked
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-28-t09-40-13-z-original-voca-li-st-scoring-complete-and-sparse-sync-availability-checked
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-28T09:40:13Z'
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

# Codex checkpoint - 2026-10-28T09-40-13Z - Original VocaLiST scoring complete and SparseSync availability checked

## Summary
原版VocaLiST下载、部署、80媒体官方评分和独立审计全部完成，最新问题转为SparseSync用途及公开可用性。

## Story
研究目标是辨别TTS驱动口型优势与SyncNet评价偏好。用户授权原版VocaLiST验证通过后重评此前云TTS/自然视频。前次下载阻塞已解决：官方2022 Release独立002 ZIP以32线程持久连接、过期URL刷新下载完毕，正常CRC解压、strict load及真实GPU推理通过。原版Dataset/calc_pdist/eval_model原样AST函数体对80媒体重新前向，冻结20来源、60云TTS配对，沿用原RMS，无新归一化、无最佳时移校正。报告71、部署指南和任务完成状态已保存读回。用户现在只询问SparseSync，未要求部署。

## Working State
权威报告 basic-memory/docs/experiments/71-vocalist-official-native-cloud-preference.md；部署 basic-memory/docs/deployment/vocalist.md；正式run runs/vocalist_official_cloud_rescore_20261028_attempt01 在Windows和Ubuntu均有小型结果、80矩阵及源码快照。Ubuntu部署 /home/wjj/av-sync-eval-20261028，venv /home/wjj/.venvs/vocalist-20261028，官方源c265b59bcf7cd8559a265aaf454a9333ec24cb0f，FP32原版80106561参数。原版BGR/255，MTD RGB0–255，不能复用张量。

原版K5 T胜18/60=30%，N胜42，平均T−N=−0.937494pp；N均值4.000898%、T3.063403%；Sync-C冲突36/60=60%（SyncT/原版N33，反向3）。20来源bootstrap CI [−1.915168,−0.020771]pp；strict12区间跨0。K7/9/11/13/15胜率25/20/15/10/5%，平均差−0.937/−1.463/−2.139/−2.902/−3.608pp，冲突65/70/70/70/75%。

6个物理±400ms控制相对时移追踪全部通过，但80媒体内部曲线原生峰值全−3帧，零点±1帧端点存在地板。N72.45%、T82.13%正确窗口位于首尾20窗口；N12/20、T42/60内部无正确窗口。不能据此证明自然口型更好或SyncNet偏置。scores.json global_offset_frames受−1000边界掩码影响；实际诊断看 independent_audit.json diagnostic_guard20_offset_counts。MTD与原版师生相关，不作为独立真值投票。

原版任务、下载、评分已完成，不要重启。外来GPU未终止，生产训练/交叉音轨/PEAVS保持暂停；不做网盘大规模额外SHA读回，不暴露历史API密钥。

SparseSync已查官方README、项目页、example.py：BMVC2022，预测音画全局时移，learnable selectors压缩稀疏时空线索。训练、评估、示例、配置、LRS3/VGGSound/AudioSet权重公开；示例自动下载。21类版本−2至+2秒、200ms步长。官方LRS3 21类权重HEAD返回200，653037857字节；未下载/部署。与Synchformer同作者路线，官方推荐后者；粗分辨率限制细粒度TFG口型排序价值。

## Changed Files
未提交且未跟踪：scripts/experiments/vocalist_official_smoke_20261028.py（补os注入）、vocalist_official_cloud_rescore_20261028.py、vocalist_official_results_audit_20261028.py及对应 tests/test_vocalist_official_{smoke,cloud_rescore,results_audit}_20261028.py。大量继承dirty/untracked保留，无commit/push。Ubuntu下载链脚本是机器本地产物。

## Verification
本次已完成真实GPU strict加载/推理和集成测试PASS，6/6物理时移控制零帧误差；80矩阵480端点360配对端点独立CPU复算PASS，本地和Ubuntu审计PASS，对应测试及py_compile通过。正常ZIP CRC解压，没有额外大文件SHA。本次直接读取本地status COMPLETE_OFFICIAL_SCORES_AND_INDEPENDENT_AUDIT、audit PASS。Git只读重查分支/根/SHA，GitHub REST确认canonical仓库和当前分支open PR=[]；SHA未远端核验，仅作本地SHA。SparseSync仅文档及权重HEAD可达性核查，无部署验证。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [VocaLiST official release](https://github.com/vskadandale/vocalist/releases/tag/interspeech2022)
- [SparseSync code and weights](https://github.com/v-iashin/SparseSync)
- [SparseSync project](https://v-iashin.github.io/SparseSync.html)
- runs/vocalist_official_cloud_rescore_20261028_attempt01/{summary,paired_scores,independent_audit,boundary_floor_diagnostic,known_shift_controls}.json

## Observations
- [result] 原版官方K5：T胜30%、平均−0.937pp、Sync-C冲突60%，全部80媒体完成。
- [decision] 官方K5主端点和K7–15全报，无最佳时移修正；地板及边界贡献必须披露。
- [decision] 评分器冲突不等于感知真值，原版与MTD不视作独立裁判。
- [next_step] 回答SparseSync用途、代码权重开放性与200ms局限；原版评分已无待办，新实验等用户指令。

## Relations
- continues [[Codex checkpoint - 2026-10-28T07-09-36Z - Official MTD rescore complete and original VocaLiST deployment pending]]
- relates_to [[71-vocalist-official-native-cloud-preference]]
- relates_to [[VocaLiST原版部署]]
