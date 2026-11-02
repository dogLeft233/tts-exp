---
title: Codex checkpoint - 2026-10-25T02-18-03Z - CoGenAV comparison and fixed PCM
  EQ concluded
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-25-t02-18-03-z-co-gen-av-comparison-and-fixed-pcm-eq-concluded
status: open
project: tts-exp
cwd: E:/Documents/tts-exp
repo_root: E:/Documents/tts-exp
repository: dogLeft233/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-25T02:18:03Z'
username: dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a16cf2-be54-7ba0-bcfb-be8433abd951
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-25T02-18-03Z - CoGenAV comparison and fixed PCM EQ concluded

## Summary
CoGenAV与Sync-C的20来源比较及固定视频谱形EQ干预均已完成并独立核验；尚不能据此认定任一指标更符合人类同步感知。

## Story
用户起初怀疑Sync-C对TTS有偏好，要求实际检查CoGenAV官方代码和权重、与Sync-C比较，再扩充数据。人评不可行后，用户授权直接执行固定视频的声学干预。官方CoGenAV Base成功严格加载；论文Eq1由本项目实现，最佳lag搜索属于本项目诊断而非官方同步benchmark。20来源138可测视频、2个40.88秒输入超过冻结30秒设置明确不可测，176比较中Eq1反向105、最佳lag反向82；来源等权最佳lag同向率53.9%。276/276真实波形错位控制通过，只支持粗错位定位。

后续20来源N/L06/L17共60实际官方评分PCM，冻结视频特征，留一来源学习±3dB长期谱差方向，加频率反转对照，完成300条件。自然N朝预定TTS谱方向ΔC=-0.003，99%描述区间[-0.043,+0.036]，方向特异对照也未通过正向门。TTS反向EQ次要ΔC约+0.075/+0.084。所有最佳lag不变。谱差方向提高相对模板距离差，但自然N只有8/20更接近TTS模板绝对距离，因此阴性不能排除其他TTS偏好。

最新用户问论文的音视频对齐能否用于TTS与自然音频对齐。本次重新核对论文2.1/2.2：训练使用本已同步的音视频；下采样使特征同帧率，同一时刻余弦相似度学习跨模态表示，并不自动输出不同语速语音之间的时间路径。可提出audio-only特征+受限单调DTW作为探索性新适配，先在已知时间变形验证，并与独立文本/音素锚点比较；尚未实现或启动。必须区分估计对应路径与实际重定时波形，不能用评分最大化定义真对齐。

## Working State
已完成实验权威记录见Relations。CoGenAV扩展完整产物在runs/cogenav_sync_full20_20261024_attempt01；EQ首选报告runs/syncnet_fixed_pcm_eq_20261025_attempt01/parent_report.md。EQ E盘仅小元数据，完整本机备份C:/Users/dogle/.codex/tts-exp-artifacts/syncnet-fixed-pcm-eq-20261025；远端/home/wjj/tts-audio/tts-exp/runs/syncnet_fixed_pcm_eq_20261025_attempt01。C副本不是百度网盘。E剩余约1.21GB，后续保持>1GiB。上次完成时GPU进程退出、322MiB桌面占用；本次未重查GPU，不把旧状态当实时证据。

工作树继承大量修改和未跟踪文件，本次git status再次确认dirty，未提交推送。不要恢复其他chat的数据补齐或WIMHF工作。当前无待完成执行任务。

## Changed Files
本会话实验代码：scripts/experiments/cogenav_sync_audit_20261024.py、cogenav_sync_compare_20261024.py、cogenav_sync_full20_20261024.py及对应tests/test_cogenav*；scripts/experiments/syncnet_fixed_pcm_eq_20261025.py及syncnet_fixed_pcm_eq_parent_audit_20261025.py、对应tests/test_syncnet_fixed_pcm_eq*。实验脚本与产物尚依赖本地未提交工作树和备份；本次问答未修改实验代码。

## Verification
前序已实际完成CoGenAV全部曲线float64重放max误差1.1142744e-7，8个契约函数直接执行通过。EQ全部60identity和300时间代理通过；父侧不导入runner独立重建20profile、300PCM、全部距离/标量/bootstrap，profile误差0、距离max3.814697e-6、标量9.536743e-6。3执行器+1父侧契约函数直接断言通过，未运行pytest。主manifest942成员175024708bytes，SHA12c4a44c974fbd7e8d5bd1f2d5f8ba36be200d8d256d842a3f0826e1e04aac74，全部逐SHA核验通过。归档包含当时0字节parent_audit_stdout.json，完整输出另存parent_audit_stdout_final.json，5成员addendum单独封存核验；不重写冻结清单。

本次只重新读取BM结论与原论文，不重跑科学计算。gh不可用；公开GitHub REST核实仓库canonical URL，当前分支PR查询返回空。本地SHA未重新核实可在GitHub解析，保留原始SHA不构造commit链接。用户级basic-memory.json不存在，采用项目.codex/basic-memory.json。

## References
- [Repository](https://github.com/dogLeft233/tts-exp)
- [CoGenAV paper](https://arxiv.org/html/2505.03186v2)
- [Official implementation](https://github.com/HumanMLLM/CoGenAV)

## Observations
- [decision] 不把CoGenAV/Sync-C分数变化直接称作人类感知同步或视觉运动改进；max lag只是整体偏移搜索，无法消除局部语速差。
- [decision] EQ结果保留目标操作有效性局限，不按分数重调方向、剂量或选择来源。
- [result] 本轮有限EQ不支持自然语音沿TTS长期谱差轴稳定提高Sync-C，不能否定广义TTS偏好。
- [next_step] 回答当前对齐可行性问题；若用户授权实施，先验证CoGenAV音频特征的受限单调时间匹配能否找回已知局部变形，不直接改现有主结果。

## Relations
- relates_to [[CoGenAV与SyncNet评分的20来源扩展对照 2026-10-24]]
- relates_to [[固定视频与实际评分PCM的留来源谱形干预 2026-10-25]]
