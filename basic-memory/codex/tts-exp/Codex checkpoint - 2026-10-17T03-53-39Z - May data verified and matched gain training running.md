---
title: Codex checkpoint - 2026-10-17T03-53-39Z - May data verified and matched gain
  training running
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t03-53-39-z-may-data-verified-and-matched-gain-training-running
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 03:53:39+00:00
username: laptop-e133gsle\dogle
hostname: LAPTOP-E133GSLE
capture: deliberate
agent: codex
session_id: 01a1445a-5cf8-7243-9764-adae839515e6
trigger: compact
model: gpt-6.1-sol
tags:
- codex
- checkpoint
---

# Codex checkpoint - 2026-10-17T03-53-39Z - May data verified and matched gain training running

## Summary
May真实视频数据已完成独立划分与全量核验；用户已授权训练，三档共同初始化的FP32先导已启动，0 dB正在训练且checkpoint持续备份F盘。

## Story
最初工作检验纯音量增益能否改变SyncTalk编码、生成和原音轨SyncNet；报告47–49完成前端及现成May推理。用户随后要求保存任务spec并准备训练数据，追问是否官方素材、自然视频和数据量足够。现已明确：原官方May素材字节一致、自然讲话监督，但采用重新隔离train/val/test，166秒训练的充分性尚未证明。最新请求明确授权−6/0/+6 dB等初始化、等预算训练，观察学习曲线及验证，再做三档测试，始终用原音频评分；旧“不训练”限制已被此次请求取代。

采用固定官方43079fb3的AVE+May，从同一完整随机初始化state训练三个生成器。首轮seed42、每档60000次成功FP32 AdamW更新、4096 rays/step，官方train_step原样调用，不启用lip阶段。帧顺序冻结，实际ray索引逐step对照，随机域分离防止验证扰动后续采样；最终固定60000 EMA，不按测试选checkpoint。每2000步对同32验证帧的三档音频评估，最终补全550帧。平台筛查预先冻结为40–48k对52–60k MSE中位数相对变化±2%，仅启发式，不能证明数据充分或感知同步。

## Working State
持久数据在F:/Datasets/tts-exp/synctalk_gain_training_20261017；42518官方源文件、4150/550/1100监督帧、九组DOUBLE纯增益WAV和冻结AVE缓存全部通过跨端QA。原包May.zip和90.5MB派生包经Windows B盘上传，再从Ubuntu百度盘挂载独立全量SHA读回；未删除历史数据。权威Task为[[准备 SyncTalk 训练电平先导数据]]，status done，数据量充分性UNVERIFIED。交付manifest v2 SHA为900cf9108a12fba087f94771dce1a411c2530e1ca97a34d9d34803d1a9c0b6b6。

进行中Task为[[训练 May 电平交叉先导]]。Ubuntu wjj@192.168.50.2执行根为/dev/shm/synctalk-may-training-pilot-20261017，数据根/dev/shm/synctalk-gain-training-data-20261017/dataset。SHM全部易失，不作为备份。隔离CUDA12.1/GCC12与已编译前后向扩展已恢复在/dev/shm/synctalk-may-gain-inference-20261016；工作provider修正独立test、完整val和已核验AVE缓存，官方网络/loss未改。根盘约580MB，后续禁止大写入根盘。

Supervisor PID14831（精确身份见supervisor_receipt.json）已经运行，依次训练0/−6/+6。最新核验0 dB 18800步、约24步/秒、GPU峰值2.95GB；尚无收敛/最终分数。完整初始tensor SHA15e74e38abec740a7e9790917658da13a0d3a0c0bf426faeb0c119b252eabbc1。运行中的protocol.json/common.py/train.py/supervisor.py已封存，不得编辑或重复启动。完成三档后supervisor会自动执行同目录postprocess.py（若不存在则停在等待后处理）。

Windows隐藏backup_watch.py PID60232持续每60秒SHA核对同步ROOT到F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017/remote；当前56文件、无失败。F盘约39GB空间。读backup_status/index/receipts判断，不把SHM路径或运行进程当持久证据。渲染/评分/独立QA脚本已写在本机，尚未上传，postprocess和绘图尚待完成；先完善并封存评估入口，再上传，不能改训练代码。

当前共享工作树9个已跟踪修改，大量未跟踪文件（本轮git status全量82163条），不得全量clean/stage/commit。当前SHA仅本地可证，GitHub解析422，分支无PR。

## Changed Files
- scripts/experiments/synctalk_gain_training_data_20261017/：准备、source-clock核验、provider资格、跨端QA、云封存和交付。
- scripts/experiments/synctalk_may_training_pilot_20261017/：冻结训练协议、共同初始化、训练/验证/supervisor、备份、待执行render_test/score_test/validate_results。
- tests/experiments/test_synctalk_gain_training_data_20261017.py 和 test_synctalk_may_training_pilot_20261017.py。
- basic-memory/Tasks/准备 SyncTalk 训练电平先导数据.md、训练 May 电平交叉先导.md及研究地图；训练Task和Task Board还需反映已启动。

## Verification
数据阶段：42518源文件、17400图像解码、5800landmark/pose、9WAV和9AVE全部通过；源视频时钟9帧bestlag0，三个12秒原WAV/视频音频bestlag0。跨平台61文件比对通过，21JSON仅LF/CRLF差异，分别封存hash；无掩盖原件。数据准备5项unittest通过。CUDA扩展前向/反向已实际编译并用于训练。训练合同5项unittest拒绝预算、初始化、ray不一致、缺臂及改评分PCM，全部通过；最终训练/9格渲染评分和独立QA尚未执行。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)
- 本地未验证推送SHA：`2b71639c3b555293375d248350999377168d515d`

## Observations
- [result] 官方May自然监督与三档增益训练数据已经全量验收并持久封存；0 dB先导训练正在运行。
- [decision] 首轮单身份单seed42，各60000次FP32更新，同初始化/图像/rays，固定最终EMA；不作完整官方两阶段复现或跨身份推断。
- [decision] 九格测试统一原自然PCM、原时钟和GT共享ROI，主guard20及次full，同时解释C/Dmin/D0/B，不将高分称为感知改善。
- [blocker] 尚未获得三档最终学习曲线/测试结果；数据量充分性仍UNVERIFIED。
- [next_step] 完成并封存postprocess/绘图代码上传到正在运行supervisor的ROOT，使三档训练后自动执行九格测试、原音频评分和独立QA；随后监测完成并交付限定结论。

## Relations
- continues [[Codex checkpoint - 2026-10-17T02-34-56Z - SyncTalk gain inference verified and normalized comparison scoped]]
- relates_to [[准备 SyncTalk 训练电平先导数据]]
- relates_to [[训练 May 电平交叉先导]]
- relates_to [[研究主线与证据地图]]