---
title: 训练 May 电平交叉先导
type: task
permalink: tts-exp/tasks/训练-may-电平交叉先导
status: done
priority: high
started: October 17, 2026
current_step: 4
completed: October 17, 2026
---

# 训练 May 电平交叉先导

用户已授权实际训练May三档、检查学习/验证曲线，再做三档测试并统一原始音频评分。本任务运行生成器训练；此前“仅准备数据”的限制已被本次授权更新。数据准备结果保持，不恢复其他暂停研究。

## 冻结设计

本轮先做一个共同初始化seed42；此前数据spec列出的seed2026留作后续重复，不在此次先导中悄然扩充。每档60000次FP32 AdamW更新、4096 rays/step、max_steps16，官方L1及默认正则、官方first-stage LR .5衰减，AVE冻结。它是控制预算的第一阶段先导，不是官方65536 rays/step加后续100000步唇部微调的完整复现。

固定原版commit43079fb3，已有数据train4150帧/166秒、val550帧/22秒、test1100帧/44秒。官方数据素材一致、新划分独立，May为已暴露英语单身份；不足以跨身份推断，数据量充分性待学习曲线检验。三个模型从同一完整参数/训练相机mask初始状态加载，实际逐step图像顺序与采样ray索引一致。Python/NumPy/Torch占用更新、ray采样和perturb使用独立的确定性种子域，验证不改变下一步随机序列；CUDA原子梯度不宣称跨设备逐位可复现。

优化器与loss调用官方实现，省去未使用的Trainer LPIPS初始化；不启用patch/lip微调。FP32避免AMP溢出跳步造成实际更新预算不等。EMA .95每1000步更新，主验证/最终生成用固定最终步EMA权重，原始权重一并保留。每2000步验证三个输入电平上的同一32个等距帧，报告L1/MSE/PSNR/口部框误差，最终补全全部550帧。主checkpoint固定60000步，不按测试或收益后选。

收敛诊断预先冻结：比较40000–48000五点与52000–60000五点验证MSE中位数，相对变化绝对值≤2%作平台筛查，恶化>2%单列；结合实际曲线/口部误差解释，不把该启发式当统计收敛证明或感知质量证据，不自动延长某一臂预算。

## Requirements

### Requirement: 系统 SHALL 在训练前封存预算、初始状态与输入身份
#### Scenario: 三档共同起点
- **WHEN** 开始三档训练
- **THEN** 机器protocol、共同初始state hash、60000步frame order、验证帧、源WAV/cache/代码SHA已封存，三档strict加载同一state，无May预训练生成器warm start。

### Requirement: 系统 SHALL 完成等预算训练并保存可核验学习曲线
#### Scenario: 固定三档更新
- **WHEN** 执行0/−6/+6dB三个模型
- **THEN** 每档成功执行60000次FP32更新，全部实际ray-index SHA逐step相同；每100步loss、每2000步共同验证/可恢复checkpoint完整；非有限/控制失败真实阻断，不补选成功结果。

### Requirement: 系统 SHALL 用固定最终模型完成九格独立测试
#### Scenario: 训练电平乘测试电平
- **WHEN** 三档训练完成
- **THEN** 每个模型对−6/0/+6dB测试输入渲染全部1100测试帧，固定四个10秒clip做SyncNet；所有格使用同一原始自然PCM与原时钟、共享GT-owned ROI、FFV1/PCM及FP32 SyncNet，主guard20/次full，保存C/Dmin/D0/B/offset与原矩阵；train0/test0首clip重复控制通过。

### Requirement: 系统 SHALL 独立验收并限定科学结论
#### Scenario: 交付先导
- **WHEN** 曲线、视频和分数齐备
- **THEN** 独立核对初始state/预算/ray schedule/最终step、原PCM/帧时钟、评分原数组，交付学习曲线、九格结果和平台/未收敛判断；仅单身份单seed工程/条件响应，不声称数据量充分、普遍最佳音量或感知改善。

### Requirement: 系统 SHALL 保留持久备份并隔离其他实验
#### Scenario: 空间受限
- **WHEN** Ubuntu root剩余不足1GB
- **THEN** 新训练与编译使用独立SHM工作目录，checkpoint和最终产物及时同步F盘并可用Windows百度盘备份；SHM不称持久存储，不修改共享环境/删除历史记录/重启其他工作。

## Steps
1. [x] 恢复隔离CUDA/GCC12并保存协议/共同初始化。
2. [x] 三档等预算训练、定期验证和备份。
3. [x] 固定最终模型九格测试/原音轨评分。
4. [x] 独立QA、图表和限定结论报告。

## 执行入口

Supervisor历史身份见Ubuntu ROOT/supervisor_receipt.json（PID14831），三档及自动后处理已完成并退出；不可重复启动或编辑已封存训练代码。后处理及补充像素/观看admission均已执行通过；事后真人诊断也得−3帧offset，原主结果未改。ROOT位于/dev/shm/synctalk-may-training-pilot-20261017，运行副本易失，持久产物以F盘备份receipt为准。共同初始化tensor SHA15e74e38abec740a7e9790917658da13a0d3a0c0bf426faeb0c119b252eabbc1。

## Observations
- [close_date] October 17, 2026
- [result] 三档各60000次FP32更新、共同initial/frame/ray完整hash一致；九条验证MSE平台筛查通过。9900测试帧/36评分矩阵/273420距离独立QA通过，F盘643文件6049784599 bytes已全量SHA核对。云归档4301034008 bytes经Windows挂载上传，Ubuntu独立全量SHA读回通过；[[50-synctalk-may-gain-training-pilot]]报告/图表/九格预览已交付。
- [status] done
- [priority] high
- [description] 执行已授权的May训练电平先导，FP32共同初始化与60000步预算，统一原PCM评价。
- [decision] 首轮单seed42；不以结果后选训练时长、checkpoint或测试电平。
- [limitation] 166秒训练数据充分性尚未验证；May单身份英语工程先导不能推断中文/跨身份效果。
- [location] scripts/experiments/synctalk_may_training_pilot_20261017；Ubuntu /dev/shm/synctalk-may-training-pilot-20261017；F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017/remote 已持续SHA核对同步；backup_worker.json记录Windows PID60232。

## Relations
- project [[50-synctalk-may-gain-training-pilot]]
- depends_on [[准备 SyncTalk 训练电平先导数据]]
- follows [[48-synctalk-may-gain-inference]]