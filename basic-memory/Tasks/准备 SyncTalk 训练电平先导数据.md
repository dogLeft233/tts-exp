---
title: 准备 SyncTalk 训练电平先导数据
type: task
permalink: tts-exp/tasks/准备-sync-talk-训练电平先导数据
status: done
priority: high
started: October 17, 2026
current_step: 6
completed: October 17, 2026
training_runs: 0
data_amount_sufficiency: UNVERIFIED
tags:
- task
- synctalk
- training-data
---


# 准备 SyncTalk 训练电平先导数据

先交付可核验的May单身份训练数据，冻结后续训练电平×测试电平设计；当前执行范围为数据准备、前端编码、存储和核验，不启动生成器训练。May属于已公开、已查看和已做推理的英语工程先导，不是中文多身份确认队列。

## 实验目标与冻结设计

固定官方SyncTalk commit43079fb3f349675f6cb33031bb07262f7566cdd5，冻结AVE及其BN运行统计，只改变生成器训练输入电平。后续每个种子从相同随机初始化训练−6/0/+6dB三个生成器，每个生成器测三档测试电平；先导种子42/2026，共6次训练、18个模型×电平评价条件。训练步数须通过先导成本/收敛资格后另行冻结，当前不猜预算或提前启动。

主要因果对比为训练×测试电平交互I=[Y(+6,+6)−Y(+6,−6)]−[Y(−6,+6)−Y(−6,−6)]。评分统一保留原自然PCM和时钟，报告C、Dmin、D0、B、offset；真实口型端点使用保留帧的唇部几何与重建误差。一次训练为处理单位，身份为跨人重复单位，种子和片段保留层级，不把帧/重叠窗口算独立样本。正式队列大小需多身份方差和最小效应分析，6–10身份/3种子只是预算提议。测试集不选checkpoint/电平；涨C不称感知改善。归一化、随机增益、HuBERT、TTS换轨/净化均为后续独立扩展。

## 数据准入

正式主队列使用中文真实单人AV，每身份至少300秒train、60秒val、120秒test，另留上下文间隔；嘴部清晰、无配音/音乐/明显削波，同一身份跨划分允许，时间块/录制段不重叠。优先独立录制段；同段划分须披露。按技术质量选择，不能按下游改善筛选。原母版、来源、语言、身份、录制段、上下游暴露/未知、帧率、音频格式和处理链必须登记。AISHELL纯音频不能直接充当视觉训练集；清单有视频路径但文件缺失不计已备妥。May短于正式门槛且官方预处理可能利用全段姿态/背景，只用于已暴露的工程先导。

May来源为官方完整May.zip，SHA256008fe3fb46430a92beff334e26f37eab229c717ff5b577dfbba90f46c3848d4b。完整解压保留原JSON、视频、所有图像/landmark/torso/背景/bs；原aud_ave.npy仅作来源存档，禁止复用为新划分特征。原立体声16k PCM16两声道须逐值相等，再选channel0；不重采样。

## Requirements

### Requirement: 系统 SHALL 在素材变换前保存任务规格与机器协议
#### Scenario: 冻结准备范围
- **WHEN** 开始准备数据
- **THEN** 本任务笔记与scripts/experiments/synctalk_gain_training_data_20261017/protocol.json已保存，含准入、划分、因果对比和验收标准；本轮training计数为0。

### Requirement: 系统 SHALL 构造无交叠的连续划分与准确source-clock映射
#### Scenario: May三划分
- **WHEN** 读取官方全部6073个transform frame
- **THEN** 音频支持块train[0,4200)、val[4250,4850)、test[4900,6050)，单位25fps源帧；块间50帧/2秒弃用间隔。每块首尾各25帧/1秒只作音频上下文，不作监督，监督分别4150/550/1100帧；aud_id=img_id−块起始帧，PCM起点=源帧×640，姿态/landmark/图像/bs保持源索引。
#### Scenario: 防止默认provider静默回退
- **WHEN** 准备正式test和validation数据入口
- **THEN** test只生成transforms_test.json，val只生成transforms_val.json，且交付明确的loader合同；官方test复用val、val截100帧、--test_train与缓存编码均不得绕过合同。真实provider兼容性必须另列资格，不以文件存在代替loader核验。

### Requirement: 系统 SHALL 从原波形构造九组纯增益驱动并固定评分PCM
#### Scenario: 三划分乘三电平
- **WHEN** 对每个支持块分别施加−6/0/+6dB
- **THEN** 9个mono16k DOUBLE驱动逐样本等于原channel0切片×10^(g/20)，无削波/限幅/重采样/归一化；每个划分只有一份原PCM16评分音轨，所有增益共享视频、姿态、landmark、bs和时钟。

### Requirement: 系统 SHALL 为九组驱动重算官方冻结AVE并核验可用支持
#### Scenario: 原前端真实编码
- **WHEN** 使用固定官方代码与AVE原权重
- **THEN** strict加载无缺失，eval/no_grad，九组原始padded和监督aligned编码均保存且shape/finite/SHA合格；0dB重复前向误差核验，所有aud_id在范围内，监督首尾各4行标注注意力边缘。此步骤为编码，0生成器训练。

### Requirement: 系统 SHALL 独立核验数据而非只核验文件存在
#### Scenario: 全量读回
- **WHEN** 准备完成
- **THEN** 独立程序重新读取原ZIP、PCM和全部监督图像/torso/landmark/pose，检查源字节身份、图像解码尺寸、landmark有限且口部可用、拆分不交叠、增益精确、原PCM不变、映射及编码shape；输出机器QA、失败清单、每split预览与明确的READY_ENGINEERING_PILOT资格。图像数不等于独立统计样本数，未做盲评/未见人确认不得写成已完成。
#### Scenario: 合同失败
- **WHEN** 交叠、偏移、削波、文件损坏或缺失
- **THEN** 验收失败并保留失败结果，不静默跳过/修复原件；相应拒绝测试通过。

### Requirement: 系统 SHALL 建立持久副本并验证空间方案
#### Scenario: 本机与Ubuntu
- **WHEN** E盘和Ubuntu根盘不足
- **THEN** F:/Datasets/tts-exp/synctalk_gain_training_20261017作为持久数据位置；Ubuntu仅用独立/dev/shm工作副本，不能称SHM备份；按manifest重建共享资产链接，不复制三份视频母版，不删除历史实验。
#### Scenario: 百度网盘备份
- **WHEN** 尝试现有百度网盘挂载
- **THEN** 使用本任务独立目录、小文件写入和另一客户端读回SHA验证，再尝试归档；记录实际成功/失败和根盘/cache变化。只能在独立读回后标记已验证云副本，挂载df剩余容量和本端回读均不等于云持久完成。

### Requirement: 系统 SHALL 将素材一致性、划分变化和数据量资格分别登记
#### Scenario: 用户核对官方数据与训练充分性
- **WHEN** 交付本轮准备结果
- **THEN** 明确记录原素材42518文件与官方ZIP逐字节一致、真实自然视频/自然语音增益干预、无TTS换轨/合成监督/帧重排；原官方train5520帧/220.8秒，新train4150帧/166秒。数据量充分性标为UNVERIFIED，尚未训练或证明收敛；不得把技术READY等同正式跨身份科学证据充分。

## 执行步骤
1. [x] 加载项目规则、前端证据与数据需求，冻结本spec/机器协议。
2. [x] 准备完整May母版、连续划分、九组音频和共享资产映射。
3. [x] 重算官方AVE与0档重复，保存原数组和provenance。
4. [x] 独立全量QA、合同拒绝测试和分层视觉预览。
5. [x] 检查云端挂载、建立持久交付manifest并核对跨端身份。
6. [x] 更新本任务的验收结论与下一训练资格边界。

## 验收结果（October 17, 2026）

✅ 本任务的数据准备Scenarios均已通过；不是训练实验已完成。持久数据根为F:/Datasets/tts-exp/synctalk_gain_training_20261017，本机与Linux两份独立全量QA均通过：42518源文件/1369015029源字节、17400图像解码、5800landmark/pose、9纯增益驱动、9组AVE编码。实际provider canonical attempt03读取4150/550/1100监督帧，27个collate检查通过，三个0档重复数组逐位同。5个泄漏/偏移/错误增益合同拒绝测试通过；36幅固定等距预览已视觉检查，未见明显遮挡/多人或损坏，不称全帧人工审核。

源MP4为512×512、25fps、6073帧、零起点；9个源帧抽样的最佳抽取索引偏移0，三个12秒音频与原MP4重新解码比对的最佳样本偏移0，相关>0.99999999997。证明未引入抽取时钟误差，不证明原视频的感知同步质量。

AVE原权重/网络计算未改；隔离provider只修正test读取test和val不截100。初次缺DataLoader和packaging.version的隔离依赖错误日志保留，canonical03成功。Windows派生JSON为CRLF、Linux为LF，跨端61文件join通过且单列21个仅换行差异，两端源素材/WAV/NPY仍须字节同。

百度网盘两个文件已从Windows挂载上传并由Ubuntu另一客户端完整读回：May.zip1225104190字节/SHA008fe3fb…，prepared_data_verified.tar.gz90538800字节/SHA1b2e107288733972d6dd90212e00230e858e07ca860c44b540871b87dd736a99。根盘云端读回前后约608MB，未额外占大缓存；没有删除历史记录。F盘持久母版和原数组为主，SHM只作工作副本。

当前训练计数0，生成器forward0。数据量充分性UNVERIFIED：May英语一个身份、新train仅166秒；尚无学习曲线/从头训练收敛或训练时长消融。它只具备工程先导资格；正式中文队列仍需按300/60/120秒每身份门槛准备，并按独立身份方差和最小效应确定人数。

权威入口：local_complete_qa.json、remote_independent_qa.json、source_clock_qa.json、cloud_storage_receipt.json、cloud_bundle_receipt.json、local_remote_join_qa.json和current_delivery.json（均在持久数据根；当前指向delivery_manifest_v2.json）；代码/复现规则在scripts/experiments/synctalk_gain_training_data_20261017/README.md。

## Observations
- [status] done
- [priority] high
- [close_date] October 17, 2026
- [result] 数据与loader技术验收完成；训练量充分性UNVERIFIED，正式中文多身份数据未完成。
- [description] 准备并核验May训练电平先导数据，另登记正式中文多身份准入及当前素材缺口。
- [decision] 当前执行仅数据准备/前端前向，训练需后续固定预算和明确执行指令。
- [limitation] 正式中文长段多身份AV尚未取得；本任务仅完成May数据先导，正式队列尚未备妥。

## Relations
- project [[47-synctalk-amplitude-frontend]]
- follows [[48-synctalk-may-gain-inference]]
- relates_to [[49-may-hubert-gain]]