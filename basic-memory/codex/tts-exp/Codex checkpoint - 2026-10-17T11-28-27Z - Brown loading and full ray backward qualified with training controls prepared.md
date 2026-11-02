---
title: Codex checkpoint - 2026-10-17T11-28-27Z - Brown loading and full ray backward
  qualified with training controls prepared
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-17-t11-28-27-z-brown-loading-and-full-ray-backward-qualified-with-training-controls-prepared
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: 2026-10-17 11:28:27.096421+00:00
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

# Codex checkpoint - 2026-10-17T11-28-27Z - Brown loading and full ray backward qualified with training controls prepared

## Summary
Brown的监督恢复、90项lazy像素/rays等价和18项原FP16全ray反向控制已实际通过，写好了保持原训练数学的epoch控制器，但四人完整训练准入未闭合，新TFG optimizer更新仍为0。

## Story
用户已授权四新身份各6–8分钟自然数据、同人不同录制测试，全部门通过后执行4身份×seed42/2026×三训练增益的24个官方完整模型，再三driver×三scoring交付27格。原预算仍是FP16/65536rays，10个基础完整epochs60000次尝试后7个lip完整epochs累计102000次尝试；实际成功optimizer更新须同预算，原AMP skips如实记录，不改scaler或追加步数掩盖差异。ADR038的v1坐标失败保留，ADR039训练共享参考点v2仍冻结。

本轮把“归档存在”与“可以加载训练”分开。监督恢复工具从F已独立逐成员验证的tar流式恢复，接收端只允许manifest中的GT/torso、逐SHA核验且拒绝覆盖不同字节。Brown恢复早中晚控制72文件，逻辑dataset绑定全8250帧但显式标记partial，full loader遇缺图即拒绝训练。随后实际比较原provider preload0/1的90个条件，三增益、早中晚、原65536粗/patch与原lip矩形、full验证/testportrait均逐像素/条件/ray索引/方向精确，原整录制attention边缘精确且不跨录制。

新增执行合同和外部ControlledTrainer保留原train_one_epoch、loss、AdamW、scaler、EMA.95每1000尝试更新与LR数学；只控制共同帧/ray随机域并记录真实sampling与optimizer post-hook计数，验证不改变训练状态。train_epoch.py按全6000帧执行单epoch、保存完整官方checkpoint及renderer非注册counter/验证后EMA状态，fine边界按官方重建model/Trainer但载入自己coarse完整state。发现原best checkpoint的EMA state_dict别名会在restore后变回raw，finalEMA改用应用EMA后深拷贝，再restore原模型。此修复只用于明确交付finalEMA，训练数学未改。

## Working State
持久根F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017；易失ROOT=/dev/shm/synctalk-multispeaker-data-20261017，SSH wjj@192.168.50.2。Task/spec/ADR038/039是设计权威；Task当前可恢复状态仍部分滞后本checkpoint，须更新。四人原材料独立QA及source-owned评分参考均通过。stage1_worker已完成四人JPEG/FAN/parser；Cathy技术receipt passed不自动代表独立人脸flags和F全持久门通过。stage2_v2_worker pid233261仍在Patrick geometry，Brown全量v2/监督/AVE通过，其余全量未通过前禁止实验训练。

Brown GT/torso仅恢复控制72文件，完整16500文件仍在F verified derived包；full训练前必须全恢复。Brown logical binding passed、local_supervision_complete=false、training_admitted=false。loading_qualification/SherrodBrown/loading_qualification.json passed90cases。capacity_qualification/SherrodBrown/capacity_qualification.json passed18cases，源SHA9d172b4ba2e90e8b21760eb5e600024b96dc502eb245eebc7723363513a98f9c，零optimizer更新且参数字节不变，粗/patch65536、max_steps16未减；lip使用原rect全部rays（本控制3844），不是声称lip强制65536。

capacity attempt01因helper错误假设bs_area=eye而失败，未到backward/更新。官方默认upper需要7维，修正后attempt02通过；同时保留fine阶段flip_finetune_lips=True，即使当前patch迭代finetune_lips=False。attempt01目录及日志保留。默认GradScaler scale65536在所有lip控制出现非有限梯度，粗/patchfinite；这只证明资源和反向可运行，不证明未来实际更新数相等。正式记录skips并于phase结束比较，不能调整政策消除这个事实。

runtime_exec.sh隔离指定已存在GCC12/CUDA/toolchain缓存，audit Python3.10/Torch2.5.1原环境不变；LPIPS官方AlexNet244408911字节已F/SHM完整SHA验证。无新生成视频，无实验TFG optimizer更新。旧capacity replacement watcher PID115672已停止，scratch replace_capacity_probe脚本废弃不可再启动。Windows stage1/flow/derived/qualification/canonical persistence observers仍在，查真实receipt再复用，不能重复启动。root约527MB、SHM约3.2GB，F后续空间须验证后仅退役本任务缓存，CloudDrive配置及未验证原数不动。

## Changed Files
未跟踪目录scripts/experiments/synctalk_multispeaker_full_20261017新增receive_supervision_restore.py、restore_supervision.py、qualify_loading.py、qualify_capacity.py、execution_contract_v1.json、execution_state.py、controlled_training.py、train_epoch.py、runtime_exec.sh；build_datasets/runtime_common加入partial/full加载准入。tests/experiments/test_synctalk_multispeaker_full_20261017.py新增RNG/counter恢复、真实CUDA toy GradScaler overflow计数、finalEMA深拷贝控制。源已上传ROOT，F execution_sources/runtime_controls_attempt04逐字节source snapshot包含sh。共享dirty tree有大量无关修改，不清理/全stage/commit。

## Verification
实际Ubuntu13个unittest 4.654秒全通过0skip；toy测试确有AdamW更新但不是TFG模型训练。Brown恢复72文件逐SHA、loading90cases、capacity18zero-updatecases实际通过，capacity峰值reserved约3.2GB。full checkpoint跨epoch/phase连续性尚须专门端到端资格；train_epoch尚未实际执行，coordinator、全四人独立data admission、render/27-grid score及独立QA未完成，不把代码存在当结果。

Git root/branch/SHA实际核验；GitHub只读fetch确认canonical仓库URL，search_prs确认该head无PR。SHA此前GitHub无法解析，当前仍标本地未验证推送，不构造commit链接。同chat两类及legacy检索has_more=false，上一checkpoint exact agent/session/started已验证。

## References
- [仓库](https://github.com/dogLeft233/tts-exp)
- 本地未验证推送SHA `2b71639c3b555293375d248350999377168d515d`
- scripts/experiments/synctalk_multispeaker_full_20261017/execution_contract_v1.json
- F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/execution_sources/runtime_controls_attempt04/source_manifest.json

## Observations
- [result] Brown90项原loader等价和18项完整ray反向控制实际通过，实验TFG更新仍0。
- [decision] 保留原默认upper7维、fine lip/patch切换、FP16 scaler/EMA/optimizer数学；资源probe不等于更新预算通过。
- [blocker] 其余三人全量几何/数据资格、完整恢复、跨epoch恢复验收以及协调/渲染/评分/独立QA未闭合。
- [next_step] 更新Task并持久化Brown新资格原数，完成跨epoch完整状态恢复资格与24模型协调器，同时复用worker完成其余三人，全部门通过后执行已授权训练。

## Relations
- continues [[Codex checkpoint - 2026-10-17T11-03-07Z - Shared training geometry passed Brown and automatic full experiment continuation active]]
- relates_to [[准备并核验四个新身份的 SyncTalk 完整训练数据]]
- relates_to [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[039-synctalk-shared-training-flow-geometry]]