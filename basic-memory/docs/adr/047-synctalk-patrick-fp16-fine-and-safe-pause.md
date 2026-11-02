---
title: 047-synctalk-patrick-fp16-fine-and-safe-pause
type: decision
permalink: tts-exp/docs/adr/047-synctalk-patrick-fp16-fine-and-safe-pause
status: accepted
project: tts-exp
decided: 2026-10-18
---



# 047-synctalk-patrick-fp16-fine-and-safe-pause

## Decision
用户明确授权：等Chris当前轮完整结束保存断点后暂停，另立版本验证精调入口，再跑Patrick三档FP16精调，精调后算27格；原训练、失败和粗结果全部保留。该授权改变ADR045的当前仅coarse执行范围，但不修改原trial合同或宣称父阶段控制通过。

## Context and rationale
Patrick seed42三档coarse10已完成并交付27格，实际成功−6/0/+6为59985/59985/59984，CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES保留。新执行版本patrick_fp16_fine_v3固定（v1保留并在执行GPU前替代）每档自己的coarse10完整raw model/optimizer/scheduler/scaler/EMA继续，统一fine11–17，每轮6000尝试，每档新增42000、累计102000，三档新增126000尝试。原FP16/defaultGradScaler/65536rays/maxsteps16/AdamW/EMA/损失/交替唇部与patch64、原fine阶段fresh-constructor计数器重置不变；不使用EMA-only预览当warmstart，不补偿跳步或改变数值政策。

## Alternatives and consequences
不杀当前GPU epoch、不从未保存的部分epoch重放、不直接复用coarse-only入口、不把CONTROL_FAILED改pass。为先验证入口，三档各从原完整state隔离执行相同前两帧，共6资格尝试且最多6真实optimizer更新；模型/优化器/scheduler/scaler/EMA恢复、完整checkpoint roundtrip、model counters、lip/nonlip交替及三档采样精确必须通过，才启动formal。资格步骤单列，不混入正式预算。

当前交接核查时Chris0dB ep2已完成F全验，旧V3先启动−6 ep2；11:35:47UTC只冻结主机调度，随后停止Windows181520，0 GPU signals。已启动job816708让其完整完成并封存，边界观察器写paused.json后才能进入fine准备。Chris最新安全断点将为0/−6 ep2、+6 ep1；未来恢复从逐receipt位置接续，不重放已成功epoch，旧V3不能盲重启。Cathy暂缓。

空间仅按ADR043 fresh F完整归档与独占lease退休Chris非controls监督，保留72controls与全部断点/证据；Patrick已退休监督必须版本化显式恢复全部16500并逐SHA通过。不得改共享环境/CloudDrive/未验证原数。

fine完成后固定epoch17 EMA和原六段test做新三轴27格，原FFV1/PCM、guard20主/full次、原0PCM16/±6float64 PCM单位、FP32SyncNet TF32off，独立矩阵/聚合/像素资格与持久交付，与coarse结果并列。父CONTROL_FAILED意味着fine只是描述性延伸，即使fine阶段本身实际更新相等也不能宣称训练gain受控或原24模型完成。

资格判定补充：固定原两次FP16尝试允许官方普通AMP skips，必须精确报告实际success/skip和finite state/完整state恢复；不以要求每步成功倒逼scaler调参，资格不等于整个fine阶段稳定性。原coarse三档scale实测33554432/134217728/67108864，进入新损失阶段可能出现正常backoff，不能将其自动称为terminal数值故障。

V2首次GPU入口资格因远端缺少主机coarse phase文件，在optimizer/trainer构造前停止，0资格attempt/0更新。原失败已F封存，V3固定携带独立phase SHA副本；新qualification不重放旧job。

资格判定补充：固定原两次FP16尝试允许官方普通AMP skips，必须精确报告实际success/skip和finite state/完整state恢复；不以要求每步成功倒逼scaler调参，资格不等于整个fine阶段稳定性。原coarse三档scale实测33554432/134217728/67108864，进入新损失阶段可能出现正常backoff，不能将其自动称为terminal数值故障。

## Observations
- [decision] 按用户新授权优先暂停Chris，验证独立入口后运行Patrick seed42三档FP16 fine11–17及27格。
- [constraint] 新版本保留原coarse、Brown失败、全部旧false；不以test分数挑训练档、checkpoint或补预算。
- [evidence] FROOT/trials/patrick_fp16_fine_v3/contract.json SHA7d6276b92c406a8496de8f230c66db0eed37351fec259ca8af7c8c3b7ed073b9，7源冻结、二进制部署freshSHA通过；3CPU父绑定/失联重放测试通过，真实GPU资格待边界后执行。

## Relations
- affects [[试跑 Patrick Chris Cathy 的 FP16 粗训练]]
- affects [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- relates_to [[045-synctalk-other-identities-fp16-diagnostic-training]]
- relates_to [[SyncTalk Patrick coarse10 seed42 27格描述性预览]]