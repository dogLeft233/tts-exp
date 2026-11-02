---
title: 试跑 Patrick Chris Cathy 的 FP16 粗训练
type: task
permalink: tts-exp/tasks/试跑-patrick-chris-cathy-的-fp16-粗训练
status: in-progress
priority: high
current_step: 2
started: October 18, 2026
---










# 试跑 Patrick Chris Cathy 的 FP16 粗训练

用户要求先训练其他三人，范围及失效规则见 [[045-synctalk-other-identities-fp16-diagnostic-training]]。本轮用独立FP16保护版本获得身份稳定性证据；不覆盖Brown失败，不算原24完整模型完成。原Brown上游增长机制核查仍未完。

## Latest user priority
用户2026-10-18新授权优先：Chris当前已启动epoch完整封存后暂停，先验证并执行Patrick三档FP16精调及新27格，见 [[暂停 Chris 并精调 Patrick 三档及27格]] / [[047-synctalk-patrick-fp16-fine-and-safe-pause]]。旧V3 Windows181520已停止，不自动继续Chris/Cathy；原trial及全部记录保留，尚未完成。当前pinned−6 ep2 job816708由boundary observer195272等待安全结束，真实paused.json优先。


## Requirements
### Requirement: 三身份获得可审计的完整粗阶段或真实终止证据
#### Scenario: 原数据准入已通过但完整训练未知
- **WHEN** 用户要求先试PatrickLeahy、ChrisVanHollen、CathyMcMorrisRodgers。
- **THEN** SHALL分别显式full恢复16500监督文件，按seed42与0/-6/+6dB执行原10粗epochs上限；保留逐步trace、验证曲线、完整初始化和成功更新计数，正常phase按原配对门验收。
### Requirement: 数值失败不能污染原版本或阻止其他身份获得证据
#### Scenario: 新试跑产生非有限state/loss或非法反缩放
- **WHEN** 已验证更新前guard拒绝或job返回失败。
- **THEN** SHALL先完整封存实际失败job/epoch/terminal state，停止该identity剩余条件并推进下一人；未跑条件不标pass，lost job不自动重播，原失败不可删除。

## Current preview handoff
实际交接已核验：Chris完整监督16500文件恢复并逐字节验证通过（16428恢复、72复用）；Windows181520状态training，remote job805176/child805186/training805187实际运行seed42/0dB/coarse ep1。一次实时trace前缀global1520、1520成功、0skip、loss有限；仅前缀，不是epoch完成或完整稳定性证明。Patrick剩余2976非control test缓存已由原controller按F全验与独占lease退休，现72controls保留；以后加载必须显式恢复。Cathy仍未启动。

Patrick三档seed42 coarse10及其完整27格描述预览已完成、独立最终QA与持久交付已通过，报告 [[SyncTalk Patrick coarse10 seed42 27格描述性预览]]。F early_score_preview_v2/all27_preview.json及delivery_v1/report.md、scores_27_cells.csv、mean_distance_curves.csv、patrick_27_grid.png、independent_review.json为结果入口。9score归档225成员freshSHA，162矩阵独立重算maxerror=5.24194470941e-06<3e-5，27格六clip等权与27PNG/head、9PCM复核。9lossless媒体既有F全成员+独立云整档读回保持，显式恢复locators齐全；不是本轮再次全电影解码。

每档60000attempts，−6/0/+6成功59985/59985/59984，15/15/16AMP skips，无terminal非有限loss，CONTROL_FAILED_UNEQUAL_ACTUAL_OPTIMIZER_UPDATES保留，不能推训练gain受控效应。三个实际模型在九种同driver/scorer设置下guard20 Sync-C始终−6>0>+6；只是单身份单seed粗阶段。原24完整模型与fine17未完成，Brown失败原档/旧false均不变。

预览controller176764已all27_ready_for_final_delivery_review正常退出；2026-10-18T10:39:55UTC已按V3 schema写controller_revision03/preview_delivery_release.json，绑定delivery_v1/delivery_receipt.json SHA9533b56022e7436774bdcb4d8e80c3d80144d477e03919644b569c1b09ee7f92。训练181520/V3获准继续Chris/Cathy，实际阶段以receipt.json为准，不重启旧controller或重放epoch。原V3自行全验Patrick F archive后退休剩余非controls，再显式full恢复Chris监督。交接前预检发现恢复Chris距180MiB安全余量差约6MB，已再次鲜核9份F score archive全部225成员与local数组，并通过Linux独占lease仅退休216个remote评分.npy重复缓存，scores.json/F原数组/归档全保留；free1330597888→1390317568B，source/intent/receipt在early_score_preview_v2/score_cache_retirement*，未来remote评分数组需显式从F恢复。

预览v1 CPU EMA筛选失败及v2修复资格记录保持；v2全部parameters映射与官方torch_ema.copy_to精确、三档完整state/60EMA计数器和9帧重复FFV1PCM资格通过。原冻结源与contract不改。Patrick此前退休13452 trainval非controls，全部test保留到预览完成；未来任何已退休输入加载必须显式恢复。每视频F重复tar已在cloud全读回后退休，不假定仍在。不得修改共享venv/CloudDrive或重复上传/cleanup。

## Latest user priority
用户要求“一个人完成后先计算分数看看结果吧”。按 [[046-synctalk-first-identity-early-score-preview]]，首张三档表已交付，现完整27格已独立核验并持久交付，按V3 release继续Chris/Cathy。原请求early_score_request_v1/request.json SHA941e1b9dade0285a9c90c8edab5852704e3910d2c337e9acd5a6e0afd7801b54保留创建时QUEUED状态，真实执行以当前头部与新receipt为准；原fine17正式renderer与失败状态不变。

## Preserved initial execution evidence
FROOT/trials/fp16_other_identities_trial_v1，remote ROOT/trials/fp16_other_identities_trial_v1；ROOT=/dev/shm/synctalk-multispeaker-data-20261017，FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017。冻结contract SHA787d917d259eeca89311016737893348578335e3623f217858797926311893bd；7执行源跨端SHA精确，原12Windows source pins通过；原failure archive SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72再次不变。

真实Windows controller PID164828（controller_revision02仅主机控制器接管，原GPU job709978/child709985/training709986连续未中断）；此写入时状态training，current={"identity": "PatrickLeahy", "seed": 42, "epoch": 1, "gain_db": 0, "relative": "training/PatrickLeahy/seed42/gain_+0/coarse/ep0001/attempt01"}。启动器隐藏运行，独立锁/不可覆盖job，旧105372/133984不重启。下一轮以receipt.json、controller_failure.json、remote jobs/*/execution.json和实际ps为准，不重复启动。controller stdout/stderr保留。source冻结在F与remote sources/，仓库对应scripts/experiments/synctalk_multispeaker_full_20261017/{other_identities_trial_v1,other_identities_trial_remote_v1,controlled_training_trial_v1,train_other_identity_epoch_v1}.py。

qualification/result.json passed：微型实际更新4，原vs试跑trace/state精确，firstskip/terminal state捕获通过；独立F逐SHA7文件通过。tests/test_synctalk_other_identities_trial_v1.py三项调度/路径测试通过。正式人体参数更新应由真实trace报告，不把资格4步算人体训练。新预算最多540000人体attempts，与先前零更新诊断32/32及原失败attempt分开。

Brown完整F archive fresh整包及所有成员验证，内存16428非控制图像通过exclusive leases退休；free_before81637376B/free_after1931939840B。Brown数据输入仍保存在F归档，72控制保留；后续诊断需要被退役路径时必须full恢复。Patrick full16500恢复已passed，16428 restored/72 reused；接管时再次全16500逐SHA通过。其他identity切换同样先全验归档再退休缓存，不改共享venv/CloudDrive或原数据pins。

控制器接管修复Windows Path把远端checkpoint路径变为反斜杠的问题，首次epoch持久化前完成；原7源/contract不改，另立controller_revision02/handoff.json及wrapper SHA4a03e7c1cf501d89c97207bcc07ee14035348c61a595a1c7f1a2a4e3eaf41d38。停止的仅Windows166604，未停止GPU或重放optimizer；旧receipt已保留。02:59:03UTC独立持久trace前缀1897/1897实际成功更新、0skip、全部finite，SHA72c102dca1da05b54da6da3f74c7a68ba0a9d69c9d07e1d046550e8314960ed4，见observations/first_actual_updates。其后实时tail到3832成功/0skip；前缀不是epoch完成证明。Chris/Cathy仍待运行。

## Observations
- [status] in-progress
- [decision] 按用户新授权先三身份FP16独立粗训练试跑；完整正式24模型/27格仍未完成。
- [result] Patrick首张固定driver0/scorer0表已真实通过独立复核：seed42、coarse10固定EMA、原六段test等权、guard20主端点；训练−6/0/+6的Sync-C分别8.213/8.068/7.770，Sync-D分别6.790/6.840/7.077。仅描述实际三个模型，不能作训练gain受控效应或感知改善结论，原CONTROL_FAILED及59985/59985/59984成功更新不变。
- [next_step] 通过经过完整交付核验的V3 release继续Chris/Cathy独立FP16粗训练，观察真实receipt/失败捕获，不重放已完成job。

## Relations
- project [[SyncTalk 四新身份官方完整训练电平实验 Spec]]
- implements [[045-synctalk-other-identities-fp16-diagnostic-training]]
- relates_to [[核查 SyncTalk FP16 适配与官方环境]]
- depends_on [[准备并核验四个新身份的 SyncTalk 完整训练数据]]