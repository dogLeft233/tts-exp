---
title: Codex checkpoint - 2026-10-18T12-08-08Z - Chris safely paused and Patrick FP16
  fine entry qualified with training running
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-18-t12-08-08-z-chris-safely-paused-and-patrick-fp16-fine-entry-qualified-with-training-running
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-18T12:08:08.190572+00:00'
username: dogle
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

# Codex checkpoint - 2026-10-18T12-08-08Z - Chris safely paused and Patrick FP16 fine entry qualified with training running

## Summary
Chris已在完整epoch封存后暂停，Patrick三档FP16精调入口已资格通过并开始正式fine11；fine17新27格仍待实际模型与真实渲染资格。

## Story
本线程要分离SyncTalk训练、驱动、评分三个环节对音量的响应，最终完成四身份、两seed、三训练gain的24个完整模型和27格。Brown原+6粗训练非有限loss失败后，用户要求保留FP16和全部失败记录、先修更新前检查并对照官方环境，再试Patrick、Chris、Cathy。Patrick三档coarse10已完成且描述性27格已交付，但实际成功更新59985/59985/59984不等，原CONTROL_FAILED保持，不能把得分排序当训练gain因果或感知改善。

最新用户明确改优先级：Chris当前轮完整结束保存后暂停，另立版本验证Patrick三档FP16精调入口，fine后重新算27格。本轮只冻结Windows调度器，已运行Chris−6 ep2正常结束并逐成员封存，无GPU信号；0/−6 ep2、+6 ep1完整断点保持，Chris/Cathy暂缓。新的patrick_fp16_fine_v3携带精确parent phase文件，三gain从各自coarse10 RAW完整state进入fine11–17；入口6实际尝试均为普通AMPskip，0optimizer更新，loss有限、完整恢复和采样记账通过。资格允许原GradScaler正常skip，不调scaler或精度，不把入口资格当全程稳定性证明。正式fine已产生真实成功更新。

## Working State
权威Task为[[暂停 Chris 并精调 Patrick 三档及27格]]，status in-progress/current_step3；决策[[047-synctalk-patrick-fp16-fine-and-safe-pause]]。FROOT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017，OUT=FROOT/trials/patrick_fp16_fine_v3，RUN=/dev/shm/synctalk-multispeaker-data-20261017/trials/patrick_fp16_fine_v3，Ubuntu wjj@192.168.50.2。F文件和receipts是持久证据；SHM、进程和当前前缀易失。

新contract SHA7d6276b92c406a8496de8f230c66db0eed37351fec259ca8af7c8c3b7ed073b9，7源固定；本次fresh核验F七源全部匹配。parent_receipts/coarse_qualification.json SHAa492c2d70d1406b92a8148d97cab64804ed9bbb8e21539f8155c695b4d119ad4在F/remote精确。Windows controller177316和capture observer182424仍运行；remote job824928/child824966/training824967持续运行0gain fine11。12:07:16UTC只读尾行global64065、累计成功64038，相对父59985为4065新尝试/4053新成功/12普通skip，尾行loss有限、scale32768。它是live前缀，不是完整epoch或已持久状态。OUT/receipt.json、controller_failure.json、remote jobs execution与ps优先；当前无controller/observer failure。新formal上限126000尝试，每档fine11–17各42000，累计102000；原FP16/loss/AdamW/EMA/原fine计数器重置和scheduler保持。

Chris权威OLD/controller_revision04_pause_for_patrick_fine_v1/paused.json（OLD=FROOT/trials/fp16_other_identities_trial_v1）。已运行job816708 rc0/795.109秒完整保存；+6 ep2未跑。0/−6 ep2成功12000/12000、+6 ep1成功6000，最新三档F checkpoint保留。原Windows181520停止，未来不能重启旧V3，须逐receipt新handoff。Chris16500监督只在F完整archive鲜验后退休16428非controls；最新三个remote checkpoint副本也鲜核F SHA、Linux独占lease后冷置，locators在fine_v2/chris_checkpoint_cache_retirement_intent.json/receipt，恢复前必须显式从F恢复，不能假定旧remote路径存在。

Patrick16500监督已显式完整恢复，v3重新逐SHA核验且0重复上传；OUT/storage/PatrickLeahy_full_restore.json。fine_v1/closed_cache_retirement_*封存724个已闭合remote训练/资格/firstskip重复cache的鲜核与lease证明，257822309逻辑字节，F原字节/曲线全部保留。capture observer仅在闭合case/epoch的F full archive全部成员鲜核后退休remote qualification checkpoint/firstskip副本，marker和F state保留，不触碰formal当前state/trace。12:07SHM free197906432B/root568545280B，真实df优先。fine完成评分前还需新版本独立鲜验归档后退休Patrick trainval13452非controls，保留3000test+48trainvalcontrols；不能用旧coarse退休receipt当新鲜fine证明。

fine_v1/v2历史均保留。v1在执行GPU前因错误要求前两次必须成功更新而替代；首次CRLF部署SHA拒绝也留档。v2首资格job821455因远端缺主机parent phase文件，在optimizer/trainer构造前失败，0attempt/0optimizer/0formal，job/case F归档；v3固定携带parent phase SHA修接线后独立运行，不重放旧job。资格独立review首次误把host derived JSON当raw archive比较物理SHA失败留review_attempt01_failure.json；修正后fresh三archive78成员+3完整checkpoint、raw receipt和采样计数通过，host derived JSON仅值相同不冒称字节相同。

评分尚未冻结或执行。候选源在scripts/experiments/synctalk_multispeaker_full_20261017/patrick_fine_score_v3.py和patrick_fine_score_controller_v3.py，预期OUT/fine_score_27_v1。原候选render receipt残留coarse10标签，本次只修未冻结评分候选，并增加纯stdlib receipt gate拒绝coarse视频、错gain/driver、错checkpoint、错EMA和错帧序；原候选精确保存在OUT/fine_score_candidate_revision01，新候选+测试及CPU receipt在revision02。新score helper必须纳入最终score source manifest。训练七源未改，0新评分GPU/视频/分数，不能宣称fine17渲染资格已通过。实际三fine17完成后，绑定fullcheckpointSHA/EMA102/epoch17/global102000与fine_ready_for_27_grid.json（identity在ready.phase.identity），再执行9固定帧重复FP16/FFV1PCM读回资格。

## Changed Files
- scripts/experiments/synctalk_multispeaker_full_20261017/train_patrick_fine_epoch_v3.py、patrick_fine_contract_v3.py、patrick_fine_controller_v3.py、receive_patrick_fine_supervision_v3.py：独立精调版本与父phase接线；v1/v2未覆盖。
- scripts/experiments/synctalk_multispeaker_full_20261017/patrick_fine_score_v3.py、patrick_fine_score_receipts_v1.py及tests/experiments/test_patrick_fine_score_receipts_v1.py：尚未部署的fine17评分候选/phase和checkpoint验证。
- .codex/scratch/中的安全边界observer、独立资格review、已核验cache退休helper及当前checkpoint脚本是机器本地未跟踪工作；F对应source/intent/receipt才是持久执行入口。
- basic-memory/Tasks/暂停 Chris 并精调 Patrick 三档及27格.md、ADR047、父Task与研究地图已更新；不按epoch追加文档日志。

## Verification
三gain入口真实6尝试/0成功更新/6普通AMPskip，loss有限；lip True/False、原两固定train帧与rays、三gain采样相同，model/optimizer/scheduler/scaler/EMA/counters恢复及roundtrip exact。独立review鲜核78archive成员+3fullstate通过。父coarse10三个F checkpoint再次鲜核未变。正式first success global60008，初始169前缀159成功/10skip另留startup_live_prefix_readback.json。4CPU fine guard合同测试及既有3EMA CPU映射测试通过；本次新3个receipt CPU unittest及两个候选py_compile通过，拒绝coarse/wrongstate/EMA/driver/帧序，不依赖GPU。

Patrick粗训练27格早已交付，详见[[SyncTalk Patrick coarse10 seed42 27格描述性预览]]与OLD/early_score_preview_v2/delivery_v1。Brown失败archive SHA55855cf8b3f28e088df0d8a056336a34ce6bb4acfaef4420feb9de97eab21e72和原实验全部保留，不重启105372/133984/job610686。原官方环境复现和FP16 guard证据保持；上游eye/sigma/权重增长根因仍未完全隔离。新精调也不消除父CONTROL_FAILED。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp) 已由GitHub只读API确认canonical；当前branch PR查询返回空。
- 本地HEAD `2b71639c3b555293375d248350999377168d515d` GitHub解析返回422，未构造commit链接。git status有9已跟踪修改和大量未跟踪条目（本次快照1051项），不是本轮全改，不得全量clean/stage/commit。
- OUT/fine_score_candidate_revision02/candidate_receipt.json记录候选SHA、3CPU通过及尚未GPU资格边界；live证据在.codex/scratch/patrick_fine_v3_checkpoint_evidence_20261018/live.json。

## Observations
- [result] Chris已完整保存暂停；Patrick三gain入口资格通过，0gain fine11正式训练正在产生成功更新。
- [decision] 保持FP16和原GradScaler，普通skip精确记账；首terminal/lost封存后停止新fine，不自动换gain或重放，不补预算。
- [decision] fine结果始终继承coarse CONTROL_FAILED，只描述实际模型；新27格固定fine17 EMA与原6testclips，不按分数选条件，主guard20/full次、SyncNet FP32 TF32off、0PCM16/±6float64原单位、FFV1PCM正式。
- [blocker] 尚未完成21个fine epoch及fine17新GPU渲染资格、9新视频/27格/独立QA和持久交付；原24full目标仍未完成。
- [next_step] 持续验收真实fine epoch与F完整state，在安全GPU边界绑定三fine17 SHA并资格/运行新的27格、独立复算与coarse并列交付；健康运行静默，heartbeat ACTIVE，Chris/Cathy继续暂缓。

## Relations
- continues [[Codex checkpoint - 2026-10-18T11-34-45Z - Verified Chris epoch checkpoints and Patrick fine handoff requirements]]
- relates_to [[暂停 Chris 并精调 Patrick 三档及27格]]
- relates_to [[047-synctalk-patrick-fp16-fine-and-safe-pause]]
- relates_to [[研究主线与证据地图]]
