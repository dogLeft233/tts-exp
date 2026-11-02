---
title: Codex checkpoint - 2026-10-19T08-22-16Z - Patrick RMS repeat metadata handoff
  repaired and next gain running
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t08-22-16-z-patrick-rms-repeat-metadata-handoff-repaired-and-next-gain-running
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-19T08:22:16.958738+00:00'
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

# Codex checkpoint - 2026-10-19T08-22-16Z - Patrick RMS repeat metadata handoff repaired and next gain running

## Summary
同初始化RMS复跑的0档首轮6000成功已完整保留；其后的provenance缺路径故障经独立审核，只补数据别名和新协调handoff，−6档首轮已真实运行，无重复训练。

## Story
用户要比较SyncTalk音量三个轴，后来授权同AVE前逐完整录音RMS0.1三档粗训练。旧RMS在0coarse4产生真实密度exp溢出；定位重现float32 exp前9个过大logit，扩张为52Inf，换旧未归一化特征或全query FP32仍溢出，故当前输入RMS直接致溢出不成立，训练轨迹因果未识别。强制恢复有限密度缓存的probe116次全skip/0更新、scaler耗尽，已另封存。用户再要求“从头开始一次带归一化的试试”，本repeat用同seed42完整随机initial、同RMS、原FP16/math，绝不继承失败状态。

本轮heartbeat发现repeat0gain coarse1已完成6000attempts/6000success/0skip、完整RAW断点及三gain验证，随后包装补写normalized_binding_sha256读取新RUN不存在的datasets路径而rc1。实际runtime一开始就有意使用父RMS datasets，仅新RUN元数据别名漏部署。这是我方post-epoch provenance接线错误，不是数值失败。原host正确归档停止，未运行−6ep1。独立fresh核验全archive/standalonecp/rawtrace后，只建立指向精确旧输入的别名，生成另存且绑定rawSHA的derived receipt，另立host-only协调器，从未运行的−6ep1继续，未重放0ep1。19冻结训练源、合同和所有数学不改。

## Working State
OUT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_repeat_v1。权威OUT/active_controller.json指向controller_revision02_metadata_handoff_v1/receipt.json与controller_failure.json；原OUT/receipt.json和bootstrap230152为历史metadata rc1，不能再误判活动数值terminal或重启。新host229476已启动−6gain coarse1 job66325/child66337/trainer66338，commandSHA0139b08517da1267a1d28f6b070f660da19ce8d38b7725940d4a2ee2b4c712e5。08:17:50UTC独立完整live前缀131行/131success/0skip，loss有限，frame/sourceaudio/pixel/rays/lip/rect/LR与0ep1前缀exact；非epoch完成。PID易失，之后读真实receipt/execution/ps。

contractSHAb44d67ce4daa7076e88648adefceb663118dd877a5b2a527a7bcf2d5439ee53b，19冻结源exact；initialSHAd6ce1f369bd2ee1695288c93f1602fcf181f0329313c710388e6b7c70071be5c。同seed42、各gain coarse1–10×6000/总180000formal，原FP16/defaultGradScaler/guard/loss/AdamW/EMA/65536rays/maxsteps16/LR/采样时钟不变，无失败权重/scaler warmstart/cache rollback/clip/precision change或补步。RMS仍gain后完整录音float64缩放到0.1再float32同AVE，无去均值/epsilon/limiter；不是编码器统一原生标准，具体processor可z-score或不归一化，Whisper用log-Mel尺度，此次不换算法。

0ep1 training archiveSHA35a7570176efca9f750ed5134d8d496044620241a43597343dd407aba146915d，43成员；standalone fullcpSHA04e329e6908deaa3c7a207fe15d122e6e6d965cfd647e258946edf591002193b。6000traceSHAe738cbbf37f841327f5181971d7f82366dd0130f3bb2af7e60738409bf8f63b0；raw epoch_receipt SHA96286a36f4105e25a654f46e88015ad38eb662c6741885159ed275f0062674de保持。三gain验证各750frames meanMSE0.00016193483100505546，验证trainstate/RNG恢复exact。job archive5成员freshSHA通过，rc1/stdout traceback原样保留。派生gain_+0_ep0001_derived_receipt.json只增加原wrapper期待6个provenance字段，不改raw所有值，F/remote字节SHA一致，不能把raw与derived物理SHA混为一谈。

remote NEW_RUN/datasets→OLD_RMS_RUN/datasets，仅供原wrapper元数据lookup；9份train/val/test×gain的binding和whole_ave freshSHA逐个一致，实际训练选用路径/缓存未变化，无GPU资格重算。handoff SHA47f4a67ecaeb965b378bcd84fd3c09b205d6715a1ecb698b7a55f080a5408e83绑定原receipt/audit/别名/source；新host拒绝非精确prefix和已存在未来job。原root与rawreceipt/archive不覆盖，新repeat全部fullcp保留，无删除。

持久入口REV=OUT/controller_revision02_metadata_handoff_v1：original_metadata_failure.json、audit.json、dataset_alias_receipt.json、qualification.json、handoff.json、startup_live_readback.json、memory_publication_receipt.json、repair_delivery_manifest.json、closure_receipt.json。closure fresh核31个不可变元数据文件，manifestSHAfe8b003042e6dc7556cbc2ad64058ad16528247809bbe8613d5ade7154d320e6；mutable receipt/stdout/stderr排除。BM地图、新repeat实验、Task当前头部/唯一Observations与任务板均已发布读回，status running/in-progress/current_step2；旧数值失败实验不改。patrick-rms自动化ACTIVE，每10分钟，已改最新authority，健康DONT_NOTIFY/无commentary，只关键最终交付/真实失败/需输入通知。旧synctalk-27已删除，不重建。

机器局部SSH wjj@192.168.50.2须-b192.168.50.1，ROOT=/dev/shm/synctalk-multispeaker-data-20261017，RUN=ROOT/trials/patrick_ave_rms_coarse_repeat_v1，父数据RUN去掉_repeat。SHM/进程易失，F为持久证据。共享venv/CloudDrive不改。早前约2.49GB两旧资源百度云归档回收及恢复locators已完成，勿重跑回收worker。CodeGraph disabled时已按banner直读源，未重建索引。

## Changed Files
- scripts/experiments/synctalk_multispeaker_full_20261017/patrick_rms_repeat_controller_revision02.py：新host-only审核接续，冻结源未改，无cp退休。
- tests/experiments/test_patrick_rms_repeat_metadata_handoff.py：CPU测试精确prefix、POSIX containment，以及实际冻结wrapper函数在numeric stub完成后缺binding失败/补齐后成功，fixture不是实验结果。
- .codex/scratch/prepare_repeat_metadata_controller_v1.py、audit_repair_repeat_metadata_v1.py、read_repeat_revision02_live_v1.py、publish_repeat_metadata_handoff_v1.py、closure/finalizer和checkpoint writer；执行源/历史准备失败在F持久。机器本地大量未跟踪文件不代表已提交，勿bulk clean/stage/commit。

## Verification
fresh核0ep1 archive43成员+1完整standalonecp、job5成员、原rawreceipt与6000trace逐行计数/finite loss通过；未load全tensor/新GPUforward/optimizer/replay。19源/contract与9数据别名binding/feature SHA fresh通过。三个CPU unittest已实际通过，metadata helper还拒绝gap/duplicate/failed prefix；live131跨gain采样及LR匹配。原入口6success/0skip/fullstate roundtrip/F13成员+3cp是前序真实资格，不证明长期稳定性。

审计准备三次失败留档：期望源码label但真实inline gain映射、嵌入NUL被Python解析器拒绝、inventory漏列已完成资格job，均先于远端别名 mutation。CPU stub首轮只接kwargs而原函数传9positional，测试修复后通过。closure两次GBK读取UTF8准备失败保留说明，finalizer显式UTF8逐manifest读回通过，不覆盖原已核source或manifest。这些不属于科学训练失败，没有掩盖旧numeric terminal。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp) 只读API确认canonical、branch无PR；远端不能解析的本地HEAD不构造commit链接。
- 新实验tts-exp/experiments/patrick-rms-归一化同初始化独立复跑；Task tts-exp/tasks/运行-patrick-的-rms-归一化-ave-三档粗训练并评测。
- 本次旧RMS完整数值诊断/强制恢复probe和云回收细节由前序同chatcheckpoint/原实验及各F报告保留；不复制长日志。

## Observations
- [result] repeat0ep1数值6000success/0skip已全验，post-epoch metadata路径故障已保留并修复，新−6ep1真实131success前缀，无0ep1重放。
- [decision] 只补精确数据别名/derived provenance和host-only接续，19冻结源/数学/预算保持；本次修复不是未来numeric terminal或lost的自动放行。
- [decision] 首真实numerical/job terminal或lost须封存后停止新job，绝不自动换gain/replay/scalertune；旧fine14/Chris/Cathy及原24full持续暂停。
- [blocker] 余下29epochs及新coarse10资格/27格/独立QA交付未完成，RMS训练轨迹因果未识别；无新视频/评分或感知结论。
- [next_step] 按active_controller真实authority审核后续闭合epoch与配对采样/LR/成功更新，健康静默；三真实coarse10后另冻结新score合同与9固定帧资格，再新27格持久交付。

Fresh GitHub evidence: {"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}

## Relations
- continues [[Codex checkpoint - 2026-10-19T08-13-09Z - Patrick RMS repeat first epoch complete with metadata handoff pending]]
- relates_to [[Patrick RMS归一化同初始化独立复跑]]
- relates_to [[运行Patrick的RMS归一化AVE三档粗训练并评测]]
- relates_to [[Patrick同AVE逐录音RMS归一化三档粗训练与27格评测]]
- relates_to [[048-synctalk-patrick-same-ave-rms-coarse]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1060条；大量其他任务改动，不批量stage/clean/commit。
