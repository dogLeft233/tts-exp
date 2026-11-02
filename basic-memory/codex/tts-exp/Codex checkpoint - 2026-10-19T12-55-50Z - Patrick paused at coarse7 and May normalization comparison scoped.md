---
title: Codex checkpoint - 2026-10-19T12-55-50Z - Patrick paused at coarse7 and May
  normalization comparison scoped
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t12-55-50-z-patrick-paused-at-coarse7-and-may-normalization-comparison-scoped
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-19T12:55:50.317717+00:00'
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

# Codex checkpoint - 2026-10-19T12-55-50Z - Patrick paused at coarse7 and May normalization comparison scoped

## Summary
Patrick RMS独立复跑已按人类要求在三档coarse7完整断点核验后主动暂停，巡检删除；最新任务转向May的小规模编码器归一化训练/推理对照，尚在范围确认与证据准备。

## Story
原目标是SyncTalk训练/驱动/评价电平三轴。旧RMS真实数值失败与强制恢复probe116全skip独立封存后，用户授权同seed42完整随机初始化的RMS复跑，保留原FP16/模型/优化器/采样。后来用户要求“完成这一轮后先保存断点并中断实验”。当前+6coarse7自然训练与验证结束、F归档与跨gainblock全验后，原host因明确pause guard退出，0GPU或host信号。不是数值失败，不以旧receipt的training字样恢复训练。

最新用户又要求“测试主流编码器在训练和推理中是否有害，请使用may进行小规模的测试和训练对照，与我们之前的实验结果做对比”。上下文涉及报告52/53输入归一化开关；已发出异步问题确认是归一化影响还是更换编码器本身。可以继续独立May准备，不恢复旧Patrick。主流编码器特征变化不能替代生成与训练结果，也不能把SyncNet涨跌称感知质量。

## Working State
持久OUT=F:/Datasets/tts-exp/synctalk_multispeaker_full_20261017/trials/patrick_ave_rms_coarse_repeat_v1。active_controller.json→user_pause_v1/paused.json，暂停UTC2026-10-19T12:43:35.174715+00:00，21/30完整epoch，各42000attempts，0/−6/+6累计成功41991/41990/41991，实际更新不等保留、无补步，coarse10/finalphase未完成。三RAW全state checkpoint SHA分别497916c17d6966bfb35062f634b96fabb0d90a13facfd8e5bfde0759602c3d0a、e042410f62a58b8c61e676f81c9322f20602294ba701534fd06528b4ac90e7bb、788fd29c2563b658c749b9dcc3973611b7c5f8077ad5c5de578b8d9dbc5eba43。所有中间fullcp/trace保留。

user_pause_v1/closure_receipt.json passed，delivery_manifest.json21文件SHA466c5a4acbb399f4b53b445e6f1838649224479fc293ff96ea7d52bbc1298a43；memory_publication_receipt.json及automation_deleted_receipt.json真实读回。无任务GPU进程、无coarse8 job、无新视频/评分，patrick-rms已通过工具delete成功，旧synctalk-27也已删除。pause_request必须保留，未来若明确恢复另做审核handoff，不能盲重启229476或旧controller。Task blocked/current_step2，实验paused/PAUSED_BY_USER/IN_PROGRESS，地图和任务板一致；旧fine14/Chris/Cathy和原24full持续暂停，不归档chat/不称总任务done。

contract b44d67ce4daa7076e88648adefceb663118dd877a5b2a527a7bcf2d5439ee53b/19冻结源不动；共同随机initial d6ce1f369bd2ee1695288c93f1602fcf181f0329313c710388e6b7c70071be5c。原post-epoch metadata别名修复和RAW/derived provenance区别仍完整保留。当前暂停产生的controller_failure.json是预期pause AssertionError，有独立分类，不混同旧RMS density exp数值terminal。旧数值诊断表明失败权重中exp前logit过大，切未归一化特征和FP32仍溢出；RMS训练轨迹间接因果未识别。

May基线报告50是单身份已暴露工程先导：自然监督4150/550/1100帧、166/22/44秒，seed42共同随机initial，三gain各60000成功FP32更新、4096rays、实际frames/rays一致；固定test0训练−6/0/+6 Sync-C9.199/8.909/8.568，原时钟offset−3。完整科学成员在F:/Datasets/tts-exp/synctalk_may_training_pilot_20261017/remote，643成员已独立SHA；冗余archive已按receipt回收，科学成员留存。百度云九片独立Linux整包读回证据见cloud_readback_receipt.json，不能重复删除或称当前临时副本持久。

报告53同权重关闭输入归一化已完成8checkpoint/512encoder forward，7/8恢复可测增益方向；0TFG/SyncNet/训练，不能当“归一化有害”或“删除后更好”。固定F:/Datasets/tts-exp/encoder_normalization_ablation_20261019，报告52/53及后续调查需读全文后选小规模代表。新准备目录F:/Datasets/tts-exp/may_encoder_normalization_pilot_20261019/orientation仅BM读回，不是训练结果。尚无新May训练合同/初始化/新encoder或TFGforward。

机器易失资源Ubuntu wjj@192.168.50.2，SSH参数-b 192.168.50.1；ROOT=/dev/shm/synctalk-multispeaker-data-20261017，运行依赖隔离runtime_exec.sh。SHM不是持久备份。新May任务需要fresh只读资源/进程/数据检查后独立命名，不改共享venv/CloudDrive，不用旧pause授权触发其他任务。

## Changed Files
暂停操作新机器局部scratch：finalize_repeat_user_pause_v1.py、observe_repeat_user_pause_v1.py、publish_repeat_user_pause_v1.py、seal_repeat_user_pause_v1.py；对应源与结果都在F/user_pause_v1。observer已退出，不能重复执行immutable publisher/finalizer。当前仅新增read_may_encoder_memory_v1.py读取BM和本checkpoint writer；未改训练数学/冻结源。分支有大量无关脏文件，不bulk clean/stage/commit。

## Verification
前一完成动作实际核验21闭合epoch的archive/rawreceipt/6000trace、7个18000行跨gain采样/LRblock，最新三fullcp F/remote freshSHA；当前重新读closure_receipt确认passed/automation_deleted/no_automatic_resume。暂停无新增GPU或optimizer重算，旧fullcp保留。

当前已读Startup Router→研究地图→实验/任务指令→两schema，再查报告50/53和May旧train/common源码。CodeGraph可用，未重建。GitHub只读证据如下；未提交/推送、无新PR。尚未运行新May科学测试，异步范围问题未回答不能冒称已启动训练。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)。HEAD若远端422保持本地SHA，不构造commit链接。
- 既有报告50、53和F/user_pause_v1是恢复入口，checkpoint不是原始证据替代。

## Observations
- [result] Patrick RMS复跑三档coarse7、21/30已完整保存并主动暂停，0第8轮/新评分，巡检已删除。
- [decision] 新May范围独立于旧Patrick暂停；先分清归一化开关与更换编码器，保持训练端×推理端交叉对照和历史匹配基线。
- [blocker] 新May具体处理范围正在异步确认；旧coarse10/new27未完成且不自动恢复。
- [next_step] 读May和编码器canonical协议、核验可用输入/运行环境，依据用户范围答复冻结小规模May训练与推理控制并实际执行；保持旧Patrick暂停。

GitHub只读API结果：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}

## Relations
- continues [[Codex checkpoint - 2026-10-19T08-22-16Z - Patrick RMS repeat metadata handoff repaired and next gain running]]
- relates_to [[运行Patrick的RMS归一化AVE三档粗训练并评测]]
- relates_to [[Patrick同AVE逐录音RMS归一化三档粗训练与27格评测]]
- relates_to [[048-synctalk-patrick-same-ave-rms-coarse]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1061条；大量其他任务改动，不批量stage/clean/commit。
