---
title: Codex checkpoint - 2026-10-19T16-41-59Z - May gain normalization factorial
  running under remote completion authority
type: coding_session
permalink: tts-exp/codex/tts-exp/codex-checkpoint-2026-10-19-t16-41-59-z-may-gain-normalization-factorial-running-under-remote-completion-authority
status: open
project: tts-exp
repository: dogLeft233/tts-exp
repo_root: E:/Documents/tts-exp
cwd: E:/Documents/tts-exp
branch: codex/cross-tfg-audio-head-20261003
git_sha: 2b71639c3b555293375d248350999377168d515d
started: '2026-10-19T16:41:59.666003+00:00'
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

# Codex checkpoint - 2026-10-19T16-41-59Z - May gain normalization factorial running under remote completion authority

## Summary
May同编码器训练音量×推理音量×输入归一化的72格实验已真实闭合7/8新增模型，第8组运行；revision02接管真实远端完成状态，原主机SSH观察故障没有导致训练重放。

## Story
原目标是研究音量对训练和口型生成的影响。用户让Patrick当前轮保存后中断，已在三gain coarse7暂停；随后要求May小规模主流编码器对照，并指出整体换编码器不能排除归一化之外的因素。报告54先完成各encoder固定权重、训练norm开关与推理norm开关及driver gain。最新用户要求再改变训练输入音量，因此本次对固定HuBERT-base及wav2vec2-base各自切native HF零均值/单位方差，训练gain和推理driver gain各−6/0/+6完整交叉。内部LN/GN、权重、帧时钟和评分原PCM不动，不用跨encoder差异识别归一化效应。

每模型共同完整随机generator initial、seed42、12000成功FP32更新、4096rays/maxsteps16、原AdamW和60k LR前缀、EMA12 final。新8组±6训练gain；报告54四个训练0gain模型按758成员全SHA导入，不重训/重渲染/重评分。全部为12模型、72条件、144个10秒clip/评分矩阵，其中新48条件96clip；短预算单May单训练seed，不能声称总体有害/最佳音量或人类感知改善。新norm是native z-score，不是旧Patrick RMS0.1/FP16，旧Inf因果未由此解决。

## Working State
持久OUT=F:/Datasets/tts-exp/may_encoder_gain_norm_factorial_20261019；科学源码scripts/experiments/may_encoder_gain_norm_factorial_20261019/。protocol SHA4fdf3fa68595be8032839e1a5cfff1c5839b51c650440e619113055755e51518，initial SHA f9cd6cfadbc72812507e293e660b27e74a61f0556373162c487ffca6f5148e7b。train SHA a885ca21447cba5afb1aae10a1f2df746cc81b3ca249b571e8f838a06d639cb2，runtime SHA47e02454f44af6e84b6e15f6ec22de088127b2c62a183523eb02b9c23e55acce，evaluate SHA d08ea47080b7f53857ffe696b846868e2c9995a8a6f87a17d5cc0d9a64fa2670。冻结GPU数学源不改。

当前权威OUT/active_controller.json和active_workflow.json指向controller_revision02_remote_completion_handoff_v1/内receipt/workflow_receipt及各自failure。原root training_failure.json/workflow_failure.json永久是历史：SSH客户退出时第六真实job仍运行，旧host误复制增长stdout被SHA拒绝。底层SSH退出原因未证实，不能叫数值terminal。revision02审计前5完整state/全F成员、现第六PID/startticks/trace/未运行后两组及3CPU控制测试后，只观察原第六PID到自然12000更新rc0，0restart/0optimizer replay，再启动从未运行的7/8。startup_observation_readback.json保存第六完整12000行；completed_job_locations.json把第六指向REV/completed_job_readback，原root第六jobs目录是被拒绝的部分副本，不能拿旧running metadata否定真实完成。

最新只读latest_live_snapshot.json为7/8闭合，第八wav2vec2_base_off_trainp6约5000/12000，finite loss0.01035894，尚非完整epoch；无当前活动failure。原4HuBERT及wav2vecnormon±6、wav2vecoff−6闭合。进度会变，读实际receipt/execution优先。本checkpoint不升级全部完成。

机器易失远端Ubuntu wjj@192.168.50.2，SSH必须-b192.168.50.1；REMOTE=/dev/shm/may-encoder-gain-norm-factorial-20261019。原GPU锁/dev/shm/synctalk-multispeaker-data-20261017/preparation_GPU.lock，TeslaV10016GB。新REMOTE/official/data/historical依赖parent SHM符号链接，不退休parent。共享venv和CloudDrive配置不动。active host约257896；工作流统一exec session28943、controller95741可读输出但不能据PID硬编码重启。不要重跑原controller.py training或原continue_may_gain_norm_factorial_v1.py；revision02工作流也只能观察现有进程，不重复执行launcher。

新postprocess使用真实remote execution/PID完成权威，科学evaluate源不变；全部8训练完成后CPU全state审计，再8新渲染jobs+1评分job，96新矩阵、24000帧独立CPU消费，加48旧矩阵。四HuBERT新媒体固定放E:/Datasets/tts-exp/may_encoder_gain_norm_factorial_20261019_media；四wav2vec新媒体F OUT/remote/renders。analysis从protocol placements及completed_job_locations读取，不把旧第六部分job目录当最终rc。

工作流.codex/scratch/continue_may_factorial_workflow_revision02_v1.py将顺序跑review_training、postprocess_controller、analysis、report，到numerical_report_and_plots_ready_pending_actual_visual_and_memory后本chat仍须实际view_image gain_matrices.png/learning_curves.png/matched_frame_grid.png，保存visual_review并改report最终解释，publish_final发布正式报告55（写前确认编号未占），更新CONTEXT/HANDOFF，seal_delivery跨F/E逐成员双SHA。未看图/未BM/未封存不能称交付。仅此次已证明主机观察错误允许接续，首真实数值/actual lost/job failure须封存停止新jobs，不能把host修复当未来重放通行证。

存储：C约135MB、E2.456GB、F2.067GB，SHM6.18GB。可继续F/E当前协议，不因假想风险停工。两次可选C缓存动作（Edge旧编译缓存删除、公开HF缓存迁移E+保留junction）被自动审批在执行前拒绝，仅返回blocked by policy，零操作；不通过换工具/解释器绕过，两个rejected.json保留，已向用户说明。若实际BM发布被空间阻塞需人工释放C，不能擅清共享环境或科学资产。

旧报告50云整包9分片独立Linuxfresh读回whole archive SHA32f6b648e2236693e97a7eb1ea0ce483f80ea30a68cb2508b6893fb090278b22、643成员6049784599bytes全SHA通过后，只窄退休18个旧本地render副本2252642715bytes，原score/CP/PNG/JSON/repeat/preview保持。OUT/parent_cloud_full_member_review.json及parent_render_retirement_receipt.json passed，旧F根render_cold_storage_locator_20261019.json给每个member/bytes/SHA/9cloudparts。父报告54全部758成员仍物理保持未退休；新实验不声称cloud备份。新全部fullcp保留。

BM新实验May训练音量与推理音量的同权重归一化交叉对照running；Task运行May训练推理音量与归一化交叉对照in-progress/current_step2。scope/training和REV/memory_publication_receipt记录实际读回。只维护当前头部/唯一Observations，无逐epoch文档日志。旧report54已完成且不可改科学数据；旧Patrick user_pause_v1/paused.json SHA545c965703b38a42058fc5c3a65a4e662722470a306a37ca7ea9f97e1a8ec090，21/30轮三gain coarse7，实际累计更新41991/41990/41991；旧巡检已删除，旧fine14/Chris/Cathy/24full仍暂停，不能恢复或归档chat。

## Changed Files
未提交新目录scripts/experiments/may_encoder_gain_norm_factorial_20261019/含protocol/features/train/runtime/controller/job_wrapper/evaluate/evaluation_wrapper/review_training/review_features/review_media_clock/review_parent_cloud/postprocess_controller/analysis/report/publish_final/seal_delivery/controller_remote_completion_v2.py；targeted tests/experiments/test_may_encoder_gain_norm_factorial_20261019.py。CONTEXT.md/HANDOFF.md当前May段更新，LF修复记录entry_documents_training_update.json。大量无关dirty不stage/commit/push/clean。新GPU执行源在execution_sources/冻结；revision02 host/analysis等实际源在REV/source_revision02_*及admission，analysis自查当前sourceSHA。

## Verification
4有意义CPU tests实际passed：完整8新增factorial不重放0gain、训练数学diff仅trainfeaturegain与固定reference、真实tartrailer消费、FP32guard/loss路径。28feature缓存的原始float64gain→float32/nativeprocessor公式/shape/finite/SHA独立核通过；新88encoder forwards，归一化on训练gain相对L2约1e−6，off约2–18%，仅特征证据不是生成结果。16真实入口更新/0skip/fullstate往返、原initial/采样同通过。parent54全758成员freshSHApassed；当前前5模型完整CPUstate/optimizerstep12000/EMA12、全member和实际第六12000行采样/ray/LR/finite/rc0 passed；所有8最终CPU审计仍待工作流执行。旧cloud643成员恢复proof、18精确退休与locator完整passed。

## References
- [项目仓库](https://github.com/dogLeft233/tts-exp)；当前HEAD本地未推送若GitHub无法resolve则不构造链接。
- 父正式报告tts-exp/docs/experiments/54-may-encoder-normalization-pilot；新实验和Task保持当前running，最终报告待完整144矩阵及看图。

## Observations
- [result] May新训练gain factorial真实7/8新增模型闭合，第八运行；旧四零gain模型精确导入，无训练重放。
- [decision] 固定encoder权重和完整随机initial，用训练/推理norm及±6/0gain交叉；不将整体换encoder效应归因归一化。
- [decision] SSH进程退出不能作为远端科学job完成；revision02依据execution及PID startticks，只接管观察原活进程。
- [blocker] C缓存可选清理/迁移自动审批拒绝，尚未阻断F/E科学工作；禁止换工具绕过。
- [next_step] 观察revision02工作流完成第八训练及96新媒体/矩阵，独立QA通过后实际看三图、发布结果并封存跨F/E manifest。

GitHub只读证据：{"repository": "https://github.com/dogLeft233/tts-exp", "pull_requests": [], "commit": {"http_status": 422}}

## Relations
- continues [[Codex checkpoint - 2026-10-19T14-33-42Z - May normalization pilot delivered with same-encoder controls]]
- relates_to [[运行May训练推理音量与归一化交叉对照]]
- relates_to [[May训练音量与推理音量的同权重归一化交叉对照]]

本地Git快照：branch `codex/cross-tfg-audio-head-20261003`，HEAD `2b71639c3b555293375d248350999377168d515d`，status --short共1066条；大量其他任务改动，不批量stage/clean/commit。
