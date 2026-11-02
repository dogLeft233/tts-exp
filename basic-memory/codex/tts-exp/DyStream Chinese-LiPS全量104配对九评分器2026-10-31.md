---
title: DyStream Chinese-LiPS全量104配对九评分器2026-10-31
type: task
permalink: tts-exp/codex/tts-exp/dy-stream-chinese-li-ps-全量-104-配对九评分器-2026-10-31
status: active
priority: high
current_step: 3
---

# DyStream Chinese-LiPS全量104配对九评分器2026-10-31

- [status] active
- [current_step] 2
- [description] 用户要求“等pc talk跑完后跑下dystream”；已排队，等待PC-Talk服务器全部九项与独立验收完成，不代表DyStream已推理或评分完成。

## Steps
- [x] 读取本次192手册、官方部署记录与源码；冻结同209 PCM/图片、协议/源权重和统计设计；CPU预检、目标测试。
- [ ] 服务器后台等待PC-Talk complete和成功验收，核验本人进程退出后执行DyStream209推理，再完成Sync-C。
- [ ] 全量九评分器与原生控制；原始数组、说话人聚合统计、环境/文件验收和归档。
- [ ] 完整下载/SHA、本机独立复算、图像与媒体QA、正式报告及记忆更新。

## 冻结范围与协议
与PC-Talk全量逐文件同SHA的105自然+104既有云Plus TTS=209媒体；104配对/21说话人，缺024_29_F_TY_139的TTS仅保留自然描述，不新调云API。原始WAV、RMS−24 mono16kPCM16、人像SHA/每人固定映射、参考ID、质量标记与配对选择保持一致。仅复用音频和图片，**不复用PC-Talk视频或任何评分**；209唯一音频对应209个新的DyStream前向。

DyStream官方commit f587541ccaa77f092d03487d7d89502a67f0a084、资产revision9ad5b1d3b0ef7aece4b9855d972c1f819f04dfbb；独立部署env不改。官方app.run_inference speaker-only、listener原生零张量，steps5/seed222/CFG .5,.5,0,1、EMA、CPU MediaPipe、原生512×512/25fps。沿用已验证渲染四组件严格加载适配，官方源码不修改。私有worker只增证据钩子：验证实际送入运动模型的音频=原PCM+官方零前缀、listener=0，捕获实际feature_extractor.do_normalize、模型seed/输出latent帧数、CUDA模块与本人PID显存。原生Wav2Vec2预处理保持原状，不声称与PC-Talk mel前端相同。

原生帧数floor(samples/640)−1：官方prefix P，window=P+2/stride1，输出后去P；输入原PCM不改，尾部支持640–1279样点不补。原AAC视频保留；独立评分MP4 stream-copy同一video elementary stream并绑定完整驱动PCM为ALAC，逐样点/视频PTS/原生帧数验证，无变速、补帧或事后offset修正。SyncNet同PCM16MKV/min_track25/原官方两CLI、FP32主端点，保留原始矩阵/stdout/单脸track与内部codec；输出为对齐512脸，不能用输入原图坐标做PC式IoU断言。

同九评分器、权重/环境、端点/原生支持；CoGenAV/AVS仍95%人脸门，缺测不放宽；分类器125完整帧门。PEAVS208配对字典序64+64+64+16，单侧自然独立batch1。6类原生±400ms诊断各418，主要配对控制416，低于80%明确CONTROL_FAILED，不当科学阴性。SyncNet/PEAVS/SyncReward未新增该类控制，保留既有限制。

统计先每人句对均值，再等权说话人；主要严格84对/17人，补充104对/21人；完全相同质量定义，不按DyStream分数筛选。保留historical pilot_seen及其67严格/83全部句敏感性，仅表示PC-Talk历史试跑句，不是DyStream看过分数或独立参与者复现。每指标每分层缺任何预定句对的单侧即排除该说话人，不补值；10k speaker bootstrap、100k signflip seed20261031、Holm9分别分层。评价器证据≠人类同步真值；训练重合和参考区间仍未知。

## 执行与恢复
源码 scripts/experiments/dystream_lips_full104_20261031；服务器源码 /home/data2/wangjiajun/av-evaluation/experiments/dystream_lips_full104_20261031。Run /home/data2/wangjiajun/av-evaluation/jobs/dystream_lips_full104_rms24_nine_20261031_attempt01。**队列PID980230**（独立session），orchestrator.pid；queue.py先每45s只读PC-Talk状态，等待 /home/data2/wangjiajun/av-evaluation/jobs/pctalk_lips_full104_rms24_nine_20261031_attempt01/orchestrator_complete，同时检查独立raw/statistics/SyncNet几何/归档凭据PASS，再运行PC terminal_check确认本人任务退出。依赖失败写queue_blocked并退出，不提前用GPU，不重跑PC。通过后exec run.sh，同PID顺序生成→SyncNet→checkpoint/track核验→八项→raw/stat独立验收→环境/源文件复核→归档。依赖是服务器验收，不要求本机agent人工图QA先结束。

个人data2、UUID7272933b-1219-b5f8-a9dd-ceade9bbde37、4线程/affinity0-3、生成Torch55%/评分60%，资源门等空余；CPU MediaPipe+Mesa避免跨卡EGL。未改共享base/驱动或他人进程。HDTF TalkLikeYou继续用户暂停，不恢复。无新云API调用、无凭证入文件。

本机 results/dystream/lips_full104_rms24_nine_20261031_attempt01；**watcher PID78182**，Popen start_new_session独立session，45s只读监听。终局自动terminal check、SFTP断点下载/SHA、九端点/原件独立重建、说话人统计复算和CSV/PNG/PDF/209视频HTML。失败只保留审查状态，不重试科学。不自动发布BM正式报告；READY_FOR_REVIEW仍需agent核看图与媒体、写报告/Task done。标准视频目录results/dystream/{natural_raw,tts_raw}/<source>/。本机休眠不影响服务器；若Mac重启只恢复watcher/下载，禁止重启已存在的科学任务。

## 已通过的预检与边界
preparation PASS209媒体/104配对/21说话人/严格84对17人，145源/资产/cache pins、446复制文件SHA核对，DyStream pip freeze匹配部署基线；官方资产pyc排除为运行时缓存。无新TFG或评分结果。prequeue_validation PASS22实验文件本地/远端SHA、绑定CoGenAV源编译；历史3.08s部署成片仅做CPU codec fixture，重封装后原PCM精确、video PTS和video elementary stream不变，**不是这轮cohort样本或新前向**。7个DyStream门槛/原生帧测试+4个PC统计/缓存测试共11PASS，Python/shell语法PASS。

初次CPU preparation在创建run之前因acceptance03.exit位置写成delivery子目录而失败，原tar/source manifest及attempt01_failed.log保留；确认实际部署根acceptance03.exit=0后更正，无GPU或媒体生成。worker do_normalize读取明确用processor.feature_extractor。准备后、任何前向前仅修source snapshot/archive纳入worker_infer.sh，initial manifest、最终本地源manifest与prequeue_source_revision均留存；科学参数未变。源manifest已在排队前重新冻结。不得把CPU预检/旧部署样片当作本轮真实CUDA验收；首条未来推理失败则停下检查。

## Relations
- follows [[PC-Talk Chinese-LiPS全量104配对九评分器2026-10-31]]
- relates_to [[DyStream 192服务器部署]]
- relates_to [[Chinese-LiPS 官方测试集 21人×5句]]

## 2026-11-01接续失败修复与恢复
PC-Talk九项/终局/terminal已PASS，原队列980230在01:49左右依约接续并通过runtime核验，首worker却在import torch时失败：torch.fx.pass_manager导入stdlib queue.Queue，被实验同目录queue.py遮蔽。失败发生在CUDA初始化/模型加载之前，0完成/0评分；不再是等PC未完成。recovery01_failed/保留全部失败日志、首worker记录、旧PID、orchestrator_failed及queue源txt；原source_snapshot/experiment_source_manifest仍在。

精确回归测试“从worker目录from queue import Queue”先RED；仅将活动入口queue.py改名wait_for_pctalk.py，GREEN，合计12目标测试PASS。原worker.py/生成/评分/模型/209输入/所有超参数不变，原22文件SHA及446输入复制SHA先核验，旧记录PID全部已退出。新增run_resume01.sh与verify_runtime_resume01.py，仅绕过已执行的启动once门、另存source_snapshot_resume01并验证experiment_source_manifest_resume01。恢复的runtime仍严格检查相同五评分环境、145DyStream源权重/cache pins和部署freeze。原run.sh不改、不重新prepare或重跑PC。服务器env真实torch2.6.0+cu124从该目录导入通过。

新orchestrator PID1614215（独立sessionbash run_resume01.sh）、首worker1614484已实际加载运动权重/EMA、CPU MediaPipe并进入Generating motion sequence，不把import通过当作完整推理验收。恢复启动时完成0/209，首条媒体/PCM/PTS验收仍待；实时数以generation receipt为准。本机旧watcher78182因FAILED正常退出，原失败状态已保留recovery01_failed_status/；新独立sessionwatcher92563、local_watcher_resume01.log，已跨调用存活并显示RUNNING/generation。recovery01_authorization.json和resume01_launch.json记录修复边界与PID，不自动修改数据或重试科学失败。

恢复后首条007_26_M_YS_064_N已实际CUDA生成并通过receipt验收：295帧/25fps，原生AAC保留，video elementary stream及PTS不变，评分PCM逐样点精确、动态有效；目前1/209成片，继续下一条，尚未评分。这条是本轮真实输入验收，取代“首条验收仍待”的恢复瞬时状态；后续实时进度以receipt和状态文件为准。

## 2026-11-01第二次恢复：原生MoviePy尾帧校验
原resume01在56已完成receipt后，于040_25_F_ZX_083_T的validation count==expected处停止，评分仍0；该条worker_result PASS且249个model/render latent已成功生成。实际官方AAC导出/无损评分副本均250帧25fps/10.00s，输入160000样点；MoviePy安装源码Clip.py:471使用np.arange(0,duration,1/fps)，249/25=9.96的浮点grid生成250时间点，最后重复尾帧。末两帧平均像素差.01308（前一相邻差1.08233），与native reader尾重复一致。官方app模型输出249而封装250，不是模型前向失败。

新增native_export.py精确复现已安装MoviePy时间grid；generate_native_export_resume02.py仅修验证原生导出数，并分别记录rendered_latent_frames、native_export_extra_frames、motion_unsupported_tail_samples和导出末帧差。原inputs/protocol、worker/原生AAC/评分MP4/PCM/PTS不修改，不裁帧或补帧，不重跑已完成前向。audit_native_export_resume02.py对应检查模型latent数和原生导出数。固定5秒支持/六类控制/评分数学和参数均不变；协议解释另存protocol_native_export_clarification.json（评分前冻结），原protocol JSON SHA保持。

原56receipts全部匹配精确exporter grid；625个原generation/video文件和原inputs/protocol SHA绑定到recovery02_preexisting_science_manifest.json。原56完成以及第57worker成功记录都复用，第57仅重做媒体验证。原日志/失败标记/旧PID/状态完整留recovery02_failed/，所有旧记录PID确认退出，原resume01科学源码SHA全部核对。13目标测试PASS（含已观察249→250与295→295和125–333帧边界）；source_manifest_resume02绑定新入口，新verify_runtime_resume02.py继续145源权重/cache、五评分env和部署freeze。run_resume02.sh记录独立source_snapshot_resume02。源码原件不回写。

新orchestrator1958449（独立sessionbash run_resume02.sh），本机watcher2773（旧92563失败后退出，recovery02_failed_status保留）；local_watcher_resume02.log。恢复启动先runtime核验，之后原57视频校验→继续剩余152新前向，再SyncNet/九项/独立复算归档。恢复时56完整receipt/57成功前向，评分0；实时数以server_status与generation receipt为准。不是全量已完成，HDTF TLY继续用户暂停。

resume02已跨进程存活并重新通过runtime；第57原成片仅重新验证通过，生成日志明确alias=same previously completed forward; validation repair only。当前57/209完整receipt、SyncNet0，继续下一条；625原文件/inputs/protocol复核凭据recovery02_reuse_check.json供独立核验，不把完成一次校验误报全部评分完成。

## 2026-11-01全量Sync-C检查点及论文协议核对
209生成和SyncNet已全完成，checkpoint PASS：按21人N5.627/T5.945/Δ+.318，95%CI[−.026,+.645]，原始signflip p=.089529，14/21正；严格84对17人N5.593/T5.951/Δ+.358，CI[−.070,+.757]，p=.126189。Holm9与最终九项审计仍待；最近进入MTD，未把检查点当九项已完成。

本次独立只读核对209 GPU receipt均success、驱动PCM tensor exact、模型seed222、processor.feature_extractor.do_normalize=True，权重missing/unexpected均0；209生成PCM/PTS核验PASS，201普通原生导出、8条MoviePy多1重复尾帧。SyncNet geometry/stdout/raw-FP64核验209 PASS，最大C误差2.3687417e−6；offset计数−2:101、−1:92、0:10、−3:6，无搜索边界堆积。官方SyncNetInstance.py:144公式median(mdist)−minval与我们的FP32主归约一致。句级媒体均值N5.625/T5.937，改为论文式可能的片均值不会消除与8.136的差距；当前主统计仍按说话人等权，不回写。

**论文复现资格仍未建立，不能把低分全部归因中文域偏移。** 论文v2 Table1 HDTF-100离线8.136；本轮Chinese-LiPS中文音频+异人HDTF人像，单独RMS−24，不是作者100片/裁剪清单，作者预处理描述含低SyncNet过滤，本轮不按SyncNet筛样本。5步Euler、CFG all1/speaker.5/listener.5与公开demo一致，但论文§4.1明确anchor w_r=.5，公开sample.yaml/app默认anchor=0，我们用0；属于已确认的论文/发布默认配置差异，尚未做消融，不能声称解释了2分多差距。

公开WrapedWav2Vec使用全上下文is_causal=False，不含论文RoPE/lookahead在线机制；作者在官方Issue3明确这是先发布的离线版本，并称其生成效果最好。因此不能误说“因为我们没有开启lookahead所以低分”，也不能擅自is_causal=True而不换配套训练权重。公开完整checkpoint加载匹配只证明发布版内部一致，不证明是论文Table1逐权重复现。公开仓库未找到与Table1绑定的完整评分参数/权重/裁脸/轨迹聚合入口；论文附录的SyncScore文字说明不足以证明与经典SyncNet V2 confidence逐项一致。结论应标“官方公开离线代码＋项目统一SyncNet协议”，不标“论文8.136复现”。

后续定位差距应另立小规模校准：同源HDTF原视频/首帧/原音频，先测GT、官方入口成片；同片对照原AAC与项目PCM评分，最后固定样本比较anchor0/.5。尚未运行这些新对照，不改当前run参数/媒体/统计。用户暂停的HDTF TalkLikeYou不得自动恢复。

来源（2026-11-01核对）：
- https://arxiv.org/html/2512.24408v2#S4.SS1 （5步/CFG）
- https://arxiv.org/html/2512.24408v2#S4.T1 （HDTF-100离线）
- https://arxiv.org/html/2512.24408v2#A5 （指标定义，未绑定完整CLI）
- https://github.com/RobinWitch/DyStream/blob/main/configs/motion_gen/sample.yaml （anchor0/steps5）
- https://github.com/RobinWitch/DyStream/issues/3 （作者离线/在线发布说明，实际页面评论核对）
本机syncnet_geometry_audit.json与syncnet_checkpoint.json已下载；实际issue评论精确锚若不可定位用issue3主页，不推断未读作者澄清。
